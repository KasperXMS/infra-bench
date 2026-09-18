"""Quality-gated reporting and measured break-even analysis for calibration_v0."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal, cast

from ..io import write_json, write_jsonl
from ..schemas.calibration_v0 import (
    CalibrationAnchorTask,
    CalibrationBreakEvenAnalysis,
    CalibrationBreakEvenPoint,
    CalibrationCellResult,
    CalibrationEvaluatorRecord,
    CalibrationQuality,
    CalibrationRun,
    CalibrationSummary,
    CalibrationTask,
    CalibrationTaskResult,
    CalibrationWorkflow,
    CalibrationWorkflowId,
    CalibrationWorkflowStep,
    CalibrationWorld,
    CalibrationWorldId,
)
from .measurement import (
    METRIC_SEMANTICS,
    build_critical_path_report,
    reconstruct_calibration_run,
)

H1 = "H1_distributed_constrained"
H2 = "H2_distributed_favorable"
CENTRALIZED = "centralized_raw"
LOCAL = "local_reduction"
DEFAULT_SWEEP_MBPS = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0)


def _parse_answer_mapping(text: str | None, question_ids: set[str]) -> dict[str, str]:
    if not text or not text.strip():
        raise ValueError("missing final answer")
    candidate = text.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        lines = candidate.splitlines()
        if len(lines) < 3:
            raise ValueError("invalid fenced JSON answer")
        candidate = "\n".join(lines[1:-1]).strip()
    try:
        raw_payload: object = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError("final answer is not valid JSON") from exc
    if not isinstance(raw_payload, dict):
        raise ValueError("final answer must be a JSON object")
    payload = cast(dict[object, object], raw_payload)
    if set(payload) != question_ids:
        raise ValueError("final answer must contain exactly the configured question IDs")
    answers: dict[str, str] = {}
    for question_id, value in payload.items():
        if not isinstance(question_id, str):
            raise ValueError("question IDs must be strings")
        if not isinstance(value, str) or value.upper() not in {"A", "B", "C", "D"}:
            raise ValueError(f"answer for {question_id!r} must be A, B, C, or D")
        answers[question_id] = value.upper()
    return answers


def evaluate_calibration_runs(
    tasks: Sequence[CalibrationTask],
    evaluators: Sequence[CalibrationEvaluatorRecord],
    runs: Sequence[CalibrationRun],
) -> list[CalibrationRun]:
    """Score final answers with evaluator-only keys; malformed answers fail closed."""
    task_by_id = {task.task_id: task for task in tasks}
    evaluator_by_id = {record.task_id: record for record in evaluators}
    if len(task_by_id) != len(tasks) or len(evaluator_by_id) != len(evaluators):
        raise ValueError("task and evaluator task IDs must be unique")
    if set(task_by_id) != set(evaluator_by_id):
        raise ValueError("evaluator records must match calibration tasks exactly")
    for task_id, task in task_by_id.items():
        evaluator = evaluator_by_id[task_id]
        if evaluator.evaluator_id != task.evaluator_id:
            raise ValueError(f"evaluator_id mismatch for task {task_id!r}")
        if {item.question_id for item in evaluator.answers} != {
            item.question_id for item in task.questions
        }:
            raise ValueError(f"evaluator question IDs mismatch task {task_id!r}")

    evaluated: list[CalibrationRun] = []
    for run in runs:
        if run.status != "completed":
            evaluated.append(run)
            continue
        task = task_by_id.get(run.task_id)
        evaluator = evaluator_by_id.get(run.task_id)
        if task is None or evaluator is None:
            raise ValueError(f"run references task without evaluator {run.task_id!r}")
        reported = run.quality
        details: dict[str, Any] = {}
        try:
            parsed = _parse_answer_mapping(
                run.final_answer if run.final_answer is not None else run.answer,
                {question.question_id for question in task.questions},
            )
            answer_key = {
                answer.question_id: answer.correct_option for answer in evaluator.answers
            }
            correct = sum(parsed[key] == answer_key[key] for key in answer_key)
            score = correct / len(answer_key)
            passed = score >= task.quality_threshold
            details.update(
                {
                    "correct_count": correct,
                    "question_count": len(answer_key),
                    "parse_valid": True,
                }
            )
        except ValueError as exc:
            score = 0.0
            passed = False
            details.update({"parse_valid": False, "parse_error": str(exc)})
        if reported is not None:
            details["runtime_reported_quality_consistent"] = (
                abs(reported.score - score) <= 1e-9
                and reported.passed == passed
                and reported.evaluator_id == task.evaluator_id
            )
        quality = CalibrationQuality(
            score=score,
            passed=passed,
            evaluator_id=task.evaluator_id,
            details=details,
        )
        evaluated.append(run.model_copy(update={"quality": quality}))
    return evaluated


def reference_workflows() -> list[CalibrationWorkflow]:
    """Return the two dataset-independent workflows on the generic operator surface."""
    return [
        CalibrationWorkflow(
            workflow_id=CENTRALIZED,
            description="Move all fixed raw chunks to 4090 and invoke the strong VLM once.",
            steps=[
                CalibrationWorkflowStep(
                    step_id="reason_over_raw_chunks",
                    operator_id="invoke_model",
                    execution_role="reasoning_4090",
                    input_kinds=["raw_video_chunk"],
                    output_kinds=["final_answer"],
                )
            ],
        ),
        CalibrationWorkflow(
            workflow_id=LOCAL,
            description=(
                "Run generic fixed-policy frame/clip sampling at each artifact-local agent, "
                "then move reduced evidence to 4090 for final reasoning."
            ),
            steps=[
                CalibrationWorkflowStep(
                    step_id="local_generic_reduction",
                    operator_id="invoke_model",
                    execution_role="artifact_local_agent",
                    input_kinds=["raw_video_chunk"],
                    output_kinds=["sampled_frames", "reduced_clip", "semantic_evidence"],
                ),
                CalibrationWorkflowStep(
                    step_id="reason_over_reduced_evidence",
                    operator_id="invoke_model",
                    execution_role="reasoning_4090",
                    input_kinds=["sampled_frames", "reduced_clip", "semantic_evidence"],
                    output_kinds=["final_answer"],
                ),
            ],
        ),
    ]


def validate_calibration_design(
    tasks: Sequence[CalibrationTask],
    worlds: Sequence[CalibrationWorld],
    workflows: Sequence[CalibrationWorkflow],
) -> None:
    """Reject confounds before measured results are compared."""
    if len(tasks) != 2:
        raise ValueError("calibration_v0 requires exactly two real long-video tasks")
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("calibration task IDs must be unique")
    by_world = {world.world_id: world for world in worlds}
    if set(by_world) != {H1, H2} or len(worlds) != 2:
        raise ValueError("calibration_v0 requires exactly the H1 and H2 worlds")
    h1 = by_world[H1]
    h2 = by_world[H2]
    if h1.network.bandwidth_mbps >= h2.network.bandwidth_mbps:
        raise ValueError("H1 bandwidth must be lower than H2 bandwidth")
    if h1.network.rtt_ms <= h2.network.rtt_ms:
        raise ValueError("H1 RTT must be higher than H2 RTT")
    stable_h1 = h1.model_dump(exclude={"world_id", "network", "metadata"})
    stable_h2 = h2.model_dump(exclude={"world_id", "network", "metadata"})
    if stable_h1 != stable_h2:
        raise ValueError("H1 and H2 may differ only in network conditions")
    workflow_ids = [workflow.workflow_id for workflow in workflows]
    if len(workflows) != 2 or set(workflow_ids) != {CENTRALIZED, LOCAL}:
        raise ValueError("calibration_v0 requires the two reference workflows")


def _median(values: Iterable[float | int]) -> float:
    return round(float(statistics.median(values)), 6)


def _cell_result(
    task: CalibrationTask,
    workflow_id: CalibrationWorkflowId,
    world_id: CalibrationWorldId,
    runs: Sequence[CalibrationRun],
    *,
    minimum_repeats: int,
) -> CalibrationCellResult:
    completed = [run for run in runs if run.status == "completed"]
    quality_runs = [run for run in completed if run.quality and run.quality.passed]
    pass_rate = len(quality_runs) / len(completed) if completed else 0.0
    quality_gate_passed = bool(completed) and all(
        run.quality is not None
        and run.quality.evaluator_id == task.evaluator_id
        and run.quality.passed
        and run.quality.score >= task.quality_threshold
        for run in completed
    )
    eligible = len(completed) >= minimum_repeats and quality_gate_passed
    medians: dict[str, float | None] = {}
    availability: dict[str, Literal["available", "unavailable"]] = {}
    if completed:
        reconstructed = [reconstruct_calibration_run(run) for run in completed]
        medians = {
            "raw_artifact_bytes": _median(
                run.artifacts.raw_bytes for run in completed if run.artifacts
            ),
            "reduced_artifact_bytes": _median(
                run.artifacts.reduced_bytes for run in completed if run.artifacts
            ),
            "cross_agent_transfer_bytes": _median(
                run.transfers.bytes for run in completed if run.transfers
            ),
            "transfer_count": _median(
                run.transfers.count for run in completed if run.transfers
            ),
            "transfer_latency_ms": _median(
                run.transfers.latency_ms for run in completed if run.transfers
            ),
            "local_preprocessing_ms": _median(
                run.service.local_preprocessing_ms for run in completed if run.service
            ),
            "remote_model_ms": _median(
                run.service.remote_model_ms for run in completed if run.service
            ),
            "service_total_ms": _median(
                run.service.total_ms for run in completed if run.service
            ),
            "input_tokens": _median(
                run.usage.input_tokens for run in completed if run.usage
            ),
            "output_tokens": _median(
                run.usage.output_tokens for run in completed if run.usage
            ),
            "e2e_latency_ms": _median(
                run.e2e_latency_ms for run in completed if run.e2e_latency_ms is not None
            ),
            "local_preprocessing_sum_ms": _median(
                item.local_preprocessing_sum_ms for item in reconstructed
            ),
            "transfer_sum_ms": _median(item.transfer_sum_ms for item in reconstructed),
            "service_sum_ms": _median(item.service_sum_ms for item in reconstructed),
            "remote_service_sum_ms": _median(
                item.remote_service_sum_ms for item in reconstructed
            ),
        }
        critical_available = all(
            item.critical_path_availability == "available" for item in reconstructed
        )
        critical_fields = {
            "local_preprocessing_critical_ms": [
                item.local_preprocessing_critical_ms for item in reconstructed
            ],
            "transfer_critical_ms": [item.transfer_critical_ms for item in reconstructed],
            "service_critical_ms": [item.service_critical_ms for item in reconstructed],
            "planner_critical_ms": [item.planner_critical_ms for item in reconstructed],
            "critical_path_ms": [item.critical_path_ms for item in reconstructed],
        }
        for name, values in critical_fields.items():
            available_values = [value for value in values if value is not None]
            is_available = critical_available and len(available_values) == len(values)
            medians[name] = _median(available_values) if is_available else None
            availability[name] = "available" if is_available else "unavailable"
        for name in (
            "local_preprocessing_sum_ms",
            "transfer_sum_ms",
            "service_sum_ms",
            "remote_service_sum_ms",
            "e2e_latency_ms",
        ):
            availability[name] = "available"
        quality_values = [run.quality.score for run in completed if run.quality]
        if quality_values:
            medians["task_quality"] = _median(quality_values)
        costs = [
            run.usage.api_cost_usd
            for run in completed
            if run.usage and run.usage.api_cost_usd is not None
        ]
        if costs:
            medians["api_cost_usd"] = _median(costs)
    return CalibrationCellResult(
        task_id=task.task_id,
        workflow_id=workflow_id,
        world_id=world_id,
        attempted_runs=len(runs),
        completed_runs=len(completed),
        quality_pass_rate=pass_rate,
        quality_gate_passed=quality_gate_passed,
        eligible_for_comparison=eligible,
        medians=medians,
        metric_availability=availability,
    )


def _winner_and_margin(
    first: CalibrationCellResult, second: CalibrationCellResult
) -> tuple[CalibrationWorkflowId, float]:
    first_ms = first.medians["e2e_latency_ms"]
    second_ms = second.medians["e2e_latency_ms"]
    if first_ms is None or second_ms is None:
        raise ValueError("eligible calibration cells require measured E2E latency")
    if first_ms <= second_ms:
        return first.workflow_id, round((second_ms - first_ms) / first_ms, 6)
    return second.workflow_id, round((first_ms - second_ms) / second_ms, 6)


def summarize_calibration(
    tasks: Sequence[CalibrationTask],
    worlds: Sequence[CalibrationWorld],
    workflows: Sequence[CalibrationWorkflow],
    runs: Sequence[CalibrationRun],
    *,
    minimum_repeats: int = 3,
    anchor_margin: float = 0.10,
) -> CalibrationSummary:
    validate_calibration_design(tasks, worlds, workflows)
    known_tasks = {task.task_id for task in tasks}
    known_worlds = {world.world_id for world in worlds}
    known_workflows = {workflow.workflow_id for workflow in workflows}
    seen_run_ids: set[str] = set()
    seen_completed_repeats: set[tuple[str, str, str, int]] = set()
    grouped: dict[tuple[str, str, str], list[CalibrationRun]] = defaultdict(list)
    for run in runs:
        if run.task_id not in known_tasks:
            raise ValueError(f"run references unknown task {run.task_id!r}")
        if run.world_id not in known_worlds or run.workflow_id not in known_workflows:
            raise ValueError(f"run {run.run_id!r} references an unknown world/workflow")
        if run.run_id in seen_run_ids:
            raise ValueError(f"duplicate run_id {run.run_id!r}")
        repeat_key = (run.task_id, run.workflow_id, run.world_id, run.repeat)
        if run.status == "completed" and repeat_key in seen_completed_repeats:
            raise ValueError(f"duplicate completed calibration cell repeat {repeat_key!r}")
        seen_run_ids.add(run.run_id)
        # Failed attempts remain in raw_runs.jsonl as diagnostics and may be retried
        # using the same planned repeat number. Successful measurements stay unique.
        if run.status == "completed":
            seen_completed_repeats.add(repeat_key)
        grouped[(run.task_id, run.workflow_id, run.world_id)].append(run)

    cells: list[CalibrationCellResult] = []
    cell_lookup: dict[tuple[str, str, str], CalibrationCellResult] = {}
    for task in tasks:
        for world_id in (H1, H2):
            for workflow_id in (CENTRALIZED, LOCAL):
                key = (task.task_id, workflow_id, world_id)
                cell = _cell_result(
                    task,
                    workflow_id,
                    world_id,
                    grouped[key],
                    minimum_repeats=minimum_repeats,
                )
                cells.append(cell)
                cell_lookup[key] = cell

    task_results: list[CalibrationTaskResult] = []
    for task in tasks:
        task_cells = [cell for cell in cells if cell.task_id == task.task_id]
        quality_ok = all(cell.quality_gate_passed for cell in task_cells)
        eligible = all(cell.eligible_for_comparison for cell in task_cells)
        if not quality_ok:
            result = CalibrationTaskResult(
                task_id=task.task_id,
                quality_gate_passed=False,
                comparison_eligible=False,
                preference_reversal=False,
                infra_sensitive_anchor_candidate=False,
                reason="quality_gate_failed; system performance was not compared",
            )
        elif not eligible:
            result = CalibrationTaskResult(
                task_id=task.task_id,
                quality_gate_passed=True,
                comparison_eligible=False,
                preference_reversal=False,
                infra_sensitive_anchor_candidate=False,
                reason=f"fewer than {minimum_repeats} completed runs in at least one cell",
            )
        else:
            h1_winner, h1_margin = _winner_and_margin(
                cell_lookup[(task.task_id, CENTRALIZED, H1)],
                cell_lookup[(task.task_id, LOCAL, H1)],
            )
            h2_winner, h2_margin = _winner_and_margin(
                cell_lookup[(task.task_id, CENTRALIZED, H2)],
                cell_lookup[(task.task_id, LOCAL, H2)],
            )
            reversal = h1_winner == LOCAL and h2_winner == CENTRALIZED
            anchor = reversal and min(h1_margin, h2_margin) >= anchor_margin
            reason = (
                "admitted as infra-sensitive anchor candidate"
                if anchor
                else "expected preference reversal not observed with the required margin"
            )
            result = CalibrationTaskResult(
                task_id=task.task_id,
                quality_gate_passed=True,
                comparison_eligible=True,
                h1_winner=h1_winner,
                h2_winner=h2_winner,
                h1_margin=h1_margin,
                h2_margin=h2_margin,
                preference_reversal=reversal,
                infra_sensitive_anchor_candidate=anchor,
                reason=reason,
            )
        task_results.append(result)

    return CalibrationSummary(
        policy={
            "minimum_completed_runs_per_cell": minimum_repeats,
            "quality_gate": "all completed runs pass original evaluator threshold",
            "comparison_requires_all_four_cells": True,
            "expected_h1_winner": LOCAL,
            "expected_h2_winner": CENTRALIZED,
            "anchor_minimum_relative_margin": anchor_margin,
            "relative_margin_formula": "(loser_e2e - winner_e2e) / winner_e2e",
        },
        metric_definitions=METRIC_SEMANTICS,
        cells=cells,
        tasks=task_results,
        totals={
            "task_count": len(tasks),
            "raw_run_count": len(runs),
            "failed_run_count": sum(run.status == "failed" for run in runs),
            "comparison_eligible_task_count": sum(item.comparison_eligible for item in task_results),
            "anchor_candidate_count": sum(
                item.infra_sensitive_anchor_candidate for item in task_results
            ),
        },
    )


def _quality_runs(runs: Sequence[CalibrationRun], task_id: str, workflow_id: str) -> list[CalibrationRun]:
    return [
        run
        for run in runs
        if run.task_id == task_id
        and run.workflow_id == workflow_id
        and run.status == "completed"
        and run.quality is not None
        and run.quality.passed
    ]


def _modeled_latency_ms(
    base_ms: float,
    round_trips: float,
    path_bytes: float,
    bandwidth_mbps: float,
    rtt_ms: float,
) -> float:
    return base_ms + round_trips * rtt_ms + path_bytes * 8.0 / (bandwidth_mbps * 1000.0)


def analyze_break_even(
    summary: CalibrationSummary,
    worlds: Sequence[CalibrationWorld],
    runs: Sequence[CalibrationRun],
    *,
    sweep_mbps: Sequence[float] = DEFAULT_SWEEP_MBPS,
) -> CalibrationBreakEvenAnalysis:
    """Estimate BW* from measured bytes and E2E-minus-transfer base time."""
    by_world = {world.world_id: world for world in worlds}
    points: list[CalibrationBreakEvenPoint] = []
    for task_result in summary.tasks:
        if not task_result.comparison_eligible:
            continue
        central = _quality_runs(runs, task_result.task_id, CENTRALIZED)
        local = _quality_runs(runs, task_result.task_id, LOCAL)
        if not central or not local:
            continue

        def measured(
            workflow_runs: Sequence[CalibrationRun],
            *,
            parallel_transfers: bool,
        ) -> tuple[float, float, float, float, float, float, float]:
            def observed_network_ms(run: CalibrationRun) -> float:
                if run.transfers is None:
                    return 0.0
                if parallel_transfers and run.transfers.records:
                    return max(record.latency_ms for record in run.transfers.records)
                return float(run.transfers.latency_ms)

            base_values = [
                float(run.e2e_latency_ms) - observed_network_ms(run)
                for run in workflow_runs
                if run.e2e_latency_ms is not None and run.transfers is not None
            ]
            path_bytes = [
                (
                    max(record.bytes for record in run.transfers.records)
                    if parallel_transfers and run.transfers.records
                    else run.transfers.bytes
                )
                for run in workflow_runs
                if run.transfers
            ]
            round_trips = [
                (
                    1
                    if parallel_transfers and run.transfers.count > 0
                    else run.transfers.count
                )
                for run in workflow_runs
                if run.transfers
            ]
            return (
                max(0.0, _median(base_values)),
                _median(run.transfers.bytes for run in workflow_runs if run.transfers),
                _median(run.transfers.count for run in workflow_runs if run.transfers),
                _median(path_bytes),
                _median(round_trips),
                _median(
                    run.service.local_preprocessing_ms
                    for run in workflow_runs
                    if run.service
                ),
                _median(
                    run.service.remote_model_ms for run in workflow_runs if run.service
                ),
            )

        (
            central_base,
            central_bytes,
            central_count,
            central_path_bytes,
            central_round_trips,
            central_local_service,
            central_remote_service,
        ) = measured(central, parallel_transfers=True)
        (
            local_base,
            local_bytes,
            local_count,
            local_path_bytes,
            local_round_trips,
            local_local_service,
            local_remote_service,
        ) = measured(local, parallel_transfers=False)
        for world_id in (H1, H2):
            world = by_world[world_id]
            added_rtt_ms = world.network.rtt_ms
            baseline_rtt_ms = float(
                world.metadata.get("break_even_physical_baseline_rtt_ms", 0.0)
            )
            rtt_ms = added_rtt_ms + baseline_rtt_ms
            denominator = (
                local_base
                - central_base
                + (local_round_trips - central_round_trips) * rtt_ms
            )
            numerator = (central_path_bytes - local_path_bytes) * 8.0
            bandwidth_star = (
                numerator / (1000.0 * denominator)
                if numerator > 0.0 and denominator > 0.0
                else None
            )
            sweep: list[dict[str, float | str]] = []
            bandwidth_values: set[float] = {
                float(value) for value in sweep_mbps if value > 0
            }
            if bandwidth_star is not None:
                bandwidth_values.add(float(bandwidth_star))
            bandwidths: list[float] = sorted(bandwidth_values)
            for bandwidth in bandwidths:
                central_ms = _modeled_latency_ms(
                    central_base,
                    central_round_trips,
                    central_path_bytes,
                    bandwidth,
                    rtt_ms,
                )
                local_ms = _modeled_latency_ms(
                    local_base,
                    local_round_trips,
                    local_path_bytes,
                    bandwidth,
                    rtt_ms,
                )
                sweep.append(
                    {
                        "bandwidth_mbps": round(bandwidth, 6),
                        "centralized_raw_ms": round(central_ms, 6),
                        "local_reduction_ms": round(local_ms, 6),
                        "winner": CENTRALIZED if central_ms <= local_ms else LOCAL,
                    }
                )
            points.append(
                CalibrationBreakEvenPoint(
                    task_id=task_result.task_id,
                    world_id=world_id,
                    rtt_ms=rtt_ms,
                    added_rtt_ms=added_rtt_ms,
                    physical_baseline_rtt_ms=baseline_rtt_ms,
                    bandwidth_star_mbps=(
                        round(bandwidth_star, 6) if bandwidth_star is not None else None
                    ),
                    centralized_base_ms=central_base,
                    local_reduction_base_ms=local_base,
                    centralized_local_preprocessing_ms=central_local_service,
                    centralized_remote_model_ms=central_remote_service,
                    local_reduction_local_preprocessing_ms=local_local_service,
                    local_reduction_remote_model_ms=local_remote_service,
                    centralized_transfer_bytes=central_bytes,
                    local_reduction_transfer_bytes=local_bytes,
                    centralized_transfer_count=central_count,
                    local_reduction_transfer_count=local_count,
                    centralized_network_path_bytes=central_path_bytes,
                    local_reduction_network_path_bytes=local_path_bytes,
                    centralized_network_round_trips=central_round_trips,
                    local_reduction_network_round_trips=local_round_trips,
                    sweep=sweep,
                    reason=(
                        None
                        if bandwidth_star is not None
                        else "no positive finite crossing for the measured inputs at this RTT"
                    ),
                )
            )
    return CalibrationBreakEvenAnalysis(
        method=(
            "This bandwidth-sensitivity estimate is distinct from measured critical-path "
            "reporting. For each workflow, base_ms is median measured E2E minus a fixed "
            "reference-workflow network-overlap assumption: centralized_raw assumes its three "
            "raw transfers overlap (max link latency/bytes and one RTT), while local_reduction "
            "assumes measured evidence transfers serialize. These assumptions are not labeled "
            "as observed critical time. The modeled network term is "
            "round_trips*RTT_ms + path_bytes*8/(bandwidth_Mbps*1000). "
            "RTT is configured added delay plus the measured physical-baseline "
            "representative stored in world metadata. "
            "Measured local preprocessing and remote model service are reported "
            "separately and are included, not treated as free, in the E2E-derived base."
        ),
        tasks=points,
    )


def _markdown(summary: CalibrationSummary, analysis: CalibrationBreakEvenAnalysis) -> str:
    lines = [
        "# calibration_v0 summary",
        "",
        "System performance is compared only after both workflows pass the original task evaluator in all worlds.",
        "",
        "| Task | Quality gate | H1 winner | H1 margin | H2 winner | H2 margin | Anchor candidate |",
        "| --- | --- | --- | ---: | --- | ---: | --- |",
    ]
    for task in summary.tasks:
        lines.append(
            f"| {task.task_id} | {'PASS' if task.quality_gate_passed else 'FAIL'} | "
            f"{task.h1_winner or '-'} | "
            f"{task.h1_margin if task.h1_margin is not None else '-'} | "
            f"{task.h2_winner or '-'} | "
            f"{task.h2_margin if task.h2_margin is not None else '-'} | "
            f"{'YES' if task.infra_sensitive_anchor_candidate else 'NO'} |"
        )
    lines.extend(
        [
            "",
            "## Measured cells",
            "",
            "`sum` metrics are aggregate work and may exceed E2E under concurrency. "
            "`critical` metrics require complete timestamps, dependencies, and trace coverage; "
            "a dash means the trace cannot support that claim.",
            "",
            "| Task | World | Workflow | Completed | Quality pass rate | E2E ms | Local sum ms | Local critical ms | Transfer sum ms | Transfer critical ms | Service sum ms | Service critical ms |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for cell in summary.cells:
        metrics = cell.medians
        local_critical = metrics.get("local_preprocessing_critical_ms")
        transfer_critical = metrics.get("transfer_critical_ms")
        service_critical = metrics.get("service_critical_ms")
        lines.append(
            f"| {cell.task_id} | {cell.world_id} | {cell.workflow_id} | "
            f"{cell.completed_runs} | {cell.quality_pass_rate:.3f} | "
            f"{metrics.get('e2e_latency_ms', '-')} | "
            f"{metrics.get('local_preprocessing_sum_ms', '-')} | "
            f"{'-' if local_critical is None else local_critical} | "
            f"{metrics.get('transfer_sum_ms', '-')} | "
            f"{'-' if transfer_critical is None else transfer_critical} | "
            f"{metrics.get('service_sum_ms', '-')} | "
            f"{'-' if service_critical is None else service_critical} |"
        )
    lines.extend(
        [
            "",
            "## Metric semantics",
            "",
            "- `local_preprocessing_ms`, `transfer_latency_ms`, and `service_total_ms` are retained as deprecated aggregate aliases.",
            "- `*_sum_ms` is total measured work across calls/links and is not a wall-clock critical path.",
            "- `*_critical_ms` is emitted only when timestamps, dependency evidence, and trace coverage are complete.",
            "- `e2e_latency_ms` is the observed runtime wall clock and remains the workflow-comparison metric.",
        ]
    )
    lines.extend(["", "## Break-even estimates", ""])
    if not analysis.tasks:
        lines.append("No quality-gated task currently has enough measurements for BW* estimation.")
    for point in analysis.tasks:
        value = (
            f"{point.bandwidth_star_mbps:.3f} Mbps"
            if point.bandwidth_star_mbps is not None
            else "no positive finite crossing"
        )
        lines.append(f"- `{point.task_id}` at RTT {point.rtt_ms:g} ms: BW* = {value}.")
    lines.append("")
    return "\n".join(lines)


def write_calibration_report(
    tasks: Sequence[CalibrationTask],
    evaluators: Sequence[CalibrationEvaluatorRecord],
    worlds: Sequence[CalibrationWorld],
    workflows: Sequence[CalibrationWorkflow],
    runs: Sequence[CalibrationRun],
    output_dir: Path,
    *,
    minimum_repeats: int = 3,
    anchor_margin: float = 0.10,
    sweep_mbps: Sequence[float] = DEFAULT_SWEEP_MBPS,
) -> tuple[CalibrationSummary, CalibrationBreakEvenAnalysis]:
    evaluated_runs = evaluate_calibration_runs(tasks, evaluators, runs)
    summary = summarize_calibration(
        tasks,
        worlds,
        workflows,
        evaluated_runs,
        minimum_repeats=minimum_repeats,
        anchor_margin=anchor_margin,
    )
    analysis = analyze_break_even(
        summary, worlds, evaluated_runs, sweep_mbps=sweep_mbps
    )
    critical_path_report = build_critical_path_report(evaluated_runs)
    output_dir.mkdir(parents=True, exist_ok=True)
    # The runtime-owned raw file is immutable evaluator input. Gold-derived quality is
    # written separately so reporting never rewrites raw_runs.jsonl.
    write_jsonl(output_dir / "evaluated_runs.jsonl", evaluated_runs)
    write_json(output_dir / "summary.json", summary.model_dump(mode="json"))
    write_json(
        output_dir / "break_even_analysis.json", analysis.model_dump(mode="json")
    )
    write_json(
        output_dir / "critical_path_report.json",
        critical_path_report.model_dump(mode="json"),
    )
    point_by_task: dict[str, list[float]] = defaultdict(list)
    for point in analysis.tasks:
        if point.bandwidth_star_mbps is not None:
            point_by_task[point.task_id].append(point.bandwidth_star_mbps)
    anchors = [
        CalibrationAnchorTask(
            task_id=task.task_id,
            h1_margin=float(task.h1_margin),
            h2_margin=float(task.h2_margin),
            bandwidth_star_mbps=(
                _median(point_by_task[task.task_id])
                if point_by_task[task.task_id]
                else None
            ),
        )
        for task in summary.tasks
        if task.infra_sensitive_anchor_candidate
        and task.h1_margin is not None
        and task.h2_margin is not None
    ]
    write_json(
        output_dir / "admitted_anchor_tasks.json",
        {
            "schema_version": "calibration-v0-anchor-list-v1",
            "tasks": [item.model_dump(mode="json") for item in anchors],
        },
    )
    (output_dir / "summary.md").write_text(
        _markdown(summary, analysis), encoding="utf-8"
    )
    return summary, analysis
