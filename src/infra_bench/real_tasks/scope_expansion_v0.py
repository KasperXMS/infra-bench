"""Evaluation and sensitivity reporting for Scope Expansion v0."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from statistics import median
from typing import Any

from ..io import write_json, write_jsonl
from ..schemas.scope_expansion_v0 import (
    FORMAL_SCOPE_PROTOCOL_ID,
    ScopeCellSummary,
    ScopeEvaluatorRecord,
    ScopeQuality,
    ScopeRun,
    ScopeSensitivityLabel,
    ScopeSensitivityReport,
    ScopeTask,
    ScopeTaskSensitivity,
    ScopeWorkflow,
    ScopeWorkflowId,
    ScopeWorld,
    ScopeWorldId,
)

CENTRALIZED: ScopeWorkflowId = "centralized_raw"
DISTRIBUTED: ScopeWorkflowId = "distributed_retrieval"
STRUCTURED_DISTRIBUTED: ScopeWorkflowId = "distributed_compute"
H1: ScopeWorldId = "H1_distributed_constrained"
H2: ScopeWorldId = "H2_distributed_favorable"
LABEL_ORDER: tuple[ScopeSensitivityLabel, ...] = (
    "workflow_reversal",
    "communication_sensitive",
    "compute_sensitive",
    "representation_sensitive",
    "quality_risk",
    "parallelism_sensitive",
    "context_cost_sensitive",
)
DEFAULT_THRESHOLDS = {
    "relative_system_difference": 0.20,
    "compute_e2e_share": 0.10,
    "parallelism_speedup": 1.25,
    "quality_drop": 0.01,
}
METRIC_DEFINITIONS = {
    "sum_ms": "Aggregate measured work; it may exceed wall clock under concurrency.",
    "critical_ms": (
        "Recovered wall-clock contribution; null unless trace timestamps and dependencies "
        "support it."
    ),
    "e2e_latency_ms": "Observed end-to-end runtime wall clock.",
    "configured_network": "Configured bandwidth cap and added RTT for the world.",
    "measured_network": "Measured effective bandwidth when a real probe is available.",
    "observed_effective_transfer_mbps": (
        "Per-run transferred bytes divided by transfer sum_ms, then median across runs. "
        "This application-observed value includes configured added RTT and is not pure link "
        "capacity."
    ),
    "workflow_preference": (
        "Lower median E2E among workflows that both pass the original quality gate."
    ),
}


def _canonical_answer(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    return re.sub(r"\s+", " ", normalized)


def _official_prediction(value: str) -> str:
    """Apply the answer extraction used by MultiHop-RAG's qa_evaluate.py."""
    match = re.search(r'The answer to the question is "(.*?)"', value)
    return match.group(1) if match else value


def _canonical_prediction(value: str) -> str:
    official = _official_prediction(value)
    if official != value:
        return official
    answer_line = re.search(r"(?im)^\s*ANSWER:\s*(.+?)\s*$", value)
    return answer_line.group(1) if answer_line else value


def _official_token_intersection(prediction: str, answer: str) -> float:
    predicted_tokens = set(_official_prediction(prediction).lower().split())
    answer_tokens = set(answer.lower().split())
    return 1.0 if predicted_tokens.intersection(answer_tokens) else 0.0


def _official_longbench_choice(prediction: str) -> str | None:
    """Apply LongBench-v2's two official answer extraction regexes in order."""
    cleaned = prediction.replace("*", "")
    for pattern in (
        r"The correct answer is \(([A-D])\)",
        r"The correct answer is ([A-D])",
    ):
        match = re.search(pattern, cleaned)
        if match:
            return match.group(1)
    return None


def _workflow_pair(task: ScopeTask) -> tuple[ScopeWorkflowId, ScopeWorkflowId]:
    return (
        (CENTRALIZED, STRUCTURED_DISTRIBUTED)
        if task.task_family == "structured_data_analysis"
        else (CENTRALIZED, DISTRIBUTED)
    )


def validate_scope_design(
    tasks: Sequence[ScopeTask],
    evaluators: Sequence[ScopeEvaluatorRecord],
    worlds: Sequence[ScopeWorld],
    workflows: Sequence[ScopeWorkflow],
) -> None:
    """Validate one family-aware scope design without exposing gold to execution."""
    if not tasks:
        raise ValueError("scope design requires at least one task")
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("scope task IDs must be unique")
    multihop = [task for task in tasks if task.dataset == "MultiHop-RAG"]
    if multihop:
        multihop_evaluators = [
            evaluator
            for evaluator in evaluators
            if evaluator.task_id in {task.task_id for task in multihop}
        ]
        if len(multihop) != 3 or sorted(
            evaluator.evidence_document_count for evaluator in multihop_evaluators
        ) != [2, 3, 4]:
            raise ValueError("MultiHop-RAG scope requires exactly one M2, M3, and M4 task")
        top_ns = {
            task.retrieval_config.top_n
            for task in multihop
            if task.retrieval_config is not None
        }
        if len(top_ns) != 1:
            raise ValueError("MultiHop-RAG BM25 top_n must be uniform")

    by_task = {record.task_id: record for record in evaluators}
    if set(by_task) != {task.task_id for task in tasks}:
        raise ValueError("evaluator task IDs must exactly match visible task IDs")
    for task in tasks:
        evaluator = by_task[task.task_id]
        if evaluator.evaluator_type != task.evaluator_type:
            raise ValueError(f"evaluator type mismatch for {task.task_id}")
        if task.dataset == "MultiHop-RAG":
            candidate_ids = set(task.artifact_document_ids.values())
            missing = set(evaluator.supporting_document_ids) - candidate_ids
            if missing:
                raise ValueError(
                    f"BM25 candidate bundle for {task.task_id} does not cover gold documents"
                )

    if {world.world_id for world in worlds} != {H1, H2} or len(worlds) != 2:
        raise ValueError("scope design requires exactly H1 and H2")
    first, second = worlds
    first_fixed = first.model_dump(exclude={"world_id", "network", "metadata"})
    second_fixed = second.model_dump(exclude={"world_id", "network", "metadata"})
    if first_fixed != second_fixed:
        raise ValueError("H1 and H2 may differ only in network conditions")
    if first.network == second.network:
        raise ValueError("H1 and H2 must use different network conditions")

    required_workflows = {
        workflow_id for task in tasks for workflow_id in _workflow_pair(task)
    }
    available_workflows = {workflow.workflow_id for workflow in workflows}
    if not required_workflows <= available_workflows:
        raise ValueError(
            f"scope design is missing workflows {sorted(required_workflows - available_workflows)}"
        )


def evaluate_scope_runs(
    tasks: Sequence[ScopeTask],
    evaluators: Sequence[ScopeEvaluatorRecord],
    runs: Sequence[ScopeRun],
) -> list[ScopeRun]:
    """Recompute quality from evaluator-only records; runtime claims are non-authoritative."""
    task_by_id = {task.task_id: task for task in tasks}
    evaluator_by_id = {record.task_id: record for record in evaluators}
    evaluated: list[ScopeRun] = []
    for run in runs:
        task = task_by_id.get(run.task_id)
        evaluator = evaluator_by_id.get(run.task_id)
        if task is None or evaluator is None:
            raise ValueError(f"run {run.run_id!r} references an unknown task")
        if run.task_family != task.task_family:
            raise ValueError(
                f"run {run.run_id!r} task_family does not match the task bank"
            )
        if run.workflow_id not in _workflow_pair(task):
            raise ValueError(
                f"run {run.run_id!r} uses workflow {run.workflow_id!r} for "
                f"task family {task.task_family!r}"
            )
        if run.status != "completed":
            evaluated.append(run)
            continue
        assert run.final_answer is not None
        assert run.demand is not None
        expected_documents_per_agent = {site: 0 for site in ("A4", "A5", "A28")}
        for site in task.artifact_placement.values():
            expected_documents_per_agent[site] += 1
        if run.demand.artifact_count != task.artifact_count:
            raise ValueError(f"run {run.run_id!r} reports the wrong artifact_count")
        if run.demand.raw_bytes != sum(task.artifact_sizes.values()):
            raise ValueError(f"run {run.run_id!r} reports the wrong raw_bytes")
        if run.demand.raw_tokens != sum(task.artifact_tokens.values()):
            raise ValueError(f"run {run.run_id!r} reports the wrong raw_tokens")
        if run.demand.documents_per_agent != expected_documents_per_agent:
            raise ValueError(
                f"run {run.run_id!r} reports placement inconsistent with the task bank"
            )
        accepted = [evaluator.answer, *evaluator.acceptable_answers]
        if evaluator.evaluator_type == "longbench_v2_official_multiple_choice":
            extracted = _official_longbench_choice(run.final_answer)
            exact = extracted in accepted
            canonical = exact
            score = float(exact)
        else:
            extracted = _canonical_prediction(run.final_answer)
            exact = any(extracted.strip() == answer.strip() for answer in accepted)
            canonical = any(
                _canonical_answer(extracted) == _canonical_answer(answer)
                for answer in accepted
            )
            score = _official_token_intersection(run.final_answer, evaluator.answer)
        evaluated.append(
            run.model_copy(
                update={
                    "quality": ScopeQuality(
                        original_benchmark_score=score,
                        exact_match=exact,
                        canonical_answer_correct=canonical,
                        passed=score >= task.quality_threshold,
                        evaluator_type=evaluator.evaluator_type,
                        details={
                            "extracted_prediction": extracted,
                            "runtime_reported_quality_consistent": (
                                run.quality is None
                                or run.quality.original_benchmark_score == score
                            )
                        },
                    )
                }
            )
        )
    return evaluated


def _series(run: ScopeRun) -> tuple[str | None, str | None]:
    series = run.measurement_series_id or run.metadata.get("measurement_series_id")
    protocol = run.protocol_id or run.metadata.get("measurement_protocol")
    return (
        str(series) if series is not None else None,
        str(protocol) if protocol is not None else None,
    )


def _select_formal_series(runs: Sequence[ScopeRun]) -> tuple[str | None, list[ScopeRun]]:
    grouped: dict[str, list[tuple[int, ScopeRun]]] = defaultdict(list)
    for index, run in enumerate(runs):
        series, protocol = _series(run)
        if series and protocol == FORMAL_SCOPE_PROTOCOL_ID:
            grouped[series].append((index, run))
    if not grouped:
        return None, []

    def slots(records: list[tuple[int, ScopeRun]]) -> dict[tuple[bool, int], tuple[int, ScopeRun]]:
        selected: dict[tuple[bool, int], tuple[int, ScopeRun]] = {}
        for index, run in records:
            selected[(run.warmup, run.repeat)] = (index, run)
        return selected

    complete: list[tuple[int, str, dict[tuple[bool, int], tuple[int, ScopeRun]]]] = []
    for series, records in grouped.items():
        selected = slots(records)
        required = [(True, 0), (False, 1), (False, 2), (False, 3)]
        if all(
            key in selected and selected[key][1].status == "completed"
            for key in required
        ):
            complete.append((max(index for index, _ in records), series, selected))
    if complete:
        _, series, selected = max(complete)
    else:
        series, records = max(grouped.items(), key=lambda item: max(i for i, _ in item[1]))
        selected = slots(records)
    selected_runs = [run for _, run in sorted(selected.values(), key=lambda item: item[0])]
    return series, selected_runs


def _median(values: Sequence[float | int]) -> float:
    return float(median(values))


def _optional_median(values: Sequence[float | int | None]) -> float | None:
    if not values or any(value is None for value in values):
        return None
    return _median([float(value) for value in values if value is not None])


def _cell_summary(
    task_id: str,
    workflow_id: ScopeWorkflowId,
    world_id: ScopeWorldId,
    runs: Sequence[ScopeRun],
) -> ScopeCellSummary:
    series, selected = _select_formal_series(runs)
    completed = [run for run in selected if run.status == "completed"]
    warmups = [run for run in completed if run.warmup and run.repeat == 0]
    measured = [
        run for run in completed if not run.warmup and run.repeat in {1, 2, 3}
    ]
    protocol_passed = len(warmups) == 1 and {run.repeat for run in measured} == {1, 2, 3}
    quality_runs = [run for run in measured if run.quality is not None]
    quality_pass_rate = (
        sum(bool(run.quality and run.quality.passed) for run in quality_runs)
        / len(quality_runs)
        if quality_runs
        else 0.0
    )
    quality_gate = protocol_passed and len(quality_runs) == 3 and quality_pass_rate == 1.0

    def med(path: Callable[[ScopeRun], float | int | None]) -> float | None:
        values = [path(run) for run in measured]
        return _optional_median(values)

    medians: dict[str, float | int | None] = {}
    if measured:
        medians = {
            "artifact_count": med(lambda run: run.demand.artifact_count if run.demand else None),
            "raw_bytes": med(lambda run: run.demand.raw_bytes if run.demand else None),
            "raw_tokens": med(lambda run: run.demand.raw_tokens if run.demand else None),
            "retrieval_calls": med(lambda run: run.retrieval.calls if run.retrieval else None),
            "retrieval_sum_ms": med(lambda run: run.retrieval.sum_ms if run.retrieval else None),
            "retrieval_critical_ms": med(
                lambda run: run.retrieval.critical_ms if run.retrieval else None
            ),
            "retrieved_document_count": med(
                lambda run: run.retrieval.retrieved_document_count if run.retrieval else None
            ),
            "local_compute_calls": med(
                lambda run: run.local_compute.calls if run.local_compute else None
            ),
            "local_compute_sum_ms": med(
                lambda run: run.local_compute.sum_ms if run.local_compute else None
            ),
            "local_compute_critical_ms": med(
                lambda run: run.local_compute.critical_ms if run.local_compute else None
            ),
            "reduced_bytes": med(lambda run: run.artifacts.reduced_bytes if run.artifacts else None),
            "reduced_tokens": med(lambda run: run.artifacts.reduced_tokens if run.artifacts else None),
            "absolute_reducible_bytes": med(
                lambda run: run.artifacts.absolute_reducible_bytes if run.artifacts else None
            ),
            "transfer_count": med(lambda run: run.transfers.count if run.transfers else None),
            "transfer_bytes": med(lambda run: run.transfers.bytes if run.transfers else None),
            "transfer_sum_ms": med(lambda run: run.transfers.latency_ms if run.transfers else None),
            "transfer_critical_ms": med(
                lambda run: run.transfers.critical_ms if run.transfers else None
            ),
            "observed_effective_transfer_mbps": med(
                lambda run: (
                    run.transfers.bytes * 8.0 / (run.transfers.latency_ms * 1000.0)
                    if run.transfers and run.transfers.latency_ms > 0
                    else None
                )
            ),
            "local_preprocessing_ms": med(
                lambda run: run.service.local_preprocessing_ms if run.service else None
            ),
            "final_model_ms": med(lambda run: run.service.final_model_ms if run.service else None),
            "service_total_ms": med(lambda run: run.service.total_ms if run.service else None),
            "input_tokens": med(lambda run: run.usage.input_tokens if run.usage else None),
            "output_tokens": med(lambda run: run.usage.output_tokens if run.usage else None),
            "api_cost_usd": med(lambda run: run.usage.api_cost_usd if run.usage else None),
            "e2e_latency_ms": med(lambda run: run.e2e_latency_ms),
        }
        for site in ("A4", "A5", "A28"):
            medians[f"documents_{site}"] = med(
                lambda run, site=site: (
                    run.demand.documents_per_agent.get(site) if run.demand else None
                )
            )
    selected_protocol = _series(selected[-1])[1] if selected else None
    return ScopeCellSummary(
        task_id=task_id,
        workflow_id=workflow_id,
        world_id=world_id,
        selected_measurement_series_id=series,
        selected_protocol_id=selected_protocol,
        completed_warmups=len(warmups),
        completed_runs=len(measured),
        profiling_protocol_passed=protocol_passed,
        quality_gate_passed=quality_gate,
        quality_pass_rate=quality_pass_rate,
        quality_median=med(
            lambda run: (
                run.quality.original_benchmark_score if run.quality else None
            )
        ),
        medians=medians,
    )


def _number(cell: ScopeCellSummary, key: str) -> float | None:
    value = cell.medians.get(key)
    return float(value) if value is not None else None


def _relative_difference(left: float | None, right: float | None) -> float:
    if left is None or right is None:
        return 0.0
    scale = max(abs(left), abs(right))
    return abs(left - right) / scale if scale else 0.0


def _task_sensitivity(
    task: ScopeTask,
    cells: dict[tuple[str, str], ScopeCellSummary],
    thresholds: dict[str, float],
) -> ScopeTaskSensitivity:
    _, distributed_workflow = _workflow_pair(task)
    selected_series = {
        cell.selected_measurement_series_id for cell in cells.values()
    }
    series_consistent = len(selected_series) == 1 and None not in selected_series
    preferences: dict[ScopeWorldId, ScopeWorkflowId | None] = {}
    for world_id in (H1, H2):
        central = cells[(CENTRALIZED, world_id)]
        distributed = cells[(distributed_workflow, world_id)]
        if not series_consistent or not (
            central.quality_gate_passed and distributed.quality_gate_passed
        ):
            preferences[world_id] = None
            continue
        central_e2e = _number(central, "e2e_latency_ms")
        distributed_e2e = _number(distributed, "e2e_latency_ms")
        if central_e2e is None or distributed_e2e is None:
            preferences[world_id] = None
        else:
            preferences[world_id] = (
                CENTRALIZED
                if central_e2e <= distributed_e2e
                else distributed_workflow
            )

    central_cells = [cells[(CENTRALIZED, world)] for world in (H1, H2)]
    distributed_cells = [
        cells[(distributed_workflow, world)] for world in (H1, H2)
    ]
    quality_by_workflow: dict[ScopeWorkflowId, float | None] = {
        CENTRALIZED: _optional_median([cell.quality_median for cell in central_cells]),
        distributed_workflow: _optional_median(
            [cell.quality_median for cell in distributed_cells]
        ),
    }
    communication_differences = [
        max(
            _relative_difference(
                _number(cells[(CENTRALIZED, world)], "transfer_bytes"),
                _number(cells[(distributed_workflow, world)], "transfer_bytes"),
            ),
            _relative_difference(
                _number(cells[(CENTRALIZED, world)], "transfer_sum_ms"),
                _number(cells[(distributed_workflow, world)], "transfer_sum_ms"),
            ),
        )
        for world in (H1, H2)
    ]
    representation_reductions: list[float] = []
    compute_shares: list[float] = []
    parallelism_speedups: list[float] = []
    context_differences: list[float] = []
    for world in (H1, H2):
        central = cells[(CENTRALIZED, world)]
        distributed = cells[(distributed_workflow, world)]
        raw = _number(distributed, "raw_bytes")
        reduced = _number(distributed, "reduced_bytes")
        representation_reductions.append(
            (raw - reduced) / raw if raw and reduced is not None else 0.0
        )
        compute_critical_key = (
            "local_compute_critical_ms"
            if task.task_family == "structured_data_analysis"
            else "retrieval_critical_ms"
        )
        compute_sum_key = (
            "local_compute_sum_ms"
            if task.task_family == "structured_data_analysis"
            else "retrieval_sum_ms"
        )
        retrieval_critical = _number(distributed, compute_critical_key)
        e2e = _number(distributed, "e2e_latency_ms")
        compute_shares.append(
            retrieval_critical / e2e if retrieval_critical is not None and e2e else 0.0
        )
        retrieval_sum = _number(distributed, compute_sum_key)
        parallelism_speedups.append(
            retrieval_sum / retrieval_critical
            if retrieval_sum is not None and retrieval_critical
            else 0.0
        )
        context_differences.append(
            _relative_difference(
                _number(central, "input_tokens"),
                _number(distributed, "input_tokens"),
            )
        )

    quality_drop = 0.0
    central_quality = quality_by_workflow[CENTRALIZED]
    distributed_quality = quality_by_workflow[distributed_workflow]
    if central_quality is not None and distributed_quality is not None:
        quality_drop = max(0.0, central_quality - distributed_quality)
    triggers: dict[ScopeSensitivityLabel, bool] = {
        "workflow_reversal": (
            preferences[H1] is not None
            and preferences[H2] is not None
            and preferences[H1] != preferences[H2]
        ),
        "communication_sensitive": max(communication_differences) >= thresholds["relative_system_difference"],
        "compute_sensitive": max(compute_shares) >= thresholds["compute_e2e_share"],
        "representation_sensitive": max(representation_reductions) >= thresholds["relative_system_difference"],
        "quality_risk": quality_drop >= thresholds["quality_drop"],
        "parallelism_sensitive": max(parallelism_speedups) >= thresholds["parallelism_speedup"],
        "context_cost_sensitive": max(context_differences) >= thresholds["relative_system_difference"],
    }
    if not series_consistent:
        triggers = {label: False for label in LABEL_ORDER}
    values: dict[ScopeSensitivityLabel, dict[str, Any]] = {
        "workflow_reversal": {
            "triggered": triggers["workflow_reversal"],
            "preferences": preferences,
            "measurement_series_consistent": series_consistent,
        },
        "communication_sensitive": {"triggered": triggers["communication_sensitive"], "max_relative_difference": max(communication_differences)},
        "compute_sensitive": {"triggered": triggers["compute_sensitive"], "max_local_compute_critical_e2e_share": max(compute_shares)},
        "representation_sensitive": {"triggered": triggers["representation_sensitive"], "max_raw_to_evidence_byte_reduction": max(representation_reductions)},
        "quality_risk": {"triggered": triggers["quality_risk"], "centralized_minus_distributed_quality": quality_drop},
        "parallelism_sensitive": {"triggered": triggers["parallelism_sensitive"], "max_local_compute_sum_over_critical": max(parallelism_speedups)},
        "context_cost_sensitive": {"triggered": triggers["context_cost_sensitive"], "max_final_input_token_relative_difference": max(context_differences)},
    }
    return ScopeTaskSensitivity(
        task_id=task.task_id,
        labels=[label for label in LABEL_ORDER if triggers[label]],
        workflow_preference=preferences,
        quality_by_workflow=quality_by_workflow,
        evidence=values,
    )


def summarize_scope_expansion(
    tasks: Sequence[ScopeTask],
    worlds: Sequence[ScopeWorld],
    workflows: Sequence[ScopeWorkflow],
    runs: Sequence[ScopeRun],
    *,
    thresholds: dict[str, float] | None = None,
) -> ScopeSensitivityReport:
    selected_thresholds = DEFAULT_THRESHOLDS | (thresholds or {})
    cells: list[ScopeCellSummary] = []
    for task in tasks:
        applicable_workflows = set(_workflow_pair(task))
        for world in worlds:
            for workflow in workflows:
                if workflow.workflow_id not in applicable_workflows:
                    continue
                matching = [
                    run
                    for run in runs
                    if run.task_id == task.task_id
                    and run.world_id == world.world_id
                    and run.workflow_id == workflow.workflow_id
                ]
                cells.append(
                    _cell_summary(
                        task.task_id, workflow.workflow_id, world.world_id, matching
                    )
                )
    task_reports: list[ScopeTaskSensitivity] = []
    for task in tasks:
        lookup = {
            (cell.workflow_id, cell.world_id): cell
            for cell in cells
            if cell.task_id == task.task_id
        }
        task_reports.append(_task_sensitivity(task, lookup, selected_thresholds))
    return ScopeSensitivityReport(
        protocol={
            "protocol_id": FORMAL_SCOPE_PROTOCOL_ID,
            "warmup_runs": 1,
            "measured_runs": 3,
            "estimator": "median",
            "warmup_excluded_from_quality_and_metrics": True,
            "quality_before_efficiency": True,
            "no_anchor_admission_objective": True,
        },
        metric_definitions=METRIC_DEFINITIONS,
        thresholds=selected_thresholds,
        cells=cells,
        tasks=task_reports,
        totals={
            "task_count": len(tasks),
            "raw_run_count": len(runs),
            "warmup_run_count": sum(run.warmup for run in runs),
            "measured_run_count": sum(not run.warmup for run in runs),
            "protocol_complete_cell_count": sum(
                cell.profiling_protocol_passed for cell in cells
            ),
            "quality_gated_cell_count": sum(cell.quality_gate_passed for cell in cells),
            "labeled_task_count": sum(bool(task.labels) for task in task_reports),
        },
    )


def _markdown(report: ScopeSensitivityReport, worlds: Sequence[ScopeWorld]) -> str:
    world_by_id = {world.world_id: world for world in worlds}
    lines = [
        "# Scope Expansion v0 sensitivity summary",
        "",
        "All latency and preference statements use measured runs only. Warm-up rows are excluded, and efficiency is compared only where both workflows pass the original evaluator.",
        "",
        "## Network worlds",
        "",
        "| World | Configured bandwidth (Mbps) | Configured added RTT (ms) | Measured effective bandwidth (Mbps) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for world_id in (H1, H2):
        network = world_by_id[world_id].network
        measured = network.measured_effective_bandwidth_mbps
        lines.append(
            f"| {world_id} | {network.bandwidth_mbps:g} | {network.rtt_ms:g} | "
            f"{measured if measured is not None else '-'} |"
        )
    lines.extend(
        [
            "",
            "## Task sensitivity",
            "",
            "| Task | Labels | H1 preference | H2 preference | Centralized quality | Distributed quality |",
            "| --- | --- | --- | --- | ---: | ---: |",
        ]
    )
    for task in report.tasks:
        distributed_workflow = next(
            workflow_id
            for workflow_id in task.quality_by_workflow
            if workflow_id != CENTRALIZED
        )
        central_quality = task.quality_by_workflow[CENTRALIZED]
        distributed_quality = task.quality_by_workflow[distributed_workflow]
        lines.append(
            f"| {task.task_id} | {', '.join(task.labels) or '-'} | "
            f"{task.workflow_preference[H1] or '-'} | "
            f"{task.workflow_preference[H2] or '-'} | "
            f"{central_quality if central_quality is not None else '-'} | "
            f"{distributed_quality if distributed_quality is not None else '-'} |"
        )
    lines.extend(
        [
            "",
            "## Measured cells",
            "",
            "| Task | World | Workflow | Protocol | Quality gate | E2E ms | Transfer bytes | Transfer sum ms | Observed effective transfer Mbps | Input tokens |",
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for cell in report.cells:
        metrics = cell.medians
        lines.append(
            f"| {cell.task_id} | {cell.world_id} | {cell.workflow_id} | "
            f"{'PASS' if cell.profiling_protocol_passed else 'FAIL'} | "
            f"{'PASS' if cell.quality_gate_passed else 'FAIL'} | "
            f"{metrics.get('e2e_latency_ms', '-')} | "
            f"{metrics.get('transfer_bytes', '-')} | "
            f"{metrics.get('transfer_sum_ms', '-')} | "
            f"{metrics.get('observed_effective_transfer_mbps', '-')} | "
            f"{metrics.get('input_tokens', '-')} |"
        )
    lines.extend(
        [
            "",
            "`*_sum_ms` is aggregate work. `*_critical_ms` is emitted only when trace evidence supports a wall-clock critical path. A sensitivity label is multi-label evidence, not anchor admission.",
            "`observed_effective_transfer_mbps` is application-observed payload throughput over transfer sum time; it includes added RTT and must not be read as pure link capacity or a separate network probe.",
            "",
        ]
    )
    return "\n".join(lines)


def write_scope_expansion_report(
    tasks: Sequence[ScopeTask],
    evaluators: Sequence[ScopeEvaluatorRecord],
    worlds: Sequence[ScopeWorld],
    workflows: Sequence[ScopeWorkflow],
    runs: Sequence[ScopeRun],
    output_dir: Path,
    *,
    thresholds: dict[str, float] | None = None,
) -> ScopeSensitivityReport:
    """Write derived artifacts without creating or mutating runtime-owned raw runs."""
    validate_scope_design(tasks, evaluators, worlds, workflows)
    evaluated = evaluate_scope_runs(tasks, evaluators, runs)
    report = summarize_scope_expansion(
        tasks, worlds, workflows, evaluated, thresholds=thresholds
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "evaluated_runs.jsonl", evaluated)
    write_json(
        output_dir / "sensitivity_summary.json", report.model_dump(mode="json")
    )
    (output_dir / "sensitivity_summary.md").write_text(
        _markdown(report, worlds), encoding="utf-8"
    )
    return report
