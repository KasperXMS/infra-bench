"""Trace-driven search for infrastructure-sensitive real task workflows.

This module intentionally separates three stages:

1. import only MAS-realizable, official-resolved runtime traces;
2. screen semantically distinct workflow pairs by measured resource demand;
3. admit a pair only after a robust winner reversal is observed in shared,
   materialized infrastructure worlds.

Modeled counterfactuals prioritize future experiments.  They are never treated
as benchmark evidence and can never admit a task by themselves.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

from infra_bench.io import write_json, write_jsonl
from infra_bench.schemas.task import TaskRecord
from infra_bench.schemas.workflow import WorkflowRecord

from .resource_calibration import (
    EmpiricalWorldMeasurement,
    WorkflowPairCalibration,
    calibrate_workflow_pair,
)
from .trace_workflows import (
    SemanticWorkflowCluster,
    TraceWorkflowRun,
    cluster_trace_workflows,
    discover_trace_runs,
)

_ORACLE_FREE_METADATA_KEYS = frozenset(
    {"base_commit", "dataset_id", "dataset_revision", "image", "repo", "version"}
)


def _empirical_measurements(
    first: SemanticWorkflowCluster,
    second: SemanticWorkflowCluster,
) -> list[EmpiricalWorldMeasurement]:
    by_world: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {first.workflow_id: [], second.workflow_id: []}
    )
    for cluster in (first, second):
        for run in cluster.runs:
            by_world[run.world_id][cluster.workflow_id].append(
                run.resource_signature.observed_e2e_ms
            )
    measurements: list[EmpiricalWorldMeasurement] = []
    for world_id, workflow_values in sorted(by_world.items()):
        if not all(workflow_values.values()):
            continue
        measurements.append(
            EmpiricalWorldMeasurement(
                world_id=world_id,
                workflow_e2e_ms={
                    workflow_id: tuple(values)
                    for workflow_id, values in workflow_values.items()
                },
            )
        )
    return measurements


def _robust_margin(calibration: WorkflowPairCalibration) -> float:
    if calibration.selected_empirical_worlds is None:
        return 0.0
    return min(item.margin for item in calibration.selected_empirical_worlds)


def _pair_rank(calibration: WorkflowPairCalibration) -> tuple[float, float, float]:
    return (
        float(calibration.admitted),
        _robust_margin(calibration),
        calibration.contrast.contrast_score,
    )


def _workflow_entry(cluster: SemanticWorkflowCluster) -> dict[str, Any]:
    worlds: dict[str, int] = defaultdict(int)
    for run in cluster.runs:
        worlds[run.world_id] += 1
    return {
        "workflow_id": cluster.workflow_id,
        "semantic_fingerprint": list(cluster.semantic_fingerprint),
        "official_resolved_run_count": len(cluster.runs),
        "run_ids": cluster.run_ids,
        "world_run_counts": dict(sorted(worlds.items())),
        "resource_signature": cluster.resource_signature.model_dump(mode="json"),
        "per_run_resource_signatures": [
            {
                "run_id": run.run_id,
                "world_id": run.world_id,
                "signature": run.resource_signature.model_dump(mode="json"),
            }
            for run in cluster.runs
        ],
        "representative_workflow": cluster.representative_workflow.model_dump(
            mode="json"
        ),
        "verification": {
            "mas_realizable": True,
            "official_resolved": True,
        },
    }


def _oracle_free_worlds(
    task: TaskRecord,
    first: SemanticWorkflowCluster,
    second: SemanticWorkflowCluster,
    calibration: WorkflowPairCalibration,
) -> list[dict[str, Any]]:
    """Build planner-safe exports without workflow or calibration labels."""

    if calibration.selected_empirical_worlds is None:
        return []
    exports: list[dict[str, Any]] = []
    planner_task = task.planner_view().model_dump(mode="json")
    planner_task["metadata"] = {
        key: value
        for key, value in planner_task["metadata"].items()
        if key in _ORACLE_FREE_METADATA_KEYS
    }
    for selected in calibration.selected_empirical_worlds:
        matching_runs = [
            run
            for cluster in (first, second)
            for run in cluster.runs
            if run.world_id == selected.world_id
        ]
        fact_variants = {
            tuple(sorted(run.world_facts.items())) for run in matching_runs
        }
        # A world ID with conflicting infrastructure facts is not safe to export.
        if len(fact_variants) != 1:
            continue
        exports.append(
            {
                "schema_version": "1.0",
                "world_id": selected.world_id,
                "task": planner_task,
                "infrastructure": dict(next(iter(fact_variants))),
            }
        )
    return exports


def _task_reason(
    *,
    excluded_reason: str | None,
    clusters: Sequence[SemanticWorkflowCluster],
    calibrations: Sequence[WorkflowPairCalibration],
) -> str:
    if excluded_reason is not None:
        return f"excluded: {excluded_reason}"
    if not clusters:
        return "no successful official-resolved MAS traces"
    if len(clusters) < 2:
        return "fewer than two semantically distinct verified workflows"
    if not any(item.contrast.sufficiently_different for item in calibrations):
        return "all workflow pairs failed the resource-signature contrast check"
    best = max(calibrations, key=_pair_rank)
    return best.reason


def analyze_trace_admission(
    tasks: Sequence[TaskRecord],
    runs: Iterable[TraceWorkflowRun],
    *,
    task_ids: Sequence[str] | None = None,
    excluded_tasks: Mapping[str, str] | None = None,
    minimum_resource_contrast: float = 0.35,
    admission_margin: float = 0.20,
) -> dict[str, Any]:
    """Analyze eligible traces without performing or scheduling new experiments."""

    selected_ids = list(task_ids or [task.task_id for task in tasks])
    task_by_id = {task.task_id: task for task in tasks}
    unknown = sorted(set(selected_ids) - set(task_by_id))
    if unknown:
        raise ValueError(f"task IDs absent from task bank: {', '.join(unknown)}")

    excluded = dict(excluded_tasks or {})
    clusters = cluster_trace_workflows(runs)
    clusters_by_task: dict[str, list[SemanticWorkflowCluster]] = defaultdict(list)
    for cluster in clusters:
        if cluster.task_id in selected_ids:
            clusters_by_task[cluster.task_id].append(cluster)

    task_reports: list[dict[str, Any]] = []
    admitted_pairs: list[dict[str, Any]] = []
    oracle_free_exports: list[dict[str, Any]] = []
    for task_id in selected_ids:
        task = task_by_id[task_id]
        task_clusters = sorted(
            clusters_by_task.get(task_id, []), key=lambda item: item.workflow_id
        )
        pair_results: list[WorkflowPairCalibration] = []
        if task_id not in excluded:
            for first, second in combinations(task_clusters, 2):
                calibration = calibrate_workflow_pair(
                    first,
                    second,
                    empirical_measurements=_empirical_measurements(first, second),
                    admission_margin=admission_margin,
                    minimum_dimension_score=minimum_resource_contrast,
                )
                pair_results.append(calibration)
                if calibration.admitted:
                    oracle_free_exports.extend(
                        _oracle_free_worlds(task, first, second, calibration)
                    )

        pair_results.sort(key=_pair_rank, reverse=True)
        task_admitted = any(item.admitted for item in pair_results)
        for calibration in pair_results:
            if not calibration.admitted:
                continue
            admitted_pairs.append(
                {
                    "task_id": task_id,
                    **calibration.model_dump(mode="json"),
                }
            )

        task_reports.append(
            {
                "task_id": task_id,
                "benchmark": task.source,
                "objective": task.instruction,
                "excluded_from_search": task_id in excluded,
                "exclusion_reason": excluded.get(task_id),
                "eligible_trace_count": sum(len(item.runs) for item in task_clusters),
                "semantic_workflow_count": len(task_clusters),
                "verified_workflows": [
                    _workflow_entry(cluster) for cluster in task_clusters
                ],
                "pair_calibrations": [
                    item.model_dump(mode="json") for item in pair_results
                ],
                "admitted": task_admitted,
                "reason": _task_reason(
                    excluded_reason=excluded.get(task_id),
                    clusters=task_clusters,
                    calibrations=pair_results,
                ),
            }
        )

    return {
        "schema_version": "1.0",
        "method": {
            "source": "real infra-aware-mas traces",
            "workflow_requirements": [
                "MAS-realizable",
                "official-resolved",
                "semantic-deduplicated",
            ],
            "resource_contrast_threshold": minimum_resource_contrast,
            "admission_margin": admission_margin,
            "modeled_counterfactuals_are_admission_evidence": False,
            "admission_rule": (
                "MAS-realizable AND benchmark-correct AND empirically infra-sensitive; "
                "opposite winners in two shared real worlds with both margins at or above "
                "the admission threshold"
            ),
        },
        "summary": {
            "task_count": len(task_reports),
            "trace_backed_task_count": sum(
                item["eligible_trace_count"] > 0 for item in task_reports
            ),
            "verified_workflow_count": sum(
                item["semantic_workflow_count"] for item in task_reports
            ),
            "admitted_task_count": sum(item["admitted"] for item in task_reports),
            "admitted_pair_count": len(admitted_pairs),
            "search_target_reached": bool(admitted_pairs),
        },
        "tasks": task_reports,
        "admitted_pairs": admitted_pairs,
        "oracle_free_world_exports": list(
            {
                (item["task"]["task_id"], item["world_id"]): item
                for item in oracle_free_exports
            }.values()
        ),
    }


def _verified_bank(report: Mapping[str, Any]) -> list[WorkflowRecord]:
    workflows: list[WorkflowRecord] = []
    for task in report["tasks"]:
        for item in task["verified_workflows"]:
            representative = WorkflowRecord.model_validate(item["representative_workflow"])
            provenance = {
                **representative.provenance,
                "trace_workflow_id": item["workflow_id"],
                "semantic_fingerprint": item["semantic_fingerprint"],
                "official_resolved_run_ids": item["run_ids"],
                "resource_signature": item["resource_signature"],
            }
            workflows.append(
                representative.model_copy(
                    update={
                        "workflow_id": item["workflow_id"],
                        "provenance": provenance,
                    }
                )
            )
    return workflows


def _format_float(value: object) -> str:
    if not isinstance(value, int | float):
        return "—"
    return f"{float(value):.3f}"


def render_trace_admission_markdown(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Trace-driven task admission search",
        "",
        "Only real MAS runs that submitted a patch and were resolved by the official "
        "benchmark evaluator enter this report. Modeled counterfactuals prioritize "
        "experiments; they do not satisfy admission.",
        "",
        "## Summary",
        "",
        "| Tasks | Trace-backed | Verified workflows | Admitted tasks | Admitted pairs |",
        "| ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {summary['task_count']} | {summary['trace_backed_task_count']} | "
            f"{summary['verified_workflow_count']} | {summary['admitted_task_count']} | "
            f"{summary['admitted_pair_count']} |"
        ),
        "",
        "## Task status",
        "",
        "| Task | Benchmark | Eligible runs | Semantic workflows | Admission | Reason |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for task in report["tasks"]:
        lines.append(
            f"| `{task['task_id']}` | {task['benchmark']} | "
            f"{task['eligible_trace_count']} | {task['semantic_workflow_count']} | "
            f"{'YES' if task['admitted'] else 'NO'} | {task['reason']} |"
        )

    for task in report["tasks"]:
        lines.extend(["", f"## `{task['task_id']}`", ""])
        if task["excluded_from_search"]:
            lines.extend(
                [
                    f"Search status: frozen/excluded (`{task['exclusion_reason']}`). Existing "
                    "traces are reported for audit only; no pair was recalibrated.",
                    "",
                ]
            )
        workflows = task["verified_workflows"]
        if not workflows:
            lines.append("No MAS-realizable, official-resolved trace workflow was found.")
            continue
        lines.extend(
            [
                "### Verified semantic workflows",
                "",
                "| Workflow | Runs | Worlds | Semantic operators | Search/read count | "
                "Local I/O ms | Remote calls/ms | Target/full count | Test ms | Context bytes | "
                "Cross-site bytes/ms | Parallelism | Verification/retry |",
                "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | "
                "---: | ---: | ---: |",
            ]
        )
        for workflow in workflows:
            signature = workflow["resource_signature"]
            worlds = ", ".join(
                f"{world}:{count}"
                for world, count in workflow["world_run_counts"].items()
            )
            operators = " -> ".join(workflow["semantic_fingerprint"])
            lines.append(
                f"| `{workflow['workflow_id']}` | {workflow['official_resolved_run_count']} | "
                f"{worlds} | {operators} | "
                f"{_format_float(signature['local_search_count'])}/"
                f"{_format_float(signature['local_read_count'])} | "
                f"{_format_float(signature['local_search_service_ms'] + signature['local_read_service_ms'])} | "
                f"{_format_float(signature['remote_reasoning_calls'])}/"
                f"{_format_float(signature['remote_reasoning_service_ms'])} | "
                f"{_format_float(signature['targeted_test_count'])}/"
                f"{_format_float(signature['full_test_count'])} | "
                f"{_format_float(signature['targeted_test_service_ms'] + signature['full_test_service_ms'])} | "
                f"{_format_float(signature['transmitted_context_bytes'])} | "
                f"{_format_float(signature['cross_site_bytes'])}/"
                f"{_format_float(signature['cross_site_transfer_ms'])} | "
                f"{_format_float(signature['max_parallelism'])} | "
                f"{_format_float(signature['verification_count'])}/"
                f"{_format_float(signature['retry_count'])} |"
            )

        calibrations = task["pair_calibrations"]
        if not calibrations:
            lines.extend(["", f"Calibration: not run. {task['reason']}."])
            continue
        lines.extend(
            [
                "",
                "### Pair screening and calibration",
                "",
                "| Workflow pair | Contrast | Perturbation dimensions | Modeled reversal | "
                "Empirical worlds | Empirical reversal | Admission | Reason |",
                "| --- | ---: | --- | --- | ---: | --- | --- | --- |",
            ]
        )
        for calibration in calibrations:
            workflow_pair = " / ".join(
                f"`{item}`" for item in calibration["workflow_ids"]
            )
            dimensions = ", ".join(calibration["contrast"]["selected_dimensions"]) or "—"
            lines.append(
                f"| {workflow_pair} | "
                f"{_format_float(calibration['contrast']['contrast_score'])} | {dimensions} | "
                f"{'YES' if calibration['modeled_reversal_found'] else 'NO'} | "
                f"{len(calibration['empirical_worlds'])} | "
                f"{'YES' if calibration['empirical_reversal_found'] else 'NO'} | "
                f"{'YES' if calibration['admitted'] else 'NO'} | {calibration['reason']} |"
            )
            for world in calibration["empirical_worlds"]:
                costs = ", ".join(
                    f"{key}={_format_float(value)} ms"
                    for key, value in world["workflow_median_e2e_ms"].items()
                )
                lines.append(
                    f"  - `{world['world_id']}`: {costs}; winner "
                    f"`{world['winner']}`, margin {_format_float(world['margin'])}."
                )

    lines.extend(
        [
            "",
            "## Export status",
            "",
            (
                "No oracle-free world was exported because no task passed empirical admission."
                if not report["admitted_pairs"]
                else "Oracle-free exports are produced only for empirically admitted pairs."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def build_trace_admission_search_report(
    tasks: Sequence[TaskRecord],
    *,
    trace_root: str | Path,
    output_dir: str | Path,
    task_ids: Sequence[str] | None = None,
    excluded_tasks: Mapping[str, str] | None = None,
    minimum_resource_contrast: float = 0.35,
    admission_margin: float = 0.20,
) -> dict[str, Any]:
    """Discover traces, run the fail-closed search, and write audit artifacts."""

    selected_ids = list(task_ids or [task.task_id for task in tasks])
    selected_tasks = [task for task in tasks if task.task_id in selected_ids]
    # Excluded tasks are still ingested so their prior evidence remains auditable.
    runs = discover_trace_runs(trace_root, selected_tasks)
    report = analyze_trace_admission(
        selected_tasks,
        runs,
        task_ids=selected_ids,
        excluded_tasks=excluded_tasks,
        minimum_resource_contrast=minimum_resource_contrast,
        admission_margin=admission_margin,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "admission_search_report.json", report)
    (destination / "admission_search_report.md").write_text(
        render_trace_admission_markdown(report), encoding="utf-8"
    )
    write_jsonl(destination / "verified_trace_workflow_bank.jsonl", _verified_bank(report))
    write_json(destination / "admitted_pairs.json", report["admitted_pairs"])
    world_export_path = destination / "oracle_free_worlds.json"
    if report["oracle_free_world_exports"]:
        write_json(
            world_export_path,
            report["oracle_free_world_exports"],
        )
    elif world_export_path.exists():
        world_export_path.unlink()
    return report
