"""Reconstruct realized-workflow metrics without assuming a reference workflow."""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime

from ..schemas.calibration_v0 import CalibrationRun
from ..schemas.measurement import (
    ArtifactLineageRecord,
    CriticalPathReport,
    RealizedWorkflowTrace,
    ReconstructedWorkflowMetrics,
    WorkflowTraceSpan,
)

METRIC_SEMANTICS = {
    "local_preprocessing_sum_ms": (
        "Sum of measured local action service durations. This is aggregate work and may "
        "exceed E2E when local actions run concurrently."
    ),
    "local_preprocessing_critical_ms": (
        "Local-action duration on the timestamped dependency critical path; unavailable "
        "unless timestamps, dependencies, and trace coverage are complete."
    ),
    "transfer_sum_ms": (
        "Sum of individual cross-site transfer durations. This is aggregate link work and "
        "may exceed E2E when transfers overlap."
    ),
    "transfer_critical_ms": (
        "Transfer duration on the timestamped dependency critical path; unavailable unless "
        "timestamps, dependencies, and trace coverage are complete."
    ),
    "service_sum_ms": (
        "Sum of measured action service durations across all sites; aggregate work, not "
        "wall-clock latency."
    ),
    "service_critical_ms": (
        "Action service duration on the timestamped dependency critical path."
    ),
    "critical_path_ms": (
        "Longest dependency-path sum of observed span wall-clock durations. It excludes "
        "untraced orchestration gaps and is never inferred from sums alone."
    ),
    "e2e_latency_ms": "Observed run wall-clock latency from runtime start to final result.",
    "legacy_aggregate_fields": (
        "local_preprocessing_ms, transfer_latency_ms, and service_total_ms are deprecated "
        "aggregate aliases for the corresponding *_sum_ms metrics."
    ),
}


def _timestamp_ms(value: str | float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, float | int):
        return float(value)
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000.0
    except ValueError:
        return None


def _elapsed_ms(span: WorkflowTraceSpan) -> float | None:
    start = _timestamp_ms(span.started_at)
    finish = _timestamp_ms(span.finished_at)
    if start is None or finish is None or finish < start:
        return None
    return finish - start


def _dependencies_and_lineage(
    trace: RealizedWorkflowTrace,
) -> tuple[list[tuple[str, str]], list[ArtifactLineageRecord]]:
    dependencies = {
        (dependency, span.span_id)
        for span in trace.spans
        for dependency in span.depends_on
    }
    producers: dict[str, str] = {}
    consumers: dict[str, set[str]] = defaultdict(set)
    transfers: dict[str, list[WorkflowTraceSpan]] = defaultdict(list)
    artifact_bytes: dict[str, int] = defaultdict(int)
    source_sites: dict[str, str] = {}
    destination_sites: dict[str, set[str]] = defaultdict(set)
    for span in trace.spans:
        for artifact_id in span.output_artifacts:
            producers.setdefault(artifact_id, span.span_id)
            artifact_bytes[artifact_id] = max(artifact_bytes[artifact_id], span.output_bytes)
            if span.site_id:
                source_sites.setdefault(artifact_id, span.site_id)
        for artifact_id in span.input_artifacts:
            consumers[artifact_id].add(span.span_id)
            artifact_bytes[artifact_id] = max(artifact_bytes[artifact_id], span.input_bytes)
        if span.span_kind == "transfer" and span.artifact_id:
            artifact_id = span.artifact_id
            transfers[artifact_id].append(span)
            artifact_bytes[artifact_id] = max(artifact_bytes[artifact_id], span.bytes)
            if span.src_site:
                source_sites.setdefault(artifact_id, span.src_site)
            if span.dst_site:
                destination_sites[artifact_id].add(span.dst_site)
            if span.producer_span_id:
                producers.setdefault(artifact_id, span.producer_span_id)
                dependencies.add((span.producer_span_id, span.span_id))
            if span.consumer_span_id:
                consumers[artifact_id].add(span.consumer_span_id)
                dependencies.add((span.span_id, span.consumer_span_id))

    artifact_ids = set(producers) | set(consumers) | set(transfers)
    lineage: list[ArtifactLineageRecord] = []
    for artifact_id in sorted(artifact_ids):
        producer = producers.get(artifact_id)
        transfer_spans = transfers.get(artifact_id, [])
        consumer_ids = sorted(consumers.get(artifact_id, set()))
        if producer and not transfer_spans:
            dependencies.update((producer, consumer) for consumer in consumer_ids)
        lineage.append(
            ArtifactLineageRecord(
                artifact_id=artifact_id,
                producer_span_id=producer,
                consumer_span_ids=consumer_ids,
                transfer_span_ids=[span.span_id for span in transfer_spans],
                bytes=artifact_bytes[artifact_id],
                source_site=source_sites.get(artifact_id),
                destination_sites=sorted(destination_sites.get(artifact_id, set())),
            )
        )
    return sorted(dependencies), lineage


def _critical_path(
    trace: RealizedWorkflowTrace,
    dependencies: list[tuple[str, str]],
) -> tuple[list[str], dict[str, float], list[str]]:
    reasons: list[str] = []
    if trace.timestamp_evidence != "complete":
        reasons.append("timestamp_evidence_not_complete")
    if trace.dependency_evidence != "complete":
        reasons.append("dependency_evidence_not_complete")
    if trace.trace_coverage != "complete":
        reasons.append("trace_coverage_not_complete")
    durations = {span.span_id: _elapsed_ms(span) for span in trace.spans}
    if any(value is None for value in durations.values()):
        reasons.append("one_or_more_spans_lack_valid_start_finish_timestamps")
    span_by_id = {span.span_id: span for span in trace.spans}
    for source, target in dependencies:
        source_finish = _timestamp_ms(span_by_id[source].finished_at)
        target_start = _timestamp_ms(span_by_id[target].started_at)
        if (
            source_finish is not None
            and target_start is not None
            and source_finish > target_start + 1e-6
        ):
            reasons.append("dependency_timestamps_are_inconsistent")
            break
    if reasons:
        return [], {}, reasons

    successors: dict[str, list[str]] = {span.span_id: [] for span in trace.spans}
    indegree = {span.span_id: 0 for span in trace.spans}
    predecessors: dict[str, list[str]] = {span.span_id: [] for span in trace.spans}
    for source, target in dependencies:
        successors[source].append(target)
        predecessors[target].append(source)
        indegree[target] += 1
    ready = deque(sorted(span_id for span_id, degree in indegree.items() if degree == 0))
    order: list[str] = []
    while ready:
        span_id = ready.popleft()
        order.append(span_id)
        for successor in sorted(successors[span_id]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
    if len(order) != len(trace.spans):
        return [], {}, ["dependency_graph_contains_cycle"]

    elapsed = {key: float(value) for key, value in durations.items() if value is not None}
    longest: dict[str, float] = {}
    previous: dict[str, str | None] = {}
    for span_id in order:
        if predecessors[span_id]:
            predecessor = max(
                predecessors[span_id], key=lambda item: (longest[item], item)
            )
            longest[span_id] = longest[predecessor] + elapsed[span_id]
            previous[span_id] = predecessor
        else:
            longest[span_id] = elapsed[span_id]
            previous[span_id] = None
    if not longest:
        return [], {}, ["trace_has_no_spans"]
    tail = max(longest, key=lambda item: (longest[item], item))
    path: list[str] = []
    cursor: str | None = tail
    while cursor is not None:
        path.append(cursor)
        cursor = previous[cursor]
    path.reverse()
    return path, elapsed, []


def reconstruct_workflow_metrics(
    trace: RealizedWorkflowTrace,
) -> ReconstructedWorkflowMetrics:
    """Reconstruct sums, lineage, and a defensible critical path from one trace."""
    dependencies, lineage = _dependencies_and_lineage(trace)
    path, elapsed, reasons = _critical_path(trace, dependencies)
    span_by_id = {span.span_id: span for span in trace.spans}
    action_spans = [span for span in trace.spans if span.span_kind == "action"]
    transfer_spans = [span for span in trace.spans if span.span_kind == "transfer"]
    planner_spans = [span for span in trace.spans if span.span_kind == "planner"]

    def duration(span: WorkflowTraceSpan) -> float:
        observed = _elapsed_ms(span)
        return observed if observed is not None else float(span.duration_ms or 0.0)

    def path_duration(*, kind: str | None = None, scope: str | None = None) -> float | None:
        if reasons:
            return None
        return sum(
            elapsed[span_id]
            for span_id in path
            if (kind is None or span_by_id[span_id].span_kind == kind)
            and (scope is None or span_by_id[span_id].service_scope == scope)
        )

    site_counts: dict[str, int] = defaultdict(int)
    operator_counts: dict[str, int] = defaultdict(int)
    model_counts: dict[str, int] = defaultdict(int)
    executor_counts: dict[str, int] = defaultdict(int)
    for span in action_spans:
        site_counts[span.site_id or "unknown"] += 1
        operator_counts[span.operator_id or "unknown"] += 1
        model_counts[span.model_id or "none"] += 1
        executor_counts[span.executor_id or "unknown"] += 1
    transfer_pairs: dict[str, int] = defaultdict(int)
    for span in transfer_spans:
        transfer_pairs[f"{span.src_site or 'unknown'}->{span.dst_site or 'unknown'}"] += (
            span.bytes
        )
    local_spans = [span for span in action_spans if span.service_scope == "local"]
    remote_spans = [span for span in action_spans if span.service_scope == "remote"]
    return ReconstructedWorkflowMetrics(
        run_id=trace.run_id,
        task_id=trace.task_id,
        workflow_id=trace.workflow_id,
        dependency_evidence=trace.dependency_evidence,
        timestamp_evidence=trace.timestamp_evidence,
        trace_coverage=trace.trace_coverage,
        artifact_lineage_availability=(
            "available" if trace.trace_coverage == "complete" else "unavailable"
        ),
        invocation_count=sum(
            span.model_id is not None or span.operator_id == "invoke_model"
            for span in action_spans
        ),
        action_count=len(action_spans),
        transfer_count=len(transfer_spans),
        dependencies=dependencies,
        artifact_lineage=lineage,
        site_action_counts=dict(sorted(site_counts.items())),
        operator_counts=dict(sorted(operator_counts.items())),
        model_counts=dict(sorted(model_counts.items())),
        executor_action_counts=dict(sorted(executor_counts.items())),
        transfer_site_pair_bytes=dict(sorted(transfer_pairs.items())),
        raw_input_bytes=sum(span.input_bytes for span in action_spans),
        produced_bytes=sum(span.output_bytes for span in action_spans),
        transfer_bytes=sum(span.bytes for span in transfer_spans),
        local_preprocessing_sum_ms=sum(duration(span) for span in local_spans),
        local_preprocessing_critical_ms=path_duration(scope="local"),
        transfer_sum_ms=sum(duration(span) for span in transfer_spans),
        transfer_critical_ms=path_duration(kind="transfer"),
        service_sum_ms=sum(duration(span) for span in action_spans),
        service_critical_ms=path_duration(kind="action"),
        local_service_sum_ms=sum(duration(span) for span in local_spans),
        remote_service_sum_ms=sum(duration(span) for span in remote_spans),
        planner_sum_ms=sum(duration(span) for span in planner_spans),
        planner_critical_ms=path_duration(kind="planner"),
        critical_path_ms=(sum(elapsed[item] for item in path) if not reasons else None),
        critical_path_span_ids=path,
        e2e_latency_ms=trace.e2e_latency_ms,
        input_tokens=sum(span.input_tokens for span in trace.spans),
        output_tokens=sum(span.output_tokens for span in trace.spans),
        quality=trace.quality,
        critical_path_availability="unavailable" if reasons else "available",
        critical_path_unavailable_reasons=reasons,
    )


def reconstruct_calibration_run(run: CalibrationRun) -> ReconstructedWorkflowMetrics:
    """Adapt legacy calibration rows; never invent missing timestamp/dependency evidence."""
    if run.trace is not None:
        trace = run.trace.model_copy(
            update={
                "quality": run.quality.score if run.quality else run.trace.quality,
                "e2e_latency_ms": run.e2e_latency_ms or run.trace.e2e_latency_ms,
            }
        )
        return reconstruct_workflow_metrics(trace)

    executions = run.executions
    site_counts: dict[str, int] = defaultdict(int)
    operator_counts: dict[str, int] = defaultdict(int)
    executor_counts: dict[str, int] = defaultdict(int)
    for execution in executions:
        site_counts[execution.site_id] += 1
        operator_counts[execution.operator_id] += 1
        executor_counts[execution.executor_id] += 1
    transfer_pairs: dict[str, int] = defaultdict(int)
    if run.transfers:
        for transfer in run.transfers.records:
            transfer_pairs[f"{transfer.src_site}->{transfer.dst_site}"] += transfer.bytes
    local_sum = sum(
        execution.service_ms
        for execution in executions
        if execution.site_id != "4090"
    )
    service_sum = sum(execution.service_ms for execution in executions)
    transfer_sum = run.transfers.latency_ms if run.transfers else 0.0
    return ReconstructedWorkflowMetrics(
        run_id=run.run_id,
        task_id=run.task_id,
        workflow_id=run.workflow_id,
        dependency_evidence="unavailable",
        timestamp_evidence="unavailable",
        trace_coverage="partial",
        artifact_lineage_availability="unavailable",
        invocation_count=sum(
            execution.operator_id == "invoke_model" for execution in executions
        ),
        action_count=len(executions),
        transfer_count=run.transfers.count if run.transfers else 0,
        dependencies=[],
        artifact_lineage=[],
        site_action_counts=dict(sorted(site_counts.items())),
        operator_counts=dict(sorted(operator_counts.items())),
        model_counts={
            "unknown_legacy_row": sum(
                execution.operator_id == "invoke_model" for execution in executions
            )
        },
        executor_action_counts=dict(sorted(executor_counts.items())),
        transfer_site_pair_bytes=dict(sorted(transfer_pairs.items())),
        raw_input_bytes=run.artifacts.raw_bytes if run.artifacts else 0,
        produced_bytes=sum(execution.output_bytes for execution in executions),
        transfer_bytes=run.transfers.bytes if run.transfers else 0,
        local_preprocessing_sum_ms=local_sum,
        local_preprocessing_critical_ms=None,
        transfer_sum_ms=transfer_sum,
        transfer_critical_ms=None,
        service_sum_ms=service_sum,
        service_critical_ms=None,
        local_service_sum_ms=local_sum,
        remote_service_sum_ms=service_sum - local_sum,
        planner_sum_ms=None,
        planner_critical_ms=None,
        critical_path_ms=None,
        critical_path_span_ids=[],
        e2e_latency_ms=run.e2e_latency_ms,
        input_tokens=run.usage.input_tokens if run.usage else 0,
        output_tokens=run.usage.output_tokens if run.usage else 0,
        quality=run.quality.score if run.quality else None,
        critical_path_availability="unavailable",
        critical_path_unavailable_reasons=[
            "runtime_row_has_no_realized_trace",
            "timestamp_evidence_unavailable",
            "dependency_evidence_unavailable",
            "artifact_lineage_unavailable",
        ],
    )


def build_critical_path_report(
    runs: list[CalibrationRun],
) -> CriticalPathReport:
    metrics = [
        reconstruct_calibration_run(run)
        for run in runs
        if run.status == "completed"
    ]
    return CriticalPathReport(
        metric_semantics=METRIC_SEMANTICS,
        runs=metrics,
        totals={
            "completed_run_count": len(metrics),
            "critical_path_available_count": sum(
                item.critical_path_availability == "available" for item in metrics
            ),
            "critical_path_unavailable_count": sum(
                item.critical_path_availability == "unavailable" for item in metrics
            ),
        },
    )


def build_trace_metrics_report(
    traces: list[RealizedWorkflowTrace],
) -> CriticalPathReport:
    """Build the same report for arbitrary future realized-workflow traces."""
    metrics = [reconstruct_workflow_metrics(trace) for trace in traces]
    return CriticalPathReport(
        metric_semantics=METRIC_SEMANTICS,
        runs=metrics,
        totals={
            "completed_run_count": len(metrics),
            "critical_path_available_count": sum(
                item.critical_path_availability == "available" for item in metrics
            ),
            "critical_path_unavailable_count": sum(
                item.critical_path_availability == "unavailable" for item in metrics
            ),
        },
    )
