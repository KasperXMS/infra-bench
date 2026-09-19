"""Generic realized-workflow trace and measurement reconstruction contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

TraceEvidence = Literal["complete", "partial", "unavailable"]
TraceSpanKind = Literal["planner", "action", "transfer"]
ServiceScope = Literal["planner", "local", "remote", "other"]


class WorkflowTraceSpan(BaseModel):
    """One timestamped planner, execution, or transfer interval."""

    model_config = ConfigDict(extra="forbid")

    span_id: str
    span_kind: TraceSpanKind
    name: str
    started_at: str | float | None = None
    finished_at: str | float | None = None
    duration_ms: float | None = Field(default=None, ge=0.0)
    depends_on: list[str] = Field(default_factory=list)
    service_scope: ServiceScope
    operator_id: str | None = None
    model_id: str | None = None
    executor_id: str | None = None
    site_id: str | None = None
    input_artifacts: list[str] = Field(default_factory=list)
    output_artifacts: list[str] = Field(default_factory=list)
    artifact_id: str | None = None
    producer_span_id: str | None = None
    consumer_span_id: str | None = None
    src_site: str | None = None
    dst_site: str | None = None
    bytes: int = Field(default=0, ge=0)
    input_bytes: int = Field(default=0, ge=0)
    output_bytes: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_transfer_fields(self) -> WorkflowTraceSpan:
        if self.span_kind == "transfer" and not self.artifact_id:
            raise ValueError("transfer spans require artifact_id")
        return self


class RealizedWorkflowTrace(BaseModel):
    """Trace for any realized workflow, independent of planner/workflow vocabulary."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["realized-workflow-trace-v1"] = (
        "realized-workflow-trace-v1"
    )
    run_id: str
    task_id: str
    workflow_id: str
    warmup: bool = False
    dependency_evidence: TraceEvidence
    timestamp_evidence: TraceEvidence
    trace_coverage: TraceEvidence
    spans: list[WorkflowTraceSpan]
    e2e_latency_ms: float | None = Field(default=None, ge=0.0)
    quality: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_references(self) -> RealizedWorkflowTrace:
        span_ids = [span.span_id for span in self.spans]
        if len(span_ids) != len(set(span_ids)):
            raise ValueError("trace span IDs must be unique")
        known = set(span_ids)
        for span in self.spans:
            unknown = set(span.depends_on) - known
            if unknown:
                raise ValueError(
                    f"span {span.span_id!r} has unknown dependencies: {sorted(unknown)}"
                )
            for reference in (span.producer_span_id, span.consumer_span_id):
                if reference is not None and reference not in known:
                    raise ValueError(
                        f"span {span.span_id!r} references unknown span {reference!r}"
                    )
        return self


class ArtifactLineageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    producer_span_id: str | None = None
    consumer_span_ids: list[str] = Field(default_factory=list)
    transfer_span_ids: list[str] = Field(default_factory=list)
    bytes: int = Field(ge=0)
    source_site: str | None = None
    destination_sites: list[str] = Field(default_factory=list)


class ReconstructedWorkflowMetrics(BaseModel):
    """Sums are work totals; critical values exist only with complete trace evidence."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["reconstructed-workflow-metrics-v1"] = (
        "reconstructed-workflow-metrics-v1"
    )
    run_id: str
    task_id: str
    workflow_id: str
    warmup: bool = False
    dependency_evidence: TraceEvidence
    timestamp_evidence: TraceEvidence
    trace_coverage: TraceEvidence
    artifact_lineage_availability: Literal["available", "unavailable"]
    invocation_count: int = Field(ge=0)
    action_count: int = Field(ge=0)
    transfer_count: int = Field(ge=0)
    dependencies: list[tuple[str, str]]
    artifact_lineage: list[ArtifactLineageRecord]
    site_action_counts: dict[str, int]
    operator_counts: dict[str, int]
    model_counts: dict[str, int]
    executor_action_counts: dict[str, int]
    transfer_site_pair_bytes: dict[str, int]
    raw_input_bytes: int = Field(ge=0)
    produced_bytes: int = Field(ge=0)
    transfer_bytes: int = Field(ge=0)
    local_preprocessing_sum_ms: float = Field(ge=0.0)
    local_preprocessing_critical_ms: float | None = Field(default=None, ge=0.0)
    transfer_sum_ms: float = Field(ge=0.0)
    transfer_critical_ms: float | None = Field(default=None, ge=0.0)
    service_sum_ms: float = Field(ge=0.0)
    service_critical_ms: float | None = Field(default=None, ge=0.0)
    local_service_sum_ms: float = Field(ge=0.0)
    remote_service_sum_ms: float = Field(ge=0.0)
    planner_sum_ms: float | None = Field(default=None, ge=0.0)
    planner_critical_ms: float | None = Field(default=None, ge=0.0)
    critical_path_ms: float | None = Field(default=None, ge=0.0)
    critical_path_span_ids: list[str] = Field(default_factory=list)
    e2e_latency_ms: float | None = Field(default=None, ge=0.0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    quality: float | None = Field(default=None, ge=0.0, le=1.0)
    critical_path_availability: Literal["available", "unavailable"]
    critical_path_unavailable_reasons: list[str] = Field(default_factory=list)


class CriticalPathReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["critical-path-report-v1"] = "critical-path-report-v1"
    metric_semantics: dict[str, str]
    runs: list[ReconstructedWorkflowMetrics]
    totals: dict[str, int]
