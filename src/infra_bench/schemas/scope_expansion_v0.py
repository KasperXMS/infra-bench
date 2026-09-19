"""Contracts for the non-video ``Scope Expansion v0`` experiments.

The task bank and evaluator records are deliberately separate models.  Runtime
code may consume :class:`ScopeTask`, but gold answers and supporting-document
identifiers can only be represented by :class:`ScopeEvaluatorRecord`.
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .interaction import DEFAULT_OPERATOR_REGISTRY
from .measurement import RealizedWorkflowTrace

ScopeWorkflowId = Literal[
    "centralized_raw", "distributed_retrieval", "distributed_compute"
]
ScopeWorldId = Literal[
    "H1_distributed_constrained", "H2_distributed_favorable"
]
ScopeSiteId = Literal["A4", "A5", "A28"]
ScopeTaskFamily = Literal["multi_document_qa", "structured_data_analysis"]
ScopeEvaluatorType = Literal[
    "multihop_rag_official_token_intersection",
    "longbench_v2_official_multiple_choice",
]
ScopeOperatorId = Literal[
    "read_artifact",
    "bm25_retrieve",
    "aggregate_artifacts",
    "invoke_model",
    "filter_records",
    "select_fields",
    "derive_fields",
    "aggregate_records",
    "top_k_records",
]
ScopeSensitivityLabel = Literal[
    "workflow_reversal",
    "communication_sensitive",
    "compute_sensitive",
    "representation_sensitive",
    "quality_risk",
    "parallelism_sensitive",
    "context_cost_sensitive",
]

SITE_BY_SHARD: tuple[ScopeSiteId, ...] = ("A4", "A5", "A28")
FORMAL_SCOPE_PROTOCOL_ID = "steady_state_1_warmup_3_measured_v1"


def stable_document_site(document_id: str) -> ScopeSiteId:
    """Map a document to a site without Python's process-randomized ``hash``."""
    digest = hashlib.sha256(document_id.encode("utf-8")).digest()
    return SITE_BY_SHARD[int.from_bytes(digest[:8], "big") % len(SITE_BY_SHARD)]


class ScopeRetrievalConfig(BaseModel):
    """The fixed, query-only candidate construction policy."""

    model_config = ConfigDict(extra="forbid")

    algorithm: Literal["BM25"] = "BM25"
    top_n: int = Field(default=30, gt=0)
    corpus_scope: Literal["full_corpus"] = "full_corpus"
    parameters: dict[str, float | int | str] = Field(default_factory=dict)


class ScopeStructuredPredicate(BaseModel):
    """One answer-independent, generic record filter."""

    model_config = ConfigDict(extra="forbid")

    field: str
    operator: Literal["eq", "ne", "lt", "le", "gt", "ge", "contains", "in"]
    value: str | int | float | bool | list[str | int | float | bool]


class ScopeStructuredDerivedField(BaseModel):
    """A generic deterministic derived field used before aggregation."""

    model_config = ConfigDict(extra="forbid")

    alias: str
    operator: Literal["add", "subtract", "multiply", "divide"]
    operands: list[str] = Field(min_length=2, max_length=2)


class ScopeStructuredAggregation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    function: Literal["count", "sum", "min", "max", "mean"]
    field: str | None = None
    alias: str


class ScopeStructuredOrder(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    direction: Literal["ascending", "descending"] = "ascending"


class ScopeStructuredPlan(BaseModel):
    """Planner-visible generic filter/project/aggregate configuration."""

    model_config = ConfigDict(extra="forbid")

    predicates: list[ScopeStructuredPredicate] = Field(default_factory=list)
    select_fields: list[str] = Field(default_factory=list)
    derived_fields: list[ScopeStructuredDerivedField] = Field(default_factory=list)
    aggregations: list[ScopeStructuredAggregation] = Field(default_factory=list)
    group_by: list[str] = Field(default_factory=list)
    order_by: list[ScopeStructuredOrder] = Field(default_factory=list)
    limit: int | None = Field(default=None, gt=0)


class ScopeProfilingProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["scope-expansion-v0-profiling-v1"] = (
        "scope-expansion-v0-profiling-v1"
    )
    protocol_id: Literal["steady_state_1_warmup_3_measured_v1"] = (
        "steady_state_1_warmup_3_measured_v1"
    )
    warmup_runs: Literal[1] = 1
    measured_runs: Literal[3] = 3
    estimator: Literal["median"] = "median"
    exclude_warmup_from_quality: Literal[True] = True
    exclude_warmup_from_metrics: Literal[True] = True
    exclude_warmup_from_preference: Literal[True] = True


class ScopeTask(BaseModel):
    """Planner/reference-visible task record; it cannot represent gold data."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["scope-expansion-v0-task-v1"] = (
        "scope-expansion-v0-task-v1"
    )
    task_id: str
    dataset: Literal["MultiHop-RAG", "LongBench-v2"]
    task_family: ScopeTaskFamily = "multi_document_qa"
    instruction: str
    query: str
    artifact_ids: list[str]
    answer_choices: dict[Literal["A", "B", "C", "D"], str] = Field(
        default_factory=dict
    )
    artifact_types: dict[str, Literal["document", "records"]]
    artifact_sizes: dict[str, int]
    artifact_tokens: dict[str, int]
    artifact_source_refs: dict[str, str]
    artifact_document_ids: dict[str, str]
    artifact_placement: dict[str, ScopeSiteId]
    retrieval_config: ScopeRetrievalConfig | None = None
    structured_plan: ScopeStructuredPlan | None = None
    placement_method: Literal["sha256_prefix_u64_mod_3"] = (
        "sha256_prefix_u64_mod_3"
    )
    agent_count: Literal[3] = 3
    artifact_count: int = Field(gt=0)
    evaluator_type: ScopeEvaluatorType
    quality_threshold: float = Field(default=1.0, ge=0.0, le=1.0)
    planner_visible: Literal[True] = True
    evaluator_only: Literal[False] = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_visible_task(self) -> ScopeTask:
        ids = self.artifact_ids
        if len(ids) != len(set(ids)):
            raise ValueError("artifact_ids must be unique")
        if self.artifact_count != len(ids):
            raise ValueError("artifact_count must equal len(artifact_ids)")
        if self.task_family == "multi_document_qa":
            if self.retrieval_config is None:
                raise ValueError("multi-document tasks require retrieval_config")
            if self.artifact_count != self.retrieval_config.top_n:
                raise ValueError("artifact_count must equal the BM25 top_n")
            if self.structured_plan is not None:
                raise ValueError("multi-document tasks cannot declare structured_plan")
            if set(self.artifact_types.values()) != {"document"}:
                raise ValueError("multi-document artifacts must be documents")
        else:
            if self.retrieval_config is not None:
                raise ValueError("structured tasks do not use document retrieval_config")
            if self.structured_plan is None:
                raise ValueError("structured tasks require structured_plan")
            if set(self.artifact_types.values()) != {"records"}:
                raise ValueError("structured task artifacts must contain records")
        if self.dataset == "MultiHop-RAG":
            if self.retrieval_config is None or self.retrieval_config.top_n not in (30, 50):
                raise ValueError("MultiHop-RAG requires the registered Top-30 or Top-50")
            if self.evaluator_type != "multihop_rag_official_token_intersection":
                raise ValueError("MultiHop-RAG must use its official evaluator")
            if self.answer_choices:
                raise ValueError("MultiHop-RAG does not expose multiple-choice options")
        else:
            if self.evaluator_type != "longbench_v2_official_multiple_choice":
                raise ValueError("LongBench-v2 must use its official evaluator")
            if set(self.answer_choices) != {"A", "B", "C", "D"}:
                raise ValueError("LongBench-v2 tasks require visible choices A-D")
        expected = set(ids)
        mirrors = {
            "artifact_types": set(self.artifact_types),
            "artifact_sizes": set(self.artifact_sizes),
            "artifact_tokens": set(self.artifact_tokens),
            "artifact_source_refs": set(self.artifact_source_refs),
            "artifact_document_ids": set(self.artifact_document_ids),
            "artifact_placement": set(self.artifact_placement),
        }
        for name, keys in mirrors.items():
            if keys != expected:
                raise ValueError(f"{name} keys must exactly match artifact_ids")
        if any(size <= 0 for size in self.artifact_sizes.values()):
            raise ValueError("artifact_sizes must be positive")
        if any(tokens < 0 for tokens in self.artifact_tokens.values()):
            raise ValueError("artifact_tokens must be non-negative")
        if len(set(self.artifact_document_ids.values())) != len(ids):
            raise ValueError("each artifact must represent a distinct document")
        for artifact_id in ids:
            document_id = self.artifact_document_ids[artifact_id]
            expected_site = stable_document_site(document_id)
            if self.artifact_placement[artifact_id] != expected_site:
                raise ValueError(
                    "artifact_placement must equal stable_hash(document_id) % 3"
                )
        hidden_fragments = ("answer", "gold", "supporting_document", "supporting_doc")
        exposed = [
            str(key)
            for key in self.metadata
            if any(fragment in str(key).casefold() for fragment in hidden_fragments)
        ]
        if exposed:
            raise ValueError(
                f"visible task metadata must not expose evaluator-only fields: {sorted(exposed)}"
            )
        return self


class ScopeEvaluatorRecord(BaseModel):
    """Gold record used only by offline selection, coverage checks, and evaluation."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["scope-expansion-v0-evaluator-v1"] = (
        "scope-expansion-v0-evaluator-v1"
    )
    task_id: str
    evaluator_type: ScopeEvaluatorType
    answer: str
    acceptable_answers: list[str] = Field(default_factory=list)
    supporting_document_ids: list[str] = Field(default_factory=list)
    evidence_document_count: int = Field(ge=0)
    question_type: str | None = None
    evaluator_config: dict[str, Any] = Field(default_factory=dict)
    original_evaluator_information: dict[str, Any] = Field(default_factory=dict)
    planner_visible: Literal[False] = False
    evaluator_only: Literal[True] = True

    @model_validator(mode="after")
    def validate_gold(self) -> ScopeEvaluatorRecord:
        if not self.answer.strip():
            raise ValueError("answer must not be blank")
        if len(self.supporting_document_ids) != len(
            set(self.supporting_document_ids)
        ):
            raise ValueError("supporting_document_ids must be unique")
        if len(self.supporting_document_ids) != self.evidence_document_count:
            raise ValueError(
                "evidence_document_count must equal supporting_document_ids length"
            )
        if any(not answer.strip() for answer in self.acceptable_answers):
            raise ValueError("acceptable_answers must not contain blank values")
        if self.evaluator_type == "multihop_rag_official_token_intersection" and not (
            2 <= self.evidence_document_count <= 4
        ):
            raise ValueError("MultiHop-RAG evidence count must be between 2 and 4")
        if self.evaluator_type == "longbench_v2_official_multiple_choice" and self.answer not in {
            "A",
            "B",
            "C",
            "D",
        }:
            raise ValueError("LongBench-v2 gold answer must be A, B, C, or D")
        return self


class ScopeTaskBankEvaluatorSummary(BaseModel):
    """Audit-only task-bank metadata; not sufficient to perform evaluation."""

    model_config = ConfigDict(extra="forbid")

    evidence_document_count: int = Field(ge=0)
    evaluator_type: ScopeEvaluatorType
    planner_visible: Literal[False] = False
    evaluator_only: Literal[True] = True


class ScopeTaskBankRecord(BaseModel):
    """Audit envelope keeping runtime input and gold-derived metadata separated."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["scope-expansion-v0-task-bank-v1"] = (
        "scope-expansion-v0-task-bank-v1"
    )
    task_id: str
    planner_visible: ScopeTask
    evaluator_only: ScopeTaskBankEvaluatorSummary

    @model_validator(mode="after")
    def validate_sections(self) -> ScopeTaskBankRecord:
        if self.task_id != self.planner_visible.task_id:
            raise ValueError("task-bank and planner-visible task IDs must match")
        if self.evaluator_only.evaluator_type != self.planner_visible.evaluator_type:
            raise ValueError("task-bank evaluator types must match")
        return self


class ScopeNetwork(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bandwidth_mbps: float = Field(gt=0.0)
    rtt_ms: float = Field(ge=0.0)
    measured_effective_bandwidth_mbps: float | None = Field(default=None, gt=0.0)


class ScopeWorld(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["scope-expansion-v0-world-v1"] = (
        "scope-expansion-v0-world-v1"
    )
    world_id: ScopeWorldId
    network: ScopeNetwork
    sites: list[ScopeSiteId] = Field(default_factory=lambda: list(SITE_BY_SHARD))
    reasoning_site: Literal["A28"] = "A28"
    final_model_id: str
    device_fingerprints: dict[ScopeSiteId, str]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_world(self) -> ScopeWorld:
        if self.sites != list(SITE_BY_SHARD):
            raise ValueError("sites must be ordered A4, A5, A28")
        if set(self.device_fingerprints) != set(SITE_BY_SHARD):
            raise ValueError("device_fingerprints must identify A4, A5, and A28")
        return self


class ScopeWorkflowStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str
    operator_id: ScopeOperatorId
    execution_role: Literal["artifact_local_agent", "final_reasoning_agent"]
    input_kinds: list[str] = Field(default_factory=list)
    output_kinds: list[str] = Field(default_factory=list)


class ScopeWorkflow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["scope-expansion-v0-workflow-v1"] = (
        "scope-expansion-v0-workflow-v1"
    )
    workflow_id: ScopeWorkflowId
    description: str
    steps: list[ScopeWorkflowStep]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_workflow(self) -> ScopeWorkflow:
        DEFAULT_OPERATOR_REGISTRY.require(
            [step.operator_id for step in self.steps]
        )
        step_ids = [step.step_id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("workflow step IDs must be unique")
        if not self.steps or self.steps[-1].operator_id != "invoke_model":
            raise ValueError("reference workflows must finish with invoke_model")
        if self.steps[-1].execution_role != "final_reasoning_agent":
            raise ValueError("final reasoning must run at the final reasoning agent")
        local_retrieval = [
            step
            for step in self.steps
            if step.operator_id == "bm25_retrieve"
            and step.execution_role == "artifact_local_agent"
        ]
        if self.workflow_id == "centralized_raw" and local_retrieval:
            raise ValueError("centralized_raw must not perform local retrieval")
        if self.workflow_id == "distributed_retrieval" and not local_retrieval:
            raise ValueError("distributed_retrieval requires artifact-local retrieval")
        local_compute = [
            step
            for step in self.steps
            if step.operator_id
            in {
                "filter_records",
                "select_fields",
                "derive_fields",
                "aggregate_records",
                "top_k_records",
            }
            and step.execution_role == "artifact_local_agent"
        ]
        if self.workflow_id == "centralized_raw" and local_compute:
            raise ValueError("centralized_raw must not perform local structured compute")
        if self.workflow_id == "distributed_compute" and not local_compute:
            raise ValueError("distributed_compute requires artifact-local structured compute")
        return self


class ScopeQuality(BaseModel):
    model_config = ConfigDict(extra="forbid")

    original_benchmark_score: float = Field(ge=0.0, le=1.0)
    exact_match: bool | None = None
    canonical_answer_correct: bool
    passed: bool
    evaluator_type: ScopeEvaluatorType
    details: dict[str, Any] = Field(default_factory=dict)


class ScopeDemandMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_count: int = Field(gt=0)
    raw_bytes: int = Field(gt=0)
    raw_tokens: int = Field(ge=0)
    documents_per_agent: dict[ScopeSiteId, int]

    @model_validator(mode="after")
    def validate_document_counts(self) -> ScopeDemandMetrics:
        if set(self.documents_per_agent) != set(SITE_BY_SHARD):
            raise ValueError("documents_per_agent must contain A4, A5, and A28")
        if any(value < 0 for value in self.documents_per_agent.values()):
            raise ValueError("documents_per_agent values must be non-negative")
        if sum(self.documents_per_agent.values()) != self.artifact_count:
            raise ValueError("documents_per_agent must sum to artifact_count")
        return self


class ScopeRetrievalMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    algorithm: Literal["bm25"] = "bm25"
    top_k_per_shard: int = Field(default=0, ge=0)
    calls: int = Field(ge=0)
    sum_ms: float = Field(ge=0.0)
    critical_ms: float | None = Field(default=None, ge=0.0)
    retrieved_document_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_critical_time(self) -> ScopeRetrievalMetrics:
        if self.critical_ms is not None and self.critical_ms > self.sum_ms + 1e-3:
            raise ValueError("retrieval critical_ms cannot exceed sum_ms")
        if self.calls == 0 and (self.sum_ms > 0 or self.retrieved_document_count > 0):
            raise ValueError("zero retrieval calls cannot report retrieval work")
        if self.calls > 0 and self.top_k_per_shard <= 0:
            raise ValueError("retrieval calls require positive top_k_per_shard")
        return self


class ScopeLocalComputeMetrics(BaseModel):
    """Generic artifact-local work for LongBench retrieval or record processing."""

    model_config = ConfigDict(extra="forbid")

    calls: int = Field(ge=0)
    sum_ms: float = Field(ge=0.0)
    critical_ms: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def validate_local_compute(self) -> ScopeLocalComputeMetrics:
        if self.critical_ms is not None and self.critical_ms > self.sum_ms + 1e-3:
            raise ValueError("local compute critical_ms cannot exceed sum_ms")
        if self.calls == 0 and self.sum_ms > 0:
            raise ValueError("zero local calls cannot report local compute time")
        return self


class ScopeArtifactMeasurement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    document_id: str | None = None
    kind: Literal[
        "raw_document",
        "raw_records",
        "retrieved_evidence",
        "filtered_records",
        "partial_aggregate",
        "aggregated_evidence",
    ]
    site_id: ScopeSiteId
    bytes: int = Field(ge=0)
    tokens: int = Field(ge=0)


class ScopeArtifactMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_bytes: int = Field(gt=0)
    reduced_bytes: int = Field(ge=0)
    reduced_tokens: int = Field(ge=0)
    absolute_reducible_bytes: int = Field(ge=0)
    records: list[ScopeArtifactMeasurement] = Field(
        default_factory=list[ScopeArtifactMeasurement]
    )

    @model_validator(mode="after")
    def validate_reduction(self) -> ScopeArtifactMetrics:
        if self.reduced_bytes > self.raw_bytes:
            raise ValueError("reduced_bytes cannot exceed raw_bytes")
        if self.absolute_reducible_bytes != self.raw_bytes - self.reduced_bytes:
            raise ValueError("absolute_reducible_bytes must equal raw_bytes - reduced_bytes")
        return self


class ScopeTransferRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transfer_id: str
    artifact_id: str
    src_agent: str
    dst_agent: str
    src_site: ScopeSiteId
    dst_site: ScopeSiteId
    bytes: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)
    started_at: str | float | None = None
    finished_at: str | float | None = None
    depends_on: list[str] = Field(default_factory=list)
    producer_action_id: str | None = None
    consumer_action_id: str | None = None


class ScopeTransferMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int = Field(ge=0)
    bytes: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)
    critical_ms: float | None = Field(default=None, ge=0.0)
    records: list[ScopeTransferRecord] = Field(
        default_factory=list[ScopeTransferRecord]
    )

    @model_validator(mode="after")
    def validate_totals(self) -> ScopeTransferMetrics:
        if self.critical_ms is not None and self.critical_ms > self.latency_ms + 1e-3:
            raise ValueError("transfer critical_ms cannot exceed latency_ms")
        if self.records:
            if self.count != len(self.records):
                raise ValueError("transfer count does not match records")
            if self.bytes != sum(record.bytes for record in self.records):
                raise ValueError("transfer bytes do not match records")
            if abs(self.latency_ms - sum(record.latency_ms for record in self.records)) > 1e-3:
                raise ValueError("transfer latency_ms does not match record sum")
        return self


class ScopeServiceMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    local_preprocessing_ms: float = Field(ge=0.0)
    final_model_ms: float = Field(ge=0.0)
    total_ms: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_total(self) -> ScopeServiceMetrics:
        if self.total_ms + 1e-3 < self.local_preprocessing_ms + self.final_model_ms:
            raise ValueError("service total_ms cannot exclude local or final-model work")
        return self


class ScopeUsageMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    api_cost_usd: float | None = Field(default=None, ge=0.0)


class ScopeExecution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: str
    operator_id: ScopeOperatorId
    executor_id: str
    worker_id: str
    site_id: ScopeSiteId
    service_ms: float = Field(ge=0.0)
    input_bytes: int = Field(ge=0)
    output_bytes: int = Field(ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    input_artifacts: list[str] = Field(default_factory=list)
    output_artifacts: list[str] = Field(default_factory=list)
    started_at: str | float | None = None
    finished_at: str | float | None = None
    depends_on: list[str] = Field(default_factory=list)
    model_id: str | None = None


class ScopeLineageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    producer_agent: str
    consumer_agent: str
    input_artifact: str
    derived_artifact: str
    artifact_bytes: int = Field(ge=0)
    operator: ScopeOperatorId


class ScopeRun(BaseModel):
    """One raw runtime row; evaluator-derived quality may initially be null."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["scope-expansion-v0-run-v1"] = (
        "scope-expansion-v0-run-v1"
    )
    run_id: str
    task_id: str
    task_family: ScopeTaskFamily = "multi_document_qa"
    workflow_id: ScopeWorkflowId
    world_id: ScopeWorldId
    repeat: int = Field(ge=0)
    warmup: bool = False
    measurement_series_id: str | None = None
    protocol_id: str | None = None
    status: Literal["completed", "failed"]
    quality: ScopeQuality | None = None
    final_answer: str | None = None
    demand: ScopeDemandMetrics | None = None
    retrieval: ScopeRetrievalMetrics | None = None
    local_compute: ScopeLocalComputeMetrics | None = None
    artifacts: ScopeArtifactMetrics | None = None
    transfers: ScopeTransferMetrics | None = None
    service: ScopeServiceMetrics | None = None
    usage: ScopeUsageMetrics | None = None
    e2e_latency_ms: float | None = Field(default=None, ge=0.0)
    executions: list[ScopeExecution] = Field(default_factory=list[ScopeExecution])
    lineage: list[ScopeLineageRecord] = Field(
        default_factory=list[ScopeLineageRecord]
    )
    trace: RealizedWorkflowTrace | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_run(self) -> ScopeRun:
        if self.warmup and self.repeat != 0:
            raise ValueError("warm-up runs must use repeat=0")
        if not self.warmup and self.repeat < 1:
            raise ValueError("measured runs must use repeat>=1")
        measured = (
            self.demand,
            self.artifacts,
            self.transfers,
            self.service,
            self.usage,
            self.e2e_latency_ms,
        )
        if self.status == "completed" and any(value is None for value in measured):
            raise ValueError("completed scope runs require all metric groups and e2e")
        if self.status == "completed" and not self.executions:
            raise ValueError("completed scope runs require executor/site measurements")
        if self.status == "completed" and not self.lineage:
            raise ValueError("completed scope runs require artifact lineage")
        if self.status == "completed" and self.final_answer is None:
            raise ValueError("completed scope runs require final_answer")
        if self.status == "failed" and not self.error:
            raise ValueError("failed scope runs require an error")
        if self.status == "completed":
            assert self.demand is not None
            assert self.artifacts is not None
            assert self.service is not None
            if self.demand.raw_bytes != self.artifacts.raw_bytes:
                raise ValueError("demand and artifact raw_bytes must agree")
            if self.workflow_id == "centralized_raw":
                if (
                    (self.retrieval is not None and self.retrieval.calls)
                    or (self.local_compute is not None and self.local_compute.calls)
                    or self.service.local_preprocessing_ms > 0
                ):
                    raise ValueError("centralized_raw cannot include local preprocessing")
            elif self.service.local_preprocessing_ms <= 0:
                raise ValueError(
                    "distributed workflows must measure non-zero local preprocessing"
                )
            if self.workflow_id == "distributed_retrieval" and (
                self.retrieval is None or self.retrieval.calls == 0
            ):
                raise ValueError("distributed_retrieval requires retrieval metrics")
            if self.workflow_id == "distributed_compute" and (
                self.local_compute is None or self.local_compute.calls == 0
            ):
                raise ValueError("distributed_compute requires local_compute metrics")
        if self.trace is not None:
            if self.trace.run_id != self.run_id:
                raise ValueError("trace run_id must match the scope run")
            if self.trace.task_id != self.task_id:
                raise ValueError("trace task_id must match the scope run")
            if self.trace.workflow_id != self.workflow_id:
                raise ValueError("trace workflow_id must match the scope run")
            if self.trace.warmup != self.warmup:
                raise ValueError("trace warmup must match the scope run")
        return self


class ScopeCellSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    workflow_id: ScopeWorkflowId
    world_id: ScopeWorldId
    selected_measurement_series_id: str | None = None
    selected_protocol_id: str | None = None
    completed_warmups: int = Field(ge=0)
    completed_runs: int = Field(ge=0)
    profiling_protocol_passed: bool
    quality_gate_passed: bool
    quality_pass_rate: float = Field(ge=0.0, le=1.0)
    quality_median: float | None = Field(default=None, ge=0.0, le=1.0)
    medians: dict[str, float | int | None]


class ScopeTaskSensitivity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    labels: list[ScopeSensitivityLabel]
    workflow_preference: dict[ScopeWorldId, ScopeWorkflowId | None]
    quality_by_workflow: dict[ScopeWorkflowId, float | None]
    evidence: dict[ScopeSensitivityLabel, dict[str, Any]]


class ScopeSensitivityReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["scope-expansion-v0-sensitivity-report-v1"] = (
        "scope-expansion-v0-sensitivity-report-v1"
    )
    protocol: dict[str, Any]
    metric_definitions: dict[str, str]
    thresholds: dict[str, float]
    cells: list[ScopeCellSummary]
    tasks: list[ScopeTaskSensitivity]
    totals: dict[str, int]
