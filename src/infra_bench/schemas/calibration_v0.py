"""Versioned contracts for the small real-system ``calibration_v0`` experiment."""

from __future__ import annotations

from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .measurement import RealizedWorkflowTrace

CalibrationWorkflowId = Literal[
    "centralized_raw", "local_reduction", "visual_reduction"
]
CalibrationWorldId = Literal[
    "H1_distributed_constrained", "H2_distributed_favorable"
]
CalibrationSiteId = Literal["A4", "A5", "A28", "4090"]
CalibrationOperatorId = Literal[
    "read_artifact",
    "sample_frames",
    "invoke_model",
    "make_contact_sheet",
    "extract_clip",
    "process_local_artifact",
    "aggregate_artifacts",
]
AnswerOption = Literal["A", "B", "C", "D"]


class CalibrationProfilingProtocol(BaseModel):
    """Fixed steady-state serving protocol used for every calibration cell."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["calibration-v0-profiling-v1"] = (
        "calibration-v0-profiling-v1"
    )
    warmup_runs: Literal[1] = 1
    measured_runs: Literal[3] = 3
    estimator: Literal["median"] = "median"
    serving_mode: Literal["steady_state"] = "steady_state"
    exclude_warmup_from_quality: Literal[True] = True
    exclude_warmup_from_metrics: Literal[True] = True
    exclude_warmup_from_admission: Literal[True] = True


class CalibrationQuestion(BaseModel):
    """Planner-visible multiple-choice question with no answer field."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    prompt: str
    options: dict[AnswerOption, str]

    @model_validator(mode="after")
    def validate_options(self) -> CalibrationQuestion:
        if set(self.options) != {"A", "B", "C", "D"}:
            raise ValueError("each calibration question must have options A, B, C, D")
        return self


class CalibrationChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    chunk_index: int = Field(ge=0, le=2)
    start_s: float = Field(ge=0.0)
    end_s: float = Field(gt=0.0)
    duration_s: float = Field(gt=0.0)
    observed_duration_s: float | None = Field(default=None, gt=0.0)
    source_ref: str
    site_id: Literal["A4", "A5", "A28"]
    raw_bytes: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_interval(self) -> CalibrationChunk:
        if self.end_s <= self.start_s:
            raise ValueError("chunk end_s must be greater than start_s")
        if abs(self.duration_s - (self.end_s - self.start_s)) > 1e-3:
            raise ValueError("chunk duration_s must match end_s - start_s")
        return self


class CalibrationTask(BaseModel):
    """A long-video task split independently of any gold evidence."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["calibration-v0-task-v1"] = "calibration-v0-task-v1"
    task_id: str
    benchmark: Literal["Video-MME", "Video-MME-v2", "MLVU"]
    instruction: str
    duration_s: float = Field(ge=1800.0)
    evaluator_id: str
    question_count: Literal[3] = 3
    sample_count_per_chunk: int = Field(default=12, gt=0)
    questions: list[CalibrationQuestion]
    quality_metric: Literal["accuracy"] = "accuracy"
    quality_threshold: float = Field(default=2.0 / 3.0, ge=0.0, le=1.0)
    segmentation: Literal["fixed_uniform_3"] = "fixed_uniform_3"
    chunks: list[CalibrationChunk]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_fixed_segmentation(self) -> CalibrationTask:
        hidden_keys = {"answer", "answers", "gold", "gold_answer", "correct_answer"}
        exposed = hidden_keys & {key.lower() for key in self.metadata}
        if exposed:
            raise ValueError(
                f"task metadata must not expose evaluator answers: {sorted(exposed)}"
            )
        if len(self.questions) != self.question_count:
            raise ValueError("calibration task must contain exactly three questions")
        question_ids = [question.question_id for question in self.questions]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("calibration question IDs must be unique")
        if len(self.chunks) != 3:
            raise ValueError("calibration_v0 tasks require exactly three chunks")
        chunks = sorted(self.chunks, key=lambda item: item.chunk_index)
        if [item.chunk_index for item in chunks] != [0, 1, 2]:
            raise ValueError("chunk_index values must be exactly 0, 1, 2")
        if [item.site_id for item in chunks] != ["A4", "A5", "A28"]:
            raise ValueError("chunks 0, 1, 2 must be placed on A4, A5, A28")
        if chunks[0].start_s != 0.0:
            raise ValueError("fixed segmentation must begin at 0 seconds")
        for previous, current in zip(chunks, chunks[1:]):
            if abs(previous.end_s - current.start_s) > 1e-3:
                raise ValueError("fixed chunks must be contiguous")
        if abs(chunks[-1].end_s - self.duration_s) > 1e-3:
            raise ValueError("fixed chunks must cover the complete video")
        lengths = [item.end_s - item.start_s for item in chunks]
        if max(lengths) - min(lengths) > 1.0:
            raise ValueError("fixed_uniform_3 chunk durations may differ by at most one second")
        return self


class CalibrationAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    correct_option: AnswerOption


class CalibrationEvaluatorRecord(BaseModel):
    """Evaluator-only answer key; never part of the runtime task payload."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["calibration-v0-evaluator-v1"] = (
        "calibration-v0-evaluator-v1"
    )
    task_id: str
    evaluator_id: str
    answers: list[CalibrationAnswer]

    @model_validator(mode="after")
    def validate_answers(self) -> CalibrationEvaluatorRecord:
        if len(self.answers) != 3:
            raise ValueError("calibration evaluator requires exactly three answers")
        question_ids = [answer.question_id for answer in self.answers]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("calibration evaluator question IDs must be unique")
        return self


class CalibrationNetwork(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bandwidth_mbps: float = Field(gt=0.0)
    rtt_ms: float = Field(ge=0.0)


class CalibrationWorld(BaseModel):
    """A controlled world. Hardware/model identity is explicit for pair validation."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["calibration-v0-world-v1"] = "calibration-v0-world-v1"
    world_id: CalibrationWorldId
    network: CalibrationNetwork
    local_sites: list[Literal["A4", "A5", "A28"]] = Field(
        default_factory=lambda: ["A4", "A5", "A28"]
    )
    reasoning_site: Literal["4090"] = "4090"
    local_model_id: str
    strong_model_id: str
    device_fingerprints: dict[CalibrationSiteId, str]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_topology(self) -> CalibrationWorld:
        if self.local_sites != ["A4", "A5", "A28"]:
            raise ValueError("local_sites must be ordered A4, A5, A28")
        if set(self.device_fingerprints) != {"A4", "A5", "A28", "4090"}:
            raise ValueError("device_fingerprints must identify A4, A5, A28, and 4090")
        return self


class CalibrationWorkflowStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str
    operator_id: CalibrationOperatorId
    execution_role: Literal["artifact_local_agent", "reasoning_4090"]
    input_kinds: list[str] = Field(default_factory=list)
    output_kinds: list[str] = Field(default_factory=list)


class CalibrationWorkflow(BaseModel):
    """Reference workflow composed only from generic registry operators."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["calibration-v0-workflow-v1"] = (
        "calibration-v0-workflow-v1"
    )
    workflow_id: CalibrationWorkflowId
    description: str
    steps: list[CalibrationWorkflowStep]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_reference_shape(self) -> CalibrationWorkflow:
        step_ids = [step.step_id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("workflow step IDs must be unique")
        if not self.steps or self.steps[-1].operator_id != "invoke_model":
            raise ValueError("reference workflows must finish with invoke_model")
        if self.steps[-1].execution_role != "reasoning_4090":
            raise ValueError("final reasoning must execute on 4090")
        local_steps = [s for s in self.steps if s.execution_role == "artifact_local_agent"]
        if self.workflow_id == "centralized_raw" and local_steps:
            raise ValueError("centralized_raw may not perform local preprocessing")
        if self.workflow_id == "local_reduction":
            if not local_steps or any(
                step.operator_id not in {"sample_frames", "invoke_model"}
                for step in local_steps
            ):
                raise ValueError(
                    "local_reduction requires generic sample_frames or invoke_model local steps"
                )
        if self.workflow_id == "visual_reduction":
            if not any(
                step.operator_id == "sample_frames"
                and step.execution_role == "artifact_local_agent"
                for step in self.steps
            ):
                raise ValueError(
                    "visual_reduction requires artifact-local generic sample_frames"
                )
            if any(
                step.operator_id not in {"sample_frames", "make_contact_sheet"}
                for step in local_steps
            ):
                raise ValueError(
                    "visual_reduction local steps may only sample frames or make a contact sheet"
                )
        return self


class CalibrationQuality(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    evaluator_id: str
    details: dict[str, Any] = Field(default_factory=dict)


class CalibrationArtifactMeasurement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    kind: Literal[
        "raw_video_chunk",
        "reduced_clip",
        "sampled_frames",
        "semantic_evidence",
        "contact_sheet",
        "visual_evidence",
        "aggregated_visual_evidence",
    ]
    site_id: CalibrationSiteId
    bytes: int = Field(ge=0)


class CalibrationArtifactMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_bytes: int = Field(ge=0)
    reduced_bytes: int = Field(ge=0)
    reduced_visual_bytes: int | None = Field(default=None, ge=0)
    semantic_evidence_bytes: int | None = Field(default=None, ge=0)
    reduction_ratio: float | None = Field(default=None, ge=0.0)
    records: list[CalibrationArtifactMeasurement] = Field(
        default_factory=list[CalibrationArtifactMeasurement]
    )

    @model_validator(mode="before")
    @classmethod
    def upgrade_legacy_reduced_byte_categories(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        raw = cast(dict[object, object], value)
        upgraded: dict[str, Any] = {str(key): item for key, item in raw.items()}
        reduced = int(upgraded.get("reduced_bytes", 0))
        visual = upgraded.get("reduced_visual_bytes")
        semantic = upgraded.get("semantic_evidence_bytes")
        if visual is None and semantic is None:
            # Historical calibration rows only contained semantic evidence.
            upgraded["reduced_visual_bytes"] = 0
            upgraded["semantic_evidence_bytes"] = reduced
        elif visual is None:
            upgraded["reduced_visual_bytes"] = reduced - int(cast(int, semantic))
        elif semantic is None:
            upgraded["semantic_evidence_bytes"] = reduced - int(cast(int, visual))
        return upgraded

    @model_validator(mode="after")
    def validate_ratio(self) -> CalibrationArtifactMetrics:
        assert self.reduced_visual_bytes is not None
        assert self.semantic_evidence_bytes is not None
        if self.reduced_visual_bytes + self.semantic_evidence_bytes != self.reduced_bytes:
            raise ValueError(
                "reduced_bytes must equal reduced_visual_bytes + semantic_evidence_bytes"
            )
        expected = self.reduced_bytes / self.raw_bytes if self.raw_bytes else None
        if self.reduction_ratio is not None and (
            expected is None or abs(self.reduction_ratio - expected) > 1e-6
        ):
            raise ValueError("reduction_ratio must equal reduced_bytes / raw_bytes")
        return self


class CalibrationTransfer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transfer_id: str
    artifact_id: str
    src_site: CalibrationSiteId
    dst_site: CalibrationSiteId
    bytes: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)
    started_at: str | float | None = None
    finished_at: str | float | None = None
    depends_on: list[str] = Field(default_factory=list)
    producer_action_id: str | None = None
    consumer_action_id: str | None = None


class CalibrationTransferMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int = Field(ge=0)
    bytes: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)
    records: list[CalibrationTransfer] = Field(default_factory=list[CalibrationTransfer])

    @model_validator(mode="after")
    def validate_totals(self) -> CalibrationTransferMetrics:
        if self.records:
            if self.count != len(self.records):
                raise ValueError("transfer count does not match records")
            if self.bytes != sum(item.bytes for item in self.records):
                raise ValueError("transfer bytes do not match records")
            measured_ms = sum(item.latency_ms for item in self.records)
            if abs(self.latency_ms - measured_ms) > 1e-3:
                raise ValueError("transfer latency_ms does not match records")
        return self


class CalibrationServiceMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    local_preprocessing_ms: float = Field(ge=0.0)
    remote_model_ms: float = Field(ge=0.0)
    total_ms: float = Field(ge=0.0)

    @model_validator(mode="after")
    def local_work_is_not_free(self) -> CalibrationServiceMetrics:
        if self.total_ms + 1e-3 < self.local_preprocessing_ms + self.remote_model_ms:
            raise ValueError("service total_ms cannot exclude local or remote service time")
        return self


class CalibrationUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    api_cost_usd: float | None = Field(default=None, ge=0.0)


class CalibrationExecution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: str
    operator_id: CalibrationOperatorId
    executor_id: str
    worker_id: str
    site_id: CalibrationSiteId
    service_ms: float = Field(ge=0.0)
    input_bytes: int = Field(ge=0)
    output_bytes: int = Field(ge=0)
    started_at: str | float | None = None
    finished_at: str | float | None = None
    depends_on: list[str] = Field(default_factory=list)
    input_artifacts: list[str] = Field(default_factory=list)
    output_artifacts: list[str] = Field(default_factory=list)
    model_id: str | None = None
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


class CalibrationRun(BaseModel):
    """One measured attempt; failed attempts remain valid first-class records."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["calibration-v0-run-v1"] = "calibration-v0-run-v1"
    run_id: str
    task_id: str
    workflow_id: CalibrationWorkflowId
    world_id: CalibrationWorldId
    repeat: int = Field(ge=0)
    warmup: bool = False
    measurement_series_id: str | None = None
    protocol_id: str | None = None
    status: Literal["completed", "failed"]
    quality: CalibrationQuality | None = None
    artifacts: CalibrationArtifactMetrics | None = None
    transfers: CalibrationTransferMetrics | None = None
    service: CalibrationServiceMetrics | None = None
    usage: CalibrationUsage | None = None
    e2e_latency_ms: float | None = Field(default=None, ge=0.0)
    executions: list[CalibrationExecution] = Field(
        default_factory=list[CalibrationExecution]
    )
    final_answer: str | None = None
    answer: str | None = None
    error: str | None = None
    trace: RealizedWorkflowTrace | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_status_payload(self) -> CalibrationRun:
        metadata_series = self.metadata.get("measurement_series_id")
        metadata_protocol = self.metadata.get("measurement_protocol")
        if (
            self.measurement_series_id is not None
            and metadata_series is not None
            and self.measurement_series_id != metadata_series
        ):
            raise ValueError("top-level and metadata measurement_series_id disagree")
        if (
            self.protocol_id is not None
            and metadata_protocol is not None
            and self.protocol_id != metadata_protocol
        ):
            raise ValueError("top-level protocol_id and metadata measurement_protocol disagree")
        if self.warmup and self.repeat != 0:
            raise ValueError("warm-up runs must use repeat=0")
        if not self.warmup and self.repeat < 1:
            raise ValueError("measured runs must use repeat>=1")
        measured = (
            self.artifacts,
            self.transfers,
            self.service,
            self.usage,
            self.e2e_latency_ms,
        )
        if self.status == "completed" and any(item is None for item in measured):
            raise ValueError("completed calibration runs require all metric groups and e2e")
        if self.status == "completed" and not self.executions:
            raise ValueError("completed calibration runs require executor/site measurements")
        if self.status == "failed" and not self.error:
            raise ValueError("failed calibration runs require an error")
        if self.workflow_id in {"local_reduction", "visual_reduction"} and self.status == "completed":
            assert self.service is not None
            assert self.artifacts is not None
            if self.service.local_preprocessing_ms <= 0.0:
                raise ValueError(
                    f"{self.workflow_id} must measure non-zero local preprocessing"
                )
            sites = {execution.site_id for execution in self.executions}
            if "4090" not in sites or not sites.intersection({"A4", "A5", "A28"}):
                raise ValueError(
                    f"{self.workflow_id} executions must include a local Orin and 4090"
                )
            if (
                self.workflow_id == "local_reduction"
                and not self.artifacts.semantic_evidence_bytes
            ):
                raise ValueError(
                    "local_reduction must report non-zero semantic_evidence_bytes"
                )
            if (
                self.workflow_id == "visual_reduction"
                and not self.artifacts.reduced_visual_bytes
            ):
                raise ValueError(
                    "visual_reduction must report non-zero reduced_visual_bytes"
                )
        return self


class CalibrationCellResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    workflow_id: CalibrationWorkflowId
    world_id: CalibrationWorldId
    selected_measurement_series_id: str | None = None
    selected_protocol_id: str | None = None
    attempted_runs: int = Field(ge=0)
    warmup_attempts: int = Field(ge=0)
    completed_warmups: int = Field(ge=0)
    measured_attempts: int = Field(ge=0)
    completed_runs: int = Field(ge=0)
    profiling_protocol_passed: bool
    quality_pass_rate: float = Field(ge=0.0, le=1.0)
    quality_gate_passed: bool
    eligible_for_comparison: bool
    pareto_optimal: bool | None = None
    pareto_dominated_by: list[CalibrationWorkflowId] = Field(
        default_factory=list[CalibrationWorkflowId]
    )
    medians: dict[str, float | None]
    metric_availability: dict[str, Literal["available", "unavailable"]] = Field(
        default_factory=dict
    )


class CalibrationTaskResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    quality_gate_passed: bool
    comparison_eligible: bool
    h1_winner: CalibrationWorkflowId | None = None
    h2_winner: CalibrationWorkflowId | None = None
    h1_margin: float | None = Field(default=None, ge=0.0)
    h2_margin: float | None = Field(default=None, ge=0.0)
    preference_reversal: bool
    infra_sensitive_anchor_candidate: bool
    reason: str


class CalibrationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["calibration-v0-summary-v2"] = (
        "calibration-v0-summary-v2"
    )
    policy: dict[str, Any]
    metric_definitions: dict[str, str]
    cells: list[CalibrationCellResult]
    tasks: list[CalibrationTaskResult]
    totals: dict[str, int]


class CalibrationBreakEvenPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    world_id: CalibrationWorldId
    rtt_ms: float = Field(ge=0.0)
    added_rtt_ms: float = Field(ge=0.0)
    physical_baseline_rtt_ms: float = Field(ge=0.0)
    bandwidth_star_mbps: float | None = Field(default=None, gt=0.0)
    centralized_base_ms: float = Field(ge=0.0)
    local_reduction_base_ms: float = Field(ge=0.0)
    centralized_local_preprocessing_ms: float = Field(ge=0.0)
    centralized_remote_model_ms: float = Field(ge=0.0)
    local_reduction_local_preprocessing_ms: float = Field(ge=0.0)
    local_reduction_remote_model_ms: float = Field(ge=0.0)
    centralized_transfer_bytes: float = Field(ge=0.0)
    local_reduction_transfer_bytes: float = Field(ge=0.0)
    centralized_transfer_count: float = Field(ge=0.0)
    local_reduction_transfer_count: float = Field(ge=0.0)
    centralized_network_path_bytes: float = Field(ge=0.0)
    local_reduction_network_path_bytes: float = Field(ge=0.0)
    centralized_network_round_trips: float = Field(ge=0.0)
    local_reduction_network_round_trips: float = Field(ge=0.0)
    sweep: list[dict[str, float | str]]
    reason: str | None = None


class CalibrationBreakEvenAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["calibration-v0-break-even-v1"] = (
        "calibration-v0-break-even-v1"
    )
    method: str
    measured_inputs_only: Literal[True] = True
    tasks: list[CalibrationBreakEvenPoint]


class CalibrationAnchorTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    workflows: tuple[Literal["centralized_raw"], Literal["local_reduction"]] = (
        "centralized_raw",
        "local_reduction",
    )
    worlds: tuple[
        Literal["H1_distributed_constrained"],
        Literal["H2_distributed_favorable"],
    ] = ("H1_distributed_constrained", "H2_distributed_favorable")
    h1_margin: float = Field(ge=0.0)
    h2_margin: float = Field(ge=0.0)
    bandwidth_star_mbps: float | None = Field(default=None, gt=0.0)
