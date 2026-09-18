"""World-independent schemas for offline multi-document scenario mining."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .interaction import TaskInteractionSpec


class ScenarioArtifact(BaseModel):
    """One benchmark-provided input boundary.

    ``content`` is optional so mined banks can retain a stable remote source
    reference without copying the raw dataset into the code checkout.
    """

    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    type: str
    content: str | None = None
    source_ref: str | None = None
    size_bytes: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_content_or_source(self) -> ScenarioArtifact:
        if self.content is None and self.source_ref is None:
            raise ValueError("scenario artifact requires content or source_ref")
        return self


class ScenarioEvidence(BaseModel):
    """Hidden benchmark evidence annotation used only by the offline miner."""

    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    span: str | None = None
    supporting_fact: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_annotation(self) -> ScenarioEvidence:
        if self.span is None and self.supporting_fact is None:
            raise ValueError("scenario evidence requires a span or supporting fact")
        return self


class ScenarioEvaluator(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["exact_match", "multiple_choice", "structured_match", "external"]
    target: str | list[str]
    deterministic: bool


class SceneFeatures(BaseModel):
    """Task-only scene statistics; no infrastructure measurements belong here."""

    model_config = ConfigDict(extra="forbid")

    artifact_count: int = Field(ge=0)
    raw_bytes: int = Field(ge=0)
    evidence_bytes: int | None = Field(default=None, ge=0)
    raw_evidence_ratio: float | None = Field(default=None, ge=0.0)
    evidence_artifact_count: int | None = Field(default=None, ge=0)
    evidence_dispersion: float | None = Field(default=None, ge=0.0, le=1.0)
    deterministic_evaluator: bool


class ScenarioRecord(BaseModel):
    """Normalized benchmark scenario with original task semantics preserved."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["scenario-v1"] = "scenario-v1"
    scenario_id: str
    dataset: str
    task_id: str
    question: str
    answer: str | list[str]
    artifacts: list[ScenarioArtifact]
    evidence: list[ScenarioEvidence] | None = None
    evidence_status: Literal[
        "known",
        "hidden_by_split",
        "absent_in_public_release",
        "not_applicable_or_incomplete",
    ]
    evaluator: ScenarioEvaluator
    scene_features: SceneFeatures
    interaction_spec: TaskInteractionSpec
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_references(self) -> ScenarioRecord:
        if self.interaction_spec.task_id != self.task_id:
            raise ValueError("scenario TaskInteractionSpec task_id mismatch")
        artifact_ids = [item.artifact_id for item in self.artifacts]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("scenario artifact IDs must be unique")
        initial_ids = [item.artifact_id for item in self.interaction_spec.initial_artifacts]
        if initial_ids != artifact_ids:
            raise ValueError("TaskInteractionSpec artifacts must match scenario artifacts")
        if self.evidence_status == "known" and self.evidence is None:
            raise ValueError("known evidence status requires evidence annotations")
        if self.evidence_status != "known" and self.evidence is not None:
            raise ValueError("unknown/incomplete evidence status must not carry evidence")
        if self.evidence is not None:
            unknown = {item.artifact_id for item in self.evidence} - set(artifact_ids)
            if unknown:
                raise ValueError(f"evidence references unknown artifacts: {sorted(unknown)}")
        if self.scene_features.artifact_count != len(self.artifacts):
            raise ValueError("scene feature artifact_count does not match artifacts")
        return self

    def planner_view(self) -> dict[str, Any]:
        """Return the only ScenarioRecord representation safe for Planner input."""

        return {
            "schema_version": self.schema_version,
            "scenario_id": self.scenario_id,
            "dataset": self.dataset,
            "task_id": self.task_id,
            "question": self.question,
            "interaction_spec": self.interaction_spec.model_dump(
                mode="json", exclude={"external_evaluator"}
            ),
        }


class WorkflowDemandTemplate(BaseModel):
    """Intrinsic demand D(G), deliberately independent of infrastructure H."""

    model_config = ConfigDict(extra="forbid")

    strategy_id: Literal["centralized_raw", "distributed_reduction"]
    artifact_bytes_consumed: int = Field(ge=0)
    artifact_bytes_produced: int | None = Field(default=None, ge=0)
    reasoning_input_bytes: int | None = Field(default=None, ge=0)
    reasoning_stages: int = Field(ge=1)
    local_preprocessing_bytes: int = Field(ge=0)
    potential_parallel_width: int = Field(ge=1)
    dependency_pattern: Literal["raw_fan_in", "parallel_map_then_reduce"]


class StrategyMultiplicity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy_multiplicity: bool
    centralized_raw_realizable: bool
    distributed_reduction_realizable: bool
    reasons: list[str] = Field(default_factory=list)


class InfrastructureCandidate(BaseModel):
    """Plausible H for later calibration; never part of intrinsic demand."""

    model_config = ConfigDict(extra="forbid")

    world_id: Literal["H_distributed", "H_colocated"]
    artifact_sites: dict[str, str]
    reasoner_site: str
    reduction_sites: list[str]
    bandwidth_mbps: float = Field(gt=0.0)
    rtt_ms: float = Field(ge=0.0)
    basis: str


class ModeledStrategyCost(BaseModel):
    """Screening-only M(D(G), H); values are not admission evidence."""

    model_config = ConfigDict(extra="forbid")

    world_id: str
    strategy_id: Literal["centralized_raw", "distributed_reduction"]
    estimated_cost_ms: float = Field(ge=0.0)
    modeled_cross_site_bytes: int = Field(ge=0)
    modeled_network_ms: float = Field(ge=0.0)
    assumptions: dict[str, float] = Field(default_factory=dict)
