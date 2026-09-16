"""Shared-on-the-wire schemas for MAS benchmark exchange."""

from __future__ import annotations

from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

Visibility = Literal["none", "static", "snapshot"]

_FORBIDDEN_EXPORT_KEYS = frozenset(
    {
        "candidate_workflows",
        "candidate_workflow_ids",
        "candidate_metrics",
        "oracle_workflow_id",
        "oracle_assignment",
        "oracle_metrics",
        "evaluator_config",
        "evaluator_answer",
        "ground_truth",
        "reference_workflow",
        "reference_workflow_names",
    }
)


def _find_forbidden_key(value: object, path: str = "$") -> str | None:
    if isinstance(value, dict):
        for key, nested in cast(dict[object, object], value).items():
            normalized = str(key).lower()
            if normalized in _FORBIDDEN_EXPORT_KEYS or normalized.startswith("oracle_"):
                return f"{path}.{key}"
            found = _find_forbidden_key(nested, f"{path}.{key}")
            if found is not None:
                return found
    elif isinstance(value, list):
        for index, nested in enumerate(cast(list[object], value)):
            found = _find_forbidden_key(nested, f"{path}[{index}]")
            if found is not None:
                return found
    return None


class ArtifactBinding(BaseModel):
    """Bind one benchmark artifact to a source and initial physical site."""

    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    source_ref: str
    site_id: str
    size_bytes: int | None = Field(default=None, ge=0)


class MASRunSpec(BaseModel):
    """Oracle-free case that can be materialized by infra-aware-mas."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["mas-run-spec-v1"] = "mas-run-spec-v1"
    case_id: str
    group_id: str
    task_id: str
    instruction: str
    artifacts: list[ArtifactBinding]
    infra_world: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def contains_no_hidden_fields(self) -> MASRunSpec:
        leaked = _find_forbidden_key(self.model_dump(mode="python"))
        if leaked is not None:
            raise ValueError(f"MAS run spec contains hidden benchmark field at {leaked}")
        return self


class RealizedAction(BaseModel):
    """One actual logical-model action reconstructed from the MAS trace."""

    model_config = ConfigDict(extra="forbid")

    action_id: str
    parent_action_id: str | None = None
    model_id: str | None = None
    role: str | None = None
    input_artifacts: list[str] = Field(default_factory=list)
    output_artifacts: list[str] = Field(default_factory=list)
    executor_id: str | None = None
    site_id: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


class RealizedWorkflow(BaseModel):
    """Actual actions and artifact-derived dependencies without a workflow ID."""

    model_config = ConfigDict(extra="forbid")

    actions: list[RealizedAction] = []
    dependencies: list[tuple[str, str]] = []


class MASExecutionResult(BaseModel):
    """Machine-readable output emitted by the real MAS benchmark runner."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["mas-execution-result-v1"] = "mas-execution-result-v1"
    case_id: str
    group_id: str
    task_id: str
    visibility: Visibility
    run_id: str
    final_answer: str
    e2e_ms: float = Field(ge=0)
    planner_latency_ms: float = Field(ge=0)
    planner_tokens: int = Field(ge=0)
    invocation_count: int = Field(ge=0)
    model_counts: dict[str, int]
    service_ms_sum: float = Field(ge=0)
    transfer_bytes: int = Field(ge=0)
    transfer_ms_sum: float = Field(ge=0)
    raw_transfer_bytes: int = Field(default=0, ge=0)
    derived_transfer_bytes: int = Field(default=0, ge=0)
    cross_worker_transfer_bytes: int = Field(default=0, ge=0)
    cross_worker_transfer_count: int = Field(default=0, ge=0)
    multimodal_invocation_count: int = Field(default=0, ge=0)
    site_aligned_multimodal_invocations: int = Field(default=0, ge=0)
    site_alignment_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    artifact_grouping: list[list[str]] = []
    realized_workflow: RealizedWorkflow


class ImportedMASResult(BaseModel):
    """Hidden-evaluator output retained by infra-bench."""

    model_config = ConfigDict(extra="forbid")

    execution: MASExecutionResult
    task_quality: dict[str, Any]
    structural_metrics: dict[str, Any]
    reference_regret: float | None = None
