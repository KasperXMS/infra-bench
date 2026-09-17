from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .interaction import (
    InitialArtifactSpec,
    ObservationSpec,
    OperatorId,
    RuntimeVerifierSpec,
    TaskInteractionSpec,
    basic_task_interaction_spec,
)


class PlannerTaskInteractionSpec(BaseModel):
    """Planner-visible task semantics; the external evaluator is omitted."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    task_id: str
    objective: str
    initial_artifacts: list[InitialArtifactSpec]
    operators: list[OperatorId]
    observations: list[ObservationSpec]
    runtime_verifier: RuntimeVerifierSpec


class PlannerTaskRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    source: str
    instruction: str
    input_type: str
    artifact_refs: list[str]
    metadata: dict[str, Any]
    interaction_spec: PlannerTaskInteractionSpec


class TaskRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    source: str
    instruction: str
    input_type: str
    artifact_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    evaluator_type: str
    evaluator_config: dict[str, Any] = Field(default_factory=dict)
    interaction_spec: TaskInteractionSpec

    @model_validator(mode="before")
    @classmethod
    def upgrade_legacy_record(cls, value: Any) -> Any:
        """Load historical banks while materializing the new mandatory contract."""
        if not isinstance(value, dict) or value.get("interaction_spec") is not None:
            return value
        upgraded = dict(value)
        input_type = str(upgraded.get("input_type", "input"))
        refs = [str(item) for item in upgraded.get("artifact_refs", [])]
        if input_type == "repository":
            artifacts = [("repository", refs[0] if refs else "repository")]
            operators: list[OperatorId] = [
                "search_code",
                "read_file",
                "edit_file",
                "apply_patch",
                "run_targeted_test",
                "run_full_test",
                "invoke_model",
                "submit_patch",
            ]
            verifier = "partial"
            evaluator = "swebench_official"
        elif input_type == "video":
            artifacts = [("raw_video", refs[0] if refs else "raw_video")]
            operators = ["sample_frames", "invoke_model"]
            verifier = "none"
            evaluator = "video_mme_v2_official"
        else:
            prefix = "img" if input_type in {"images", "image-set"} else "input"
            artifacts = [
                (f"{prefix}_{index:02d}", ref) for index, ref in enumerate(refs, 1)
            ]
            operators = ["invoke_model"]
            verifier = "none"
            evaluator = str(upgraded.get("evaluator_type", "external"))
        upgraded["interaction_spec"] = basic_task_interaction_spec(
            task_id=str(upgraded.get("task_id", "task")),
            objective=str(upgraded.get("instruction", "")),
            artifacts=artifacts,
            operators=operators,
            evaluator_id=evaluator,
            verifier_level=verifier,
        ).model_dump(mode="json")
        return upgraded

    @model_validator(mode="after")
    def interaction_matches_task(self) -> "TaskRecord":
        if self.interaction_spec.task_id != self.task_id:
            raise ValueError("TaskInteractionSpec task_id does not match TaskRecord")
        if self.interaction_spec.objective != self.instruction:
            raise ValueError("TaskInteractionSpec objective does not match task instruction")
        return self

    def planner_view(self) -> PlannerTaskRecord:
        interaction = self.interaction_spec.model_dump(
            mode="python", exclude={"external_evaluator"}
        )
        return PlannerTaskRecord(
            task_id=self.task_id,
            source=self.source,
            instruction=self.instruction,
            input_type=self.input_type,
            artifact_refs=self.artifact_refs,
            metadata=self.metadata,
            interaction_spec=PlannerTaskInteractionSpec.model_validate(interaction),
        )
