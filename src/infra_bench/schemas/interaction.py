"""Task semantics shared with the real MAS runtime."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

VerifierLevel = Literal["none", "partial", "terminal"]
OperatorId = Literal[
    "search_code",
    "read_file",
    "edit_file",
    "apply_patch",
    "run_targeted_test",
    "run_full_test",
    "invoke_model",
    "submit_patch",
    "read_artifact",
    "sample_frames",
]


class InitialArtifactSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    kind: str
    source_ref: str


class ObservationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation_id: str
    produced_by: list[OperatorId]
    description: str


class RuntimeVerifierSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: VerifierLevel
    signals: list[str] = Field(default_factory=list)


class ExternalEvaluatorSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluator_id: str
    visibility: Literal["external_only"] = "external_only"


class TaskInteractionSpec(BaseModel):
    """Semantic task contract; infrastructure is deliberately absent."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["task-interaction-v1"] = "task-interaction-v1"
    task_id: str
    objective: str
    initial_artifacts: list[InitialArtifactSpec]
    operators: list[OperatorId]
    observations: list[ObservationSpec]
    runtime_verifier: RuntimeVerifierSpec
    external_evaluator: ExternalEvaluatorSpec

    @model_validator(mode="after")
    def validate_references(self) -> TaskInteractionSpec:
        if len(self.operators) != len(set(self.operators)):
            raise ValueError("TaskInteractionSpec operators must be unique")
        allowed = set(self.operators)
        for observation in self.observations:
            unknown = set(observation.produced_by) - allowed
            if unknown:
                raise ValueError(
                    f"observation {observation.observation_id!r} references disallowed "
                    f"operators: {sorted(unknown)}"
                )
        return self


class OperatorDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operator_id: OperatorId
    description: str
    mutates_artifacts: bool = False


class OperatorRegistry:
    """Canonical stable semantic operator IDs, independent of runtime bindings."""

    def __init__(self, definitions: list[OperatorDefinition]) -> None:
        self._definitions = {item.operator_id: item for item in definitions}
        if len(self._definitions) != len(definitions):
            raise ValueError("operator IDs must be unique")

    def contains(self, operator_id: str) -> bool:
        return operator_id in self._definitions

    def require(self, operator_ids: list[str]) -> None:
        unknown = sorted(set(operator_ids) - self._definitions.keys())
        if unknown:
            raise ValueError(f"unknown semantic operators: {unknown}")

    def ids(self) -> frozenset[str]:
        return frozenset(self._definitions)


DEFAULT_OPERATOR_REGISTRY = OperatorRegistry(
    [
        OperatorDefinition(operator_id="search_code", description="Search repository text."),
        OperatorDefinition(operator_id="read_file", description="Read repository text."),
        OperatorDefinition(
            operator_id="edit_file", description="Apply an exact file edit.", mutates_artifacts=True
        ),
        OperatorDefinition(
            operator_id="apply_patch", description="Apply a validated patch.", mutates_artifacts=True
        ),
        OperatorDefinition(
            operator_id="run_targeted_test", description="Run a bounded targeted test."
        ),
        OperatorDefinition(operator_id="run_full_test", description="Run the full test command."),
        OperatorDefinition(operator_id="invoke_model", description="Invoke a logical model."),
        OperatorDefinition(
            operator_id="submit_patch", description="Submit the current patch.", mutates_artifacts=True
        ),
        OperatorDefinition(operator_id="read_artifact", description="Inspect a text artifact."),
        OperatorDefinition(operator_id="sample_frames", description="Sample video frames."),
    ]
)

# This is the deployable semantic surface shared by the current general and
# code-task MAS runtimes.  A registered operator is not necessarily bound yet
# (for example, ``sample_frames`` is intentionally not bound in this phase).
CODE_MAS_OPERATOR_BINDINGS = frozenset(
    {
        "search_code",
        "read_file",
        "edit_file",
        "apply_patch",
        "run_targeted_test",
        "run_full_test",
        "invoke_model",
        "submit_patch",
    }
)
GENERAL_MAS_OPERATOR_BINDINGS = frozenset({"invoke_model", "read_artifact"})
CURRENT_MAS_OPERATOR_BINDINGS = (
    CODE_MAS_OPERATOR_BINDINGS | GENERAL_MAS_OPERATOR_BINDINGS
)


def runtime_bindings_for_task(task: TaskInteractionSpec) -> frozenset[str]:
    """Select one real runtime surface; never accept a fictitious union runtime."""
    code_only = CODE_MAS_OPERATOR_BINDINGS - GENERAL_MAS_OPERATOR_BINDINGS
    if set(task.operators) & code_only:
        return CODE_MAS_OPERATOR_BINDINGS
    return GENERAL_MAS_OPERATOR_BINDINGS


class WorkflowRealizability(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["realizable", "not_realizable"]
    unsupported_operators: list[str] = Field(default_factory=list)
    disallowed_operators: list[str] = Field(default_factory=list)
    artifact_errors: list[str] = Field(default_factory=list)


def basic_task_interaction_spec(
    *,
    task_id: str,
    objective: str,
    artifacts: list[tuple[str, str]],
    operators: list[OperatorId],
    evaluator_id: str,
    verifier_level: VerifierLevel = "none",
) -> TaskInteractionSpec:
    """Small explicit constructor used by synthetic and smoke fixtures."""
    return TaskInteractionSpec(
        task_id=task_id,
        objective=objective,
        initial_artifacts=[
            InitialArtifactSpec(artifact_id=item_id, kind="input", source_ref=source)
            for item_id, source in artifacts
        ],
        operators=operators,
        observations=[
            ObservationSpec(
                observation_id="operation_result",
                produced_by=operators,
                description="Bounded result returned by an allowed semantic operation.",
            )
        ],
        runtime_verifier=RuntimeVerifierSpec(level=verifier_level),
        external_evaluator=ExternalEvaluatorSpec(evaluator_id=evaluator_id),
    )
