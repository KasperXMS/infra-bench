from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .infra import InfraState
from .task import PlannerTaskRecord, TaskRecord
from .workflow import WorkflowRecord

CaseType = Literal["semantic_switch", "placement_only", "invariance"]


class PlannerInput(BaseModel):
    """Oracle-free view exposed to planners."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    group_id: str
    task: PlannerTaskRecord
    infra: InfraState | None
    candidate_workflows: list[WorkflowRecord]


class BenchmarkCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    group_id: str
    task_id: str
    infra_id: str
    candidate_workflow_ids: list[str]
    perturbation: dict[str, Any]
    case_type: CaseType
    oracle_workflow_id: str
    oracle_assignment: dict[str, str]
    oracle_metrics: dict[str, float]
    candidate_metrics: dict[str, dict[str, Any]]
    task: TaskRecord
    infra: InfraState
    candidate_workflows: list[WorkflowRecord]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def ids_are_consistent(self) -> "BenchmarkCase":
        if self.task.task_id != self.task_id:
            raise ValueError("embedded task does not match task_id")
        if self.infra.infra_id != self.infra_id:
            raise ValueError("embedded infra does not match infra_id")
        workflow_ids = [workflow.workflow_id for workflow in self.candidate_workflows]
        if workflow_ids != self.candidate_workflow_ids:
            raise ValueError("embedded candidate workflows do not match candidate_workflow_ids")
        if self.oracle_workflow_id not in workflow_ids:
            raise ValueError("oracle workflow is not a candidate")
        return self

    def planner_input(self, *, include_infra: bool = True) -> PlannerInput:
        # Evaluator configuration may contain gold answers or hidden tests. It is
        # intentionally stripped from every planner-facing representation.
        return PlannerInput(
            case_id=self.case_id,
            group_id=self.group_id,
            task=self.task.planner_view(),
            infra=self.infra if include_infra else None,
            candidate_workflows=self.candidate_workflows,
        )
