from ..schemas import PlannerDecision, PlannerInput
from .base import Planner


class ResourceBlindPlanner(Planner):
    """Stable task-only heuristic that never receives infrastructure state."""

    name = "resource-blind"

    def select(self, case: PlannerInput) -> PlannerDecision:
        selected = min(
            case.candidate_workflows,
            key=lambda workflow: (len(workflow.nodes), workflow.workflow_id),
        )
        return PlannerDecision(
            selected_workflow=selected.workflow_id,
            rationale="Selected the shortest semantic workflow without infrastructure input.",
        )

