from ..schemas import PlannerDecision, PlannerInput
from .base import Planner


class OraclePlanner(Planner):
    name = "oracle"

    def __init__(self, selections: dict[str, str]) -> None:
        self._selections = selections

    def select(self, case: PlannerInput) -> PlannerDecision:
        return PlannerDecision(
            selected_workflow=self._selections[case.case_id],
            rationale="Reproduced stored oracle selection.",
        )

