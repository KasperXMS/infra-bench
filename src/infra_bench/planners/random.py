import hashlib

from ..schemas import PlannerDecision, PlannerInput
from .base import Planner


class RandomPlanner(Planner):
    name = "random"

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def select(self, case: PlannerInput) -> PlannerDecision:
        workflow_ids = sorted(workflow.workflow_id for workflow in case.candidate_workflows)
        digest = hashlib.sha256(f"{self.seed}:{case.case_id}".encode()).digest()
        index = int.from_bytes(digest[:8], "big") % len(workflow_ids)
        return PlannerDecision(selected_workflow=workflow_ids[index])
