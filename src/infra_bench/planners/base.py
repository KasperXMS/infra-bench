from abc import ABC, abstractmethod

from ..schemas import PlannerDecision, PlannerInput


class Planner(ABC):
    name: str

    @abstractmethod
    def select(self, case: PlannerInput) -> PlannerDecision:
        raise NotImplementedError

