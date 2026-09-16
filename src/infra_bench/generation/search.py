from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from ..schemas import ExecutionResult, InfraState, OperatorProfile, WorkflowRecord
from ..simulator.scheduler import optimize_workflow
from ..simulator.utility import metric_cost


@dataclass(frozen=True)
class SearchPoint:
    value: Any
    infra: InfraState
    oracle_workflow_id: str
    oracle_assignment: dict[str, str]
    executions: dict[str, ExecutionResult]


def select_oracle(
    executions: dict[str, ExecutionResult], objective: str = "latency_s"
) -> tuple[str, ExecutionResult]:
    feasible = [
        (workflow_id, result)
        for workflow_id, result in executions.items()
        if result.feasible
    ]
    if not feasible:
        raise ValueError("no feasible workflow")
    return min(feasible, key=lambda item: (metric_cost(item[1], objective), item[0]))


def search_counterfactual(
    workflows: Iterable[WorkflowRecord],
    anchor: InfraState,
    profiles: dict[str, OperatorProfile],
    candidate_values: Iterable[Any],
    apply_value: Callable[[InfraState, Any, int], InfraState],
    *,
    objective: str = "latency_s",
    load_alpha: float = 0.25,
) -> list[SearchPoint]:
    results: list[SearchPoint] = []
    workflows = list(workflows)
    for index, value in enumerate(candidate_values):
        infra = apply_value(anchor, value, index)
        executions = {
            workflow.workflow_id: optimize_workflow(
                workflow,
                infra,
                profiles,
                objective=objective,
                load_alpha=load_alpha,
            )
            for workflow in workflows
        }
        workflow_id, execution = select_oracle(executions, objective)
        results.append(
            SearchPoint(
                value=value,
                infra=infra,
                oracle_workflow_id=workflow_id,
                oracle_assignment=execution.assignment,
                executions=executions,
            )
        )
    return results


def classify_transitions(points: list[SearchPoint]) -> list[tuple[str, SearchPoint, SearchPoint]]:
    transitions: list[tuple[str, SearchPoint, SearchPoint]] = []
    for previous, current in zip(points, points[1:], strict=False):
        if previous.oracle_workflow_id != current.oracle_workflow_id:
            case_type = "semantic_switch"
        elif previous.oracle_assignment != current.oracle_assignment:
            case_type = "placement_only"
        else:
            case_type = "invariance"
        transitions.append((case_type, previous, current))
    return transitions

