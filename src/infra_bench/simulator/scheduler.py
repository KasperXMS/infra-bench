from itertools import product

from ..schemas import ExecutionResult, InfraState, OperatorProfile, WorkflowRecord
from .execution import simulate_assignment

SUPPORTED_OBJECTIVES = {"latency_s", "wan_mb", "monetary_cost"}


def _objective_value(result: ExecutionResult, objective: str) -> float:
    if result.metrics is None:
        return float("inf")
    return float(getattr(result.metrics, objective))


def optimize_workflow(
    workflow: WorkflowRecord,
    infra: InfraState,
    profiles: dict[str, OperatorProfile],
    *,
    objective: str = "latency_s",
    load_alpha: float = 0.25,
    max_assignments: int = 200_000,
) -> ExecutionResult:
    if objective not in SUPPORTED_OBJECTIVES:
        raise ValueError(f"unsupported objective: {objective}")

    node_order = workflow.topological_order()
    node_by_id = {node.node_id: node for node in workflow.nodes}
    choices: list[list[str]] = []
    for node_id in node_order:
        node = node_by_id[node_id]
        profile = profiles.get(node.operator)
        if profile is None:
            return ExecutionResult(feasible=False, reason=f"missing profile: {node.operator}")
        required = set(profile.required_capabilities) | set(node.required_capabilities)
        compatible = sorted(
            executor.executor_id
            for executor in infra.executors()
            if required.issubset(executor.capabilities)
        )
        if not compatible:
            return ExecutionResult(
                feasible=False, reason=f"no capable executor for node {node_id}"
            )
        choices.append(compatible)

    combination_count = 1
    for node_choices in choices:
        combination_count *= len(node_choices)
    if combination_count > max_assignments:
        return _greedy_optimize(workflow, infra, profiles, objective, load_alpha)

    best: ExecutionResult | None = None
    best_key: tuple[float, tuple[str, ...]] | None = None
    for selected in product(*choices):
        assignment = dict(zip(node_order, selected, strict=True))
        result = simulate_assignment(
            workflow, infra, profiles, assignment, load_alpha=load_alpha
        )
        if not result.feasible:
            continue
        key = (_objective_value(result, objective), selected)
        if best_key is None or key < best_key:
            best = result
            best_key = key
    return best or ExecutionResult(feasible=False, reason="no feasible assignment")


def _greedy_optimize(
    workflow: WorkflowRecord,
    infra: InfraState,
    profiles: dict[str, OperatorProfile],
    objective: str,
    load_alpha: float,
) -> ExecutionResult:
    """Deterministic fallback for unusually large assignment spaces."""
    assignment: dict[str, str] = {}
    nodes = {node.node_id: node for node in workflow.nodes}
    for node_id in workflow.topological_order():
        node = nodes[node_id]
        profile = profiles[node.operator]
        required = set(profile.required_capabilities) | set(node.required_capabilities)
        compatible = sorted(
            executor.executor_id
            for executor in infra.executors()
            if required.issubset(executor.capabilities)
        )
        # Use intrinsic execution time as a stable local proxy. The final result is
        # always recomputed with full network and dependency costs.
        compatible.sort(
            key=lambda executor_id: (
                next(
                    executor.speed_factors.get(
                        node.operator, executor.speed_factors.get("default", 1.0)
                    )
                    * (1.0 + load_alpha * executor.load / (1.0 - executor.load))
                    for executor in infra.executors()
                    if executor.executor_id == executor_id
                ),
                executor_id,
            )
        )
        assignment[node_id] = compatible[0]
    return simulate_assignment(workflow, infra, profiles, assignment, load_alpha=load_alpha)

