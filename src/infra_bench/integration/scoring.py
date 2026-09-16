"""Open-ended MAS structural and reference scoring."""

from __future__ import annotations

from collections import defaultdict, deque

from .mas_case import RealizedWorkflow


def structural_metrics(workflow: RealizedWorkflow) -> dict[str, object]:
    """Measure dependency structure without matching a candidate workflow ID."""
    action_ids = [action.action_id for action in workflow.actions]
    predecessors: dict[str, set[str]] = {action_id: set() for action_id in action_ids}
    successors: dict[str, set[str]] = {action_id: set() for action_id in action_ids}
    for source, target in workflow.dependencies:
        if source in successors and target in predecessors:
            successors[source].add(target)
            predecessors[target].add(source)

    indegree = {action_id: len(values) for action_id, values in predecessors.items()}
    ready = deque(sorted(key for key, degree in indegree.items() if degree == 0))
    depth_by_action: dict[str, int] = {}
    while ready:
        action_id = ready.popleft()
        depth_by_action[action_id] = (
            1 + max((depth_by_action[item] for item in predecessors[action_id]), default=-1)
        )
        for target in sorted(successors[action_id]):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    groups: dict[int, list[str]] = defaultdict(list)
    for action_id, depth in depth_by_action.items():
        groups[depth].append(action_id)
    artifact_group_sizes = [len(action.input_artifacts) for action in workflow.actions]
    model_composition: dict[str, int] = {}
    for action in workflow.actions:
        if action.model_id is not None:
            model_composition[action.model_id] = model_composition.get(action.model_id, 0) + 1
    return {
        "action_count": len(workflow.actions),
        "workflow_depth": max(depth_by_action.values(), default=-1) + 1,
        "parallel_action_groups": [sorted(values) for _, values in sorted(groups.items())],
        "max_parallel_width": max((len(values) for values in groups.values()), default=0),
        "artifact_group_sizes": artifact_group_sizes,
        "model_composition": model_composition,
        "refinement_action_count": sum(
            1
            for action in workflow.actions
            if action.role
            and any(
                token in action.role.lower()
                for token in ("refine", "verify", "audit", "adjudicat", "review")
            )
        ),
    }


def reference_regret(actual_cost: float, best_reference_cost: float, epsilon: float = 1e-9) -> float:
    """Compare with a calibrated reference bank without claiming global optimality."""
    return (actual_cost - best_reference_cost) / max(best_reference_cost, epsilon)
