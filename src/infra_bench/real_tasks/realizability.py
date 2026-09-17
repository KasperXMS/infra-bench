"""Hard realizability gate for workflow-bank and admission records."""

from __future__ import annotations

from infra_bench.schemas.interaction import (
    DEFAULT_OPERATOR_REGISTRY,
    TaskInteractionSpec,
    WorkflowRealizability,
)
from infra_bench.schemas.workflow import WorkflowRecord


def validate_workflow_realizability(
    workflow: WorkflowRecord,
    task: TaskInteractionSpec,
    *,
    runtime_bindings: set[str] | frozenset[str],
) -> WorkflowRealizability:
    operators = [node.operator for node in workflow.nodes]
    unsupported = sorted(
        {item for item in operators if not DEFAULT_OPERATOR_REGISTRY.contains(item)}
        | (set(operators) - set(runtime_bindings))
    )
    disallowed = sorted(set(operators) - set(task.operators))

    available = {item.artifact_id for item in task.initial_artifacts}
    artifact_errors: list[str] = []
    nodes = {node.node_id: node for node in workflow.nodes}
    for node_id in workflow.topological_order():
        node = nodes[node_id]
        missing = sorted(set(node.input_artifacts) - available)
        if missing:
            artifact_errors.append(f"{node_id}: missing inputs {missing}")
        available.update(node.output_artifacts)

    return WorkflowRealizability(
        status=(
            "realizable"
            if not unsupported and not disallowed and not artifact_errors
            else "not_realizable"
        ),
        unsupported_operators=unsupported,
        disallowed_operators=disallowed,
        artifact_errors=artifact_errors,
    )
