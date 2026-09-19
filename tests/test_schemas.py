import pytest
from pydantic import ValidationError

from infra_bench.schemas import (
    CURRENT_MAS_OPERATOR_BINDINGS,
    DEFAULT_OPERATOR_REGISTRY,
    GENERAL_MAS_OPERATOR_BINDINGS,
    WorkflowEdge,
    WorkflowNode,
    WorkflowRecord,
)


def test_workflow_round_trip_and_topological_order():
    workflow = WorkflowRecord(
        workflow_id="w",
        task_id="t",
        source="test",
        nodes=[
            WorkflowNode(node_id="b", operator="reason"),
            WorkflowNode(node_id="a", operator="reason"),
        ],
        edges=[WorkflowEdge(src="a", dst="b")],
        success=True,
    )
    restored = WorkflowRecord.model_validate_json(workflow.model_dump_json())
    assert restored == workflow
    assert restored.topological_order() == ["a", "b"]


def test_cycle_is_rejected():
    with pytest.raises(ValidationError, match="DAG"):
        WorkflowRecord(
            workflow_id="cycle",
            task_id="t",
            source="test",
            nodes=[
                WorkflowNode(node_id="a", operator="reason"),
                WorkflowNode(node_id="b", operator="reason"),
            ],
            edges=[WorkflowEdge(src="a", dst="b"), WorkflowEdge(src="b", dst="a")],
            success=True,
        )


def test_visual_reduction_operators_are_registered_and_generically_bound() -> None:
    visual_operators = {
        "make_contact_sheet",
        "extract_clip",
        "process_local_artifact",
        "aggregate_artifacts",
    }
    DEFAULT_OPERATOR_REGISTRY.require(sorted(visual_operators))
    assert visual_operators <= GENERAL_MAS_OPERATOR_BINDINGS
    assert visual_operators <= CURRENT_MAS_OPERATOR_BINDINGS
