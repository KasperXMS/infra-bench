import pytest
from pydantic import ValidationError

from infra_bench.schemas import WorkflowEdge, WorkflowNode, WorkflowRecord


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

