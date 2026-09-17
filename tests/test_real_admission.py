from infra_bench.real_tasks.admission import is_evaluator_verified
from infra_bench.schemas import WorkflowNode, WorkflowRecord


def _workflow(success: bool, verification: dict | None = None) -> WorkflowRecord:
    provenance = {"type": "real_execution"}
    if verification is not None:
        provenance["verification"] = verification
    return WorkflowRecord(
        workflow_id="workflow",
        task_id="task",
        source="test",
        nodes=[WorkflowNode(node_id="run", operator="reason")],
        success=success,
        provenance=provenance,
    )


def test_unverified_success_does_not_pass_admission() -> None:
    assert not is_evaluator_verified(_workflow(True))


def test_original_evaluator_passes_admission() -> None:
    workflow = _workflow(
        True,
        {
            "evaluator": "video_mme_v2_official",
            "passed": True,
            "quality_threshold_met": True,
        },
    )
    assert is_evaluator_verified(workflow)


def test_threshold_failure_does_not_pass_admission() -> None:
    workflow = _workflow(
        True,
        {
            "evaluator": "swebench_official",
            "passed": True,
            "quality_threshold_met": False,
        },
    )
    assert not is_evaluator_verified(workflow)
