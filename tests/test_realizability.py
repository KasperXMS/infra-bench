from infra_bench.adapters import SweBenchVerifiedAdapter
from infra_bench.real_tasks.admission import is_admission_eligible
from infra_bench.real_tasks.realizability import validate_workflow_realizability
from infra_bench.schemas import WorkflowNode, WorkflowRecord


def _task():
    return SweBenchVerifiedAdapter().task_from_row(
        {
            "instance_id": "owner__repo-1",
            "repo": "owner/repo",
            "base_commit": "abc",
            "problem_statement": "Fix it",
            "FAIL_TO_PASS": [],
            "PASS_TO_PASS": [],
        },
        "rev",
    )


def test_realizability_requires_registered_bound_allowed_operators_and_artifacts():
    task = _task()
    valid = WorkflowRecord(
        workflow_id="valid",
        task_id=task.task_id,
        source="test",
        success=True,
        nodes=[
            WorkflowNode(
                node_id="search",
                operator="search_code",
                input_artifacts=["repository"],
                output_artifacts=["matches"],
            )
        ],
    )
    assert (
        validate_workflow_realizability(
            valid, task.interaction_spec, runtime_bindings={"search_code"}
        ).status
        == "realizable"
    )

    unsupported = valid.model_copy(
        update={"nodes": [valid.nodes[0].model_copy(update={"operator": "inspect_repo"})]}
    )
    result = validate_workflow_realizability(
        unsupported, task.interaction_spec, runtime_bindings={"search_code"}
    )
    assert result.status == "not_realizable"
    assert result.unsupported_operators == ["inspect_repo"]

    missing_artifact = valid.model_copy(
        update={
            "nodes": [
                valid.nodes[0].model_copy(update={"input_artifacts": ["hidden_context"]})
            ]
        }
    )
    result = validate_workflow_realizability(
        missing_artifact, task.interaction_spec, runtime_bindings={"search_code"}
    )
    assert result.status == "not_realizable"
    assert result.artifact_errors


def test_admission_is_three_way_conjunction():
    task = _task()
    workflow = WorkflowRecord(
        workflow_id="verified",
        task_id=task.task_id,
        source="test",
        nodes=[
            WorkflowNode(
                node_id="search",
                operator="search_code",
                input_artifacts=["repository"],
                output_artifacts=["matches"],
            )
        ],
        success=True,
        provenance={
            "verification": {
                "evaluator": "swebench_official",
                "passed": True,
                "quality_threshold_met": True,
            }
        },
    )
    assert is_admission_eligible(
        workflow,
        task.interaction_spec,
        infra_sensitive=True,
        runtime_bindings={"search_code"},
    )
    assert not is_admission_eligible(
        workflow,
        task.interaction_spec,
        infra_sensitive=False,
        runtime_bindings={"search_code"},
    )
    assert not is_admission_eligible(
        workflow,
        task.interaction_spec,
        infra_sensitive=True,
        runtime_bindings=set(),
    )
