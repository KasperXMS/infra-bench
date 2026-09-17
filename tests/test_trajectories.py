from infra_bench.adapters import SweBenchVerifiedAdapter
from infra_bench.trajectories import (
    TraceAction,
    TrajectoryRecord,
    build_workflow_bank,
    canonicalize_trajectory,
)


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


def test_canonicalizer_maps_agent_specific_actions_and_removes_noise():
    task = _task()
    trajectory = TrajectoryRecord(
        task_id=task.task_id,
        success=True,
        actions=[
            TraceAction(name="cd /repo"),
            TraceAction(name="ripgrep symbol"),
            TraceAction(name="open_file"),
            TraceAction(name="apply_patch"),
            TraceAction(name="pytest tests/test_bug.py"),
        ],
        provenance={"run_id": "run-1"},
    )
    workflow = canonicalize_trajectory(trajectory, task)
    assert workflow is not None
    assert [node.operator for node in workflow.nodes] == [
        "search_code",
        "read_file",
        "apply_patch",
        "run_targeted_test",
    ]
    assert workflow.provenance["type"] == "real_successful_trajectory"


def test_real_trajectory_is_preferred_but_fallback_preserves_minimum_diversity():
    task = _task()
    adapter = SweBenchVerifiedAdapter()
    trace = TrajectoryRecord(
        task_id=task.task_id,
        success=True,
        actions=[TraceAction(name="grep"), TraceAction(name="apply_patch")],
    )
    bank = build_workflow_bank([task], adapter.ingest_workflows, [trace])
    assert any(item.provenance["type"] == "real_successful_trajectory" for item in bank)
    assert len(bank) == 1
    assert bank[0].provenance["realizability"]["status"] == "realizable"
