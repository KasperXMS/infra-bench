from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from infra_bench.real_tasks.trace_workflows import (
    cluster_trace_workflows,
    discover_trace_runs,
)
from infra_bench.schemas.interaction import basic_task_interaction_spec
from infra_bench.schemas.task import TaskRecord


def _task(task_id: str = "project__repo-1") -> TaskRecord:
    objective = "Fix the repository bug and submit a tested patch."
    return TaskRecord(
        task_id=task_id,
        source="swebench_verified",
        instruction=objective,
        input_type="repository",
        artifact_refs=["repo://project/repo@base"],
        evaluator_type="swebench_official",
        interaction_spec=basic_task_interaction_spec(
            task_id=task_id,
            objective=objective,
            artifacts=[("repository", "repo://project/repo@base")],
            operators=[
                "search_code",
                "read_file",
                "edit_file",
                "apply_patch",
                "run_targeted_test",
                "run_full_test",
                "invoke_model",
                "submit_patch",
            ],
            evaluator_id="swebench_official",
            verifier_level="partial",
        ),
    )


def _write_run(
    root: Path,
    run_id: str,
    *,
    operators: list[tuple[str, bool]],
    resolved: bool = True,
    success: bool = True,
    submitted: bool = True,
    task_id: str = "project__repo-1",
    world_id: str = "edge-world",
) -> None:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    result = {
        "task_id": task_id,
        "run_id": run_id,
        "world_id": world_id,
        "success": success,
        "submitted_patch": submitted,
        "retry_replanning_count": 2,
        "transmitted_code_context_bytes": 600,
        "cross_site_transfer_bytes": 500,
        "cross_site_transfer_ms": 40,
        "e2e_ms": 900,
    }
    (run_dir / "code_result.json").write_text(json.dumps(result), encoding="utf-8")
    evaluation = {"resolved_ids": [task_id] if resolved else []}
    (run_dir / "official_evaluation.json").write_text(
        json.dumps(evaluation), encoding="utf-8"
    )
    (run_dir / "config.yaml").write_text(
        "\n".join(
            [
                "world:",
                "  repository_site: edge",
                "  repository_size_bytes: 12345",
                "  bandwidth_mbps: 10.0",
                "  rtt_ms: 5.0",
            ]
        ),
        encoding="utf-8",
    )

    events: list[dict[str, Any]] = []
    base_second = 0
    for index, (operator, action_success) in enumerate(operators, 1):
        action_id = f"a-{index}"
        if operator == "invoke_model":
            prefix = "planner.llm"
            end_fields: dict[str, Any] = {
                "latency_ms": 100,
                "success": action_success,
            }
        elif operator == "shell_noise":
            prefix = "code_tool"
            end_fields = {"tool": "shell", "service_ms": 500, "success": True}
        else:
            prefix = "code_tool"
            end_fields = {
                "tool": operator,
                "service_ms": 10 * index,
                "planner_context_bytes": 100,
                "cross_site": True,
                "cross_site_transfer_bytes": 50,
                "cross_site_transfer_ms": 4,
                "success": action_success,
            }
        # The first two tool actions overlap; all other actions are serial.
        start_second = base_second
        duration = 3 if index in {2, 3} else 1
        if index == 3:
            start_second -= 1
        events.append(
            {
                "event_type": f"{prefix}.start",
                "action_id": action_id,
                "timestamp": f"2026-01-01T00:00:{start_second:02d}Z",
            }
        )
        events.append(
            {
                "event_type": f"{prefix}.end",
                "action_id": action_id,
                "timestamp": f"2026-01-01T00:00:{start_second + duration:02d}Z",
                **end_fields,
            }
        )
        base_second += 2
    (run_dir / "trace.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )


def test_discovery_filters_runs_and_reconstructs_realizable_workflows(
    tmp_path: Path,
) -> None:
    common = [
        ("invoke_model", True),
        ("search_code", True),
        ("read_file", True),
        ("edit_file", True),
        ("run_targeted_test", True),
        ("submit_patch", True),
    ]
    _write_run(tmp_path, "valid", operators=common)
    _write_run(tmp_path, "unresolved", operators=common, resolved=False)
    _write_run(tmp_path, "not-submitted", operators=common, submitted=False)
    _write_run(tmp_path, "other-task", operators=common, task_id="other__repo-2")
    (tmp_path / "incomplete").mkdir()
    (tmp_path / "incomplete" / "trace.jsonl").write_text("", encoding="utf-8")

    runs = discover_trace_runs(tmp_path, [_task()])

    assert [run.run_id for run in runs] == ["valid"]
    run = runs[0]
    assert run.world_id == "edge-world"
    assert run.world_facts == {
        "repository_site": "edge",
        "bandwidth_mbps": 10.0,
        "rtt_ms": 5.0,
        "repository_size_bytes": 12345,
    }
    assert run.semantic_fingerprint == (
        "remote_reasoning",
        "repository_inspection",
        "code_mutation",
        "targeted_verification",
        "patch_submission",
    )
    assert all(
        node.operator in _task().interaction_spec.operators for node in run.workflow.nodes
    )
    assert run.workflow.provenance["official_resolved"] is True
    assert run.resource_signature.remote_reasoning_calls == 1
    assert run.resource_signature.local_search_count == 1
    assert run.resource_signature.targeted_test_count == 1
    assert run.resource_signature.max_parallelism == 2
    assert run.resource_signature.parallel_action_count == 2


def test_semantic_dedup_ignores_repetition_retry_and_shell_noise(tmp_path: Path) -> None:
    baseline = [
        ("invoke_model", True),
        ("search_code", True),
        ("read_file", True),
        ("edit_file", True),
        ("run_targeted_test", True),
        ("submit_patch", True),
    ]
    repeated = [
        ("invoke_model", True),
        ("invoke_model", True),
        ("search_code", False),
        ("search_code", True),
        ("shell_noise", True),
        ("read_file", True),
        ("edit_file", True),
        ("run_targeted_test", False),
        ("run_targeted_test", True),
        ("submit_patch", True),
    ]
    with_full_test = [*baseline[:-1], ("run_full_test", True), baseline[-1]]
    _write_run(tmp_path, "baseline", operators=baseline)
    _write_run(tmp_path, "repeated", operators=repeated)
    _write_run(tmp_path, "full-test", operators=with_full_test)

    clusters = cluster_trace_workflows(discover_trace_runs(tmp_path, [_task()]))

    assert len(clusters) == 2
    compact = next(
        cluster for cluster in clusters if "full_verification" not in cluster.semantic_fingerprint
    )
    full = next(
        cluster for cluster in clusters if "full_verification" in cluster.semantic_fingerprint
    )
    assert compact.run_ids == ["baseline", "repeated"]
    assert [run.world_id for run in compact.runs] == ["edge-world", "edge-world"]
    assert compact.workflow_id.startswith("mas-semantic-project__repo-1-")
    assert compact.resource_signature.local_search_count == 1.5
    assert compact.resource_signature.targeted_test_count == 1.5
    assert compact.resource_signature.failed_action_count == 1
    assert compact.resource_signature.retry_count == 2
    assert full.resource_signature.full_test_count == 1


def test_cluster_workflow_id_is_stable_across_run_names(tmp_path: Path) -> None:
    operators = [
        ("invoke_model", True),
        ("search_code", True),
        ("submit_patch", True),
    ]
    _write_run(tmp_path / "left", "one", operators=operators)
    _write_run(tmp_path / "right", "different-name", operators=operators)

    left = cluster_trace_workflows(discover_trace_runs(tmp_path / "left", [_task()]))[0]
    right = cluster_trace_workflows(discover_trace_runs(tmp_path / "right", [_task()]))[0]

    assert left.workflow_id == right.workflow_id


def test_excluded_task_ids_are_not_ingested(tmp_path: Path) -> None:
    _write_run(
        tmp_path,
        "excluded",
        operators=[("invoke_model", True), ("submit_patch", True)],
    )

    assert discover_trace_runs(
        tmp_path, [_task()], excluded_task_ids={"project__repo-1"}
    ) == []


def test_new_trace_semantics_merge_start_inputs_with_end_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "new-trace"
    run_dir.mkdir()
    (run_dir / "code_result.json").write_text(
        json.dumps(
            {
                "task_id": "project__repo-1",
                "run_id": "new-trace",
                "world_id": "cloud-world",
                "success": True,
                "submitted_patch": True,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "official_evaluation.json").write_text(
        json.dumps({"resolved_ids": ["project__repo-1"]}), encoding="utf-8"
    )
    events: list[dict[str, Any]] = [
        {
            "event_type": "planner.llm.start",
            "action_id": "llm-1",
            "timestamp": "2026-01-01T00:00:00Z",
            "semantic_operator": "invoke_model",
            "input_artifacts": [],
        },
        {
            "event_type": "planner.llm.end",
            "action_id": "llm-1",
            "timestamp": "2026-01-01T00:00:01Z",
            "semantic_operator": "invoke_model",
            "output_artifacts": ["llm-1/reasoning"],
            "success": True,
            "latency_ms": 1000,
        },
        {
            "event_type": "code_tool.start",
            "action_id": "tool-1",
            "timestamp": "2026-01-01T00:00:01Z",
            "semantic_operator": "submit_patch",
            "input_artifacts": [
                "repo://project/repo@base@v0",
                "llm-1/reasoning",
            ],
        },
        {
            "event_type": "code_tool.end",
            "action_id": "tool-1",
            "timestamp": "2026-01-01T00:00:02Z",
            "semantic_operator": "submit_patch",
            "tool": "submit_patch",
            "input_artifacts": [
                "repo://project/repo@base@v0",
                "llm-1/reasoning",
            ],
            "output_artifacts": ["tool-1/submitted_patch"],
            "success": True,
            "service_ms": 10,
        },
    ]
    (run_dir / "trace.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )

    runs = discover_trace_runs(tmp_path, [_task()])

    assert len(runs) == 1
    run = runs[0]
    assert len(run.workflow.nodes) == 2
    assert run.resource_signature.remote_reasoning_calls == 1
    assert run.workflow.nodes[1].input_artifacts == ["repository", "llm-1/reasoning"]
    assert run.workflow.edges[0].artifact == "llm-1/reasoning"
