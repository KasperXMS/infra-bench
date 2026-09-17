from __future__ import annotations

from pathlib import Path

from infra_bench.real_tasks.admission_search import (
    analyze_trace_admission,
    build_trace_admission_search_report,
    render_trace_admission_markdown,
)
from infra_bench.real_tasks.trace_workflows import ResourceSignature, TraceWorkflowRun
from infra_bench.schemas.interaction import basic_task_interaction_spec
from infra_bench.schemas.task import TaskRecord
from infra_bench.schemas.workflow import WorkflowNode, WorkflowRecord


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


def _run(
    run_id: str,
    world_id: str,
    fingerprint: tuple[str, ...],
    *,
    e2e_ms: float,
    full_tests: float,
    context_bytes: float,
) -> TraceWorkflowRun:
    task_id = "project__repo-1"
    operators = [
        "invoke_model",
        "search_code",
        "edit_file",
        "run_targeted_test",
        *(["run_full_test"] if full_tests else []),
        "submit_patch",
    ]
    workflow = WorkflowRecord(
        workflow_id=f"trace-{run_id}",
        task_id=task_id,
        source="real_mas_trace",
        nodes=[
            WorkflowNode(node_id=f"n{index}", operator=operator)
            for index, operator in enumerate(operators)
        ],
        success=True,
        provenance={"official_resolved": True},
    )
    return TraceWorkflowRun(
        task_id=task_id,
        run_id=run_id,
        run_dir=f"/runs/{run_id}",
        world_id=world_id,
        semantic_fingerprint=fingerprint,
        workflow=workflow,
        resource_signature=ResourceSignature(
            local_search_count=1,
            local_search_service_ms=100,
            local_read_count=1,
            local_read_service_ms=100,
            remote_reasoning_calls=1 + full_tests,
            remote_reasoning_service_ms=100 + 500 * full_tests,
            transmitted_context_bytes=context_bytes,
            cross_site_bytes=context_bytes,
            cross_site_transfer_ms=context_bytes / 1000,
            cross_site_round_trips=1 + full_tests,
            targeted_test_count=1,
            targeted_test_service_ms=100,
            full_test_count=full_tests,
            full_test_service_ms=400 * full_tests,
            verification_count=1 + full_tests,
            total_tool_service_ms=300 + 400 * full_tests,
            observed_e2e_ms=e2e_ms,
        ),
    )


def _reversal_runs() -> list[TraceWorkflowRun]:
    compact = (
        "remote_reasoning",
        "repository_inspection",
        "code_mutation",
        "targeted_verification",
        "patch_submission",
    )
    full = (*compact[:-1], "full_verification", compact[-1])
    return [
        _run("compact-h1", "H1", compact, e2e_ms=100, full_tests=0, context_bytes=1_000),
        _run("compact-h2", "H2", compact, e2e_ms=180, full_tests=0, context_bytes=1_000),
        _run("full-h1", "H1", full, e2e_ms=150, full_tests=1, context_bytes=100_000),
        _run("full-h2", "H2", full, e2e_ms=100, full_tests=1, context_bytes=100_000),
    ]


def test_admission_requires_resource_contrast_and_empirical_reversal() -> None:
    report = analyze_trace_admission([_task()], _reversal_runs())

    assert report["summary"]["admitted_task_count"] == 1
    assert report["summary"]["admitted_pair_count"] == 1
    assert {item["world_id"] for item in report["oracle_free_world_exports"]} == {
        "H1",
        "H2",
    }
    assert all(
        "workflow" not in item and "winner" not in item
        for item in report["oracle_free_world_exports"]
    )
    task = report["tasks"][0]
    assert task["semantic_workflow_count"] == 2
    calibration = task["pair_calibrations"][0]
    assert calibration["contrast"]["sufficiently_different"] is True
    assert calibration["empirical_reversal_found"] is True
    assert calibration["admitted"] is True
    assert {item["winner"] for item in calibration["empirical_worlds"]} == set(
        calibration["workflow_ids"]
    )


def test_excluded_task_is_audited_but_not_searched() -> None:
    report = analyze_trace_admission(
        [_task()],
        _reversal_runs(),
        excluded_tasks={"project__repo-1": "confirmed_not_semantic_switch"},
    )

    task = report["tasks"][0]
    assert task["eligible_trace_count"] == 4
    assert task["semantic_workflow_count"] == 2
    assert task["pair_calibrations"] == []
    assert task["admitted"] is False
    assert task["reason"] == "excluded: confirmed_not_semantic_switch"


def test_no_trace_report_is_fail_closed_and_writes_no_world_export(
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    (report_dir / "oracle_free_worlds.json").write_text("stale", encoding="utf-8")
    report = build_trace_admission_search_report(
        [_task()],
        trace_root=tmp_path / "empty-traces",
        output_dir=report_dir,
    )

    assert report["summary"]["admitted_pair_count"] == 0
    assert report["oracle_free_world_exports"] == []
    assert (report_dir / "admission_search_report.json").is_file()
    assert (report_dir / "admission_search_report.md").is_file()
    assert (report_dir / "verified_trace_workflow_bank.jsonl").is_file()
    assert not (report_dir / "oracle_free_worlds.json").exists()
    markdown = render_trace_admission_markdown(report)
    assert "No MAS-realizable, official-resolved trace workflow" in markdown
