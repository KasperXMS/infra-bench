from pathlib import Path

import pytest

from infra_bench.integration.export import export_case, export_group
from infra_bench.integration.import_result import import_mas_result
from infra_bench.integration.mas_case import (
    MASExecutionResult,
    MASRunSpec,
    RealizedAction,
    RealizedWorkflow,
)
from infra_bench.integration.smoke import V1_GROUP_ID, build_v1_smoke_cases
from infra_bench.schemas import (
    BenchmarkCase,
    DataArtifact,
    Executor,
    InfraState,
    NetworkLink,
    Site,
    TaskRecord,
    WorkflowNode,
    WorkflowRecord,
)


def _case(tmp_path: Path, *, case_id: str = "v1:colocated") -> BenchmarkCase:
    images = [tmp_path / "one.jpg", tmp_path / "two.jpg"]
    images[0].write_bytes(b"one")
    images[1].write_bytes(b"two")
    workflow = WorkflowRecord(
        workflow_id="hidden-reference",
        task_id="v1",
        source="smoke",
        nodes=[WorkflowNode(node_id="inspect", operator="vlm")],
        success=True,
    )
    task = TaskRecord(
        task_id="v1",
        source="smoke",
        instruction="Find the matching image.",
        input_type="images",
        artifact_refs=[str(path) for path in images],
        evaluator_type="exact_image_id",
        evaluator_config={"image_id": "img_99"},
    )
    infra = InfraState(
        infra_id=f"{case_id}:infra",
        sites=[
            Site(
                site_id="A",
                executors=[
                    Executor(
                        executor_id="a-vlm",
                        site_id="A",
                        capabilities=["vision"],
                        load=0,
                    )
                ],
            ),
            Site(site_id="B"),
        ],
        links=[NetworkLink(src_site="A", dst_site="B", rtt_ms=2, bandwidth_mbps=100)],
        artifacts=[
            DataArtifact(artifact_id="img_01", site_id="A", size_mb=0, kind="image"),
            DataArtifact(artifact_id="img_02", site_id="B", size_mb=0, kind="image"),
        ],
    )
    return BenchmarkCase(
        case_id=case_id,
        group_id="v1:paired",
        task_id="v1",
        infra_id=infra.infra_id,
        candidate_workflow_ids=[workflow.workflow_id],
        perturbation={"placement": case_id},
        case_type="invariance",
        oracle_workflow_id=workflow.workflow_id,
        oracle_assignment={"inspect": "a-vlm"},
        oracle_metrics={"latency_s": 1.0},
        candidate_metrics={workflow.workflow_id: {"metrics": {"latency_s": 1.0}}},
        task=task,
        infra=infra,
        candidate_workflows=[workflow],
        metadata={"calibrated_reference_costs_ms": {"single": 100.0}},
    )


def _execution(case: BenchmarkCase) -> MASExecutionResult:
    workflow = RealizedWorkflow(
        actions=[
            RealizedAction(
                action_id="a",
                model_id="vlm",
                role="inspect-one",
                input_artifacts=["img_01"],
                output_artifacts=["finding-a"],
            ),
            RealizedAction(
                action_id="b",
                model_id="vlm",
                role="inspect-two",
                input_artifacts=["img_02"],
                output_artifacts=["finding-b"],
            ),
            RealizedAction(
                action_id="c",
                model_id="vlm",
                role="verify",
                input_artifacts=["finding-a", "finding-b"],
                output_artifacts=["answer"],
            ),
        ],
        dependencies=[("a", "c"), ("b", "c")],
    )
    return MASExecutionResult(
        case_id=case.case_id,
        group_id=case.group_id,
        task_id=case.task_id,
        visibility="snapshot",
        run_id="run-1",
        final_answer="img_99",
        e2e_ms=150,
        planner_latency_ms=10,
        planner_tokens=20,
        invocation_count=3,
        model_counts={"vlm": 3},
        service_ms_sum=100,
        transfer_bytes=10,
        transfer_ms_sum=2,
        realized_workflow=workflow,
    )


def test_export_is_oracle_free_and_hides_evaluator(tmp_path: Path) -> None:
    case = _case(tmp_path)
    spec = export_case(case, dataset_directory=tmp_path)
    payload = spec.model_dump_json()

    for forbidden in (
        "candidate_workflows",
        "candidate_metrics",
        "oracle_workflow_id",
        "oracle_assignment",
        "oracle_metrics",
        "evaluator_config",
        "img_99",
        "hidden-reference",
    ):
        assert forbidden not in payload
    assert [item.site_id for item in spec.artifacts] == ["A", "B"]


def test_schema_rejects_hidden_fields_even_when_nested() -> None:
    with pytest.raises(ValueError, match="hidden benchmark field"):
        MASRunSpec.model_validate(
            {
                "schema_version": "mas-run-spec-v1",
                "case_id": "c",
                "group_id": "g",
                "task_id": "t",
                "instruction": "x",
                "artifacts": [],
                "infra_world": {},
                "metadata": {"nested": {"oracle_metrics": {}}},
            }
        )


def test_group_export_is_deterministic(tmp_path: Path) -> None:
    first = _case(tmp_path, case_id="v1:b")
    second = _case(tmp_path, case_id="v1:a")
    output_one = tmp_path / "one"
    output_two = tmp_path / "two"

    export_group(
        [first, second], "v1:paired", output_one, dataset_directory=tmp_path
    )
    export_group(
        [second, first], "v1:paired", output_two, dataset_directory=tmp_path
    )

    assert (output_one / "manifest.json").read_bytes() == (
        output_two / "manifest.json"
    ).read_bytes()
    assert (output_one / "world-a.json").read_bytes() == (
        output_two / "world-a.json"
    ).read_bytes()


def test_import_associates_case_runs_hidden_evaluator_and_structure(tmp_path: Path) -> None:
    case = _case(tmp_path)
    imported = import_mas_result([case], _execution(case))

    assert imported.task_quality["correct"] is True
    assert imported.structural_metrics["workflow_depth"] == 2
    assert imported.structural_metrics["max_parallel_width"] == 2
    assert imported.structural_metrics["refinement_action_count"] == 1
    assert imported.reference_regret is not None
    assert abs(imported.reference_regret - 0.5) < 1e-9

    mismatched = _execution(case).model_copy(update={"case_id": "missing"})
    with pytest.raises(ValueError, match="expected one benchmark case"):
        import_mas_result([case], mismatched)

    wrong = _execution(case).model_copy(
        update={"final_answer": "ANSWER: img_01. img_99 was considered and rejected."}
    )
    assert import_mas_result([case], wrong).task_quality["correct"] is False


def test_v1_smoke_builder_keeps_hidden_fields_out_of_export(tmp_path: Path) -> None:
    image_directory = tmp_path / "images"
    image_directory.mkdir()
    for index in range(1, 7):
        (image_directory / f"V1_img_{index:02d}.jpg").write_bytes(bytes([index]))

    cases = build_v1_smoke_cases(image_directory)
    assert {case.group_id for case in cases} == {V1_GROUP_ID}
    assert [artifact.site_id for artifact in cases[0].infra.artifacts] == ["A28"] * 6
    assert [artifact.site_id for artifact in cases[1].infra.artifacts] == [
        "A4",
        "A4",
        "A5",
        "A5",
        "A28",
        "A28",
    ]
    exported = export_case(cases[0], dataset_directory=tmp_path).model_dump_json()
    assert "evaluator" not in exported
    assert "candidate" not in exported
    assert "reference" not in exported
