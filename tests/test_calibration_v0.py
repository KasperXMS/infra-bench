import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from infra_bench.cli import _read_calibration_records
from infra_bench.real_tasks.calibration_v0 import (
    analyze_break_even,
    evaluate_calibration_runs,
    reference_workflows,
    summarize_calibration,
    validate_calibration_design,
    write_calibration_report,
)
from infra_bench.schemas import (
    CalibrationAnswer,
    CalibrationArtifactMetrics,
    CalibrationChunk,
    CalibrationEvaluatorRecord,
    CalibrationExecution,
    CalibrationNetwork,
    CalibrationQuality,
    CalibrationQuestion,
    CalibrationRun,
    CalibrationServiceMetrics,
    CalibrationTask,
    CalibrationTransfer,
    CalibrationTransferMetrics,
    CalibrationUsage,
    CalibrationWorld,
)


def _task(task_id: str) -> CalibrationTask:
    return CalibrationTask(
        task_id=task_id,
        benchmark="Video-MME",
        instruction="Answer three questions requiring evidence from multiple time periods.",
        duration_s=1_800,
        evaluator_id="video_mme_multiple_choice",
        questions=[
            CalibrationQuestion(
                question_id=f"{task_id}:q{index}",
                prompt=f"Question {index}",
                options={"A": "one", "B": "two", "C": "three", "D": "four"},
            )
            for index in range(1, 4)
        ],
        quality_threshold=2 / 3,
        chunks=[
            CalibrationChunk(
                artifact_id=f"{task_id}:chunk:{index}",
                chunk_index=index,
                start_s=index * 600,
                end_s=(index + 1) * 600,
                duration_s=600,
                source_ref=f"chunk_{index}.mp4",
                site_id=site,
                raw_bytes=100_000_000,
            )
            for index, site in enumerate(("A4", "A5", "A28"))
        ],
    )


def _world(world_id: str, bandwidth: float, rtt: float) -> CalibrationWorld:
    return CalibrationWorld(
        world_id=world_id,
        network=CalibrationNetwork(bandwidth_mbps=bandwidth, rtt_ms=rtt),
        local_model_id="light-vlm-v1",
        strong_model_id="strong-vlm-v1",
        device_fingerprints={
            "A4": "orin-a4",
            "A5": "orin-a5",
            "A28": "orin-a28",
            "4090": "dual-4090",
        },
    )


def _evaluators(tasks: list[CalibrationTask]) -> list[CalibrationEvaluatorRecord]:
    return [
        CalibrationEvaluatorRecord(
            task_id=task.task_id,
            evaluator_id=task.evaluator_id,
            answers=[
                CalibrationAnswer(question_id=question.question_id, correct_option=option)
                for question, option in zip(task.questions, ("A", "B", "C"), strict=True)
            ],
        )
        for task in tasks
    ]


def _run(
    task_id: str,
    workflow_id: str,
    world_id: str,
    repeat: int,
    *,
    e2e_ms: float,
    transfer_ms: float,
    quality_passed: bool = True,
    warmup: bool = False,
) -> CalibrationRun:
    is_local = workflow_id in {"local_reduction", "visual_reduction"}
    transfer_bytes = 3_000_000 if is_local else 300_000_000
    records = [
        CalibrationTransfer(
            transfer_id=f"transfer-{index}",
            artifact_id=f"{task_id}:chunk:{index}",
            src_site=site,
            dst_site="4090",
            bytes=transfer_bytes // 3,
            latency_ms=transfer_ms / 3,
        )
        for index, site in enumerate(("A4", "A5", "A28"))
    ]
    executions = []
    if is_local:
        executions.append(
            CalibrationExecution(
                action_id="reduce-a4",
                operator_id="invoke_model",
                executor_id="a4-vlm",
                worker_id="worker-a4",
                site_id="A4",
                service_ms=300,
                input_bytes=100_000_000,
                output_bytes=1_000_000,
            )
        )
    executions.append(
        CalibrationExecution(
            action_id="reason",
            operator_id="invoke_model",
            executor_id="4090-vlm",
            worker_id="worker-4090",
            site_id="4090",
            service_ms=400,
            input_bytes=transfer_bytes,
            output_bytes=100,
        )
    )
    return CalibrationRun(
        run_id=f"{task_id}-{workflow_id}-{world_id}-{repeat}",
        task_id=task_id,
        workflow_id=workflow_id,
        world_id=world_id,
        repeat=repeat,
        warmup=warmup,
        measurement_series_id="formal-series-v1",
        protocol_id="steady_state_1_warmup_3_measured_v1",
        status="completed",
        quality=CalibrationQuality(
            score=1.0 if quality_passed else 1 / 3,
            passed=quality_passed,
            evaluator_id="video_mme_multiple_choice",
        ),
        artifacts=CalibrationArtifactMetrics(
            raw_bytes=300_000_000,
            reduced_bytes=3_000_000 if is_local else 0,
            reduced_visual_bytes=(
                3_000_000 if workflow_id == "visual_reduction" else 0
            ),
            semantic_evidence_bytes=(
                3_000_000 if workflow_id == "local_reduction" else 0
            ),
            reduction_ratio=0.01 if is_local else 0.0,
        ),
        transfers=CalibrationTransferMetrics(
            count=3,
            bytes=transfer_bytes,
            latency_ms=transfer_ms,
            records=records,
        ),
        service=CalibrationServiceMetrics(
            local_preprocessing_ms=300 if is_local else 0,
            remote_model_ms=400,
            total_ms=700 if is_local else 400,
        ),
        usage=CalibrationUsage(
            input_tokens=1_000,
            output_tokens=20,
            api_cost_usd=0.01,
        ),
        e2e_latency_ms=e2e_ms,
        executions=executions,
        final_answer=json.dumps(
            {
                f"{task_id}:q1": "A",
                f"{task_id}:q2": "B",
                f"{task_id}:q3": "C",
            }
        ),
    )


def _experiment():
    tasks = [_task("video-795"), _task("video-857")]
    worlds = [
        _world("H1_distributed_constrained", 20, 80),
        _world("H2_distributed_favorable", 1_000, 2),
    ]
    workflows = reference_workflows()
    runs = []
    for task in tasks:
        runs.extend(
            [
                _run(
                    task.task_id,
                    "centralized_raw",
                    "H1_distributed_constrained",
                    0,
                    e2e_ms=2_100,
                    transfer_ms=1_050,
                    warmup=True,
                ),
                _run(
                    task.task_id,
                    "local_reduction",
                    "H1_distributed_constrained",
                    0,
                    e2e_ms=1_500,
                    transfer_ms=110,
                    warmup=True,
                ),
                _run(
                    task.task_id,
                    "visual_reduction",
                    "H1_distributed_constrained",
                    0,
                    e2e_ms=1_300,
                    transfer_ms=105,
                    warmup=True,
                ),
                _run(
                    task.task_id,
                    "centralized_raw",
                    "H2_distributed_favorable",
                    0,
                    e2e_ms=600,
                    transfer_ms=110,
                    warmup=True,
                ),
                _run(
                    task.task_id,
                    "local_reduction",
                    "H2_distributed_favorable",
                    0,
                    e2e_ms=1_500,
                    transfer_ms=110,
                    warmup=True,
                ),
                _run(
                    task.task_id,
                    "visual_reduction",
                    "H2_distributed_favorable",
                    0,
                    e2e_ms=900,
                    transfer_ms=105,
                    warmup=True,
                ),
            ]
        )
        for repeat in range(1, 4):
            runs.extend(
                [
                    _run(
                        task.task_id,
                        "centralized_raw",
                        "H1_distributed_constrained",
                        repeat,
                        e2e_ms=2_000,
                        transfer_ms=1_000,
                    ),
                    _run(
                        task.task_id,
                        "local_reduction",
                        "H1_distributed_constrained",
                        repeat,
                        e2e_ms=1_400,
                        transfer_ms=100,
                    ),
                    _run(
                        task.task_id,
                        "visual_reduction",
                        "H1_distributed_constrained",
                        repeat,
                        e2e_ms=1_200,
                        transfer_ms=95,
                    ),
                    _run(
                        task.task_id,
                        "centralized_raw",
                        "H2_distributed_favorable",
                        repeat,
                        e2e_ms=500,
                        transfer_ms=100,
                    ),
                    _run(
                        task.task_id,
                        "local_reduction",
                        "H2_distributed_favorable",
                        repeat,
                        e2e_ms=1_400,
                        transfer_ms=100,
                    ),
                    _run(
                        task.task_id,
                        "visual_reduction",
                        "H2_distributed_favorable",
                        repeat,
                        e2e_ms=800,
                        transfer_ms=95,
                    ),
                ]
            )
    return tasks, worlds, workflows, runs


def test_schema_rejects_gold_and_nonuniform_chunks() -> None:
    payload = _task("video").model_dump()
    payload["metadata"] = {"gold_answer": "A"}
    with pytest.raises(ValidationError, match="must not expose evaluator answers"):
        CalibrationTask.model_validate(payload)


def test_reduced_byte_categories_must_sum_to_reduced_bytes() -> None:
    with pytest.raises(ValidationError, match="must equal"):
        CalibrationArtifactMetrics(
            raw_bytes=100,
            reduced_bytes=20,
            reduced_visual_bytes=12,
            semantic_evidence_bytes=9,
            reduction_ratio=0.2,
        )
    legacy = CalibrationArtifactMetrics(
        raw_bytes=100,
        reduced_bytes=20,
        reduction_ratio=0.2,
    )
    assert legacy.reduced_visual_bytes == 0
    assert legacy.semantic_evidence_bytes == 20

    visual_payload = _run(
        "video-795",
        "visual_reduction",
        "H1_distributed_constrained",
        1,
        e2e_ms=1_200,
        transfer_ms=95,
    ).model_dump()
    visual_payload["artifacts"].pop("reduced_visual_bytes")
    visual_payload["artifacts"].pop("semantic_evidence_bytes")
    with pytest.raises(ValidationError, match="reduced_visual_bytes"):
        CalibrationRun.model_validate(visual_payload)
    payload = _task("video").model_dump()
    payload["chunks"][0]["end_s"] = 500
    with pytest.raises(ValidationError, match="duration_s|contiguous"):
        CalibrationTask.model_validate(payload)


def test_failed_run_is_valid_but_completed_run_requires_measurements() -> None:
    failed = CalibrationRun(
        run_id="failed-1",
        task_id="video",
        workflow_id="centralized_raw",
        world_id="H1_distributed_constrained",
        repeat=1,
        measurement_series_id="formal-series-v1",
        protocol_id="steady_state_1_warmup_3_measured_v1",
        status="failed",
        error="remote model timed out",
    )
    assert failed.error == "remote model timed out"
    with pytest.raises(ValidationError, match="require all metric groups"):
        CalibrationRun(
            run_id="invalid",
            task_id="video",
            workflow_id="centralized_raw",
            world_id="H1_distributed_constrained",
            repeat=1,
            status="completed",
        )


def test_completed_runtime_row_accepts_pending_evaluator_quality() -> None:
    pending = _run(
        "video-795",
        "centralized_raw",
        "H1_distributed_constrained",
        1,
        e2e_ms=2_000,
        transfer_ms=1_000,
    ).model_copy(update={"quality": None})
    restored = CalibrationRun.model_validate_json(pending.model_dump_json())
    assert restored.status == "completed"
    assert restored.quality is None
    assert json.loads(restored.model_dump_json())["warmup"] is False


def test_warmup_and_measured_repeat_numbers_are_explicit() -> None:
    measured = _run(
        "video-795",
        "centralized_raw",
        "H1_distributed_constrained",
        1,
        e2e_ms=2_000,
        transfer_ms=1_000,
    )
    with pytest.raises(ValidationError, match="warm-up runs must use repeat=0"):
        CalibrationRun.model_validate(
            measured.model_dump() | {"warmup": True, "repeat": 1}
        )
    with pytest.raises(ValidationError, match="measured runs must use repeat>=1"):
        CalibrationRun.model_validate(
            measured.model_dump() | {"warmup": False, "repeat": 0}
        )


def test_failed_attempt_may_be_retried_with_same_repeat_number() -> None:
    tasks, worlds, workflows, runs = _experiment()
    failed = CalibrationRun(
        run_id="failed-probe-before-retry",
        task_id=tasks[0].task_id,
        workflow_id="centralized_raw",
        world_id="H1_distributed_constrained",
        repeat=1,
        status="failed",
        error="deployment probe failed",
    )

    summary = summarize_calibration(tasks, worlds, workflows, [failed, *runs])

    assert summary.totals["raw_run_count"] == 49
    assert summary.totals["failed_run_count"] == 1


def test_design_allows_only_network_contrast() -> None:
    tasks, worlds, workflows, _ = _experiment()
    validate_calibration_design(tasks, worlds, workflows)
    validate_calibration_design(tasks[:1], worlds, workflows)
    changed = worlds[1].model_copy(update={"strong_model_id": "other-model"})
    with pytest.raises(ValueError, match="only in network"):
        validate_calibration_design(tasks, [worlds[0], changed], workflows)


def test_quality_gated_reversal_and_measured_break_even() -> None:
    tasks, worlds, workflows, runs = _experiment()
    summary = summarize_calibration(tasks, worlds, workflows, runs)
    assert summary.totals["raw_run_count"] == 48
    assert summary.totals["warmup_run_count"] == 12
    assert all(cell.completed_warmups == 1 for cell in summary.cells)
    assert all(cell.completed_runs == 3 for cell in summary.cells)
    assert summary.totals["anchor_candidate_count"] == 2
    assert all(task.h1_winner == "local_reduction" for task in summary.tasks)
    assert all(task.h2_winner == "centralized_raw" for task in summary.tasks)
    visual_cells = [cell for cell in summary.cells if cell.workflow_id == "visual_reduction"]
    assert len(visual_cells) == 4
    assert all(cell.quality_gate_passed for cell in visual_cells)
    assert all(cell.pareto_optimal is True for cell in visual_cells)
    analysis = analyze_break_even(summary, worlds, runs)
    assert len(analysis.tasks) == 4
    assert all(point.bandwidth_star_mbps is not None for point in analysis.tasks)
    assert all(point.centralized_transfer_bytes == 300_000_000 for point in analysis.tasks)


def test_quality_failure_suppresses_system_comparison() -> None:
    tasks, worlds, workflows, runs = _experiment()
    target = next(
        index
        for index, run in enumerate(runs)
        if not run.warmup
        and run.task_id == "video-795"
        and run.workflow_id == "centralized_raw"
        and run.world_id == "H1_distributed_constrained"
        and run.repeat == 1
    )
    runs[target] = _run(
        "video-795",
        "centralized_raw",
        "H1_distributed_constrained",
        1,
        e2e_ms=2_000,
        transfer_ms=1_000,
        quality_passed=False,
    )
    summary = summarize_calibration(tasks, worlds, workflows, runs)
    failed = next(task for task in summary.tasks if task.task_id == "video-795")
    assert failed.quality_gate_passed is False
    assert failed.comparison_eligible is False
    assert failed.h1_winner is None


def test_visual_quality_failure_does_not_change_two_arm_anchor_admission() -> None:
    tasks, worlds, workflows, runs = _experiment()
    target = next(
        index
        for index, run in enumerate(runs)
        if not run.warmup
        and run.task_id == "video-795"
        and run.workflow_id == "visual_reduction"
        and run.world_id == "H1_distributed_constrained"
        and run.repeat == 1
    )
    runs[target] = runs[target].model_copy(
        update={
            "quality": CalibrationQuality(
                score=1 / 3,
                passed=False,
                evaluator_id="video_mme_multiple_choice",
            )
        }
    )
    summary = summarize_calibration(tasks, worlds, workflows, runs)
    task = next(item for item in summary.tasks if item.task_id == "video-795")
    visual = next(
        cell
        for cell in summary.cells
        if cell.task_id == "video-795"
        and cell.workflow_id == "visual_reduction"
        and cell.world_id == "H1_distributed_constrained"
    )
    assert visual.quality_gate_passed is False
    assert visual.pareto_optimal is None
    assert task.infra_sensitive_anchor_candidate is True


def test_visual_cells_are_only_emitted_for_tasks_with_visual_runs() -> None:
    tasks, worlds, workflows, runs = _experiment()
    runs = [
        run
        for run in runs
        if not (
            run.task_id == "video-857" and run.workflow_id == "visual_reduction"
        )
    ]
    summary = summarize_calibration(tasks, worlds, workflows, runs)
    assert not any(
        cell.task_id == "video-857" and cell.workflow_id == "visual_reduction"
        for cell in summary.cells
    )
    assert sum(
        cell.task_id == "video-795" and cell.workflow_id == "visual_reduction"
        for cell in summary.cells
    ) == 2


def test_warmup_is_excluded_from_medians_quality_and_admission() -> None:
    tasks, worlds, workflows, runs = _experiment()
    warmup = next(run for run in runs if run.warmup)
    replacement = warmup.model_copy(
        update={
            "e2e_latency_ms": 99_999_999.0,
            "quality": CalibrationQuality(
                score=0.0,
                passed=False,
                evaluator_id="video_mme_multiple_choice",
            ),
        }
    )
    runs[runs.index(warmup)] = replacement
    summary = summarize_calibration(tasks, worlds, workflows, runs)
    cell = next(
        item
        for item in summary.cells
        if item.task_id == warmup.task_id
        and item.workflow_id == warmup.workflow_id
        and item.world_id == warmup.world_id
    )
    assert cell.profiling_protocol_passed is True
    assert cell.quality_pass_rate == 1.0
    assert cell.medians["e2e_latency_ms"] == 2_000
    assert summary.totals["anchor_candidate_count"] == 2
    assert len(analyze_break_even(summary, worlds, runs).tasks) == 4


def test_missing_warmup_blocks_formal_admission_but_legacy_rows_still_parse() -> None:
    tasks, worlds, workflows, runs = _experiment()
    legacy_measured = [run.model_copy(update={"warmup": False}) for run in runs if not run.warmup]
    summary = summarize_calibration(tasks, worlds, workflows, legacy_measured)
    assert all(not cell.profiling_protocol_passed for cell in summary.cells)
    assert summary.totals["anchor_candidate_count"] == 0
    legacy_payload = legacy_measured[0].model_dump(exclude={"warmup"})
    assert CalibrationRun.model_validate(legacy_payload).warmup is False


def test_report_selects_latest_complete_formal_series_without_mixing_repeats() -> None:
    tasks, worlds, workflows, runs = _experiment()
    first_cell = [
        run
        for run in runs
        if run.task_id == "video-795"
        and run.workflow_id == "centralized_raw"
        and run.world_id == "H1_distributed_constrained"
    ]
    replacement_series = [
        run.model_copy(
            update={
                "run_id": f"{run.run_id}-series-v2",
                "measurement_series_id": "formal-series-v2",
                "e2e_latency_ms": 1_234.0,
            }
        )
        for run in first_cell
    ]
    summary = summarize_calibration(
        tasks, worlds, workflows, [*runs, *replacement_series]
    )
    cell = next(
        item
        for item in summary.cells
        if item.task_id == "video-795"
        and item.workflow_id == "centralized_raw"
        and item.world_id == "H1_distributed_constrained"
    )
    assert cell.selected_measurement_series_id == "formal-series-v2"
    assert cell.selected_protocol_id == "steady_state_1_warmup_3_measured_v1"
    assert cell.completed_runs == 3
    assert cell.medians["e2e_latency_ms"] == 1_234.0


def test_runtime_metadata_series_fields_are_accepted() -> None:
    run = _run(
        "video-795",
        "centralized_raw",
        "H1_distributed_constrained",
        1,
        e2e_ms=2_000,
        transfer_ms=1_000,
    )
    payload = run.model_dump(exclude={"measurement_series_id", "protocol_id"})
    payload["metadata"] = {
        "measurement_series_id": "calibration-v0-795-all-workflows",
        "measurement_protocol": "steady_state_1_warmup_3_measured_v1",
        "run_prefix": "calibration-v0-795-all-workflows",
    }
    parsed = CalibrationRun.model_validate(payload)
    assert parsed.measurement_series_id is None
    assert parsed.metadata["measurement_protocol"] == (
        "steady_state_1_warmup_3_measured_v1"
    )


def test_evaluator_overrides_reported_quality_and_fails_closed() -> None:
    tasks, _, _, runs = _experiment()
    runs[0] = runs[0].model_copy(
        update={
            "quality": CalibrationQuality(
                score=0.0,
                passed=False,
                evaluator_id="untrusted-runtime-evaluator",
            )
        }
    )
    runs[1] = runs[1].model_copy(update={"final_answer": "not JSON"})
    evaluated = evaluate_calibration_runs(tasks, _evaluators(tasks), runs[:2])
    assert evaluated[0].quality is not None
    assert evaluated[0].quality.score == 1.0
    assert evaluated[0].quality.passed is True
    assert evaluated[0].quality.details["runtime_reported_quality_consistent"] is False
    assert evaluated[1].quality is not None
    assert evaluated[1].quality.score == 0.0
    assert evaluated[1].quality.passed is False
    assert evaluated[1].quality.details["parse_valid"] is False


def test_report_writes_requested_artifacts(tmp_path) -> None:
    tasks, worlds, workflows, runs = _experiment()
    output_dir = tmp_path / "calibration_v0"
    output_dir.mkdir()
    raw_path = output_dir / "raw_runs.jsonl"
    raw_path.write_text("runtime-owned-sentinel\n", encoding="utf-8")
    summary, _ = write_calibration_report(
        tasks,
        _evaluators(tasks),
        worlds,
        workflows,
        runs,
        output_dir,
    )
    assert summary.totals["anchor_candidate_count"] == 2
    expected = {
        "raw_runs.jsonl",
        "evaluated_runs.jsonl",
        "summary.json",
        "summary.md",
        "break_even_analysis.json",
        "critical_path_report.json",
        "admitted_anchor_tasks.json",
    }
    assert {path.name for path in output_dir.iterdir()} == expected
    assert raw_path.read_text(encoding="utf-8") == "runtime-owned-sentinel\n"
    admitted = json.loads(
        (output_dir / "admitted_anchor_tasks.json").read_text()
    )
    assert len(admitted["tasks"]) == 2
    critical = json.loads((output_dir / "critical_path_report.json").read_text())
    assert critical["totals"]["critical_path_available_count"] == 0
    assert all(
        run["critical_path_ms"] is None
        and run["critical_path_availability"] == "unavailable"
        for run in critical["runs"]
    )


def test_checked_in_manifests_are_valid_and_keep_answers_separate() -> None:
    root = Path(__file__).parents[1] / "configs" / "calibration_v0"
    tasks = _read_calibration_records(str(root / "tasks.yaml"), CalibrationTask, "tasks")
    worlds = _read_calibration_records(str(root / "worlds.yaml"), CalibrationWorld, "worlds")
    workflows = _read_calibration_records(
        str(root / "workflows.yaml"), type(reference_workflows()[0]), "workflows"
    )
    validate_calibration_design(tasks, worlds, workflows)
    assert [workflow.model_dump() for workflow in workflows] == [
        workflow.model_dump() for workflow in reference_workflows()
    ]
    assert {workflow.workflow_id for workflow in workflows} == {
        "centralized_raw",
        "local_reduction",
        "visual_reduction",
    }
    assert [task.task_id for task in tasks] == [
        "video_mme:795",
        "video_mme:848",
        "video_mme:747",
    ]
    assert [[chunk.site_id for chunk in task.chunks] for task in tasks] == [
        ["A4", "A5", "A28"],
        ["A4", "A5", "A28"],
        ["A4", "A5", "A28"],
    ]
    assert worlds[0].network.bandwidth_mbps == 3
    assert worlds[1].network.bandwidth_mbps == 100
    assert worlds[0].metadata["rtt_semantics"] == (
        "added_delay_above_shared_physical_baseline"
    )
    assert all(task.sample_count_per_chunk == 12 for task in tasks)
    task_747 = next(task for task in tasks if task.task_id == "video_mme:747")
    assert sum(chunk.raw_bytes for chunk in task_747.chunks) == 330076886
    assert [chunk.observed_duration_s for chunk in task_747.chunks] == [
        1091.341,
        1092.547,
        1093.755,
    ]

    answer_payload = json.loads(
        (root / "evaluator_only" / "answers.json").read_text(encoding="utf-8")
    )
    answer_records = [
        CalibrationEvaluatorRecord.model_validate(item)
        for item in answer_payload["records"]
    ]
    for task, answers in zip(tasks, answer_records, strict=True):
        assert task.task_id == answers.task_id
        assert {question.question_id for question in task.questions} == {
            answer.question_id for answer in answers.answers
        }
        assert "correct_option" not in task.model_dump_json()

    evaluator_747 = next(
        record for record in answer_records if record.task_id == "video_mme:747"
    )
    assert [answer.correct_option for answer in evaluator_747.answers] == ["D", "A", "D"]
