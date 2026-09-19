from __future__ import annotations

import json
from pathlib import Path

import pytest

from infra_bench.real_tasks.scope_expansion_v0 import (
    evaluate_scope_runs,
    summarize_scope_expansion,
    validate_scope_design,
)
from infra_bench.schemas.scope_expansion_v0 import (
    ScopeArtifactMetrics,
    ScopeDemandMetrics,
    ScopeEvaluatorRecord,
    ScopeExecution,
    ScopeLineageRecord,
    ScopeLocalComputeMetrics,
    ScopeNetwork,
    ScopeRun,
    ScopeServiceMetrics,
    ScopeTask,
    ScopeTaskBankRecord,
    ScopeTransferMetrics,
    ScopeUsageMetrics,
    ScopeWorkflow,
    ScopeWorkflowStep,
    ScopeWorld,
)
from infra_bench.scope_expansion.longbench_v2 import (
    MULTIDOC_SOURCE_ID,
    STRUCTURED_SOURCE_ID,
    LongBenchV2Adapter,
    LongBenchV2Row,
    materialize_longbench_v2,
    select_longbench_v2_tasks,
)


def _row(
    source_id: str,
    *,
    domain: str,
    sub_domain: str,
    question: str,
    context: str,
) -> LongBenchV2Row:
    return LongBenchV2Row(
        source_ordinal=0,
        source_id=source_id,
        domain=domain,
        sub_domain=sub_domain,
        difficulty="hard",
        length="long",
        question=question,
        choices={"A": "first", "B": "second", "C": "third", "D": "fourth"},
        answer="D",
        context=context,
    )


def _adapter() -> LongBenchV2Adapter:
    lines = [""] * 17_725
    lines[0] = "1"
    lines[1] = "2020 JD.com Environmental, Social and Governance Report"
    lines[2] = " ".join(["word"] * 100_001)
    lines[3277] = "2021 JD.com Environmental, Social and Governance Report"
    lines[9091] = "About This Report"
    lines[9153] = "2023 JD.com, Inc."
    lines[9154] = "Environmental, Social and Governance Report"
    lines[17721] = "2022"
    lines[17722] = "Environmental, Social and"
    lines[17723] = "Governance Report"
    lines[17724] = "last report body"
    multidoc = _row(
        MULTIDOC_SOURCE_ID,
        domain="Multi-Document QA",
        sub_domain="Financial",
        question="What changed over all four reports?",
        context="\n".join(lines),
    )
    header = (
        "Symbol EndDate IndustryCode TotalAssets TotalLiability IntangibleAsset "
        "NetProfit OperatingEvenue OperatingCost OperationProfit"
    )
    rows = [
        f"{index} 2021-12-31 J66 {1000 + index}.0 {20 + index}.0 0.0 "
        f"{'NaN' if index == 1 else '1.0'} 2.0 3.0 4.0"
        for index in range(1, 101)
    ]
    structured = _row(
        STRUCTURED_SOURCE_ID,
        domain="Long Structured Data Understanding",
        sub_domain="Table QA",
        question="Which company has the maximum asset to liability ratio?",
        context="\n".join([header, *rows]),
    )
    return LongBenchV2Adapter(
        rows=(multidoc, structured),
        source_sha256="fixture",
        dataset_revision="fixture",
    )


def test_selection_uses_natural_boundaries_and_all_sites() -> None:
    selection = select_longbench_v2_tasks(_adapter())

    assert len(selection.documents) == 4
    assert "2020 JD.com" in selection.documents[0].text[:100]
    assert selection.documents[1].text.startswith(
        "2021 JD.com Environmental, Social and Governance Report"
    )
    assert selection.documents[2].text.startswith("About This Report")
    assert selection.documents[3].text.startswith(
        "2022\nEnvironmental, Social and\nGovernance Report"
    )
    assert sum(shard.record_count for shard in selection.structured_shards) == 100
    assert {shard.site for shard in selection.structured_shards} == {"A4", "A5", "A28"}
    all_records = []
    numeric_columns = set(selection.structured_columns) - {
        "Symbol",
        "EndDate",
        "IndustryCode",
    }
    for shard in selection.structured_shards:
        records = json.loads(shard.payload)["records"]
        all_records.extend(records)
        assert len(records) == shard.record_count
        assert all("record_id" in record and "Symbol" in record for record in records)
        assert all(
            all(
                value is None
                or (isinstance(value, (int, float)) and not isinstance(value, bool))
                for column in numeric_columns
                for value in [record[column]]
            )
            for record in records
        )
    assert next(record for record in all_records if record["Symbol"] == "1")[
        "NetProfit"
    ] is None
    assert any(
        record["EndDate"] == "2021-12-31"
        and isinstance(record["TotalLiability"], (int, float))
        and record["TotalLiability"] > 0
        for record in all_records
    )


def test_materialization_separates_gold_and_emits_generic_plan(tmp_path: Path) -> None:
    adapter = _adapter()
    selection = select_longbench_v2_tasks(adapter)
    result = materialize_longbench_v2(adapter, selection, tmp_path)

    assert result["source_ids"] == [MULTIDOC_SOURCE_ID, STRUCTURED_SOURCE_ID]
    records = [
        ScopeTaskBankRecord.model_validate_json(line)
        for line in (tmp_path / "task_bank.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 2
    tasks = {record.planner_visible.task_family: record.planner_visible for record in records}
    multidoc = tasks["multi_document_qa"]
    structured = tasks["structured_data_analysis"]
    assert multidoc.artifact_count == 4
    assert set(multidoc.artifact_placement.values()) == {"A4", "A5", "A28"}
    assert multidoc.retrieval_config is not None
    assert multidoc.retrieval_config.parameters["runtime_local_top_k"] == 1
    assert structured.artifact_count == 3
    assert structured.retrieval_config is None
    assert structured.structured_plan is not None
    assert structured.structured_plan.derived_fields[0].operator == "divide"
    assert structured.structured_plan.order_by[0].direction == "descending"
    assert structured.structured_plan.limit == 1
    assert "asset_liability_ratio" not in structured.structured_plan.select_fields

    visible = "\n".join(
        json.dumps(task.model_dump(mode="json"), sort_keys=True) for task in tasks.values()
    )
    assert '"answer":' not in visible
    evaluator = (tmp_path / "evaluator_only" / "longbench_v2.jsonl").read_text(
        encoding="utf-8"
    )
    assert '"answer":"D"' in evaluator
    assert (tmp_path / "task_selection" / "longbench_selection.md").is_file()
    selection_report = json.loads(
        (tmp_path / "task_selection" / "longbench_selection.json").read_text(
            encoding="utf-8"
        )
    )
    reported_artifacts = [
        *selection_report["multidoc"]["artifacts"],
        *selection_report["structured"]["artifacts"],
    ]
    assert len(reported_artifacts) == 7
    assert all(len(artifact["sha256"]) == 64 for artifact in reported_artifacts)


def _worlds() -> list[ScopeWorld]:
    return [
        ScopeWorld(
            world_id=world_id,
            network=ScopeNetwork(bandwidth_mbps=bandwidth, rtt_ms=rtt),
            final_model_id="edge-text-reasoner",
            device_fingerprints={"A4": "a4", "A5": "a5", "A28": "a28"},
        )
        for world_id, bandwidth, rtt in (
            ("H1_distributed_constrained", 3.0, 83.0),
            ("H2_distributed_favorable", 100.0, 33.0),
        )
    ]


def _structured_workflows() -> list[ScopeWorkflow]:
    return [
        ScopeWorkflow(
            workflow_id="centralized_raw",
            description="Centralize raw records.",
            steps=[
                ScopeWorkflowStep(
                    step_id="reason",
                    operator_id="invoke_model",
                    execution_role="final_reasoning_agent",
                )
            ],
        ),
        ScopeWorkflow(
            workflow_id="distributed_compute",
            description="Compute generic local top-k records.",
            steps=[
                ScopeWorkflowStep(
                    step_id="filter",
                    operator_id="filter_records",
                    execution_role="artifact_local_agent",
                ),
                ScopeWorkflowStep(
                    step_id="reason",
                    operator_id="invoke_model",
                    execution_role="final_reasoning_agent",
                ),
            ],
        ),
    ]


def _structured_run(
    task: ScopeTask,
    workflow: str,
    world: str,
    repeat: int,
) -> ScopeRun:
    distributed = workflow == "distributed_compute"
    counts = {site: list(task.artifact_placement.values()).count(site) for site in ("A4", "A5", "A28")}
    raw_bytes = sum(task.artifact_sizes.values())
    raw_tokens = sum(task.artifact_tokens.values())
    reduced = 300 if distributed else raw_bytes
    operator = "filter_records" if distributed else "invoke_model"
    return ScopeRun(
        run_id=f"structured-{workflow}-{world}-{repeat}",
        task_id=task.task_id,
        task_family="structured_data_analysis",
        workflow_id=workflow,  # type: ignore[arg-type]
        world_id=world,  # type: ignore[arg-type]
        repeat=repeat,
        warmup=repeat == 0,
        measurement_series_id="lb2-series",
        protocol_id="steady_state_1_warmup_3_measured_v1",
        status="completed",
        final_answer="The correct answer is (D)",
        demand=ScopeDemandMetrics(
            artifact_count=task.artifact_count,
            raw_bytes=raw_bytes,
            raw_tokens=raw_tokens,
            documents_per_agent=counts,  # type: ignore[arg-type]
        ),
        local_compute=ScopeLocalComputeMetrics(
            calls=3 if distributed else 0,
            sum_ms=300 if distributed else 0,
            critical_ms=100 if distributed else 0,
        ),
        artifacts=ScopeArtifactMetrics(
            raw_bytes=raw_bytes,
            reduced_bytes=reduced,
            reduced_tokens=30 if distributed else raw_tokens,
            absolute_reducible_bytes=raw_bytes - reduced,
        ),
        transfers=ScopeTransferMetrics(
            count=0,
            bytes=300 if distributed else raw_bytes,
            latency_ms=30 if distributed else 300,
            critical_ms=15 if distributed else 150,
        ),
        service=ScopeServiceMetrics(
            local_preprocessing_ms=300 if distributed else 0,
            final_model_ms=400,
            total_ms=700 if distributed else 400,
        ),
        usage=ScopeUsageMetrics(
            input_tokens=100 if distributed else raw_tokens,
            output_tokens=8,
        ),
        e2e_latency_ms=700 if distributed else 1_000,
        executions=[
            ScopeExecution(
                action_id="work",
                operator_id=operator,  # type: ignore[arg-type]
                executor_id="worker",
                worker_id="a28",
                site_id="A28",
                service_ms=300 if distributed else 400,
                input_bytes=raw_bytes,
                output_bytes=reduced,
            )
        ],
        lineage=[
            ScopeLineageRecord(
                producer_agent="A4",
                consumer_agent="A28",
                input_artifact=task.artifact_ids[0],
                derived_artifact="partial",
                artifact_bytes=reduced,
                operator=operator,  # type: ignore[arg-type]
            )
        ],
    )


def test_longbench_mcq_and_structured_report_are_family_aware(tmp_path: Path) -> None:
    adapter = _adapter()
    materialize_longbench_v2(adapter, select_longbench_v2_tasks(adapter), tmp_path)
    records = [
        ScopeTaskBankRecord.model_validate_json(line)
        for line in (tmp_path / "task_bank.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    task = next(
        record.planner_visible
        for record in records
        if record.planner_visible.task_family == "structured_data_analysis"
    )
    evaluator = next(
        ScopeEvaluatorRecord.model_validate_json(line)
        for line in (tmp_path / "evaluator_only" / "longbench_v2.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if task.task_id in line
    )
    worlds = _worlds()
    workflows = _structured_workflows()
    runs = [
        _structured_run(task, workflow.workflow_id, world.world_id, repeat)
        for world in worlds
        for workflow in workflows
        for repeat in range(4)
    ]
    validate_scope_design([task], [evaluator], worlds, workflows)
    evaluated = evaluate_scope_runs([task], [evaluator], runs)
    assert all(run.quality and run.quality.original_benchmark_score == 1.0 for run in evaluated)
    report = summarize_scope_expansion([task], worlds, workflows, evaluated)
    assert len(report.cells) == 4
    assert set(report.tasks[0].quality_by_workflow) == {
        "centralized_raw",
        "distributed_compute",
    }
    assert "compute_sensitive" in report.tasks[0].labels

    wrong_family = runs[0].model_copy(update={"task_family": "multi_document_qa"})
    with pytest.raises(ValueError, match="task_family does not match"):
        evaluate_scope_runs([task], [evaluator], [wrong_family])
