import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from infra_bench.config import load_yaml
from infra_bench.real_tasks.scope_expansion_v0 import (
    evaluate_scope_runs,
    summarize_scope_expansion,
    validate_scope_design,
    write_scope_expansion_report,
)
from infra_bench.schemas import (
    DEFAULT_OPERATOR_REGISTRY,
    GENERAL_MAS_OPERATOR_BINDINGS,
    RealizedWorkflowTrace,
    ScopeArtifactMeasurement,
    ScopeArtifactMetrics,
    ScopeDemandMetrics,
    ScopeEvaluatorRecord,
    ScopeExecution,
    ScopeLineageRecord,
    ScopeNetwork,
    ScopeProfilingProtocol,
    ScopeRetrievalConfig,
    ScopeRetrievalMetrics,
    ScopeRun,
    ScopeServiceMetrics,
    ScopeSiteId,
    ScopeTask,
    ScopeTaskBankEvaluatorSummary,
    ScopeTaskBankRecord,
    ScopeTransferMetrics,
    ScopeTransferRecord,
    ScopeUsageMetrics,
    ScopeWorkflow,
    ScopeWorkflowId,
    ScopeWorkflowStep,
    ScopeWorld,
    ScopeWorldId,
    WorkflowTraceSpan,
    stable_document_site,
)

PROTOCOL = "steady_state_1_warmup_3_measured_v1"


def _task(index: int, evidence_count: int) -> ScopeTask:
    artifact_ids = [f"m{index}-artifact-{item:02d}" for item in range(30)]
    document_ids = {
        artifact_id: f"m{index}-document-{item:02d}"
        for item, artifact_id in enumerate(artifact_ids)
    }
    return ScopeTask(
        task_id=f"multihop_rag:m{evidence_count}",
        dataset="MultiHop-RAG",
        instruction="Answer the question from the candidate documents.",
        query=f"question {index}",
        artifact_ids=artifact_ids,
        artifact_types={artifact_id: "document" for artifact_id in artifact_ids},
        artifact_sizes={artifact_id: 100 for artifact_id in artifact_ids},
        artifact_tokens={artifact_id: 20 for artifact_id in artifact_ids},
        artifact_source_refs={
            artifact_id: f"documents/{artifact_id}.json" for artifact_id in artifact_ids
        },
        artifact_document_ids=document_ids,
        artifact_placement={
            artifact_id: stable_document_site(document_id)
            for artifact_id, document_id in document_ids.items()
        },
        retrieval_config=ScopeRetrievalConfig(),
        artifact_count=30,
        evaluator_type="multihop_rag_official_token_intersection",
    )


def _evaluator(task: ScopeTask, evidence_count: int) -> ScopeEvaluatorRecord:
    return ScopeEvaluatorRecord(
        task_id=task.task_id,
        evaluator_type="multihop_rag_official_token_intersection",
        answer="Alpha Beta",
        supporting_document_ids=list(task.artifact_document_ids.values())[:evidence_count],
        evidence_document_count=evidence_count,
        question_type="multi-hop",
        original_evaluator_information={"script": "qa_evaluate.py"},
    )


def _worlds() -> list[ScopeWorld]:
    return [
        ScopeWorld(
            world_id="H1_distributed_constrained",
            network=ScopeNetwork(bandwidth_mbps=3, rtt_ms=50),
            final_model_id="edge-text-reasoner",
            device_fingerprints={
                "A4": "orin-a4",
                "A5": "orin-a5",
                "A28": "orin-a28",
            },
        ),
        ScopeWorld(
            world_id="H2_distributed_favorable",
            network=ScopeNetwork(bandwidth_mbps=100, rtt_ms=0),
            final_model_id="edge-text-reasoner",
            device_fingerprints={
                "A4": "orin-a4",
                "A5": "orin-a5",
                "A28": "orin-a28",
            },
        ),
    ]


def _workflows() -> list[ScopeWorkflow]:
    return [
        ScopeWorkflow(
            workflow_id="centralized_raw",
            description="Move raw candidate documents to A28 and reason once.",
            steps=[
                ScopeWorkflowStep(
                    step_id="aggregate_raw",
                    operator_id="aggregate_artifacts",
                    execution_role="final_reasoning_agent",
                ),
                ScopeWorkflowStep(
                    step_id="reason",
                    operator_id="invoke_model",
                    execution_role="final_reasoning_agent",
                ),
            ],
        ),
        ScopeWorkflow(
            workflow_id="distributed_retrieval",
            description="Retrieve on each document shard and move evidence to A28.",
            steps=[
                ScopeWorkflowStep(
                    step_id="retrieve",
                    operator_id="bm25_retrieve",
                    execution_role="artifact_local_agent",
                ),
                ScopeWorkflowStep(
                    step_id="aggregate_evidence",
                    operator_id="aggregate_artifacts",
                    execution_role="final_reasoning_agent",
                ),
                ScopeWorkflowStep(
                    step_id="reason",
                    operator_id="invoke_model",
                    execution_role="final_reasoning_agent",
                ),
            ],
        ),
    ]


def _run(
    task: ScopeTask,
    workflow: ScopeWorkflowId,
    world: ScopeWorldId,
    repeat: int,
    *,
    warmup: bool,
    e2e_ms: float,
    answer: str = 'The answer to the question is "Alpha Beta"',
) -> ScopeRun:
    distributed = workflow == "distributed_retrieval"
    documents_per_agent: dict[ScopeSiteId, int] = {"A4": 0, "A5": 0, "A28": 0}
    for site in task.artifact_placement.values():
        documents_per_agent[site] += 1
    raw_bytes = sum(task.artifact_sizes.values())
    raw_tokens = sum(task.artifact_tokens.values())
    reduced_bytes = 300 if distributed else raw_bytes
    return ScopeRun(
        run_id=f"{task.task_id}-{workflow}-{world}-{repeat}-{'w' if warmup else 'm'}",
        task_id=task.task_id,
        workflow_id=workflow,
        world_id=world,
        repeat=repeat,
        warmup=warmup,
        measurement_series_id="formal-series-1",
        protocol_id=PROTOCOL,
        status="completed",
        final_answer=answer,
        demand=ScopeDemandMetrics(
            artifact_count=30,
            raw_bytes=raw_bytes,
            raw_tokens=raw_tokens,
            documents_per_agent=documents_per_agent,
        ),
        retrieval=ScopeRetrievalMetrics(
            top_k_per_shard=2 if distributed else 0,
            calls=3 if distributed else 0,
            sum_ms=300 if distributed else 0,
            critical_ms=100 if distributed else 0,
            retrieved_document_count=6 if distributed else 0,
        ),
        artifacts=ScopeArtifactMetrics(
            raw_bytes=raw_bytes,
            reduced_bytes=reduced_bytes,
            reduced_tokens=60 if distributed else raw_tokens,
            absolute_reducible_bytes=raw_bytes - reduced_bytes,
            records=[
                ScopeArtifactMeasurement(
                    artifact_id=task.artifact_ids[0],
                    document_id=task.artifact_document_ids[task.artifact_ids[0]],
                    kind="raw_document",
                    site_id="A4",
                    bytes=100,
                    tokens=20,
                ),
                ScopeArtifactMeasurement(
                    artifact_id="evidence-0",
                    kind="retrieved_evidence" if distributed else "aggregated_evidence",
                    site_id="A28",
                    bytes=reduced_bytes,
                    tokens=60 if distributed else raw_tokens,
                ),
            ],
        ),
        transfers=ScopeTransferMetrics(
            count=2,
            bytes=200 if distributed else 2_000,
            latency_ms=50 if distributed else 500,
            critical_ms=25 if distributed else 250,
            records=[
                ScopeTransferRecord(
                    transfer_id=f"transfer-{index}",
                    artifact_id=f"payload-{index}",
                    src_agent=site,
                    dst_agent="A28",
                    src_site=site,
                    dst_site="A28",
                    bytes=100 if distributed else 1_000,
                    latency_ms=25 if distributed else 250,
                    producer_action_id="retrieve" if distributed else None,
                    consumer_action_id="reason",
                )
                for index, site in enumerate(("A4", "A5"), start=1)
            ],
        ),
        service=ScopeServiceMetrics(
            local_preprocessing_ms=300 if distributed else 0,
            final_model_ms=400,
            total_ms=700 if distributed else 400,
        ),
        usage=ScopeUsageMetrics(
            input_tokens=100 if distributed else 600,
            output_tokens=10,
        ),
        e2e_latency_ms=e2e_ms,
        executions=[
            ScopeExecution(
                action_id="retrieve" if distributed else "reason",
                operator_id="bm25_retrieve" if distributed else "invoke_model",
                executor_id="worker",
                worker_id="worker-a28",
                site_id="A28",
                service_ms=300 if distributed else 400,
                input_bytes=raw_bytes,
                output_bytes=reduced_bytes,
            )
        ],
        lineage=[
            ScopeLineageRecord(
                producer_agent="A4",
                consumer_agent="A28",
                input_artifact=task.artifact_ids[0],
                derived_artifact="evidence-0" if distributed else task.artifact_ids[0],
                artifact_bytes=100,
                operator="bm25_retrieve" if distributed else "read_artifact",
            )
        ],
        trace=RealizedWorkflowTrace(
            run_id=f"{task.task_id}-{workflow}-{world}-{repeat}-{'w' if warmup else 'm'}",
            task_id=task.task_id,
            workflow_id=workflow,
            warmup=warmup,
            dependency_evidence="partial",
            timestamp_evidence="partial",
            trace_coverage="partial",
            spans=[
                WorkflowTraceSpan(
                    span_id="action-1",
                    span_kind="action",
                    name="reference workflow action",
                    duration_ms=300 if distributed else 400,
                    service_scope="local" if distributed else "remote",
                    operator_id="bm25_retrieve" if distributed else "invoke_model",
                    executor_id="worker",
                    site_id="A28",
                    input_bytes=raw_bytes,
                    output_bytes=reduced_bytes,
                )
            ],
            e2e_latency_ms=e2e_ms,
        ),
    )


def _experiment() -> tuple[
    list[ScopeTask],
    list[ScopeEvaluatorRecord],
    list[ScopeWorld],
    list[ScopeWorkflow],
    list[ScopeRun],
]:
    tasks = [_task(index, evidence) for index, evidence in enumerate((2, 3, 4), start=1)]
    evaluators = [
        _evaluator(task, evidence)
        for task, evidence in zip(tasks, (2, 3, 4), strict=True)
    ]
    worlds = _worlds()
    workflows = _workflows()
    runs: list[ScopeRun] = []
    for task in tasks:
        for world in worlds:
            for workflow in workflows:
                is_h1 = world.world_id == "H1_distributed_constrained"
                central = workflow.workflow_id == "centralized_raw"
                e2e = 2_000 if is_h1 and central else 1_000
                if not is_h1 and central:
                    e2e = 500
                for repeat in range(4):
                    answer = (
                        "no-overlap"
                        if task.task_id.endswith("m4") and not central
                        else 'The answer to the question is "Alpha elaboration"'
                    )
                    runs.append(
                        _run(
                            task,
                            workflow.workflow_id,
                            world.world_id,
                            repeat,
                            warmup=repeat == 0,
                            e2e_ms=e2e,
                            answer=answer,
                        )
                    )
    return tasks, evaluators, worlds, workflows, runs


def test_task_and_evaluator_visibility_are_strictly_separated() -> None:
    task = _task(1, 2)
    serialized = task.model_dump_json()
    assert "supporting_document_ids" not in serialized
    assert "evidence_document_count" not in serialized
    assert "answer" not in task.metadata
    bad = task.model_dump()
    bad["metadata"] = {"gold_answer": "leak"}
    with pytest.raises(ValidationError, match="evaluator-only"):
        ScopeTask.model_validate(bad)
    envelope = ScopeTaskBankRecord(
        task_id=task.task_id,
        planner_visible=task,
        evaluator_only=ScopeTaskBankEvaluatorSummary(
            evidence_document_count=2,
            evaluator_type="multihop_rag_official_token_intersection",
        ),
    )
    assert envelope.evaluator_only.evidence_document_count == 2
    assert "supporting_document_ids" not in envelope.model_dump_json()


def test_stable_placement_and_operator_are_shared_contracts() -> None:
    assert stable_document_site("same-document") == stable_document_site("same-document")
    assert DEFAULT_OPERATOR_REGISTRY.contains("bm25_retrieve")
    assert "bm25_retrieve" in GENERAL_MAS_OPERATOR_BINDINGS
    bad = _task(1, 2).model_dump()
    first = bad["artifact_ids"][0]
    bad["artifact_placement"][first] = (
        "A5" if bad["artifact_placement"][first] != "A5" else "A4"
    )
    with pytest.raises(ValidationError, match="stable_hash"):
        ScopeTask.model_validate(bad)


def test_official_evaluator_is_token_intersection_not_exact_match() -> None:
    tasks, evaluators, _, _, runs = _experiment()
    evaluated = evaluate_scope_runs(tasks, evaluators, [runs[0]])[0]
    assert evaluated.quality is not None
    assert evaluated.quality.original_benchmark_score == 1.0
    assert evaluated.quality.canonical_answer_correct is False
    assert evaluated.quality.exact_match is False

    answer_prefix = runs[0].model_copy(update={"final_answer": "ANSWER: Alpha Beta"})
    evaluated_prefix = evaluate_scope_runs(tasks, evaluators, [answer_prefix])[0]
    assert evaluated_prefix.quality is not None
    assert evaluated_prefix.quality.original_benchmark_score == 1.0
    assert evaluated_prefix.quality.canonical_answer_correct is True


def test_mas_raw_row_round_trips_through_strict_scope_schema() -> None:
    task = _task(1, 2)
    row = _run(
        task,
        "distributed_retrieval",
        "H1_distributed_constrained",
        1,
        warmup=False,
        e2e_ms=1_000,
    ).model_dump(mode="json")
    parsed = ScopeRun.model_validate(row)
    assert parsed.retrieval is not None
    assert parsed.retrieval.algorithm == "bm25"
    assert parsed.transfers is not None
    assert parsed.transfers.records[0].producer_action_id == "retrieve"
    assert parsed.executions[0].worker_id == "worker-a28"
    assert parsed.trace is not None and parsed.trace.spans[0].span_kind == "action"


def test_design_requires_gold_coverage_but_keeps_it_off_runtime_task() -> None:
    tasks, evaluators, worlds, workflows, _ = _experiment()
    validate_scope_design(tasks, evaluators, worlds, workflows)
    bad = evaluators[0].model_copy(
        update={"supporting_document_ids": ["not-retrieved", "also-not-retrieved"]}
    )
    with pytest.raises(ValueError, match="does not cover gold"):
        validate_scope_design(tasks, [bad, *evaluators[1:]], worlds, workflows)


def test_warmup_exclusion_quality_gate_and_multilabel_sensitivity() -> None:
    tasks, evaluators, worlds, workflows, runs = _experiment()
    evaluated = evaluate_scope_runs(tasks, evaluators, runs)
    report = summarize_scope_expansion(tasks, worlds, workflows, evaluated)
    assert report.totals["warmup_run_count"] == 12
    assert report.totals["measured_run_count"] == 36
    assert report.totals["protocol_complete_cell_count"] == 12
    assert all(
        cell.medians["observed_effective_transfer_mbps"] is not None
        and abs(float(cell.medians["observed_effective_transfer_mbps"]) - 0.032) < 1e-9
        for cell in report.cells
    )
    m2 = next(task for task in report.tasks if task.task_id.endswith("m2"))
    assert "workflow_reversal" in m2.labels
    assert "communication_sensitive" in m2.labels
    assert "representation_sensitive" in m2.labels
    assert "parallelism_sensitive" in m2.labels
    assert "context_cost_sensitive" in m2.labels
    m4 = next(task for task in report.tasks if task.task_id.endswith("m4"))
    assert "quality_risk" in m4.labels
    assert m4.workflow_preference["H1_distributed_constrained"] is None


def test_report_does_not_mutate_runtime_owned_raw_runs(tmp_path: Path) -> None:
    tasks, evaluators, worlds, workflows, runs = _experiment()
    output = tmp_path / "scope_expansion_v0"
    output.mkdir()
    raw = output / "raw_runs.jsonl"
    raw.write_text("runtime-owned-sentinel\n", encoding="utf-8")
    report = write_scope_expansion_report(
        tasks, evaluators, worlds, workflows, runs, output
    )
    assert raw.read_text(encoding="utf-8") == "runtime-owned-sentinel\n"
    assert {path.name for path in output.iterdir()} == {
        "raw_runs.jsonl",
        "evaluated_runs.jsonl",
        "sensitivity_summary.json",
        "sensitivity_summary.md",
    }
    payload = json.loads((output / "sensitivity_summary.json").read_text())
    assert payload["totals"] == report.totals


def test_checked_in_scope_world_workflow_and_protocol_configs() -> None:
    root = Path(__file__).parents[1] / "configs" / "scope_expansion_v0"
    world_payload = load_yaml(root / "worlds.yaml")
    workflow_payload = load_yaml(root / "workflows.yaml")
    worlds = [ScopeWorld.model_validate(item) for item in world_payload["worlds"]]
    workflows = [
        ScopeWorkflow.model_validate(item) for item in workflow_payload["workflows"]
    ]
    protocol = ScopeProfilingProtocol.model_validate(
        load_yaml(root / "profiling.yaml")
    )
    assert [(world.network.bandwidth_mbps, world.network.rtt_ms) for world in worlds] == [
        (3.0, 83.0),
        (100.0, 33.0),
    ]
    assert {world.final_model_id for world in worlds} == {"edge-text-reasoner"}
    assert [step.operator_id for step in workflows[0].steps] == ["invoke_model"]
    assert [step.operator_id for step in workflows[1].steps] == [
        "bm25_retrieve",
        "invoke_model",
    ]
    assert protocol.warmup_runs == 1 and protocol.measured_runs == 3
