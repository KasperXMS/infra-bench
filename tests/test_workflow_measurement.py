from infra_bench.real_tasks.measurement import reconstruct_workflow_metrics
from infra_bench.schemas import RealizedWorkflowTrace, WorkflowTraceSpan


def _span(
    span_id: str,
    kind: str,
    scope: str,
    start: float,
    finish: float,
    *,
    depends_on: list[str] | None = None,
    input_artifacts: list[str] | None = None,
    output_artifacts: list[str] | None = None,
    artifact_id: str | None = None,
    producer: str | None = None,
    consumer: str | None = None,
    byte_count: int = 0,
    site: str | None = None,
) -> WorkflowTraceSpan:
    return WorkflowTraceSpan(
        span_id=span_id,
        span_kind=kind,
        name=span_id,
        service_scope=scope,
        started_at=start,
        finished_at=finish,
        depends_on=depends_on or [],
        input_artifacts=input_artifacts or [],
        output_artifacts=output_artifacts or [],
        artifact_id=artifact_id,
        producer_span_id=producer,
        consumer_span_id=consumer,
        bytes=byte_count,
        input_bytes=byte_count if input_artifacts else 0,
        output_bytes=byte_count if output_artifacts else 0,
        site_id=site,
        operator_id="invoke_model" if kind == "action" else None,
    )


def _complete_trace() -> RealizedWorkflowTrace:
    return RealizedWorkflowTrace(
        run_id="run-1",
        task_id="task-1",
        workflow_id="arbitrary-realized-workflow",
        dependency_evidence="complete",
        timestamp_evidence="complete",
        trace_coverage="complete",
        e2e_latency_ms=310,
        quality=1.0,
        spans=[
            _span("planner", "planner", "planner", 0, 10),
            _span(
                "local-a",
                "action",
                "local",
                10,
                110,
                depends_on=["planner"],
                output_artifacts=["evidence-a"],
                byte_count=100,
                site="A4",
            ),
            _span(
                "local-b",
                "action",
                "local",
                10,
                210,
                depends_on=["planner"],
                output_artifacts=["evidence-b"],
                byte_count=200,
                site="A5",
            ),
            _span(
                "transfer-a",
                "transfer",
                "other",
                110,
                130,
                depends_on=["local-a"],
                artifact_id="evidence-a",
                producer="local-a",
                consumer="remote",
                byte_count=100,
            ),
            _span(
                "transfer-b",
                "transfer",
                "other",
                210,
                240,
                depends_on=["local-b"],
                artifact_id="evidence-b",
                producer="local-b",
                consumer="remote",
                byte_count=200,
            ),
            _span(
                "remote",
                "action",
                "remote",
                240,
                290,
                depends_on=["transfer-a", "transfer-b"],
                input_artifacts=["evidence-a", "evidence-b"],
                byte_count=300,
                site="4090",
            ),
        ],
    )


def test_reconstructs_sum_and_timestamped_dependency_critical_path() -> None:
    metrics = reconstruct_workflow_metrics(_complete_trace())
    assert metrics.local_preprocessing_sum_ms == 300
    assert metrics.local_preprocessing_critical_ms == 200
    assert metrics.transfer_sum_ms == 50
    assert metrics.transfer_critical_ms == 30
    assert metrics.service_sum_ms == 350
    assert metrics.service_critical_ms == 250
    assert metrics.planner_sum_ms == 10
    assert metrics.planner_critical_ms == 10
    assert metrics.critical_path_ms == 290
    assert metrics.critical_path_span_ids == [
        "planner",
        "local-b",
        "transfer-b",
        "remote",
    ]
    assert metrics.invocation_count == 3
    assert metrics.transfer_bytes == 300
    assert metrics.site_action_counts == {"4090": 1, "A4": 1, "A5": 1}
    assert metrics.operator_counts == {"invoke_model": 3}
    assert metrics.transfer_site_pair_bytes == {"unknown->unknown": 300}
    assert {item.artifact_id for item in metrics.artifact_lineage} == {
        "evidence-a",
        "evidence-b",
    }


def test_critical_metrics_fail_closed_when_dependency_evidence_is_partial() -> None:
    trace = _complete_trace().model_copy(update={"dependency_evidence": "partial"})
    metrics = reconstruct_workflow_metrics(trace)
    assert metrics.local_preprocessing_sum_ms == 300
    assert metrics.transfer_sum_ms == 50
    assert metrics.service_sum_ms == 350
    assert metrics.local_preprocessing_critical_ms is None
    assert metrics.transfer_critical_ms is None
    assert metrics.service_critical_ms is None
    assert metrics.critical_path_ms is None
    assert metrics.critical_path_availability == "unavailable"
    assert "dependency_evidence_not_complete" in metrics.critical_path_unavailable_reasons
