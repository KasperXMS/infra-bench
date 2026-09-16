
from infra_bench.schemas import (
    DataArtifact,
    Executor,
    InfraState,
    NetworkLink,
    Site,
    WorkflowNode,
    WorkflowRecord,
)
from infra_bench.simulator.execution import execution_time_s
from infra_bench.simulator.network import transfer_time_s
from infra_bench.simulator.scheduler import optimize_workflow


def test_lower_bandwidth_never_reduces_transfer_time():
    assert transfer_time_s(100, 20, 10) > transfer_time_s(100, 20, 1000)


def test_higher_load_never_reduces_execution_time(profiles):
    low = Executor(
        executor_id="low",
        site_id="s",
        capabilities=["reason"],
        speed_factors={},
        load=0.1,
    )
    high = low.model_copy(update={"executor_id": "high", "load": 0.9})
    assert execution_time_s(profiles["reason"], high) > execution_time_s(
        profiles["reason"], low
    )


def test_locality_eliminates_edge_transfer(profiles):
    workflow = WorkflowRecord(
        workflow_id="w",
        task_id="t",
        source="test",
        nodes=[
            WorkflowNode(
                node_id="n",
                operator="reason",
                input_artifacts=["input"],
                output_artifacts=["output"],
            )
        ],
        success=True,
    )
    sites = [
        Site(
            site_id="a",
            executors=[
                Executor(
                    executor_id="worker",
                    site_id="a",
                    capabilities=["reason"],
                    speed_factors={},
                    load=0.1,
                )
            ],
        ),
        Site(site_id="b", executors=[]),
    ]
    base = InfraState(
        infra_id="remote",
        sites=sites,
        links=[NetworkLink(src_site="a", dst_site="b", rtt_ms=20, bandwidth_mbps=10)],
        artifacts=[DataArtifact(artifact_id="input", site_id="b", size_mb=10, kind="text")],
    )
    remote = optimize_workflow(workflow, base, profiles)
    local_infra = base.model_copy(deep=True, update={"infra_id": "local"})
    local_infra.artifacts[0].site_id = "a"
    local = optimize_workflow(workflow, local_infra, profiles)
    assert remote.metrics is not None and local.metrics is not None
    assert remote.metrics.wan_mb == 10
    assert local.metrics.wan_mb == 0
    assert local.metrics.latency_s < remote.metrics.latency_s


def test_capability_filtering_reports_infeasible(profiles):
    workflow = WorkflowRecord(
        workflow_id="w",
        task_id="t",
        source="test",
        nodes=[WorkflowNode(node_id="n", operator="strong_vlm")],
        success=True,
    )
    infra = InfraState(
        infra_id="i",
        sites=[
            Site(
                site_id="s",
                executors=[
                    Executor(
                        executor_id="cpu",
                        site_id="s",
                        capabilities=["reason"],
                        speed_factors={},
                        load=0.1,
                    )
                ],
            )
        ],
    )
    result = optimize_workflow(workflow, infra, profiles)
    assert not result.feasible
    assert "no capable executor" in (result.reason or "")

