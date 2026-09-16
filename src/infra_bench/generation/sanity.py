from collections.abc import Iterable

from ..schemas import (
    BenchmarkCase,
    DataArtifact,
    Executor,
    InfraState,
    NetworkLink,
    OperatorProfile,
    Site,
    TaskRecord,
    WorkflowEdge,
    WorkflowNode,
    WorkflowRecord,
)
from ..simulator.scheduler import optimize_workflow
from .search import select_oracle


def _metrics_payload(result: object) -> dict:
    if not getattr(result, "feasible"):
        return {"feasible": False, "reason": getattr(result, "reason")}
    return {
        "feasible": True,
        "assignment": getattr(result, "assignment"),
        "metrics": getattr(result, "metrics").model_dump(),
    }


def _case(
    *,
    case_id: str,
    group_id: str,
    case_type: str,
    task: TaskRecord,
    infra: InfraState,
    workflows: Iterable[WorkflowRecord],
    profiles: dict[str, OperatorProfile],
    perturbation: dict,
    seed: int,
) -> BenchmarkCase:
    workflows = list(workflows)
    executions = {
        workflow.workflow_id: optimize_workflow(workflow, infra, profiles)
        for workflow in workflows
    }
    oracle_id, oracle = select_oracle(executions)
    assert oracle.metrics is not None
    return BenchmarkCase(
        case_id=case_id,
        group_id=group_id,
        task_id=task.task_id,
        infra_id=infra.infra_id,
        candidate_workflow_ids=[workflow.workflow_id for workflow in workflows],
        perturbation=perturbation,
        case_type=case_type,
        oracle_workflow_id=oracle_id,
        oracle_assignment=oracle.assignment,
        oracle_metrics=oracle.metrics.model_dump(),
        candidate_metrics={
            workflow_id: _metrics_payload(result)
            for workflow_id, result in executions.items()
        },
        task=task,
        infra=infra,
        candidate_workflows=workflows,
        metadata={
            "source_benchmark_version": "synthetic-v1",
            "source_task_id": task.task_id,
            "workflow_bank_version": "synthetic-v1",
            "infra_config": "configs/mvp.yaml",
            "random_seed": seed,
            "generator_version": "0.1.0",
            "synthetic_sanity": True,
        },
    )


def _video_group(profiles: dict[str, OperatorProfile], seed: int) -> list[BenchmarkCase]:
    task = TaskRecord(
        task_id="sanity_video",
        source="synthetic",
        instruction="Answer a question about a 500 MB video.",
        input_type="video",
        artifact_refs=["raw_video"],
        evaluator_type="synthetic",
    )
    direct = WorkflowRecord(
        workflow_id="video_direct_vlm",
        task_id=task.task_id,
        source="synthetic",
        nodes=[
            WorkflowNode(
                node_id="vlm",
                operator="strong_vlm",
                input_artifacts=["raw_video"],
                output_artifacts=["answer"],
            )
        ],
        success=True,
        provenance={"type": "manual_fallback"},
    )
    filtered = WorkflowRecord(
        workflow_id="video_sample_then_vlm",
        task_id=task.task_id,
        source="synthetic",
        nodes=[
            WorkflowNode(
                node_id="sample",
                operator="sample_frames",
                input_artifacts=["raw_video"],
                output_artifacts=["frames"],
            ),
            WorkflowNode(
                node_id="vlm",
                operator="strong_vlm",
                input_artifacts=["frames"],
                output_artifacts=["answer"],
            ),
        ],
        edges=[WorkflowEdge(src="sample", dst="vlm", artifact="frames")],
        success=True,
        provenance={"type": "manual_fallback"},
    )

    def infra(bandwidth: float, suffix: str) -> InfraState:
        return InfraState(
            infra_id=f"video_{suffix}",
            sites=[
                Site(
                    site_id="edge",
                    executors=[
                        Executor(
                            executor_id="edge_cpu",
                            site_id="edge",
                            capabilities=["video_preprocess"],
                            speed_factors={"default": 1.0},
                            load=0.1,
                        )
                    ],
                ),
                Site(
                    site_id="cloud",
                    executors=[
                        Executor(
                            executor_id="cloud_vlm",
                            site_id="cloud",
                            capabilities=["strong_vlm"],
                            speed_factors={"default": 1.0},
                            load=0.1,
                            monetary_cost_per_s=0.02,
                        )
                    ],
                ),
            ],
            links=[
                NetworkLink(
                    src_site="edge",
                    dst_site="cloud",
                    rtt_ms=20,
                    bandwidth_mbps=bandwidth,
                )
            ],
            artifacts=[
                DataArtifact(
                    artifact_id="raw_video", site_id="edge", size_mb=500, kind="video"
                )
            ],
        )

    workflows = [direct, filtered]
    return [
        _case(
            case_id="semantic_high_bw",
            group_id="semantic_video_bandwidth",
            case_type="semantic_switch",
            task=task,
            infra=infra(10_000, "high_bw"),
            workflows=workflows,
            profiles=profiles,
            perturbation={"dimension": "bandwidth_mbps", "value": 10_000},
            seed=seed,
        ),
        _case(
            case_id="semantic_low_bw",
            group_id="semantic_video_bandwidth",
            case_type="semantic_switch",
            task=task,
            infra=infra(20, "low_bw"),
            workflows=workflows,
            profiles=profiles,
            perturbation={"dimension": "bandwidth_mbps", "value": 20},
            seed=seed,
        ),
    ]


def _placement_group(profiles: dict[str, OperatorProfile], seed: int) -> list[BenchmarkCase]:
    task = TaskRecord(
        task_id="sanity_reasoning",
        source="synthetic",
        instruction="Reason over a compact local fact set.",
        input_type="text",
        artifact_refs=["facts"],
        evaluator_type="synthetic",
    )
    workflow = WorkflowRecord(
        workflow_id="reason_once",
        task_id=task.task_id,
        source="synthetic",
        nodes=[
            WorkflowNode(
                node_id="reason",
                operator="reason",
                input_artifacts=["facts"],
                output_artifacts=["answer"],
            )
        ],
        success=True,
        provenance={"type": "manual_fallback"},
    )

    def infra(load_a: float, load_b: float, suffix: str) -> InfraState:
        return InfraState(
            infra_id=f"placement_{suffix}",
            sites=[
                Site(
                    site_id="local",
                    executors=[
                        Executor(
                            executor_id="reason_a",
                            site_id="local",
                            capabilities=["reason"],
                            speed_factors={"default": 1.0},
                            load=load_a,
                        ),
                        Executor(
                            executor_id="reason_b",
                            site_id="local",
                            capabilities=["reason"],
                            speed_factors={"default": 1.0},
                            load=load_b,
                        ),
                    ],
                )
            ],
            artifacts=[
                DataArtifact(artifact_id="facts", site_id="local", size_mb=0.01, kind="text")
            ],
        )

    return [
        _case(
            case_id="placement_a_free",
            group_id="placement_reason_load",
            case_type="placement_only",
            task=task,
            infra=infra(0.1, 0.8, "a_free"),
            workflows=[workflow],
            profiles=profiles,
            perturbation={"dimension": "executor_load", "reason_a": 0.1, "reason_b": 0.8},
            seed=seed,
        ),
        _case(
            case_id="placement_b_free",
            group_id="placement_reason_load",
            case_type="placement_only",
            task=task,
            infra=infra(0.9, 0.1, "b_free"),
            workflows=[workflow],
            profiles=profiles,
            perturbation={"dimension": "executor_load", "reason_a": 0.9, "reason_b": 0.1},
            seed=seed,
        ),
    ]


def _invariance_group(profiles: dict[str, OperatorProfile], seed: int) -> list[BenchmarkCase]:
    task = TaskRecord(
        task_id="sanity_invariance",
        source="synthetic",
        instruction="Reason over a compact local fact set while an unrelated OCR worker changes load.",
        input_type="text",
        artifact_refs=["facts"],
        evaluator_type="synthetic",
    )
    workflow = WorkflowRecord(
        workflow_id="invariant_reason",
        task_id=task.task_id,
        source="synthetic",
        nodes=[
            WorkflowNode(
                node_id="reason",
                operator="reason",
                input_artifacts=["facts"],
                output_artifacts=["answer"],
            )
        ],
        success=True,
        provenance={"type": "manual_fallback"},
    )

    def infra(ocr_load: float, suffix: str) -> InfraState:
        return InfraState(
            infra_id=f"invariance_{suffix}",
            sites=[
                Site(
                    site_id="local",
                    executors=[
                        Executor(
                            executor_id="reason_fixed",
                            site_id="local",
                            capabilities=["reason"],
                            speed_factors={"default": 1.0},
                            load=0.2,
                        ),
                        Executor(
                            executor_id="unused_ocr",
                            site_id="local",
                            capabilities=["ocr"],
                            speed_factors={"default": 1.0},
                            load=ocr_load,
                        ),
                    ],
                )
            ],
            artifacts=[
                DataArtifact(artifact_id="facts", site_id="local", size_mb=0.01, kind="text")
            ],
        )

    return [
        _case(
            case_id="invariance_ocr_low",
            group_id="invariance_unused_ocr",
            case_type="invariance",
            task=task,
            infra=infra(0.1, "ocr_low"),
            workflows=[workflow],
            profiles=profiles,
            perturbation={"dimension": "unused_executor_load", "value": 0.1},
            seed=seed,
        ),
        _case(
            case_id="invariance_ocr_high",
            group_id="invariance_unused_ocr",
            case_type="invariance",
            task=task,
            infra=infra(0.9, "ocr_high"),
            workflows=[workflow],
            profiles=profiles,
            perturbation={"dimension": "unused_executor_load", "value": 0.9},
            seed=seed,
        ),
    ]


def build_sanity_cases(
    profiles: dict[str, OperatorProfile], *, seed: int = 42
) -> list[BenchmarkCase]:
    return [
        *_video_group(profiles, seed),
        *_placement_group(profiles, seed),
        *_invariance_group(profiles, seed),
    ]

