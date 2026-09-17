from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from ..real_tasks.admission import is_evaluator_verified
from ..schemas import (
    BenchmarkCase,
    DataArtifact,
    Executor,
    InfraState,
    NetworkLink,
    OperatorProfile,
    Site,
    TaskRecord,
    WorkflowRecord,
)
from ..simulator.scheduler import optimize_workflow
from .search import select_oracle


def _executor(
    executor_id: str,
    site_id: str,
    capabilities: list[str],
    load: float,
    cost: float = 0.0,
) -> Executor:
    return Executor(
        executor_id=executor_id,
        site_id=site_id,
        capabilities=capabilities,
        speed_factors={"default": 1.0},
        load=load,
        monetary_cost_per_s=cost,
    )


def _infra(
    task: TaskRecord,
    template: Mapping[str, Any],
    *,
    infra_id: str,
    bandwidth_mbps: float,
    cloud_a_load: float,
    cloud_b_load: float,
    unused_ocr_load: float,
) -> InfraState:
    low_load = float(template["low_load"])
    cloud_cost = float(template.get("cloud_cost_per_s", 0.0))
    if task.input_type == "repository":
        artifact_id = "repository"
        artifact_kind = "repository"
        local_capabilities = [
            "local_repo",
            "repo_access",
            "code_search",
            "static_analysis",
            "code_edit",
            "test_runner",
            "verify",
        ]
        cloud_capabilities = [
            "cloud_full_repo",
            "repo_access",
            "code_search",
            "code_edit",
            "test_runner",
            "reason",
            "verify",
        ]
    elif task.input_type == "video":
        artifact_id = "raw_video"
        artifact_kind = "video"
        local_capabilities = ["video_preprocess"]
        cloud_capabilities = ["strong_vlm", "reason", "verify"]
    else:
        raise ValueError(f"unsupported task input_type: {task.input_type}")

    return InfraState(
        infra_id=infra_id,
        sites=[
            Site(
                site_id="edge",
                executors=[
                    _executor("edge_worker", "edge", local_capabilities, low_load),
                    _executor("unused_ocr", "edge", ["ocr"], unused_ocr_load),
                ],
            ),
            Site(
                site_id="cloud",
                executors=[
                    _executor(
                        "cloud_a", "cloud", cloud_capabilities, cloud_a_load, cloud_cost
                    ),
                    _executor(
                        "cloud_b", "cloud", cloud_capabilities, cloud_b_load, cloud_cost
                    ),
                ],
            ),
        ],
        links=[
            NetworkLink(
                src_site="edge",
                dst_site="cloud",
                rtt_ms=float(template["rtt_ms"]),
                bandwidth_mbps=bandwidth_mbps,
            )
        ],
        artifacts=[
            DataArtifact(
                artifact_id=artifact_id,
                site_id="edge",
                size_mb=float(task.metadata.get("artifact_size_mb", template["artifact_size_mb"])),
                kind=artifact_kind,
            )
        ],
        metadata={
            "template": "coding" if task.input_type == "repository" else "video",
            "artifact_size_is_synthetic": "artifact_size_mb" not in task.metadata,
        },
    )


def _case(
    task: TaskRecord,
    workflows: list[WorkflowRecord],
    infra: InfraState,
    profiles: dict[str, OperatorProfile],
    *,
    case_id: str,
    group_id: str,
    case_type: str,
    perturbation: dict[str, Any],
    seed: int,
) -> BenchmarkCase:
    executions = {
        workflow.workflow_id: optimize_workflow(workflow, infra, profiles)
        for workflow in workflows
    }
    oracle_id, oracle = select_oracle(executions)
    if oracle.metrics is None:
        raise ValueError(f"oracle has no metrics for {case_id}")
    candidate_metrics: dict[str, dict[str, Any]] = {}
    for workflow_id, execution in executions.items():
        if execution.feasible and execution.metrics is not None:
            candidate_metrics[workflow_id] = {
                "feasible": True,
                "assignment": execution.assignment,
                "metrics": execution.metrics.model_dump(),
            }
        else:
            candidate_metrics[workflow_id] = {
                "feasible": False,
                "reason": execution.reason,
            }
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
        candidate_metrics=candidate_metrics,
        task=task,
        infra=infra,
        candidate_workflows=workflows,
        metadata={
            "source_benchmark_version": task.metadata.get("dataset_revision"),
            "source_task_id": task.task_id,
            "workflow_bank_version": "manual-fallback-v1",
            "infra_config": infra.metadata.get("template"),
            "random_seed": seed,
            "generator_version": "0.2.0",
            "synthetic_sanity": False,
        },
    )


def _valid_group(group: list[BenchmarkCase], case_type: str) -> bool:
    workflows = {case.oracle_workflow_id for case in group}
    assignments = {tuple(sorted(case.oracle_assignment.items())) for case in group}
    if case_type == "semantic_switch":
        return len(workflows) > 1
    if case_type == "placement_only":
        return len(workflows) == 1 and len(assignments) > 1
    if case_type == "invariance":
        return len(workflows) == 1 and len(assignments) == 1 and len(
            {tuple(sorted(case.oracle_metrics.items())) for case in group}
        ) == 1
    raise ValueError(f"unknown case_type: {case_type}")


def build_counterfactual_cases(
    tasks: Iterable[TaskRecord],
    workflows: Iterable[WorkflowRecord],
    profiles: dict[str, OperatorProfile],
    templates: Mapping[str, Mapping[str, Any]],
    *,
    seed: int = 42,
) -> list[BenchmarkCase]:
    workflows_by_task: dict[str, list[WorkflowRecord]] = defaultdict(list)
    for workflow in workflows:
        if is_evaluator_verified(workflow):
            workflows_by_task[workflow.task_id].append(workflow)

    output: list[BenchmarkCase] = []
    for index, task in enumerate(tasks):
        candidates = sorted(workflows_by_task[task.task_id], key=lambda item: item.workflow_id)
        if len(candidates) < 2:
            continue
        template_name = "coding" if task.input_type == "repository" else "video"
        template = templates[template_name]
        low_load = float(template["low_load"])
        high_load = float(template["high_load"])
        high_bw = float(template["high_bandwidth_mbps"])
        low_bw = float(template["low_bandwidth_mbps"])
        stem = task.task_id.replace(":", "_").replace("/", "_")
        semantic_cloud_b_load = high_load if index % 2 else low_load

        semantic_group = [
            _case(
                task,
                candidates,
                _infra(
                    task,
                    template,
                    infra_id=f"{stem}_semantic_high",
                    bandwidth_mbps=high_bw,
                    cloud_a_load=low_load,
                    cloud_b_load=semantic_cloud_b_load,
                    unused_ocr_load=low_load,
                ),
                profiles,
                case_id=f"{stem}:semantic:high",
                group_id=f"{stem}:semantic_bandwidth",
                case_type="semantic_switch",
                perturbation={"dimension": "bandwidth_mbps", "value": high_bw},
                seed=seed,
            ),
            _case(
                task,
                candidates,
                _infra(
                    task,
                    template,
                    infra_id=f"{stem}_semantic_low",
                    bandwidth_mbps=low_bw,
                    cloud_a_load=low_load,
                    cloud_b_load=semantic_cloud_b_load,
                    unused_ocr_load=low_load,
                ),
                profiles,
                case_id=f"{stem}:semantic:low",
                group_id=f"{stem}:semantic_bandwidth",
                case_type="semantic_switch",
                perturbation={"dimension": "bandwidth_mbps", "value": low_bw},
                seed=seed,
            ),
        ]
        if _valid_group(semantic_group, "semantic_switch"):
            output.extend(semantic_group)

        if index % 2 == 0:
            placement_group = [
                _case(
                    task,
                    candidates,
                    _infra(
                        task,
                        template,
                        infra_id=f"{stem}_placement_a",
                        bandwidth_mbps=high_bw,
                        cloud_a_load=low_load,
                        cloud_b_load=high_load,
                        unused_ocr_load=low_load,
                    ),
                    profiles,
                    case_id=f"{stem}:placement:a",
                    group_id=f"{stem}:placement_load",
                    case_type="placement_only",
                    perturbation={
                        "dimension": "executor_load",
                        "cloud_a": low_load,
                        "cloud_b": high_load,
                    },
                    seed=seed,
                ),
                _case(
                    task,
                    candidates,
                    _infra(
                        task,
                        template,
                        infra_id=f"{stem}_placement_b",
                        bandwidth_mbps=high_bw,
                        cloud_a_load=high_load,
                        cloud_b_load=low_load,
                        unused_ocr_load=low_load,
                    ),
                    profiles,
                    case_id=f"{stem}:placement:b",
                    group_id=f"{stem}:placement_load",
                    case_type="placement_only",
                    perturbation={
                        "dimension": "executor_load",
                        "cloud_a": high_load,
                        "cloud_b": low_load,
                    },
                    seed=seed,
                ),
            ]
            if _valid_group(placement_group, "placement_only"):
                output.extend(placement_group)
        else:
            invariance_group = [
                _case(
                    task,
                    candidates,
                    _infra(
                        task,
                        template,
                        infra_id=f"{stem}_invariance_low",
                        bandwidth_mbps=high_bw,
                        cloud_a_load=low_load,
                        cloud_b_load=high_load,
                        unused_ocr_load=low_load,
                    ),
                    profiles,
                    case_id=f"{stem}:invariance:low",
                    group_id=f"{stem}:invariance_ocr",
                    case_type="invariance",
                    perturbation={"dimension": "unused_ocr_load", "value": low_load},
                    seed=seed,
                ),
                _case(
                    task,
                    candidates,
                    _infra(
                        task,
                        template,
                        infra_id=f"{stem}_invariance_high",
                        bandwidth_mbps=high_bw,
                        cloud_a_load=low_load,
                        cloud_b_load=high_load,
                        unused_ocr_load=high_load,
                    ),
                    profiles,
                    case_id=f"{stem}:invariance:high",
                    group_id=f"{stem}:invariance_ocr",
                    case_type="invariance",
                    perturbation={"dimension": "unused_ocr_load", "value": high_load},
                    seed=seed,
                ),
            ]
            if _valid_group(invariance_group, "invariance"):
                output.extend(invariance_group)
    return output
