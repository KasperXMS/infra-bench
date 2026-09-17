"""Build the real-system V1 bridge smoke-test cases."""

from __future__ import annotations

from pathlib import Path

from infra_bench.schemas import (
    BenchmarkCase,
    DataArtifact,
    Executor,
    InfraState,
    NetworkLink,
    Site,
    TaskRecord,
    WorkflowEdge,
    WorkflowNode,
    WorkflowRecord,
    basic_task_interaction_spec,
)

V1_GROUP_ID = "v1-six-image-real-system-smoke"
V1_TASK_ID = "v1-blue-airplane-and-truck"

_CALIBRATED_REFERENCE_COSTS_MS = {
    "colocated": {
        "single-six-image": 53_375.85849990137,
        "three-pairs-plus-synthesis": 25_733.939200057648,
    },
    "distributed": {
        "single-six-image": 55_877.314100041986,
        "three-pairs-plus-synthesis": 34_851.07189998962,
    },
}


def _sites() -> list[Site]:
    return [
        Site(
            site_id=site_id,
            executors=[
                Executor(
                    executor_id=f"{site_id.lower()}-vlm",
                    site_id=site_id,
                    capabilities=["edge-vlm"],
                    speed_factors={"vlm": 1.0},
                    load=0.0,
                )
            ],
        )
        for site_id in ("A4", "A5", "A28")
    ]


def _links() -> list[NetworkLink]:
    return [
        NetworkLink(src_site=src, dst_site=dst, bandwidth_mbps=100.0, rtt_ms=2.0)
        for src in ("A4", "A5", "A28")
        for dst in ("A4", "A5", "A28")
        if src != dst
    ]


def _reference_workflows() -> list[WorkflowRecord]:
    single = WorkflowRecord(
        workflow_id="single-six-image",
        task_id=V1_TASK_ID,
        source="hidden-calibration",
        nodes=[
            WorkflowNode(
                node_id="inspect-all",
                operator="invoke_model",
                input_artifacts=[f"img_{index:02d}" for index in range(1, 7)],
                output_artifacts=["answer"],
                required_capabilities=["edge-vlm"],
            )
        ],
        success=True,
    )
    pair_nodes = [
        WorkflowNode(
            node_id=f"inspect-pair-{pair}",
            operator="invoke_model",
            input_artifacts=[f"img_{2 * pair - 1:02d}", f"img_{2 * pair:02d}"],
            output_artifacts=[f"pair-{pair}-finding"],
            required_capabilities=["edge-vlm"],
        )
        for pair in range(1, 4)
    ]
    synthesis = WorkflowRecord(
        workflow_id="three-pairs-plus-synthesis",
        task_id=V1_TASK_ID,
        source="hidden-calibration",
        nodes=[
            *pair_nodes,
            WorkflowNode(
                node_id="synthesize",
                operator="invoke_model",
                input_artifacts=[f"pair-{pair}-finding" for pair in range(1, 4)],
                output_artifacts=["answer"],
                required_capabilities=["edge-vlm"],
            ),
        ],
        edges=[
            WorkflowEdge(
                src=f"inspect-pair-{pair}",
                dst="synthesize",
                artifact=f"pair-{pair}-finding",
            )
            for pair in range(1, 4)
        ],
        success=True,
    )
    return [single, synthesis]


def build_v1_smoke_cases(image_directory: Path) -> list[BenchmarkCase]:
    """Create paired colocated/distributed cases from the six V1 images."""
    image_directory = image_directory.resolve()
    images = [image_directory / f"V1_img_{index:02d}.jpg" for index in range(1, 7)]
    missing = [path for path in images if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing V1 image(s): {', '.join(map(str, missing))}")

    task = TaskRecord(
        task_id=V1_TASK_ID,
        source="v1-real-system-pilot",
        instruction=(
            "Find the image that contains both a blue airplane and a truck. "
            "Return the image ID and briefly justify the answer."
        ),
        input_type="image-set",
        artifact_refs=[str(path) for path in images],
        evaluator_type="exact_image_id",
        evaluator_config={"image_id": "img_06"},
        interaction_spec=basic_task_interaction_spec(
            task_id=V1_TASK_ID,
            objective=(
                "Find the image that contains both a blue airplane and a truck. "
                "Return the image ID and briefly justify the answer."
            ),
            artifacts=[
                (f"img_{index:02d}", str(path))
                for index, path in enumerate(images, start=1)
            ],
            operators=["invoke_model"],
            evaluator_id="exact_image_id",
        ),
    )
    workflows = _reference_workflows()
    workflow_ids = [workflow.workflow_id for workflow in workflows]
    placements = {
        "colocated": ["A28"] * 6,
        "distributed": ["A4", "A4", "A5", "A5", "A28", "A28"],
    }
    cases: list[BenchmarkCase] = []
    for world, sites in placements.items():
        reference_costs = _CALIBRATED_REFERENCE_COSTS_MS[world]
        oracle = min(reference_costs, key=reference_costs.__getitem__)
        artifacts = [
            DataArtifact(
                artifact_id=f"img_{index:02d}",
                site_id=site_id,
                size_mb=path.stat().st_size / 1_000_000,
                kind="image/jpeg",
            )
            for index, (path, site_id) in enumerate(zip(images, sites, strict=True), start=1)
        ]
        cases.append(
            BenchmarkCase(
                case_id=f"{V1_TASK_ID}-{world}",
                group_id=V1_GROUP_ID,
                task_id=V1_TASK_ID,
                infra_id=f"v1-{world}",
                candidate_workflow_ids=workflow_ids,
                perturbation={"initial_placement": world},
                case_type="invariance",
                oracle_workflow_id=oracle,
                oracle_assignment={},
                oracle_metrics={"latency_ms": reference_costs[oracle]},
                candidate_metrics={
                    workflow_id: {"latency_ms": cost}
                    for workflow_id, cost in reference_costs.items()
                },
                task=task,
                infra=InfraState(
                    infra_id=f"v1-{world}",
                    sites=_sites(),
                    links=_links(),
                    artifacts=artifacts,
                    metadata={"world": world},
                ),
                candidate_workflows=workflows,
                metadata={
                    "world": world,
                    "calibrated_reference_costs_ms": reference_costs,
                    "calibration_preference_reversed": False,
                    "admission": "generic_infra_sensitivity_smoke",
                },
            )
        )
    return cases
