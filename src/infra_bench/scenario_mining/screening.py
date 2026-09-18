"""Plausible infrastructure candidates and screening-only cost estimates."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from infra_bench.schemas import (
    InfrastructureCandidate,
    ModeledStrategyCost,
    ScenarioRecord,
    WorkflowDemandTemplate,
)

DEFAULT_SCREENING_ASSUMPTIONS: dict[str, float] = {
    # Transparent coarse service model; none of these values enter D(G).
    "central_reasoning_base_ms": 400.0,
    "reduction_base_ms": 250.0,
    "synthesis_base_ms": 400.0,
    "central_reasoning_bytes_per_s": 50_000.0,
    "reduction_bytes_per_s": 100_000.0,
    "local_reducer_replicas": 3.0,
}


def infrastructure_candidates(
    scenario: ScenarioRecord,
) -> tuple[InfrastructureCandidate, InfrastructureCandidate]:
    """Generate two non-extreme worlds grounded in existing project profiles."""

    edge_sites = ("A4", "A5", "A28")
    distributed_sites = {
        artifact.artifact_id: edge_sites[index % len(edge_sites)]
        for index, artifact in enumerate(scenario.artifacts)
    }
    distributed = InfrastructureCandidate(
        world_id="H_distributed",
        artifact_sites=distributed_sites,
        reasoner_site="cloud",
        reduction_sites=list(edge_sites),
        bandwidth_mbps=20.0,
        rtt_ms=20.0,
        basis=(
            "Existing infra-bench constrained-link profile: 20 Mbps and 20 ms RTT; "
            "documents round-robin across A4/A5/A28 with cloud synthesis."
        ),
    )
    colocated = InfrastructureCandidate(
        world_id="H_colocated",
        artifact_sites={artifact.artifact_id: "cloud" for artifact in scenario.artifacts},
        reasoner_site="cloud",
        reduction_sites=["cloud"],
        bandwidth_mbps=10_000.0,
        rtt_ms=0.1,
        basis=(
            "Existing infra-bench high-bandwidth profile: inputs colocated with the "
            "cloud reasoner; 10 Gbps/0.1 ms values document the local-link envelope."
        ),
    )
    return distributed, colocated


def _network_ms(byte_count: int, transfers: int, world: InfrastructureCandidate) -> float:
    if byte_count <= 0 or transfers <= 0:
        return 0.0
    serialization = byte_count * 8.0 / (world.bandwidth_mbps * 1_000_000.0) * 1_000.0
    return serialization + transfers * world.rtt_ms


def _central_cost(
    scenario: ScenarioRecord,
    demand: WorkflowDemandTemplate,
    world: InfrastructureCandidate,
    assumptions: Mapping[str, float],
) -> ModeledStrategyCost:
    cross_site = sum(
        artifact.size_bytes
        for artifact in scenario.artifacts
        if world.artifact_sites[artifact.artifact_id] != world.reasoner_site
    )
    transfers = sum(
        world.artifact_sites[artifact.artifact_id] != world.reasoner_site
        for artifact in scenario.artifacts
    )
    network = _network_ms(cross_site, transfers, world)
    reasoning_bytes = demand.reasoning_input_bytes or 0
    service = assumptions["central_reasoning_base_ms"] + (
        reasoning_bytes / assumptions["central_reasoning_bytes_per_s"] * 1_000.0
    )
    return ModeledStrategyCost(
        world_id=world.world_id,
        strategy_id=demand.strategy_id,
        estimated_cost_ms=service + network,
        modeled_cross_site_bytes=cross_site,
        modeled_network_ms=network,
        assumptions=dict(assumptions),
    )


def _distributed_cost(
    scenario: ScenarioRecord,
    demand: WorkflowDemandTemplate,
    world: InfrastructureCandidate,
    assumptions: Mapping[str, float],
) -> ModeledStrategyCost | None:
    if scenario.scene_features.evidence_bytes is None:
        return None
    artifact_count = scenario.scene_features.artifact_count
    reducer_replicas = max(1, int(assumptions["local_reducer_replicas"]))
    usable_width = min(max(1, artifact_count), reducer_replicas)
    waves = math.ceil(artifact_count / usable_width)
    reduction_service = waves * assumptions["reduction_base_ms"] + (
        scenario.scene_features.raw_bytes
        / (assumptions["reduction_bytes_per_s"] * usable_width)
        * 1_000.0
    )
    reduced_bytes = scenario.scene_features.evidence_bytes
    cross_site = (
        reduced_bytes
        if any(site != world.reasoner_site for site in world.reduction_sites)
        else 0
    )
    transfers = (
        min(artifact_count, len(world.reduction_sites)) if cross_site > 0 else 0
    )
    network = _network_ms(cross_site, transfers, world)
    synthesis = assumptions["synthesis_base_ms"] + (
        reduced_bytes / assumptions["central_reasoning_bytes_per_s"] * 1_000.0
    )
    return ModeledStrategyCost(
        world_id=world.world_id,
        strategy_id=demand.strategy_id,
        estimated_cost_ms=reduction_service + synthesis + network,
        modeled_cross_site_bytes=cross_site,
        modeled_network_ms=network,
        assumptions=dict(assumptions),
    )


def modeled_screening(
    scenario: ScenarioRecord,
    centralized: WorkflowDemandTemplate,
    distributed: WorkflowDemandTemplate,
    *,
    assumptions: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Estimate M(D, H) and flag reversal candidates without admitting them."""

    profile = {**DEFAULT_SCREENING_ASSUMPTIONS, **dict(assumptions or {})}
    worlds = infrastructure_candidates(scenario)
    costs: list[ModeledStrategyCost] = []
    winners: dict[str, str] = {}
    for world in worlds:
        central_cost = _central_cost(scenario, centralized, world, profile)
        distributed_cost = _distributed_cost(scenario, distributed, world, profile)
        costs.append(central_cost)
        if distributed_cost is None:
            continue
        costs.append(distributed_cost)
        winners[world.world_id] = min(
            (central_cost, distributed_cost),
            key=lambda item: (item.estimated_cost_ms, item.strategy_id),
        ).strategy_id
    reversal = len(set(winners.values())) >= 2 and len(winners) == len(worlds)
    return {
        "scenario_id": scenario.scenario_id,
        "task_id": scenario.task_id,
        "dataset": scenario.dataset,
        "worlds": [item.model_dump(mode="json") for item in worlds],
        "costs": [item.model_dump(mode="json") for item in costs],
        "winner_by_world": winners,
        "modeled_reversal": reversal,
        "screening_only": True,
        "admission": False,
        "evidence_compression_estimate": (
            "offline optimistic lower bound from hidden benchmark evidence size; "
            "not an executable reduction output and never used to select artifacts"
        ),
        "recommended_infra_perturbation": "artifact locality and bandwidth/RTT",
    }
