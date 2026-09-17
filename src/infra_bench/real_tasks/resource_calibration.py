from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import median
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .trace_workflows import ResourceSignature, SemanticWorkflowCluster

PerturbationDimension = Literal[
    "locality_bandwidth",
    "replica_availability_load",
    "test_runner_speed_locality",
    "reasoning_service_profile",
]


class DimensionEvidence(BaseModel):
    """Evidence that a workflow pair couples to an infrastructure dimension."""

    model_config = ConfigDict(frozen=True)

    dimension: PerturbationDimension
    score: float = Field(ge=0.0, le=1.0)
    feature_deltas: dict[str, float]
    explanation: str


class ResourceSignatureContrast(BaseModel):
    """Cheap screening result computed before counterfactual calibration."""

    model_config = ConfigDict(frozen=True)

    sufficiently_different: bool
    contrast_score: float = Field(ge=0.0, le=1.0)
    selected_dimensions: tuple[PerturbationDimension, ...]
    dimension_evidence: tuple[DimensionEvidence, ...]
    skip_reason: str | None = None


class CounterfactualWorld(BaseModel):
    """An explicit, auditable infrastructure perturbation used for screening."""

    model_config = ConfigDict(frozen=True)

    world_id: str
    dimension: PerturbationDimension
    bandwidth_mbps: float = Field(default=1_000.0, gt=0.0)
    rtt_ms: float = Field(default=1.0, ge=0.0)
    transfer_fraction: float = Field(default=1.0, ge=0.0, le=1.0)
    local_io_multiplier: float = Field(default=1.0, gt=0.0)
    test_service_multiplier: float = Field(default=1.0, gt=0.0)
    reasoning_service_multiplier: float = Field(default=1.0, gt=0.0)
    replica_count: int = Field(default=1, ge=1)
    load_multiplier: float = Field(default=1.0, gt=0.0)


class ModeledWorldResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    world: CounterfactualWorld
    workflow_cost_ms: dict[str, float]
    winner: str
    margin: float = Field(ge=0.0)


class EmpiricalWorldMeasurement(BaseModel):
    """Repeated real E2E observations for workflows in one materialized world."""

    model_config = ConfigDict(frozen=True)

    world_id: str
    workflow_e2e_ms: dict[str, tuple[float, ...]]

    @field_validator("workflow_e2e_ms")
    @classmethod
    def validate_measurements(
        cls, value: dict[str, tuple[float, ...]]
    ) -> dict[str, tuple[float, ...]]:
        for workflow_id, observations in value.items():
            if not observations:
                raise ValueError(f"{workflow_id!r} has no E2E observations")
            if any(item < 0.0 for item in observations):
                raise ValueError("E2E observations must be non-negative")
        return value


class EmpiricalWorldResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    world_id: str
    workflow_median_e2e_ms: dict[str, float]
    winner: str
    margin: float = Field(ge=0.0)


class WorkflowPairCalibration(BaseModel):
    """Calibration outcome; modeled results can screen, but cannot admit a pair."""

    model_config = ConfigDict(frozen=True)

    workflow_ids: tuple[str, str]
    contrast: ResourceSignatureContrast
    modeled_worlds: tuple[ModeledWorldResult, ...]
    modeled_reversal_found: bool
    empirical_worlds: tuple[EmpiricalWorldResult, ...]
    empirical_reversal_found: bool
    admission_margin: float = Field(ge=0.0)
    admitted: bool
    selected_empirical_worlds: tuple[EmpiricalWorldResult, EmpiricalWorldResult] | None
    reason: str


def _number(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _feature(signature: ResourceSignature, name: str) -> float:
    return _number(getattr(signature, name, 0.0))


def _relative_delta(first: float, second: float, *, floor: float = 1.0) -> float:
    return min(1.0, abs(first - second) / max(abs(first), abs(second), floor))


def _deltas(
    first: ResourceSignature, second: ResourceSignature, features: Sequence[str]
) -> dict[str, float]:
    return {
        feature: round(
            _relative_delta(_feature(first, feature), _feature(second, feature)), 6
        )
        for feature in features
    }


def _weighted_score(deltas: Mapping[str, float], weights: Mapping[str, float]) -> float:
    total_weight = sum(weights.values())
    if total_weight <= 0.0:
        return 0.0
    return min(
        1.0,
        sum(deltas.get(feature, 0.0) * weight for feature, weight in weights.items())
        / total_weight,
    )


def contrast_resource_signatures(
    first: ResourceSignature,
    second: ResourceSignature,
    *,
    minimum_dimension_score: float = 0.25,
) -> ResourceSignatureContrast:
    """Rank infrastructure dimensions that explain meaningful resource contrast.

    The score is deliberately based on semantic resource groups rather than exact
    action sequences, so shell spelling and harmless repetition do not manufacture
    an infrastructure-sensitive workflow pair.
    """

    locality = _deltas(
        first,
        second,
        (
            "local_search_count",
            "local_read_count",
            "local_search_service_ms",
            "local_read_service_ms",
            "transmitted_context_bytes",
            "cross_site_bytes",
            "cross_site_transfer_ms",
            "cross_site_round_trips",
        ),
    )
    parallelism = _deltas(
        first, second, ("max_parallelism", "parallel_action_count")
    )
    tests = _deltas(
        first,
        second,
        (
            "targeted_test_count",
            "full_test_count",
            "targeted_test_service_ms",
            "full_test_service_ms",
            "verification_count",
            "retry_count",
            "failed_action_count",
        ),
    )
    reasoning = _deltas(
        first,
        second,
        (
            "remote_reasoning_calls",
            "remote_reasoning_service_ms",
            "transmitted_context_bytes",
        ),
    )

    evidence = (
        DimensionEvidence(
            dimension="locality_bandwidth",
            score=_weighted_score(
                locality,
                {
                    "transmitted_context_bytes": 3.0,
                    "cross_site_bytes": 3.0,
                    "cross_site_transfer_ms": 1.0,
                    "cross_site_round_trips": 1.0,
                    "local_search_count": 0.5,
                    "local_read_count": 0.5,
                    "local_search_service_ms": 0.5,
                    "local_read_service_ms": 0.5,
                },
            ),
            feature_deltas=locality,
            explanation=(
                "context/network volume and local search/read work couple to artifact "
                "locality, bandwidth, and RTT"
            ),
        ),
        DimensionEvidence(
            dimension="replica_availability_load",
            score=_weighted_score(
                parallelism, {"max_parallelism": 1.0, "parallel_action_count": 1.0}
            ),
            feature_deltas=parallelism,
            explanation=(
                "available workflow parallelism couples to replica count and replica load"
            ),
        ),
        DimensionEvidence(
            dimension="test_runner_speed_locality",
            score=_weighted_score(
                tests,
                {
                    "targeted_test_count": 1.5,
                    "full_test_count": 2.0,
                    "targeted_test_service_ms": 1.0,
                    "full_test_service_ms": 2.0,
                    "verification_count": 0.5,
                    "retry_count": 0.25,
                    "failed_action_count": 0.25,
                },
            ),
            feature_deltas=tests,
            explanation=(
                "targeted/full verification mix and measured test cost couple to test-runner "
                "speed and locality"
            ),
        ),
        DimensionEvidence(
            dimension="reasoning_service_profile",
            score=_weighted_score(
                reasoning,
                {
                    "remote_reasoning_calls": 2.0,
                    "remote_reasoning_service_ms": 2.0,
                    "transmitted_context_bytes": 0.5,
                },
            ),
            feature_deltas=reasoning,
            explanation=(
                "remote reasoning count, service demand, and context payload couple to the "
                "reasoning service profile"
            ),
        ),
    )
    ranked = tuple(sorted(evidence, key=lambda item: (-item.score, item.dimension)))
    top_score = ranked[0].score
    selected_dimensions: list[PerturbationDimension] = []
    for item in ranked:
        if item.score >= minimum_dimension_score and item.score >= max(
            minimum_dimension_score, top_score * 0.75
        ):
            selected_dimensions.append(item.dimension)
    selected = tuple(selected_dimensions)
    sufficiently_different = bool(selected)
    return ResourceSignatureContrast(
        sufficiently_different=sufficiently_different,
        contrast_score=top_score,
        selected_dimensions=selected,
        dimension_evidence=ranked,
        skip_reason=(
            None
            if sufficiently_different
            else "resource signatures have no infrastructure-coupled contrast above threshold"
        ),
    )


def _workflow_id(cluster: SemanticWorkflowCluster | ResourceSignature, fallback: str) -> str:
    for name in ("cluster_id", "workflow_id", "semantic_id", "fingerprint"):
        value = getattr(cluster, name, None)
        if isinstance(value, str) and value:
            return value
    representative = getattr(cluster, "representative_workflow", None)
    representative_id = getattr(representative, "workflow_id", None)
    if isinstance(representative_id, str) and representative_id:
        return representative_id
    return fallback


def _signature(cluster: SemanticWorkflowCluster | ResourceSignature) -> ResourceSignature:
    if isinstance(cluster, ResourceSignature):
        return cluster
    for name in ("resource_signature", "signature"):
        value = getattr(cluster, name, None)
        if isinstance(value, ResourceSignature):
            return value
    raise TypeError("workflow cluster does not expose a ResourceSignature")


def _network_ms(byte_count: float, round_trips: float, world: CounterfactualWorld) -> float:
    transferred = max(0.0, byte_count) * world.transfer_fraction
    if transferred <= 0.0:
        return 0.0
    serialization_ms = transferred * 8.0 / (world.bandwidth_mbps * 1_000_000.0) * 1_000.0
    return serialization_ms + max(1.0, round_trips) * world.rtt_ms


def _modeled_cost(signature: ResourceSignature, world: CounterfactualWorld) -> float:
    local_io = _feature(signature, "local_search_service_ms") + _feature(
        signature, "local_read_service_ms"
    )
    reasoning = _feature(signature, "remote_reasoning_service_ms")
    tests = _feature(signature, "targeted_test_service_ms") + _feature(
        signature, "full_test_service_ms"
    )
    other = _feature(signature, "other_tool_service_ms")
    component_total = local_io + reasoning + tests + other
    observed_e2e = _feature(signature, "observed_e2e_ms")
    fixed_overhead = max(0.0, observed_e2e - component_total)

    service = (
        local_io * world.local_io_multiplier
        + reasoning * world.reasoning_service_multiplier
        + tests * world.test_service_multiplier
        + other
    )
    action_count = (
        _feature(signature, "local_search_count")
        + _feature(signature, "local_read_count")
        + _feature(signature, "remote_reasoning_calls")
        + _feature(signature, "targeted_test_count")
        + _feature(signature, "full_test_count")
    )
    parallel_actions = min(_feature(signature, "parallel_action_count"), action_count)
    parallel_share = parallel_actions / action_count if action_count > 0 else 0.0
    usable_parallelism = min(
        max(1.0, _feature(signature, "max_parallelism")), float(world.replica_count)
    )
    service *= 1.0 - parallel_share * (1.0 - 1.0 / usable_parallelism)
    service *= world.load_multiplier

    transfer_bytes = max(
        _feature(signature, "cross_site_bytes"),
        _feature(signature, "transmitted_context_bytes"),
    )
    network = _network_ms(
        transfer_bytes, _feature(signature, "cross_site_round_trips"), world
    )
    return max(0.0, fixed_overhead + service + network)


def _worlds_for(dimension: PerturbationDimension) -> tuple[CounterfactualWorld, ...]:
    if dimension == "locality_bandwidth":
        return (
            CounterfactualWorld(
                world_id="transfer_constrained",
                dimension=dimension,
                bandwidth_mbps=10.0,
                rtt_ms=50.0,
            ),
            CounterfactualWorld(
                world_id="transfer_colocated",
                dimension=dimension,
                bandwidth_mbps=10_000.0,
                rtt_ms=0.1,
                transfer_fraction=0.0,
            ),
        )
    if dimension == "replica_availability_load":
        return (
            CounterfactualWorld(
                world_id="single_replica_loaded",
                dimension=dimension,
                replica_count=1,
                load_multiplier=1.5,
            ),
            CounterfactualWorld(
                world_id="four_replicas_available",
                dimension=dimension,
                replica_count=4,
                load_multiplier=0.8,
            ),
        )
    if dimension == "test_runner_speed_locality":
        return (
            CounterfactualWorld(
                world_id="slow_remote_test_runner",
                dimension=dimension,
                test_service_multiplier=4.0,
                bandwidth_mbps=50.0,
                rtt_ms=25.0,
            ),
            CounterfactualWorld(
                world_id="fast_local_test_runner",
                dimension=dimension,
                test_service_multiplier=0.25,
                transfer_fraction=0.0,
            ),
        )
    return (
        CounterfactualWorld(
            world_id="reasoning_scarce",
            dimension=dimension,
            reasoning_service_multiplier=3.0,
        ),
        CounterfactualWorld(
            world_id="reasoning_available",
            dimension=dimension,
            reasoning_service_multiplier=0.35,
        ),
    )


def _winner(costs: Mapping[str, float]) -> tuple[str, float]:
    ordered = sorted(costs.items(), key=lambda item: (item[1], item[0]))
    winning_id, winning_cost = ordered[0]
    losing_cost = ordered[1][1]
    denominator = max(winning_cost, 1e-9)
    return winning_id, max(0.0, (losing_cost - winning_cost) / denominator)


def _coerce_empirical(
    measurements: Sequence[EmpiricalWorldMeasurement | Mapping[str, Any]],
) -> tuple[EmpiricalWorldMeasurement, ...]:
    return tuple(
        item
        if isinstance(item, EmpiricalWorldMeasurement)
        else EmpiricalWorldMeasurement.model_validate(item)
        for item in measurements
    )


def calibrate_workflow_pair(
    first: SemanticWorkflowCluster | ResourceSignature,
    second: SemanticWorkflowCluster | ResourceSignature,
    *,
    empirical_measurements: Sequence[
        EmpiricalWorldMeasurement | Mapping[str, Any]
    ] = (),
    empirical_worlds: Sequence[EmpiricalWorldMeasurement | Mapping[str, Any]] | None = None,
    admission_margin: float = 0.20,
    minimum_dimension_score: float = 0.25,
) -> WorkflowPairCalibration:
    """Screen a workflow pair and admit it only on robust empirical reversal.

    Counterfactual modeling selects promising worlds. It is never sufficient for
    admission: both workflows must have real E2E observations in the same two worlds.
    """

    first_signature = _signature(first)
    second_signature = _signature(second)
    first_id = _workflow_id(first, "workflow_a")
    second_id = _workflow_id(second, "workflow_b")
    if first_id == second_id:
        second_id = f"{second_id}_b"
    contrast = contrast_resource_signatures(
        first_signature,
        second_signature,
        minimum_dimension_score=minimum_dimension_score,
    )

    modeled: list[ModeledWorldResult] = []
    seen_worlds: set[tuple[object, ...]] = set()
    if contrast.sufficiently_different:
        for dimension in contrast.selected_dimensions:
            for world in _worlds_for(dimension):
                identity = tuple(world.model_dump().values())
                if identity in seen_worlds:
                    continue
                seen_worlds.add(identity)
                costs = {
                    first_id: _modeled_cost(first_signature, world),
                    second_id: _modeled_cost(second_signature, world),
                }
                winner, margin = _winner(costs)
                modeled.append(
                    ModeledWorldResult(
                        world=world,
                        workflow_cost_ms={
                            key: round(value, 6) for key, value in costs.items()
                        },
                        winner=winner,
                        margin=round(margin, 6),
                    )
                )
    modeled_reversal = len({item.winner for item in modeled}) >= 2

    supplied = empirical_worlds if empirical_worlds is not None else empirical_measurements
    empirical: list[EmpiricalWorldResult] = []
    for measurement in _coerce_empirical(supplied):
        if first_id not in measurement.workflow_e2e_ms or second_id not in measurement.workflow_e2e_ms:
            continue
        costs = {
            first_id: median(measurement.workflow_e2e_ms[first_id]),
            second_id: median(measurement.workflow_e2e_ms[second_id]),
        }
        winner, margin = _winner(costs)
        empirical.append(
            EmpiricalWorldResult(
                world_id=measurement.world_id,
                workflow_median_e2e_ms={
                    key: round(value, 6) for key, value in costs.items()
                },
                winner=winner,
                margin=round(margin, 6),
            )
        )

    candidates: list[tuple[float, EmpiricalWorldResult, EmpiricalWorldResult]] = []
    for index, one in enumerate(empirical):
        for two in empirical[index + 1 :]:
            if one.winner == two.winner:
                continue
            robust_margin = min(one.margin, two.margin)
            if robust_margin >= admission_margin:
                candidates.append((robust_margin, one, two))
    candidates.sort(key=lambda item: (-item[0], item[1].world_id, item[2].world_id))
    selected = candidates[0][1:] if candidates else None
    empirical_reversal = len({item.winner for item in empirical}) >= 2
    admitted = contrast.sufficiently_different and selected is not None
    if admitted:
        reason = "empirical winner reversal observed with robust margins in shared worlds"
    elif not contrast.sufficiently_different:
        reason = cast(str, contrast.skip_reason)
    elif not modeled_reversal:
        reason = "resource contrast found, but modeled worlds did not produce a reversal"
    elif not empirical_reversal:
        reason = "modeled reversal is screening-only; no empirical winner reversal was observed"
    elif selected is None:
        reason = "empirical reversal exists, but one or both margins are below admission threshold"
    else:
        reason = "empirical calibration did not satisfy admission"
    return WorkflowPairCalibration(
        workflow_ids=(first_id, second_id),
        contrast=contrast,
        modeled_worlds=tuple(modeled),
        modeled_reversal_found=modeled_reversal,
        empirical_worlds=tuple(empirical),
        empirical_reversal_found=empirical_reversal,
        admission_margin=admission_margin,
        admitted=admitted,
        selected_empirical_worlds=selected,
        reason=reason,
    )
