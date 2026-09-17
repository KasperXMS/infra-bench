from infra_bench.real_tasks.resource_calibration import (
    EmpiricalWorldMeasurement,
    calibrate_workflow_pair,
    contrast_resource_signatures,
)
from infra_bench.real_tasks.trace_workflows import ResourceSignature


def _signature(**overrides: float) -> ResourceSignature:
    values = {
        "local_search_count": 1.0,
        "local_search_service_ms": 100.0,
        "local_read_count": 1.0,
        "local_read_service_ms": 100.0,
        "remote_reasoning_calls": 1.0,
        "remote_reasoning_service_ms": 100.0,
        "transmitted_context_bytes": 10_000.0,
        "cross_site_bytes": 10_000.0,
        "cross_site_transfer_ms": 10.0,
        "cross_site_round_trips": 1.0,
        "targeted_test_count": 1.0,
        "targeted_test_service_ms": 100.0,
        "full_test_count": 0.0,
        "full_test_service_ms": 0.0,
        "max_parallelism": 1.0,
        "parallel_action_count": 0.0,
        "verification_count": 1.0,
        "failed_action_count": 0.0,
        "retry_count": 0.0,
        "total_tool_service_ms": 300.0,
        "other_tool_service_ms": 0.0,
        "observed_e2e_ms": 400.0,
    }
    values.update(overrides)
    return ResourceSignature(**values)


def test_small_resource_differences_are_skipped() -> None:
    first = _signature()
    second = _signature(
        local_search_service_ms=105.0,
        remote_reasoning_service_ms=105.0,
        transmitted_context_bytes=10_500.0,
        cross_site_bytes=10_500.0,
    )

    contrast = contrast_resource_signatures(first, second)

    assert contrast.sufficiently_different is False
    assert contrast.selected_dimensions == ()
    assert contrast.skip_reason is not None


def test_transfer_contrast_selects_locality_and_bandwidth_first() -> None:
    compact = _signature(
        transmitted_context_bytes=10_000.0,
        cross_site_bytes=10_000.0,
    )
    verbose = _signature(
        transmitted_context_bytes=5_000_000.0,
        cross_site_bytes=5_000_000.0,
        cross_site_transfer_ms=2_000.0,
    )

    contrast = contrast_resource_signatures(compact, verbose)

    assert contrast.sufficiently_different is True
    assert contrast.selected_dimensions[0] == "locality_bandwidth"
    assert contrast.dimension_evidence[0].dimension == "locality_bandwidth"


def test_modeled_reversal_is_only_a_screening_signal() -> None:
    local_heavy = _signature(
        local_search_service_ms=900.0,
        local_read_service_ms=100.0,
        remote_reasoning_service_ms=100.0,
        observed_e2e_ms=1_100.0,
    )
    reasoning_heavy = _signature(
        local_search_service_ms=50.0,
        local_read_service_ms=50.0,
        remote_reasoning_calls=5.0,
        remote_reasoning_service_ms=1_000.0,
        observed_e2e_ms=1_100.0,
    )

    result = calibrate_workflow_pair(local_heavy, reasoning_heavy)

    assert result.modeled_reversal_found is True
    assert result.empirical_reversal_found is False
    assert result.admitted is False
    assert "screening-only" in result.reason


def test_empirical_shared_world_robust_reversal_is_admitted() -> None:
    local_heavy = _signature(
        local_search_service_ms=900.0,
        local_read_service_ms=100.0,
        remote_reasoning_service_ms=100.0,
        observed_e2e_ms=1_100.0,
    )
    reasoning_heavy = _signature(
        local_search_service_ms=50.0,
        local_read_service_ms=50.0,
        remote_reasoning_calls=5.0,
        remote_reasoning_service_ms=1_000.0,
        observed_e2e_ms=1_100.0,
    )
    measurements = [
        EmpiricalWorldMeasurement(
            world_id="reasoning_scarce",
            workflow_e2e_ms={
                "workflow_a": (100.0, 102.0, 98.0),
                "workflow_b": (150.0, 152.0, 148.0),
            },
        ),
        EmpiricalWorldMeasurement(
            world_id="reasoning_available",
            workflow_e2e_ms={
                "workflow_a": (180.0, 182.0, 178.0),
                "workflow_b": (100.0, 102.0, 98.0),
            },
        ),
    ]

    result = calibrate_workflow_pair(
        local_heavy,
        reasoning_heavy,
        empirical_measurements=measurements,
    )

    assert result.admitted is True
    assert result.empirical_reversal_found is True
    assert result.selected_empirical_worlds is not None
    assert {item.winner for item in result.selected_empirical_worlds} == {
        "workflow_a",
        "workflow_b",
    }
    assert all(item.margin >= 0.20 for item in result.selected_empirical_worlds)


def test_unpaired_empirical_observations_cannot_admit() -> None:
    first = _signature(transmitted_context_bytes=1_000.0, cross_site_bytes=1_000.0)
    second = _signature(
        transmitted_context_bytes=10_000_000.0, cross_site_bytes=10_000_000.0
    )
    measurements = [
        EmpiricalWorldMeasurement(
            world_id="only-first-observed",
            workflow_e2e_ms={"workflow_a": (100.0,)},
        ),
        EmpiricalWorldMeasurement(
            world_id="only-second-observed",
            workflow_e2e_ms={"workflow_b": (100.0,)},
        ),
    ]

    result = calibrate_workflow_pair(
        first, second, empirical_measurements=measurements
    )

    assert result.empirical_worlds == ()
    assert result.admitted is False
