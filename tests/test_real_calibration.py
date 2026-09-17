from infra_bench.real_tasks.calibration import calibrate_swebench_pair, calibrate_video_pair
from infra_bench.real_tasks.video_mme import WORKFLOW_DENSE, WORKFLOW_LOCALIZED


def _result(
    *, preprocess_ms: float, model_ms: float, request_bytes: int, source_bytes: int
) -> dict:
    return {
        "preprocess_service_ms": preprocess_ms,
        "model_service_ms": model_ms,
        "request_bytes": request_bytes,
        "source_artifact_bytes": source_bytes,
    }


def test_calibration_requires_opposite_robust_winners() -> None:
    dense = _result(
        preprocess_ms=100,
        model_ms=100,
        request_bytes=100_000_000,
        source_bytes=500_000_000,
    )
    localized = _result(
        preprocess_ms=1_000,
        model_ms=100,
        request_bytes=1_000_000,
        source_bytes=500_000_000,
    )
    result = calibrate_video_pair(dense, localized)
    assert result["semantic_switch_pair_found"] is True
    assert {world["winner"] for world in result["selected_worlds"]} == {
        WORKFLOW_DENSE,
        WORKFLOW_LOCALIZED,
    }
    assert all(world["margin"] >= 0.20 for world in result["selected_worlds"])


def test_swebench_calibration_prefers_controlled_single_dimension_pair() -> None:
    remote = {
        "repository_bytes": 80_000_000,
        "context_bytes": 150_000,
        "trace": [
            {"phase": "remote_file_selection", "service_ms": 100_000},
            {"phase": "remote_patch_reasoning", "service_ms": 200_000},
            {"phase": "patch_apply_repair", "service_ms": 1_000},
        ],
    }
    compact = {
        "repository_bytes": 80_000_000,
        "context_bytes": 180_000,
        "trace": [
            {"phase": "local_search_static_filter", "service_ms": 10},
            {"phase": "remote_patch_reasoning", "service_ms": 450_000},
            {"phase": "patch_apply_repair", "service_ms": 1_000},
        ],
    }
    result = calibrate_swebench_pair(remote, compact)
    assert result["semantic_switch_pair_found"] is True
    first, second = result["selected_worlds"]
    assert first["artifact_locality"] == second["artifact_locality"]
    assert first["rtt_ms"] == second["rtt_ms"]
    changed = {
        key
        for key in ("bandwidth_mbps", "remote_selection_service_scale")
        if first[key] != second[key]
    }
    assert len(changed) == 1
