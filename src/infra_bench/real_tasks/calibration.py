from __future__ import annotations

from itertools import product
from typing import Any

from .video_mme import WORKFLOW_DENSE, WORKFLOW_LOCALIZED


def _world_distance(first: dict[str, Any], second: dict[str, Any]) -> int:
    ignored = {"world_id", "cost_ms", "winner", "margin", "replicas"}
    keys = (set(first) | set(second)) - ignored
    return sum(first.get(key) != second.get(key) for key in keys)


def _transfer_ms(byte_count: int, bandwidth_mbps: float, rtt_ms: float) -> float:
    if byte_count <= 0:
        return 0.0
    return rtt_ms + byte_count * 8.0 / (bandwidth_mbps * 1_000_000.0) * 1000.0


def calibrate_video_pair(
    dense: dict[str, Any], localized: dict[str, Any], *, admission_margin: float = 0.20
) -> dict[str, Any]:
    """Scan explicit infrastructure counterfactuals using measured workflow services."""
    worlds: list[dict[str, Any]] = []
    for locality, bandwidth, rtt, edge_scale in product(
        ("edge", "cloud"), (10.0, 50.0, 200.0, 1000.0), (10.0, 50.0), (0.25, 0.5, 1.0, 2.0)
    ):
        dense_preprocess = float(dense["preprocess_service_ms"])
        localized_preprocess = float(localized["preprocess_service_ms"]) * edge_scale
        dense_transfer = 0.0
        localized_transfer = 0.0
        if locality == "edge":
            dense_preprocess *= edge_scale
            dense_transfer = _transfer_ms(int(dense["request_bytes"]), bandwidth, rtt)
            localized_transfer = _transfer_ms(
                int(localized["request_bytes"]), bandwidth, rtt
            )
        else:
            # Dense sampling has a cloud replica colocated with the video. The motion
            # localizer is edge-only, so it fetches the source and returns compact evidence.
            localized_transfer = _transfer_ms(
                int(localized["source_artifact_bytes"]), bandwidth, rtt
            ) + _transfer_ms(int(localized["request_bytes"]), bandwidth, rtt)

        costs = {
            WORKFLOW_DENSE: dense_preprocess
            + dense_transfer
            + float(dense["model_service_ms"]),
            WORKFLOW_LOCALIZED: localized_preprocess
            + localized_transfer
            + float(localized["model_service_ms"]),
        }
        winner = min(costs, key=lambda key: (costs[key], key))
        loser = next(key for key in costs if key != winner)
        margin = (costs[loser] - costs[winner]) / costs[winner]
        worlds.append(
            {
                "world_id": f"{locality}_bw{bandwidth:g}_rtt{rtt:g}_edge{edge_scale:g}",
                "artifact_locality": locality,
                "bandwidth_mbps": bandwidth,
                "rtt_ms": rtt,
                "edge_preprocess_service_scale": edge_scale,
                "replicas": {
                    "uniform_sampler": [locality],
                    "motion_localizer": ["edge"],
                    "strong_vlm": ["cloud"],
                },
                "cost_ms": {key: round(value, 3) for key, value in costs.items()},
                "winner": winner,
                "margin": round(margin, 6),
            }
        )

    pairs: list[tuple[float, float, dict[str, Any], dict[str, Any]]] = []
    for first_index, first in enumerate(worlds):
        for second in worlds[first_index + 1 :]:
            if first["winner"] == second["winner"]:
                continue
            minimum = min(float(first["margin"]), float(second["margin"]))
            pairs.append((minimum, float(first["margin"]) + float(second["margin"]), first, second))
    robust_pairs = [item for item in pairs if item[0] >= admission_margin]
    robust_pairs.sort(
        key=lambda item: (
            _world_distance(item[2], item[3]),
            max(float(item[2]["margin"]), float(item[3]["margin"])),
            item[1],
            item[2]["world_id"],
            item[3]["world_id"],
        )
    )
    robust = robust_pairs[0] if robust_pairs else None
    selected = None
    if robust is not None:
        _, _, first, second = robust
        selected = [first, second]
    return {
        "objective": "measured_service_plus_network_e2e_ms",
        "world_count": len(worlds),
        "admission_margin": admission_margin,
        "semantic_switch_pair_found": selected is not None,
        "selected_worlds": selected,
        "worlds": worlds,
    }


def calibrate_swebench_pair(
    remote: dict[str, Any], compact: dict[str, Any], *, admission_margin: float = 0.20
) -> dict[str, Any]:
    def phase_ms(result: dict[str, Any], phase: str) -> float:
        return sum(
            float(item.get("service_ms", 0.0))
            for item in result["trace"]
            if item.get("phase") == phase
        )

    remote_select = phase_ms(remote, "remote_file_selection")
    remote_common = phase_ms(remote, "remote_patch_reasoning") + phase_ms(
        remote, "patch_apply_repair"
    )
    compact_local = phase_ms(compact, "local_search_static_filter")
    compact_common = phase_ms(compact, "remote_patch_reasoning") + phase_ms(
        compact, "patch_apply_repair"
    )
    worlds: list[dict[str, Any]] = []
    for locality, bandwidth, rtt, selection_scale in product(
        ("edge", "cloud"),
        (1.0, 10.0, 100.0, 1000.0),
        (10.0, 50.0),
        (0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0),
    ):
        remote_transfer = 0.0
        compact_transfer = 0.0
        if locality == "edge":
            remote_transfer = _transfer_ms(int(remote["context_bytes"]), bandwidth, rtt)
            compact_transfer = _transfer_ms(int(compact["context_bytes"]), bandwidth, rtt)
        else:
            compact_transfer = _transfer_ms(
                int(compact["repository_bytes"]), bandwidth, rtt
            ) + _transfer_ms(int(compact["context_bytes"]), bandwidth, rtt)
        costs = {
            "whole_repo_remote_reasoning": remote_select * selection_scale
            + remote_common
            + remote_transfer,
            "local_search_test_compact_remote_reasoning": compact_local
            + compact_common
            + compact_transfer,
        }
        winner = min(costs, key=lambda key: (costs[key], key))
        loser = next(key for key in costs if key != winner)
        margin = (costs[loser] - costs[winner]) / costs[winner]
        worlds.append(
            {
                "world_id": (
                    f"{locality}_bw{bandwidth:g}_rtt{rtt:g}_"
                    f"remote_selection{selection_scale:g}x"
                ),
                "artifact_locality": locality,
                "bandwidth_mbps": bandwidth,
                "rtt_ms": rtt,
                "remote_selection_service_scale": selection_scale,
                "replicas": {
                    "remote_repo_selection": ["cloud"],
                    "local_search_static_filter": ["edge"],
                    "remote_patch_reasoning": ["cloud"],
                },
                "service_time_estimates_ms": {
                    "remote_repository_inspection": round(
                        remote_select * selection_scale, 3
                    ),
                    "local_search_static_filter": round(compact_local, 3),
                    "remote_reasoning_typical": round(
                        (remote_common + compact_common) / 2.0, 3
                    ),
                },
                "cost_ms": {key: round(value, 3) for key, value in costs.items()},
                "winner": winner,
                "margin": round(margin, 6),
            }
        )
    pairs: list[tuple[float, float, dict[str, Any], dict[str, Any]]] = []
    for index, first in enumerate(worlds):
        for second in worlds[index + 1 :]:
            if first["winner"] == second["winner"]:
                continue
            minimum = min(float(first["margin"]), float(second["margin"]))
            pairs.append((minimum, float(first["margin"]) + float(second["margin"]), first, second))
    robust_pairs = [item for item in pairs if item[0] >= admission_margin]
    robust_pairs.sort(
        key=lambda item: (
            _world_distance(item[2], item[3]),
            -item[0],
            -item[1],
            item[2]["world_id"],
            item[3]["world_id"],
        )
    )
    robust = robust_pairs[0] if robust_pairs else None
    selected = [robust[2], robust[3]] if robust is not None else None
    return {
        "objective": "measured_service_plus_network_e2e_ms",
        "world_count": len(worlds),
        "admission_margin": admission_margin,
        "semantic_switch_pair_found": selected is not None,
        "selected_worlds": selected,
        "worlds": worlds,
        "measurement_note": (
            "Service times are real workflow traces; repository/context byte sizes are measured "
            "from the checked-out artifact and exact bounded contexts."
        ),
    }
