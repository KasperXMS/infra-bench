"""Tests for strict seven-task Scope Expansion v0 aggregation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from infra_bench.cli import build_parser
from infra_bench.real_tasks.scope_aggregate import (
    H1,
    H2,
    ScopeAggregationError,
    _read_stable_bytes,  # pyright: ignore[reportPrivateUsage]
    aggregate_scope_expansion_v0,
    write_scope_expansion_top_level_report,
)

VIDEO_IDS = ("video:reversal", "video:quality-risk")
MULTIHOP_IDS = ("mhr:m2", "mhr:m3", "mhr:m4")
LONG_MULTI_ID = "longbench:multidoc"
STRUCTURED_ID = "longbench:structured"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _write_inputs(root: Path) -> dict[str, Path]:
    paths = {
        "manifest": root / "core_set_manifest.json",
        "task_bank": root / "task_bank.jsonl",
        "calibration": root / "calibration.json",
        "multihop": root / "multihop.json",
        "multidoc": root / "multidoc.json",
        "structured": root / "structured.json",
    }
    manifest_tasks = [
        {
            "task_id": VIDEO_IDS[0],
            "dataset": "Video-MME",
            "task_family": "long_video_qa",
            "role": "workflow_reversal_positive_case",
        },
        {
            "task_id": VIDEO_IDS[1],
            "dataset": "Video-MME",
            "task_family": "long_video_qa",
            "role": "quality_risk_case",
        },
        *[
            {
                "task_id": task_id,
                "dataset": "MultiHop-RAG",
                "task_family": "multi_document_qa",
                "role": role,
            }
            for task_id, role in zip(MULTIHOP_IDS, ("M2", "M3", "M4"), strict=True)
        ],
        {
            "task_id": LONG_MULTI_ID,
            "dataset": "LongBench-v2",
            "task_family": "multi_document_qa",
            "role": "extra_long_multidoc",
        },
        {
            "task_id": STRUCTURED_ID,
            "dataset": "LongBench-v2",
            "task_family": "structured_data_analysis",
            "role": "long_structured_data",
        },
    ]
    _write_json(
        paths["manifest"],
        {
            "schema_version": "scope-expansion-v0-core-set-v1",
            "expected_task_count": 7,
            "task_count": 7,
            "tasks": manifest_tasks,
        },
    )
    non_video = [*MULTIHOP_IDS, LONG_MULTI_ID, STRUCTURED_ID]
    paths["task_bank"].write_text(
        "".join(
            json.dumps({"planner_visible": {"task_id": task_id}}) + "\n"
            for task_id in non_video
        ),
        encoding="utf-8",
    )
    _write_json(paths["calibration"], _calibration_summary())
    _write_json(
        paths["multihop"],
        _scope_summary(list(MULTIHOP_IDS), "distributed_retrieval", reversal_ids={MULTIHOP_IDS[0]}),
    )
    _write_json(
        paths["multidoc"],
        _scope_summary([LONG_MULTI_ID], "distributed_retrieval"),
    )
    _write_json(
        paths["structured"],
        _scope_summary([STRUCTURED_ID], "distributed_compute", labels=["compute_sensitive"]),
    )
    return paths


def _calibration_summary() -> dict[str, object]:
    cells: list[dict[str, object]] = []
    for task_id in VIDEO_IDS:
        for world in (H1, H2):
            for workflow in ("centralized_raw", "local_reduction"):
                central = workflow == "centralized_raw"
                e2e = (
                    (100.0 if central else 80.0)
                    if world == H1
                    else (50.0 if central else 90.0)
                )
                quality_gate = not (
                    task_id == VIDEO_IDS[1]
                    and world == H1
                    and workflow == "local_reduction"
                )
                cells.append(
                    {
                        "task_id": task_id,
                        "workflow_id": workflow,
                        "world_id": world,
                        "attempted_runs": 3,
                        "completed_runs": 3,
                        "eligible_for_comparison": quality_gate,
                        "quality_gate_passed": quality_gate,
                        "quality_pass_rate": 1.0 if quality_gate else 2 / 3,
                        "medians": {
                            "e2e_latency_ms": e2e,
                            "task_quality": 1.0 if quality_gate else 0.5,
                            "cross_agent_transfer_bytes": 1_000_000 if central else 1_000,
                            "transfer_sum_ms": 1_000 if central else 10,
                            "raw_artifact_bytes": 10_000_000,
                            "reduced_artifact_bytes": 0 if central else 1_000,
                        },
                    }
                )
    return {
        "schema_version": "calibration-v0-summary-v2",
        "policy": {"minimum_completed_runs_per_cell": 3},
        "cells": cells,
        "tasks": [],
        "totals": {},
    }


def _scope_summary(
    task_ids: list[str],
    distributed_workflow: str,
    *,
    reversal_ids: set[str] | None = None,
    labels: list[str] | None = None,
) -> dict[str, object]:
    reversal_ids = reversal_ids or set()
    base_labels = labels or ["communication_sensitive"]
    cells: list[dict[str, object]] = []
    tasks: list[dict[str, object]] = []
    for task_id in task_ids:
        reversal = task_id in reversal_ids
        preferences: dict[str, str] = {}
        for world in (H1, H2):
            for workflow in ("centralized_raw", distributed_workflow):
                if reversal:
                    e2e = (
                        (100.0 if workflow == "centralized_raw" else 80.0)
                        if world == H1
                        else (50.0 if workflow == "centralized_raw" else 90.0)
                    )
                else:
                    e2e = 50.0 if workflow == "centralized_raw" else 90.0
                cells.append(
                    {
                        "task_id": task_id,
                        "workflow_id": workflow,
                        "world_id": world,
                        "selected_measurement_series_id": "series-1",
                        "selected_protocol_id": "steady_state_1_warmup_3_measured_v1",
                        "completed_warmups": 1,
                        "completed_runs": 3,
                        "profiling_protocol_passed": True,
                        "quality_gate_passed": True,
                        "quality_pass_rate": 1.0,
                        "quality_median": 1.0,
                        "medians": {"e2e_latency_ms": e2e},
                    }
                )
            preferences[world] = (
                distributed_workflow if reversal and world == H1 else "centralized_raw"
            )
        task_labels = [*base_labels]
        if reversal:
            task_labels.insert(0, "workflow_reversal")
        tasks.append(
            {
                "task_id": task_id,
                "labels": task_labels,
                "workflow_preference": preferences,
                "quality_by_workflow": {
                    "centralized_raw": 1.0,
                    distributed_workflow: 1.0,
                },
                "evidence": {
                    label: {"triggered": label in task_labels} for label in task_labels
                },
            }
        )
    return {
        "schema_version": "scope-expansion-v0-sensitivity-report-v1",
        "protocol": {},
        "metric_definitions": {},
        "thresholds": {
            "relative_system_difference": 0.2,
            "compute_e2e_share": 0.1,
            "parallelism_speedup": 1.25,
            "quality_drop": 0.01,
        },
        "cells": cells,
        "tasks": tasks,
        "totals": {},
    }


def _aggregate(paths: dict[str, Path], *, strict: bool = True):
    return aggregate_scope_expansion_v0(
        manifest_path=paths["manifest"],
        task_bank_path=paths["task_bank"],
        calibration_summary_path=paths["calibration"],
        multihop_summary_path=paths["multihop"],
        longbench_multidoc_summary_path=paths["multidoc"],
        longbench_structured_summary_path=paths["structured"],
        strict=strict,
    )


def test_aggregate_builds_strict_seven_task_family_aware_report(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    report, snapshots = _aggregate(paths)

    assert report["complete"] is True
    completeness = report["completeness"]
    assert isinstance(completeness, dict)
    assert completeness["reported_task_count"] == 7
    assert completeness["reported_cell_count"] == 28
    assert completeness["protocol_complete_cell_count"] == 28
    assert len(snapshots) == 6
    task_rows = cast(list[dict[str, Any]], report["tasks"])
    tasks = {str(item["task_id"]): item for item in task_rows}
    reversal = tasks[VIDEO_IDS[0]]
    assert reversal["workflow_reversal"] is True
    assert reversal["winner_by_world"][H1]["winner"] == "local_reduction"
    assert reversal["winner_by_world"][H2]["winner"] == "centralized_raw"
    assert reversal["label_status"]["compute_sensitive"] == "not_evaluable"
    assert reversal["label_status"]["context_cost_sensitive"] == "not_applicable"
    assert "communication_sensitive" in reversal["labels"]
    assert "representation_sensitive" in reversal["labels"]

    quality_risk = tasks[VIDEO_IDS[1]]
    assert quality_risk["comparison_eligible"] is False
    assert quality_risk["winner_by_world"][H1]["winner"] is None
    assert quality_risk["winner_by_world"][H1]["status"] == "quality_gate_failed"
    assert quality_risk["winner_by_world"][H2]["winner"] == "centralized_raw"
    assert "quality_risk" in quality_risk["labels"]
    assert report["families"]["structured_data_analysis"]["label_counts"][  # type: ignore[index]
        "compute_sensitive"
    ] == 1


def test_writer_emits_json_and_markdown_from_tmp_fixtures(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path / "inputs")
    output = tmp_path / "output"
    report = write_scope_expansion_top_level_report(
        manifest_path=paths["manifest"],
        task_bank_path=paths["task_bank"],
        calibration_summary_path=paths["calibration"],
        multihop_summary_path=paths["multihop"],
        longbench_multidoc_summary_path=paths["multidoc"],
        longbench_structured_summary_path=paths["structured"],
        output_dir=output,
    )

    assert report["complete"] is True
    encoded = json.loads((output / "sensitivity_summary.json").read_text(encoding="utf-8"))
    assert encoded["completeness"]["reported_cell_count"] == 28
    markdown = (output / "sensitivity_summary.md").read_text(encoding="utf-8")
    assert "Quality-comparable" in markdown
    assert VIDEO_IDS[0] in markdown


def test_strict_mode_fails_closed_before_writing_when_family_report_missing(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path / "inputs")
    paths["structured"].unlink()
    output = tmp_path / "output"

    with pytest.raises(ScopeAggregationError, match="strict Scope Expansion"):
        write_scope_expansion_top_level_report(
            manifest_path=paths["manifest"],
            task_bank_path=paths["task_bank"],
            calibration_summary_path=paths["calibration"],
            multihop_summary_path=paths["multihop"],
            longbench_multidoc_summary_path=paths["multidoc"],
            longbench_structured_summary_path=paths["structured"],
            output_dir=output,
        )

    assert not output.exists()


def test_strict_mode_rejects_protocol_incomplete_cell(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    payload: dict[str, Any] = json.loads(paths["structured"].read_text(encoding="utf-8"))
    payload["cells"][0]["completed_warmups"] = 0
    payload["cells"][0]["profiling_protocol_passed"] = False
    _write_json(paths["structured"], payload)

    with pytest.raises(ScopeAggregationError, match="protocol_complete"):
        _aggregate(paths)


def test_stable_reader_rejects_content_change_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    original = Path.read_bytes
    calls = 0

    def changing_read(path: Path) -> bytes:
        nonlocal calls
        if path == source:
            calls += 1
            if calls == 2:
                return b'{"changed":true}'
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", changing_read)
    with pytest.raises(ScopeAggregationError, match="changed while reading"):
        _read_stable_bytes(source)


def test_cli_registers_strict_top_level_aggregation() -> None:
    args = build_parser().parse_args(["aggregate-scope-expansion-v0"])
    assert args.strict is True
    assert args.output_dir == "runs/scope_expansion_v0"
    relaxed = build_parser().parse_args(
        ["aggregate-scope-expansion-v0", "--no-strict"]
    )
    assert relaxed.strict is False
