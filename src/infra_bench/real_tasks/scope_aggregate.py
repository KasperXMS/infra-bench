"""Strict top-level aggregation for the fixed seven-task Scope Expansion v0 set."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Literal, cast
from uuid import uuid4

from infra_bench.schemas.scope_expansion_v0 import (
    FORMAL_SCOPE_PROTOCOL_ID,
    ScopeSensitivityLabel,
)

H1 = "H1_distributed_constrained"
H2 = "H2_distributed_favorable"
WORLDS = (H1, H2)
CENTRALIZED = "centralized_raw"
LABELS: tuple[ScopeSensitivityLabel, ...] = (
    "workflow_reversal",
    "communication_sensitive",
    "compute_sensitive",
    "representation_sensitive",
    "quality_risk",
    "parallelism_sensitive",
    "context_cost_sensitive",
)
FAMILY_DISTRIBUTED_WORKFLOW = {
    "long_video_qa": "local_reduction",
    "multi_document_qa": "distributed_retrieval",
    "structured_data_analysis": "distributed_compute",
}
EXPECTED_FAMILY_COUNTS = {
    "long_video_qa": 2,
    "multi_document_qa": 4,
    "structured_data_analysis": 1,
}
EXPECTED_TASK_COUNT = 7
EXPECTED_CELL_COUNT = 28
SYSTEM_DIFFERENCE_THRESHOLD = 0.20

LabelStatus = Literal[
    "triggered", "not_triggered", "not_evaluable", "not_applicable"
]


class ScopeAggregationError(ValueError):
    """Raised when a source snapshot cannot support an honest top-level report."""


@dataclass(frozen=True)
class _FileState:
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class SourceSnapshot:
    path: Path
    state: _FileState
    sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path.as_posix(),
            "size_bytes": self.state.size,
            "mtime_ns": self.state.mtime_ns,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class _Cell:
    task_id: str
    workflow_id: str
    world_id: str
    protocol_complete: bool
    quality_gate_passed: bool
    quality_pass_rate: float
    quality_median: float | None
    median_e2e_ms: float | None
    medians: dict[str, float | int | None]
    measurement_series_id: str | None
    protocol_id: str | None


@dataclass(frozen=True)
class _TaskSource:
    source_name: str
    cells: tuple[_Cell, ...]
    labels: tuple[ScopeSensitivityLabel, ...]
    evidence: dict[str, object]
    quality_by_workflow: dict[str, float | None]
    series_consistent: bool


def _file_state(path: Path) -> _FileState:
    stat = path.stat()
    return _FileState(size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def _read_stable_bytes(path: Path) -> tuple[bytes, SourceSnapshot]:
    """Read twice and require an unchanged file before accepting its contents."""
    if not path.is_file():
        raise ScopeAggregationError(f"required aggregation input is missing: {path}")
    before = _file_state(path)
    first = path.read_bytes()
    middle = _file_state(path)
    second = path.read_bytes()
    after = _file_state(path)
    if before != middle or middle != after or first != second:
        raise ScopeAggregationError(f"aggregation input changed while reading: {path}")
    digest = hashlib.sha256(first).hexdigest()
    return first, SourceSnapshot(path=path, state=after, sha256=digest)


def _read_stable_json(path: Path) -> tuple[object, SourceSnapshot]:
    encoded, snapshot = _read_stable_bytes(path)
    try:
        return json.loads(encoded), snapshot
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ScopeAggregationError(f"invalid JSON aggregation input {path}: {error}") from error


def _read_stable_jsonl(path: Path) -> tuple[list[dict[str, Any]], SourceSnapshot]:
    encoded, snapshot = _read_stable_bytes(path)
    records: list[dict[str, Any]] = []
    try:
        text = encoded.decode("utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            value: object = json.loads(line)
            if not isinstance(value, dict):
                raise ScopeAggregationError(
                    f"{path}:{line_number} must contain a JSON object"
                )
            records.append(cast(dict[str, Any], value))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ScopeAggregationError(f"invalid JSONL aggregation input {path}: {error}") from error
    return records, snapshot


def _verify_snapshots(snapshots: Sequence[SourceSnapshot]) -> None:
    """Ensure the complete multi-file input set still matches the accepted snapshot."""
    for snapshot in snapshots:
        if not snapshot.path.is_file() or _file_state(snapshot.path) != snapshot.state:
            raise ScopeAggregationError(
                f"aggregation input changed after snapshot: {snapshot.path}"
            )
        digest = hashlib.sha256(snapshot.path.read_bytes()).hexdigest()
        if digest != snapshot.sha256:
            raise ScopeAggregationError(
                f"aggregation input content changed after snapshot: {snapshot.path}"
            )


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ScopeAggregationError(f"{context} must be a JSON object")
    return cast(dict[str, Any], value)


def _list(value: object, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ScopeAggregationError(f"{context} must be a JSON array")
    return cast(list[Any], value)


def _optional_number(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScopeAggregationError(f"expected a number or null, got {value!r}")
    return float(value)


def _relative_difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    scale = max(abs(left), abs(right))
    return abs(left - right) / scale if scale else 0.0


def _median_optional(values: Sequence[float | None]) -> float | None:
    if not values or any(value is None for value in values):
        return None
    return float(median(cast(Sequence[float], values)))


def _manifest_tasks(payload: object) -> list[dict[str, str]]:
    root = _mapping(payload, "core set manifest")
    if root.get("schema_version") != "scope-expansion-v0-core-set-v1":
        raise ScopeAggregationError("core set manifest has an unsupported schema_version")
    if root.get("expected_task_count") != EXPECTED_TASK_COUNT:
        raise ScopeAggregationError("core set manifest expected_task_count must be 7")
    if root.get("task_count") != EXPECTED_TASK_COUNT:
        raise ScopeAggregationError("core set manifest task_count must be 7")
    tasks: list[dict[str, str]] = []
    for index, raw in enumerate(_list(root.get("tasks"), "core set manifest tasks")):
        item = _mapping(raw, f"core set manifest task {index}")
        tasks.append(
            {
                "task_id": str(item.get("task_id", "")),
                "dataset": str(item.get("dataset", "")),
                "task_family": str(item.get("task_family", "")),
                "role": str(item.get("role", "")),
            }
        )
    ids = [item["task_id"] for item in tasks]
    if any(not task_id for task_id in ids) or len(ids) != len(set(ids)):
        raise ScopeAggregationError("core set manifest task IDs must be non-empty and unique")
    return tasks


def _cell_from_scope(raw: object, source_name: str) -> _Cell:
    item = _mapping(raw, f"{source_name} cell")
    medians_raw = _mapping(item.get("medians"), f"{source_name} cell medians")
    medians = {
        str(key): _optional_number(value) for key, value in medians_raw.items()
    }
    completed_warmups = int(item.get("completed_warmups", -1))
    completed_runs = int(item.get("completed_runs", -1))
    declared_protocol = bool(item.get("profiling_protocol_passed", False))
    protocol_complete = (
        declared_protocol
        and completed_warmups == 1
        and completed_runs == 3
        and item.get("selected_protocol_id") == FORMAL_SCOPE_PROTOCOL_ID
    )
    return _Cell(
        task_id=str(item.get("task_id", "")),
        workflow_id=str(item.get("workflow_id", "")),
        world_id=str(item.get("world_id", "")),
        protocol_complete=protocol_complete,
        quality_gate_passed=bool(item.get("quality_gate_passed", False)),
        quality_pass_rate=float(item.get("quality_pass_rate", 0.0)),
        quality_median=_optional_number(item.get("quality_median")),
        median_e2e_ms=medians.get("e2e_latency_ms"),
        medians=medians,
        measurement_series_id=(
            str(item["selected_measurement_series_id"])
            if item.get("selected_measurement_series_id") is not None
            else None
        ),
        protocol_id=(
            str(item["selected_protocol_id"])
            if item.get("selected_protocol_id") is not None
            else None
        ),
    )


def _scope_report_tasks(
    payload: object, source_name: str
) -> tuple[dict[str, _TaskSource], dict[str, float]]:
    root = _mapping(payload, source_name)
    if root.get("schema_version") != "scope-expansion-v0-sensitivity-report-v1":
        raise ScopeAggregationError(f"{source_name} has an unsupported schema_version")
    cells_by_task: dict[str, list[_Cell]] = defaultdict(list)
    for raw in _list(root.get("cells"), f"{source_name} cells"):
        cell = _cell_from_scope(raw, source_name)
        cells_by_task[cell.task_id].append(cell)
    output: dict[str, _TaskSource] = {}
    for raw in _list(root.get("tasks"), f"{source_name} tasks"):
        item = _mapping(raw, f"{source_name} task")
        task_id = str(item.get("task_id", ""))
        if not task_id or task_id in output:
            raise ScopeAggregationError(
                f"{source_name} task IDs must be non-empty and unique: {task_id!r}"
            )
        labels_raw = [str(value) for value in _list(item.get("labels"), "task labels")]
        unknown_labels = sorted(set(labels_raw) - set(LABELS))
        if unknown_labels:
            raise ScopeAggregationError(
                f"{source_name} task {task_id!r} has unknown labels: {unknown_labels}"
            )
        cells = tuple(cells_by_task.pop(task_id, []))
        series = {cell.measurement_series_id for cell in cells}
        series_consistent = len(series) == 1 and None not in series
        quality_raw = _mapping(
            item.get("quality_by_workflow"), f"{source_name} quality_by_workflow"
        )
        output[task_id] = _TaskSource(
            source_name=source_name,
            cells=cells,
            labels=cast(tuple[ScopeSensitivityLabel, ...], tuple(labels_raw)),
            evidence=cast(
                dict[str, object],
                _mapping(item.get("evidence"), f"{source_name} evidence"),
            ),
            quality_by_workflow={
                str(key): _optional_number(value) for key, value in quality_raw.items()
            },
            series_consistent=series_consistent,
        )
    if cells_by_task:
        raise ScopeAggregationError(
            f"{source_name} has cells without task summaries: {sorted(cells_by_task)}"
        )
    thresholds_raw = _mapping(root.get("thresholds", {}), f"{source_name} thresholds")
    thresholds = {
        str(key): float(value)
        for key, value in thresholds_raw.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    return output, thresholds


def _cell_from_calibration(raw: object, minimum_runs: int) -> _Cell:
    item = _mapping(raw, "calibration cell")
    medians_raw = _mapping(item.get("medians"), "calibration cell medians")
    medians = {
        str(key): _optional_number(value) for key, value in medians_raw.items()
    }
    completed_runs = int(item.get("completed_runs", -1))
    return _Cell(
        task_id=str(item.get("task_id", "")),
        workflow_id=str(item.get("workflow_id", "")),
        world_id=str(item.get("world_id", "")),
        protocol_complete=completed_runs == minimum_runs,
        quality_gate_passed=bool(item.get("quality_gate_passed", False)),
        quality_pass_rate=float(item.get("quality_pass_rate", 0.0)),
        quality_median=medians.get("task_quality"),
        median_e2e_ms=medians.get("e2e_latency_ms"),
        medians=medians,
        measurement_series_id="calibration-v0-legacy-measured-series",
        protocol_id="calibration-v0-legacy-3-measured",
    )


def _video_labels(cells: Sequence[_Cell]) -> tuple[
    tuple[ScopeSensitivityLabel, ...], dict[str, object], dict[ScopeSensitivityLabel, LabelStatus]
]:
    lookup = {(cell.workflow_id, cell.world_id): cell for cell in cells}
    communication: list[float] = []
    reductions: list[float] = []
    quality_pass_rate_drops: list[float] = []
    for world in WORLDS:
        central = lookup.get((CENTRALIZED, world))
        reduced = lookup.get(("local_reduction", world))
        if central is None or reduced is None:
            continue
        byte_difference = _relative_difference(
            central.medians.get("cross_agent_transfer_bytes"),
            reduced.medians.get("cross_agent_transfer_bytes"),
        )
        latency_difference = _relative_difference(
            central.medians.get("transfer_sum_ms")
            or central.medians.get("transfer_latency_ms"),
            reduced.medians.get("transfer_sum_ms")
            or reduced.medians.get("transfer_latency_ms"),
        )
        communication.append(max(byte_difference or 0.0, latency_difference or 0.0))
        raw = reduced.medians.get("raw_artifact_bytes")
        compact = reduced.medians.get("reduced_artifact_bytes")
        reductions.append((raw - compact) / raw if raw and compact is not None else 0.0)
        quality_pass_rate_drops.append(
            max(0.0, central.quality_pass_rate - reduced.quality_pass_rate)
        )
    communication_value = max(communication, default=0.0)
    representation_value = max(reductions, default=0.0)
    quality_drop = max(quality_pass_rate_drops, default=0.0)
    status: dict[ScopeSensitivityLabel, LabelStatus] = {
        "workflow_reversal": "not_triggered",
        "communication_sensitive": (
            "triggered"
            if communication_value >= SYSTEM_DIFFERENCE_THRESHOLD
            else "not_triggered"
        ),
        "compute_sensitive": "not_evaluable",
        "representation_sensitive": (
            "triggered"
            if representation_value >= SYSTEM_DIFFERENCE_THRESHOLD
            else "not_triggered"
        ),
        "quality_risk": "triggered" if quality_drop > 0 else "not_triggered",
        "parallelism_sensitive": "not_evaluable",
        "context_cost_sensitive": "not_applicable",
    }
    evidence: dict[str, object] = {
        "communication_sensitive": {
            "triggered": status["communication_sensitive"] == "triggered",
            "max_relative_difference": communication_value,
        },
        "representation_sensitive": {
            "triggered": status["representation_sensitive"] == "triggered",
            "max_raw_to_evidence_byte_reduction": representation_value,
        },
        "quality_risk": {
            "triggered": status["quality_risk"] == "triggered",
            "max_centralized_minus_reduced_quality_pass_rate": quality_drop,
        },
        "compute_sensitive": {"triggered": False, "reason": "critical path unavailable"},
        "parallelism_sensitive": {
            "triggered": False,
            "reason": "critical path unavailable",
        },
        "context_cost_sensitive": {
            "triggered": False,
            "reason": "multimodal and text token accounting are not comparable",
        },
    }
    labels = cast(
        tuple[ScopeSensitivityLabel, ...],
        tuple(label for label in LABELS if status[label] == "triggered"),
    )
    return labels, evidence, status


def _calibration_tasks(payload: object) -> dict[str, _TaskSource]:
    root = _mapping(payload, "calibration summary")
    if not str(root.get("schema_version", "")).startswith("calibration-v0-summary-"):
        raise ScopeAggregationError("calibration summary has an unsupported schema_version")
    policy = _mapping(root.get("policy", {}), "calibration policy")
    minimum_runs = int(policy.get("minimum_completed_runs_per_cell", 3))
    cells_by_task: dict[str, list[_Cell]] = defaultdict(list)
    for raw in _list(root.get("cells"), "calibration cells"):
        cell = _cell_from_calibration(raw, minimum_runs)
        cells_by_task[cell.task_id].append(cell)
    output: dict[str, _TaskSource] = {}
    for task_id, cells in cells_by_task.items():
        labels, evidence, _ = _video_labels(cells)
        quality_by_workflow = {
            workflow: _median_optional(
                [cell.quality_median for cell in cells if cell.workflow_id == workflow]
            )
            for workflow in (CENTRALIZED, "local_reduction")
        }
        output[task_id] = _TaskSource(
            source_name="calibration_v0",
            cells=tuple(cells),
            labels=labels,
            evidence=evidence,
            quality_by_workflow=quality_by_workflow,
            series_consistent=True,
        )
    return output


def _winner_for_world(
    cells: Mapping[tuple[str, str], _Cell],
    distributed_workflow: str,
    world: str,
    *,
    series_consistent: bool,
) -> dict[str, object]:
    central = cells.get((CENTRALIZED, world))
    distributed = cells.get((distributed_workflow, world))
    medians = {
        CENTRALIZED: central.median_e2e_ms if central else None,
        distributed_workflow: distributed.median_e2e_ms if distributed else None,
    }
    if central is None or distributed is None:
        return {"winner": None, "margin": None, "status": "cell_missing", "e2e_ms": medians}
    if not series_consistent:
        return {
            "winner": None,
            "margin": None,
            "status": "measurement_series_mismatch",
            "e2e_ms": medians,
        }
    if not (central.protocol_complete and distributed.protocol_complete):
        return {
            "winner": None,
            "margin": None,
            "status": "protocol_incomplete",
            "e2e_ms": medians,
        }
    if not (central.quality_gate_passed and distributed.quality_gate_passed):
        return {
            "winner": None,
            "margin": None,
            "status": "quality_gate_failed",
            "e2e_ms": medians,
        }
    if central.median_e2e_ms is None or distributed.median_e2e_ms is None:
        return {
            "winner": None,
            "margin": None,
            "status": "metric_missing",
            "e2e_ms": medians,
        }
    if central.median_e2e_ms <= distributed.median_e2e_ms:
        winner = CENTRALIZED
        winner_e2e = central.median_e2e_ms
        loser_e2e = distributed.median_e2e_ms
    else:
        winner = distributed_workflow
        winner_e2e = distributed.median_e2e_ms
        loser_e2e = central.median_e2e_ms
    margin = (loser_e2e - winner_e2e) / winner_e2e if winner_e2e else 0.0
    return {"winner": winner, "margin": margin, "status": "comparable", "e2e_ms": medians}


def _task_report(
    manifest: Mapping[str, str], source: _TaskSource
) -> dict[str, object]:
    family = manifest["task_family"]
    distributed = FAMILY_DISTRIBUTED_WORKFLOW[family]
    lookup = {(cell.workflow_id, cell.world_id): cell for cell in source.cells}
    winners = {
        world: _winner_for_world(
            lookup, distributed, world, series_consistent=source.series_consistent
        )
        for world in WORLDS
    }
    comparison_eligible = all(
        winners[world]["status"] == "comparable" for world in WORLDS
    )
    reversal = bool(
        comparison_eligible
        and winners[H1]["winner"] != winners[H2]["winner"]
    )
    labels = list(source.labels)
    evidence = dict(source.evidence)
    if family == "long_video_qa":
        _, _, label_status = _video_labels(source.cells)
    else:
        label_status = {
            label: ("triggered" if label in source.labels else "not_triggered")
            for label in LABELS
        }
    label_status["workflow_reversal"] = (
        "triggered" if reversal else "not_triggered"
    )
    if reversal and "workflow_reversal" not in labels:
        labels.insert(0, "workflow_reversal")
    if not reversal and "workflow_reversal" in labels:
        labels.remove("workflow_reversal")
    evidence["workflow_reversal"] = {
        "triggered": reversal,
        "preferences": {world: winners[world]["winner"] for world in WORLDS},
        "quality_gated": comparison_eligible,
        "measurement_series_consistent": source.series_consistent,
    }
    cell_rows = [
        {
            "workflow_id": cell.workflow_id,
            "world_id": cell.world_id,
            "protocol_complete": cell.protocol_complete,
            "quality_gate_passed": cell.quality_gate_passed,
            "quality_pass_rate": cell.quality_pass_rate,
            "quality_median": cell.quality_median,
            "median_e2e_latency_ms": cell.median_e2e_ms,
            "measurement_series_id": cell.measurement_series_id,
            "protocol_id": cell.protocol_id,
        }
        for cell in sorted(source.cells, key=lambda item: (item.world_id, item.workflow_id))
    ]
    return {
        **dict(manifest),
        "source_report": source.source_name,
        "workflow_pair": [CENTRALIZED, distributed],
        "protocol_complete": len(source.cells) == 4
        and all(cell.protocol_complete for cell in source.cells)
        and source.series_consistent,
        "measurement_series_consistent": source.series_consistent,
        "cells": cell_rows,
        "quality_by_workflow": source.quality_by_workflow,
        "winner_by_world": winners,
        "comparison_eligible": comparison_eligible,
        "workflow_reversal": reversal,
        "labels": labels,
        "label_status": label_status,
        "evidence": evidence,
    }


def _markdown(report: Mapping[str, Any]) -> str:
    completeness = cast(dict[str, Any], report["completeness"])
    lines = [
        "# Scope Expansion v0 top-level sensitivity summary",
        "",
        "Efficiency winners are reported only when both workflows pass the original evaluator quality gate. Warm-ups never enter medians or quality decisions.",
        "",
        "## Completeness",
        "",
        f"- Complete: `{str(report['complete']).lower()}`",
        f"- Tasks: {completeness['reported_task_count']}/{completeness['expected_task_count']}",
        f"- Cells: {completeness['reported_cell_count']}/{completeness['expected_cell_count']}",
        f"- Protocol-complete cells: {completeness['protocol_complete_cell_count']}/{completeness['expected_cell_count']}",
        "",
        "## Tasks",
        "",
        "| Task | Family | Protocol | Quality-comparable | H1 winner | H2 winner | Labels |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for task in cast(list[dict[str, Any]], report["tasks"]):
        winners = cast(dict[str, dict[str, Any]], task["winner_by_world"])
        labels = ", ".join(cast(list[str], task["labels"])) or "-"
        lines.append(
            f"| {task['task_id']} | {task['task_family']} | "
            f"{'yes' if task['protocol_complete'] else 'no'} | "
            f"{'yes' if task['comparison_eligible'] else 'no'} | "
            f"{winners[H1]['winner'] or '-'} | {winners[H2]['winner'] or '-'} | "
            f"{labels} |"
        )
    lines.extend(
        [
            "",
            "## Family summary",
            "",
            "| Family | Tasks | Protocol complete | Quality-comparable | Triggered labels |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for family, values in cast(dict[str, dict[str, Any]], report["families"]).items():
        label_counts = cast(dict[str, int], values["label_counts"])
        triggered = ", ".join(
            f"{label}={count}" for label, count in label_counts.items() if count
        ) or "-"
        lines.append(
            f"| {family} | {values['task_count']} | {values['protocol_complete_task_count']} | "
            f"{values['quality_comparable_task_count']} | {triggered} |"
        )
    lines.extend(["", "## Sources", ""])
    for source in cast(list[dict[str, Any]], report["sources"]):
        lines.append(
            f"- `{source['path']}` — `{source['sha256']}` ({source['size_bytes']} bytes)"
        )
    lines.append("")
    return "\n".join(lines)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def aggregate_scope_expansion_v0(
    *,
    manifest_path: Path,
    task_bank_path: Path,
    calibration_summary_path: Path,
    multihop_summary_path: Path,
    longbench_multidoc_summary_path: Path,
    longbench_structured_summary_path: Path,
    strict: bool = True,
) -> tuple[dict[str, object], tuple[SourceSnapshot, ...]]:
    """Load stable source summaries and build one family-aware seven-task report."""
    snapshots: list[SourceSnapshot] = []
    manifest_payload, snapshot = _read_stable_json(manifest_path)
    snapshots.append(snapshot)
    manifest_tasks = _manifest_tasks(manifest_payload)

    task_bank_rows, snapshot = _read_stable_jsonl(task_bank_path)
    snapshots.append(snapshot)
    task_bank_ids: list[str] = []
    for row in task_bank_rows:
        visible = row.get("planner_visible", row)
        task_bank_ids.append(str(_mapping(visible, "task bank planner_visible").get("task_id", "")))

    sources: dict[str, _TaskSource] = {}
    source_thresholds: list[dict[str, float]] = []
    source_specs = (
        ("calibration_v0", calibration_summary_path, "calibration"),
        ("multihop_rag", multihop_summary_path, "scope"),
        ("longbench_multidoc", longbench_multidoc_summary_path, "scope"),
        ("longbench_structured", longbench_structured_summary_path, "scope"),
    )
    missing_source_paths: list[str] = []
    for source_name, path, source_type in source_specs:
        if not path.is_file():
            missing_source_paths.append(path.as_posix())
            continue
        payload, snapshot = _read_stable_json(path)
        snapshots.append(snapshot)
        if source_type == "calibration":
            loaded = _calibration_tasks(payload)
        else:
            loaded, thresholds = _scope_report_tasks(payload, source_name)
            source_thresholds.append(thresholds)
        duplicate = sorted(set(sources) & set(loaded))
        if duplicate:
            raise ScopeAggregationError(
                f"tasks occur in more than one source report: {duplicate}"
            )
        sources.update(loaded)

    _verify_snapshots(snapshots)
    manifest_by_id = {item["task_id"]: item for item in manifest_tasks}
    expected_ids = set(manifest_by_id)
    reported_ids = set(sources)
    family_counts = Counter(item["task_family"] for item in manifest_tasks)
    non_video_ids = {
        item["task_id"]
        for item in manifest_tasks
        if item["task_family"] != "long_video_qa"
    }
    task_bank_duplicates = sorted(
        task_id for task_id, count in Counter(task_bank_ids).items() if count > 1
    )
    missing_task_bank_ids = sorted(non_video_ids - set(task_bank_ids))
    unexpected_task_bank_ids = sorted(set(task_bank_ids) - non_video_ids)
    missing_report_ids = sorted(expected_ids - reported_ids)
    unexpected_report_ids = sorted(reported_ids - expected_ids)
    expected_source_by_task = {
        item["task_id"]: (
            "calibration_v0"
            if item["task_family"] == "long_video_qa"
            else "multihop_rag"
            if item["dataset"] == "MultiHop-RAG"
            else "longbench_multidoc"
            if item["role"] == "extra_long_multidoc"
            else "longbench_structured"
        )
        for item in manifest_tasks
    }
    source_mismatch_tasks = sorted(
        task_id
        for task_id in expected_ids & reported_ids
        if sources[task_id].source_name != expected_source_by_task[task_id]
    )

    task_reports: list[dict[str, object]] = []
    missing_cells: list[str] = []
    unexpected_cells: list[str] = []
    protocol_incomplete_cells: list[str] = []
    series_mismatch_tasks: list[str] = []
    for task_id in sorted(expected_ids & reported_ids):
        manifest = manifest_by_id[task_id]
        family = manifest["task_family"]
        if family not in FAMILY_DISTRIBUTED_WORKFLOW:
            raise ScopeAggregationError(f"task {task_id!r} has unsupported family {family!r}")
        source = sources[task_id]
        if not source.series_consistent:
            series_mismatch_tasks.append(task_id)
        expected_keys = {
            (workflow, world)
            for workflow in (CENTRALIZED, FAMILY_DISTRIBUTED_WORKFLOW[family])
            for world in WORLDS
        }
        actual_keys = {(cell.workflow_id, cell.world_id) for cell in source.cells}
        for workflow, world in sorted(expected_keys - actual_keys):
            missing_cells.append(f"{task_id}:{workflow}:{world}")
        for workflow, world in sorted(actual_keys - expected_keys):
            unexpected_cells.append(f"{task_id}:{workflow}:{world}")
        if len(actual_keys) != len(source.cells):
            unexpected_cells.append(f"{task_id}:duplicate_cell")
        for cell in source.cells:
            if not cell.protocol_complete:
                protocol_incomplete_cells.append(
                    f"{task_id}:{cell.workflow_id}:{cell.world_id}"
                )
        task_reports.append(_task_report(manifest, source))

    reported_cell_count = sum(len(sources[task_id].cells) for task_id in expected_ids & reported_ids)
    protocol_complete_cell_count = sum(
        cell.protocol_complete
        for task_id in expected_ids & reported_ids
        for cell in sources[task_id].cells
    )
    identity_complete = (
        len(manifest_tasks) == EXPECTED_TASK_COUNT
        and family_counts == Counter(EXPECTED_FAMILY_COUNTS)
    )
    task_bank_complete = (
        not task_bank_duplicates
        and not missing_task_bank_ids
        and not unexpected_task_bank_ids
    )
    report_coverage_complete = (
        not missing_source_paths
        and not missing_report_ids
        and not unexpected_report_ids
        and not source_mismatch_tasks
        and len(reported_ids) == EXPECTED_TASK_COUNT
    )
    cell_matrix_complete = (
        not missing_cells
        and not unexpected_cells
        and reported_cell_count == EXPECTED_CELL_COUNT
    )
    protocol_complete = (
        cell_matrix_complete
        and not protocol_incomplete_cells
        and not series_mismatch_tasks
        and protocol_complete_cell_count == EXPECTED_CELL_COUNT
    )
    complete = all(
        (
            identity_complete,
            task_bank_complete,
            report_coverage_complete,
            cell_matrix_complete,
            protocol_complete,
        )
    )
    completeness: dict[str, object] = {
        "expected_task_count": EXPECTED_TASK_COUNT,
        "manifest_task_count": len(manifest_tasks),
        "reported_task_count": len(reported_ids & expected_ids),
        "expected_cell_count": EXPECTED_CELL_COUNT,
        "reported_cell_count": reported_cell_count,
        "protocol_complete_cell_count": protocol_complete_cell_count,
        "identity_complete": identity_complete,
        "task_bank_complete": task_bank_complete,
        "report_coverage_complete": report_coverage_complete,
        "cell_matrix_complete": cell_matrix_complete,
        "protocol_complete": protocol_complete,
        "missing_source_paths": missing_source_paths,
        "missing_task_bank_ids": missing_task_bank_ids,
        "duplicate_task_bank_ids": task_bank_duplicates,
        "unexpected_task_bank_ids": unexpected_task_bank_ids,
        "missing_report_task_ids": missing_report_ids,
        "unexpected_report_task_ids": unexpected_report_ids,
        "source_mismatch_task_ids": source_mismatch_tasks,
        "missing_cells": missing_cells,
        "unexpected_cells": unexpected_cells,
        "protocol_incomplete_cells": protocol_incomplete_cells,
        "measurement_series_mismatch_task_ids": series_mismatch_tasks,
    }
    if strict and not complete:
        failures = [
            key
            for key in (
                "identity_complete",
                "task_bank_complete",
                "report_coverage_complete",
                "cell_matrix_complete",
                "protocol_complete",
            )
            if not completeness[key]
        ]
        raise ScopeAggregationError(
            "strict Scope Expansion aggregation failed: " + ", ".join(failures)
        )

    families: dict[str, dict[str, object]] = {}
    for family in FAMILY_DISTRIBUTED_WORKFLOW:
        family_tasks = [task for task in task_reports if task["task_family"] == family]
        families[family] = {
            "task_count": len(family_tasks),
            "protocol_complete_task_count": sum(
                bool(task["protocol_complete"]) for task in family_tasks
            ),
            "quality_comparable_task_count": sum(
                bool(task["comparison_eligible"]) for task in family_tasks
            ),
            "workflow_reversal_task_count": sum(
                bool(task["workflow_reversal"]) for task in family_tasks
            ),
            "label_counts": {
                label: sum(label in cast(list[str], task["labels"]) for task in family_tasks)
                for label in LABELS
            },
        }
    thresholds = source_thresholds[0] if source_thresholds else {
        "relative_system_difference": SYSTEM_DIFFERENCE_THRESHOLD
    }
    if strict and any(item != thresholds for item in source_thresholds[1:]):
        raise ScopeAggregationError("Scope family reports use inconsistent thresholds")
    report: dict[str, object] = {
        "schema_version": "scope-expansion-v0-top-level-summary-v1",
        "complete": complete,
        "policy": {
            "quality_before_efficiency": True,
            "winner": "lower measured median E2E only after both workflows pass quality",
            "cross_world_reversal_requires_both_worlds_comparable": True,
            "warmups_excluded": True,
            "calibration_protocol_note": (
                "The archived calibration_v0 source uses its legacy three-measured-run "
                "protocol and has no warmup field."
            ),
        },
        "thresholds": thresholds,
        "completeness": completeness,
        "families": families,
        "tasks": sorted(task_reports, key=lambda item: str(item["task_id"])),
        "sources": [snapshot.as_dict() for snapshot in snapshots],
        "totals": {
            "task_count": len(task_reports),
            "cell_count": reported_cell_count,
            "protocol_complete_task_count": sum(
                bool(task["protocol_complete"]) for task in task_reports
            ),
            "quality_comparable_task_count": sum(
                bool(task["comparison_eligible"]) for task in task_reports
            ),
            "workflow_reversal_task_count": sum(
                bool(task["workflow_reversal"]) for task in task_reports
            ),
            "quality_blocked_task_count": sum(
                not bool(task["comparison_eligible"]) for task in task_reports
            ),
        },
    }
    return report, tuple(snapshots)


def write_scope_expansion_top_level_report(
    *,
    manifest_path: Path,
    task_bank_path: Path,
    calibration_summary_path: Path,
    multihop_summary_path: Path,
    longbench_multidoc_summary_path: Path,
    longbench_structured_summary_path: Path,
    output_dir: Path,
    strict: bool = True,
) -> dict[str, object]:
    """Aggregate stable summaries and atomically emit top-level JSON and Markdown."""
    report, snapshots = aggregate_scope_expansion_v0(
        manifest_path=manifest_path,
        task_bank_path=task_bank_path,
        calibration_summary_path=calibration_summary_path,
        multihop_summary_path=multihop_summary_path,
        longbench_multidoc_summary_path=longbench_multidoc_summary_path,
        longbench_structured_summary_path=longbench_structured_summary_path,
        strict=strict,
    )
    _verify_snapshots(snapshots)
    encoded = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    markdown = _markdown(cast(Mapping[str, Any], report))
    _atomic_write(output_dir / "sensitivity_summary.json", encoded)
    _atomic_write(output_dir / "sensitivity_summary.md", markdown)
    return report
