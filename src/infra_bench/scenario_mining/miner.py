"""Offline shortlist construction for normalized ScenarioRecord objects."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from infra_bench.io import write_json, write_jsonl
from infra_bench.schemas import ScenarioRecord

from .screening import modeled_screening
from .strategy import estimate_demands, validate_strategy_multiplicity


def shortlist_score(scenario: ScenarioRecord, strategy_multiplicity: bool) -> float | None:
    features = scenario.scene_features
    if features.raw_evidence_ratio is None or features.evidence_dispersion is None:
        return None
    return (
        math.log1p(features.raw_evidence_ratio)
        + math.log1p(features.artifact_count)
        + features.evidence_dispersion
        + float(strategy_multiplicity)
    )


def _classify(
    scenario: ScenarioRecord, strategy_multiplicity: bool
) -> tuple[str, list[str]]:
    features = scenario.scene_features
    reasons: list[str] = []
    if features.artifact_count < 3:
        reasons.append("artifact_count < 3")
    if not features.deterministic_evaluator:
        reasons.append("evaluator is not deterministic")
    if not strategy_multiplicity:
        reasons.append("generic strategies are not both MAS-realizable")
    evidence_unknown = (
        features.evidence_bytes is None
        or features.raw_evidence_ratio is None
        or features.evidence_artifact_count is None
        or features.evidence_dispersion is None
    )
    if evidence_unknown:
        return "evidence_unknown", reasons
    evidence_artifact_count = features.evidence_artifact_count
    raw_evidence_ratio = features.raw_evidence_ratio
    if evidence_artifact_count is None or raw_evidence_ratio is None:
        raise AssertionError("known evidence classification lost required features")
    if evidence_artifact_count < 2:
        reasons.append("evidence_artifact_count < 2")
    if raw_evidence_ratio < 4.0:
        reasons.append("raw_evidence_ratio < 4")
    return ("strong_candidate" if not reasons else "rejected"), reasons


def assess_scenario(scenario: ScenarioRecord) -> dict[str, Any]:
    multiplicity = validate_strategy_multiplicity(scenario)
    centralized, distributed = estimate_demands(scenario)
    queue, rejection_reasons = _classify(
        scenario, multiplicity.strategy_multiplicity
    )
    score = shortlist_score(scenario, multiplicity.strategy_multiplicity)
    screening = modeled_screening(scenario, centralized, distributed)
    return {
        "scenario_id": scenario.scenario_id,
        "dataset": scenario.dataset,
        "task_id": scenario.task_id,
        "features": scenario.scene_features.model_dump(mode="json"),
        "strategy_multiplicity": multiplicity.model_dump(mode="json"),
        "demands": {
            "centralized_raw": centralized.model_dump(mode="json"),
            "distributed_reduction": distributed.model_dump(mode="json"),
        },
        "queue": queue,
        "shortlist_score": score,
        "shortlist_reason": (
            "passes the fixed multi-source/compressible/verifiable/realizable gates"
            if queue == "strong_candidate"
            else (
                "benchmark evidence size/dispersion is unavailable; retained separately"
                + (
                    f"; other gate observations: {'; '.join(rejection_reasons)}"
                    if rejection_reasons
                    else ""
                )
                if queue == "evidence_unknown"
                else "; ".join(rejection_reasons)
            )
        ),
        "modeled_screening": screening,
    }


def _rank_key(item: Mapping[str, Any]) -> tuple[float, str]:
    score = item.get("shortlist_score")
    numeric = float(score) if isinstance(score, int | float) else -1.0
    return (-numeric, str(item["scenario_id"]))


def _queue_entry(item: Mapping[str, Any]) -> dict[str, Any]:
    """Keep fail-closed queue records auditable without duplicating full screenings."""

    return {
        "scenario_id": item["scenario_id"],
        "dataset": item["dataset"],
        "task_id": item["task_id"],
        "features": item["features"],
        "strategy_multiplicity": item["strategy_multiplicity"],
        "queue": item["queue"],
        "shortlist_score": item["shortlist_score"],
        "shortlist_reason": item["shortlist_reason"],
    }


def _balanced_shortlist(
    candidates: Sequence[dict[str, Any]],
    *,
    limit: int,
    source_minimums: Mapping[str, int],
) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in sorted(candidates, key=_rank_key):
        by_source[str(item["dataset"])].append(item)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for dataset, minimum in source_minimums.items():
        for item in by_source.get(dataset, [])[:minimum]:
            if len(selected) >= limit:
                break
            selected.append(item)
            selected_ids.add(str(item["scenario_id"]))
    for item in sorted(candidates, key=_rank_key):
        if len(selected) >= limit:
            break
        if str(item["scenario_id"]) not in selected_ids:
            selected.append(item)
            selected_ids.add(str(item["scenario_id"]))
    return sorted(selected, key=_rank_key)


def _dataset_stats(
    scenarios: Sequence[ScenarioRecord],
    assessments: Sequence[Mapping[str, Any]],
    raw_candidates: Sequence[Mapping[str, Any]],
    shortlist: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    totals = Counter(item.dataset for item in scenarios)
    queues: dict[str, Counter[str]] = defaultdict(Counter)
    for item in assessments:
        queues[str(item["dataset"])][str(item["queue"])] += 1
    raw_counts = Counter(str(item["dataset"]) for item in raw_candidates)
    shortlist_counts = Counter(str(item["dataset"]) for item in shortlist)
    return {
        "total_scenarios": len(scenarios),
        "datasets": {
            dataset: {
                "total": total,
                **dict(queues[dataset]),
                "raw_candidates": raw_counts[dataset],
                "shortlist": shortlist_counts[dataset],
            }
            for dataset, total in sorted(totals.items())
        },
    }


def render_scenario_mining_report(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Multi-document QA scenario mining report",
        "",
        "This is a fully offline structural screen. A modeled reversal is a calibration "
        "candidate, not admission evidence.",
        "",
        "## Summary",
        "",
        "| Input tasks | Raw candidates | Strong shortlist | Evidence unknown | "
        "Calibration candidates | Modeled reversals |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {summary['input_scenarios']} | {summary['raw_candidates']} | "
            f"{summary['shortlist']} | {summary['evidence_unknown']} | "
            f"{summary['calibration_candidates']} | "
            f"{summary['modeled_reversal_candidates']} |"
        ),
        "",
        "## Strong shortlist",
        "",
        "| Rank | Dataset | Task | K | Raw bytes | Evidence bytes | Raw/evidence | "
        "Evidence artifacts | Dispersion | Strategies realizable | Central demand | "
        "Distributed demand | Perturbation | Modeled reversal | Reason |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | "
        "--- | --- | --- | --- |",
    ]
    for rank, item in enumerate(report["shortlist"], 1):
        features = item["features"]
        demands = item["demands"]
        central = demands["centralized_raw"]
        distributed = demands["distributed_reduction"]
        ratio = features["raw_evidence_ratio"]
        lines.append(
            f"| {rank} | {item['dataset']} | `{item['task_id']}` | "
            f"{features['artifact_count']} | {features['raw_bytes']} | "
            f"{features['evidence_bytes']} | {ratio:.3f} | "
            f"{features['evidence_artifact_count']} | "
            f"{features['evidence_dispersion']:.3f} | "
            f"{'YES' if item['strategy_multiplicity']['strategy_multiplicity'] else 'NO'} | "
            f"input={central['reasoning_input_bytes']}, stages={central['reasoning_stages']} | "
            f"input={distributed['reasoning_input_bytes']}, width={distributed['potential_parallel_width']} | "
            f"{item['modeled_screening']['recommended_infra_perturbation']} | "
            f"{'YES' if item['modeled_screening']['modeled_reversal'] else 'NO'} | "
            f"{item['shortlist_reason']} |"
        )
    lines.extend(
        [
            "",
            "## Evidence-unknown queue",
            "",
            "Tasks without published evidence are retained separately and never assigned "
            "invented evidence statistics. Dataset-level counts:",
            "",
            "| Dataset | Evidence unknown | Other gate outcomes |",
            "| --- | ---: | --- |",
        ]
    )
    for dataset, stats in report["dataset_stats"]["datasets"].items():
        unknown_count = stats.get("evidence_unknown", 0)
        if not unknown_count:
            continue
        lines.append(
            f"| {dataset} | {unknown_count} | Raw candidates: "
            f"{stats['raw_candidates']}; shortlist: {stats['shortlist']} |"
        )
    longbench_stats = report["dataset_stats"]["datasets"].get("longbench-v2", {})
    if longbench_stats.get("evidence_unknown", 0):
        lines.extend(
            [
                "",
                "LongBench-v2 publishes each selected task as one opaque `context` string "
                "without evidence or document/file/table boundaries. The miner therefore "
                "retains these tasks as evidence-unknown but does not invent chunks to make "
                "them pass the multi-artifact gate.",
            ]
        )
    lines.extend(
        [
            "",
            "## Research boundary",
            "",
            "- Scenario features describe task semantics T only.",
            "- WorkflowDemandTemplate describes intrinsic D(G) only.",
            "- Placement, modeled cross-site bytes, network time, and service assumptions "
            "exist only in M(D, H).",
            "- Supporting facts and answers are hidden mining/evaluator annotations; neither "
            "generic strategy uses them to preselect artifacts.",
            "- Distributed output size uses hidden evidence bytes only as an optimistic offline "
            "lower bound; real calibration must execute reduction over every artifact.",
            "- No modeled result is written to `admitted_pairs.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def mine_scenarios(
    scenarios: Sequence[ScenarioRecord],
    *,
    output_dir: str | Path,
    raw_candidate_limit: int = 40,
    shortlist_limit: int = 15,
    evidence_unknown_limit: int = 10,
    source_minimums: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Mine, rank, and write the complete offline scenario artifact set."""

    assessments = [assess_scenario(item) for item in scenarios]
    strong = sorted(
        [item for item in assessments if item["queue"] == "strong_candidate"],
        key=_rank_key,
    )
    unknown = sorted(
        [item for item in assessments if item["queue"] == "evidence_unknown"],
        key=lambda item: str(item["scenario_id"]),
    )
    shortlist = _balanced_shortlist(
        strong,
        limit=shortlist_limit,
        source_minimums=source_minimums
        or {"2wikimultihopqa": 5, "musique": 3, "longbench-v2": 3},
    )
    raw_assessments = list(shortlist)
    raw_ids = {str(item["scenario_id"]) for item in raw_assessments}
    for item in strong:
        if len(raw_assessments) >= raw_candidate_limit:
            break
        scenario_id = str(item["scenario_id"])
        if scenario_id not in raw_ids:
            raw_assessments.append(item)
            raw_ids.add(scenario_id)
    unknown_added = 0
    for item in unknown:
        if len(raw_assessments) >= raw_candidate_limit or unknown_added >= evidence_unknown_limit:
            break
        scenario_id = str(item["scenario_id"])
        if scenario_id not in raw_ids:
            raw_assessments.append(item)
            raw_ids.add(scenario_id)
            unknown_added += 1
    raw_assessments.sort(key=_rank_key)
    scenario_by_id = {item.scenario_id: item for item in scenarios}
    raw_scenarios = [
        scenario_by_id[str(item["scenario_id"])] for item in raw_assessments
    ]
    calibration_candidates = [item["modeled_screening"] for item in shortlist]
    modeled_reversal_count = sum(
        item["modeled_screening"]["modeled_reversal"] for item in shortlist
    )
    stats = _dataset_stats(scenarios, assessments, raw_assessments, shortlist)
    report: dict[str, Any] = {
        "schema_version": "scenario-mining-report-v1",
        "summary": {
            "input_scenarios": len(scenarios),
            "raw_candidates": len(raw_scenarios),
            "shortlist": len(shortlist),
            "evidence_unknown": len(unknown),
            "calibration_candidates": len(calibration_candidates),
            "modeled_reversal_candidates": modeled_reversal_count,
            "admitted_pairs": 0,
            "model_api_calls": 0,
        },
        "thresholds": {
            "artifact_count": 3,
            "evidence_artifact_count": 2,
            "raw_evidence_ratio": 4.0,
            "deterministic_evaluator": True,
        },
        "shortlist": shortlist,
        "evidence_unknown": [_queue_entry(item) for item in unknown],
        "rejected": [
            _queue_entry(item)
            for item in assessments
            if item["queue"] == "rejected"
        ],
        "calibration_candidates": calibration_candidates,
        "dataset_stats": stats,
        "research_boundary": {
            "modeled_reversal_is_admission": False,
            "supporting_facts_planner_visible": False,
            "answers_planner_visible": False,
        },
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    write_jsonl(destination / "scenario_bank.jsonl", raw_scenarios)
    write_json(destination / "shortlist.json", shortlist)
    write_json(destination / "calibration_candidates.json", calibration_candidates)
    write_json(destination / "dataset_stats.json", stats)
    write_json(destination / "scenario_mining_report.json", report)
    (destination / "scenario_mining_report.md").write_text(
        render_scenario_mining_report(report), encoding="utf-8"
    )
    return report
