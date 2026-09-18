"""Configuration-driven offline scenario mining entry point."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from infra_bench.config import load_yaml
from infra_bench.schemas import ScenarioRecord

from .adapters import (
    LongBenchV2Adapter,
    MuSiQueAdapter,
    OfflineScenarioAdapter,
    TwoWikiMultiHopQAAdapter,
    iter_json_records,
)
from .miner import mine_scenarios


def _adapter(settings: Mapping[str, Any]) -> OfflineScenarioAdapter:
    revision = str(settings["revision"])
    split = str(settings["split"])
    embed_content = bool(settings.get("embed_content", False))
    name = str(settings["adapter"])
    if name == "2wikimultihopqa":
        return TwoWikiMultiHopQAAdapter(
            revision=revision,
            split=split,
            embed_content=embed_content,
            variant=str(settings.get("variant", "official-v1.0")),
        )
    if name == "musique":
        return MuSiQueAdapter(
            revision=revision,
            split=split,
            embed_content=embed_content,
            variant=str(settings.get("variant", "ans-v1.0")),
        )
    if name == "longbench-v2":
        return LongBenchV2Adapter(
            revision=revision, split=split, embed_content=embed_content
        )
    raise ValueError(f"unknown scenario adapter: {name}")


def build_scenario_bank(config: Mapping[str, Any]) -> list[ScenarioRecord]:
    scenarios: list[ScenarioRecord] = []
    datasets = config.get("datasets", [])
    if not isinstance(datasets, list):
        raise ValueError("scenario datasets must be a list")
    for raw_settings in cast(list[object], datasets):
        if not isinstance(raw_settings, dict):
            raise ValueError("scenario dataset settings must be mappings")
        settings = cast(dict[str, Any], raw_settings)
        source = Path(str(settings["path"])).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"scenario dataset input not found: {source}")
        adapter = _adapter(settings)
        raw_limit = settings.get("limit")
        limit = int(raw_limit) if raw_limit is not None else None
        scenarios.extend(adapter.ingest(iter_json_records(source), limit=limit))
    ids = [item.scenario_id for item in scenarios]
    if len(ids) != len(set(ids)):
        duplicates = sorted({item for item in ids if ids.count(item) > 1})
        raise ValueError(f"duplicate scenario IDs: {duplicates[:10]}")
    return scenarios


def run_scenario_mining(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    scenarios = build_scenario_bank(config)
    source_minimums = {
        str(key): int(value)
        for key, value in config.get("source_minimums", {}).items()
    }
    return mine_scenarios(
        scenarios,
        output_dir=Path(str(config.get("output_dir", "runs/scenario_mining"))).expanduser(),
        raw_candidate_limit=int(config.get("raw_candidate_limit", 40)),
        shortlist_limit=int(config.get("shortlist_limit", 15)),
        evidence_unknown_limit=int(config.get("evidence_unknown_limit", 10)),
        source_minimums=source_minimums,
    )
