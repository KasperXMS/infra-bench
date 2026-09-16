import random
from collections import defaultdict
from typing import Any

from ..schemas import BenchmarkCase


def _cases_by_task(cases: list[BenchmarkCase]) -> dict[str, list[BenchmarkCase]]:
    grouped: dict[str, list[BenchmarkCase]] = defaultdict(list)
    for case in cases:
        grouped[case.task_id].append(case)
    return grouped


def _partition_tasks(
    task_ids: list[str], seed: int, train_ratio: float = 0.6, dev_ratio: float = 0.2
) -> dict[str, list[str]]:
    shuffled = sorted(task_ids)
    random.Random(seed).shuffle(shuffled)
    count = len(shuffled)
    train_end = round(count * train_ratio)
    dev_end = train_end + round(count * dev_ratio)
    return {
        "train": sorted(shuffled[:train_end]),
        "dev": sorted(shuffled[train_end:dev_end]),
        "test": sorted(shuffled[dev_end:]),
    }


def _expand_cases(
    by_task: dict[str, list[BenchmarkCase]], task_splits: dict[str, list[str]]
) -> dict[str, dict[str, list[str]]]:
    return {
        split: {
            "task_ids": task_ids,
            "case_ids": sorted(
                case.case_id for task_id in task_ids for case in by_task[task_id]
            ),
        }
        for split, task_ids in task_splits.items()
    }


def _is_low_bandwidth_high_load(case: BenchmarkCase) -> bool:
    low_bandwidth = bool(case.infra.links) and min(
        link.bandwidth_mbps for link in case.infra.links
    ) <= 50.0
    relevant_loads = [
        executor.load
        for executor in case.infra.executors()
        if executor.executor_id != "unused_ocr"
    ]
    high_load = bool(relevant_loads) and max(relevant_loads) >= 0.8
    return low_bandwidth and high_load


def build_split_manifest(cases: list[BenchmarkCase], *, seed: int = 42) -> dict[str, Any]:
    by_task = _cases_by_task(cases)
    task_ids = sorted(by_task)

    random_tasks = _partition_tasks(task_ids, seed)

    composition_test = sorted(
        task_id
        for task_id, task_cases in by_task.items()
        if any(_is_low_bandwidth_high_load(case) for case in task_cases)
    )
    composition_remaining = sorted(set(task_ids) - set(composition_test))
    composition_base = _partition_tasks(
        composition_remaining, seed + 1, train_ratio=0.75, dev_ratio=0.25
    )
    composition_tasks = {
        "train": composition_base["train"],
        "dev": [*composition_base["dev"], *composition_base["test"]],
        "test": composition_test,
    }

    coding_tasks = sorted(
        task_id
        for task_id, task_cases in by_task.items()
        if task_cases[0].task.input_type == "repository"
    )
    video_tasks = sorted(set(task_ids) - set(coding_tasks))
    coding_partition = _partition_tasks(
        coding_tasks, seed + 2, train_ratio=0.8, dev_ratio=0.2
    )
    cross_domain_tasks = {
        "train": coding_partition["train"],
        "dev": [*coding_partition["dev"], *coding_partition["test"]],
        "test": video_tasks,
    }

    manifest = {
        "version": "1",
        "seed": seed,
        "splits": {
            "random_task": _expand_cases(by_task, random_tasks),
            "unseen_infra_composition": _expand_cases(by_task, composition_tasks),
            "cross_domain": _expand_cases(by_task, cross_domain_tasks),
        },
        "unseen_infra_composition": {
            "held_out_signature": {
                "bandwidth_mbps_lte": 50.0,
                "relevant_executor_load_gte": 0.8,
            }
        },
    }
    validate_split_manifest(manifest)
    return manifest


def validate_split_manifest(manifest: dict[str, Any]) -> None:
    for split_name, splits in manifest["splits"].items():
        seen: set[str] = set()
        for partition in ("train", "dev", "test"):
            task_ids = set(splits[partition]["task_ids"])
            overlap = seen & task_ids
            if overlap:
                raise ValueError(
                    f"{split_name}: task leakage into {partition}: {sorted(overlap)}"
                )
            seen.update(task_ids)
