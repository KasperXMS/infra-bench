from collections import defaultdict
from statistics import fmean
from typing import Any

from ..schemas import EvaluationResult


def _aggregate(results: list[EvaluationResult]) -> dict[str, float | int]:
    if not results:
        return {"count": 0, "accuracy": 0.0, "mean_regret": 0.0}
    return {
        "count": len(results),
        "accuracy": sum(result.correct for result in results) / len(results),
        "mean_regret": fmean(result.regret for result in results),
    }


def summarize_results(results: list[EvaluationResult]) -> dict[str, Any]:
    by_category: dict[str, list[EvaluationResult]] = defaultdict(list)
    by_group: dict[str, list[EvaluationResult]] = defaultdict(list)
    for result in results:
        by_category[result.case_type].append(result)
        by_group[result.group_id].append(result)

    group_consistency = (
        sum(all(result.correct for result in group) for group in by_group.values())
        / len(by_group)
        if by_group
        else 0.0
    )

    semantic_groups = [
        group for group in by_group.values() if group[0].case_type == "semantic_switch"
    ]
    placement_groups = [
        group for group in by_group.values() if group[0].case_type == "placement_only"
    ]
    invariance_groups = [
        group for group in by_group.values() if group[0].case_type == "invariance"
    ]

    def group_rate(groups: list[list[EvaluationResult]], predicate) -> float:
        return sum(bool(predicate(group)) for group in groups) / len(groups) if groups else 0.0

    return {
        "planner": results[0].planner if results else None,
        "overall": _aggregate(results),
        "by_category": {
            category: _aggregate(category_results)
            for category, category_results in sorted(by_category.items())
        },
        "pairwise_consistency": group_consistency,
        "counterfactual_adaptation_accuracy": group_rate(
            semantic_groups,
            lambda group: all(result.correct for result in group)
            and len({result.selected_workflow for result in group}) > 1,
        ),
        "placement_only_accuracy": group_rate(
            placement_groups, lambda group: all(result.correct for result in group)
        ),
        "invariance_accuracy": group_rate(
            invariance_groups,
            lambda group: len({result.selected_workflow for result in group}) == 1,
        ),
        "oracle_aligned_invariance_accuracy": group_rate(
            invariance_groups, lambda group: all(result.correct for result in group)
        ),
    }
