import csv
from collections import defaultdict
from pathlib import Path

from ..schemas import EvaluationResult
from .metrics import summarize_results


CATEGORIES = ("semantic_switch", "placement_only", "invariance")


def write_report(
    results: list[EvaluationResult], output_dir: str | Path
) -> tuple[Path, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    by_planner: dict[str, list[EvaluationResult]] = defaultdict(list)
    for result in results:
        by_planner[result.planner].append(result)
    summaries = [
        summarize_results(by_planner[planner]) for planner in sorted(by_planner)
    ]

    csv_path = destination / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "planner",
                "overall_accuracy",
                *CATEGORIES,
                "mean_regret",
                "pairwise_consistency",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            overall = summary["overall"]
            by_category = summary["by_category"]
            writer.writerow(
                {
                    "planner": summary["planner"],
                    "overall_accuracy": overall["accuracy"],
                    "semantic_switch": summary["counterfactual_adaptation_accuracy"],
                    "placement_only": summary["placement_only_accuracy"],
                    "invariance": summary["invariance_accuracy"],
                    "mean_regret": overall["mean_regret"],
                    "pairwise_consistency": summary["pairwise_consistency"],
                }
            )

    markdown_path = destination / "summary.md"
    lines = [
        "| Planner | Overall Acc. | Semantic Switch | Placement-only | Invariance | Mean Regret | Pairwise Consistency |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        overall = summary["overall"]
        by_category = summary["by_category"]
        values = [
            str(summary["planner"] or "unknown"),
            f"{overall['accuracy']:.3f}",
            f"{summary['counterfactual_adaptation_accuracy']:.3f}",
            f"{summary['placement_only_accuracy']:.3f}",
            f"{summary['invariance_accuracy']:.3f}",
            f"{overall['mean_regret']:.3f}",
            f"{summary['pairwise_consistency']:.3f}",
        ]
        lines.append(f"| {' | '.join(values)} |")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, markdown_path
