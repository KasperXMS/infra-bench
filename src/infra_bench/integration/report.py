"""Small paired MAS report writer."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .mas_case import ImportedMASResult


def write_mas_report(
    results: list[ImportedMASResult], output_directory: Path
) -> tuple[Path, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "group_id": item.execution.group_id,
            "case_id": item.execution.case_id,
            "visibility": item.execution.visibility,
            "task_score": item.task_quality.get("score"),
            "e2e_ms": item.execution.e2e_ms,
            "planner_latency_ms": item.execution.planner_latency_ms,
            "planner_tokens": item.execution.planner_tokens,
            "invocation_count": item.execution.invocation_count,
            "model_counts": json.dumps(item.execution.model_counts, sort_keys=True),
            "service_ms_sum": item.execution.service_ms_sum,
            "transfer_bytes": item.execution.transfer_bytes,
            "transfer_ms_sum": item.execution.transfer_ms_sum,
            "raw_transfer_bytes": item.execution.raw_transfer_bytes,
            "derived_transfer_bytes": item.execution.derived_transfer_bytes,
            "cross_worker_transfer_bytes": item.execution.cross_worker_transfer_bytes,
            "cross_worker_transfer_count": item.execution.cross_worker_transfer_count,
            "multimodal_invocation_count": item.execution.multimodal_invocation_count,
            "site_aligned_multimodal_invocations": (
                item.execution.site_aligned_multimodal_invocations
            ),
            "site_alignment_rate": item.execution.site_alignment_rate,
            "artifact_grouping": json.dumps(item.execution.artifact_grouping),
            "action_count": item.structural_metrics.get("action_count"),
            "workflow_depth": item.structural_metrics.get("workflow_depth"),
            "max_parallel_width": item.structural_metrics.get("max_parallel_width"),
            "artifact_group_sizes": json.dumps(
                item.structural_metrics.get("artifact_group_sizes")
            ),
            "model_composition": json.dumps(
                item.structural_metrics.get("model_composition"), sort_keys=True
            ),
            "refinement_action_count": item.structural_metrics.get(
                "refinement_action_count"
            ),
            "reference_regret": item.reference_regret,
        }
        for item in sorted(
            results,
            key=lambda value: (
                value.execution.group_id,
                value.execution.case_id,
                value.execution.visibility,
            ),
        )
    ]
    csv_path = output_directory / "mas_report.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    markdown_path = output_directory / "mas_report.md"
    headers = list(rows[0]) if rows else []
    lines = ["# MAS benchmark report", ""]
    if rows:
        lines.extend(
            [
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join("---" for _ in headers) + " |",
                *[
                    "| " + " | ".join(str(row[key]) for key in headers) + " |"
                    for row in rows
                ],
            ]
        )
    else:
        lines.append("No MAS results.")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, markdown_path
