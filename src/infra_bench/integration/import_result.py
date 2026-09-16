"""Import real MAS results and apply hidden task evaluators."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

from infra_bench.schemas import BenchmarkCase, TaskRecord
from infra_bench.task_evaluation import score_video_mme

from .mas_case import ImportedMASResult, MASExecutionResult
from .scoring import reference_regret, structural_metrics


def read_mas_result(run_path: Path) -> MASExecutionResult:
    """Load `mas_result.json` from a run directory or a direct JSON path."""
    path = run_path / "mas_result.json" if run_path.is_dir() else run_path
    return MASExecutionResult.model_validate_json(path.read_text(encoding="utf-8"))


def _evaluate_task(task: TaskRecord, final_answer: str) -> dict[str, Any]:
    if task.evaluator_type == "exact_image_id":
        expected = str(task.evaluator_config["image_id"])
        normalized_expected = expected.lower().removeprefix("img_").lstrip("0") or "0"
        explicit = re.search(
            r"(?:answer|image\s*id)\s*[:#*` ]+[^\n]*?"
            r"(?:img(?:age)?[_ -]?|input-00)([0-9]+)",
            final_answer.lower(),
        )
        if explicit is None:
            explicit = re.fullmatch(
                r"\s*(?:img(?:age)?[_ -]?|input-00)([0-9]+)\s*",
                final_answer.lower(),
            )
        candidates: set[str] = set()
        if explicit is not None:
            candidates.add(explicit.group(1).lstrip("0") or "0")
        return {
            "evaluator_type": task.evaluator_type,
            "score": float(normalized_expected in candidates),
            "correct": normalized_expected in candidates,
        }
    if task.evaluator_type == "video_mme":
        metrics = score_video_mme([task], {task.task_id: final_answer})
        return {"evaluator_type": task.evaluator_type, **metrics}
    if task.evaluator_type == "swebench":
        return {
            "evaluator_type": task.evaluator_type,
            "status": "external_harness_required",
            "instance_id": task.evaluator_config.get("instance_id", task.task_id),
        }
    return {"evaluator_type": task.evaluator_type, "status": "unsupported"}


def import_mas_result(
    cases: list[BenchmarkCase],
    execution: MASExecutionResult,
) -> ImportedMASResult:
    """Associate a real run with its hidden source case and evaluate it."""
    matches = [case for case in cases if case.case_id == execution.case_id]
    if len(matches) != 1:
        raise ValueError(
            f"expected one benchmark case for {execution.case_id!r}, found {len(matches)}"
        )
    case = matches[0]
    if case.group_id != execution.group_id or case.task_id != execution.task_id:
        raise ValueError("MAS result identifiers do not match the source benchmark case")

    regret: float | None = None
    reference_costs = case.metadata.get("calibrated_reference_costs_ms")
    if isinstance(reference_costs, dict) and reference_costs:
        numeric = [
            float(cast(float | int | str, value))
            for value in cast(dict[object, object], reference_costs).values()
        ]
        regret = reference_regret(execution.e2e_ms, min(numeric))
    return ImportedMASResult(
        execution=execution,
        task_quality=_evaluate_task(case.task, execution.final_answer),
        structural_metrics=structural_metrics(execution.realized_workflow),
        reference_regret=regret,
    )


def write_imported_result(path: Path, result: ImportedMASResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
