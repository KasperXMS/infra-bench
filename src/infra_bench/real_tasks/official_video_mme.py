from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from ..schemas import TaskRecord

_OFFICIAL_FUNCTIONS = {
    "extract_characters_regex_v2",
    "cal_relevance_rating",
    "cal_logic_rating",
    "get_final_rating",
    "get_final_acc",
    "get_final_metrics",
}


def score_with_official_script(
    tasks: list[TaskRecord], predictions: dict[str, str], script: Path
) -> tuple[dict[str, Any], str]:
    """Execute the scoring functions from the pinned upstream Video-MME-v2 script."""
    import pandas as pd

    source = script.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(script))
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in _OFFICIAL_FUNCTIONS
    ]
    found = {node.name for node in selected}
    if found != _OFFICIAL_FUNCTIONS:
        raise ValueError(f"official evaluator functions missing: {sorted(_OFFICIAL_FUNCTIONS - found)}")
    namespace: dict[str, Any] = {"ast": ast, "json": json, "re": re}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(script), "exec"), namespace)  # noqa: S102

    rows: list[dict[str, Any]] = []
    extractor = namespace["extract_characters_regex_v2"]
    for task in sorted(tasks, key=lambda item: str(item.metadata["question_id"])):
        prediction = predictions.get(task.task_id, "")
        extracted = extractor(prediction)
        answer = str(task.evaluator_config["answer"])
        rows.append(
            {
                "group_type": task.evaluator_config["group_type"],
                "group_structure": task.evaluator_config["group_structure"],
                "level": task.metadata["level"],
                "second_head": task.metadata["second_head"],
                "third_head": task.metadata["third_head"],
                "score": int(extracted == answer) if extracted else -1,
            }
        )
    metrics = namespace["get_final_metrics"](pd.DataFrame(rows))
    metrics["question_accuracy"] = metrics["acc"]["overall_acc"]
    metrics["grouped_rating"] = metrics["rating"]["overall_rating"]["total"]
    return metrics, hashlib.sha256(source.encode("utf-8")).hexdigest()
