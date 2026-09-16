import ast
import re
from collections import defaultdict
from statistics import fmean
from typing import Any

from ..schemas import TaskRecord


def extract_choice(prediction: str) -> str | None:
    matches = re.findall(r"(?<![A-Z])([A-H])(?![A-Z])", prediction.upper())
    return matches[-1] if matches else None


def relevance_rating(scores: list[int]) -> float:
    if len(scores) != 4:
        raise ValueError("Video-MME-v2 relevance groups must contain four questions")
    correct = sum(bool(score) for score in scores)
    return {0: 0.0, 1: 100.0 / 16, 2: 100.0 * 4 / 16, 3: 100.0 * 9 / 16, 4: 100.0}[correct]


def logic_rating(scores: list[int], group_structure: str | list[Any]) -> float:
    if len(scores) != 4:
        raise ValueError("Video-MME-v2 logic groups must contain four questions")
    structure = ast.literal_eval(group_structure) if isinstance(group_structure, str) else group_structure
    last_correct_idx = -1
    for index, value in enumerate(scores):
        if value:
            last_correct_idx = index
        else:
            break

    if structure == [1, 2, 3, 4]:
        score_map = {0: 0.0, 1: 100.0 / 16, 2: 100.0 * 4 / 16, 3: 100.0 * 9 / 16, 4: 100.0}
    elif structure == [1, [2, 3], 4]:
        score_map = {0: 0.0, 1: 100.0 / 12, 2: 100.0 * 4 / 12, 3: 100.0 * 7 / 12, 4: 100.0}
        if last_correct_idx == 0 and scores[2]:
            last_correct_idx += 1
    elif structure == [[1, 2], 3, 4]:
        score_map = {0: 0.0, 1: 100.0 / 10, 2: 100.0 * 2 / 10, 3: 100.0 * 5 / 10, 4: 100.0}
        if last_correct_idx == -1 and scores[1]:
            last_correct_idx += 1
    else:
        raise ValueError(f"unknown Video-MME-v2 group structure: {structure}")
    return score_map.get(last_correct_idx + 1, 0.0)


def _question_index(task: TaskRecord) -> int:
    question_id = str(task.metadata["question_id"])
    return int(question_id.rsplit("-", 1)[-1])


def score_video_mme(
    tasks: list[TaskRecord], predictions: dict[str, str]
) -> dict[str, Any]:
    groups: dict[str, list[TaskRecord]] = defaultdict(list)
    for task in tasks:
        if task.evaluator_type != "video_mme_v2_grouped":
            raise ValueError(f"task {task.task_id} is not a Video-MME-v2 task")
        groups[str(task.metadata["video_id"])].append(task)

    group_ratings: list[float] = []
    group_type_ratings: dict[str, list[float]] = defaultdict(list)
    correct_scores: list[int] = []
    for video_id, group in sorted(groups.items()):
        ordered = sorted(group, key=_question_index)
        if len(ordered) != 4:
            raise ValueError(f"video {video_id} has {len(ordered)} questions; expected 4")
        scores: list[int] = []
        for task in ordered:
            prediction = predictions.get(task.task_id, "")
            extracted = extract_choice(prediction)
            answer = str(task.evaluator_config["answer"]).upper()
            scores.append(int(extracted == answer))
        correct_scores.extend(scores)
        group_type = str(ordered[-1].evaluator_config["group_type"])
        structure = ordered[-1].evaluator_config["group_structure"]
        if group_type == "relevance":
            rating = relevance_rating(scores)
        elif group_type == "logic":
            rating = logic_rating(scores, structure)
        else:
            raise ValueError(f"unknown Video-MME-v2 group type: {group_type}")
        group_ratings.append(rating)
        group_type_ratings[group_type].append(rating)

    return {
        "question_accuracy": 100.0 * fmean(correct_scores) if correct_scores else 0.0,
        "grouped_rating": fmean(group_ratings) if group_ratings else 0.0,
        "group_type_rating": {
            key: fmean(values) for key, values in sorted(group_type_ratings.items())
        },
        "question_count": len(correct_scores),
        "group_count": len(group_ratings),
    }
