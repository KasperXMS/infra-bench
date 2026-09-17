from __future__ import annotations

import base64
import json
import re
import time
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..io import write_json
from ..schemas import TaskRecord
from ..task_evaluation import score_video_mme

WORKFLOW_DENSE = "full_video_dense_128"
WORKFLOW_LOCALIZED = "local_temporal_clip_64"


def _read_credentials(path: Path) -> tuple[str, str]:
    api_key = ""
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            name, value = line.split("=", 1)
            if name.strip() == "DASHSCOPE_API_KEY":
                api_key = value.strip().strip("\"'")
        elif line.startswith("https://"):
            base_url = line.rstrip("/")
    if not api_key:
        raise ValueError(f"DASHSCOPE_API_KEY not found in {path}")
    return api_key, base_url


def _video_metadata(path: Path) -> tuple[int, float]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open video: {path}")
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), float(
            capture.get(cv2.CAP_PROP_FPS)
        )
    finally:
        capture.release()


def _uniform_indices(frame_count: int, count: int) -> list[int]:
    import numpy as np

    return [int(value) for value in np.linspace(0, frame_count - 1, count)]


def _motion_window_indices(
    path: Path,
    frame_count: int,
    fps: float,
    count: int,
    tasks: list[TaskRecord],
) -> list[int]:
    import cv2
    import numpy as np

    stride = max(1, round(fps))
    candidates: list[tuple[float, int]] = []
    previous = None
    capture = cv2.VideoCapture(str(path))
    try:
        for frame_index in range(0, frame_count, stride):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
            if previous is not None:
                score = float(np.mean(cv2.absdiff(previous, gray)))
                candidates.append((score, frame_index))
            previous = gray
    finally:
        capture.release()

    joined_instruction = "\n".join(task.instruction for task in tasks)
    explicit = re.search(r"first\s+(\d+(?:\.\d+)?)\s+seconds?", joined_instruction, re.I)
    if explicit:
        end_frame = min(frame_count - 1, round(float(explicit.group(1)) * fps))
        return _uniform_indices(end_frame + 1, count)

    duration_s = frame_count / fps
    window_s = min(90.0, max(30.0, duration_s * 0.40))
    window_frames = max(stride, round(window_s * fps))
    ordered = sorted(candidates, key=lambda item: item[1])
    best_start = 0
    best_score = -1.0
    left = 0
    running = 0.0
    for right, (score, frame_index) in enumerate(ordered):
        running += score
        while ordered[left][1] < frame_index - window_frames:
            running -= ordered[left][0]
            left += 1
        start = max(0, frame_index - window_frames)
        if running > best_score:
            best_score = running
            best_start = start
    best_end = min(frame_count - 1, best_start + window_frames)
    local = _uniform_indices(best_end - best_start + 1, count)
    return [best_start + index for index in local]


def _encode_frames(path: Path, indices: list[int]) -> tuple[list[str], int]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    encoded: list[str] = []
    byte_count = 0
    try:
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"failed to decode frame {frame_index} from {path}")
            height, width = frame.shape[:2]
            scale = min(1.0, 640.0 / max(height, width))
            if scale < 1.0:
                frame = cv2.resize(
                    frame,
                    (round(width * scale), round(height * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            ok, payload = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                raise ValueError(f"failed to encode frame {frame_index} from {path}")
            raw = payload.tobytes()
            byte_count += len(raw)
            encoded.append("data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii"))
    finally:
        capture.release()
    return encoded, byte_count


def _prompt(tasks: list[TaskRecord]) -> str:
    questions = "\n\n".join(
        f"[{task.metadata['question_id']}]\n{task.instruction}" for task in tasks
    )
    ids = ", ".join(str(task.metadata["question_id"]) for task in tasks)
    return (
        "These are temporally ordered frames of one video. Answer every multiple-choice "
        "question from visual evidence. Return only one JSON object mapping each question ID "
        f"to one capital answer letter A-H. Required IDs: {ids}.\n\n{questions}"
    )


def _request_qwen(
    *,
    api_key: str,
    base_url: str,
    model: str,
    frames: list[str],
    prompt: str,
    thinking: bool,
) -> tuple[str, dict[str, Any], int]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "temperature": 0,
        "enable_thinking": thinking,
        "max_tokens": 4096 if thinking else 1024,
    }
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=encoded,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=900) as response:  # noqa: S310
        body = json.loads(response.read().decode("utf-8"))
    content = str(body["choices"][0]["message"]["content"])
    return content, dict(body.get("usage") or {}), len(encoded)


def _extract_predictions(tasks: list[TaskRecord], response: str) -> dict[str, str]:
    text = response.strip()
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        parsed = json.loads(fenced)
    except json.JSONDecodeError:
        parsed = None
    predictions: dict[str, str] = {}
    if isinstance(parsed, dict):
        for task in tasks:
            question_id = str(task.metadata["question_id"])
            value = str(parsed.get(question_id, ""))
            match = re.search(r"(?<![A-Z])([A-H])(?![A-Z])", value.upper())
            predictions[task.task_id] = match.group(1) if match else ""
        return predictions
    for task in tasks:
        question_id = re.escape(str(task.metadata["question_id"]))
        match = re.search(rf"{question_id}[^A-H]+([A-H])", text.upper())
        predictions[task.task_id] = match.group(1) if match else ""
    return predictions


def _run_one(
    *,
    tasks: list[TaskRecord],
    video_path: Path,
    workflow_id: str,
    api_key: str,
    base_url: str,
    model: str,
    thinking: bool,
) -> dict[str, Any]:
    frame_count, fps = _video_metadata(video_path)
    started = time.perf_counter()
    if workflow_id == WORKFLOW_DENSE:
        indices = _uniform_indices(frame_count, 128)
    elif workflow_id == WORKFLOW_LOCALIZED:
        indices = _motion_window_indices(video_path, frame_count, fps, 64, tasks)
    else:
        raise ValueError(f"unknown workflow: {workflow_id}")
    frames, visual_bytes = _encode_frames(video_path, indices)
    preprocess_s = time.perf_counter() - started

    request_started = time.perf_counter()
    response, usage, request_bytes = _request_qwen(
        api_key=api_key,
        base_url=base_url,
        model=model,
        frames=frames,
        prompt=_prompt(tasks),
        thinking=thinking,
    )
    model_service_s = time.perf_counter() - request_started
    predictions = _extract_predictions(tasks, response)
    metrics = score_video_mme(tasks, predictions)
    return {
        "benchmark": "Video-MME-v2",
        "video_id": str(tasks[0].metadata["video_id"]),
        "workflow_id": workflow_id,
        "model": model,
        "thinking": thinking,
        "frame_count": len(indices),
        "selected_frame_indices": indices,
        "selected_timestamps_s": [round(index / fps, 3) for index in indices],
        "source_artifact_bytes": video_path.stat().st_size,
        "visual_payload_bytes": visual_bytes,
        "request_bytes": request_bytes,
        "preprocess_service_ms": round(preprocess_s * 1000, 3),
        "model_service_ms": round(model_service_s * 1000, 3),
        "e2e_ms": round((time.perf_counter() - started) * 1000, 3),
        "usage": usage,
        "predictions": predictions,
        "model_response": response,
        "official_metrics": metrics,
        "quality_threshold": {"question_accuracy": 100.0, "grouped_rating": 100.0},
        "quality_threshold_met": bool(
            metrics["question_accuracy"] == 100.0 and metrics["grouped_rating"] == 100.0
        ),
    }


def run_video_mme_workflows(
    tasks: Iterable[TaskRecord],
    *,
    video_dir: Path,
    credential_file: Path,
    output_dir: Path,
    video_ids: Iterable[str],
    model: str = "qwen3-vl-plus",
    thinking: bool = False,
) -> list[Path]:
    selected = set(video_ids)
    groups: dict[str, list[TaskRecord]] = {}
    for task in tasks:
        video_id = str(task.metadata.get("video_id", ""))
        if video_id in selected:
            groups.setdefault(video_id, []).append(task)
    missing = selected - groups.keys()
    if missing:
        raise ValueError(f"missing Video-MME task groups: {sorted(missing)}")

    api_key, base_url = _read_credentials(credential_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for video_id in sorted(selected):
        group = sorted(groups[video_id], key=lambda task: str(task.metadata["question_id"]))
        if len(group) != 4:
            raise ValueError(f"video {video_id} requires exactly four tasks")
        video_path = video_dir / f"{video_id}.mp4"
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        for workflow_id in (WORKFLOW_DENSE, WORKFLOW_LOCALIZED):
            output = output_dir / f"{video_id}__{workflow_id}.json"
            if output.exists():
                outputs.append(output)
                continue
            result = _run_one(
                tasks=group,
                video_path=video_path,
                workflow_id=workflow_id,
                api_key=api_key,
                base_url=base_url,
                model=model,
                thinking=thinking,
            )
            write_json(output, result)
            outputs.append(output)
    return outputs
