from __future__ import annotations

import json
import re
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

from ..io import write_json
from ..schemas import TaskRecord
from .video_mme import _read_credentials

WORKFLOW_REMOTE = "whole_repo_remote_reasoning"
WORKFLOW_COMPACT = "local_search_test_compact_remote_reasoning"


def _run(command: list[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(  # noqa: S603
        command, cwd=cwd, check=True, text=True, capture_output=True
    )
    return completed.stdout


def _qwen_text(
    *,
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    thinking: bool,
) -> tuple[str, dict[str, Any], float]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "enable_thinking": thinking,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=900) as response:  # noqa: S310
        body = json.loads(response.read().decode("utf-8"))
    elapsed = time.perf_counter() - started
    return str(body["choices"][0]["message"]["content"]), dict(body.get("usage") or {}), elapsed


def _extract_json(text: str) -> dict[str, Any]:
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        value = json.loads(fenced)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if match is None:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("model response is not a JSON object")
    return value


def _extract_patch(text: str) -> str:
    match = re.search(r"```(?:diff|patch)?\s*(.*?)```", text, re.S | re.I)
    patch = match.group(1) if match else text
    start = patch.find("diff --git ")
    if start < 0:
        raise ValueError("model response contains no unified diff")
    return patch[start:].strip() + "\n"


def _safe_file_context(repo: Path, paths: list[str], *, limit: int = 180_000) -> str:
    chunks: list[str] = []
    used = 0
    for relative in paths:
        candidate = (repo / relative).resolve()
        if not candidate.is_relative_to(repo.resolve()) or not candidate.is_file():
            continue
        try:
            content = candidate.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        chunk = f"\n===== {relative} =====\n{content}"
        remaining = limit - used
        if remaining <= 0:
            break
        chunks.append(chunk[:remaining])
        used += len(chunks[-1])
    return "".join(chunks)


def _remote_selected_context(
    task: TaskRecord, repo: Path, *, api_key: str, base_url: str, model: str
) -> tuple[str, list[str], list[dict[str, Any]]]:
    tree = _run(["git", "ls-files"], cwd=repo)
    prompt = (
        "You are triaging a SWE-bench issue. From the repository file tree, select at most 8 "
        "existing source or test files whose contents are needed to implement the fix. Return "
        "only JSON as {\"files\":[\"path\"]}. Do not solve the issue yet.\n\nISSUE:\n"
        f"{task.instruction}\n\nREPOSITORY TREE:\n{tree[:40_000]}"
    )
    response, usage, elapsed = _qwen_text(
        api_key=api_key,
        base_url=base_url,
        model=model,
        prompt=prompt,
        max_tokens=512,
        thinking=False,
    )
    selected = [str(path) for path in _extract_json(response).get("files", [])][:8]
    return _safe_file_context(repo, selected), selected, [
        {"phase": "remote_file_selection", "service_ms": elapsed * 1000, "usage": usage}
    ]


def _compact_context(task: TaskRecord, repo: Path) -> tuple[str, list[str], list[dict[str, Any]]]:
    started = time.perf_counter()
    files: list[str] = []
    for value in [task.instruction, *task.evaluator_config.get("fail_to_pass", [])]:
        for match in re.findall(r"(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.py", str(value)):
            if (repo / match).is_file() and match not in files:
                files.append(match)
    terms = [
        term
        for term in re.findall(r"[A-Za-z_][A-Za-z0-9_]{5,}", task.instruction)
        if term.lower() not in {"description", "expected", "behavior", "python", "version"}
    ]
    for term in terms[:12]:
        completed = subprocess.run(  # noqa: S603
            ["git", "grep", "-l", "-i", term, "--", "*.py"],
            cwd=repo,
            check=False,
            text=True,
            capture_output=True,
        )
        for path in completed.stdout.splitlines()[:3]:
            if path not in files:
                files.append(path)
        if len(files) >= 8:
            break
    selected = files[:8]
    elapsed = time.perf_counter() - started
    return _safe_file_context(repo, selected), selected, [
        {
            "phase": "local_search_static_filter",
            "service_ms": elapsed * 1000,
            "search_terms": terms[:12],
        }
    ]


def _prepare_repo(task: TaskRecord, repo_root: Path) -> Path:
    repo = repo_root / task.task_id
    if not (repo / ".git").is_dir():
        repo.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--filter=blob:none", "https://github.com/astropy/astropy.git", str(repo)])
    _run(["git", "reset", "--hard", str(task.metadata["base_commit"])], cwd=repo)
    _run(["git", "clean", "-fd"], cwd=repo)
    return repo


def _check_patch(repo: Path, patch_file: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        ["git", "apply", "--check", str(patch_file.resolve())],
        cwd=repo,
        check=False,
        text=True,
        capture_output=True,
    )


def _repair_patch(
    *,
    repo: Path,
    patch: str,
    error: str,
    api_key: str,
    base_url: str,
    model: str,
) -> tuple[str, dict[str, Any]]:
    paths = re.findall(r"^\+\+\+ b/(.+)$", patch, re.M)
    context = _safe_file_context(repo, paths, limit=120_000)
    prompt = (
        "The following proposed unified diff failed `git apply --check`. Return only a corrected "
        "unified git diff against the exact current files below. Preserve the intended fix; do not "
        "invent unrelated edits.\n\nAPPLY ERROR:\n"
        f"{error}\n\nFAILED DIFF:\n{patch}\n\nEXACT CURRENT FILES:\n{context}"
    )
    response, usage, elapsed = _qwen_text(
        api_key=api_key,
        base_url=base_url,
        model=model,
        prompt=prompt,
        max_tokens=2048,
        thinking=False,
    )
    return _extract_patch(response), {
        "phase": "patch_apply_repair",
        "service_ms": elapsed * 1000,
        "usage": usage,
    }


def run_swebench_workflows(
    tasks: list[TaskRecord],
    *,
    task_ids: list[str],
    repo_root: Path,
    credential_file: Path,
    output_dir: Path,
    model: str = "qwen3.8-max",
) -> list[Path]:
    selected = {task.task_id: task for task in tasks if task.task_id in task_ids}
    missing = set(task_ids) - selected.keys()
    if missing:
        raise ValueError(f"missing SWE-bench tasks: {sorted(missing)}")
    api_key, base_url = _read_credentials(credential_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for task_id in task_ids:
        task = selected[task_id]
        for workflow_id in (WORKFLOW_REMOTE, WORKFLOW_COMPACT):
            output = output_dir / f"{task_id}__{workflow_id}.json"
            if output.is_file():
                existing = json.loads(output.read_text(encoding="utf-8"))
                if existing.get("patch_apply_check"):
                    outputs.append(output)
                    continue
                repo = _prepare_repo(task, repo_root / workflow_id)
                patch_file = output.with_suffix(".patch")
                if not patch_file.is_file():
                    outputs.append(output)
                    continue
                repaired, repair_trace = _repair_patch(
                    repo=repo,
                    patch=patch_file.read_text(encoding="utf-8"),
                    error=str(existing.get("patch_apply_error", "")),
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                )
                patch_file.write_text(repaired, encoding="utf-8")
                applied = _check_patch(repo, patch_file)
                existing["trace"].append(repair_trace)
                existing["patch_apply_check"] = applied.returncode == 0
                existing["patch_apply_error"] = applied.stderr
                write_json(output, existing)
                outputs.append(output)
                continue
            repo = _prepare_repo(task, repo_root / workflow_id)
            if workflow_id == WORKFLOW_REMOTE:
                context, files, trace = _remote_selected_context(
                    task, repo, api_key=api_key, base_url=base_url, model=model
                )
            else:
                context, files, trace = _compact_context(task, repo)
            prompt = (
                "Solve this SWE-bench issue against the checked-out base commit. Return only a "
                "valid unified git diff. Make the minimal production-code change and add or adjust "
                "tests only when necessary. Do not claim tests were run.\n\nISSUE:\n"
                f"{task.instruction}\n\nSELECTED REPOSITORY CONTEXT:\n{context}"
            )
            response, usage, elapsed = _qwen_text(
                api_key=api_key,
                base_url=base_url,
                model=model,
                prompt=prompt,
                max_tokens=4096,
                thinking=False,
            )
            trace.append({"phase": "remote_patch_reasoning", "service_ms": elapsed * 1000, "usage": usage})
            try:
                patch = _extract_patch(response)
            except ValueError as exc:
                write_json(
                    output,
                    {
                        "benchmark": "SWE-bench Verified",
                        "task_id": task_id,
                        "workflow_id": workflow_id,
                        "model": model,
                        "selected_files": files,
                        "trace": trace,
                        "patch_file": None,
                        "patch_apply_check": False,
                        "patch_apply_error": str(exc),
                        "official_evaluation": None,
                    },
                )
                outputs.append(output)
                continue
            patch_file = output.with_suffix(".patch")
            patch_file.write_text(patch, encoding="utf-8")
            applied = _check_patch(repo, patch_file)
            if applied.returncode != 0:
                patch, repair_trace = _repair_patch(
                    repo=repo,
                    patch=patch,
                    error=applied.stderr,
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                )
                trace.append(repair_trace)
                patch_file.write_text(patch, encoding="utf-8")
                applied = _check_patch(repo, patch_file)
            result = {
                "benchmark": "SWE-bench Verified",
                "task_id": task_id,
                "workflow_id": workflow_id,
                "model": model,
                "selected_files": files,
                "trace": trace,
                "patch_file": str(patch_file),
                "patch_apply_check": applied.returncode == 0,
                "patch_apply_error": applied.stderr,
                "official_evaluation": None,
            }
            write_json(output, result)
            outputs.append(output)
    return outputs


def build_swebench_predictions(
    result_dir: Path, *, workflow_id: str, output: Path
) -> list[dict[str, str]]:
    predictions: list[dict[str, str]] = []
    for result_path in sorted(result_dir.glob(f"*__{workflow_id}.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if not result.get("patch_apply_check"):
            continue
        patch_path = Path(str(result["patch_file"]))
        if not patch_path.is_absolute():
            patch_path = result_path.parent / patch_path.name
        predictions.append(
            {
                "instance_id": str(result["task_id"]),
                "model_name_or_path": f"qwen3.8-max/{workflow_id}",
                "model_patch": patch_path.read_text(encoding="utf-8"),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in predictions),
        encoding="utf-8",
    )
    return predictions
