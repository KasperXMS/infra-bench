from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..io import write_json
from ..schemas import TaskRecord
from .calibration import calibrate_swebench_pair, calibrate_video_pair
from .official_video_mme import score_with_official_script
from .swebench import WORKFLOW_COMPACT, WORKFLOW_REMOTE
from .video_mme import WORKFLOW_DENSE, WORKFLOW_LOCALIZED


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_admission_report(
    tasks: Iterable[TaskRecord],
    *,
    result_dir: Path,
    official_script: Path,
    output_dir: Path,
    swebench_result_dir: Path,
    swebench_eval_dir: Path,
    video_ids: Iterable[str],
    swebench_ids: Iterable[str],
    admission_margin: float = 0.20,
) -> dict[str, Any]:
    tasks = list(tasks)
    by_video: dict[str, list[TaskRecord]] = {}
    for task in tasks:
        video_id = str(task.metadata.get("video_id", ""))
        if video_id:
            by_video.setdefault(video_id, []).append(task)

    task_rows: list[dict[str, Any]] = []
    admitted: list[dict[str, Any]] = []
    verified_workflows: list[dict[str, Any]] = []
    official_hashes: set[str] = set()
    for video_id in video_ids:
        workflows: list[dict[str, Any]] = []
        results: dict[str, dict[str, Any]] = {}
        for workflow_id in (WORKFLOW_DENSE, WORKFLOW_LOCALIZED):
            path = result_dir / f"{video_id}__{workflow_id}.json"
            if not path.is_file():
                workflows.append(
                    {"workflow_id": workflow_id, "verified": False, "reason": "not_run"}
                )
                continue
            result = _load(path)
            official, source_hash = score_with_official_script(
                by_video[video_id], dict(result["predictions"]), official_script
            )
            official_hashes.add(source_hash)
            passed = bool(
                official["question_accuracy"] == 100.0
                and official["grouped_rating"] == 100.0
            )
            result["official_metrics"] = official
            result["official_evaluator_sha256"] = source_hash
            result["quality_threshold_met"] = passed
            write_json(path, result)
            results[workflow_id] = result
            workflow_row = {
                "workflow_id": workflow_id,
                "model": result["model"],
                "verified": passed,
                "quality": official,
                "e2e_ms": result["e2e_ms"],
                "preprocess_service_ms": result["preprocess_service_ms"],
                "model_service_ms": result["model_service_ms"],
                "transfer_payload_bytes": result["request_bytes"],
            }
            workflows.append(workflow_row)
            if passed:
                verified_workflows.append(
                    {
                        "task_id": f"video_mme_v2:{video_id}",
                        "benchmark": "Video-MME-v2",
                        **workflow_row,
                        "provenance": {
                            "type": "real_execution",
                            "evaluator": "video_mme_v2_official",
                            "official_evaluator_sha256": source_hash,
                        },
                    }
                )

        calibration = None
        passed = False
        reason = "fewer_than_two_workflows_meet_quality_threshold"
        if len(results) == 2 and all(
            bool(result["quality_threshold_met"]) for result in results.values()
        ):
            calibration = calibrate_video_pair(
                results[WORKFLOW_DENSE],
                results[WORKFLOW_LOCALIZED],
                admission_margin=admission_margin,
            )
            passed = bool(calibration["semantic_switch_pair_found"])
            reason = "admitted" if passed else "no_robust_preference_reversal"
            if passed:
                admitted.append(
                    {
                        "task_id": f"video_mme_v2:{video_id}",
                        "benchmark": "Video-MME-v2",
                        "workflows": [WORKFLOW_DENSE, WORKFLOW_LOCALIZED],
                        "worlds": calibration["selected_worlds"],
                    }
                )
        task_rows.append(
            {
                "task_id": f"video_mme_v2:{video_id}",
                "benchmark": "Video-MME-v2",
                "verified_workflows": workflows,
                "calibration": calibration,
                "semantic_switch_admitted": passed,
                "reason": reason,
            }
        )

    for task_id in swebench_ids:
        workflow_rows: list[dict[str, Any]] = []
        workflow_results: dict[str, dict[str, Any]] = {}
        for workflow_id in (WORKFLOW_REMOTE, WORKFLOW_COMPACT):
            result_path = swebench_result_dir / f"{task_id}__{workflow_id}.json"
            result = _load(result_path) if result_path.is_file() else None
            resolved = False
            evaluator_report = None
            for candidate in swebench_eval_dir.glob(f"*{workflow_id}*.json"):
                payload = _load(candidate)
                if task_id in payload.get("submitted_ids", []):
                    evaluator_report = candidate.name
                    resolved = task_id in payload.get("resolved_ids", [])
                    break
            row: dict[str, Any] = {
                "workflow_id": workflow_id,
                "verified": resolved,
                "quality": {"resolved": resolved} if evaluator_report else None,
                "official_evaluator_report": evaluator_report,
            }
            if result is not None:
                row["patch_apply_check"] = result.get("patch_apply_check", False)
                row["service_ms_sum"] = round(
                    sum(float(item.get("service_ms", 0.0)) for item in result["trace"]), 3
                )
            workflow_rows.append(row)
            if resolved and result is not None:
                workflow_results[workflow_id] = result
                verified_workflows.append(
                    {
                        "task_id": task_id,
                        "benchmark": "SWE-bench Verified",
                        **row,
                        "provenance": {
                            "type": "real_execution",
                            "evaluator": "swebench_official",
                            "official_evaluator_report": evaluator_report,
                        },
                    }
                )
        calibration = None
        passed = False
        generated = [
            _load(swebench_result_dir / f"{task_id}__{workflow_id}.json")
            for workflow_id in (WORKFLOW_REMOTE, WORKFLOW_COMPACT)
            if (swebench_result_dir / f"{task_id}__{workflow_id}.json").is_file()
        ]
        if generated and any(not item.get("patch_apply_check") for item in generated):
            reason = "workflow_patch_generation_failed"
        elif generated:
            reason = "official_evaluation_pending"
        else:
            reason = "workflow_not_run"
        if len(workflow_results) == 2:
            calibration = calibrate_swebench_pair(
                workflow_results[WORKFLOW_REMOTE],
                workflow_results[WORKFLOW_COMPACT],
                admission_margin=admission_margin,
            )
            passed = bool(calibration["semantic_switch_pair_found"])
            reason = "admitted" if passed else "no_robust_preference_reversal"
            if passed:
                admitted.append(
                    {
                        "task_id": task_id,
                        "benchmark": "SWE-bench Verified",
                        "workflows": [WORKFLOW_REMOTE, WORKFLOW_COMPACT],
                        "worlds": calibration["selected_worlds"],
                    }
                )
        task_rows.append(
            {
                "task_id": task_id,
                "benchmark": "SWE-bench Verified",
                "verified_workflows": workflow_rows,
                "calibration": calibration,
                "semantic_switch_admitted": passed,
                "reason": reason,
            }
        )

    report = {
        "version": 1,
        "policy": {
            "manual_fallback_is_valid": False,
            "minimum_verified_workflows": 2,
            "semantic_switch_margin": admission_margin,
            "export_only_admitted": True,
        },
        "official_video_mme_evaluator_sha256": sorted(official_hashes),
        "tasks": task_rows,
        "admitted_task_world_pairs": admitted,
        "summary": {
            "task_count": len(task_rows),
            "admitted_count": len(admitted),
            "target_reached": len(admitted) >= 1,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "task_admission_report.json", report)
    (output_dir / "verified_workflow_bank.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in verified_workflows),
        encoding="utf-8",
    )
    write_json(output_dir / "admitted_pairs.json", {"pairs": admitted})
    _write_mas_exports(task_rows, tasks, output_dir / "mas_exports")
    (output_dir / "task_admission_report.md").write_text(
        _markdown(report), encoding="utf-8"
    )
    return report


def _write_mas_exports(
    task_rows: list[dict[str, Any]], tasks: list[TaskRecord], output_dir: Path
) -> None:
    task_by_id = {task.task_id: task for task in tasks}
    for row in task_rows:
        if not row["semantic_switch_admitted"]:
            continue
        task = task_by_id.get(str(row["task_id"]))
        calibration = row.get("calibration")
        if task is None or not calibration or not calibration.get("selected_worlds"):
            continue
        task_dir = output_dir / task.task_id.replace(":", "_")
        task_dir.mkdir(parents=True, exist_ok=True)
        for stale in task_dir.glob("*.json"):
            stale.unlink()
        for index, world in enumerate(calibration["selected_worlds"], start=1):
            source_services = world.get("service_time_estimates_ms", {})
            payload = {
                "schema_version": 1,
                "task": {
                    "task_id": task.task_id,
                    "benchmark": row["benchmark"],
                    "instruction": task.instruction,
                    "artifact_refs": task.artifact_refs,
                },
                "infrastructure": {
                    "world_id": f"admitted_world_{index}",
                    "artifact_locality": world.get("artifact_locality"),
                    "network": {
                        "bandwidth_mbps": world.get("bandwidth_mbps"),
                        "rtt_ms": world.get("rtt_ms"),
                    },
                    "replicas": {
                        "repository_inspection": ["cloud"],
                        "code_search": ["edge"],
                        "reasoning_model": ["cloud"],
                    },
                    "service_time_estimates_ms": {
                        "repository_inspection": source_services.get(
                            "remote_repository_inspection"
                        ),
                        "code_search": source_services.get(
                            "local_search_static_filter"
                        ),
                        "reasoning_model": source_services.get(
                            "remote_reasoning_typical"
                        ),
                    },
                },
                "planner_contract": {
                    "oracle_free": True,
                    "dynamic_state_only": True,
                },
            }
            write_json(task_dir / f"world_{index}.json", payload)


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Real task admission report",
        "",
        "Only workflows that pass the original benchmark evaluator are eligible. "
        "Unverified templates and `manual_fallback` records are rejected.",
        "",
        "| Task | Benchmark | Verified workflows | Calibration pair | Admission | Reason |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for task in report["tasks"]:
        verified = sum(bool(item.get("verified")) for item in task["verified_workflows"])
        pair = "-"
        calibration = task.get("calibration")
        if calibration and calibration.get("selected_worlds"):
            pair = " <-> ".join(world["world_id"] for world in calibration["selected_worlds"])
        lines.append(
            f"| {task['task_id']} | {task['benchmark']} | {verified} | {pair} | "
            f"{'PASS' if task['semantic_switch_admitted'] else 'FAIL'} | {task['reason']} |"
        )
    lines.extend(["", "## Workflow quality and selected worlds", ""])
    for task in report["tasks"]:
        lines.append(f"### {task['task_id']}")
        lines.append("")
        for workflow in task["verified_workflows"]:
            quality = workflow.get("quality")
            detail = "not evaluator-verified"
            if quality:
                if "question_accuracy" in quality:
                    detail = (
                        f"accuracy={quality['question_accuracy']:.1f}, "
                        f"grouped_rating={quality['grouped_rating']:.1f}"
                    )
                else:
                    detail = f"resolved={str(quality.get('resolved', False)).lower()}"
            lines.append(f"- `{workflow['workflow_id']}`: {detail}")
        calibration = task.get("calibration")
        if calibration and calibration.get("selected_worlds"):
            for world in calibration["selected_worlds"]:
                lines.append(
                    f"- `{world['world_id']}`: winner `{world['winner']}`, "
                    f"margin {100 * world['margin']:.1f}%, cost_ms={world['cost_ms']}"
                )
        lines.append("")
    return "\n".join(lines)
