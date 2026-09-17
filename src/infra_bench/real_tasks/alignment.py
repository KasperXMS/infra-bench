"""Offline audit aligning real workflow evidence with the deployed MAS action space."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from ..io import write_json
from ..schemas import (
    CODE_MAS_OPERATOR_BINDINGS,
    ExternalEvaluatorSpec,
    InitialArtifactSpec,
    ObservationSpec,
    RuntimeVerifierSpec,
    TaskInteractionSpec,
    WorkflowEdge,
    WorkflowNode,
    WorkflowRecord,
)
from .realizability import validate_workflow_realizability

TASK_ID = "astropy__astropy-14309"
BASE_COMMIT = "cdb66059a2feb44ee49021874605ba90801f9986"
REPOSITORY_REF = f"repo://astropy/astropy@{BASE_COMMIT}"


def astropy_interaction_spec(objective: str) -> TaskInteractionSpec:
    return TaskInteractionSpec(
        task_id=TASK_ID,
        objective=objective,
        initial_artifacts=[
            InitialArtifactSpec(
                artifact_id="repository",
                kind="git_repository",
                source_ref=REPOSITORY_REF,
            )
        ],
        operators=[
            "search_code",
            "read_file",
            "edit_file",
            "apply_patch",
            "run_targeted_test",
            "run_full_test",
            "invoke_model",
            "submit_patch",
        ],
        observations=[
            ObservationSpec(
                observation_id="repository_observation",
                produced_by=["search_code", "read_file"],
                description="Bounded search matches and file contents.",
            ),
            ObservationSpec(
                observation_id="mutation_feedback",
                produced_by=["edit_file", "apply_patch", "submit_patch"],
                description="Mutation or submission result.",
            ),
            ObservationSpec(
                observation_id="test_feedback",
                produced_by=["run_targeted_test", "run_full_test"],
                description="Local test exit status and bounded output.",
            ),
            ObservationSpec(
                observation_id="model_output",
                produced_by=["invoke_model"],
                description="Planner model output.",
            ),
        ],
        runtime_verifier=RuntimeVerifierSpec(
            level="partial", signals=["targeted_test_result", "full_test_result"]
        ),
        external_evaluator=ExternalEvaluatorSpec(evaluator_id="swebench_official"),
    )


def _load_events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _resolved(run_directory: Path) -> bool:
    payload = json.loads(
        (run_directory / "official_evaluation.json").read_text(encoding="utf-8")
    )
    return TASK_ID in payload.get("resolved_ids", [])


def _workflow_from_mas_run(
    run_directory: Path, workflow_id: str, spec: TaskInteractionSpec
) -> tuple[WorkflowRecord, dict[str, float | int]]:
    events = _load_events(run_directory / "trace.jsonl")
    semantic_events = [
        event
        for event in events
        if (
            event.get("event_type") == "code_tool.end"
            or (
                event.get("event_type") == "planner.llm.end"
                and event.get("success") is True
            )
        )
    ]
    nodes: list[WorkflowNode] = []
    edges: list[WorkflowEdge] = []
    producer: dict[str, str] = {}
    repository = "repository"
    last_reasoning: str | None = None
    pending_observations: list[str] = []
    service_ms = 0.0
    planner_ms = 0.0
    transfer_bytes = 0
    tool_calls = 0
    for index, event in enumerate(semantic_events, start=1):
        event_type = str(event.get("event_type"))
        node_id = f"n{index:03d}"
        if event_type == "planner.llm.end":
            operator = "invoke_model"
            inputs = pending_observations or [repository]
            output = f"{node_id}/reasoning"
            pending_observations = []
            last_reasoning = output
            planner_ms += float(event.get("latency_ms", 0.0))
        else:
            operator = str(event.get("tool"))
            inputs = [repository]
            if last_reasoning is not None:
                inputs.append(last_reasoning)
            output = f"{node_id}/observation"
            pending_observations.append(output)
            service_ms += float(event.get("service_ms", 0.0))
            transfer_bytes += int(event.get("request_bytes", 0)) + int(
                event.get("response_bytes", 0)
            )
            tool_calls += 1
            if event.get("success") is True and operator in {"edit_file", "apply_patch"}:
                repository = f"{node_id}/repository"
        outputs = [output]
        if repository.startswith(f"{node_id}/"):
            outputs.append(repository)
        nodes.append(
            WorkflowNode(
                node_id=node_id,
                operator=operator,
                input_artifacts=inputs,
                output_artifacts=outputs,
                metadata={"runtime_event": event_type, "run_id": run_directory.name},
            )
        )
        for artifact in inputs:
            if artifact in producer:
                edges.append(
                    WorkflowEdge(src=producer[artifact], dst=node_id, artifact=artifact)
                )
        for artifact in outputs:
            producer[artifact] = node_id
    resolved = _resolved(run_directory)
    workflow = WorkflowRecord(
        workflow_id=workflow_id,
        task_id=TASK_ID,
        source="infra-aware-mas-trace",
        nodes=nodes,
        edges=edges,
        success=resolved,
        provenance={
            "type": "real_mas_execution",
            "run_id": run_directory.name,
            "verification": {
                "evaluator": "swebench_official",
                "passed": resolved,
                "quality_threshold_met": resolved,
                "runtime_verifier_level": spec.runtime_verifier.level,
            },
        },
    )
    return workflow, {
        "planner_ms": planner_ms,
        "tool_service_ms": service_ms,
        "transfer_payload_bytes": transfer_bytes,
        "tool_calls": tool_calls,
    }


def _legacy_workflows() -> list[WorkflowRecord]:
    definitions = {
        "whole_repo_remote_reasoning": ["inspect_repo", "invoke_model", "apply_patch"],
        "local_search_test_compact_remote_reasoning": [
            "search_code",
            "static_analysis",
            "invoke_model",
            "apply_patch",
            "run_targeted_test",
        ],
    }
    workflows: list[WorkflowRecord] = []
    for workflow_id, operators in definitions.items():
        nodes: list[WorkflowNode] = []
        previous = "repository"
        for index, operator in enumerate(operators, start=1):
            output = f"legacy-{index}"
            nodes.append(
                WorkflowNode(
                    node_id=f"n{index}",
                    operator=operator,
                    input_artifacts=[previous],
                    output_artifacts=[output],
                )
            )
            previous = output
        workflows.append(
            WorkflowRecord(
                workflow_id=workflow_id,
                task_id=TASK_ID,
                source="legacy-admission",
                nodes=nodes,
                success=True,
            )
        )
    return workflows


def _counterfactual(
    profiles: dict[str, dict[str, float | int]], *, bandwidth_mbps: float = 1.0, rtt_ms: float = 50.0
) -> list[dict[str, Any]]:
    worlds: list[dict[str, Any]] = []
    for world_id, locality in (("H1_edge", "edge"), ("H2_cloud", "cloud")):
        costs: dict[str, float] = {}
        for workflow_id, profile in profiles.items():
            transfer_ms = 0.0
            if locality == "edge":
                transfer_ms = (
                    float(profile["transfer_payload_bytes"]) * 8 / (bandwidth_mbps * 1_000_000) * 1000
                    + float(profile["tool_calls"]) * rtt_ms
                )
            costs[workflow_id] = round(
                float(profile["planner_ms"])
                + float(profile["tool_service_ms"])
                + transfer_ms,
                3,
            )
        winner = min(costs, key=costs.__getitem__)
        loser = max(costs, key=costs.__getitem__)
        worlds.append(
            {
                "world_id": world_id,
                "artifact_locality": locality,
                "bandwidth_mbps": bandwidth_mbps,
                "rtt_ms": rtt_ms,
                "cost_ms": costs,
                "winner": winner,
                "margin": round((costs[loser] - costs[winner]) / costs[winner], 6),
            }
        )
    return worlds


def build_astropy_alignment_audit(
    task_bank_path: Path,
    mas_runs_root: Path,
    output_directory: Path,
    *,
    admission_directory: Path | None = None,
) -> dict[str, Any]:
    raw_task = next(
        json.loads(line)
        for line in task_bank_path.read_text(encoding="utf-8").splitlines()
        if json.loads(line).get("task_id") == TASK_ID
    )
    spec = astropy_interaction_spec(str(raw_task["instruction"]))
    legacy = [
        {
            "workflow_id": workflow.workflow_id,
            "realizability": validate_workflow_realizability(
                workflow, spec, runtime_bindings=CODE_MAS_OPERATOR_BINDINGS
            ).model_dump(mode="json"),
        }
        for workflow in _legacy_workflows()
    ]
    sources = {
        "mas_targeted_validation": mas_runs_root / "formal-h1-static-r3",
        "mas_full_validation": mas_runs_root / "formal-h1-snapshot-r2",
    }
    workflows: list[WorkflowRecord] = []
    profiles: dict[str, dict[str, float | int]] = {}
    workflow_rows: list[dict[str, Any]] = []
    for workflow_id, run_directory in sources.items():
        workflow, profile = _workflow_from_mas_run(run_directory, workflow_id, spec)
        validation = validate_workflow_realizability(
            workflow, spec, runtime_bindings=CODE_MAS_OPERATOR_BINDINGS
        )
        workflows.append(workflow)
        profiles[workflow_id] = profile
        workflow_rows.append(
            {
                "workflow_id": workflow_id,
                "source_run": run_directory.name,
                "mas_realizable": validation.status == "realizable",
                "realizability": validation.model_dump(mode="json"),
                "benchmark_correct": workflow.success,
                "official_evaluator": "swebench_official",
                "semantic_operators": [node.operator for node in workflow.nodes],
                "structure": {
                    "model_invocations": sum(
                        node.operator == "invoke_model" for node in workflow.nodes
                    ),
                    "targeted_tests": sum(
                        node.operator == "run_targeted_test" for node in workflow.nodes
                    ),
                    "full_tests": sum(
                        node.operator == "run_full_test" for node in workflow.nodes
                    ),
                },
                "profile": profile,
            }
        )
    worlds = _counterfactual(profiles)
    reversal = len({world["winner"] for world in worlds}) > 1
    robust = reversal and all(float(world["margin"]) >= 0.2 for world in worlds)
    report = {
        "schema_version": "task-action-alignment-v1",
        "task_id": TASK_ID,
        "task_interaction_spec": spec.model_dump(mode="json"),
        "admission_rule": "MAS-realizable AND benchmark-correct AND infra-sensitive",
        "legacy_admission": {
            "status": "invalidated",
            "reason": "legacy reference workflows are not MAS-realizable",
            "workflows": legacy,
            "previous_crossover_must_not_be_used": True,
        },
        "verified_mas_workflows": workflow_rows,
        "calibration": {
            "method": "offline measured-action counterfactual; no Planner/API calls",
            "replications_per_workflow": 1,
            "worlds": worlds,
            "winner_reversal": reversal,
            "robust_margin": robust,
        },
        "semantic_switch_admitted": robust,
        "reason": (
            "admitted"
            if robust
            else "no robust preference reversal from the two verified MAS workflows"
        ),
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    write_json(output_directory / "astropy__astropy-14309.json", report)
    write_json(output_directory / "task_interaction_spec.json", spec.model_dump(mode="json"))
    (output_directory / "verified_mas_workflow_bank.jsonl").write_text(
        "".join(workflow.model_dump_json() + "\n" for workflow in workflows),
        encoding="utf-8",
    )
    (output_directory / "astropy__astropy-14309.md").write_text(
        _markdown(report), encoding="utf-8"
    )
    if admission_directory is not None:
        _invalidate_prior_admission(admission_directory, report, workflows, spec)
    return report


def _invalidate_prior_admission(
    admission_directory: Path,
    alignment: dict[str, Any],
    workflows: list[WorkflowRecord],
    spec: TaskInteractionSpec,
) -> None:
    report_path = admission_directory / "task_admission_report.json"
    report = cast(dict[str, Any], json.loads(report_path.read_text(encoding="utf-8")))
    for row in cast(list[dict[str, Any]], report["tasks"]):
        if row.get("task_id") != TASK_ID:
            continue
        for workflow in cast(list[dict[str, Any]], row["verified_workflows"]):
            workflow["mas_realizable"] = False
            workflow["realizability_status"] = "not_realizable"
        if isinstance(row.get("calibration"), dict):
            row["calibration"]["status"] = "invalidated"
            row["calibration"]["reason"] = "reference workflows are not MAS-realizable"
        row["semantic_switch_admitted"] = False
        row["reason"] = "legacy_reference_workflows_not_mas_realizable"
        row["action_space_alignment_audit"] = (
            "action_space_alignment/astropy__astropy-14309.json"
        )
    report["admitted_task_world_pairs"] = [
        row
        for row in cast(list[dict[str, Any]], report["admitted_task_world_pairs"])
        if row.get("task_id") != TASK_ID
    ]
    report["policy"]["admission_rule"] = (
        "MAS-realizable AND benchmark-correct AND infra-sensitive"
    )
    report["summary"]["admitted_count"] = len(report["admitted_task_world_pairs"])
    report["summary"]["target_reached"] = bool(report["admitted_task_world_pairs"])
    write_json(report_path, report)
    write_json(
        admission_directory / "admitted_pairs.json",
        {"pairs": report["admitted_task_world_pairs"]},
    )

    bank_path = admission_directory / "verified_workflow_bank.jsonl"
    retained = [
        json.loads(line)
        for line in bank_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("task_id") != TASK_ID
    ]
    retained.extend(
        {
            "task_id": TASK_ID,
            "benchmark": "SWE-bench Verified",
            "workflow_id": workflow.workflow_id,
            "mas_realizable": True,
            "verified": True,
            "quality": {"resolved": True},
            "provenance": workflow.provenance,
        }
        for workflow in workflows
    )
    bank_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in retained),
        encoding="utf-8",
    )

    export_directory = admission_directory / "mas_exports" / TASK_ID
    for path in export_directory.glob("world_*.json"):
        payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
        payload["task"]["interaction_spec"] = spec.model_dump(mode="json")
        payload["planner_contract"] = {
            "oracle_free": True,
            "dynamic_state_only": True,
            "admitted": False,
            "realizability_status": "not_realizable",
            "reason": "legacy reference workflows are not MAS-realizable",
        }
        write_json(path, payload)

    notice = (
        "# Real task admission report\n\n"
        "> **Action-space audit update:** `astropy__astropy-14309` is no longer "
        "admitted. Its legacy reference workflows are not MAS-realizable; the old "
        "49%/43% crossover is invalid for Planner evaluation. See "
        "`action_space_alignment/astropy__astropy-14309.md`.\n\n"
    )
    old_markdown = (admission_directory / "task_admission_report.md").read_text(
        encoding="utf-8"
    )
    old_markdown = old_markdown.replace(
        "Only workflows that pass the original benchmark evaluator are eligible. "
        "Unverified templates and `manual_fallback` records are rejected.",
        "Admission requires MAS-realizability, original benchmark correctness, and "
        "infrastructure sensitivity. Unverified templates, `manual_fallback` records, "
        "and workflows with unbound semantic operators are rejected.",
    )
    if "Action-space audit update" not in old_markdown:
        if old_markdown.startswith("# Real task admission report\n"):
            old_markdown = old_markdown.split("\n", 1)[1].lstrip()
        old_markdown = notice + old_markdown
    lines = []
    for line in old_markdown.splitlines():
        if line.startswith(f"| {TASK_ID} |"):
            columns = line.split("|")
            if len(columns) >= 8:
                columns[4] = " - "
                columns[-3] = " FAIL "
                columns[-2] = " legacy_reference_workflows_not_mas_realizable "
                line = "|".join(columns)
        lines.append(line)
    (admission_directory / "task_admission_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# astropy__astropy-14309 action-space alignment audit",
        "",
        "Admission now requires `MAS-realizable AND benchmark-correct AND infra-sensitive`.",
        "The prior 49%/43% crossover is invalid and must not be used to score the Planner.",
        "",
        "## Legacy references",
        "",
        "| Workflow | Status | Unsupported operators |",
        "| --- | --- | --- |",
    ]
    for row in cast(list[dict[str, Any]], report["legacy_admission"]["workflows"]):
        validation = row["realizability"]
        lines.append(
            f"| {row['workflow_id']} | {validation['status']} | "
            f"{', '.join(validation['unsupported_operators'])} |"
        )
    lines.extend(
        [
            "",
            "## Verified workflows from actual MAS traces",
            "",
            "| Workflow | Source run | MAS-realizable | Official resolved | Test strategy |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in cast(list[dict[str, Any]], report["verified_mas_workflows"]):
        lines.append(
            f"| {row['workflow_id']} | {row['source_run']} | "
            f"{row['mas_realizable']} | {row['benchmark_correct']} | "
            f"targeted={row['structure']['targeted_tests']}, "
            f"full={row['structure']['full_tests']} |"
        )
    lines.extend(
        [
            "",
            "## Offline counterfactual calibration",
            "",
            "| World | Winner | Margin | Costs (ms) |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for world in cast(list[dict[str, Any]], report["calibration"]["worlds"]):
        lines.append(
            f"| {world['world_id']} | {world['winner']} | {world['margin']:.1%} | "
            f"`{json.dumps(world['cost_ms'], sort_keys=True)}` |"
        )
    lines.extend(
        [
            "",
            f"Semantic-switch admitted: **{report['semantic_switch_admitted']}**.",
            "",
            "Local targeted/full tests are `partial` runtime verification. The hidden "
            "SWE-bench official evaluator is the terminal external evaluator.",
        ]
    )
    return "\n".join(lines) + "\n"
