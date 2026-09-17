"""Ingest successful real-MAS traces into a semantic workflow bank.

The importer deliberately derives workflows from completed runtime actions rather
than Planner prose or run names.  It also keeps resource measurements separate
from the semantic fingerprint: repetitions and retries affect cost, but do not
manufacture a new workflow shape.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from statistics import median
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field

from infra_bench.config import load_yaml
from infra_bench.real_tasks.realizability import validate_workflow_realizability
from infra_bench.schemas.interaction import CODE_MAS_OPERATOR_BINDINGS
from infra_bench.schemas.task import TaskRecord
from infra_bench.schemas.workflow import WorkflowEdge, WorkflowNode, WorkflowRecord

_TOOL_OPERATOR_ALIASES = {
    "search_code": "search_code",
    "read_file": "read_file",
    "edit": "edit_file",
    "edit_file": "edit_file",
    "apply_patch": "apply_patch",
    "run_targeted_test": "run_targeted_test",
    "targeted_test": "run_targeted_test",
    "run_full_test": "run_full_test",
    "full_test": "run_full_test",
    "submit_patch": "submit_patch",
}
_MUTATING_OPERATORS = frozenset({"edit_file", "apply_patch"})
_TEST_OPERATORS = frozenset({"run_targeted_test", "run_full_test"})
_SEMANTIC_PHASE = {
    "invoke_model": "remote_reasoning",
    "search_code": "repository_inspection",
    "read_file": "repository_inspection",
    "edit_file": "code_mutation",
    "apply_patch": "code_mutation",
    "run_targeted_test": "targeted_verification",
    "run_full_test": "full_verification",
    "submit_patch": "patch_submission",
}


class ResourceSignature(BaseModel):
    """Measured resource demand for one run or the median of a cluster."""

    model_config = ConfigDict(extra="forbid")

    local_search_count: float = 0
    local_search_service_ms: float = 0
    local_read_count: float = 0
    local_read_service_ms: float = 0
    remote_reasoning_calls: float = 0
    remote_reasoning_service_ms: float = 0
    transmitted_context_bytes: float = 0
    cross_site_bytes: float = 0
    cross_site_transfer_ms: float = 0
    cross_site_round_trips: float = 0
    targeted_test_count: float = 0
    targeted_test_service_ms: float = 0
    full_test_count: float = 0
    full_test_service_ms: float = 0
    max_parallelism: float = 1
    parallel_action_count: float = 0
    verification_count: float = 0
    failed_action_count: float = 0
    retry_count: float = 0
    total_tool_service_ms: float = 0
    other_tool_service_ms: float = 0
    observed_e2e_ms: float = 0


class TraceWorkflowRun(BaseModel):
    """One official-resolved run reconstructed from its runtime trace."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    run_id: str
    run_dir: str
    world_id: str
    world_facts: dict[str, object] = Field(default_factory=dict)
    semantic_fingerprint: tuple[str, ...]
    workflow: WorkflowRecord
    resource_signature: ResourceSignature


class SemanticWorkflowCluster(BaseModel):
    """Runs sharing a semantic phase shape for the same benchmark task."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    workflow_id: str
    semantic_fingerprint: tuple[str, ...]
    run_ids: list[str]
    runs: list[TraceWorkflowRun]
    representative_workflow: WorkflowRecord
    per_run_signatures: list[ResourceSignature]
    resource_signature: ResourceSignature


def discover_trace_runs(
    trace_root: str | Path,
    tasks: Mapping[str, TaskRecord] | Iterable[TaskRecord],
    excluded_task_ids: Iterable[str] = (),
) -> list[TraceWorkflowRun]:
    """Discover and import eligible run directories below ``trace_root``.

    A directory is eligible only when all three run artifacts exist, the MAS run
    reports success and a submitted patch, and the official evaluator resolves
    the matching task.  Malformed or irrelevant directories are ignored so one
    incomplete experiment cannot poison a larger discovery pass.
    """

    task_by_id = _task_mapping(tasks)
    excluded = set(excluded_task_ids)
    imported: list[TraceWorkflowRun] = []
    for trace_path in sorted(Path(trace_root).rglob("trace.jsonl")):
        run_dir = trace_path.parent
        result_path = run_dir / "code_result.json"
        evaluation_path = run_dir / "official_evaluation.json"
        if not result_path.is_file() or not evaluation_path.is_file():
            continue
        try:
            result = _load_json(result_path)
            evaluation = _load_json(evaluation_path)
            task_id = str(result["task_id"])
            task = task_by_id.get(task_id)
            if task is None or task_id in excluded:
                continue
            if not result.get("success") or not result.get("submitted_patch"):
                continue
            if not _officially_resolved(evaluation, task_id):
                continue
            events = _load_events(trace_path)
            workflow, fingerprint = _reconstruct_workflow(task, result, events, run_dir)
            realizability = validate_workflow_realizability(
                workflow,
                task.interaction_spec,
                runtime_bindings=CODE_MAS_OPERATOR_BINDINGS,
            )
            if realizability.status != "realizable":
                continue
            imported.append(
                TraceWorkflowRun(
                    task_id=task_id,
                    run_id=str(result.get("run_id") or run_dir.name),
                    run_dir=str(run_dir.resolve()),
                    world_id=str(result.get("world_id") or "unknown"),
                    world_facts=_load_world_facts(run_dir),
                    semantic_fingerprint=fingerprint,
                    workflow=workflow,
                    resource_signature=_resource_signature(result, events),
                )
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return imported


def cluster_trace_workflows(
    runs: Iterable[TraceWorkflowRun],
) -> list[SemanticWorkflowCluster]:
    """Semantic-deduplicate runs and aggregate their measured resource demand."""

    groups: dict[tuple[str, tuple[str, ...]], list[TraceWorkflowRun]] = defaultdict(list)
    for run in runs:
        groups[(run.task_id, run.semantic_fingerprint)].append(run)

    clusters: list[SemanticWorkflowCluster] = []
    for (task_id, fingerprint), members in sorted(groups.items()):
        ordered = sorted(members, key=lambda item: item.run_id)
        signatures = [item.resource_signature for item in ordered]
        clusters.append(
            SemanticWorkflowCluster(
                task_id=task_id,
                workflow_id=_stable_workflow_id(task_id, fingerprint),
                semantic_fingerprint=fingerprint,
                run_ids=[item.run_id for item in ordered],
                runs=ordered,
                representative_workflow=ordered[0].workflow,
                per_run_signatures=signatures,
                resource_signature=_median_signature(signatures),
            )
        )
    return clusters


def _stable_workflow_id(task_id: str, fingerprint: tuple[str, ...]) -> str:
    semantic_key = "\x1f".join(fingerprint).encode()
    digest = sha256(semantic_key).hexdigest()[:12]
    return f"mas-semantic-{task_id}-{digest}"


def _load_world_facts(run_dir: Path) -> dict[str, object]:
    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        return {}
    config = load_yaml(config_path)
    world = config.get("world")
    if not isinstance(world, dict):
        return {}
    typed_world = cast(dict[str, Any], world)
    fact_names = (
        "repository_site",
        "bandwidth_mbps",
        "rtt_ms",
        "repository_size_bytes",
    )
    return {
        name: cast(object, typed_world[name])
        for name in fact_names
        if name in typed_world
    }


def _task_mapping(
    tasks: Mapping[str, TaskRecord] | Iterable[TaskRecord],
) -> dict[str, TaskRecord]:
    if isinstance(tasks, Mapping):
        mapping = cast(Mapping[str, TaskRecord], tasks)
        return {task_id: task for task_id, task in mapping.items()}
    return {task.task_id: task for task in tasks}


def _load_json(path: Path) -> dict[str, Any]:
    value = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return cast(dict[str, Any], value)


def _load_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = cast(object, json.loads(line))
        if isinstance(value, dict):
            events.append(cast(dict[str, Any], value))
    return events


def _officially_resolved(evaluation: dict[str, Any], task_id: str) -> bool:
    resolved_ids = evaluation.get("resolved_ids")
    if isinstance(resolved_ids, list):
        return task_id in {str(item) for item in cast(list[object], resolved_ids)}
    resolved = evaluation.get("resolved")
    if isinstance(resolved, bool):
        return resolved
    return evaluation.get("resolved_instances") == 1 and evaluation.get(
        "total_instances"
    ) == 1


def _event_operator(event: dict[str, Any]) -> str | None:
    event_type = event.get("event_type")
    if event_type not in {"planner.llm.end", "code_tool.end"}:
        return None
    semantic = event.get("semantic_operator")
    if isinstance(semantic, str) and semantic in CODE_MAS_OPERATOR_BINDINGS:
        return semantic
    if event_type == "planner.llm.end":
        return "invoke_model"
    if event_type == "code_tool.end":
        tool = event.get("tool")
        if isinstance(tool, str):
            return _TOOL_OPERATOR_ALIASES.get(tool)
    return None


def _semantic_fingerprint(operators: Iterable[str]) -> tuple[str, ...]:
    """Keep first-occurrence semantic phases, not low-level tool spelling."""

    seen: set[str] = set()
    phases: list[str] = []
    for operator in operators:
        phase = _SEMANTIC_PHASE[operator]
        if phase not in seen:
            seen.add(phase)
            phases.append(phase)
    return tuple(phases)


def _reconstruct_workflow(
    task: TaskRecord,
    result: dict[str, Any],
    events: list[dict[str, Any]],
    run_dir: Path,
) -> tuple[WorkflowRecord, tuple[str, ...]]:
    initial_ids = [item.artifact_id for item in task.interaction_spec.initial_artifacts]
    repository = initial_ids[0]
    pending_observations: list[str] = []
    last_reasoning: str | None = None
    nodes: list[WorkflowNode] = []
    edges: list[WorkflowEdge] = []
    artifact_producer: dict[str, str] = {}
    operators: list[str] = []
    repository_version = 0
    starts = {
        str(event["action_id"]): event
        for event in events
        if event.get("event_type") in {"planner.llm.start", "code_tool.start"}
        and isinstance(event.get("action_id"), str)
    }

    for event in events:
        operator = _event_operator(event)
        if operator is None:
            continue
        operators.append(operator)
        node_id = f"n{len(nodes) + 1:04d}"
        start = starts.get(str(event.get("action_id")), {})
        explicit_inputs = event.get("input_artifacts", start.get("input_artifacts"))
        explicit_outputs = event.get("output_artifacts")
        if isinstance(explicit_inputs, list) or isinstance(explicit_outputs, list):
            inputs = _normalize_runtime_inputs(task, explicit_inputs)
            outputs = (
                [str(item) for item in cast(list[object], explicit_outputs)]
                if isinstance(explicit_outputs, list)
                else []
            )
            if operator == "invoke_model" and outputs:
                last_reasoning = outputs[-1]
            repository_outputs = [
                artifact
                for artifact in outputs
                if "@v" in artifact and artifact.startswith("repo://")
            ]
            if repository_outputs:
                repository = repository_outputs[-1]
        elif operator == "invoke_model":
            inputs = [repository, *pending_observations]
            output = f"reasoning_{len(nodes) + 1:04d}"
            outputs = [output]
            pending_observations = []
            last_reasoning = output
        else:
            inputs = [repository]
            if last_reasoning is not None:
                inputs.append(last_reasoning)
            observation = f"observation_{len(nodes) + 1:04d}"
            outputs = [observation]
            pending_observations.append(observation)
            if operator in _MUTATING_OPERATORS and bool(event.get("success", True)):
                repository_version += 1
                repository = f"repository_v{repository_version}"
                outputs.append(repository)
            elif operator == "submit_patch":
                outputs.append(f"submitted_patch_{len(nodes) + 1:04d}")

        metadata = {
            "runtime_action_id": event.get("action_id"),
            "success": bool(event.get("success", True)),
            "site": event.get("site"),
        }
        nodes.append(
            WorkflowNode(
                node_id=node_id,
                operator=operator,
                input_artifacts=inputs,
                output_artifacts=outputs,
                metadata=metadata,
            )
        )
        for artifact in inputs:
            producer = artifact_producer.get(artifact)
            if producer is not None:
                edges.append(WorkflowEdge(src=producer, dst=node_id, artifact=artifact))
        for artifact in outputs:
            artifact_producer[artifact] = node_id

    fingerprint = _semantic_fingerprint(operators)
    workflow = WorkflowRecord(
        workflow_id=f"mas-trace-{task.task_id}-{result.get('run_id') or run_dir.name}",
        task_id=task.task_id,
        source="real_mas_trace",
        nodes=nodes,
        edges=edges,
        success=True,
        provenance={
            "run_dir": str(run_dir.resolve()),
            "official_evaluator": task.interaction_spec.external_evaluator.evaluator_id,
            "official_resolved": True,
            "semantic_fingerprint": list(fingerprint),
        },
    )
    return workflow, fingerprint


def _normalize_runtime_inputs(task: TaskRecord, value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    initial = task.interaction_spec.initial_artifacts
    runtime_aliases = {
        f"{artifact.source_ref}@v0": artifact.artifact_id for artifact in initial
    }
    return [
        runtime_aliases.get(str(item), str(item))
        for item in cast(list[object], value)
    ]


def _resource_signature(
    result: dict[str, Any], events: list[dict[str, Any]]
) -> ResourceSignature:
    completed = [
        (event, operator)
        for event in events
        if (operator := _event_operator(event)) is not None
    ]
    tools = [(event, operator) for event, operator in completed if operator != "invoke_model"]
    reasoning = [event for event, operator in completed if operator == "invoke_model"]

    def tool_events(operator: str) -> list[dict[str, Any]]:
        return [event for event, item in tools if item == operator]

    searches = tool_events("search_code")
    reads = tool_events("read_file")
    targeted = tool_events("run_targeted_test")
    full = tool_events("run_full_test")
    tool_service = sum(_number(event.get("service_ms")) for event, _ in tools)
    categorized = sum(
        _number(event.get("service_ms"))
        for event, operator in tools
        if operator in {"search_code", "read_file", *_TEST_OPERATORS}
    )
    intervals = _action_intervals(events)
    max_parallelism, parallel_count = _parallelism(intervals)
    event_context_bytes = sum(_number(event.get("planner_context_bytes")) for event, _ in tools)
    event_cross_bytes = sum(
        _number(event.get("cross_site_transfer_bytes")) for event, _ in tools
    )
    event_cross_ms = sum(
        _number(event.get("cross_site_transfer_ms")) for event, _ in tools
    )
    return ResourceSignature(
        local_search_count=len(searches),
        local_search_service_ms=sum(_number(item.get("service_ms")) for item in searches),
        local_read_count=len(reads),
        local_read_service_ms=sum(_number(item.get("service_ms")) for item in reads),
        remote_reasoning_calls=len(reasoning),
        remote_reasoning_service_ms=sum(_number(item.get("latency_ms")) for item in reasoning),
        transmitted_context_bytes=_number(
            result.get("transmitted_code_context_bytes"), event_context_bytes
        ),
        cross_site_bytes=_number(result.get("cross_site_transfer_bytes"), event_cross_bytes),
        cross_site_transfer_ms=_number(result.get("cross_site_transfer_ms"), event_cross_ms),
        cross_site_round_trips=sum(
            1 for event, _ in tools if bool(event.get("cross_site"))
        ),
        targeted_test_count=len(targeted),
        targeted_test_service_ms=sum(_number(item.get("service_ms")) for item in targeted),
        full_test_count=len(full),
        full_test_service_ms=sum(_number(item.get("service_ms")) for item in full),
        max_parallelism=max_parallelism,
        parallel_action_count=parallel_count,
        verification_count=len(targeted) + len(full),
        failed_action_count=sum(
            1 for event, _ in completed if not bool(event.get("success", True))
        ),
        retry_count=_number(result.get("retry_replanning_count")),
        total_tool_service_ms=tool_service,
        other_tool_service_ms=max(0.0, tool_service - categorized),
        observed_e2e_ms=_number(result.get("e2e_ms")),
    )


def _number(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return default


def _parse_timestamp(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _action_intervals(events: list[dict[str, Any]]) -> list[tuple[float, float]]:
    starts: dict[str, float] = {}
    intervals: list[tuple[float, float]] = []
    for event in events:
        event_type = event.get("event_type")
        action_id = event.get("action_id")
        timestamp = _parse_timestamp(event.get("timestamp"))
        if not isinstance(action_id, str) or timestamp is None:
            continue
        if event_type in {"planner.llm.start", "code_tool.start"}:
            starts[action_id] = timestamp
        elif event_type in {"planner.llm.end", "code_tool.end"}:
            start = starts.get(action_id)
            if start is not None and timestamp >= start:
                intervals.append((start, timestamp))
    return intervals


def _parallelism(intervals: list[tuple[float, float]]) -> tuple[int, int]:
    if not intervals:
        return 1, 0
    endpoints: list[tuple[float, int]] = []
    overlapping: set[int] = set()
    for index, (start, end) in enumerate(intervals):
        endpoints.extend(((start, 1), (end, -1)))
        for other_index, (other_start, other_end) in enumerate(intervals[:index]):
            if start < other_end and other_start < end:
                overlapping.update((index, other_index))
    active = 0
    maximum = 0
    for _, delta in sorted(endpoints, key=lambda item: (item[0], item[1])):
        active += delta
        maximum = max(maximum, active)
    return max(1, maximum), len(overlapping)


def _median_signature(signatures: list[ResourceSignature]) -> ResourceSignature:
    if not signatures:
        return ResourceSignature()
    values: dict[str, float] = {}
    for field_name in ResourceSignature.model_fields:
        values[field_name] = float(
            median(getattr(signature, field_name) for signature in signatures)
        )
    return ResourceSignature(**values)
