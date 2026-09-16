import hashlib
import json
import re

from ..schemas import TaskRecord, WorkflowEdge, WorkflowNode, WorkflowRecord
from .parser import TrajectoryRecord


OPERATOR_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?:ripgrep|\brg\b|grep|code.?search|search_code)", re.I), "search_code"),
    (re.compile(r"(?:read.?file|open.?file|\bcat\b)", re.I), "read_file"),
    (re.compile(r"(?:static.?analysis|lint|type.?check)", re.I), "static_analysis"),
    (re.compile(r"(?:pytest|targeted.?test|test.?single|test.?file)", re.I), "run_targeted_test"),
    (re.compile(r"(?:full.?test|test.?suite)", re.I), "run_full_test"),
    (re.compile(r"(?:apply.?patch|edit.?code|write.?file)", re.I), "edit_code"),
    (re.compile(r"(?:inspect.?repo|repo.?tree|list.?files)", re.I), "inspect_repo"),
    (re.compile(r"(?:sample.?frames|extract.?frames)", re.I), "sample_frames"),
    (re.compile(r"(?:extract.?clip|video.?clip)", re.I), "extract_clip"),
    (re.compile(r"(?:detect|object.?detection)", re.I), "detect"),
    (re.compile(r"(?:\bocr\b|text.?recognition)", re.I), "ocr"),
    (re.compile(r"(?:strong.?vlm|video.?model)", re.I), "strong_vlm"),
    (re.compile(r"(?:small.?vlm|local.?vlm)", re.I), "small_vlm"),
    (re.compile(r"(?:compress.?evidence|summari[sz]e.?evidence)", re.I), "compress_evidence"),
    (re.compile(r"(?:verify|validation|check.?answer)", re.I), "verify"),
    (re.compile(r"(?:reason|think|analy[sz]e)", re.I), "reason"),
)


def canonical_operator(action_name: str) -> str | None:
    normalized = action_name.replace("_", " ").replace("-", " ")
    for pattern, operator in OPERATOR_PATTERNS:
        if pattern.search(normalized):
            return operator
    return None


def canonicalize_trajectory(
    trajectory: TrajectoryRecord, task: TaskRecord
) -> WorkflowRecord | None:
    if not trajectory.success:
        return None
    initial_artifact = "repository" if task.input_type == "repository" else "raw_video"
    nodes: list[WorkflowNode] = []
    edges: list[WorkflowEdge] = []
    previous_output = initial_artifact
    previous_node: str | None = None
    operator_counts: dict[str, int] = {}

    for action in trajectory.actions:
        operator = canonical_operator(action.name)
        if operator is None:
            continue
        operator_counts[operator] = operator_counts.get(operator, 0) + 1
        node_id = f"{operator}_{operator_counts[operator]}"
        inputs = action.input_artifacts or [previous_output]
        outputs = action.output_artifacts or [f"artifact_{len(nodes) + 1}"]
        nodes.append(
            WorkflowNode(
                node_id=node_id,
                operator=operator,
                input_artifacts=inputs,
                output_artifacts=outputs,
                metadata={"source_action": action.name},
            )
        )
        if previous_node is not None:
            edges.append(
                WorkflowEdge(src=previous_node, dst=node_id, artifact=previous_output)
            )
        previous_node = node_id
        previous_output = outputs[0]

    if not nodes:
        return None
    signature = json.dumps(
        {"operators": [node.operator for node in nodes]}, sort_keys=True
    )
    suffix = hashlib.sha256(signature.encode()).hexdigest()[:12]
    return WorkflowRecord(
        workflow_id=f"{task.task_id}:trace:{suffix}",
        task_id=task.task_id,
        source=task.source,
        nodes=nodes,
        edges=edges,
        success=True,
        provenance={
            "type": "real_successful_trajectory",
            **trajectory.provenance,
        },
    )


def workflow_signature(workflow: WorkflowRecord) -> str:
    nodes = {node.node_id: node for node in workflow.nodes}
    order = workflow.topological_order()
    position = {node_id: index for index, node_id in enumerate(order)}
    payload = {
        "operators": [nodes[node_id].operator for node_id in order],
        "edges": sorted((position[edge.src], position[edge.dst]) for edge in workflow.edges),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def deduplicate_workflows(workflows: list[WorkflowRecord]) -> list[WorkflowRecord]:
    unique: dict[str, WorkflowRecord] = {}
    for workflow in workflows:
        signature = workflow_signature(workflow)
        current = unique.get(signature)
        if current is None or workflow.workflow_id < current.workflow_id:
            unique[signature] = workflow
    return sorted(unique.values(), key=lambda workflow: workflow.workflow_id)
