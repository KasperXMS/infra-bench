from collections import deque
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class WorkflowNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str
    operator: str
    input_artifacts: list[str] = Field(default_factory=list)
    output_artifacts: list[str] = Field(default_factory=list)
    required_capabilities: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    src: str
    dst: str
    artifact: str | None = None


class WorkflowRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workflow_id: str
    task_id: str
    source: str
    nodes: list[WorkflowNode]
    edges: list[WorkflowEdge] = Field(default_factory=list)
    success: bool
    provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_dag(self) -> "WorkflowRecord":
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("workflow node IDs must be unique")
        known = set(node_ids)
        for edge in self.edges:
            if edge.src not in known or edge.dst not in known:
                raise ValueError(f"edge references unknown node: {edge.src} -> {edge.dst}")
            if edge.src == edge.dst:
                raise ValueError("self edges are not allowed")
        self.topological_order()
        return self

    def topological_order(self) -> list[str]:
        successors: dict[str, list[str]] = {node.node_id: [] for node in self.nodes}
        indegree = {node.node_id: 0 for node in self.nodes}
        for edge in self.edges:
            successors[edge.src].append(edge.dst)
            indegree[edge.dst] += 1

        ready = deque(sorted(node_id for node_id, degree in indegree.items() if degree == 0))
        order: list[str] = []
        while ready:
            node_id = ready.popleft()
            order.append(node_id)
            for dst in sorted(successors[node_id]):
                indegree[dst] -= 1
                if indegree[dst] == 0:
                    ready.append(dst)
        if len(order) != len(self.nodes):
            raise ValueError("workflow must be a DAG")
        return order

