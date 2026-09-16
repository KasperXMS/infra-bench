from dataclasses import dataclass

from ..schemas import (
    ExecutionMetrics,
    ExecutionResult,
    Executor,
    InfraState,
    OperatorProfile,
    WorkflowRecord,
)
from .network import transfer_between_sites_s


@dataclass(frozen=True)
class ArtifactState:
    site_id: str
    size_mb: float
    ready_s: float


def load_slowdown(load: float, alpha: float = 0.25) -> float:
    if not 0.0 <= load < 1.0:
        raise ValueError("load must be in [0, 1)")
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")
    return 1.0 + alpha * load / (1.0 - load)


def execution_time_s(
    profile: OperatorProfile, executor: Executor, alpha: float = 0.25
) -> float:
    speed_factor = executor.speed_factors.get(
        profile.operator, executor.speed_factors.get("default", 1.0)
    )
    if speed_factor <= 0:
        raise ValueError("executor speed factors must be positive")
    return profile.base_latency_s * speed_factor * load_slowdown(executor.load, alpha)


def output_size_mb(profile: OperatorProfile, total_input_mb: float) -> float:
    if profile.fixed_output_mb is not None:
        return profile.fixed_output_mb
    if profile.output_ratio is not None:
        return profile.output_ratio * total_input_mb
    return total_input_mb


def simulate_assignment(
    workflow: WorkflowRecord,
    infra: InfraState,
    profiles: dict[str, OperatorProfile],
    assignment: dict[str, str],
    *,
    load_alpha: float = 0.25,
) -> ExecutionResult:
    executors = {executor.executor_id: executor for executor in infra.executors()}
    nodes = {node.node_id: node for node in workflow.nodes}
    incoming = {node.node_id: [] for node in workflow.nodes}
    for edge in workflow.edges:
        incoming[edge.dst].append(edge)

    artifact_states = {
        artifact.artifact_id: ArtifactState(artifact.site_id, artifact.size_mb, 0.0)
        for artifact in infra.artifacts
    }
    timings: dict[str, dict[str, float]] = {}
    wan_mb = 0.0
    monetary_cost = 0.0

    for node_id in workflow.topological_order():
        node = nodes[node_id]
        profile = profiles.get(node.operator)
        if profile is None:
            return ExecutionResult(feasible=False, reason=f"missing profile: {node.operator}")
        executor = executors.get(assignment.get(node_id, ""))
        if executor is None:
            return ExecutionResult(feasible=False, reason=f"missing executor assignment: {node_id}")
        requirements = set(profile.required_capabilities) | set(node.required_capabilities)
        if not requirements.issubset(executor.capabilities):
            return ExecutionResult(
                feasible=False,
                assignment=assignment,
                reason=f"executor {executor.executor_id} lacks capabilities for {node_id}",
            )

        dependency_ready = 0.0
        edge_artifacts: set[str] = set()
        for edge in incoming[node_id]:
            dependency_ready = max(dependency_ready, timings[edge.src]["finish_s"])
            if edge.artifact is not None:
                edge_artifacts.add(edge.artifact)

        input_ids = list(dict.fromkeys([*node.input_artifacts, *sorted(edge_artifacts)]))
        total_input_mb = 0.0
        start_s = dependency_ready
        for artifact_id in input_ids:
            state = artifact_states.get(artifact_id)
            if state is None:
                return ExecutionResult(
                    feasible=False,
                    assignment=assignment,
                    reason=f"unknown input artifact {artifact_id!r} for node {node_id}",
                )
            total_input_mb += state.size_mb
            try:
                transfer_s = transfer_between_sites_s(
                    infra, state.site_id, executor.site_id, state.size_mb
                )
            except ValueError as exc:
                return ExecutionResult(feasible=False, assignment=assignment, reason=str(exc))
            if state.site_id != executor.site_id:
                wan_mb += state.size_mb
            start_s = max(start_s, state.ready_s + transfer_s)

        duration_s = execution_time_s(profile, executor, load_alpha)
        finish_s = start_s + duration_s
        timings[node_id] = {
            "start_s": start_s,
            "duration_s": duration_s,
            "finish_s": finish_s,
        }
        monetary_cost += duration_s * executor.monetary_cost_per_s

        produced_size = output_size_mb(profile, total_input_mb)
        per_artifact_size = produced_size / max(len(node.output_artifacts), 1)
        for artifact_id in node.output_artifacts:
            artifact_states[artifact_id] = ArtifactState(
                executor.site_id, per_artifact_size, finish_s
            )

    latency_s = max((timing["finish_s"] for timing in timings.values()), default=0.0)
    critical_node = max(timings, key=lambda item: timings[item]["finish_s"], default=None)
    return ExecutionResult(
        feasible=True,
        assignment=assignment,
        metrics=ExecutionMetrics(
            latency_s=latency_s,
            wan_mb=wan_mb,
            monetary_cost=monetary_cost,
        ),
        node_timings=timings,
        bottleneck={"critical_node": critical_node},
    )

