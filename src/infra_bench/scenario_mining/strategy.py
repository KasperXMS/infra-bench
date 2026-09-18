"""Generic MAS-realizable strategy templates and world-independent demand."""

from __future__ import annotations

from infra_bench.real_tasks.realizability import validate_workflow_realizability
from infra_bench.schemas import (
    GENERAL_MAS_OPERATOR_BINDINGS,
    ScenarioRecord,
    StrategyMultiplicity,
    WorkflowDemandTemplate,
    WorkflowEdge,
    WorkflowNode,
    WorkflowRecord,
)

from .features import utf8_size


def _centralized_template(scenario: ScenarioRecord) -> WorkflowRecord:
    artifact_ids = [item.artifact_id for item in scenario.artifacts]
    return WorkflowRecord(
        workflow_id=f"template-centralized-{scenario.scenario_id}",
        task_id=scenario.task_id,
        source="generic_strategy_template",
        nodes=[
            WorkflowNode(
                node_id="central_reasoning",
                operator="invoke_model",
                input_artifacts=artifact_ids,
                output_artifacts=["final_answer"],
            )
        ],
        success=False,
        provenance={"strategy_family": "centralized_raw", "screening_only": True},
    )


def _distributed_template(scenario: ScenarioRecord) -> WorkflowRecord:
    reducers: list[WorkflowNode] = []
    edges: list[WorkflowEdge] = []
    reduction_ids: list[str] = []
    for index, artifact in enumerate(scenario.artifacts, 1):
        reduction_id = f"reduction_{index:04d}"
        node_id = f"reduce_{index:04d}"
        reduction_ids.append(reduction_id)
        reducers.append(
            WorkflowNode(
                node_id=node_id,
                operator="invoke_model",
                input_artifacts=[artifact.artifact_id],
                output_artifacts=[reduction_id],
            )
        )
        edges.append(
            WorkflowEdge(src=node_id, dst="central_synthesis", artifact=reduction_id)
        )
    return WorkflowRecord(
        workflow_id=f"template-distributed-{scenario.scenario_id}",
        task_id=scenario.task_id,
        source="generic_strategy_template",
        nodes=[
            *reducers,
            WorkflowNode(
                node_id="central_synthesis",
                operator="invoke_model",
                input_artifacts=reduction_ids,
                output_artifacts=["final_answer"],
            ),
        ],
        edges=edges,
        success=False,
        provenance={
            "strategy_family": "distributed_reduction",
            "screening_only": True,
        },
    )


def build_strategy_validation_templates(
    scenario: ScenarioRecord,
) -> tuple[WorkflowRecord, WorkflowRecord]:
    """Build non-executed structural witnesses used only for realizability checks."""

    return _centralized_template(scenario), _distributed_template(scenario)


def _centralized_shape_valid(
    workflow: WorkflowRecord, scenario: ScenarioRecord
) -> bool:
    return (
        len(workflow.nodes) == 1
        and workflow.nodes[0].operator == "invoke_model"
        and set(workflow.nodes[0].input_artifacts)
        == {item.artifact_id for item in scenario.artifacts}
        and workflow.nodes[0].output_artifacts == ["final_answer"]
    )


def _distributed_shape_valid(
    workflow: WorkflowRecord, scenario: ScenarioRecord
) -> bool:
    artifact_ids = {item.artifact_id for item in scenario.artifacts}
    reducers = [item for item in workflow.nodes if item.node_id.startswith("reduce_")]
    synthesis = [item for item in workflow.nodes if item.node_id == "central_synthesis"]
    if len(reducers) != len(artifact_ids) or len(synthesis) != 1:
        return False
    reducer_inputs = [item.input_artifacts for item in reducers]
    if any(len(items) != 1 for items in reducer_inputs):
        return False
    if {items[0] for items in reducer_inputs} != artifact_ids:
        return False
    reducer_outputs = [item.output_artifacts for item in reducers]
    if any(len(items) != 1 for items in reducer_outputs):
        return False
    produced = {items[0] for items in reducer_outputs}
    final = synthesis[0]
    if set(final.input_artifacts) != produced or final.output_artifacts != ["final_answer"]:
        return False
    producers = [artifact for node in workflow.nodes for artifact in node.output_artifacts]
    return len(producers) == len(set(producers))


def validate_strategy_multiplicity(scenario: ScenarioRecord) -> StrategyMultiplicity:
    """Verify both generic DAG shapes against the actual general MAS surface."""

    centralized, distributed = build_strategy_validation_templates(scenario)
    central_realizability = validate_workflow_realizability(
        centralized,
        scenario.interaction_spec,
        runtime_bindings=GENERAL_MAS_OPERATOR_BINDINGS,
    )
    distributed_realizability = validate_workflow_realizability(
        distributed,
        scenario.interaction_spec,
        runtime_bindings=GENERAL_MAS_OPERATOR_BINDINGS,
    )
    central_ok = (
        central_realizability.status == "realizable"
        and _centralized_shape_valid(centralized, scenario)
        and bool(scenario.artifacts)
    )
    distributed_ok = (
        distributed_realizability.status == "realizable"
        and _distributed_shape_valid(distributed, scenario)
        and len(scenario.artifacts) >= 2
    )
    reasons: list[str] = []
    if not central_ok:
        reasons.append("centralized_raw is not structurally realizable")
    if not distributed_ok:
        reasons.append("distributed_reduction is not structurally realizable")
    if len(scenario.artifacts) < 2:
        reasons.append("distributed_reduction requires at least two artifacts")
    return StrategyMultiplicity(
        strategy_multiplicity=central_ok and distributed_ok,
        centralized_raw_realizable=central_ok,
        distributed_reduction_realizable=distributed_ok,
        reasons=reasons,
    )


def estimate_demands(
    scenario: ScenarioRecord,
) -> tuple[WorkflowDemandTemplate, WorkflowDemandTemplate]:
    """Estimate intrinsic D(G) without latency, placement, or transfer fields."""

    features = scenario.scene_features
    question_bytes = utf8_size(scenario.question)
    answers = scenario.answer if isinstance(scenario.answer, list) else [scenario.answer]
    answer_bytes = max((utf8_size(item) for item in answers), default=0)
    centralized = WorkflowDemandTemplate(
        strategy_id="centralized_raw",
        artifact_bytes_consumed=features.raw_bytes,
        artifact_bytes_produced=answer_bytes,
        reasoning_input_bytes=features.raw_bytes + question_bytes,
        reasoning_stages=1,
        local_preprocessing_bytes=0,
        potential_parallel_width=1,
        dependency_pattern="raw_fan_in",
    )
    if features.evidence_bytes is None:
        produced = None
        reasoning_input = None
    else:
        produced = features.evidence_bytes + answer_bytes
        reasoning_input = (
            features.raw_bytes
            + (features.artifact_count + 1) * question_bytes
            + features.evidence_bytes
        )
    distributed = WorkflowDemandTemplate(
        strategy_id="distributed_reduction",
        artifact_bytes_consumed=features.raw_bytes,
        artifact_bytes_produced=produced,
        reasoning_input_bytes=reasoning_input,
        reasoning_stages=2,
        local_preprocessing_bytes=features.raw_bytes,
        potential_parallel_width=max(1, features.artifact_count),
        dependency_pattern="parallel_map_then_reduce",
    )
    return centralized, distributed
