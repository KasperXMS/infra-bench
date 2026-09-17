import json
from collections.abc import Iterable, Mapping
from typing import Any

from ..schemas import (
    ExternalEvaluatorSpec,
    InitialArtifactSpec,
    ObservationSpec,
    RuntimeVerifierSpec,
    TaskInteractionSpec,
    TaskRecord,
    WorkflowEdge,
    WorkflowNode,
    WorkflowRecord,
)
from .base import BenchmarkAdapter
from .huggingface import HuggingFaceRowsClient


class SweBenchVerifiedAdapter(BenchmarkAdapter):
    dataset_id = "SWE-bench/SWE-bench_Verified"
    source = "swebench_verified"

    def __init__(self, client: HuggingFaceRowsClient | None = None) -> None:
        self.client = client or HuggingFaceRowsClient()

    def ingest_tasks(self, limit: int | None = None) -> Iterable[TaskRecord]:
        revision = self.client.revision(self.dataset_id)
        return (
            self.task_from_row(row, revision)
            for row in self.client.rows(self.dataset_id, limit=limit)
        )

    def task_from_row(self, row: Mapping[str, Any], revision: str) -> TaskRecord:
        instance_id = str(row["instance_id"])
        repo = str(row["repo"])
        base_commit = str(row["base_commit"])
        return TaskRecord(
            task_id=instance_id,
            source=self.source,
            instruction=str(row["problem_statement"]),
            input_type="repository",
            artifact_refs=[f"repo://{repo}@{base_commit}"],
            metadata={
                "dataset_id": self.dataset_id,
                "dataset_revision": revision,
                "repo": repo,
                "base_commit": base_commit,
                "version": row.get("version"),
                "difficulty": row.get("difficulty"),
                "image": row.get("image"),
            },
            evaluator_type="swebench",
            evaluator_config={
                "instance_id": instance_id,
                "fail_to_pass": _string_list(row.get("FAIL_TO_PASS")),
                "pass_to_pass": _string_list(row.get("PASS_TO_PASS")),
                "dataset_id": self.dataset_id,
                "dataset_revision": revision,
            },
            interaction_spec=TaskInteractionSpec(
                task_id=instance_id,
                objective=str(row["problem_statement"]),
                initial_artifacts=[
                    InitialArtifactSpec(
                        artifact_id="repository",
                        kind="git_repository",
                        source_ref=f"repo://{repo}@{base_commit}",
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
                        observation_id="test_feedback",
                        produced_by=["run_targeted_test", "run_full_test"],
                        description="Exit status and bounded test output.",
                    ),
                    ObservationSpec(
                        observation_id="mutation_feedback",
                        produced_by=["edit_file", "apply_patch", "submit_patch"],
                        description="Mutation or submission success and bounded diagnostics.",
                    ),
                    ObservationSpec(
                        observation_id="model_output",
                        produced_by=["invoke_model"],
                        description="Model reasoning output used by the Planner.",
                    ),
                ],
                runtime_verifier=RuntimeVerifierSpec(
                    level="partial",
                    signals=["targeted_test_result", "full_test_result"],
                ),
                external_evaluator=ExternalEvaluatorSpec(
                    evaluator_id="swebench_official"
                ),
            ),
        )

    def ingest_workflows(self, task: TaskRecord) -> Iterable[WorkflowRecord]:
        prefix = task.task_id
        full_remote = WorkflowRecord(
            workflow_id=f"{prefix}:remote_inspect",
            task_id=task.task_id,
            source=self.source,
            nodes=[
                WorkflowNode(node_id="inspect", operator="inspect_repo", input_artifacts=["repository"], output_artifacts=["repo_index"], required_capabilities=["cloud_full_repo"]),
                WorkflowNode(node_id="search", operator="search_code", input_artifacts=["repo_index"], output_artifacts=["candidate_files"]),
                WorkflowNode(node_id="read", operator="read_file", input_artifacts=["candidate_files"], output_artifacts=["code_context"]),
                WorkflowNode(node_id="reason", operator="reason", input_artifacts=["code_context"], output_artifacts=["edit_plan"]),
                WorkflowNode(node_id="edit", operator="edit_code", input_artifacts=["candidate_files", "edit_plan"], output_artifacts=["patch"]),
                WorkflowNode(node_id="test", operator="run_targeted_test", input_artifacts=["patch"], output_artifacts=["test_result"]),
                WorkflowNode(node_id="verify", operator="verify", input_artifacts=["test_result"], output_artifacts=["final_patch"]),
            ],
            edges=[
                WorkflowEdge(src="inspect", dst="search", artifact="repo_index"),
                WorkflowEdge(src="search", dst="read", artifact="candidate_files"),
                WorkflowEdge(src="read", dst="reason", artifact="code_context"),
                WorkflowEdge(src="reason", dst="edit", artifact="edit_plan"),
                WorkflowEdge(src="search", dst="edit", artifact="candidate_files"),
                WorkflowEdge(src="edit", dst="test", artifact="patch"),
                WorkflowEdge(src="test", dst="verify", artifact="test_result"),
            ],
            success=False,
            provenance={
                "type": "unverified_template",
                "correctness_basis": "not_evaluated",
                "admission_eligible": False,
            },
        )
        local_filter = WorkflowRecord(
            workflow_id=f"{prefix}:local_filter_remote_reason",
            task_id=task.task_id,
            source=self.source,
            nodes=[
                WorkflowNode(node_id="search", operator="search_code", input_artifacts=["repository"], output_artifacts=["candidate_files"], required_capabilities=["local_repo"]),
                WorkflowNode(node_id="analyze", operator="static_analysis", input_artifacts=["candidate_files"], output_artifacts=["analysis"]),
                WorkflowNode(node_id="reason", operator="reason", input_artifacts=["analysis"], output_artifacts=["edit_plan"]),
                WorkflowNode(node_id="edit", operator="edit_code", input_artifacts=["candidate_files", "edit_plan"], output_artifacts=["patch"]),
                WorkflowNode(node_id="test", operator="run_targeted_test", input_artifacts=["patch"], output_artifacts=["test_result"]),
                WorkflowNode(node_id="verify", operator="verify", input_artifacts=["test_result"], output_artifacts=["final_patch"]),
            ],
            edges=[
                WorkflowEdge(src="search", dst="analyze", artifact="candidate_files"),
                WorkflowEdge(src="analyze", dst="reason", artifact="analysis"),
                WorkflowEdge(src="reason", dst="edit", artifact="edit_plan"),
                WorkflowEdge(src="search", dst="edit", artifact="candidate_files"),
                WorkflowEdge(src="edit", dst="test", artifact="patch"),
                WorkflowEdge(src="test", dst="verify", artifact="test_result"),
            ],
            success=False,
            provenance={
                "type": "unverified_template",
                "correctness_basis": "not_evaluated",
                "admission_eligible": False,
            },
        )
        return [full_remote, local_filter]


def _string_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        decoded = json.loads(value)
        if not isinstance(decoded, list):
            raise ValueError("expected a JSON list")
        return [str(item) for item in decoded]
    return [str(item) for item in value]
