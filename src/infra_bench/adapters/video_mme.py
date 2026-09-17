from collections.abc import Iterable, Mapping
from typing import Any

from ..schemas import TaskRecord, WorkflowEdge, WorkflowNode, WorkflowRecord
from .base import BenchmarkAdapter
from .huggingface import HuggingFaceRowsClient


class VideoMMEV2Adapter(BenchmarkAdapter):
    dataset_id = "MME-Benchmarks/Video-MME-v2"
    source = "video_mme_v2"

    def __init__(self, client: HuggingFaceRowsClient | None = None) -> None:
        self.client = client or HuggingFaceRowsClient()

    def ingest_tasks(self, limit: int | None = None) -> Iterable[TaskRecord]:
        revision = self.client.revision(self.dataset_id)
        return (
            self.task_from_row(row, revision)
            for row in self.client.rows(self.dataset_id, limit=limit)
        )

    def task_from_row(self, row: Mapping[str, Any], revision: str) -> TaskRecord:
        video_id = str(row["video_id"])
        question_id = str(row["question_id"])
        instruction = f"{row['question']}\n{row['options']}"
        return TaskRecord(
            task_id=f"video_mme_v2:{question_id}",
            source=self.source,
            instruction=instruction,
            input_type="video",
            artifact_refs=[f"video-mme-v2://{video_id}"],
            metadata={
                "dataset_id": self.dataset_id,
                "dataset_revision": revision,
                "video_id": video_id,
                "question_id": question_id,
                "video_url": row.get("url"),
                "level": row.get("level"),
                "second_head": row.get("second_head"),
                "third_head": row.get("third_head"),
            },
            evaluator_type="video_mme_v2_grouped",
            evaluator_config={
                "answer": str(row["answer"]),
                "group_type": row.get("group_type"),
                "group_structure": row.get("group_structure"),
                "video_id": video_id,
                "dataset_id": self.dataset_id,
                "dataset_revision": revision,
            },
        )

    def ingest_workflows(self, task: TaskRecord) -> Iterable[WorkflowRecord]:
        prefix = task.task_id
        direct = WorkflowRecord(
            workflow_id=f"{prefix}:direct_strong_vlm",
            task_id=task.task_id,
            source=self.source,
            nodes=[
                WorkflowNode(node_id="vlm", operator="strong_vlm", input_artifacts=["raw_video"], output_artifacts=["evidence"]),
                WorkflowNode(node_id="reason", operator="reason", input_artifacts=["evidence"], output_artifacts=["answer"]),
            ],
            edges=[WorkflowEdge(src="vlm", dst="reason", artifact="evidence")],
            success=False,
            provenance={
                "type": "unverified_template",
                "correctness_basis": "not_evaluated",
                "admission_eligible": False,
            },
        )
        sampled = WorkflowRecord(
            workflow_id=f"{prefix}:sample_then_strong_vlm",
            task_id=task.task_id,
            source=self.source,
            nodes=[
                WorkflowNode(node_id="sample", operator="sample_frames", input_artifacts=["raw_video"], output_artifacts=["frames"]),
                WorkflowNode(node_id="vlm", operator="strong_vlm", input_artifacts=["frames"], output_artifacts=["evidence"]),
                WorkflowNode(node_id="reason", operator="reason", input_artifacts=["evidence"], output_artifacts=["answer"]),
            ],
            edges=[
                WorkflowEdge(src="sample", dst="vlm", artifact="frames"),
                WorkflowEdge(src="vlm", dst="reason", artifact="evidence"),
            ],
            success=False,
            provenance={
                "type": "unverified_template",
                "correctness_basis": "not_evaluated",
                "admission_eligible": False,
            },
        )
        return [direct, sampled]
