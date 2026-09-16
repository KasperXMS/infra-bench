from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TaskRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    source: str
    instruction: str
    input_type: str
    artifact_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    evaluator_type: str
    evaluator_config: dict[str, Any] = Field(default_factory=dict)

