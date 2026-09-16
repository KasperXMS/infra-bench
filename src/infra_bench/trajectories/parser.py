from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..io import read_jsonl


class TraceAction(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    input_artifacts: list[str] = Field(default_factory=list)
    output_artifacts: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrajectoryRecord(BaseModel):
    """Normalized interchange format for benchmark-agent traces."""

    model_config = ConfigDict(extra="allow")

    task_id: str
    success: bool
    actions: list[TraceAction]
    provenance: dict[str, Any] = Field(default_factory=dict)


def read_trajectories(path: str | Path) -> list[TrajectoryRecord]:
    return read_jsonl(path, TrajectoryRecord)
