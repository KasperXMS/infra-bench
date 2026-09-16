from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ExecutionMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    latency_s: float = Field(ge=0.0)
    wan_mb: float = Field(ge=0.0)
    monetary_cost: float = Field(ge=0.0)


class ExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feasible: bool
    assignment: dict[str, str] = Field(default_factory=dict)
    metrics: ExecutionMetrics | None = None
    node_timings: dict[str, dict[str, float]] = Field(default_factory=dict)
    bottleneck: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None


class PlannerDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selected_workflow: str
    rationale: str | None = None
    raw_response: str | None = None


class EvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    group_id: str
    case_type: str
    planner: str
    selected_workflow: str
    oracle_workflow: str
    correct: bool
    selected_metrics: dict[str, float]
    oracle_metrics: dict[str, float]
    regret: float
    raw_response: str | None = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

