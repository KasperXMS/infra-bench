from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class OperatorProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operator: str
    base_latency_s: float = Field(ge=0.0)
    required_capabilities: list[str] = Field(default_factory=list)
    cpu_weight: float = Field(default=0.0, ge=0.0)
    gpu_weight: float = Field(default=0.0, ge=0.0)
    memory_mb: float = Field(default=0.0, ge=0.0)
    output_ratio: float | None = Field(default=None, ge=0.0)
    fixed_output_mb: float | None = Field(default=None, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def output_rule_is_unambiguous(self) -> "OperatorProfile":
        if self.output_ratio is not None and self.fixed_output_mb is not None:
            raise ValueError("set only one of output_ratio and fixed_output_mb")
        return self

