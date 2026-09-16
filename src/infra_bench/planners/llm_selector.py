import json
from typing import Any

from ..schemas import PlannerDecision, PlannerInput
from .base import Planner


class LLMSelector(Planner):
    """Infrastructure-aware selector backed by the OpenAI Responses API."""

    def __init__(
        self,
        model: str,
        *,
        client: Any | None = None,
        max_output_tokens: int = 300,
    ) -> None:
        if not model:
            raise ValueError("model is required")
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "LLM evaluation requires the optional dependency: uv sync --extra llm"
                ) from exc
            client = OpenAI()
        self.client = client
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.name = f"llm:{model}"

    def select(self, case: PlannerInput) -> PlannerDecision:
        if case.infra is None:
            raise ValueError("the resource-aware LLM selector requires infrastructure input")
        workflow_ids = [workflow.workflow_id for workflow in case.candidate_workflows]
        schema = {
            "type": "object",
            "properties": {
                "selected_workflow": {"type": "string", "enum": workflow_ids},
                "rationale": {"type": "string"},
            },
            "required": ["selected_workflow", "rationale"],
            "additionalProperties": False,
        }
        prompt = json.dumps(
            {
                "task": case.task.model_dump(exclude={"evaluator_config"}),
                "infrastructure": case.infra.model_dump(),
                "candidate_workflows": [
                    workflow.model_dump() for workflow in case.candidate_workflows
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        response = self.client.responses.create(
            model=self.model,
            instructions=(
                "Choose the semantic workflow that minimizes end-to-end latency under the "
                "given infrastructure. Account for data locality, transfer size, executor "
                "capabilities, and load. Do not invent or modify a workflow."
            ),
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "workflow_selection",
                    "strict": True,
                    "schema": schema,
                }
            },
            max_output_tokens=self.max_output_tokens,
            store=False,
        )
        raw = response.output_text
        payload = json.loads(raw)
        selected = payload["selected_workflow"]
        if selected not in workflow_ids:
            raise ValueError(f"model selected unknown workflow: {selected}")
        return PlannerDecision(
            selected_workflow=selected,
            rationale=payload.get("rationale"),
            raw_response=raw,
        )
