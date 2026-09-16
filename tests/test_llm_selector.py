import json
from types import SimpleNamespace

from infra_bench.generation.sanity import build_sanity_cases
from infra_bench.planners import LLMSelector


class FakeResponses:
    def __init__(self, selected: str) -> None:
        self.selected = selected
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            output_text=json.dumps(
                {"selected_workflow": self.selected, "rationale": "lowest transfer cost"}
            )
        )


def test_llm_selector_uses_strict_schema_and_oracle_free_prompt(profiles):
    case = build_sanity_cases(profiles)[0]
    case.task.evaluator_config = {"answer": "SECRET"}
    responses = FakeResponses(case.candidate_workflow_ids[0])
    client = SimpleNamespace(responses=responses)
    selector = LLMSelector("test-model", client=client)
    decision = selector.select(case.planner_input())

    assert decision.selected_workflow == case.candidate_workflow_ids[0]
    assert responses.kwargs["store"] is False
    assert responses.kwargs["text"]["format"]["strict"] is True
    assert "SECRET" not in responses.kwargs["input"]
    assert "oracle_workflow" not in responses.kwargs["input"]
