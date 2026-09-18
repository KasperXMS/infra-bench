from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from infra_bench.real_tasks.realizability import validate_workflow_realizability
from infra_bench.scenario_mining.adapters import (
    LongBenchV2Adapter,
    MuSiQueAdapter,
    ScenarioRowSkipped,
    TwoWikiMultiHopQAAdapter,
    iter_json_records,
)
from infra_bench.scenario_mining.miner import assess_scenario, mine_scenarios
from infra_bench.scenario_mining.strategy import (
    build_strategy_validation_templates,
    estimate_demands,
    validate_strategy_multiplicity,
)
from infra_bench.schemas import ScenarioEvidence


def _two_wiki_row(*, bad_support: bool = False) -> dict[str, Any]:
    title = "missing" if bad_support else "Doc A"
    return {
        "_id": "2wiki-1",
        "question": "Which facts connect the two entities?",
        "answer": "answer",
        "type": "bridge_comparison",
        "context": [
            ["Doc A", ["first fact", "x" * 100]],
            ["Doc B", ["second fact", "y" * 100]],
            ["Doc C", ["irrelevant", "z" * 100]],
        ],
        "supporting_facts": [[title, 0], ["Doc B", 0]],
        "evidences": [["a", "relation", "b"]],
    }


def _musique_row(*, raw_id: str = "m-1") -> dict[str, Any]:
    return {
        "id": raw_id,
        "question": "What is the linked answer?",
        "answer": "final",
        "answer_aliases": ["the final"],
        "answerable": True,
        "paragraphs": [
            {
                "idx": 0,
                "title": "A",
                "paragraph_text": "support one",
                "is_supporting": True,
            },
            {
                "idx": 1,
                "title": "B",
                "paragraph_text": "support two",
                "is_supporting": True,
            },
            {
                "idx": 2,
                "title": "C",
                "paragraph_text": "noise " + "x" * 200,
                "is_supporting": False,
            },
        ],
        "question_decomposition": [
            {"id": 0, "question": "q1", "answer": "a1", "paragraph_support_idx": 0},
            {"id": 1, "question": "q2", "answer": "a2", "paragraph_support_idx": 1},
        ],
    }


def _longbench_row(domain: str = "Multi-Document QA") -> dict[str, Any]:
    subdomain = "Academic" if domain == "Multi-Document QA" else "Other"
    return {
        "_id": "123456789012345678901234",
        "domain": domain,
        "sub_domain": subdomain,
        "difficulty": "easy",
        "length": "short",
        "question": "Choose the answer.",
        "choice_A": "one",
        "choice_B": "two",
        "choice_C": "three",
        "choice_D": "four",
        "answer": "B",
        "context": "opaque context with no official document boundaries",
    }


def test_2wiki_adapter_preserves_boundaries_and_hidden_evidence() -> None:
    adapter = TwoWikiMultiHopQAAdapter(revision="rev", split="validation")
    scenario = adapter.scenario_from_row(_two_wiki_row(), row_ordinal=7)

    assert scenario.dataset == "2wikimultihopqa"
    assert len(scenario.artifacts) == 3
    assert all(item.content is None for item in scenario.artifacts)
    assert scenario.evidence_status == "known"
    assert scenario.evidence is not None
    assert {item.artifact_id for item in scenario.evidence} == {"doc-0000", "doc-0001"}
    assert scenario.scene_features.evidence_artifact_count == 2
    assert scenario.scene_features.raw_evidence_ratio is not None
    assert "answer" not in scenario.interaction_spec.objective
    assert scenario.interaction_spec.operators == ["invoke_model"]
    planner_payload = scenario.planner_view()
    serialized_payload = json.dumps(planner_payload, sort_keys=True)
    for forbidden_key in (
        '"answer"',
        '"evidence"',
        '"supporting_fact"',
        '"question_decomposition"',
        '"external_evaluator"',
    ):
        assert forbidden_key not in serialized_payload


def test_unresolvable_2wiki_evidence_becomes_unknown_not_empty() -> None:
    adapter = TwoWikiMultiHopQAAdapter(revision="rev", split="validation")
    scenario = adapter.scenario_from_row(_two_wiki_row(bad_support=True), row_ordinal=0)

    assert scenario.evidence is None
    assert scenario.evidence_status == "not_applicable_or_incomplete"
    assert scenario.scene_features.evidence_bytes is None
    assert scenario.scene_features.evidence_artifact_count is None


def test_musique_full_uses_content_digest_and_paragraph_evidence() -> None:
    adapter = MuSiQueAdapter(revision="rev", split="dev", variant="full-v1.0")
    first = adapter.scenario_from_row(_musique_row(), row_ordinal=0)
    changed = _musique_row()
    changed["paragraphs"][2]["paragraph_text"] += " changed"
    second = adapter.scenario_from_row(changed, row_ordinal=1)

    assert first.task_id != second.task_id
    assert first.evidence_status == "known"
    assert first.scene_features.evidence_artifact_count == 2
    assert first.evaluator.target == ["final", "the final"]


def test_longbench_keeps_public_context_opaque_and_evidence_unknown() -> None:
    adapter = LongBenchV2Adapter(revision="pinned", split="train")
    scenario = adapter.scenario_from_row(_longbench_row(), row_ordinal=0)

    assert len(scenario.artifacts) == 1
    assert scenario.artifacts[0].metadata["derived_chunks"] is False
    assert scenario.evidence_status == "absent_in_public_release"
    assert scenario.scene_features.evidence_bytes is None
    assert "A. one" in scenario.question
    assert validate_strategy_multiplicity(scenario).strategy_multiplicity is False
    try:
        adapter.scenario_from_row(_longbench_row("Single-Document QA"), row_ordinal=1)
    except ScenarioRowSkipped:
        pass
    else:
        raise AssertionError("non-target LongBench category was not skipped")


def test_generic_strategies_are_realizable_and_demand_is_world_independent() -> None:
    scenario = TwoWikiMultiHopQAAdapter(
        revision="rev", split="validation"
    ).scenario_from_row(_two_wiki_row(), row_ordinal=0)

    multiplicity = validate_strategy_multiplicity(scenario)
    centralized, distributed = estimate_demands(scenario)
    centralized_dag, distributed_dag = build_strategy_validation_templates(scenario)
    raw_ids = {item.artifact_id for item in scenario.artifacts}

    assert multiplicity.strategy_multiplicity is True
    assert centralized_dag.success is False
    assert distributed_dag.success is False
    assert {item.operator for item in centralized_dag.nodes} == {"invoke_model"}
    assert {item.operator for item in distributed_dag.nodes} == {"invoke_model"}
    assert set(centralized_dag.nodes[0].input_artifacts) == raw_ids
    reducers = [
        item for item in distributed_dag.nodes if item.node_id.startswith("reduce_")
    ]
    synthesis = next(
        item for item in distributed_dag.nodes if item.node_id == "central_synthesis"
    )
    assert len(reducers) == len(raw_ids)
    assert {item.input_artifacts[0] for item in reducers} == raw_ids
    assert all(len(item.input_artifacts) == 1 for item in reducers)
    assert len(distributed_dag.edges) == len(raw_ids)
    assert set(synthesis.input_artifacts) == {
        item.output_artifacts[0] for item in reducers
    }
    assert centralized.dependency_pattern == "raw_fan_in"
    assert centralized.reasoning_stages == 1
    assert distributed.dependency_pattern == "parallel_map_then_reduce"
    assert distributed.reasoning_stages == 2
    assert distributed.potential_parallel_width == 3
    for demand in (centralized, distributed):
        dumped = demand.model_dump()
        assert "cross_site_bytes" not in dumped
        assert "network_latency" not in dumped
        assert "e2e" not in dumped


def test_strategy_realizability_fails_closed_and_ignores_gold_evidence() -> None:
    scenario = TwoWikiMultiHopQAAdapter(
        revision="rev", split="validation"
    ).scenario_from_row(_two_wiki_row(), row_ordinal=0)
    centralized, distributed = build_strategy_validation_templates(scenario)
    assert (
        validate_workflow_realizability(
            centralized,
            scenario.interaction_spec,
            runtime_bindings=frozenset(),
        ).status
        == "not_realizable"
    )

    blocked_interaction = scenario.interaction_spec.model_copy(
        update={"operators": ["read_artifact"], "observations": []}
    )
    blocked = scenario.model_copy(update={"interaction_spec": blocked_interaction})
    assert validate_strategy_multiplicity(blocked).strategy_multiplicity is False

    changed_evidence = scenario.model_copy(
        update={
            "evidence": [
                ScenarioEvidence(
                    artifact_id="doc-0002",
                    span="sentence:1",
                    supporting_fact="entirely different hidden gold evidence",
                )
            ]
        }
    )
    changed_centralized, changed_distributed = build_strategy_validation_templates(
        changed_evidence
    )
    assert changed_centralized.nodes == centralized.nodes
    assert changed_centralized.edges == centralized.edges
    assert changed_distributed.nodes == distributed.nodes
    assert changed_distributed.edges == distributed.edges


def test_miner_hard_gates_unknown_queue_and_never_admits(tmp_path: Path) -> None:
    two_wiki = TwoWikiMultiHopQAAdapter(
        revision="rev", split="validation"
    ).scenario_from_row(_two_wiki_row(), row_ordinal=0)
    musique = MuSiQueAdapter(revision="rev", split="dev").scenario_from_row(
        _musique_row(), row_ordinal=0
    )
    longbench = LongBenchV2Adapter(revision="pinned", split="train").scenario_from_row(
        _longbench_row(), row_ordinal=0
    )

    longbench_assessment = assess_scenario(longbench)
    assert longbench_assessment["queue"] == "evidence_unknown"
    assert "artifact_count < 3" in longbench_assessment["shortlist_reason"]
    report = mine_scenarios(
        [two_wiki, musique, longbench],
        output_dir=tmp_path,
        raw_candidate_limit=40,
        shortlist_limit=15,
    )

    assert report["summary"]["input_scenarios"] == 3
    assert report["summary"]["shortlist"] == 2
    assert report["summary"]["evidence_unknown"] == 1
    assert report["summary"]["calibration_candidates"] == 2
    assert report["summary"]["admitted_pairs"] == 0
    assert report["summary"]["model_api_calls"] == 0
    assert report["research_boundary"]["modeled_reversal_is_admission"] is False
    assert (tmp_path / "scenario_bank.jsonl").is_file()
    bank_ids = {
        json.loads(line)["scenario_id"]
        for line in (tmp_path / "scenario_bank.jsonl").read_text().splitlines()
    }
    assert {item["scenario_id"] for item in report["shortlist"]} <= bank_ids
    assert json.loads((tmp_path / "dataset_stats.json").read_text())["total_scenarios"] == 3


def test_json_and_jsonl_record_reader(tmp_path: Path) -> None:
    json_path = tmp_path / "rows.json"
    json_path.write_text(json.dumps([{"id": 1}, {"id": 2}]), encoding="utf-8")
    jsonl_path = tmp_path / "rows.jsonl"
    jsonl_path.write_text('{"id": 3}\n{"id": 4}\n', encoding="utf-8")

    assert [row["id"] for row in iter_json_records(json_path)] == [1, 2]
    assert [row["id"] for row in iter_json_records(jsonl_path)] == [3, 4]
