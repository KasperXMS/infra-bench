from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from infra_bench.schemas.scope_expansion_v0 import (
    ScopeTaskBankRecord,
    stable_document_site,
)
from infra_bench.scope_expansion.multihop_rag import (
    BM25DocumentIndex,
    MultiHopRAGAdapter,
    materialize_multihop_rag,
    official_answer_score,
    select_multihop_rag_tasks,
    stable_document_id,
)


def _document(index: int, terms: str) -> dict[str, object]:
    return {
        "title": f"Document {index}",
        "body": terms,
        "author": "Author",
        "source": "Source",
        "published_at": f"2024-01-{index + 1:02d}",
        "category": "technology",
        "url": f"https://example.test/{index}",
    }


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, list[dict[str, object]]]:
    documents = [
        _document(0, "alpha first bridge"),
        _document(1, "beta second bridge"),
        _document(2, "gamma third bridge"),
        _document(3, "delta fourth bridge"),
        *[_document(index, f"noise item {index}") for index in range(4, 35)],
    ]
    queries: list[dict[str, Any]] = []
    for hop in (2, 3, 4):
        queries.append(
            {
                "query": f"alpha beta gamma delta bridge hop {hop}",
                "answer": f"answer-{hop}",
                "question_type": "inference_query",
                "evidence_list": [
                    {
                        key: value
                        for key, value in documents[index].items()
                        if key != "body"
                    }
                    | {"fact": f"fact-{index}"}
                    for index in range(hop)
                ],
            }
        )
    corpus_path = tmp_path / "corpus.json"
    query_path = tmp_path / "MultiHopRAG.json"
    corpus_path.write_text(json.dumps(documents), encoding="utf-8")
    query_path.write_text(json.dumps(queries), encoding="utf-8")
    return corpus_path, query_path, documents


def test_adapter_resolves_evidence_and_stable_ids(tmp_path: Path) -> None:
    corpus_path, query_path, documents = _write_fixture(tmp_path)
    adapter = MultiHopRAGAdapter.load(corpus_path, query_path, dataset_revision="rev")

    assert len(adapter.documents) == 35
    assert len(adapter.queries) == 3
    assert adapter.queries[2].evidence_document_count == 4
    assert adapter.documents[0].document_id == stable_document_id(documents[0])
    assert stable_document_site(adapter.documents[0].document_id) in {"A4", "A5", "A28"}


def test_bm25_and_selection_do_not_need_gold_to_rank(tmp_path: Path) -> None:
    corpus_path, query_path, _ = _write_fixture(tmp_path)
    adapter = MultiHopRAGAdapter.load(corpus_path, query_path)
    index = BM25DocumentIndex(adapter.documents)
    before = index.rank(adapter.queries[0].query, top_n=30)
    changed_query = replace(
        adapter.queries[0],
        supporting_document_ids=tuple(
            reversed(adapter.queries[0].supporting_document_ids)
        ),
        answer="changed",
    )
    assert index.rank(changed_query.query, top_n=30) == before

    selection = select_multihop_rag_tasks(adapter, top_n=30, local_top_k=3)
    assert set(selection.selected) == {"M2", "M3", "M4"}
    assert all(item.coverage_complete for item in selection.selected.values())
    assert all(len(item.ranked_documents) == 30 for item in selection.selected.values())


def test_materialization_separates_visible_and_evaluator_only(tmp_path: Path) -> None:
    corpus_path, query_path, _ = _write_fixture(tmp_path)
    adapter = MultiHopRAGAdapter.load(corpus_path, query_path)
    selection = select_multihop_rag_tasks(adapter)
    output = tmp_path / "scope"
    result = materialize_multihop_rag(adapter, selection, output)

    assert result["task_count"] == 3
    records = [
        ScopeTaskBankRecord.model_validate_json(line)
        for line in (output / "task_bank.jsonl").read_text().splitlines()
    ]
    tasks = [record.planner_visible for record in records]
    assert len(tasks) == 3
    assert sorted(record.evaluator_only.evidence_document_count for record in records) == [2, 3, 4]
    for task in tasks:
        visible = json.dumps(task.model_dump(mode="json"), sort_keys=True)
        assert '"answer"' not in visible
        assert '"supporting_document_ids"' not in visible
        assert task.artifact_count == 30
        for artifact_id in task.artifact_ids:
            site = task.artifact_placement[artifact_id]
            assert (output / "multihop_rag" / f"task_{task.metadata['selection_label'].lower()}" / "artifacts" / site / f"{artifact_id}.json").is_file()
    evaluator_text = (output / "evaluator_only" / "multihop_rag.jsonl").read_text()
    assert "supporting_document_ids" in evaluator_text
    assert "answer-2" in evaluator_text


def test_official_answer_score_matches_upstream_token_overlap() -> None:
    assert official_answer_score('The answer to the question is "Sam Altman"', "Altman") == 1.0
    assert official_answer_score("unrelated", "Sam Altman") == 0.0
