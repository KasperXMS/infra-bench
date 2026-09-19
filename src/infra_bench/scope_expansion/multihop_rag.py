"""Deterministic MultiHop-RAG ingestion, retrieval, selection, and materialization.

Gold evidence is used only after retrieval to measure coverage and select a fixed
M2/M3/M4 cohort.  Neither BM25 scoring nor artifact placement accepts gold data.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, cast

from infra_bench.schemas.scope_expansion_v0 import (
    SITE_BY_SHARD,
    ScopeEvaluatorRecord,
    ScopeRetrievalConfig,
    ScopeSiteId,
    ScopeTask,
    ScopeTaskBankEvaluatorSummary,
    ScopeTaskBankRecord,
    stable_document_site,
)

DATASET_NAME = "MultiHop-RAG"
DATASET_REPOSITORY = "yixuantt/MultiHopRAG"
DATASET_REVISION = "71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82"
REFERENCE_REPOSITORY = "yixuantt/MultiHop-RAG"
REFERENCE_REVISION = "c1c1287aa60a94acf9c4d20c891c9cd611a0f6e8"
DEFAULT_TOP_N = 30
FALLBACK_TOP_N = 50
PREREGISTERED_LOCAL_TOP_K = 3
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest_id(prefix: str, value: object) -> str:
    digest = sha256(_canonical_json(value).encode("utf-8")).hexdigest()[:24]
    return f"{prefix}-{digest}"


def stable_document_id(row: Mapping[str, Any]) -> str:
    """Return a corpus-order-independent identifier for one natural document."""

    identity = {
        "url": row.get("url"),
        "title": row.get("title"),
        "source": row.get("source"),
        "published_at": row.get("published_at"),
    }
    return _digest_id("mhr-doc", identity)


def stable_task_id(query: str) -> str:
    return _digest_id("multihop-rag-train", {"query": query})


def tokenize(text: str) -> tuple[str, ...]:
    """Use the preregistered dependency-free deterministic lexical tokenizer."""

    return tuple(TOKEN_PATTERN.findall(text.casefold()))


def _render_document(row: Mapping[str, Any]) -> bytes:
    visible = {
        "author": row.get("author"),
        "body": row.get("body"),
        "category": row.get("category"),
        "published_at": row.get("published_at"),
        "source": row.get("source"),
        "title": row.get("title"),
        "url": row.get("url"),
    }
    return (_canonical_json(visible) + "\n").encode("utf-8")


def _retrieval_text(row: Mapping[str, Any]) -> str:
    # Metadata matters for this benchmark.  The fixed field order is part of the
    # registered retriever and is independent of query annotations.
    fields = ("title", "source", "published_at", "author", "category", "body", "url")
    return "\n".join(str(row.get(field) or "") for field in fields)


def _evidence_identity(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("url") or ""),
        str(row.get("title") or ""),
        str(row.get("source") or ""),
        str(row.get("published_at") or ""),
    )


@dataclass(frozen=True)
class MultiHopRAGDocument:
    document_id: str
    source_ordinal: int
    row: Mapping[str, Any]
    payload: bytes
    tokens: tuple[str, ...]

    @property
    def size_bytes(self) -> int:
        return len(self.payload)


@dataclass(frozen=True)
class MultiHopRAGQuery:
    task_id: str
    source_ordinal: int
    query: str
    answer: str
    question_type: str
    supporting_document_ids: tuple[str, ...]
    evidence_list: tuple[Mapping[str, Any], ...]

    @property
    def evidence_document_count(self) -> int:
        return len(self.supporting_document_ids)


@dataclass(frozen=True)
class RankedDocument:
    document_id: str
    score: float
    rank: int


@dataclass(frozen=True)
class MultiHopRAGCandidate:
    query: MultiHopRAGQuery
    ranked_documents: tuple[RankedDocument, ...]
    coverage_complete: bool
    covered_support_count: int
    bundle_bytes: int
    bundle_tokens: int
    documents_per_agent: Mapping[str, int]
    support_sites: tuple[str, ...]
    answer_fact_mentions: int

    @property
    def structural_multihop_passed(self) -> bool:
        return self.query.question_type in {
            "comparison_query",
            "temporal_query",
        } or self.answer_fact_mentions == 0


@dataclass(frozen=True)
class MultiHopRAGSelection:
    top_n: int
    local_top_k: int
    candidates: tuple[MultiHopRAGCandidate, ...]
    selected: Mapping[str, MultiHopRAGCandidate]
    coverage_by_hop: Mapping[int, Mapping[str, int | float]]


class MultiHopRAGAdapter:
    """Load the official corpus and QA rows while preserving document boundaries."""

    def __init__(
        self,
        *,
        documents: Sequence[MultiHopRAGDocument],
        queries: Sequence[MultiHopRAGQuery],
        corpus_sha256: str,
        queries_sha256: str,
        dataset_revision: str = DATASET_REVISION,
    ) -> None:
        self.documents = tuple(documents)
        self.queries = tuple(queries)
        self.corpus_sha256 = corpus_sha256
        self.queries_sha256 = queries_sha256
        self.dataset_revision = dataset_revision

    @classmethod
    def load(
        cls,
        corpus_path: str | Path,
        queries_path: str | Path,
        *,
        dataset_revision: str = DATASET_REVISION,
    ) -> MultiHopRAGAdapter:
        corpus_source = Path(corpus_path)
        query_source = Path(queries_path)
        corpus_bytes = corpus_source.read_bytes()
        query_bytes = query_source.read_bytes()
        raw_corpus = cast(object, json.loads(corpus_bytes))
        raw_queries = cast(object, json.loads(query_bytes))
        if not isinstance(raw_corpus, list) or not isinstance(raw_queries, list):
            raise ValueError("MultiHop-RAG corpus and query files must be JSON arrays")

        documents: list[MultiHopRAGDocument] = []
        identities: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
        seen_document_ids: set[str] = set()
        for ordinal, value in enumerate(cast(list[object], raw_corpus)):
            if not isinstance(value, dict):
                raise ValueError(f"invalid corpus row {ordinal}")
            row = cast(dict[str, Any], value)
            if not str(row.get("title") or "").strip() or not str(
                row.get("body") or ""
            ).strip():
                raise ValueError(f"corpus row {ordinal} lacks title or body")
            document_id = stable_document_id(row)
            if document_id in seen_document_ids:
                raise ValueError(f"duplicate stable document ID: {document_id}")
            seen_document_ids.add(document_id)
            document = MultiHopRAGDocument(
                document_id=document_id,
                source_ordinal=ordinal,
                row=row,
                payload=_render_document(row),
                tokens=tokenize(_retrieval_text(row)),
            )
            documents.append(document)
            identities[_evidence_identity(row)].append(document_id)

        queries: list[MultiHopRAGQuery] = []
        seen_task_ids: set[str] = set()
        for ordinal, value in enumerate(cast(list[object], raw_queries)):
            if not isinstance(value, dict):
                raise ValueError(f"invalid query row {ordinal}")
            row = cast(dict[str, Any], value)
            query = str(row.get("query") or "").strip()
            answer = str(row.get("answer") or "").strip()
            question_type = str(row.get("question_type") or "").strip()
            evidence_value = row.get("evidence_list")
            if not query or not answer or not question_type or not isinstance(
                evidence_value, list
            ):
                raise ValueError(f"invalid query row {ordinal}")
            task_id = stable_task_id(query)
            if task_id in seen_task_ids:
                raise ValueError(f"duplicate stable task ID: {task_id}")
            seen_task_ids.add(task_id)

            evidence_rows: list[Mapping[str, Any]] = []
            supporting_ids: list[str] = []
            for evidence in cast(list[object], evidence_value):
                if not isinstance(evidence, dict):
                    raise ValueError(f"invalid evidence in query row {ordinal}")
                evidence_row = cast(dict[str, Any], evidence)
                matches = identities.get(_evidence_identity(evidence_row), [])
                if len(matches) != 1:
                    raise ValueError(
                        f"query row {ordinal} evidence resolves to {len(matches)} corpus documents"
                    )
                evidence_rows.append(evidence_row)
                supporting_ids.append(matches[0])
            queries.append(
                MultiHopRAGQuery(
                    task_id=task_id,
                    source_ordinal=ordinal,
                    query=query,
                    answer=answer,
                    question_type=question_type,
                    supporting_document_ids=tuple(dict.fromkeys(supporting_ids)),
                    evidence_list=tuple(evidence_rows),
                )
            )
        return cls(
            documents=documents,
            queries=queries,
            corpus_sha256=sha256(corpus_bytes).hexdigest(),
            queries_sha256=sha256(query_bytes).hexdigest(),
            dataset_revision=dataset_revision,
        )


class BM25DocumentIndex:
    """Small, dependency-free BM25 index over the complete document corpus."""

    def __init__(
        self,
        documents: Sequence[MultiHopRAGDocument],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not documents:
            raise ValueError("BM25 corpus cannot be empty")
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("invalid BM25 parameters")
        self.documents = tuple(documents)
        self.k1 = k1
        self.b = b
        self._term_frequencies = tuple(Counter(item.tokens) for item in documents)
        self._lengths = tuple(len(item.tokens) for item in documents)
        self._average_length = sum(self._lengths) / len(self._lengths)
        document_frequency: Counter[str] = Counter()
        for item in self._term_frequencies:
            document_frequency.update(item.keys())
        corpus_size = len(documents)
        self._idf = {
            term: math.log(1.0 + (corpus_size - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def rank(self, query: str, *, top_n: int) -> tuple[RankedDocument, ...]:
        if top_n <= 0:
            raise ValueError("top_n must be positive")
        query_terms = Counter(tokenize(query))
        scored: list[tuple[float, str]] = []
        for index, document in enumerate(self.documents):
            score = 0.0
            frequencies = self._term_frequencies[index]
            length_norm = 1.0 - self.b + self.b * (
                self._lengths[index] / self._average_length
            )
            for term, query_frequency in query_terms.items():
                frequency = frequencies.get(term, 0)
                if frequency == 0:
                    continue
                numerator = frequency * (self.k1 + 1.0)
                denominator = frequency + self.k1 * length_norm
                score += self._idf.get(term, 0.0) * numerator / denominator
                # Repeated query terms are intentionally not given extra weight,
                # matching the usual BM25 query treatment.
                _ = query_frequency
            scored.append((score, document.document_id))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return tuple(
            RankedDocument(document_id=document_id, score=score, rank=rank)
            for rank, (score, document_id) in enumerate(scored[:top_n], 1)
        )


def _coverage_by_hop(
    candidates: Iterable[MultiHopRAGCandidate],
) -> dict[int, dict[str, int | float]]:
    counts: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for candidate in candidates:
        hop = candidate.query.evidence_document_count
        if hop not in (2, 3, 4):
            continue
        counts[hop][0] += 1
        counts[hop][1] += int(candidate.coverage_complete)
    return {
        hop: {
            "eligible_queries": counts[hop][0],
            "fully_covered_queries": counts[hop][1],
            "full_coverage_rate": (
                counts[hop][1] / counts[hop][0] if counts[hop][0] else 0.0
            ),
        }
        for hop in (2, 3, 4)
    }


def _answer_fact_mentions(query: MultiHopRAGQuery) -> int:
    answer_tokens = tokenize(query.answer)
    if not answer_tokens:
        return 0
    answer_phrase = " ".join(answer_tokens)
    return sum(
        answer_phrase in " ".join(tokenize(str(item.get("fact") or "")))
        for item in query.evidence_list
    )


def select_multihop_rag_tasks(
    adapter: MultiHopRAGAdapter,
    *,
    top_n: int = DEFAULT_TOP_N,
    local_top_k: int = PREREGISTERED_LOCAL_TOP_K,
) -> MultiHopRAGSelection:
    """Run one fixed full-corpus retrieval and select real M2/M3/M4 tasks."""

    if top_n not in (DEFAULT_TOP_N, FALLBACK_TOP_N):
        raise ValueError("the registered candidate size must be globally 30 or 50")
    if local_top_k <= 0:
        raise ValueError("local_top_k must be positive")
    index = BM25DocumentIndex(adapter.documents)
    document_by_id = {item.document_id: item for item in adapter.documents}
    candidates: list[MultiHopRAGCandidate] = []
    for query in adapter.queries:
        if query.evidence_document_count not in (2, 3, 4):
            continue
        ranked = index.rank(query.query, top_n=top_n)
        ranked_ids = {item.document_id for item in ranked}
        support_ids = set(query.supporting_document_ids)
        documents = [document_by_id[item.document_id] for item in ranked]
        site_counts = Counter(stable_document_site(item.document_id) for item in documents)
        support_sites = tuple(
            sorted(
                {stable_document_site(item) for item in query.supporting_document_ids},
                key=SITE_BY_SHARD.index,
            )
        )
        candidates.append(
            MultiHopRAGCandidate(
                query=query,
                ranked_documents=ranked,
                coverage_complete=support_ids <= ranked_ids,
                covered_support_count=len(support_ids & ranked_ids),
                bundle_bytes=sum(item.size_bytes for item in documents),
                bundle_tokens=sum(len(item.tokens) for item in documents),
                documents_per_agent={site: site_counts[site] for site in SITE_BY_SHARD},
                support_sites=support_sites,
                answer_fact_mentions=_answer_fact_mentions(query),
            )
        )

    selected: dict[str, MultiHopRAGCandidate] = {}
    for hop in (2, 3, 4):
        eligible = [
            item
            for item in candidates
            if item.query.evidence_document_count == hop
            and item.coverage_complete
            and len(item.support_sites) >= 2
            and item.query.question_type != "null_query"
            and item.structural_multihop_passed
        ]
        if not eligible:
            raise ValueError(f"no fully covered distributed M{hop} candidate")
        eligible.sort(
            key=lambda item: (
                item.query.question_type not in {"comparison_query", "temporal_query"},
                -len(item.support_sites),
                -item.bundle_bytes,
                item.query.task_id,
            )
        )
        selected[f"M{hop}"] = eligible[0]
    return MultiHopRAGSelection(
        top_n=top_n,
        local_top_k=local_top_k,
        candidates=tuple(candidates),
        selected=selected,
        coverage_by_hop=_coverage_by_hop(candidates),
    )


def official_answer_score(prediction: str, gold: str) -> float:
    """Reproduce the upstream QA evaluator's case-insensitive token overlap."""

    match = re.search(r'The answer to the question is "(.*?)"', prediction)
    extracted = match.group(1) if match else prediction
    return float(bool(set(extracted.lower().split()) & set(gold.lower().split())))


def _source_ref(task_label: str, site: str, artifact_id: str) -> str:
    return f"scope-expansion-v0://multihop-rag/{task_label}/{site}/{artifact_id}.json"


def _candidate_record(candidate: MultiHopRAGCandidate) -> dict[str, Any]:
    return {
        "task_id": candidate.query.task_id,
        "source_row_ordinal": candidate.query.source_ordinal,
        "query": candidate.query.query,
        "question_type": candidate.query.question_type,
        "evidence_document_count": candidate.query.evidence_document_count,
        "coverage_complete": candidate.coverage_complete,
        "covered_support_count": candidate.covered_support_count,
        "candidate_document_ids": [
            item.document_id for item in candidate.ranked_documents
        ],
        "candidate_scores": [item.score for item in candidate.ranked_documents],
        "bundle_bytes": candidate.bundle_bytes,
        "bundle_tokens_lexical": candidate.bundle_tokens,
        "documents_per_agent": dict(candidate.documents_per_agent),
        "evaluator_only": {
            "answer": candidate.query.answer,
            "supporting_document_ids": list(candidate.query.supporting_document_ids),
            "support_sites": list(candidate.support_sites),
            "answer_fact_mentions": candidate.answer_fact_mentions,
            "structural_multihop_passed": candidate.structural_multihop_passed,
        },
    }


def _markdown_report(
    adapter: MultiHopRAGAdapter, selection: MultiHopRAGSelection
) -> str:
    lines = [
        "# MultiHop-RAG task selection",
        "",
        "This is an offline selection report. Gold annotations below are evaluator-only and",
        "are not serialized into runtime task records or retrieval inputs.",
        "",
        "## Fixed protocol",
        "",
        f"- Dataset: `{DATASET_REPOSITORY}` at revision `{adapter.dataset_revision}`",
        f"- Dataset files: corpus `{adapter.corpus_sha256}`, queries `{adapter.queries_sha256}`",
        f"- Reference evaluator code: `{REFERENCE_REPOSITORY}` at `{REFERENCE_REVISION}`",
        f"- Candidate retrieval: document-level BM25 over all {len(adapter.documents)} documents, global Top-{selection.top_n}",
        "- Tokenizer: lower-case regex `[a-z0-9]+`; BM25 k1=1.5, b=0.75",
        f"- Runtime distributed retrieval is preregistered as per-shard Top-{selection.local_top_k}",
        "- Placement: SHA-256(document_id), first 8 bytes big-endian modulo 3",
        "- Gold evidence affects only offline coverage, selection, and evaluation; never ranking or placement",
        "- Multi-hop selection gate: comparison/temporal query, or an inference query whose answer is absent from every individual support fact",
        "",
        "Top-30 was retained. Top-50 is permitted only as a single global fallback if overall",
        "coverage proves inadequate; it was not tuned per task.",
        "",
        "## Coverage",
        "",
        "| support documents | eligible | fully covered | rate |",
        "|---:|---:|---:|---:|",
    ]
    for hop, values in selection.coverage_by_hop.items():
        lines.append(
            f"| {hop} | {values['eligible_queries']} | {values['fully_covered_queries']} | "
            f"{float(values['full_coverage_rate']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Selected tasks",
            "",
            "| label | task ID | type | bytes | lexical tokens | docs A4/A5/A28 | support sites (offline only) |",
            "|---|---|---|---:|---:|---|---|",
        ]
    )
    for label, candidate in selection.selected.items():
        counts = candidate.documents_per_agent
        lines.append(
            f"| {label} | `{candidate.query.task_id}` | `{candidate.query.question_type}` | "
            f"{candidate.bundle_bytes} | {candidate.bundle_tokens} | "
            f"{counts['A4']}/{counts['A5']}/{counts['A28']} | "
            f"{', '.join(candidate.support_sites)} |"
        )
        lines.extend(["", f"**{label} query:** {candidate.query.query}", ""])
    lines.extend(
        [
            "## Selection rule",
            "",
            "For each required support-document count, retain only non-null queries whose fixed",
            "Top-N covers every official supporting document and whose support maps to at least",
            "two shards. Exclude obvious single-fact answer exposures using the preregistered",
            "multi-hop gate above. Prefer comparison/temporal structure, then number of support",
            "shards, candidate bundle bytes, and stable task ID. This deterministic rule does not",
            "alter retrieval results or add gold documents.",
            "",
        ]
    )
    return "\n".join(lines)


def materialize_multihop_rag(
    adapter: MultiHopRAGAdapter,
    selection: MultiHopRAGSelection,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write selected task artifacts plus physically separate evaluator records."""

    root = Path(output_dir)
    selection_dir = root / "task_selection"
    evaluator_dir = root / "evaluator_only"
    multihop_dir = root / "multihop_rag"
    for directory in (selection_dir, evaluator_dir, multihop_dir):
        directory.mkdir(parents=True, exist_ok=True)

    document_by_id = {item.document_id: item for item in adapter.documents}
    visible_tasks: list[ScopeTask] = []
    task_bank_records: list[ScopeTaskBankRecord] = []
    evaluator_records: list[ScopeEvaluatorRecord] = []
    selected_ids = {item.query.task_id for item in selection.selected.values()}

    for label, candidate in selection.selected.items():
        task_dir = multihop_dir / f"task_{label.casefold()}"
        artifact_ids: list[str] = []
        artifact_sizes: dict[str, int] = {}
        artifact_tokens: dict[str, int] = {}
        artifact_types: dict[str, Literal["document"]] = {}
        artifact_source_refs: dict[str, str] = {}
        artifact_document_ids: dict[str, str] = {}
        artifact_placement: dict[str, ScopeSiteId] = {}
        for ranked in candidate.ranked_documents:
            document = document_by_id[ranked.document_id]
            artifact_id = ranked.document_id
            site = stable_document_site(document.document_id)
            artifact_path = task_dir / "artifacts" / site / f"{artifact_id}.json"
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_bytes(document.payload)
            artifact_ids.append(artifact_id)
            artifact_sizes[artifact_id] = document.size_bytes
            artifact_tokens[artifact_id] = len(document.tokens)
            artifact_types[artifact_id] = "document"
            artifact_source_refs[artifact_id] = _source_ref(label, site, artifact_id)
            artifact_document_ids[artifact_id] = document.document_id
            artifact_placement[artifact_id] = site

        task = ScopeTask(
            task_id=candidate.query.task_id,
            dataset=DATASET_NAME,
            instruction=(
                "Answer the query using the candidate documents. Return a concise answer."
            ),
            query=candidate.query.query,
            artifact_ids=artifact_ids,
            artifact_types=artifact_types,
            artifact_sizes=artifact_sizes,
            artifact_tokens=artifact_tokens,
            artifact_source_refs=artifact_source_refs,
            artifact_document_ids=artifact_document_ids,
            artifact_placement=artifact_placement,
            retrieval_config=ScopeRetrievalConfig(
                top_n=cast(Literal[30, 50], selection.top_n),
                parameters={
                    "k1": 1.5,
                    "b": 0.75,
                    "runtime_local_top_k": selection.local_top_k,
                    "tokenizer": "lowercase_ascii_alnum_regex",
                },
            ),
            artifact_count=len(artifact_ids),
            evaluator_type="multihop_rag_official_token_intersection",
            metadata={
                "selection_label": label,
                "source": {
                    "repository": DATASET_REPOSITORY,
                    "revision": adapter.dataset_revision,
                    "split": "train",
                    "row_ordinal": candidate.query.source_ordinal,
                },
                "retrieval": {
                    "algorithm": "bm25_document_level_v1",
                    "corpus_scope": "full_609_document_corpus",
                    "candidate_top_n": selection.top_n,
                    "runtime_local_top_k": selection.local_top_k,
                    "k1": 1.5,
                    "b": 0.75,
                    "tokenizer": "lowercase_ascii_alnum_regex",
                },
                "candidate_bundle_bytes": candidate.bundle_bytes,
                "candidate_bundle_tokens_lexical": candidate.bundle_tokens,
                "question_type": candidate.query.question_type,
            },
        )
        evaluator = ScopeEvaluatorRecord(
            task_id=candidate.query.task_id,
            evaluator_type="multihop_rag_official_token_intersection",
            answer=candidate.query.answer,
            acceptable_answers=[],
            supporting_document_ids=list(candidate.query.supporting_document_ids),
            evidence_document_count=candidate.query.evidence_document_count,
            original_evaluator_information={
                "repository": REFERENCE_REPOSITORY,
                "revision": REFERENCE_REVISION,
                "file": "qa_evaluate.py",
                "extract_rule": 'The answer to the question is "..." or full response',
                "score_rule": "case-insensitive whitespace-token intersection",
                "question_type": candidate.query.question_type,
                "source_evidence_list": list(candidate.query.evidence_list),
            },
        )
        visible_tasks.append(task)
        task_bank_records.append(
            ScopeTaskBankRecord(
                task_id=task.task_id,
                planner_visible=task,
                evaluator_only=ScopeTaskBankEvaluatorSummary(
                    evidence_document_count=candidate.query.evidence_document_count,
                    evaluator_type="multihop_rag_official_token_intersection",
                ),
            )
        )
        evaluator_records.append(evaluator)
        (task_dir / "task.json").write_text(
            json.dumps(task.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    candidate_output = {
        "schema_version": "scope-expansion-v0-multihop-selection-v1",
        "offline_evaluator_only": True,
        "source": {
            "dataset_repository": DATASET_REPOSITORY,
            "dataset_revision": adapter.dataset_revision,
            "corpus_sha256": adapter.corpus_sha256,
            "queries_sha256": adapter.queries_sha256,
            "reference_repository": REFERENCE_REPOSITORY,
            "reference_revision": REFERENCE_REVISION,
        },
        "retrieval": {
            "algorithm": "bm25_document_level_v1",
            "corpus_document_count": len(adapter.documents),
            "top_n": selection.top_n,
            "fallback_top_n": FALLBACK_TOP_N,
            "fallback_used": selection.top_n == FALLBACK_TOP_N,
            "runtime_local_top_k": selection.local_top_k,
        },
        "coverage_by_hop": selection.coverage_by_hop,
        "selected_task_ids": sorted(selected_ids),
        "candidates": [_candidate_record(item) for item in selection.candidates],
    }
    (selection_dir / "multihop_rag_candidates.json").write_text(
        json.dumps(candidate_output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (selection_dir / "multihop_rag_selection.md").write_text(
        _markdown_report(adapter, selection), encoding="utf-8"
    )
    (root / "task_bank.jsonl").write_text(
        "".join(
            json.dumps(item.model_dump(mode="json"), ensure_ascii=False) + "\n"
            for item in task_bank_records
        ),
        encoding="utf-8",
    )
    (evaluator_dir / "multihop_rag.jsonl").write_text(
        "".join(
            json.dumps(item.model_dump(mode="json"), ensure_ascii=False) + "\n"
            for item in evaluator_records
        ),
        encoding="utf-8",
    )
    return {
        "output_dir": str(root),
        "task_count": len(visible_tasks),
        "artifact_count": sum(item.artifact_count for item in visible_tasks),
        "selected_task_ids": {
            label: item.query.task_id for label, item in selection.selected.items()
        },
        "coverage_by_hop": selection.coverage_by_hop,
    }
