"""Offline adapters for the first multi-document scenario-mining datasets."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote

from infra_bench.schemas import (
    ExternalEvaluatorSpec,
    InitialArtifactSpec,
    ObservationSpec,
    RuntimeVerifierSpec,
    ScenarioArtifact,
    ScenarioEvaluator,
    ScenarioEvidence,
    ScenarioRecord,
    TaskInteractionSpec,
)

from .features import compute_scene_features, utf8_size


class ScenarioRowSkipped(ValueError):
    """A well-formed row outside the configured scenario subset."""


def iter_json_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Read JSON/JSONL, or optional PyArrow Parquet, without model services."""

    source = Path(path).expanduser()
    if source.suffix.lower() == ".parquet":
        try:
            parquet = import_module("pyarrow.parquet")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Parquet input requires `uv run --extra scenario-mining ...`"
            ) from exc
        parquet_file = parquet.ParquetFile(source)
        for batch in parquet_file.iter_batches():
            for item in batch.to_pylist():
                if not isinstance(item, dict):
                    raise ValueError(f"expected Parquet object row in {source}")
                yield cast(dict[str, Any], item)
        return
    if source.suffix.lower() == ".jsonl":
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = cast(object, json.loads(line))
                if not isinstance(value, dict):
                    raise ValueError(f"expected object at {source}:{line_number}")
                yield cast(dict[str, Any], value)
        return
    value = cast(object, json.loads(source.read_text(encoding="utf-8")))
    if not isinstance(value, list):
        raise ValueError(f"expected a JSON array in {source}")
    for index, item in enumerate(cast(list[object], value)):
        if not isinstance(item, dict):
            raise ValueError(f"expected object at {source}[{index}]")
        yield cast(dict[str, Any], item)


def _task_interaction(
    *,
    task_id: str,
    objective: str,
    artifacts: Sequence[ScenarioArtifact],
    evaluator_id: str,
) -> TaskInteractionSpec:
    return TaskInteractionSpec(
        task_id=task_id,
        objective=objective,
        initial_artifacts=[
            InitialArtifactSpec(
                artifact_id=item.artifact_id,
                kind=item.type,
                source_ref=cast(str, item.source_ref),
            )
            for item in artifacts
        ],
        operators=["invoke_model"],
        observations=[
            ObservationSpec(
                observation_id="model_result",
                produced_by=["invoke_model"],
                description="A model-produced text artifact derived from declared inputs.",
            )
        ],
        runtime_verifier=RuntimeVerifierSpec(level="none"),
        external_evaluator=ExternalEvaluatorSpec(evaluator_id=evaluator_id),
    )


def _source_ref(
    dataset: str, revision: str, split: str, raw_id: str, artifact_index: int
) -> str:
    identity = "/".join(quote(item, safe="") for item in (revision, split, raw_id))
    return f"dataset://{dataset}/{identity}#artifact={artifact_index}"


def _artifact(
    *,
    artifact_id: str,
    artifact_type: str,
    text: str,
    source_ref: str,
    embed_content: bool,
    metadata: Mapping[str, Any] | None = None,
) -> ScenarioArtifact:
    return ScenarioArtifact(
        artifact_id=artifact_id,
        type=artifact_type,
        content=text if embed_content else None,
        source_ref=source_ref,
        size_bytes=utf8_size(text),
        metadata=dict(metadata or {}),
    )


class OfflineScenarioAdapter:
    dataset: str

    def __init__(
        self,
        *,
        revision: str,
        split: str,
        embed_content: bool = False,
    ) -> None:
        self.revision = revision
        self.split = split
        self.embed_content = embed_content

    def scenario_from_row(
        self, row: Mapping[str, Any], *, row_ordinal: int
    ) -> ScenarioRecord:
        raise NotImplementedError

    def ingest(
        self, rows: Iterable[Mapping[str, Any]], *, limit: int | None = None
    ) -> Iterator[ScenarioRecord]:
        emitted = 0
        for ordinal, row in enumerate(rows):
            try:
                scenario = self.scenario_from_row(row, row_ordinal=ordinal)
            except ScenarioRowSkipped:
                continue
            yield scenario
            emitted += 1
            if limit is not None and emitted >= limit:
                break


def _two_wiki_context(value: object) -> list[tuple[str, list[str]]]:
    if isinstance(value, dict):
        mapping = cast(Mapping[str, object], value)
        titles = mapping.get("title")
        sentences = mapping.get("sentences", mapping.get("sentence"))
        if isinstance(titles, list) and isinstance(sentences, list):
            title_values = cast(list[object], titles)
            sentence_values = cast(list[object], sentences)
            return [
                (str(title), [str(sentence) for sentence in cast(list[object], paragraph)])
                for title, paragraph in zip(
                    title_values, sentence_values, strict=True
                )
                if isinstance(paragraph, list)
            ]
    if isinstance(value, list):
        contexts: list[tuple[str, list[str]]] = []
        for item in cast(list[object], value):
            if not isinstance(item, list | tuple):
                raise ValueError("invalid 2Wiki context entry")
            pair = cast(Sequence[object], item)
            if len(pair) != 2:
                raise ValueError("invalid 2Wiki context entry")
            title, sentences = pair
            if not isinstance(sentences, list):
                raise ValueError("invalid 2Wiki sentence list")
            sentence_values = cast(list[object], sentences)
            contexts.append((str(title), [str(sentence) for sentence in sentence_values]))
        return contexts
    raise ValueError("invalid 2Wiki context")


def _two_wiki_supporting_facts(value: object) -> list[tuple[str, int]] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        mapping = cast(Mapping[str, object], value)
        titles = mapping.get("title")
        sentence_ids = mapping.get("sent_id", mapping.get("sentence_id"))
        if isinstance(titles, list) and isinstance(sentence_ids, list):
            title_values = cast(list[object], titles)
            sentence_values = cast(list[object], sentence_ids)
            return [
                (str(title), int(str(sentence_id)))
                for title, sentence_id in zip(
                    title_values, sentence_values, strict=True
                )
            ]
    if isinstance(value, list):
        facts: list[tuple[str, int]] = []
        for item in cast(list[object], value):
            if not isinstance(item, list | tuple):
                return None
            pair = cast(Sequence[object], item)
            if len(pair) != 2:
                return None
            facts.append((str(pair[0]), int(str(pair[1]))))
        return facts
    return None


class TwoWikiMultiHopQAAdapter(OfflineScenarioAdapter):
    dataset = "2wikimultihopqa"

    def __init__(
        self,
        *,
        revision: str,
        split: str,
        variant: str = "official-v1.0",
        embed_content: bool = False,
    ) -> None:
        super().__init__(revision=revision, split=split, embed_content=embed_content)
        self.variant = variant

    def scenario_from_row(
        self, row: Mapping[str, Any], *, row_ordinal: int
    ) -> ScenarioRecord:
        raw_id = str(row.get("_id", row.get("id", "")))
        question = str(row.get("question", ""))
        answer = row.get("answer")
        if not raw_id or not question or answer is None:
            raise ValueError("2Wiki mining requires id, question, and public answer")
        contexts = _two_wiki_context(row.get("context"))
        artifacts: list[ScenarioArtifact] = []
        title_indices: dict[str, list[int]] = {}
        for index, (title, sentences) in enumerate(contexts):
            text = title + "\n" + "".join(sentences)
            artifacts.append(
                _artifact(
                    artifact_id=f"doc-{index:04d}",
                    artifact_type="document",
                    text=text,
                    source_ref=_source_ref(
                        self.dataset, self.revision, self.split, raw_id, index
                    ),
                    embed_content=self.embed_content,
                    metadata={"title": title, "context_ordinal": index},
                )
            )
            title_indices.setdefault(title, []).append(index)

        support = _two_wiki_supporting_facts(row.get("supporting_facts"))
        resolved_evidence: list[ScenarioEvidence] = []
        evidence: list[ScenarioEvidence] | None = (
            resolved_evidence if support is not None else None
        )
        evidence_status = "known" if support is not None else "hidden_by_split"
        if support is not None:
            for title, sentence_id in support:
                indices = title_indices.get(title, [])
                if (
                    len(indices) != 1
                    or sentence_id < 0
                    or sentence_id >= len(contexts[indices[0]][1])
                ):
                    evidence = None
                    evidence_status = "not_applicable_or_incomplete"
                    break
                artifact_index = indices[0]
                resolved_evidence.append(
                    ScenarioEvidence(
                        artifact_id=artifacts[artifact_index].artifact_id,
                        span=f"sentence:{sentence_id}",
                        supporting_fact=contexts[artifact_index][1][sentence_id],
                        metadata={"title": title},
                    )
                )
        evaluator = ScenarioEvaluator(
            type="exact_match", target=str(answer), deterministic=True
        )
        task_id = f"2wikimultihopqa:{self.variant}:{self.split}:{raw_id}"
        features = compute_scene_features(artifacts, evidence, evaluator)
        interaction = _task_interaction(
            task_id=task_id,
            objective=question,
            artifacts=artifacts,
            evaluator_id="2wikimultihopqa_official",
        )
        return ScenarioRecord(
            scenario_id=task_id,
            dataset=self.dataset,
            task_id=task_id,
            question=question,
            answer=str(answer),
            artifacts=artifacts,
            evidence=evidence,
            evidence_status=cast(Any, evidence_status),
            evaluator=evaluator,
            scene_features=features,
            interaction_spec=interaction,
            metadata={
                "source_identity": {
                    "dataset": self.dataset,
                    "revision": self.revision,
                    "variant": self.variant,
                    "split": self.split,
                    "raw_id": raw_id,
                },
                "source_row_ordinal": row_ordinal,
                "question_type": row.get("type"),
                "evidence_granularity": "sentence",
                "evidence_path": row.get("evidences"),
                "dataset_license": "Apache-2.0 repository; third-party context rights vary",
            },
        )


class MuSiQueAdapter(OfflineScenarioAdapter):
    dataset = "musique"

    def __init__(
        self,
        *,
        revision: str,
        split: str,
        variant: str = "ans-v1.0",
        embed_content: bool = False,
    ) -> None:
        super().__init__(revision=revision, split=split, embed_content=embed_content)
        self.variant = variant

    def scenario_from_row(
        self, row: Mapping[str, Any], *, row_ordinal: int
    ) -> ScenarioRecord:
        raw_id = str(row.get("id", ""))
        question = str(row.get("question", ""))
        answer = row.get("answer")
        paragraphs_value = row.get("paragraphs")
        if not raw_id or not question or answer is None or not isinstance(paragraphs_value, list):
            raise ValueError("MuSiQue mining requires id, question, paragraphs, and answer")
        paragraphs = [
            cast(Mapping[str, Any], item)
            for item in cast(list[object], paragraphs_value)
            if isinstance(item, dict)
        ]
        indices = [int(item["idx"]) for item in paragraphs]
        if len(indices) != len(paragraphs) or sorted(indices) != list(range(len(indices))):
            raise ValueError("MuSiQue paragraph idx values must be unique and contiguous")
        paragraphs.sort(key=lambda item: int(item["idx"]))
        identity_payload = json.dumps(
            [question, [(item.get("title"), item.get("paragraph_text")) for item in paragraphs]],
            sort_keys=True,
            ensure_ascii=False,
        )
        suffix = raw_id
        if self.variant.startswith("full"):
            suffix = f"{raw_id}:{sha256(identity_payload.encode()).hexdigest()[:16]}"
        task_id = f"musique:{self.variant}:{self.split}:{suffix}"
        artifacts = [
            _artifact(
                artifact_id=f"paragraph-{int(item['idx']):04d}",
                artifact_type="document",
                text=f"{item.get('title', '')}\n{item.get('paragraph_text', '')}",
                source_ref=_source_ref(
                    self.dataset,
                    self.revision,
                    self.split,
                    suffix,
                    int(item["idx"]),
                ),
                embed_content=self.embed_content,
                metadata={"title": item.get("title"), "paragraph_idx": int(item["idx"])},
            )
            for item in paragraphs
        ]
        answerable = row.get("answerable")
        support_known = answerable is True and all(
            isinstance(item.get("is_supporting"), bool) for item in paragraphs
        )
        if support_known:
            supporting = [item for item in paragraphs if item["is_supporting"]]
            evidence = [
                ScenarioEvidence(
                    artifact_id=f"paragraph-{int(item['idx']):04d}",
                    span="paragraph",
                    supporting_fact=str(item.get("paragraph_text", "")),
                    metadata={"title": item.get("title")},
                )
                for item in supporting
            ]
            if not evidence:
                evidence = None
                evidence_status = "not_applicable_or_incomplete"
            else:
                evidence_status = "known"
        else:
            evidence = None
            evidence_status = (
                "hidden_by_split"
                if answerable is None
                else "not_applicable_or_incomplete"
            )
        aliases = [str(item) for item in row.get("answer_aliases", [])]
        targets = list(dict.fromkeys([str(answer), *aliases]))
        evaluator = ScenarioEvaluator(
            type="exact_match", target=targets, deterministic=True
        )
        features = compute_scene_features(artifacts, evidence, evaluator)
        interaction = _task_interaction(
            task_id=task_id,
            objective=question,
            artifacts=artifacts,
            evaluator_id="musique_v1_official",
        )
        return ScenarioRecord(
            scenario_id=task_id,
            dataset=self.dataset,
            task_id=task_id,
            question=question,
            answer=targets,
            artifacts=artifacts,
            evidence=evidence,
            evidence_status=cast(Any, evidence_status),
            evaluator=evaluator,
            scene_features=features,
            interaction_spec=interaction,
            metadata={
                "source_identity": {
                    "dataset": self.dataset,
                    "revision": self.revision,
                    "variant": self.variant,
                    "split": self.split,
                    "raw_id": raw_id,
                },
                "source_row_ordinal": row_ordinal,
                "answerable": answerable,
                "evidence_granularity": "paragraph",
                "question_decomposition": row.get("question_decomposition"),
                "dataset_license": "CC BY 4.0",
            },
        )


_LONGBENCH_SUBDOMAINS = {
    "Multi-Document QA": {
        "Academic",
        "Legal",
        "Financial",
        "Governmental",
        "Multi-news",
    },
    "Code Repository Understanding": {"Code repo QA"},
    "Long Structured Data Understanding": {"Table QA", "Knowledge graph reasoning"},
}


class LongBenchV2Adapter(OfflineScenarioAdapter):
    dataset = "longbench-v2"

    def scenario_from_row(
        self, row: Mapping[str, Any], *, row_ordinal: int
    ) -> ScenarioRecord:
        domain = str(row.get("domain", ""))
        subdomain = str(row.get("sub_domain", ""))
        if subdomain not in _LONGBENCH_SUBDOMAINS.get(domain, set()):
            raise ScenarioRowSkipped(f"LongBench category not selected: {domain}/{subdomain}")
        raw_id = str(row.get("_id", ""))
        raw_question = str(row.get("question", ""))
        context = row.get("context")
        answer = str(row.get("answer", ""))
        choices = {letter: str(row.get(f"choice_{letter}", "")) for letter in "ABCD"}
        if (
            not raw_id
            or not raw_question
            or not isinstance(context, str)
            or answer not in choices
            or any(not item for item in choices.values())
        ):
            raise ValueError("invalid LongBench-v2 selected row")
        visible_question = "\n".join(
            [raw_question, *(f"{letter}. {choices[letter]}" for letter in "ABCD")]
        )
        task_id = f"longbench-v2:{self.revision}:{raw_id}"
        if domain == "Code Repository Understanding":
            artifact_type = "opaque_code_context"
        elif domain == "Long Structured Data Understanding":
            artifact_type = "opaque_structured_context"
        else:
            artifact_type = "opaque_document_context"
        artifacts = [
            _artifact(
                artifact_id="context-0000",
                artifact_type=artifact_type,
                text=context,
                source_ref=_source_ref(
                    self.dataset, self.revision, self.split, raw_id, 0
                ),
                embed_content=self.embed_content,
                metadata={
                    "official_boundary": "single opaque context string",
                    "derived_chunks": False,
                },
            )
        ]
        evaluator = ScenarioEvaluator(
            type="multiple_choice", target=answer, deterministic=True
        )
        evidence = None
        features = compute_scene_features(artifacts, evidence, evaluator)
        interaction = _task_interaction(
            task_id=task_id,
            objective=visible_question,
            artifacts=artifacts,
            evaluator_id="longbench_v2_official_mc",
        )
        return ScenarioRecord(
            scenario_id=task_id,
            dataset=self.dataset,
            task_id=task_id,
            question=visible_question,
            answer=answer,
            artifacts=artifacts,
            evidence=None,
            evidence_status="absent_in_public_release",
            evaluator=evaluator,
            scene_features=features,
            interaction_spec=interaction,
            metadata={
                "source_identity": {
                    "dataset": self.dataset,
                    "revision": self.revision,
                    "variant": "official-public",
                    "split": self.split,
                    "raw_id": raw_id,
                },
                "source_row_ordinal": row_ordinal,
                "domain": domain,
                "sub_domain": subdomain,
                "difficulty": row.get("difficulty"),
                "length_category": row.get("length"),
                "original_question": raw_question,
                "choices": choices,
                "evidence_granularity": None,
                "dataset_license": "Apache-2.0 data card; third-party context rights vary",
                "evaluator_code_license": "MIT",
            },
        )
