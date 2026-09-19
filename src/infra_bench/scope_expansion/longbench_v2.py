"""LongBench-v2 selection and natural-boundary materialization.

This module is deliberately offline-only.  It recovers source document or row
boundaries, creates deterministic A4/A5/A28 placements, and writes gold labels
to a separate evaluator file.  Runtime operators consume only the generic task
configuration and artifact payloads.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

from infra_bench.io import write_json, write_jsonl
from infra_bench.schemas.scope_expansion_v0 import (
    SITE_BY_SHARD,
    ScopeEvaluatorRecord,
    ScopeRetrievalConfig,
    ScopeSiteId,
    ScopeStructuredPlan,
    ScopeTask,
    ScopeTaskBankEvaluatorSummary,
    ScopeTaskBankRecord,
    stable_document_site,
)
from infra_bench.scope_expansion.multihop_rag import tokenize

DATASET_NAME = "LongBench-v2"
DATASET_REPOSITORY = "zai-org/LongBench-v2"
DATASET_REVISION = "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"
DATASET_SHA256 = "15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2"
REFERENCE_REPOSITORY = "THUDM/LongBench"
REFERENCE_REVISION = "2e00731f8d0bff23dc4325161044d0ed8af94c1e"

MULTIDOC_SOURCE_ID = "66f7c780bb02136c067c35e8"
STRUCTURED_SOURCE_ID = "66f3ac0b821e116aacb2e203"
MULTIDOC_TASK_ID = f"longbench-v2-{MULTIDOC_SOURCE_ID}"
STRUCTURED_TASK_ID = f"longbench-v2-{STRUCTURED_SOURCE_ID}"
WORD_PATTERN = re.compile(r"\S+")
STRUCTURED_TEXT_COLUMNS = frozenset({"Symbol", "EndDate", "IndustryCode"})


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest_id(prefix: str, value: object) -> str:
    digest = sha256(_canonical_json(value).encode("utf-8")).hexdigest()[:24]
    return f"{prefix}-{digest}"


@dataclass(frozen=True)
class LongBenchV2Row:
    source_ordinal: int
    source_id: str
    domain: str
    sub_domain: str
    difficulty: str
    length: str
    question: str
    choices: dict[str, str]
    answer: str
    context: str


@dataclass(frozen=True)
class NaturalDocument:
    document_id: str
    title: str
    natural_boundary: str
    text: str
    payload: bytes


@dataclass(frozen=True)
class StructuredShard:
    shard_id: str
    site: ScopeSiteId
    record_count: int
    payload: bytes


@dataclass(frozen=True)
class LongBenchV2Selection:
    multidoc: LongBenchV2Row
    documents: tuple[NaturalDocument, ...]
    structured: LongBenchV2Row
    structured_columns: tuple[str, ...]
    structured_shards: tuple[StructuredShard, ...]


class LongBenchV2Adapter:
    """Load the official JSON array without modifying its concatenated contexts."""

    def __init__(
        self,
        *,
        rows: tuple[LongBenchV2Row, ...],
        source_sha256: str,
        dataset_revision: str = DATASET_REVISION,
    ) -> None:
        self.rows = rows
        self.source_sha256 = source_sha256
        self.dataset_revision = dataset_revision

    @classmethod
    def load(
        cls,
        source_path: str | Path,
        *,
        dataset_revision: str = DATASET_REVISION,
        verify_official_checksum: bool = True,
    ) -> LongBenchV2Adapter:
        source = Path(source_path)
        source_bytes = source.read_bytes()
        source_sha256 = sha256(source_bytes).hexdigest()
        if verify_official_checksum and source_sha256 != DATASET_SHA256:
            raise ValueError(f"LongBench-v2 checksum mismatch: {source_sha256} != {DATASET_SHA256}")
        value = cast(object, json.loads(source_bytes))
        if not isinstance(value, list):
            raise ValueError("LongBench-v2 data.json must be a JSON array")
        rows: list[LongBenchV2Row] = []
        seen: set[str] = set()
        for ordinal, item in enumerate(cast(list[object], value)):
            if not isinstance(item, dict):
                raise ValueError(f"LongBench-v2 row {ordinal} is not an object")
            raw = cast(dict[str, Any], item)
            source_id = str(raw.get("_id") or "").strip()
            if not source_id or source_id in seen:
                raise ValueError(f"invalid or duplicate LongBench-v2 ID at row {ordinal}")
            seen.add(source_id)
            choices = {letter: str(raw.get(f"choice_{letter}") or "").strip() for letter in "ABCD"}
            answer = str(raw.get("answer") or "").strip()
            context = str(raw.get("context") or "")
            question = str(raw.get("question") or "").strip()
            if any(not choice for choice in choices.values()) or answer not in choices:
                raise ValueError(f"invalid choices or answer for LongBench-v2 row {source_id}")
            if not context or not question:
                raise ValueError(f"blank question or context for LongBench-v2 row {source_id}")
            rows.append(
                LongBenchV2Row(
                    source_ordinal=ordinal,
                    source_id=source_id,
                    domain=str(raw.get("domain") or ""),
                    sub_domain=str(raw.get("sub_domain") or ""),
                    difficulty=str(raw.get("difficulty") or ""),
                    length=str(raw.get("length") or ""),
                    question=question,
                    choices=choices,
                    answer=answer,
                    context=context,
                )
            )
        return cls(
            rows=tuple(rows),
            source_sha256=source_sha256,
            dataset_revision=dataset_revision,
        )

    def by_id(self, source_id: str) -> LongBenchV2Row:
        matches = [row for row in self.rows if row.source_id == source_id]
        if len(matches) != 1:
            raise ValueError(f"expected one LongBench-v2 row {source_id}, found {len(matches)}")
        return matches[0]


def _render_document(title: str, text: str) -> bytes:
    return (_canonical_json({"body": text, "title": title}) + "\n").encode("utf-8")


def recover_jd_esg_report_documents(row: LongBenchV2Row) -> tuple[NaturalDocument, ...]:
    """Recover four complete source PDFs from audited report transition blocks."""

    if row.source_id != MULTIDOC_SOURCE_ID:
        raise ValueError("the registered boundary rule only applies to the selected source row")
    lines = row.context.splitlines(keepends=True)
    if len(lines) <= 17_223:
        raise ValueError("selected context is shorter than its audited report boundaries")
    normalized = [line.rstrip("\r\n") for line in lines]
    if normalized[1] != "2020 JD.com Environmental, Social and Governance Report":
        raise ValueError("missing audited 2020 report cover")
    if normalized[3277] != "2021 JD.com Environmental, Social and Governance Report":
        raise ValueError("missing audited 2021 report cover")
    if not (
        normalized[9091] == "About This Report"
        and normalized[9153] == "2023 JD.com, Inc."
        and normalized[9154] == "Environmental, Social and Governance Report"
    ):
        raise ValueError("missing audited 2023 report start/self-identification block")
    if tuple(normalized[17721:17724]) != (
        "2022",
        "Environmental, Social and",
        "Governance Report",
    ):
        raise ValueError("missing audited 2022 report cover")
    starts = (0, 3277, 9091, 17721)
    texts = tuple(
        "".join(lines[start:end])
        for start, end in zip(starts, (*starts[1:], len(lines)), strict=True)
    )
    titles = (
        "2020 JD.com Environmental, Social and Governance Report",
        "2021 JD.com Environmental, Social and Governance Report",
        "2023 JD.com Environmental, Social and Governance Report",
        "2022 JD.com Environmental, Social and Governance Report",
    )
    if not all(text.strip() for text in texts):
        raise ValueError("natural document recovery produced an empty document")
    documents: list[NaturalDocument] = []
    for ordinal, (title, text) in enumerate(zip(titles, texts, strict=True)):
        identity = {
            "source_id": row.source_id,
            "natural_boundary_ordinal": ordinal,
            "title": title,
        }
        documents.append(
            NaturalDocument(
                document_id=_digest_id("lb2-doc", identity),
                title=title,
                natural_boundary=(
                    "source context start"
                    if ordinal == 0
                    else {
                        1: "explicit 2021 report cover",
                        2: "post-2021 publication break and 2023 report identity block",
                        3: "explicit 2022 report cover",
                    }[ordinal]
                ),
                text=text,
                payload=_render_document(title, text),
            )
        )
    return tuple(documents)


def _stable_record_id(source_id: str, ordinal: int, values: dict[str, str]) -> str:
    return _digest_id(
        "lb2-record",
        {
            "source_id": source_id,
            "natural_row_ordinal": ordinal,
            "symbol": values.get("Symbol"),
            "end_date": values.get("EndDate"),
        },
    )


def _parse_structured_numeric(value: str, *, column: str, ordinal: int) -> int | float | None:
    """Recover a finite JSON number; source missing/non-finite values become null."""
    normalized = value.strip()
    if not normalized or normalized.casefold() in {"nan", "na", "n/a", "null", "none"}:
        return None
    try:
        if re.fullmatch(r"[+-]?\d+", normalized):
            return int(normalized)
        parsed = float(normalized)
    except ValueError as exc:
        raise ValueError(
            f"natural row {ordinal} column {column} is not numeric: {value!r}"
        ) from exc
    return parsed if math.isfinite(parsed) else None


def _stable_shard_id(source_id: str, site: ScopeSiteId) -> str:
    # The nonce is chosen only from source ID + target site.  It cannot depend on
    # question answers or record contents; SHA-256 remains the placement rule.
    for nonce in range(1000):
        shard_id = _digest_id(
            "lb2-record-shard", {"source_id": source_id, "site": site, "nonce": nonce}
        )
        if stable_document_site(shard_id) == site:
            return shard_id
    raise RuntimeError(f"could not derive deterministic shard ID for {site}")


def recover_fixed_width_record_shards(
    row: LongBenchV2Row,
) -> tuple[tuple[str, ...], tuple[StructuredShard, ...]]:
    """Parse natural fixed-width rows and hash each intact row to one site."""

    if row.source_id != STRUCTURED_SOURCE_ID:
        raise ValueError("the registered row parser only applies to the selected source row")
    lines = row.context.splitlines()
    if not lines:
        raise ValueError("structured context is empty")
    columns = tuple(lines[0].split())
    expected = (
        "Symbol",
        "EndDate",
        "IndustryCode",
        "TotalAssets",
        "TotalLiability",
        "IntangibleAsset",
        "NetProfit",
        "OperatingEvenue",
        "OperatingCost",
        "OperationProfit",
    )
    if columns != expected:
        raise ValueError(f"unexpected fixed-width schema: {columns}")
    records_by_site: dict[ScopeSiteId, list[dict[str, Any]]] = {site: [] for site in SITE_BY_SHARD}
    for ordinal, line in enumerate(lines[1:], 1):
        values = line.split()
        if len(values) != len(columns):
            raise ValueError(f"natural row {ordinal} has {len(values)} fields")
        source_record = dict(zip(columns, values, strict=True))
        record_id = _stable_record_id(row.source_id, ordinal, source_record)
        site = stable_document_site(record_id)
        record: dict[str, Any] = {
            column: (
                value
                if column in STRUCTURED_TEXT_COLUMNS
                else _parse_structured_numeric(value, column=column, ordinal=ordinal)
            )
            for column, value in source_record.items()
        }
        records_by_site[site].append({"record_id": record_id, **record})
    if any(not records for records in records_by_site.values()):
        raise ValueError("stable row placement must populate A4, A5, and A28")
    shards: list[StructuredShard] = []
    for site in SITE_BY_SHARD:
        payload = (_canonical_json({"records": records_by_site[site]}) + "\n").encode("utf-8")
        shards.append(
            StructuredShard(
                shard_id=_stable_shard_id(row.source_id, site),
                site=site,
                record_count=len(records_by_site[site]),
                payload=payload,
            )
        )
    return columns, tuple(shards)


def select_longbench_v2_tasks(adapter: LongBenchV2Adapter) -> LongBenchV2Selection:
    """Apply the preregistered structural gates to exactly two official rows."""

    multidoc = adapter.by_id(MULTIDOC_SOURCE_ID)
    if multidoc.domain != "Multi-Document QA" or multidoc.length != "long":
        raise ValueError("selected multi-document row no longer has its audited category/length")
    word_count = len(WORD_PATTERN.findall(multidoc.context))
    if word_count < 100_000:
        raise ValueError(f"multi-document source is below 100K words: {word_count}")
    documents = recover_jd_esg_report_documents(multidoc)
    if len({stable_document_site(item.document_id) for item in documents}) < 2:
        raise ValueError("selected natural documents are not distributed across sites")

    structured = adapter.by_id(STRUCTURED_SOURCE_ID)
    if (
        structured.domain != "Long Structured Data Understanding"
        or structured.sub_domain != "Table QA"
        or structured.length != "long"
    ):
        raise ValueError("selected structured row no longer has its audited category/length")
    columns, shards = recover_fixed_width_record_shards(structured)
    return LongBenchV2Selection(
        multidoc=multidoc,
        documents=documents,
        structured=structured,
        structured_columns=columns,
        structured_shards=shards,
    )


def _source_ref(family: str, site: str, artifact_id: str, suffix: str) -> str:
    return f"scope-expansion-v0://{family}/{site}/{artifact_id}.{suffix}"


def _visible_choices(row: LongBenchV2Row) -> dict[str, str]:
    return {letter: _runtime_text(value) for letter, value in row.choices.items()}


def _runtime_text(value: str) -> str:
    """Normalize compatibility whitespace/punctuation without guessing encodings."""
    return unicodedata.normalize("NFKC", value)


def _multidoc_task(selection: LongBenchV2Selection) -> ScopeTask:
    artifacts = {
        _digest_id("lb2-artifact", {"document_id": document.document_id}): document
        for document in selection.documents
    }
    placement = {
        artifact_id: stable_document_site(document.document_id)
        for artifact_id, document in artifacts.items()
    }
    return ScopeTask(
        task_id=MULTIDOC_TASK_ID,
        dataset=DATASET_NAME,
        task_family="multi_document_qa",
        instruction=(
            "Answer the multiple-choice question from the complete natural-document bundle. "
            "Return the selected choice letter A, B, C, or D."
        ),
        query=_runtime_text(selection.multidoc.question),
        answer_choices=_visible_choices(selection.multidoc),
        artifact_ids=list(artifacts),
        artifact_types={artifact_id: "document" for artifact_id in artifacts},
        artifact_sizes={
            artifact_id: len(document.payload) for artifact_id, document in artifacts.items()
        },
        artifact_tokens={
            artifact_id: len(tokenize(document.text)) for artifact_id, document in artifacts.items()
        },
        artifact_source_refs={
            artifact_id: _source_ref(
                "longbench-multidoc", placement[artifact_id], artifact_id, "json"
            )
            for artifact_id in artifacts
        },
        artifact_document_ids={
            artifact_id: document.document_id for artifact_id, document in artifacts.items()
        },
        artifact_placement=placement,
        retrieval_config=ScopeRetrievalConfig(
            top_n=len(artifacts),
            parameters={"runtime_local_top_k": 1, "tokenizer": "lowercase_a-z0-9"},
        ),
        artifact_count=len(artifacts),
        evaluator_type="longbench_v2_official_multiple_choice",
        metadata={
            "source_id": selection.multidoc.source_id,
            "source_domain": selection.multidoc.domain,
            "source_sub_domain": selection.multidoc.sub_domain,
            "source_difficulty": selection.multidoc.difficulty,
            "source_length": selection.multidoc.length,
            "context_word_count": len(WORD_PATTERN.findall(selection.multidoc.context)),
            "boundary_recovery": (
                "four audited complete-report transitions at source lines "
                "0, 3277, 9091, and 17721; no token-based splitting"
            ),
            "visible_text_normalization": "Unicode NFKC (gold answer letter unchanged)",
        },
    )


def _structured_plan() -> ScopeStructuredPlan:
    return ScopeStructuredPlan.model_validate(
        {
            "predicates": [
                {"field": "EndDate", "operator": "eq", "value": "2021-12-31"},
                {"field": "TotalLiability", "operator": "gt", "value": 0},
            ],
            "select_fields": [
                "Symbol",
                "EndDate",
                "TotalAssets",
                "TotalLiability",
            ],
            "derived_fields": [
                {
                    "alias": "asset_liability_ratio",
                    "operator": "divide",
                    "operands": ["TotalAssets", "TotalLiability"],
                }
            ],
            "aggregations": [],
            "group_by": [],
            "order_by": [{"field": "asset_liability_ratio", "direction": "descending"}],
            "limit": 1,
        }
    )


def _structured_task(selection: LongBenchV2Selection) -> ScopeTask:
    artifacts = {
        _digest_id("lb2-artifact", {"shard_id": shard.shard_id}): shard
        for shard in selection.structured_shards
    }
    return ScopeTask(
        task_id=STRUCTURED_TASK_ID,
        dataset=DATASET_NAME,
        task_family="structured_data_analysis",
        instruction=(
            "Execute the generic structured plan over all natural records, combine partial "
            "results, and return the selected choice letter A, B, C, or D."
        ),
        query=_runtime_text(selection.structured.question),
        answer_choices=_visible_choices(selection.structured),
        artifact_ids=list(artifacts),
        artifact_types={artifact_id: "records" for artifact_id in artifacts},
        artifact_sizes={
            artifact_id: len(shard.payload) for artifact_id, shard in artifacts.items()
        },
        artifact_tokens={
            artifact_id: len(tokenize(shard.payload.decode("utf-8")))
            for artifact_id, shard in artifacts.items()
        },
        artifact_source_refs={
            artifact_id: _source_ref("longbench-structured", shard.site, artifact_id, "json")
            for artifact_id, shard in artifacts.items()
        },
        artifact_document_ids={
            artifact_id: shard.shard_id for artifact_id, shard in artifacts.items()
        },
        artifact_placement={artifact_id: shard.site for artifact_id, shard in artifacts.items()},
        structured_plan=_structured_plan(),
        artifact_count=len(artifacts),
        evaluator_type="longbench_v2_official_multiple_choice",
        metadata={
            "source_id": selection.structured.source_id,
            "source_domain": selection.structured.domain,
            "source_sub_domain": selection.structured.sub_domain,
            "source_difficulty": selection.structured.difficulty,
            "source_length": selection.structured.length,
            "natural_record_count": sum(
                shard.record_count for shard in selection.structured_shards
            ),
            "record_schema": list(selection.structured_columns),
            "numeric_columns": [
                column
                for column in selection.structured_columns
                if column not in STRUCTURED_TEXT_COLUMNS
            ],
            "records_per_agent": {
                shard.site: shard.record_count for shard in selection.structured_shards
            },
            "record_boundary_recovery": "one audited fixed-width source line per record",
            "record_placement": "sha256(record_id) prefix u64 modulo 3",
            "numeric_value_recovery": (
                "finite JSON int/float; source NaN, missing, and non-finite values map to null"
            ),
            "local_operation": "filter -> derive(divide) -> local top-1 -> global top-1",
            "visible_text_normalization": "Unicode NFKC (gold answer letter unchanged)",
        },
    )


def _evaluator(row: LongBenchV2Row, task_id: str) -> ScopeEvaluatorRecord:
    return ScopeEvaluatorRecord(
        task_id=task_id,
        evaluator_type="longbench_v2_official_multiple_choice",
        answer=row.answer,
        supporting_document_ids=[],
        evidence_document_count=0,
        question_type=row.sub_domain,
        evaluator_config={
            "prediction_extraction_regexes": [
                r"The correct answer is \(([A-D])\)",
                r"The correct answer is ([A-D])",
            ],
            "score": "exact_match_on_extracted_choice",
        },
        original_evaluator_information={
            "dataset_repository": DATASET_REPOSITORY,
            "dataset_revision": DATASET_REVISION,
            "dataset_source_id": row.source_id,
            "reference_repository": REFERENCE_REPOSITORY,
            "reference_revision": REFERENCE_REVISION,
        },
    )


def _task_bank_record(task: ScopeTask) -> ScopeTaskBankRecord:
    return ScopeTaskBankRecord(
        task_id=task.task_id,
        planner_visible=task,
        evaluator_only=ScopeTaskBankEvaluatorSummary(
            evidence_document_count=0,
            evaluator_type=task.evaluator_type,
        ),
    )


def _upsert_task_bank(path: Path, additions: list[ScopeTaskBankRecord]) -> None:
    records: list[ScopeTaskBankRecord] = []
    if path.exists():
        replaced_ids = {record.task_id for record in additions}
        replaced_families = {record.planner_visible.task_family for record in additions}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = cast(dict[str, Any], json.loads(line))
            if str(raw.get("task_id")) in replaced_ids:
                continue
            visible = cast(dict[str, Any], raw.get("planner_visible", {}))
            if (
                visible.get("dataset") == DATASET_NAME
                and visible.get("task_family") in replaced_families
            ):
                continue
            records.append(ScopeTaskBankRecord.model_validate(raw))
    by_id = {record.task_id: record for record in records}
    by_id.update({record.task_id: record for record in additions})
    write_jsonl(path, list(by_id.values()))


def _selection_markdown(
    adapter: LongBenchV2Adapter,
    selection: LongBenchV2Selection,
    multidoc_task: ScopeTask,
    structured_task: ScopeTask,
) -> str:
    multidoc_sites = sorted(set(multidoc_task.artifact_placement.values()))
    return "\n".join(
        [
            "# LongBench-v2 task selection",
            "",
            "## Source and evaluator pinning",
            "",
            f"- Dataset: `{DATASET_REPOSITORY}` at `{adapter.dataset_revision}`",
            f"- Official `data.json` SHA-256: `{adapter.source_sha256}`",
            f"- Official evaluator: `{REFERENCE_REPOSITORY}` at `{REFERENCE_REVISION}`",
            "- Quality: exact match after the two official answer-choice extraction regexes",
            "- Gold answer letters are stored only in `evaluator_only/longbench_v2.jsonl`",
            "- Planner-visible questions/options use deterministic Unicode NFKC normalization (for example NBSP to space and full-width semicolon to ASCII); evaluator-only answer letters are unchanged",
            "",
            "## Selected extra-long multi-document QA",
            "",
            f"- Source ID / task ID: `{selection.multidoc.source_id}` / `{MULTIDOC_TASK_ID}`",
            f"- Category: `{selection.multidoc.domain}` / `{selection.multidoc.sub_domain}`",
            f"- Official length label / measured words: `{selection.multidoc.length}` / {len(WORD_PATTERN.findall(selection.multidoc.context)):,}",
            f"- Recovered natural documents: {len(selection.documents)} on {', '.join(multidoc_sites)}",
            "- Boundary evidence: four complete JD.com ESG report transitions at source lines 0, 3,277, 9,091, and 17,721 (2020, 2021, 2023, 2022)",
            "- No equal-token or equal-byte split is used",
            "- The question asks what changed over four years, so it requires cross-report synthesis",
            "- LongBench-only runtime local BM25 is fixed at Top-1 per shard; MultiHop-RAG remains Top-3",
            "",
            "## Selected long structured-data task",
            "",
            f"- Source ID / task ID: `{selection.structured.source_id}` / `{STRUCTURED_TASK_ID}`",
            f"- Category: `{selection.structured.domain}` / `{selection.structured.sub_domain}`",
            f"- Official length label: `{selection.structured.length}`",
            f"- Natural records: {sum(shard.record_count for shard in selection.structured_shards):,} fixed-width source rows plus one header",
            "- Every intact row is assigned by `stable_hash(record_id) % 3`; three JSON record-array artifacts preserve row boundaries",
            "- Numeric columns are recovered as finite JSON numbers; source NaN/missing/non-finite cells become `null`",
            f"- Records per site: {structured_task.metadata['records_per_agent']}",
            "- Nontriviality: filter a date slice, reject zero liabilities, derive an asset/liability ratio for every eligible company, then compute a distributed top-1",
            "- Generic local semantics: `filter_records` + derived divide + order descending + limit; no dataset-specific operator or gold value",
            "",
        ]
    )


def _write_core_set_manifest(output_dir: Path, longbench_tasks: list[ScopeTask]) -> None:
    task_bank_path = output_dir / "task_bank.jsonl"
    scope_records = [
        ScopeTaskBankRecord.model_validate_json(line)
        for line in task_bank_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    multihop = [
        record for record in scope_records if record.planner_visible.dataset == "MultiHop-RAG"
    ]
    entries: list[dict[str, Any]] = [
        {
            "task_id": "video_mme:795",
            "dataset": "Video-MME",
            "task_family": "long_video_qa",
            "record_ref": "../calibration_v0",
            "role": "workflow_reversal_positive_case",
        },
        {
            "task_id": "video_mme:848",
            "dataset": "Video-MME",
            "task_family": "long_video_qa",
            "record_ref": "../calibration_v0",
            "role": "quality_risk_case",
        },
    ]
    entries.extend(
        {
            "task_id": record.task_id,
            "dataset": record.planner_visible.dataset,
            "task_family": record.planner_visible.task_family,
            "record_ref": "task_bank.jsonl",
            "role": record.planner_visible.metadata.get("selection_label", "multihop_rag"),
        }
        for record in multihop
    )
    entries.extend(
        {
            "task_id": task.task_id,
            "dataset": task.dataset,
            "task_family": task.task_family,
            "record_ref": "task_bank.jsonl",
            "role": "extra_long_multidoc"
            if task.task_family == "multi_document_qa"
            else "long_structured_data",
        }
        for task in longbench_tasks
    )
    manifest = {
        "schema_version": "scope-expansion-v0-core-set-v1",
        "expected_task_count": 7,
        "task_count": len(entries),
        "tasks": entries,
    }
    write_json(output_dir / "core_set_manifest.json", manifest)
    lines = [
        "# Scope Expansion v0 core problem-validity set",
        "",
        "Video tasks remain referenced from `calibration_v0`; they are not coerced into the non-video task schema.",
        "",
        "| Task | Dataset | Family | Role | Record |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| {item['task_id']} | {item['dataset']} | {item['task_family']} | {item['role']} | {item['record_ref']} |"
        for item in entries
    )
    (output_dir / "core_set_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def materialize_longbench_v2(
    adapter: LongBenchV2Adapter,
    selection: LongBenchV2Selection,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write the two selected tasks and evaluator-only gold records."""

    output = Path(output_dir)
    multidoc_task = _multidoc_task(selection)
    structured_task = _structured_task(selection)
    tasks = [multidoc_task, structured_task]

    multidoc_by_id = {
        _digest_id("lb2-artifact", {"document_id": document.document_id}): document
        for document in selection.documents
    }
    for artifact_id, document in multidoc_by_id.items():
        site = multidoc_task.artifact_placement[artifact_id]
        path = output / "longbench_multidoc" / "artifacts" / site / f"{artifact_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(document.payload)
    write_json(
        output / "longbench_multidoc" / "task.json",
        multidoc_task.model_dump(mode="json"),
    )

    structured_by_id = {
        _digest_id("lb2-artifact", {"shard_id": shard.shard_id}): shard
        for shard in selection.structured_shards
    }
    for artifact_id, shard in structured_by_id.items():
        path = output / "longbench_structured" / "artifacts" / shard.site / f"{artifact_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(shard.payload)
    write_json(
        output / "longbench_structured" / "task.json",
        structured_task.model_dump(mode="json"),
    )

    evaluator_dir = output / "evaluator_only"
    evaluator_dir.mkdir(parents=True, exist_ok=True)
    evaluators = [
        _evaluator(selection.multidoc, MULTIDOC_TASK_ID),
        _evaluator(selection.structured, STRUCTURED_TASK_ID),
    ]
    write_jsonl(evaluator_dir / "longbench_v2.jsonl", evaluators)
    _upsert_task_bank(output / "task_bank.jsonl", [_task_bank_record(task) for task in tasks])

    selection_dir = output / "task_selection"
    selection_dir.mkdir(parents=True, exist_ok=True)
    (selection_dir / "longbench_selection.md").write_text(
        _selection_markdown(adapter, selection, multidoc_task, structured_task),
        encoding="utf-8",
    )
    write_json(
        selection_dir / "longbench_selection.json",
        {
            "dataset_repository": DATASET_REPOSITORY,
            "dataset_revision": adapter.dataset_revision,
            "dataset_sha256": adapter.source_sha256,
            "reference_repository": REFERENCE_REPOSITORY,
            "reference_revision": REFERENCE_REVISION,
            "selected_source_ids": [MULTIDOC_SOURCE_ID, STRUCTURED_SOURCE_ID],
            "selected_task_ids": [MULTIDOC_TASK_ID, STRUCTURED_TASK_ID],
            "multidoc": {
                "word_count": len(WORD_PATTERN.findall(selection.multidoc.context)),
                "natural_document_count": len(selection.documents),
                "document_ids": [document.document_id for document in selection.documents],
                "sites": list(multidoc_task.artifact_placement.values()),
                "artifacts": [
                    {
                        "artifact_id": artifact_id,
                        "document_id": document.document_id,
                        "site": multidoc_task.artifact_placement[artifact_id],
                        "bytes": len(document.payload),
                        "sha256": sha256(document.payload).hexdigest(),
                    }
                    for artifact_id, document in multidoc_by_id.items()
                ],
                "boundary_rule": (
                    "audited complete-report starts at source lines 0,3277,9091,17721"
                ),
            },
            "structured": {
                "natural_record_count": sum(
                    shard.record_count for shard in selection.structured_shards
                ),
                "columns": list(selection.structured_columns),
                "numeric_columns": [
                    column
                    for column in selection.structured_columns
                    if column not in STRUCTURED_TEXT_COLUMNS
                ],
                "missing_value_policy": (
                    "source NaN, missing, and non-finite numeric values map to JSON null"
                ),
                "records_per_site": structured_task.metadata["records_per_agent"],
                "artifacts": [
                    {
                        "artifact_id": artifact_id,
                        "shard_id": shard.shard_id,
                        "site": shard.site,
                        "record_count": shard.record_count,
                        "bytes": len(shard.payload),
                        "sha256": sha256(shard.payload).hexdigest(),
                    }
                    for artifact_id, shard in structured_by_id.items()
                ],
                "structured_plan": structured_task.structured_plan.model_dump(mode="json")
                if structured_task.structured_plan
                else None,
            },
        },
    )
    _write_core_set_manifest(output, tasks)
    return {
        "task_count": len(tasks),
        "task_ids": [task.task_id for task in tasks],
        "source_ids": [MULTIDOC_SOURCE_ID, STRUCTURED_SOURCE_ID],
        "artifact_count": sum(task.artifact_count for task in tasks),
        "artifact_bytes": sum(sum(task.artifact_sizes.values()) for task in tasks),
    }
