# MultiHop-RAG task selection

This is an offline selection report. Gold annotations below are evaluator-only and
are not serialized into runtime task records or retrieval inputs.

## Fixed protocol

- Dataset: `yixuantt/MultiHopRAG` at revision `71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82`
- Dataset files: corpus `20b61b5ab84de84a927420c5d265b7ec8d859ae49980699958a787ade9e4d28f`, queries `03cfb4926461f868684903aadc8024447bdda5bb3f6804741424cce338515bff`
- Reference evaluator code: `yixuantt/MultiHop-RAG` at `c1c1287aa60a94acf9c4d20c891c9cd611a0f6e8`
- Candidate retrieval: document-level BM25 over all 609 documents, global Top-30
- Tokenizer: lower-case regex `[a-z0-9]+`; BM25 k1=1.5, b=0.75
- Runtime distributed retrieval is preregistered as per-shard Top-3
- Placement: SHA-256(document_id), first 8 bytes big-endian modulo 3
- Gold evidence affects only offline coverage, selection, and evaluation; never ranking or placement
- Multi-hop selection gate: comparison/temporal query, or an inference query whose answer is absent from every individual support fact

Top-30 was retained. Top-50 is permitted only as a single global fallback if overall
coverage proves inadequate; it was not tuned per task.

## Coverage

| support documents | eligible | fully covered | rate |
|---:|---:|---:|---:|
| 2 | 1169 | 1143 | 0.978 |
| 3 | 774 | 677 | 0.875 |
| 4 | 312 | 259 | 0.830 |

## Selected tasks

| label | task ID | type | bytes | lexical tokens | docs A4/A5/A28 | support sites (offline only) |
|---|---|---|---:|---:|---|---|
| M2 | `multihop-rag-train-9bae0079038050a37a1ae583` | `comparison_query` | 671018 | 115390 | 8/14/8 | A4, A5 |

**M2 query:** Does the Polygon Diablo 4 guide for Sorcerer builds in season 2 provide simplified versions of the builds in a similar manner to how the Polygon Diablo 4 guide for Barbarian builds does for the same season?

| M3 | `multihop-rag-train-0071e3a68e90c2896fc7fce7` | `temporal_query` | 572513 | 99037 | 6/13/11 | A4, A5, A28 |

**M3 query:** After the Polygon report on the Steam Deck OLED published at 18:00:00 on November 9, 2023, and the Engadget review of the Steam Deck OLED published at 18:00:38 on the same day, was the information about the improvements in the new iteration of the Steam Deck consistent?

| M4 | `multihop-rag-train-9fec8a884e51ff8e037c394f` | `inference_query` | 451189 | 74658 | 9/12/9 | A4, A5, A28 |

**M4 query:** Considering the reported details from 'The Independent - Life and Style' regarding Jada Pinkett Smith's memoir, the undisclosed efforts in their relationship, the timeframe of their separation mentioned at the 2022 Academy Awards, and Jada's pre-marital views on divorce, what longstanding Hollywood couple's union embodies these aspects and has faced public scrutiny despite a commitment made before their marriage to remain under the same roof?

## Selection rule

For each required support-document count, retain only non-null queries whose fixed
Top-N covers every official supporting document and whose support maps to at least
two shards. Exclude obvious single-fact answer exposures using the preregistered
multi-hop gate above. Prefer comparison/temporal structure, then number of support
shards, candidate bundle bytes, and stable task ID. This deterministic rule does not
alter retrieval results or add gold documents.
