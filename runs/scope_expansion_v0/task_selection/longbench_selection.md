# LongBench-v2 task selection

## Source and evaluator pinning

- Dataset: `zai-org/LongBench-v2` at `2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9`
- Official `data.json` SHA-256: `15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2`
- Official evaluator: `THUDM/LongBench` at `2e00731f8d0bff23dc4325161044d0ed8af94c1e`
- Quality: exact match after the two official answer-choice extraction regexes
- Gold answer letters are stored only in `evaluator_only/longbench_v2.jsonl`
- Planner-visible questions/options use deterministic Unicode NFKC normalization (for example NBSP to space and full-width semicolon to ASCII); evaluator-only answer letters are unchanged

## Selected extra-long multi-document QA

- Source ID / task ID: `66f7c780bb02136c067c35e8` / `longbench-v2-66f7c780bb02136c067c35e8`
- Category: `Multi-Document QA` / `Financial`
- Official length label / measured words: `long` / 143,802
- Recovered natural documents: 4 on A28, A4, A5
- Boundary evidence: four complete JD.com ESG report transitions at source lines 0, 3,277, 9,091, and 17,721 (2020, 2021, 2023, 2022)
- No equal-token or equal-byte split is used
- The question asks what changed over four years, so it requires cross-report synthesis
- LongBench-only runtime local BM25 is fixed at Top-1 per shard; MultiHop-RAG remains Top-3

## Selected long structured-data task

- Source ID / task ID: `66f3ac0b821e116aacb2e203` / `longbench-v2-66f3ac0b821e116aacb2e203`
- Category: `Long Structured Data Understanding` / `Table QA`
- Official length label: `long`
- Natural records: 17,071 fixed-width source rows plus one header
- Every intact row is assigned by `stable_hash(record_id) % 3`; three JSON record-array artifacts preserve row boundaries
- Numeric columns are recovered as finite JSON numbers; source NaN/missing/non-finite cells become `null`
- Records per site: {'A4': 5662, 'A5': 5654, 'A28': 5755}
- Nontriviality: filter a date slice, reject zero liabilities, derive an asset/liability ratio for every eligible company, then compute a distributed top-1
- Generic local semantics: `filter_records` + derived divide + order descending + limit; no dataset-specific operator or gold value
