# Scope Expansion v0 top-level sensitivity summary

Efficiency winners are reported only when both workflows pass the original evaluator quality gate. Warm-ups never enter medians or quality decisions.

## Completeness

- Complete: `true`
- Tasks: 7/7
- Cells: 28/28
- Protocol-complete cells: 28/28

## Tasks

| Task | Family | Protocol | Quality-comparable | H1 winner | H2 winner | Labels |
| --- | --- | --- | --- | --- | --- | --- |
| longbench-v2-66f3ac0b821e116aacb2e203 | structured_data_analysis | yes | no | - | - | communication_sensitive, representation_sensitive, parallelism_sensitive, context_cost_sensitive |
| longbench-v2-66f7c780bb02136c067c35e8 | multi_document_qa | yes | no | - | - | communication_sensitive, parallelism_sensitive |
| multihop-rag-train-0071e3a68e90c2896fc7fce7 | multi_document_qa | yes | no | - | - | communication_sensitive, representation_sensitive, parallelism_sensitive, context_cost_sensitive |
| multihop-rag-train-9bae0079038050a37a1ae583 | multi_document_qa | yes | no | - | - | communication_sensitive, representation_sensitive, parallelism_sensitive |
| multihop-rag-train-9fec8a884e51ff8e037c394f | multi_document_qa | yes | yes | centralized_raw | centralized_raw | communication_sensitive, representation_sensitive, parallelism_sensitive, context_cost_sensitive |
| video_mme:795 | long_video_qa | yes | yes | local_reduction | centralized_raw | workflow_reversal, communication_sensitive, representation_sensitive |
| video_mme:848 | long_video_qa | yes | no | - | centralized_raw | communication_sensitive, representation_sensitive, quality_risk |

## Family summary

| Family | Tasks | Protocol complete | Quality-comparable | Triggered labels |
| --- | ---: | ---: | ---: | --- |
| long_video_qa | 2 | 2 | 1 | workflow_reversal=1, communication_sensitive=2, representation_sensitive=2, quality_risk=1 |
| multi_document_qa | 4 | 4 | 1 | communication_sensitive=4, representation_sensitive=3, parallelism_sensitive=4, context_cost_sensitive=2 |
| structured_data_analysis | 1 | 1 | 0 | communication_sensitive=1, representation_sensitive=1, parallelism_sensitive=1, context_cost_sensitive=1 |

## Sources

- `runs/scope_expansion_v0/core_set_manifest.json` — `835c143cbfd8c664b2a0f7a251d79e070aa39fc074fe691413197528de388d5c` (1651 bytes)
- `runs/scope_expansion_v0/task_bank.jsonl` — `47e5a5d8595ca26458cf01a12ad290a1639f3e9016278cf5ddcb7ca39535f22b` (50283 bytes)
- `runs/calibration_v0/summary.json` — `affccf180ffd0fa97d87d689960339ec6c5df35cbd08784f4ccd925b8a03d605` (16685 bytes)
- `runs/scope_expansion_v0/multihop_rag/sensitivity_summary.json` — `17db868bf96efe9d0fe85c249cc219df72965c59bb9ae6bb70d966aec1c58db2` (24927 bytes)
- `runs/scope_expansion_v0/longbench_multidoc/sensitivity_summary.json` — `9c1dc99ebec97d7162aab4b83d65401c6c86eaf66ccc3b65d0fdd3b04d18b94f` (9734 bytes)
- `runs/scope_expansion_v0/longbench_structured/sensitivity_summary.json` — `c22f2d39f7e314c3f28296f6858bda637922c60d91cacc6cb2d2899b229f0c8b` (9805 bytes)
