# Scope Expansion v0 sensitivity summary

All latency and preference statements use measured runs only. Warm-up rows are excluded, and efficiency is compared only where both workflows pass the original evaluator.

## Network worlds

| World | Configured bandwidth (Mbps) | Configured added RTT (ms) | Measured effective bandwidth (Mbps) |
| --- | ---: | ---: | ---: |
| H1_distributed_constrained | 3 | 83 | - |
| H2_distributed_favorable | 100 | 33 | - |

## Task sensitivity

| Task | Labels | H1 preference | H2 preference | Centralized quality | Distributed quality |
| --- | --- | --- | --- | ---: | ---: |
| multihop-rag-train-9bae0079038050a37a1ae583 | communication_sensitive, representation_sensitive, parallelism_sensitive | - | - | 0.0 | 0.0 |
| multihop-rag-train-0071e3a68e90c2896fc7fce7 | communication_sensitive, representation_sensitive, parallelism_sensitive, context_cost_sensitive | - | - | 0.0 | 0.0 |
| multihop-rag-train-9fec8a884e51ff8e037c394f | communication_sensitive, representation_sensitive, parallelism_sensitive, context_cost_sensitive | centralized_raw | centralized_raw | 1.0 | 1.0 |

## Measured cells

| Task | World | Workflow | Protocol | Quality gate | E2E ms | Transfer bytes | Transfer sum ms | Observed effective transfer Mbps | Input tokens |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| multihop-rag-train-9bae0079038050a37a1ae583 | H1_distributed_constrained | centralized_raw | PASS | FAIL | 56826.27130008768 | 541938.0 | 3779.812670312822 | 1.1470155740922443 | 4197.0 |
| multihop-rag-train-9bae0079038050a37a1ae583 | H1_distributed_constrained | distributed_retrieval | PASS | FAIL | 103193.09139996767 | 139053.0 | 584.5007810276002 | 1.903203615988789 | 4098.0 |
| multihop-rag-train-9bae0079038050a37a1ae583 | H2_distributed_favorable | centralized_raw | PASS | FAIL | 54221.34439996444 | 541938.0 | 1312.2403880115598 | 3.3038946519315693 | 4197.0 |
| multihop-rag-train-9bae0079038050a37a1ae583 | H2_distributed_favorable | distributed_retrieval | PASS | FAIL | 100983.67430001963 | 139053.0 | 196.52013876475394 | 5.660610698690969 | 4098.0 |
| multihop-rag-train-0071e3a68e90c2896fc7fce7 | H1_distributed_constrained | centralized_raw | PASS | FAIL | 34697.889799950644 | 385878.0 | 3041.1743104923517 | 1.0150763109333991 | 2322.0 |
| multihop-rag-train-0071e3a68e90c2896fc7fce7 | H1_distributed_constrained | distributed_retrieval | PASS | FAIL | 103601.13339999225 | 121360.0 | 536.0892438329756 | 1.8110417456957748 | 4098.0 |
| multihop-rag-train-0071e3a68e90c2896fc7fce7 | H2_distributed_favorable | centralized_raw | PASS | FAIL | 32862.8030999098 | 385878.0 | 1150.3402355592698 | 2.683574741258318 | 2322.0 |
| multihop-rag-train-0071e3a68e90c2896fc7fce7 | H2_distributed_favorable | distributed_retrieval | PASS | FAIL | 103140.72819997091 | 121360.0 | 125.7905752863735 | 7.718225294619289 | 4098.0 |
| multihop-rag-train-9fec8a884e51ff8e037c394f | H1_distributed_constrained | centralized_raw | PASS | PASS | 51742.36459995154 | 314750.0 | 3058.1456581130624 | 0.8233747772346648 | 1475.0 |
| multihop-rag-train-9fec8a884e51ff8e037c394f | H1_distributed_constrained | distributed_retrieval | PASS | PASS | 93101.01099999156 | 55480.0 | 360.59361114166677 | 1.2308593005704367 | 4098.0 |
| multihop-rag-train-9fec8a884e51ff8e037c394f | H2_distributed_favorable | centralized_raw | PASS | PASS | 49918.87119994499 | 314750.0 | 1203.6420949734747 | 2.091983996335298 | 1475.0 |
| multihop-rag-train-9fec8a884e51ff8e037c394f | H2_distributed_favorable | distributed_retrieval | PASS | PASS | 92965.06369998679 | 55480.0 | 113.47677116282284 | 3.9112850625891835 | 4098.0 |

`*_sum_ms` is aggregate work. `*_critical_ms` is emitted only when trace evidence supports a wall-clock critical path. A sensitivity label is multi-label evidence, not anchor admission.
`observed_effective_transfer_mbps` is application-observed payload throughput over transfer sum time; it includes added RTT and must not be read as pure link capacity or a separate network probe.
