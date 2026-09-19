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
| longbench-v2-66f7c780bb02136c067c35e8 | communication_sensitive, parallelism_sensitive | - | - | 0.0 | 0.0 |

## Measured cells

| Task | World | Workflow | Protocol | Quality gate | E2E ms | Transfer bytes | Transfer sum ms | Observed effective transfer Mbps | Input tokens |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| longbench-v2-66f7c780bb02136c067c35e8 | H1_distributed_constrained | centralized_raw | PASS | FAIL | 100017.91669998784 | 798391.0 | 2534.230706980452 | 2.520341965081109 | 4098.0 |
| longbench-v2-66f7c780bb02136c067c35e8 | H1_distributed_constrained | distributed_retrieval | PASS | FAIL | 98815.01969997771 | 682476.0 | 2047.207438852638 | 2.6669539668437126 | 4098.0 |
| longbench-v2-66f7c780bb02136c067c35e8 | H2_distributed_favorable | centralized_raw | PASS | FAIL | 97991.40709999483 | 798391.0 | 518.4224490076303 | 12.32031524141423 | 4098.0 |
| longbench-v2-66f7c780bb02136c067c35e8 | H2_distributed_favorable | distributed_retrieval | PASS | FAIL | 97138.07410001755 | 682476.0 | 244.50425687246025 | 22.330114288553993 | 4098.0 |

`*_sum_ms` is aggregate work. `*_critical_ms` is emitted only when trace evidence supports a wall-clock critical path. A sensitivity label is multi-label evidence, not anchor admission.
`observed_effective_transfer_mbps` is application-observed payload throughput over transfer sum time; it includes added RTT and must not be read as pure link capacity or a separate network probe.
