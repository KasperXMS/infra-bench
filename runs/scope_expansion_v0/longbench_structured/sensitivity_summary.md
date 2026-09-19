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
| longbench-v2-66f3ac0b821e116aacb2e203 | communication_sensitive, representation_sensitive, parallelism_sensitive, context_cost_sensitive | - | - | 0.0 | 1.0 |

## Measured cells

| Task | World | Workflow | Protocol | Quality gate | E2E ms | Transfer bytes | Transfer sum ms | Observed effective transfer Mbps | Input tokens |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| longbench-v2-66f3ac0b821e116aacb2e203 | H1_distributed_constrained | centralized_raw | PASS | FAIL | 117635.86739997845 | 3519986.0 | 9713.614317122847 | 2.899012363540176 | 4098.0 |
| longbench-v2-66f3ac0b821e116aacb2e203 | H1_distributed_constrained | distributed_compute | PASS | PASS | 3602.535799960606 | 438.0 | 247.00710107572377 | 0.014185826985297058 | 583.0 |
| longbench-v2-66f3ac0b821e116aacb2e203 | H2_distributed_favorable | centralized_raw | PASS | FAIL | 108677.30069998652 | 3519986.0 | 797.8869848884642 | 35.293078510281056 | 4098.0 |
| longbench-v2-66f3ac0b821e116aacb2e203 | H2_distributed_favorable | distributed_compute | PASS | PASS | 3215.524499886669 | 438.0 | 150.33236076124012 | 0.02330835478307362 | 583.0 |

`*_sum_ms` is aggregate work. `*_critical_ms` is emitted only when trace evidence supports a wall-clock critical path. A sensitivity label is multi-label evidence, not anchor admission.
`observed_effective_transfer_mbps` is application-observed payload throughput over transfer sum time; it includes added RTT and must not be read as pure link capacity or a separate network probe.
