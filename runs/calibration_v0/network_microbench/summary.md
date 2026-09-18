# A4/A5/A28 network microbenchmark

All payloads moved directly between the three Jetson Workers; no 4090 endpoint was contacted. `single` runs one flow at a time across all six directions. `concurrent` runs the three-flow ring A4->A5->A28->A4 simultaneously.

| Scenario | Mode | Payload | Flows | Mean/flow Mbps | Aggregate Mbps | Critical latency |
|---|---|---:|---:|---:|---:|---:|
| physical_baseline | single | 1 MiB | 6 | 64.08 | 384.47 | 0.222s |
| physical_baseline | single | 10 MiB | 6 | 124.31 | 745.89 | 1.304s |
| physical_baseline | single | 100 MiB | 6 | 143.48 | 860.87 | 9.749s |
| physical_baseline | single | 300 MiB | 6 | 130.39 | 782.34 | 36.560s |
| physical_baseline | concurrent | 1 MiB | 3 | 36.67 | 110.02 | 0.609s |
| physical_baseline | concurrent | 10 MiB | 3 | 45.99 | 137.98 | 3.630s |
| physical_baseline | concurrent | 100 MiB | 3 | 69.57 | 208.70 | 24.750s |
| physical_baseline | concurrent | 300 MiB | 3 | 53.88 | 161.64 | 75.921s |
| H2_distributed_favorable | single | 1 MiB | 6 | 57.79 | 346.72 | 0.174s |
| H2_distributed_favorable | single | 10 MiB | 6 | 69.75 | 418.48 | 1.599s |
| H2_distributed_favorable | single | 100 MiB | 6 | 70.89 | 425.37 | 15.987s |
| H2_distributed_favorable | single | 300 MiB | 6 | 78.26 | 469.54 | 36.693s |
| H2_distributed_favorable | concurrent | 1 MiB | 3 | 41.19 | 123.57 | 0.263s |
| H2_distributed_favorable | concurrent | 10 MiB | 3 | 42.53 | 127.58 | 3.316s |
| H2_distributed_favorable | concurrent | 100 MiB | 3 | 64.05 | 192.14 | 21.020s |
| H2_distributed_favorable | concurrent | 300 MiB | 3 | 51.30 | 153.90 | 68.672s |
| H1_distributed_constrained | single | 1 MiB | 6 | 2.90 | 17.39 | 2.914s |
| H1_distributed_constrained | single | 10 MiB | 6 | 2.97 | 17.81 | 28.328s |
| H1_distributed_constrained | concurrent | 1 MiB | 3 | 2.90 | 8.69 | 2.909s |
| H1_distributed_constrained | concurrent | 10 MiB | 3 | 2.97 | 8.91 | 28.294s |

H1 100/300 MiB cells are explicitly unavailable: at 3 Mbps their ideal transfer times are about 280/839 seconds before physical-network overhead; the deployed Worker's source read timeout is 300 seconds. No latency was fabricated for them.

The simple `bytes / configured_bandwidth + RTT` model should be compared with `transfer_latency_ms`: deviations include HTTP setup, application pacing granularity, Wi-Fi jitter, and contention. Concurrent aggregate throughput exposes whether the three flows share a physical WLAN bottleneck.

## Interpretation

- H1 shaping is accurate for payload-dominated transfers: single-flow mean throughput is 2.90 Mbps at 1 MiB and 2.97 Mbps at 10 MiB versus the configured 3 Mbps. Concurrent flows remain independently paced at about 2.97 Mbps per flow.
- H2 is an upper bound, not a guaranteed observed rate. At 100/300 MiB the single-flow mean is 70.89/78.26 Mbps because the physical WLAN and HTTP path are slower than the configured 100 Mbps cap on several directions.
- Physical RTT is highly variable: directional means range from 18.43 to 36.23 ms, maxima from 88.76 to 138.39 ms, and measured jitter from 32.16 to 47.32 ms.
- The WLAN is shared and directionally asymmetric. For 300 MiB, physical concurrent mean per-flow throughput falls from 130.39 to 53.88 Mbps and critical latency grows from 36.56 to 75.92 seconds.
- A4/A5/A28 used `mq` plus `fq_codel` on `wlP1p1s0` before and after the benchmark. No `tc/netem` rule was installed; H1/H2 were enforced by the existing Worker application-layer byte pacing and added delay.
