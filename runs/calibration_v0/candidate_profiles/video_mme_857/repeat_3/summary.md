# calibration_v0 Jetson local-reduction profiles

These are new Jetson-only measurements. The 4090 worker was neither contacted nor used.
Each branch runs the generic `sample_frames` operator followed by one local VLM call; the three branches execute concurrently.

| Task | Raw bytes | Evidence bytes | Reduction | Sum compute | Critical branch | Bottleneck |
|---|---:|---:|---:|---:|---:|---|
| `video_mme:857` | 215,488,462 | 7,200 | 29929x | 415.53s | a4 (179.53s) | uniform frame sampling |

`sum` is total resource time over all branches; `critical` is the measured longest branch from controller timestamps, not `sum / 3`. The reduction path is raw MP4 -> 12 fixed uniform frames -> one chronological JPEG contact sheet -> compact JSON semantic evidence. No clips were selected in this implementation.
For video_mme:857, the measured critical local cost is 179.53s (415.53s summed resource time) and avoids transferring 215,481,262 raw bytes.
This is a compute-for-communication trade: preprocessing is not free, and it only wins when saved transfer time exceeds the measured local critical path plus quality risk.
