# calibration_v0 Jetson local-reduction profiles

These are new Jetson-only measurements. The 4090 worker was neither contacted nor used.
Each branch runs the generic `sample_frames` operator followed by one local VLM call; the three branches execute concurrently.

| Task | Raw bytes | Evidence bytes | Reduction | Sum compute | Critical branch | Bottleneck |
|---|---:|---:|---:|---:|---:|---|
| `video_mme:747` | 330,076,886 | 3,739 | 88279x | 339.43s | a28 (127.51s) | local VLM |
| `video_mme:865` | 123,466,172 | 3,075 | 40152x | 394.21s | a28 (137.80s) | uniform frame sampling |

`sum` is total resource time over all branches; `critical` is the measured longest branch from controller timestamps, not `sum / 3`. The reduction path is raw MP4 -> 12 fixed uniform frames -> one chronological JPEG contact sheet -> compact JSON semantic evidence. No clips were selected in this implementation.
For video_mme:747, the measured critical local cost is 127.51s (339.43s summed resource time) and avoids transferring 330,073,147 raw bytes.
For video_mme:865, the measured critical local cost is 137.80s (394.21s summed resource time) and avoids transferring 123,463,097 raw bytes.
This is a compute-for-communication trade: preprocessing is not free, and it only wins when saved transfer time exceeds the measured local critical path plus quality risk.
