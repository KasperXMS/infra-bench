# infra-bench

`infra-bench` is a standalone, deterministic benchmark for infrastructure-aware
semantic workflow selection. It intentionally lives beside `infra-aware-mas` and
does not import from or modify that project.

The current milestone implements:

- strict Pydantic schemas for tasks, workflows, infrastructure, cases, and results;
- DAG validation and deterministic topological ordering;
- network, load, artifact-size, latency, WAN-traffic, and monetary-cost simulation;
- a shared capability-aware exhaustive optimizer with deterministic tie-breaking;
- cost-derived semantic-switch, placement-only, and invariance sanity groups;
- dataset validation, oracle/random/resource-blind baselines, metrics, and reports.
- metadata-only adapters for SWE-bench Verified and Video-MME-v2;
- unverified semantic workflow templates that are explicitly ineligible for admission;
- normalized successful-trajectory import, canonicalization, and semantic deduplication.

Real benchmark ingestion was gated on the synthetic sanity benchmark, as required
by the implementation spec. That gate now passes. Ingestion fetches only task
metadata through Hugging Face's
dataset service; it does not clone repositories or download video archives.

## Quick start

From this directory:

```bash
uv sync --extra dev
uv run pytest
uv run infra-bench ingest --source swebench --limit 20
uv run infra-bench ingest --source video-mme --limit 20
uv run infra-bench workflows build --source swebench
uv run infra-bench workflows build --source video-mme
uv run infra-bench generate --config configs/mvp.yaml --output data/generated/mvp.jsonl
uv run infra-bench validate data/generated/mvp.jsonl
uv run infra-bench split data/generated/mvp.jsonl
uv run infra-bench evaluate --dataset data/generated/mvp.jsonl --planner random
uv run infra-bench report data/results/latest.jsonl
```

When successful agent traces are available, provide normalized JSONL records with
`task_id`, `success`, and an `actions` array. Templates preserve structural
diversity but remain `success=false`; they never count as verified workflows:

```bash
uv run infra-bench workflows build --source swebench --trajectories data/raw/swe_traces.jsonl
```

The report command accepts multiple result files to produce a comparison table:

```bash
uv run infra-bench report data/results/oracle.jsonl data/results/random.jsonl data/results/resource-blind.jsonl
```

To regenerate only the six hand-authored gate cases, add `--sanity-only` to the
`generate` command. The default generator reads both task/workflow banks and keeps
only counterfactual groups whose recomputed oracle behavior proves the requested
semantic-switch, placement-only, or invariance label.

Use `--planner oracle` for a zero-regret simulator consistency check, or
`--planner resource-blind` for the task-only baseline.

For an infrastructure-aware model selector, install the optional API dependency,
set `OPENAI_API_KEY`, and pass an explicit model ID:

```bash
uv sync --extra dev --extra llm
uv run infra-bench evaluate --dataset data/generated/mvp.jsonl --planner llm --model YOUR_MODEL --output data/results/llm.jsonl
```

The selector uses the Responses API with a strict JSON Schema constrained to the
candidate workflow IDs and sends `store=false`. Evaluator configuration and gold
answers are removed before prompt construction.

## Original task evaluators

Video-MME-v2 predictions use JSONL records containing `task_id` and `prediction`.
The built-in grader reproduces the official four-question relevance/logic scoring:

```bash
uv run infra-bench grade video-mme --predictions predictions.jsonl
```

SWE-bench correctness is delegated to its official Docker harness. Inspect the
exact command before installing/running that external harness:

```bash
uv run infra-bench grade swebench --predictions patches.jsonl --run-id experiment-1 --dry-run
```

## Real task admission

The real-task milestone uses the narrow candidate set in
`configs/real_task_admission.yaml`: three SWE-bench Verified instances and three
Video-MME-v2 four-question groups. Two semantically different workflows are
executed per candidate. A record enters `verified_workflow_bank.jsonl` only after
the original benchmark evaluator meets the configured quality threshold.

Video execution is intended for a storage/GPU host with the official MP4 files:

```bash
uv run --extra real-video infra-bench run-real-video \
  --video-dir /path/to/videos \
  --credential-file /path/to/api_key.txt \
  --video-ids 002 003 004
```

SWE-bench patch generation keeps the model constant while changing semantic
processing: remote file selection from the repository tree versus local
search/static filtering followed by compact remote reasoning.

```bash
uv run infra-bench run-real-swebench \
  --repo-root /path/to/checkouts \
  --credential-file /path/to/api_key.txt \
  --task-ids astropy__astropy-14309 astropy__astropy-14995 astropy__astropy-7166
```

Generate the fail-closed report by executing the scoring functions directly from
a pinned checkout of the official Video-MME-v2 evaluation script:

```bash
uv run --extra real-video infra-bench report-real-admission \
  --official-video-script /path/to/Video-MME-v2/evaluation/test_video_mme_v2.py \
  --video-ids 002 003 004 \
  --swebench-ids astropy__astropy-14309 astropy__astropy-14995 astropy__astropy-7166
```

Counterfactual calibration starts only when both workflows for a task pass. It
uses measured preprocessing/model service times and actual payload sizes, then
scans artifact locality, bandwidth/RTT, replica availability, and preprocessing
service factors. Only pairs with opposite winners and at least 20% margin on both
sides are listed in `admitted_pairs.json`; only those pairs are eligible for MAS
export.

## Design boundary

The planner selects the semantic workflow. The shared physical optimizer only
binds existing workflow nodes to compatible executors; it never rewrites the DAG.
Planner inputs are produced through an oracle-free schema so labels and oracle
metrics cannot leak into selection prompts.

## Open-ended real-system MAS bridge

The `integration/` layer is separate from the finite workflow-selection benchmark above. It exports
only the instruction, artifact sources and initial sites, and planner-visible infrastructure facts.
Candidate/reference workflows, oracle metrics, and evaluator configuration stay in this repository.
The sibling `infra-aware-mas` remains an open-ended `dynamic_models` Planner.

Build and export the reproducible six-image V1 smoke pair:

```bash
uv run infra-bench prepare-mas-smoke \
  --images ../taskset_v0/images \
  --output data/generated/v1-mas-smoke.jsonl
uv run infra-bench export-mas \
  --dataset data/generated/v1-mas-smoke.jsonl \
  --group v1-six-image-real-system-smoke \
  --output data/mas_exports/v1-smoke
```

After `infra-mas-bench` has run the static/snapshot arms on both worlds, import and report them:

```bash
uv run infra-bench import-mas \
  --dataset data/generated/v1-mas-smoke.jsonl \
  --run ../infra-aware-mas/runs/example-static-a \
  --output data/results/mas/example-static-a.json
uv run infra-bench report-mas data/results/mas/*.json \
  --output-dir data/results/mas/report
```

Import associates the result with the original case before applying its hidden evaluator. The
report keeps task quality, real E2E/Planner/service/transfer measurements, structural workflow
metrics, and calibrated `reference_regret` separate rather than collapsing them into one score.
