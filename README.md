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
- canonical manual-fallback workflow-bank construction with explicit provenance.
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
`task_id`, `success`, and an `actions` array. Real successful trajectories are
preferred; manual strategies are used only to preserve minimum diversity:

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

## Design boundary

The planner selects the semantic workflow. The shared physical optimizer only
binds existing workflow nodes to compatible executors; it never rewrites the DAG.
Planner inputs are produced through an oracle-free schema so labels and oracle
metrics cannot leak into selection prompts.
