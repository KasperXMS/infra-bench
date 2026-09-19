# Scope Expansion v0

This directory defines only the fixed two-world experiment surface. Each task
family compares `centralized_raw` with one generic distributed workflow:
`distributed_retrieval` for document QA or `distributed_compute` for structured
data. Real M2/M3/M4 task records are produced by the offline selector after
full-corpus BM25 coverage validation; no gold answer or supporting-document
identifier belongs in a runtime task.

`task_bank.jsonl` uses `ScopeTaskBankRecord`: its nested `planner_visible` value is the only runtime-safe task payload, while nested `evaluator_only` contains only audit metadata. Full answers and supporting-document IDs remain in a separate evaluator-only file.

The formal protocol is one warm-up (`warmup=true`, `repeat=0`) followed by measured repeats 1--3 for every `(task, workflow, world)` cell. `report-scope-expansion-v0` reads runtime-owned raw rows and writes derived evaluated rows plus sensitivity summaries; it never rewrites `raw_runs.jsonl`.

After the stable MultiHop-RAG pass, `prepare-scope-longbench-v2` adds exactly two
official LongBench-v2 rows pinned in `longbench_v2.yaml`. The multi-document
row is recovered as four complete source reports at audited publication
transitions, never by token length. The structured row retains each fixed-width
source row as a natural record and hashes intact records to A4/A5/A28 before
writing one JSON record-array artifact per site. Multiple-choice options are
execution-visible; the correct choice is written only to
`runs/scope_expansion_v0/evaluator_only/longbench_v2.jsonl`.
