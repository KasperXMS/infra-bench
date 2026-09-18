# video_mme:848 H1/local_reduction failure analysis

## Outcome

| Repeat | Final answer | Score | Interpretation |
|---|---|---:|---|
| r1 | D / C / A | 3/3 | Correct, but evidence is not semantically sufficient |
| r2 | D / C / A | 3/3 | Correct, but evidence is not semantically sufficient |
| r3 | C / C / D | 1/3 | Fails 848-1 and 848-3 |

Gold is **D / C / A**. All outputs are valid JSON and the unchanged `video_mme_multiple_choice` evaluator parsed all three runs successfully. No rerun was performed.

## Controlled evidence comparison

The fixed uniform contact sheet for each site is byte-identical across r1/r2/r3. A28 semantic evidence is identical in all three runs; A5 evidence is identical in successful r1 and failed r3; A4 r1/r2 are identical while r3 is shorter but does not omit a fact that directly supports either flipped answer. All nine evidence artifacts, hashes, byte counts, remote source paths, and reconstructed final reasoning messages are preserved in `video_mme_848_run_comparison.json`.

None of the three evidence sets directly establishes the requested four-chapter order, enumerates four major setbacks, or states that failed Onitsuka negotiations caused the 1971 name change. Thus r1/r2 are fragile correct outcomes under an evidence gap, not proof that reduction preserved sufficient information.

## Required taxonomy

- Primary: **`final reasoning error`**
- Contributing: **`stochastic output variance`**
- Shared upstream risks: **`frame/clip selection missed`**, **`semantic reduction information loss`**

Not assigned as the repeat-3 cause:

- `OCR/detail loss`: OCR-like extraction is stable and no run has the missing decisive wording.
- `local VLM perception error`: failed r3 shares A5 evidence exactly with successful r1 and A28 evidence with both successes.
- `temporal evidence missed`: temporal support is absent, but the observed mechanisms are sparse uniform selection and subsequent semantic compression.
- `other`: no additional mechanism is needed.

The task must remain quality-gate failed for H1/local_reduction. The evaluator and evidence were not changed, and no gold-guided rerun was used.
