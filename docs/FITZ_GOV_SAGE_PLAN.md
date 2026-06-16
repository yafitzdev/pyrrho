# fitz-gov-sage v1 Plan

## Decision

Use **fitz-gov V10.0.0** as the highest clean source line for the sage experiment.

- Source prep: `data/multitask_g5_1_v10_repaired`
- Source row count: **53,503**
- Included versions: V6 **2,980**, V7 **7,520**, V8 **14,092**, V9 **16,163**, V10 **12,748**
- Excluded lines: V11 strict-owner repair and all V12 local fitz-sage repair rows

The first model line should be named `pyrrho-sage-nano-g1` locally. It is a new production-shaped encoder line, not a continuation of `pyrrho-nano-g5.5`.

## Why This Exists

Clean fitz-gov rows are good for label truth, but they are too tidy compared with fitz-sage runtime evidence packs. The downstream failures show that Pyrrho needs training examples shaped like production:

- ranked packs, not clean equal-status contexts
- table/code/prose companion evidence
- stale or earlier evidence near the correct final evidence
- conflict packs where `DISPUTED` must survive strong single-source support
- incomplete packs where `ABSTAIN` must remain the answer

The clean fitz-gov dataset remains the canonical truth source. `fitz-gov-sage` is a derived training view.

## Training Shape

Each selected source row becomes two stage rows.

| Stage | Input Shape | Trained Heads | Masked Heads |
|---|---|---|---|
| `query_planning` | query-only for now | query contract, route, retrieval action, gap type, answerability shape, retrieval modality, retrieval obligation | governance, taxonomy, scalars |
| `evidence_governance` | query + messy ranked evidence pack | governance, taxonomy, scalars, retrieval action, gap type, answerability shape, retrieval modality, retrieval obligation | none by default |

The multitask trainer now treats any class id of `-1` as masked, so one dataset can carry both stages without training the wrong head on the wrong input.

## First Batch

Target: **10,000 source rows** from V10-clean prep.

Expected output: **20,000 training rows** after stage expansion.

Default source selection:

- at least **65%** obligation-labeled rows when available
- balanced by label, route, query contract, retrieval modality, retrieval obligation, answerability shape, taxonomy, difficulty, dataset version
- deterministic selection by source id hash
- no V11/V12 rows

Command:

```powershell
python scripts/prepare_fitz_gov_sage_workpacks.py --target-source-rows 10000
```

Output is under `data/fitz_gov_sage_v1_workpacks/` and is ignored by git because it is generated.

## Subagent Contract

Subagents transform each source row into the two sage stage rows. The prompt is:

`docs/prompts/FITZ_GOV_SAGE_TRANSFORM_SUBAGENT.md`

Rules:

- Do not change the governance label.
- Do not change the target query.
- Do not copy fitz-sage benchmark documents or entity names.
- Preserve the source row id in every output row.
- Produce exactly two rows per source row unless the source row is structurally invalid.
- Make evidence packs messy but still honest: distractors and stale/conflict variants must be plausible and label-consistent.

## Effort Estimate

With 10,000 source rows and 50 source rows per workpack:

- **200 workpacks**
- **20,000 final stage rows**
- First useful pilot: 200 source rows / 400 stage rows
- Full pass: depends on subagent throughput, but the pipeline is resumable because every source id is stable

## Gates

Before training `pyrrho-sage-nano-g1`:

- schema validation: every source id has exactly one `query_planning` row and one `evidence_governance` row
- no V11/V12 source rows
- no missing labels for trained heads
- no governance labels on query-planning rows
- exact-query leakage check against fitz-sage benchmark cases
- source-id dedupe clean
- smoke train dry-run with `scripts/train_multitask_encoder.py --dry-run`

