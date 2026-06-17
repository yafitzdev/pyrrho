# fitz-gov-sage Manual Messy-Pack Procedure

This document is the canonical repeatable procedure for generating
fitz-sage-shaped `fitz-gov-sage` rows.

The rule is strict:

- **Code may select rows, split workpacks, and audit outputs.**
- **Code must not write the transformed evidence-pack rows.**
- **GPT-5.4 subagents write the row content by hand.**

## Current Clean Pilot

The procedure was validated on:

```text
data/fitz_gov_sage_messy_pilot_100_20260617
data/fitz_gov_sage_v1_messy_repair_batch_0000
```

Final status:

```text
source rows:                       100
stage rows:                        200
semantic QA before repair:         85 accept / 15 repair
semantic repair rows:              15
structural violations after repair: 0
exact source-context-list copies:  0 / 100
changed evidence packs:            100 / 100
mean contexts per evidence row:    4.33
pack shape:                        100 / 100 retrieval_pack_4_7
label mismatches:                  0
scalar mismatches:                 0
planning shape issues:             0
```

Final artifact:

```text
data/fitz_gov_sage_messy_pilot_100_20260617/pilot_status_after_semantic_repair.json
```

The first original-10k replacement wave was validated on:

```text
data/fitz_gov_sage_v1_messy_repair_batch_0000
```

Final status:

```text
source rows:                       100
stage rows:                        200
semantic QA before repair:         87 accept / 13 repair
semantic repair rows:              13
final repaired-row QA:             13 accept / 0 repair
structural violations after repair: 0
exact source-context-list copies:  0 / 100
changed evidence packs:            100 / 100
mean contexts per evidence row:    4.18
pack shape:                        100 / 100 retrieval_pack_4_7
label mismatches:                  0
scalar mismatches:                 0
```

Final artifact:

```text
data/fitz_gov_sage_v1_messy_repair_batch_0000/batch_status_final.json
```

## Worker Prompt Files

Use these prompt templates exactly:

```text
docs/prompts/FITZ_GOV_SAGE_MESSY_TRANSFORM_WORKER.md
docs/prompts/FITZ_GOV_SAGE_SEMANTIC_QA_WORKER.md
docs/prompts/FITZ_GOV_SAGE_REPAIR_WORKER.md
```

Do not use the older loose prompt as the authority for new messy-pack work:

```text
docs/prompts/FITZ_GOV_SAGE_TRANSFORM_SUBAGENT.md
```

That older prompt produced mostly source-preserving rows in the original 10k.

## Current Throughput Rule

Future waves should **not** run per-wave semantic QA.

The fast path is:

```text
prepare 100 source rows
six GPT-5.4 workers manually rewrite them from the source rows
run structural audit
run shape audit
run label/scalar preservation audit
accumulate mechanically clean outputs
```

Semantic QA is deferred into one consolidated pass across the accumulated
mechanically clean waves. The already completed `batch_0000` kept its full
per-wave QA history because that was the procedure-validation wave, not the
normal throughput path.

## Batch Preparation

Use 100-row waves. This is the proven manual-quality unit:

```text
100 source rows
200 stage rows
6 GPT-5.4 transform workers
15-17 source rows per worker
```

Do not use 500-row waves without a separate serialization layer. The first
500-row attempt put about 85 source rows on each worker and stalled before
content generation. The repeatable procedure is smaller waves with hard audits.

Prepare workpacks with:

```powershell
.venv\Scripts\python.exe scripts\prepare_fitz_gov_sage_manual_batch.py `
  --source-selection data\fitz_gov_sage_v1_workpacks\source_selection.jsonl `
  --output-dir data\fitz_gov_sage_v1_messy_repair_batch_0000 `
  --target-source-rows 100 `
  --start-index 0 `
  --selection-mode sequential `
  --workpack-size 17 `
  --worker-count 6 `
  --target-dataset fitz-gov-sage-v1-messy-repair-batch-0000 `
  --target-model-line pyrrho-sage-nano-messy-repair
```

For later batches, either increase `--start-index` or pass previous
`source_selection.jsonl` files with `--exclude-source-selection`.

The script writes:

```text
<output-dir>/source_selection.jsonl
<output-dir>/batch_manifest.json
<output-dir>/workpacks/pack_0000.json ...
<output-dir>/subagent_outputs/
```

The script does **not** write transformed rows.

## Manual Generation

Spawn GPT-5.4 workers using:

```text
docs/prompts/FITZ_GOV_SAGE_MESSY_TRANSFORM_WORKER.md
```

Each worker gets a disjoint workpack or workpack range and writes only its
assigned JSONL output file(s) under:

```text
<output-dir>/subagent_outputs/
```

Worker hard requirements:

```text
2 JSONL rows per source row
query_planning row has no contexts
evidence_governance row has 4-9 realistic fitz-sage evidence contexts
do not copy the source context list unchanged
preserve labels, query, source_id, scalar_targets
preserve all numeric label *_id fields from the source row
no fake "missing evidence" context
no internal source_id strings inside evidence contexts or pack metadata
one pack_metadata.items entry per context
```

## Structural Audit

Run:

```powershell
.venv\Scripts\python.exe scripts\audit_fitz_gov_sage_outputs.py `
  --workpack-dir <output-dir> `
  --input-dir <output-dir>\subagent_outputs `
  --output <output-dir>\subagent_output_audit.json
```

Gate:

```text
violations = 0
rows_seen = source_rows * 2
source_ids_seen = source_rows
```

This audit also rejects placeholder text, synthetic "missing evidence" filler,
pack/context count mismatches, and internal source-id leakage inside evidence
contexts or pack metadata.

## Shape Audit

Run:

```powershell
.venv\Scripts\python.exe scripts\audit_fitz_gov_sage_shape.py `
  --workpack-dir <output-dir> `
  --input-dir <output-dir>\subagent_outputs `
  --output <output-dir>\shape_audit.json
```

Gate:

```text
exact_source_context_list_fraction <= 0.10
changed_context_set_fraction >= 0.80
mean_context_count >= 4.0
short_pack_fraction <= 0.15
violations = 0
```

The clean 100-row pilot was stricter:

```text
exact_source_context_list_fraction = 0.00
changed_context_set_fraction = 1.00
mean_context_count = 4.33
short_pack_fraction = 0.00
```

## Label / Scalar Preservation

Run a label/scalar preservation check after generation and after every repair
pass. The gate is:

```powershell
.venv\Scripts\python.exe scripts\audit_fitz_gov_sage_label_preservation.py `
  --workpack-dir <output-dir> `
  --input-dir <output-dir>\subagent_outputs `
  --output <output-dir>\label_preservation_audit.json
```

Gate:

```text
violations = 0
stage_counts.query_planning = source_rows
stage_counts.evidence_governance = source_rows
```

This check is mandatory because the 100-row pilot repair pass caught one worker
changing `retrieval_obligation`; it was restored before the pilot was accepted.
It also verifies the numeric `*_id` label fields, because a later batch repair
showed that preserving label names without the IDs is not sufficient for
training the heads.

## Deferred Semantic QA

Do not run this after every 100-row wave by default. Run it later as one
consolidated QA pass across accumulated mechanically clean waves.

When the consolidated QA pass starts, spawn GPT-5.4 QA workers using:

```text
docs/prompts/FITZ_GOV_SAGE_SEMANTIC_QA_WORKER.md
```

Each QA worker writes one report:

```text
<output-dir>/semantic_qa/pack_0000.semantic_qa.jsonl
```

Then summarize:

```powershell
.venv\Scripts\python.exe scripts\summarize_fitz_gov_sage_semantic_qa.py `
  --workpack-dir <output-dir> `
  --qa-dir <output-dir>\semantic_qa `
  --output <output-dir>\semantic_qa\semantic_qa_summary.json `
  --repair-output <output-dir>\semantic_qa\repair_manifest.jsonl
```

Semantic QA verdicts:

```text
accept = row is usable as-is
repair = row must be manually edited before it can count
```

Common repair reasons from the clean pilot:

```text
ABSTAIN answer leakage
unsupported auxiliary context
fake missing-evidence scaffolding
scope erasure
metadata/content mismatch
```

## Repair

Only run repair after the deferred semantic QA pass identifies rows that need
repair. Spawn GPT-5.4 repair workers using:

```text
docs/prompts/FITZ_GOV_SAGE_REPAIR_WORKER.md
```

Each repair worker gets:

```text
the generated output JSONL file
the source workpack JSON
the semantic QA report
the exact source_id list it is allowed to edit
```

After repair, rerun:

```text
structural audit
shape audit
label/scalar preservation check
focused semantic QA on every repaired source_id
```

The focused semantic QA report for a repaired batch should live under:

```text
<output-dir>/semantic_qa/final_repaired_rows_semantic_check.jsonl
```

## Row State Definitions

A row counts as **mechanically clean** after:

```text
structural audit passes
shape audit passes
label/scalar preservation passes
```

A row counts as **QA accepted** only after:

```text
mechanical gates pass
semantic QA says accept, or repair was completed and post-repair gates pass
```

Mechanically clean rows may be accumulated for throughput. Do not use them for
final training/publication until the deferred semantic QA pass is complete.

## Fixing The Original 10k

The old `fitz_gov_sage_v1_workpacks` data is mostly source-preserving:

```text
exact source-context-list copies: 90.75%
changed evidence packs:           9.25%
mean contexts:                    2.04
short_pack_1_3:                   9,980 / 10,000
```

Do not edit it in place. Keep it as the source-preserving control.

Create replacement messy-pack batches from the same source selection:

```text
data/fitz_gov_sage_v1_messy_repair_batch_0000
data/fitz_gov_sage_v1_messy_repair_batch_0001
...
```

Once enough repaired-clean batches exist, materialize a separate dataset:

```text
data/fitz_gov_sage_v1_messy_repair
```

That dataset, not the old source-preserving v1, is the candidate for the next
serious `pyrrho-sage-nano` training run.
