# fitz-gov-sage Generation Runbook

This is the exact reproducible procedure for the current base `fitz-gov-sage`
transform path.

## Base State

Use repo commit:

```text
1863c7e Fix sage planning heads and train g1.1 control
```

Keep these generated data directories:

```text
data/fitz_gov_sage_v1
data/fitz_gov_sage_v1_1
data/fitz_gov_sage_v1_workpacks
data/multitask_sage_g1_1_v10_clean_plus_stage
```

Do not use later `fitz_gov_sage_v1_2` or `fitz_gov_sage_v2` artifacts for this
base state.

## Source Data

Clean source prep:

```text
data/multitask_g5_1_v10_repaired
```

Original source selection:

```text
data/fitz_gov_sage_v1_workpacks/source_selection.jsonl
```

Original full workpacks:

```text
data/fitz_gov_sage_v1_workpacks/workpacks/pack_0000.json
...
data/fitz_gov_sage_v1_workpacks/workpacks/pack_0199.json
```

Original subagent outputs:

```text
data/fitz_gov_sage_v1_workpacks/subagent_outputs/pack_0000.jsonl
...
data/fitz_gov_sage_v1_workpacks/subagent_outputs/pack_0199.jsonl
```

Original transform prompt:

```text
docs/prompts/FITZ_GOV_SAGE_TRANSFORM_SUBAGENT.md
```

## Full-Scale Shape

The original v1 generation shape is:

- **10,000** source rows
- **200** workpacks
- **50** source rows per workpack
- **20,000** stage rows expected
- exactly two rows per source row:
  - `query_planning`
  - `evidence_governance`

## Worker Prompt Template

Use one Codex GPT-5.4 worker per workpack. The worker gets exactly one
workpack and exactly one allowed output file.

Template:

```text
You are Worker {N} for a pyrrho fitz-gov-sage pilot. You are not alone in the codebase. Do not revert or modify any files outside your assigned output file.

Task: transform exactly the items in {WORKPACK_PATH} using the contract in C:\Users\yanfi\PycharmProjects\pyrrho\docs\prompts\FITZ_GOV_SAGE_TRANSFORM_SUBAGENT.md.

Write exactly one file: {OUTPUT_PATH}

Rules:
- Produce exactly 2 JSONL rows per input item: query_planning and evidence_governance.
- Expected rows for this pack: {EXPECTED_ROWS}.
- Preserve source_id, query, labels, and scalar_targets according to the prompt.
- Do not write scripts, helper programs, docs, logs, summaries, validation reports, or git changes.
- Do not use generic repeated templates. Build each evidence pack from the actual source row content.
- Do not output anything except the final brief status message after the file is written.
- If blocked, write no partial file and report the blocker.
```

## 100-Row Pilot Procedure

The 2026-06-17 reset pilot used the first **100** rows from the original v1
source selection and split them across six GPT-5.4 workers:

```text
data/fitz_gov_sage_pilot_100_20260617/workpacks/pack_0000.json  17 source rows / 34 output rows
data/fitz_gov_sage_pilot_100_20260617/workpacks/pack_0001.json  17 source rows / 34 output rows
data/fitz_gov_sage_pilot_100_20260617/workpacks/pack_0002.json  17 source rows / 34 output rows
data/fitz_gov_sage_pilot_100_20260617/workpacks/pack_0003.json  17 source rows / 34 output rows
data/fitz_gov_sage_pilot_100_20260617/workpacks/pack_0004.json  16 source rows / 32 output rows
data/fitz_gov_sage_pilot_100_20260617/workpacks/pack_0005.json  16 source rows / 32 output rows
```

The six workers wrote:

```text
data/fitz_gov_sage_pilot_100_20260617/subagent_outputs/pack_0000.jsonl
...
data/fitz_gov_sage_pilot_100_20260617/subagent_outputs/pack_0005.jsonl
```

## Structural Audit

Run:

```powershell
.venv\Scripts\python.exe scripts\audit_fitz_gov_sage_outputs.py `
  --workpack-dir data\fitz_gov_sage_pilot_100_20260617 `
  --input-dir data\fitz_gov_sage_pilot_100_20260617\subagent_outputs `
  --output data\fitz_gov_sage_pilot_100_20260617\subagent_output_audit.json
```

Pilot result:

```json
{
  "files": 6,
  "rows_seen": 200,
  "source_ids_seen": 100,
  "violations": 0
}
```

## Comparison Against Original v1

For the same 100 source IDs, compare the pilot output to:

```text
data/fitz_gov_sage_v1_workpacks/subagent_outputs
```

Observed pilot-vs-baseline match:

```text
same labels rows:      200 / 200
same context rows:     200 / 200
same pack-shape rows:  200 / 200
```

Both the pilot and the original v1 baseline for these 100 IDs have:

```text
evidence rows:                 100
pack_shape short_pack_1_3:     100 / 100
mean contexts per evidence row: 1.98
exact source-context lists:    100 / 100
duplicate contexts:            0
```

## Interpretation

The pilot reproduced the earlier v1 transform behavior. It did not reproduce
the later rejected v2 mechanical-template failure.

Important caveat: for the sampled 100 rows, the earlier v1 behavior is mostly a
source-context-preserving stage transform. It preserves labels and stage masks,
but it does not create larger messy retrieval packs for those rows. That is the
base state we are returning to, not proof that more source-preserving rows will
solve fitz-sage downstream quality.

