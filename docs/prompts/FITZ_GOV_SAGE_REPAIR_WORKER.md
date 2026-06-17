# fitz-gov-sage Repair Worker Prompt

Use this prompt after semantic QA. Give each GPT-5.4 repair worker one output
file and the exact `source_id` list it is allowed to edit.

Replace:

- `{WORKER_NAME}`
- `{OUTPUT_PATH}`
- `{WORKPACK_PATH}`
- `{QA_REPORT_PATH}`
- `{REPAIR_LIST}`

```text
You are {WORKER_NAME} for the pyrrho fitz-gov-sage messy-pack batch. You are not alone in the codebase. Do not revert or modify any files outside your assigned output file.

Task: manually repair only the flagged rows in:
{OUTPUT_PATH}

Use the source workpack for truth:
{WORKPACK_PATH}

Use QA report:
{QA_REPORT_PATH}

Repair only these source_ids:
{REPAIR_LIST}

Hard rules:
- Do not change labels, source_id, query, scalar_targets, or query_planning rows unless a query_planning row is malformed.
- Keep exactly 2 rows per source_id.
- For each repaired evidence_governance row, keep realistic fitz-sage shape: 4-9 contexts, pack_shape retrieval_pack_4_7 or retrieval_pack_8_10, one pack_metadata.items entry per context.
- For ABSTAIN rows, do not add decisive answer evidence or fake missing-evidence scaffolding.
- For DISPUTED rows, preserve visible material contradiction.
- For TRUSTWORTHY rows, keep enough decisive evidence to answer.
- Leave all unflagged rows byte-for-byte as much as practical.
- Do not write summaries, docs, scripts, or extra validation reports.
- Final answer: state only which source_ids you repaired.
```

