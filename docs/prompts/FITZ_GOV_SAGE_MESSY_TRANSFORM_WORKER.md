# fitz-gov-sage Manual Messy-Pack Transform Worker Prompt

Use this prompt when spawning a GPT-5.4 worker to manually transform one
workpack into fitz-sage-shaped rows.

Replace:

- `{WORKER_NAME}`
- `{WORKPACK_PATH}`
- `{OUTPUT_PATH}`
- `{EXPECTED_INPUT_ITEMS}`
- `{EXPECTED_OUTPUT_ROWS}`

```text
You are {WORKER_NAME} for a pyrrho fitz-gov-sage messy-pack batch. You are not alone in the codebase. Do not revert or modify any files outside your assigned output file.

Task: manually transform exactly the items in {WORKPACK_PATH} into fitz-sage-shaped JSONL rows.

Write exactly one file:
{OUTPUT_PATH}

Expected input items: {EXPECTED_INPUT_ITEMS}. Expected output rows: {EXPECTED_OUTPUT_ROWS}.

Hard rules:
- Do this by hand/reasoning, not with a script and not by mechanically copying input rows.
- Produce exactly 2 JSONL rows per input item: one query_planning row and one evidence_governance row.
- Preserve source_id, query, labels, scalar_targets, and every numeric label *_id field. Do not relabel.
- query_planning row: contexts=[], text="Question: <query>", labels copied from source labels except label_id=-1, taxonomy_pattern=null, and taxonomy_pattern_id=null or -1, scalar_targets={}. Keep pre-retrieval label IDs such as query_contract_id, route_id, retrieval_action_id, gap_type_id, answerability_shape_id, retrieval_modality_id, and retrieval_obligation_id. This row trains pre-retrieval planning only.
- evidence_governance row: build a realistic fitz-sage runtime evidence pack. Normally 4-9 contexts. Do not copy the source context list unchanged. Add/split/merge/reorder/rewrite around the source truth. Include plausible distractors, stale evidence, partial evidence, or companion modalities when label-consistent.
- For TRUSTWORTHY, include enough decisive evidence to answer.
- For DISPUTED, include visible material contradiction, not vague ambiguity.
- For ABSTAIN, keep decisive evidence absent; represent missing evidence by absence, never by a fake "missing evidence" context.
- Include pack_metadata with pack_shape retrieval_pack_4_7 or retrieval_pack_8_10 where possible, and one pack_metadata.items entry per context with rank, role, modality, anchor, why_present.
- Do not use placeholders like [entity], [date], [project], [product], [system], [source], or [document].
- Do not leak internal source_id strings inside contexts, anchors, or why_present text. Use natural document/table/log identifiers instead.
- Do not copy fitz-sage benchmark entity names or documents.
- Do not output explanations outside JSONL.
- If blocked, write no partial file and report the blocker.
```
