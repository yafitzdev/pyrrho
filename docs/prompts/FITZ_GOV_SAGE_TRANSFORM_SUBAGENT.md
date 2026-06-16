# fitz-gov-sage v1 Subagent Prompt

You are transforming clean fitz-gov V10 source rows into production-shaped `fitz-gov-sage` rows for Pyrrho.

Your job is not to relabel. Your job is to preserve the source truth while rewriting the evidence presentation into the kind of ranked, messy evidence pack that fitz-sage gives Pyrrho at runtime.

## Input

You receive a JSON workpack with `items`. Each item contains:

- `source_id`
- `query`
- `contexts`
- `context_features`
- `labels`
- `scalar_targets`
- optional `evidence_chain`
- optional `grounding_targets`

## Output

Return JSONL. Produce exactly **two** JSON objects per input item:

1. a `query_planning` row
2. an `evidence_governance` row

Do not wrap the JSONL in markdown.

## Shared Required Fields

Every output row must include:

```json
{
  "source_id": "original source_id",
  "id": "source_id::sage::query_planning or source_id::sage::evidence_governance",
  "stage": "query_planning or evidence_governance",
  "query": "unchanged source query",
  "query_text": "Question: unchanged source query",
  "labels": {},
  "scalar_targets": {},
  "sage_metadata": {}
}
```

Preserve source labels. Do not invent a new governance class.

## query_planning Row

Purpose: train Pyrrho's pre-retrieval planning heads.

Required shape:

```json
{
  "stage": "query_planning",
  "text": "Question: <query>",
  "contexts": [],
  "labels": {
    "label_id": -1,
    "query_contract": "<source label>",
    "route": "<source label>",
    "taxonomy_pattern": null,
    "retrieval_action": "<source label>",
    "gap_type": "<source label>",
    "answerability_shape": "<source label>",
    "retrieval_modality": "<source label>",
    "retrieval_obligation": "<source label or null>"
  },
  "scalar_targets": {}
}
```

Use `label_id: -1` and `taxonomy_pattern: null` because governance and taxonomy are not trained from query-only text.

## evidence_governance Row

Purpose: train Pyrrho's post-retrieval governance and evidence-contract behavior on runtime-shaped evidence.

Required shape:

```json
{
  "stage": "evidence_governance",
  "text": "Question: <query>\n\nSources:\n[1] ...",
  "contexts": ["ranked evidence text 1", "ranked evidence text 2"],
  "pack_metadata": {
    "pack_shape": "short_pack_1_3 | retrieval_pack_4_7 | retrieval_pack_8_10 | long_pack_11_plus",
    "items": [
      {
        "rank": 1,
        "role": "supporting | conflicting | stale | partial | distractor",
        "modality": "unstructured_text | structured_table | code | configuration | log_trace | pdf_layout",
        "anchor": "short source identifier",
        "why_present": "one short reason this item appears in the pack"
      }
    ]
  },
  "labels": {
    "label": "ABSTAIN | DISPUTED | TRUSTWORTHY",
    "label_id": 0,
    "query_contract": "<source label>",
    "route": "<source label>",
    "taxonomy_pattern": "<source label>",
    "retrieval_action": "<source label>",
    "gap_type": "<source label>",
    "answerability_shape": "<source label>",
    "retrieval_modality": "<source label>",
    "retrieval_obligation": "<source label or null>"
  },
  "scalar_targets": "<source scalar_targets>"
}
```

## Evidence Pack Rules

Make the pack realistic:

- Keep the original correct/supporting contexts.
- Keep concrete names, dates, IDs, product names, function names, table keys, policy names, and source anchors from the source row whenever they are relevant.
- Add plausible distractors only when they are label-consistent.
- For `TRUSTWORTHY`, the final pack must contain enough evidence to answer.
- For `DISPUTED`, the final pack must contain material contradiction, not just two wordings.
- For `ABSTAIN`, the final pack must still lack the required answer. Represent missing evidence by absence, not by adding a fake "missing item" context.
- For temporal/final/latest questions, include stale or earlier evidence only if the correct final evidence is clearly present or clearly missing according to the source label.
- For mixed obligations, include companion evidence types where possible: prose plus table, prose plus code, table plus config, code plus changelog, log plus config.
- Every `pack_metadata.items[]` entry must correspond to a real context entry. Do not include a metadata item for evidence that is absent.

Do not:

- copy fitz-sage benchmark entity names or documents
- anonymize source evidence with placeholders like `[entity]`, `[date]`, `[project]`, `[product]`, or `[system]`
- add synthetic context strings such as "Missing item", "missing evidence", "the pack lacks", or "does not include the decisive evidence"
- change the query
- change the governance label
- add facts that contradict the intended label
- output explanations outside JSONL

## Quality Standard

The output should look like a retrieval system's ranked evidence pack, not like a clean benchmark row. The model must learn how to judge imperfect evidence packs while Pyrrho remains the owner of pre-retrieval planning and post-retrieval governance.
