# Fitz-Gov V2 Production Recipe — 2026-06-22

## 2026-06-23 Broad-Schema Update

Active v2 now uses the broad label contract only:

```text
label
route
query_contract
evidence_need
failure_family
```

Removed active heads:

```text
answerability_shape
retrieval_modality
retrieval_obligation
retrieval_action
gap_type
taxonomy_pattern
```

`evidence_need` replaces modality/obligation. `failure_family` replaces the
old gap/taxonomy split. `retrieval_action` belongs to the RAG package, not the
encoder head.

## Naming Decision

The frozen classic line remains:

```text
fitz-gov-v1
```

The existing broad 50k generated corpus that was previously called
`fitz-gov-v2` is demoted to:

```text
fitz-gov-v1.5
```

Status:

```text
private / frozen / ignored for now
do not train pyrrho-v2 from it
do not bulk-relabel it
do not spend more Codex tokens repairing it
```

The next dataset created from scratch is the active:

```text
fitz-gov-v2
```

External Hugging Face privacy/rename work is intended but was not performed in
the session that wrote this recipe.

## Why We Reset

The failed 50k corpus used exactly 8 or 12 contexts per row, balanced every hard
label/dimension, and produced too many rows where the label was not uniquely
recoverable from visible `query + contexts`.

For production RAG governance, that is the wrong base shape. `fitz-sage` usually
retrieves a small evidence pack, not 8-12 contexts by default.

New rule:

```text
Hard does not mean ambiguous.
A valid row must be label-observable from query + contexts.
```

## First Tranche

Create **20,000** rows total:

```text
10,000 base rows
  Purpose: teach core governance ontology.
  Contexts: mostly 1-3, with 1-context easy rows allowed.
  Labels should be obvious to blind labelers.

7,000 production rows
  Purpose: match normal fitz-sage retrieval behavior.
  Contexts: mostly 3-5.
  Include realistic retrieval noise and decoys, but keep the gold label visible.

3,000 hard rows
  Purpose: controlled edge cases.
  Contexts: exactly 6.
  Include temporal/final/current, authority conflict, stale decoys, table/code/config/log cases.
```

Do not create 8-12 context packs. Active v2 uses only 2, 4, or 6 contexts.

## Curriculum Target

If the 20k alpha looks promising, scale to **60,000** rows:

```text
30,000 base rows
20,000 production rows
10,000 hard rows
```

Do not scrape or reuse old datasets for the initial v2. The new dataset should
be created from scratch.

## Difficulty Contract

Difficulty must control context count and ambiguity:

```text
easy:
  2 contexts
  direct answer, clear absence, or clear conflict

medium:
  4 contexts
  one or two decoys
  realistic production retrieval shape

hard:
  6 contexts
  still label-observable after applying written policy
```

## QA Gate

Before training `pyrrho-v2-nano-g1-alpha`, run a small blind QA gate:

```text
sample size: 500-1,000 rows
minimum main-label agreement target: >= 90%
core agreement target: label + query_contract + evidence_need + failure_family
manual review: disagreement rows must show row ambiguity or fixable prompt policy
```

The QA must evaluate whether the row is label-observable, not only whether an
LLM can assign some label.

## First Model

Train only after the 20k set passes QA:

```text
model name: pyrrho-v2-nano-g1-alpha
data: new production-shaped fitz-gov-v2 first tranche
expected role: alpha quality check, not publishable release by default
```
