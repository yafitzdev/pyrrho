# V2 Label Instability Investigation — 2026-06-22

## Question

Why did `fitz-gov-v1` blind labeling behave cleanly while `fitz-gov-v2`
produces large label disagreement? Can `pyrrho-nano-g5.6` help?

## Short Answer

`fitz-gov-v2` is not just a fresh cleaner replacement for v1. It is a much
harder corpus by construction: balanced labels, long evidence packs, every
modality, every obligation family, and many decoy/profile dimensions. The
current v2 rows often encode hard policy-boundary questions where the intended
label is not recoverable enough from visible `query + contexts`.

`pyrrho-nano-g5.6` can help as a cheap local critic, but not as a label oracle.
On the 100-row v2.4 pilot, it agreed with the original v2 label only **56/100**
and with the v2.4 GPT-5.4 majority only **44/98**. Even on the **76** rows where
the three GPT-5.4 annotators were unanimous, g5.6 agreed with only **39**.

## Key Evidence

### v1/g5.6 Shape

Processed active v1/g5.6 corpus:

```text
rows: 67,944
labels: TRUSTWORTHY=23,806, ABSTAIN=23,008, DISPUTED=21,130
avg contexts/row: 3.66
context counts: many 1-3 context rows, plus focused 8/12 context blocks
avg context chars/row: 726
avg query chars: 111
```

The strongest v1-focused addition, the **7,061** V12/g5.6 rows, is not balanced.
Its golden distribution is intentionally narrow and downstream-focused:

```text
TRUSTWORTHY=4,923
DISPUTED=2,012
ABSTAIN=126
taxonomy only: direct_answer, scope_conflict, resolved_candidate_selection,
               factual_contradiction, evidence_absent
```

The accepted V13 round-2 probe matched this distribution exactly at 100 rows:

```text
70 TRUSTWORTHY / 28 DISPUTED / 2 ABSTAIN
```

### v2 Shape

Processed v2 50k corpus:

```text
rows: 50,000
labels: ABSTAIN=16,686, DISPUTED=16,661, TRUSTWORTHY=16,653
avg contexts/row: 10.0
context counts: exactly 8 or 12 for every row
avg context chars/row: 2,154
avg query chars: 171
modalities: ~7.1k each across 7 modalities
query contracts: ~8.3k each across 6 contracts
answerability: ~12.5k each across 4 shapes
```

This means v2 massively increases the share of hard ABSTAIN/DISPUTED cases and
forces coverage of every difficult dimension.

### v2 Blind QA

Work-laptop Sonnet 4.6 blind QA:

```text
5k slice label agreement:      76.7%
remaining 45k label agreement: 73.9%
taxonomy agreement:            11-13%
```

The taxonomy drift is partly taxonomy-boundary noise, but the main label drift
is still too high.

### v2.4 Three-Annotator Governance Pilot

The corrected Stage A governance-only pilot was structurally clean:

```text
100 rows
300/300 valid annotations
0 missing
0 validation errors
```

But it failed reproducibility:

```text
unanimous: 76/100
contentious rows: 49/70 unanimous
controls: 27/30 unanimous
pairwise agreement: 250/300
```

The non-unanimous rows cluster around:

```text
exhaustive "list every" queries
official/current/final source-of-record status
policy-vs-implementation authority
stale logs / corrected logs
source hierarchy and date/version precedence
ABSTAIN vs DISPUTED when partial evidence also conflicts
```

## Interpretation

The problem is not that GPT-5.4 cannot reason. The problem is that v2 asks the
labeler to recover a private governance policy from difficult evidence packs.
Many rows are answerable under one reasonable policy and unsafe under another.

In v1, successful batches were narrow enough that the intended label was usually
observable from the evidence. In v2, many labels are closer to intention labels:
the generation target says what the row was supposed to be, but the visible
pack does not always make that label uniquely recoverable.

## Role of pyrrho-nano-g5.6

Useful:

- cheap no-token critic over v2 rows
- detect rows that look v1-policy-compatible
- identify rows where original labels, GPT labels, Sonnet labels, and g5.6 all
  disagree
- help build triage buckets for the work laptop

Not useful:

- not a final label oracle
- not a replacement for blind labeling
- not a reason to auto-accept labels

Small local probe on the v2.4 100-row set:

```text
g5.6 predictions: TRUSTWORTHY=30, DISPUTED=27, ABSTAIN=43
agrees original v2 label: 56/100
agrees v2.4 majority: 44/98
agrees v2.4 unanimous rows: 39/76
high-confidence g5.6 rows (>=0.8): 54
high-confidence agreement with original: 37/54
high-confidence agreement with v2.4 majority: 30/54
```

## Practical Next Step

Do not scale another full relabel yet. First build a small work-laptop protocol
that separates row quality from label policy:

1. Sample rows by disagreement bucket.
2. For each row, ask the labeler first: "is this row uniquely labelable from
   query + contexts under the written policy?"
3. Only then assign `ABSTAIN / DISPUTED / TRUSTWORTHY`.
4. Accept rows only when labelers agree on both row validity and label.
5. Rewrite or discard rows that are not uniquely labelable.

The goal is not more labels. The goal is to make v2 rows as label-observable as
the successful v1 rows.
