# pyrrho

Pyrrho is a CPU-friendly planning and governance encoder for RAG systems. It
can run before retrieval on the user query and after retrieval on the query plus
evidence. The post-retrieval pass decides whether evidence is sufficient,
disputed, or insufficient; both passes expose retrieval/evidence metadata that a
RAG runtime can act on.

Current release:

| Artifact | Link |
|---|---|
| Model | [`yafitzdev/pyrrho-v2-nano-g1`](https://huggingface.co/yafitzdev/pyrrho-v2-nano-g1) |
| Dataset | [`yafitzdev/fitz-gov-v2`](https://huggingface.co/datasets/yafitzdev/fitz-gov-v2) |
| Runtime consumer | [`fitz-sage`](https://github.com/yafitzdev/fitz-sage) |

## Current Model

`pyrrho-v2-nano-g1` is a ModernBERT-base classifier with four native v2 heads:

| Head | Labels |
|---|---|
| `evidence_verdict` | `SUFFICIENT`, `DISPUTED`, `INSUFFICIENT` |
| `failure_mode` | `none`, `unresolved_conflict`, `missing_or_incomplete_evidence`, `wrong_scope_or_version`, `ambiguous_request` |
| `retrieval_intents` | `needs_lookup`, `needs_temporal_resolution`, `needs_comparison_or_set`, `needs_broad_coverage` |
| `evidence_kinds` | `needs_text`, `needs_table_or_record`, `needs_code_or_symbol`, `needs_config_or_setting`, `needs_log_or_run_result`, `needs_document_layout` |

The model is used by `fitz-sage` in two passes:

- `[PYRRHO_PRE]`: query-only planning using `retrieval_intents` and
  `evidence_kinds`.
- `[PYRRHO_POST]`: evidence governance using all four heads.

`fitz-sage` bridges the native v2 verdict into its runtime `AnswerMode`, while
the Pyrrho metadata keeps the native v2 heads visible.

## Input Format

Pre-retrieval input:

```text
[PYRRHO_PRE]
Question: <user query>
```

Post-retrieval input:

```text
[PYRRHO_POST]
Question: <user query>

Sources:
[1] <retrieved source text>
[2] <retrieved source text>
```

## Quick Start

```python
import torch
from transformers import AutoModelForSequenceClassification, PreTrainedTokenizerFast

model_id = "yafitzdev/pyrrho-v2-nano-g1"
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id).eval()

text = """[PYRRHO_POST]
Question: Has the company achieved profitability?

Sources:
[1] The company posted net income of $4 million in Q2.
[2] The company recorded a quarterly loss of $12 million in Q3.
"""

encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
with torch.no_grad():
    logits = model(**encoded).logits[0]

verdict_labels = ["INSUFFICIENT", "DISPUTED", "SUFFICIENT"]
failure_labels = [
    "none",
    "unresolved_conflict",
    "missing_or_incomplete_evidence",
    "wrong_scope_or_version",
    "ambiguous_request",
]
intent_labels = [
    "needs_lookup",
    "needs_temporal_resolution",
    "needs_comparison_or_set",
    "needs_broad_coverage",
]
kind_labels = [
    "needs_text",
    "needs_table_or_record",
    "needs_code_or_symbol",
    "needs_config_or_setting",
    "needs_log_or_run_result",
    "needs_document_layout",
]

verdict = verdict_labels[int(torch.softmax(logits[0:3], dim=-1).argmax())]
failure = failure_labels[int(torch.softmax(logits[3:8], dim=-1).argmax())]
intents = [
    label
    for label, score in zip(intent_labels, torch.sigmoid(logits[8:12]))
    if float(score) >= 0.5
]
kinds = [
    label
    for label, score in zip(kind_labels, torch.sigmoid(logits[12:18]))
    if float(score) >= 0.5
]

print(verdict, failure, intents, kinds)
```

## Training Snapshot

| Field | Value |
|---|---|
| Dataset | `fitz-gov-v2` |
| Clean active training rows | 41,358 |
| Training source pointer | `fitz_gov_v2_41358_20260703` |
| Base model | `answerdotai/ModernBERT-base` |
| Seed | 42 |

## Evaluation Snapshot

Held-out post-retrieval eval from
`outputs/modernbert_base_v2_dual_from_g1_41358_active_20260704_seed42`:

| Metric | Value |
|---|---:|
| overall score | 0.9471 |
| verdict accuracy | 0.9703 |
| false sufficient rate | 0.0484 |
| failure accuracy | 0.9567 |
| retrieval exact match | 0.8308 |
| retrieval macro F1 | 0.9277 |
| evidence-kind exact match | 0.9809 |
| evidence-kind macro F1 | 0.9950 |

Held-out pre-retrieval query eval:

| Metric | Value |
|---|---:|
| retrieval exact match | 0.8248 |
| retrieval macro F1 | 0.9266 |
| evidence-kind exact match | 0.9637 |
| evidence-kind macro F1 | 0.9873 |

Fitz-sage release-candidate checks:

| Benchmark | Result |
|---|---:|
| balanced fixed-evidence governance sanity suite | 120/120 |
| live fitz-sage benchmark | 97/120 |
| core | 19/20 |
| holdout | 43/50 |
| holdout2 | 35/50 |

## Repository Layout

```text
pyrrho/
├── scripts/              # training, export, evaluation, and hub utilities
├── src/pyrrho/           # package helpers and manifests
├── configs/              # training configs
├── docs/                 # development notes and historical planning docs
└── outputs/              # local experiment outputs, ignored unless promoted
```

## Related Projects

- [`fitz-sage`](https://github.com/yafitzdev/fitz-sage): RAG runtime that uses Pyrrho for evidence governance.
- [`fitz-gov-v2`](https://huggingface.co/datasets/yafitzdev/fitz-gov-v2): active v2 training dataset.
- [`pyrrho-v2-nano-g1`](https://huggingface.co/yafitzdev/pyrrho-v2-nano-g1): current v2 model release.

## License

CC BY-NC 4.0. Free for research, evaluation, and personal use; commercial use
requires a separate license.
