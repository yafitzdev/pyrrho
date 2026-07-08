"""Build the Hugging Face README.md for the current Pyrrho v2 encoder release.

The card is intentionally product-facing first and implementation-facing
second: explain where Pyrrho sits in a RAG pipeline, then document the native
v2 heads and decoding contract.

Run from project root:
    python scripts/build_model_card.py --output models/pyrrho-v2-nano-g1/README.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Where to write README.md")
    parser.add_argument("--model-id", default="yafitzdev/pyrrho-v2-nano-g1")
    parser.add_argument("--dataset-id", default="yafitzdev/fitz-gov-v2")
    parser.add_argument("--base-model", default="answerdotai/ModernBERT-base")
    parser.add_argument("--training-rows", type=int, default=41358)
    parser.add_argument("--snapshot-id", default="fitz_gov_v2_41358_20260703")
    return parser.parse_args()


def build_card(args: argparse.Namespace) -> str:
    model_name = args.model_id.split("/")[-1]
    dataset_name = args.dataset_id.split("/")[-1]
    return f"""---
license: cc-by-nc-4.0
base_model: {args.base_model}
library_name: transformers
pipeline_tag: text-classification
language:
  - en
tags:
  - rag
  - governance
  - hallucination-detection
  - evidence-verification
  - pyrrho
  - fitz-gov-v2
  - modernbert
  - multi-label-classification
datasets:
  - {args.dataset_id}
metrics:
  - accuracy
  - f1
---

# {model_name}

`{model_name}` is a small local RAG planning and governance co-processor. It can
run before retrieval on a user query and after retrieval on the query plus
source passages. The post-retrieval pass returns whether the evidence is
`SUFFICIENT`, `DISPUTED`, or `INSUFFICIENT` before an answer is generated.

It is not an answer generator, not a retriever, and not an open-world fact
checker. It sits between retrieval and generation, or beside a retrieval
pipeline as a fast evidence-quality layer, so downstream systems can answer,
show a dispute, retry retrieval, or ask for missing evidence.

Compared with the older Pyrrho v1 line, v2 exposes a smaller native head shape:
one evidence verdict, one failure reason, and two compact multi-label metadata
heads. The same model owns the `fitz-sage` pre-retrieval query-planning pass and
the post-retrieval evidence-governance pass.

## Native V2 Heads

| Head | Labels / values | Intended use |
|---|---|---|
| `evidence_verdict` | `SUFFICIENT`, `DISPUTED`, `INSUFFICIENT` | Post-retrieval evidence sufficiency and conflict decision. |
| `failure_mode` | `none`, `unresolved_conflict`, `missing_or_incomplete_evidence`, `wrong_scope_or_version`, `ambiguous_request` | Actionable reason when evidence is disputed or insufficient. |
| `retrieval_intents` | `needs_lookup`, `needs_temporal_resolution`, `needs_comparison_or_set`, `needs_broad_coverage` | Pre-retrieval planning and post-retrieval retry metadata. |
| `evidence_kinds` | `needs_text`, `needs_table_or_record`, `needs_code_or_symbol`, `needs_config_or_setting`, `needs_log_or_run_result`, `needs_document_layout` | Evidence-surface metadata for routing, audit, and missing-source hints. |

## Output Contract

The raw Hugging Face model output is an 18-logit vector. It is not one flat
softmax. Decode it by head:

| Logit slice | Head | Decoding |
|---|---|---|
| `0:3` | `evidence_verdict` | softmax over `INSUFFICIENT`, `DISPUTED`, `SUFFICIENT` |
| `3:8` | `failure_mode` | softmax over the five failure labels |
| `8:12` | `retrieval_intents` | sigmoid multi-label scores |
| `12:18` | `evidence_kinds` | sigmoid multi-label scores |

Most integrations should expose structured objects derived from those logits.
For `[PYRRHO_PRE]`, use only `retrieval_intents` and `evidence_kinds`. For
`[PYRRHO_POST]`, use all four heads.

| Field | Meaning |
|---|---|
| `evidence_verdict.final_label` | Final v2 verdict: `SUFFICIENT`, `DISPUTED`, or `INSUFFICIENT`. |
| `evidence_verdict.probabilities` | Softmax probability distribution over the three verdict labels. |
| `failure_mode.final_label` | Most likely failure reason, or `none` for sufficient evidence. |
| `retrieval_intents.final_labels` | Intent labels above the configured sigmoid threshold. |
| `evidence_kinds.final_labels` | Evidence-kind labels above the configured sigmoid threshold. |
| `confidence` | Probability or score assigned to the selected label. |

Example normalized output:

```json
{{
  "schema_version": "pyrrho_v2_prediction",
  "evidence_verdict": {{
    "final_label": "DISPUTED",
    "confidence": 0.86,
    "probabilities": {{
      "INSUFFICIENT": 0.08,
      "DISPUTED": 0.86,
      "SUFFICIENT": 0.06
    }}
  }},
  "failure_mode": {{
    "final_label": "unresolved_conflict",
    "confidence": 0.81
  }},
  "retrieval_intents": {{
    "final_labels": ["needs_comparison_or_set"],
    "scores": {{
      "needs_comparison_or_set": 0.77
    }}
  }},
  "evidence_kinds": {{
    "final_labels": ["needs_text", "needs_table_or_record"],
    "scores": {{
      "needs_text": 0.91,
      "needs_table_or_record": 0.63
    }}
  }}
}}
```

The model does not generate answers, citations, source spans, retrieval results,
or natural-language explanations. It classifies and scores query intent before
retrieval and the `(query, retrieved_contexts)` evidence state after retrieval.

## Intended Use

Use this model when a RAG or retrieval system needs fast local signals about:

- whether retrieved evidence is enough to answer,
- whether retrieved evidence contains an unresolved conflict,
- why evidence is insufficient or disputed,
- whether another retrieval pass should focus on lookup, time, comparison, or broad coverage,
- which source surface appears relevant or missing,
- how to log governance decisions for later audit.

This model is not intended to verify facts outside the provided sources, replace
a retriever, write answers, or replace human review in high-stakes settings.

## Input Format

Pre-retrieval query planning:

```text
[PYRRHO_PRE]
Question: <user query>
```

Post-retrieval evidence governance:

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
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "{args.model_id}"

VERDICT_LABELS = ["INSUFFICIENT", "DISPUTED", "SUFFICIENT"]
FAILURE_LABELS = [
    "none",
    "unresolved_conflict",
    "missing_or_incomplete_evidence",
    "wrong_scope_or_version",
    "ambiguous_request",
]
INTENT_LABELS = [
    "needs_lookup",
    "needs_temporal_resolution",
    "needs_comparison_or_set",
    "needs_broad_coverage",
]
KIND_LABELS = [
    "needs_text",
    "needs_table_or_record",
    "needs_code_or_symbol",
    "needs_config_or_setting",
    "needs_log_or_run_result",
    "needs_document_layout",
]

query = "Has the company achieved profitability?"
contexts = [
    "The company posted net income of $4 million in Q2.",
    "The company recorded a quarterly loss of $12 million in Q3.",
]
text = "[PYRRHO_POST]\\nQuestion: " + query + "\\n\\nSources:\\n" + "\\n".join(
    f"[{{i}}] {{context}}" for i, context in enumerate(contexts, start=1)
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()

inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
with torch.no_grad():
    logits = model(**inputs).logits[0]

verdict_probs = torch.softmax(logits[0:3], dim=-1)
failure_probs = torch.softmax(logits[3:8], dim=-1)
intent_scores = torch.sigmoid(logits[8:12])
kind_scores = torch.sigmoid(logits[12:18])

verdict = VERDICT_LABELS[int(verdict_probs.argmax())]
failure = FAILURE_LABELS[int(failure_probs.argmax())]
intents = [
    label for label, score in zip(INTENT_LABELS, intent_scores)
    if float(score) >= 0.5
]
kinds = [
    label for label, score in zip(KIND_LABELS, kind_scores)
    if float(score) >= 0.5
]

print(verdict)
print(failure)
print(intents)
print(kinds)
```

## CPU ONNX

The repository includes both FP32 and INT8 ONNX exports:

- `model.onnx`
- `model_quantized.onnx`

`fitz-sage` loads `model.onnx` by default for governance accuracy. The quantized
graph is included for integrations that explicitly trade some accuracy margin
for smaller artifacts. Decode the resulting 18 logits using the same slices
shown above.

## Evaluation

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

The live benchmark result is the practical integration target. The fixed-evidence
suite is a minimal sanity check for the governance head.

## Training Data

| Field | Value |
|---|---|
| Dataset | [`{dataset_name}`](https://huggingface.co/datasets/{args.dataset_id}) |
| Clean active training rows | {args.training_rows:,} |
| Training source pointer | `{args.snapshot_id}` |
| Base model | `{args.base_model}` |
| Seed | 42 |

## Artifacts

This repository contains:

- `model.safetensors`: Transformers checkpoint
- `model.onnx`: FP32 ONNX export
- `model_quantized.onnx`: INT8 dynamic ONNX export
- tokenizer/config files
- `manifest.json`: release metadata

## Limitations

1. **Evidence-bounded judgment.** Pyrrho judges only the retrieved evidence it
   is given. It does not retrieve new evidence or verify claims against outside
   knowledge.
2. **English synthetic training data.** The v2 dataset is English synthetic RAG
   governance data. Multilingual behavior is not established.
3. **Metadata heads are policy signals, not formal proof.** `retrieval_intents`
   and `evidence_kinds` are useful routing and audit hints. They do not prove
   SQL correctness, code execution behavior, or complete corpus coverage.
4. **RAG integration still matters.** Bad retrieval can produce bad evidence
   packs. Pyrrho can flag insufficiency or conflict, but it cannot recover
   source material that was never retrieved.

## License

CC BY-NC 4.0. Free for research, evaluation, and personal use; commercial use
requires a separate license.
"""


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_card(args), encoding="utf-8", newline="\n")
    print(f"Wrote model card -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
