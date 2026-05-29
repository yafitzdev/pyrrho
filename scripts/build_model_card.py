"""build_model_card.py — Generate the HuggingFace README.md (model card) for a pyrrho release.

Reads the per-seed `final_metrics.json` and the multi-seed `summary.json`, plus the
optional `eval_report.json` (full per-breakdown), and produces a model card that pins
the validated 3-seed mean ± std numbers as the headline.

Run from project root:
    python scripts/build_model_card.py \\
        --checkpoint outputs/multi_seed/seed_42/checkpoint-730 \\
        --summary outputs/multi_seed/summary.json \\
        --output models/pyrrho-nano-g1/README.md
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from statistics import median

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")


# Hard-coded baseline numbers from fitz-sage v0.11 README.md L340-346
BASELINE = {
    "accuracy": 0.787,
    "recall_abstain": 0.865,
    "recall_disputed": 0.861,
    "recall_trustworthy": 0.700,
    "false_trustworthy_rate": 0.057,
}


def _get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True, help="Path to the checkpoint being shipped")
    p.add_argument("--summary", type=Path, required=True, help="outputs/multi_seed/summary.json (3-seed aggregate)")
    p.add_argument("--eval-report", type=Path, default=None, help="Optional eval_report.json with per-breakdown metrics")
    p.add_argument("--output", type=Path, required=True, help="Where to write README.md")
    p.add_argument("--model-id", type=str, default="yafitzdev/pyrrho-nano-g1")
    p.add_argument("--base-model", type=str, default="answerdotai/ModernBERT-base")
    p.add_argument("--fitz-gov-version", type=str, default="V5.1")
    p.add_argument("--fitz-gov-revision", type=str, default=None)
    p.add_argument("--fitz-gov-commit", type=str, default=None)
    p.add_argument("--dataset-config", type=str, default=None)
    return p.parse_args()


def fmt_mean_std(stat: dict, as_pct: bool = True) -> str:
    m, s = stat["mean"], stat["std"]
    if as_pct:
        return f"{m * 100:.2f} ± {s * 100:.2f}"
    return f"{m:.4f} ± {s:.4f}"


def fmt_delta(new_mean: float, baseline: float, as_pct: bool = True) -> str:
    delta = new_mean - baseline
    if as_pct:
        return f"{delta * 100:+.2f}"
    return f"{delta:+.4f}"


def fitz_gov_commit() -> str:
    """Best-effort: return the git commit of the user's local fitz-gov."""
    repo = Path("../fitz-gov").resolve()
    if not repo.exists():
        return "(fitz-gov directory not accessible from this run; please fill in manually)"
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo), capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "(not available)"


def main() -> int:
    args = parse_args()
    if not args.summary.exists():
        print(f"ERROR: summary not found: {args.summary}", file=sys.stderr)
        return 1

    with args.summary.open("r", encoding="utf-8") as fh:
        summary = json.load(fh)

    seeds = summary.get("seeds", [])
    agg = summary["aggregate"]
    headline_key = "test_calibrated" if "test_calibrated" in agg else "eval_calibrated"
    eval_cal = agg[headline_key]
    uses_test_split = headline_key == "test_calibrated"

    fitz_commit = args.fitz_gov_commit or fitz_gov_commit()
    model_name = args.model_id.split("/")[-1]
    citation_key = "".join(ch if ch.isalnum() else "_" for ch in model_name).strip("_")
    baseline_label = "fitz-sage v0.11 gate/baseline"
    dataset_config = args.dataset_config or ("v7" if uses_test_split else None)
    thresholds = [
        float(_get(row, "metrics", "threshold"))
        for row in summary.get("per_seed", [])
        if _get(row, "metrics", "threshold") is not None
    ]
    if thresholds:
        tau_range = f"{min(thresholds):.2f}-{max(thresholds):.2f}"
        default_tau = median(thresholds)
    else:
        tau_range = "see per-seed metrics"
        default_tau = 0.50

    if uses_test_split:
        if dataset_config == "v8":
            validation_size = "2,459"
            test_size = "2,459"
            train_size = "19,674"
            total_rows = "24,592"
            dataset_composition = "2,980 V6 rows, 7,520 V7 rows, and 14,092 V8 target-50 rows"
            qa_clause = (
                "uses query-grouped leakage-safe train/validation/test splits, exposes canonical SDGP breakdown fields, "
                "has target-50 coverage complete across 483/483 taxonomy cells, and passed stricter second-pass blind-label QA before training"
            )
        else:
            validation_size = "1,050"
            test_size = "1,050"
            train_size = "8,400"
            total_rows = "10,500"
            dataset_composition = "2,980 V6 rows and 7,520 V7 SDGP rows"
            qa_clause = (
                "uses query-grouped leakage-safe train/validation/test splits, exposes canonical SDGP breakdown fields, "
                "and passed blind-label, dedup, and cross-label semantic QA before training"
            )
        split_blurb = (
            f"Validated on [fitz-gov](https://github.com/yafitzdev/fitz-gov) {args.fitz_gov_version} "
            f"default `{dataset_config}` splits. Checkpoint and threshold are selected on the "
            f"{validation_size}-row validation split; headline numbers below are from the separate {test_size}-row held-out test split. "
            f"All numbers are **3-seed mean ± std** across seeds {seeds}."
        )
        training_data = (
            f"Training data: fitz-gov {args.fitz_gov_version} default `{dataset_config}` config, "
            f"published query-grouped splits: train={train_size}, validation={validation_size}, held-out test={test_size}. "
            "The validation split is used for checkpoint and TRUSTWORTHY-threshold selection; the test split is used for the headline release metrics."
        )
        dataset_blurb = (
            f"This model is trained and evaluated on [**fitz-gov {args.fitz_gov_version}**](https://github.com/yafitzdev/fitz-gov), "
            f"a {total_rows}-row RAG governance benchmark. The default {dataset_config.upper()} config combines {dataset_composition}, "
            f"{qa_clause}."
        )
        known_limitations = f"""1. **Not a generator.** This is a classification head. It decides whether retrieved evidence supports answering; it does not write the answer.

2. **Low-n breakdowns should not be overread.** {args.fitz_gov_version} reports canonical SDGP axes only: `taxonomy.pattern`, `taxonomy.cell_id`, `routing.expert_fired`, and `meta.difficulty`. The headline test split is large ({test_size} rows), but some taxonomy cells are still small; use aggregate metrics before drawing product conclusions from a single cell.

3. **English-only benchmark.** fitz-gov is English-only, so multilingual governance is not claimed for this release."""
    else:
        split_blurb = (
            f"Validated on the [fitz-gov](https://github.com/yafitzdev/fitz-gov) {args.fitz_gov_version} eval split "
            f"(584 cases, stratified 20% hold-out from `tier1_core`). All numbers are **3-seed mean ± std** across seeds {seeds}."
        )
        training_data = (
            f"Training data: fitz-gov {args.fitz_gov_version} `tier1_core`, stratified 80/20 split by `(label, difficulty)` for train/eval. "
            "The 60-case `tier0_sanity` set is held out separately as a noise-prone diagnostic."
        )
        dataset_blurb = (
            f"This model is trained and evaluated on [**fitz-gov {args.fitz_gov_version}**](https://github.com/yafitzdev/fitz-gov), "
            "a 2,980-case benchmark for RAG governance (epistemic honesty). The eval split (584 cases) is a stratified 20% hold-out from "
            "`tier1_core` (2,920 cases, 62.7% hard difficulty, 17 domains, 113+ subcategories)."
        )
        known_limitations = """1. **Multi-source-convergence cases can be misclassified as DISPUTED.** When multiple authoritative sources state the same fact with slight numerical variation that falls within measurement tolerance (e.g., 4 climate agencies citing 1.09-1.20 degrees C of warming, or NIST and IUPAC both giving the speed of light), the model occasionally classifies the case as DISPUTED with high confidence. On the relevant fitz-gov subcategory (`multi_source_convergence`, n=7) the error rate is ~57%. A v2 release with augmented training data targeting this pattern is planned.

2. **Short, direct factual contexts can trigger over-abstention.** Smoke-test example: query *"When was the iPhone released?"* + a single-sentence context confirming June 29, 2007 -> predicted `ABSTAIN` with P(ABSTAIN)=0.92. The model was trained on 62.7% hard tier1 cases (rich methodological contexts), so it underweights the short-clean-answer pattern. Production RAG chunks (typically 200-500 chars) are tier1-like and largely unaffected."""

    revision_line = f"\nfitz-gov HF revision/tag: `{args.fitz_gov_revision}`\n" if args.fitz_gov_revision else ""

    card = f"""---
license: cc-by-nc-4.0
library_name: transformers
pipeline_tag: text-classification
language:
  - en
base_model: {args.base_model}
tags:
  - rag
  - governance
  - hallucination-detection
  - epistemic-honesty
  - classification
  - fitz-gov
  - pyrrho
datasets:
  - yafitzdev/fitz-gov
metrics:
  - accuracy
  - f1
  - false-trustworthy-rate
---

# {model_name}

> Decide whether your retrieved sources support a confident answer, contradict each other, or simply don't contain it — **without an LLM call**.

This is a fine-tune of [`{args.base_model}`]({"https://huggingface.co/" + args.base_model}) on [fitz-gov](https://github.com/yafitzdev/fitz-gov) {args.fitz_gov_version} for **3-class RAG governance classification**: given a `(query, retrieved contexts)` pair, predicts one of:

| Verdict | Meaning |
|---|---|
| `ABSTAIN` | The sources do not contain enough information to answer. |
| `DISPUTED` | The sources contradict each other on the answer. |
| `TRUSTWORTHY` | The sources consistently and sufficiently support an answer. |

A drop-in replacement for the constraint+sklearn governance pipeline in [fitz-sage](https://github.com/yafitzdev/fitz-sage). Single forward pass, ~30 ms on CPU after INT8 ONNX quantization, no external LLM dependency.

---

## Results

{split_blurb}

| Metric | {model_name} | {baseline_label} | Δ |
|---|---|---|---|
| Overall accuracy (calibrated) | **{fmt_mean_std(eval_cal['accuracy'])}%** | {BASELINE['accuracy'] * 100:.1f}% | **{fmt_delta(eval_cal['accuracy']['mean'], BASELINE['accuracy'])}** |
| False-trustworthy rate (safety) | **{fmt_mean_std(eval_cal['false_trustworthy_rate'])}%** | {BASELINE['false_trustworthy_rate'] * 100:.1f}% | **{fmt_delta(eval_cal['false_trustworthy_rate']['mean'], BASELINE['false_trustworthy_rate'])}** (safer) |
| Trustworthy recall | **{fmt_mean_std(eval_cal['recall_trustworthy'])}%** | {BASELINE['recall_trustworthy'] * 100:.1f}% | **{fmt_delta(eval_cal['recall_trustworthy']['mean'], BASELINE['recall_trustworthy'])}** |
| Disputed recall | **{fmt_mean_std(eval_cal['recall_disputed'])}%** | {BASELINE['recall_disputed'] * 100:.1f}% | **{fmt_delta(eval_cal['recall_disputed']['mean'], BASELINE['recall_disputed'])}** |
| Abstain recall | **{fmt_mean_std(eval_cal['recall_abstain'])}%** | {BASELINE['recall_abstain'] * 100:.1f}% | **{fmt_delta(eval_cal['recall_abstain']['mean'], BASELINE['recall_abstain'])}** |
| Macro F1 | {fmt_mean_std(eval_cal['macro_f1'])}% | n/a | — |

---

## Known limitations

{known_limitations}

---

## Usage

### Direct (transformers)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("{args.model_id}")
model = AutoModelForSequenceClassification.from_pretrained("{args.model_id}").eval()

query = "Has the company achieved profitability?"
contexts = [
    "The company posted its first profitable quarter, with net income of $4 million.",
    "The company recorded a quarterly loss of $12 million, the third consecutive losing quarter.",
]

# Build the input the same way training data was formatted
text = f"Question: {{query}}\\n\\nSources:\\n" + "\\n".join(
    f"[{{i}}] {{c}}" for i, c in enumerate(contexts, start=1)
)

enc = tokenizer(text, truncation=True, max_length=4096, return_tensors="pt")
with torch.no_grad():
    logits = model(**enc).logits[0]
probs = torch.softmax(logits, dim=-1).numpy()
labels = ["ABSTAIN", "DISPUTED", "TRUSTWORTHY"]
print(f"Predicted: {{labels[int(probs.argmax())]}}")
print(f"Probs    : A={{probs[0]:.3f}} D={{probs[1]:.3f}} T={{probs[2]:.3f}}")
```

### CPU-optimized (ONNX + INT8)

For production CPU inference, load the INT8 ONNX variant with `onnxruntime`. The `.onnx.data` file must stay next to `model_quantized.onnx`, so use `snapshot_download` rather than downloading the `.onnx` alone:

```python
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
import numpy as np
import onnxruntime as ort

model_dir = Path(snapshot_download("{args.model_id}"))
tokenizer = AutoTokenizer.from_pretrained(model_dir)
session = ort.InferenceSession(
    str(model_dir / "model_quantized.onnx"),
    providers=["CPUExecutionProvider"],
)

enc = tokenizer(text, truncation=True, max_length=4096, return_tensors="np")
logits = session.run(
    ["logits"],
    {{"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}},
)[0][0]
probs = np.exp(logits - logits.max())
probs = probs / probs.sum()
```

### Calibrated decision rule

The headline numbers above use **threshold calibration** on the TRUSTWORTHY softmax probability. To match the published numbers, fall back from `TRUSTWORTHY` to the runner-up class when `P(TRUSTWORTHY) < tau`. The per-seed selected `tau` varied across runs ({tau_range}); a practical default is `tau = {default_tau:.2f}`.

```python
TAU = {default_tau:.2f}
pred = int(probs.argmax())
if pred == 2 and probs[2] < TAU:  # TRUSTWORTHY id is 2
    pred = int(probs[:2].argmax())   # fall back to runner-up between ABSTAIN/DISPUTED
```

---

## Training

| Hyperparameter | Value |
|---|---|
| Base model | `{args.base_model}` |
| Architecture | ModernBERT (sequence classification head) |
| Labels (3-class) | ABSTAIN (0), DISPUTED (1), TRUSTWORTHY (2) |
| Max sequence length | 4096 tokens |
| Epochs | 5 (with early stopping, patience 2) |
| Per-device batch size | 16 |
| Effective batch size | 16 |
| Learning rate | 5e-5 |
| LR scheduler | cosine, 10% warmup |
| Weight decay | 0.01 |
| Label smoothing | 0.15 |
| Class weights | [2.3, 2.3, 1.0] (asymmetric safety recipe for the false-TRUSTWORTHY gate) |
| Loss | Weighted cross-entropy + label smoothing |
| Selection metric | `ft_penalized_accuracy = accuracy - 3 * max(0, FT - 0.057)` |
| Optimizer | adamw_torch_fused (bf16) |
| Hardware | NVIDIA RTX 5090 (Blackwell sm_120) |
| Training time | ~80–500 s per run depending on GPU contention |

{training_data}

---

## Dataset

{dataset_blurb}

fitz-gov commit at training time: `{fitz_commit}`
{revision_line}

---

## Limitations & intended use

**Intended use:** as a CPU-friendly governance head inside a RAG pipeline that needs to decide when to answer, abstain, or flag a dispute. Drop-in replacement for the constraint+sklearn cascade in [fitz-sage](https://github.com/yafitzdev/fitz-sage).

**Not intended for:**
- Generating answers (this is a classification model, not a generator).
- Token-level hallucination localization (see [LettuceDetect](https://github.com/KRLabsOrg/LettuceDetect) for that — complementary use).
- Languages other than English. fitz-gov is English-only; multilingual variants are a v3+ consideration.

**Safety axis:** the false-trustworthy rate is the production safety metric (a case wrongly classified as `TRUSTWORTHY` is the dangerous error — the system would confidently surface a hallucinated or unsupported answer). Threshold calibration is tuned to keep this rate at or below the fitz-sage baseline (5.7%).

---

## Citation

```bibtex
@misc{{{citation_key}_2026,
  title  = {{ {model_name} }},
  author = {{ Yan Fitzner }},
  year   = {{ 2026 }},
  url    = {{ https://huggingface.co/{args.model_id} }},
}}
```

## License

CC BY-NC 4.0 — see [LICENSE](https://github.com/yafitzdev/pyrrho/blob/main/LICENSE). Free for research, evaluation, and personal use; commercial use requires a separate license.

## Related projects

- [**fitz-sage**](https://github.com/yafitzdev/fitz-sage) — production RAG library that uses this model.
- [**fitz-gov**](https://github.com/yafitzdev/fitz-gov) — the benchmark dataset.
- [**pyrrho**](https://github.com/yafitzdev/pyrrho) — training code and roadmap for the full model family.
"""

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        fh.write(card)
    print(f"Wrote model card -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
