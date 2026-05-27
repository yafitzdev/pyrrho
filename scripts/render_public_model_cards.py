"""Render public-facing pyrrho model cards.

This script keeps Hugging Face cards user-facing: what the model does, how to
use it, what data/metrics support it, and what it should not be used for.
It intentionally avoids internal dataset-schema and generation-pipeline terms.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev

LABEL_IDS = {"ABSTAIN": 0, "DISPUTED": 1, "TRUSTWORTHY": 2}
LABEL_KEYS = {"ABSTAIN": "abstain", "DISPUTED": "disputed", "TRUSTWORTHY": "trustworthy"}

LABELS_MD = """| Label | Meaning |
|---|---|
| `ABSTAIN` | The retrieved sources do not contain enough evidence to answer the question. |
| `DISPUTED` | The retrieved sources conflict on the answer. |
| `TRUSTWORTHY` | The retrieved sources consistently support answering the question. |"""

ENCODER_CODE = """```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

MODEL_ID = "__MODEL_ID__"
LABELS = ["ABSTAIN", "DISPUTED", "TRUSTWORTHY"]
TAU = __TAU__

query = "Has the company achieved profitability?"
contexts = [
    "The company posted its first profitable quarter, with net income of $4 million.",
    "The company recorded a quarterly loss of $12 million, the third consecutive losing quarter.",
]

text = "Question: " + query + "\\n\\nSources:\\n" + "\\n".join(
    f"[{i}] {context}" for i, context in enumerate(contexts, start=1)
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()

inputs = tokenizer(text, truncation=True, max_length=4096, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits[0]
probs = torch.softmax(logits, dim=-1)
probs_np = probs.detach().numpy()

raw_pred = int(probs_np.argmax())
final_pred = raw_pred
used_threshold_fallback = False
if raw_pred == 2 and probs_np[2] < TAU:
    final_pred = int(probs_np[:2].argmax())
    used_threshold_fallback = True

decision = {
    "label": LABELS[final_pred],
    "raw_label": LABELS[raw_pred],
    "logits": dict(zip(LABELS, logits.detach().numpy().tolist(), strict=True)),
    "probabilities": dict(zip(LABELS, probs_np.tolist(), strict=True)),
    "confidence": float(probs_np[final_pred]),
    "trustworthy_probability": float(probs_np[2]),
    "threshold": TAU,
    "used_threshold_fallback": used_threshold_fallback,
}
print(decision)
```"""

ONNX_CODE = """```python
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
import numpy as np
import onnxruntime as ort

MODEL_ID = "__MODEL_ID__"
query = "Has the company achieved profitability?"
contexts = [
    "The company posted its first profitable quarter, with net income of $4 million.",
    "The company recorded a quarterly loss of $12 million, the third consecutive losing quarter.",
]
text = "Question: " + query + "\\n\\nSources:\\n" + "\\n".join(
    f"[{i}] {context}" for i, context in enumerate(contexts, start=1)
)

model_dir = Path(snapshot_download(MODEL_ID))

tokenizer = AutoTokenizer.from_pretrained(model_dir)
session = ort.InferenceSession(
    str(model_dir / "model_quantized.onnx"),
    providers=["CPUExecutionProvider"],
)

inputs = tokenizer(text, truncation=True, max_length=4096, return_tensors="np")
logits = session.run(
    ["logits"],
    {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]},
)[0][0]
probs = np.exp(logits - logits.max())
probs = probs / probs.sum()
```"""

CALIBRATION_CODE = """```python
LABELS = ["ABSTAIN", "DISPUTED", "TRUSTWORTHY"]
TAU = __TAU__
probs_np = probs.detach().cpu().numpy() if hasattr(probs, "detach") else probs

raw_pred = int(probs_np.argmax())
final_pred = raw_pred
used_threshold_fallback = False
if raw_pred == 2 and probs_np[2] < TAU:
    final_pred = int(probs_np[:2].argmax())
    used_threshold_fallback = True

decision = {
    "label": LABELS[final_pred],
    "raw_label": LABELS[raw_pred],
    "probabilities": dict(zip(LABELS, probs_np.tolist(), strict=True)),
    "confidence": float(probs_np[final_pred]),
    "trustworthy_probability": float(probs_np[2]),
    "threshold": TAU,
    "used_threshold_fallback": used_threshold_fallback,
}
```"""

SLM_CODE = """```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen3.5-0.8B"
ADAPTER_ID = "__MODEL_ID__"

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, ADAPTER_ID).eval()
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

system = (
    "You classify whether retrieved sources support answering a question. "
    "Return exactly one label: ABSTAIN, DISPUTED, or TRUSTWORTHY."
)
query = "Has the company achieved profitability?"
contexts = [
    "Posted its first profitable quarter, net income $4M.",
    "Recorded a quarterly loss of $12M, third consecutive losing quarter.",
]
sources = "\\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, start=1))
user = f"Question: {query}\\n\\nSources:\\n{sources}"

messages = [
    {"role": "system", "content": system},
    {"role": "user", "content": user},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
text = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).upper()

parsed_label = None
for label in ("ABSTAIN", "DISPUTED", "TRUSTWORTHY"):
    if label in text:
        parsed_label = label
        break

decision = {
    "label": parsed_label or "ABSTAIN",
    "raw_text": text,
    "fallback_used": parsed_label is None,
}
print(decision)
```"""

ENCODER_OUTPUTS_MD = """The raw Hugging Face model output is a three-class `logits` vector:

| Raw field | Meaning |
|---|---|
| `logits[ABSTAIN]` | Unnormalized score for insufficient evidence. |
| `logits[DISPUTED]` | Unnormalized score for conflicting evidence. |
| `logits[TRUSTWORTHY]` | Unnormalized score for consistently supported evidence. |

Most integrations should expose a structured decision object derived from those logits:

| Field | Meaning |
|---|---|
| `label` | Final calibrated label: `ABSTAIN`, `DISPUTED`, or `TRUSTWORTHY`. |
| `raw_label` | Highest-probability label before threshold calibration. |
| `logits` | Raw score for each label, keyed by label name. |
| `probabilities` | Softmax probability distribution over the three labels. |
| `confidence` | Probability assigned to the final calibrated label. |
| `trustworthy_probability` | `P(TRUSTWORTHY)`, used by the calibrated decision rule. |
| `threshold` | TRUSTWORTHY probability threshold used for calibrated reporting. |
| `used_threshold_fallback` | Whether a low-confidence `TRUSTWORTHY` argmax was changed to `ABSTAIN` or `DISPUTED`. |

Example normalized JSON output:

```json
{
  "label": "DISPUTED",
  "raw_label": "DISPUTED",
  "logits": {
    "ABSTAIN": -1.42,
    "DISPUTED": 2.31,
    "TRUSTWORTHY": 0.18
  },
  "probabilities": {
    "ABSTAIN": 0.02,
    "DISPUTED": 0.86,
    "TRUSTWORTHY": 0.12
  },
  "confidence": 0.86,
  "trustworthy_probability": 0.12,
  "threshold": 0.60,
  "used_threshold_fallback": false
}
```

The encoder does not output generated answers, explanations, citations, source spans, retrieval results, taxonomy/category tags, route IDs, scalar diagnostics, or experimental multitask-head fields. Taxonomy/category fields that appear in evaluation reports are benchmark metadata used for breakdowns; route, taxonomy, and scalar heads are part of the experimental MoE track, not the published nano encoder output contract."""

SLM_OUTPUTS_MD = """The adapter is a label generator. The raw output is generated text, and the practical governance output should be parsed into a small decision object:

| Field | Meaning |
|---|---|
| `label` | Parsed label: `ABSTAIN`, `DISPUTED`, or `TRUSTWORTHY`. |
| `raw_text` | Raw generated text before parsing. |
| `fallback_used` | Whether the parser had to fall back because no valid label was found. |

Example normalized JSON output:

```json
{
  "label": "DISPUTED",
  "raw_text": "DISPUTED",
  "fallback_used": false
}
```

The adapter does not return calibrated probabilities. If probability scores are needed, use an encoder release or add a separate scoring wrapper."""

TEMPLATE_DOC = """# Pyrrho Model Card Template

Use this shape for every pyrrho model card. Keep the card public-facing: no
internal pipeline names, no private dataset/category terminology, no future-plan language,
and no implementation history unless it affects model use.

## Required Shape

1. YAML metadata
   - license, library_name, pipeline_tag, language, base_model, tags, datasets, metrics.
2. Title
   - The public model name only.
3. Model Summary
   - Frame pyrrho as a RAG governance co-processor: an anti-hallucination evidence gate that sits between retrieval and generation, or beside a generator.
   - Explain that the model classifies `(question, retrieved sources)` into `ABSTAIN`, `DISPUTED`, or `TRUSTWORTHY`.
   - State clearly whether the artifact is a full model, ONNX export, or LoRA adapter.
   - State clearly that it is not an answer generator and not an open-world fact checker.
4. Labels
   - A three-row table defining `ABSTAIN`, `DISPUTED`, and `TRUSTWORTHY`.
5. Outputs
   - Distinguish raw artifact outputs from the decision object a product integration should expose.
   - For encoder releases, document raw class logits plus derived fields such as `label`, `raw_label`, `probabilities`, `confidence`, `trustworthy_probability`, `threshold`, and `used_threshold_fallback`.
   - For encoder releases, state explicitly that taxonomy/category tags, route IDs, and scalar diagnostics are not inference outputs; they are evaluation metadata or MoE-only research outputs.
   - For adapters, document raw generated text and parsed label output.
   - Include one compact JSON example of the normalized decision object.
   - Do not claim explanations, citations, spans, auxiliary research-head fields, or retrieval results unless the artifact actually returns them.
6. Intended Use
   - Explain pre-generation answer gating, retrieval retry/escalation, abstention, dispute detection, and evidence-quality logging.
   - Explain the anti-hallucination scope: reducing cases where unsupported or contradictory retrieved evidence gets treated as safe to answer from.
   - Explain what it should not be used for: generating answers, checking facts outside provided sources, span-level hallucination localization, or high-stakes autonomous decisions.
7. Quick Start
   - Minimal runnable code for the main loading path.
   - For encoder releases, include the ONNX CPU path when ONNX artifacts are shipped.
   - For adapters, show base-model plus adapter loading.
8. Decision Rule
   - Explain any TRUSTWORTHY threshold used for reported metrics.
   - Keep it practical: what probability is thresholded, and what fallback is used.
9. Results
   - Name dataset version and split sizes.
   - Use one headline table with rows `OVERALL`, `ABSTAIN`, `DISPUTED`, `TRUSTWORTHY`.
   - Use columns `Recall`, `Precision`, and `False-rate`, reported as 3-seed mean ± std.
   - Define false-rate in plain language. For label rows, it is the share of non-label cases incorrectly predicted as that label. For `TRUSTWORTHY`, this is the safety false-trustworthy rate.
   - Omit F1 from the headline table unless there is a specific reason to include it. If F1 is mentioned, define it as the harmonic mean of precision and recall.
10. Training Data
   - Public dataset name/version, language, total examples, train/validation/test sizes, and leakage-safe grouping if relevant.
   - Avoid internal category or generation-pipeline names.
11. Training Recipe
   - Base model, architecture/method, max length, labels, epochs, batch size, learning rate, loss, class weights/smoothing if used, selection metric, and seeds.
12. Limitations
   - English-only status.
   - Scope of evidence: only provided sources are judged.
   - Known weak cases that affect use.
   - Safety/threshold tradeoff.
13. Citation
   - BibTeX entry for the model.
14. License
   - CC BY-NC 4.0 plus commercial-use note.

## Style Rules

- Prefer plain task language over project-internal names.
- Do not mention internal schema names, cell-coverage machinery, generator batches, QA provider details, or future-plan phases.
- Do not compare to private baselines unless the comparison is necessary to interpret the model.
- Keep metrics precise, but do not overload the card with every diagnostic table. Link to repo docs or artifacts for deep breakdowns.
- Make limitations concrete and operational.
"""


def frontmatter(library: str, pipeline: str, base_model: str, extra_tags: list[str] | None = None) -> str:
    tags = ["rag", "governance", "hallucination-detection", "classification", "fitz-gov", "pyrrho"]
    tags.extend(extra_tags or [])
    tag_lines = "\n".join(f"  - {tag}" for tag in tags)
    return f"""---
license: cc-by-nc-4.0
library_name: {library}
pipeline_tag: {pipeline}
language:
  - en
base_model: {base_model}
tags:
{tag_lines}
datasets:
  - yafitzdev/fitz-gov
metrics:
  - accuracy
  - f1
---"""


def fmt_metric(mean: float, std: float) -> str:
    return f"{mean:.2f} ± {std:.2f}%"


def fmt_values(values: list[float]) -> str:
    if not values:
        return "not reported"
    mu = mean(values)
    sigma = stdev(values) if len(values) > 1 else 0.0
    return fmt_metric(mu * 100, sigma * 100)


def read_label_supports(path: str) -> tuple[dict[str, int], int]:
    counts = {label: 0 for label in LABEL_IDS}
    total = 0
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            label = row.get("label")
            if label is None and "label_id" in row:
                label = next(k for k, v in LABEL_IDS.items() if v == int(row["label_id"]))
            label = str(label).upper()
            if label not in counts:
                raise ValueError(f"unexpected label {label!r} in {path}")
            counts[label] += 1
            total += 1
    return counts, total


def split_metrics(summary: dict, split: str) -> list[dict[str, float]]:
    return [row["metrics"][split] for row in summary["per_seed"]]


def metric_values(metrics: list[dict[str, float]], key: str) -> list[float]:
    return [float(row[key]) for row in metrics if key in row]


def false_rate_values(
    metrics: list[dict[str, float]],
    label: str,
    supports: dict[str, int],
    total: int,
) -> list[float]:
    if label == "TRUSTWORTHY":
        return metric_values(metrics, "false_trustworthy_rate")

    key = LABEL_KEYS[label]
    support = supports[label]
    non_label = total - support
    out = []
    for row in metrics:
        recall = float(row[f"recall_{key}"])
        precision = float(row[f"precision_{key}"])
        if non_label <= 0:
            out.append(0.0)
            continue
        if precision <= 0:
            out.append(0.0)
            continue
        true_positive = recall * support
        false_positive = true_positive * ((1.0 / precision) - 1.0)
        out.append(min(1.0, max(0.0, false_positive / non_label)))
    return out


def render_results_table(rows: list[tuple[str, str, str, str]]) -> str:
    body = "\n".join(f"| {name} | {recall} | {precision} | {false_rate} |" for name, recall, precision, false_rate in rows)
    return f"""| Decision | Recall | Precision | False-rate |
|---|---:|---:|---:|
{body}"""


def results_from_summary(path: str, split: str, label_data: str) -> str:
    summary = json.loads(Path(path).read_text(encoding="utf-8"))
    metrics = split_metrics(summary, split)
    supports, total = read_label_supports(label_data)

    accuracy = metric_values(metrics, "accuracy")
    rows = [
        (
            "`OVERALL`",
            fmt_values(accuracy),
            fmt_values(accuracy),
            fmt_values([1.0 - value for value in accuracy]),
        )
    ]
    for label in ("ABSTAIN", "DISPUTED", "TRUSTWORTHY"):
        key = LABEL_KEYS[label]
        rows.append(
            (
                f"`{label}`",
                fmt_values(metric_values(metrics, f"recall_{key}")),
                fmt_values(metric_values(metrics, f"precision_{key}")),
                fmt_values(false_rate_values(metrics, label, supports, total)),
            )
        )
    return render_results_table(rows)


G1_RESULTS_TABLE = render_results_table(
    [
        ("`OVERALL`", fmt_metric(86.13, 0.86), fmt_metric(86.13, 0.86), fmt_metric(13.87, 0.86)),
        ("`ABSTAIN`", fmt_metric(92.94, 1.11), "not reported", "not reported"),
        ("`DISPUTED`", fmt_metric(94.81, 1.28), "not reported", "not reported"),
        ("`TRUSTWORTHY`", fmt_metric(79.38, 1.64), "not reported", fmt_metric(5.27, 0.21)),
    ]
)


def per_seed_from_summary(path: str) -> list[tuple[int, str, float, float]]:
    summary = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for row in summary["per_seed"]:
        metrics = row["metrics"]
        test = metrics["test_calibrated"]
        out.append(
            (
                row["seed"],
                f"{metrics['threshold']:.2f}",
                test["accuracy"] * 100,
                test["false_trustworthy_rate"] * 100,
            )
        )
    return out


def slm_metrics(path: str) -> list[tuple[str, float, float]]:
    summary = json.loads(Path(path).read_text(encoding="utf-8"))
    agg = summary["aggregate"]["eval"]
    metrics = [
        ("Accuracy", agg["accuracy"]["mean"] * 100, agg["accuracy"]["std"] * 100),
        ("Macro F1", agg["macro_f1"]["mean"] * 100, agg["macro_f1"]["std"] * 100),
        (
            "False-trustworthy rate",
            agg["false_trustworthy_rate"]["mean"] * 100,
            agg["false_trustworthy_rate"]["std"] * 100,
        ),
        ("ABSTAIN recall", agg["recall_abstain"]["mean"] * 100, agg["recall_abstain"]["std"] * 100),
        ("DISPUTED recall", agg["recall_disputed"]["mean"] * 100, agg["recall_disputed"]["std"] * 100),
        (
            "TRUSTWORTHY recall",
            agg["recall_trustworthy"]["mean"] * 100,
            agg["recall_trustworthy"]["std"] * 100,
        ),
        (
            "ABSTAIN precision",
            agg["precision_abstain"]["mean"] * 100,
            agg["precision_abstain"]["std"] * 100,
        ),
        (
            "DISPUTED precision",
            agg["precision_disputed"]["mean"] * 100,
            agg["precision_disputed"]["std"] * 100,
        ),
        (
            "TRUSTWORTHY precision",
            agg["precision_trustworthy"]["mean"] * 100,
            agg["precision_trustworthy"]["std"] * 100,
        ),
    ]
    tier0 = summary["aggregate"].get("tier0_sanity")
    if tier0:
        metrics.extend(
            [
                ("Small sanity-set accuracy", tier0["accuracy"]["mean"] * 100, tier0["accuracy"]["std"] * 100),
                (
                    "Small sanity-set false-trustworthy rate",
                    tier0["false_trustworthy_rate"]["mean"] * 100,
                    tier0["false_trustworthy_rate"]["std"] * 100,
                ),
            ]
        )
    return metrics


def encoder_card(meta: dict) -> str:
    per_seed = "\n".join(
        f"| {seed} | {tau} | {acc:.2f}% | {ft:.2f}% |" for seed, tau, acc, ft in meta.get("per_seed", [])
    )
    per_seed_section = ""
    if per_seed:
        per_seed_section = f"""

Per-seed held-out test results:

| Seed | TRUSTWORTHY threshold | Accuracy | False-trustworthy rate |
|---|---:|---:|---:|
{per_seed}"""

    limitations = "\n".join(f"- {item}" for item in meta["limitations"])
    encoder_code = ENCODER_CODE.replace("__MODEL_ID__", meta["model_id"]).replace(
        "__TAU__", f"{meta['default_tau']:.2f}"
    )
    onnx_code = ONNX_CODE.replace("__MODEL_ID__", meta["model_id"])
    calibration_code = CALIBRATION_CODE.replace("__TAU__", f"{meta['default_tau']:.2f}")

    return f"""{frontmatter("transformers", "text-classification", "answerdotai/ModernBERT-base")}

# {meta["name"]}

{meta["name"]} is a small RAG governance co-processor for anti-hallucination pipelines. It reads a user question plus retrieved source passages and returns an evidence-state decision a RAG application can use before answering: `ABSTAIN`, `DISPUTED`, or `TRUSTWORTHY`.

It is not an answer generator and not an open-world fact checker. It sits between retrieval and generation, or beside a generator as a fast guardrail, to reduce cases where unsupported or contradictory retrieved evidence gets treated as safe to answer from.

## Labels

{LABELS_MD}

## Outputs

{ENCODER_OUTPUTS_MD}

## Intended Use

Use this model when a RAG system needs a fast decision about whether retrieved evidence is good enough to answer. Typical uses include pre-generation answer gating, retrieval retry or escalation triggers, abstention decisions, dispute detection, and logging evidence-quality signals for later review.

This model is not intended to write answers, verify facts outside the provided sources, localize hallucinated spans, or replace human review in high-stakes settings.

## Quick Start

### Transformers

{encoder_code}

### CPU ONNX

The repository includes an INT8 ONNX export for CPU inference. Download the full repository so any external ONNX data files stay next to the `.onnx` file.

{onnx_code}

### Calibrated Decision Rule

The reported metrics use a validation-selected threshold on `P(TRUSTWORTHY)`. If the model's top class is `TRUSTWORTHY` but its probability is below the threshold, fall back to the stronger of `ABSTAIN` and `DISPUTED`.

{calibration_code}

## Results

{meta["eval_text"]}

{meta["results_table"]}

For `OVERALL`, recall and precision are micro-averages; in single-label three-class classification they both equal accuracy. For label rows, false-rate is the share of cases that were not that label but were incorrectly predicted as that label. The `TRUSTWORTHY` false-rate is the main safety metric: it measures cases where the model says `TRUSTWORTHY` even though the sources do not support that decision.

F1 is not shown in the headline table. It is the harmonic mean of precision and recall (`2 * precision * recall / (precision + recall)`), useful as a compact balance score but less direct than the operating metrics above.{per_seed_section}

## Training Data

{meta["data_text"]}

The validation split was used for checkpoint and threshold selection. The held-out test split, when present, was used only for final reporting.

## Training Recipe

| Item | Value |
|---|---|
| Base model | `answerdotai/ModernBERT-base` |
| Architecture | Encoder with sequence-classification head |
| Max sequence length | 4096 tokens |
| Labels | `ABSTAIN`, `DISPUTED`, `TRUSTWORTHY` |
| Epochs | 5 with early stopping |
| Batch size | 16 |
| Learning rate | 5e-5 |
| Scheduler | Cosine with 10% warmup |
| Weight decay | 0.01 |
| Loss | Weighted cross-entropy with label smoothing |
| Class weights | `[2.3, 2.3, 1.0]` |
| Label smoothing | 0.15 |
| Selection metric | Accuracy with an explicit penalty for false-trustworthy errors |
| Seeds | 42, 1337, 7 |

## Limitations

{limitations}

## Citation

```bibtex
@misc{{{meta["citation_key"]}_2026,
  title  = {{{meta["name"]}}},
  author = {{Yan Fitzner}},
  year   = {{2026}},
  url    = {{https://huggingface.co/{meta["model_id"]}}},
}}
```

## License

CC BY-NC 4.0. Free for research, evaluation, and personal use; commercial use requires a separate license.
"""


def slm_card(meta: dict) -> str:
    limitations = "\n".join(f"- {item}" for item in meta["limitations"])
    status = f"\n\n**Release status:** {meta['status']}" if meta.get("status") else ""
    slm_code = SLM_CODE.replace("__MODEL_ID__", meta["model_id"])

    return f"""{frontmatter("peft", "text-generation", "Qwen/Qwen3.5-0.8B", ["qlora", "adapter"])}

# {meta["name"]}

{meta["name"]} is a research LoRA adapter that turns `Qwen/Qwen3.5-0.8B` into a RAG governance label generator. It reads a user question plus retrieved source passages and generates an evidence-state decision a RAG application can use before answering: `ABSTAIN`, `DISPUTED`, or `TRUSTWORTHY`.

This is an adapter, not a standalone model. It is not an answer generator and not an open-world fact checker. Load it on top of the base model with `peft`.{status}

## Labels

{LABELS_MD}

## Outputs

{SLM_OUTPUTS_MD}

## Intended Use

Use this adapter for experiments where a small generative model is useful in the governance path. For production CPU inference, the encoder releases are smaller, faster, and safer on the false-trustworthy metric.

This adapter is not intended to write answers, provide free-form rationales, verify facts outside the provided sources, or run in safety-critical production settings without additional evaluation.

## Quick Start

{slm_code}

## Results

{meta["eval_text"]}

{meta["results_table"]}

For `OVERALL`, recall and precision are micro-averages; in single-label three-class classification they both equal accuracy. For label rows, false-rate is the share of cases that were not that label but were incorrectly predicted as that label. The `TRUSTWORTHY` false-rate is the main safety metric: it measures cases where the model returns `TRUSTWORTHY` even though the sources do not support that decision.

F1 is not shown in the headline table. It is the harmonic mean of precision and recall (`2 * precision * recall / (precision + recall)`), useful as a compact balance score but less direct than the operating metrics above.

## Training Data

{meta["data_text"]}

## Training Recipe

| Item | Value |
|---|---|
| Base model | `Qwen/Qwen3.5-0.8B` |
| Method | QLoRA adapter training |
| Quantization during training | 4-bit NF4 with bfloat16 compute |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Max sequence length | 4096 tokens |
| Output format | Single generated label |
| Epochs | 3 |
| Effective batch size | 16 |
| Learning rate | 2e-4 |
| Scheduler | Cosine with 5% warmup |
| Optimizer | paged AdamW 8-bit |
| Seeds | 42, 1337, 7 |

## Limitations

{limitations}

## Citation

```bibtex
@misc{{{meta["citation_key"]}_2026,
  title  = {{{meta["name"]}}},
  author = {{Yan Fitzner}},
  year   = {{2026}},
  url    = {{https://huggingface.co/{meta["model_id"]}}},
}}
```

## License

CC BY-NC 4.0. Free for research, evaluation, and personal use; commercial use requires a separate license.
"""


def card_specs() -> dict[str, str]:
    return {
        "models/pyrrho-modernbert-base-v1/README.md": encoder_card(
            {
                "name": "pyrrho-nano-g1",
                "model_id": "yafitzdev/pyrrho-nano-g1",
                "citation_key": "pyrrho_nano_g1",
                "default_tau": 0.50,
                "results_table": G1_RESULTS_TABLE,
                "eval_text": "Reported on the fitz-gov V5.1 held-out evaluation split: 584 examples, 3 seeds, validation-selected TRUSTWORTHY thresholds.",
                "data_text": "Trained and evaluated on fitz-gov V5.1, an English benchmark of 2,980 RAG evidence-governance cases. The main benchmark split used 2,336 training examples and 584 held-out evaluation examples; a separate 60-example sanity set was kept for diagnostics.",
                "limitations": [
                    "English-only training and evaluation data.",
                    "This release is weaker on cases where multiple sources agree in substance but use slightly different numbers or phrasing.",
                    "Short, direct single-source answers can be over-abstained compared with longer production-style retrieved passages.",
                    "The model judges only the provided sources; it does not check the open web or hidden background knowledge.",
                ],
            }
        ),
        "models/pyrrho-nano-g2/README.md": encoder_card(
            {
                "name": "pyrrho-nano-g2",
                "model_id": "yafitzdev/pyrrho-nano-g2",
                "citation_key": "pyrrho_nano_g2",
                "default_tau": 0.50,
                "results_table": results_from_summary(
                    "outputs/multi_seed_g2/summary.json",
                    "test_calibrated",
                    "data/processed_v7/test.jsonl",
                ),
                "per_seed": per_seed_from_summary("outputs/multi_seed_g2/summary.json"),
                "eval_text": "Reported on the fitz-gov V7.0.1 held-out test split: 1,050 examples, 3 seeds. Checkpoints and TRUSTWORTHY thresholds were selected on a separate 1,050-example validation split.",
                "data_text": "Trained and evaluated on fitz-gov V7.0.1, an English benchmark of 10,500 RAG evidence-governance examples with query-grouped train, validation, and test splits. Split sizes: 8,400 train, 1,050 validation, 1,050 held-out test.",
                "limitations": [
                    "English-only training and evaluation data.",
                    "The model judges only the provided sources; unsupported retrieval input can still lead to abstention or dispute decisions that require application-level handling.",
                    "Small per-category slices should not be treated as standalone product guarantees; use aggregate metrics and run domain-specific checks for deployment.",
                    "The decision threshold is tuned for low false-trustworthy rate, so some answerable cases may be classified as ABSTAIN or DISPUTED.",
                ],
            }
        ),
        "models/pyrrho-nano-g3/README.md": encoder_card(
            {
                "name": "pyrrho-nano-g3",
                "model_id": "yafitzdev/pyrrho-nano-g3",
                "citation_key": "pyrrho_nano_g3",
                "default_tau": 0.60,
                "results_table": results_from_summary(
                    "outputs/multi_seed_g3_v8/summary.json",
                    "test_calibrated",
                    "data/processed_v8/test.jsonl",
                ),
                "per_seed": per_seed_from_summary("outputs/multi_seed_g3_v8/summary.json"),
                "eval_text": "Reported on the fitz-gov V8.0.0 held-out test split: 2,459 examples, 3 seeds. Checkpoints and TRUSTWORTHY thresholds were selected on a separate 2,459-example validation split.",
                "data_text": "Trained and evaluated on fitz-gov V8.0.0, an English benchmark of 24,592 RAG evidence-governance examples with query-grouped train, validation, and test splits. Split sizes: 19,674 train, 2,459 validation, 2,459 held-out test.",
                "limitations": [
                    "English-only training and evaluation data.",
                    "The model judges only the provided sources; it does not retrieve new evidence or verify claims against outside knowledge.",
                    "Numeric agreement is learned from examples, not enforced by a hard tolerance rule. Evaluate exact numeric workflows before deployment.",
                    "The decision threshold is tuned for low false-trustworthy rate, so some answerable cases may be classified as ABSTAIN or DISPUTED.",
                ],
            }
        ),
        "models/pyrrho-small-g1/README.md": slm_card(
            {
                "name": "pyrrho-small-g1",
                "model_id": "yafitzdev/pyrrho-small-g1",
                "citation_key": "pyrrho_small_g1",
                "results_table": results_from_summary(
                    "outputs/multi_seed_slm/summary.json",
                    "eval",
                    "data/processed/eval.jsonl",
                ),
                "eval_text": "Reported on the fitz-gov V5.1 held-out evaluation split: 584 examples, 3 seeds. Evaluation uses greedy generation and parses the generated text into one of the three labels.",
                "data_text": "Trained and evaluated on fitz-gov V5.1, an English benchmark of 2,980 RAG evidence-governance cases. The adapter was trained on the same train/eval split used for pyrrho-nano-g1.",
                "status": "Research artifact only. It fails the false-trustworthy safety target and is not recommended as the production default.",
                "limitations": [
                    "Fails the false-trustworthy safety target: 12.13% ± 1.27%, well above the 5.7% release gate.",
                    "Slower and more operationally complex than the encoder releases because it requires autoregressive decoding and a base model plus adapter.",
                    "English-only training and evaluation data.",
                    "Classification-only adapter; it is not trained to write answers or explanations.",
                ],
            }
        ),
        "models/pyrrho-small-g1.1/README.md": slm_card(
            {
                "name": "pyrrho-small-g1.1",
                "model_id": "yafitzdev/pyrrho-small-g1.1",
                "citation_key": "pyrrho_small_g1_1",
                "results_table": results_from_summary(
                    "outputs/multi_seed_slm_g1_1/summary.json",
                    "eval",
                    "data/processed/eval.jsonl",
                ),
                "eval_text": "Reported on the fitz-gov V5.1 held-out evaluation split: 584 examples, 3 seeds. Evaluation uses greedy generation and parses the generated text into one of the three labels.",
                "data_text": "Trained and evaluated on fitz-gov V5.1, an English benchmark of 2,980 RAG evidence-governance cases. This run adds stronger safety pressure than pyrrho-small-g1 but keeps the same split and base model.",
                "status": "Research artifact only. It improves over pyrrho-small-g1 on false-trustworthy rate but still misses the safety target.",
                "limitations": [
                    "Still fails the false-trustworthy safety target: 9.31% ± 1.06%, above the 5.7% release gate.",
                    "Slower and more operationally complex than the encoder releases because it requires autoregressive decoding and a base model plus adapter.",
                    "English-only training and evaluation data.",
                    "Classification-only adapter; it is not trained to write answers or explanations.",
                ],
            }
        ),
    }


def main() -> int:
    for path, text in card_specs().items():
        target = Path(path)
        target.write_text(text, encoding="utf-8")
        print(f"wrote {target}")

    template_path = Path("docs/MODEL_CARD_TEMPLATE.md")
    template_path.write_text(TEMPLATE_DOC, encoding="utf-8")
    print(f"wrote {template_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
