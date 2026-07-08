"""Legacy v1 SLM card builder.

This is retained to reproduce old SLM experiments. It is not used for the
current Pyrrho v2 encoder release.

Generate the HuggingFace README.md for a pyrrho SLM release.

Reads the multi-seed summary.json written by scripts/aggregate_slm_seeds.py and
writes a model card pinning the 3-seed mean +/- std as the headline. Compares
against both the sklearn baseline and pyrrho-nano-g1 (the encoder on the same
benchmark), so the encoder-vs-SLM trade-off is visible at a glance.

Run from project root:
    python scripts/build_slm_model_card.py \\
        --summary outputs/multi_seed_slm/summary.json \\
        --adapter outputs/multi_seed_slm/seed_42/final \\
        --output models/pyrrho-small-g1/README.md
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


# Locked baselines (fitz-sage v0.11 sklearn) — same numbers as build_model_card.py.
BASELINE_SKLEARN = {
    "accuracy": 0.787,
    "recall_abstain": 0.865,
    "recall_disputed": 0.861,
    "recall_trustworthy": 0.700,
    "false_trustworthy_rate": 0.057,
}

# pyrrho-nano-g1 (the encoder on the same V5.1 split). Numbers from
# docs/HANDOFF.md "Validated v1 metrics" — 3-seed mean. Used as the
# head-to-head comparison row in the SLM card.
NANO_G1 = {
    "accuracy": 0.8613,
    "false_trustworthy_rate": 0.0527,
    "recall_trustworthy": 0.7938,
    "recall_disputed": 0.9481,
    "recall_abstain": 0.9294,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary", type=Path, required=True, help="summary.json from aggregate_slm_seeds.py")
    p.add_argument("--adapter", type=Path, required=True, help="Path to the chosen seed's final/ adapter dir")
    p.add_argument("--output", type=Path, required=True, help="Where to write README.md")
    p.add_argument("--model-id", type=str, default="yafitzdev/pyrrho-small-g1")
    p.add_argument("--base-model", type=str, default="Qwen/Qwen3.5-0.8B")
    p.add_argument("--fitz-gov-version", type=str, default="V5.1")
    return p.parse_args()


def fmt_pct(stat: dict) -> str:
    m, s = stat["mean"], stat["std"]
    return f"{m * 100:.2f} ± {s * 100:.2f}"


def fmt_delta(new_mean: float, baseline: float) -> str:
    return f"{(new_mean - baseline) * 100:+.2f}"


def fitz_gov_commit() -> str:
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
    if not args.adapter.exists():
        print(f"ERROR: adapter dir not found: {args.adapter}", file=sys.stderr)
        return 1

    with args.summary.open("r", encoding="utf-8") as fh:
        summary = json.load(fh)

    seeds = summary.get("seeds", [])
    agg = summary["aggregate"]
    eval_agg = agg.get("eval", {})
    tier0_agg = agg.get("tier0_sanity", {})

    fitz_commit = fitz_gov_commit()
    model_name = args.model_id.split("/")[-1]

    # Comparative table rows — handle the case where SLM didn't beat encoder
    # gracefully: show the actual delta whether positive or negative.
    eval_acc = eval_agg["accuracy"]
    eval_ft = eval_agg["false_trustworthy_rate"]

    card = f"""---
license: cc-by-nc-4.0
library_name: peft
pipeline_tag: text-generation
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
  - qwen3.5
  - qlora
  - sft
datasets:
  - yafitzdev/fitz-gov
metrics:
  - accuracy
  - f1
  - false-trustworthy-rate
---

# {model_name}

> Generative SLM version of pyrrho's RAG governance classifier. QLoRA fine-tune of [`{args.base_model}`](https://huggingface.co/{args.base_model}) on [fitz-gov](https://huggingface.co/datasets/yafitzdev/fitz-gov) {args.fitz_gov_version}.
> Same 3-class output as [`pyrrho-nano-g1`](https://huggingface.co/yafitzdev/pyrrho-nano-g1) (`ABSTAIN` / `DISPUTED` / `TRUSTWORTHY`); different architecture, same benchmark, intended as the head-to-head data point that anchors the encoder-vs-SLM story in the pyrrho family.

This is a **LoRA adapter**, not a full fine-tune — load it on top of the base `{args.base_model}` model with `peft`.

| Verdict | Meaning |
|---|---|
| `ABSTAIN` | The sources do not contain enough information to answer. |
| `DISPUTED` | The sources contradict each other on the answer. |
| `TRUSTWORTHY` | The sources consistently and sufficiently support an answer. |

---

## Results

3-seed mean ± std on the fitz-gov {args.fitz_gov_version} eval split (584 cases, stratified 20% hold-out from `tier1_core`), seeds {seeds}. Decode-based eval (greedy generation, parse the assistant turn into a label).

| Metric | **{model_name}** | pyrrho-nano-g1 (encoder) | sklearn baseline | Δ vs baseline |
|---|---|---|---|---|
| Overall accuracy | **{fmt_pct(eval_agg['accuracy'])}** | {NANO_G1['accuracy']*100:.2f} | {BASELINE_SKLEARN['accuracy']*100:.1f} | **{fmt_delta(eval_acc['mean'], BASELINE_SKLEARN['accuracy'])}** |
| False-trustworthy rate (safety) | **{fmt_pct(eval_agg['false_trustworthy_rate'])}** | {NANO_G1['false_trustworthy_rate']*100:.2f} | {BASELINE_SKLEARN['false_trustworthy_rate']*100:.1f} | **{fmt_delta(eval_ft['mean'], BASELINE_SKLEARN['false_trustworthy_rate'])}** |
| Trustworthy recall | **{fmt_pct(eval_agg['recall_trustworthy'])}** | {NANO_G1['recall_trustworthy']*100:.2f} | {BASELINE_SKLEARN['recall_trustworthy']*100:.1f} | **{fmt_delta(eval_agg['recall_trustworthy']['mean'], BASELINE_SKLEARN['recall_trustworthy'])}** |
| Disputed recall | **{fmt_pct(eval_agg['recall_disputed'])}** | {NANO_G1['recall_disputed']*100:.2f} | {BASELINE_SKLEARN['recall_disputed']*100:.1f} | **{fmt_delta(eval_agg['recall_disputed']['mean'], BASELINE_SKLEARN['recall_disputed'])}** |
| Abstain recall | **{fmt_pct(eval_agg['recall_abstain'])}** | {NANO_G1['recall_abstain']*100:.2f} | {BASELINE_SKLEARN['recall_abstain']*100:.1f} | **{fmt_delta(eval_agg['recall_abstain']['mean'], BASELINE_SKLEARN['recall_abstain'])}** |
| Macro F1 | {fmt_pct(eval_agg['macro_f1'])} | n/a | n/a | — |

Tier0 sanity (60-case held-out diagnostic, not a release gate):

| Metric | **{model_name}** |
|---|---|
| Accuracy | {fmt_pct(tier0_agg['accuracy'])} |
| False-trustworthy rate | {fmt_pct(tier0_agg['false_trustworthy_rate'])} |

---

## Usage

This is a PEFT/LoRA adapter. Load the base model first, then attach the adapter:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "{args.base_model}",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, "{args.model_id}")
model.eval()

tokenizer = AutoTokenizer.from_pretrained("{args.base_model}")

SYSTEM = (
    "You are a RAG governance classifier. Given a user question and retrieved "
    "sources, decide whether the sources support a confident answer.\\n\\n"
    "Output exactly one token: ABSTAIN, DISPUTED, or TRUSTWORTHY.\\n"
    "- ABSTAIN: sources do not contain enough information to answer.\\n"
    "- DISPUTED: sources contradict each other on the answer.\\n"
    "- TRUSTWORTHY: sources consistently and sufficiently support an answer."
)

query = "Has the company achieved profitability?"
contexts = [
    "Posted its first profitable quarter, net income $4M.",
    "Recorded a quarterly loss of $12M, third consecutive losing quarter.",
]
sources = "\\n".join(f"[{{i}}] {{c}}" for i, c in enumerate(contexts, start=1))
user = f"Question: {{query}}\\n\\nSources:\\n{{sources}}"

messages = [
    {{"role": "system", "content": SYSTEM}},
    {{"role": "user", "content": user}},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=16, do_sample=False, pad_token_id=tokenizer.eos_token_id)
new = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

# The Qwen3.5 chat template injects an empty <think></think> block; strip it
# and take the first label token that appears.
payload = new.split("</think>", 1)[-1].upper()
for lab in ("ABSTAIN", "DISPUTED", "TRUSTWORTHY"):
    if lab in payload:
        print(lab); break
```

For batch decoding, use left-padding (the tokenizer needs `padding_side='left'`) so the generated tokens align right.

---

## When to use {model_name} vs pyrrho-nano-g1

`pyrrho-nano-g1` (149M ModernBERT encoder, single forward pass, ~30 ms on CPU after INT8 ONNX) is the right default for production. It is faster, smaller, and ships as a self-contained ONNX artifact with no LoRA-adapter loading step.

`{model_name}` (this model: 0.8B SLM + 6.4M LoRA, generative) is the right pick when:
- You want the same governance decision **plus** the model's reasoning trace (this version is trained classification-only, but the architecture supports adding rationale generation in v2).
- You already have GPU budget on the inference path for a 0.8B SLM.
- You want to evaluate whether pre-trained world knowledge helps on the multi-source-convergence subcategory that bottlenecks the encoder.

For raw CPU latency on a fixed 3-class output, the encoder still wins.

---

## Training

| Hyperparameter | Value |
|---|---|
| Base model | `{args.base_model}` |
| Architecture | Qwen3.5 (Gated DeltaNet + Gated Attention hybrid, 24 layers) |
| Method | QLoRA (4-bit NF4 + bf16 compute) + LoRA adapter |
| LoRA rank / alpha / dropout | r=16, alpha=32, dropout=0.05 |
| LoRA targets | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` (standard transformer projections; linear-attention `in_proj_*` / `out_proj` not targeted) |
| Trainable params | ~6.4M / 510M (1.25%) |
| Max sequence length | 4096 tokens |
| Epochs | 3 |
| Per-device batch size | 4 |
| Effective batch size | 16 (grad accum 4) |
| Learning rate | 2e-4 |
| LR scheduler | cosine, 5% warmup |
| Weight decay | 0.01 |
| NEFTune noise | 5 |
| Optimizer | paged_adamw_8bit (bf16 compute) |
| Loss | TRL `SFTTrainer` with `assistant_only_loss=True` (loss only on the assistant label tokens, masking the full `system+user` prompt) |
| Chat template | Qwen3-style ChatML; the Qwen3 patched training template from TRL (`{{% generation %}}` markers) is used during training |
| Selection metric | `eval_loss` (the decode-time accuracy/FT pass is a post-training step, not a checkpoint-selection signal) |
| Hardware | NVIDIA RTX 5090 (Blackwell sm_120) |
| Training time | ~30-60 min per seed (depending on GPU contention) |
| Seeds | {seeds} |

Training data: fitz-gov {args.fitz_gov_version} `tier1_core`, same stratified 80/20 split by `(label, difficulty)` as pyrrho-nano-g1 — so the two models are directly comparable case-for-case. The 60-case `tier0_sanity` set is held out as a noise-prone diagnostic and not used for training.

---

## Dataset

This model is trained and evaluated on [**fitz-gov {args.fitz_gov_version}**](https://huggingface.co/datasets/yafitzdev/fitz-gov), a 2,980-case benchmark for RAG governance (epistemic honesty). Eval split: 584 cases stratified from `tier1_core` (2,920 cases, 62.7% hard difficulty, 17 domains, 113+ subcategories).

fitz-gov commit at training time: `{fitz_commit}`

---

## Intended use

A drop-in alternative governance head for any RAG pipeline that wants a small generative SLM in the governance path (vs the encoder `pyrrho-nano-g1`). The output is a single classification label per `(query, contexts)` pair.

**Not intended for:**
- Generating answers (this is a classification fine-tune, not a generator).
- Token-level hallucination localization (see [LettuceDetect](https://github.com/KRLabsOrg/LettuceDetect)).
- Languages other than English. fitz-gov is English-only.

**Safety axis:** the false-trustworthy rate is the production safety metric — a wrong `TRUSTWORTHY` is the dangerous error (would confidently surface an unsupported answer).

---

## Limitations

This is the first generative SLM in the pyrrho family, fine-tuned on the same V5.1 data as `pyrrho-nano-g1`. It carries forward the encoder's known limitation on multi-source-convergence cases (multiple authoritative sources agreeing within measurement tolerance — sometimes misclassified as `DISPUTED`) unless the SLM's pre-trained world knowledge happens to override the decoder pattern. A V6 retrain (`pyrrho-small-g2`, ROADMAP Phase 3) targets this directly with the synthetic data pipeline.

---

## Citation

```bibtex
@misc{{pyrrho_small_g1_2026,
  title  = {{ {model_name} }},
  author = {{ Yan Fitzner }},
  year   = {{ 2026 }},
  url    = {{ https://huggingface.co/{args.model_id} }},
}}
```

## License

CC BY-NC 4.0 — see [LICENSE](https://github.com/yafitzdev/pyrrho/blob/main/LICENSE). Free for research, evaluation, and personal use; commercial use requires a separate license.

## Related projects

- [**pyrrho-nano-g1**](https://huggingface.co/yafitzdev/pyrrho-nano-g1) — the encoder counterpart (149M ModernBERT, CPU-only, INT8 ONNX).
- [**fitz-sage**](https://github.com/yafitzdev/fitz-sage) — production RAG library that uses pyrrho models.
- [**fitz-gov**](https://huggingface.co/datasets/yafitzdev/fitz-gov) — the benchmark dataset.
- [**pyrrho**](https://github.com/yafitzdev/pyrrho) — training code, configs, and roadmap.
"""

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        fh.write(card)
    print(f"Wrote model card -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
