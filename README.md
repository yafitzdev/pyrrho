

<div align="center">

# pyrrho

### Fine-tuned classification models that decide when your RAG should answer — without an LLM call.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-v1-green.svg)](docs/LOG.md)
[![🤗 Model](https://img.shields.io/badge/🤗%20Model-yafitzdev%2Fpyrrho--nano--g1-yellow)](https://huggingface.co/yafitzdev/pyrrho-nano-g1)
[![🤗 Dataset](https://img.shields.io/badge/🤗%20Dataset-yafitzdev%2Ffitz--gov-yellow)](https://huggingface.co/datasets/yafitzdev/fitz-gov)

[Why pyrrho?](#why-pyrrho) • [Results](#headline-results) • [Roadmap](#family-roadmap) • [Usage](#-where-to-start) • [Docs](#documentation) • [GitHub](https://github.com/yafitzdev/pyrrho) • [🤗 HuggingFace](https://huggingface.co/yafitzdev/pyrrho-nano-g1)

</div>

<br />

---

<div align="center">
<table>
  <tr>
    <td align="center" colspan="2">
      <pre><strong>Query: "Has the company achieved profitability?"</strong>
Sources:
  [1] "Posted its first profitable quarter, net income $4M."
  [2] "Recorded a quarterly loss of $12M, third consecutive losing quarter."</pre>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <strong>❌ Standard governance (constraint + sklearn cascade)</strong>
<pre>
5 LLM calls. 108 hand-crafted features.
Verdict: TRUSTWORTHY  (misses the conflict)
Latency: ~1–2 s on CPU
Requires:  local LLM or paid cloud API
</pre>
    </td>
    <td align="center" width="50%">
      <strong>🛡️ pyrrho-nano-g1</strong>
<pre>
1 forward pass. No features. No LLM.
Verdict: DISPUTED  (correct, P(D)=0.55)
Latency: ~30 ms on CPU (INT8 ONNX)
Requires:  nothing — self-contained
</pre>
    </td>
  </tr>
</table>

  → A 150 MB CPU-friendly classifier that beats the prior pipeline by **+7.43 accuracy points** and **~50× speedup**, with no LLM dependency at inference.

</div>

---

### 🚀 Where to start

> [!IMPORTANT]
> The model lives on **🤗 HuggingFace** as [`yafitzdev/pyrrho-nano-g1`](https://huggingface.co/yafitzdev/pyrrho-nano-g1). Drop it into any RAG pipeline that needs a governance gate.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("yafitzdev/pyrrho-nano-g1")
model = AutoModelForSequenceClassification.from_pretrained("yafitzdev/pyrrho-nano-g1").eval()

query = "Has the company achieved profitability?"
contexts = [
    "Posted its first profitable quarter, net income $4M.",
    "Recorded a quarterly loss of $12M, third consecutive losing quarter.",
]
text = f"Question: {query}\n\nSources:\n" + "\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, 1))

with torch.no_grad():
    probs = torch.softmax(model(**tokenizer(text, return_tensors="pt", truncation=True)).logits[0], dim=-1).numpy()
print({"ABSTAIN": probs[0], "DISPUTED": probs[1], "TRUSTWORTHY": probs[2]})
# → DISPUTED ≈ 0.55
```

For **production CPU inference at ~30 ms/query**, use the INT8 ONNX variant via `optimum`. Full usage in the [model card](https://huggingface.co/yafitzdev/pyrrho-nano-g1#cpu-optimized-onnx--int8).

---

### About

Most RAG governance is either **(a) a black-box LLM call** ("ask GPT-4 if these sources support the answer" — slow, expensive, non-deterministic) or **(b) a feature-engineered classifier** (~108 hand-crafted signals fed into sklearn — cheap but capped at ~79% accuracy on hard benchmarks). I built `pyrrho` to replace both with a single fine-tuned encoder that runs at **30 ms on CPU** and **beats both approaches on a public benchmark**.

The architecture call: pure encoder (ModernBERT-base, 149M params) — not a generative SLM, not an LLM. For 3-class classification with constrained label space, encoder + INT8 ONNX is 50–100× faster on CPU than the same task with a generative model, and **doesn't lose accuracy** when the labels are categorical and the input fits in 4K tokens (as RAG retrievals almost always do).

It's the model that powers governance in [`fitz-sage`](https://github.com/yafitzdev/fitz-sage) (the RAG library) and is benchmarked against [`fitz-gov`](https://github.com/yafitzdev/fitz-gov) (2,920 adversarial cases, 5-fold CV). The three projects form a triangle — benchmark, models, library.

Yan Fitzner — ([LinkedIn](https://www.linkedin.com/in/yan-fitzner/), [GitHub](https://github.com/yafitzdev)).

---

### Headline results

Release v1 — `pyrrho-nano-g1` vs the published `fitz-sage` v0.11 sklearn baseline. 3-seed mean ± std on the [`fitz-gov`](https://huggingface.co/datasets/yafitzdev/fitz-gov) V5.1 eval hold-out (584 cases, stratified 20% from `tier1_core`):

| Metric | **pyrrho v1** | sklearn baseline | Δ |
|---|---|---|---|
| Overall accuracy | **86.13 ± 0.86 %** | 78.7 % | **+7.43** |
| False-trustworthy rate (safety) | **5.27 ± 0.21 %** | 5.7 % | **-0.43** (safer) |
| Trustworthy recall | **79.38 ± 1.64 %** | 70.0 % | **+9.38** |
| Disputed recall | **94.81 ± 1.28 %** | 86.1 % | **+8.71** |
| Abstain recall | **92.94 ± 1.11 %** | 86.5 % | **+6.44** |
| CPU inference (estimated) | **~30 ms** | ~500–2000 ms (5 LLM calls) | **~50× faster** |
| External dependencies | **none** | requires LLM | self-contained |

Every margin is multiple standard deviations larger than seed noise — not a lucky-run artifact. Independently verifiable by running the published model against the published benchmark: `load_dataset("yafitzdev/fitz-gov")` + `AutoModelForSequenceClassification.from_pretrained("yafitzdev/pyrrho-nano-g1")`.

> [!NOTE]
> **Known limitation:** the model occasionally classifies *multi-source-convergence* cases (multiple authoritative sources agreeing within measurement tolerance) as `DISPUTED`. ~57% error on this fitz-gov subcategory (n=7). Fixed in v2 with augmented training data. Documented in the model card.

---

### Why `pyrrho`?

**No LLM dependency 🪶** → [Model card](https://huggingface.co/yafitzdev/pyrrho-nano-g1)
> Standard governance pipelines route every query through 5+ LLM calls to extract constraint signals (contradiction detection, evidence sufficiency, causal attribution, …) before the classifier even fires. `pyrrho` reads the raw query and contexts and emits a verdict in **one forward pass**. No cloud API spend, no GPU swap, no rate limits.

**Beats the baseline by 7 points 📊** → [Benchmark](https://huggingface.co/datasets/yafitzdev/fitz-gov)
> 86.13% accuracy vs 78.7% for the prior constraint+sklearn pipeline, on the same 2,920-case `fitz-gov` benchmark. The biggest gain is on **trustworthy recall** (+9.4 pts) — the bucket where hand-crafted features couldn't read positive evidence-agreement signals. Attention over raw text can.

**Safer than the baseline 🛡️**
> False-trustworthy rate (the production safety metric: how often a *confident hallucination path* gets greenlit) is **5.27%**, below the prior pipeline's 5.7%. Threshold calibration on top can push this lower at a small accuracy cost.

**Production-grade CPU inference ⚡** → [INT8 ONNX](https://huggingface.co/yafitzdev/pyrrho-nano-g1/blob/main/model_quantized.onnx)
> ~30 ms per query on commodity CPU after INT8 dynamic quantization. Ship the 150 MB `model_quantized.onnx` and serve governance inline — no GPU, no API, no LLM. Fits into latency-sensitive RAG paths that previously couldn't afford a governance step.

**Reproducible end-to-end 🔬**
> Training data, model weights, and the evaluation pipeline are all public. The `final_metrics.json` and `manifest.json` that ship alongside the weights pin: git commit, pip freeze, hardware, seed, training duration. Anyone can re-run the smoke test (`pytest tests/test_smoke.py`) against the published model.

**Cross-linked with the triangle 🔗**
> Benchmark: [`fitz-gov`](https://github.com/yafitzdev/fitz-gov). Models: [`yafitzdev/pyrrho-*`](https://huggingface.co/yafitzdev). Production library: [`fitz-sage`](https://github.com/yafitzdev/fitz-sage). Each reinforces the others — `fitz-gov` defines the eval contract, `pyrrho` ships the models, `fitz-sage` consumes them in production.

---

### Family roadmap

Three tiers, generation-suffixed by the fitz-gov data version they were trained on. All CPU-runnable.

- **`pyrrho-nano`** — fine-tuned encoder (ModernBERT-class). Single forward pass, fastest, smallest. The production governance head.
- **`pyrrho-small`** — fine-tuned generative SLM (1–3B dense). Classification + reasoning trace.
- **`pyrrho-MoE`** — sparse Mixture-of-Experts trained from scratch (4B total / 0.4B active). The terminal architecture; covers 16 RAG governance capabilities in v1.

<br>

| Model | Tier | Params | Status |
|---|---|---|---|
| [`pyrrho-nano-g1`](https://huggingface.co/yafitzdev/pyrrho-nano-g1) | encoder | 149M | ✅ **live on HF** (fitz-gov V5.1) |
| `pyrrho-nano-g1.1` | encoder | 149M | planned — V5.1-enriched apples-to-apples retrain (ROADMAP Phase 1) |
| `pyrrho-nano-g2` | encoder | 149M | planned — fitz-gov V6 retrain, 5K–10K cases (ROADMAP Phase 3) |
| `pyrrho-small-g2` | generative SLM | 1–3B dense | planned — first SLM on V6, classification + rationale, RL fine-tuned (ROADMAP Phase 3) |
| `pyrrho-MoE-g3` | sparse MoE | 4B total / 0.4B active | planned — trained from scratch on V7; 7–8 domain experts + conflict-detection meta-expert; full 16-capability RAG runtime (ROADMAP Phase 5) |
| `pyrrho-MoE-g4` | sparse MoE | 4B total / 0.4B active | planned — V8+ retrain, infrastructure-grade reliability (ROADMAP Phase 6) |

Full release roadmap, expert specifications, evaluation metrics, and publication strategy in [`docs/ROADMAP.md`](docs/ROADMAP.md). The older 10-release breakdown in [`docs/PROJECT.md §10`](docs/PROJECT.md) is retained for historical context but superseded.

---

<details>

<summary><strong>📦 Repository structure</strong></summary>

<br>

```
pyrrho/
├── README.md           ← you are here
├── CLAUDE.md           ← project conventions (HANDOFF/LOG update rules, banned models, style)
├── LICENSE             ← CC BY-NC 4.0
├── pyproject.toml      ← Python deps; encoder / slm / hub / dev extras
├── docs/
│   ├── INDEX.md        ← reading-order entry point for any new contributor
│   ├── HANDOFF.md      ← current status snapshot (overwritten as state changes)
│   ├── LOG.md          ← append-only project history
│   ├── PROJECT.md      ← full vision, model picks, roadmap, training recipes
│   ├── METHODOLOGY.md  ← end-to-end pipeline; release gates; W&B conventions
│   └── SETUP.md        ← RTX 5090 / Blackwell / Windows specifics
├── src/pyrrho/         ← Python package: data, metrics, training, manifest
├── scripts/            ← all CLI scripts (train, eval, sweep, compare, push, …)
├── configs/
│   ├── encoder/        ← ModernBERT-base, DeBERTa-v3-large (3-class + 4-class)
│   ├── slm/            ← Qwen3.5-2B, LFM2.5-1.2B, LFM2-8B-A1B MoE
│   └── sweep_grids/    ← hyperparameter sweep grids
├── tests/              ← pytest suites (smoke regression guard)
├── data/               ← (gitignored) processed splits from prepare_data.py
└── outputs/            ← (gitignored) training runs, checkpoints, eval reports
```

</details>

---

<details>

<summary><strong>📦 Train your own pyrrho variant from scratch</strong></summary>

<br>

Reproduces the published numbers end-to-end. Requires an RTX 50-series GPU (see [`docs/SETUP.md`](docs/SETUP.md) for Blackwell / Windows / WSL2 specifics).

```bash
# 1. Install
git clone https://github.com/yafitzdev/pyrrho.git
cd pyrrho
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\Activate.ps1 on Windows
pip install torch --index-url https://download.pytorch.org/whl/cu128   # Blackwell wheels
pip install -e ".[encoder,hub,dev]"

# 2. Prepare data — either pull from the published HF dataset,
# or use a local clone of yafitzdev/fitz-gov.
python scripts/prepare_data.py --fitz-gov ../fitz-gov/data --output data/processed

# 3. Verify the environment (driver / CUDA / bitsandbytes / Blackwell)
python scripts/verify_env.py

# 4. Train release #1 (~80–500 s on RTX 5090 depending on contention)
python scripts/train_encoder.py --config configs/encoder/modernbert_base.yaml --no-wandb

# 5. Multi-seed validation — produces the published mean ± std
python scripts/run_seeds.py --seeds 42 1337 7

# 6. Full per-breakdown evaluation (per domain / difficulty / reasoning_type / subcategory)
python scripts/eval_report.py --checkpoint outputs/multi_seed/seed_42/checkpoint-XXX

# 7. Compare to the sklearn baseline OR an existing pyrrho release
python scripts/compare_runs.py baseline outputs/multi_seed/summary.json

# 8. Smoke test regression guard (10 handcrafted cases)
pytest tests/test_smoke.py -v
```

Full methodology, release gates, and W&B conventions in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

</details>

---

### Documentation

| Document | Purpose |
|---|---|
| [`docs/INDEX.md`](docs/INDEX.md) | **Fresh session entry point.** Reading order for any new contributor. |
| [`docs/HANDOFF.md`](docs/HANDOFF.md) | Current status snapshot — what's trained, headline metrics, next actions. |
| [`docs/LOG.md`](docs/LOG.md) | Append-only project history (findings, decisions, experiments). |
| [`docs/PROJECT.md`](docs/PROJECT.md) | Full plan: vision, model picks, training recipes, roadmap. |
| [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) | End-to-end model-development pipeline, release gates, W&B conventions. |
| [`docs/SETUP.md`](docs/SETUP.md) | RTX 5090 / Blackwell / Windows environment specifics. |

---

### Related projects

- [**`fitz-sage`**](https://github.com/yafitzdev/fitz-sage) — production RAG library that uses `pyrrho` for governance.
- [**`fitz-gov`**](https://github.com/yafitzdev/fitz-gov) — 2,980-case benchmark for RAG epistemic honesty. The dataset `pyrrho` is trained and evaluated against. Also on HF: [`yafitzdev/fitz-gov`](https://huggingface.co/datasets/yafitzdev/fitz-gov).

The three projects form a triangle: `fitz-gov` defines the eval contract, `pyrrho` produces the models, `fitz-sage` consumes them in production.

---

### License

CC BY-NC 4.0 — see [LICENSE](LICENSE). Free for research, evaluation, and personal use; commercial use requires a separate license.
