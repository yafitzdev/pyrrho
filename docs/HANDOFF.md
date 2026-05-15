# HANDOFF — pyrrho project current status

> Fresh-session entry point. Read this first.
> For the *history* of how we got here, see [LOG.md](LOG.md).
> For the full plan and roadmap, see [PROJECT.md](PROJECT.md).

This file is **overwritten** as the state changes. It always reflects only the current state.

---

## What pyrrho is, in 30 seconds

Fine-tuned classification models for RAG governance. Given a (query, retrieved contexts) pair, predicts `ABSTAIN` / `DISPUTED` / `TRUSTWORTHY`. Drop-in replacement for the constraint+sklearn pipeline in [fitz-sage](https://github.com/yafitzdev/fitz-sage). Encoders for CPU production; generative SLMs (Qwen 3.5, Gemma 4, Phi-4-mini, LFM2.5) as a HuggingFace portfolio. Benchmarked against [fitz-gov](https://github.com/yafitzdev/fitz-gov) v5.

The brand name is from Pyrrho of Elis — the Greek philosopher whose school practiced suspension of judgment when evidence was insufficient.

## Current state of the family

| Release | Status |
|---|---|
| `pyrrho-modernbert-base-v1` | Trained + 3-seed validated. On HuggingFace. **In production — it is the governance backend of fitz-sage v0.13.0.** |
| #2–#10 (Track A long-context / accuracy mode, Track B SLMs, grounding sidecar) | Not started |

## Validated v1 metrics (3-seed mean ± std on fitz-gov v5 eval hold-out, 584 cases)

| Metric | pyrrho v1 | sklearn baseline | Δ |
|---|---|---|---|
| Overall accuracy (calibrated) | **86.13 ± 0.86%** | 78.7% | **+7.43** |
| False-trustworthy rate | **5.27 ± 0.21%** | 5.7% | **-0.43** (safer) |
| Trustworthy recall | **79.38 ± 1.64%** | 70.0% | **+9.38** |
| Disputed recall | **94.81 ± 1.28%** | 86.1% | **+8.71** |
| Abstain recall | **92.94 ± 1.11%** | 86.5% | **+6.44** |
| CPU inference (est.) | ~30 ms | ~500–2000 ms (5 LLM calls) | ~50× faster |

Every margin is multiple standard deviations larger than seed noise. Confirmed not a lucky-run artifact (LOG entry 2026-05-14 afternoon).

## Known limitations of v1

1. **Multi-source-convergence misclassified as DISPUTED.** When multiple authoritative sources agree on a fact with slight numerical variation (within measurement tolerance), the model frequently classifies the case as DISPUTED with high confidence. 57% error rate on this fitz-gov subcategory (n=7). Examples: "What is the speed of light?" + 4 sources all citing 299,792,458 m/s → P(DISPUTED) = 0.79. Attempted fix with 30 training-side supplement cases (2026-05-14 evening LOG entry) only moved the needle to 43% err and regressed other axes — proper fix needs ~100-200 cases with boundary counter-examples; deferred to v2.
2. **Short clean TRUSTWORTHY contexts trigger over-abstention.** Smell test showed: `"Q: When was the iPhone released? Ctx: Apple released the original iPhone on June 29, 2007..."` → predicts ABSTAIN with P(A)=0.92. Tier1 training is 62.7% hard cases; the model never learned the short-clean-answer pattern. Fixable in v2 alongside #1.

## Pipeline / tooling — what exists now

| Script | Purpose |
|---|---|
| [`scripts/prepare_data.py`](../scripts/prepare_data.py) | fitz-gov → train/eval/tier0 splits + HF DatasetDict |
| [`scripts/train_encoder.py`](../scripts/train_encoder.py) | Single-run encoder fine-tuning, config-driven, writes manifest.json |
| [`scripts/run_seeds.py`](../scripts/run_seeds.py) | Multi-seed orchestrator, aggregates mean ± std |
| [`scripts/sweep.py`](../scripts/sweep.py) | Hyperparameter sweep (coordinate-descent or grid) |
| [`scripts/eval.py`](../scripts/eval.py) | 5-fold CV runner |
| [`scripts/eval_report.py`](../scripts/eval_report.py) | Full per-breakdown evaluation report on a checkpoint |
| [`scripts/compare_runs.py`](../scripts/compare_runs.py) | Diff two runs (or vs baseline), markdown table out |
| [`scripts/inspect_tier0_failures.py`](../scripts/inspect_tier0_failures.py) | Dump misclassified tier0 cases with full context |
| [`scripts/smell_test.py`](../scripts/smell_test.py) | 10-case sanity check (ad-hoc) |
| [`tests/test_smoke.py`](../tests/test_smoke.py) | pytest version of the smell test for CI regression |

Reproducibility: every artifact-producing script writes `manifest.json` (git/pip/hw/seed/timing) via `pyrrho.manifest.write_manifest`.

Full methodology, release gates, and W&B conventions in [METHODOLOGY.md](METHODOLOGY.md).

## What's live

- **pyrrho model on HF**: https://huggingface.co/yafitzdev/pyrrho-modernbert-base-v1 (public, Apache-2.0, 1.35 GB: safetensors + FP32 ONNX + INT8 ONNX). Model card aligned to user's trimmed shape (no philosophical aside, no uncalibrated table) — `build_model_card.py` produces this by default. Cross-linked to `yafitzdev/fitz-gov` dataset.
- **fitz-gov dataset on HF**: https://huggingface.co/datasets/yafitzdev/fitz-gov (public, MIT, V5.1, three configs: tier1_core / tier0_sanity / validation). Verified `load_dataset(...)` loads cleanly.
- **pyrrho GitHub repo**: public, redesigned README in the fitz-sage style.
- **pyrrho is in production.** fitz-sage **v0.13.0** (shipped 2026-05-15, PyPI + GitHub) replaced its constraint+sklearn governance cascade with `yafitzdev/pyrrho-modernbert-base-v1` — loaded as INT8 ONNX, ~30 ms/decision on CPU, zero LLM calls on the governance path. The same release also swapped fitz-sage's chat-call reranker for `Alibaba-NLP/gte-reranker-modernbert-base` (a separate ONNX cross-encoder — fitz-sage's call, applying pyrrho's pattern). See LOG 2026-05-15.
  - Release: https://github.com/yafitzdev/fitz-sage/releases/tag/v0.13.0
  - PyPI: https://pypi.org/project/fitz-sage/0.13.0/

## Immediate next actions

The integration milestone is **closed** — pyrrho v1 is shipped and is
the production governance backend. Everything below is model-quality
upside on an already-live baseline; nothing here blocks anything.

1. **SLM track** — fine-tune Qwen3.5-0.8B to see if it fixes `multi_source_convergence` (v1's only real failure mode — see Known limitations #1). Pretraining world knowledge + reasoning depth should address it. ~1 hr for the fine-tune.

2. **v2 augmentation set** — bigger / more diverse data (~100-200 cases per failing subcategory, plus boundary counter-examples in the DISPUTED class) targeting both known limitations, then retrain `pyrrho-modernbert-base-v2`.

3. **Cross-architecture validation** (optional). Train DeBERTa-v3-base with the same encoder config. Defensive content for a v1.5 model card. ~10 min.

4. **fitz-gov v6** — only after a v2 pyrrho ships and validates. Long-context / domain expansion. Still gated (see "Things NOT to do").

## Release gates (the bar any pyrrho model must clear before shipping)

Measured on the eval split, mean across 3 seeds:

- **Overall accuracy ≥ 78.7%** — matches fitz-sage v0.11 sklearn baseline.
- **False-trustworthy rate ≤ 5.7%** — matches baseline; the production safety axis.

The originally-planned **tier0 95% sanity gate has been dropped** (see LOG 2026-05-14 afternoon). With 60 cases, run-to-run variance is ±3.5 pts and ~5 of the 60 cases have ambiguous gold labels. Tier0 is reported as a diagnostic in every model card, not a gate.

## Things NOT to do (already decided — don't relitigate)

- ❌ Don't propose Qwen 2.5 anything — stale (Nov 2024). Use Qwen 3.5+ family.
- ❌ Don't propose 35B-class MoE bases — violates the universal CPU-runnable constraint.
- ❌ Don't propose Llama-family bases — license is more restrictive than Apache-2.0.
- ❌ Don't start fitz-gov v6 (long-context / domain expansion) before a pyrrho **v2** ships. Release #1 is now in production (fitz-sage v0.13.0); the gate moved to v2 so v6 data work doesn't outrun model validation.
- ❌ Don't rebrand pyrrho — chosen after going through Doxa/Aegis/Sift/Minos/Themis.
- ❌ Don't generate emojis in code/docs unless explicitly asked.
- ❌ Don't propose running pyrrho as a remote endpoint / hosted service. It runs CPU-side via INT8 ONNX inside fitz-sage — that's the architectural commitment (no embeddings, no vector DB, no per-vendor providers in fitz-sage's hot path).

## Where to look for more

| Need | Where |
|---|---|
| What happened, why, and when | [LOG.md](LOG.md) |
| Full vision + 10-release roadmap | [PROJECT.md](PROJECT.md) |
| End-to-end model-development pipeline | [METHODOLOGY.md](METHODOLOGY.md) |
| RTX 5090 / Blackwell / Windows specifics | [SETUP.md](SETUP.md) |
| Repository overview + quickstart | [../README.md](../README.md) |
| Persistent memory (user prefs, banned models, conventions) | `C:\Users\yanfi\.claude\projects\C--Users-yanfi-PycharmProjects-pyrrho\memory\` |
