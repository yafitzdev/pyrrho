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
| `pyrrho-modernbert-base-v1` | Trained + 3-seed validated. **Not yet packaged for HuggingFace.** |
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

## Immediate next actions

Pick one (or do in order):

1. **Push to GitHub.** Repo is initialized; 5 commits ready.
   ```powershell
   cd C:\Users\yanfi\PycharmProjects\pyrrho
   gh repo create pyrrho --public --source=. --remote=origin --push --description "Fine-tuned classification models for RAG governance"
   ```

2. **Ship v1 to HuggingFace** — baseline confirmed as winner. Three pieces still to write:
   - `scripts/export_onnx.py` — `optimum[onnxruntime]` ONNX + INT8 quantization
   - `scripts/push_to_hub.py` — upload safetensors + ONNX + tokenizer + model card to `yafitzdev/pyrrho-modernbert-base-v1`
   - Model card auto-generated from `final_metrics.json` + `eval_report.json`

3. **Cross-architecture validation.** Train DeBERTa-v3-base with the same config. If within 1–2 pts of ModernBERT-base, result is architecture-robust. ~10 min.

4. **Start v2 work** (post-v1 ship). Add 30–50 short-context TRUSTWORTHY cases to training to close the known limitation.

## Release gates (the bar any pyrrho model must clear before shipping)

Measured on the eval split, mean across 3 seeds:

- **Overall accuracy ≥ 78.7%** — matches fitz-sage v0.11 sklearn baseline.
- **False-trustworthy rate ≤ 5.7%** — matches baseline; the production safety axis.

The originally-planned **tier0 95% sanity gate has been dropped** (see LOG 2026-05-14 afternoon). With 60 cases, run-to-run variance is ±3.5 pts and ~5 of the 60 cases have ambiguous gold labels. Tier0 is reported as a diagnostic in every model card, not a gate.

## Things NOT to do (already decided — don't relitigate)

- ❌ Don't propose Qwen 2.5 anything — stale (Nov 2024). Use Qwen 3.5+ family.
- ❌ Don't propose 35B-class MoE bases — violates the universal CPU-runnable constraint.
- ❌ Don't propose Llama-family bases — license is more restrictive than Apache-2.0.
- ❌ Don't start fitz-gov v6 (long-context / domain expansion) before release #1 ships.
- ❌ Don't rebrand pyrrho — chosen after going through Doxa/Aegis/Sift/Minos/Themis.
- ❌ Don't generate emojis in code/docs unless explicitly asked.

## Where to look for more

| Need | Where |
|---|---|
| What happened, why, and when | [LOG.md](LOG.md) |
| Full vision + 10-release roadmap | [PROJECT.md](PROJECT.md) |
| End-to-end model-development pipeline | [METHODOLOGY.md](METHODOLOGY.md) |
| RTX 5090 / Blackwell / Windows specifics | [SETUP.md](SETUP.md) |
| Repository overview + quickstart | [../README.md](../README.md) |
| Persistent memory (user prefs, banned models, conventions) | `C:\Users\yanfi\.claude\projects\C--Users-yanfi-PycharmProjects-pyrrho\memory\` |
