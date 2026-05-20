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
| `pyrrho-nano-g1` (ModernBERT 149M encoder) | Trained + 3-seed validated. On HuggingFace. **In production — it is the governance backend of fitz-sage v0.13.0.** |
| `pyrrho-small-g1` (Qwen3.5-0.8B + LoRA, V5.1 SFT) | Trained + 3-seed validated locally. **Not on HF — fails the false-trustworthy gate (12.13% vs 5.7% bar).** Release dir staged at `models/pyrrho-small-g1/`. See LOG 2026-05-20 evening for the finding. |
| `pyrrho-nano-g1.1`, `pyrrho-nano-g2`, `pyrrho-small-g2`, `pyrrho-MoE-g3` (and beyond) | Not started. See [ROADMAP.md](ROADMAP.md). |

## Validated metrics

### `pyrrho-nano-g1` (encoder, calibrated, 3-seed mean ± std on V5.1 eval, 584 cases)

| Metric | pyrrho-nano-g1 | sklearn baseline | Δ |
|---|---|---|---|
| Overall accuracy | **86.13 ± 0.86%** | 78.7% | **+7.43** |
| False-trustworthy rate | **5.27 ± 0.21%** | 5.7% | **-0.43** (safer) |
| Trustworthy recall | **79.38 ± 1.64%** | 70.0% | **+9.38** |
| Disputed recall | **94.81 ± 1.28%** | 86.1% | **+8.71** |
| Abstain recall | **92.94 ± 1.11%** | 86.5% | **+6.44** |
| CPU inference (est.) | ~30 ms | ~500–2000 ms (5 LLM calls) | ~50× faster |

Every margin is multiple standard deviations larger than seed noise (LOG 2026-05-14).

### `pyrrho-small-g1` (SLM, plain SFT, 3-seed mean ± std on same eval split)

| Metric | pyrrho-small-g1 | vs `nano-g1` | vs sklearn baseline |
|---|---|---|---|
| Overall accuracy | **90.01 ± 0.55%** | **+3.88** | **+11.31** |
| False-trustworthy rate | **12.13 ± 1.27%** | **+6.86 (worse)** | **+6.43 (worse)** — **fails gate** |
| Trustworthy recall | **92.09 ± 0.19%** | +12.71 | +22.09 |
| Disputed recall | **87.16 ± 1.54%** | -7.65 | +1.06 |
| Abstain recall | **88.08 ± 2.23%** | -4.86 | +1.58 |
| Tier0 sanity accuracy | **99.44 ± 0.96%** (60-case set) | vs ~83% (encoder) | n/a |
| Decode-time fallback rate | **0.00%** (every case produced a parseable label) | — | — |

Headline finding: pre-trained world knowledge + reasoning depth genuinely lifts overall accuracy and nearly-perfects tier0, but the SLM systematically over-predicts TRUSTWORTHY because plain SFT has no safety-asymmetric signal (no class weights, no label smoothing, no FT-penalized selection metric — all the things `nano-g1` had). The 12.13% FT rate is stable across seeds (11.40 / 11.40 / 13.60), so it's a recipe-level finding, not noise.

## Known limitations

### `pyrrho-nano-g1` (in production)

1. **Multi-source-convergence misclassified as DISPUTED.** When multiple authoritative sources agree on a fact with slight numerical variation, ~57% error rate on this fitz-gov subcategory (n=7). Deferred to v2.
2. **Short clean TRUSTWORTHY contexts trigger over-abstention.** Tier1 training is 62.7% hard cases; the model never learned the short-clean-answer pattern. Fixable in v2.

### `pyrrho-small-g1` (not shipped)

1. **Fails the false-trustworthy gate (12.13% vs 5.7%).** Plain SFT has no anti-FT pressure. Fix paths for a `small-g1.1` re-spin: add class weights / label smoothing to SFT, or — better — do GRPO with asymmetric FT penalty per [ROADMAP §8 Phase 3](ROADMAP.md). Not shipped because deploying it as the production backend would *double* the production false-trustworthy rate vs `nano-g1`, which is exactly the safety axis the encoder was tuned to protect.
2. **Disputed/abstain recall regressed vs encoder.** The SLM's preference for TRUSTWORTHY pulls cases out of those buckets — same root cause as #1.

## Pipeline / tooling — what exists now

| Script | Purpose |
|---|---|
| [`scripts/prepare_data.py`](../scripts/prepare_data.py) | fitz-gov → train/eval/tier0 splits + HF DatasetDict |
| [`scripts/train_encoder.py`](../scripts/train_encoder.py) | Single-run encoder fine-tuning, config-driven, writes manifest.json |
| [`scripts/train_slm.py`](../scripts/train_slm.py) | Single-run SLM QLoRA fine-tune (TRL SFTTrainer + PEFT) with decode-based eval |
| [`scripts/run_seeds.py`](../scripts/run_seeds.py) | Multi-seed orchestrator (encoder-shaped output), aggregates mean ± std |
| [`scripts/aggregate_slm_seeds.py`](../scripts/aggregate_slm_seeds.py) | Multi-seed aggregator for `train_slm.py` outputs (no threshold calibration) |
| [`scripts/sweep.py`](../scripts/sweep.py) | Hyperparameter sweep (coordinate-descent or grid) |
| [`scripts/eval.py`](../scripts/eval.py) | 5-fold CV runner |
| [`scripts/eval_report.py`](../scripts/eval_report.py) | Full per-breakdown evaluation report on a checkpoint |
| [`scripts/compare_runs.py`](../scripts/compare_runs.py) | Diff two runs (or vs baseline), markdown table out |
| [`scripts/inspect_tier0_failures.py`](../scripts/inspect_tier0_failures.py) | Dump misclassified tier0 cases with full context |
| [`scripts/smell_test.py`](../scripts/smell_test.py) | 10-case sanity check (ad-hoc) |
| [`scripts/build_model_card.py`](../scripts/build_model_card.py) | Encoder HF model card builder |
| [`scripts/build_slm_model_card.py`](../scripts/build_slm_model_card.py) | SLM HF model card builder (uses summary.json + LoRA adapter dir) |
| [`tests/test_smoke.py`](../tests/test_smoke.py) | pytest version of the smell test for CI regression |

Reproducibility: every artifact-producing script writes `manifest.json` (git/pip/hw/seed/timing) via `pyrrho.manifest.write_manifest`.

Full methodology, release gates, and W&B conventions in [METHODOLOGY.md](METHODOLOGY.md).

## What's live

- **pyrrho model on HF**: https://huggingface.co/yafitzdev/pyrrho-nano-g1 (public, CC BY-NC 4.0, 1.35 GB: safetensors + FP32 ONNX + INT8 ONNX). Model card aligned to user's trimmed shape (no philosophical aside, no uncalibrated table) — `build_model_card.py` produces this by default. Cross-linked to `yafitzdev/fitz-gov` dataset. (Was `pyrrho-modernbert-base-v1` + Apache-2.0 through 2026-05-19; renamed under the new `pyrrho-{tier}-{generation}` scheme.)
- **fitz-gov dataset on HF**: https://huggingface.co/datasets/yafitzdev/fitz-gov (public, CC BY-NC 4.0, V5.1, three configs: tier1_core / tier0_sanity / validation). Verified `load_dataset(...)` loads cleanly.
- **pyrrho GitHub repo**: public, redesigned README in the fitz-sage style.
- **pyrrho is in production.** fitz-sage **v0.13.0** (shipped 2026-05-15, PyPI + GitHub) replaced its constraint+sklearn governance cascade with `yafitzdev/pyrrho-nano-g1` — loaded as INT8 ONNX, ~30 ms/decision on CPU, zero LLM calls on the governance path. The same release also swapped fitz-sage's chat-call reranker for `Alibaba-NLP/gte-reranker-modernbert-base` (a separate ONNX cross-encoder — fitz-sage's call, applying pyrrho's pattern). See LOG 2026-05-15.
  - Release: https://github.com/yafitzdev/fitz-sage/releases/tag/v0.13.0
  - PyPI: https://pypi.org/project/fitz-sage/0.13.0/
- **`pyrrho-small-g1` release dir staged locally** at `models/pyrrho-small-g1/` — LoRA adapter (`adapter_model.safetensors` + `adapter_config.json`), tokenizer files, chat template, and a 3-seed model card. **Not on HF** pending a fix for the false-trustworthy gate (see Known limitations above). If the user does want to push it as a research artifact (not a production replacement), `huggingface-cli upload yafitzdev/pyrrho-small-g1 models/pyrrho-small-g1/` after a `huggingface-cli login`.

## Immediate next actions

The integration milestone is **closed** — pyrrho v1 is shipped and
`pyrrho-small-g1` is the first generative SLM data point in the family.
Everything below is model-quality upside on an already-live baseline.

1. **Phase 0: V5.1 schema enrichment** — per [ROADMAP.md §8 Phase 0](ROADMAP.md). Retrofit the MoE schema fields (routing, conflict_density, evidence_sufficiency, boundary_proximity, etc.) onto the existing 2,900 V5.1 cases. Cheaper than scaling data first — discoveries here directly improve the synthetic pipeline prompts for V6.

2. **Phase 1: `pyrrho-nano-g1.1`** — retrain the encoder on V5.1-enriched as the apples-to-apples baseline before scaling.

3. **`pyrrho-small-g1.1` (optional fix)** — re-spin the SLM with safety-asymmetric training: either SFT with class weights + label smoothing, or DPO/GRPO with FT-penalized reward. The goal is to land FT ≤ 5.7% while keeping the +3.88 accuracy gain. Skip if the team prefers to wait until V6 and do `pyrrho-small-g2` directly with RL (ROADMAP Phase 3).

4. **fitz-gov v6** — only after Phase 1 ships and validates the enriched schema. See "Things NOT to do".

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
