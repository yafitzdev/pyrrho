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
| `pyrrho-small-g1` (Qwen3.5-0.8B + LoRA, V5.1 plain SFT) | Trained + 3-seed validated locally. **Not on HF — fails the FT gate (12.13% vs 5.7% bar).** Release dir staged at `models/pyrrho-small-g1/`. See LOG 2026-05-20 evening. |
| `pyrrho-small-g1.1` (Qwen3.5-0.8B + LoRA, V5.1 SFT with class weights + label smoothing) | Trained + 3-seed validated locally. **Still not on HF — FT improved 12.13 → 9.31% but still misses 5.7% gate by ~3.6 pts.** Release dir staged at `models/pyrrho-small-g1.1/`. See LOG 2026-05-20 night. |
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

Headline finding: pre-trained world knowledge + reasoning depth genuinely lifts overall accuracy and nearly-perfects tier0, but the SLM systematically over-predicts TRUSTWORTHY because plain SFT has no safety-asymmetric signal. 12.13% FT rate is stable across seeds (11.40 / 11.40 / 13.60), so it's a recipe-level finding, not noise.

### `pyrrho-small-g1.1` (SLM, same SFT + class_weights=[2.3, 2.3, 1.0] + label_smoothing=0.15, 3-seed mean ± std)

Recipe-fix re-spin of g1 with the encoder's anti-FT regularization transplanted onto the token-level CE loss (per-example weighting in `WeightedLossSFTTrainer`).

| Metric | pyrrho-small-g1.1 | vs g1 | vs `nano-g1` |
|---|---|---|---|
| Overall accuracy | **89.55 ± 1.40%** | -0.46 | +3.42 |
| False-trustworthy rate | **9.31 ± 1.06%** | **-2.82 (improved)** | +4.04 — **still fails gate by ~3.6 pts** |
| Trustworthy recall | 89.00 ± 2.45% | -3.09 (model less aggressive on T — by design) | +9.62 |
| Disputed recall | **91.60 ± 1.13%** | **+4.44 (improved)** | -3.21 |
| Abstain recall | 88.81 ± 2.56% | +0.73 | -4.13 |
| Tier0 sanity accuracy | **96.67 ± 0.00%** | -2.77 | vs ~83% (encoder) |
| Decode-time fallback rate | **0.00%** | — | — |
| Per-seed FT rate | 8.09 / 9.93 / 9.93 | — | — |

Direction is exactly what the recipe predicts: model is less aggressive on TRUSTWORTHY (TR ↓ 3.09), more aggressive on DISPUTED (DR ↑ 4.44), FT drops ~3 pts. But the absolute FT of 9.31% is still ~3.6 pts above the 5.7% gate — the encoder's [2.3, 2.3, 1.0] + 0.15 smoothing recipe lands 5.27% on the encoder but only 9.31% on the SLM. Token-level CE on the assistant turn diffuses the safety pressure across many tokens (~11/example: 6 think-block + 3–5 label tokens + im_end) while the encoder's class-weighted CE on a single classification head concentrates it. Bumping to more aggressive interventions (e.g., class_weights=[5, 5, 1], label_smoothing=0.25, or moving to ft_penalized_accuracy selection / threshold-based post-processing / DPO) is the next lever set for a `small-g1.2`.

## Known limitations

### `pyrrho-nano-g1` (in production)

1. **Multi-source-convergence misclassified as DISPUTED.** When multiple authoritative sources agree on a fact with slight numerical variation, ~57% error rate on this fitz-gov subcategory (n=7). Deferred to v2.
2. **Short clean TRUSTWORTHY contexts trigger over-abstention.** Tier1 training is 62.7% hard cases; the model never learned the short-clean-answer pattern. Fixable in v2.

### `pyrrho-small-g1` (not shipped)

1. **Fails the false-trustworthy gate (12.13% vs 5.7%).** Plain SFT has no anti-FT pressure. See g1.1 below for the partial fix.
2. **Disputed/abstain recall regressed vs encoder.** The SLM's preference for TRUSTWORTHY pulls cases out of those buckets — same root cause as #1.

### `pyrrho-small-g1.1` (not shipped)

1. **Still fails the false-trustworthy gate (9.31% vs 5.7%).** Class weights + label smoothing closed ~40% of the gap from g1 (12.13 → 9.31), but the encoder-style recipe transplant under-delivers on the SLM because token-level CE diffuses the safety pressure across many assistant-turn tokens. To clear the 5.7% gate without ditching SFT, would need stronger weights (e.g., 5.0/5.0/1.0), stronger smoothing (0.25+), or `ft_penalized_accuracy` checkpoint selection (currently `eval_loss`). Cleaner long-term fix: DPO/GRPO with asymmetric FT-penalized reward per [ROADMAP §8 Phase 3](ROADMAP.md).
2. **Tier0 dropped from 99.44 → 96.67%.** The class-weight pressure makes the model more cautious in general, costing it 2 tier0 cases. Within the dropped-95%-gate budget, but worth noting.

## Pipeline / tooling — what exists now

| Script | Purpose |
|---|---|
| [`scripts/prepare_data.py`](../scripts/prepare_data.py) | fitz-gov → train/eval/tier0 splits + HF DatasetDict |
| [`scripts/train_encoder.py`](../scripts/train_encoder.py) | Single-run encoder fine-tuning, config-driven, writes manifest.json |
| [`scripts/train_slm.py`](../scripts/train_slm.py) | Single-run SLM QLoRA fine-tune (TRL SFTTrainer + PEFT) with decode-based eval; auto-uses `WeightedLossSFTTrainer` (per-example class weights + label smoothing) when the config sets `training.class_weights` or `training.label_smoothing` |
| [`scripts/eval_slm.py`](../scripts/eval_slm.py) | Eval-only path for a saved SLM LoRA adapter — re-runs the decode-based eval pass without re-training. Use when the in-script eval was interrupted (e.g., the stdout-buffering hang we hit on g1.1 seed 1337) |
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
- **fitz-gov dataset on HF**: https://huggingface.co/datasets/yafitzdev/fitz-gov (public, CC BY-NC 4.0, **V6.0.0**, three configs: tier1_core / tier0_sanity / validation). V6 adds LLM-enriched signals to all 2,980 cases: `query_rewritten`, per-context `summary`/`relevance_to_query`/`anchor_period`, governance signals (`hallucination_pressure`, `retrieval_retry_value`, `query_evidence_alignment`, `answer_coverage`, `boundary_proximity.distance`), and `near_miss_reason`. Convenience top-level `label` and `tier` fields added. Uploaded 2026-05-20.
- **pyrrho GitHub repo**: public, redesigned README in the fitz-sage style.
- **pyrrho is in production.** fitz-sage **v0.13.0** (shipped 2026-05-15, PyPI + GitHub) replaced its constraint+sklearn governance cascade with `yafitzdev/pyrrho-nano-g1` — loaded as INT8 ONNX, ~30 ms/decision on CPU, zero LLM calls on the governance path. The same release also swapped fitz-sage's chat-call reranker for `Alibaba-NLP/gte-reranker-modernbert-base` (a separate ONNX cross-encoder — fitz-sage's call, applying pyrrho's pattern). See LOG 2026-05-15.
  - Release: https://github.com/yafitzdev/fitz-sage/releases/tag/v0.13.0
  - PyPI: https://pypi.org/project/fitz-sage/0.13.0/
- **`pyrrho-small-g1` release dir staged locally** at `models/pyrrho-small-g1/` — LoRA adapter (`adapter_model.safetensors` + `adapter_config.json`), tokenizer files, chat template, and a 3-seed model card. **Not on HF** pending a fix for the false-trustworthy gate (see Known limitations above).
- **`pyrrho-small-g1.1` release dir staged locally** at `models/pyrrho-small-g1.1/` — same layout, includes the model card that documents the class-weight + label-smoothing recipe. **Not on HF** — still fails the FT gate (9.31% vs 5.7%), though it's strictly closer to the target than g1.
- **Run `huggingface-cli upload yafitzdev/pyrrho-small-g1.1 models/pyrrho-small-g1.1/`** after `huggingface-cli login` if you want to push either as a research artifact.

## Immediate next actions

The integration milestone is **closed** — pyrrho v1 is shipped and
`pyrrho-small-g1` is the first generative SLM data point in the family.
Everything below is model-quality upside on an already-live baseline.

1. ~~**Phase 0: V5.1 schema enrichment**~~ **COMPLETE (2026-05-20).** All 2,980 V5.1 cases LLM-enriched with V6+ schema fields (query_rewritten, context summaries, governance signals, boundary_proximity, near_miss_reason) via Sonnet subagents + LM Studio local worker. 0 TODO markers remain in vault. See LOG 2026-05-20 evening.

   **Phase 0c: V6 completion** **COMPLETE (2026-05-21).** Added the 4 MoE-training fields: per-chunk `boundary_quality`, per-case `governance.evidence_bias_score`, per-case `input.evidence_chain.{order,reasoning}` (multi-chunk only), and per-case `meta.grounding_targets` (`gold_answer` + per-sentence `attributions`, TRUSTWORTHY only). 2,979/2,980 cases complete (99.97%); one residual case (`t1_qualify_medium_101`, Terravax vaccine query) couldn't get through Sonnet's safety classifier — backfill via LM Studio when next available. Re-uploaded to `yafitzdev/fitz-gov` v6.0.0 (16.4 MB, up from 12.9 MB).

2. **Phase 1: `pyrrho-nano-g1.1`** — retrain the encoder on fitz-gov V6 (the V5.1-enriched dataset now live on HF) as the apples-to-apples baseline before scaling. First step: update `scripts/prepare_data.py` to read the V6 vault JSONL schema instead of the legacy flat tier JSON files.

3. **`pyrrho-small-g1.2` (optional, only if SLM is on critical path)** — g1.1 closed ~40% of the FT gap with the encoder's recipe. Next lever set: more aggressive class weights (e.g., 5.0/5.0/1.0), stronger label smoothing (0.25+), `ft_penalized_accuracy` checkpoint selection, or threshold-based post-processing on the TRUSTWORTHY token logit at decode time. Cleanest fix is DPO/GRPO with asymmetric FT reward but that's properly a Phase 3 (V6) item. Skip if the team is happy to wait for `small-g2` on V6.

4. ~~**fitz-gov v6**~~ **DONE (2026-05-20).** V5.1-enriched = V6 (executive call). Uploaded to `yafitzdev/fitz-gov` as V6.0.0. See LOG 2026-05-20 evening.

## Release gates (the bar any pyrrho model must clear before shipping)

Measured on the eval split, mean across 3 seeds:

- **Overall accuracy ≥ 78.7%** — matches fitz-sage v0.11 sklearn baseline.
- **False-trustworthy rate ≤ 5.7%** — matches baseline; the production safety axis.

The originally-planned **tier0 95% sanity gate has been dropped** (see LOG 2026-05-14 afternoon). With 60 cases, run-to-run variance is ±3.5 pts and ~5 of the 60 cases have ambiguous gold labels. Tier0 is reported as a diagnostic in every model card, not a gate.

## Things NOT to do (already decided — don't relitigate)

- ❌ Don't propose Qwen 2.5 anything — stale (Nov 2024). Use Qwen 3.5+ family.
- ❌ Don't propose 35B-class MoE bases — violates the universal CPU-runnable constraint.
- ❌ Don't propose Llama-family bases — license is more restrictive than Apache-2.0.
- ~~❌ Don't start fitz-gov v6~~ fitz-gov V6.0.0 shipped 2026-05-20 (V5.1-enriched schema overlay). Long-context / domain expansion for V7+ still deferred until pyrrho v2.
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
