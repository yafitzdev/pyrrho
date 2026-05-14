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

## What's live

- **Model on HF**: https://huggingface.co/yafitzdev/pyrrho-modernbert-base-v1 (public, Apache-2.0, 1.35 GB: safetensors + FP32 ONNX + INT8 ONNX). Model card aligned to user's trimmed shape (no philosophical aside, no uncalibrated table) — `build_model_card.py` produces this by default. Cross-linked to `yafitzdev/fitz-gov` dataset.
- **Dataset on HF**: https://huggingface.co/datasets/yafitzdev/fitz-gov (public, MIT, V5.1, three configs: tier1_core / tier0_sanity / validation). Verified `load_dataset(...)` loads cleanly.
- **pyrrho GitHub repo**: public, **redesigned README** matches fitz-sage style (centered header, hero comparison table, "Why pyrrho?" feature blockquotes, collapsible details). 12+ commits pushed.
- **fitz-gov README**: has new 🤗 HuggingFace banner + callout block with `load_dataset` quickstart and link to pyrrho baseline. New `scripts/upload_to_hf.py` in that repo (uncommitted in fitz-gov as of paused state).

## IN-FLIGHT WORK — fitz-sage v0.12.0 push (paused 2026-05-14 evening, context limit)

User wanted to push fitz-sage v0.12.0 before integrating pyrrho into it. Release commit was sitting locally but tests were failing. Mid-debug when context window filled.

**Status at pause:**

1. ✅ **Lint clean**: ruff (1 manual `# noqa: F401` on `LLMReranker` re-export + 1 auto-fixed unused pytest import), black (15 files reformatted), isort (2 files fixed). All commits sit uncommitted in fitz-sage working tree.
2. ✅ **216 fixture errors fixed**: `FitzKragConfig` rejected legacy `embedding` / `vector_db` / `vector_db_kwargs` keys (Cloud-removal migration). Patched `tests/e2e_krag/runner.py`, `tests/e2e_krag/config.py`, and `tests/test_config.yaml` to drop legacy keys + switch `ollama`/`cohere` providers → `endpoint` with explicit `chat_base_url` per tier (Ollama on `:11434/v1`, OpenAI on api).
3. ✅ **`openai` dep installed** into fitz-sage venv (was missing; needed by `endpoint` provider).
4. ✅ **Test isolation bug diagnosed**: `tests/unit/test_krag_detection.py` and `test_krag_engine.py` use `@patch("...SqliteConnectionManager")` without the `reset_sqlite_singleton` fixture → the singleton `_instance` leaks a `MagicMock` into subsequent tests, causing **25 cascading failures** in test_vocabulary / test_section_store / test_krag_guardrails (all of which pass in isolation but fail in full-suite order).
5. ⚠️ **Fix attempted but caused hang**: changed `reset_sqlite_singleton` in `tests/unit/conftest.py` to `autouse=True`. Test run hung for 29+ min with 0 bytes of output. **Likely a deadlock on the singleton's `_lock`** when `reset_instance()` is called in `before/after` and an inner test holds the lock. User killed the run and asked me to stop and update HANDOFF before resuming.

**Files modified in fitz-sage (working tree, NOT committed):**
- `fitz_sage/llm/providers/__init__.py` (added F401 noqa)
- 15 black-reformatted files across `fitz_sage/` and `tests/`
- `tests/unit/conftest.py` (autouse change — **this is the suspect for the deadlock**)
- `tests/e2e_krag/runner.py` (dropped 3 legacy config keys from both `config_dict` builds; plumbed `chat_base_url` + `chat_api_key_env`)
- `tests/e2e_krag/config.py` (simplified `get_tier_config` to drop embedding/vector_db)
- `tests/test_config.yaml` (rewrote tiers to use `endpoint` provider)

**Resume plan when fitz-sage push picks up again:**

a. **Revert the autouse change** in `tests/unit/conftest.py` (back to opt-in `reset_sqlite_singleton`). The autouse is too risky given the deadlock.
b. **Instead, add explicit `reset_sqlite_singleton` requests** to the two offending test files (`test_krag_detection.py` and `test_krag_engine.py`) — either as method-level fixture parameters or a class-level autouse inside those files only.
c. Re-run `pytest tests/unit/ -q` — expect ~1573 pass / 0 fail.
d. Commit the formatting + test-fixture migration as a single pre-release commit. Tag v0.12.0. Push.
e. Then resume the pyrrho-into-fitz-sage integration plan.

## Immediate next actions (after fitz-sage v0.12.0 ships)

1. **Build pyrrho into fitz-sage as the default governance backend.** The moat-realizing step. Replace the constraint+sklearn pipeline with an inference call to `yafitzdev/pyrrho-modernbert-base-v1` (INT8 ONNX). Real users get the +7 pt accuracy, +50× CPU speedup, and zero LLM dependency. Triggers a fitz-sage v0.13.0 release with "now powered by pyrrho" as the headline. ~4–8 hr.

2. **SLM track** — fine-tune Qwen3.5-0.8B to see if it fixes `multi_source_convergence` (the only real v1 limitation). Pretraining world knowledge + reasoning depth should address the failure mode. ~1 hr for the fine-tune. Independent from #1, can run in parallel.

3. **Cross-link the triangle in fitz-sage README** — already partially done; verify "powered by pyrrho" / "models trained on this benchmark" sections cover the model + dataset HF links. ~10 min.

4. **Cross-architecture validation** (optional). Train DeBERTa-v3-base with the same encoder config. Defensive content for v1.5 model card. ~10 min.

5. **v2 work** — bigger / more diverse augmentation set (~100-200 cases per failing subcategory, plus boundary counter-examples in the DISPUTED class), then retrain.

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
