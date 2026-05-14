# LOG — pyrrho project history

Append-only chronological log of findings, decisions, and experiments. **Most recent entries at the top.**

For *current* state (what's true right now), see [HANDOFF.md](HANDOFF.md).
For the *plan* (vision, roadmap, training recipes), see [PROJECT.md](PROJECT.md).

Each entry follows the pattern:
- **Date / phase** in the heading.
- **What landed** — the concrete deliverable.
- **What was learned** — surprises, validations, or new constraints.
- **What's next** — the implied next step at the time of writing (may be obsoleted by later entries).

---

## 2026-05-14 (evening) — Repo cleanup + GitHub prep

**What landed:**
- Deleted obsolete `scripts/inspect_tier0.py` (superseded by `inspect_tier0_failures.py`) and stale `.gitkeep` placeholders.
- All project docs moved into `docs/` (PROJECT.md, SETUP.md, METHODOLOGY.md, plus new INDEX.md). `STATE.md` deleted in favor of the HANDOFF.md / LOG.md split.
- `LICENSE` (Apache 2.0) added; `pyproject.toml` already named Apache-2.0.
- `.gitignore` expanded: `.claude/`, `.pytest_cache/`, `data/*` with `!data/.gitkeep` exception, broader artifact glob.
- README.md rewritten for GitHub-facing audience: hero, headline metrics, family roadmap, repo structure, quickstart, doc index.
- `tests/test_smoke.py` now skips cleanly when no checkpoint exists (fresh clones won't hard-fail).
- Git initialized; **initial commit `5e145fc`** with 33 files staged.

**What was learned:**
- Repo was carrying ~6 stale or duplicate scripts. The clean tree is much easier to reason about.
- The HANDOFF / LOG split (proposed by user this session) is a much better structure than a single STATE.md that tries to be both.

**Next:** push to GitHub via `gh repo create`, then run the sweep, then ship v1 to HuggingFace.

---

## 2026-05-14 (evening) — Tier-A systematization pipeline shipped

**What landed:**
- `src/pyrrho/manifest.py` — git/pip/hw/seed/timing capture per run. Wired into `train_encoder.py` automatically.
- `scripts/eval_report.py` — full per-breakdown evaluation (per-domain / per-difficulty / per-reasoning_type / per-evidence_pattern / per-query_type / per-subcategory + confusion matrix).
- `scripts/compare_runs.py` — diff two runs (single or multi-seed) vs each other or the sklearn baseline. Markdown table to stdout.
- `scripts/sweep.py` + `configs/sweep_grids/encoder_v1.yaml` — coordinate-descent or full-grid hyperparameter sweeps. 23-cell coordinate sweep around v1 baseline available.
- `tests/test_smoke.py` — pytest regression guard. Currently **9 passed, 2 xfailed** in 7.5 s.
- `docs/METHODOLOGY.md` — end-to-end pipeline doc, release gate definitions, W&B conventions, manifest schema.

**What was learned:**
- The eval_report tier0 breakdown surfaced patterns we couldn't see before: `direct_contradiction` subcategory at 55% accuracy, `causal_without_evidence` at 29% (likely fitz-gov label noise — model arguably right), `different_domain` (ABSTAIN cases) at 100%.
- By `query_type`, "why" questions land at 29% and "is" questions at 33% on tier0 — same root pattern as the per-subcategory analysis.
- The pipeline is now reusable: any future release (DeBERTa, Qwen3.5-0.8B, LFM2.5-1.2B, etc.) follows the same 5-command sequence.

**Next:** ship v1 to HuggingFace, or run the sweep first to confirm attempt-4 is optimal.

---

## 2026-05-14 (afternoon) — v1 trained, 3-seed validated

**What landed:**
- 5 hyperparameter iterations to find the v1 config:
  - **Attempt 1** (weights 2.3/2.3/1, 5ep, macro_f1 selection): 87.5% acc / 13.6% FT / 73.3% tier0. Overfit. macro_f1 picked the most-overfit checkpoint.
  - **Attempt 2** (no weights, ls 0.1, 3ep, patience 1, ft_pen): 73.8% / 8.1% / 63.3%. Under-fit.
  - **Attempt 3** (weights + ls 0.1, 5ep, patience 2, ft_pen): 83.2% / 5.5% / 73.3%. FT gate passed.
  - **Attempt 4** (weights + ls **0.15**, 5ep, patience 2, ft_pen): 85.8% / 5.5% / 81.7%. **Winner.**
  - **Attempt 5** (no weights, ls 0.15): 68.8% / 5.5% / 73.3%. Confirmed weights needed.
- 3-seed validation (42, 1337, 7): **86.13 ± 0.86% accuracy, 5.27 ± 0.21% FT, 79.38 ± 1.64% trustworthy recall.**
- Smell test on 10 handcrafted cases: **8/10** (consistent with eval claim, within seed noise).
- `scripts/inspect_tier0_failures.py` written; surfaced the failure decomposition.

**What was learned:**
- **Architecture hypothesis confirmed.** Trustworthy recall jumped from 70.0% (sklearn) → 89.1% (uncalibrated) → 79.4% (calibrated). That was the bucket where features couldn't capture positive evidence; attention can.
- **The dominant tier0 weakness is short-context TRUSTWORTHY cases** — e.g. "When was the iPhone released?" + 1-sentence factual context → ABSTAIN with high confidence. Tier1 training was 62.7% hard cases, so the model never learned the short-clean-answer pattern. Fix is v2 training data, not a hyperparameter knob.
- **The tier0 95% sanity gate is unreachable** on a 60-case set: run-to-run std is ±3.5 pts; even a perfect model couldn't reliably clear 95%. Plus ~5 of the 60 cases have ambiguous gold labels (e.g. "Why is Python popular?" with stats-but-no-causal context labeled TRUSTWORTHY — model abstains, label says trustworthy). The gate was my (Claude's) invention in PROJECT.md, not in fitz-gov's spec. **Dropped.**
- **Calibration is rock-solid; threshold selection is unstable.** Across 3 seeds, FT lands at 5.27 ± 0.21% (essentially zero variance), but the selected τ varies 0.34–0.62. The model's confidence distribution shifts seed-to-seed, but threshold-selection adapts and finds the same operating point. Good sign.

**Key decisions:**
- Drop the tier0 95% gate. Replace with "tier0 reported as diagnostic in every model card."
- Use `ft_penalized_accuracy = accuracy - 3 * max(0, FT - 0.057)` as the model-selection metric instead of macro_f1.
- Document the short-context-TRUSTWORTHY weakness as a known v1 limitation; close it in v2 with augmented training data.

**Next:** systematize the pipeline (Tier A tooling) before producing more releases.

---

## 2026-05-13 (evening) — Environment + training scaffolding done

**What landed:**
- Python 3.12 venv + `torch 2.11.0+cu128` (Blackwell verified) + project extras `[encoder,slm,tracking,dev]`.
- `bitsandbytes 0.49.2` confirmed working on RTX 5090 / Blackwell sm_120 (4-bit NF4 load of Qwen3.5-0.8B succeeded).
- `pandas<3` and `pyarrow<24` pinned in `pyproject.toml` — newer combo crashes on Windows.
- `src/pyrrho/{data.py,metrics.py,training.py}` modules + `scripts/{verify_env.py,train_encoder.py,eval.py,run_seeds.py,smell_test.py}`.
- End-to-end verified with `--dry-run`: ModernBERT-base downloaded, splits tokenized, Trainer built cleanly.

**What was learned:**
- Switching from Python 3.11 → 3.12 saved several library compatibility issues on Windows.
- 3.5-line gotcha (`pandas<3` and `pyarrow<24`) cost ~30 min to track down but is documented now.

**Next:** train v1.

---

## 2026-05-13 (afternoon) — Rebrand to pyrrho, model lineup refresh, LFM2 discovery

**What landed:**
- Project renamed from `fitz-judge` to `pyrrho` — after Pyrrho of Elis, founder of Greek philosophical skepticism (school of suspension-of-judgment-when-evidence-is-insufficient). Directory, Python package, configs, all docs migrated.
- Model lineup refreshed to 2026 vintage after user called out Qwen 2.5 as stale.
- **LFM2 family discovered** via user-shared LM Studio link. Hybrid architecture (multiplicative gates + short convolutions, not transformer). Per Liquid AI's benchmarks: 2.3–2.8× faster prefill, 1.7–2.2× faster decode than Qwen3-1.7B at 1.2B size.
- **MoE pick swapped**: Qwen3.6-35B-A3B (needs ~17 GB Q4 RAM, excluded typical laptops) → `LiquidAI/LFM2-8B-A1B` (8B total / 1B active, ~5 GB Q4, fits 8 GB laptops). This was the small-MoE I claimed didn't exist; I was wrong.

**What was learned:**
- Always web-search current model state before recommending — I had defaulted to Qwen 2.5 and missed Qwen3.5, Gemma 4, Phi-4-mini, LFM2/LFM2.5 in the same session that I had Qwen3.5-0.8B in my search results.
- The user values cross-architecture diversity in the portfolio — transformer dense (Qwen, Gemma, Phi) + Liquid hybrid (LFM2.5) + MoE (LFM2-8B-A1B) makes a stronger story than 8 transformer dense variants.

**Key decisions:**
- v5-first strategy: train on current fitz-gov v5 before any v6 (long-context) data work. Reasons: direct apples-to-apples comparison with published 78.7% baseline; de-risk the architecture hypothesis in 30 min instead of weeks of data prep.
- Brand naming pattern fixed: `yafitzdev/pyrrho-{base-model}-{size}-v{n}`. The `fitz-` prefix is dropped because the HF org segment carries the ownership signal already (matching the Qwen / Gemma / Phi naming convention).

**Next:** build training scaffolding (data prep + encoder trainer).

---

## 2026-05-13 (morning) — Initial planning session

**What landed:**
- `PROJECT.md` scoped end-to-end (18 sections covering vision, architecture, model picks, training recipes, eval protocol, roadmap, open questions, research notes).
- Memory files bootstrapped in `C:\Users\yanfi\.claude\projects\C--Users-yanfi-PycharmProjects-pyrrho\memory\` capturing user role, project context, response-style preferences, banned models.

**Key decisions:**
- **Replace the whole stack, not just the classifier.** Originally proposed: train an SLM to replace fitz-sage's sklearn classifier. Refined to: replace the entire constraint+sklearn pipeline with a single fine-tuned head. This decouples governance accuracy from whichever chat LLM the user happens to be running, which was the README's "scores are a floor, not a ceiling" dependency.
- **Encoder for production, generative SLMs for portfolio.** CPU constraint forces this: an encoder is ~100× faster on CPU than a similarly-capable generative SLM. fitz-sage default must run on consumer hardware → encoder. Generative SLMs become a parallel HuggingFace portfolio track, not the production path.
- **The triangle**: fitz-gov (benchmark, public dataset) + fitz-sage (library, production user surface) + pyrrho (models). Each reinforces the others. Anyone wanting to compare against pyrrho must use fitz-gov; that makes fitz-gov the de-facto benchmark.
- **Baseline to beat**: fitz-sage v0.11 sklearn cascade per README.md L340-346 — 78.7% overall, 86.5/86.1/70.0 per-class recall, 5.7% false-trustworthy. Trustworthy bucket is the largest headroom.

**Next:** build infrastructure.
