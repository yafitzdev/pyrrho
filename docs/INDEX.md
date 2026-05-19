# docs/ — start here

This directory holds everything a fresh contributor (or Claude session) needs to pick up the pyrrho project mid-stream. Read in the order below.

---

## Reading order

| # | File | What you'll learn | When to read |
|---|---|---|---|
| 1 | [HANDOFF.md](HANDOFF.md) | **Current status snapshot.** What's trained, validated numbers, immediate next actions, decisions not to relitigate. Gets overwritten as state changes. | First thing, always. |
| 2 | [LOG.md](LOG.md) | **Project history.** Append-only reverse-chronological log of findings, decisions, and experiments. Read for the *why* and the *when*. | When HANDOFF.md mentions something you don't have context on. |
| 3 | [PROJECT.md](PROJECT.md) §1–§5 | Vision, the fitz-gov / fitz-sage / pyrrho triangle, baseline to beat, encoder-vs-SLM rationale. | When you need *why* anything was decided structurally. |
| 4 | [METHODOLOGY.md](METHODOLOGY.md) | End-to-end model-development pipeline. The 8-step process every release follows. | Before producing any new release. |
| 5 | [SETUP.md](SETUP.md) | Environment specifics — RTX 5090 / Blackwell / Windows / WSL2. | First time setting up the project locally. |
| 6 | [PROJECT.md](PROJECT.md) §6–§18 | Full plan: hardware reality, model picks, training recipes, release roadmap, open questions, research notes, original session history. | When you need the deep context behind a model pick or hyperparameter choice. |

---

## Quick reference — what's where

| Need | Path |
|---|---|
| Repository-wide overview | [../README.md](../README.md) |
| Training configs (encoder) | [../configs/encoder/](../configs/encoder/) |
| Training configs (SLM) | [../configs/slm/](../configs/slm/) |
| Hyperparameter sweep grids | [../configs/sweep_grids/](../configs/sweep_grids/) |
| Python library | [../src/pyrrho/](../src/pyrrho/) |
| CLI scripts | [../scripts/](../scripts/) |
| Pytest suites | [../tests/](../tests/) |
| Sibling repo — RAG library | `C:/Users/yanfi/PycharmProjects/fitz-sage` |
| Sibling repo — benchmark dataset | `C:/Users/yanfi/PycharmProjects/fitz-gov` |

---

## Headline result so far

`pyrrho-nano-g1` (3-seed mean ± std, vs published fitz-sage v0.11 sklearn baseline):

- Overall accuracy: **86.13 ± 0.86%** (baseline: 78.7%, **Δ +7.43**)
- False-trustworthy rate: **5.27 ± 0.21%** (baseline: 5.7%, **safer**)
- Trustworthy recall: **79.38 ± 1.64%** (baseline: 70.0%, **Δ +9.38**)

Full validation methodology and per-seed numbers in [HANDOFF.md](HANDOFF.md). Pipeline that produced these in [METHODOLOGY.md](METHODOLOGY.md). The story of how we got these numbers (5 hyperparameter attempts, 3-seed validation, smoke test) lives in [LOG.md](LOG.md).

---

## If you're a Claude session, also check memory

Persistent memory for this project lives at:
`C:\Users\yanfi\.claude\projects\C--Users-yanfi-PycharmProjects-pyrrho\memory\`

`MEMORY.md` is the index. The user-role, feedback, and reference files there capture conventions that aren't repeated in these docs:
- The user is a senior IC. Terse, concrete, no hand-waving.
- Always web-search current model state before recommending — never rely on training priors past ~3 months stale.
- Specific models that are *banned* from suggestions (Qwen 2.5, 35B-class MoEs, Llama-family bases). See PROJECT.md §18 for why.
