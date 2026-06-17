# docs/ — start here

This directory holds everything a fresh contributor (or Claude session) needs to pick up the pyrrho project mid-stream. Read in the order below.

---

## Reading order

| # | File | What you'll learn | When to read |
|---|---|---|---|
| 1 | [HANDOFF.md](HANDOFF.md) | **Current status snapshot.** What's trained, validated numbers, immediate next actions, decisions not to relitigate. Gets overwritten as state changes. | First thing, always. |
| 2 | [GOAL.md](GOAL.md) | **Active north-star.** The current target is `pyrrho-MoE-g4-real`: clean ~4B/A0.4B sparse MoE with stock runtime compatibility as gate zero. | Before doing MoE work or choosing the next task. |
| 3 | [LOG.md](LOG.md) | **Project history.** Append-only reverse-chronological log of findings, decisions, and experiments. Read for the *why* and the *when*. | When HANDOFF.md mentions something you don't have context on. |
| 4 | [PROJECT.md](PROJECT.md) §1–§5 | Vision, the fitz-gov / fitz-sage / pyrrho triangle, baseline to beat, encoder-vs-SLM rationale. | When you need *why* anything was decided structurally. |
| 5 | [METHODOLOGY.md](METHODOLOGY.md) | End-to-end model-development pipeline. The 8-step process every release follows. | Before producing any new release. |
| 6 | [CODE_MODALITY_AXES.md](CODE_MODALITY_AXES.md) | Current structured/code modality axis decision, code coverage audit, targeted patch status, and local-control results. | Before touching structured/code candidate rows or code OOD probes. |
| 7 | [FITZ_GOV_SAGE_PLAN.md](FITZ_GOV_SAGE_PLAN.md) | V10-clean `fitz-gov-sage` plan: source version, stage-aware row shape, subagent prompt, workpack tooling, and training gates. | Before creating or training the sage-shaped encoder line. |
| 8 | [FITZ_GOV_SAGE_GENERATION_RUNBOOK.md](FITZ_GOV_SAGE_GENERATION_RUNBOOK.md) | Exact reproducible `fitz-gov-sage` generation procedure, 100-row pilot split, worker prompt template, audit command, and baseline comparison. | Before generating any more sage-shaped rows. |
| 9 | [FITZ_GOV_SAGE_MANUAL_MESSY_PROCEDURE.md](FITZ_GOV_SAGE_MANUAL_MESSY_PROCEDURE.md) | Canonical repeatable procedure for manual GPT-5.4 messy-pack generation, semantic QA, repair, and gates. | Before scaling or repairing sage-shaped data. |
| 10 | [PYRRHO_MOE_ARCHITECTURE.md](PYRRHO_MOE_ARCHITECTURE.md) | Canonical `pyrrho-MoE` architecture spec: 4B-A0.4B parameter math, expert layout, upcycling/distillation plan, training gates. | Before touching MoE implementation or model selection. |
| 11 | [PYRRHO_MOE_MVP_RUN_GUIDE.md](PYRRHO_MOE_MVP_RUN_GUIDE.md) | Short operator guide for running the published `pyrrho-MoE-g3-mvp` GGUF package with full-sequence label scoring. | Before trying the MoE MVP locally or writing downstream consumer docs. |
| 12 | [OLMOE_TRAINING_PATH_2026-05-31.md](OLMOE_TRAINING_PATH_2026-05-31.md) | Current `g4-real` OLMoE carrier training path: SFT smoke, donor audit, donor initialization result, and next bounded training step. | Before starting `pyrrho-MoE-g4-real` training/upcycling. |
| 13 | [SETUP.md](SETUP.md) | Environment specifics — RTX 5090 / Blackwell / Windows / WSL2. | First time setting up the project locally. |
| 14 | [PROJECT.md](PROJECT.md) §6–§18 | Full plan: hardware reality, model picks, training recipes, release roadmap, open questions, research notes, original session history. | When you need the deep context behind a model pick or hyperparameter choice. |

---

## Quick reference — what's where

| Need | Path |
|---|---|
| Repository-wide overview | [../README.md](../README.md) |
| Training configs (encoder) | [../configs/encoder/](../configs/encoder/) |
| Training configs (SLM) | [../configs/slm/](../configs/slm/) |
| Training configs (MoE) | [../configs/moe/](../configs/moe/) |
| Hyperparameter sweep grids | [../configs/sweep_grids/](../configs/sweep_grids/) |
| `pyrrho-MoE` architecture spec | [PYRRHO_MOE_ARCHITECTURE.md](PYRRHO_MOE_ARCHITECTURE.md) |
| Current MoE seed scan | [MOE_SEED_SEARCH_2026-05-26.md](MOE_SEED_SEARCH_2026-05-26.md) |
| Current MoE upcycling decision | [MOE_UPCYCLING_DECISION_2026-05-26.md](MOE_UPCYCLING_DECISION_2026-05-26.md) |
| Current OLMoE training path | [OLMOE_TRAINING_PATH_2026-05-31.md](OLMOE_TRAINING_PATH_2026-05-31.md) |
| Active MoE MVP goal | [GOAL.md](GOAL.md) |
| Published MoE MVP run guide | [PYRRHO_MOE_MVP_RUN_GUIDE.md](PYRRHO_MOE_MVP_RUN_GUIDE.md) |
| Sage-shaped encoder plan | [FITZ_GOV_SAGE_PLAN.md](FITZ_GOV_SAGE_PLAN.md) |
| Sage generation runbook | [FITZ_GOV_SAGE_GENERATION_RUNBOOK.md](FITZ_GOV_SAGE_GENERATION_RUNBOOK.md) |
| Sage manual messy-pack procedure | [FITZ_GOV_SAGE_MANUAL_MESSY_PROCEDURE.md](FITZ_GOV_SAGE_MANUAL_MESSY_PROCEDURE.md) |
| Sage transform prompt | [prompts/FITZ_GOV_SAGE_TRANSFORM_SUBAGENT.md](prompts/FITZ_GOV_SAGE_TRANSFORM_SUBAGENT.md) |
| Sage messy transform worker prompt | [prompts/FITZ_GOV_SAGE_MESSY_TRANSFORM_WORKER.md](prompts/FITZ_GOV_SAGE_MESSY_TRANSFORM_WORKER.md) |
| Sage semantic QA worker prompt | [prompts/FITZ_GOV_SAGE_SEMANTIC_QA_WORKER.md](prompts/FITZ_GOV_SAGE_SEMANTIC_QA_WORKER.md) |
| Sage repair worker prompt | [prompts/FITZ_GOV_SAGE_REPAIR_WORKER.md](prompts/FITZ_GOV_SAGE_REPAIR_WORKER.md) |
| Structured/code modality status | [CODE_MODALITY_AXES.md](CODE_MODALITY_AXES.md) |
| Public model card template | [MODEL_CARD_TEMPLATE.md](MODEL_CARD_TEMPLATE.md) |
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

`pyrrho-nano-g2` is published at [`yafitzdev/pyrrho-nano-g2`](https://huggingface.co/yafitzdev/pyrrho-nano-g2). Held-out V7 test, 3-seed mean ± std: **95.24 ± 0.48%** accuracy and **3.48 ± 0.40%** false-trustworthy. The local release mirror is `models/pyrrho-nano-g2/`.

`pyrrho-nano-g3` is published at [`yafitzdev/pyrrho-nano-g3`](https://huggingface.co/yafitzdev/pyrrho-nano-g3). Held-out V8 test, 3-seed mean ± std: **97.52 ± 0.43%** accuracy and **1.42 ± 0.16%** false-trustworthy. The local release mirror is `models/pyrrho-nano-g3/`.

---

## Current dataset state

Published benchmark contract for new work: `yafitzdev/fitz-gov` **V8.0.1**, 24,592 query-grouped rows with explicit `meta.modality: "unstructured"`. V7.0.1 remains the published `pyrrho-nano-g2` contract, and V6.0.0 remains the 2,980-row enriched baseline for V6 apples-to-apples comparisons.

Published V8: `yafitzdev/fitz-gov` **v8.0.1** on Hugging Face, **24,592 rows** total: V6 2,980 + V7 7,520 + V8 14,092. Default config `v8` has query-grouped splits: train=19,674 / validation=2,459 / test=2,459, and current rows carry `meta.modality: "unstructured"`. Local pyrrho prep is available at `data/processed_v8`; MoE multitask prep is available at `data/moe_v8` with a strict audit showing **0** required-field misses.

---

## If you're a Codex or Claude session, also check memory

Persistent memory for this project lives at:
`C:\Users\yanfi\.Codex\projects\C--Users-yanfi-PycharmProjects-pyrrho\memory\`

`MEMORY.md` is the index. The user-role, feedback, and reference files there capture conventions that aren't repeated in these docs:
- The user is a senior IC. Terse, concrete, no hand-waving.
- Always web-search current model state before recommending — never rely on training priors past ~3 months stale.
- Specific models that are *banned* from suggestions (Qwen 2.5, 35B-class MoEs, Llama-family bases). See PROJECT.md §18 for why.
