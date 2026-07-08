# docs - start here

This directory contains the current handoff plus historical planning notes.
Read in this order.

| # | File | Status | Purpose |
|---|---|---|---|
| 1 | [HANDOFF.md](HANDOFF.md) | Current | Active v2 release state, metrics, links, next actions. |
| 2 | [../README.md](../README.md) | Current | Public project overview and quick start. |
| 3 | [METHODOLOGY.md](METHODOLOGY.md) | Current v2 summary | Release workflow and current script map. |
| 4 | [LOG.md](LOG.md) | Historical | Append-only project history. |
| 5 | [PROJECT.md](PROJECT.md) | Historical v1 plan | Original long-form v1 planning and decisions. |
| 6 | [ROADMAP.md](ROADMAP.md) | Historical roadmap | Older expansion roadmap; useful context, not the v2 contract. |
| 7 | [SETUP.md](SETUP.md) | Environment | Windows/GPU/local setup notes. |

## Current Release

| Artifact | Link |
|---|---|
| Model | [`yafitzdev/pyrrho-v2-nano-g1`](https://huggingface.co/yafitzdev/pyrrho-v2-nano-g1) |
| Dataset | [`yafitzdev/fitz-gov-v2`](https://huggingface.co/datasets/yafitzdev/fitz-gov-v2) |
| Runtime consumer | [`fitz-sage`](https://github.com/yafitzdev/fitz-sage) |

## Current Shape

Pyrrho v2 emits:

- `evidence_verdict`: `SUFFICIENT`, `DISPUTED`, `INSUFFICIENT`
- `failure_mode`: actionable reason for insufficient or disputed evidence
- `retrieval_intents`: multi-label retrieval task metadata
- `evidence_kinds`: multi-label evidence-surface metadata

Older v1 wording appears in historical docs and in fitz-sage runtime
`AnswerMode` API references. It is not the native v2 model label shape.
