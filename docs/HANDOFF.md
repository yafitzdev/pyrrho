# HANDOFF - pyrrho current status

Fresh-session entry point. This file is overwritten as state changes and should
describe only the current active release surface.

## Current Release

| Item | Value |
|---|---|
| Model | `yafitzdev/pyrrho-v2-nano-g1` |
| Dataset | `yafitzdev/fitz-gov-v2` |
| Runtime consumer | `fitz-sage` |
| Active data snapshot | `fitz_gov_v2_41358_20260703` |
| Clean training rows | 41,358 |
| Base model | `answerdotai/ModernBERT-base` |
| License | CC BY-NC 4.0 |

Pyrrho v2 is a dual-pass planning and evidence-governance encoder:

- `[PYRRHO_PRE]` reads the user query and emits pre-retrieval
  `retrieval_intents` plus `evidence_kinds`.
- `[PYRRHO_POST]` reads `Question + Sources` and emits all native v2 heads.

| Head | Labels |
|---|---|
| `evidence_verdict` | `SUFFICIENT`, `DISPUTED`, `INSUFFICIENT` |
| `failure_mode` | `none`, `unresolved_conflict`, `missing_or_incomplete_evidence`, `wrong_scope_or_version`, `ambiguous_request` |
| `retrieval_intents` | `needs_lookup`, `needs_temporal_resolution`, `needs_comparison_or_set`, `needs_broad_coverage` |
| `evidence_kinds` | `needs_text`, `needs_table_or_record`, `needs_code_or_symbol`, `needs_config_or_setting`, `needs_log_or_run_result`, `needs_document_layout` |

`fitz-sage` uses the PRE heads for retrieval planning and the POST heads for
governance metadata. Its public runtime `AnswerMode` still serializes the
existing runtime values, but the native model verdict is `evidence_verdict`.

## Release Metrics

Held-out post-retrieval eval from
`outputs/modernbert_base_v2_dual_from_g1_41358_active_20260704_seed42`:

| Metric | Value |
|---|---:|
| overall score | 0.9471 |
| verdict accuracy | 0.9703 |
| false sufficient rate | 0.0484 |
| failure accuracy | 0.9567 |
| retrieval exact match | 0.8308 |
| retrieval macro F1 | 0.9277 |
| evidence-kind exact match | 0.9809 |
| evidence-kind macro F1 | 0.9950 |

Held-out pre-retrieval query eval:

| Metric | Value |
|---|---:|
| retrieval exact match | 0.8248 |
| retrieval macro F1 | 0.9266 |
| evidence-kind exact match | 0.9637 |
| evidence-kind macro F1 | 0.9873 |

Fitz-sage release-candidate checks:

| Benchmark | Result |
|---|---:|
| balanced fixed-evidence governance sanity suite | 120/120 |
| live fitz-sage benchmark | 97/120 |
| core | 19/20 |
| holdout | 43/50 |
| holdout2 | 35/50 |

## Live Links

- Model card: https://huggingface.co/yafitzdev/pyrrho-v2-nano-g1
- Dataset card: https://huggingface.co/datasets/yafitzdev/fitz-gov-v2
- fitz-sage docs: https://github.com/yafitzdev/fitz-sage

## Current Tooling

| Script | Purpose |
|---|---|
| `scripts/prepare_v2_data.py` | Convert accepted fitz-gov-v2 rows into train/eval datasets. |
| `scripts/train_v2_encoder.py` | Train the 18-logit v2 multi-head encoder. |
| `scripts/export_onnx.py` | Export Transformers checkpoint plus FP32/INT8 ONNX artifacts. |
| `scripts/build_model_card.py` | Build the current v2 HF model card. |
| `scripts/push_to_hub.py` | Upload a release directory to Hugging Face. |

Legacy v1 scripts and docs are retained only to reproduce earlier experiments.
Do not use them for current v2 release work unless explicitly doing historical
comparison.

## Immediate Next Actions

1. Commit and push the current v2 release/docs changes in `pyrrho`.
2. Commit and push the matching v2 integration/docs changes in `fitz-sage`.
3. Keep future data generation and training work pointed at `fitz-gov-v2` and
   the four native v2 heads above.

## Historical Context

Use `docs/LOG.md`, `docs/PROJECT.md`, and `docs/ROADMAP.md` for history. Those
files contain v1 terminology and old plans; they are not the current release
contract.
