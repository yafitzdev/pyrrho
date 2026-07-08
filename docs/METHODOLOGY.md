# pyrrho v2 methodology

This page describes the current v2 release workflow. Older v1 methodology and
SLM planning lives in `docs/PROJECT.md`, `docs/ROADMAP.md`, and `docs/LOG.md`
as historical context.

## Pipeline

```text
accepted fitz-gov-v2 rows
  -> scripts/prepare_v2_data.py
  -> scripts/train_v2_encoder.py
  -> scripts/export_onnx.py
  -> scripts/build_model_card.py
  -> scripts/push_to_hub.py
```

## Data Prep

Input is the accepted v2 training vault:

```bash
python scripts/prepare_v2_data.py ^
  --accepted-rows C:/path/to/accepted.rows.jsonl ^
  --output data/v2_alpha
```

The prepared dataset stores:

- `text`: `[PYRRHO_POST]` plus `Question` and `Sources`
- `query_only_text`: `[PYRRHO_PRE]` plus the user query
- `labels`: the 18-dimensional v2 label vector
- `label_mask`: which heads are trained for each input mode
- label metadata for distribution and audit checks

## Training

```bash
python scripts/train_v2_encoder.py ^
  --config configs/encoder/modernbert_base_v2_alpha.yaml ^
  --no-wandb
```

The active v2 model uses one 18-logit vector:

| Slice | Head | Loss |
|---|---|---|
| `0:3` | `evidence_verdict` | categorical cross entropy |
| `3:8` | `failure_mode` | categorical cross entropy |
| `8:12` | `retrieval_intents` | binary cross entropy |
| `12:18` | `evidence_kinds` | binary cross entropy |

Default training is dual-mode. POST rows train all four heads. PRE rows train
only `retrieval_intents` and `evidence_kinds`; verdict and failure labels are
masked out for query-only inputs.

## Release Checks

Before publishing a v2 model:

1. Confirm held-out training eval reports verdict, failure, retrieval-intent,
   and evidence-kind metrics.
2. Run the fixed-evidence balanced governance sanity suite in `fitz-sage`.
3. Run the live fitz-sage benchmark.
4. Confirm the model card and dataset card use the native v2 heads:
   `SUFFICIENT`, `DISPUTED`, `INSUFFICIENT`.
5. Confirm no discarded, unapproved, or throwaway generation data is referenced
   by the active snapshot.

## Current Release Snapshot

| Item | Value |
|---|---|
| Model | `yafitzdev/pyrrho-v2-nano-g1` |
| Dataset | `yafitzdev/fitz-gov-v2` |
| Active rows | 41,358 |
| Training source pointer | `fitz_gov_v2_41358_20260703` |
| Balanced governance sanity suite | 120/120 |
| Live fitz-sage benchmark | 97/120 |
| Core | 19/20 |
| Holdout | 43/50 |
| Holdout2 | 35/50 |

## Notes

The old v1 class metrics are historical. The v2 safety metric is expressed as
false sufficient rate, and the native verdict labels are `SUFFICIENT`,
`DISPUTED`, and `INSUFFICIENT`.
