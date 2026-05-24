# pyrrho — Model Development Methodology

How to take any base model (encoder or SLM) from "I'd like to train this" to
"pushed to HuggingFace with a model card" reproducibly. Every release in the
pyrrho family follows the same pipeline. If a step is skipped, the model card
must say so explicitly.

---

## The pipeline

```
  configs/<family>/<model>.yaml
            │
            ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  1. prepare_data.py    (one-time per fitz-gov version)        │
  └──────────────────────────────────────────────────────────────┘
            │
            ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  2. sweep.py           coordinate-descent or grid sweep        │
  │                        finds a good hyperparameter point        │
  └──────────────────────────────────────────────────────────────┘
            │  (best cell config)
            ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  3. run_seeds.py       3-seed validation                        │
  │                        reports mean +/- std on the headline    │
  │                        metrics; catches lucky-seed artifacts    │
  └──────────────────────────────────────────────────────────────┘
            │  (best per-seed checkpoint)
            ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  4. eval_report.py     full per-breakdown report                │
  │                        difficulty / expert / taxonomy pattern / │
  │                        taxonomy cell                            │
  └──────────────────────────────────────────────────────────────┘
            │
            ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  5. compare_runs.py    diff against sklearn baseline AND       │
  │                        the previous pyrrho release              │
  └──────────────────────────────────────────────────────────────┘
            │
            ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  6. pytest tests/      smoke test gate                          │
  │                        regression detection vs known-good      │
  └──────────────────────────────────────────────────────────────┘
            │
            ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  7. export_onnx.py     (encoder track only)                     │
  │     export_gguf.py     (SLM track only)                         │
  └──────────────────────────────────────────────────────────────┘
            │
            ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  8. write model card   from final_metrics.json + eval_report   │
  │     push_to_hub.py     yafitzdev/pyrrho-<base>-<size>-v<n>     │
  └──────────────────────────────────────────────────────────────┘
```

Every step writes a `manifest.json` next to its output. Every script in `scripts/`
is config-driven so the same code works across the whole pyrrho family.

---

## What each step produces

| Step | Script | Outputs |
|---|---|---|
| 1. Data prep | `prepare_data.py` | `data/processed/{train,eval,test}.jsonl` + `hf_dataset/`; `tier0_sanity.jsonl` is optional/legacy |
| 2. Hyperparameter sweep | `sweep.py` | `outputs/sweeps/<name>/<cell>/{final_metrics.json,manifest.json,checkpoint-*}` + `sweep_summary.json` |
| 3. Multi-seed validation | `run_seeds.py` | `outputs/multi_seed/<run>/seed_<N>/` + `summary.json` |
| 4. Full evaluation | `eval_report.py` | `<checkpoint_parent>/eval_report.json` |
| 5. Diff | `compare_runs.py` | stdout markdown + optional JSON |
| 6. Smoke test | `pytest tests/test_smoke.py` | exit code; floor is 70% on the 10-case set |
| 7. Export | `export_onnx.py` / `export_gguf.py` | `models/<name>.{onnx,gguf}` |
| 8. Ship | `push_to_hub.py` | HF repo `yafitzdev/<name>` |

---

## Reproducibility manifest schema

Every artifact-producing script (training, sweep cells, multi-seed runs) writes
`manifest.json` via `pyrrho.manifest.write_manifest`. Schema:

```json
{
  "schema_version": 1,
  "created_at": "2026-05-14T15:00:00+00:00",
  "seed": 42,
  "config": {
    "path": "configs/encoder/modernbert_base.yaml",
    "raw":  "<full file text>",
    "parsed": { ... }
  },
  "hardware": {
    "gpu_name": "NVIDIA GeForce RTX 5090",
    "gpu_compute_capability": "12.0",
    "torch_version": "2.11.0+cu128",
    "platform": "Windows-11-..."
  },
  "pip_freeze": [ "transformers==4.46.0", ... ],
  "git": {
    "pyrrho":   { "commit": "<sha>", "branch": "main", "dirty": false },
    "fitz_gov": { "commit": "<sha>", "branch": "main", "dirty": false }
  },
  "timing": { "start_unix": 1715690400, "end_unix": 1715690484, "duration_seconds": 84.0 },
  "extra": { "base_model": "answerdotai/ModernBERT-base", "threshold_selected": 0.5, ... }
}
```

If you cannot reproduce a published number from `manifest.json` + the
corresponding fitz-gov commit, that's a bug — open an issue.

---

## Release gates (revised 2026-05-14)

A pyrrho model release MUST pass these two as a mean across 3 seeds. On V5/V6-style data this means the eval split; on V7+ data with a dedicated held-out test split, checkpoint/threshold selection happens on validation and the gates are applied to held-out test:

- **Overall accuracy ≥ 78.7%** — matches fitz-sage v0.11 sklearn baseline.
- **False-trustworthy rate ≤ 5.7%** — matches baseline; this is the safety axis.

The original tier0 95% sanity gate has been dropped (see PROJECT.md §18 item 19).
With N=60 cases the gate is unreachable purely from sample variance (±3.5 pts std)
and ~5 of the 60 cases have ambiguous gold labels. **Tier0 results are reported
in every model card as a diagnostic, not a gate.**

A release SHOULD also:
- Beat the previous pyrrho release on overall accuracy OR justify why in the model card.
- Document known failure modes from `eval_report.py` per-breakdown analysis.

---

## W&B conventions

When `--no-wandb` is not passed, training scripts log to W&B with:

- **Project**: `pyrrho`
- **Run name**: from config `training.run_name`, e.g. `pyrrho-nano-g1`
- **Tags**: `["v1", "encoder", "modernbert"]` or equivalent. Set in config.
- **Logged metrics** (every epoch + final): accuracy, macro_f1, per-class P/R/F1,
  false_trustworthy_rate, ft_penalized_accuracy.
- **Logged artifacts**: best checkpoint, final_metrics.json, manifest.json.

For multi-seed runs, append `-seed{N}` to the run name. The W&B project view will
group these visually.

---

## Picking hyperparameters for a new model

For each new base model (e.g. `Qwen3.5-0.8B`, `DeBERTa-v3-base`, `LFM2.5-1.2B`):

1. **Copy the closest existing config** as a starting point.
2. **Tune the must-set knobs** (batch size for VRAM, max_seq_length, etc.).
3. **Run a coordinate-descent sweep** around the existing v1 baseline using
   `configs/sweep_grids/encoder_v1.yaml` (or a family-specific one).
4. **Pick the winner** from `sweep_summary.json` based on eval_calibrated.accuracy
   with the FT gate respected.
5. **Validate the winner with 3 seeds** via `run_seeds.py`.
6. **If 3-seed std > ±2% accuracy**, run 2 more seeds — the result is unstable.
7. **Run `eval_report.py`** to surface where the model wins/loses by breakdown.
8. **Run `compare_runs.py` vs baseline + previous release.**
9. **Ship if gates pass; document if not.**

For SLM training (track B), replace step 1's config with `configs/slm/*.yaml`
and use `scripts/train_slm.py` (not yet written — write following `train_encoder.py`'s
structure).

---

## What to put in the model card

Auto-generated section (from `final_metrics.json` + `eval_report.json`):

- Headline table: 3-seed mean ± std for accuracy / FT / per-class recall.
- Comparison row vs `fitz-sage v0.11 sklearn baseline`.
- Comparison row vs previous pyrrho release.
- Per-breakdown table: top 5 strongest and weakest buckets by accuracy.

Hand-written section:

- 2-paragraph overview tying this release to the pyrrho roadmap.
- Specific known limitations (with example failing cases).
- Citation: dataset commit hash + model card commit hash.

The fitz-gov commit hash MUST be in the model card so anyone can re-run the
exact same evaluation.

---

## Reading order for new contributors / new sessions

1. [HANDOFF.md](HANDOFF.md) — current status, what's next.
   - [LOG.md](LOG.md) — append-only project history.
2. [PROJECT.md](PROJECT.md) §1-§5 — vision and architecture decisions.
3. This file (METHODOLOGY.md) — how to actually produce a release.
4. [SETUP.md](SETUP.md) — environment specifics (RTX 5090 / Windows / WSL2).
5. PROJECT.md §18 — full conversation history for context behind every decision.
