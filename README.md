# pyrrho

> Fine-tuned classification models for RAG governance. Decide whether your retrieved sources support a confident answer, contradict each other, or simply don't contain it — without an LLM call.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Named for [Pyrrho of Elis](https://en.wikipedia.org/wiki/Pyrrho), the Greek philosopher whose school practiced suspension of judgment when evidence was insufficient — the same principle this model family encodes.

---

## What it does

Given a `query + retrieved contexts`, returns one of:

| Verdict | Meaning |
|---|---|
| `ABSTAIN` | The sources do not contain enough information to answer. |
| `DISPUTED` | The sources contradict each other on the answer. |
| `TRUSTWORTHY` | The sources consistently and sufficiently support an answer. |

Drop-in replacement for the constraint+sklearn governance pipeline in [fitz-sage](https://github.com/yafitzdev/fitz-sage). Single forward pass, ~30 ms on CPU, no LLM calls at inference.

---

## Headline results (release v1)

`pyrrho-modernbert-base-v1` vs the published [fitz-sage v0.11](https://github.com/yafitzdev/fitz-sage) sklearn baseline, 3-seed mean ± std on the [fitz-gov](https://github.com/yafitzdev/fitz-gov) v5 eval hold-out (584 cases):

| Metric | pyrrho v1 | sklearn baseline | Δ |
|---|---|---|---|
| Overall accuracy | **86.13 ± 0.86%** | 78.7% | **+7.43** |
| False-trustworthy (safety) | **5.27 ± 0.21%** | 5.7% | **-0.43** (safer) |
| Trustworthy recall | **79.38 ± 1.64%** | 70.0% | **+9.38** |
| Disputed recall | **94.81 ± 1.28%** | 86.1% | **+8.71** |
| Abstain recall | **92.94 ± 1.11%** | 86.5% | **+6.44** |
| CPU inference (est.) | ~30 ms | ~500–2000 ms (5 LLM calls) | **~50× faster** |
| External dependencies | none | requires LLM | self-contained |

Every improvement margin is multiple standard deviations larger than seed noise. Not a lucky-run artifact.

Known limitation: model over-abstains on short, clean factual contexts (e.g. one-sentence answers). Production RAG chunks are typically 200–500 chars and look like the training distribution; v2 will add short-context training data.

---

## Family roadmap

**Track A — production encoders (CPU-only):**

| Model | Params | Status |
|---|---|---|
| `pyrrho-modernbert-base-v1` | 149M | ✅ trained + 3-seed validated |
| `pyrrho-modernbert-base-v2-long` | 149M | planned (long-context augmentation) |
| `pyrrho-deberta-v3-large-v1` | 435M | planned (accuracy mode) |

**Track B — portfolio generative SLMs (all CPU-runnable):**

| Model | Params | Status |
|---|---|---|
| `pyrrho-qwen3.5-0.8b-v1` | 0.8B | planned |
| `pyrrho-qwen3.5-2b-v1` | 2B | planned |
| `pyrrho-lfm2.5-1.2b-v1` | 1.2B (Liquid hybrid) | planned |
| `pyrrho-gemma-4-E2B-v1` | 2.3B | planned |
| `pyrrho-qwen3.5-4b-v1` | 4B | planned |
| `pyrrho-gemma-4-E4B-v1` | 4.5B | planned |
| `pyrrho-phi-4-mini-v1` | 3.8B | planned |
| `pyrrho-lfm2-8b-a1b-v1` | 8B / 1B-active MoE | planned |

**Sidecar:** `pyrrho-grounding-modernbert-base-v1` — answer-level hallucination detection.

Full roadmap in [docs/PROJECT.md §10](docs/PROJECT.md).

---

## Repository structure

```
pyrrho/
├── README.md           ← you are here
├── LICENSE             ← Apache 2.0
├── pyproject.toml      ← Python deps (encoder / slm / dev extras)
├── docs/               ← all project docs (start with INDEX.md)
├── src/pyrrho/         ← Python package (data, metrics, training, manifest)
├── scripts/            ← all CLI scripts (train, eval, sweep, compare, …)
├── configs/            ← training configs (encoder/, slm/, sweep_grids/)
├── tests/              ← pytest suites
├── data/               ← (gitignored) processed splits from prepare_data.py
└── outputs/            ← (gitignored) training runs, checkpoints, eval reports
```

---

## Documentation

All docs live under [`docs/`](docs/). Start here:

| Document | Purpose |
|---|---|
| [docs/INDEX.md](docs/INDEX.md) | **Fresh session entry point.** Read this first. |
| [docs/STATE.md](docs/STATE.md) | Current status, what was done, what's next |
| [docs/PROJECT.md](docs/PROJECT.md) | Full plan: vision, model picks, training recipes, roadmap |
| [docs/METHODOLOGY.md](docs/METHODOLOGY.md) | End-to-end model-development pipeline, release gates, W&B conventions |
| [docs/SETUP.md](docs/SETUP.md) | RTX 5090 / Blackwell / Windows environment specifics |

---

## Quickstart

```bash
# 1. Install
git clone https://github.com/yafitzdev/pyrrho.git
cd pyrrho
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\Activate.ps1 on Windows
pip install torch --index-url https://download.pytorch.org/whl/cu128   # for RTX 50-series
pip install -e ".[encoder,dev]"

# 2. Prepare data (requires a local clone of yafitzdev/fitz-gov)
python scripts/prepare_data.py --fitz-gov ../fitz-gov/data --output data/processed

# 3. Verify the environment
python scripts/verify_env.py

# 4. Train release #1
python scripts/train_encoder.py --config configs/encoder/modernbert_base.yaml --no-wandb

# 5. Multi-seed validation
python scripts/run_seeds.py --seeds 42 1337 7

# 6. Full per-breakdown evaluation report
python scripts/eval_report.py --checkpoint outputs/modernbert_base_v1/checkpoint-XXX

# 7. Compare to the sklearn baseline
python scripts/compare_runs.py baseline outputs/multi_seed/summary.json

# 8. Regression smoke test
pytest tests/test_smoke.py -v
```

Full setup details (Blackwell-specific install quirks, WSL2 fallback) in [docs/SETUP.md](docs/SETUP.md).

---

## Related projects

- [**fitz-sage**](https://github.com/yafitzdev/fitz-sage) — production RAG library that uses pyrrho models for governance.
- [**fitz-gov**](https://github.com/yafitzdev/fitz-gov) — 2,980-case benchmark for RAG epistemic honesty; the dataset pyrrho is trained and evaluated against.

The three projects form a triangle: fitz-gov defines the eval contract, pyrrho produces the models, fitz-sage consumes them in production.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
