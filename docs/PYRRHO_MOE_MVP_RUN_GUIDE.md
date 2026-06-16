# pyrrho-MoE-g3-mvp Run Guide

This is the shortest supported path for running the published `pyrrho-MoE-g3-mvp` MVP locally.

The important rule: use full-sequence label scoring as the decision source. Do not use raw generation as the governance verdict.

## What You Are Running

| Field | Value |
|---|---|
| HF model | `yafitzdev/pyrrho-MoE-g3-mvp` |
| Project architecture | Pyrrho MoE MVP |
| Runtime / loader architecture | Qwen3MoE-compatible sparse MoE |
| Recommended local artifact | `gguf/pyrrho-MoE-g3-mvp-merged-Q4_K_M.gguf` |
| Decision mode | `sequence-label-score` |
| TRUSTWORTHY threshold | `0.50` |
| Expected peak RSS | about 4.2 GiB on the validated full-test smoke |

Hugging Face or GGUF tooling may display `qwen3moe`. That is the loader compatibility family, not the project architecture name.

## 1. Clone pyrrho and Install the Python Helpers

```powershell
git clone https://github.com/yafitzdev/pyrrho
Set-Location pyrrho

# Install PyTorch first for your machine, then:
python -m pip install -e ".[slm,hub]"
```

The GGUF runner uses `scripts/smoke_moe_gguf_server.py` from this repo. The model files alone are not a standalone Python package.

## 2. Download the Model Package

```powershell
hf download yafitzdev/pyrrho-MoE-g3-mvp `
  --repo-type model `
  --local-dir models\pyrrho-MoE-g3-mvp
```

The low-memory model should be here afterward:

```text
models/pyrrho-MoE-g3-mvp/gguf/pyrrho-MoE-g3-mvp-merged-Q4_K_M.gguf
```

## 3. Build the Patched llama.cpp Server

The current GGUF needs the bundled Qwen3MoE loader patch:

```text
models/pyrrho-MoE-g3-mvp/patches/llama_cpp_qwen3moe_mlp_only_layers.patch
```

Example build:

```powershell
git clone https://github.com/ggml-org/llama.cpp C:\work\llama.cpp
Set-Location C:\work\llama.cpp

git apply C:\path\to\pyrrho\models\pyrrho-MoE-g3-mvp\patches\llama_cpp_qwen3moe_mlp_only_layers.patch

cmake -B build
cmake --build build --config Release -j
```

The validated local checkout was `C:/Users/yanfi/.unsloth/llama.cpp` at commit `568aec82d2fc48341c54cae565768ac75072a31d`, with the patch applied. If the patch does not apply cleanly to a newer llama.cpp checkout, use a checkout close to that commit.

## LM Studio Status

LM Studio is not a supported runtime for this MVP GGUF right now. The app uses its own bundled llama.cpp build, and that build does not include the pyrrho patch above. The expected symptom is a generic load failure such as:

```text
Failed to load the model
```

This does not mean the GGUF file is corrupt. Use the patched `llama-server` path in this guide. LM Studio can become viable later only if its bundled runtime includes equivalent support for the Qwen3MoE dense `mlp_only_layers` tensors and `norm_topk_prob=false` routing behavior.

## 4. Prepare Input JSONL

Minimal input rows need `query` and `contexts`.

```json
{"id":"demo-001","query":"Has the company achieved profitability?","contexts":["The company posted its first profitable quarter, net income $4M.","The same report lists a quarterly loss of $12M."]}
```

Gold fields such as `label`, `route`, and `taxonomy_pattern` are optional. If `label` is present, the runner writes metrics to `report.json`; otherwise it just writes predictions.

## 5. Run the Recommended GGUF Decision Path

```powershell
python scripts\smoke_moe_gguf_server.py `
  --model models\pyrrho-MoE-g3-mvp\gguf\pyrrho-MoE-g3-mvp-merged-Q4_K_M.gguf `
  --llama-server C:\work\llama.cpp\build\bin\Release\llama-server.exe `
  --input data\moe_v8\test.jsonl `
  --output-dir outputs\moe\gguf\pyrrho_moe_g3_mvp_quick_smoke `
  --max-samples 8 `
  --decision-mode sequence-label-score `
  --label-threshold 0.50 `
  --n-probs 5000
```

For your own file, replace `--input data\moe_v8\test.jsonl` with your JSONL path.

## 6. Read the Output

The runner writes:

```text
outputs/moe/gguf/pyrrho_moe_g3_mvp_quick_smoke/predictions.jsonl
outputs/moe/gguf/pyrrho_moe_g3_mvp_quick_smoke/report.json
```

For GGUF `sequence-label-score`, trust:

```json
{
  "classification": "ABSTAIN | DISPUTED | TRUSTWORTHY",
  "label_score": {
    "mode": "sequence-label-score",
    "trustworthy_threshold": 0.5,
    "probabilities": {
      "ABSTAIN": 0.0,
      "DISPUTED": 0.0,
      "TRUSTWORTHY": 0.0
    }
  }
}
```

Do not consume `raw_generation` as the decision. In sequence-label mode it is intentionally empty. In raw-generation mode it is parseable but unsafe.

## Validated Evidence

Full held-out Q4_K_M GGUF sequence-label scoring:

| Metric | Value |
|---|---:|
| Rows | 2,459 |
| Accuracy | 82.15% |
| False-TRUSTWORTHY | 5.27% |
| TRUSTWORTHY recall | 72.63% |
| Peak RSS | 4.224 GiB |
| Agreement with HF selected-output path | 96.38% |

Downloaded-snapshot random-32 smoke:

| Metric | Value |
|---|---:|
| Accuracy | 93.75% |
| False-TRUSTWORTHY | 0.00% |
| Label parse | 100% |
| Peak RSS | 4.220 GiB |

## What Not To Do

- Do not use raw generated `classification` as the governance decision.
- Do not use first-token label scoring; the labels are multi-token and the method failed at scale.
- Do not load the current GGUF in LM Studio; use patched llama.cpp directly.
- Do not use local Transformers + bitsandbytes 4-bit as the release runtime; packaged-adapter inference timed out after 900 seconds.
- Do not call this a from-scratch Pyrrho pretrain. The honest public name is Pyrrho MoE MVP, Qwen3MoE-compatible and Qwen-seeded.

## Slow Internal Adapter Path

The package also includes the PEFT adapter path used for local validation:

```powershell
python scripts\infer_moe_qwen_sft.py `
  --adapter-path models\pyrrho-MoE-g3-mvp\adapter `
  --input data\moe_v8\test.jsonl `
  --max-samples 8 `
  --threshold 0.50 `
  --skip-generation `
  --output outputs\moe\pyrrho_moe_g3_mvp_skipgen_smoke.jsonl
```

This path is useful for debugging against the original selected-output reports. It is not the recommended low-memory CPU path; BF16 CPU smokes peaked around 9.8-10.2 GiB.
