# OLMoE Training Path - 2026-05-31

Status: current working decision for `pyrrho-MoE-g4-real` after the stock runtime gate and first donor initialization passed.

## What Is Proven

The current carrier shape is `configs/moe/pyrrho_moe_g4_real_olmoe_stock_runtime.yaml`:

| Field | Value |
|---|---:|
| Runtime carrier | `OlmoeForCausalLM` |
| Layers | 24 |
| Hidden size | 1024 |
| Attention heads / KV heads | 16 / 16 |
| Experts/layer | 19 |
| Experts used/token | 1 |
| FFN dim | 2688 |
| Vocab | 50,304 |
| Total params | 3.969692721B |
| Active inclusive params | 0.402437169B |

Full random-weight structural proof passes:

- Clean upstream llama.cpp conversion and load/generate.
- LM Studio CLI load after hard-link import.
- Report: `outputs/moe/g4_real_stock_runtime_carrier/olmoe_full_stock_runtime_probe_report.json`.

## Mechanical SFT Smoke

Added `scripts/train_moe_olmoe_sft.py`, a wrapper around the existing generative SFT harness with OLMoE-safe defaults.

Full-shape smoke:

```powershell
python scripts\train_moe_olmoe_sft.py `
  --output-dir outputs\moe\olmoe_g4_real_sft_full_smoke_fp32loss `
  --max-steps 1 `
  --max-train-samples 2 `
  --max-eval-samples 2 `
  --sample-mode prefix `
  --max-length 128 `
  --batch-size 1 `
  --eval-batch-size 1 `
  --lora-r 4 `
  --lora-alpha 8
```

Result:

- Base params loaded: **3,970,475,008**.
- LoRA trainable params: **786,432**.
- One training step completed with finite loss: **11.070787**.
- Adapter saved at `outputs/moe/olmoe_g4_real_sft_full_smoke_fp32loss/final_adapter/`.
- Reload-only smoke passed at `outputs/moe/olmoe_g4_real_sft_full_smoke_fp32loss_reload/`.

Quality is meaningless here. The base weights are random and the smoke used two rows. This only proves the training path works mechanically.

## Donor Audit

Current source checks used Hugging Face model metadata and local downloaded configs.

| Candidate | Fit | Decision |
|---|---|---|
| `allenai/OLMoE-1B-7B-0924` | Best family/tokenizer match. Apache-2.0, `model_type: olmoe`, vocab 50,304. Shape is **2048 hidden / 16 layers / 64 experts / top-8 / ffn 1024**, while target is **1024 hidden / 24 layers / 19 experts / top-1 / ffn 2688**. | Good donor/teacher, not a direct load. Use only with explicit compression/slicing/distillation. |
| `allenai/OLMo-2-0425-1B` | Apache-2.0 dense teacher candidate. Shape is **2048 hidden / 16 layers / ffn 8192**, but tokenizer is 100,352 vocab and model type is `olmo2`. | Teacher only, not direct seed. |
| `allenai/Olmo-Hybrid-7B` | Apache-2.0 and current, but 7B hybrid architecture. | Teacher/reference only, not a seed for this OLMoE carrier. |

## First Donor Initialization Attempt

The first controlled transplant now exists. It preserves the exact target carrier shape and fills it from `allenai/OLMoE-1B-7B-0924` with explicit resizing instead of pretending the donor and target shapes match.

- Script: `scripts/init_olmoe_from_donor.py`.
- Donor mirror: `outputs/moe/g4_real_olmoe_training_path/donors/OLMoE-1B-7B-0924/` (**26** files / **13,841,165,654** bytes).
- HF output: `outputs/moe/g4_real_olmoe_donor_init/olmoe_1b7b_block_resize_hf/` (**15** files / **7,943,366,778** bytes).
- Transplant report: `outputs/moe/g4_real_olmoe_donor_init/olmoe_1b7b_block_resize_hf/donor_init_report.json`.
- Result: **3,969,688,576 / 3,969,688,576** target CausalLM params filled, **0** missing target parameters, **267** tensors resized.
- GGUF: `outputs/moe/g4_real_olmoe_donor_init/pyrrho_g4_real_olmoe_1b7b_block_resize_f16.gguf` (**7,942,296,800** bytes, SHA256 `e646c254f4bf7f11605ef6b9dc463994245ff104b6d4e711f635b05baacefaf4`).
- Clean upstream `llama-cli` load/generate passed, with **7,623 MiB** host memory reported.
- LM Studio CLI load passed as `pyrrho-moe-g4-real-olmoe-donor-init`, CPU/local, **5.69s** reported load time, **7.40 GiB**.
- Runtime report: `outputs/moe/g4_real_olmoe_donor_init/donor_init_runtime_probe_report.json`.

`scripts/init_olmoe_from_donor.py` now also supports `--resize-strategy head-topnorm-slice` for a controlled alternative seed. It keeps the strongest donor hidden channels per attention head instead of block-averaging every 2048-to-1024 hidden axis.

Quality is not proven. The 12-row label-score baseline smoke only scored **1/12 accuracy** with **0/12 false-TRUSTWORTHY** at tau **0.50**. That is a mechanical eval proof, not a capability result.

## Bounded SFT Probes

Bounded SFT from the donor-initialized seed is now tested and should be treated as a negative branch, not an obvious scale-up path.

Summary artifact: `outputs/moe/g4_real_olmoe_donor_init/sft_probe_summary_2026-05-31.json`.

| Probe | Evidence | Result |
|---|---|---|
| Donor baseline | 12-row label-score smoke | **8.33%** accuracy / **0.00%** FT. Mechanical eval proof only. |
| Label-JSON attention LoRA, 256 train / 64 eval | `outputs/moe/g4_real_olmoe_donor_init/sft_label_json_256x64_tau050/` | Tau **0.50** label-score: **37.50%** / **2.33%** FT. Best gated tau **0.55**: **39.06%** / **0.00%** FT. |
| Label-JSON attention LoRA generation smoke | `outputs/moe/g4_real_olmoe_donor_init/sft_label_json_256x64_generation_smoke/` | **0.00%** JSON parse, **43.75%** label parse, **25.00%** accuracy, **27.27%** FT. Free generation unsafe. |
| Label-only attention LoRA, 256 train / 64 eval | `outputs/moe/g4_real_olmoe_donor_init/sft_label_only_256x64_tau050/` | **32.81%** / **0.00%** FT; collapsed to DISPUTED. |
| Label-only attention LoRA generation smoke | `outputs/moe/g4_real_olmoe_donor_init/sft_label_only_256x64_generation_smoke/` | **100.00%** label parse but only **31.25%** accuracy; collapsed to DISPUTED. |
| Label-only attention LoRA + lm_head | `outputs/moe/g4_real_olmoe_donor_init/sft_label_only_lmhead_256x64_rawtau/` | Raw tau **0.00**: **39.06%** / **76.74%** FT. Tau **0.60** removes FT by collapsing to DISPUTED at **32.81%** accuracy. |
| Label-JSON attention LoRA, 1,024 train / 256 eval | `outputs/moe/g4_real_olmoe_donor_init/sft_label_json_1024x256_tau050/` | **33.59%** / **0.58%** FT at tau **0.50**; scaling the best-shaped recipe worsened selected-label quality and mostly collapsed to ABSTAIN. |
| Label-JSON 1,024-row generation smoke | `outputs/moe/g4_real_olmoe_donor_init/sft_label_json_1024x256_generation_smoke/` | **6.25%** JSON parse, **100.00%** label parse, **37.50%** accuracy, **0.00%** FT; all selected classifications were ABSTAIN. |
| Router/lm_head raw-parameter unfreeze | `outputs/moe/g4_real_olmoe_donor_init/sft_label_only_router_lmhead_fullparam_256x64_rawtau/` | **51.98M** trainable params; float16 training went **NaN** from step 2 onward. |

Implementation note: OLMoE experts and routers use raw `nn.Parameter` tensors (`OlmoeExperts.gate_up_proj`, `OlmoeExperts.down_proj`, `OlmoeTopKRouter.weight`), not ordinary `nn.Linear` modules. Standard PEFT LoRA hits attention and `lm_head`, but not expert/router tensors. `scripts/train_moe_qwen_sft.py` now has `--unfreeze-parameter-patterns` for explicit raw-parameter diagnostics; the first naive router/lm_head attempt was unstable, so this is a diagnostic hook, not a release recipe.

## Stable Expert/Router Adaptation Probes

The trainer now has OLMoE-specific adapter hooks for raw expert/router tensors:

- `--olmoe-expert-down-lora-r`
- `--olmoe-expert-down-lora-alpha`
- `--olmoe-expert-gate-up-lora-r`
- `--olmoe-expert-gate-up-lora-alpha`
- `--olmoe-router-lora-r`
- `--olmoe-router-lora-alpha`

These hooks attach float32 LoRA diagnostics to OLMoE expert `gate_up_proj` / `down_proj` tensors and router logits. They are useful because PEFT cannot target OLMoE's raw 3D expert parameters.

Summary artifact: `outputs/moe/g4_real_olmoe_donor_init/stable_adaptation_probe_summary_2026-05-31.json`.

| Probe | Evidence | Result |
|---|---|---|
| BF16 low-LR raw router/lm_head unfreeze | `outputs/moe/g4_real_olmoe_donor_init/sft_label_only_router_lmhead_fullparam_bf16_lr1e5_128x64_rawtau/` | Avoided the earlier float16 NaN, but raw label-score was unsafe: **34.38%** accuracy / **62.79%** FT. Safe calibration collapses to **32.81%** accuracy. |
| Expert-down LoRA rank 4 | `outputs/moe/g4_real_olmoe_donor_init/sft_label_json_expertdown_lora_r4_256x64_tau050/` | Stable, **6.77M** trainable params, last loss **1.9716**, label-score **37.50%** / **2.33%** FT. This ties but does not beat the earlier attention-LoRA small probe. |
| Attention LoRA + expert-down LoRA rank 4 | `outputs/moe/g4_real_olmoe_donor_init/sft_label_json_attention_expertdown_lora_r4_256x64_tau050/` | Stable, **8.34M** trainable params, last loss **1.1349**, but collapsed mostly to DISPUTED: **31.25%** / **0.00%** FT. |

Conclusion: the stable expert-adapter surface is mechanically useful, but supervised-only SFT is still not the quality path.

## Teacher Distillation Probes

Teacher-logit distillation is now wired into `scripts/train_moe_qwen_sft.py`.

New arguments:

- `--teacher-logits-dir`
- `--label-distillation-weight`
- `--distillation-temperature`
- `--label-distillation-length-normalization`

The teacher sidecar used here is `outputs/moe/teacher_logits/pyrrho_nano_g3_full_v8/`. On the exact 64-row eval sample used by these probes, the teacher is strong: **96.88%** accuracy / **2.33%** FT with balanced predictions (**22** ABSTAIN / **21** DISPUTED / **21** TRUSTWORTHY). The student failure is not because the attached teacher labels are bad.

Summary artifact: `outputs/moe/g4_real_olmoe_donor_init/teacher_distillation_probe_summary_2026-05-31.json`.

| Probe | Evidence | Result |
|---|---|---|
| Expert-down LoRA rank 4, teacher weight 1, label-JSON | `outputs/moe/g4_real_olmoe_donor_init/distill_label_json_expertdown_lora_r4_128x64_tau050/` | Best bounded distillation result, but only **39.06%** / **2.33%** FT at tau **0.50**. This ties the earlier small ceiling and does not justify scale-up. |
| Expert-down LoRA rank 4, teacher weight 4, label-JSON | `outputs/moe/g4_real_olmoe_donor_init/distill_w4_label_json_expertdown_lora_r4_128x64_tau050/` | **37.50%** / **0.00%** FT, but no TRUSTWORTHY recall under the safe gate. |
| Expert-down LoRA rank 4, teacher weight 1, label-only | `outputs/moe/g4_real_olmoe_donor_init/distill_w1_label_only_expertdown_lora_r4_128x64_tau050/` | **34.38%** / **0.00%** FT; collapsed to all ABSTAIN at the safe gate. |
| Expert-down + router LoRA rank 4, teacher weight 1, label-JSON | `outputs/moe/g4_real_olmoe_donor_init/distill_w1_label_json_expertdown_router_lora_r4_128x64_tau050/` | Tau **0.50** was unsafe (**32.81%** / **6.98%** FT); safe tau collapsed to **32.81%** / **0.00%** FT. |
| BF16 low-LR raw router/lm_head fullparam, teacher weight 4, label-only | `outputs/moe/g4_real_olmoe_donor_init/distill_w4_label_only_router_lmhead_fullparam_bf16_lr1e5_128x64_tau050/` | Stayed finite with **51.98M** trainable params, but safe calibration collapsed mostly to DISPUTED at **32.81%** / **0.00%** FT. |

Conclusion: bounded teacher distillation is mechanically implemented, but the tested student surfaces do not absorb the strong teacher signal. Do not scale these exact tiny adapter recipes.

## Gate/Up Expert Surface Probe

The trainer now reaches the remaining raw expert tensor surface: `OlmoeExperts.gate_up_proj`, alongside the earlier `down_proj` hook. This is still a diagnostic adapter surface, not a release recipe.

Summary artifact: `outputs/moe/g4_real_olmoe_donor_init/gate_up_expert_surface_probe_summary_2026-05-31.json`.

| Probe | Evidence | Result |
|---|---|---|
| Gate/up + down LoRA smoke, rank 2 | `outputs/moe/g4_real_olmoe_donor_init/gateup_down_distill_smoke_r2_4x4/` | Mechanically positive: hook attaches to the donor checkpoint, trains/evals for 2 steps, and reports **9.22M** trainable params. Eval quality is meaningless at 4 rows. |
| Gate/up + down LoRA rank 4, LR 1e-4, teacher weight 1 | `outputs/moe/g4_real_olmoe_donor_init/distill_w1_label_json_gateup_down_lora_r4_128x64_tau050/` | Raw tau **0.50** moved TRUSTWORTHY behavior but was unsafe: **35.94%** / **48.84%** FT with 32/64 TRUSTWORTHY predictions. Best safe gate tau **0.65** was only **34.38%** / **2.33%** FT and mostly ABSTAIN. |
| Gate/up + down LoRA rank 4, LR 3e-5, teacher weight 1 | `outputs/moe/g4_real_olmoe_donor_init/distill_w1_label_json_gateup_down_lora_r4_lr3e5_128x64_tau050/` | Lower LR avoided the unsafe TRUSTWORTHY surge but collapsed useful recall: tau **0.50** was **31.25%** / **2.33%** FT; best safe gate tau **0.55** was **32.81%** / **0.00%** FT with 63/64 ABSTAIN and 0 TRUSTWORTHY predictions. |

Conclusion: the gate/up expert surface is real and can move class behavior, but this adapter recipe is still quality-negative. Do not scale gate/up+down LoRA as-is.

## Head-Topnorm Donor Resize Probe

The second donor seed changes only the hidden-axis shrink rule. It preserves the same OLMoE carrier shape and fills all target parameters, but uses a head-wise top-norm hidden slice for 2048-to-1024 donor hidden axes.

Summary artifact: `outputs/moe/g4_real_olmoe_donor_init/head_topnorm_slice_init_probe_summary_2026-05-31.json`.

| Probe | Evidence | Result |
|---|---|---|
| Head-topnorm donor seed | `outputs/moe/g4_real_olmoe_donor_init/olmoe_1b7b_head_topnorm_slice_hf/donor_init_report.json` | Mechanically positive: **3,969,688,576 / 3,969,688,576** params copied, **0** missing, **1,024** hidden channels selected, **267** resized tensors. |
| No-training label-score baseline | `outputs/moe/g4_real_olmoe_donor_init/head_topnorm_baseline_labelscore_64_tau050/` | Tau **0.50** was unsafe: **35.94%** / **46.51%** FT. Best safe gate tau **0.65** was **37.50%** / **0.00%** FT with 62/64 ABSTAIN and only 1 TRUSTWORTHY prediction. |
| Expert-down LoRA rank 4, teacher weight 1, label-JSON | `outputs/moe/g4_real_olmoe_donor_init/headtopnorm_distill_label_json_expertdown_lora_r4_128x64_tau050/` | Raw tau **0.50** was unsafe: **31.25%** / **32.56%** FT. Best safe gate tau **0.70** tied the old block-resize accuracy ceiling at **39.06%**, but with worse FT (**4.65%** vs old **2.33%**) and the same TRUSTWORTHY recall. |

Conclusion: head-topnorm slicing is a valid seed-building option, but not a quality improvement. Do not scale this initialization with the same tiny adapter recipe.

## Decision

Do not scale random-only SFT. It can learn the mechanics but not useful language competence.

Do not keep scaling donor-seed attention-LoRA SFT. It can learn label tokens, but current probes collapse to ABSTAIN/DISPUTED or become unsafe on TRUSTWORTHY, and generation still is not reliable pyrrho JSON.

1. Keep the exact OLMoE carrier shape unchanged.
2. Do not keep scaling the tested tiny adapter SFT/distillation recipes, including gate/up+down expert LoRA and the head-topnorm donor-resize seed.
3. Move next to stronger initialization/upcycling or a materially broader training surface so the teacher signal can actually move class behavior.
4. Re-convert trained weights and rerun both clean llama.cpp and LM Studio load gates only after bounded held-out quality improves beyond the current ~39% ceiling.

Any trained-weight artifact must rerun the same full GGUF + LM Studio load gate before release.
