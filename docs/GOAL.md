# GOAL - pyrrho-MoE-g4-real

Status: reset on 2026-05-31. `pyrrho-MoE-g3-mvp` is frozen as an experimental proof-of-life artifact. The active goal is now a clean real MoE: `pyrrho-MoE-g4-real`.

This document is the project objective. If another doc frames the work as polishing `pyrrho-MoE-g3-mvp`, Qwen SFT adapters, alpha quorum packages, or patched-runtime GGUFs, this file supersedes that framing.

## Objective

Build a real Pyrrho-owned sparse Mixture-of-Experts generative model that is:

- about **4B total / 0.4B active**,
- trained for pyrrho RAG governance behavior,
- CPU-runnable after quantization,
- loadable through **stock runtime support** before release,
- honest in public naming and model-card claims.

The immediate target name is `pyrrho-MoE-g4-real`.

## Gate Zero: Runtime Portability

No model advances to serious training or release packaging unless it first passes the runtime-shape gate.

The runtime-shape gate requires:

- No patched llama.cpp.
- No LM Studio load failure caused by custom tensor layout.
- No Qwen3MoE `mlp_only_layers` dependency.
- No custom GGUF converter/runtime behavior required for normal load.
- A stock llama.cpp-compatible MoE layout selected before long training starts.
- A short local smoke proving the exported tiny/random-weight shape loads in an unpatched runtime.
- No public release whose visible architecture metadata says `qwen3moe` or `mixtral`.

If a candidate needs a runtime patch, it is a research experiment, not `g4-real`.

## What Counts

`pyrrho-MoE-g4-real` must be:

- A sparse MoE model, not a quorum wrapper around multiple checkpoints.
- Pyrrho-owned in architecture intent: semantic expert groups, supervised routing from fitz-gov, and pyrrho governance output contract.
- Implemented on a **stock-supported runtime carrier** if needed, but not an off-the-shelf MoE checkpoint renamed as Pyrrho.
- Approximately 4B total / 0.4B active under the project parameter counter.
- Able to generate compact governance output, not only classify.
- Evaluated on fitz-gov V8.0.1 held-out data or the current published successor.
- Publicly documented with exact limitations.

Using a pretrained tokenizer or dense teacher is allowed. Requiring a patched loader is not.

## What Does Not Count

These do not satisfy the active goal:

- `pyrrho-MoE-g3-mvp` as currently published. It is Qwen3MoE-compatible, Qwen-seeded, and patched-runtime experimental evidence.
- `pyrrho-MoE-g3-alpha`. It is a Stage 0.7 quorum/research package.
- More threshold work around the alpha.
- More Qwen SFT adapter packaging.
- More patched llama.cpp work as the primary path.
- A model that only works through full-sequence label scoring while normal generation remains unsafe.
- A model that cannot load in stock llama.cpp / LM Studio-class runtimes because of custom tensor layout.
- A public model whose visible runtime metadata ties `pyrrho-MoE-g4-real` to Mixtral or Qwen. Mixtral is allowed only as an internal compatibility proof.

## Release Bar

Release requires all of the following:

- Runtime-shape gate passes before training scale-up.
- Loads from a release directory with no local runtime patch.
- Quantized CPU path is documented and smoke-tested.
- Normal generation produces parseable pyrrho governance output.
- If a safer label-scoring mode is also offered, it is secondary safety tooling, not the only usable decision path.
- Reports total and active parameter counts.
- Reports held-out fitz-gov metrics.
- Documents false-TRUSTWORTHY behavior and known failure modes.

Preferred quality bar remains:

- Overall accuracy >= **78.7%**.
- False-TRUSTWORTHY <= **5.7%**.
- Route and taxonomy behavior reported.

If the runtime-shape gate fails, quality numbers do not matter.

## First Build Path

Start from runtime shape, not training.

1. Define a clean `g4-real` architecture config with no Qwen3MoE `mlp_only_layers`.
2. Count total/active parameters and keep the design inside 4B/A0.4B.
3. Pick a stock-supported GGUF carrier layout for the MoE blocks.
4. Export a tiny/random-weight structural checkpoint.
5. Confirm it loads in unpatched llama.cpp.
6. Confirm LM Studio-class loadability or identify the exact stock-runtime blocker before any long training.
7. Only then start upcycling/distillation/training.

Current first-build status:

- Steps 1-2 have passed for the first candidate.
- A tiny internal stock-carrier proof passed with `MixtralForCausalLM`: `outputs/moe/g4_real_stock_runtime_carrier/stock_runtime_carrier_probe_report.json`.
- Mixtral is now explicitly internal proof only. It is not the final public carrier unless this document is changed again.
- A non-Mixtral/non-Qwen tiny carrier proof now passes with `OlmoeForCausalLM`: `outputs/moe/g4_real_stock_runtime_carrier/olmoe_stock_runtime_carrier_probe_report.json`.
- The current public-carrier candidate config is `configs/moe/pyrrho_moe_g4_real_olmoe_stock_runtime.yaml`: **3.969692721B total / 0.402437169B active inclusive / 0.299414577B active excluding embeddings**.
- Full random-weight OLMoE structural proof now passes in both clean upstream llama.cpp and LM Studio CLI: `outputs/moe/g4_real_stock_runtime_carrier/olmoe_full_stock_runtime_probe_report.json`.
- Gate zero is passed for this carrier shape.
- The full OLMoE SFT harness smoke passes mechanically via `scripts/train_moe_olmoe_sft.py`: `outputs/moe/olmoe_g4_real_sft_full_smoke_fp32loss/train_report.json`, with finite one-step loss **11.070787** and reload smoke passing.
- The first donor-initialized OLMoE seed now exists at `outputs/moe/g4_real_olmoe_donor_init/olmoe_1b7b_block_resize_hf/`, created from `allenai/OLMoE-1B-7B-0924` with explicit resizing.
- The donor-initialized F16 GGUF loads in clean upstream llama.cpp and LM Studio; report: `outputs/moe/g4_real_olmoe_donor_init/donor_init_runtime_probe_report.json`.
- Bounded donor-seed SFT has now been tested and is quality-negative. Best small gated label-score probe reached only **39.06% accuracy / 0.00% false-TRUSTWORTHY**, the 1,024-row scale-up worsened to **33.59% / 0.58% FT**, and free generation still did not produce reliable JSON/taxonomy. Summary: `outputs/moe/g4_real_olmoe_donor_init/sft_probe_summary_2026-05-31.json`.
- Stable expert/router adaptation tooling now exists, but supervised-only probes remain quality-negative. BF16 low-LR raw router/lm_head unfreeze avoided NaN but was unsafe (**34.38% / 62.79% FT** raw); rank-4 OLMoE expert-down LoRA reached only **37.50% / 2.33% FT**; attention+expert LoRA collapsed to **31.25% / 0.00% FT**. Summary: `outputs/moe/g4_real_olmoe_donor_init/stable_adaptation_probe_summary_2026-05-31.json`.
- Bounded teacher distillation is also quality-negative at this scale. The `pyrrho-nano-g3` teacher is strong on the exact 64-row eval sample (**96.88% / 2.33% FT**), but the best student probe still only reached **39.06% / 2.33% FT** and the other tested surfaces collapsed. Summary: `outputs/moe/g4_real_olmoe_donor_init/teacher_distillation_probe_summary_2026-05-31.json`.
- A fuller raw expert `gate_up_proj` + `down_proj` OLMoE LoRA surface now works mechanically, but it is also quality-negative. Rank-4 gate/up+down distillation at LR **1e-4** became unsafe at tau **0.50** (**35.94% / 48.84% FT**) and only became safe by collapsing to **34.38% / 2.33% FT** at tau **0.65**. LR **3e-5** was safe at tau **0.50** (**31.25% / 2.33% FT**) but had no TRUSTWORTHY recall, and its best safe gate was only **32.81% / 0.00% FT**. Summary: `outputs/moe/g4_real_olmoe_donor_init/gate_up_expert_surface_probe_summary_2026-05-31.json`.
- A second donor initialization strategy, `--resize-strategy head-topnorm-slice`, now exists and produced a valid full HF seed at `outputs/moe/g4_real_olmoe_donor_init/olmoe_1b7b_head_topnorm_slice_hf/`, but it is also quality-negative. The no-training 64-row baseline was unsafe at tau **0.50** (**35.94% / 46.51% FT**) and only safe by near-ABSTAIN collapse. The controlled expert-down teacher-distill comparison tied the old accuracy ceiling at **39.06%**, but with worse FT (**4.65%**) than the old block-resize same-recipe result (**2.33%**). Summary: `outputs/moe/g4_real_olmoe_donor_init/head_topnorm_slice_init_probe_summary_2026-05-31.json`.
- External current-base controls are not an easy escape hatch. `Qwen/Qwen3.5-0.8B-Base` and `LiquidAI/LFM2.5-8B-A1B` both load locally and were tested on the same 64-row label-score control plus a 128-row LoRA+teacher probe. Raw Qwen tied the **39.06%** safe ceiling only at tau **0.90** with no DISPUTED predictions, and its tiny LoRA/distill run regressed to **35.94% / 2.33% FT**. Raw LFM had a higher unsafe tradeoff (**42.19% / 11.63% FT**) but its best safe gate was only **34.38% / 4.65% FT**, and tiny LoRA/distill collapsed to **35.94% / 0.00% FT** with no TRUSTWORTHY predictions. Summary: `outputs/external_baselines/pyrrho_moe_external_baseline_control_summary_2026-05-31.json`.
- Enrichment is easier than governance. A side probe against the real fitz-sage `KragEnricher` contract shows `Qwen/Qwen3.5-0.8B-Base` can handle a small local enrichment-bus fixture when given **1024** output tokens: **2/2** batch calls parsed with correct item count, **8/8** enriched item shapes, **8/8** nonempty keyword lists, and **35/39** anchor hits. The LM Studio `Qwen3.5-0.8B-Q4_K_M.gguf` variant also passes CPU-only enrichment via `--gpu off`, averaging **39.26 completion tok/s** across three runs and **34/39** anchor hits. The same probe failed for local HF `LiquidAI/LFM2.5-8B-A1B` in this setup and for the current LM Studio endpoint model `pyrrho-moe-g4-real-olmoe-donor-init`. Script: `scripts/probe_fitz_sage_enrichment_bus.py`; reports: `outputs/enrichment_bus_probe/`.
- Next is stronger initialization/training-surface work before any longer fitz-gov scale-up. Do not call the OLMoE seed capable until bounded held-out quality evidence moves beyond the current ~39% ceiling.

## Output Contract

The model should generate compact governance JSON:

```json
{
  "classification": "ABSTAIN | DISPUTED | TRUSTWORTHY",
  "rationale": "short evidence-grounded explanation",
  "route": "science_medicine | law_policy | ...",
  "taxonomy_pattern": "fitz-gov taxonomy pattern",
  "signals": {
    "false_trustworthy_risk": 0.12,
    "retrieval_retry_value": 0.64,
    "query_evidence_alignment": 0.41,
    "answer_coverage": 0.37
  }
}
```

Raw generation cannot be ignored as a release blocker. If it is unsafe, the model is not a clean release.

## Current Reality

- `pyrrho-MoE-g3-mvp` proved a 4B/A0.4B-class governance MoE path can run on CPU around **4.224 GiB** with Q4 GGUF and full-sequence label scoring.
- It does **not** satisfy the new goal because it requires patched llama.cpp, fails LM Studio load, depends on Qwen3MoE `mlp_only_layers`, and relies on label scoring as the safe decision path.
- The first clean `g4-real` architecture candidate passes the local budget/runtime-shape audit: **4.092512305B total / 0.412010545B active inclusive / 0.346474545B active excluding embeddings**.
- A tiny stock carrier probe passes with `MixtralForCausalLM`: a random HF checkpoint converted through an unpatched upstream llama.cpp converter and loaded/generated through unpatched upstream `llama-cli` at commit `568aec82d2fc48341c54cae565768ac75072a31d`.
- The user does not want public Mixtral association. Treat that Mixtral proof as internal evidence only.
- The non-Mixtral stock carrier candidate is now `OlmoeForCausalLM`. A tiny random OLMoE checkpoint converted and loaded through the same unpatched upstream llama.cpp path; report: `outputs/moe/g4_real_stock_runtime_carrier/olmoe_stock_runtime_carrier_probe_report.json`.
- OLMoE compatibility forced a re-budgeted full candidate: **24** all-MoE layers, **19** experts/layer, `ffn_dim=2688`, full `kv_heads=16`, untied embeddings, and 50,304-token OLMoE tokenizer seed.
- The full random OLMoE-shaped GGUF is **7,942,296,864 bytes** and loads in clean upstream `llama-cli` with **7,623 MiB** host memory reported; SHA256 `5bc7c7e6a9f6ec4095477279937e34c95167af454148d489cfc3d242046d62da`.
- The same GGUF imports into LM Studio by hard link and loads through `lms load pyrrho-moe-g4-real-olmoe-structural --gpu off -c 128 --ttl 5 -y`, reporting **5.68s** load time and **7.40 GiB**.
- The OLMoE SFT wrapper proves the full carrier can run the pyrrho generative SFT loop and save/reload a LoRA adapter, but quality is meaningless because the base is random.
- The first OLMoE donor transplant filled **3,969,688,576 / 3,969,688,576** target CausalLM params with **0** missing tensors, converted to a **7,942,296,800** byte F16 GGUF, and loaded in both clean llama.cpp and LM Studio as `pyrrho-moe-g4-real-olmoe-donor-init`.
- The donor baseline was not trained for pyrrho behavior: a 12-row label-score smoke scored **1/12** accuracy.
- Bounded attention-LoRA SFT on the donor seed is now a negative branch: small probes learned some label text but not useful governance decisions, the 1,024-row scale-up regressed, and naive router/lm_head raw unfreeze was numerically unstable.
- The NaN can be avoided with BF16/lower LR, and raw expert tensors can now be adapted with OLMoE-specific expert-down LoRA, but supervised-only quality still does not move past the earlier 39% small-probe ceiling.
- Teacher distillation on the donor-initialized OLMoE seed has now been tested across expert-down, router, gate/up expert surfaces, a second head-topnorm donor resize seed, and external Qwen/LFM controls. All are quality-negative at this small scale. The next valid work is stronger initialization/upcycling or a materially broader training surface for the OLMoE carrier, not `g3-mvp` polish, not random-only scaling, not more supervised-only adapter scaling, and not more 64-128 row adapter probes with the same recipe.
