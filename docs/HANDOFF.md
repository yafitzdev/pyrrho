# HANDOFF — pyrrho project current status

> Fresh-session entry point. Read this first.
> For the *history* of how we got here, see [LOG.md](LOG.md).
> For the full plan and roadmap, see [PROJECT.md](PROJECT.md).

This file is **overwritten** as the state changes. It always reflects only the current state.

---

## What pyrrho is, in 30 seconds

Fine-tuned classification models for RAG governance. Given a (query, retrieved contexts) pair, predicts `ABSTAIN` / `DISPUTED` / `TRUSTWORTHY`. Drop-in replacement for the constraint+sklearn pipeline in [fitz-sage](https://github.com/yafitzdev/fitz-sage). Encoders for CPU production; generative SLMs as a HuggingFace portfolio. Published benchmark contract for new work is [fitz-gov](https://github.com/yafitzdev/fitz-gov) V8.0.1, the modality-labeled patch over the V8.0.0 row set; `pyrrho-nano-g3` is the published V8 encoder release, `pyrrho-nano-g2` remains the V7.0.1 encoder release, and `pyrrho-nano-g1` metrics remain from the V5.1 eval hold-out.

The brand name is from Pyrrho of Elis — the Greek philosopher whose school practiced suspension of judgment when evidence was insufficient.

## Current state of the family

| Release | Status |
|---|---|
| `pyrrho-nano-g1` (ModernBERT 149M encoder) | Trained + 3-seed validated. On HuggingFace. **In production — it is the governance backend of fitz-sage v0.13.0.** |
| `pyrrho-small-g1` (Qwen3.5-0.8B + LoRA, V5.1 plain SFT) | Trained + 3-seed validated locally. **Not on HF — fails the FT gate (12.13% vs 5.7% bar).** Release dir staged at `models/pyrrho-small-g1/`. See LOG 2026-05-20 evening. |
| `pyrrho-small-g1.1` (Qwen3.5-0.8B + LoRA, V5.1 SFT with class weights + label smoothing) | Trained + 3-seed validated locally. **Still not on HF — FT improved 12.13 → 9.31% but still misses 5.7% gate by ~3.6 pts.** Release dir staged at `models/pyrrho-small-g1.1/`. See LOG 2026-05-20 night. |
| `pyrrho-nano-g1.1` (ModernBERT V6 retrain) | Attempted locally on V6; **not released**. 3-seed result was 81.54 ± 5.97% accuracy / 5.31 ± 0.21% false-trustworthy, with high variance after toolchain drift. Superseded by direct V7 `g2` work. See LOG 2026-05-21. |
| `pyrrho-nano-g2` (ModernBERT V7 retrain) | **Trained + 3-seed validated on fitz-gov V7.0.1 schema-clean contract (same rows/splits/labels as V7.0.0). On Hugging Face.** Held-out test result: **95.24 ± 0.48% accuracy / 3.48 ± 0.40% false-trustworthy**. Passes gates by a wide margin. |
| `pyrrho-nano-g3` (ModernBERT V8 retrain) | **Trained + 3-seed validated on the fitz-gov V8 row set, now published as V8.0.1 with explicit `meta.modality: "unstructured"`. On Hugging Face at [`yafitzdev/pyrrho-nano-g3`](https://huggingface.co/yafitzdev/pyrrho-nano-g3).** Held-out V8 test result: **97.52 ± 0.43% accuracy / 1.42 ± 0.16% false-trustworthy** on 2,459 cases. Release artifact uses seed **1337** because it passed the packaged smoke check while also clearing held-out gates (**97.68% / 1.54% FT**). |
| `pyrrho-nano-g2.1-v8-probe` (ModernBERT local V8 probe retrain) | **Local-only experimental retrain** on published V7 splits plus the 525-row V8 cohort appended by manifest. Mixed held-out test result: **95.51 ± 0.43% accuracy / 3.56 ± 0.38% false-trustworthy** on 1,107 rows. Automotive ECU OOD probe improved from **7.00/10** to **8.33/10** mean across the same 3 seeds. Not published. |
| `pyrrho-nano-g2.1-v8-verdict-patch` (ModernBERT local ablation) | **Failed local ablation; do not publish.** Added 105 hard `verdict_conflict` rows on top of the 525-row V8 probe. Held-out test still passed gates (**94.92 ± 0.41% / 4.08 ± 0.92% FT**), and `ecu_04` improved **1/3 -> 2/3**, but ECU OOD mean regressed **8.33/10 -> 7.33/10** by over-predicting DISPUTED on nearby TRUSTWORTHY/ABSTAIN controls. |
| `pyrrho-nano-g2.2` (ModernBERT local V8 balanced-controls retrain) | **Trained + 3-seed validated locally; do not publish.** Uses published V7 plus the older **840-row** V8 clean set. Held-out test passes gates at **95.49 ± 0.15% accuracy / 3.06 ± 0.61% false-trustworthy**, but it is now superseded by the active **14,092-row** V8 target-50 vault and should be treated as an ablation checkpoint, not a release candidate. |
| `pyrrho-nano-g2.3-v8-claude4200` (ModernBERT local V8 expanded retrain) | **Older data prep only; do not train as the next run.** It used published V7 plus the earlier **4,200-row** V8 vault. Prepared data: `data/processed_v8_claude4200` with train=11,773 / eval=1,439 / test=1,488. Superseded by the active **14,092-row** V8 target-50 vault; rebuild prep before the next retrain. |
| `pyrrho-small-g2` | Not started. See [ROADMAP.md](ROADMAP.md). |
| `pyrrho-MoE-g3-alpha` | **Custom Stage 0.7 support aggregation is the current quality-positive MoE baseline; Stage 0.6 remains the safety reference; post-hoc Stage 0.7 verifier is the first positive guard result; Qwen3-MoE Stage 1 remains negative and is not a release candidate.** Config/parameter counter, V8 MoE data prep, seed-search report, tiny route prototype, g3 teacher-logit sidecars, repaired Qwen upcycling config, local 30-shard seed pack, governance wrapper, Stage 1 negative probes, Stage 0 route-first diagnostics, Stage 0.5/0.6 route-coupled custom trunks, Stage 0.6b-e recipe sweep, Stage 0.7 support-aggregation 3-seed validation, Stage 0.8/0.9 guard probes, Stage 0.7 frozen-output post-hoc verifier, and Stage 0.7 verifier packaging/reload are complete. Stage 0.7 held-out test: **89.49 ± 0.47%** calibrated accuracy / **3.06 ± 0.45%** false-trustworthy / **82.61 ± 2.50%** route / **75.78 ± 0.21%** taxonomy. Stage 0.7 post-hoc verifier has two validated operating points: safety-heavy **88.97 ± 0.51% / 1.99 ± 0.17% FT** and preferred support-retaining **89.29 ± 0.69% / 2.37 ± 0.26% FT**. Best full-eval frozen-head Qwen checkpoint is only **54.66%** calibrated accuracy / **5.35%** false-trustworthy, well below the encoder gate; current Qwen adapter/distillation variants also fail the continuation gate. |

## pyrrho-MoE architecture clarification

Canonical spec: `C:/Users/yanfi/PycharmProjects/pyrrho/docs/PYRRHO_MOE_ARCHITECTURE.md`.

The terminal `pyrrho-MoE` target is still **custom 4B total / 0.4B active**, CPU-runnable, with pyrrho-defined experts and supervised routing from fitz-gov (`routing.expert_fired`). It is not an off-the-shelf LFM2/Qwen/other MoE checkpoint renamed as pyrrho. Because full general-language pretraining from scratch is outside the project budget, the intended path is **dense-to-MoE upcycling plus distillation**: pick a compatible pretrained dense SLM seed, clone its dense FFNs into sparse experts, preserve general competence through teacher distillation, then train the custom sparse expert layout on fitz-gov governance/routing tasks. Baseline spec is 24 layers, hidden size 1024, 20 MoE layers, 16 physical experts/layer grouped into 8 semantic expert groups, top-1 routing, and **3.950935086B total / 0.411991086B active inclusive** under the implemented counter. `LiquidAI/LFM2-8B-A1B` is only a proxy/teacher/comparison candidate, not the final 4B-A0.4B architecture.

Current MoE scaffold:
- Config: `configs/moe/pyrrho_moe_g3_alpha.yaml`.
- Parameter counter: `scripts/count_moe_params.py`.
- MoE V8 prep/audit: `scripts/prepare_moe_data.py`; strict audit against the V8 row set wrote `data/moe_v8` with **train=19,674 / eval=2,459 / test=2,459**, **0** missing required fields, **23** taxonomy patterns, and route IDs for the 8 semantic expert groups. Future prep should pull HF `v8.0.1` so every row carries `meta.modality: "unstructured"`.
- Stage 0 trainer: `scripts/train_moe.py`; full local run at `outputs/moe/stage0_route_proto/` trained a **10,505,009-param** hashed-token top-1 MoE prototype for 3 epochs in ~38s. Held-out test: **82.47%** governance accuracy, **5.63%** false-trustworthy, **81.09%** route accuracy, **65.80%** taxonomy accuracy.
- Stage 0 g3 distillation diagnostics: `scripts/generate_moe_teacher_logits.py` wrote full `pyrrho-nano-g3` sidecars at `outputs/moe/teacher_logits/pyrrho_nano_g3_full_v8/` (**train=19,674 / eval=2,459 / test=2,459**). `outputs/moe/stage0_route_proto_distill_g3_route15/final_metrics.json` used governance KL distillation plus higher route loss and reached held-out test **82.43%** calibrated accuracy / **5.45%** FT / **82.80%** route / **64.99%** taxonomy. This confirms the V8 route signal is learnable when supervised semantic route is the actual active expert path; it does **not** rescue the current Qwen adapter path.
- Stage 0.5 route-coupled custom student: `configs/moe/pyrrho_moe_stage0_5_route_coupled.yaml` trains a **53,861,425-param** hash-token student where the selected semantic route drives every residual expert layer. Three-seed full V8 run at `outputs/moe/stage0_5_route_coupled_g3_3seed/summary.json`: held-out test **83.91 ± 1.18%** calibrated accuracy / **5.55 ± 0.03%** FT / **82.92 ± 0.35%** route / **67.64 ± 1.23%** taxonomy. Per-seed calibrated test accuracy / FT: seed 42 **84.47% / 5.51%**, seed 1337 **84.71% / 5.57%**, seed 7 **82.55% / 5.57%**. Gold-route mean was **84.64 ± 1.05%** calibrated / **5.29 ± 0.42%** FT, so the remaining gap is not primarily route prediction. A too-conservative route-heavy variant (`stage0_5_route_coupled_g3_route15`) scored only **77.51%** calibrated / **0.77%** FT.
- Stage 0.5 failure analysis: `scripts/analyze_moe_failures.py` writes per-case predictions, per-route/per-taxonomy breakdowns, seed-overlap histograms, and top hard-error lists. Test report at `outputs/moe/stage0_5_route_coupled_g3_3seed/failure_analysis_test/failure_report.md`; eval report at `outputs/moe/stage0_5_route_coupled_g3_3seed/failure_analysis_eval/failure_report.md`. Test split has **109/2,459** all-seed hard errors (**4.43%**), **12** all-seed false-TRUSTWORTHY cases, and **202** rows with FT in at least one seed. Weak test routes: `science_medicine` (**78.80%** accuracy / **12.38%** FT), then `technology_computing` and `general_commonsense`. Weak test taxonomy groups: `consistent_chain` (**66.43%**), `multi_source_corroboration` (**67.38%**), `quantitative_consensus` (**71.11%**), and `factual_contradiction` (**77.88%**, **12.98%** FT). Eval split repeats the core pattern: **106** all-seed hard errors and `consistent_chain` is still weakest (**62.99%**).
- Stage 0.6 token route-coupled baseline: `TokenRouteCoupledMoEForGovernance` in `src/pyrrho/moe/modeling.py` is a **55,728,817-param** hash-token student with RoPE self-attention, RMSNorm pre-norms, route-selected SwiGLU FFNs, and last-token/mean pooled heads. Config: `configs/moe/pyrrho_moe_stage0_6_token_route_coupled.yaml`. Three-seed run at `outputs/moe/stage0_6_token_route_coupled_g3_3seed/summary.json`: held-out test **87.23 ± 1.29%** calibrated accuracy / **2.92 ± 1.06%** FT / **86.06 ± 0.94%** route / **71.97 ± 0.72%** taxonomy. Per-seed calibrated test accuracy / FT: seed 42 **88.33% / 2.37%**, seed 1337 **87.56% / 4.15%**, seed 7 **85.81% / 2.25%**. Gold-route mean was **87.61 ± 1.51%** calibrated / **2.86 ± 1.22%** FT, so the remaining gap is still mostly trunk/pattern handling rather than route prediction.
- Stage 0.6 failure analysis: reports at `outputs/moe/stage0_6_token_route_coupled_g3_3seed/failure_analysis_{eval,test}/`. Test split has **140/2,459** all-seed hard errors (**5.69%**), **18** all-seed false-TRUSTWORTHY cases, and **92** rows with FT in at least one seed. Compared with Stage 0.5, headline metrics and safety improved strongly, especially `science_medicine` (**78.80% / 12.38% FT → 82.22% / 4.08% FT**) and `factual_contradiction` (**77.88% / 12.98% FT → 89.68% / 5.31% FT**). The unresolved regression is support-pattern TRUSTWORTHY recall: `multi_source_corroboration` fell **67.38% → 59.50%** and `quantitative_consensus` fell **71.11% → 65.40%**, while `consistent_chain` only improved **66.43% → 69.27%**.
- Stage 0.6b-e support-recall recipe sweep: added per-pattern support weights in `src/pyrrho/moe/losses.py` and configs `pyrrho_moe_stage0_6{b,c,d,e}_*.yaml`. Seed-42 probes show scalar weighting is not enough. 0.6c/0.6d can recover `multi_source_corroboration` / `quantitative_consensus`, but leak FT; 0.6e guards FT (**1.66%**) but collapses support recall. Do **not** scale 0.6b/0.6c/0.6d/0.6e to 3 seeds.
- Stage 0.7 support-aggregation baseline: `SupportAggregatingMoEForGovernance` in `src/pyrrho/moe/modeling.py` keeps the Stage 0.6 flat token route-coupled trunk, adds query/source tensors in `src/pyrrho/moe/data.py`, and fuses query-source support pooling into the terminal heads. Config: `configs/moe/pyrrho_moe_stage0_7_support_aggregation.yaml` (**4 epochs**; the same recipe overfit/shifted unsafe at 5 epochs on seed 42). Three-seed run at `outputs/moe/stage0_7_support_aggregation_g3_3seed/summary.json`: held-out test **89.49 ± 0.47%** calibrated accuracy / **3.06 ± 0.45%** FT / **82.61 ± 2.50%** route / **75.78 ± 0.21%** taxonomy. Per-seed calibrated test accuracy / FT: seed 42 **90.04% / 3.26%**, seed 1337 **89.18% / 2.55%**, seed 7 **89.26% / 3.38%**. Gold-route mean was **89.60 ± 0.55%** calibrated / **3.14 ± 0.52%** FT, so route prediction is not the limiting factor. Support patterns improved versus Stage 0.6: `consistent_chain` **69.27% → 75.18%**, `multi_source_corroboration` **59.50% → 69.53%**, `quantitative_consensus` **65.40% → 79.05%**. Safety caveat: `science_medicine` accuracy improved **82.22% → 85.21%**, but FT worsened **4.08% → 5.58%**; `factual_contradiction` stayed near Stage 0.6 accuracy but FT worsened **5.31% → 6.19%**. Next work should preserve Stage 0.7 support gains while restoring Stage 0.6 safety on these slices.
- Stage 0.7b-d guarded support probes: configs `pyrrho_moe_stage0_7{b,c,d}_*.yaml` ran seed-42 probes. 0.7b restored safety best (**88.98%** accuracy / **2.13%** FT; `science_medicine` FT **2.86%**, `factual_contradiction` FT **3.54%**) but cut support recall (`multi_source_corroboration` **61.29%**, `quantitative_consensus` **73.33%**). 0.7c restored multi-source recall but not safety; 0.7d did not dominate. Post-hoc stricter TRUSTWORTHY thresholds and the existing `false_trustworthy_risk` scalar head also trade off support recall too quickly. Do **not** scale 0.7b/0.7c/0.7d to 3 seeds; move to an architectural guard head.
- Stage 0.8 guarded-head scaffold: `GuardedSupportAggregatingMoEForGovernance` adds a learned positive TRUSTWORTHY penalty on top of the Stage 0.7 support-fused state. Config: `configs/moe/pyrrho_moe_stage0_8_guarded_support_aggregation.yaml`. CUDA smoke/reload passed, but seed-42 quality is negative. Four epochs overfit to **85.28%** calibrated accuracy / **5.57%** FT; the 3-epoch checkpoint was safer but too conservative at **85.40%** accuracy / **2.31%** FT / **76.78%** route / **73.32%** taxonomy. Do **not** scale this Stage 0.8 implementation.
- Stage 0.9 explicit trust-guard probe: `TrustGuardedSupportAggregatingMoEForGovernance` adds a separately supervised binary trust verifier over Stage 0.7 candidate logits/support state. Config: `configs/moe/pyrrho_moe_stage0_9_trust_guarded_support_aggregation.yaml`. CUDA smoke/reload/tests passed, but seed-42 quality is negative. Three epochs reached **86.50%** calibrated accuracy / **1.24%** FT / **82.19%** route / **72.83%** taxonomy; four epochs reached **84.75%** / **0.59%** FT / **84.38%** route / **73.97%** taxonomy. It restores safety by collapsing support recall (`multi_source_corroboration` **45.16%** at 3 epochs, **38.71%** at 4 epochs). Do **not** scale this implementation.
- Stage 0.7 post-hoc verifier: `scripts/train_moe_posthoc_verifier.py` trains a separate HGB verifier on frozen Stage 0.7 candidate outputs and only demotes candidate TRUSTWORTHY predictions. Safety-heavy 2.5% eval-FT target artifact: `outputs/moe/stage0_7_posthoc_verifier_g3_3seed/summary.json` (**88.97 ± 0.51%** accuracy / **1.99 ± 0.17%** FT). Preferred support-retaining 2.8% eval-FT target artifact: `outputs/moe/stage0_7_posthoc_verifier_g3_3seed_ft028/summary.json`, which moves the verifier-script baseline **89.37 ± 0.59% / 2.94 ± 0.36% FT** to guarded **89.29 ± 0.69% / 2.37 ± 0.26% FT**. Minimal-intervention 3.0% eval-FT target artifact: `outputs/moe/stage0_7_posthoc_verifier_g3_3seed_ft030/summary.json` (**89.35 ± 0.64% / 2.61 ± 0.36% FT**) preserves support best but is a weaker safety move and no-ops seed 7. `scripts/package_moe_posthoc_verifier.py` packages the preferred 2.8% reranker as `outputs/moe/stage0_7_posthoc_verifier_g3_ft028_package/`; reload evaluation over eval+test reproduced the packaged per-seed metrics with **0** max absolute delta and feature schema width **120**. Treat the verifier as the current positive guard direction, not a trunk architecture.
- Stage 0 evaluator: `scripts/eval_moe.py`; reproduces saved tiny, pooled route-coupled, token route-coupled, and support-aggregating checkpoints and writes eval/test reports without retraining.
- Seed/upcycling decision: `docs/MOE_UPCYCLING_DECISION_2026-05-26.md`; first upcycling target is `configs/moe/pyrrho_moe_g3_alpha_qwen.yaml` using `Qwen/Qwen3-0.6B-Base`, Qwen's full 151,936-token vocab, 28 layers, KV=8, explicit `head_dim=128`, 24 MoE layers, 48 experts/layer, and FFN dim 1056. Count with the 15 V8 scalar heads: **4.083139633B total / 0.423871537B active inclusive**. This keeps the active budget but requires FFN compression from Qwen's 3072-wide dense FFNs.
- Upcycling inspector: `scripts/upcycle_dense_to_moe.py --inspect-only` validates the Qwen seed/config match and wrote `outputs/moe/upcycling/qwen_alpha_inspect.json`. `--real-weight-smoke` verifies real Qwen safetensors on one layer and wrote `outputs/moe/upcycling/qwen_alpha_real_weight_smoke.json`. Direct-copy surfaces are embeddings, attention, Q/K attention norms, and final norm; FFNs require **3072 -> 1056** compression before cloning into experts.
- FFN compression utility: `src/pyrrho/moe/upcycling.py` selects the strongest seed FFN channels by combined gate/up/down norm and slices `gate_proj`, `up_proj`, and `down_proj` consistently into the 1056-wide target.
- Upcycled seed pack: `scripts/upcycle_dense_to_moe.py --write-seed-pack outputs/moe/upcycling/qwen_alpha_seed_pack --output outputs/moe/upcycling/qwen_alpha_seed_pack_plan.json` materialized a Qwen3-MoE-compatible local seed pack: **30** safetensors shards, **310** tensors, **8.166 GB** total tensor bytes. Shape validation passed via `outputs/moe/upcycling/qwen_alpha_seed_pack/load_shape_report.json`: expected 311 model tensors, manifest has 310, with tied `lm_head.weight` intentionally omitted and mapped to `model.embed_tokens.weight`.
- Governance wrapper smoke: `scripts/smoke_moe_qwen_wrapper.py --max-length 64 --batch-size 2` loaded `outputs/moe/upcycling/qwen_alpha_seed_pack/` on CUDA, attached pyrrho heads via `src/pyrrho/moe/qwen_governance.py`, and wrote `outputs/moe/upcycling/qwen_alpha_wrapper_smoke.json`. Output shapes passed: governance `[2,3]`, route `[2,8]`, taxonomy `[2,23]`, scalar `[2,15]`; multitask loss computed (`11.4245`) with no-training random heads.
- Stage 1 Qwen trainer: `scripts/train_moe_qwen_heads.py` freezes the Qwen3-MoE trunk by default and trains pyrrho governance/route/taxonomy/scalar heads. It supports deterministic bounded sampling, calibrated TRUSTWORTHY threshold reporting, `heads.pt` eval-only reload, split head/router/trunk LRs, final-dense partial unfreeze, PEFT LoRA, trainable physical-expert residual adapters (`--expert-adapter-r`), trainable semantic-route pooled adapters (`--semantic-adapter-r`), optional g3 governance-logit distillation via `--teacher-logits-dir` / `--loss-distillation`, and oracle-route comparison via `--eval-compare-gold-routes`.
- Stage 1 Qwen result: bounded heads-only learning is real but weak. Best full-eval artifact remains `outputs/moe/qwen_heads_stage1_8192_random_lr3e4_ft12_eval_full/eval_report.json`: raw **57.18%** accuracy / **15.99%** FT; calibrated **54.66%** accuracy / **5.35%** FT at tau **0.54**; route **43.51%**, taxonomy **34.40%**. Negative probes now include internal-router-only (**39.45%** calibrated accuracy / **3.85%** FT), final dense unfreeze (**38.09%** calibrated accuracy on the 512-row slice), attention-only LoRA (**43.55%** calibrated accuracy / **5.22%** FT / **26.17%** route), physical expert adapters + g3 distillation (**50.00%** calibrated accuracy / **4.40%** FT / **24.02%** route on the 512-row slice), and semantic-route adapters + g3 distillation (**44.34%** calibrated accuracy / **1.65%** FT / **26.37%** route on the 512-row slice; only **45.51%** calibrated accuracy when forced to gold routes). Do not scale these Qwen Stage 1 variants as release candidates.

## Validated metrics

### `pyrrho-nano-g1` (encoder, calibrated, 3-seed mean ± std on V5.1 eval, 584 cases)

| Metric | pyrrho-nano-g1 | sklearn baseline | Δ |
|---|---|---|---|
| Overall accuracy | **86.13 ± 0.86%** | 78.7% | **+7.43** |
| False-trustworthy rate | **5.27 ± 0.21%** | 5.7% | **-0.43** (safer) |
| Trustworthy recall | **79.38 ± 1.64%** | 70.0% | **+9.38** |
| Disputed recall | **94.81 ± 1.28%** | 86.1% | **+8.71** |
| Abstain recall | **92.94 ± 1.11%** | 86.5% | **+6.44** |
| CPU inference (est.) | ~30 ms | ~500–2000 ms (5 LLM calls) | ~50× faster |

Every margin is multiple standard deviations larger than seed noise (LOG 2026-05-14).

### `pyrrho-nano-g2` (encoder, calibrated, 3-seed mean ± std on V7 held-out test, 1,050 cases)

Trained on published fitz-gov **V7** default `v7` splits: train=8,400 / validation=1,050 / test=1,050. V7.0.1 is a schema-clean republish of V7.0.0 with the same rows, labels, and splits; no retrain was required. Checkpoint and TRUSTWORTHY threshold are selected on validation; headline numbers below are from the separate held-out test split.

| Metric | pyrrho-nano-g2 | release gate / baseline | Δ |
|---|---|---|---|
| Overall accuracy | **95.24 ± 0.48%** | 78.7% | **+16.54** |
| False-trustworthy rate | **3.48 ± 0.40%** | 5.7% | **-2.22** (safer) |
| Trustworthy recall | **93.66 ± 0.30%** | 70.0% | **+23.66** |
| Disputed recall | **97.00 ± 1.17%** | 86.1% | **+10.90** |
| Abstain recall | **95.25 ± 0.00%** | 86.5% | **+8.75** |
| Trustworthy precision | **94.06 ± 0.66%** | n/a | — |

Validation-split calibrated metrics were **94.92 ± 0.29% accuracy / 2.89 ± 0.26% false-trustworthy**. Every seed passed both gates on validation and held-out test. Training artifacts: `outputs/multi_seed_g2/summary.json`, per-seed best checkpoints under `outputs/multi_seed_g2/seed_*/best_model/`, and per-seed breakdown reports at `outputs/multi_seed_g2/seed_*/eval_report.json`. Release dir: `models/pyrrho-nano-g2/`.

### `pyrrho-nano-g3` (encoder, calibrated, 3-seed mean ± std on V8 held-out test, 2,459 cases)

Trained on the published fitz-gov V8 row set, now available as **V8.0.1** with explicit row-level `meta.modality: "unstructured"` and the same default `v8` splits: train=19,674 / validation=2,459 / test=2,459. Config: `configs/encoder/modernbert_base_g3_v8.yaml`. Checkpoint and TRUSTWORTHY threshold are selected on validation; headline numbers below are from the separate held-out test split.

| Metric | pyrrho-nano-g3 | vs published `g2` |
|---|---|---|
| Overall accuracy | **97.52 ± 0.43%** | **+2.28** |
| False-trustworthy rate | **1.42 ± 0.16%** | **-2.06** (safer) |
| Trustworthy recall | **96.28 ± 0.83%** | **+2.62** |
| Disputed recall | **98.34 ± 0.24%** | **+1.34** |
| Abstain recall | **97.83 ± 0.76%** | **+2.58** |
| Trustworthy precision | **96.87 ± 0.34%** | **+2.81** |

Validation-split calibrated metrics were **97.84 ± 0.07% accuracy / 1.23 ± 0.10% false-trustworthy**. Every seed passed both gates on validation and held-out test. Per-seed held-out calibrated results: seed 42 **97.03% / 1.48% FT** at tau **0.68**; seed 1337 **97.68% / 1.54% FT** at tau **0.58**; seed 7 **97.84% / 1.24% FT** at tau **0.60**. Training artifacts: `outputs/multi_seed_g3_v8/summary.json`, per-seed best checkpoints under `outputs/multi_seed_g3_v8/seed_*/best_model/`, and per-seed breakdown reports at `outputs/multi_seed_g3_v8/seed_*/eval_report.json`. Release dir: `models/pyrrho-nano-g3/`, rebuilt from seed **1337** after packaged smoke selection; published at [`yafitzdev/pyrrho-nano-g3`](https://huggingface.co/yafitzdev/pyrrho-nano-g3), HF commit `397393718985e7bfa101042e89ecc60103e9c447`.

### `pyrrho-nano-g2.1-v8-probe` (encoder, calibrated, 3-seed mean ± std on local V7+V8 mixed held-out test, 1,107 cases)

Local experiment only. Data prep preserved the published V7 train/validation/test contract and appended the **525-row V8** cohort from the local fitz-gov vault by manifest assignment: **train +414 / eval +54 / test +57**, producing `data/processed_v8_probe` with train=8,814 / eval=1,104 / test=1,107.

| Metric | pyrrho-nano-g2.1-v8-probe | vs published `g2` |
|---|---|---|
| Overall accuracy | **95.51 ± 0.43%** | **+0.27** |
| False-trustworthy rate | **3.56 ± 0.38%** | **+0.08** (slightly worse) |
| Trustworthy recall | **94.72 ± 0.29%** | **+1.06** |
| Disputed recall | **96.21 ± 0.27%** | **-0.79** |
| Abstain recall | **95.71 ± 0.96%** | **+0.46** |

Automotive ECU/test-management OOD probe on the recovered 10-case fixture stays exact-string OOD against both `data/processed_v7` and `data/processed_v8_probe` (**0/10 exact query matches** in each). Mean calibrated score improved from **7.00/10** on `g2` to **8.33/10** on `g2.1-v8-probe`; per-seed scores moved **7/10 -> 8/10** (seed 42), **6/10 -> 9/10** (seed 1337), and **8/10 -> 8/10** (seed 7). Biggest gains were `resolved_candidate_selection`-style and wrong-release abstain behavior (`ecu_02`, `ecu_07`); explicit PASS/FAIL conflict resolution (`ecu_04`) improved only **0/3 -> 1/3**.

### `pyrrho-nano-g2.1-v8-verdict-patch` (failed local ablation, 1,115-case mixed held-out test)

Local experiment only. Added 105 hard final-verdict PASS/FAIL conflict rows to the V8 probe, producing `data/processed_v8_verdict_patch` with train=8,901 / eval=1,114 / test=1,115. Three-seed held-out test passed gates at **94.92 ± 0.41% accuracy / 4.08 ± 0.92% false-trustworthy**, but it is not a release candidate. ECU OOD mean dropped from the initial V8 probe's **8.33/10** to **7.33/10**. The target `ecu_04_disputed_dtc_powercycle` improved **1/3 -> 2/3**, but nearby controls regressed: `ecu_01` **2/3 -> 1/3**, `ecu_02` **2/3 -> 1/3**, and `ecu_07` **2/3 -> 0/3**. Artifact: `outputs/automotive_ood_probe/comparison_v8_verdict_patch.json`.

### `pyrrho-nano-g2.2` (local V7+840-row V8 balanced-controls retrain, 1,132-case mixed held-out test)

Local experiment only. `configs/encoder/modernbert_base_g2_2.yaml` retrained ModernBERT on `data/processed_v8_balanced_controls`, preserving the published V7 split contract and appending the 840-row V8 cohort by manifest (**+661 train / +97 eval / +82 test**). Prepared split sizes are train=9,061 / eval=1,147 / test=1,132.

| Metric | `pyrrho-nano-g2.2` | vs published `g2` |
|---|---|---|
| Overall accuracy | **95.49 ± 0.15%** | **+0.25** |
| False-trustworthy rate | **3.06 ± 0.61%** | **-0.42** (safer) |
| Trustworthy recall | **93.61 ± 0.61%** | -0.05 |
| Disputed recall | **96.27 ± 0.53%** | -0.73 |
| Abstain recall | **96.91 ± 0.44%** | +1.66 |

ECU OOD comparison artifact: `outputs/automotive_ood_probe/comparison_g2_2.json`. Exact-query leakage check stayed clean: **0/10** exact query matches in `data/processed_v7`, `data/processed_v8_probe`, and `data/processed_v8_balanced_controls`. OOD mean is **8.00/10** (per-seed **8/10, 8/10, 8/10**), which improves over published `g2` (**7.00/10**) and the failed verdict patch (**7.33/10**) but does not beat the original 525-row V8 probe (**8.33/10**). It fixes `ecu_04_disputed_dtc_powercycle` completely (**1/3 -> 3/3** vs V8 probe) and `ecu_01` (**2/3 -> 3/3**), but regresses `ecu_02_trustworthy_acceptance_run` (**2/3 -> 0/3**) and `ecu_07_abstain_wrong_ecu_release` (**2/3 -> 0/3**).

### Aviation maintenance / airworthiness OOD probe

The 10-case aviation maintenance probe is exact-query OOD against all three checked processed datasets: **0/10** exact query matches in `data/processed_v7`, `data/processed_v8_probe`, and `data/processed_v8_balanced_controls`. Artifact: `outputs/aviation_ood_probe/comparison_g2_g21_g22.json`.

| Run | Mean score | Seed scores (42 / 1337 / 7) |
|---|---:|---|
| `g2` | **7.00/10** | 7/10, 6/10, 8/10 |
| `g2.1-v8-probe` | **8.00/10** | 8/10, 9/10, 7/10 |
| `g2.2` | **8.67/10** | 9/10, 7/10, 10/10 |

This did **not** uncover a grave new field-specific taxonomy gap. The persistent hard case is `air_02_trustworthy_superseded_sb_resolved` (**1/3** on `g2.2`), which looks like the existing `resolved_candidate_selection` / superseded-candidate boundary. The other residuals are one-seed `authority_status_conflict` drift (`air_05`) and revision mismatch (`air_10`, **2/3** on `g2.2`).

### `pyrrho-small-g1` (SLM, plain SFT, 3-seed mean ± std on same eval split)

| Metric | pyrrho-small-g1 | vs `nano-g1` | vs sklearn baseline |
|---|---|---|---|
| Overall accuracy | **90.01 ± 0.55%** | **+3.88** | **+11.31** |
| False-trustworthy rate | **12.13 ± 1.27%** | **+6.86 (worse)** | **+6.43 (worse)** — **fails gate** |
| Trustworthy recall | **92.09 ± 0.19%** | +12.71 | +22.09 |
| Disputed recall | **87.16 ± 1.54%** | -7.65 | +1.06 |
| Abstain recall | **88.08 ± 2.23%** | -4.86 | +1.58 |
| Tier0 sanity accuracy | **99.44 ± 0.96%** (60-case set) | vs ~83% (encoder) | n/a |
| Decode-time fallback rate | **0.00%** (every case produced a parseable label) | — | — |

Headline finding: pre-trained world knowledge + reasoning depth genuinely lifts overall accuracy and nearly-perfects tier0, but the SLM systematically over-predicts TRUSTWORTHY because plain SFT has no safety-asymmetric signal. 12.13% FT rate is stable across seeds (11.40 / 11.40 / 13.60), so it's a recipe-level finding, not noise.

### `pyrrho-small-g1.1` (SLM, same SFT + class_weights=[2.3, 2.3, 1.0] + label_smoothing=0.15, 3-seed mean ± std)

Recipe-fix re-spin of g1 with the encoder's anti-FT regularization transplanted onto the token-level CE loss (per-example weighting in `WeightedLossSFTTrainer`).

| Metric | pyrrho-small-g1.1 | vs g1 | vs `nano-g1` |
|---|---|---|---|
| Overall accuracy | **89.55 ± 1.40%** | -0.46 | +3.42 |
| False-trustworthy rate | **9.31 ± 1.06%** | **-2.82 (improved)** | +4.04 — **still fails gate by ~3.6 pts** |
| Trustworthy recall | 89.00 ± 2.45% | -3.09 (model less aggressive on T — by design) | +9.62 |
| Disputed recall | **91.60 ± 1.13%** | **+4.44 (improved)** | -3.21 |
| Abstain recall | 88.81 ± 2.56% | +0.73 | -4.13 |
| Tier0 sanity accuracy | **96.67 ± 0.00%** | -2.77 | vs ~83% (encoder) |
| Decode-time fallback rate | **0.00%** | — | — |
| Per-seed FT rate | 8.09 / 9.93 / 9.93 | — | — |

Direction is exactly what the recipe predicts: model is less aggressive on TRUSTWORTHY (TR ↓ 3.09), more aggressive on DISPUTED (DR ↑ 4.44), FT drops ~3 pts. But the absolute FT of 9.31% is still ~3.6 pts above the 5.7% gate — the encoder's [2.3, 2.3, 1.0] + 0.15 smoothing recipe lands 5.27% on the encoder but only 9.31% on the SLM. Token-level CE on the assistant turn diffuses the safety pressure across many tokens (~11/example: 6 think-block + 3–5 label tokens + im_end) while the encoder's class-weighted CE on a single classification head concentrates it. Bumping to more aggressive interventions (e.g., class_weights=[5, 5, 1], label_smoothing=0.25, or moving to ft_penalized_accuracy selection / threshold-based post-processing / DPO) is the next lever set for a `small-g1.2`.

## Hard V8 Dataset Contract

V8 must be a coherent SDGP dataset, not a compatibility layer. The
source-of-truth contract is
`C:/Users/yanfi/PycharmProjects/fitz-gov/docs/V8_SCHEMA_CONTRACT.md`.

- No legacy shims, no compatibility configs, no old pre-SDGP report axes.
- V8 keeps the current V7.0.1 SDGP row shape: `id`, `version`, `input`, `governance`, `taxonomy`, `routing`, `meta`, `evaluation` plus local `_vault` provenance.
- Taxonomy gaps are first-class `taxonomy.pattern` values, not `taxonomy.subpattern` or side-channel fields.
- Existing 10,500 rows are not rewritten for this additive taxonomy expansion.
- New V8 rows use the existing cohort marker: `version: "fitz-gov-8.0"` and `meta.dataset_version: "v8"`.
- Any future V8 testcase addition must follow `C:/Users/yanfi/PycharmProjects/fitz-gov/docs/SDGP_TESTCASE_ADDITION_CYCLE.md`: candidate rows get structural dry-run and offline blind-label QA before active-vault merge. The current local Qwen QA settings are pinned there (`qwen3.6-35b-a3b@q5_k_s`, `max_tokens=2048`, `request_timeout_s=300`, `temperature=0.0`).
- Initial V8 taxonomy-gap implementation is in fitz-gov: five new primary patterns (`resolved_candidate_selection`, `verdict_conflict`, `authority_status_conflict`, `version_build_mismatch`, `missing_execution_result`) expanded across 7 current domains x 3 difficulties = 105 cells. The active local V8 set is now **14,092 QA-clean rows**: original 840 clean rows + 3,360 repaired Claude-handoff rows + 5,198 target-40 rows + 4,694 target-50 rows. Plan file: `C:/Users/yanfi/PycharmProjects/fitz-gov/docs/V8_TAXONOMY_EXPANSION_PLAN.md`.
- Expansion rows are generated, merged, structurally audited, repaired, and blind-label clean locally: vault is **24,592 rows** = 10,500 V6/V7 + **14,092 V8**. Current clean manifest is `C:/Users/yanfi/PycharmProjects/fitz-gov/data/sdgp_v8_qa/blind_label_manifest.jsonl`. Training-schema audit is **14,092/14,092 complete** at `C:/Users/yanfi/PycharmProjects/fitz-gov/data/sdgp_v8_qa/training_schema_summary.json`.
- The Claude-generated candidate handoff at `C:/Users/yanfi/PycharmProjects/fitz-gov/data/sdgp_handoff_v8_candidate_20260525_claude_expand/` has been repaired and merged. Raw intake initially failed structural dry-run (**1,915 accepted / 0 existing / 624 rejected**). The normalized pack had **124** Codex subagent triage rows; those rows were replaced with deterministic V8 template cases in `subagent_outputs_patched_124_template/`. Patched candidate QA passed **3,360/3,360 agreement**, **0 missing / 0 invalid / 0 error**, **0 triage**, then merged as batch `v8_candidate_20260525_claude_expand_patched_124_template` (**3,360 added / 0 duplicate**).
- Merge recheck on 2026-05-25 confirmed there are no pending Claude rows to add: active vault contains **3,360** rows from the patched Claude batch, and the separate **315-row** `standalone_35cell_topup_outputs` dry-run reports **0 accepted / 315 existing / 0 rejected**. Do not re-merge either pack.
- Gap detector check after merge: the active five V8 gap patterns have **105/105 cells at 40/cell** and **0 gap**. The Claude patched pack alone is not full 40/cell coverage: `authority_status_conflict` and `missing_execution_result` contribute **35/cell**, while `resolved_candidate_selection`, `verdict_conflict`, and `version_build_mismatch` contribute **30/cell**. The preexisting V8 rows supply the remaining 5 or 10 rows/cell. Report: `C:/Users/yanfi/PycharmProjects/fitz-gov/data/sdgp_v8_qa/gap_report_20260525_after_claude_patch.json`.
- Whole-dataset target **40/cell is complete**. Across all **483** canonical generation cells, target 40 has **483/483** cells at target and **0** gap. Report: `C:/Users/yanfi/PycharmProjects/fitz-gov/data/sdgp_v8_qa/full_dataset_gap_target40_after_merge.json`.
- Whole-dataset target-40 pack is generated, Codex-subagent blind-label QA clean, and merged. Candidate pack: `C:/Users/yanfi/PycharmProjects/fitz-gov/data/sdgp_handoff_v8_target40/subagent_outputs/` (**174** `batch_*.jsonl` files / **5,198** rows). Final Codex blind score: `C:/Users/yanfi/PycharmProjects/fitz-gov/data/sdgp_qa_v8_target40/score_codex_subagents_combined/` = **5,198/5,198 agreement**, **0 missing / 0 invalid / 0 error**, **0 triage**. Merge batch: `v8_target40_template_20260526`, **5,198 added / 0 duplicate**. Pre-merge backup: `C:/Users/yanfi/PycharmProjects/fitz-gov/data/sdgp_vault_v51_enriched/cases.before_v8_target40_merge_20260526_005837.jsonl`.
- Whole-dataset target **50/cell is complete**. The target-50 pack at `C:/Users/yanfi/PycharmProjects/fitz-gov/data/sdgp_handoff_v8_target50/subagent_outputs/` has **157** `batch_*.jsonl` files / **4,694** rows. Initial Codex-subagent blind QA found **82** triage rows; template repairs fixed `factual_contradiction`, `numerical_conflict`, and `resolved_candidate_selection`. Final Codex blind score: `C:/Users/yanfi/PycharmProjects/fitz-gov/data/sdgp_qa_v8_target50/score_codex_subagents_combined/` = **4,694/4,694 agreement**, **0 missing / 0 invalid / 0 error**, **0 triage**. Merge batch: `v8_target50_template_20260526`, **4,694 added / 0 duplicate**. Pre-merge backup: `C:/Users/yanfi/PycharmProjects/fitz-gov/data/sdgp_vault_v51_enriched/cases.before_v8_target50_merge_20260526_013413.jsonl`. Active target-50 report: `C:/Users/yanfi/PycharmProjects/fitz-gov/data/sdgp_v8_qa/full_dataset_gap_target50_after_merge.json` = **483/483** cells at target, **0** gap.

## Future modality expansion decision

Structured-data and code governance should live under the **same fitz-gov benchmark family**, not separate benchmark repos. Future modality-expanded releases should add a row-level `meta.modality` axis with allowed values `unstructured`, `structured`, and `code`; existing V8 rows are implicitly `unstructured` until a published backfill/migration makes the field explicit. Keep `meta.modality` separate from `routing.expert_fired`: modality describes evidence representation, while route remains the semantic/domain expert target. Current local fitz-gov seeds are only probes: `data/modality_probes/structured/cases.jsonl` and `data/modality_probes/code/cases.jsonl`, 10 rows each, not active-vault training data.

## Known limitations

### `pyrrho-nano-g1` (in production)

1. **Multi-source-convergence misclassified as DISPUTED.** When multiple authoritative sources agree on a fact with slight numerical variation, ~57% error rate on this fitz-gov subcategory (n=7). Deferred to v2.
2. **Short clean TRUSTWORTHY contexts trigger over-abstention.** Tier1 training is 62.7% hard cases; the model never learned the short-clean-answer pattern. Fixable in v2.

### `pyrrho-nano-g2`

1. **Not the production default yet.** `fitz-sage` still uses `pyrrho-nano-g1`; `g2` is live as the V7 benchmark release but has not been integrated into fitz-sage's production path.
2. **Breakdowns are SDGP-only.** V7.0.1 public rows and regenerated pyrrho processed data no longer expose pre-SDGP report axes (`meta.domain`, `meta.subcategory`, `meta.reasoning_type`, `meta.query_type`, `meta.evidence_pattern`). Use `taxonomy.pattern`, `taxonomy.cell_id`, `routing.expert_fired`/processed `expert`, and `meta.difficulty`/processed `difficulty`.
3. **Weakest held-out expert domain is science/medicine.** Across the 3 seed reports, calibrated held-out test metrics by `expert` put `science_medicine` last at **90.93 ± 0.74% accuracy / 5.99 ± 1.06% false-trustworthy** (n=169 per seed). It is the first V8 candidate for targeted eval-probe and augmentation.
4. **Automotive/ECU test-management OOD probe exposed generic taxonomy gaps.** A 10-case synthetic ECU-test probe with exact query strings absent from `data/processed_v7` scored **7/10, 6/10, 8/10** across seeds 42/1337/7. Manual gold-label audit found **10/10 expected labels defensible**. The misses are now represented as V8 primary taxonomy gaps and expanded across all current domains, not as an automotive-only domain.

### `pyrrho-nano-g3`

1. **Not the production default yet.** `fitz-sage` still uses `pyrrho-nano-g1`; `g3` is live as the V8 benchmark release and is the next encoder candidate to integrate.
2. **Output contract is intentionally narrow.** The published nano encoder emits three-class governance logits only. Normalized product output is derived from logits/probabilities/thresholds; taxonomy/category, route, and scalar fields in reports are benchmark metadata or MoE-only research outputs.
3. **Aggregate metrics are very strong, but deployment still needs field-specific checks.** The held-out V8 result is **97.52 ± 0.43% accuracy / 1.42 ± 0.16% false-trustworthy**, but production integrations should still run domain-specific probes before relying on it in high-cost workflows.

### `pyrrho-nano-g2.1-v8-probe`

1. **Still local-only; not benchmark-contract clean enough to publish as `g2.1` yet.** The run is on a mixed local contract (`data/processed_v8_probe`) that preserves V7 splits and appends V8 rows by manifest. It is the right ablation, but not yet the public release.
2. **Verdict-conflict robustness is still the weak spot.** The recovered ECU PASS/FAIL conflict case (`ecu_04_disputed_dtc_powercycle`) improved only **0/3 -> 1/3** across seeds after V8 retraining. Candidate-selection and wrong-release abstain gaps improved more cleanly than explicit final-verdict contradiction handling.
3. **One seed traded fixes instead of improving cleanly.** Seed 7 stayed **8/10** overall by fixing `ecu_02` and `ecu_04` but regressing `ecu_01` and `ecu_07`. The V8 pack is directionally useful, but not yet a fully stable OOD fix.
4. **Verdict-only densification was not enough.** The 105-row hard `verdict_conflict` patch improved `ecu_04` but pushed adjacent clean cases toward DISPUTED. The later 840-row g2.2 retrain fixed `ecu_04` completely but still traded off adjacent controls, so more data is not automatically better unless the OOD probe improves too.

### `pyrrho-nano-g2.2`

1. **Not a publish candidate yet despite better FT.** It has the best held-out false-trustworthy rate so far (**3.06 ± 0.61%**) and passes gates, but OOD mean is **8.00/10**, below the original V8 probe's **8.33/10**.
2. **OOD tradeoff moved rather than vanished.** g2.2 fixed explicit same-build PASS/FAIL conflict handling (`ecu_04` **3/3**) but lost resolved-candidate and wrong-release controls (`ecu_02` **0/3**, `ecu_07` **0/3**).
3. **Aviation probe is stronger, but not clean.** g2.2 is best on the new aviation maintenance probe (**8.67/10**), but `air_02_trustworthy_superseded_sb_resolved` is still only **1/3** and maps to the same resolved/superseded-candidate boundary rather than a new aviation-specific taxonomy.

### `pyrrho-small-g1` (not shipped)

1. **Fails the false-trustworthy gate (12.13% vs 5.7%).** Plain SFT has no anti-FT pressure. See g1.1 below for the partial fix.
2. **Disputed/abstain recall regressed vs encoder.** The SLM's preference for TRUSTWORTHY pulls cases out of those buckets — same root cause as #1.

### `pyrrho-small-g1.1` (not shipped)

1. **Still fails the false-trustworthy gate (9.31% vs 5.7%).** Class weights + label smoothing closed ~40% of the gap from g1 (12.13 → 9.31), but the encoder-style recipe transplant under-delivers on the SLM because token-level CE diffuses the safety pressure across many assistant-turn tokens. To clear the 5.7% gate without ditching SFT, would need stronger weights (e.g., 5.0/5.0/1.0), stronger smoothing (0.25+), or `ft_penalized_accuracy` checkpoint selection (currently `eval_loss`). Cleaner long-term fix: DPO/GRPO with asymmetric FT-penalized reward per [ROADMAP §8 Phase 3](ROADMAP.md).
2. **Tier0 dropped from 99.44 → 96.67%.** The class-weight pressure makes the model more cautious in general, costing it 2 tier0 cases. Within the dropped-95%-gate budget, but worth noting.

## Pipeline / tooling — what exists now

| Script | Purpose |
|---|---|
| [`scripts/prepare_data.py`](../scripts/prepare_data.py) | fitz-gov HF/Vault → train/eval/test JSONL + HF DatasetDict; current default reads published V8.0.1 (`yafitzdev/fitz-gov`, config `v8`, revision `v8.0.1`); explicit V7 args still reproduce `data/processed_v7`; older V8 probe mode appended local cohorts by manifest |
| [`scripts/prepare_moe_data.py`](../scripts/prepare_moe_data.py) | Published fitz-gov V8 → flattened MoE multitask JSONL with governance labels, route IDs, taxonomy IDs, scalar targets, context features, metadata, and strict required-field audit |
| [`scripts/count_moe_params.py`](../scripts/count_moe_params.py) | Exact pyrrho-MoE total/active parameter accounting from `configs/moe/pyrrho_moe_g3_alpha.yaml` |
| [`scripts/analyze_moe_seed_budget.py`](../scripts/analyze_moe_seed_budget.py) | Compares canonical and Qwen-aligned MoE budget variants; selected `pyrrho_moe_g3_alpha_qwen.yaml` for first upcycling attempt |
| [`scripts/train_moe.py`](../scripts/train_moe.py) | Stage 0-0.7 MoE route prototype trainer/evaluator; supports tiny, pooled route-coupled, token route-coupled, and support-aggregating custom students, top-1 expert selection, route/governance/taxonomy/scalar losses, g3 teacher-logit distillation, oracle-route comparison, and expert traffic reporting |
| [`scripts/eval_moe.py`](../scripts/eval_moe.py) | Standalone Stage 0-0.7 MoE checkpoint evaluator; loads tiny, route-coupled, token route-coupled, or support-aggregating checkpoints and writes eval/test reports without retraining |
| [`scripts/upcycle_dense_to_moe.py`](../scripts/upcycle_dense_to_moe.py) | Dense-to-MoE upcycling path; validates Qwen seed/config compatibility, real-weight FFN compression, sharded seed-pack writing, and seed-pack shape validation |
| [`scripts/smoke_moe_qwen_wrapper.py`](../scripts/smoke_moe_qwen_wrapper.py) | Loads the Qwen3-MoE seed pack with pyrrho governance heads and runs a tiny no-training forward/loss smoke |
| [`scripts/generate_moe_teacher_logits.py`](../scripts/generate_moe_teacher_logits.py) | Generates sidecar governance logits from `models/pyrrho-nano-g3` for MoE distillation without rewriting canonical MoE JSONL |
| [`scripts/analyze_moe_failures.py`](../scripts/analyze_moe_failures.py) | Reloads Stage 0-0.7 checkpoints, emits per-case predictions, per-route/per-taxonomy breakdowns, seed-overlap histograms, and markdown failure reports |
| [`scripts/train_moe_posthoc_verifier.py`](../scripts/train_moe_posthoc_verifier.py) | Trains a separate frozen-output Stage 0.7 verifier/reranker over candidate governance/route/taxonomy/scalar features and selects demotion thresholds on eval |
| [`scripts/package_moe_posthoc_verifier.py`](../scripts/package_moe_posthoc_verifier.py) | Packages Stage 0.7 post-hoc verifier artifacts with manifest, feature schema, checksums, and reload evaluation against frozen base checkpoints |
| [`scripts/train_moe_qwen_heads.py`](../scripts/train_moe_qwen_heads.py) | Stage 1 Qwen3-MoE trainer; freezes trunk by default and trains pyrrho heads, optional internal routers/final dense layers/LoRA, physical expert adapters, semantic-route adapters, and g3 logit distillation |
| [`scripts/train_encoder.py`](../scripts/train_encoder.py) | Single-run encoder fine-tuning, config-driven, writes manifest.json; supports optional held-out `test` and optional `tier0_sanity` |
| [`scripts/train_slm.py`](../scripts/train_slm.py) | Single-run SLM QLoRA fine-tune (TRL SFTTrainer + PEFT) with decode-based eval; auto-uses `WeightedLossSFTTrainer` (per-example class weights + label smoothing) when the config sets `training.class_weights` or `training.label_smoothing` |
| [`scripts/eval_slm.py`](../scripts/eval_slm.py) | Eval-only path for a saved SLM LoRA adapter — re-runs the decode-based eval pass without re-training. Use when the in-script eval was interrupted (e.g., the stdout-buffering hang we hit on g1.1 seed 1337) |
| [`scripts/run_seeds.py`](../scripts/run_seeds.py) | Multi-seed orchestrator (encoder-shaped output), aggregates mean ± std |
| [`scripts/aggregate_slm_seeds.py`](../scripts/aggregate_slm_seeds.py) | Multi-seed aggregator for `train_slm.py` outputs (no threshold calibration) |
| [`scripts/sweep.py`](../scripts/sweep.py) | Hyperparameter sweep (coordinate-descent or grid) |
| [`scripts/eval.py`](../scripts/eval.py) | 5-fold CV runner |
| [`scripts/eval_report.py`](../scripts/eval_report.py) | Full per-breakdown evaluation report on a checkpoint |
| [`scripts/compare_runs.py`](../scripts/compare_runs.py) | Diff two runs (or vs baseline), markdown table out |
| [`scripts/automotive_ood_probe.py`](../scripts/automotive_ood_probe.py) | Recovered 10-case ECU/test-management OOD probe; exact-query leakage check against processed datasets plus calibrated old-vs-new multi-seed checkpoint comparison |
| [`scripts/aviation_ood_probe.py`](../scripts/aviation_ood_probe.py) | 10-case aviation maintenance / airworthiness OOD probe; compares `g2`, `g2.1-v8-probe`, and `g2.2` with exact-query leakage checks |
| [`scripts/inspect_tier0_failures.py`](../scripts/inspect_tier0_failures.py) | Dump misclassified tier0 cases with full context |
| [`scripts/smell_test.py`](../scripts/smell_test.py) | 10-case sanity check (ad-hoc) |
| [`scripts/build_model_card.py`](../scripts/build_model_card.py) | Encoder HF model card builder |
| [`scripts/build_slm_model_card.py`](../scripts/build_slm_model_card.py) | SLM HF model card builder (uses summary.json + LoRA adapter dir) |
| [`scripts/render_public_model_cards.py`](../scripts/render_public_model_cards.py) | Public-facing model card renderer for current pyrrho release dirs; follows [`docs/MODEL_CARD_TEMPLATE.md`](MODEL_CARD_TEMPLATE.md), documents raw vs derived outputs, clarifies nano encoder vs MoE-only outputs, uses `±` metric notation, and avoids internal dataset/pipeline terms |
| [`tests/test_smoke.py`](../tests/test_smoke.py) | pytest version of the smell test for CI regression |

Reproducibility: every artifact-producing script writes `manifest.json` (git/pip/hw/seed/timing) via `pyrrho.manifest.write_manifest`.

Full methodology, release gates, and W&B conventions in [METHODOLOGY.md](METHODOLOGY.md).

## What's live

- **pyrrho-nano-g1 on HF**: https://huggingface.co/yafitzdev/pyrrho-nano-g1 (public, CC BY-NC 4.0, 1.35 GB: safetensors + FP32 ONNX + INT8 ONNX). README-only public-card refresh commit: `29e4eecba2676a0fca03637d1515ab03a6e7379f`. The card frames pyrrho as a RAG governance co-processor / anti-hallucination evidence gate, includes explicit Outputs with normalized JSON, label-oriented Results, uses `±` notation, and states that taxonomy/category tags, route IDs, and scalar diagnostics are not published nano encoder outputs. (Was `pyrrho-modernbert-base-v1` + Apache-2.0 through 2026-05-19; renamed under the new `pyrrho-{tier}-{generation}` scheme.)
- **pyrrho-nano-g2 on HF**: https://huggingface.co/yafitzdev/pyrrho-nano-g2 (public, CC BY-NC 4.0, README-only public-card refresh commit `4b66447636c14155640461a84639bb6ea7ebcd09`). Release has 10 files: `model.safetensors`, FP32 ONNX external-data pair, INT8 ONNX external-data pair, tokenizer/config, README, and `.gitattributes`.
- **pyrrho-nano-g3 on HF**: https://huggingface.co/yafitzdev/pyrrho-nano-g3 (public, CC BY-NC 4.0, README-only public-card refresh commit `f52f4a6a1ff6a008086aa3d1352b560b32e851cb`; original model upload commit `397393718985e7bfa101042e89ecc60103e9c447`). Release has 10 files: `model.safetensors`, FP32 ONNX external-data pair, INT8 ONNX external-data pair, tokenizer/config, README, and `.gitattributes`. Local mirror is `models/pyrrho-nano-g3/` with 9 source files / **1.506 GB**; the Hub adds `.gitattributes`.
- **fitz-gov dataset on HF**: https://huggingface.co/datasets/yafitzdev/fitz-gov (public, CC BY-NC 4.0, **V8.0.1**, default config `v8`, tag `v8.0.1`, data/tag commit `0d01bb999e80e4c6b01027763b054b4aa48d2334`). Default V8 splits are query-grouped and leakage-safe: `train=19,674`, `validation=2,459`, `test=2,459`. Public V8 has **24,592 rows** = V6 **2,980** + V7 **7,520** + V8 **14,092**, one canonical config, no compatibility configs, no public pre-SDGP report axes, and all rows carry `meta.modality: "unstructured"`.
- **fitz-gov V8 local rows**: the active local data access point is `C:/Users/yanfi/PycharmProjects/fitz-gov/data/fitz-gov/cases.jsonl`. There is no separate canonical `v8_manifest.jsonl`; derive V8-only indexes from `cases.jsonl` when needed. Full V8 audit reports **14,092** V8 rows, training-schema completeness is **14,092/14,092**, target-50 coverage is **483/483 cells / 0 gap**, and the stricter all-Claude/Codex full V8 second pass is **14,092/14,092 agreement**, **0 missing / 0 invalid / 0 error**, **0 triage**. Local verification shows **24,592/24,592** active rows as `unstructured` with V8 cohort count **14,092**; HF upload verified by loading revision `v8.0.1` from the Hub.
- **pyrrho V8 local prep**: `scripts/prepare_data.py` now defaults to HF `v8` / `v8.0.1`; existing `data/processed_v8` was built on the same row-equivalent V8 splits with **train=19,674 / eval=2,459 / test=2,459**. `scripts/prepare_moe_data.py --strict` wrote `data/moe_v8` with the same splits, **0** required-field misses, **23** taxonomy patterns, and route IDs for all 8 semantic expert groups; rebuild from `v8.0.1` before any new training prep that needs `meta.modality`.
- **Stopped partial LM Studio V8 blind-label pass**: user explicitly asked to stop the overnight LM Studio run on 2026-05-26 morning. The background worker PID **3712** was killed, LM Studio/server processes were stopped, and the partial predictions file is preserved at `C:/Users/yanfi/PycharmProjects/fitz-gov/data/sdgp_v8_qa/blind_label_predictions_v8_target50_full_lmstudio_qwen36_35b_q5_20260526.jsonl` with **4,164/14,092** rows. Do not treat this as a completed full-cohort QA score. The run is resumable with `scripts/sdgp_run_blind_label.py` because the output file contains case IDs for completed rows.
- **Mixed LM Studio + Claude remainder QA did not pass**: Claude Code labeled the **9,928** rows not covered by the stopped LM Studio pass in `C:/Users/yanfi/PycharmProjects/fitz-gov/data/sdgp_v8_qa/claude_remainder_blind/`. The combined file `C:/Users/yanfi/PycharmProjects/fitz-gov/data/sdgp_v8_qa/blind_label_predictions_lmstudio4164_claude_remainder_combined.jsonl` has **14,092** unique case IDs and scores **13,871/14,092 agreement** with **221** triage rows at `C:/Users/yanfi/PycharmProjects/fitz-gov/data/sdgp_v8_qa/score_lmstudio4164_claude_remainder_combined_20260526/`. All **221** disagreements are from provider `lm_studio`; Claude residual rows had **0** disagreements. Treat this as evidence that the stopped LM Studio partial is not usable as a clean QA component.
- **Stricter full V8 second-pass QA is now clean after 87-row repair**: the all-Claude/Codex full second pass initially found **87** false-trustworthy triage rows in the 4,164-row hard V8-gap replacement subset. Those active vault rows were rewritten to remove ambiguous source-of-record/final-status wording and make the intended conflict or exact-version gap explicit. Repair backup: `C:/Users/yanfi/PycharmProjects/fitz-gov/data/sdgp_vault_v51_enriched/cases.before_v8_second_pass_triage87_repair_20260526_102013.jsonl`; repair batch marker: `v8_second_pass_triage87_repair_20260526`. Narrow repair blind recheck: `C:/Users/yanfi/PycharmProjects/fitz-gov/data/sdgp_v8_qa/score_second_pass_triage87_repair_only_20260526/` = **87/87 agreement**, **0 triage**. Final all-Claude/Codex full score: `C:/Users/yanfi/PycharmProjects/fitz-gov/data/sdgp_v8_qa/score_claude_full_repaired87_combined_20260526/` = **14,092/14,092 agreement**, **0 missing / 0 invalid / 0 error**, **0 triage**. Final combined predictions: `C:/Users/yanfi/PycharmProjects/fitz-gov/data/sdgp_v8_qa/blind_label_predictions_claude_full_repaired87_combined_20260526.jsonl`.
- **V7 10.5k target reached + schema unified + blind-label triage repaired + cross-label query review clean + schema-clean V7.0.1 published:** SDGP cell-targeted generation totals **10,500 rows**: 2,980 V6 + **7,520 V7**. Strict full training-schema audit reports V6 **2,980/2,980** and V7 **7,520/7,520** complete against the rich V6/MoE schema after removing pre-SDGP report axes. Every row has canonical `evaluation`; duplicate legacy/compatibility aliases have been removed. All **2,348 V7 TRUSTWORTHY rows** carry evaluator quality constraints, and **0** V6/V7 TRUSTWORTHY rows are missing them. Gap detector rerun on 2026-05-24: target 20/cell and 25/cell are complete across all **378/378** primary taxonomy cells; target 30/cell is a stretch backlog with **20 / 378** cells at target and **1,575** rows remaining. Fresh reports are under `fitz-gov/data/sdgp_vault_v51_enriched/coverage_report_v7_target{20,25,30}.md`. QA audit artifacts exist under `fitz-gov/data/sdgp_v7_qa/`: query-grouped split assignments with **0 query-group leakage**, duplicate reports, full blind-label resolution ledgers, and cross-label semantic review artifacts. Exact dedup remains clean (**0 duplicate IDs, 0 duplicate exact inputs, 0 duplicate checker hashes**). Repeated raw queries remain by design (**562 exact-query duplicate groups**, **218 cross-label exact-query groups**) because the governed input is `(query, contexts)`; semantic review passed with **0** cross-label pairs sharing the same exact context set, **1** shared-context pair adjudicated valid, and **0** unresolved review pairs. Full LM Studio `qwen3.6-35b-a3b` blind-label coverage is **7,520 / 7,520 V7 rows**: **7,520 validated / 0 triage**. The original **842** triage rows were closed by strict prompt/parser recheck (**362**), provider-assisted repair passes (**389 + 52 + 21**), and manual holdout repair (**18**). `fitz-gov/data/sdgp_v7_qa/training_excluded_triage_case_ids.txt` is empty. **pyrrho-side `g2` encoder training/evaluation has passed.**
- **`pyrrho-nano-g2` local release dir**: `models/pyrrho-nano-g2/` mirrors the HF release. Training summary: `outputs/multi_seed_g2/summary.json` reports held-out test **95.24 ± 0.48% accuracy / 3.48 ± 0.40% false-trustworthy** across seeds 42/1337/7. Config: `configs/encoder/modernbert_base_g2.yaml`. Prepared V7.0.1 data: `data/processed_v7` with train=8,400 / eval=1,050 / test=1,050 / tier0=0 and canonical breakdown columns only.
- **`pyrrho-nano-g3` V8 encoder release**: `configs/encoder/modernbert_base_g3_v8.yaml` trained ModernBERT on the fitz-gov V8 row set via `data/processed_v8` (train=19,674 / eval=2,459 / test=2,459); that same row set is now published as `v8.0.1` with explicit unstructured modality metadata. Training summary at `outputs/multi_seed_g3_v8/summary.json`: held-out test **97.52 ± 0.43% accuracy / 1.42 ± 0.16% false-trustworthy** across seeds 42/1337/7. Per-seed best checkpoints and detailed reports are under `outputs/multi_seed_g3_v8/seed_*/`. Release dir `models/pyrrho-nano-g3/` is built from seed **1337** and live on HF at `yafitzdev/pyrrho-nano-g3`.
- **`pyrrho-nano-g2.1-v8-probe` local experiment**: `configs/encoder/modernbert_base_g2_v8_probe.yaml` retrained ModernBERT on `data/processed_v8_probe`, which preserves the published V7 split contract and appends the 525-row V8 cohort by manifest (`+414 train / +54 eval / +57 test`). Training summary at `outputs/multi_seed_g2_1_v8_probe/summary.json`: mixed held-out test **95.51 ± 0.43% accuracy / 3.56 ± 0.38% false-trustworthy** across seeds 42/1337/7. Recovered automotive ECU OOD comparison artifact: `outputs/automotive_ood_probe/comparison.json`.
- **`pyrrho-nano-g2.1-v8-verdict-patch` failed ablation**: `configs/encoder/modernbert_base_g2_v8_verdict_patch.yaml` is historical and points to the old clean 630-row manifest. Training summary at `outputs/multi_seed_g2_1_v8_verdict_patch/summary.json`: held-out test **94.92 ± 0.41% accuracy / 4.08 ± 0.92% false-trustworthy**. ECU OOD comparison artifact: `outputs/automotive_ood_probe/comparison_v8_verdict_patch.json`. Do not publish; it regressed OOD mean to **7.33/10**.
- **`pyrrho-nano-g2.2` local retrain**: `configs/encoder/modernbert_base_g2_2.yaml` trained ModernBERT on `data/processed_v8_balanced_controls`: published V7 plus local V8 append counts **train +661 / eval +97 / test +82**, producing train=9,061 / eval=1,147 / test=1,132. Training summary at `outputs/multi_seed_g2_2/summary.json`: held-out test **95.49 ± 0.15% accuracy / 3.06 ± 0.61% false-trustworthy**. ECU OOD artifact: `outputs/automotive_ood_probe/comparison_g2_2.json`. Do not publish yet; it improves FT but does not beat the original V8 probe on OOD.
- **`pyrrho-nano-g2.3-v8-claude4200` local prep**: `configs/encoder/modernbert_base_g2_3_v8_claude4200.yaml` and `data/processed_v8_claude4200` are now historical because they target the older 4,200-row V8 vault. Rebuild pyrrho prep from the active 14,092-row V8 target-50 vault before the next 3-seed run.
- **Aviation maintenance OOD probe**: `scripts/aviation_ood_probe.py` scored `g2`, `g2.1-v8-probe`, and `g2.2` on 10 new airworthiness/maintenance cases. Exact-query leakage check is clean (**0/10** matches in V7, V8 probe, and V8 balanced-control processed data). `g2.2` is best at **8.67/10**, but the hardest miss is still a known resolved/superseded-candidate boundary, not a new aviation-only taxonomy gap.
- **pyrrho GitHub repo**: public, redesigned README in the fitz-sage style.
- **pyrrho is in production.** fitz-sage **v0.13.0** (shipped 2026-05-15, PyPI + GitHub) replaced its constraint+sklearn governance cascade with `yafitzdev/pyrrho-nano-g1` — loaded as INT8 ONNX, ~30 ms/decision on CPU, zero LLM calls on the governance path. The same release also swapped fitz-sage's chat-call reranker for `Alibaba-NLP/gte-reranker-modernbert-base` (a separate ONNX cross-encoder — fitz-sage's call, applying pyrrho's pattern). See LOG 2026-05-15.
  - Release: https://github.com/yafitzdev/fitz-sage/releases/tag/v0.13.0
  - PyPI: https://pypi.org/project/fitz-sage/0.13.0/
- **`pyrrho-small-g1` release dir staged locally** at `models/pyrrho-small-g1/` — LoRA adapter (`adapter_model.safetensors` + `adapter_config.json`), tokenizer files, chat template, and a 3-seed model card. **Not on HF** pending a fix for the false-trustworthy gate (see Known limitations above).
- **`pyrrho-small-g1.1` release dir staged locally** at `models/pyrrho-small-g1.1/` — same layout, includes the model card that documents the class-weight + label-smoothing recipe. **Not on HF** — still fails the FT gate (9.31% vs 5.7%), though it's strictly closer to the target than g1.
- **Run `huggingface-cli upload yafitzdev/pyrrho-small-g1.1 models/pyrrho-small-g1.1/`** after `huggingface-cli login` if you want to push either as a research artifact.

## Immediate next actions

The integration milestone is **closed** — pyrrho v1 is shipped and
`pyrrho-small-g1` is the first generative SLM data point in the family.
Everything below is model-quality upside on an already-live baseline.

1. ~~**Phase 0: V5.1 schema enrichment**~~ **COMPLETE (2026-05-20).** All 2,980 V5.1 cases LLM-enriched with V6+ schema fields (query_rewritten, context summaries, governance signals, boundary_proximity, near_miss_reason) via Sonnet subagents + LM Studio local worker. 0 TODO markers remain in vault. See LOG 2026-05-20 evening.

   **Phase 0c: V6 completion** **COMPLETE (2026-05-21).** Added the 4 MoE-training fields: per-chunk `boundary_quality`, per-case `governance.evidence_bias_score`, per-case `input.evidence_chain.{order,reasoning}` (multi-chunk only), and per-case `meta.grounding_targets` (`gold_answer` + per-sentence `attributions`, TRUSTWORTHY only). **All 2,980/2,980 cases complete (100%).** Final case (`t1_qualify_medium_101`, Terravax vaccine query — denied by Sonnet's safety classifier on every attempt) backfilled via LM Studio (qwen3.6-27b local). Re-uploaded to `yafitzdev/fitz-gov` v6.0.0 (16.4 MB, up from 12.9 MB).

2. ~~**V7 schema completion + expansion to 10.5k + QA + publish**~~ **COMPLETE (2026-05-24).** Hugging Face `yafitzdev/fitz-gov` reached **V7.0.1** with default `v7` query-grouped splits: train=8,400 / validation=1,050 / test=1,050. V7.0.1 is the schema-clean public contract used by `pyrrho-nano-g2`: same rows/splits/labels as V7.0.0, no `meta.domain`, `meta.subcategory`, `meta.reasoning_type`, `meta.query_type`, or `meta.evidence_pattern` in public rows, rich training-schema audit clean for V6/V7, full Qwen second-pass coverage **7,520/7,520** with **0 triage**, and cross-label exact-query semantic review **0** unresolved pairs. It is now superseded as the dataset default by V8.0.1, but remains the `g2` model contract.

3. ~~**Phase 3 / `pyrrho-nano-g2` train/package/publish**~~ **COMPLETE (2026-05-24).** `scripts/prepare_data.py` now reads HF V7 and preserves published train/validation/test splits. 3-seed encoder validation passed on held-out test: **95.24 ± 0.48% accuracy / 3.48 ± 0.40% false-trustworthy**. `models/pyrrho-nano-g2/` is staged locally and live at `yafitzdev/pyrrho-nano-g2`.

4. ~~**Rebuild pyrrho V8 prep from published fitz-gov V8.0.0**~~ **COMPLETE (2026-05-26 late morning; defaults updated to V8.0.1 on 2026-05-27).** `scripts/prepare_data.py` now defaults to HF revision `v8.0.1` / config `v8`; the existing local `data/processed_v8` row set has **train=19,674 / eval=2,459 / test=2,459**.

5. ~~**Next encoder run: `pyrrho-nano-g3` / V8 ModernBERT ablation**~~ **COMPLETE (2026-05-26 evening).** `configs/encoder/modernbert_base_g3_v8.yaml` trains ModernBERT on `data/processed_v8`; 3-seed validation passed on held-out V8 test with **97.52 ± 0.43% accuracy / 1.42 ± 0.16% false-trustworthy**.

6. ~~**Package `pyrrho-nano-g3` for release**~~ **COMPLETE (2026-05-26 evening).** `models/pyrrho-nano-g3/` is built from seed 1337, includes safetensors plus FP32/INT8 ONNX external-data pairs, passed packaged smoke (**11/11**) and ruff checks, and is live at `yafitzdev/pyrrho-nano-g3` (HF commit `397393718985e7bfa101042e89ecc60103e9c447`).

7. ~~**MoE Stage 0 tiny route prototype**~~ **COMPLETE (2026-05-26 late morning).** `scripts/train_moe.py` runs end-to-end on `data/moe_v8`. Full 3-epoch prototype: **82.47%** test accuracy, **5.63%** FT, **81.09%** route accuracy, **65.80%** taxonomy accuracy. This is a plumbing/prototype result, not a release candidate.

8. ~~**MoE upcycling decision**~~ **COMPLETE (2026-05-26 late morning; repaired after real-weight inspection).** First upcycling target is `configs/moe/pyrrho_moe_g3_alpha_qwen.yaml`: Qwen tokenizer/vocab preserved, 28 layers, KV=8, `head_dim=128`, 48 experts/layer, FFN dim 1056, 15 scalar heads, **4.083B total / 0.424B active inclusive**.

9. ~~**MoE upcycling inspector**~~ **COMPLETE (2026-05-26 late morning).** `scripts/upcycle_dense_to_moe.py --inspect-only` validates Qwen/config compatibility and writes `outputs/moe/upcycling/qwen_alpha_inspect.json`; `--real-weight-smoke` verifies real Qwen safetensors and writes `outputs/moe/upcycling/qwen_alpha_real_weight_smoke.json`; all direct-copy checks pass and FFN compression is explicitly **3072 -> 1056**.

10. ~~**MoE seed-pack materialization**~~ **COMPLETE (2026-05-26 late morning).** Local artifact: `outputs/moe/upcycling/qwen_alpha_seed_pack/` with 30 shards / 310 tensors / 8.166 GB and a passing meta-model shape validation report.

11. ~~**MoE governance wrapper smoke**~~ **COMPLETE (2026-05-26).** `src/pyrrho/moe/qwen_governance.py` wraps the Qwen3-MoE trunk with governance / route / taxonomy / scalar heads; `scripts/smoke_moe_qwen_wrapper.py` loaded the seed pack on CUDA and produced valid no-training outputs.

12. ~~**MoE Stage 1 heads-only training smoke**~~ **COMPLETE (2026-05-26).** `scripts/train_moe_qwen_heads.py` ran 2 CUDA optimizer steps with the trunk frozen and wrote `outputs/moe/qwen_heads_stage1_smoke/train_report.json`.

13. ~~**MoE Stage 1 bounded Qwen-head sweep**~~ **COMPLETE (2026-05-26 afternoon).** Frozen-trunk Qwen heads train end-to-end and calibration works, but full-eval quality plateaus around **53.5-54.7%** calibrated accuracy at the FT gate; full-data scaling and a first split-LR internal-router probe were worse. Treat this as a negative Stage 1 result for release purposes.

14. ~~**MoE Stage 1 adapter/distillation v2 probes**~~ **COMPLETE (2026-05-26 night).** Added g3 teacher-logit sidecars, governance KL distillation, physical Qwen expert residual adapters, semantic-route pooled adapters, save/load support, and CUDA smokes. Bounded 2,048-row / 512-eval probes were negative: physical expert adapters + distillation scored **50.00%** calibrated accuracy / **4.40%** FT / **24.02%** route; semantic-route adapters + distillation scored **44.34%** / **1.65%** FT / **26.37%** route. These do not clear the **>80%** continuation gate.

15. ~~**MoE Stage 0 route-first distillation diagnostics**~~ **COMPLETE (2026-05-26 night).** Added full g3 teacher-logit sidecars, Stage 0 governance KL distillation, loss-weight CLI overrides, and oracle-route eval. Best route-first Stage 0 diagnostic (`outputs/moe/stage0_route_proto_distill_g3_route15/final_metrics.json`) reached **82.43%** calibrated test accuracy / **5.45%** FT / **82.80%** route. This clears the per-run thresholds locally and shows the route signal is learnable when it controls the active expert path.

16. ~~**MoE Stage 0.5 route-coupled custom student**~~ **COMPLETE (2026-05-26 night).** Added `RouteCoupledMoEForGovernance`, a 53.86M-param hash-token student where the selected semantic route drives every residual expert layer; `train_moe.py` / `eval_moe.py` now load both tiny and route-coupled checkpoints. Three-seed full V8 run (`outputs/moe/stage0_5_route_coupled_g3_3seed/summary.json`) reached **83.91 ± 1.18%** calibrated held-out accuracy / **5.55 ± 0.03%** FT / **82.92 ± 0.35%** route / **67.64 ± 1.23%** taxonomy. This clears the 3-seed continuation bar and shows route-coupled capacity can scale above Stage 0 without using the failed Qwen adapter path.

17. ~~**MoE Stage 0.5 failure reporting**~~ **COMPLETE (2026-05-26 night).** Added `scripts/analyze_moe_failures.py` and generated eval/test reports under `outputs/moe/stage0_5_route_coupled_g3_3seed/failure_analysis_{eval,test}/`. The main design signal is that route prediction is good enough for the next probe, while taxonomy/support-pattern handling is weak: `consistent_chain`, `multi_source_corroboration`, and `quantitative_consensus` are low-accuracy TRUSTWORTHY support patterns, and `factual_contradiction` plus partial-overlap/absence patterns drive FT risk.

18. ~~**MoE Stage 0.6 token route-coupled scaffold**~~ **COMPLETE (2026-05-26 night).** Added `TokenRouteCoupledMoEForGovernance`, `configs/moe/pyrrho_moe_stage0_6_token_route_coupled.yaml`, loader support in `train_moe.py` / `eval_moe.py` / `analyze_moe_failures.py`, and targeted governance sample weights for support-pattern recall plus science/medicine and factual-contradiction FT risk. Bounded CUDA smoke and standalone reload passed at `outputs/moe/stage0_6_token_route_coupled_smoke/`.

19. ~~**MoE Stage 0.6 quality probe / 3-seed stability**~~ **COMPLETE (2026-05-27 morning).** `outputs/moe/stage0_6_token_route_coupled_g3_3seed/summary.json` reached **87.23 ± 1.29%** calibrated held-out accuracy / **2.92 ± 1.06%** FT / **86.06 ± 0.94%** route / **71.97 ± 0.72%** taxonomy. This is the new custom-trunk baseline.

20. ~~**MoE Stage 0.6b-e support-recall recipe sweep**~~ **COMPLETE (2026-05-27 morning).** Added per-pattern support weighting and ran seed-42 probes for 0.6b/0.6c/0.6d/0.6e. Findings: 0.6c/0.6d can improve `multi_source_corroboration` and `quantitative_consensus`, but leak FT; 0.6e guards FT but collapses support recall. Do not scale these recipes.

21. ~~**MoE Stage 0.7 support-aggregation architecture**~~ **COMPLETE (2026-05-27).** Added `SupportAggregatingMoEForGovernance`, query/source dataset tensors, loader/evaluator/failure-analysis support for `model_kind: support_aggregating_token`, and `configs/moe/pyrrho_moe_stage0_7_support_aggregation.yaml`. Four-epoch 3-seed run (`outputs/moe/stage0_7_support_aggregation_g3_3seed/summary.json`) reached **89.49 ± 0.47%** calibrated held-out accuracy / **3.06 ± 0.45%** FT / **82.61 ± 2.50%** route / **75.78 ± 0.21%** taxonomy. It is quality-positive and fixes support-pattern recall, but it is not a clean safety domination of Stage 0.6 because `science_medicine` and `factual_contradiction` FT worsened.

22. ~~**MoE Stage 0.7b-d guarded support aggregation probes**~~ **COMPLETE (2026-05-27).** Added guarded recipe configs and ran seed-42 probes for 0.7b/0.7c/0.7d. 0.7b restored safety best but gave back too much support recall; 0.7c and 0.7d did not dominate. Existing scalar risk and threshold gating also fail to preserve support while restoring safety. Do not scale these recipes.

23. ~~**MoE Stage 0.8 architectural guard scaffold**~~ **COMPLETE / NEGATIVE (2026-05-27).** Added `GuardedSupportAggregatingMoEForGovernance`, a learned TRUSTWORTHY-penalty path on the Stage 0.7 support state. Smoke/reload/tests passed, but seed-42 quality was negative: 4 epochs **85.28% / 5.57% FT**, 3 epochs **85.40% / 2.31% FT**. It is too conservative/unstable and should not be scaled.

24. ~~**MoE Stage 0.9 explicit trust-guard target**~~ **COMPLETE / NEGATIVE (2026-05-27).** Added `TrustGuardedSupportAggregatingMoEForGovernance`, a decoupled binary trust verifier over Stage 0.7 candidate logits/support state, plus `configs/moe/pyrrho_moe_stage0_9_trust_guarded_support_aggregation.yaml`. Smoke/reload/tests passed, but seed-42 quality was negative: 3 epochs **86.50% / 1.24% FT**, 4 epochs **84.75% / 0.59% FT**. It restores safety by suppressing TRUSTWORTHY support recall too aggressively (`multi_source_corroboration` **45.16%** at 3 epochs, **38.71%** at 4 epochs). Do not scale this implementation.

25. ~~**MoE Stage 0.7 frozen-output post-hoc verifier**~~ **COMPLETE / POSITIVE (2026-05-27).** Added `scripts/train_moe_posthoc_verifier.py`, trained HGB verifiers on frozen Stage 0.7 candidate outputs for seeds 42/1337/7, selected verifier thresholds on eval with target FT **2.5%** and max eval accuracy drop **1.5%**, and wrote `outputs/moe/stage0_7_posthoc_verifier_g3_3seed/summary.json`. Held-out test moved from the verifier-script baseline **89.37 ± 0.59% / 2.94 ± 0.36% FT** to guarded **88.97 ± 0.51% / 1.99 ± 0.17% FT**. This is the first positive guard path, but it is a post-hoc reranker over frozen logits and still costs support recall.

26. ~~**MoE Stage 0.7 post-hoc verifier support-retention tune**~~ **COMPLETE / POSITIVE (2026-05-27).** Reran the frozen-output verifier with target eval FT **2.8%** and max eval accuracy drop **1.5%** across seeds 42/1337/7. Artifact: `outputs/moe/stage0_7_posthoc_verifier_g3_3seed_ft028/summary.json`. Held-out test is **89.29 ± 0.69% / 2.37 ± 0.26% FT**, with T recall **80.20 ± 1.64%**. This is the preferred post-hoc guard operating point because it preserves support substantially better than the 2.5% safety-heavy setting.

27. ~~**MoE Stage 0.7 post-hoc verifier minimal-intervention/support-aware check**~~ **COMPLETE (2026-05-27).** Ran target eval FT **3.0%** across seeds 42/1337/7 and wrote `outputs/moe/stage0_7_posthoc_verifier_g3_3seed_ft030/summary.json`: **89.35 ± 0.64% / 2.61 ± 0.36% FT**, T recall **80.85 ± 1.30%**. It preserves support best but is a weaker safety improvement and leaves seed 7 unchanged. Added support/risk metrics plus optional support-aware selection constraints to `scripts/train_moe_posthoc_verifier.py`; a seed-42 support-aware probe showed the 2.8% target cannot keep eval support drop under 3 pts, so target-FT remains the practical knob.

28. ~~**MoE Stage 0.7 post-hoc verifier package/reload**~~ **COMPLETE (2026-05-27).** Added `scripts/package_moe_posthoc_verifier.py` with `create` and `evaluate` subcommands. Preferred 2.8% verifier package: `outputs/moe/stage0_7_posthoc_verifier_g3_ft028_package/` with `manifest.json`, copied per-seed verifier artifacts, feature schema width **120**, and `package_eval_report.json`. Full eval+test reload reproduced packaged per-seed metrics with **0** max absolute delta; held-out test remains **89.29 ± 0.69% / 2.37 ± 0.26% FT**.

29. **Next MoE step if continuing custom trunk:** keep Stage 0.7 as the quality baseline and the packaged Stage 0.7 post-hoc verifier as the current positive safety guard artifact. Do not continue scalar weighting, the simple penalty-head path, or the in-model binary trust-guard target. If continuing the verifier branch, harden the package into an inference-facing reranker API or compare package policies; do not backpropagate it through the Stage 0.7 trunk.

30. **`pyrrho-small-g2`** — after the V8 encoder/MoE scaffold stop point, search current permissive 2026 CPU-runnable SLM bases, update `train_slm.py`/`eval_slm.py` for V8 split shape, then run the SLM baseline with asymmetric safety pressure or DPO/GRPO.

31. **fitz-gov modality expansion planning:** promote the current structured/code seed probes into a formal future fitz-gov release plan. Add row-level `meta.modality`, modality-stratified split/report support, and pyrrho prep filters before training any structured-data or code specialist. Do not silently retrofit the published V8 contract.

## Release gates (the bar any pyrrho model must clear before shipping)

Measured mean across 3 seeds. For datasets with a held-out test split (V7+), checkpoint/threshold selection happens on validation and gates are applied to the held-out test report.

- **Overall accuracy ≥ 78.7%** — matches fitz-sage v0.11 sklearn baseline.
- **False-trustworthy rate ≤ 5.7%** — matches baseline; the production safety axis.

The originally-planned **tier0 95% sanity gate has been dropped** (see LOG 2026-05-14 afternoon). With 60 cases, run-to-run variance is ±3.5 pts and ~5 of the 60 cases have ambiguous gold labels. Tier0 is reported as a diagnostic in every model card, not a gate.

## Things NOT to do (already decided — don't relitigate)

- ❌ Don't propose Qwen 2.5 anything — stale (Nov 2024). Use Qwen 3.5+ family.
- ❌ Don't propose 35B-class MoE bases — violates the universal CPU-runnable constraint.
- ❌ Don't propose Llama-family bases — license is more restrictive than Apache-2.0.
- ~~❌ Don't start fitz-gov v6~~ fitz-gov V6.0.0 shipped 2026-05-20 (V5.1-enriched schema overlay). ~~Do not publish/train on V7 until QA passes~~ V7.0.1 shipped to Hugging Face on 2026-05-24 after blind-label, dedup/leakage, cross-label review, and schema-clean gates passed. Do not add new primary domains in V7; V8.0.1 is now the published target-50 expansion release with the unstructured modality field backfilled.
- ❌ Don't rebrand pyrrho — chosen after going through Doxa/Aegis/Sift/Minos/Themis.
- ❌ Don't generate emojis in code/docs unless explicitly asked.
- ❌ Don't propose running pyrrho as a remote endpoint / hosted service. It runs CPU-side via INT8 ONNX inside fitz-sage — that's the architectural commitment (no embeddings, no vector DB, no per-vendor providers in fitz-sage's hot path).

## Where to look for more

| Need | Where |
|---|---|
| What happened, why, and when | [LOG.md](LOG.md) |
| Full vision + 10-release roadmap | [PROJECT.md](PROJECT.md) |
| End-to-end model-development pipeline | [METHODOLOGY.md](METHODOLOGY.md) |
| RTX 5090 / Blackwell / Windows specifics | [SETUP.md](SETUP.md) |
| Repository overview + quickstart | [../README.md](../README.md) |
| Persistent memory (user prefs, banned models, conventions) | `C:\Users\yanfi\.Codex\projects\C--Users-yanfi-PycharmProjects-pyrrho\memory\` |
