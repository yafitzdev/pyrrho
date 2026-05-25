# HANDOFF — pyrrho project current status

> Fresh-session entry point. Read this first.
> For the *history* of how we got here, see [LOG.md](LOG.md).
> For the full plan and roadmap, see [PROJECT.md](PROJECT.md).

This file is **overwritten** as the state changes. It always reflects only the current state.

---

## What pyrrho is, in 30 seconds

Fine-tuned classification models for RAG governance. Given a (query, retrieved contexts) pair, predicts `ABSTAIN` / `DISPUTED` / `TRUSTWORTHY`. Drop-in replacement for the constraint+sklearn pipeline in [fitz-sage](https://github.com/yafitzdev/fitz-sage). Encoders for CPU production; generative SLMs as a HuggingFace portfolio. Published benchmark contract for new `g2` work is [fitz-gov](https://github.com/yafitzdev/fitz-gov) V7.0.1; `pyrrho-nano-g1` metrics remain from the V5.1 eval hold-out.

The brand name is from Pyrrho of Elis — the Greek philosopher whose school practiced suspension of judgment when evidence was insufficient.

## Current state of the family

| Release | Status |
|---|---|
| `pyrrho-nano-g1` (ModernBERT 149M encoder) | Trained + 3-seed validated. On HuggingFace. **In production — it is the governance backend of fitz-sage v0.13.0.** |
| `pyrrho-small-g1` (Qwen3.5-0.8B + LoRA, V5.1 plain SFT) | Trained + 3-seed validated locally. **Not on HF — fails the FT gate (12.13% vs 5.7% bar).** Release dir staged at `models/pyrrho-small-g1/`. See LOG 2026-05-20 evening. |
| `pyrrho-small-g1.1` (Qwen3.5-0.8B + LoRA, V5.1 SFT with class weights + label smoothing) | Trained + 3-seed validated locally. **Still not on HF — FT improved 12.13 → 9.31% but still misses 5.7% gate by ~3.6 pts.** Release dir staged at `models/pyrrho-small-g1.1/`. See LOG 2026-05-20 night. |
| `pyrrho-nano-g1.1` (ModernBERT V6 retrain) | Attempted locally on V6; **not released**. 3-seed result was 81.54 ± 5.97% accuracy / 5.31 ± 0.21% false-trustworthy, with high variance after toolchain drift. Superseded by direct V7 `g2` work. See LOG 2026-05-21. |
| `pyrrho-nano-g2` (ModernBERT V7 retrain) | **Trained + 3-seed validated on fitz-gov V7.0.1 schema-clean contract (same rows/splits/labels as V7.0.0). On Hugging Face.** Held-out test result: **95.24 ± 0.48% accuracy / 3.48 ± 0.40% false-trustworthy**. Passes gates by a wide margin. |
| `pyrrho-nano-g2.1-v8-probe` (ModernBERT local V8 probe retrain) | **Local-only experimental retrain** on published V7 splits plus the 525-row V8 cohort appended by manifest. Mixed held-out test result: **95.51 ± 0.43% accuracy / 3.56 ± 0.38% false-trustworthy** on 1,107 rows. Automotive ECU OOD probe improved from **7.00/10** to **8.33/10** mean across the same 3 seeds. Not published. |
| `pyrrho-nano-g2.1-v8-verdict-patch` (ModernBERT local ablation) | **Failed local ablation; do not publish.** Added 105 hard `verdict_conflict` rows on top of the 525-row V8 probe. Held-out test still passed gates (**94.92 ± 0.41% / 4.08 ± 0.92% FT**), and `ecu_04` improved **1/3 -> 2/3**, but ECU OOD mean regressed **8.33/10 -> 7.33/10** by over-predicting DISPUTED on nearby TRUSTWORTHY/ABSTAIN controls. |
| `pyrrho-small-g2`, `pyrrho-MoE-g3` (and beyond) | Not started. See [ROADMAP.md](ROADMAP.md). |

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
- Initial V8 taxonomy-gap implementation is in fitz-gov: five new primary patterns (`resolved_candidate_selection`, `verdict_conflict`, `authority_status_conflict`, `version_build_mismatch`, `missing_execution_result`) expanded across 7 current domains x 3 difficulties = 105 new cells / 525 rows at 5 rows/cell. A later 105-row hard `verdict_conflict` patch is QA-clean but not model-quality positive overall. A later 210-row balanced-control attempt is **quarantined** because blind-label QA failed on `version_build_mismatch` controls (the labeler treated explicit "no final row exists" evidence as a TRUSTWORTHY negative answer). Plan file: `C:/Users/yanfi/PycharmProjects/fitz-gov/docs/V8_TAXONOMY_EXPANSION_PLAN.md`.
- Expansion rows are generated, merged, structurally audited, repaired, and blind-label clean locally: vault is **11,025 rows** = 10,500 V6/V7 + **525 V8**. Initial LM Studio `qwen3.6-35b-a3b@q5_k_s` scoring found **502 validated / 23 triage**; all 210 rows in the two affected patterns (`missing_execution_result`, `authority_status_conflict`) were repaired and rerun. Final result is **525/525 validated / 0 triage**, **0 missing / 0 invalid / 0 error**. V8 QA artifacts are under `C:/Users/yanfi/PycharmProjects/fitz-gov/data/sdgp_v8_qa/`.

## Known limitations

### `pyrrho-nano-g1` (in production)

1. **Multi-source-convergence misclassified as DISPUTED.** When multiple authoritative sources agree on a fact with slight numerical variation, ~57% error rate on this fitz-gov subcategory (n=7). Deferred to v2.
2. **Short clean TRUSTWORTHY contexts trigger over-abstention.** Tier1 training is 62.7% hard cases; the model never learned the short-clean-answer pattern. Fixable in v2.

### `pyrrho-nano-g2`

1. **Not the production default yet.** `fitz-sage` still uses `pyrrho-nano-g1`; `g2` is live as the V7 benchmark release but has not been integrated into fitz-sage's production path.
2. **Breakdowns are SDGP-only.** V7.0.1 public rows and regenerated pyrrho processed data no longer expose pre-SDGP report axes (`meta.domain`, `meta.subcategory`, `meta.reasoning_type`, `meta.query_type`, `meta.evidence_pattern`). Use `taxonomy.pattern`, `taxonomy.cell_id`, `routing.expert_fired`/processed `expert`, and `meta.difficulty`/processed `difficulty`.
3. **Weakest held-out expert domain is science/medicine.** Across the 3 seed reports, calibrated held-out test metrics by `expert` put `science_medicine` last at **90.93 ± 0.74% accuracy / 5.99 ± 1.06% false-trustworthy** (n=169 per seed). It is the first V8 candidate for targeted eval-probe and augmentation.
4. **Automotive/ECU test-management OOD probe exposed generic taxonomy gaps.** A 10-case synthetic ECU-test probe with exact query strings absent from `data/processed_v7` scored **7/10, 6/10, 8/10** across seeds 42/1337/7. Manual gold-label audit found **10/10 expected labels defensible**. The misses are now represented as V8 primary taxonomy gaps and expanded across all current domains, not as an automotive-only domain.

### `pyrrho-nano-g2.1-v8-probe`

1. **Still local-only; not benchmark-contract clean enough to publish as `g2.1` yet.** The run is on a mixed local contract (`data/processed_v8_probe`) that preserves V7 splits and appends V8 rows by manifest. It is the right ablation, but not yet the public release.
2. **Verdict-conflict robustness is still the weak spot.** The recovered ECU PASS/FAIL conflict case (`ecu_04_disputed_dtc_powercycle`) improved only **0/3 -> 1/3** across seeds after V8 retraining. Candidate-selection and wrong-release abstain gaps improved more cleanly than explicit final-verdict contradiction handling.
3. **One seed traded fixes instead of improving cleanly.** Seed 7 stayed **8/10** overall by fixing `ecu_02` and `ecu_04` but regressing `ecu_01` and `ecu_07`. The V8 pack is directionally useful, but not yet a fully stable OOD fix.
4. **Verdict-only densification is not enough.** The 105-row hard `verdict_conflict` patch improved `ecu_04` but pushed adjacent clean cases toward DISPUTED. Balanced controls are required, but the first 210-row control attempt is quarantined because `version_build_mismatch` examples were not blind-label clean.

### `pyrrho-small-g1` (not shipped)

1. **Fails the false-trustworthy gate (12.13% vs 5.7%).** Plain SFT has no anti-FT pressure. See g1.1 below for the partial fix.
2. **Disputed/abstain recall regressed vs encoder.** The SLM's preference for TRUSTWORTHY pulls cases out of those buckets — same root cause as #1.

### `pyrrho-small-g1.1` (not shipped)

1. **Still fails the false-trustworthy gate (9.31% vs 5.7%).** Class weights + label smoothing closed ~40% of the gap from g1 (12.13 → 9.31), but the encoder-style recipe transplant under-delivers on the SLM because token-level CE diffuses the safety pressure across many assistant-turn tokens. To clear the 5.7% gate without ditching SFT, would need stronger weights (e.g., 5.0/5.0/1.0), stronger smoothing (0.25+), or `ft_penalized_accuracy` checkpoint selection (currently `eval_loss`). Cleaner long-term fix: DPO/GRPO with asymmetric FT-penalized reward per [ROADMAP §8 Phase 3](ROADMAP.md).
2. **Tier0 dropped from 99.44 → 96.67%.** The class-weight pressure makes the model more cautious in general, costing it 2 tier0 cases. Within the dropped-95%-gate budget, but worth noting.

## Pipeline / tooling — what exists now

| Script | Purpose |
|---|---|
| [`scripts/prepare_data.py`](../scripts/prepare_data.py) | fitz-gov HF/Vault → train/eval/test JSONL + HF DatasetDict; V7 default reads `yafitzdev/fitz-gov` config `v7` revision `v7.0.1`; V8 probe mode preserves the published V7 split contract and appends a local cohort by QA manifest |
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
| [`scripts/inspect_tier0_failures.py`](../scripts/inspect_tier0_failures.py) | Dump misclassified tier0 cases with full context |
| [`scripts/smell_test.py`](../scripts/smell_test.py) | 10-case sanity check (ad-hoc) |
| [`scripts/build_model_card.py`](../scripts/build_model_card.py) | Encoder HF model card builder |
| [`scripts/build_slm_model_card.py`](../scripts/build_slm_model_card.py) | SLM HF model card builder (uses summary.json + LoRA adapter dir) |
| [`tests/test_smoke.py`](../tests/test_smoke.py) | pytest version of the smell test for CI regression |

Reproducibility: every artifact-producing script writes `manifest.json` (git/pip/hw/seed/timing) via `pyrrho.manifest.write_manifest`.

Full methodology, release gates, and W&B conventions in [METHODOLOGY.md](METHODOLOGY.md).

## What's live

- **pyrrho model on HF**: https://huggingface.co/yafitzdev/pyrrho-nano-g1 (public, CC BY-NC 4.0, 1.35 GB: safetensors + FP32 ONNX + INT8 ONNX). Model card aligned to user's trimmed shape (no philosophical aside, no uncalibrated table) — `build_model_card.py` produces this by default. Cross-linked to `yafitzdev/fitz-gov` dataset. (Was `pyrrho-modernbert-base-v1` + Apache-2.0 through 2026-05-19; renamed under the new `pyrrho-{tier}-{generation}` scheme.)
- **pyrrho-nano-g2 on HF**: https://huggingface.co/yafitzdev/pyrrho-nano-g2 (public, CC BY-NC 4.0, HF commit `83453ad96c31250dd4f5d000dfaf8974a1daf42d` for the schema-clean model-card update). Release has 10 files: `model.safetensors`, FP32 ONNX external-data pair, INT8 ONNX external-data pair, tokenizer/config, README, and `.gitattributes`. Model card pins fitz-gov HF revision `v7.0.1` and commit `b74c085c0261369c05dc318bab36c3ae48adc27c`.
- **fitz-gov dataset on HF**: https://huggingface.co/datasets/yafitzdev/fitz-gov (public, CC BY-NC 4.0, **V7.0.1**, default config `v7`, tag `v7.0.1`, commit `b74c085c0261369c05dc318bab36c3ae48adc27c`). Default V7 splits are query-grouped and leakage-safe: `train=8,400`, `validation=1,050`, `test=1,050`. V7.0.1 is a schema-clean republish of V7.0.0 with the same rows/splits/labels and no public pre-SDGP report axes; HF config list is now only `v7`. V6.0.0 was the 2,980-row V5.1 schema overlay baseline; V7 is the 10,500-row SDGP-scaled training/eval contract.
- **fitz-gov V8 local rows**: the active local vault now has **630 V8 rows** at `C:/Users/yanfi/PycharmProjects/fitz-gov/data/sdgp_vault_v51_enriched/cases.jsonl` (**11,130 total rows** including V6/V7). The clean training manifest is `C:/Users/yanfi/PycharmProjects/fitz-gov/data/sdgp_v8_qa/blind_label_manifest_clean_630.jsonl`: original 525 QA-clean V8 rows + 105 QA-clean hard `verdict_conflict` rows. The later 210 balanced-control rows are **not in the active vault**; their quarantined IDs are in `C:/Users/yanfi/PycharmProjects/fitz-gov/data/sdgp_v8_qa/quarantined_balanced_control_case_ids.txt`. Repaired offline control outputs exist under `C:/Users/yanfi/PycharmProjects/fitz-gov/data/sdgp_handoff_v8_balanced_controls/subagent_outputs_repaired`, but they are still unvalidated for training use: the first repaired blind-label pass scored **77/82 parsed agreements** with **128 invalid** at `max_tokens=128`, and the follow-up `max_tokens=2048` retry was intentionally interrupted after writing **107/128** rows to `C:/Users/yanfi/PycharmProjects/fitz-gov/data/sdgp_v8_qa/blind_label_predictions_balanced_controls_invalid_retry_qwen36_35b_q5_max2048.jsonl`. Do not merge or train from repaired balanced-control artifacts in this state.
- **V7 10.5k target reached + schema unified + blind-label triage repaired + cross-label query review clean + schema-clean V7.0.1 published:** SDGP cell-targeted generation totals **10,500 rows**: 2,980 V6 + **7,520 V7**. Strict full training-schema audit reports V6 **2,980/2,980** and V7 **7,520/7,520** complete against the rich V6/MoE schema after removing pre-SDGP report axes. Every row has canonical `evaluation`; duplicate legacy/compatibility aliases have been removed. All **2,348 V7 TRUSTWORTHY rows** carry evaluator quality constraints, and **0** V6/V7 TRUSTWORTHY rows are missing them. Gap detector rerun on 2026-05-24: target 20/cell and 25/cell are complete across all **378/378** primary taxonomy cells; target 30/cell is a stretch backlog with **20 / 378** cells at target and **1,575** rows remaining. Fresh reports are under `fitz-gov/data/sdgp_vault_v51_enriched/coverage_report_v7_target{20,25,30}.md`. QA audit artifacts exist under `fitz-gov/data/sdgp_v7_qa/`: query-grouped split assignments with **0 query-group leakage**, duplicate reports, full blind-label resolution ledgers, and cross-label semantic review artifacts. Exact dedup remains clean (**0 duplicate IDs, 0 duplicate exact inputs, 0 duplicate checker hashes**). Repeated raw queries remain by design (**562 exact-query duplicate groups**, **218 cross-label exact-query groups**) because the governed input is `(query, contexts)`; semantic review passed with **0** cross-label pairs sharing the same exact context set, **1** shared-context pair adjudicated valid, and **0** unresolved review pairs. Full LM Studio `qwen3.6-35b-a3b` blind-label coverage is **7,520 / 7,520 V7 rows**: **7,520 validated / 0 triage**. The original **842** triage rows were closed by strict prompt/parser recheck (**362**), provider-assisted repair passes (**389 + 52 + 21**), and manual holdout repair (**18**). `fitz-gov/data/sdgp_v7_qa/training_excluded_triage_case_ids.txt` is empty. **pyrrho-side `g2` encoder training/evaluation has passed.**
- **`pyrrho-nano-g2` local release dir**: `models/pyrrho-nano-g2/` mirrors the HF release. Training summary: `outputs/multi_seed_g2/summary.json` reports held-out test **95.24 ± 0.48% accuracy / 3.48 ± 0.40% false-trustworthy** across seeds 42/1337/7. Config: `configs/encoder/modernbert_base_g2.yaml`. Prepared V7.0.1 data: `data/processed_v7` with train=8,400 / eval=1,050 / test=1,050 / tier0=0 and canonical breakdown columns only.
- **`pyrrho-nano-g2.1-v8-probe` local experiment**: `configs/encoder/modernbert_base_g2_v8_probe.yaml` retrained ModernBERT on `data/processed_v8_probe`, which preserves the published V7 split contract and appends the 525-row V8 cohort by manifest (`+414 train / +54 eval / +57 test`). Training summary at `outputs/multi_seed_g2_1_v8_probe/summary.json`: mixed held-out test **95.51 ± 0.43% accuracy / 3.56 ± 0.38% false-trustworthy** across seeds 42/1337/7. Recovered automotive ECU OOD comparison artifact: `outputs/automotive_ood_probe/comparison.json`.
- **`pyrrho-nano-g2.1-v8-verdict-patch` failed ablation**: `configs/encoder/modernbert_base_g2_v8_verdict_patch.yaml` now points to the clean 630-row V8 manifest, matching the already-prepared `data/processed_v8_verdict_patch`. Training summary at `outputs/multi_seed_g2_1_v8_verdict_patch/summary.json`: held-out test **94.92 ± 0.41% accuracy / 4.08 ± 0.92% false-trustworthy**. ECU OOD comparison artifact: `outputs/automotive_ood_probe/comparison_v8_verdict_patch.json`. Do not publish; it regressed OOD mean to **7.33/10**.
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

2. ~~**V7 schema completion + expansion to 10.5k + QA + publish**~~ **COMPLETE (2026-05-24).** Hugging Face `yafitzdev/fitz-gov` is now **V7.0.1** with default `v7` query-grouped splits: train=8,400 / validation=1,050 / test=1,050. V7.0.1 is the schema-clean public contract: same rows/splits/labels as V7.0.0, no `meta.domain`, `meta.subcategory`, `meta.reasoning_type`, `meta.query_type`, or `meta.evidence_pattern` in public rows, and HF config list is only `v7`. Rich training-schema audit is clean for V6 and V7, every row has canonical `evaluation`, full Qwen second-pass coverage is **7,520/7,520** V7 rows with **7,520 validated / 0 triage**, and cross-label exact-query semantic review has **0** unresolved pairs. A separate 5,000-row expansion can resume after `g2` baselines if wanted.

3. ~~**Phase 3 / `pyrrho-nano-g2` train/package/publish**~~ **COMPLETE (2026-05-24).** `scripts/prepare_data.py` now reads HF V7 and preserves published train/validation/test splits. 3-seed encoder validation passed on held-out test: **95.24 ± 0.48% accuracy / 3.48 ± 0.40% false-trustworthy**. `models/pyrrho-nano-g2/` is staged locally and live at `yafitzdev/pyrrho-nano-g2`.

4. **V8 stop point** — keep the active **630-row** clean manifest as the only safe local V8 training input. Do **not** promote the verdict-patch ablation. The best current V8 candidate remains the original 525-row probe: **95.51 ± 0.43% / 3.56 ± 0.38% FT** and ECU OOD **8.33/10**. The repaired balanced-control attempt is still offline-only and incomplete; if this work is ever resumed, delete/rebuild the partial retry artifacts and rerun repaired-control blind-label QA from scratch before any merge or retrain.

5. **`pyrrho-small-g2`** — after the V8 stop point, or if staying on V7 permanently, search current permissive 2026 CPU-runnable SLM bases, update `train_slm.py`/`eval_slm.py` for V7/V8 split shape, then run the SLM baseline with asymmetric safety pressure or DPO/GRPO.

6. ~~**fitz-gov v6**~~ **DONE (2026-05-20).** V5.1-enriched = V6 (executive call). Uploaded to `yafitzdev/fitz-gov` as V6.0.0. See LOG 2026-05-20 evening.

## Release gates (the bar any pyrrho model must clear before shipping)

Measured mean across 3 seeds. For datasets with a held-out test split (V7+), checkpoint/threshold selection happens on validation and gates are applied to the held-out test report.

- **Overall accuracy ≥ 78.7%** — matches fitz-sage v0.11 sklearn baseline.
- **False-trustworthy rate ≤ 5.7%** — matches baseline; the production safety axis.

The originally-planned **tier0 95% sanity gate has been dropped** (see LOG 2026-05-14 afternoon). With 60 cases, run-to-run variance is ±3.5 pts and ~5 of the 60 cases have ambiguous gold labels. Tier0 is reported as a diagnostic in every model card, not a gate.

## Things NOT to do (already decided — don't relitigate)

- ❌ Don't propose Qwen 2.5 anything — stale (Nov 2024). Use Qwen 3.5+ family.
- ❌ Don't propose 35B-class MoE bases — violates the universal CPU-runnable constraint.
- ❌ Don't propose Llama-family bases — license is more restrictive than Apache-2.0.
- ~~❌ Don't start fitz-gov v6~~ fitz-gov V6.0.0 shipped 2026-05-20 (V5.1-enriched schema overlay). ~~Do not publish/train on V7 until QA passes~~ V7.0.1 shipped to Hugging Face on 2026-05-24 after blind-label, dedup/leakage, cross-label review, and schema-clean gates passed. Do not add new primary domains in V7; V8 is the domain-focused expansion release.
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
