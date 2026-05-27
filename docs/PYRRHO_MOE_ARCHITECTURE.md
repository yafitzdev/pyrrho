# pyrrho-MoE Architecture Spec

Status: draft baseline for alignment, 2026-05-25.

This document defines the target architecture for `pyrrho-MoE-g3`. It supersedes any older language that treats an off-the-shelf MoE checkpoint as the final pyrrho-MoE model. LFM2/Qwen/other models may be teachers, baselines, or proxy experiments, but the deployed pyrrho-MoE target is custom.

## 1. Goal

Build a CPU-runnable sparse MoE governance model with:

- Total parameters: 4B class, hard target window 3.9B-4.1B.
- Active parameters: A0.4B class, hard target window 0.38B-0.43B under the project counting convention.
- Routing: supervised by fitz-gov `routing.expert_fired`.
- Experts: pyrrho-defined semantic expert groups, not generic opaque experts.
- Training: general language competence transferred from pretrained teacher/seed model(s); governance behavior trained on fitz-gov.
- Deployment: 4-bit quantized CPU target, GGUF/llama.cpp compatibility preferred; custom CPU runtime acceptable only if GGUF support blocks the architecture.

This is feasible only as a dense-to-MoE upcycling + distillation project. It is not feasible as random-initialized general LLM pretraining on fitz-gov alone.

## 2. Baseline Architecture: `pyrrho-moe-g3-alpha`

The baseline is a decoder-only Transformer with sparse SwiGLU FFN experts.

| Field | Value |
|---|---:|
| Layers | 24 |
| Hidden size | 1024 |
| Attention heads | 16 |
| KV heads | 4 |
| Head dim | 64 |
| Attention type | GQA |
| Position encoding | RoPE |
| Norm | RMSNorm, pre-norm |
| FFN activation | SwiGLU |
| Expert FFN dim | 3840 |
| Dense FFN layers | 4 |
| MoE FFN layers | 20 |
| Experts per MoE layer | 16 physical experts |
| Routing | top-1 for CPU path |
| Tokenizer assumption | 64k vocab, tied input/output embeddings |
| Context target | 4096 minimum, 8192 stretch |

Layer layout:

```text
layers 00-01: dense FFN warm-up
layers 02-21: MoE FFN, top-1 expert
layers 22-23: dense FFN stabilization
```

Top-2 routing is not part of the CPU release path because it breaks the A0.4B active budget. It may be used only as an offline ablation.

## 3. Parameter Budget

Assumptions:

- Vocab size: 64,000.
- Embedding and LM head are tied.
- Attention uses GQA: Q and O project to 1024, K/V project to 256.
- Each SwiGLU FFN/expert has `3 * hidden_size * ffn_dim` parameters.

| Component | Formula | Params |
|---|---:|---:|
| Tied embedding / LM head | `64,000 * 1024` | 65.5M |
| Attention blocks | `24 * ((1024*1024) + 2*(1024*256) + (1024*1024))` | 62.9M |
| Dense FFNs | `4 * (3*1024*3840)` | 47.2M |
| MoE expert bank | `20 * 16 * (3*1024*3840)` | 3774.9M |
| Routers | `20 * 1024 * 16` | 0.3M |
| Norms + task heads + small adapters | allowance | 1M-15M |
| **Total** | | **~3.95B-3.97B** |

Active parameter budget with top-1 routing:

| Active component | Params |
|---|---:|
| Selected experts, one per MoE layer | 235.9M |
| Dense FFNs | 47.2M |
| Attention blocks | 62.9M |
| Tied embedding / LM head, full resident-matrix convention | 65.5M |
| Routers + heads | ~1M-5M |
| **Active total, inclusive convention** | **~0.412B-0.416B** |
| **Active total, excluding full embedding matrix** | **~0.346B-0.351B** |

Release naming should use `4B-A0.4B`. Manifests must report exact total and active counts under both conventions.

## 4. Expert Layout

The 16 physical experts are grouped into 8 semantic expert groups. Each semantic group owns 2 physical shards per MoE layer.

| Semantic expert group | Physical shards | Supervised target source |
|---|---:|---|
| `science_medicine` | 2 | `routing.expert_fired` |
| `law_policy` | 2 | `routing.expert_fired` |
| `history_geography` | 2 | `routing.expert_fired` |
| `technology_computing` | 2 | `routing.expert_fired` |
| `economics_finance` | 2 | `routing.expert_fired` |
| `culture_society` | 2 | `routing.expert_fired` |
| `general_commonsense` | 2 | `routing.expert_fired` |
| `conflict_detection` | 2 | `routing.expert_fired` plus conflict-density supervision |

The phrase "7-8 experts" in older docs means semantic expert groups. The physical MoE needs more shards to hit the 4B total / A0.4B active ratio.

Conflict handling:

- The CPU release path still activates only one physical expert per MoE layer.
- `conflict_detection` is a routeable semantic group, not a mandatory second expert.
- A separate conflict scalar head can fire alongside any route and is used for output signals.
- Optional secondary conflict routing is an offline ablation only unless active params stay inside budget.

## 5. Construction Method

Preferred construction is dense-to-MoE upcycling.

1. Select a permissive pretrained dense SLM seed with architecture close to the baseline:
   - hidden size near 1024,
   - 20-28 layers,
   - SwiGLU FFNs,
   - GQA preferred,
   - tokenizer and license compatible with pyrrho publication.
2. Copy seed embeddings, attention blocks, norms, and dense layers.
3. Convert selected FFN layers into MoE layers by compressing the seed FFN if needed, then cloning it into each physical expert.
4. Add small per-expert perturbations or adapters so experts can specialize without losing seed competence.
5. Initialize routers from fitz-gov domain priors, then train routers with supervised losses.
6. Distill from stronger teacher model(s) to preserve instruction following and evidence-sensitive reasoning.

If no compatible dense seed exists, the fallback is direct student distillation into the same architecture. That remains possible but is substantially higher risk and should not be the first path.

2026-05-26 upcycling decision: the first seed-aligned alpha uses
`Qwen/Qwen3-0.6B-Base` and keeps Qwen's tokenizer/vocab rather than forcing the
old 64k-tokenizer assumption. The selected config is
`configs/moe/pyrrho_moe_g3_alpha_qwen.yaml`: 28 layers, hidden 1024, KV=8,
explicit `head_dim=128`, 24 MoE layers, 48 physical experts/layer, FFN dim 1056,
and 151,936 tied vocab. Exact count with 15 V8 scalar heads: **4.083139633B total
/ 0.423871537B active inclusive**. This preserves the CPU budget but requires FFN compression from
Qwen's 3072-wide FFNs into 1056-wide experts; attention, embeddings, Q/K
attention norms, and final norm can be copied directly. Initial FFN compression
utility landed in `src/pyrrho/moe/upcycling.py`: score channels by combined
gate/up/down norm, select the strongest 1056 channels, and slice all three
projection matrices consistently.

## 6. Training Data Contract

fitz-gov provides the governance and routing supervision. A row is MoE-ready only if it has:

- `governance.classification` or processed class label: ABSTAIN / DISPUTED / TRUSTWORTHY.
- `routing.expert_fired`: semantic route target.
- `taxonomy.pattern`: specialist behavior target.
- `meta.difficulty`: difficulty conditioning and reporting axis.
- Query plus retrieved contexts in the current SDGP shape.
- Split assignment with query-group leakage protection.

Recommended before the first serious g3 run:

- Use published fitz-gov V8.0.0 (`yafitzdev/fitz-gov`, config `v8`, revision `v8.0.0`). The former target-40/target-50 gap is closed; public splits are train=19,674 / validation=2,459 / test=2,459.
- Generate teacher traces for all train rows and a locked subset of validation rows.
- Keep gold labels authoritative. Teacher traces are auxiliary targets, not replacements for fitz-gov labels.

Teacher trace fields:

- concise rationale,
- predicted governance label,
- route explanation,
- uncertainty explanation,
- evidence summary,
- optional query rewrite,
- optional per-context relevance/authority notes.

## 7. Losses

Core losses:

- Governance classification CE, class-weighted against false TRUSTWORTHY.
- Router semantic CE against `routing.expert_fired`.
- Physical expert load-balancing loss inside each semantic group.
- Taxonomy pattern CE against `taxonomy.pattern`.
- Scalar regression losses for available governance signals.
- Distillation KL from teacher logits where available.
- Sequence NLL on short teacher rationales/traces where used.

Safety losses:

- False-trustworthy weighted penalty.
- DPO/GRPO preference stage with heavy penalty for confident wrong TRUSTWORTHY.
- Calibration loss or evidential head loss for uncertainty quality.

Router stability losses:

- Auxiliary load balance.
- Router z-loss.
- Entropy floor early in training, annealed down after routes specialize.
- Per-expert minimum traffic constraint during early epochs.

## 8. Training Stages

### Stage 0: tiny route prototype

Build a smaller model with the same code path, roughly 100M-300M total. Goal is not quality. Goal is proving:

- data loader,
- router supervision,
- expert grouping,
- loss plumbing,
- eval reports,
- no expert collapse.

Exit gate:

- training runs end-to-end,
- router learns above majority baseline,
- all semantic routes receive traffic,
- smoke eval does not regress obvious cases.

Status 2026-05-26: Stage 0 plumbing is implemented and passes the intended
prototype gate for the seven primary route groups present in V8. `scripts/train_moe.py`
trained a 10,505,009-param hashed-token top-1 MoE on `data/moe_v8` in ~38s.
Held-out test: 82.47% governance accuracy, 5.63% false-trustworthy, 81.09%
route accuracy, and 65.80% taxonomy accuracy. `conflict_detection` has no gold
primary-route rows in V8, so it cannot satisfy "all semantic routes receive
traffic" as a primary route until data or routing targets expose it directly.

Stage 0 route-first diagnostics also landed on 2026-05-26. Full
`pyrrho-nano-g3` teacher-logit sidecars were generated at
`outputs/moe/teacher_logits/pyrrho_nano_g3_full_v8/`, and Stage 0 now supports
governance KL distillation, CLI loss-weight overrides, and oracle gold-route
eval. The best diagnostic run so far is
`outputs/moe/stage0_route_proto_distill_g3_route15/final_metrics.json`:
82.43% calibrated test accuracy, 5.45% false-trustworthy, 82.80% route
accuracy, and 64.99% taxonomy accuracy with `loss_route=1.5` and
`loss_distillation=0.5`. This confirms the V8 route signal is learnable when
the supervised semantic route controls the active expert path.

### Stage 0.5: route-coupled custom student

Stage 0.5 is the bridge between the tiny proof and the terminal 4B skeleton.
It deliberately avoids the failed Qwen adapter path. The selected semantic
route is the active expert path through every residual expert layer, so a
successful result is evidence for route-coupled custom scaling rather than
opaque-router adaptation.

Status 2026-05-26 night: `src/pyrrho/moe/modeling.py` now includes
`RouteCoupledMoEForGovernance`, and
`configs/moe/pyrrho_moe_stage0_5_route_coupled.yaml` defines a 53,861,425-param
hash-token student with hidden size 384, expert hidden size 768, four
route-coupled residual expert layers, governance KL from `pyrrho-nano-g3`, and
lighter false-trustworthy pressure than the tiny route-first diagnostic. The
three-seed full V8 run at
`outputs/moe/stage0_5_route_coupled_g3_3seed/summary.json` reached 83.91 +/-
1.18% calibrated test accuracy, 5.55 +/- 0.03% false-trustworthy, 82.92 +/-
0.35% route accuracy, and 67.64 +/- 1.23% taxonomy accuracy. Per-seed
calibrated test accuracy / false-trustworthy: seed 42 84.47% / 5.51%, seed
1337 84.71% / 5.57%, seed 7 82.55% / 5.57%. Gold-route mean was 84.64 +/- 1.05%
calibrated accuracy / 5.29 +/- 0.42% false-trustworthy, so the remaining gap is
not primarily route prediction. A route-heavy/high-FT-weight variant
(`stage0_5_route_coupled_g3_route15`) was too conservative at 77.51% calibrated
accuracy / 0.77% false-trustworthy.

Stage 0.5 failure reporting also landed on 2026-05-26. `scripts/analyze_moe_failures.py`
reloads the 3-seed checkpoints, applies the saved calibrated TRUSTWORTHY
thresholds, writes per-case predictions, and reports route/taxonomy/seed-overlap
breakdowns. Test report:
`outputs/moe/stage0_5_route_coupled_g3_3seed/failure_analysis_test/failure_report.md`.
Eval report:
`outputs/moe/stage0_5_route_coupled_g3_3seed/failure_analysis_eval/failure_report.md`.
The test split has 109 all-seed hard errors (4.43%), 12 all-seed
false-TRUSTWORTHY cases, and 202 rows with a false-TRUSTWORTHY prediction in at
least one seed. Weak test routes are `science_medicine` (78.80% accuracy /
12.38% FT), `technology_computing`, and `general_commonsense`. Weak taxonomy
groups are support-pattern TRUSTWORTHY rows (`consistent_chain` 66.43%,
`multi_source_corroboration` 67.38%, `quantitative_consensus` 71.11%) plus
`factual_contradiction` safety risk (77.88% accuracy / 12.98% FT). Eval repeats
the core pattern: 106 all-seed hard errors and `consistent_chain` still weakest
at 62.99%.

### Stage 0.6: token route-coupled custom trunk

Stage 0.6 keeps the positive Stage 0.5 property that supervised semantic route
is the actual expert execution path, but removes the pooled-only bottleneck.
The probe is still hash-token based and not a release candidate; its purpose is
to test whether real token interaction closes the support-pattern and
science/medicine risk gaps before returning to a 4B skeleton.

Status 2026-05-26 night: `TokenRouteCoupledMoEForGovernance` landed in
`src/pyrrho/moe/modeling.py`, with config
`configs/moe/pyrrho_moe_stage0_6_token_route_coupled.yaml`. The model has
55,728,817 parameters and uses RoPE self-attention, RMSNorm pre-norms,
route-selected SwiGLU FFNs in every block, and last-token/mean pooled heads.
`scripts/train_moe.py`, `scripts/eval_moe.py`, and
`scripts/analyze_moe_failures.py` now load `model_kind: route_coupled_token`.
The config adds governance sample weights for the known weak slices:
TRUSTWORTHY support patterns (`consistent_chain`,
`multi_source_corroboration`, `quantitative_consensus`) and non-TRUSTWORTHY
FT-risk rows in `science_medicine` and `factual_contradiction`.

Bounded CUDA smoke artifact:
`outputs/moe/stage0_6_token_route_coupled_smoke/`. The smoke used 8 rows per
split plus full `pyrrho-nano-g3` teacher-logit sidecars, trained for one epoch,
saved `model.pt`, and reloaded through `scripts/eval_moe.py`.

Full 3-seed result landed on 2026-05-27 at
`outputs/moe/stage0_6_token_route_coupled_g3_3seed/summary.json`: held-out test
87.23 +/- 1.29% calibrated accuracy, 2.92 +/- 1.06% false-trustworthy, 86.06
+/- 0.94% route accuracy, and 71.97 +/- 0.72% taxonomy accuracy. This is a clear
headline improvement over Stage 0.5 (83.91 +/- 1.18% / 5.55 +/- 0.03% FT /
82.92 +/- 0.35% route / 67.64 +/- 1.23% taxonomy) and makes Stage 0.6 the
current custom-trunk baseline.

Failure reports:
`outputs/moe/stage0_6_token_route_coupled_g3_3seed/failure_analysis_test/failure_report.md`
and
`outputs/moe/stage0_6_token_route_coupled_g3_3seed/failure_analysis_eval/failure_report.md`.
The intended safety targets improved: test `science_medicine` moved from
78.80% accuracy / 12.38% FT in Stage 0.5 to 82.22% / 4.08% FT in Stage 0.6, and
`factual_contradiction` moved from 77.88% / 12.98% FT to 89.68% / 5.31% FT.
The support-pattern target was still unresolved: `consistent_chain` improved only
66.43% -> 69.27%, while `multi_source_corroboration` regressed 67.38% -> 59.50%
and `quantitative_consensus` regressed 71.11% -> 65.40%. This motivated the
0.6b-e recipe sweep and then the Stage 0.7 support-aggregation path below.

Stage 0.6b-e support-recall recipe sweep landed on 2026-05-27. Per-pattern
support weights are supported through
`stage0.governance_sample_weights.support_taxonomy_pattern_weights`, and the
seed-42 recipe artifacts are under:
`outputs/moe/stage0_6b_support_recall_g3_seed42/`,
`outputs/moe/stage0_6c_pattern_weighted_g3_seed42/`,
`outputs/moe/stage0_6d_balanced_pattern_g3_seed42/`, and
`outputs/moe/stage0_6e_guarded_pattern_g3_seed42/`.

The result was informative but negative for simple recipe scaling. 0.6c/0.6d
show that weighting can recover `multi_source_corroboration` and
`quantitative_consensus`, but they leak false-TRUSTWORTHY on absence,
partial-overlap, and contradiction families. 0.6e guards FT strongly (seed-42
test 87.60% calibrated accuracy / 1.66% FT), but collapses support recall
(`consistent_chain` 58.16%, `multi_source_corroboration` 56.99%,
`quantitative_consensus` 62.86%). Do not scale 0.6b/0.6c/0.6d/0.6e. The next
move should be architectural: a Stage 0.7 support-aggregation path or
support-pattern auxiliary head that helps multi-source TRUSTWORTHY recognition
without globally increasing TRUSTWORTHY bias.

### Stage 0.7: support aggregation

Stage 0.7 keeps Stage 0.6's flat token route-coupled trunk, but adds an
explicit query/source interface and terminal support aggregation path:

- `MoEJsonlDataset` now returns hashed `query_input_ids`, per-source
  `source_input_ids`, source attention masks, and source-valid masks from the
  canonical V8 `query` and `contexts` fields.
- `SupportAggregatingMoEForGovernance` pools the query and each source with the
  shared hash embedding, scores query-source alignment, builds a weighted
  support state plus source max-pool, and fuses that state into the governance,
  taxonomy, and scalar heads.
- The supervised semantic route still controls the token-trunk expert path from
  the start; support aggregation changes the terminal evidence state, not the
  route/expert contract.

Config: `configs/moe/pyrrho_moe_stage0_7_support_aggregation.yaml`. The
validated recipe uses **4 epochs**. The same recipe at 5 epochs overfit/shifted
unsafe on seed 42: final held-out test fell to 86.95% calibrated accuracy /
3.79% FT after eval peaked at epoch 4.

Three-seed artifact:
`outputs/moe/stage0_7_support_aggregation_g3_3seed/summary.json`.
Held-out test result: **89.49 +/- 0.47%** calibrated accuracy, **3.06 +/-
0.45%** false-trustworthy, **82.61 +/- 2.50%** route accuracy, and **75.78 +/-
0.21%** taxonomy accuracy. Per-seed calibrated accuracy / FT: seed 42
**90.04% / 3.26%**, seed 1337 **89.18% / 2.55%**, seed 7 **89.26% / 3.38%**.
Gold-route mean was **89.60 +/- 0.55%** calibrated accuracy / **3.14 +/- 0.52%**
FT, so the remaining gap is not route prediction.

Failure reports:
`outputs/moe/stage0_7_support_aggregation_g3_3seed/failure_analysis_test/failure_report.md`
and
`outputs/moe/stage0_7_support_aggregation_g3_3seed/failure_analysis_eval/failure_report.md`.
Compared with Stage 0.6, the support target moved strongly:
`consistent_chain` **69.27% -> 75.18%**, `multi_source_corroboration`
**59.50% -> 69.53%**, and `quantitative_consensus` **65.40% -> 79.05%**.
The caveat is safety: `science_medicine` improved in accuracy
**82.22% -> 85.21%** but FT worsened **4.08% -> 5.58%**; `factual_contradiction`
stayed near the Stage 0.6 accuracy level but FT worsened **5.31% -> 6.19%**.
Stage 0.7 is therefore the current quality baseline, while Stage 0.6 remains
the safety reference. The next probe should preserve support aggregation and
restore the Stage 0.6 FT profile on science/medicine, factual contradiction,
and adjacent ABSTAIN risk families.

Stage 0.7b-d guarded recipe probes landed on 2026-05-27 and should not be
scaled. 0.7b restored seed-42 safety best (88.98% accuracy / 2.13% FT,
`science_medicine` FT 2.86%, `factual_contradiction` FT 3.54%) but cut
support recall (`multi_source_corroboration` 61.29%,
`quantitative_consensus` 73.33%). 0.7c softened the guard and recovered
`multi_source_corroboration` to 68.82%, but did not restore safety enough.
0.7d targeted only the strongest risk slices but did not dominate either.
Post-hoc TRUSTWORTHY-threshold sweeps and the existing
`false_trustworthy_risk` scalar head reduce FT only by eroding support recall.
The next move should be architectural: add an explicit guarded governance head
or TRUSTWORTHY-penalty path that is trained through the terminal support state.

### Stage 0.8: guarded-head scaffold

Stage 0.8 added `GuardedSupportAggregatingMoEForGovernance`, a small learned
positive TRUSTWORTHY-penalty head on the Stage 0.7 support-fused state. Config:
`configs/moe/pyrrho_moe_stage0_8_guarded_support_aggregation.yaml`. The model
has 57,501,747 parameters. The scaffold trains, saves, reloads, and passes
failure-analysis plumbing, but the first quality result is negative.

Seed-42 artifacts:

- `outputs/moe/stage0_8_guarded_support_aggregation_g3_seed42/` (4 epochs)
- `outputs/moe/stage0_8_guarded_support_aggregation_g3_seed42_e3/` (3 epochs)

The 4-epoch run overfit/shifted unsafe: held-out test **85.28%** calibrated
accuracy / **5.57%** FT. The 3-epoch checkpoint was safer but too conservative:
**85.40%** accuracy / **2.31%** FT / **76.78%** route / **73.32%** taxonomy.
Do not scale this guarded-head implementation. The useful next guard must be
less entangled with the main governance head.

### Stage 0.9: explicit trust-guard target

Stage 0.9 added `TrustGuardedSupportAggregatingMoEForGovernance`, a binary
trust verifier over the Stage 0.7 support-fused state plus detached candidate
governance logits. Config:
`configs/moe/pyrrho_moe_stage0_9_trust_guarded_support_aggregation.yaml`. The
verifier is explicitly supervised to accept TRUSTWORTHY rows and reject
non-TRUSTWORTHY rows, with extra weight on the risk slices that drove Stage 0.7
false-TRUSTWORTHY regressions.

Seed-42 artifacts:

- `outputs/moe/stage0_9_trust_guarded_support_aggregation_g3_seed42/` (4 epochs)
- `outputs/moe/stage0_9_trust_guarded_support_aggregation_g3_seed42_e3/` (3 epochs)

CUDA smoke/reload/tests passed, but quality is negative. The 3-epoch checkpoint
reached **86.50%** calibrated test accuracy / **1.24%** false-trustworthy /
**82.19%** route / **72.83%** taxonomy. The 4-epoch checkpoint was even more
conservative at **84.75%** / **0.59%** FT / **84.38%** route / **73.97%**
taxonomy. Failure reports:

- `outputs/moe/stage0_9_trust_guarded_support_aggregation_g3_seed42_e3/failure_analysis_test/failure_report.md`
- `outputs/moe/stage0_9_trust_guarded_support_aggregation_g3_seed42/failure_analysis_test/failure_report.md`

The explicit binary verifier restores safety by collapsing support-pattern
TRUSTWORTHY recall: at 3 epochs, `multi_source_corroboration` falls to
**45.16%**, `consistent_chain` to **58.16%**, and `quantitative_consensus` to
**61.90%**; at 4 epochs they fall further to **38.71%**, **46.81%**, and
**51.43%**. Do not scale this implementation. If custom-trunk work continues,
the next verifier should be a true post-hoc reranker over frozen Stage 0.7
candidate logits, not another in-model TRUSTWORTHY penalty path.

### Stage 0.7 post-hoc verifier

The first true post-hoc verifier landed after the Stage 0.9 in-model guard
failed. Script: `scripts/train_moe_posthoc_verifier.py`. It loads frozen Stage
0.7 checkpoints, collects candidate governance/route/taxonomy/scalar outputs
for train/eval/test, trains a separate HGB binary verifier only on rows the
frozen model predicts as TRUSTWORTHY, selects a verifier threshold on eval, and
only demotes candidate TRUSTWORTHY predictions at test time. The Stage 0.7 trunk
is not updated.

Safety-heavy three-seed artifact:
`outputs/moe/stage0_7_posthoc_verifier_g3_3seed/summary.json`.

Using target eval FT **2.5%** and max eval accuracy drop **1.5%**, held-out test
moved from the verifier-script baseline **89.37 +/- 0.59%** accuracy / **2.94
+/- 0.36%** false-trustworthy to guarded **88.97 +/- 0.51%** accuracy / **1.99
+/- 0.17%** false-trustworthy. TRUSTWORTHY recall moved **81.50 +/- 1.97% ->
78.56 +/- 1.67%**.

Safety improvements are real: `science_medicine` FT moved **5.44% -> 1.90%**,
`evidence_absent` **4.60% -> 1.44%**, `wrong_entity` **6.46% -> 4.08%**, and
`numerical_conflict` **7.14% -> 5.44%**. The caveat is support recall:
`consistent_chain` accuracy moved **74.70% -> 65.96%**,
`multi_source_corroboration` **68.82% -> 66.31%**, and
`quantitative_consensus` **79.05% -> 74.92%**. This is the first positive guard
direction, but it should remain a separate reranker/verifier unless support
retention is explicitly tuned.

Support-retaining three-seed artifact:
`outputs/moe/stage0_7_posthoc_verifier_g3_3seed_ft028/summary.json`.

Using target eval FT **2.8%** and the same max eval accuracy drop, held-out test
moved to **89.29 +/- 0.69%** accuracy / **2.37 +/- 0.26%**
false-trustworthy, with TRUSTWORTHY recall **80.20 +/- 1.64%**. This is the
preferred verifier operating point so far: it gives up less accuracy and support
than the 2.5% target while still reducing FT versus the verifier-script baseline.
Support slices are much better retained (`multi_source_corroboration` **68.82%
-> 67.38%**, `quantitative_consensus` **79.05% -> 78.10%**), though
`consistent_chain` still drops **74.70% -> 70.92%**. Route safety still improves:
`science_medicine` FT **5.44% -> 3.13%**, `economics_finance` **2.99% ->
2.49%**.

Minimal-intervention three-seed artifact:
`outputs/moe/stage0_7_posthoc_verifier_g3_3seed_ft030/summary.json`.

Using target eval FT **3.0%** gives **89.35 +/- 0.64%** accuracy / **2.61 +/-
0.36%** false-trustworthy, with TRUSTWORTHY recall **80.85 +/- 1.30%**. This
retains support best (`consistent_chain` **74.70% -> 72.81%**,
`multi_source_corroboration` **68.82% -> 67.74%**, `quantitative_consensus`
**79.05% -> 78.73%**), but the safety move is weaker and seed 7 is a no-op
because its eval FT is already under the 3.0% target. Treat it as a
minimal-intervention option, not the preferred safety setting.

The verifier script now reports support/risk metrics in the threshold sweep and
supports optional support-aware selection constraints. A seed-42 support-aware
probe showed that with target eval FT **2.8%**, no threshold can keep eval
support accuracy within 3 points of baseline while meeting the FT target. For
this verifier, the target-FT setting is therefore the practical control surface.

Packaging/reload check:
`outputs/moe/stage0_7_posthoc_verifier_g3_ft028_package/`.
`scripts/package_moe_posthoc_verifier.py` now creates a lightweight package with
copied per-seed verifier artifacts, `manifest.json`, feature schema
`pyrrho_moe_posthoc_features_v1` width **120**, checksums, and a reload evaluator
that points back to the frozen Stage 0.7 base checkpoints rather than copying
them. Full eval+test reload reproduced the packaged per-seed metrics with **0**
max absolute delta. The reload report preserved the preferred operating point:
eval **89.33 +/- 0.06%** accuracy / **2.76 +/- 0.00%** false-trustworthy, test
**89.29 +/- 0.69%** accuracy / **2.37 +/- 0.26%** false-trustworthy.

### Stage 1: upcycled 4B skeleton

Create the 4B-A0.4B model from the dense seed.

Initial trainable set:

- routers,
- task heads,
- per-expert adapters or LoRA blocks,
- optionally final dense layers.

Shared attention/embedding weights start frozen or low-LR to preserve seed competence.

Exit gate:

- model loads,
- exact param count is inside target window,
- one forward pass activates one expert per MoE layer,
- router labels are trainable without collapse.

Status 2026-05-26 night: the Qwen3-MoE seed pack loads and trains with pyrrho
heads, physical expert residual adapters, semantic-route pooled adapters, and
`pyrrho-nano-g3` governance-logit distillation. The bounded adapter/distillation
probes did not clear the continuation gate: physical expert adapters reached
50.00% calibrated accuracy / 4.40% false-trustworthy / 24.02% route accuracy on
the 512-row eval slice, and semantic-route adapters reached 44.34% / 1.65% /
26.37%. Forcing gold semantic routes on the semantic-adapter checkpoint only
reached 45.51% calibrated accuracy, so the failure is not explained by predicted
route errors alone. These are below the frozen-head 8,192-row best and far below
the >80% probe gate, so the current Qwen Stage 1 adapter variants should not be
scaled.

### Stage 2: supervised governance post-training

Train on fitz-gov with classification, routing, taxonomy, scalar, and trace losses.

Exit gate:

- validation accuracy clears baseline gate,
- false-trustworthy is below release gate or moving clearly in that direction,
- per-expert accuracy is not hiding a failed expert,
- route confusion matrix is interpretable.

### Stage 3: preference / safety tuning

Apply DPO/GRPO-style preference tuning focused on false TRUSTWORTHY and confidence calibration.

Exit gate:

- false-trustworthy <= 5.7% on held-out eval,
- no large collapse in ABSTAIN/DISPUTED recall,
- OOD probes do not regress.

### Stage 4: quantization and CPU runtime

Quantize and test CPU inference.

Exit gate:

- Q4 artifact fits the memory target,
- CPU latency is measured on fixed hardware,
- outputs match PyTorch within acceptable tolerance,
- runtime supports the custom router and expert selection.

## 9. Evaluation Gates

Inherited release gates:

- Overall accuracy >= 78.7%.
- False-trustworthy <= 5.7%.
- Three-seed mean, not best seed.

MoE-specific gates:

- Route accuracy by semantic expert.
- Per-expert classification accuracy.
- Expert traffic distribution: no collapsed or dead expert group.
- False-trustworthy by expert, not only global.
- Calibration metrics on borderline rows.
- CPU memory and latency measured after quantization.
- OOD probes: automotive, aviation, and any future domain probes.

## 10. Implementation Work Items

Required code:

- `src/pyrrho/moe/config.py`: architecture config and param-count helpers. **Landed 2026-05-26.**
- `src/pyrrho/moe/modeling.py`: PyTorch model with dense-to-MoE upcycling support. **Stage 0 tiny route prototype, oracle-route eval hook, Stage 0.5 route-coupled student, and Stage 0.6 token route-coupled trunk landed 2026-05-26.**
- `src/pyrrho/moe/router.py`: semantic routing and physical shard selection.
- `src/pyrrho/moe/losses.py`: multi-task, safety, distillation, and router losses. **Stage 0 losses, governance-logit distillation, and Stage 0.6 targeted governance sample weights landed 2026-05-26.**
- `scripts/prepare_moe_data.py`: fitz-gov to MoE multi-task format. **Landed 2026-05-26.**
- `scripts/generate_teacher_traces.py`: teacher trace generation with locked manifests.
- `scripts/generate_moe_teacher_logits.py`: g3 governance-logit sidecars for MoE distillation. **Landed 2026-05-26.**
- `scripts/analyze_moe_failures.py`: per-case Stage 0/0.5/0.6 failure analysis. **Landed 2026-05-26.**
- `scripts/upcycle_dense_to_moe.py`: seed model to pyrrho-MoE checkpoint. **Inspect-only compatibility planner landed 2026-05-26.**
- `scripts/train_moe.py`: staged training loop. **Stage 0/0.5/0.6 trainer, g3 distillation, loss overrides, and oracle-route reporting landed 2026-05-26.**
- `scripts/eval_moe.py`: classification, routing, expert, calibration, OOD reports. **Stage 0/0.5/0.6 evaluator landed 2026-05-26.**
- `scripts/count_moe_params.py`: exact total/active parameter manifest. **Landed 2026-05-26.**
- `scripts/train_moe_posthoc_verifier.py`: frozen-output Stage 0.7 verifier/reranker. **Landed 2026-05-27.**
- `scripts/package_moe_posthoc_verifier.py`: lightweight verifier package and reload evaluator. **Landed 2026-05-27.**

Required docs/artifacts:

- seed model selection report. **Initial scan and Qwen upcycling decision landed 2026-05-26.**
- teacher trace manifest,
- MoE data audit,
- param-count manifest,
- CPU quantization report,
- model card section explaining upcycling and fitz-gov supervision.

## 11. Main Risks

| Risk | Severity | Mitigation |
|---|---:|---|
| No compatible dense seed model | High | Search current permissive models before locking dimensions; adjust d/layers if needed while preserving 4B-A0.4B math. |
| Router collapse | High | Supervised routing CE, load-balance loss, entropy floor, per-expert traffic gates. |
| Loss of seed general competence | High | Dense-to-MoE cloning, freeze/low-LR shared trunk, teacher distillation before aggressive fitz-gov specialization. |
| Single-GPU optimizer memory | High | Start with router/adapters, use 8-bit optimizer, gradient checkpointing, expert freezing, staged expert unfreezing. |
| Custom GGUF support | Medium | Prototype PyTorch first; plan llama.cpp custom arch work only after quality clears gates. |
| fitz-gov too small for all heads | Medium | V8.0.0 target-50 contract is live; still use teacher traces and keep heads scoped to signals with reliable labels. |

## 12. Immediate Next Steps

1. ~~Run a current model search for dense seed candidates matching the baseline dimensions and license constraints.~~ Done 2026-05-26; see `docs/MOE_SEED_SEARCH_2026-05-26.md`.
2. ~~Implement `count_moe_params.py` and verify the baseline count from code.~~ Done 2026-05-26; implemented count is **3.950935086B total / 0.411991086B active inclusive**.
3. ~~Prepare/audit published V8 for MoE training.~~ Done 2026-05-26; `scripts/prepare_moe_data.py --strict` wrote `data/moe_v8` with **0** required-field misses.
4. ~~Build the tiny route prototype.~~ Done 2026-05-26; see `outputs/moe/stage0_route_proto/final_metrics.json`.
5. ~~Add standalone `eval_moe.py` so Stage 0 and later MoE checkpoints can be re-reported without retraining.~~ Done 2026-05-26.
6. ~~Resolve tokenizer/embedding handling for `Qwen/Qwen3-0.6B-Base` or choose a wider alternate seed.~~ Done 2026-05-26; keep Qwen tokenizer and use `pyrrho_moe_g3_alpha_qwen.yaml`.
7. ~~Implement inspect-only dense-to-MoE upcycling planner.~~ Done 2026-05-26; `outputs/moe/upcycling/qwen_alpha_inspect.json` validates direct-copy surfaces and FFN compression requirements.
8. ~~Implement actual dense-to-MoE weight transform using `src/pyrrho/moe/upcycling.py`.~~ Done 2026-05-26; `outputs/moe/upcycling/qwen_alpha_seed_pack/` contains a 30-shard Qwen3-MoE-compatible seed pack and passing meta-model shape validation.
9. ~~Build the governance/router wrapper around the Qwen3-MoE trunk, load the seed pack, attach pyrrho task heads, and run a no-training forward smoke.~~ Done 2026-05-26; `scripts/smoke_moe_qwen_wrapper.py` produced valid governance `[2,3]`, route `[2,8]`, taxonomy `[2,23]`, and scalar `[2,15]` outputs from the seed pack on CUDA.
10. ~~Run the first Stage 1 heads-only training smoke with the trunk frozen.~~ Done 2026-05-26; `scripts/train_moe_qwen_heads.py` ran 2 CUDA optimizer steps and wrote `outputs/moe/qwen_heads_stage1_smoke/train_report.json`.
11. ~~Run a longer Stage 1 heads-only training pass on a bounded V8 subset, then evaluate whether route/governance signals justify internal-router training or adapter insertion.~~ Done 2026-05-26; frozen heads, internal routers, final dense unfreeze, and attention LoRA were all negative for release quality.
12. ~~Run a bounded adapter/distillation v2 probe with `pyrrho-nano-g3` governance logits and trainable expert-local capacity.~~ Done 2026-05-26; physical expert adapters and semantic-route adapters both stayed below the >80% continuation gate.
13. ~~Run Stage 0 route-first distillation diagnostics.~~ Done 2026-05-26; route-heavy g3-distilled Stage 0 reached **82.43%** calibrated test accuracy / **5.45%** FT / **82.80%** route, showing route labels are learnable when they control expert execution.
14. ~~Train a route-coupled custom student/trunk where the supervised semantic route is the actual active expert path from the start.~~ Done 2026-05-26; Stage 0.5 reached **83.91 +/- 1.18%** calibrated test accuracy / **5.55 +/- 0.03%** false-trustworthy / **82.92 +/- 0.35%** route across 3 seeds with a 53.86M-param route-coupled student.
15. ~~Add Stage 0.5 per-route/per-taxonomy failure reports.~~ Done 2026-05-26; `scripts/analyze_moe_failures.py` wrote eval/test reports under `outputs/moe/stage0_5_route_coupled_g3_3seed/failure_analysis_{eval,test}/`.
16. ~~Map the positive 3-seed signal into Stage 0.6 without returning to the failed Qwen adapter variants.~~ Done 2026-05-26; `TokenRouteCoupledMoEForGovernance` and `configs/moe/pyrrho_moe_stage0_6_token_route_coupled.yaml` add real token interaction, decoder-shaped route-coupled blocks, and targeted governance sample weights for support-pattern recall and FT-risk slices. Bounded CUDA smoke/reload passed at `outputs/moe/stage0_6_token_route_coupled_smoke/`.
17. ~~Run the full Stage 0.6 seed-42 quality probe, then 3-seed stability if positive.~~ Done 2026-05-27; `outputs/moe/stage0_6_token_route_coupled_g3_3seed/summary.json` reached **87.23 +/- 1.29%** calibrated held-out accuracy / **2.92 +/- 1.06%** false-trustworthy / **86.06 +/- 0.94%** route / **71.97 +/- 0.72%** taxonomy.
18. ~~Run Stage 0.6b support-recall recipe work focused on `multi_source_corroboration` and `quantitative_consensus`, while preserving the Stage 0.6 safety gains.~~ Done 2026-05-27; 0.6b/0.6c/0.6d/0.6e seed-42 probes show scalar/per-pattern weighting alone does not dominate Stage 0.6.
19. ~~Add a Stage 0.7 support-aggregation architecture path instead of continuing scalar loss-weight sweeps.~~ Done 2026-05-27; `SupportAggregatingMoEForGovernance` plus query/source dataset tensors reached **89.49 +/- 0.47%** calibrated held-out accuracy / **3.06 +/- 0.45%** FT / **82.61 +/- 2.50%** route / **75.78 +/- 0.21%** taxonomy across 3 seeds at `outputs/moe/stage0_7_support_aggregation_g3_3seed/summary.json`.
20. ~~Run a Stage 0.7b guarded support-aggregation probe that keeps the Stage 0.7 support gains but restores Stage 0.6 safety on `science_medicine`, `factual_contradiction`, and adjacent ABSTAIN risk families.~~ Done 2026-05-27; 0.7b/0.7c/0.7d recipe probes and post-hoc risk/threshold sweeps showed weighting/gating trades off support too quickly and should not be scaled.
21. ~~Add a Stage 0.8 architectural guard head on top of the Stage 0.7 support state.~~ Done 2026-05-27; scaffold landed and passed smoke/reload/tests, but seed-42 quality was negative (3-epoch **85.40% / 2.31% FT**, 4-epoch **85.28% / 5.57% FT**). Do not scale this implementation.
22. ~~Try an explicit binary FT-risk/trust-accept guard target rather than scalar weighting or a simple positive penalty on the main support state.~~ Done 2026-05-27; Stage 0.9 trained and reloads, but is negative because it collapses support-pattern TRUSTWORTHY recall.
23. ~~Try a true post-hoc reranker/verifier over frozen Stage 0.7 candidate logits.~~ Done 2026-05-27; the HGB verifier is positive at **88.97 +/- 0.51%** accuracy / **1.99 +/- 0.17%** FT, with support-recall caveats.
24. ~~Tune the post-hoc verifier for support retention.~~ Done 2026-05-27; the preferred 2.8% eval-FT target is **89.29 +/- 0.69%** accuracy / **2.37 +/- 0.26%** FT and keeps support slices much closer to Stage 0.7.
25. ~~Check minimal-intervention/support-aware verifier selection.~~ Done 2026-05-27; the 3.0% target preserves support best at **89.35 +/- 0.64%** / **2.61 +/- 0.36%** FT but is a weaker safety move, and explicit support-aware selection does not beat target-FT tuning.
26. ~~Package/evaluate the preferred Stage 0.7 post-hoc verifier as a separate reranker.~~ Done 2026-05-27; `outputs/moe/stage0_7_posthoc_verifier_g3_ft028_package/package_eval_report.json` reloads the package over eval+test and reproduces all packaged per-seed metrics with **0** max absolute delta.
27. If continuing the custom trunk, keep Stage 0.7 as the quality baseline and the packaged Stage 0.7 post-hoc verifier as the positive guard artifact. Do not return to in-model penalty heads unless the support-retention objective is explicit.
