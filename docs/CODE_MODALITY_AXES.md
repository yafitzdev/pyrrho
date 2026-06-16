# Code Modality Axes

This note resolves the current modality/domain question for pyrrho and fitz-gov
candidate work. It is a modeling/data-design note, not a V8 schema change.

## Decision

`routing.expert_fired` remains the semantic subject route. It answers: what is
the retrieved evidence about?

`meta.modality` is the evidence representation. It answers: what form does the
retrieved evidence take?

So unstructured, structured, and code rows can share semantic domains. A billing
service implementation can route to `economics_finance`; a clinical pipeline
script can route to `science_medicine`; most general code routes to
`technology_computing`.

Code languages are relevant, but they are not primary domains. They are
code-modality coverage axes. Python vs TypeScript vs YAML can change syntax and
retrieval shape, but the label boundary is usually about whether the evidence
supports the query, conflicts with another artifact, or is missing the exact
artifact/result/field needed.

## Axes To Track For Code

Keep these as generation, audit, and evaluation axes unless fitz-gov explicitly
adds a future schema field:

| Axis | Why it matters |
|---|---|
| `label` | The governance target: `ABSTAIN`, `DISPUTED`, `TRUSTWORTHY`. |
| `taxonomy.pattern` | The canonical V8 failure/support pattern. |
| `routing.expert_fired` | Subject/domain route, separate from modality. |
| `meta.difficulty` | Coverage balance across easy/medium/hard. |
| `meta.modality` | Evidence representation: `code`. |
| Code language/family | Parser/syntax surface: Python, TS/JS, SQL, YAML, JSON, shell/CI, Go/Rust/JVM. |
| Artifact type | Source, config, test, trace/log, API spec, generated client, docs, CI/build. |
| Question target | Function/symbol, route/endpoint, config/default, type/field, version/build, trace/test result, security/auth. |
| Failure mode | Wrong symbol, wrong version, missing specific field, missing execution result, config/runtime conflict, docs/code conflict, test/impl conflict. |
| Retrieval serialization | Code excerpt, review packet, diff context, trace excerpt, search-result packet. |

## Current Candidate Audit

`scripts/audit_code_modality_axes.py` audits the current 10,000-row code
candidate pack without modifying it. Latest output:

- Report: `outputs/code_modality_axis_audit/modality_code_v1_20260527/report.md`
- JSON: `outputs/code_modality_axis_audit/modality_code_v1_20260527/audit.json`

Key findings:

- All **10,000/10,000** rows carry `meta.modality: "code"`.
- Semantic routing is not code-specific: `technology_computing` **7,144**,
  `economics_finance` **1,428**, `science_medicine` **1,428**.
- Language signals are broad but uneven and partly synthetic: Python **3,865**,
  Markdown/docs **3,709**, TS/JS **3,360**, YAML **2,539**, runtime traces
  **1,179**, JSON **370**, unknown **239**, and only ~100 each for SQL,
  shell/CI, Go, and Java/Kotlin.
- Artifact coverage is useful but not complete: source code **8,963**, docs
  **3,709**, tests **1,181**, traces/logs **1,179**, config **1,074**,
  search-results **667**, build/CI **370**, API specs **333**.
- The hard-code OOD target gaps are concrete: **0** direct control-flow support
  rows and **0** missing-specific-field rows by the current audit mapping.
- **239/10,000** rows have extension/syntax mismatch flags, mostly rows like
  `.sql`, `.yaml`, `.go`, or `.java_kotlin` paths containing JS-like
  `function ...` syntax. That means some apparent language coverage is path
  decoration, not true syntax coverage.

## Targeted Patch v1

The first targeted patch is generated in fitz-gov by
`scripts/sdgp_generate_code_modality_patch.py`.

Candidate workspace:

- `C:/Users/yanfi/PycharmProjects/fitz-gov/data/_workspaces/handoff/modality_code_patch_v1_20260528/`

Pyrrho audit outputs:

- Patch only: `outputs/code_modality_axis_audit/modality_code_patch_v1_20260528/`
- Original 10k + patch: `outputs/code_modality_axis_audit/modality_code_v1_plus_patch_v1_20260528/`

Patch status:

- **720** candidate-only code rows.
- Label-balanced: **240 TRUSTWORTHY / 240 ABSTAIN / 240 DISPUTED**.
- Mechanisms: `control_flow_support` **80**, `decorator_guard_support` **80**,
  `transaction_order_support` **80**, `missing_specific_field` **60**,
  `wrong_symbol` **60**, `wrong_api_version` **60**,
  `test_definition_without_run` **60**, unresolved runtime-status conflict **80**,
  `docs_code_conflict` **80**, `test_impl_conflict` **80**.
- Structural validation: **0 errors** via fitz-gov `Checker(require_training_schema=True)`.
- Patch-only syntax mismatch audit: **0/720**.
- Original 10k + patch closes the audited hard-code target gaps: **10,720** rows,
  **0** hard-OOD target gaps, **239** extension/syntax mismatch flags inherited
  from the original pack.
- Blind-label QA queue exists at
  `C:/Users/yanfi/PycharmProjects/fitz-gov/data/_workspaces/qa/modality_code_patch_v1_20260528/`.
- Codex blind shards are prepared at
  `C:/Users/yanfi/PycharmProjects/fitz-gov/data/_workspaces/qa/modality_code_patch_v1_20260528/codex_subagent_blind/`
  (**720** blind rows across **12** shards).
- Qwen/LM Studio blind QA was paused by user decision. Targeted slices after
  repair were clean: config/runtime **10/10**, wrong-symbol **10/10**. The
  partial full run reached **115** predictions before stopping. Treat labels as
  trusted only for local controls until full blind QA is completed later.

The label-trusted 3-seed local control has been trained on published V8.0.1 +
the 20k structured/code candidate pack + this 720-row patch:

- Data: `data/processed_v8_plus_structured_code_patch_candidate`
- Seed checkpoints: `outputs/modality_retraining/structured_code_patch_seed{42,1337,7}/best_model`
- Summary: `outputs/modality_retraining/structured_code_patch_3seed_summary.json`
- Held-out test: **98.52 ± 0.13%** calibrated accuracy /
  **1.06 ± 0.22%** false-TRUSTWORTHY
- Modality test slices: code **100.00 ± 0.00% / 0.00 ± 0.00% FT** (n=1,071),
  structured **100.00 ± 0.00% / 0.00 ± 0.00% FT** (n=995), unstructured
  **97.28 ± 0.25% / 1.92 ± 0.40% FT** (n=2,459)
- Hand-authored code OOD probe:
  `outputs/code_ood_probe/structured_code_patch_3seed_summary.json`
  scored **95.37 ± 3.21%** accuracy / **6.94 ± 4.81%** FT, improving the
  unpatched joint branch from **77.78 ± 14.70% / 29.17 ± 27.32% FT**.
- The remaining code OOD misses all come from
  `constant_config_conflict`: retry-limit code/config numerical conflict. Seeds
  42 and 1337 miss only the `review_packet` serialization; seed 7 misses
  `code_excerpt`, `review_packet`, and `diff_context`.

This is not release evidence: the patch is still not fully blind-label-QA-clean.

## Targeted Retry Patch v1

The second targeted patch is generated in fitz-gov by
`scripts/sdgp_generate_code_retry_conflict_patch.py`.

Candidate workspace:

- `C:/Users/yanfi/PycharmProjects/fitz-gov/data/_workspaces/handoff/modality_code_retry_conflict_patch_v1_20260529/`

Pyrrho audit outputs:

- Patch only: `outputs/code_modality_axis_audit/modality_code_retry_conflict_patch_v1_20260529/`
- Original 10k + patch v1 + retry patch:
  `outputs/code_modality_axis_audit/modality_code_v1_plus_patch_v1_plus_retry_patch_v1_20260529/`

Patch status:

- **360** candidate-only code rows.
- Label-balanced: **120 TRUSTWORTHY / 120 ABSTAIN / 120 DISPUTED**.
- Mechanisms: `retry_limit_code_config_conflict` **120**,
  `retry_limit_code_config_agreement` **120**, and
  `retry_limit_wrong_service` **120**.
- Serialization-balanced: `code_excerpt` **120**, `review_packet` **120**,
  `diff_context` **120**.
- Structural validation: **0 errors** via fitz-gov `Checker(require_training_schema=True)`.
- Patch-only syntax mismatch audit: **0/360**.
- Original 10k + patch v1 + retry patch has **11,080** code rows,
  **0** hard-OOD target gaps, and **239** syntax mismatch flags inherited from
  the original 10k pack.

The label-trusted 3-seed local control has been trained on published V8.0.1 +
the 20k structured/code candidate pack + patch v1 + this retry patch:

- Data: `data/processed_v8_plus_structured_code_retry_patch_candidate`
- Seed checkpoints:
  `outputs/modality_retraining/structured_code_retry_patch_seed{42,1337,7}/best_model`
- Summary: `outputs/modality_retraining/structured_code_retry_patch_3seed_summary.json`
- Held-out test: **98.62 ± 0.15%** calibrated accuracy /
  **1.06 ± 0.05%** false-TRUSTWORTHY
- Modality test slices: code **100.00 ± 0.00% / 0.00 ± 0.00% FT**
  (n=1,083), structured **100.00 ± 0.00% / 0.00 ± 0.00% FT**
  (n=991), unstructured **97.47 ± 0.28% / 1.94 ± 0.09% FT**
  (n=2,459)
- Hand-authored code OOD probe:
  `outputs/code_ood_probe/structured_code_retry_patch_3seed_summary.json`
  scored **99.07 ± 1.60%** accuracy / **1.39 ± 2.41%** FT.
- The targeted retry-limit mechanism is now clean across seeds:
  `constant_config_conflict` is **100.00 ± 0.00% / 0.00 ± 0.00% FT**.
- The only remaining OOD failure is inherited from the earlier OOD set:
  seed 42 `code_10_missing_audit__diff_context`, expected ABSTAIN but predicted
  TRUSTWORTHY (`missing_specific_field` / `evidence_absent`).

This is still not release evidence: labels are trusted only for local controls,
and full blind-label QA remains required before merge/publish.

## Targeted Missing-Evidence Patch v1

The third targeted patch is generated in fitz-gov by
`scripts/sdgp_generate_missing_evidence_patch.py`.

Candidate workspace:

- `C:/Users/yanfi/PycharmProjects/fitz-gov/data/_workspaces/handoff/modality_missing_evidence_patch_v1_20260529/`

Pyrrho audit outputs:

- Patch only: `outputs/code_modality_axis_audit/modality_missing_evidence_patch_v1_20260529/`
- Original 10k + patch v1 + retry patch + missing-evidence patch:
  `outputs/code_modality_axis_audit/modality_code_v1_plus_patch_v1_plus_retry_patch_v1_plus_missing_evidence_patch_v1_20260529/`

Patch status:

- **360** candidate-only rows.
- Modality-balanced: **180 code** / **180 structured**.
- Label-balanced: **120 TRUSTWORTHY / 120 ABSTAIN / 120 DISPUTED**.
- Code serialization: `diff_context` only, targeting the inherited
  `missing_specific_field` OOD miss plus exact-support and docs/code-conflict
  controls.
- Structured serializations: `markdown_table`, `csv_extract`, and
  `evidence_packet`, targeting missing result grids plus exact filtered rows
  and same-metric conflicting values.
- Structural validation: **0 errors** via fitz-gov `Checker(require_training_schema=True)`.
- Patch-only syntax mismatch audit: **0/180** code rows.
- Original 10k + patch v1 + retry patch + missing-evidence patch has
  **11,260** code rows and **180** structured rows in the code-axis audit,
  **0** hard-OOD target gaps, and **239** syntax mismatch flags inherited from
  the original 10k code pack.

The label-trusted exposed-seed control has been trained on published V8.0.1 +
the 20k structured/code candidate pack + patch v1 + retry patch + this
missing-evidence patch:

- Data: `data/processed_v8_plus_structured_code_missing_evidence_patch_candidate`
- Seed checkpoints:
  `outputs/modality_retraining/structured_code_missing_evidence_patch_seed{42,7}/best_model`
- Summary: `outputs/modality_retraining/structured_code_missing_evidence_patch_2seed_summary.json`
- Held-out test: **98.70 ± 0.18%** calibrated accuracy /
  **0.75 ± 0.25%** false-TRUSTWORTHY
- Modality test slices: code **100.00 ± 0.00% / 0.00 ± 0.00% FT**
  (n=1,121), structured **100.00 ± 0.00% / 0.00 ± 0.00% FT**
  (n=1,023), unstructured **97.56 ± 0.35% / 1.39 ± 0.46% FT**
  (n=2,459)
- Hand-authored code OOD probe:
  `outputs/code_ood_probe/structured_code_missing_evidence_patch_2seed_summary.json`
  scored **93.06 ± 5.89%** accuracy / **2.08 ± 2.95%** FT. The inherited
  seed-42 `missing_specific_field` false-TRUSTWORTHY is fixed, but seed 42 now
  over-demotes TRUSTWORTHY support examples and seed 7 still false-trusts one
  retry-limit conflict.
- Hand-authored tabular OOD probe:
  `outputs/tabular_ood_probe/structured_code_missing_evidence_patch_2seed_summary.json`
  scored **87.50 ± 13.75%** accuracy / **0.00 ± 0.00%** FT, with large seed
  spread from seed-42 over-demotion of exact-row/SLA TRUSTWORTHY cases.

This is still not release evidence: labels are trusted only for local controls,
full blind-label QA remains required, and the branch is a mixed tradeoff rather
than a clear successor to the retry-patch branch.

## Threshold/Policy Sweep

The local OOD-only threshold sweep is written to:

- `outputs/modality_policy_sweep/retry_vs_missing_evidence/summary.json`
- `outputs/modality_policy_sweep/retry_vs_missing_evidence/report.md`

It used the saved code/tabular OOD prediction probabilities only; no retraining,
LM Studio, or API calls were used.

Readout:

- Retry-patch remains the current local baseline. At eval-selected thresholds,
  code OOD is **99.07 ± 1.60% / 1.39 ± 2.41% FT** and tabular OOD is
  **93.52 ± 8.93% / 1.39 ± 2.41% FT**.
- The diagnostic best fixed OOD threshold for retry-patch is tau **0.73**,
  giving code OOD **100.00 ± 0.00% / 0.00 ± 0.00% FT** and tabular OOD
  **93.52 ± 11.23% / 0.00 ± 0.00% FT** on the small probes.
- Missing-evidence does not become a clean successor under threshold tuning:
  its best fixed OOD threshold is tau **0.34**, combined **93.06%** accuracy /
  **1.04%** FT, and its exposed seeds need thresholds in opposite directions.

## Specialist Routing Diagnostic

The smallest local specialist comparison is complete and written to:

- `outputs/modality_specialist_compare/retry_patch_seed42_router/summary.json`
- `outputs/modality_specialist_compare/retry_patch_seed42_router/report.md`
- `outputs/code_ood_probe/code_specialist_seed42/summary.json`
- `outputs/tabular_ood_probe/structured_specialist_seed42/summary.json`
- `outputs/modality_specialist_compare/retry_patch_seed42_patch_aware_code_router/summary.json`
- `outputs/code_ood_probe/code_retry_patch_seed42/summary.json`
- `outputs/modality_retraining/code_retry_patch_seed42/eval_report.json`

It used fixed checkpoints only, routed by the existing `modality` column, and
made no schema changes, row generation, LM Studio calls, or API calls.

Readout:

- Seed-42 retry-patch generalist remains stronger on the retry-patch mixed
  held-out test: **98.74%** accuracy / **1.07%** FT.
- Routing code rows to the existing seed-42 code-only specialist regressed the
  mixed held-out test to **98.26%** / **1.46%** FT because the code slice fell
  to **97.97%** / **1.66%** FT.
- Routing structured rows to the existing seed-42 structured-only specialist
  tied the retry-patch generalist on the in-distribution structured candidate
  slice, so it does not add evidence for augmentation.
- On the hand-authored OOD probes, the code-only specialist was safer but too
  conservative (**86.11%** / **0.00%** FT, TRUSTWORTHY recall **58.33%**), while
  the structured-only specialist was unsafe (**91.67%** / **8.33%** FT).

Patch-aware code-specialist follow-up:

- `scripts/filter_processed_modalities.py` prepared
  `data/processed_v8_plus_code_retry_patch_candidate` from the retry-patch
  processed set by keeping only `unstructured` and `code` rows. Splits are
  **train=28,564**, **eval=3,566**, and **test=3,542**.
- Seed 42 trained at `outputs/modality_retraining/code_retry_patch_seed42/`.
  Held-out code+unstructured test is **98.25%** accuracy / **1.49%** FT;
  candidate code is **100.00%** / **0.00%** FT, but unstructured is
  **97.48%** / **2.13%** FT.
- Code OOD is **97.22%** / **4.17%** FT, tying the retry-patch seed-42
  generalist. It fixes the earlier missing-field miss but reintroduces a
  retry-limit `constant_config_conflict` false-TRUSTWORTHY on `code_excerpt`.
- Routing code rows to this patch-aware specialist ties the retry-patch
  seed-42 generalist on mixed held-out test (**98.74%** / **1.07%** FT), because
  both are already perfect on in-distribution candidate code rows.

Conclusion: do not replace or augment retry-patch with either the existing
specialists or the patch-aware separate code encoder.

Patch-aware structured-specialist follow-up:

- `scripts/filter_processed_modalities.py` prepared
  `data/processed_v8_plus_structured_retry_patch_candidate` from the retry-patch
  processed set by keeping only `unstructured` and `structured` rows. Splits
  are **train=27,665**, **eval=3,477**, and **test=3,450**.
- Seed 42 trained at `outputs/modality_retraining/structured_retry_patch_seed42/`.
  Held-out structured+unstructured test is **98.32%** accuracy / **1.02%** FT;
  candidate structured is **100.00%** / **0.00%** FT, but unstructured is
  **97.64%** / **1.42%** FT.
- Routing structured rows to this specialist ties the retry-patch seed-42
  generalist on mixed held-out test (**98.74%** / **1.07%** FT), because both
  are already perfect on in-distribution candidate structured rows.
- Tabular OOD is unsafe: **88.89%** accuracy / **16.67%** FT, with ABSTAIN
  metric-mismatch and missing-result rows false-trusted.

Conclusion: do not replace or augment retry-patch with the patch-aware separate
structured encoder either. The joint retry-patch generalist remains the local
baseline.

## Modality Threshold Diagnostic

`scripts/modality_threshold_sweep.py` tests the simplest no-training policy
augmentation: choose TRUSTWORTHY thresholds per modality on eval, then apply
them to held-out test for fixed checkpoints.

Retry-patch 3-seed artifact:

- `outputs/modality_threshold_sweep/retry_patch_3seed/summary.json`
- `outputs/modality_threshold_sweep/retry_patch_3seed/report.md`

Readout:

- Global retry-patch thresholds remain best on held-out accuracy:
  **98.62 ± 0.12%** accuracy / **1.06 ± 0.04%** FT.
- Per-modality thresholds selected to avoid eval FT regression scored
  **98.61 ± 0.11%** accuracy / **1.04 ± 0.05%** FT, a **-0.01 pp** accuracy
  trade for **-0.02 pp** FT.
- The only movement was seed 42 unstructured threshold **0.34 -> 0.53**, which
  reduced test FT by **0.07 pp** but reduced accuracy by **0.04 pp**. Seeds
  1337 and 7 kept their global thresholds for every modality.

Conclusion: simple per-modality thresholding does not justify augmenting the
retry-patch joint generalist.

## Concrete Next Data Work

Do not merge or publish the current structured/code candidate rows yet, and do
not generate more rows for the current specialist question. The retry patch has
the best 3-seed code OOD evidence so far, while the missing-evidence patch and
patch-aware code specialist are diagnostic tradeoffs rather than successors; the
threshold diagnostic also leaves the global retry-patch policy in place. Patch
labels are still label-trusted only. Full blind-label QA for
`modality_code_patch_v1_20260528`,
`modality_code_retry_conflict_patch_v1_20260529`, and
`modality_missing_evidence_patch_v1_20260529` is the next required gate before
any merge or publish decision.

QA readiness status:

- `modality_code_patch_v1_20260528` has a 720-row blind queue/manifest and
  **12** Codex blind shards at
  `C:/Users/yanfi/PycharmProjects/fitz-gov/data/_workspaces/qa/modality_code_patch_v1_20260528/`.
  Existing scores are partial/targeted, not a completed full pass.
- `modality_code_retry_conflict_patch_v1_20260529` now has a 360-row blind
  queue/manifest and **12** Codex blind shards at
  `C:/Users/yanfi/PycharmProjects/fitz-gov/data/_workspaces/qa/modality_code_retry_conflict_patch_v1_20260529/`.
- `modality_missing_evidence_patch_v1_20260529` now has a 360-row blind
  queue/manifest and **12** Codex blind shards at
  `C:/Users/yanfi/PycharmProjects/fitz-gov/data/_workspaces/qa/modality_missing_evidence_patch_v1_20260529/`.
- No full blind-label scoring has run for these three patch packs yet.

Blind labeling is intentionally deferred for rapid local progress, but patch
labels remain label-trusted only and candidate rows must not be merged or
published. Further specialist work should only happen if it tests a genuinely
new architecture or training-objective question, not another simple
separate-encoder reroute or threshold policy.

The release question is not whether code has separate domains. It is whether the
model can safely handle code-specific evidence failure modes without raising
false-TRUSTWORTHY on ABSTAIN code rows or regressing unstructured V8 safety.
