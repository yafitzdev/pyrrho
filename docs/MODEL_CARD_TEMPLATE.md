# Pyrrho Model Card Template

Use this shape for every pyrrho model card. Keep the card public-facing: no
internal pipeline names, no private dataset/category terminology, no future-plan language,
and no implementation history unless it affects model use.

## Required Shape

1. YAML metadata
   - license, library_name, pipeline_tag, language, base_model, tags, datasets, metrics.
2. Title
   - The public model name only.
3. Model Summary
   - Frame pyrrho as a RAG governance co-processor: an anti-hallucination evidence gate that sits between retrieval and generation, or beside a generator.
   - Explain that the model classifies `(question, retrieved sources)` into `ABSTAIN`, `DISPUTED`, or `TRUSTWORTHY`.
   - State clearly whether the artifact is a full model, ONNX export, or LoRA adapter.
   - State clearly that it is not an answer generator and not an open-world fact checker.
4. Labels
   - A three-row table defining `ABSTAIN`, `DISPUTED`, and `TRUSTWORTHY`.
5. Outputs
   - Distinguish raw artifact outputs from the decision object a product integration should expose.
   - For encoder releases, document raw class logits plus derived fields such as `label`, `raw_label`, `probabilities`, `confidence`, `trustworthy_probability`, `threshold`, and `used_threshold_fallback`.
   - For encoder releases, state explicitly that taxonomy/category tags, route IDs, and scalar diagnostics are not inference outputs; they are evaluation metadata or MoE-only research outputs.
   - For adapters, document raw generated text and parsed label output.
   - Include one compact JSON example of the normalized decision object.
   - Do not claim explanations, citations, spans, auxiliary research-head fields, or retrieval results unless the artifact actually returns them.
6. Intended Use
   - Explain pre-generation answer gating, retrieval retry/escalation, abstention, dispute detection, and evidence-quality logging.
   - Explain the anti-hallucination scope: reducing cases where unsupported or contradictory retrieved evidence gets treated as safe to answer from.
   - Explain what it should not be used for: generating answers, checking facts outside provided sources, span-level hallucination localization, or high-stakes autonomous decisions.
7. Quick Start
   - Minimal runnable code for the main loading path.
   - For encoder releases, include the ONNX CPU path when ONNX artifacts are shipped.
   - For adapters, show base-model plus adapter loading.
8. Decision Rule
   - Explain any TRUSTWORTHY threshold used for reported metrics.
   - Keep it practical: what probability is thresholded, and what fallback is used.
9. Results
   - Name dataset version and split sizes.
   - Use one headline table with rows `OVERALL`, `ABSTAIN`, `DISPUTED`, `TRUSTWORTHY`.
   - Use columns `Recall`, `Precision`, and `False-rate`, reported as 3-seed mean ± std.
   - Define false-rate in plain language. For label rows, it is the share of non-label cases incorrectly predicted as that label. For `TRUSTWORTHY`, this is the safety false-trustworthy rate.
   - Omit F1 from the headline table unless there is a specific reason to include it. If F1 is mentioned, define it as the harmonic mean of precision and recall.
10. Training Data
   - Public dataset name/version, language, total examples, train/validation/test sizes, and leakage-safe grouping if relevant.
   - Avoid internal category or generation-pipeline names.
11. Training Recipe
   - Base model, architecture/method, max length, labels, epochs, batch size, learning rate, loss, class weights/smoothing if used, selection metric, and seeds.
12. Limitations
   - English-only status.
   - Scope of evidence: only provided sources are judged.
   - Known weak cases that affect use.
   - Safety/threshold tradeoff.
13. Citation
   - BibTeX entry for the model.
14. License
   - CC BY-NC 4.0 plus commercial-use note.

## Style Rules

- Prefer plain task language over project-internal names.
- Do not mention internal schema names, cell-coverage machinery, generator batches, QA provider details, or future-plan phases.
- Do not compare to private baselines unless the comparison is necessary to interpret the model.
- Keep metrics precise, but do not overload the card with every diagnostic table. Link to repo docs or artifacts for deep breakdowns.
- Make limitations concrete and operational.
