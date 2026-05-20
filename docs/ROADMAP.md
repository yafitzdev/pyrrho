# Pyrrho Training Roadmap

> Comprehensive specification for finetuning, training, and reinforcement learning path toward pyrrho-MoE — a CPU-native sparse Mixture-of-Experts RAG runtime engine.

---

## 1. Vision

Pyrrho is a purpose-built RAG co-processor. Not a general assistant, not a classifier bolted onto a pipeline — a governance runtime that wraps the entire LLM interaction lifecycle. Every stage between a user query and a trustworthy LLM response has a pyrrho touchpoint.

The terminal architecture is **pyrrho-MoE: a 4B-A0.4B sparse Mixture-of-Experts model** that runs entirely on CPU, covers 16 RAG lifecycle capabilities in v1, and is deployable anywhere — laptops, edge devices, air-gapped workstations, cheap VMs — with no GPU dependency.

The dataset that makes this possible is **fitz-gov**: a versioned, domain-balanced benchmark for epistemic honesty in RAG systems. fitz-gov and pyrrho are mutually reinforcing — better data produces better models, better models validate and enrich the benchmark.

---

## 2. Model Family & Naming Convention

Model names follow the pattern: `pyrrho-{tier}-{generation}`

### Tiers

| Tier | Architecture | Description |
|---|---|---|
| `pyrrho-nano` | Fine-tuned encoder (ModernBERT-class) | CPU-efficient classifier, fast single forward pass |
| `pyrrho-small` | Fine-tuned generative SLM | Reasoning-capable, classification + rationale output |
| `pyrrho-MoE` | Sparse MoE, trained from scratch | Full RAG runtime, 4B total / 0.4B active parameters |

### Generations

Generations are suffixed by dataset version trained on:

| Suffix | Meaning |
|---|---|
| `-g1` | Trained on fitz-gov V5.1 |
| `-g1.x` | Trained on fitz-gov V6 (V5.1 + LLM-enriched schema overlay; same case IDs) |
| `-g2` | Trained on fitz-gov V7 |
| `-g3` | Trained on fitz-gov V8 |
| etc. | Increments with each major dataset version |

### Examples

- `pyrrho-nano-g1` — ModernBERT encoder fine-tuned on fitz-gov V5.1 (live on HF as `yafitzdev/pyrrho-nano-g1`)
- `pyrrho-nano-g1.1` — same architecture, retrained on fitz-gov V6 for apples-to-apples comparison on the enriched schema
- `pyrrho-nano-g2` — retrained on fitz-gov V7 (SDGP-scaled, 5K–10K cases)
- `pyrrho-small-g2` — generative SLM fine-tuned on fitz-gov V7
- `pyrrho-MoE-g3` — full sparse MoE trained from scratch on fitz-gov V8

---

## 3. fitz-gov Dataset Roadmap

fitz-gov is the benchmark that gives pyrrho credibility. It must scale ahead of the model — you need the data before you can train.

### Version Targets

| Version | Target size | Primary goal |
|---|---|---|
| V5.1 | 2,980 cases | Original baseline (no schema enrichment) |
| **V6** | **2,980 cases** | **LLM-enriched schema overlay on V5.1 — CURRENT PUBLISHED VERSION (`yafitzdev/fitz-gov` v6.0.0, shipped 2026-05-20)** |
| V7 | 5,000–10,000 cases | SDGP-scaled benchmark, taxonomy × domain × difficulty matrix coverage, MoE training base |
| V8 | 15,000–20,000 cases | Generalization, uncertainty calibration, adversarial cases |
| V9+ | 30,000+ cases | Infrastructure-grade reliability, false-trustworthy floor |

### Data Volume vs. Capability

| Range | What it buys |
|---|---|
| 2,000–5,000 | Proof of concept. Routing learns, accuracy meaningful but brittle on unseen distributions |
| 5,000–15,000 | Honest comparison vs encoder and SLM. Expert specialization emerges. Paper is writable |
| 15,000–30,000 | Generalization kicks in. Governance signals become calibrated, not just directional |
| 30,000+ | False-trustworthy approaches floor. Routing stable enough to expose as signal. Infrastructure-grade |

### Distribution Requirements

The distribution matters more than total count. Monitor and enforce:

- **Expert domain balance** — rough parity across all 7–8 MoE expert domains
- **Governance class balance per expert** — not just globally; each expert needs ABSTAIN/DISPUTED/TRUSTWORTHY representation
- **Difficulty distribution** — hard/medium/easy ratio maintained per expert
- **Taxonomy pattern coverage** — all patterns represented within each expert domain (see taxonomy below)
- **Near-miss ratio** — 20–25% of every expert's data must be borderline/near-miss cases
- **Query type diversity** — what, how, why, when, who distributed across experts
- **Reasoning type coverage** — factual, inferential, comparative, causal all represented

---

### Case Taxonomy

The governance classification problem is not infinitely varied. Failure modes are finite and enumerable. The taxonomy defines canonical evidence patterns that map deterministically to governance labels. This is the skeleton the entire dataset hangs on.

**The generation space is three-dimensional:**

```
case_taxonomy  ×  domain  ×  difficulty
```

All three dimensions are enumerable and controllable. Every generated case instantiates a specific cell in this matrix. The distribution monitor tracks cell coverage, not just marginal counts. A cell below minimum threshold gets prioritized in the next generation batch.

**Matrix size estimate:**

- Case taxonomy: ~18 patterns (6 per governance class)
- Domains: 7–8 expert domains
- Difficulty: 3 levels (easy / medium / hard)

→ ~18 × 8 × 3 = **~432 cells**. At 20 examples per cell: ~8,600 cases with guaranteed full coverage. At 25 per cell: ~10,800 cases. You hit the V6 target AND guarantee no taxonomy blind spots.

#### ABSTAIN Patterns

| Pattern | Description | Example signal |
|---|---|---|
| `wrong_specificity` | Right entity, wrong aspect or sub-topic | Hannibal/Alps when asked about Zama |
| `wrong_entity` | Evidence covers a different entity entirely | Apple iPhone when asked about apple fruit prices |
| `partial_overlap` | Evidence touches the topic but cannot answer the specific question | General medical info when asked about a specific drug dosage |
| `evidence_absent` | Nothing retrieved is remotely relevant | Complete topic mismatch |
| `too_general` | Evidence is true but too broad to answer the specific query | "Germany has a market economy" when asked for specific GDP figure |
| `temporal_mismatch` | Evidence exists but is anchored to the wrong time period | 2019 regulation when asked about current law |

#### DISPUTED Patterns

| Pattern | Description | Example signal |
|---|---|---|
| `numerical_conflict` | Multiple sources provide different numerical values for the same entity and attribute | Apple costs €5 vs €3 in Germany |
| `temporal_conflict` | Sources describe different states at different times presented without temporal framing | Source A: X was true in 2020 / Source B: Y is true now |
| `definitional_conflict` | Sources disagree on what something IS | Two sources define the same medical term differently |
| `factual_contradiction` | Direct logical incompatibility between sources | Source A: person X was born in Berlin / Source B: person X was born in Munich |
| `authority_conflict` | One high-authority source contradicts one low-authority source | Peer-reviewed paper vs blog post on same claim |
| `scope_conflict` | Sources are both correct but apply to different scopes presented as equivalent | EU regulation vs German national regulation on same topic |

#### TRUSTWORTHY Patterns

| Pattern | Description | Example signal |
|---|---|---|
| `multi_source_corroboration` | Multiple independent sources agree on the same claim | Three sources confirm same historical date |
| `single_authoritative` | One high-authority source, no contradictions, directly answers query | Official government source answers a policy question |
| `consistent_chain` | Multiple chunks from same or related sources form a coherent evidence chain | Wikipedia + cited source both support same factual claim |
| `quantitative_consensus` | Multiple sources provide same or consistent numerical values | Three sources agree price is approximately €3 |
| `expert_consensus` | Multiple domain-expert sources converge on same conclusion | Multiple medical studies agree on treatment efficacy |
| `direct_answer` | Single chunk directly and completely answers the query with no ambiguity | Definition query answered by a definitional source |

#### Taxonomy Schema Field

Every row carries the taxonomy classification as a structured field:

```json
"taxonomy": {
  "governance_class": "ABSTAIN",
  "pattern": "wrong_specificity",
  "pattern_description": "evidence covers the right entity but addresses a different aspect than the query requires",
  "cell_id": "wrong_specificity__history_geography__hard"
}
```

`cell_id` is the unique identifier for the taxonomy × domain × difficulty cell. Used by the distribution monitor to track coverage and by the generator to receive cell-specific prompts.

#### How Taxonomy Changes the Generator

The case generator takes a cell specification as input rather than an open-ended gap vector:

```python
generate(
  taxonomy_pattern = "numerical_conflict",
  domain = "medical",
  difficulty = "hard",
  expert = "science_medicine"
)
```

The generator's job is now constrained: instantiate a known pattern in a specific domain at a specific difficulty. This is a well-defined creative task — not inventing governance scenarios from scratch. Generator reliability improves significantly because the output is structurally checkable against the pattern specification.

The validator's job also sharpens: not "is this label correct in the abstract" but "does this case correctly instantiate the specified taxonomy pattern." That is a structural yes/no check, not an open-ended quality judgment.

#### Taxonomy as Interpretability Signal

In deployed pyrrho-MoE, the taxonomy pattern becomes an output signal alongside the governance classification:

```json
{
  "classification": "DISPUTED",
  "taxonomy_pattern": "numerical_conflict",
  "signals": { ... }
}
```

Downstream agents and systems receive not just the governance verdict but which canonical failure mode triggered it. A `numerical_conflict` DISPUTED warrants different handling than a `scope_conflict` DISPUTED — the taxonomy makes that distinction actionable.

### Schema: Target Data Row (fitz-gov V6+)

Every row must carry multi-task ground truth for all pyrrho-MoE output heads simultaneously.

```json
{
  "id": "moe_t1_abstain_hard_001",
  "version": "fitz-gov-6.0",

  "input": {
    "query": "What specific battle tactics did Hannibal use at the Battle of Zama in 202 BCE?",
    "query_rewritten": "What were Hannibal Barca's tactical decisions and formations at the Battle of Zama in 202 BCE?",
    "contexts": [
      {
        "id": "ctx_001",
        "text": "Hannibal Barca is considered one of history's greatest military commanders, known for crossing the Alps with war elephants in 218 BCE.",
        "authority_score": 0.71,
        "authority_signal": "encyclopedic_general",
        "temporality": {
          "is_time_sensitive": false,
          "anchor_period": "218 BCE",
          "staleness_risk": "none"
        },
        "summary": "Hannibal was a renowned Carthaginian general famous for his Alpine crossing with elephants.",
        "relevance_to_query": 0.31
      }
    ]
  },

  "governance": {
    "classification": "ABSTAIN",
    "trustworthy": 0.04,
    "disputed": 0.09,
    "abstain": 0.87,
    "confidence": 0.91,
    "grounding": 0.12,
    "conflict_density": 0.06,
    "evidence_sufficiency": 0.09,
    "boundary_proximity": {
      "nearest_class": "DISPUTED",
      "distance": 0.78
    },
    "domain_familiarity": 0.94,
    "false_trustworthy_risk": 0.03,
    "hallucination_pressure": 0.89,
    "retrieval_retry_value": 0.96,
    "human_escalation_score": 0.11,
    "query_evidence_alignment": 0.19,
    "answer_coverage": 0.08
  },

  "routing": {
    "expert_fired": "history_geography",
    "secondary_expert": null,
    "routing_confidence": 0.97
  },

  "meta": {
    "difficulty": "hard",
    "subcategory": "wrong_specificity",
    "domain": "history",
    "query_type": "what",
    "reasoning_type": "factual",
    "evidence_pattern": "absent",
    "confidence_level": "high",
    "near_miss_class": "DISPUTED",
    "near_miss_reason": "contexts discuss Hannibal authoritatively but cover wrong battles entirely, not conflicting evidence",
    "annotator_agreement": "unanimous",
    "category": "abstention"
  }
}
```

### New Fields Required for MoE Training (vs V5.1)

Fields that must be added to every row for MoE training that do not exist in V5.1:

- `query_rewritten` — reformulated query for better retrieval
- `contexts[].authority_score` + `authority_signal` — per-chunk source credibility
- `contexts[].temporality` — staleness, anchor period, time-sensitivity
- `contexts[].summary` — reference compression output per chunk
- `contexts[].relevance_to_query` — per-chunk alignment score
- `routing.expert_fired` + `routing_confidence` — MoE routing ground truth
- `governance.conflict_density` — inter-source disagreement scalar
- `governance.evidence_sufficiency` — distinct from ABSTAIN, measures evidence volume
- `governance.boundary_proximity` — nearest class and distance
- `governance.domain_familiarity` — OOD detection signal
- `governance.false_trustworthy_risk` — explicit high-stakes failure signal
- `governance.hallucination_pressure` — generation risk given evidence quality
- `governance.retrieval_retry_value` — would more retrieval change classification
- `governance.human_escalation_score` — composite escalation signal
- `governance.query_evidence_alignment` — query vs context alignment
- `governance.answer_coverage` — how much of the query the evidence covers
- `meta.confidence_level` — high/medium/borderline label certainty
- `meta.near_miss_class` + `near_miss_reason` — boundary case annotation
- `meta.annotator_agreement` — unanimous/disputed/borderline

---

## 4. Synthetic Data Generation Pipeline

### Architecture

```
Distribution Monitor (tracks cell coverage in real time)
        ↓
Cell Gap Vector (which taxonomy × domain × difficulty cell is underrepresented)
        ↓
Case Generator (Claude or Codex, prompted with cell specification)
        ↓
Label Validator (opposite model — if Claude generates, Codex validates)
        ↓
Consistency Checker (schema validation, internal signal coherence, taxonomy pattern match)
        ↓
Conflict Resolver (disagreements → human review queue)
        ↓
Distribution Monitor (updated)
```

### Core Principles

- **Generator and validator must never be the same model.** Claude generates, Codex scores — or vice versa. Disagreements flag for human review, never auto-resolved.
- **Taxonomy defines the generation space.** The generator receives a cell specification (taxonomy pattern × domain × difficulty), not an open-ended prompt. This constrains output to structurally checkable cases.
- **Generation targets cells, not volume.** The distribution monitor tracks coverage across all ~432 cells. Full cells stop generating; sparse cells get prioritized. Total count is a byproduct of full coverage, not a target in itself.
- **Borderline cases are first-class.** 20–25% of every expert's data must be near-miss rows at decision boundaries. These teach calibrated uncertainty.
- **Every case has provenance.** Generated to fill cell `{pattern}__{domain}__{difficulty}`. Auditable for benchmark credibility. Version-to-version deltas are explainable by which cells were filled.

### Consistency Checks Per Row

- Governance signals internally coherent (e.g. high ABSTAIN + high `hallucination_pressure`, not high TRUSTWORTHY + high `hallucination_pressure`)
- `taxonomy.pattern` correctly instantiated — does the generated evidence actually exhibit the specified pattern
- `taxonomy.cell_id` matches `routing.expert_fired` + `meta.difficulty` combination
- Routing assignment matches domain label
- Near-miss class differs from classification
- Per-chunk relevance scores consistent with overall `query_evidence_alignment`
- Authority scores consistent with `source_type`
- `conflict_density` consistent with taxonomy class (DISPUTED patterns should have high conflict density, TRUSTWORTHY patterns low)

### What the Monitor Tracks

- **Cell coverage** — count per `taxonomy × domain × difficulty` cell vs minimum threshold (primary signal)
- Expert domain count vs target (per expert)
- Governance class distribution per expert (not just global)
- Difficulty ratio per expert
- Taxonomy pattern distribution per expert — all 18 patterns represented within each domain
- Near-miss ratio per expert
- Query type distribution
- Reasoning type distribution

---

## 5. MoE Architecture Specification

### Target Architecture

```
pyrrho-MoE
├── Total parameters:   4B
├── Active parameters:  0.4B (10:1 sparsity ratio)
├── Quantization:       4-bit (GGUF / llama.cpp compatible)
├── Memory footprint:   ~2GB on disk, ~200MB active weights per pass
├── Inference target:   CPU-only, universally deployable
└── Parallelism:        Per-chunk tasks parallelizable across CPU cores
```

### Expert Domains (7–8 experts)

| Expert | Domains covered | Specialization |
|---|---|---|
| `science_medicine` | Medical, scientific, empirical | High-precision claims, study evidence, statistical data |
| `law_policy` | Legal, regulatory, compliance | Jurisdiction-sensitivity, normative claims, versioned docs |
| `history_geography` | Historical, geographical | Temporal anchoring, factual claims, wrong-specificity ABSTAIN |
| `technology_computing` | Tech, software, AI | Rapidly changing facts, documentation, version sensitivity |
| `economics_finance` | Financial, economic | Quantitative claims, time-sensitivity, source credibility variance |
| `culture_society` | Pop culture, social topics | Subjective claims mixed with factual, natural DISPUTED patterns |
| `general_commonsense` | Cross-domain, simple factual | Catch-all, handles routing failures gracefully |
| `conflict_detection` | All domains | Meta-expert: fires on high inter-source disagreement regardless of domain |

The conflict detection expert is a reasoning pattern expert, not a subject matter expert. It specializes in DISPUTED classification across all domains and directly maps to the `conflict_density` signal.

### Routing

- Primary expert fires per input based on domain classification
- Conflict detection expert can fire as secondary expert alongside any primary
- Routing confidence exposed as `routing.routing_confidence` signal
- Which expert fired is an interpretable governance signal in itself

### Parallelism Model

```
Query arrives
    ├── Query rewriting (1 pass, non-blocking)
    └── Retrieval (concurrent)
            ↓
    Chunks arrive → [chunk 1 ... chunk N]
            ↓ parallel across CPU cores
    Per chunk (all simultaneous):
        - Authority detection
        - Temporality detection  
        - Evidence bias detection
        - Chunk boundary detection
        - Chunk summarization
            ↓ join — all chunk scores available
    Evidence chain construction  (sequential — needs all chunks)
    Governed reranking           (sequential — needs all chunk scores)
            ↓
    Generation gate              (1 pass)
    Uncertainty expression       (1 pass)
            ↓
    LLM generates
            ↓
    Answer grounding verification (parallel per sentence)
```

Effective wallclock for full v1 pass over 5 chunks on 8-core CPU: **~150–400ms** end to end.

### CPU Inference Estimates (4-bit quantization, 8-core, 512 token input)

| Active params | Scoring task | Short generative | Long generative |
|---|---|---|---|
| 0.2B | 8–15ms | 40–80ms | 200–400ms |
| 0.3B | 12–25ms | 60–120ms | 300–600ms |
| **0.4B ★** | **18–35ms** | **80–160ms** | **400–800ms** |
| 0.5B | 25–45ms | 100–200ms | 500ms–1s |
| 0.6B | 30–60ms | 130–260ms | 700ms–1.4s |

0.4B is the sweet spot. Scoring tasks stay under 35ms. Query rewriting (short generative) at 80–160ms is acceptable inline. Chunk summarization runs async to avoid blocking the critical path.

---

## 6. Pyrrho Signal API

The complete output of a pyrrho-MoE inference pass:

```json
{
  "classification": "TRUSTWORTHY",
  "signals": {
    "trustworthy": 0.847,
    "disputed": 0.112,
    "abstain": 0.041,
    "confidence": 0.891,
    "grounding": 0.763,
    "conflict_density": 0.134,
    "evidence_sufficiency": 0.812,
    "boundary_proximity": {
      "nearest_class": "DISPUTED",
      "distance": 0.735
    },
    "domain_familiarity": 0.923,
    "false_trustworthy_risk": 0.087,
    "hallucination_pressure": 0.201,
    "retrieval_retry_value": 0.143,
    "human_escalation_score": 0.044,
    "query_evidence_alignment": 0.881,
    "answer_coverage": 0.794
  },
  "meta": {
    "model": "pyrrho-MoE-g3",
    "inference_ms": 24,
    "expert_fired": "science_medicine",
    "routing_confidence": 0.961,
    "fitz_gov_version": "7.0"
  }
}
```

`false_trustworthy_risk` is never zero — zero would be overconfident. The `meta` block is mandatory for production auditability: two pyrrho generations are not directly comparable without model and benchmark version context.

---

## 7. Feature Scope

### v1 Features (16 total — 4B-A0.4B)

#### Governance core
| Feature | Type | MoE native |
|---|---|---|
| Governance classification (ABSTAIN / TRUSTWORTHY / DISPUTED) | scoring | yes |
| Full pyrrho signal suite (15+ signals) | scoring | yes |
| Generation gate (go / no-go before LLM generates) | scoring | yes |
| Uncertainty expression (instructs LLM how to hedge) | scoring | yes |

#### Retrieval
| Feature | Type | MoE native |
|---|---|---|
| Governed reranking (relevance + trust combined) | scoring | yes |
| Query intent classification (factual / causal / comparative / procedural) | scoring | partial |
| Query rewriting (reformulate for better retrieval) ↑ pulled from v2 | generative | partial |

#### Context
| Feature | Type | MoE native |
|---|---|---|
| Authority detection (source credibility per chunk) | scoring | yes |
| Temporality detection (staleness, anchor period, time-sensitivity) | scoring | yes |
| Evidence bias detection (one-sided sourcing signal) | scoring | yes |
| Chunk boundary detection (flag bad semantic cuts) | scoring | partial |
| Evidence chain construction (logical ordering across chunks) ↑ pulled from v2 | scoring | partial |
| Chunk summarization (compress context before assembly) ↑ pulled from v2 | generative | yes |

#### Generation
| Feature | Type | MoE native |
|---|---|---|
| Answer grounding verification (sentence-level source attribution) | scoring | partial |

#### Agentic
| Feature | Type | MoE native |
|---|---|---|
| Action confidence gating (evidence quality before irreversible action) | scoring | yes |
| Tool selection confidence (evidence-conditional tool routing) | scoring | yes |
| Memory trustworthiness (score stored memories before retrieval) | scoring | yes |

### v2 Features (10 remaining)

| Feature | Layer | Reason deferred |
|---|---|---|
| Query decomposition | retrieval | generative, longer output |
| HyDE generation | retrieval | generative, high size pressure |
| Redundancy detection | context | medium priority |
| Cross-chunk coreference | context | extra architecture head needed |
| Claim extraction | context | generative, medium complexity |
| Citation suggestion | generation | medium priority |
| Factual consistency scoring | generation | extra architecture head needed |
| Loop detection | agentic | medium priority |
| Pipeline health monitoring | agentic | low priority |

---

## 8. Training Path

### Phase 0 — Schema enrichment (fitz-gov V5.1 → V6) — **COMPLETE 2026-05-20**

**Goal:** Retrofit all new schema fields onto existing 2,980 validated cases before generating any new data.

- Map existing 17 domains to 7–8 MoE expert domains → add `routing.expert_fired`
- Derive temporality signals from existing `domain` and `evidence_pattern` fields programmatically
- Bootstrap authority scores from `source_type` and context content
- LLM-assisted annotation for fields requiring reasoning: `query_rewritten`, `near_miss_reason`, `hallucination_pressure`, `retrieval_retry_value`, per-chunk `relevance_to_query`
- Spot-check enriched fields for consistency, do not auto-accept LLM annotations blindly
- Add `evidence_chain` ordering labels for multi-chunk cases
- Add reference `summary` per context chunk

**Output:** 2,980 fully enriched rows in the V6 schema. Published as `yafitzdev/fitz-gov` v6.0.0 on 2026-05-20.

**Why first:** Enrichment teaches you which fields are hard to label consistently before you scale. Discoveries here directly improve synthetic pipeline prompts. Cheaper to learn on 2,900 rows than 10,000.

---

### Phase 1 — pyrrho-nano-g1.1 (apples-to-apples baseline)

**Goal:** Retrain the encoder on fitz-gov V6 for a clean controlled comparison.

- Same ModernBERT architecture as g1
- Trained and evaluated on fitz-gov V6 (fixed dataset)
- Produces three-way comparison: ML classifier vs pyrrho-nano-g1 vs pyrrho-nano-g1.1
- Cleans up version drift in existing benchmark numbers
- Documents evaluation methodology before scaling

**Why:** Current comparisons are muddied by dataset version differences. This establishes a clean baseline the paper can cite.

---

### Phase 2 — Synthetic data pipeline + fitz-gov V7

**Goal:** Scale to 5,000–10,000 cases with taxonomy × domain × difficulty matrix coverage.

- Define final taxonomy: 18 patterns across 3 governance classes (6 per class)
- Map existing V6 cases to taxonomy cells — retroactively assign `taxonomy.cell_id` to all 2,980 rows (already in V6 vault)
- Identify which cells are empty or sparse after mapping — these are the generation targets
- Build distribution monitor: tracks coverage across all ~432 cells (taxonomy × domain × difficulty)
- Build case generator: Claude or Codex, prompted with cell specification `(pattern, domain, difficulty, expert)`
- Build label validator: opposite model from generator, independent scoring
- Build consistency checker: schema validation + signal coherence + taxonomy pattern match verification
- Build conflict resolver: disagreements → human review queue
- Generate toward empty/sparse cells until all cells meet minimum threshold (target: 20–25 examples per cell)
- Maintain 20–25% near-miss / borderline cases per expert
- Add evidence chain construction cases (multi-chunk ordered reasoning) — new evidence pattern not in V6
- Add reference chunk summaries to all cases

**Output:** fitz-gov V7, 5,000–10,000 cases, fully enriched schema, complete taxonomy × domain × difficulty coverage, auditable cell provenance.

---

### Phase 3 — pyrrho-nano-g2 + pyrrho-small-g2

**Goal:** Establish encoder and SLM benchmarks on V7 before MoE training begins.

#### pyrrho-nano-g2
- ModernBERT fine-tune on fitz-gov V7
- Classification only + signal suite
- Establishes V7 accuracy baseline
- Comparison point for MoE paper

#### pyrrho-small-g2
- Generative SLM fine-tune (1–3B dense, e.g. Phi-3-mini, Qwen2.5-1.5B, or Gemma-3-1B)
- Classification + reasoning trace output
- Tests whether rationale generation improves accuracy
- Introduces RL fine-tuning:
  - Reward correct classifications
  - Heavy penalty on false-trustworthy predictions
  - Reward calibrated uncertainty (model uncertain when wrong, confident when right)
  - Tooling: TRL (HuggingFace) with GRPO or simplified PPO
  - Compute: LoRA/QLoRA on consumer hardware

**Key result to document:** Three-way comparison — ML classifier vs pyrrho-nano vs pyrrho-small — across overall accuracy, per-class breakdown, false-trustworthy rate, inference speed, pipeline complexity, and development effort (the 300x effort ratio of ML vs encoder is worth stating explicitly in the paper).

---

### Phase 4 — fitz-gov V8

**Goal:** Scale to 15,000–20,000 cases with adversarial and uncertainty-focused data.

- Continue synthetic pipeline with updated cell targets — increase minimum per cell to 40–50 examples
- Add adversarial variants of existing cells: same taxonomy pattern × domain × difficulty but designed to fool the model on exactly the pattern it should be most reliable on
- Systematically target cells where pyrrho-nano-g2 and pyrrho-small-g2 show lowest accuracy — taxonomy makes this surgical rather than speculative
- Add cases where correct answer requires chaining evidence across chunks in specific order
- Deliberately over-sample borderline cells — cases that sit at the boundary between two taxonomy patterns
- Add `confidence_level` and `near_miss_reason` fields with careful annotation for borderline rows
- Adversarial evaluation set held out — not used for training, only evaluation
- Publish adversarial cell distribution alongside dataset so others can target the same failure modes

**Output:** fitz-gov V8, 15,000–20,000 cases. Adversarial eval set published separately with cell-level breakdown.

---

### Phase 5 — pyrrho-MoE-g3 (train from scratch)

**Goal:** Train the terminal architecture on fitz-gov V8.

#### Architecture decisions
- 4B total / 0.4B active parameters
- 7–8 domain experts + conflict detection meta-expert
- Expert domains defined upfront, routing supervised by `routing.expert_fired` labels in fitz-gov
- Evidential Deep Learning output layer (Dirichlet-based) for native uncertainty quantification — not softmax post-hoc confidence
- 4-bit quantization target for deployment (GGUF compatible)

#### Training stages

**Stage 1 — Supervised multi-task pre-training**
- All 16 v1 output heads trained simultaneously on fitz-gov V8
- Routing loss + per-task losses combined
- Verify expert specialization emerging every few epochs: inspect which experts fire on which domains

**Stage 2 — Reinforcement learning**
- GRPO or PPO via TRL
- Reward signals:
  - Correct governance classification
  - Heavy penalty on false-trustworthy (asymmetric — this is the highest-stakes failure)
  - Calibrated uncertainty reward: penalize confident wrong predictions, reward uncertain correct ones
  - Governance-aware query rewrite quality: does the rewrite improve simulated retrieval precision
  - Grounding faithfulness: does generated summary stay within chunk content
- RL shapes risk aversion directly — more principled than adding labeled data for edge cases

**Stage 3 — Adversarial hardening**
- Fine-tune on adversarial eval set from V8
- Focus on false-trustworthy rate specifically
- Re-evaluate on held-out adversarial cases

#### Validation during training
- Routing analysis every N epochs: are experts specializing by domain
- Per-expert accuracy tracked separately — global accuracy can mask individual expert failure
- False-trustworthy rate tracked as primary metric throughout, not overall accuracy
- Uncertainty calibration tracked: ECE (Expected Calibration Error) on borderline cases

---

### Phase 6 — fitz-gov V9+ and pyrrho-MoE-g4

**Goal:** Infrastructure-grade reliability.

- 30,000+ cases
- Adversarial evaluation becomes primary quality signal, not dataset size
- False-trustworthy rate approaches floor
- Routing stable enough to expose expert firing as a first-class downstream signal
- Consider: custom tokenizer or domain-specific pretraining corpus before fine-tuning for further gains
- Consider: v2 feature integration (query decomposition, cross-chunk coreference, factual consistency)

---

## 9. Evaluation Metrics

### Primary (always tracked)
- `false_trustworthy_rate` — the metric that matters most; a confident wrong TRUSTWORTHY is the highest-stakes failure
- Per-class accuracy: ABSTAIN, TRUSTWORTHY, DISPUTED separately — global accuracy masks class-level failures
- Overall accuracy on fitz-gov benchmark

### Secondary
- ECE (Expected Calibration Error) on borderline cases — measures uncertainty calibration quality
- Expert routing accuracy — does the router fire the right expert per domain
- Per-expert accuracy breakdown — global balance can hide individual expert weakness
- Inference time on standard hardware (document: CPU, 4-bit, 8-core, 512 token input)

### Paper comparison dimensions
- Overall accuracy
- Per-class breakdown
- False-trustworthy rate
- Inference speed (ms)
- CPU deployability (yes/no)
- Pipeline complexity (lines of code / components)
- Development effort (qualitative — the 300x effort ratio is worth documenting)

---

## 10. Publication Strategy

### Artifacts to publish (in order)

1. **fitz-gov on HuggingFace** — canonical home, dataset card, version changelog, split statistics, distribution dashboard. GitHub repo private or archived — HuggingFace is the single source of truth.

2. **pyrrho model family on HuggingFace** — model cards with benchmark numbers, per-class accuracy, false-trustworthy rate, inference benchmarks per hardware class.

3. **arXiv preprint** — not a traditional ML paper. Self-published, citable, indexed. Target: after pyrrho-nano-g2 and pyrrho-small-g2 are published, before MoE training begins. Core argument: RAG systems have no lightweight CPU-native governance signal. Here is the problem definition, the benchmark, the model family, the numbers, and why false-trustworthy is the metric that matters most.

4. **pyrrho-MoE paper** — the capstone. "Pyrrho: A Sparse Mixture-of-Experts Runtime for Trustworthy Retrieval-Augmented Generation." Complete system description, three-way architecture comparison, full v1 capability suite, CPU inference benchmarks, adversarial evaluation results.

### Leaderboard
Publish a fitz-gov leaderboard on HuggingFace where third parties can submit their own classifiers. This converts fitz-gov from a dataset into a benchmark in the community's eyes — the same transition that makes a personal project into infrastructure.

---

## 11. Broader Context

### Why this matters beyond research

- **EU AI Act compliance:** High-risk AI systems require transparency, human oversight, and documented reliability. Pyrrho is an auditable evidence quality layer that compliance teams can point to with published accuracy numbers. Not using it becomes a conscious choice rather than a gap.
- **Agentic systems:** Any agent that takes actions needs to know when it has sufficient trustworthy evidence to act. Pyrrho is the decision gate that currently does not exist cheaply.
- **Air-gapped / edge deployment:** Medical, legal, automotive, defense — industries that cannot send data to an API and cannot provision GPUs. CPU-native governance that actually works is rare in these contexts.
- **The infrastructure trajectory:** Reliable enough + accessible enough = people build on top of it. The same arc seatbelts and code linters went through. First optional best practice, then expected, then required. fitz-gov defines what "good" looks like; pyrrho is the reference implementation.

### The compounding flywheel
fitz-gov credibility → pyrrho adoption → usage generates feedback → feedback improves labels → labels improve next generation → stronger credibility. The dataset and the model reinforce each other. That flywheel, once turning, is self-funding.
