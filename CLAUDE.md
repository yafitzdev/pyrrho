# CLAUDE.md — pyrrho project instructions

Loaded into every Claude Code session that opens this directory. Keep it short and high-signal.

## What pyrrho is

Fine-tuned classification models for RAG governance. Given a `(query, retrieved_contexts)` pair, predicts one of `ABSTAIN / DISPUTED / TRUSTWORTHY`. Drop-in replacement for the constraint+sklearn pipeline in `fitz-sage`. Benchmarked against `fitz-gov` V6 (V5.1 + LLM-enriched schema overlay, 2,980 cases).

## Read these in order on any fresh session

1. **[docs/HANDOFF.md](docs/HANDOFF.md)** — current state snapshot. Always reflects "what's true right now." Read this first.
2. **[docs/ROADMAP.md](docs/ROADMAP.md)** — current vision: model tiers (`pyrrho-nano` / `-small` / `-MoE`), fitz-gov data path (V5.1 → V6 → V7 → V8), training phases through pyrrho-MoE. Supersedes PROJECT.md §10 release table.
3. **[docs/LOG.md](docs/LOG.md)** — append-only project history. Read when HANDOFF mentions something you don't have context on.
4. **[docs/PROJECT.md](docs/PROJECT.md)** — original vision document. Still load-bearing for §1–§9, §11–§18. Section §10 release table is superseded by ROADMAP.md.
5. **[docs/METHODOLOGY.md](docs/METHODOLOGY.md)** — the 8-step pipeline every release follows.
6. **[docs/SETUP.md](docs/SETUP.md)** — RTX 5090 / Blackwell / Windows specifics.
7. **[docs/INDEX.md](docs/INDEX.md)** — index of all docs with reading order.

---

## Log and handoff convention — IMPORTANT

This project deliberately separates **current state** from **history**:

- `docs/HANDOFF.md` is the **snapshot**. It gets *overwritten*. It should always answer: "what's true right now, what's next."
- `docs/LOG.md` is the **history**. It is *append-only*. Past entries are never edited. New entries go *at the top*.

### When to update HANDOFF.md

After **any** state change that affects what a fresh session would need to know:

- A new model release has been trained → update the family-status table and validated-metrics table.
- The immediate next action changes → update the "Immediate next actions" section.
- A new known limitation is discovered → add to "Known limitations of v1" (or current release section).
- A previously-blocked decision is resolved → remove it from "Things NOT to do (already decided)" if no longer relevant, or add new entries when new decisions land.
- New scripts or tooling shipped → update the "Pipeline / tooling" table.

**Overwrite, don't append.** HANDOFF.md must stay concise — if a section grows past ~10 lines, the historical detail belongs in LOG.md instead.

### When to append a LOG.md entry

After **any concrete deliverable**:

- A training run finished (success or failure — both produce learnings).
- A piece of code shipped (a new script, module, config, doc).
- A decision was made that changes the plan.
- An experiment surfaced a finding that wasn't expected.
- A piece of infrastructure was rebuilt or removed.

### LOG.md entry format

New entries go at the **top** of the file (most recent first). Each entry follows:

```markdown
## YYYY-MM-DD (morning|afternoon|evening) — Short title

**What landed:**
- Concrete deliverable 1
- Concrete deliverable 2

**What was learned:**
- Surprise, validation, or new constraint that wasn't obvious before
- Numbers/file paths to anchor the claim

**Next:** one sentence on the implied next step at the time of writing.

---
```

Don't edit past entries — if an earlier finding is later contradicted, write a new entry that supersedes it. Truth in this file is "this is what we believed at this time," not "this is final truth."

### Don't ask permission to update logs

If meaningful work happened in your session, **update LOG.md and HANDOFF.md before ending the turn.** Don't ask the user for permission to log. Don't wait until the session ends. Treat it like running tests — part of finishing the work.

---

## Hard constraints — do not relitigate

These were decided in earlier sessions. Don't re-propose alternatives unless explicitly asked.

- **Brand**: `pyrrho` (after Pyrrho of Elis). Don't suggest renames.
- **Production track**: encoder only, must run on CPU. No generative SLMs in `fitz-sage`'s default path.
- **Portfolio track**: generative SLMs, all CPU-runnable (≤8 GB RAM at Q4). No 35B+ MoE bases.
- **Model bases**: 2026-vintage permissive bases. No Qwen 2.5 (stale), no Llama family (license-restrictive).
- **License (pyrrho + fitz-gov)**: CC BY-NC 4.0. Don't suggest re-permissive licensing. Commercial use requires a separate license; research/eval/personal use is free.
- **Benchmark**: fitz-gov V6 (V5.1-enriched, shipped 2026-05-20 as `yafitzdev/fitz-gov` v6.0.0). pyrrho-nano-g1.1 is the V6 retrain per ROADMAP.md Phase 1. The next major dataset version (scaled to 5K–10K cases via SDGP) is V7 — no V7 data work until pyrrho v2 ships.
- **HF naming**: `yafitzdev/pyrrho-{tier}-{generation}` where tier ∈ {nano, small, MoE} and generation ∈ {g1, g1.1, g2, ...}. Live model is `pyrrho-nano-g1`. No `fitz-` prefix on model names. (Was `yafitzdev/pyrrho-{base-model}-{size}-v{n}` until 2026-05-20.)
- **Dataset distribution**: fitz-gov dataset rows (`data/`) live on HuggingFace only — `yafitzdev/fitz-gov`. Gitignored in the fitz-gov repo. Don't suggest re-committing.
- **Release gates**: overall accuracy ≥ 78.7% AND false-trustworthy ≤ 5.7%, mean across 3 seeds. The tier0 95% sanity gate was dropped (unreachable on N=60 with ~5 ambiguous labels).

Full rationale for each: PROJECT.md §18.

## Working style preferences

- **Terse, concrete, no hand-waving.** Lead with the answer. Use real numbers and file paths.
- **No emojis** in code, configs, or docs unless the user explicitly asks.
- **Don't over-ask for confirmation.** When HANDOFF.md or PROJECT.md names the next step, just do it. Reserve confirmation requests for destructive or hard-to-reverse actions.
- **Always web-search current model state** before recommending a base model. Never rely on training priors past ~3 months stale — this project has been burned by it.
- **Run the smoke test (`pytest tests/test_smoke.py`)** after any code change that touches training, eval, or inference paths. It runs in 7.5 s and catches regressions on the 10 handcrafted cases.

## Common commands

```powershell
# From the project root, .venv activated
python scripts/prepare_data.py --fitz-gov ../fitz-gov/data --output data/processed   # one-time
python scripts/verify_env.py                                                          # env sanity
python scripts/train_encoder.py --config configs/encoder/modernbert_base.yaml --no-wandb
python scripts/run_seeds.py --seeds 42 1337 7                                         # multi-seed validation
python scripts/sweep.py --grid configs/sweep_grids/encoder_v1.yaml                    # hyperparameter sweep
python scripts/eval_report.py --checkpoint <path>                                     # full breakdown
python scripts/compare_runs.py baseline outputs/multi_seed/<run>/summary.json         # diff vs baseline
pytest tests/test_smoke.py -v                                                         # smoke test
```

## Related repos

| Path | Role |
|---|---|
| `C:/Users/yanfi/PycharmProjects/fitz-sage` | Production RAG library that uses pyrrho models |
| `C:/Users/yanfi/PycharmProjects/fitz-gov` | Benchmark dataset (the eval contract) |

The three projects form a triangle: fitz-gov defines the eval, pyrrho produces the models, fitz-sage consumes them.

## Memory directory

Persistent cross-session memory lives at:
`C:\Users\yanfi\.claude\projects\C--Users-yanfi-PycharmProjects-pyrrho\memory\`

It contains user-role, response-style, and reference memories. Read `MEMORY.md` there to see the index. Update memory files when something durable changes about the user, the project, or the working agreements — but for in-project state, **prefer HANDOFF.md and LOG.md** over memory files.
