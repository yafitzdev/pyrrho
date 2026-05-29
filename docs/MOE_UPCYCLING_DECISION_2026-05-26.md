# MoE Upcycling Decision — 2026-05-26

## Decision

Use `Qwen/Qwen3-0.6B-Base` as the first dense upcycling seed, with a repaired
Qwen-aligned alpha config:

- Config: `configs/moe/pyrrho_moe_g3_alpha_qwen.yaml`
- Seed: `Qwen/Qwen3-0.6B-Base`
- Layers: 28
- Hidden size: 1024
- Attention heads / KV heads / head dim: 16 / 8 / 128
- Vocab: 151,936, tied embeddings
- Dense FFN layers: 4
- MoE FFN layers: 24
- Physical experts per MoE layer: 48
- Semantic expert groups: 8, with 6 physical shards/group
- Expert FFN dim: 1056
- Routing: top-1

Parameter count from `scripts/analyze_moe_seed_budget.py`:

| Variant | Total | Active inclusive | Active excl. embedding | Result |
|---|---:|---:|---:|---|
| Canonical 64k baseline | 3.951B | 0.412B | 0.346B | Budget pass, not seed-tokenizer aligned |
| Qwen vocab + old shape + real attention | 4.129B | 0.590B | 0.435B | Fails active inclusive |
| Qwen exact trunk + FFN 3072 + real attention | 3.994B | 0.596B | 0.441B | Fails active inclusive |
| Former Qwen alpha, 24 experts, FFN 2112 | 4.096B | 0.514B | 0.359B | Invalid after real-weight inspection |
| **Repaired Qwen alpha, 48 experts, FFN 1056** | **4.083B** | **0.424B** | **0.268B** | **Selected** |
| Alternate Qwen alpha, 48 experts, FFN 1024 | 3.970B | 0.421B | 0.266B | Budget pass, lower capacity |

## Rationale

The old 64k-tokenizer assumption made the original 4B/A0.4B arithmetic work,
but it blocks clean embedding upcycling from Qwen. Keeping Qwen's tokenizer and
embedding matrix preserves more pretrained competence and avoids introducing a
custom tokenizer/retraining path before the sparse trunk exists.

The first Qwen-aligned alpha also missed one real seed detail: Qwen3-0.6B has an
explicit `head_dim=128`, so its attention tensors are `q_proj=2048x1024`,
`k_proj/v_proj=1024x1024`, and `o_proj=1024x2048`. The old counter inferred
`head_dim=64` from `hidden_size / attention_heads`, understating shared active
attention parameters by roughly 88M. The real-weight smoke test caught this
before any 4B checkpoint was materialized.

The repaired alpha restores the budget by:

- Keeping Qwen's 28-layer / 1024-hidden / KV=8 / head-dim=128 / vocab shape.
- Increasing physical experts from 24 to 48 per MoE layer, preserving 4B-class
  total capacity after shrinking the selected expert FFN.
- Shrinking expert FFN dim from 3072 to 1056, keeping active inclusive at
  **0.424B**, inside the A0.38B-A0.43B window.
- Matching the V8 MoE metadata's 15 scalar targets. This changes only 3,075
  task-head parameters, but keeps the config, data, and wrapper consistent.

## Upcycling Implication

This is not a pure FFN clone. Qwen's dense FFN dim is 3072; the selected expert
FFN dim is 1056. The upcycling script initializes experts by selecting the
strongest seed FFN channels by combined gate/up/down norm, then slicing
`gate_proj`, `up_proj`, and `down_proj` consistently.

Attention, embeddings, tokenizer, final norm, and Q/K attention norms can be
copied directly. Routers and task heads are new and trained from fitz-gov V8.

## Verification

`scripts/upcycle_dense_to_moe.py --inspect-only --real-weight-smoke` now checks
real Qwen safetensors keys and tensors. On 2026-05-26 it passed on layer 2:

- Seed FFN shapes: `3072x1024`, `3072x1024`, `1024x3072`
- Target compressed shapes: `1056x1024`, `1056x1024`, `1024x1056`
- Selected channels: 1056 of 3072
- Output: `outputs/moe/upcycling/qwen_alpha_real_weight_smoke.json`

`scripts/upcycle_dense_to_moe.py --write-seed-pack
outputs/moe/upcycling/qwen_alpha_seed_pack` also materialized the first local
seed pack:

- 30 safetensors shards
- 310 tensors
- 8.166 GB total tensor bytes
- Qwen3-MoE-compatible tensor names (`mlp.experts.gate_up_proj`,
  `mlp.experts.down_proj`, `mlp.gate.weight`)
- Shape validation passed against a meta-initialized `Qwen3MoeForCausalLM`
- `lm_head.weight` is intentionally omitted because embeddings are tied

The first governance-wrapper smoke also passed:

- Script: `scripts/smoke_moe_qwen_wrapper.py`
- Output: `outputs/moe/upcycling/qwen_alpha_wrapper_smoke.json`
- Device / dtype: CUDA / bfloat16
- Batch: 2 V8 test rows, max length 64
- Shapes: governance `[2,3]`, route `[2,8]`, taxonomy `[2,23]`, scalar `[2,15]`
- Multitask loss computed with no-training random heads: `11.4245`

The first heads-only Stage 1 training smoke passed after that:

- Script: `scripts/train_moe_qwen_heads.py`
- Output: `outputs/moe/qwen_heads_stage1_smoke/train_report.json`
- Trainable params: 50,225 pyrrho head parameters
- Internal Qwen3-MoE router params present but frozen by default: 1,179,648
- Smoke: 2 optimizer steps on 4 train rows, eval on 4 rows, CUDA / bfloat16

## Rejected Paths

- **Custom 64k tokenizer now:** preserves the original spec but gives up clean
  embedding upcycling and creates a tokenizer-training project before there is a
  working 4B skeleton.
- **Accept Qwen vocab and relax A0.4B:** violates the CPU-runnable target.
- **Keep the 24-expert / 2112-FFN alpha:** invalid after counting Qwen's real
  `head_dim=128` attention tensors; active inclusive becomes roughly 0.514B.
- **SmolLM2 2048-wide alpha now:** possible, but it changes the hidden-size
  target more drastically and makes the current 1024-wide architecture math less
  relevant. Keep it as fallback if Qwen FFN compression fails.
