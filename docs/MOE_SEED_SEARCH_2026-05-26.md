# MoE Seed Search — 2026-05-26

Purpose: first-pass dense-seed scan for `pyrrho-MoE-g3-alpha` upcycling. This
does not lock the final seed; it records the current public model state before
implementation starts.

## Constraints

- License compatible with pyrrho publication; Apache-2.0/MIT preferred.
- Dense SLM seed, not an off-the-shelf MoE renamed as pyrrho.
- Close to baseline architecture: hidden size near 1024, 20-28 layers, SwiGLU,
  GQA preferred.
- CPU-runnable final artifact remains custom 4B total / 0.4B active.

## HF API Findings

| Model | Public | License | Shape | Fit |
|---|---:|---|---|---|
| [`Qwen/Qwen3-0.6B-Base`](https://huggingface.co/Qwen/Qwen3-0.6B-Base) | yes | Apache-2.0 | hidden 1024, 28 layers, FFN 3072, 16 heads / 8 KV, vocab 151,936 | Best structural seed candidate; tokenizer size exceeds 64k baseline budget. |
| [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) | yes | Apache-2.0 | hidden 1024, 28 layers, FFN 3072, 16 heads / 8 KV, vocab 151,936 | Instruct/post-trained variant; useful for behavior distillation, less clean for upcycling than Base. |
| [`HuggingFaceTB/SmolLM2-1.7B`](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B) | yes | Apache-2.0 | hidden 2048, 24 layers, FFN 8192, 32 heads / 32 KV, vocab 49,152 | Good license/tokenizer; width mismatch means alpha dimensions would need to change. |
| [`HuggingFaceTB/SmolLM3-3B-Base`](https://huggingface.co/HuggingFaceTB/SmolLM3-3B-Base) | yes | Apache-2.0 | hidden 2048, 36 layers, FFN 11008, 16 heads / 4 KV, vocab 128,256 | Strong teacher/alternate architecture candidate; too wide/deep for current alpha. |
| [`allenai/OLMo-2-0425-1B-Instruct`](https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct) | yes | Apache-2.0 | hidden 2048, 16 layers, FFN 8192, 16 heads / 16 KV, vocab 100,352 | Clean license; shape is less compatible and context is short. |
| [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it) | yes | Apache-2.0 | config uses Gemma-4 nested/multimodal fields | Better teacher/proxy than upcycling seed until text-only weight layout is inspected. |
| [`microsoft/Phi-4-mini-instruct`](https://huggingface.co/microsoft/Phi-4-mini-instruct) | yes | MIT | hidden 3072, 32 layers, FFN 8192, vocab 200,064 | Teacher candidate, not structurally close. |
| [`LiquidAI/LFM2.5-1.2B-Instruct`](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) | yes | other | hidden 2048, 16 layers, FFN 12288, hybrid LFM2 | Proxy/teacher only unless license terms and hybrid upcycling path are accepted. |

The older `Qwen/Qwen3.5-0.8B-Instruct` and `Qwen/Qwen3.5-2B-Instruct` IDs
returned HTTP 401 from the HF API in this environment, so they are not treated
as public seed candidates for this pass.

## Preliminary Decision

Use [`Qwen/Qwen3-0.6B-Base`](https://huggingface.co/Qwen/Qwen3-0.6B-Base) as the
first upcycling-seed candidate because it is the only checked public dense model
that matches the 1024-wide baseline. Before actual upcycling, resolve the
tokenizer/embedding budget issue:

- Keep the pyrrho 64k tokenizer assumption and distill/remap embeddings, or
- Accept the Qwen tokenizer and revise the inclusive active budget, or
- Adjust the alpha dimensions around a 2048-wide seed such as SmolLM2.

For now, code and parameter counting stay tokenizer-neutral at the canonical
64k baseline.

## Superseding Decision

The tokenizer/embedding issue was resolved later on 2026-05-26. See
[`MOE_UPCYCLING_DECISION_2026-05-26.md`](MOE_UPCYCLING_DECISION_2026-05-26.md).

Selected first upcycling config:
`configs/moe/pyrrho_moe_g3_alpha_qwen.yaml`, using Qwen's full tokenizer/vocab,
28 layers, KV=8, explicit `head_dim=128`, 24 MoE layers, 48 experts/layer, and
FFN dim 1056. Count with 15 V8 scalar heads:
**4.083139633B total / 0.423871537B active inclusive**.

Note: an earlier 24-expert / 2112-FFN alpha was invalidated by real-weight
inspection. Qwen3-0.6B's attention tensors use `q_proj=2048x1024`, not
`1024x1024`, so the counter now models explicit head dimensions.
