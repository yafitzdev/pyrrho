"""Run llama.cpp's HF-to-GGUF converter with pyrrho Qwen3-MoE fixes.

The local pyrrho Qwen3-MoE seed pack uses Qwen's `mlp_only_layers`, so a few
blocks contain dense FFN tensors (`mlp.gate_proj`, `mlp.up_proj`,
`mlp.down_proj`) instead of expert tensors. Some llama.cpp converter builds
support Qwen3-MoE but omit those dense FFN tensors from the Qwen3-MoE tensor
allow-list. This wrapper patches that table at runtime and then delegates to
the upstream converter without editing the llama.cpp checkout.

Example:
    python scripts/convert_qwen3moe_hf_to_gguf.py \
      --llama-cpp C:/Users/yanfi/.unsloth/llama.cpp \
      --outtype bf16 \
      --outfile outputs/moe/gguf/pyrrho-MoE-g3-seed-bf16.gguf \
      outputs/moe/upcycling/qwen_alpha_seed_pack
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llama-cpp", type=Path, default=Path(r"C:/Users/yanfi/.unsloth/llama.cpp"))
    args, converter_args = parser.parse_known_args()
    args.converter_args = converter_args
    if args.converter_args and args.converter_args[0] == "--":
        args.converter_args = args.converter_args[1:]
    if not args.converter_args:
        parser.error("pass convert_hf_to_gguf.py arguments after --llama-cpp")
    return args


def patch_qwen3moe_dense_mlp(llama_cpp: Path) -> None:
    sys.path.insert(0, str(llama_cpp.resolve()))
    sys.path.insert(1, str((llama_cpp / "gguf-py").resolve()))
    import gguf

    tensors = gguf.MODEL_TENSORS[gguf.MODEL_ARCH.QWEN3MOE]
    needed = [
        gguf.MODEL_TENSOR.FFN_GATE,
        gguf.MODEL_TENSOR.FFN_DOWN,
        gguf.MODEL_TENSOR.FFN_UP,
    ]
    if all(tensor in tensors for tensor in needed):
        return

    insert_after = gguf.MODEL_TENSOR.FFN_NORM
    insert_at = tensors.index(insert_after) + 1 if insert_after in tensors else len(tensors)
    for tensor in reversed(needed):
        if tensor not in tensors:
            tensors.insert(insert_at, tensor)


def main() -> None:
    args = parse_args()
    llama_cpp = args.llama_cpp.resolve()
    converter = llama_cpp / "convert_hf_to_gguf.py"
    if not converter.exists():
        raise FileNotFoundError(converter)

    patch_qwen3moe_dense_mlp(llama_cpp)
    sys.argv = [str(converter), *args.converter_args]
    runpy.run_path(str(converter), run_name="__main__")


if __name__ == "__main__":
    main()
