"""Merge the Qwen pyrrho-MoE PEFT adapter into a plain HF model directory.

This is a release-runtime preparation step: llama.cpp conversion expects normal
HF weights, while the trained local MVP is a PEFT adapter over the upcycled
Qwen3-MoE seed pack.

Example:
    python scripts/materialize_moe_qwen_sft_merged.py \
      --seed-pack outputs/moe/upcycling/qwen_alpha_seed_pack \
      --adapter-path models/pyrrho-MoE-g3-mvp/adapter \
      --output-dir outputs/moe/pyrrho_moe_g3_mvp_merged_hf
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def load_training_module() -> Any:
    path = Path(__file__).with_name("train_moe_qwen_sft.py")
    spec = importlib.util.spec_from_file_location("pyrrho_qwen_sft_training", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import training helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed-pack", type=Path, default=Path("outputs/moe/upcycling/qwen_alpha_seed_pack"))
    parser.add_argument("--adapter-path", type=Path, default=Path("models/pyrrho-MoE-g3-mvp/adapter"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device-map", default="none", help='Use "none" for a plain CPU load or "auto" for Accelerate.')
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-shard-size", default="2GB")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_loader_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        seed_pack=args.seed_pack,
        dtype=args.dtype,
        device_map=args.device_map,
        quantization="none",
        bnb_4bit_quant_type="nf4",
        bnb_4bit_double_quant=True,
        attn_implementation=args.attn_implementation,
        adapter_path=args.adapter_path,
        eval_only=True,
        lora_r=0,
        lora_alpha=0,
        lora_dropout=0.0,
        lora_target_modules="",
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    helpers = load_training_module()
    model, tokenizer = helpers.load_model_and_tokenizer(build_loader_args(args))
    helpers.disable_router_aux_outputs(model)
    model.eval()

    if not hasattr(model, "merge_and_unload"):
        raise TypeError("loaded model is not a PEFT model with merge_and_unload()")
    merged = model.merge_and_unload()
    helpers.disable_router_aux_outputs(merged)
    merged.config.use_cache = True
    merged.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(output_dir)

    for filename in ("pyrrho_moe_config.json", "upcycling_manifest.json", "load_shape_report.json"):
        source = args.seed_pack / filename
        if source.exists():
            shutil.copy2(source, output_dir / filename)

    report = {
        "ok": True,
        "seed_pack": str(args.seed_pack),
        "adapter_path": str(args.adapter_path),
        "output_dir": str(output_dir),
        "dtype": args.dtype,
        "device_map": args.device_map,
        "max_shard_size": args.max_shard_size,
        "elapsed_seconds": time.perf_counter() - started,
    }
    with (output_dir / "merge_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
