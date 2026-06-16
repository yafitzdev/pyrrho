"""Export a pyrrho OLMoE structural checkpoint for stock runtime proof.

This writes random weights. It is for architecture/loadability proof only, not
for training quality.

Example:
    python scripts/export_olmoe_structural_checkpoint.py \
      --config configs/moe/pyrrho_moe_g4_real_olmoe_stock_runtime.yaml \
      --output-dir outputs/moe/g4_real_stock_runtime_carrier/olmoe_g4_real_full_random_hf \
      --tokenizer-source outputs/moe/g4_real_stock_runtime_carrier/olmoe_tokenizer_seed
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

from pyrrho.moe import PyrrhoMoEConfig

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/moe/pyrrho_moe_g4_real_olmoe_stock_runtime.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination HF checkpoint directory.",
    )
    parser.add_argument(
        "--tokenizer-source",
        type=Path,
        default=Path("outputs/moe/g4_real_stock_runtime_carrier/olmoe_tokenizer_seed"),
        help="Directory containing OLMoE tokenizer files.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
        help="Tensor dtype to materialize on disk. float16 is enough for load proof.",
    )
    parser.add_argument("--max-shard-size", default="1GB")
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> tuple[dict[str, Any], PyrrhoMoEConfig]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    arch_raw = raw.get("architecture") or {}
    cfg = PyrrhoMoEConfig.from_mapping(arch_raw)
    carrier = str(arch_raw.get("runtime_carrier") or raw.get("runtime_gate", {}).get("selected_carrier") or "")
    model_type = str(arch_raw.get("model_type") or raw.get("runtime_gate", {}).get("selected_model_type") or "")
    if carrier != "OlmoeForCausalLM" or model_type != "olmoe":
        raise ValueError("config must target OlmoeForCausalLM / model_type olmoe")
    if cfg.dense_ffn_layers != 0 or cfg.moe_ffn_layers != cfg.layers:
        raise ValueError("OLMoE structural proof expects every FFN layer to be sparse MoE")
    if cfg.top_k != 1:
        raise ValueError("OLMoE structural proof expects top_k=1")
    if cfg.kv_heads != cfg.attention_heads:
        raise ValueError("stock OLMoE proof requires kv_heads == attention_heads")
    if cfg.tied_embeddings:
        raise ValueError("stock OLMoE proof requires untied output embeddings")
    return raw, cfg


def copy_tokenizer_files(source: Path, output_dir: Path) -> list[str]:
    copied: list[str] = []
    for name in TOKENIZER_FILES:
        src = source / name
        if src.exists():
            shutil.copy2(src, output_dir / name)
            copied.append(name)
    missing = sorted(set(TOKENIZER_FILES[:3]) - set(copied))
    if missing:
        raise FileNotFoundError(f"missing required tokenizer files in {source}: {missing}")
    return copied


def main() -> int:
    args = parse_args()
    raw, cfg = load_config(args.config)
    budget = cfg.budget_report()
    if not all(budget["budget_checks"].values()):
        raise ValueError("config does not pass pyrrho 4B/A0.4B budget checks")

    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} exists; pass --overwrite to replace it")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import OlmoeConfig, OlmoeForCausalLM

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    old_default_dtype = torch.get_default_dtype()

    torch.manual_seed(args.seed)
    try:
        torch.set_default_dtype(dtype)
        hf_cfg = OlmoeConfig(
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.ffn_dim,
            num_hidden_layers=cfg.layers,
            num_attention_heads=cfg.attention_heads,
            num_key_value_heads=cfg.kv_heads,
            num_experts=cfg.experts_per_moe_layer,
            num_experts_per_tok=cfg.top_k,
            tie_word_embeddings=cfg.tied_embeddings,
            max_position_embeddings=int((raw.get("data") or {}).get("max_seq_length") or 4096),
            bos_token_id=None,
            eos_token_id=50279,
            pad_token_id=1,
            norm_topk_prob=False,
        )
        hf_cfg.architectures = ["OlmoeForCausalLM"]
        hf_cfg.dtype = args.dtype
        model = OlmoeForCausalLM(hf_cfg)
    finally:
        torch.set_default_dtype(old_default_dtype)

    parameter_count = sum(param.numel() for param in model.parameters())
    model.save_pretrained(args.output_dir, safe_serialization=True, max_shard_size=args.max_shard_size)
    copied_tokenizer_files = copy_tokenizer_files(args.tokenizer_source, args.output_dir)

    report = {
        "schema_version": "pyrrho_olmoe_structural_export_v1",
        "status": "pass",
        "not_a_trained_model": True,
        "config": str(args.config),
        "output_dir": str(args.output_dir),
        "tokenizer_source": str(args.tokenizer_source),
        "copied_tokenizer_files": copied_tokenizer_files,
        "dtype": args.dtype,
        "seed": args.seed,
        "parameter_count": parameter_count,
        "pyrrho_budget": budget["parameters"],
        "hf_config": hf_cfg.to_dict(),
    }
    report_path = args.output_dir / "pyrrho_structural_export_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
