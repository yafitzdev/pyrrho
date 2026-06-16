"""Initialize the g4-real OLMoE carrier from an OLMoE donor checkpoint.

This is an explicit shape-transplant experiment, not a claim that the donor
directly fits the target. The target shape remains the stock-loadable
`pyrrho-MoE-g4-real` OLMoE carrier; donor tensors are resized deterministically
where dimensions differ.

Example:
    python scripts/init_olmoe_from_donor.py \
      --donor outputs/moe/g4_real_olmoe_training_path/donors/OLMoE-1B-7B-0924 \
      --config configs/moe/pyrrho_moe_g4_real_olmoe_stock_runtime.yaml \
      --output-dir outputs/moe/g4_real_olmoe_donor_init/olmoe_1b7b_block_resize_hf \
      --dtype float16 \
      --overwrite
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from pyrrho.moe import PyrrhoMoEConfig

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--donor", required=True, help="Local donor checkpoint path or HF repo id.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/moe/pyrrho_moe_g4_real_olmoe_stock_runtime.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--max-shard-size", default="1GB")
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not download donor files through Transformers.",
    )
    parser.add_argument(
        "--resize-strategy",
        choices=("block-mean", "head-topnorm-slice"),
        default="block-mean",
        help=(
            "How to shrink donor hidden axes. block-mean preserves the original behavior; "
            "head-topnorm-slice keeps the strongest donor hidden channels per attention head."
        ),
    )
    return parser.parse_args()


def load_target_config(path: Path) -> tuple[dict[str, Any], PyrrhoMoEConfig]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    arch = raw.get("architecture") or {}
    cfg = PyrrhoMoEConfig.from_mapping(arch)
    if arch.get("runtime_carrier") != "OlmoeForCausalLM" or arch.get("model_type") != "olmoe":
        raise ValueError("target config must be OlmoeForCausalLM / model_type olmoe")
    if cfg.kv_heads != cfg.attention_heads or cfg.tied_embeddings:
        raise ValueError("target config must keep the proven stock OLMoE constraints")
    return raw, cfg


def build_head_topnorm_hidden_index(donor: Any, target_cfg: PyrrhoMoEConfig) -> Any:
    """Select a consistent hidden-axis slice while preserving per-head width."""
    import torch

    source_hidden = int(donor.config.hidden_size)
    target_hidden = int(target_cfg.hidden_size)
    source_heads = int(donor.config.num_attention_heads)
    target_heads = int(target_cfg.attention_heads)
    if source_heads != target_heads:
        raise ValueError(
            "head-topnorm-slice requires donor/target attention head counts to match "
            f"({source_heads} != {target_heads})"
        )
    if source_hidden % source_heads != 0 or target_hidden % target_heads != 0:
        raise ValueError("hidden sizes must divide evenly by attention heads")
    source_head_dim = source_hidden // source_heads
    target_head_dim = target_hidden // target_heads
    if target_head_dim > source_head_dim:
        raise ValueError("head-topnorm-slice only supports shrinking hidden axes")

    embeddings = donor.get_input_embeddings().weight.detach().float().cpu()
    importance = embeddings.square().sum(dim=0)
    chosen: list[Any] = []
    for head_idx in range(source_heads):
        start = head_idx * source_head_dim
        head_scores = importance[start : start + source_head_dim]
        local = torch.topk(head_scores, k=target_head_dim, largest=True).indices.sort().values
        chosen.append(local + start)
    return torch.cat(chosen).long()


def resize_axis(
    tensor: Any,
    dim: int,
    target_size: int,
    *,
    hidden_index: Any | None = None,
    source_hidden_size: int | None = None,
) -> Any:
    """Resize one tensor axis deterministically by block-mean or nearest repeat."""
    import torch

    source_size = int(tensor.shape[dim])
    if source_size == target_size:
        return tensor
    if (
        hidden_index is not None
        and source_hidden_size is not None
        and source_size == int(source_hidden_size)
        and target_size == int(hidden_index.numel())
    ):
        return tensor.index_select(dim, hidden_index.to(device=tensor.device)).contiguous()
    if source_size > target_size and source_size % target_size == 0:
        factor = source_size // target_size
        moved = tensor.movedim(dim, 0)
        shape = (target_size, factor, *moved.shape[1:])
        resized = moved.reshape(shape).float().mean(dim=1).to(tensor.dtype)
        return resized.movedim(0, dim).contiguous()

    indices = torch.floor(
        torch.arange(target_size, device=tensor.device, dtype=torch.float32)
        * (source_size / target_size)
    ).clamp_max(source_size - 1).long()
    return tensor.index_select(dim, indices).contiguous()


def resize_like(
    source: Any,
    target_shape: tuple[int, ...],
    *,
    hidden_index: Any | None = None,
    source_hidden_size: int | None = None,
) -> Any:
    out = source
    for dim, target_size in enumerate(target_shape):
        out = resize_axis(
            out,
            dim,
            int(target_size),
            hidden_index=hidden_index,
            source_hidden_size=source_hidden_size,
        )
    return out


def layer_source_index(target_layer: int, target_layers: int, donor_layers: int) -> int:
    if target_layers <= 1:
        return 0
    return int(round(target_layer * (donor_layers - 1) / (target_layers - 1)))


def direct_name(name: str) -> str | None:
    if name in {
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    }:
        return name
    return None


def mapped_layer_name(name: str, target_layers: int, donor_layers: int) -> tuple[str, int, int] | None:
    prefix = "model.layers."
    if not name.startswith(prefix):
        return None
    rest = name[len(prefix):]
    layer_text, _, suffix = rest.partition(".")
    if not layer_text.isdigit() or not suffix:
        return None
    target_layer = int(layer_text)
    donor_layer = layer_source_index(target_layer, target_layers, donor_layers)
    return f"{prefix}{donor_layer}.{suffix}", target_layer, donor_layer


def build_hf_config(raw: dict[str, Any], cfg: PyrrhoMoEConfig, dtype_name: str) -> Any:
    from transformers import OlmoeConfig

    data_cfg = raw.get("data") or {}
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
        max_position_embeddings=int(data_cfg.get("max_seq_length") or 4096),
        bos_token_id=None,
        eos_token_id=50279,
        pad_token_id=1,
        norm_topk_prob=False,
    )
    hf_cfg.architectures = ["OlmoeForCausalLM"]
    hf_cfg.dtype = dtype_name
    return hf_cfg


def main() -> int:
    args = parse_args()
    started = time.time()
    raw, target_cfg = load_target_config(args.config)
    budget = target_cfg.budget_report()
    if not all(budget["budget_checks"].values()):
        raise ValueError("target config does not pass pyrrho budget checks")

    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} exists; pass --overwrite")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, OlmoeForCausalLM

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    torch.manual_seed(args.seed)
    donor = AutoModelForCausalLM.from_pretrained(
        args.donor,
        dtype=dtype,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    ).eval()
    donor_config = donor.config.to_dict()
    if getattr(donor.config, "model_type", None) != "olmoe":
        raise ValueError(f"donor model_type must be olmoe, got {donor.config.model_type!r}")

    old_default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        target = OlmoeForCausalLM(build_hf_config(raw, target_cfg, args.dtype)).eval()
    finally:
        torch.set_default_dtype(old_default_dtype)

    donor_params = dict(donor.named_parameters())
    target_layers = int(target.config.num_hidden_layers)
    donor_layers = int(donor.config.num_hidden_layers)
    hidden_index = None
    if args.resize_strategy == "head-topnorm-slice":
        hidden_index = build_head_topnorm_hidden_index(donor, target_cfg)
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    copied = 0

    with torch.no_grad():
        for target_name, target_param in target.named_parameters():
            donor_name = direct_name(target_name)
            target_layer = None
            donor_layer = None
            if donor_name is None:
                mapped = mapped_layer_name(target_name, target_layers, donor_layers)
                if mapped is not None:
                    donor_name, target_layer, donor_layer = mapped
            if donor_name is None or donor_name not in donor_params:
                missing.append(target_name)
                continue

            source = donor_params[donor_name].detach().to(dtype=dtype, device="cpu")
            resized = resize_like(
                source,
                tuple(target_param.shape),
                hidden_index=hidden_index,
                source_hidden_size=int(donor.config.hidden_size),
            )
            target_param.copy_(resized.to(dtype=target_param.dtype, device=target_param.device))
            copied += int(target_param.numel())
            records.append(
                {
                    "target": target_name,
                    "donor": donor_name,
                    "target_shape": list(target_param.shape),
                    "donor_shape": list(source.shape),
                    "target_layer": target_layer,
                    "donor_layer": donor_layer,
                    "resized": list(source.shape) != list(target_param.shape),
                }
            )

    resize_counter = Counter("resized" if row["resized"] else "exact" for row in records)
    target_param_count = sum(param.numel() for param in target.parameters())

    tokenizer = AutoTokenizer.from_pretrained(
        args.donor,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    target.save_pretrained(args.output_dir, safe_serialization=True, max_shard_size=args.max_shard_size)
    tokenizer.save_pretrained(args.output_dir)

    del donor
    del target
    gc.collect()

    report = {
        "schema_version": "pyrrho_olmoe_donor_init_v1",
        "status": "complete",
        "not_trained": True,
        "donor": args.donor,
        "target_config": str(args.config),
        "output_dir": str(args.output_dir),
        "dtype": args.dtype,
        "seed": args.seed,
        "strategy": (
            "layer_interpolation_plus_head_topnorm_hidden_slice_or_nearest_repeat"
            if args.resize_strategy == "head-topnorm-slice"
            else "layer_interpolation_plus_axis_block_mean_or_nearest_repeat"
        ),
        "resize_strategy": args.resize_strategy,
        "hidden_index_count": int(hidden_index.numel()) if hidden_index is not None else 0,
        "target_parameter_count": target_param_count,
        "copied_parameter_count": copied,
        "missing_target_parameters": missing,
        "mapping_summary": dict(resize_counter),
        "donor_config": donor_config,
        "target_budget": budget["parameters"],
        "elapsed_seconds": round(time.time() - started, 3),
        "mapping_records": records,
    }
    report_path = args.output_dir / "donor_init_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "mapping_records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
