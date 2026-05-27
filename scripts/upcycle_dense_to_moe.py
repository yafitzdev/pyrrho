"""Inspect or build dense-to-MoE upcycling plans.

The default mode validates the selected dense seed against a target pyrrho-MoE
config and writes the layer/key/compression plan. `--real-weight-smoke` verifies
the plan against actual seed tensors for one layer. `--write-seed-pack` streams
the seed weights into a sharded Qwen3-MoE-compatible checkpoint skeleton.

Run from project root:
    python scripts/upcycle_dense_to_moe.py --inspect-only
    python scripts/upcycle_dense_to_moe.py --inspect-only --real-weight-smoke
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import yaml
from transformers import AutoConfig

from pyrrho.moe import PyrrhoMoEConfig
from pyrrho.moe.upcycling import (
    compress_qwen_ffn,
    qwen_mlp_weight_keys,
    resolve_safetensor_files_for_keys,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/moe/pyrrho_moe_g3_alpha_qwen.yaml"),
        help="Target MoE YAML config",
    )
    p.add_argument(
        "--seed-model",
        type=str,
        default=None,
        help="Override seed_candidate.model_id from the config",
    )
    p.add_argument(
        "--seed-revision",
        type=str,
        default=None,
        help="Optional Hugging Face revision for the seed model",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/moe/upcycling/qwen_alpha_inspect.json"),
        help="Output JSON plan path",
    )
    p.add_argument(
        "--inspect-only",
        action="store_true",
        help="Write only the JSON inspection plan unless paired with smoke/write flags",
    )
    p.add_argument(
        "--real-weight-smoke",
        action="store_true",
        help="Download the needed seed safetensors shard and compress one layer FFN",
    )
    p.add_argument(
        "--smoke-layer",
        type=int,
        default=None,
        help="Seed layer to use for --real-weight-smoke; defaults to the first MoE layer",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache dir for seed weight downloads",
    )
    p.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only locally cached Hugging Face files for --real-weight-smoke",
    )
    p.add_argument(
        "--write-seed-pack",
        type=Path,
        default=None,
        help="Write a sharded Qwen3-MoE-compatible upcycled seed pack to this directory",
    )
    p.add_argument(
        "--validate-seed-pack",
        type=Path,
        default=None,
        help="Validate a written seed pack's manifest against a meta-initialized Qwen3-MoE model",
    )
    return p.parse_args()


def seed_attr(seed_cfg: Any, name: str) -> Any:
    return getattr(seed_cfg, name, None)


def dense_layer_ids(total_layers: int, dense_ffn_layers: int) -> list[int]:
    """Use first/last dense layers, matching the architecture-doc pattern."""
    if dense_ffn_layers == 0:
        return []
    first = dense_ffn_layers // 2
    last = dense_ffn_layers - first
    return [*range(first), *range(total_layers - last, total_layers)]


def qwen_key_shapes(
    layer: int,
    hidden: int,
    q_dim: int,
    kv_dim: int,
    head_dim: int,
    ffn_dim: int,
) -> dict[str, list[int]]:
    prefix = f"model.layers.{layer}"
    return {
        f"{prefix}.self_attn.q_proj.weight": [q_dim, hidden],
        f"{prefix}.self_attn.k_proj.weight": [kv_dim, hidden],
        f"{prefix}.self_attn.v_proj.weight": [kv_dim, hidden],
        f"{prefix}.self_attn.o_proj.weight": [hidden, q_dim],
        f"{prefix}.self_attn.q_norm.weight": [head_dim],
        f"{prefix}.self_attn.k_norm.weight": [head_dim],
        f"{prefix}.mlp.gate_proj.weight": [ffn_dim, hidden],
        f"{prefix}.mlp.up_proj.weight": [ffn_dim, hidden],
        f"{prefix}.mlp.down_proj.weight": [hidden, ffn_dim],
        f"{prefix}.input_layernorm.weight": [hidden],
        f"{prefix}.post_attention_layernorm.weight": [hidden],
    }


def qwen_direct_layer_keys(layer: int) -> tuple[str, ...]:
    prefix = f"model.layers.{layer}"
    return (
        f"{prefix}.self_attn.q_proj.weight",
        f"{prefix}.self_attn.k_proj.weight",
        f"{prefix}.self_attn.v_proj.weight",
        f"{prefix}.self_attn.o_proj.weight",
        f"{prefix}.self_attn.q_norm.weight",
        f"{prefix}.self_attn.k_norm.weight",
        f"{prefix}.input_layernorm.weight",
        f"{prefix}.post_attention_layernorm.weight",
    )


def build_plan(target: PyrrhoMoEConfig, seed_cfg: Any, seed_model: str) -> dict[str, Any]:
    target_counts = target.budget_report()
    seed_hidden = int(seed_attr(seed_cfg, "hidden_size"))
    seed_layers = int(seed_attr(seed_cfg, "num_hidden_layers"))
    seed_heads = int(seed_attr(seed_cfg, "num_attention_heads"))
    seed_kv_heads = int(seed_attr(seed_cfg, "num_key_value_heads"))
    seed_vocab = int(seed_attr(seed_cfg, "vocab_size"))
    seed_ffn = int(seed_attr(seed_cfg, "intermediate_size"))
    seed_head_dim = int(seed_attr(seed_cfg, "head_dim") or (seed_hidden // seed_heads))
    seed_q_dim = seed_heads * seed_head_dim
    seed_kv_dim = seed_kv_heads * seed_head_dim

    checks = {
        "hidden_size_match": seed_hidden == target.hidden_size,
        "layer_count_match": seed_layers == target.layers,
        "attention_heads_match": seed_heads == target.attention_heads,
        "head_dim_match": seed_head_dim == target.head_dim,
        "kv_heads_match": seed_kv_heads == target.kv_heads,
        "vocab_size_match": seed_vocab == target.vocab_size,
        "budget_pass": all(target_counts["budget_checks"].values()),
    }

    dense_ids = set(dense_layer_ids(target.layers, target.dense_ffn_layers))
    layers = []
    for layer in range(target.layers):
        layer_type = "dense_ffn" if layer in dense_ids else "moe_ffn"
        if layer_type == "dense_ffn":
            ffn_action = "compress_seed_ffn_to_dense_layer"
            target_experts = 0
        else:
            ffn_action = "compress_seed_ffn_then_clone_to_all_experts"
            target_experts = target.experts_per_moe_layer
        layers.append(
            {
                "layer": layer,
                "layer_type": layer_type,
                "target_experts": target_experts,
                "attention": "copy_direct",
                "norms": "copy_direct",
                "ffn": ffn_action,
                "seed_key_shapes": qwen_key_shapes(
                    layer, seed_hidden, seed_q_dim, seed_kv_dim, seed_head_dim, seed_ffn
                ),
                "target_ffn_dim": target.ffn_dim,
                "seed_ffn_dim": seed_ffn,
                "ffn_keep_ratio": target.ffn_dim / seed_ffn,
            }
        )

    return {
        "status": "inspect_only",
        "seed_model": seed_model,
        "seed_config": {
            "model_type": seed_attr(seed_cfg, "model_type"),
            "hidden_size": seed_hidden,
            "num_hidden_layers": seed_layers,
            "intermediate_size": seed_ffn,
            "num_attention_heads": seed_heads,
            "num_key_value_heads": seed_kv_heads,
            "head_dim": seed_head_dim,
            "q_dim": seed_q_dim,
            "vocab_size": seed_vocab,
            "tie_word_embeddings": seed_attr(seed_cfg, "tie_word_embeddings"),
        },
        "target_config": target.budget_report()["config"],
        "target_derived": target_counts["derived"],
        "target_parameters": target_counts["parameters"],
        "compatibility_checks": checks,
        "direct_copy": {
            "embeddings": checks["hidden_size_match"] and checks["vocab_size_match"],
            "attention": checks["hidden_size_match"]
            and checks["attention_heads_match"]
            and checks["head_dim_match"]
            and checks["kv_heads_match"],
            "norms": checks["hidden_size_match"],
            "lm_head": seed_attr(seed_cfg, "tie_word_embeddings")
            and checks["hidden_size_match"]
            and checks["vocab_size_match"],
        },
        "ffn_compression": {
            "required": seed_ffn != target.ffn_dim,
            "seed_ffn_dim": seed_ffn,
            "target_ffn_dim": target.ffn_dim,
            "keep_ratio": target.ffn_dim / seed_ffn,
            "allowed_first_strategy": "select_ffn_channels_by_combined_gate_up_down_norm",
            "fallback_strategy": "svd_or_projection_init",
        },
        "layer_layout": {
            "dense_layers": sorted(dense_ids),
            "moe_layers": [i for i in range(target.layers) if i not in dense_ids],
        },
        "layers": layers,
    }


def hf_hub_file(
    *,
    repo_id: str,
    filename: str,
    revision: str | None,
    cache_dir: Path | None,
    local_files_only: bool,
) -> Path:
    from huggingface_hub import hf_hub_download

    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "filename": filename,
        "revision": revision,
        "local_files_only": local_files_only,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    return Path(hf_hub_download(**kwargs))


def load_safetensors_weight_map(
    *,
    repo_id: str,
    revision: str | None,
    cache_dir: Path | None,
    local_files_only: bool,
) -> Mapping[str, str] | None:
    try:
        index_path = hf_hub_file(
            repo_id=repo_id,
            filename="model.safetensors.index.json",
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    except Exception as exc:
        missing_error_names = {
            "EntryNotFoundError",
            "LocalEntryNotFoundError",
            "RemoteEntryNotFoundError",
        }
        if exc.__class__.__name__ in missing_error_names:
            return None
        raise

    raw = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = raw.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"{index_path} has no safetensors weight_map")
    return weight_map


def load_seed_tensors(
    *,
    repo_id: str,
    revision: str | None,
    cache_dir: Path | None,
    local_files_only: bool,
    key_to_file: Mapping[str, str],
) -> dict[str, Any]:
    from safetensors import safe_open

    file_to_keys: dict[str, list[str]] = {}
    for key, filename in key_to_file.items():
        file_to_keys.setdefault(filename, []).append(key)

    tensors: dict[str, Any] = {}
    for filename, keys in sorted(file_to_keys.items()):
        path = hf_hub_file(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            missing = [key for key in keys if key not in available]
            if missing:
                missing_text = ", ".join(missing)
                raise ValueError(f"{path} is missing expected tensors: {missing_text}")
            for key in keys:
                tensors[key] = handle.get_tensor(key)
    return tensors


def real_weight_smoke(
    *,
    target: PyrrhoMoEConfig,
    seed_model: str,
    seed_revision: str | None,
    layer: int,
    cache_dir: Path | None,
    local_files_only: bool,
) -> dict[str, Any]:
    keys = qwen_mlp_weight_keys(layer)
    weight_map = load_safetensors_weight_map(
        repo_id=seed_model,
        revision=seed_revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    key_to_file = resolve_safetensor_files_for_keys(weight_map, keys)
    tensors = load_seed_tensors(
        repo_id=seed_model,
        revision=seed_revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        key_to_file=key_to_file,
    )

    gate = tensors[keys[0]]
    up = tensors[keys[1]]
    down = tensors[keys[2]]
    compressed = compress_qwen_ffn(gate, up, down, target.ffn_dim)
    selected = compressed.selected_indices

    return {
        "status": "pass",
        "layer": layer,
        "seed_model": seed_model,
        "seed_revision": seed_revision,
        "files": sorted(set(key_to_file.values())),
        "seed_shapes": {key: list(tensors[key].shape) for key in keys},
        "seed_dtypes": {key: str(tensors[key].dtype) for key in keys},
        "compressed_shapes": {
            "gate_proj": list(compressed.gate_proj.shape),
            "up_proj": list(compressed.up_proj.shape),
            "down_proj": list(compressed.down_proj.shape),
        },
        "selected_channels": {
            "count": int(selected.numel()),
            "min": int(selected.min().item()),
            "max": int(selected.max().item()),
            "sum": int(selected.long().sum().item()),
            "first_8": [int(x) for x in selected[:8].tolist()],
            "last_8": [int(x) for x in selected[-8:].tolist()],
        },
    }


def tensor_size_bytes(tensor: Any) -> int:
    return int(tensor.numel() * tensor.element_size())


def write_qwen3_moe_config(target: PyrrhoMoEConfig, seed_cfg: Any, output_dir: Path) -> None:
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

    dense_ids = dense_layer_ids(target.layers, target.dense_ffn_layers)
    cfg = Qwen3MoeConfig(
        architectures=["Qwen3MoeForCausalLM"],
        dtype="bfloat16",
        vocab_size=target.vocab_size,
        hidden_size=target.hidden_size,
        intermediate_size=target.ffn_dim,
        num_hidden_layers=target.layers,
        num_attention_heads=target.attention_heads,
        num_key_value_heads=target.kv_heads,
        head_dim=target.head_dim,
        hidden_act=seed_attr(seed_cfg, "hidden_act") or "silu",
        max_position_embeddings=seed_attr(seed_cfg, "max_position_embeddings") or 32768,
        initializer_range=seed_attr(seed_cfg, "initializer_range") or 0.02,
        rms_norm_eps=seed_attr(seed_cfg, "rms_norm_eps") or 1e-6,
        use_cache=True,
        tie_word_embeddings=target.tied_embeddings,
        rope_parameters=seed_attr(seed_cfg, "rope_parameters"),
        attention_bias=bool(seed_attr(seed_cfg, "attention_bias") or False),
        use_sliding_window=bool(seed_attr(seed_cfg, "use_sliding_window") or False),
        sliding_window=seed_attr(seed_cfg, "sliding_window"),
        attention_dropout=seed_attr(seed_cfg, "attention_dropout") or 0.0,
        decoder_sparse_step=1,
        moe_intermediate_size=target.ffn_dim,
        num_experts_per_tok=target.top_k,
        num_experts=target.experts_per_moe_layer,
        norm_topk_prob=False,
        output_router_logits=True,
        router_aux_loss_coef=0.001,
        mlp_only_layers=dense_ids,
        pad_token_id=seed_attr(seed_cfg, "pad_token_id"),
        bos_token_id=seed_attr(seed_cfg, "bos_token_id"),
        eos_token_id=seed_attr(seed_cfg, "eos_token_id"),
    )
    cfg.save_pretrained(output_dir)


def write_seed_tokenizer(
    *,
    seed_model: str,
    seed_revision: str | None,
    output_dir: Path,
    cache_dir: Path | None,
    local_files_only: bool,
) -> None:
    from transformers import AutoTokenizer

    kwargs: dict[str, Any] = {
        "revision": seed_revision,
        "trust_remote_code": True,
        "local_files_only": local_files_only,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(seed_model, **kwargs)
    tokenizer.save_pretrained(output_dir)


def save_safetensor_shard(
    *,
    output_dir: Path,
    shard_index: int,
    shard_count: int,
    tensors: Mapping[str, Any],
    weight_map: dict[str, str],
    tensor_manifest: dict[str, Any],
) -> int:
    from safetensors.torch import save_file

    filename = f"model-{shard_index:05d}-of-{shard_count:05d}.safetensors"
    path = output_dir / filename
    save_file(dict(tensors), str(path), metadata={"format": "pt"})

    shard_bytes = 0
    for key, tensor in tensors.items():
        nbytes = tensor_size_bytes(tensor)
        shard_bytes += nbytes
        weight_map[key] = filename
        tensor_manifest[key] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "file": filename,
            "bytes": nbytes,
        }
    return shard_bytes


def all_required_seed_keys(target: PyrrhoMoEConfig) -> list[str]:
    keys = ["model.embed_tokens.weight", "model.norm.weight"]
    for layer in range(target.layers):
        keys.extend(qwen_direct_layer_keys(layer))
        keys.extend(qwen_mlp_weight_keys(layer))
    return keys


def write_seed_pack(
    *,
    target: PyrrhoMoEConfig,
    seed_cfg: Any,
    seed_model: str,
    seed_revision: str | None,
    output_dir: Path,
    cache_dir: Path | None,
    local_files_only: bool,
) -> dict[str, Any]:
    import torch

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} already exists and is not empty")
    output_dir.mkdir(parents=True, exist_ok=True)

    write_qwen3_moe_config(target, seed_cfg, output_dir)
    write_seed_tokenizer(
        seed_model=seed_model,
        seed_revision=seed_revision,
        output_dir=output_dir,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )

    hf_weight_map = load_safetensors_weight_map(
        repo_id=seed_model,
        revision=seed_revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    required_keys = all_required_seed_keys(target)
    seed_key_to_file = resolve_safetensor_files_for_keys(hf_weight_map, required_keys)

    shard_count = target.layers + 2
    shard_index = 1
    total_size = 0
    weight_map: dict[str, str] = {}
    tensor_manifest: dict[str, Any] = {}
    compressed_layers: list[dict[str, Any]] = []

    tensors = load_seed_tensors(
        repo_id=seed_model,
        revision=seed_revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        key_to_file={"model.embed_tokens.weight": seed_key_to_file["model.embed_tokens.weight"]},
    )
    total_size += save_safetensor_shard(
        output_dir=output_dir,
        shard_index=shard_index,
        shard_count=shard_count,
        tensors=tensors,
        weight_map=weight_map,
        tensor_manifest=tensor_manifest,
    )
    shard_index += 1
    del tensors
    gc.collect()

    dense_ids = set(dense_layer_ids(target.layers, target.dense_ffn_layers))
    for layer in range(target.layers):
        layer_seed_keys = [*qwen_direct_layer_keys(layer), *qwen_mlp_weight_keys(layer)]
        tensors = load_seed_tensors(
            repo_id=seed_model,
            revision=seed_revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            key_to_file={key: seed_key_to_file[key] for key in layer_seed_keys},
        )
        gate_key, up_key, down_key = qwen_mlp_weight_keys(layer)
        compressed = compress_qwen_ffn(
            tensors[gate_key],
            tensors[up_key],
            tensors[down_key],
            target.ffn_dim,
        )

        shard_tensors: dict[str, Any] = {
            key: tensors[key] for key in qwen_direct_layer_keys(layer)
        }
        if layer in dense_ids:
            shard_tensors[gate_key] = compressed.gate_proj
            shard_tensors[up_key] = compressed.up_proj
            shard_tensors[down_key] = compressed.down_proj
            layer_type = "dense_ffn"
        else:
            prefix = f"model.layers.{layer}.mlp"
            gate_up = torch.cat([compressed.gate_proj, compressed.up_proj], dim=0)
            shard_tensors[f"{prefix}.experts.gate_up_proj"] = gate_up.unsqueeze(0).repeat(
                target.experts_per_moe_layer, 1, 1
            ).contiguous()
            shard_tensors[f"{prefix}.experts.down_proj"] = compressed.down_proj.unsqueeze(0).repeat(
                target.experts_per_moe_layer, 1, 1
            ).contiguous()
            shard_tensors[f"{prefix}.gate.weight"] = torch.zeros(
                (target.experts_per_moe_layer, target.hidden_size),
                dtype=compressed.gate_proj.dtype,
            )
            layer_type = "moe_ffn"

        total_size += save_safetensor_shard(
            output_dir=output_dir,
            shard_index=shard_index,
            shard_count=shard_count,
            tensors=shard_tensors,
            weight_map=weight_map,
            tensor_manifest=tensor_manifest,
        )
        selected = compressed.selected_indices
        compressed_layers.append(
            {
                "layer": layer,
                "layer_type": layer_type,
                "selected_count": int(selected.numel()),
                "selected_min": int(selected.min().item()),
                "selected_max": int(selected.max().item()),
                "selected_sum": int(selected.long().sum().item()),
            }
        )
        shard_index += 1
        del tensors, shard_tensors, compressed
        gc.collect()

    tensors = load_seed_tensors(
        repo_id=seed_model,
        revision=seed_revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        key_to_file={"model.norm.weight": seed_key_to_file["model.norm.weight"]},
    )
    total_size += save_safetensor_shard(
        output_dir=output_dir,
        shard_index=shard_index,
        shard_count=shard_count,
        tensors=tensors,
        weight_map=weight_map,
        tensor_manifest=tensor_manifest,
    )
    del tensors
    gc.collect()

    index = {
        "metadata": {
            "total_size": total_size,
            "format": "pt",
            "source_model": seed_model,
            "source_revision": seed_revision,
            "omitted_tied_weights": ["lm_head.weight"] if target.tied_embeddings else [],
        },
        "weight_map": weight_map,
    }
    (output_dir / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2), encoding="utf-8"
    )

    manifest = {
        "status": "complete",
        "seed_model": seed_model,
        "seed_revision": seed_revision,
        "target_config": target.budget_report(),
        "shard_count": shard_count,
        "total_size": total_size,
        "dense_layers": sorted(dense_ids),
        "moe_layers": [i for i in range(target.layers) if i not in dense_ids],
        "ffn_compression": {
            "seed_ffn_dim": int(seed_attr(seed_cfg, "intermediate_size")),
            "target_ffn_dim": target.ffn_dim,
            "strategy": "select_ffn_channels_by_combined_gate_up_down_norm",
            "compressed_layers": compressed_layers,
        },
        "tensors": tensor_manifest,
    }
    (output_dir / "upcycling_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (output_dir / "pyrrho_moe_config.json").write_text(
        json.dumps(target.budget_report(), indent=2), encoding="utf-8"
    )
    return {
        "status": "complete",
        "output_dir": str(output_dir),
        "shard_count": shard_count,
        "tensor_count": len(tensor_manifest),
        "total_size": total_size,
    }


def validate_seed_pack_shapes(pack_dir: Path) -> dict[str, Any]:
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    manifest_path = pack_dir / "upcycling_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tensors = manifest["tensors"]

    cfg = AutoConfig.from_pretrained(pack_dir, trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    expected = {key: list(tensor.shape) for key, tensor in model.state_dict().items()}
    tied = getattr(model, "_tied_weights_keys", {}) or {}

    missing: list[str] = []
    shape_mismatches: list[dict[str, Any]] = []
    for key, shape in expected.items():
        tied_to = tied.get(key) if isinstance(tied, dict) else None
        if key not in tensors:
            if tied_to and tied_to in tensors:
                continue
            missing.append(key)
            continue
        if tensors[key]["shape"] != shape:
            shape_mismatches.append(
                {"key": key, "expected": shape, "actual": tensors[key]["shape"]}
            )

    unexpected = sorted(key for key in tensors if key not in expected)
    report = {
        "status": "pass" if not missing and not unexpected and not shape_mismatches else "fail",
        "pack_dir": str(pack_dir),
        "expected_tensor_count": len(expected),
        "manifest_tensor_count": len(tensors),
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatches": shape_mismatches,
        "ignored_tied_weights": ["lm_head.weight"] if "lm_head.weight" in expected else [],
    }
    (pack_dir / "load_shape_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> int:
    args = parse_args()
    if not (args.inspect_only or args.real_weight_smoke or args.write_seed_pack or args.validate_seed_pack):
        print(
            "ERROR: choose --inspect-only, --real-weight-smoke, --write-seed-pack, "
            "or --validate-seed-pack",
            file=sys.stderr,
        )
        return 2

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    seed_model = args.seed_model or raw.get("seed_candidate", {}).get("model_id")
    seed_revision = args.seed_revision or raw.get("seed_candidate", {}).get("revision")
    if not seed_model:
        print("ERROR: seed model not supplied and config has no seed_candidate.model_id", file=sys.stderr)
        return 1

    target = PyrrhoMoEConfig.from_mapping(raw.get("architecture"))
    seed_cfg = AutoConfig.from_pretrained(seed_model, trust_remote_code=True)
    plan = build_plan(target, seed_cfg, seed_model)
    if seed_revision is not None:
        plan["seed_revision"] = seed_revision

    if args.real_weight_smoke:
        smoke_layer = args.smoke_layer
        if smoke_layer is None:
            smoke_layer = plan["layer_layout"]["moe_layers"][0]
        plan["real_weight_smoke"] = real_weight_smoke(
            target=target,
            seed_model=seed_model,
            seed_revision=seed_revision,
            layer=smoke_layer,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
        )

    if args.write_seed_pack is not None:
        plan["seed_pack"] = write_seed_pack(
            target=target,
            seed_cfg=seed_cfg,
            seed_model=seed_model,
            seed_revision=seed_revision,
            output_dir=args.write_seed_pack,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
        )
        plan["seed_pack_validation"] = validate_seed_pack_shapes(args.write_seed_pack)

    if args.validate_seed_pack is not None:
        plan["seed_pack_validation"] = validate_seed_pack_shapes(args.validate_seed_pack)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    checks = plan["compatibility_checks"]
    print(f"Seed model      : {seed_model}")
    print(f"Target config   : {args.config}")
    print(f"Output plan     : {args.output}")
    for key, passed in checks.items():
        print(f"{key:24s}: {'PASS' if passed else 'FAIL'}")
    ffn = plan["ffn_compression"]
    print(
        "ffn compression        : "
        f"{ffn['seed_ffn_dim']} -> {ffn['target_ffn_dim']} "
        f"({ffn['keep_ratio']:.3f})"
    )
    if "real_weight_smoke" in plan:
        smoke = plan["real_weight_smoke"]
        selected = smoke["selected_channels"]
        print(f"real weight smoke      : PASS layer={smoke['layer']} files={len(smoke['files'])}")
        print(
            "selected channels      : "
            f"{selected['count']} ({selected['min']}..{selected['max']})"
        )
    if "seed_pack" in plan:
        pack = plan["seed_pack"]
        print(f"seed pack              : {pack['output_dir']}")
        print(f"seed pack tensors      : {pack['tensor_count']} in {pack['shard_count']} shards")
        print(f"seed pack size         : {pack['total_size'] / 1_000_000_000:.3f} GB")
    if "seed_pack_validation" in plan:
        validation = plan["seed_pack_validation"]
        print(f"seed pack validation   : {validation['status'].upper()}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
