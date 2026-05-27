"""Dense-to-MoE upcycling helpers.

These functions are small and deterministic on purpose. The first real
upcycling path needs to compress Qwen's 3072-wide dense FFNs into the selected
2112-wide pyrrho expert FFNs before cloning those weights across experts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CompressedFFN:
    gate_proj: torch.Tensor
    up_proj: torch.Tensor
    down_proj: torch.Tensor
    selected_indices: torch.Tensor


def qwen_mlp_weight_keys(layer: int) -> tuple[str, str, str]:
    if layer < 0:
        raise ValueError("layer must be non-negative")
    prefix = f"model.layers.{layer}.mlp"
    return (
        f"{prefix}.gate_proj.weight",
        f"{prefix}.up_proj.weight",
        f"{prefix}.down_proj.weight",
    )


def resolve_safetensor_files_for_keys(
    weight_map: Mapping[str, str] | None,
    keys: Iterable[str],
    *,
    default_filename: str = "model.safetensors",
) -> dict[str, str]:
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for key in keys:
        if weight_map is None:
            resolved[key] = default_filename
        elif key in weight_map:
            resolved[key] = weight_map[key]
        else:
            missing.append(key)
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"missing keys in safetensors weight map: {missing_text}")
    return resolved


def ffn_channel_scores(
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
) -> torch.Tensor:
    """Score seed FFN channels by combined gate/up row and down column norm."""
    validate_qwen_ffn_shapes(gate_proj, up_proj, down_proj)
    return (
        gate_proj.float().pow(2).sum(dim=1)
        + up_proj.float().pow(2).sum(dim=1)
        + down_proj.float().pow(2).sum(dim=0)
    )


def select_ffn_channels_by_norm(
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    target_dim: int,
) -> torch.Tensor:
    """Select the strongest FFN channels, returned in original index order."""
    scores = ffn_channel_scores(gate_proj, up_proj, down_proj)
    if target_dim <= 0:
        raise ValueError("target_dim must be positive")
    if target_dim > scores.numel():
        raise ValueError(
            f"target_dim={target_dim} exceeds seed FFN dim={scores.numel()}"
        )
    selected = torch.topk(scores, k=target_dim, largest=True, sorted=False).indices
    return torch.sort(selected).values


def validate_qwen_ffn_shapes(
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
) -> None:
    if gate_proj.ndim != 2 or up_proj.ndim != 2 or down_proj.ndim != 2:
        raise ValueError("FFN projection tensors must be rank-2")
    if gate_proj.shape != up_proj.shape:
        raise ValueError(
            f"gate/up shapes must match, got {tuple(gate_proj.shape)} and {tuple(up_proj.shape)}"
        )
    if down_proj.shape != (gate_proj.shape[1], gate_proj.shape[0]):
        raise ValueError(
            "down_proj shape must be [hidden, intermediate], got "
            f"{tuple(down_proj.shape)} for gate shape {tuple(gate_proj.shape)}"
        )


def compress_qwen_ffn(
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    target_dim: int,
) -> CompressedFFN:
    """Compress Qwen-style SwiGLU FFN tensors to `target_dim` channels."""
    selected = select_ffn_channels_by_norm(gate_proj, up_proj, down_proj, target_dim)
    return CompressedFFN(
        gate_proj=gate_proj.index_select(0, selected).contiguous(),
        up_proj=up_proj.index_select(0, selected).contiguous(),
        down_proj=down_proj.index_select(1, selected).contiguous(),
        selected_indices=selected,
    )
