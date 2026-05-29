"""Qwen3-MoE trunk wrapper for pyrrho governance heads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from torch import nn


@dataclass(frozen=True)
class QwenMoEGovernanceConfig:
    num_routes: int = 8
    num_taxonomy_patterns: int = 23
    num_scalar_targets: int = 15
    num_labels: int = 3
    dropout: float = 0.0
    freeze_trunk: bool = True


def last_token_pool(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Pool causal hidden states at the last non-padding token."""
    if attention_mask is None:
        return hidden_states[:, -1]
    lengths = attention_mask.to(dtype=torch.long).sum(dim=1).clamp_min(1) - 1
    batch_ids = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[batch_ids, lengths]


def resolve_torch_dtype(dtype: str | torch.dtype | None) -> torch.dtype | str | None:
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    normalized = dtype.lower()
    if normalized in {"auto", "none"}:
        return normalized
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {dtype}")


def set_final_dense_layer_trainability(
    model: QwenMoEForGovernance,
    num_layers: int,
    trainable: bool,
) -> tuple[int, list[int]]:
    if num_layers <= 0:
        return 0, []
    trunk = model.trunk
    layers = getattr(trunk, "layers", None)
    config = getattr(trunk, "config", None)
    if layers is None and hasattr(trunk, "base_model"):
        base_model = getattr(trunk.base_model, "model", trunk.base_model)
        layers = getattr(base_model, "layers", None)
        config = getattr(base_model, "config", config)
    if layers is None:
        raise ValueError("Qwen trunk does not expose a top-level layers module list")
    mlp_only_layers = list(getattr(config, "mlp_only_layers", []) or [])
    dense_layer_ids = [idx for idx in mlp_only_layers if 0 <= int(idx) < len(layers)]
    if not dense_layer_ids:
        raise ValueError("Qwen trunk config does not expose dense mlp_only_layers")
    selected = [int(idx) for idx in dense_layer_ids[-num_layers:]]
    count = 0
    for idx in selected:
        for param in layers[idx].parameters():
            param.requires_grad_(trainable)
            count += param.numel()
    return count, selected


class SparseExpertAdapterExperts(nn.Module):
    """Frozen Qwen expert bank plus trainable per-expert low-rank residuals."""

    def __init__(
        self,
        base_experts: nn.Module,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base_experts = base_experts
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.num_experts = int(getattr(base_experts, "num_experts"))
        self.hidden_dim = int(getattr(base_experts, "hidden_dim"))
        self.adapter_a = nn.Parameter(torch.empty(self.num_experts, self.rank, self.hidden_dim))
        self.adapter_b = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim, self.rank))
        self.dropout = nn.Dropout(float(dropout))
        nn.init.kaiming_uniform_(self.adapter_a, a=5**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        base_output = self.base_experts(hidden_states, top_k_index, top_k_weights)
        adapter_output = torch.zeros_like(base_output)
        with torch.no_grad():
            expert_mask = functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx_tensor in expert_hit:
            expert_idx = int(expert_idx_tensor[0].item())
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx].to(dtype=self.adapter_a.dtype)
            current_hidden = functional.linear(current_state, self.adapter_a[expert_idx])
            current_hidden = functional.silu(current_hidden)
            current_hidden = self.dropout(current_hidden)
            current_output = functional.linear(current_hidden, self.adapter_b[expert_idx])
            current_output = current_output * self.scaling
            current_output = current_output * top_k_weights[
                token_idx,
                top_k_pos,
                None,
            ].to(dtype=current_output.dtype)
            adapter_output.index_add_(0, token_idx, current_output.to(adapter_output.dtype))
        return base_output + adapter_output


class SemanticRouteAdapter(nn.Module):
    """Route-supervised sparse residual adapter over pooled trunk states."""

    def __init__(
        self,
        *,
        num_routes: int,
        hidden_size: int,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.num_routes = int(num_routes)
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.adapter_a = nn.Parameter(torch.empty(self.num_routes, self.rank, self.hidden_size))
        self.adapter_b = nn.Parameter(torch.zeros(self.num_routes, self.hidden_size, self.rank))
        self.dropout = nn.Dropout(float(dropout))
        nn.init.kaiming_uniform_(self.adapter_a, a=5**0.5)

    def forward(self, hidden_states: torch.Tensor, route_ids: torch.Tensor) -> torch.Tensor:
        if route_ids.ndim != 1 or route_ids.shape[0] != hidden_states.shape[0]:
            raise ValueError("route_ids must be a batch vector matching hidden_states")
        if bool((route_ids < 0).any() or (route_ids >= self.num_routes).any()):
            raise ValueError("route_ids contain values outside adapter route range")
        adapter_input = hidden_states.to(dtype=self.adapter_a.dtype)
        a = self.adapter_a.index_select(0, route_ids)
        b = self.adapter_b.index_select(0, route_ids)
        hidden = torch.bmm(a, adapter_input.unsqueeze(-1)).squeeze(-1)
        hidden = functional.silu(hidden)
        hidden = self.dropout(hidden)
        residual = torch.bmm(b, hidden.unsqueeze(-1)).squeeze(-1) * self.scaling
        return hidden_states + residual.to(dtype=hidden_states.dtype)


def _resolve_trunk_layers(trunk: nn.Module) -> nn.ModuleList:
    layers = getattr(trunk, "layers", None)
    if layers is None and hasattr(trunk, "base_model"):
        base_model = getattr(trunk.base_model, "model", trunk.base_model)
        layers = getattr(base_model, "layers", None)
    if layers is None:
        raise ValueError("Qwen trunk does not expose a top-level layers module list")
    return layers


def add_sparse_expert_adapters(
    model: QwenMoEForGovernance,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    num_layers: int,
) -> tuple[int, list[int]]:
    """Attach trainable low-rank residual adapters to Qwen sparse expert banks."""
    if rank <= 0:
        return 0, []
    layers = _resolve_trunk_layers(model.trunk)
    sparse_layers = [
        (idx, layer)
        for idx, layer in enumerate(layers)
        if hasattr(getattr(layer, "mlp", None), "experts")
    ]
    if not sparse_layers:
        raise ValueError("Qwen trunk has no sparse MoE layers with expert banks")
    selected = sparse_layers[-num_layers:] if num_layers > 0 else sparse_layers
    count = 0
    selected_indices: list[int] = []
    for idx, layer in selected:
        experts = layer.mlp.experts
        if isinstance(experts, SparseExpertAdapterExperts):
            adapter = experts
        else:
            adapter = SparseExpertAdapterExperts(
                experts,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
            layer.mlp.experts = adapter
        count += adapter.adapter_a.numel() + adapter.adapter_b.numel()
        selected_indices.append(int(idx))
    return count, selected_indices


def add_semantic_route_adapter(
    model: QwenMoEForGovernance,
    *,
    rank: int,
    alpha: float,
    dropout: float,
) -> int:
    """Attach a trainable semantic-route sparse adapter after trunk pooling."""
    if rank <= 0:
        return 0
    hidden_size = int(getattr(model.trunk.config, "hidden_size"))
    model.semantic_route_adapter = SemanticRouteAdapter(
        num_routes=model.config.num_routes,
        hidden_size=hidden_size,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    )
    return (
        model.semantic_route_adapter.adapter_a.numel()
        + model.semantic_route_adapter.adapter_b.numel()
    )


class QwenMoEForGovernance(nn.Module):
    """Attach pyrrho multitask governance heads to a Qwen3-MoE trunk."""

    def __init__(self, trunk: nn.Module, config: QwenMoEGovernanceConfig) -> None:
        super().__init__()
        self.trunk = trunk
        self.config = config
        hidden_size = int(getattr(trunk.config, "hidden_size"))
        self.dropout = nn.Dropout(config.dropout)
        self.route_head = nn.Linear(hidden_size, config.num_routes)
        self.governance_head = nn.Linear(hidden_size, config.num_labels)
        self.taxonomy_head = nn.Linear(hidden_size, config.num_taxonomy_patterns)
        self.scalar_head = nn.Linear(hidden_size, config.num_scalar_targets)
        self.semantic_route_adapter: SemanticRouteAdapter | None = None
        if config.freeze_trunk:
            for param in self.trunk.parameters():
                param.requires_grad_(False)

    @classmethod
    def from_seed_pack(
        cls,
        seed_pack: str | Path,
        config: QwenMoEGovernanceConfig,
        *,
        dtype: str | torch.dtype | None = "bfloat16",
        device_map: str | dict[str, Any] | None = None,
        local_files_only: bool = True,
        attn_implementation: str | None = "sdpa",
    ) -> QwenMoEForGovernance:
        from transformers import AutoModel

        kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "local_files_only": local_files_only,
        }
        resolved_dtype = resolve_torch_dtype(dtype)
        if resolved_dtype is not None:
            kwargs["dtype"] = resolved_dtype
        if device_map is not None:
            kwargs["device_map"] = device_map
        if attn_implementation is not None:
            kwargs["attn_implementation"] = attn_implementation

        trunk = AutoModel.from_pretrained(seed_pack, **kwargs)
        return cls(trunk=trunk, config=config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        route_ids: torch.Tensor | None = None,
        force_route_ids: bool = False,
    ) -> dict[str, torch.Tensor]:
        trunk_outputs = self.trunk(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        pooled = last_token_pool(trunk_outputs.last_hidden_state, attention_mask)
        pooled = self.dropout(pooled)
        head_state = pooled.to(dtype=self.route_head.weight.dtype)
        route_logits = self.route_head(head_state)
        if route_ids is not None and (self.training or force_route_ids):
            selected_routes = route_ids
        else:
            selected_routes = route_logits.argmax(dim=-1)
        expert_state = head_state
        if self.semantic_route_adapter is not None:
            expert_state = self.semantic_route_adapter(head_state, selected_routes)
        return {
            "route_logits": route_logits,
            "selected_routes": selected_routes,
            "governance_logits": self.governance_head(expert_state),
            "taxonomy_logits": self.taxonomy_head(expert_state),
            "scalar_preds": torch.sigmoid(self.scalar_head(expert_state)),
        }
