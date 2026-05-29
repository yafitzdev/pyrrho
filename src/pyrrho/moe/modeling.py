"""Route-coupled MoE prototype models used before the 4B skeleton."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn


@dataclass(frozen=True)
class TinyMoEConfig:
    token_vocab_size: int = 32768
    hidden_size: int = 256
    expert_hidden_size: int = 512
    num_routes: int = 8
    num_taxonomy_patterns: int = 23
    num_scalar_targets: int = 15
    num_labels: int = 3
    dropout: float = 0.1


@dataclass(frozen=True)
class RouteCoupledMoEConfig:
    token_vocab_size: int = 65536
    hidden_size: int = 384
    expert_hidden_size: int = 768
    num_expert_layers: int = 4
    num_routes: int = 8
    num_taxonomy_patterns: int = 23
    num_scalar_targets: int = 15
    num_labels: int = 3
    dropout: float = 0.1


@dataclass(frozen=True)
class TokenRouteCoupledMoEConfig:
    token_vocab_size: int = 65536
    hidden_size: int = 384
    expert_hidden_size: int = 768
    num_expert_layers: int = 4
    num_attention_heads: int = 6
    num_key_value_heads: int = 2
    num_routes: int = 8
    num_taxonomy_patterns: int = 23
    num_scalar_targets: int = 15
    num_labels: int = 3
    dropout: float = 0.1
    rope_theta: float = 10000.0


@dataclass(frozen=True)
class SupportAggregatingMoEConfig(TokenRouteCoupledMoEConfig):
    max_query_length: int = 96
    max_sources: int = 8
    max_source_length: int = 192
    uses_support_aggregation: bool = True


@dataclass(frozen=True)
class GuardedSupportAggregatingMoEConfig(SupportAggregatingMoEConfig):
    trust_penalty_scale: float = 1.0
    trust_penalty_init_bias: float = -2.0


@dataclass(frozen=True)
class TrustGuardedSupportAggregatingMoEConfig(SupportAggregatingMoEConfig):
    trust_guard_scale: float = 0.75
    trust_guard_init_bias: float = 2.0
    trust_guard_detach_candidate_logits: bool = True


class RMSNorm(nn.Module):
    """RMSNorm block matching the terminal decoder-style MoE plan."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


def _rope_cache(
    *,
    seq_len: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos = freqs.cos().to(dtype=dtype)[None, None, :, :]
    sin = freqs.sin().to(dtype=dtype)[None, None, :, :]
    return cos, sin


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack(
        (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos),
        dim=-1,
    )
    return rotated.flatten(-2)


def _repeat_kv(x: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return x
    batch, kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, kv_heads, repeats, seq_len, head_dim)
    return x.reshape(batch, kv_heads * repeats, seq_len, head_dim)


class TinyMoEForGovernance(nn.Module):
    """Hash-token prototype with supervised top-1 expert selection.

    This is intentionally not the terminal 4B architecture. It exercises the
    same task surfaces: route logits, selected expert path, governance logits,
    taxonomy logits, and scalar targets.
    """

    def __init__(self, config: TinyMoEConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            config.token_vocab_size,
            config.hidden_size,
            padding_idx=0,
        )
        self.norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.router = nn.Linear(config.hidden_size, config.num_routes)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.hidden_size, config.expert_hidden_size),
                    nn.SiLU(),
                    nn.Linear(config.expert_hidden_size, config.hidden_size),
                    nn.SiLU(),
                )
                for _ in range(config.num_routes)
            ]
        )
        self.governance_head = nn.Linear(config.hidden_size, config.num_labels)
        self.taxonomy_head = nn.Linear(config.hidden_size, config.num_taxonomy_patterns)
        self.scalar_head = nn.Linear(config.hidden_size, config.num_scalar_targets)

    def pool(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1).to(dtype=emb.dtype)
        pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.dropout(self.norm(pooled))

    def select_experts(
        self,
        pooled: torch.Tensor,
        route_logits: torch.Tensor,
        route_ids: torch.Tensor | None,
        force_route_ids: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if route_ids is not None and (self.training or force_route_ids):
            selected = route_ids
        else:
            selected = route_logits.argmax(dim=-1)

        expert_out = torch.empty_like(pooled)
        for expert_id, expert in enumerate(self.experts):
            mask = selected == expert_id
            if mask.any():
                expert_out[mask] = expert(pooled[mask])
        return expert_out, selected

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        route_ids: torch.Tensor | None = None,
        force_route_ids: bool = False,
    ) -> dict[str, torch.Tensor]:
        pooled = self.pool(input_ids, attention_mask)
        route_logits = self.router(pooled)
        expert_state, selected_routes = self.select_experts(
            pooled,
            route_logits,
            route_ids,
            force_route_ids=force_route_ids,
        )
        expert_state = self.dropout(expert_state)
        return {
            "route_logits": route_logits,
            "selected_routes": selected_routes,
            "governance_logits": self.governance_head(expert_state),
            "taxonomy_logits": self.taxonomy_head(expert_state),
            "scalar_preds": torch.sigmoid(self.scalar_head(expert_state)),
        }


class SwiGLUExpert(nn.Module):
    """Small route-local SwiGLU MLP used by the pooled Stage 0.5 student."""

    def __init__(self, hidden_size: int, expert_hidden_size: int) -> None:
        super().__init__()
        self.gate_up = nn.Linear(hidden_size, expert_hidden_size * 2)
        self.down = nn.Linear(expert_hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(torch.nn.functional.silu(gate) * up)


class RouteCoupledResidualLayer(nn.Module):
    """One residual layer where each example executes only its selected route."""

    def __init__(
        self,
        *,
        hidden_size: int,
        expert_hidden_size: int,
        num_routes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [SwiGLUExpert(hidden_size, expert_hidden_size) for _ in range(num_routes)]
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, state: torch.Tensor, selected_routes: torch.Tensor) -> torch.Tensor:
        expert_delta = torch.empty_like(state)
        for route_id, expert in enumerate(self.experts):
            mask = selected_routes == route_id
            if mask.any():
                expert_delta[mask] = expert(state[mask])
        return self.norm(state + self.dropout(expert_delta))


class RouteCoupledMoEForGovernance(nn.Module):
    """Stage 0.5 custom student with route-coupled execution from the first block.

    The model stays intentionally small and hash-token based, but differs from
    `TinyMoEForGovernance` in the key experimental property: after an early
    pooled route decision, every residual block executes only the selected
    semantic expert path. During training the gold route is used as the active
    path; during normal eval the predicted route controls execution.
    """

    def __init__(self, config: RouteCoupledMoEConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            config.token_vocab_size,
            config.hidden_size,
            padding_idx=0,
        )
        self.input_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.stem = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.SiLU(),
            nn.LayerNorm(config.hidden_size),
        )
        self.router = nn.Linear(config.hidden_size, config.num_routes)
        self.route_embedding = nn.Embedding(config.num_routes, config.hidden_size)
        self.route_norm = nn.LayerNorm(config.hidden_size)
        self.layers = nn.ModuleList(
            [
                RouteCoupledResidualLayer(
                    hidden_size=config.hidden_size,
                    expert_hidden_size=config.expert_hidden_size,
                    num_routes=config.num_routes,
                    dropout=config.dropout,
                )
                for _ in range(config.num_expert_layers)
            ]
        )
        self.governance_head = nn.Linear(config.hidden_size, config.num_labels)
        self.taxonomy_head = nn.Linear(config.hidden_size, config.num_taxonomy_patterns)
        self.scalar_head = nn.Linear(config.hidden_size, config.num_scalar_targets)

    def pool(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        emb = self.input_norm(self.embedding(input_ids))
        mask = attention_mask.unsqueeze(-1).to(dtype=emb.dtype)
        mean_pool = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        masked = emb.masked_fill(mask == 0, torch.finfo(emb.dtype).min)
        max_pool = masked.max(dim=1).values
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))

        return self.dropout(self.stem(torch.cat([mean_pool, max_pool], dim=-1)))

    def select_routes(
        self,
        route_logits: torch.Tensor,
        route_ids: torch.Tensor | None,
        force_route_ids: bool = False,
    ) -> torch.Tensor:
        if route_ids is not None and (self.training or force_route_ids):
            return route_ids
        return route_logits.argmax(dim=-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        route_ids: torch.Tensor | None = None,
        force_route_ids: bool = False,
    ) -> dict[str, torch.Tensor]:
        state = self.pool(input_ids, attention_mask)
        route_logits = self.router(state)
        selected_routes = self.select_routes(
            route_logits,
            route_ids,
            force_route_ids=force_route_ids,
        )
        state = self.route_norm(state + self.route_embedding(selected_routes))
        for layer in self.layers:
            state = layer(state, selected_routes)
        state = self.dropout(state)
        return {
            "route_logits": route_logits,
            "selected_routes": selected_routes,
            "governance_logits": self.governance_head(state),
            "taxonomy_logits": self.taxonomy_head(state),
            "scalar_preds": torch.sigmoid(self.scalar_head(state)),
        }


class CausalSelfAttention(nn.Module):
    """Small GQA self-attention with RoPE for the Stage 0.6 token trunk."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        dropout: float,
        rope_theta: float,
    ) -> None:
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        if self.head_dim % 2 != 0:
            raise ValueError("attention head_dim must be even for RoPE")
        self.kv_repeats = num_attention_heads // num_key_value_heads
        self.rope_theta = rope_theta
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _shape_heads(self, x: torch.Tensor, heads: int) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self._shape_heads(self.q_proj(x), self.num_attention_heads)
        k = self._shape_heads(self.k_proj(x), self.num_key_value_heads)
        v = self._shape_heads(self.v_proj(x), self.num_key_value_heads)
        cos, sin = _rope_cache(
            seq_len=seq_len,
            head_dim=self.head_dim,
            device=x.device,
            dtype=x.dtype,
            theta=self.rope_theta,
        )
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        k = _repeat_kv(k, self.kv_repeats)
        v = _repeat_kv(v, self.kv_repeats)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
        key_mask = attention_mask.to(dtype=torch.bool)[:, None, None, :]
        scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
        attn = functional.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_size)
        return self.o_proj(out)


class RouteCoupledTokenLayer(nn.Module):
    """Decoder-style block: shared attention, route-selected SwiGLU expert FFN."""

    def __init__(
        self,
        *,
        hidden_size: int,
        expert_hidden_size: int,
        num_routes: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        dropout: float,
        rope_theta: float,
    ) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(hidden_size)
        self.attention = CausalSelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            dropout=dropout,
            rope_theta=rope_theta,
        )
        self.expert_norm = RMSNorm(hidden_size)
        self.experts = nn.ModuleList(
            [SwiGLUExpert(hidden_size, expert_hidden_size) for _ in range(num_routes)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        state: torch.Tensor,
        attention_mask: torch.Tensor,
        selected_routes: torch.Tensor,
    ) -> torch.Tensor:
        state = state + self.dropout(self.attention(self.attention_norm(state), attention_mask))
        expert_input = self.expert_norm(state)
        expert_delta = torch.zeros_like(state)
        for route_id, expert in enumerate(self.experts):
            mask = selected_routes == route_id
            if mask.any():
                expert_delta[mask] = expert(expert_input[mask])
        return state + self.dropout(expert_delta)


class TokenRouteCoupledMoEForGovernance(nn.Module):
    """Stage 0.6 token-interaction student with route-coupled expert execution.

    This remains a hash-token prototype, but it is shaped like the terminal
    sparse decoder path: RoPE self-attention in every block, RMSNorm pre-norms,
    SwiGLU route-local experts, and last-token/mean pooled governance heads.
    The selected semantic route controls the expert FFN in every block.
    """

    def __init__(self, config: TokenRouteCoupledMoEConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            config.token_vocab_size,
            config.hidden_size,
            padding_idx=0,
        )
        self.input_norm = RMSNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.route_stem = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.SiLU(),
            RMSNorm(config.hidden_size),
        )
        self.router = nn.Linear(config.hidden_size, config.num_routes)
        self.route_embedding = nn.Embedding(config.num_routes, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                RouteCoupledTokenLayer(
                    hidden_size=config.hidden_size,
                    expert_hidden_size=config.expert_hidden_size,
                    num_routes=config.num_routes,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    dropout=config.dropout,
                    rope_theta=config.rope_theta,
                )
                for _ in range(config.num_expert_layers)
            ]
        )
        self.final_norm = RMSNorm(config.hidden_size)
        self.head_stem = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.SiLU(),
            RMSNorm(config.hidden_size),
        )
        self.governance_head = nn.Linear(config.hidden_size, config.num_labels)
        self.taxonomy_head = nn.Linear(config.hidden_size, config.num_taxonomy_patterns)
        self.scalar_head = nn.Linear(config.hidden_size, config.num_scalar_targets)

    def initial_route_state(
        self,
        token_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(dtype=token_state.dtype)
        mean_pool = (token_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        masked = token_state.masked_fill(mask == 0, torch.finfo(token_state.dtype).min)
        max_pool = masked.max(dim=1).values
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        return self.dropout(self.route_stem(torch.cat([mean_pool, max_pool], dim=-1)))

    def select_routes(
        self,
        route_logits: torch.Tensor,
        route_ids: torch.Tensor | None,
        force_route_ids: bool = False,
    ) -> torch.Tensor:
        if route_ids is not None and (self.training or force_route_ids):
            return route_ids
        return route_logits.argmax(dim=-1)

    def pool_heads(self, state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(dtype=state.dtype)
        mean_pool = (state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        lengths = attention_mask.to(dtype=torch.long).sum(dim=1).clamp_min(1) - 1
        batch_idx = torch.arange(state.shape[0], device=state.device)
        last_pool = state[batch_idx, lengths]
        return self.dropout(self.head_stem(torch.cat([last_pool, mean_pool], dim=-1)))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        route_ids: torch.Tensor | None = None,
        force_route_ids: bool = False,
    ) -> dict[str, torch.Tensor]:
        state = self.input_norm(self.embedding(input_ids))
        route_state = self.initial_route_state(state, attention_mask)
        route_logits = self.router(route_state)
        selected_routes = self.select_routes(
            route_logits,
            route_ids,
            force_route_ids=force_route_ids,
        )
        state = self.dropout(state + self.route_embedding(selected_routes).unsqueeze(1))
        for layer in self.layers:
            state = layer(state, attention_mask, selected_routes)
        pooled = self.pool_heads(self.final_norm(state), attention_mask)
        return {
            "route_logits": route_logits,
            "selected_routes": selected_routes,
            "governance_logits": self.governance_head(pooled),
            "taxonomy_logits": self.taxonomy_head(pooled),
            "scalar_preds": torch.sigmoid(self.scalar_head(pooled)),
        }


class SupportAggregatingMoEForGovernance(TokenRouteCoupledMoEForGovernance):
    """Stage 0.7 token route-coupled student with source-level evidence pooling.

    The flat rendered sequence still drives route prediction and route-selected
    expert execution. The added path separately pools the query and each source,
    scores query-source alignment, and fuses the weighted support state into
    the terminal governance/taxonomy/scalar heads.
    """

    def __init__(self, config: SupportAggregatingMoEConfig) -> None:
        super().__init__(config)
        self.config = config
        self.support_query_stem = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.SiLU(),
            RMSNorm(config.hidden_size),
        )
        self.support_source_stem = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.SiLU(),
            RMSNorm(config.hidden_size),
        )
        self.source_score = nn.Sequential(
            nn.Linear(config.hidden_size * 4, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, 1),
        )
        self.support_fuse = nn.Sequential(
            nn.Linear(config.hidden_size * 4, config.hidden_size),
            nn.SiLU(),
            RMSNorm(config.hidden_size),
        )

    def pool_support_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        stem: nn.Module,
    ) -> torch.Tensor:
        emb = self.input_norm(self.embedding(input_ids))
        mask = attention_mask.unsqueeze(-1).to(dtype=emb.dtype)
        token_counts = mask.sum(dim=1)
        mean_pool = (emb * mask).sum(dim=1) / token_counts.clamp_min(1.0)
        masked = emb.masked_fill(mask == 0, torch.finfo(emb.dtype).min)
        max_pool = masked.max(dim=1).values
        max_pool = torch.where(
            token_counts > 0,
            max_pool,
            torch.zeros_like(max_pool),
        )
        return stem(torch.cat([mean_pool, max_pool], dim=-1))

    def aggregate_support(
        self,
        flat_pooled: torch.Tensor,
        *,
        query_input_ids: torch.Tensor | None,
        query_attention_mask: torch.Tensor | None,
        source_input_ids: torch.Tensor | None,
        source_attention_mask: torch.Tensor | None,
        source_valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if (
            query_input_ids is None
            or query_attention_mask is None
            or source_input_ids is None
            or source_attention_mask is None
            or source_valid_mask is None
        ):
            return flat_pooled

        query_state = self.pool_support_text(
            query_input_ids,
            query_attention_mask,
            self.support_query_stem,
        )
        batch, sources, source_len = source_input_ids.shape
        flat_sources = source_input_ids.reshape(batch * sources, source_len)
        flat_source_mask = source_attention_mask.reshape(batch * sources, source_len)
        source_state = self.pool_support_text(
            flat_sources,
            flat_source_mask,
            self.support_source_stem,
        ).reshape(batch, sources, -1)

        query_by_source = query_state.unsqueeze(1).expand(-1, sources, -1)
        score_features = torch.cat(
            [
                source_state,
                query_by_source,
                source_state * query_by_source,
                torch.abs(source_state - query_by_source),
            ],
            dim=-1,
        )
        source_scores = self.source_score(score_features).squeeze(-1)
        valid_mask = source_valid_mask.to(dtype=torch.bool)
        source_scores = source_scores.masked_fill(
            ~valid_mask,
            torch.finfo(source_scores.dtype).min,
        )
        source_weights = functional.softmax(source_scores, dim=-1)
        has_source = valid_mask.any(dim=1, keepdim=True)
        source_weights = torch.where(
            has_source,
            source_weights,
            torch.zeros_like(source_weights),
        )
        support_state = (source_state * source_weights.unsqueeze(-1)).sum(dim=1)

        masked_sources = source_state.masked_fill(
            ~valid_mask.unsqueeze(-1),
            torch.finfo(source_state.dtype).min,
        )
        source_max = masked_sources.max(dim=1).values
        source_max = torch.where(torch.isfinite(source_max), source_max, torch.zeros_like(source_max))

        return self.dropout(
            self.support_fuse(
                torch.cat([flat_pooled, query_state, support_state, source_max], dim=-1)
            )
        )

    def encode_with_support(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        route_ids: torch.Tensor | None = None,
        force_route_ids: bool = False,
        query_input_ids: torch.Tensor | None = None,
        query_attention_mask: torch.Tensor | None = None,
        source_input_ids: torch.Tensor | None = None,
        source_attention_mask: torch.Tensor | None = None,
        source_valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self.input_norm(self.embedding(input_ids))
        route_state = self.initial_route_state(state, attention_mask)
        route_logits = self.router(route_state)
        selected_routes = self.select_routes(
            route_logits,
            route_ids,
            force_route_ids=force_route_ids,
        )
        state = self.dropout(state + self.route_embedding(selected_routes).unsqueeze(1))
        for layer in self.layers:
            state = layer(state, attention_mask, selected_routes)
        flat_pooled = self.pool_heads(self.final_norm(state), attention_mask)
        pooled = self.aggregate_support(
            flat_pooled,
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
            source_input_ids=source_input_ids,
            source_attention_mask=source_attention_mask,
            source_valid_mask=source_valid_mask,
        )
        return route_logits, selected_routes, pooled

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        route_ids: torch.Tensor | None = None,
        force_route_ids: bool = False,
        query_input_ids: torch.Tensor | None = None,
        query_attention_mask: torch.Tensor | None = None,
        source_input_ids: torch.Tensor | None = None,
        source_attention_mask: torch.Tensor | None = None,
        source_valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        route_logits, selected_routes, pooled = self.encode_with_support(
            input_ids,
            attention_mask,
            route_ids=route_ids,
            force_route_ids=force_route_ids,
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
            source_input_ids=source_input_ids,
            source_attention_mask=source_attention_mask,
            source_valid_mask=source_valid_mask,
        )
        return {
            "route_logits": route_logits,
            "selected_routes": selected_routes,
            "governance_logits": self.governance_head(pooled),
            "taxonomy_logits": self.taxonomy_head(pooled),
            "scalar_preds": torch.sigmoid(self.scalar_head(pooled)),
        }


class GuardedSupportAggregatingMoEForGovernance(SupportAggregatingMoEForGovernance):
    """Stage 0.8 support aggregator with a learned TRUSTWORTHY penalty path."""

    def __init__(self, config: GuardedSupportAggregatingMoEConfig) -> None:
        super().__init__(config)
        self.config = config
        self.trust_penalty_head = nn.Linear(config.hidden_size, 1)
        nn.init.zeros_(self.trust_penalty_head.weight)
        nn.init.constant_(self.trust_penalty_head.bias, config.trust_penalty_init_bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        route_ids: torch.Tensor | None = None,
        force_route_ids: bool = False,
        query_input_ids: torch.Tensor | None = None,
        query_attention_mask: torch.Tensor | None = None,
        source_input_ids: torch.Tensor | None = None,
        source_attention_mask: torch.Tensor | None = None,
        source_valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        route_logits, selected_routes, pooled = self.encode_with_support(
            input_ids,
            attention_mask,
            route_ids,
            force_route_ids=force_route_ids,
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
            source_input_ids=source_input_ids,
            source_attention_mask=source_attention_mask,
            source_valid_mask=source_valid_mask,
        )
        governance_logits = self.governance_head(pooled)
        trust_penalty = functional.softplus(self.trust_penalty_head(pooled)).squeeze(-1)
        trust_penalty = trust_penalty * float(self.config.trust_penalty_scale)
        governance_logits = governance_logits.clone()
        governance_logits[:, 2] = governance_logits[:, 2] - trust_penalty
        return {
            "route_logits": route_logits,
            "selected_routes": selected_routes,
            "governance_logits": governance_logits,
            "trust_penalty": trust_penalty,
            "taxonomy_logits": self.taxonomy_head(pooled),
            "scalar_preds": torch.sigmoid(self.scalar_head(pooled)),
        }


class TrustGuardedSupportAggregatingMoEForGovernance(SupportAggregatingMoEForGovernance):
    """Stage 0.9 support aggregator with a separately supervised trust verifier.

    The main governance head still proposes candidate logits. A small verifier
    then sees the support-fused state plus those candidate logits and predicts
    whether TRUSTWORTHY should be accepted. Low verifier confidence subtracts a
    bounded penalty from the TRUSTWORTHY candidate logit.
    """

    def __init__(self, config: TrustGuardedSupportAggregatingMoEConfig) -> None:
        super().__init__(config)
        self.config = config
        self.trust_guard_head = nn.Sequential(
            nn.Linear(config.hidden_size + config.num_labels, config.hidden_size),
            nn.SiLU(),
            RMSNorm(config.hidden_size),
            nn.Linear(config.hidden_size, 1),
        )
        nn.init.constant_(self.trust_guard_head[-1].bias, config.trust_guard_init_bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        route_ids: torch.Tensor | None = None,
        force_route_ids: bool = False,
        query_input_ids: torch.Tensor | None = None,
        query_attention_mask: torch.Tensor | None = None,
        source_input_ids: torch.Tensor | None = None,
        source_attention_mask: torch.Tensor | None = None,
        source_valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        route_logits, selected_routes, pooled = self.encode_with_support(
            input_ids,
            attention_mask,
            route_ids=route_ids,
            force_route_ids=force_route_ids,
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
            source_input_ids=source_input_ids,
            source_attention_mask=source_attention_mask,
            source_valid_mask=source_valid_mask,
        )
        candidate_logits = self.governance_head(pooled)
        guard_logits_source = (
            candidate_logits.detach()
            if self.config.trust_guard_detach_candidate_logits
            else candidate_logits
        )
        trust_guard_logits = self.trust_guard_head(
            torch.cat([pooled, guard_logits_source], dim=-1)
        ).squeeze(-1)
        trust_penalty = functional.softplus(-trust_guard_logits)
        trust_penalty = trust_penalty * float(self.config.trust_guard_scale)
        governance_logits = candidate_logits.clone()
        governance_logits[:, 2] = governance_logits[:, 2] - trust_penalty
        return {
            "route_logits": route_logits,
            "selected_routes": selected_routes,
            "candidate_governance_logits": candidate_logits,
            "governance_logits": governance_logits,
            "trust_guard_logits": trust_guard_logits,
            "trust_guard_penalty": trust_penalty,
            "taxonomy_logits": self.taxonomy_head(pooled),
            "scalar_preds": torch.sigmoid(self.scalar_head(pooled)),
        }
