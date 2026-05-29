"""Configuration and parameter accounting for the custom pyrrho-MoE target."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

DEFAULT_SEMANTIC_EXPERT_GROUPS: tuple[str, ...] = (
    "science_medicine",
    "law_policy",
    "history_geography",
    "technology_computing",
    "economics_finance",
    "culture_society",
    "general_commonsense",
    "conflict_detection",
)


@dataclass(frozen=True)
class MoEParameterCounts:
    """Exact component counts for a pyrrho-MoE config.

    The active count follows the project convention from
    `docs/PYRRHO_MOE_ARCHITECTURE.md`: top-1 selected experts plus the shared
    trunk, with both inclusive and embedding-excluded variants reported.
    """

    embedding: int
    attention_blocks: int
    dense_ffns: int
    moe_expert_bank: int
    routers: int
    norms: int
    task_heads: int
    total: int
    active_selected_experts: int
    active_inclusive: int
    active_excluding_embedding: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class PyrrhoMoEConfig:
    """Architecture math for `pyrrho-moe-g3-alpha`.

    Defaults match the canonical baseline: 24 layers, hidden size 1024, 20 MoE
    layers, 16 physical experts per MoE layer, top-1 routing, and a 64k tied
    tokenizer assumption.
    """

    layers: int = 24
    hidden_size: int = 1024
    attention_heads: int = 16
    attention_head_dim: int | None = None
    kv_heads: int = 4
    attention_qk_norms: bool = False
    ffn_dim: int = 3840
    dense_ffn_layers: int = 4
    moe_ffn_layers: int = 20
    experts_per_moe_layer: int = 16
    top_k: int = 1
    vocab_size: int = 64_000
    tied_embeddings: bool = True
    semantic_expert_groups: tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_SEMANTIC_EXPERT_GROUPS
    )
    taxonomy_patterns: int = 23
    scalar_heads: int = 15
    governance_classes: int = 3

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> PyrrhoMoEConfig:
        """Build from a YAML/JSON mapping, ignoring unknown keys."""
        raw = dict(raw or {})
        if "semantic_expert_groups" in raw and isinstance(raw["semantic_expert_groups"], list):
            raw["semantic_expert_groups"] = tuple(raw["semantic_expert_groups"])
        if "head_dim" in raw and "attention_head_dim" not in raw:
            raw["attention_head_dim"] = raw.pop("head_dim")
        allowed = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in raw.items() if k in allowed})

    @property
    def head_dim(self) -> int:
        self.validate()
        if self.attention_head_dim is not None:
            return self.attention_head_dim
        return self.hidden_size // self.attention_heads

    @property
    def q_dim(self) -> int:
        return self.attention_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def semantic_group_count(self) -> int:
        return len(self.semantic_expert_groups)

    @property
    def physical_shards_per_group(self) -> int:
        self.validate()
        return self.experts_per_moe_layer // self.semantic_group_count

    def validate(self) -> None:
        """Raise `ValueError` if dimensions cannot produce the intended topology."""
        if self.layers <= 0:
            raise ValueError("layers must be positive")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.attention_heads <= 0:
            raise ValueError("attention_heads must be positive")
        if self.attention_head_dim is None and self.hidden_size % self.attention_heads != 0:
            raise ValueError("hidden_size must be divisible by attention_heads")
        if self.attention_head_dim is not None and self.attention_head_dim <= 0:
            raise ValueError("attention_head_dim must be positive")
        if self.kv_heads <= 0 or self.kv_heads > self.attention_heads:
            raise ValueError("kv_heads must be in 1..attention_heads")
        if self.dense_ffn_layers < 0 or self.moe_ffn_layers < 0:
            raise ValueError("FFN layer counts cannot be negative")
        if self.dense_ffn_layers + self.moe_ffn_layers != self.layers:
            raise ValueError("dense_ffn_layers + moe_ffn_layers must equal layers")
        if self.top_k != 1:
            raise ValueError("CPU release config currently supports only top_k=1")
        if not self.semantic_expert_groups:
            raise ValueError("at least one semantic expert group is required")
        if self.experts_per_moe_layer % len(self.semantic_expert_groups) != 0:
            raise ValueError("experts_per_moe_layer must divide evenly across semantic groups")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

    def parameter_counts(self) -> MoEParameterCounts:
        """Return component-level parameter counts for the configured architecture."""
        self.validate()
        h = self.hidden_size
        head_dim = self.head_dim
        kv_dim = self.kv_heads * head_dim

        embedding = self.vocab_size * h
        if not self.tied_embeddings:
            embedding *= 2

        q_dim = self.q_dim
        attention_per_layer = (h * q_dim) + (2 * h * kv_dim) + (h * q_dim)
        attention_blocks = self.layers * attention_per_layer

        ffn_params = 3 * h * self.ffn_dim
        dense_ffns = self.dense_ffn_layers * ffn_params
        moe_expert_bank = self.moe_ffn_layers * self.experts_per_moe_layer * ffn_params
        routers = self.moe_ffn_layers * h * self.experts_per_moe_layer

        # Two RMSNorm vectors per block plus final norm. Biases are excluded
        # because the baseline uses RMSNorm/pre-norm and bias-free projections.
        norms = (2 * self.layers + 1) * h
        if self.attention_qk_norms:
            norms += 2 * self.layers * head_dim

        task_heads = (
            h * self.governance_classes
            + self.governance_classes
            + h * self.semantic_group_count
            + self.semantic_group_count
            + h * self.taxonomy_patterns
            + self.taxonomy_patterns
            + h * self.scalar_heads
            + self.scalar_heads
        )

        total = (
            embedding
            + attention_blocks
            + dense_ffns
            + moe_expert_bank
            + routers
            + norms
            + task_heads
        )

        active_selected_experts = self.moe_ffn_layers * self.top_k * ffn_params
        active_inclusive = (
            embedding
            + attention_blocks
            + dense_ffns
            + routers
            + norms
            + task_heads
            + active_selected_experts
        )

        return MoEParameterCounts(
            embedding=embedding,
            attention_blocks=attention_blocks,
            dense_ffns=dense_ffns,
            moe_expert_bank=moe_expert_bank,
            routers=routers,
            norms=norms,
            task_heads=task_heads,
            total=total,
            active_selected_experts=active_selected_experts,
            active_inclusive=active_inclusive,
            active_excluding_embedding=active_inclusive - embedding,
        )

    def budget_report(self) -> dict[str, Any]:
        counts = self.parameter_counts()
        return {
            "config": asdict(self),
            "derived": {
                "head_dim": self.head_dim,
                "q_dim": self.q_dim,
                "kv_dim": self.kv_dim,
                "semantic_group_count": self.semantic_group_count,
                "physical_shards_per_group": self.physical_shards_per_group,
            },
            "parameters": counts.to_dict(),
            "budget_checks": {
                "total_3_9b_to_4_1b": 3_900_000_000 <= counts.total <= 4_100_000_000,
                "active_inclusive_0_38b_to_0_43b": 380_000_000
                <= counts.active_inclusive
                <= 430_000_000,
                "active_excluding_embedding_under_0_43b": counts.active_excluding_embedding
                <= 430_000_000,
            },
        }
