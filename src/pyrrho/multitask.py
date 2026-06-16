"""ModernBERT multi-head encoder for pyrrho nano g3.1."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoConfig, AutoModel


@dataclass(frozen=True)
class PyrrhoMultiTaskConfig:
    base_model: str
    num_governance_labels: int
    num_query_contract_labels: int
    num_routes: int
    num_taxonomy_patterns: int
    scalar_fields: tuple[str, ...]
    id2label: dict[int, str]
    query_contract_id2label: dict[int, str]
    route_id2label: dict[int, str]
    taxonomy_id2label: dict[int, str]
    retrieval_action_id2label: dict[int, str] | None = None
    gap_type_id2label: dict[int, str] | None = None
    answerability_shape_id2label: dict[int, str] | None = None
    retrieval_modality_id2label: dict[int, str] | None = None
    retrieval_obligation_id2label: dict[int, str] | None = None
    dropout: float = 0.0

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> PyrrhoMultiTaskConfig:
        return cls(
            base_model=str(raw["base_model"]),
            num_governance_labels=int(raw["num_governance_labels"]),
            num_query_contract_labels=int(raw["num_query_contract_labels"]),
            num_routes=int(raw["num_routes"]),
            num_taxonomy_patterns=int(raw["num_taxonomy_patterns"]),
            scalar_fields=tuple(str(v) for v in raw.get("scalar_fields", [])),
            id2label={int(k): str(v) for k, v in dict(raw["id2label"]).items()},
            query_contract_id2label={
                int(k): str(v) for k, v in dict(raw["query_contract_id2label"]).items()
            },
            route_id2label={int(k): str(v) for k, v in dict(raw["route_id2label"]).items()},
            taxonomy_id2label={int(k): str(v) for k, v in dict(raw["taxonomy_id2label"]).items()},
            retrieval_action_id2label=_optional_id2label(raw.get("retrieval_action_id2label")),
            gap_type_id2label=_optional_id2label(raw.get("gap_type_id2label")),
            answerability_shape_id2label=_optional_id2label(
                raw.get("answerability_shape_id2label")
            ),
            retrieval_modality_id2label=_optional_id2label(
                raw.get("retrieval_modality_id2label")
            ),
            retrieval_obligation_id2label=_optional_id2label(
                raw.get("retrieval_obligation_id2label")
            ),
            dropout=float(raw.get("dropout", 0.0)),
        )

    def to_mapping(self) -> dict[str, Any]:
        data = asdict(self)
        data["scalar_fields"] = list(self.scalar_fields)
        data["id2label"] = {str(k): v for k, v in self.id2label.items()}
        data["query_contract_id2label"] = {
            str(k): v for k, v in self.query_contract_id2label.items()
        }
        data["route_id2label"] = {str(k): v for k, v in self.route_id2label.items()}
        data["taxonomy_id2label"] = {str(k): v for k, v in self.taxonomy_id2label.items()}
        for key in (
            "retrieval_action_id2label",
            "gap_type_id2label",
            "answerability_shape_id2label",
            "retrieval_modality_id2label",
            "retrieval_obligation_id2label",
        ):
            mapping = getattr(self, key)
            data[key] = {str(k): v for k, v in mapping.items()} if mapping else None
        return data

    @property
    def num_retrieval_action_labels(self) -> int:
        return len(self.retrieval_action_id2label or {})

    @property
    def num_gap_type_labels(self) -> int:
        return len(self.gap_type_id2label or {})

    @property
    def num_answerability_shape_labels(self) -> int:
        return len(self.answerability_shape_id2label or {})

    @property
    def num_retrieval_modality_labels(self) -> int:
        return len(self.retrieval_modality_id2label or {})

    @property
    def num_retrieval_obligation_labels(self) -> int:
        return len(self.retrieval_obligation_id2label or {})


def _optional_id2label(raw: Any) -> dict[int, str] | None:
    if raw is None:
        return None
    mapping = {int(k): str(v) for k, v in dict(raw).items()}
    return mapping or None


class PyrrhoMultiTaskModernBert(nn.Module):
    """Shared ModernBERT trunk with query-only and evidence-conditioned heads."""

    def __init__(
        self,
        config: PyrrhoMultiTaskConfig,
        *,
        backbone_config: Any | None = None,
        load_pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.pyrrho_config = config
        backbone_config = backbone_config or AutoConfig.from_pretrained(config.base_model)
        self.backbone = (
            AutoModel.from_pretrained(config.base_model, config=backbone_config)
            if load_pretrained_backbone
            else AutoModel.from_config(backbone_config)
        )
        hidden_size = int(backbone_config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.governance_head = nn.Linear(hidden_size, config.num_governance_labels)
        self.query_contract_head = nn.Linear(hidden_size, config.num_query_contract_labels)
        self.route_head = nn.Linear(hidden_size, config.num_routes)
        self.taxonomy_head = nn.Linear(hidden_size, config.num_taxonomy_patterns)
        self.scalar_head = nn.Linear(hidden_size, len(config.scalar_fields))
        self.retrieval_action_head = (
            nn.Linear(hidden_size, config.num_retrieval_action_labels)
            if config.num_retrieval_action_labels
            else None
        )
        self.gap_type_head = (
            nn.Linear(hidden_size, config.num_gap_type_labels)
            if config.num_gap_type_labels
            else None
        )
        self.answerability_shape_head = (
            nn.Linear(hidden_size, config.num_answerability_shape_labels)
            if config.num_answerability_shape_labels
            else None
        )
        self.retrieval_modality_head = (
            nn.Linear(hidden_size, config.num_retrieval_modality_labels)
            if config.num_retrieval_modality_labels
            else None
        )
        self.retrieval_obligation_head = (
            nn.Linear(hidden_size, config.num_retrieval_obligation_labels)
            if config.num_retrieval_obligation_labels
            else None
        )

    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
        return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def _encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self.dropout(self._mean_pool(outputs.last_hidden_state, attention_mask))

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        evidence_state = self._encode(input_ids, attention_mask)
        query_state = self._encode(query_input_ids, query_attention_mask)
        outputs = {
            "governance_logits": self.governance_head(evidence_state),
            "query_contract_logits": self.query_contract_head(query_state),
            "route_logits": self.route_head(query_state),
            "taxonomy_logits": self.taxonomy_head(evidence_state),
            "scalar_preds": torch.sigmoid(self.scalar_head(evidence_state)),
        }
        if self.retrieval_action_head is not None:
            outputs["retrieval_action_logits"] = self.retrieval_action_head(evidence_state)
        if self.gap_type_head is not None:
            outputs["gap_type_logits"] = self.gap_type_head(evidence_state)
        if self.answerability_shape_head is not None:
            outputs["answerability_shape_logits"] = self.answerability_shape_head(query_state)
        if self.retrieval_modality_head is not None:
            outputs["retrieval_modality_logits"] = self.retrieval_modality_head(query_state)
        if self.retrieval_obligation_head is not None:
            outputs["retrieval_obligation_logits"] = self.retrieval_obligation_head(query_state)
        return outputs

    def save_pretrained(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.backbone.config.save_pretrained(output)
        save_file(self.state_dict(), output / "model.safetensors")
        (output / "pyrrho_multitask_config.json").write_text(
            json.dumps(self.pyrrho_config.to_mapping(), indent=2),
            encoding="utf-8",
        )

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> PyrrhoMultiTaskModernBert:
        model_path = Path(model_dir)
        raw = json.loads((model_path / "pyrrho_multitask_config.json").read_text(encoding="utf-8"))
        config = PyrrhoMultiTaskConfig.from_mapping(raw)
        backbone_config_ref = (
            model_path if (model_path / "config.json").exists() else config.base_model
        )
        backbone_config = AutoConfig.from_pretrained(backbone_config_ref)
        model = cls(
            config,
            backbone_config=backbone_config,
            load_pretrained_backbone=False,
        )
        state = load_file(model_path / "model.safetensors", device=str(map_location))
        model.load_state_dict(state)
        return model
