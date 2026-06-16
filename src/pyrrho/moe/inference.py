"""End-to-end inference runtime for Stage 0 MoE governance checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml

from pyrrho.data import ID2LABEL, build_encoder_text
from pyrrho.data import LABEL2ID
from pyrrho.metrics import gated_predictions
from pyrrho.moe.data import MoEVocab, hash_tokenize
from pyrrho.moe.modeling import (
    GuardedSupportAggregatingMoEConfig,
    GuardedSupportAggregatingMoEForGovernance,
    RouteCoupledMoEConfig,
    RouteCoupledMoEForGovernance,
    SupportAggregatingMoEConfig,
    SupportAggregatingMoEForGovernance,
    TinyMoEConfig,
    TinyMoEForGovernance,
    TokenRouteCoupledMoEConfig,
    TokenRouteCoupledMoEForGovernance,
    TrustGuardedSupportAggregatingMoEConfig,
    TrustGuardedSupportAggregatingMoEForGovernance,
)
from pyrrho.moe.posthoc_verifier import (
    PosthocVerifierPackage,
    PosthocVerifierResult,
    build_posthoc_features,
    runner_up_non_trustworthy,
    softmax,
)
from pyrrho.moe.posthoc_policies import (
    majority_vote,
    trustworthy_quorum,
)

ENSEMBLE_POLICY_NAMES = (
    "majority_guarded_safety_tie",
    "trustworthy_quorum_2_of_3",
    "trustworthy_unanimous",
)


@dataclass(frozen=True)
class MoEInferenceLengths:
    max_length: int = 768
    max_query_length: int = 96
    max_sources: int = 8
    max_source_length: int = 192


@dataclass(frozen=True)
class PreparedInferenceRow:
    id: str
    query: str
    contexts: list[str]
    text: str
    input_ids: list[int]
    query_input_ids: list[int]
    source_input_ids: list[list[int]]
    source_valid_mask: list[float]


def read_jsonl(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            if not raw.strip():
                continue
            rows.append(json.loads(raw))
            if limit is not None and len(rows) >= int(limit):
                break
    if not rows:
        raise ValueError(f"no rows loaded from {path}")
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _label_id(name: str) -> int:
    try:
        return int(LABEL2ID[str(name)])
    except KeyError:
        raise ValueError(f"unknown governance label {name!r}") from None


def _row_probabilities(row: dict[str, Any]) -> np.ndarray:
    raw = row.get("governance_probabilities")
    if not isinstance(raw, dict):
        raise ValueError(f"prediction row {row.get('id')!r} is missing governance_probabilities")
    return np.asarray([float(raw[ID2LABEL[idx]]) for idx in sorted(ID2LABEL)], dtype=np.float32)


def combine_seed_prediction_rows(
    seed_outputs: dict[int, list[dict[str, Any]]],
    *,
    policy: str,
) -> list[dict[str, Any]]:
    """Combine per-seed inference JSON rows into one ensemble prediction stream."""

    if policy not in ENSEMBLE_POLICY_NAMES:
        raise ValueError(f"unknown ensemble policy {policy!r}")
    seed_ids = sorted(seed_outputs)
    if not seed_ids:
        raise ValueError("at least one seed output is required")
    reference = seed_outputs[seed_ids[0]]
    if not reference:
        raise ValueError("seed outputs must not be empty")
    reference_ids = [str(row["id"]) for row in reference]
    for seed in seed_ids[1:]:
        ids = [str(row["id"]) for row in seed_outputs[seed]]
        if ids != reference_ids:
            raise ValueError(f"seed {seed} prediction rows are not aligned")
    pred_matrix = np.asarray(
        [
            [_label_id(str(row["classification"])) for row in seed_outputs[seed]]
            for seed in seed_ids
        ],
        dtype=np.int64,
    )
    probs = np.asarray(
        [[_row_probabilities(row) for row in seed_outputs[seed]] for seed in seed_ids],
        dtype=np.float32,
    )
    mean_probs = probs.mean(axis=0)
    if policy == "majority_guarded_safety_tie":
        combined = majority_vote(pred_matrix, mean_probs=mean_probs, name=policy)
    elif policy == "trustworthy_quorum_2_of_3":
        combined = trustworthy_quorum(
            pred_matrix,
            quorum=min(2, len(seed_ids)),
            mean_probs=mean_probs,
            name=policy,
        )
    else:
        combined = trustworthy_quorum(
            pred_matrix,
            quorum=len(seed_ids),
            mean_probs=mean_probs,
            name=policy,
        )
    rows = []
    for row_idx, row_id in enumerate(reference_ids):
        rows.append(
            {
                "id": row_id,
                "classification": ID2LABEL[int(combined.predictions[row_idx])],
                "policy": policy,
                "seed_classifications": {
                    str(seed): str(seed_outputs[seed][row_idx]["classification"])
                    for seed in seed_ids
                },
                "seed_rejected": {
                    str(seed): bool(seed_outputs[seed][row_idx].get("verifier_rejected", False))
                    for seed in seed_ids
                },
                "mean_governance_probabilities": {
                    ID2LABEL[label_id]: float(mean_probs[row_idx, label_id])
                    for label_id in sorted(ID2LABEL)
                },
            }
        )
    return rows


def invert_mapping(mapping: dict[str, int]) -> dict[int, str]:
    return {int(v): str(k) for k, v in mapping.items()}


def _extract_context_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "content", "chunk", "summary"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate
    return str(value)


def normalize_inference_row(row: dict[str, Any], index: int) -> tuple[str, str, list[str], str]:
    nested_input = row.get("input") if isinstance(row.get("input"), dict) else {}
    row_id = str(row.get("id") or nested_input.get("id") or f"row_{index}")
    query = str(row.get("query") or nested_input.get("query") or "")
    raw_contexts = row.get("contexts")
    if raw_contexts is None:
        raw_contexts = nested_input.get("contexts")
    if raw_contexts is None:
        raw_contexts = []
    if not isinstance(raw_contexts, list):
        raw_contexts = [raw_contexts]
    contexts = [_extract_context_text(context) for context in raw_contexts]
    text = str(row.get("text") or build_encoder_text(query, contexts))
    return row_id, query, contexts, text


def prepare_inference_row(
    row: dict[str, Any],
    *,
    index: int,
    token_vocab_size: int,
    lengths: MoEInferenceLengths,
) -> PreparedInferenceRow:
    row_id, query, contexts, text = normalize_inference_row(row, index)
    source_ids = [
        hash_tokenize(context, token_vocab_size, lengths.max_source_length)
        for context in contexts[: lengths.max_sources]
    ]
    source_valid = [1.0] * len(source_ids)
    while len(source_ids) < lengths.max_sources:
        source_ids.append([0])
        source_valid.append(0.0)
    return PreparedInferenceRow(
        id=row_id,
        query=query,
        contexts=contexts,
        text=text,
        input_ids=hash_tokenize(text, token_vocab_size, lengths.max_length),
        query_input_ids=hash_tokenize(query, token_vocab_size, lengths.max_query_length),
        source_input_ids=source_ids,
        source_valid_mask=source_valid,
    )


def _pad_2d(rows: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    width = max((len(row) for row in rows), default=1)
    values = []
    mask = []
    for row in rows:
        pad = width - len(row)
        values.append(row + [0] * pad)
        mask.append([1] * len(row) + [0] * pad)
    return torch.tensor(values, dtype=torch.long), torch.tensor(mask, dtype=torch.float32)


def collate_inference_rows(rows: list[PreparedInferenceRow]) -> dict[str, Any]:
    input_ids, attention_mask = _pad_2d([row.input_ids for row in rows])
    query_input_ids, query_attention_mask = _pad_2d([row.query_input_ids for row in rows])
    max_sources = max((len(row.source_input_ids) for row in rows), default=1)
    max_source_len = 1
    for row in rows:
        for source in row.source_input_ids:
            max_source_len = max(max_source_len, len(source))
    source_input_ids = []
    source_attention_mask = []
    source_valid_mask = []
    for row in rows:
        row_sources = list(row.source_input_ids[:max_sources])
        row_valid = list(row.source_valid_mask[:max_sources])
        while len(row_sources) < max_sources:
            row_sources.append([0])
            row_valid.append(0.0)
        padded_sources = []
        padded_masks = []
        for source in row_sources:
            pad = max_source_len - len(source)
            padded_sources.append(source + [0] * pad)
            padded_masks.append([1] * len(source) + [0] * pad)
        source_input_ids.append(padded_sources)
        source_attention_mask.append(padded_masks)
        source_valid_mask.append(row_valid)
    return {
        "ids": [row.id for row in rows],
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "query_input_ids": query_input_ids,
        "query_attention_mask": query_attention_mask,
        "source_input_ids": torch.tensor(source_input_ids, dtype=torch.long),
        "source_attention_mask": torch.tensor(source_attention_mask, dtype=torch.float32),
        "source_valid_mask": torch.tensor(source_valid_mask, dtype=torch.float32),
    }


def load_stage0_model(payload: dict[str, Any]) -> torch.nn.Module:
    model_kind = str(payload.get("model_kind", "tiny"))
    if model_kind == "tiny":
        cfg = TinyMoEConfig(**payload["config"])
        model = TinyMoEForGovernance(cfg)
    elif model_kind == "route_coupled":
        cfg = RouteCoupledMoEConfig(**payload["config"])
        model = RouteCoupledMoEForGovernance(cfg)
    elif model_kind == "route_coupled_token":
        cfg = TokenRouteCoupledMoEConfig(**payload["config"])
        model = TokenRouteCoupledMoEForGovernance(cfg)
    elif model_kind == "support_aggregating_token":
        cfg = SupportAggregatingMoEConfig(**payload["config"])
        model = SupportAggregatingMoEForGovernance(cfg)
    elif model_kind == "guarded_support_aggregating_token":
        cfg = GuardedSupportAggregatingMoEConfig(**payload["config"])
        model = GuardedSupportAggregatingMoEForGovernance(cfg)
    elif model_kind == "trust_guarded_support_aggregating_token":
        cfg = TrustGuardedSupportAggregatingMoEConfig(**payload["config"])
        model = TrustGuardedSupportAggregatingMoEForGovernance(cfg)
    else:
        raise ValueError(f"unknown checkpoint model_kind: {model_kind!r}")
    model.load_state_dict(payload["model_state_dict"])
    return model.eval()


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def model_forward(model: torch.nn.Module, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    kwargs: dict[str, Any] = {"route_ids": None, "force_route_ids": False}
    if bool(getattr(model.config, "uses_support_aggregation", False)):
        kwargs.update(
            {
                "query_input_ids": batch["query_input_ids"],
                "query_attention_mask": batch["query_attention_mask"],
                "source_input_ids": batch["source_input_ids"],
                "source_attention_mask": batch["source_attention_mask"],
                "source_valid_mask": batch["source_valid_mask"],
            }
        )
    return model(batch["input_ids"], batch["attention_mask"], **kwargs)


def lengths_from_config(config_path: Path | None, model_config: Any) -> MoEInferenceLengths:
    stage_cfg: dict[str, Any] = {}
    data_cfg: dict[str, Any] = {}
    if config_path is not None and config_path.exists():
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        stage_cfg = cfg.get("stage0", {}) or {}
        data_cfg = cfg.get("data", {}) or {}
    return MoEInferenceLengths(
        max_length=int(stage_cfg.get("max_seq_length", data_cfg.get("max_seq_length", 768))),
        max_query_length=int(
            stage_cfg.get("max_query_length", getattr(model_config, "max_query_length", 96))
        ),
        max_sources=int(stage_cfg.get("max_sources", getattr(model_config, "max_sources", 8))),
        max_source_length=int(
            stage_cfg.get("max_source_length", getattr(model_config, "max_source_length", 192))
        ),
    )


class MoEInferenceRuntime:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        vocab: MoEVocab,
        lengths: MoEInferenceLengths,
        device: torch.device,
        verifier_package: PosthocVerifierPackage | None = None,
    ) -> None:
        self.model = model.to(device).eval()
        self.vocab = vocab
        self.lengths = lengths
        self.device = device
        self.verifier_package = verifier_package
        self.route_names_by_id = invert_mapping(vocab.route2id)
        self.taxonomy_names_by_id = invert_mapping(vocab.taxonomy_pattern2id)

    @classmethod
    def from_checkpoint(
        cls,
        *,
        checkpoint: Path,
        metadata_path: Path,
        config_path: Path | None = None,
        verifier_package: PosthocVerifierPackage | None = None,
        device: torch.device | None = None,
    ) -> "MoEInferenceRuntime":
        payload = torch.load(checkpoint, map_location="cpu")
        model = load_stage0_model(payload)
        vocab = MoEVocab.from_metadata(metadata_path)
        return cls(
            model=model,
            vocab=vocab,
            lengths=lengths_from_config(config_path, model.config),
            device=device or torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            verifier_package=verifier_package,
        )

    @property
    def token_vocab_size(self) -> int:
        return int(self.model.config.token_vocab_size)

    def predict_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        batch_size: int = 32,
        base_threshold: float = 0.34,
        verifier_seed: int | None = None,
        verifier_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        prepared = [
            prepare_inference_row(
                row,
                index=idx,
                token_vocab_size=self.token_vocab_size,
                lengths=self.lengths,
            )
            for idx, row in enumerate(rows)
        ]
        out: list[dict[str, Any]] = []
        for start in range(0, len(prepared), int(batch_size)):
            chunk = prepared[start : start + int(batch_size)]
            batch = move_batch(collate_inference_rows(chunk), self.device)
            with torch.no_grad():
                outputs = model_forward(self.model, batch)
            governance = outputs["governance_logits"].detach().cpu().numpy()
            route = outputs["route_logits"].detach().cpu().numpy()
            taxonomy = outputs["taxonomy_logits"].detach().cpu().numpy()
            scalars = outputs["scalar_preds"].detach().cpu().numpy()
            result = self._guarded_result(
                governance_logits=governance,
                route_logits=route,
                taxonomy_logits=taxonomy,
                scalar_preds=scalars,
                base_threshold=base_threshold,
                verifier_seed=verifier_seed,
                verifier_threshold=verifier_threshold,
            )
            out.extend(
                self._format_predictions(
                    rows=chunk,
                    governance_logits=governance,
                    route_logits=route,
                    taxonomy_logits=taxonomy,
                    scalar_preds=scalars,
                    result=result,
                    verifier_seed=verifier_seed,
                    verifier_threshold=verifier_threshold,
                    base_threshold=base_threshold,
                )
            )
        return out

    def _guarded_result(
        self,
        *,
        governance_logits: np.ndarray,
        route_logits: np.ndarray,
        taxonomy_logits: np.ndarray,
        scalar_preds: np.ndarray,
        base_threshold: float,
        verifier_seed: int | None,
        verifier_threshold: float | None,
    ) -> PosthocVerifierResult | None:
        if self.verifier_package is None:
            return None
        if verifier_seed is None:
            seed_ids = self.verifier_package.seed_ids
            if len(seed_ids) != 1:
                raise ValueError(f"verifier seed required; package seeds={seed_ids}")
            verifier_seed = seed_ids[0]
        features = build_posthoc_features(
            governance_logits=governance_logits,
            route_logits=route_logits,
            taxonomy_logits=taxonomy_logits,
            scalar_preds=scalar_preds,
        )
        return self.verifier_package.apply_features(
            seed=int(verifier_seed),
            features=features,
            governance_logits=governance_logits,
            base_threshold=base_threshold,
            verifier_threshold=verifier_threshold,
        )

    def _format_predictions(
        self,
        *,
        rows: list[PreparedInferenceRow],
        governance_logits: np.ndarray,
        route_logits: np.ndarray,
        taxonomy_logits: np.ndarray,
        scalar_preds: np.ndarray,
        result: PosthocVerifierResult | None,
        verifier_seed: int | None,
        verifier_threshold: float | None,
        base_threshold: float,
    ) -> list[dict[str, Any]]:
        probs = softmax(governance_logits)
        route_probs = softmax(route_logits)
        taxonomy_probs = softmax(taxonomy_logits)
        base_preds = (
            result.base_predictions
            if result is not None
            else gated_predictions(governance_logits, base_threshold, num_classes=governance_logits.shape[1])
        )
        guarded_preds = result.guarded_predictions if result is not None else base_preds
        runner_up = runner_up_non_trustworthy(governance_logits)
        out = []
        for idx, row in enumerate(rows):
            route_id = int(route_logits[idx].argmax())
            taxonomy_id = int(taxonomy_logits[idx].argmax())
            payload: dict[str, Any] = {
                "id": row.id,
                "classification": ID2LABEL[int(guarded_preds[idx])],
                "base_classification": ID2LABEL[int(base_preds[idx])],
                "guarded_classification": ID2LABEL[int(guarded_preds[idx])],
                "base_threshold": float(base_threshold),
                "governance_probabilities": {
                    ID2LABEL[class_id]: float(probs[idx, class_id])
                    for class_id in sorted(ID2LABEL)
                },
                "selected_route_id": route_id,
                "selected_route": self.route_names_by_id.get(route_id, str(route_id)),
                "taxonomy_pattern_id": taxonomy_id,
                "taxonomy_pattern": self.taxonomy_names_by_id.get(taxonomy_id, str(taxonomy_id)),
                "runner_up_non_trustworthy": ID2LABEL[int(runner_up[idx])],
                "route_confidence": float(route_probs[idx, route_id]),
                "taxonomy_confidence": float(taxonomy_probs[idx, taxonomy_id]),
                "scalar_preds": {
                    field: float(scalar_preds[idx, field_idx])
                    for field_idx, field in enumerate(self.vocab.scalar_fields)
                },
            }
            if result is not None:
                active_seed = int(verifier_seed) if verifier_seed is not None else int(
                    self.verifier_package.seed_ids[0]  # type: ignore[union-attr]
                )
                active_threshold = (
                    float(verifier_threshold)
                    if verifier_threshold is not None
                    else float(self.verifier_package.seed(active_seed).verifier_threshold)  # type: ignore[union-attr]
                )
                payload.update(
                    {
                        "verifier_seed": active_seed,
                        "verifier_threshold": active_threshold,
                        "verifier_accept_score": float(result.accept_scores[idx]),
                        "verifier_rejected": bool(result.rejected_mask[idx]),
                    }
                )
            out.append(payload)
        return out
