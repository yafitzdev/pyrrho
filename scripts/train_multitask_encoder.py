"""Train the pyrrho-nano-g3.1 ModernBERT multi-head encoder.

The g3.1 encoder keeps governance classification as the primary safety head and
adds trained metadata heads:

- pre-retrieval query contract, from query text only
- semantic route/domain, from query text only
- taxonomy pattern, from query + retrieved evidence
- selected governance scalar signals, from query + retrieved evidence
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from pyrrho.data import ID2LABEL
from pyrrho.manifest import write_manifest
from pyrrho.metrics import (
    BASELINE_FALSE_TRUSTWORTHY,
    BASELINE_OVERALL,
    compute_classification_metrics,
    compute_multiclass_metrics,
    find_optimal_threshold,
    gated_predictions,
)
from pyrrho.multitask import PyrrhoMultiTaskConfig, PyrrhoMultiTaskModernBert
from pyrrho.training import set_all_seeds


class MultiTaskJsonlDataset(Dataset):
    """JSONL rows written by `scripts/prepare_moe_data.py`."""

    def __init__(self, path: Path, *, scalar_fields: tuple[str, ...]) -> None:
        self.path = path
        self.scalar_fields = scalar_fields
        self.rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as fh:
            for raw in fh:
                if raw.strip():
                    row = json.loads(raw)
                    self.rows.append(row)
        if not self.rows:
            raise ValueError(f"no rows loaded from {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        scalar_targets = row.get("scalar_targets") or {}
        scalar_values = []
        scalar_mask = []
        for field in self.scalar_fields:
            value = scalar_targets.get(field)
            if isinstance(value, int | float):
                scalar_values.append(float(value))
                scalar_mask.append(1.0)
            else:
                scalar_values.append(0.0)
                scalar_mask.append(0.0)
        return {
            "id": row["id"],
            "text": row["text"],
            "query_text": row.get("query_text") or f"Question: {row.get('query', '')}",
            "label_id": int(row.get("label_id", -1)),
            "query_contract_id": int(row.get("query_contract_id", -1)),
            "route_id": int(row.get("route_id", -1)),
            "taxonomy_pattern_id": int(row.get("taxonomy_pattern_id", -1)),
            "retrieval_action_id": int(row.get("retrieval_action_id", -1)),
            "gap_type_id": int(row.get("gap_type_id", -1)),
            "answerability_shape_id": int(row.get("answerability_shape_id", -1)),
            "retrieval_modality_id": int(row.get("retrieval_modality_id", -1)),
            "retrieval_obligation_id": int(row.get("retrieval_obligation_id", -1)),
            "scalar_targets": scalar_values,
            "scalar_mask": scalar_mask,
        }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def invert_mapping(mapping: dict[str, int]) -> dict[int, str]:
    return {int(v): str(k) for k, v in mapping.items()}


def load_partial_init(
    model: PyrrhoMultiTaskModernBert,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    """Initialize matching tensors from a previous multitask checkpoint."""
    weights_path = checkpoint_dir / "model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(weights_path)
    source_state = load_file(weights_path, device="cpu")
    target_state = model.state_dict()
    copied: list[str] = []
    skipped_shape: list[dict[str, Any]] = []
    skipped_missing: list[str] = []
    for name, target_tensor in target_state.items():
        source_tensor = source_state.get(name)
        if source_tensor is None:
            skipped_missing.append(name)
            continue
        if tuple(source_tensor.shape) != tuple(target_tensor.shape):
            skipped_shape.append(
                {
                    "name": name,
                    "source_shape": list(source_tensor.shape),
                    "target_shape": list(target_tensor.shape),
                }
            )
            continue
        target_state[name] = source_tensor.to(dtype=target_tensor.dtype)
        copied.append(name)
    model.load_state_dict(target_state)
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "copied_tensors": len(copied),
        "skipped_shape": skipped_shape,
        "skipped_missing": skipped_missing,
    }


def class_weights_from_counts(labels: list[int], num_classes: int) -> list[float]:
    valid_labels = [int(label) for label in labels if int(label) >= 0]
    if not valid_labels:
        return [1.0] * num_classes
    counts = Counter(valid_labels)
    total = sum(counts.values())
    weights = []
    for idx in range(num_classes):
        count = counts.get(idx, 0)
        weights.append(float(total / (num_classes * count)) if count else 1.0)
    mean_weight = sum(weights) / len(weights)
    return [float(w / mean_weight) for w in weights]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def collate_builder(
    tokenizer,
    *,
    max_seq_length: int,
    max_query_length: int,
):
    def collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        full = tokenizer(
            [row["text"] for row in rows],
            truncation=True,
            max_length=max_seq_length,
            padding=True,
            return_tensors="pt",
        )
        query = tokenizer(
            [row["query_text"] for row in rows],
            truncation=True,
            max_length=max_query_length,
            padding=True,
            return_tensors="pt",
        )
        return {
            "ids": [row["id"] for row in rows],
            "input_ids": full["input_ids"],
            "attention_mask": full["attention_mask"],
            "query_input_ids": query["input_ids"],
            "query_attention_mask": query["attention_mask"],
            "labels": torch.tensor([row["label_id"] for row in rows], dtype=torch.long),
            "query_contract_ids": torch.tensor(
                [row["query_contract_id"] for row in rows],
                dtype=torch.long,
            ),
            "route_ids": torch.tensor([row["route_id"] for row in rows], dtype=torch.long),
            "taxonomy_ids": torch.tensor(
                [row["taxonomy_pattern_id"] for row in rows],
                dtype=torch.long,
            ),
            "retrieval_action_ids": torch.tensor(
                [row["retrieval_action_id"] for row in rows],
                dtype=torch.long,
            ),
            "gap_type_ids": torch.tensor([row["gap_type_id"] for row in rows], dtype=torch.long),
            "answerability_shape_ids": torch.tensor(
                [row["answerability_shape_id"] for row in rows],
                dtype=torch.long,
            ),
            "retrieval_modality_ids": torch.tensor(
                [row["retrieval_modality_id"] for row in rows],
                dtype=torch.long,
            ),
            "retrieval_obligation_ids": torch.tensor(
                [row["retrieval_obligation_id"] for row in rows],
                dtype=torch.long,
            ),
            "scalar_targets": torch.tensor(
                [row["scalar_targets"] for row in rows],
                dtype=torch.float32,
            ),
            "scalar_mask": torch.tensor([row["scalar_mask"] for row in rows], dtype=torch.float32),
        }

    return collate


def scalar_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    scalar_fields: tuple[str, ...],
) -> dict[str, float]:
    valid = mask > 0
    errors = np.abs(preds - targets)
    sq_errors = (preds - targets) ** 2
    out: dict[str, float] = {
        "scalar_mae": float(errors[valid].mean()) if valid.any() else 0.0,
        "scalar_rmse": float(np.sqrt(sq_errors[valid].mean())) if valid.any() else 0.0,
    }
    for idx, field in enumerate(scalar_fields):
        field_valid = valid[:, idx]
        if field_valid.any():
            out[f"scalar_mae_{field}"] = float(errors[:, idx][field_valid].mean())
            out[f"scalar_rmse_{field}"] = float(np.sqrt(sq_errors[:, idx][field_valid].mean()))
    return out


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = dict(batch)
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
    return out


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    weights: dict[str, float],
    governance_class_weights: torch.Tensor | None,
    query_contract_class_weights: torch.Tensor | None,
    label_smoothing: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    def masked_cross_entropy(
        logits: torch.Tensor | None,
        labels: torch.Tensor | None,
        *,
        weight: torch.Tensor | None = None,
        smoothing: float = 0.0,
    ) -> torch.Tensor:
        if logits is None or labels is None:
            return outputs["governance_logits"].sum() * 0.0
        valid = labels >= 0
        if not bool(valid.any().item()):
            return logits.sum() * 0.0
        return F.cross_entropy(
            logits[valid],
            labels[valid],
            weight=weight,
            label_smoothing=smoothing,
        )

    def optional_cross_entropy(output_key: str, target_key: str) -> torch.Tensor:
        return masked_cross_entropy(outputs.get(output_key), batch.get(target_key))

    gov_loss = masked_cross_entropy(
        outputs["governance_logits"],
        batch["labels"],
        weight=governance_class_weights,
        smoothing=label_smoothing,
    )
    query_contract_loss = masked_cross_entropy(
        outputs["query_contract_logits"],
        batch["query_contract_ids"],
        weight=query_contract_class_weights,
    )
    route_loss = masked_cross_entropy(outputs["route_logits"], batch["route_ids"])
    taxonomy_loss = masked_cross_entropy(outputs["taxonomy_logits"], batch["taxonomy_ids"])
    scalar_diff = (outputs["scalar_preds"] - batch["scalar_targets"]) ** 2
    scalar_mask = batch["scalar_mask"]
    scalar_loss = (scalar_diff * scalar_mask).sum() / scalar_mask.sum().clamp_min(1.0)
    retrieval_action_loss = optional_cross_entropy(
        "retrieval_action_logits", "retrieval_action_ids"
    )
    gap_type_loss = optional_cross_entropy("gap_type_logits", "gap_type_ids")
    answerability_shape_loss = optional_cross_entropy(
        "answerability_shape_logits", "answerability_shape_ids"
    )
    retrieval_modality_loss = optional_cross_entropy(
        "retrieval_modality_logits", "retrieval_modality_ids"
    )
    retrieval_obligation_loss = optional_cross_entropy(
        "retrieval_obligation_logits", "retrieval_obligation_ids"
    )
    total = (
        weights["governance"] * gov_loss
        + weights["query_contract"] * query_contract_loss
        + weights["route"] * route_loss
        + weights["taxonomy"] * taxonomy_loss
        + weights["scalars"] * scalar_loss
        + weights.get("retrieval_action", 0.0) * retrieval_action_loss
        + weights.get("gap_type", 0.0) * gap_type_loss
        + weights.get("answerability_shape", 0.0) * answerability_shape_loss
        + weights.get("retrieval_modality", 0.0) * retrieval_modality_loss
        + weights.get("retrieval_obligation", 0.0) * retrieval_obligation_loss
    )
    return total, {
        "governance": float(gov_loss.detach().cpu()),
        "query_contract": float(query_contract_loss.detach().cpu()),
        "route": float(route_loss.detach().cpu()),
        "taxonomy": float(taxonomy_loss.detach().cpu()),
        "scalars": float(scalar_loss.detach().cpu()),
        "retrieval_action": float(retrieval_action_loss.detach().cpu()),
        "gap_type": float(gap_type_loss.detach().cpu()),
        "answerability_shape": float(answerability_shape_loss.detach().cpu()),
        "retrieval_modality": float(retrieval_modality_loss.detach().cpu()),
        "retrieval_obligation": float(retrieval_obligation_loss.detach().cpu()),
        "total": float(total.detach().cpu()),
    }


def optional_multiclass_metrics(
    logits_chunks: list[np.ndarray],
    label_chunks: list[np.ndarray],
    *,
    id2label: dict[int, str],
    prefix: str,
) -> dict[str, float] | None:
    if not logits_chunks or not label_chunks or not id2label:
        return None
    logits = np.concatenate(logits_chunks, axis=0)
    labels = np.concatenate(label_chunks, axis=0)
    return multiclass_metrics_for_valid(
        logits,
        labels,
        id2label=id2label,
        prefix=prefix,
    )


def multiclass_metrics_for_valid(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    id2label: dict[int, str],
    prefix: str,
) -> dict[str, float] | None:
    if not id2label:
        return None
    valid = labels >= 0
    if not valid.any():
        return None
    return compute_multiclass_metrics(
        logits[valid],
        labels[valid],
        label_ids=sorted(id2label),
        label_names=id2label,
        prefix=prefix,
    )


def required_multiclass_metrics_for_valid(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    id2label: dict[int, str],
    prefix: str,
) -> dict[str, float]:
    metrics = multiclass_metrics_for_valid(
        logits,
        labels,
        id2label=id2label,
        prefix=prefix,
    )
    if metrics is None:
        raise ValueError(f"evaluation split has no valid {prefix} labels")
    return metrics


@torch.no_grad()
def evaluate(
    model: PyrrhoMultiTaskModernBert,
    loader: DataLoader,
    *,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    query_contract_id2label: dict[int, str],
    route_id2label: dict[int, str],
    taxonomy_id2label: dict[int, str],
    retrieval_action_id2label: dict[int, str],
    gap_type_id2label: dict[int, str],
    answerability_shape_id2label: dict[int, str],
    retrieval_modality_id2label: dict[int, str],
    retrieval_obligation_id2label: dict[int, str],
    scalar_fields: tuple[str, ...],
    threshold: float | None = None,
) -> dict[str, Any]:
    model.eval()
    gov_logits: list[np.ndarray] = []
    qc_logits: list[np.ndarray] = []
    route_logits: list[np.ndarray] = []
    taxonomy_logits: list[np.ndarray] = []
    retrieval_action_logits: list[np.ndarray] = []
    gap_type_logits: list[np.ndarray] = []
    answerability_shape_logits: list[np.ndarray] = []
    retrieval_modality_logits: list[np.ndarray] = []
    retrieval_obligation_logits: list[np.ndarray] = []
    scalar_preds: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    qc_labels: list[np.ndarray] = []
    route_labels: list[np.ndarray] = []
    taxonomy_labels: list[np.ndarray] = []
    retrieval_action_labels: list[np.ndarray] = []
    gap_type_labels: list[np.ndarray] = []
    answerability_shape_labels: list[np.ndarray] = []
    retrieval_modality_labels: list[np.ndarray] = []
    retrieval_obligation_labels: list[np.ndarray] = []
    scalar_targets: list[np.ndarray] = []
    scalar_masks: list[np.ndarray] = []

    for raw_batch in tqdm(loader, desc="eval", leave=False):
        batch = move_batch(raw_batch, device)
        with torch.amp.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=autocast_dtype is not None,
        ):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                query_input_ids=batch["query_input_ids"],
                query_attention_mask=batch["query_attention_mask"],
            )
        gov_logits.append(outputs["governance_logits"].float().cpu().numpy())
        qc_logits.append(outputs["query_contract_logits"].float().cpu().numpy())
        route_logits.append(outputs["route_logits"].float().cpu().numpy())
        taxonomy_logits.append(outputs["taxonomy_logits"].float().cpu().numpy())
        if "retrieval_action_logits" in outputs:
            retrieval_action_logits.append(outputs["retrieval_action_logits"].float().cpu().numpy())
            retrieval_action_labels.append(batch["retrieval_action_ids"].cpu().numpy())
        if "gap_type_logits" in outputs:
            gap_type_logits.append(outputs["gap_type_logits"].float().cpu().numpy())
            gap_type_labels.append(batch["gap_type_ids"].cpu().numpy())
        if "answerability_shape_logits" in outputs:
            answerability_shape_logits.append(outputs["answerability_shape_logits"].float().cpu().numpy())
            answerability_shape_labels.append(batch["answerability_shape_ids"].cpu().numpy())
        if "retrieval_modality_logits" in outputs:
            retrieval_modality_logits.append(outputs["retrieval_modality_logits"].float().cpu().numpy())
            retrieval_modality_labels.append(batch["retrieval_modality_ids"].cpu().numpy())
        if "retrieval_obligation_logits" in outputs:
            retrieval_obligation_logits.append(
                outputs["retrieval_obligation_logits"].float().cpu().numpy()
            )
            retrieval_obligation_labels.append(batch["retrieval_obligation_ids"].cpu().numpy())
        scalar_preds.append(outputs["scalar_preds"].float().cpu().numpy())
        labels.append(batch["labels"].cpu().numpy())
        qc_labels.append(batch["query_contract_ids"].cpu().numpy())
        route_labels.append(batch["route_ids"].cpu().numpy())
        taxonomy_labels.append(batch["taxonomy_ids"].cpu().numpy())
        scalar_targets.append(batch["scalar_targets"].cpu().numpy())
        scalar_masks.append(batch["scalar_mask"].cpu().numpy())

    gov = np.concatenate(gov_logits, axis=0)
    qc = np.concatenate(qc_logits, axis=0)
    route = np.concatenate(route_logits, axis=0)
    tax = np.concatenate(taxonomy_logits, axis=0)
    scalar = np.concatenate(scalar_preds, axis=0)
    y = np.concatenate(labels, axis=0)
    qc_y = np.concatenate(qc_labels, axis=0)
    route_y = np.concatenate(route_labels, axis=0)
    tax_y = np.concatenate(taxonomy_labels, axis=0)
    scalar_y = np.concatenate(scalar_targets, axis=0)
    scalar_mask = np.concatenate(scalar_masks, axis=0)
    gov_valid = y >= 0
    if not gov_valid.any():
        raise ValueError("evaluation split has no valid governance labels")
    gov_for_metrics = gov[gov_valid]
    y_for_metrics = y[gov_valid]

    if threshold is None:
        gov_calibrated = find_optimal_threshold(gov_for_metrics, y_for_metrics)
        tau = float(gov_calibrated["threshold"])
    else:
        tau = float(threshold)
        calibrated_preds = gated_predictions(gov_for_metrics, tau, num_classes=3)
        gov_calibrated = compute_classification_metrics(calibrated_preds, y_for_metrics)
        gov_calibrated["threshold"] = tau
        gov_calibrated["target_met"] = bool(
            gov_calibrated["false_trustworthy_rate"] <= BASELINE_FALSE_TRUSTWORTHY
        )

    metrics = {
        "governance_uncalibrated": compute_classification_metrics(gov_for_metrics, y_for_metrics),
        "governance_calibrated": gov_calibrated,
        "threshold": tau,
        "query_contract": required_multiclass_metrics_for_valid(
            qc,
            qc_y,
            id2label=query_contract_id2label,
            prefix="query_contract",
        ),
        "route": required_multiclass_metrics_for_valid(
            route,
            route_y,
            id2label=route_id2label,
            prefix="route",
        ),
        "taxonomy": required_multiclass_metrics_for_valid(
            tax,
            tax_y,
            id2label=taxonomy_id2label,
            prefix="taxonomy",
        ),
        "scalars": scalar_metrics(scalar, scalar_y, scalar_mask, scalar_fields),
    }
    optional_heads = {
        "retrieval_action": optional_multiclass_metrics(
            retrieval_action_logits,
            retrieval_action_labels,
            id2label=retrieval_action_id2label,
            prefix="retrieval_action",
        ),
        "gap_type": optional_multiclass_metrics(
            gap_type_logits,
            gap_type_labels,
            id2label=gap_type_id2label,
            prefix="gap_type",
        ),
        "answerability_shape": optional_multiclass_metrics(
            answerability_shape_logits,
            answerability_shape_labels,
            id2label=answerability_shape_id2label,
            prefix="answerability_shape",
        ),
        "retrieval_modality": optional_multiclass_metrics(
            retrieval_modality_logits,
            retrieval_modality_labels,
            id2label=retrieval_modality_id2label,
            prefix="retrieval_modality",
        ),
        "retrieval_obligation": optional_multiclass_metrics(
            retrieval_obligation_logits,
            retrieval_obligation_labels,
            id2label=retrieval_obligation_id2label,
            prefix="retrieval_obligation",
        ),
    }
    metrics.update({key: value for key, value in optional_heads.items() if value is not None})
    return metrics


def composite_score(metrics: dict[str, Any]) -> float:
    gov = metrics["governance_calibrated"]["ft_penalized_accuracy"]
    qc = metrics["query_contract"]["query_contract_macro_f1"]
    route = metrics["route"]["route_accuracy"]
    taxonomy = metrics["taxonomy"]["taxonomy_accuracy"]
    scalar = 1.0 - metrics["scalars"]["scalar_mae"]
    score = gov + 0.35 * qc + 0.10 * route + 0.10 * taxonomy + 0.05 * scalar
    optional_heads = (
        ("retrieval_action", "retrieval_action_macro_f1", 0.08),
        ("gap_type", "gap_type_macro_f1", 0.08),
        ("answerability_shape", "answerability_shape_macro_f1", 0.04),
        ("retrieval_modality", "retrieval_modality_macro_f1", 0.04),
        ("retrieval_obligation", "retrieval_obligation_macro_f1", 0.06),
    )
    for section, key, weight in optional_heads:
        if section in metrics:
            score += weight * metrics[section][key]
    return float(score)


def print_report(name: str, metrics: dict[str, Any]) -> None:
    gov = metrics["governance_calibrated"]
    qc = metrics["query_contract"]
    route = metrics["route"]
    taxonomy = metrics["taxonomy"]
    scalars = metrics["scalars"]
    optional = []
    if "retrieval_action" in metrics:
        optional.append(f"retrieval_action_f1={metrics['retrieval_action']['retrieval_action_macro_f1']:.4f}")
    if "gap_type" in metrics:
        optional.append(f"gap_type_f1={metrics['gap_type']['gap_type_macro_f1']:.4f}")
    if "answerability_shape" in metrics:
        optional.append(
            f"answerability_shape_f1={metrics['answerability_shape']['answerability_shape_macro_f1']:.4f}"
        )
    if "retrieval_modality" in metrics:
        optional.append(
            f"retrieval_modality_f1={metrics['retrieval_modality']['retrieval_modality_macro_f1']:.4f}"
        )
    if "retrieval_obligation" in metrics:
        optional.append(
            f"retrieval_obligation_f1={metrics['retrieval_obligation']['retrieval_obligation_macro_f1']:.4f}"
        )
    print(
        f"{name}: gov_acc={gov['accuracy']:.4f} gov_FT={gov['false_trustworthy_rate']:.4f} "
        f"tau={metrics['threshold']:.2f} query_contract_f1={qc['query_contract_macro_f1']:.4f} "
        f"route_acc={route['route_accuracy']:.4f} taxonomy_acc={taxonomy['taxonomy_accuracy']:.4f} "
        f"scalar_mae={scalars['scalar_mae']:.4f}"
        + (" " + " ".join(optional) if optional else "")
    )


def main() -> int:
    args = parse_args()
    start = time.time()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    loss_cfg = cfg.get("loss", {})

    data_dir = (args.data_dir or Path(data_cfg["input_dir"])).resolve()
    output_dir = (args.output_dir or Path(train_cfg["output_dir"])).resolve()
    seed = int(args.seed if args.seed is not None else train_cfg.get("seed", 42))
    set_all_seeds(seed)

    metadata = load_json(data_dir / "metadata.json")
    scalar_fields = tuple(str(v) for v in metadata["scalar_fields"])
    query_contract_id2label = invert_mapping(metadata["query_contract2id"])
    route_id2label = invert_mapping(metadata["route2id"])
    taxonomy_id2label = invert_mapping(metadata["taxonomy_pattern2id"])
    has_retrieval_control = bool(metadata.get("require_retrieval_control", False))
    retrieval_action_id2label = (
        invert_mapping(metadata.get("retrieval_action2id", {})) if has_retrieval_control else {}
    )
    gap_type_id2label = (
        invert_mapping(metadata.get("gap_type2id", {})) if has_retrieval_control else {}
    )
    answerability_shape_id2label = (
        invert_mapping(metadata.get("answerability_shape2id", {})) if has_retrieval_control else {}
    )
    retrieval_modality_id2label = (
        invert_mapping(metadata.get("retrieval_modality2id", {})) if has_retrieval_control else {}
    )
    retrieval_obligation_id2label = (
        invert_mapping(metadata.get("retrieval_obligation2id", {})) if has_retrieval_control else {}
    )

    train_ds = MultiTaskJsonlDataset(data_dir / "train.jsonl", scalar_fields=scalar_fields)
    eval_ds = MultiTaskJsonlDataset(data_dir / "eval.jsonl", scalar_fields=scalar_fields)
    test_ds = (
        MultiTaskJsonlDataset(data_dir / "test.jsonl", scalar_fields=scalar_fields)
        if (data_dir / "test.jsonl").exists()
        else None
    )

    tokenizer_source = model_cfg.get("tokenizer_from") or model_cfg.get("init_from") or model_cfg["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    collate = collate_builder(
        tokenizer,
        max_seq_length=int(data_cfg.get("max_seq_length", 4096)),
        max_query_length=int(data_cfg.get("max_query_length", 256)),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg["per_device_train_batch_size"]),
        shuffle=True,
        collate_fn=collate,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=int(train_cfg["per_device_eval_batch_size"]),
        shuffle=False,
        collate_fn=collate,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = (
        DataLoader(
            test_ds,
            batch_size=int(train_cfg["per_device_eval_batch_size"]),
            shuffle=False,
            collate_fn=collate,
            num_workers=int(train_cfg.get("num_workers", 0)),
            pin_memory=torch.cuda.is_available(),
        )
        if test_ds is not None
        else None
    )

    mt_config = PyrrhoMultiTaskConfig(
        base_model=str(model_cfg["base_model"]),
        num_governance_labels=3,
        num_query_contract_labels=len(query_contract_id2label),
        num_routes=len(route_id2label),
        num_taxonomy_patterns=len(taxonomy_id2label),
        scalar_fields=scalar_fields,
        id2label=ID2LABEL,
        query_contract_id2label=query_contract_id2label,
        route_id2label=route_id2label,
        taxonomy_id2label=taxonomy_id2label,
        retrieval_action_id2label=retrieval_action_id2label or None,
        gap_type_id2label=gap_type_id2label or None,
        answerability_shape_id2label=answerability_shape_id2label or None,
        retrieval_modality_id2label=retrieval_modality_id2label or None,
        retrieval_obligation_id2label=retrieval_obligation_id2label or None,
        dropout=float(model_cfg.get("dropout", 0.0)),
    )
    model = PyrrhoMultiTaskModernBert(mt_config)
    init_report = None
    init_from = model_cfg.get("init_from")
    if init_from:
        init_report = load_partial_init(model, Path(str(init_from)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    autocast_dtype = (
        torch.bfloat16
        if device.type == "cuda" and bool(train_cfg.get("bf16", True))
        else None
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Config        : {args.config}")
    print(f"Data dir      : {data_dir}")
    print(f"Output dir    : {output_dir}")
    print(f"Device        : {device}")
    print(f"Seed          : {seed}")
    if init_report is not None:
        print(f"Init from     : {init_report['checkpoint_dir']}")
        print(
            "Init tensors  : copied={copied} shape_skipped={shape} missing={missing}".format(
                copied=init_report["copied_tensors"],
                shape=len(init_report["skipped_shape"]),
                missing=len(init_report["skipped_missing"]),
            )
        )
    print(f"Splits        : train={len(train_ds)} eval={len(eval_ds)} test={len(test_ds) if test_ds else 0}")
    print(
        "Heads         : "
        f"query_contract={len(query_contract_id2label)} "
        f"route={len(route_id2label)} "
        f"taxonomy={len(taxonomy_id2label)} "
        f"retrieval_action={len(retrieval_action_id2label)} "
        f"gap_type={len(gap_type_id2label)} "
        f"answerability_shape={len(answerability_shape_id2label)} "
        f"retrieval_modality={len(retrieval_modality_id2label)} "
        f"retrieval_obligation={len(retrieval_obligation_id2label)} "
        f"scalars={len(scalar_fields)}"
    )

    if args.dry_run:
        print("DRY RUN: built dataset/model/loaders; exiting before training.")
        return 0

    governance_weights = train_cfg.get("governance_class_weights", [2.3, 2.3, 1.0])
    governance_class_weights = torch.tensor(governance_weights, dtype=torch.float32, device=device)
    query_contract_weights = None
    if bool(train_cfg.get("balance_query_contract", True)):
        query_contract_weights = class_weights_from_counts(
            [int(row.get("query_contract_id", -1)) for row in train_ds.rows],
            len(query_contract_id2label),
        )
    query_contract_class_weights = (
        torch.tensor(query_contract_weights, dtype=torch.float32, device=device)
        if query_contract_weights
        else None
    )
    weights = {
        "governance": float(loss_cfg.get("governance", 1.0)),
        "query_contract": float(loss_cfg.get("query_contract", 0.5)),
        "route": float(loss_cfg.get("route", 0.2)),
        "taxonomy": float(loss_cfg.get("taxonomy", 0.2)),
        "scalars": float(loss_cfg.get("scalars", 0.2)),
        "retrieval_action": float(loss_cfg.get("retrieval_action", 0.0)),
        "gap_type": float(loss_cfg.get("gap_type", 0.0)),
        "answerability_shape": float(loss_cfg.get("answerability_shape", 0.0)),
        "retrieval_modality": float(loss_cfg.get("retrieval_modality", 0.0)),
        "retrieval_obligation": float(loss_cfg.get("retrieval_obligation", 0.0)),
    }
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 1))
    epochs = float(train_cfg["num_train_epochs"])
    max_steps = math.ceil(len(train_loader) * epochs / grad_accum)
    warmup_steps = int(max_steps * float(train_cfg.get("warmup_ratio", 0.0)))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )

    best_score = -1e9
    best_epoch = 0
    best_eval: dict[str, Any] | None = None
    global_step = 0
    model.train()
    for epoch_idx in range(1, int(math.ceil(epochs)) + 1):
        loss_meter: Counter[str] = Counter()
        seen_batches = 0
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(train_loader, desc=f"epoch {epoch_idx}")
        for step_idx, raw_batch in enumerate(progress, start=1):
            batch = move_batch(raw_batch, device)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_dtype is not None,
            ):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    query_input_ids=batch["query_input_ids"],
                    query_attention_mask=batch["query_attention_mask"],
                )
                loss, loss_parts = compute_loss(
                    outputs,
                    batch,
                    weights=weights,
                    governance_class_weights=governance_class_weights,
                    query_contract_class_weights=query_contract_class_weights,
                    label_smoothing=label_smoothing,
                )
                loss = loss / grad_accum
            loss.backward()
            for key, value in loss_parts.items():
                loss_meter[key] += value
            seen_batches += 1
            if step_idx % grad_accum == 0 or step_idx == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("max_grad_norm", 1.0)))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            progress.set_postfix(total=f"{loss_parts['total']:.4f}")

        avg_loss = {key: value / max(seen_batches, 1) for key, value in loss_meter.items()}
        print(f"\nEpoch {epoch_idx} train loss: {avg_loss}")
        eval_metrics = evaluate(
            model,
            eval_loader,
            device=device,
            autocast_dtype=autocast_dtype,
            query_contract_id2label=query_contract_id2label,
            route_id2label=route_id2label,
            taxonomy_id2label=taxonomy_id2label,
            retrieval_action_id2label=retrieval_action_id2label,
            gap_type_id2label=gap_type_id2label,
            answerability_shape_id2label=answerability_shape_id2label,
            retrieval_modality_id2label=retrieval_modality_id2label,
            retrieval_obligation_id2label=retrieval_obligation_id2label,
            scalar_fields=scalar_fields,
        )
        score = composite_score(eval_metrics)
        print_report(f"Epoch {epoch_idx} eval", eval_metrics)
        print(f"Epoch {epoch_idx} composite={score:.4f}")
        if score > best_score:
            best_score = score
            best_epoch = epoch_idx
            best_eval = eval_metrics
            best_dir = output_dir / "best_model"
            if best_dir.exists():
                shutil.rmtree(best_dir)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"Saved best model -> {best_dir}")

    if best_eval is None:
        raise RuntimeError("training finished without an eval pass")

    best_model = PyrrhoMultiTaskModernBert.from_pretrained(output_dir / "best_model")
    best_model.to(device)
    best_model.eval()
    test_metrics = None
    if test_loader is not None:
        test_metrics = evaluate(
            best_model,
            test_loader,
            device=device,
            autocast_dtype=autocast_dtype,
            query_contract_id2label=query_contract_id2label,
            route_id2label=route_id2label,
            taxonomy_id2label=taxonomy_id2label,
            retrieval_action_id2label=retrieval_action_id2label,
            gap_type_id2label=gap_type_id2label,
            answerability_shape_id2label=answerability_shape_id2label,
            retrieval_modality_id2label=retrieval_modality_id2label,
            retrieval_obligation_id2label=retrieval_obligation_id2label,
            scalar_fields=scalar_fields,
            threshold=float(best_eval["threshold"]),
        )
        print_report("Best held-out test", test_metrics)

    final_metrics = {
        "config_path": str(args.config),
        "data_dir": str(data_dir),
        "seed": seed,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "eval": best_eval,
        "test": test_metrics,
        "target_met_on_test": bool(
            test_metrics
            and test_metrics["governance_calibrated"]["accuracy"] >= BASELINE_OVERALL
            and test_metrics["governance_calibrated"]["false_trustworthy_rate"] <= BASELINE_FALSE_TRUSTWORTHY
        ),
        "loss_weights": weights,
        "governance_class_weights": governance_weights,
        "query_contract_class_weights": query_contract_weights,
        "init_from": str(init_from) if init_from else None,
        "init_report": init_report,
        "global_step": global_step,
        "elapsed_seconds": time.time() - start,
    }
    (output_dir / "final_metrics.json").write_text(
        json.dumps(final_metrics, indent=2),
        encoding="utf-8",
    )
    write_manifest(
        output_dir=output_dir,
        config_path=args.config,
        seed=seed,
        pyrrho_repo=Path.cwd(),
        fitz_gov_repo=(Path.cwd().parent / "fitz-gov"),
        start_time=start,
        extra={
            "script": "train_multitask_encoder.py",
            "base_model": model_cfg["base_model"],
            "init_from": str(init_from) if init_from else None,
            "data_dir": str(data_dir),
            "best_epoch": best_epoch,
            "best_score": best_score,
            "heads": {
                "governance": 3,
                "query_contract": len(query_contract_id2label),
                "route": len(route_id2label),
                "taxonomy": len(taxonomy_id2label),
                "retrieval_action": len(retrieval_action_id2label),
                "gap_type": len(gap_type_id2label),
                "answerability_shape": len(answerability_shape_id2label),
                "retrieval_modality": len(retrieval_modality_id2label),
                "retrieval_obligation": len(retrieval_obligation_id2label),
                "scalars": list(scalar_fields),
            },
        },
    )
    print(f"\nWrote final metrics -> {output_dir / 'final_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
