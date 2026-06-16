"""Minimal generative SFT smoke for local pyrrho-MoE CausalLM checkpoints.

This started as the Qwen-seeded MVP harness and now also serves stock carrier
checks such as the OLMoE g4-real structural checkpoint.
It trains a small LoRA adapter on compact pyrrho JSON output, then runs greedy
generation and scores JSON parseability plus governance/route/taxonomy fields.

Run from project root:
    python scripts/train_moe_qwen_sft.py --max-steps 1 --max-train-samples 2 --max-eval-samples 2
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from pyrrho.metrics import compute_classification_metrics
from pyrrho.moe.data import MoEVocab, load_teacher_logits
from pyrrho.training import set_all_seeds

LABELS = ("ABSTAIN", "DISPUTED", "TRUSTWORTHY")
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
DEFAULT_SIGNAL_FIELDS = (
    "false_trustworthy_risk",
    "retrieval_retry_value",
    "query_evidence_alignment",
    "answer_coverage",
)
DEFAULT_RATIONALES = {
    "ABSTAIN": "The retrieved sources do not provide sufficient direct evidence for the query.",
    "DISPUTED": "The retrieved sources contain conflicting evidence for the query.",
    "TRUSTWORTHY": "The retrieved sources directly support the requested answer.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--seed-pack",
        type=Path,
        default=Path("outputs/moe/upcycling/qwen_alpha_seed_pack"),
        help="Local AutoModelForCausalLM checkpoint or seed pack",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/moe_v8"))
    parser.add_argument("--train-split", choices=("train", "eval", "test"), default="train")
    parser.add_argument("--eval-split", choices=("train", "eval", "test"), default="eval")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/moe/qwen_generative_sft_smoke"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run-label",
        default="generative-sft",
        help="Short label used in progress bars and run reports.",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--max-train-samples", type=int, default=64)
    parser.add_argument("--max-eval-samples", type=int, default=64)
    parser.add_argument(
        "--sample-mode",
        choices=("random", "prefix", "balanced-label"),
        default="random",
        help="How to choose bounded train/eval subsets",
    )
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--class-weights",
        default="1,1,1",
        help="Comma-separated ABSTAIN,DISPUTED,TRUSTWORTHY target-token loss weights",
    )
    parser.add_argument(
        "--classification-loss-weight",
        type=float,
        default=1.0,
        help="Extra multiplier for the classification value tokens inside the JSON target",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Token-level CE label smoothing for the target sequence",
    )
    parser.add_argument(
        "--aux-classifier-weight",
        type=float,
        default=0.0,
        help="Auxiliary prompt-state governance classifier loss weight",
    )
    parser.add_argument(
        "--aux-classifier-lr",
        type=float,
        default=0.0,
        help="Auxiliary classifier learning rate; defaults to --learning-rate when <= 0",
    )
    parser.add_argument(
        "--aux-label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing for the auxiliary governance classifier",
    )
    parser.add_argument(
        "--aux-detach",
        action="store_true",
        help="Train the auxiliary classifier without backpropagating through the causal LM",
    )
    parser.add_argument(
        "--eval-label-source",
        choices=("generation", "aux", "label-score"),
        default="generation",
        help="Which label source drives reported classification metrics",
    )
    parser.add_argument(
        "--label-score-length-normalization",
        choices=("mean", "sum"),
        default="mean",
        help="How to normalize candidate label logprobs for label-score eval.",
    )
    parser.add_argument(
        "--label-score-trustworthy-threshold",
        type=float,
        default=0.0,
        help=(
            "If >0, demote label-score TRUSTWORTHY predictions below this "
            "candidate probability to the best non-TRUSTWORTHY label."
        ),
    )
    parser.add_argument(
        "--eval-skip-generation",
        action="store_true",
        help=(
            "During eval, skip free-text generation and report only the selected "
            "non-generative label source. Requires --eval-label-source != generation."
        ),
    )
    parser.add_argument(
        "--teacher-logits-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing train/eval/test JSONL teacher logits keyed by row id. "
            "Used by --label-distillation-weight."
        ),
    )
    parser.add_argument(
        "--label-distillation-weight",
        type=float,
        default=0.0,
        help=(
            "KL weight for distilling label-candidate probabilities from teacher logits. "
            "The student scores ABSTAIN/DISPUTED/TRUSTWORTHY continuations from the prompt."
        ),
    )
    parser.add_argument(
        "--distillation-temperature",
        type=float,
        default=2.0,
        help="Temperature for label-candidate teacher distillation.",
    )
    parser.add_argument(
        "--label-distillation-length-normalization",
        choices=("sum", "mean"),
        default="mean",
        help="How to normalize candidate label sequence log probabilities for distillation.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help='Transformers device_map value. Use "none" to load on the default device, or "cpu" for {"": "cpu"}.',
    )
    parser.add_argument(
        "--quantization",
        choices=("none", "bnb-4bit", "bnb-8bit"),
        default="none",
        help="Optional bitsandbytes quantized load mode for inference/runtime probes.",
    )
    parser.add_argument(
        "--bnb-4bit-quant-type",
        choices=("nf4", "fp4"),
        default="nf4",
        help="bitsandbytes 4-bit quantization type when --quantization bnb-4bit is used.",
    )
    parser.add_argument(
        "--bnb-4bit-double-quant",
        dest="bnb_4bit_double_quant",
        action="store_true",
        default=True,
        help="Use nested quantization for bitsandbytes 4-bit loads.",
    )
    parser.add_argument(
        "--no-bnb-4bit-double-quant",
        dest="bnb_4bit_double_quant",
        action="store_false",
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated Linear module suffixes for PEFT LoRA",
    )
    parser.add_argument(
        "--unfreeze-parameter-patterns",
        default="",
        help=(
            "Comma-separated substrings for base parameters to unfreeze after "
            "the normal LoRA/freeze setup. Intended for controlled diagnostics "
            "such as OLMoE router or lm_head adaptation."
        ),
    )
    parser.add_argument(
        "--olmoe-expert-down-lora-r",
        type=int,
        default=0,
        help=(
            "Rank for an OLMoE-only float32 LoRA diagnostic on raw expert down_proj tensors. "
            "This reaches expert parameters that PEFT cannot target because OLMoE stores "
            "experts as 3D nn.Parameter tensors."
        ),
    )
    parser.add_argument(
        "--olmoe-expert-down-lora-alpha",
        type=float,
        default=8.0,
        help="Scaling alpha for --olmoe-expert-down-lora-r.",
    )
    parser.add_argument(
        "--olmoe-expert-gate-up-lora-r",
        type=int,
        default=0,
        help=(
            "Rank for an OLMoE-only float32 LoRA diagnostic on raw expert gate_up_proj "
            "tensors. This is a stronger expert adaptation surface than down_proj-only LoRA."
        ),
    )
    parser.add_argument(
        "--olmoe-expert-gate-up-lora-alpha",
        type=float,
        default=8.0,
        help="Scaling alpha for --olmoe-expert-gate-up-lora-r.",
    )
    parser.add_argument(
        "--olmoe-router-lora-r",
        type=int,
        default=0,
        help=(
            "Rank for an OLMoE-only float32 LoRA diagnostic on router logits. "
            "This wraps OlmoeTopKRouter without changing the stock carrier config."
        ),
    )
    parser.add_argument(
        "--olmoe-router-lora-alpha",
        type=float,
        default=8.0,
        help="Scaling alpha for --olmoe-router-lora-r.",
    )
    parser.add_argument(
        "--fallback-label",
        choices=LABELS,
        default="ABSTAIN",
        help="Classification used when generation does not contain a parseable label",
    )
    parser.add_argument(
        "--short-rationale",
        action="store_true",
        help="Use a short template rationale to isolate JSON/label format acquisition",
    )
    parser.add_argument(
        "--target-mode",
        choices=("json", "label-only", "label-json"),
        default="json",
        help="Assistant target format. label-only and label-json are decision diagnostics.",
    )
    parser.add_argument(
        "--save-adapter",
        action="store_true",
        help="Save the LoRA adapter under output-dir/final_adapter",
    )
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--adapter-path", type=Path, default=None)
    return parser.parse_args()


def parse_class_weights(raw: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if len(parts) != len(LABELS):
        raise ValueError(
            f"--class-weights must have {len(LABELS)} comma-separated values "
            f"for {','.join(LABELS)}"
        )
    weights = tuple(float(part) for part in parts)
    if any(weight <= 0 for weight in weights):
        raise ValueError("--class-weights values must be positive")
    return weights  # type: ignore[return-value]


def resolve_dtype(raw: str) -> torch.dtype | str:
    if raw == "auto":
        return "auto"
    if raw == "bfloat16":
        return torch.bfloat16
    if raw == "float16":
        return torch.float16
    if raw == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {raw}")


def resolve_device_map(raw: str) -> Any:
    if raw == "none":
        return None
    if raw == "cpu":
        return {"": "cpu"}
    return raw


def build_quantization_config(args: argparse.Namespace, dtype: torch.dtype | str) -> Any | None:
    if getattr(args, "quantization", "none") == "none":
        return None
    from transformers import BitsAndBytesConfig

    compute_dtype = dtype if isinstance(dtype, torch.dtype) else torch.bfloat16
    if args.quantization == "bnb-4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.bnb_4bit_double_quant,
        )
    if args.quantization == "bnb-8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError(f"unsupported quantization: {args.quantization}")


def read_rows(
    path: Path,
    *,
    limit: int | None,
    sample_mode: str,
    sample_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                rows.append(json.loads(raw))
    if not rows:
        raise ValueError(f"no rows loaded from {path}")
    if limit is not None and limit < len(rows):
        if sample_mode == "prefix":
            rows = rows[:limit]
        elif sample_mode == "random":
            rng = random.Random(sample_seed)
            indexes = list(range(len(rows)))
            rng.shuffle(indexes)
            selected = sorted(indexes[:limit])
            rows = [rows[idx] for idx in selected]
        elif sample_mode == "balanced-label":
            rng = random.Random(sample_seed)
            by_label: dict[str, list[int]] = {}
            for idx, row in enumerate(rows):
                by_label.setdefault(str(row.get("label")), []).append(idx)
            labels = sorted(by_label)
            if not labels:
                rows = rows[:limit]
            else:
                base = limit // len(labels)
                remainder = limit % len(labels)
                selected: list[int] = []
                for label_index, label in enumerate(labels):
                    candidates = list(by_label[label])
                    rng.shuffle(candidates)
                    take = base + int(label_index < remainder)
                    selected.extend(candidates[:take])
                if len(selected) < limit:
                    remaining = [idx for idx in range(len(rows)) if idx not in set(selected)]
                    rng.shuffle(remaining)
                    selected.extend(remaining[: limit - len(selected)])
                rows = [rows[idx] for idx in sorted(selected[:limit])]
        else:
            raise ValueError(f"unsupported sample_mode: {sample_mode}")
    return rows


def teacher_sidecar_path(
    teacher_logits_dir: Path | None,
    split: str,
    *,
    required: bool,
) -> Path | None:
    if teacher_logits_dir is None:
        if required:
            raise ValueError("--teacher-logits-dir is required when label distillation is enabled")
        return None
    path = teacher_logits_dir / f"{split}.jsonl"
    if required and not path.exists():
        raise FileNotFoundError(f"teacher logits sidecar not found: {path}")
    return path if path.exists() else None


def attach_teacher_logits_to_rows(
    rows: list[dict[str, Any]],
    path: Path | None,
    *,
    required: bool,
) -> int:
    if path is None:
        return 0
    sidecar = load_teacher_logits(path)
    attached = 0
    missing: list[str] = []
    for row in rows:
        row_id = str(row["id"])
        logits = sidecar.get(row_id)
        if logits is None:
            missing.append(row_id)
            continue
        row["teacher_logits"] = logits
        attached += 1
    if missing and required:
        sample = ", ".join(missing[:5])
        raise ValueError(
            f"teacher logits missing for {len(missing)} selected rows in {path}; "
            f"examples: {sample}"
        )
    return attached


def format_sources(contexts: Iterable[Any]) -> str:
    lines = []
    for idx, context in enumerate(contexts or [], start=1):
        lines.append(f"[{idx}] {str(context).strip()}")
    return "\n".join(lines)


def build_prompt(row: dict[str, Any], *, target_mode: str = "json") -> str:
    if target_mode == "label-only":
        return (
            "You are pyrrho-MoE, a RAG governance model. Given a question and "
            "retrieved sources, return only one classification label: ABSTAIN, "
            "DISPUTED, or TRUSTWORTHY.\n\n"
            f"Question: {str(row.get('query') or '').strip()}\n\n"
            f"Sources:\n{format_sources(row.get('contexts') or [])}\n\n"
            "Classification:"
        )
    if target_mode == "label-json":
        return (
            "You are pyrrho-MoE, a RAG governance model. Given a question and "
            "retrieved sources, return one classification label on the first line, "
            "then compact JSON with keys classification, rationale, route, "
            "taxonomy_pattern, and signals. The label must be one of ABSTAIN, "
            "DISPUTED, TRUSTWORTHY.\n\n"
            f"Question: {str(row.get('query') or '').strip()}\n\n"
            f"Sources:\n{format_sources(row.get('contexts') or [])}\n\n"
            "Output:"
        )
    return (
        "You are pyrrho-MoE, a RAG governance model. Given a question and retrieved "
        "sources, return only compact JSON with keys classification, rationale, route, "
        "taxonomy_pattern, and signals. classification must be one of ABSTAIN, "
        "DISPUTED, TRUSTWORTHY.\n\n"
        f"Question: {str(row.get('query') or '').strip()}\n\n"
        f"Sources:\n{format_sources(row.get('contexts') or [])}\n\n"
        "JSON:"
    )


def short_rationale(row: dict[str, Any]) -> str:
    chain = row.get("evidence_chain") or {}
    reasoning = chain.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()
    reason = row.get("near_miss_reason")
    if isinstance(reason, str) and reason.strip():
        return reason.strip()
    label = str(row.get("label") or "").upper()
    if label == "TRUSTWORTHY":
        return "The retrieved sources directly support the requested answer."
    if label == "DISPUTED":
        return "The retrieved sources contain conflicting evidence for the query."
    return "The retrieved sources do not provide sufficient direct evidence for the query."


def build_target_json(
    row: dict[str, Any],
    signal_fields: Iterable[str],
    *,
    short_rationale_only: bool,
) -> str:
    scalar_targets = row.get("scalar_targets") or {}
    signals: dict[str, float] = {}
    for field in signal_fields:
        value = scalar_targets.get(field)
        if isinstance(value, int | float):
            signals[str(field)] = round(float(value), 4)
    payload = {
        "classification": str(row["label"]),
        "rationale": (
            f"Evidence state is {str(row['label']).lower()} for the requested query."
            if short_rationale_only
            else short_rationale(row)
        ),
        "route": str(row["route"]),
        "taxonomy_pattern": str(row["taxonomy_pattern"]),
        "signals": signals,
    }
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def build_target(
    row: dict[str, Any],
    signal_fields: Iterable[str],
    *,
    short_rationale_only: bool,
    target_mode: str,
) -> str:
    if target_mode == "label-only":
        return str(row["label"])
    if target_mode == "label-json":
        return (
            f"{str(row['label'])}\n"
            + build_target_json(
                row,
                signal_fields,
                short_rationale_only=short_rationale_only,
            )
        )
    return build_target_json(
        row,
        signal_fields,
        short_rationale_only=short_rationale_only,
    )


class GenerativeMoEDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        tokenizer: Any,
        max_length: int,
        signal_fields: Iterable[str],
        include_target: bool,
        short_rationale_only: bool,
        class_weights: tuple[float, float, float],
        classification_loss_weight: float,
        target_mode: str,
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.signal_fields = tuple(signal_fields)
        self.include_target = bool(include_target)
        self.short_rationale_only = bool(short_rationale_only)
        self.class_weights = tuple(float(weight) for weight in class_weights)
        self.classification_loss_weight = float(classification_loss_weight)
        self.target_mode = str(target_mode)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        prompt = build_prompt(row, target_mode=self.target_mode)
        target = build_target(
            row,
            self.signal_fields,
            short_rationale_only=self.short_rationale_only,
            target_mode=self.target_mode,
        )
        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]
        if not self.include_target:
            prompt_ids = prompt_ids[: self.max_length]
            return {
                "id": row["id"],
                "prompt": prompt,
                "input_ids": prompt_ids,
                "prompt_len": len(prompt_ids),
                "label_id": int(row["label_id"]),
                "route_id": int(row["route_id"]),
                "taxonomy_pattern_id": int(row["taxonomy_pattern_id"]),
                "label": str(row["label"]),
                "route": str(row["route"]),
                "taxonomy_pattern": str(row["taxonomy_pattern"]),
                "target": target,
                "teacher_logits": row.get("teacher_logits"),
            }

        eos = self.tokenizer.eos_token or ""
        target_ids = self.tokenizer(
            target + eos,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        if len(target_ids) >= self.max_length:
            target_ids = target_ids[: self.max_length - 1]
        max_prompt_len = max(1, self.max_length - len(target_ids))
        prompt_ids = prompt_ids[:max_prompt_len]
        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        class_weight = self.class_weights[int(row["label_id"])]
        loss_weights = [0.0] * len(prompt_ids) + [class_weight] * len(target_ids)
        label_token_ids = self.tokenizer(
            str(row["label"]),
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        if label_token_ids and self.classification_loss_weight != 1.0:
            for offset in range(0, len(target_ids) - len(label_token_ids) + 1):
                if target_ids[offset : offset + len(label_token_ids)] == label_token_ids:
                    start = len(prompt_ids) + offset
                    end = start + len(label_token_ids)
                    for weight_idx in range(start, end):
                        loss_weights[weight_idx] *= self.classification_loss_weight
                    break
        return {
            "id": row["id"],
            "input_ids": input_ids,
            "labels": labels,
            "loss_weights": loss_weights,
            "prompt_len": len(prompt_ids),
            "label_id": int(row["label_id"]),
            "route_id": int(row["route_id"]),
            "taxonomy_pattern_id": int(row["taxonomy_pattern_id"]),
            "label": str(row["label"]),
            "route": str(row["route"]),
            "taxonomy_pattern": str(row["taxonomy_pattern"]),
            "target": target,
            "teacher_logits": row.get("teacher_logits"),
        }


class CausalCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        max_len = max(len(row["input_ids"]) for row in rows)
        input_ids = []
        attention_mask = []
        labels = []
        loss_weights = []
        teacher_values = []
        teacher_masks = []
        prompt_indices = []
        has_labels = "labels" in rows[0]
        for row in rows:
            pad = max_len - len(row["input_ids"])
            if has_labels:
                input_ids.append(row["input_ids"] + [self.pad_token_id] * pad)
                attention_mask.append([1] * len(row["input_ids"]) + [0] * pad)
                prompt_indices.append(max(0, int(row["prompt_len"]) - 1))
            else:
                input_ids.append([self.pad_token_id] * pad + row["input_ids"])
                attention_mask.append([0] * pad + [1] * len(row["input_ids"]))
                prompt_indices.append(max_len - 1)
            if has_labels:
                labels.append(row["labels"] + [-100] * pad)
                loss_weights.append(row["loss_weights"] + [0.0] * pad)
            teacher_logits = row.get("teacher_logits")
            if isinstance(teacher_logits, list) and len(teacher_logits) == len(LABELS):
                teacher_values.append([float(value) for value in teacher_logits])
                teacher_masks.append(1.0)
            else:
                teacher_values.append([0.0, 0.0, 0.0])
                teacher_masks.append(0.0)
        batch: dict[str, Any] = {
            "ids": [row["id"] for row in rows],
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "prompt_indices": torch.tensor(prompt_indices, dtype=torch.long),
            "label_ids": torch.tensor([row["label_id"] for row in rows], dtype=torch.long),
            "route_ids": torch.tensor([row["route_id"] for row in rows], dtype=torch.long),
            "taxonomy_ids": torch.tensor(
                [row["taxonomy_pattern_id"] for row in rows],
                dtype=torch.long,
            ),
            "labels_text": [row["label"] for row in rows],
            "routes_text": [row["route"] for row in rows],
            "taxonomy_text": [row["taxonomy_pattern"] for row in rows],
            "targets": [row["target"] for row in rows],
            "teacher_logits": torch.tensor(teacher_values, dtype=torch.float32),
            "teacher_mask": torch.tensor(teacher_masks, dtype=torch.float32),
        }
        if has_labels:
            batch["labels"] = torch.tensor(labels, dtype=torch.long)
            batch["loss_weights"] = torch.tensor(loss_weights, dtype=torch.float32)
        return batch


def model_input_device(model: torch.nn.Module) -> torch.device:
    embeddings = model.get_input_embeddings()
    return embeddings.weight.device


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def disable_router_aux_outputs(model: Any) -> None:
    candidates = [model]
    for attr in ("base_model", "model"):
        current = getattr(model, attr, None)
        if current is not None:
            candidates.append(current)
            nested = getattr(current, "model", None)
            if nested is not None:
                candidates.append(nested)
    for candidate in candidates:
        config = getattr(candidate, "config", None)
        if config is not None:
            setattr(config, "output_router_logits", False)
            setattr(config, "router_aux_loss_coef", 0.0)
        generation_config = getattr(candidate, "generation_config", None)
        if generation_config is not None:
            setattr(generation_config, "output_router_logits", False)


def parse_unfreeze_parameter_patterns(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def apply_extra_unfreeze_patterns(
    model: torch.nn.Module,
    raw_patterns: str,
    *,
    enabled: bool,
) -> dict[str, Any]:
    patterns = parse_unfreeze_parameter_patterns(raw_patterns)
    if not patterns or not enabled:
        return {"patterns": patterns, "parameters": 0, "tensors": 0}

    matched_tensors = 0
    matched_parameters = 0
    for name, param in model.named_parameters():
        if any(pattern in name for pattern in patterns):
            param.requires_grad_(True)
            matched_tensors += 1
            matched_parameters += int(param.numel())

    if matched_tensors == 0:
        raise ValueError(
            "--unfreeze-parameter-patterns did not match any parameters: "
            + ", ".join(patterns)
        )
    return {
        "patterns": patterns,
        "parameters": matched_parameters,
        "tensors": matched_tensors,
    }


class OlmoeExpertLora(torch.nn.Module):
    """LoRA diagnostic for OLMoE expert projections stored as raw tensors."""

    def __init__(
        self,
        base_experts: torch.nn.Module,
        *,
        down_rank: int,
        down_alpha: float,
        gate_up_rank: int,
        gate_up_alpha: float,
    ) -> None:
        super().__init__()
        if down_rank <= 0 and gate_up_rank <= 0:
            raise ValueError("at least one expert LoRA rank must be positive")
        self.base_experts = base_experts
        self.num_experts = int(base_experts.num_experts)
        self.hidden_dim = int(base_experts.hidden_dim)
        self.intermediate_dim = int(base_experts.intermediate_dim)
        self.down_rank = int(down_rank)
        self.gate_up_rank = int(gate_up_rank)
        self.down_scaling = (
            float(down_alpha) / float(self.down_rank) if self.down_rank > 0 else 0.0
        )
        self.gate_up_scaling = (
            float(gate_up_alpha) / float(self.gate_up_rank) if self.gate_up_rank > 0 else 0.0
        )

        down_device = base_experts.down_proj.device
        gate_up_device = base_experts.gate_up_proj.device
        if self.down_rank > 0:
            self.down_lora_a = torch.nn.Parameter(
                torch.empty(
                    self.num_experts,
                    self.down_rank,
                    self.intermediate_dim,
                    dtype=torch.float32,
                    device=down_device,
                )
            )
            self.down_lora_b = torch.nn.Parameter(
                torch.zeros(
                    self.num_experts,
                    self.hidden_dim,
                    self.down_rank,
                    dtype=torch.float32,
                    device=down_device,
                )
            )
            torch.nn.init.kaiming_uniform_(self.down_lora_a, a=math.sqrt(5))
        if self.gate_up_rank > 0:
            self.gate_up_lora_a = torch.nn.Parameter(
                torch.empty(
                    self.num_experts,
                    self.gate_up_rank,
                    self.hidden_dim,
                    dtype=torch.float32,
                    device=gate_up_device,
                )
            )
            self.gate_up_lora_b = torch.nn.Parameter(
                torch.zeros(
                    self.num_experts,
                    self.intermediate_dim * 2,
                    self.gate_up_rank,
                    dtype=torch.float32,
                    device=gate_up_device,
                )
            )
            torch.nn.init.kaiming_uniform_(self.gate_up_lora_a, a=math.sqrt(5))

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(
                top_k_index,
                num_classes=self.num_experts,
            )
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx_tensor in expert_hit:
            expert_idx = int(expert_idx_tensor[0])
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate_up = torch.nn.functional.linear(
                current_state,
                self.base_experts.gate_up_proj[expert_idx],
            )
            if self.gate_up_rank > 0:
                gate_up_lora_hidden = torch.nn.functional.linear(
                    current_state.float(),
                    self.gate_up_lora_a[expert_idx],
                )
                gate_up_lora_output = torch.nn.functional.linear(
                    gate_up_lora_hidden,
                    self.gate_up_lora_b[expert_idx],
                )
                gate_up = gate_up + (gate_up_lora_output * self.gate_up_scaling).to(gate_up.dtype)
            gate, up = gate_up.chunk(2, dim=-1)
            expert_hidden = self.base_experts.act_fn(gate) * up
            expert_output = torch.nn.functional.linear(
                expert_hidden,
                self.base_experts.down_proj[expert_idx],
            )
            if self.down_rank > 0:
                down_lora_hidden = torch.nn.functional.linear(
                    expert_hidden.float(),
                    self.down_lora_a[expert_idx],
                )
                down_lora_output = torch.nn.functional.linear(
                    down_lora_hidden,
                    self.down_lora_b[expert_idx],
                )
                expert_output = expert_output + (down_lora_output * self.down_scaling).to(
                    expert_output.dtype
                )
            expert_output = expert_output * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, expert_output.to(final_hidden_states.dtype))

        return final_hidden_states


class OlmoeRouterLora(torch.nn.Module):
    """LoRA diagnostic for OLMoE router logits stored as a raw weight tensor."""

    def __init__(self, base_router: torch.nn.Module, *, rank: int, alpha: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base_router = base_router
        self.top_k = int(base_router.top_k)
        self.num_experts = int(base_router.num_experts)
        self.norm_topk_prob = bool(base_router.norm_topk_prob)
        self.hidden_dim = int(base_router.hidden_dim)
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        device = base_router.weight.device
        self.lora_a = torch.nn.Parameter(
            torch.empty(self.rank, self.hidden_dim, dtype=torch.float32, device=device)
        )
        self.lora_b = torch.nn.Parameter(
            torch.zeros(self.num_experts, self.rank, dtype=torch.float32, device=device)
        )
        torch.nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = torch.nn.functional.linear(hidden_states, self.base_router.weight)
        lora_hidden = torch.nn.functional.linear(hidden_states.float(), self.lora_a)
        lora_logits = torch.nn.functional.linear(lora_hidden, self.lora_b)
        router_logits = router_logits + (lora_logits * self.scaling).to(router_logits.dtype)
        router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        router_scores = router_top_value.to(router_logits.dtype)
        return router_logits, router_scores, router_indices


def attach_olmoe_raw_lora_adapters(
    model: torch.nn.Module,
    args: argparse.Namespace,
) -> dict[str, Any]:
    down_rank = int(args.olmoe_expert_down_lora_r)
    gate_up_rank = int(args.olmoe_expert_gate_up_lora_r)
    router_rank = int(args.olmoe_router_lora_r)
    if down_rank <= 0 and gate_up_rank <= 0 and router_rank <= 0:
        return {
            "expert_down_lora": {"enabled": False, "rank": down_rank, "layers": 0, "parameters": 0},
            "expert_gate_up_lora": {
                "enabled": False,
                "rank": gate_up_rank,
                "layers": 0,
                "parameters": 0,
            },
            "router_lora": {"enabled": False, "rank": router_rank, "layers": 0, "parameters": 0},
        }

    expert_layers = 0
    router_layers = 0
    router_params = 0
    for module in model.modules():
        if module.__class__.__name__ != "OlmoeSparseMoeBlock":
            continue
        if down_rank > 0 or gate_up_rank > 0:
            module.experts = OlmoeExpertLora(
                module.experts,
                down_rank=down_rank,
                down_alpha=args.olmoe_expert_down_lora_alpha,
                gate_up_rank=gate_up_rank,
                gate_up_alpha=args.olmoe_expert_gate_up_lora_alpha,
            )
            expert_layers += 1
        if router_rank > 0:
            module.gate = OlmoeRouterLora(
                module.gate,
                rank=router_rank,
                alpha=args.olmoe_router_lora_alpha,
            )
            router_layers += 1
            router_params += sum(
                int(param.numel()) for param in module.gate.parameters() if param.requires_grad
            )

    if (down_rank > 0 or gate_up_rank > 0) and expert_layers == 0:
        raise ValueError(
            "OLMoE expert LoRA was requested, but no OlmoeSparseMoeBlock modules were found"
        )
    if router_rank > 0 and router_layers == 0:
        raise ValueError("--olmoe-router-lora-r was set, but no OlmoeSparseMoeBlock modules were found")

    return {
        "expert_down_lora": {
            "enabled": down_rank > 0,
            "rank": down_rank,
            "alpha": float(args.olmoe_expert_down_lora_alpha),
            "layers": expert_layers,
            "parameters": sum(
                int(param.numel())
                for module in model.modules()
                if module.__class__.__name__ == "OlmoeExpertLora"
                for name, param in module.named_parameters()
                if name.startswith("down_lora_") and param.requires_grad
            ),
        },
        "expert_gate_up_lora": {
            "enabled": gate_up_rank > 0,
            "rank": gate_up_rank,
            "alpha": float(args.olmoe_expert_gate_up_lora_alpha),
            "layers": expert_layers,
            "parameters": sum(
                int(param.numel())
                for module in model.modules()
                if module.__class__.__name__ == "OlmoeExpertLora"
                for name, param in module.named_parameters()
                if name.startswith("gate_up_lora_") and param.requires_grad
            ),
        },
        "router_lora": {
            "enabled": router_rank > 0,
            "rank": router_rank,
            "alpha": float(args.olmoe_router_lora_alpha),
            "layers": router_layers,
            "parameters": router_params,
        },
    }


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any]:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.seed_pack,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": True,
        "low_cpu_mem_usage": True,
    }
    dtype = resolve_dtype(args.dtype)
    if dtype is not None:
        kwargs["dtype"] = dtype
    quantization_config = build_quantization_config(args, dtype)
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    device_map = resolve_device_map(args.device_map)
    if device_map is not None:
        kwargs["device_map"] = device_map
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(args.seed_pack, **kwargs)
    disable_router_aux_outputs(model)
    model.config.use_cache = False

    if args.adapter_path is not None:
        model = load_peft_adapter_with_conversion_fallback(
            PeftModel,
            model,
            args.adapter_path,
            is_trainable=not args.eval_only,
        )
    elif args.lora_r > 0:
        targets = [
            target.strip()
            for target in args.lora_target_modules.split(",")
            if target.strip()
        ]
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=targets,
        )
        model = get_peft_model(model, lora_config)
    else:
        for param in model.parameters():
            param.requires_grad_(False)

    extra_unfrozen = apply_extra_unfreeze_patterns(
        model,
        args.unfreeze_parameter_patterns,
        enabled=not args.eval_only,
    )
    olmoe_adapters = attach_olmoe_raw_lora_adapters(model, args)
    setattr(model, "_pyrrho_extra_unfrozen", extra_unfrozen)
    setattr(model, "_pyrrho_olmoe_adapters", olmoe_adapters)
    disable_router_aux_outputs(model)
    return model, tokenizer


def load_peft_adapter_with_conversion_fallback(
    peft_model_cls: Any,
    model: torch.nn.Module,
    adapter_path: Path,
    *,
    is_trainable: bool,
) -> torch.nn.Module:
    model_type = str(getattr(getattr(model, "config", None), "model_type", ""))
    if model_type in {"qwen2_moe", "qwen3_moe"}:
        return load_peft_adapter_without_transformers_v5_conversion(
            peft_model_cls,
            model,
            adapter_path,
            is_trainable=is_trainable,
        )
    try:
        return peft_model_cls.from_pretrained(
            model,
            adapter_path,
            is_trainable=is_trainable,
        )
    except TypeError as exc:
        message = str(exc)
        if "WeightConverter.__init__()" not in message or "distributed_operation" not in message:
            raise

    # PEFT 0.19.1 + Transformers 5.x attempts an MoE adapter key conversion that
    # is incompatible with the local Qwen3-MoE converter class. These adapters are
    # trained against the same local seed pack, so no architecture conversion is
    # needed on reload.
    return load_peft_adapter_without_transformers_v5_conversion(
        peft_model_cls,
        model,
        adapter_path,
        is_trainable=is_trainable,
    )


def load_peft_adapter_without_transformers_v5_conversion(
    peft_model_cls: Any,
    model: torch.nn.Module,
    adapter_path: Path,
    *,
    is_trainable: bool,
) -> torch.nn.Module:
    import peft.utils.save_and_load as peft_save_and_load

    previous = peft_save_and_load.is_transformers_ge_v5
    peft_save_and_load.is_transformers_ge_v5 = False
    try:
        return peft_model_cls.from_pretrained(
            model,
            adapter_path,
            is_trainable=is_trainable,
        )
    finally:
        peft_save_and_load.is_transformers_ge_v5 = previous


def trainable_parameter_report(model: torch.nn.Module) -> dict[str, int | float]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        "total": int(total),
        "trainable": int(trainable),
        "trainable_fraction": float(trainable / total) if total else 0.0,
    }


class AuxClassificationHead(torch.nn.Module):
    def __init__(self, hidden_size: int, num_labels: int = len(LABELS)) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(hidden_size)
        self.classifier = torch.nn.Linear(hidden_size, num_labels)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.norm(states.float()))


def build_aux_classifier(model: torch.nn.Module, args: argparse.Namespace) -> AuxClassificationHead | None:
    if args.aux_classifier_weight <= 0 and args.eval_label_source != "aux":
        return None
    hidden_size = int(model.get_input_embeddings().weight.shape[1])
    head = AuxClassificationHead(hidden_size)
    return head.to(model_input_device(model))


def aux_parameter_report(head: torch.nn.Module | None) -> dict[str, int | float]:
    if head is None:
        return {"enabled": False, "total": 0, "trainable": 0}
    total = sum(param.numel() for param in head.parameters())
    trainable = sum(param.numel() for param in head.parameters() if param.requires_grad)
    return {"enabled": True, "total": int(total), "trainable": int(trainable)}


def encode_label_candidates(tokenizer: Any) -> list[list[int]]:
    candidates: list[list[int]] = []
    for label in LABELS:
        token_ids = tokenizer(
            label,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        if not token_ids:
            raise ValueError(f"label candidate tokenized to an empty sequence: {label}")
        candidates.append([int(token_id) for token_id in token_ids])
    return candidates


def label_candidate_scores(
    model: torch.nn.Module,
    batch: dict[str, Any],
    label_token_ids: list[list[int]],
    *,
    length_normalization: str,
) -> torch.Tensor:
    device = model_input_device(model)
    row_scores: list[torch.Tensor] = []
    for row_idx in range(batch["input_ids"].shape[0]):
        prompt_len = int(batch["prompt_indices"][row_idx].item()) + 1
        prompt_ids = batch["input_ids"][row_idx, :prompt_len].detach().tolist()
        candidate_sequences = [prompt_ids + candidate for candidate in label_token_ids]
        max_len = max(len(sequence) for sequence in candidate_sequences)
        pad_token_id = int(batch["input_ids"][row_idx, 0].item())
        input_rows = []
        attention_rows = []
        for sequence in candidate_sequences:
            pad = max_len - len(sequence)
            input_rows.append(sequence + [pad_token_id] * pad)
            attention_rows.append([1] * len(sequence) + [0] * pad)
        candidate_input_ids = torch.tensor(input_rows, dtype=torch.long, device=device)
        candidate_attention = torch.tensor(attention_rows, dtype=torch.long, device=device)
        outputs = model(
            input_ids=candidate_input_ids,
            attention_mask=candidate_attention,
            use_cache=False,
        )
        scores = []
        for candidate_idx, candidate in enumerate(label_token_ids):
            token_scores = []
            for offset, token_id in enumerate(candidate):
                pos = prompt_len - 1 + offset
                log_probs = torch.nn.functional.log_softmax(
                    outputs.logits[candidate_idx, pos].float(),
                    dim=-1,
                )
                token_scores.append(log_probs[int(token_id)])
            score = torch.stack(token_scores).sum()
            if length_normalization == "mean":
                score = score / float(len(candidate))
            scores.append(score)
        row_scores.append(torch.stack(scores))
    return torch.stack(row_scores, dim=0)


def label_distillation_loss(
    model: torch.nn.Module,
    batch: dict[str, Any],
    label_token_ids: list[list[int]],
    *,
    weight: float,
    temperature: float,
    length_normalization: str,
) -> torch.Tensor | None:
    if weight <= 0:
        return None
    teacher_mask = batch.get("teacher_mask")
    if teacher_mask is None or not bool(teacher_mask.bool().any().item()):
        return None
    scores = label_candidate_scores(
        model,
        batch,
        label_token_ids,
        length_normalization=length_normalization,
    )
    active = teacher_mask.to(device=scores.device, dtype=torch.bool)
    teacher_logits = batch["teacher_logits"].to(device=scores.device)
    temp = float(temperature)
    student_log_probs = torch.nn.functional.log_softmax(scores[active].float() / temp, dim=-1)
    teacher_probs = torch.nn.functional.softmax(teacher_logits[active].float() / temp, dim=-1)
    return torch.nn.functional.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean",
    ) * (temp * temp)


def prompt_states_from_outputs(outputs: Any, prompt_indices: torch.Tensor) -> torch.Tensor:
    hidden_states = outputs.hidden_states
    if not hidden_states:
        raise RuntimeError("auxiliary classifier requires model hidden states")
    states = hidden_states[-1]
    indexes = prompt_indices.to(states.device).clamp(min=0, max=states.shape[1] - 1)
    batch_indexes = torch.arange(states.shape[0], device=states.device)
    return states[batch_indexes, indexes]


def label_candidate_token_ids(tokenizer: Any) -> list[list[int]]:
    candidates: list[list[int]] = []
    for label in LABELS:
        token_ids = tokenizer(
            label,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        if not token_ids:
            raise ValueError(f"label {label} produced no token ids")
        candidates.append([int(token_id) for token_id in token_ids])
    return candidates


def score_label_candidates(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    label_token_ids: list[list[int]],
    pad_token_id: int,
    length_normalization: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = input_ids.device
    sequences: list[list[int]] = []
    metadata: list[tuple[int, int, int, int]] = []
    for row_idx in range(input_ids.shape[0]):
        prompt_ids = input_ids[row_idx][attention_mask[row_idx].bool()].detach().cpu().tolist()
        prompt_len = len(prompt_ids)
        for label_idx, candidate_ids in enumerate(label_token_ids):
            sequence = prompt_ids + candidate_ids
            sequences.append(sequence)
            metadata.append((row_idx, label_idx, prompt_len, len(candidate_ids)))

    max_len = max(len(sequence) for sequence in sequences)
    score_input_ids = torch.full(
        (len(sequences), max_len),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    score_attention_mask = torch.zeros_like(score_input_ids)
    for seq_idx, sequence in enumerate(sequences):
        values = torch.tensor(sequence, dtype=torch.long, device=device)
        score_input_ids[seq_idx, : values.numel()] = values
        score_attention_mask[seq_idx, : values.numel()] = 1

    outputs = model(
        input_ids=score_input_ids,
        attention_mask=score_attention_mask,
        use_cache=False,
    )
    log_probs = torch.log_softmax(outputs.logits.float(), dim=-1)
    scores = torch.full(
        (input_ids.shape[0], len(label_token_ids)),
        -torch.inf,
        dtype=torch.float32,
        device=device,
    )
    for seq_idx, (row_idx, label_idx, prompt_len, label_len) in enumerate(metadata):
        pieces = []
        for offset, token_id in enumerate(label_token_ids[label_idx]):
            logit_idx = prompt_len - 1 + offset
            pieces.append(log_probs[seq_idx, logit_idx, int(token_id)])
        label_score = torch.stack(pieces).sum()
        if length_normalization == "mean":
            label_score = label_score / max(1, label_len)
        scores[row_idx, label_idx] = label_score

    probs = torch.softmax(scores, dim=-1)
    return scores, probs


def apply_trustworthy_score_threshold(
    probs: torch.Tensor,
    *,
    threshold: float,
) -> torch.Tensor:
    preds = probs.argmax(dim=-1)
    if threshold <= 0:
        return preds
    trustworthy_id = LABEL2ID["TRUSTWORTHY"]
    for row_idx in range(probs.shape[0]):
        if int(preds[row_idx]) == trustworthy_id and float(probs[row_idx, trustworthy_id]) < threshold:
            non_trust_probs = probs[row_idx].clone()
            non_trust_probs[trustworthy_id] = -1.0
            preds[row_idx] = int(non_trust_probs.argmax().item())
    return preds


JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_object(text: str) -> dict[str, Any] | None:
    match = JSON_RE.search(text or "")
    if not match:
        return None
    raw = match.group(0)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def parse_label(text: str, fallback_label: str) -> tuple[int, bool, str]:
    upper = (text or "").upper()
    for label in LABELS:
        if label in upper:
            return LABEL2ID[label], True, label
    return LABEL2ID[fallback_label], False, fallback_label


def parse_generation(
    text: str,
    *,
    route2id: dict[str, int],
    taxonomy2id: dict[str, int],
    fallback_label: str,
) -> dict[str, Any]:
    parsed_json = extract_json_object(text)
    payload = parsed_json if parsed_json is not None else {}
    classification_value = str(payload.get("classification") or "")
    label_id, label_parsed, label = parse_label(classification_value, fallback_label)
    if not label_parsed:
        label_id, label_parsed, label = parse_label(text, fallback_label)

    route_value = str(payload.get("route") or "")
    route_id = route2id.get(route_value, -1)
    if route_id < 0:
        for route, idx in route2id.items():
            if route in text:
                route_id = idx
                route_value = route
                break

    taxonomy_value = str(payload.get("taxonomy_pattern") or "")
    taxonomy_id = taxonomy2id.get(taxonomy_value, -1)
    if taxonomy_id < 0:
        for pattern, idx in taxonomy2id.items():
            if pattern in text:
                taxonomy_id = idx
                taxonomy_value = pattern
                break

    return {
        "json_parsed": parsed_json is not None,
        "label_parsed": label_parsed,
        "classification": label,
        "classification_id": int(label_id),
        "route": route_value,
        "route_id": int(route_id),
        "taxonomy_pattern": taxonomy_value,
        "taxonomy_pattern_id": int(taxonomy_id),
    }


def selected_governance_output(
    text: str,
    parsed: dict[str, Any],
    *,
    selected_label_id: int,
    label_source: str,
) -> dict[str, Any]:
    selected_label = LABELS[int(selected_label_id)]
    payload = extract_json_object(text)
    if payload is None:
        payload = {}
    else:
        payload = dict(payload)

    parsed_classification_text = str(parsed.get("classification") or "").upper()
    classification_overridden = (
        bool((text or "").strip())
        and bool(parsed_classification_text)
        and parsed_classification_text != selected_label
    )
    payload["classification"] = selected_label

    rationale = payload.get("rationale")
    if classification_overridden or not isinstance(rationale, str) or not rationale.strip():
        payload["rationale"] = DEFAULT_RATIONALES[selected_label]

    route = payload.get("route")
    if (not isinstance(route, str) or not route.strip()) and int(parsed["route_id"]) >= 0:
        payload["route"] = str(parsed["route"])
    payload.setdefault("route", "")

    taxonomy = payload.get("taxonomy_pattern")
    if (
        (not isinstance(taxonomy, str) or not taxonomy.strip())
        and int(parsed["taxonomy_pattern_id"]) >= 0
    ):
        payload["taxonomy_pattern"] = str(parsed["taxonomy_pattern"])
    payload.setdefault("taxonomy_pattern", "")

    signals = payload.get("signals")
    if not isinstance(signals, dict):
        payload["signals"] = {}

    return {
        "json": payload,
        "text": json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
        "classification": selected_label,
        "classification_id": int(selected_label_id),
        "label_source": label_source,
        "classification_overridden": classification_overridden,
    }


def train(
    model: torch.nn.Module,
    aux_head: torch.nn.Module | None,
    loader: DataLoader,
    *,
    tokenizer: Any,
    args: argparse.Namespace,
    class_weights: tuple[float, float, float],
) -> list[dict[str, float]]:
    device = model_input_device(model)
    label_token_ids = encode_label_candidates(tokenizer)
    model_params = [param for param in model.parameters() if param.requires_grad]
    param_groups: list[dict[str, Any]] = [
        {"params": model_params, "lr": args.learning_rate},
    ]
    aux_params: list[torch.nn.Parameter] = []
    if aux_head is not None:
        aux_head.train()
        aux_params = [param for param in aux_head.parameters() if param.requires_grad]
        param_groups.append(
            {
                "params": aux_params,
                "lr": args.aux_classifier_lr
                if args.aux_classifier_lr > 0
                else args.learning_rate,
            }
        )
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=args.weight_decay,
    )
    model.train()
    history: list[dict[str, float]] = []
    step = 0
    progress = tqdm(total=args.max_steps, desc=args.run_label, leave=False)
    while step < args.max_steps:
        for raw_batch in loader:
            if step >= args.max_steps:
                break
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
                output_hidden_states=aux_head is not None,
            )
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = batch["labels"][..., 1:].contiguous()
            shift_weights = batch["loss_weights"][..., 1:].contiguous().to(shift_logits.dtype)
            token_losses = F.cross_entropy(
                shift_logits.float().view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="none",
                label_smoothing=args.label_smoothing,
            )
            weights = shift_weights.view(-1)
            valid = shift_labels.view(-1).ne(-100)
            weights = weights * valid.to(weights.dtype)
            sft_loss = (token_losses * weights).sum() / weights.sum().clamp_min(1.0)
            loss = sft_loss
            aux_loss = None
            if aux_head is not None and args.aux_classifier_weight > 0:
                prompt_states = prompt_states_from_outputs(outputs, batch["prompt_indices"])
                if args.aux_detach:
                    prompt_states = prompt_states.detach()
                aux_logits = aux_head(prompt_states)
                aux_weights = torch.tensor(
                    class_weights,
                    dtype=torch.float32,
                    device=aux_logits.device,
                )
                aux_loss = F.cross_entropy(
                    aux_logits.float(),
                    batch["label_ids"].to(aux_logits.device),
                    weight=aux_weights,
                    label_smoothing=args.aux_label_smoothing,
                )
                loss = loss + args.aux_classifier_weight * aux_loss
            distillation_loss = label_distillation_loss(
                model,
                batch,
                label_token_ids,
                weight=args.label_distillation_weight,
                temperature=args.distillation_temperature,
                length_normalization=args.label_distillation_length_normalization,
            )
            if distillation_loss is not None:
                loss = loss + args.label_distillation_weight * distillation_loss
            loss.backward()
            if args.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model_params, args.gradient_clip_norm)
                if aux_params:
                    torch.nn.utils.clip_grad_norm_(aux_params, args.gradient_clip_norm)
            optimizer.step()
            step += 1
            item = {
                "step": float(step),
                "loss": float(loss.detach().cpu()),
                "sft_loss": float(sft_loss.detach().cpu()),
            }
            if aux_loss is not None:
                item["aux_loss"] = float(aux_loss.detach().cpu())
            if distillation_loss is not None:
                item["label_distillation_loss"] = float(distillation_loss.detach().cpu())
            history.append(item)
            progress.update(1)
            progress.set_postfix(loss=f"{item['loss']:.3f}")
    progress.close()
    return history


def evaluate_generation(
    model: torch.nn.Module,
    aux_head: torch.nn.Module | None,
    loader: DataLoader,
    *,
    tokenizer: Any,
    vocab: MoEVocab,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.eval_skip_generation and args.eval_label_source == "generation":
        raise ValueError("--eval-skip-generation requires --eval-label-source != generation")

    device = model_input_device(model)
    model.eval()
    if aux_head is not None:
        aux_head.eval()
    model.config.use_cache = True
    disable_router_aux_outputs(model)

    pred_labels: list[int] = []
    generation_pred_labels: list[int] = []
    aux_pred_labels: list[int] = []
    label_score_pred_labels: list[int] = []
    gold_labels: list[int] = []
    pred_routes: list[int] = []
    gold_routes: list[int] = []
    pred_taxonomy: list[int] = []
    gold_taxonomy: list[int] = []
    predictions: list[dict[str, Any]] = []
    json_parsed = 0
    label_parsed = 0
    fallback_labels = 0
    selected_output_overrides = 0
    started = time.time()
    label_token_ids = label_candidate_token_ids(tokenizer)
    generation_skipped = bool(args.eval_skip_generation)

    with torch.inference_mode():
        desc = "score-eval" if generation_skipped else "generate-eval"
        for raw_batch in tqdm(loader, desc=desc, leave=False):
            batch = move_batch(raw_batch, device)
            aux_batch_preds: list[int] | None = None
            aux_batch_scores: list[list[float]] | None = None
            label_score_batch_preds: list[int] | None = None
            label_score_batch_scores: list[list[float]] | None = None
            label_score_batch_probs: list[list[float]] | None = None
            if aux_head is not None:
                aux_outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                    output_hidden_states=True,
                )
                prompt_states = prompt_states_from_outputs(
                    aux_outputs,
                    batch["prompt_indices"],
                )
                aux_logits = aux_head(prompt_states)
                aux_probs = torch.softmax(aux_logits.float(), dim=-1)
                aux_batch_preds = aux_probs.argmax(dim=-1).detach().cpu().tolist()
                aux_batch_scores = aux_probs.detach().cpu().tolist()
            if args.eval_label_source == "label-score":
                label_scores, label_probs = score_label_candidates(
                    model,
                    batch["input_ids"],
                    batch["attention_mask"],
                    label_token_ids=label_token_ids,
                    pad_token_id=tokenizer.pad_token_id,
                    length_normalization=args.label_score_length_normalization,
                )
                label_score_preds = apply_trustworthy_score_threshold(
                    label_probs,
                    threshold=args.label_score_trustworthy_threshold,
                )
                label_score_batch_preds = label_score_preds.detach().cpu().tolist()
                label_score_batch_scores = label_scores.detach().cpu().tolist()
                label_score_batch_probs = label_probs.detach().cpu().tolist()
            generated_texts: list[str] = [""] * int(batch["input_ids"].shape[0])
            if not generation_skipped:
                output_ids = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
                prompt_len = batch["input_ids"].shape[1]
                generated_texts = [
                    tokenizer.decode(
                        output_ids[idx, prompt_len:],
                        skip_special_tokens=True,
                    )
                    for idx in range(output_ids.shape[0])
                ]
            for idx, text in enumerate(generated_texts):
                if generation_skipped:
                    parsed = {
                        "json_parsed": False,
                        "label_parsed": False,
                        "classification": args.fallback_label,
                        "classification_id": int(LABEL2ID[args.fallback_label]),
                        "route": "",
                        "route_id": -1,
                        "taxonomy_pattern": "",
                        "taxonomy_pattern_id": -1,
                    }
                    generation_label_id = None
                    selected_label_id = int(LABEL2ID[args.fallback_label])
                    selected_label_source = "none"
                else:
                    parsed = parse_generation(
                        text,
                        route2id=vocab.route2id,
                        taxonomy2id=vocab.taxonomy_pattern2id,
                        fallback_label=args.fallback_label,
                    )
                    json_parsed += int(parsed["json_parsed"])
                    label_parsed += int(parsed["label_parsed"])
                    fallback_labels += int(not parsed["label_parsed"])
                    generation_label_id = int(parsed["classification_id"])
                    selected_label_id = generation_label_id
                    selected_label_source = "generation"
                aux_entry = None
                label_score_entry = None
                if aux_batch_preds is not None and aux_batch_scores is not None:
                    aux_label_id = int(aux_batch_preds[idx])
                    aux_entry = {
                        "classification": LABELS[aux_label_id],
                        "classification_id": aux_label_id,
                        "scores": {
                            label: float(aux_batch_scores[idx][label_idx])
                            for label_idx, label in enumerate(LABELS)
                        },
                    }
                    aux_pred_labels.append(aux_label_id)
                    if args.eval_label_source == "aux":
                        selected_label_id = aux_label_id
                        selected_label_source = "aux"
                if (
                    label_score_batch_preds is not None
                    and label_score_batch_scores is not None
                    and label_score_batch_probs is not None
                ):
                    label_score_label_id = int(label_score_batch_preds[idx])
                    label_score_entry = {
                        "classification": LABELS[label_score_label_id],
                        "classification_id": label_score_label_id,
                        "scores": {
                            label: float(label_score_batch_scores[idx][label_idx])
                            for label_idx, label in enumerate(LABELS)
                        },
                        "probabilities": {
                            label: float(label_score_batch_probs[idx][label_idx])
                            for label_idx, label in enumerate(LABELS)
                        },
                        "length_normalization": args.label_score_length_normalization,
                        "trustworthy_threshold": args.label_score_trustworthy_threshold,
                    }
                    label_score_pred_labels.append(label_score_label_id)
                    if args.eval_label_source == "label-score":
                        selected_label_id = label_score_label_id
                        selected_label_source = "label-score"
                selected_output = selected_governance_output(
                    text,
                    parsed,
                    selected_label_id=selected_label_id,
                    label_source=selected_label_source,
                )
                selected_output_overrides += int(
                    selected_output["classification_overridden"]
                )
                if generation_label_id is not None:
                    generation_pred_labels.append(generation_label_id)
                pred_labels.append(selected_label_id)
                pred_routes.append(parsed["route_id"])
                pred_taxonomy.append(parsed["taxonomy_pattern_id"])
                gold_labels.append(int(raw_batch["label_ids"][idx]))
                gold_routes.append(int(raw_batch["route_ids"][idx]))
                gold_taxonomy.append(int(raw_batch["taxonomy_ids"][idx]))
                predictions.append(
                    {
                        "id": raw_batch["ids"][idx],
                        "gold": {
                            "classification": raw_batch["labels_text"][idx],
                            "route": raw_batch["routes_text"][idx],
                            "taxonomy_pattern": raw_batch["taxonomy_text"][idx],
                        },
                        "parsed": parsed,
                        "aux": aux_entry,
                        "label_score": label_score_entry,
                        "selected_classification": LABELS[selected_label_id],
                        "selected_classification_id": selected_label_id,
                        "selected_label_source": selected_label_source,
                        "selected_output": selected_output,
                        "raw_generation": text,
                        "generation_skipped": generation_skipped,
                        "target": raw_batch["targets"][idx],
                    }
                )

    labels_arr = np.array(gold_labels, dtype=np.int64)
    preds_arr = np.array(pred_labels, dtype=np.int64)
    route_arr = np.array(gold_routes, dtype=np.int64)
    pred_route_arr = np.array(pred_routes, dtype=np.int64)
    taxonomy_arr = np.array(gold_taxonomy, dtype=np.int64)
    pred_taxonomy_arr = np.array(pred_taxonomy, dtype=np.int64)
    elapsed = time.time() - started
    n = len(gold_labels)
    parsed_mask = np.array(
        [bool(row["parsed"]["label_parsed"]) for row in predictions],
        dtype=bool,
    )
    parsed_label_accuracy = None
    if parsed_mask.any():
        parsed_label_accuracy = float((preds_arr[parsed_mask] == labels_arr[parsed_mask]).mean())
    report = {
        "rows": n,
        "classification": compute_classification_metrics(preds_arr, labels_arr),
        "classification_label_source": args.eval_label_source,
        "classification_scored_with_fallback": args.eval_label_source == "generation",
        "generation_skipped": generation_skipped,
        "generation_classification": None,
        "parsed_label_accuracy": parsed_label_accuracy,
        "route_accuracy": None
        if generation_skipped
        else float((pred_route_arr == route_arr).mean()) if n else 0.0,
        "taxonomy_accuracy": None
        if generation_skipped
        else float((pred_taxonomy_arr == taxonomy_arr).mean()) if n else 0.0,
        "json_parse_rate": None if generation_skipped else float(json_parsed / n) if n else 0.0,
        "label_parse_rate": None if generation_skipped else float(label_parsed / n) if n else 0.0,
        "fallback_label_count": int(fallback_labels),
        "fallback_label": args.fallback_label,
        "selected_output_classification_overrides": int(selected_output_overrides),
        "selected_output_classification_override_rate": (
            float(selected_output_overrides / n) if n else 0.0
        ),
        "invalid_route_rate": None
        if generation_skipped
        else float((pred_route_arr < 0).mean()) if n else 0.0,
        "invalid_taxonomy_rate": None
        if generation_skipped
        else float((pred_taxonomy_arr < 0).mean()) if n else 0.0,
        "elapsed_seconds": round(elapsed, 3),
        "rows_per_second": float(n / elapsed) if elapsed > 0 else 0.0,
        "predictions": predictions,
    }
    if generation_pred_labels:
        generation_preds_arr = np.array(generation_pred_labels, dtype=np.int64)
        report["generation_classification"] = compute_classification_metrics(
            generation_preds_arr,
            labels_arr,
        )
    if aux_pred_labels:
        aux_preds_arr = np.array(aux_pred_labels, dtype=np.int64)
        report["aux_classification"] = compute_classification_metrics(aux_preds_arr, labels_arr)
    if label_score_pred_labels:
        label_score_preds_arr = np.array(label_score_pred_labels, dtype=np.int64)
        report["label_score_classification"] = compute_classification_metrics(
            label_score_preds_arr,
            labels_arr,
        )
        report["label_score_length_normalization"] = args.label_score_length_normalization
        report["label_score_trustworthy_threshold"] = args.label_score_trustworthy_threshold
    return report


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    start = time.time()
    set_all_seeds(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    class_weights = parse_class_weights(args.class_weights)

    vocab = MoEVocab.from_metadata(args.data_dir / "metadata.json")
    train_rows = read_rows(
        args.data_dir / f"{args.train_split}.jsonl",
        limit=args.max_train_samples,
        sample_mode=args.sample_mode,
        sample_seed=args.seed,
    )
    if args.eval_split == args.train_split:
        eval_rows = train_rows[: min(args.max_eval_samples, len(train_rows))]
    else:
        eval_rows = read_rows(
            args.data_dir / f"{args.eval_split}.jsonl",
            limit=args.max_eval_samples,
            sample_mode=args.sample_mode,
            sample_seed=args.seed + 1,
        )
    teacher_required = bool(args.label_distillation_weight > 0 and not args.eval_only)
    train_teacher_count = attach_teacher_logits_to_rows(
        train_rows,
        teacher_sidecar_path(
            args.teacher_logits_dir,
            args.train_split,
            required=teacher_required,
        ),
        required=teacher_required,
    )
    eval_teacher_count = attach_teacher_logits_to_rows(
        eval_rows,
        teacher_sidecar_path(
            args.teacher_logits_dir,
            args.eval_split,
            required=False,
        ),
        required=False,
    )

    model, tokenizer = load_model_and_tokenizer(args)
    aux_head = build_aux_classifier(model, args)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    collator = CausalCollator(pad_token_id=pad_token_id)
    train_ds = GenerativeMoEDataset(
        train_rows,
        tokenizer=tokenizer,
        max_length=args.max_length,
        signal_fields=DEFAULT_SIGNAL_FIELDS,
        include_target=True,
        short_rationale_only=args.short_rationale,
        class_weights=class_weights,
        classification_loss_weight=args.classification_loss_weight,
        target_mode=args.target_mode,
    )
    eval_ds = GenerativeMoEDataset(
        eval_rows,
        tokenizer=tokenizer,
        max_length=args.max_length,
        signal_fields=DEFAULT_SIGNAL_FIELDS,
        include_target=False,
        short_rationale_only=args.short_rationale,
        class_weights=class_weights,
        classification_loss_weight=args.classification_loss_weight,
        target_mode=args.target_mode,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    params = trainable_parameter_report(model)
    aux_params = aux_parameter_report(aux_head)
    print(f"Seed pack       : {args.seed_pack}")
    print(f"Output dir      : {args.output_dir}")
    print(f"Rows            : train={len(train_ds)} eval={len(eval_ds)}")
    print(f"Splits          : train={args.train_split} eval={args.eval_split}")
    if args.teacher_logits_dir is not None:
        print(
            "Teacher logits  : "
            f"train={train_teacher_count}/{len(train_ds)} "
            f"eval={eval_teacher_count}/{len(eval_ds)} "
            f"weight={args.label_distillation_weight:g} "
            f"temp={args.distillation_temperature:g}"
        )
    print(f"Max length      : {args.max_length}")
    print(f"Max new tokens  : {args.max_new_tokens}")
    print(f"Params          : total={params['total']:,} trainable={params['trainable']:,}")
    if aux_params["enabled"]:
        print(
            "Aux classifier  : "
            f"trainable={int(aux_params['trainable']):,} "
            f"weight={args.aux_classifier_weight:g} "
            f"eval_source={args.eval_label_source}"
        )
    print(f"LoRA targets    : {args.lora_target_modules if args.lora_r > 0 else 'disabled'}")
    extra_unfrozen = getattr(
        model,
        "_pyrrho_extra_unfrozen",
        {"patterns": [], "parameters": 0, "tensors": 0},
    )
    if extra_unfrozen["patterns"]:
        print(
            "Extra unfreeze  : "
            f"patterns={extra_unfrozen['patterns']} "
            f"tensors={extra_unfrozen['tensors']:,} "
            f"params={extra_unfrozen['parameters']:,}"
        )
    olmoe_adapters = getattr(
        model,
        "_pyrrho_olmoe_adapters",
        {
            "expert_down_lora": {"enabled": False, "rank": 0, "layers": 0, "parameters": 0},
            "expert_gate_up_lora": {"enabled": False, "rank": 0, "layers": 0, "parameters": 0},
            "router_lora": {"enabled": False, "rank": 0, "layers": 0, "parameters": 0},
        },
    )
    if (
        olmoe_adapters["expert_down_lora"]["enabled"]
        or olmoe_adapters["expert_gate_up_lora"]["enabled"]
        or olmoe_adapters["router_lora"]["enabled"]
    ):
        print(f"OLMoE adapters  : {olmoe_adapters}")

    train_history: list[dict[str, float]] = []
    if not args.eval_only and args.max_steps > 0:
        train_history = train(
            model,
            aux_head,
            train_loader,
            tokenizer=tokenizer,
            args=args,
            class_weights=class_weights,
        )

    eval_report = evaluate_generation(
        model,
        aux_head,
        eval_loader,
        tokenizer=tokenizer,
        vocab=vocab,
        args=args,
    )

    adapter_path = None
    if args.save_adapter and hasattr(model, "save_pretrained"):
        adapter_dir = args.output_dir / "final_adapter"
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        adapter_path = str(adapter_dir)
        if aux_head is not None:
            torch.save(aux_head.state_dict(), adapter_dir / "aux_classifier.pt")

    predictions = eval_report.pop("predictions")
    write_jsonl(args.output_dir / "eval_generations.jsonl", predictions)
    report = {
        "status": "complete",
        "seed_pack": str(args.seed_pack),
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "run_label": args.run_label,
        "seed": args.seed,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "eval_skip_generation": bool(args.eval_skip_generation),
        "max_steps": args.max_steps,
        "train_rows": len(train_ds),
        "eval_rows": len(eval_ds),
        "teacher_logits": {
            "dir": str(args.teacher_logits_dir) if args.teacher_logits_dir else None,
            "train_attached": int(train_teacher_count),
            "eval_attached": int(eval_teacher_count),
            "label_distillation_weight": args.label_distillation_weight,
            "distillation_temperature": args.distillation_temperature,
            "label_distillation_length_normalization": args.label_distillation_length_normalization,
        },
        "dtype": args.dtype,
        "device_map": args.device_map,
        "lora": {
            "enabled": args.lora_r > 0,
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": [
                target.strip()
                for target in args.lora_target_modules.split(",")
                if target.strip()
            ],
            "adapter_path": adapter_path,
            "loaded_adapter_path": str(args.adapter_path) if args.adapter_path else None,
        },
        "extra_unfrozen": extra_unfrozen,
        "olmoe_adapters": olmoe_adapters,
        "aux_classifier": {
            **aux_params,
            "loss_weight": args.aux_classifier_weight,
            "learning_rate": args.aux_classifier_lr
            if args.aux_classifier_lr > 0
            else args.learning_rate,
            "label_smoothing": args.aux_label_smoothing,
            "detach": bool(args.aux_detach),
            "eval_label_source": args.eval_label_source,
        },
        "short_rationale": bool(args.short_rationale),
        "target_mode": args.target_mode,
        "loss": {
            "class_weights": {
                label: class_weights[idx]
                for idx, label in enumerate(LABELS)
            },
            "classification_loss_weight": args.classification_loss_weight,
            "label_smoothing": args.label_smoothing,
        },
        "parameters": params,
        "train_history": train_history,
        "last_train_loss": train_history[-1]["loss"] if train_history else None,
        "eval": eval_report,
        "elapsed_seconds": round(time.time() - start, 3),
    }
    (args.output_dir / "train_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    classification = eval_report["classification"]
    json_rate = eval_report["json_parse_rate"]
    label_rate = eval_report["label_parse_rate"]
    route_accuracy = eval_report["route_accuracy"]
    taxonomy_accuracy = eval_report["taxonomy_accuracy"]
    eval_summary = (
        "Eval generation : "
        + (f"json={json_rate:.3f} " if json_rate is not None else "json=n/a ")
        + (f"label={label_rate:.3f} " if label_rate is not None else "label=n/a ")
        + f"source={eval_report['classification_label_source']} "
        + f"acc={classification['accuracy']:.3f} "
        + f"ft={classification['false_trustworthy_rate']:.3f} "
        + (f"route={route_accuracy:.3f} " if route_accuracy is not None else "route=n/a ")
        + (
            f"taxonomy={taxonomy_accuracy:.3f}"
            if taxonomy_accuracy is not None
            else "taxonomy=n/a"
        )
    )
    print(
        eval_summary
    )
    print(f"Generations     : {args.output_dir / 'eval_generations.jsonl'}")
    print(f"Report          : {args.output_dir / 'train_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
