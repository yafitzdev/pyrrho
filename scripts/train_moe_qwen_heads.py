"""Stage 1 training smoke for pyrrho heads on the Qwen3-MoE seed pack.

This trainer intentionally starts conservative: by default it freezes the
Qwen3-MoE trunk and trains only the pyrrho governance/route/taxonomy/scalar
heads. Use `--train-internal-routers` only for a later router-specific run.

Run from project root:
    python scripts/train_moe_qwen_heads.py --max-steps 1 --max-train-samples 2 --max-eval-samples 2
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from pyrrho.moe.data import MoEVocab
from pyrrho.moe.losses import MoELossWeights, multitask_loss
from pyrrho.moe.metrics import moe_eval_metrics
from pyrrho.moe.qwen_governance import (
    QwenMoEForGovernance,
    QwenMoEGovernanceConfig,
    add_semantic_route_adapter,
    add_sparse_expert_adapters,
    set_final_dense_layer_trainability,
)
from pyrrho.training import set_all_seeds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed-pack", type=Path, default=Path("outputs/moe/upcycling/qwen_alpha_seed_pack"))
    p.add_argument("--data-dir", type=Path, default=Path("data/moe_v8"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/moe/qwen_heads_stage1_smoke"))
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=4)
    p.add_argument("--max-train-samples", type=int, default=16)
    p.add_argument("--max-eval-samples", type=int, default=16)
    p.add_argument(
        "--sample-mode",
        choices=["random", "prefix"],
        default="random",
        help="How to choose bounded train/eval subsets",
    )
    p.add_argument("--learning-rate", type=float, default=1.0e-4)
    p.add_argument(
        "--router-learning-rate",
        type=float,
        default=None,
        help="Optional separate LR for trainable Qwen internal router gates",
    )
    p.add_argument(
        "--trunk-learning-rate",
        type=float,
        default=None,
        help="Optional separate LR for partially unfrozen non-router Qwen trunk params",
    )
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--loss-governance", type=float, default=1.0)
    p.add_argument("--loss-route", type=float, default=0.7)
    p.add_argument("--loss-taxonomy", type=float, default=0.3)
    p.add_argument("--loss-scalar", type=float, default=0.2)
    p.add_argument("--loss-distillation", type=float, default=0.0)
    p.add_argument("--distillation-temperature", type=float, default=2.0)
    p.add_argument("--loss-load-balance", type=float, default=0.02)
    p.add_argument("--false-trustworthy-weight", type=float, default=2.3)
    p.add_argument("--calibration-grid-size", type=int, default=66)
    p.add_argument("--heads-path", type=Path, default=None)
    p.add_argument(
        "--teacher-logits-dir",
        type=Path,
        default=None,
        help="Optional sidecar dir with train.jsonl/eval.jsonl teacher governance logits keyed by id",
    )
    p.add_argument("--lora-adapter-path", type=Path, default=None)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument(
        "--eval-compare-gold-routes",
        action="store_true",
        help="Also report governance metrics when semantic adapters are forced to use gold route IDs at eval time",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32", "auto"], default="bfloat16")
    p.add_argument("--train-internal-routers", action="store_true")
    p.add_argument(
        "--train-final-dense-layers",
        type=int,
        default=0,
        help="Unfreeze this many final dense Qwen decoder layers in addition to pyrrho heads",
    )
    p.add_argument("--lora-r", type=int, default=0)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated Linear module suffixes for PEFT LoRA; empty disables target parsing",
    )
    p.add_argument(
        "--expert-adapter-r",
        type=int,
        default=0,
        help="Enable trainable per-physical-expert residual adapters with this rank",
    )
    p.add_argument("--expert-adapter-alpha", type=float, default=16.0)
    p.add_argument("--expert-adapter-dropout", type=float, default=0.05)
    p.add_argument(
        "--expert-adapter-layers",
        type=int,
        default=4,
        help="Number of final sparse MoE layers to adapt; use 0 for all sparse layers",
    )
    p.add_argument(
        "--semantic-adapter-r",
        type=int,
        default=0,
        help="Enable trainable route-supervised pooled-state sparse adapters with this rank",
    )
    p.add_argument("--semantic-adapter-alpha", type=float, default=16.0)
    p.add_argument("--semantic-adapter-dropout", type=float, default=0.05)
    return p.parse_args()


def choose_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


class QwenMoEJsonlDataset(Dataset):
    def __init__(
        self,
        path: Path,
        *,
        teacher_logits_path: Path | None = None,
        limit: int | None = None,
        sample_mode: str = "random",
        sample_seed: int = 42,
    ) -> None:
        self.rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                self.rows.append(json.loads(line))
        if teacher_logits_path is not None:
            teacher_logits = load_teacher_logits(teacher_logits_path)
            for row in self.rows:
                logits = teacher_logits.get(str(row["id"]))
                if logits is not None:
                    row["teacher_logits"] = logits
        if not self.rows:
            raise ValueError(f"no rows loaded from {path}")
        if limit is not None and limit < len(self.rows):
            if sample_mode == "prefix":
                self.rows = self.rows[:limit]
            elif sample_mode == "random":
                rng = random.Random(sample_seed)
                indices = list(range(len(self.rows)))
                rng.shuffle(indices)
                selected = sorted(indices[:limit])
                self.rows = [self.rows[i] for i in selected]
            else:
                raise ValueError(f"unsupported sample_mode: {sample_mode}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]

    def summary(self) -> dict[str, dict[str, int]]:
        return {
            "labels": dict(Counter(str(row["label"]) for row in self.rows)),
            "routes": dict(Counter(str(row["route"]) for row in self.rows)),
            "taxonomy_patterns": dict(
                Counter(str(row["taxonomy_pattern"]) for row in self.rows)
            ),
        }


def load_teacher_logits(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        raise FileNotFoundError(f"teacher logits sidecar not found: {path}")
    out: dict[str, list[float]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            if not raw.strip():
                continue
            row = json.loads(raw)
            logits = row.get("logits") or row.get("teacher_logits")
            if not isinstance(logits, list) or len(logits) != 3:
                raise ValueError(f"invalid teacher logits row for id={row.get('id')!r}")
            out[str(row["id"])] = [float(v) for v in logits]
    return out


class QwenBatchCollator:
    def __init__(self, tokenizer: Any, vocab: MoEVocab, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_length = max_length

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = self.tokenizer(
            [row["text"] for row in rows],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        scalar_values = []
        scalar_masks = []
        teacher_values = []
        teacher_masks = []
        for row in rows:
            targets = row.get("scalar_targets") or {}
            values = []
            masks = []
            for field in self.vocab.scalar_fields:
                value = targets.get(field)
                if isinstance(value, int | float):
                    values.append(float(value))
                    masks.append(1.0)
                else:
                    values.append(0.0)
                    masks.append(0.0)
            scalar_values.append(values)
            scalar_masks.append(masks)
            teacher_logits = row.get("teacher_logits")
            if isinstance(teacher_logits, list) and len(teacher_logits) == 3:
                teacher_values.append([float(v) for v in teacher_logits])
                teacher_masks.append(1.0)
            else:
                teacher_values.append([0.0, 0.0, 0.0])
                teacher_masks.append(0.0)

        return {
            "ids": [row["id"] for row in rows],
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": torch.tensor([int(row["label_id"]) for row in rows], dtype=torch.long),
            "route_ids": torch.tensor([int(row["route_id"]) for row in rows], dtype=torch.long),
            "taxonomy_ids": torch.tensor(
                [int(row["taxonomy_pattern_id"]) for row in rows],
                dtype=torch.long,
            ),
            "scalar_targets": torch.tensor(scalar_values, dtype=torch.float32),
            "scalar_mask": torch.tensor(scalar_masks, dtype=torch.float32),
            "teacher_logits": torch.tensor(teacher_values, dtype=torch.float32),
            "teacher_mask": torch.tensor(teacher_masks, dtype=torch.float32),
        }


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def set_internal_router_trainability(model: QwenMoEForGovernance, trainable: bool) -> int:
    count = 0
    for name, param in model.trunk.named_parameters():
        if name.endswith(".mlp.gate.weight"):
            param.requires_grad_(trainable)
            count += param.numel()
    return count


def apply_lora_to_trunk(
    model: QwenMoEForGovernance,
    *,
    r: int,
    alpha: int,
    dropout: float,
    target_modules: list[str],
    adapter_path: Path | None = None,
    trainable: bool = True,
) -> int:
    if r <= 0 and adapter_path is None:
        return 0
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    if adapter_path is not None:
        model.trunk = PeftModel.from_pretrained(model.trunk, adapter_path, is_trainable=trainable)
    else:
        config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            target_modules=target_modules,
        )
        model.trunk = get_peft_model(model.trunk, config)
    return sum(param.numel() for param in model.trunk.parameters() if param.requires_grad)


def trainable_parameter_count(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def optimizer_param_groups(
    model: torch.nn.Module,
    *,
    head_lr: float,
    router_lr: float | None,
    trunk_lr: float | None,
) -> list[dict[str, Any]]:
    head_params = []
    router_params = []
    trunk_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("trunk."):
            if name.endswith(".mlp.gate.weight"):
                router_params.append(param)
            else:
                trunk_params.append(param)
        else:
            head_params.append(param)

    groups: list[dict[str, Any]] = []
    router_group_lr = router_lr if router_lr is not None else head_lr
    trunk_group_lr = trunk_lr if trunk_lr is not None else router_group_lr
    if head_params:
        groups.append({"params": head_params, "lr": head_lr})
    if router_params:
        groups.append({"params": router_params, "lr": router_group_lr})
    if trunk_params:
        groups.append(
            {
                "params": trunk_params,
                "lr": trunk_group_lr,
            }
        )
    return groups


def load_heads(model: QwenMoEForGovernance, heads_path: Path) -> None:
    checkpoint = torch.load(heads_path, map_location="cpu", weights_only=False)
    trainable_state = checkpoint.get("trainable_state_dict")
    if trainable_state is not None:
        model.load_state_dict(trainable_state, strict=False)
        return
    model.route_head.load_state_dict(checkpoint["route_head"])
    model.governance_head.load_state_dict(checkpoint["governance_head"])
    model.taxonomy_head.load_state_dict(checkpoint["taxonomy_head"])
    model.scalar_head.load_state_dict(checkpoint["scalar_head"])


def trainable_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def hydrate_adapter_args_from_checkpoint(args: argparse.Namespace) -> None:
    if args.heads_path is None or not args.heads_path.exists():
        return
    checkpoint = torch.load(args.heads_path, map_location="cpu", weights_only=False)
    expert_adapter = checkpoint.get("expert_adapter") or {}
    if expert_adapter.get("enabled") and args.expert_adapter_r <= 0:
        args.expert_adapter_r = int(expert_adapter["rank"])
        args.expert_adapter_alpha = float(expert_adapter["alpha"])
        args.expert_adapter_dropout = float(expert_adapter["dropout"])
        args.expert_adapter_layers = int(expert_adapter["num_layers_arg"])
    semantic_adapter = checkpoint.get("semantic_adapter") or {}
    if semantic_adapter.get("enabled") and args.semantic_adapter_r <= 0:
        args.semantic_adapter_r = int(semantic_adapter["rank"])
        args.semantic_adapter_alpha = float(semantic_adapter["alpha"])
        args.semantic_adapter_dropout = float(semantic_adapter["dropout"])


def evaluate(
    model: QwenMoEForGovernance,
    loader: DataLoader,
    device: torch.device,
    weights: MoELossWeights,
    calibration_grid_size: int,
    distillation_temperature: float,
    force_route_ids: bool = False,
) -> dict[str, Any]:
    model.eval()
    losses = []
    gov_logits = []
    route_logits = []
    taxonomy_logits = []
    labels = []
    route_labels = []
    taxonomy_labels = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                route_ids=batch["route_ids"],
                force_route_ids=force_route_ids,
            )
            _, parts = multitask_loss(
                outputs,
                labels=batch["labels"],
                route_ids=batch["route_ids"],
                taxonomy_ids=batch["taxonomy_ids"],
                scalar_targets=batch["scalar_targets"],
                scalar_mask=batch["scalar_mask"],
                weights=weights,
                teacher_logits=batch.get("teacher_logits"),
                teacher_mask=batch.get("teacher_mask"),
                distillation_temperature=distillation_temperature,
            )
            losses.append(parts["loss"])
            gov_logits.append(outputs["governance_logits"].detach().cpu().numpy())
            route_logits.append(outputs["route_logits"].detach().cpu().numpy())
            taxonomy_logits.append(outputs["taxonomy_logits"].detach().cpu().numpy())
            labels.append(batch["labels"].detach().cpu().numpy())
            route_labels.append(batch["route_ids"].detach().cpu().numpy())
            taxonomy_labels.append(batch["taxonomy_ids"].detach().cpu().numpy())

    metrics = moe_eval_metrics(
        governance_logits=np.concatenate(gov_logits, axis=0),
        labels=np.concatenate(labels, axis=0),
        route_logits=np.concatenate(route_logits, axis=0),
        route_labels=np.concatenate(route_labels, axis=0),
        taxonomy_logits=np.concatenate(taxonomy_logits, axis=0),
        taxonomy_labels=np.concatenate(taxonomy_labels, axis=0),
        calibration_grid_size=calibration_grid_size,
    )
    metrics["loss"] = float(mean(losses)) if losses else 0.0
    metrics["force_route_ids"] = bool(force_route_ids)
    return metrics


def main() -> int:
    args = parse_args()
    start = time.time()
    set_all_seeds(args.seed)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hydrate_adapter_args_from_checkpoint(args)

    vocab = MoEVocab.from_metadata(args.data_dir / "metadata.json")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.seed_pack, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    collator = QwenBatchCollator(tokenizer, vocab, args.max_length)
    train_ds = QwenMoEJsonlDataset(
        args.data_dir / "train.jsonl",
        teacher_logits_path=(args.teacher_logits_dir / "train.jsonl")
        if args.teacher_logits_dir is not None
        else None,
        limit=args.max_train_samples,
        sample_mode=args.sample_mode,
        sample_seed=args.seed,
    )
    eval_ds = QwenMoEJsonlDataset(
        args.data_dir / "eval.jsonl",
        teacher_logits_path=(args.teacher_logits_dir / "eval.jsonl")
        if args.teacher_logits_dir is not None
        else None,
        limit=args.max_eval_samples,
        sample_mode=args.sample_mode,
        sample_seed=args.seed + 1,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    eval_loader = DataLoader(eval_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collator)

    model_cfg = QwenMoEGovernanceConfig(
        num_routes=len(vocab.route2id),
        num_taxonomy_patterns=len(vocab.taxonomy_pattern2id),
        num_scalar_targets=len(vocab.scalar_fields),
        freeze_trunk=True,
    )
    model = QwenMoEForGovernance.from_seed_pack(
        args.seed_pack,
        model_cfg,
        dtype=args.dtype,
        local_files_only=True,
    )
    expert_adapter_params, expert_adapter_layer_indices = add_sparse_expert_adapters(
        model,
        rank=args.expert_adapter_r,
        alpha=args.expert_adapter_alpha,
        dropout=args.expert_adapter_dropout,
        num_layers=args.expert_adapter_layers,
    )
    semantic_adapter_params = add_semantic_route_adapter(
        model,
        rank=args.semantic_adapter_r,
        alpha=args.semantic_adapter_alpha,
        dropout=args.semantic_adapter_dropout,
    )
    lora_target_modules = [
        target.strip() for target in args.lora_target_modules.split(",") if target.strip()
    ]
    lora_params = apply_lora_to_trunk(
        model,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=lora_target_modules,
        adapter_path=args.lora_adapter_path,
        trainable=not args.eval_only,
    )
    model.to(device)
    router_params = set_internal_router_trainability(model, args.train_internal_routers)
    final_dense_layer_params, final_dense_layer_indices = set_final_dense_layer_trainability(
        model,
        args.train_final_dense_layers,
        True,
    )
    if args.heads_path is not None:
        load_heads(model, args.heads_path)
    model.train()

    trainable = trainable_parameter_count(model)
    weights = MoELossWeights(
        governance=args.loss_governance,
        route=args.loss_route,
        taxonomy=args.loss_taxonomy,
        scalar=args.loss_scalar,
        distillation=args.loss_distillation,
        load_balance=args.loss_load_balance,
        false_trustworthy_weight=args.false_trustworthy_weight,
    )
    loss_history = []

    print(f"Seed pack       : {args.seed_pack}")
    print(f"Output dir      : {args.output_dir}")
    print(f"Device / dtype  : {device} / {args.dtype}")
    print(f"Rows            : train={len(train_ds)} eval={len(eval_ds)}")
    print(f"Sample mode     : {args.sample_mode}")
    print(f"Train labels    : {train_ds.summary()['labels']}")
    print(f"Eval labels     : {eval_ds.summary()['labels']}")
    print(f"Trainable params: {trainable:,}")
    print(
        "Expert adapters : "
        f"{expert_adapter_layer_indices if expert_adapter_layer_indices else 'disabled'} "
        f"({expert_adapter_params:,})"
    )
    print(
        "Semantic adapter: "
        f"{'enabled' if semantic_adapter_params else 'disabled'} "
        f"({semantic_adapter_params:,})"
    )
    print(f"LoRA params     : {lora_params:,}")
    print(f"Internal routers: {'trainable' if args.train_internal_routers else 'frozen'} ({router_params:,})")
    print(
        "Final dense Lyrs: "
        f"{final_dense_layer_indices if final_dense_layer_indices else 'frozen'} "
        f"({final_dense_layer_params:,})"
    )
    if args.heads_path is not None:
        print(f"Loaded heads    : {args.heads_path}")

    step = 0
    if not args.eval_only:
        optimizer = torch.optim.AdamW(
            optimizer_param_groups(
                model,
                head_lr=args.learning_rate,
                router_lr=args.router_learning_rate,
                trunk_lr=args.trunk_learning_rate,
            ),
            weight_decay=args.weight_decay,
        )
        progress = tqdm(total=args.max_steps, desc="stage1-heads", leave=False)
        while step < args.max_steps:
            for raw_batch in train_loader:
                if step >= args.max_steps:
                    break
                batch = move_batch(raw_batch, device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    route_ids=batch["route_ids"],
                )
                loss, parts = multitask_loss(
                    outputs,
                    labels=batch["labels"],
                    route_ids=batch["route_ids"],
                    taxonomy_ids=batch["taxonomy_ids"],
                    scalar_targets=batch["scalar_targets"],
                    scalar_mask=batch["scalar_mask"],
                    weights=weights,
                    teacher_logits=batch.get("teacher_logits"),
                    teacher_mask=batch.get("teacher_mask"),
                    distillation_temperature=args.distillation_temperature,
                )
                loss.backward()
                optimizer.step()
                loss_history.append(parts)
                step += 1
                progress.update(1)
                progress.set_postfix(loss=f"{parts['loss']:.3f}")
        progress.close()

    eval_metrics = evaluate(
        model,
        eval_loader,
        device,
        weights,
        args.calibration_grid_size,
        args.distillation_temperature,
    )
    eval_metrics_gold_routes = None
    if args.eval_compare_gold_routes:
        eval_metrics_gold_routes = evaluate(
            model,
            eval_loader,
            device,
            weights,
            args.calibration_grid_size,
            args.distillation_temperature,
            force_route_ids=True,
        )
    heads_path = args.heads_path if args.eval_only and args.heads_path is not None else args.output_dir / "heads.pt"
    lora_save_path = args.output_dir / "lora_adapter"
    if not args.eval_only:
        torch.save(
            {
                "route_head": model.route_head.state_dict(),
                "governance_head": model.governance_head.state_dict(),
                "taxonomy_head": model.taxonomy_head.state_dict(),
                "scalar_head": model.scalar_head.state_dict(),
                "trainable_state_dict": trainable_state_dict(model),
                "config": model_cfg,
                "expert_adapter": {
                    "enabled": expert_adapter_params > 0,
                    "rank": args.expert_adapter_r,
                    "alpha": args.expert_adapter_alpha,
                    "dropout": args.expert_adapter_dropout,
                    "num_layers_arg": args.expert_adapter_layers,
                    "layer_indices": expert_adapter_layer_indices,
                    "trainable_params": expert_adapter_params,
                },
                "semantic_adapter": {
                    "enabled": semantic_adapter_params > 0,
                    "rank": args.semantic_adapter_r,
                    "alpha": args.semantic_adapter_alpha,
                    "dropout": args.semantic_adapter_dropout,
                    "trainable_params": semantic_adapter_params,
                },
            },
            heads_path,
        )
        if lora_params > 0 and hasattr(model.trunk, "save_pretrained"):
            model.trunk.save_pretrained(lora_save_path)
    report = {
        "status": "complete",
        "seed_pack": str(args.seed_pack),
        "data_dir": str(args.data_dir),
        "device": str(device),
        "dtype": args.dtype,
        "max_length": args.max_length,
        "max_steps": args.max_steps,
        "eval_only": args.eval_only,
        "sample_mode": args.sample_mode,
        "train_rows": len(train_ds),
        "eval_rows": len(eval_ds),
        "train_summary": train_ds.summary(),
        "eval_summary": eval_ds.summary(),
        "teacher_logits_dir": str(args.teacher_logits_dir)
        if args.teacher_logits_dir is not None
        else None,
        "trainable_params": trainable,
        "expert_adapter": {
            "enabled": expert_adapter_params > 0,
            "rank": args.expert_adapter_r,
            "alpha": args.expert_adapter_alpha,
            "dropout": args.expert_adapter_dropout,
            "num_layers_arg": args.expert_adapter_layers,
            "layer_indices": expert_adapter_layer_indices,
            "trainable_params": expert_adapter_params,
        },
        "semantic_adapter": {
            "enabled": semantic_adapter_params > 0,
            "rank": args.semantic_adapter_r,
            "alpha": args.semantic_adapter_alpha,
            "dropout": args.semantic_adapter_dropout,
            "trainable_params": semantic_adapter_params,
        },
        "internal_router_params": router_params,
        "train_internal_routers": args.train_internal_routers,
        "train_final_dense_layers": args.train_final_dense_layers,
        "trainable_final_dense_layer_indices": final_dense_layer_indices,
        "final_dense_layer_params": final_dense_layer_params,
        "lora": {
            "enabled": lora_params > 0,
            "trainable_params": lora_params,
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": lora_target_modules,
            "adapter_path": str(args.lora_adapter_path)
            if args.lora_adapter_path is not None
            else None,
            "saved_adapter_path": str(lora_save_path)
            if not args.eval_only and lora_params > 0
            else None,
        },
        "loss_weights": {
            "governance": args.loss_governance,
            "route": args.loss_route,
            "taxonomy": args.loss_taxonomy,
            "scalar": args.loss_scalar,
            "distillation": args.loss_distillation,
            "distillation_temperature": args.distillation_temperature,
            "load_balance": args.loss_load_balance,
            "false_trustworthy_weight": args.false_trustworthy_weight,
        },
        "learning_rates": {
            "heads": args.learning_rate,
            "routers": args.router_learning_rate
            if args.router_learning_rate is not None
            else args.learning_rate,
            "trunk": args.trunk_learning_rate
            if args.trunk_learning_rate is not None
            else args.router_learning_rate
            if args.router_learning_rate is not None
            else args.learning_rate,
        },
        "last_train_loss": loss_history[-1] if loss_history else None,
        "eval_metrics": eval_metrics,
        "eval_metrics_gold_routes": eval_metrics_gold_routes,
        "heads_path": str(heads_path),
        "loaded_heads_path": str(args.heads_path) if args.heads_path is not None else None,
        "elapsed_seconds": round(time.time() - start, 3),
    }
    report_name = "eval_report.json" if args.eval_only else "train_report.json"
    (args.output_dir / report_name).write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    if report["last_train_loss"] is not None:
        print(f"Last train loss : {report['last_train_loss']['loss']:.4f}")
    governance = eval_metrics["governance"]
    calibrated = eval_metrics["governance_calibrated"]
    print(
        "Eval            : "
        f"acc={governance['accuracy']:.4f} "
        f"ft={governance['false_trustworthy_rate']:.4f} "
        f"route={eval_metrics['route_accuracy']:.4f}"
    )
    print(
        "Calibrated      : "
        f"acc={calibrated['accuracy']:.4f} "
        f"ft={calibrated['false_trustworthy_rate']:.4f} "
        f"tau={calibrated['threshold']:.4f}"
    )
    if eval_metrics_gold_routes is not None:
        gold_governance = eval_metrics_gold_routes["governance"]
        gold_calibrated = eval_metrics_gold_routes["governance_calibrated"]
        print(
            "Eval gold routes: "
            f"acc={gold_governance['accuracy']:.4f} "
            f"ft={gold_governance['false_trustworthy_rate']:.4f} "
            f"route={eval_metrics_gold_routes['route_accuracy']:.4f}"
        )
        print(
            "Cal gold routes : "
            f"acc={gold_calibrated['accuracy']:.4f} "
            f"ft={gold_calibrated['false_trustworthy_rate']:.4f} "
            f"tau={gold_calibrated['threshold']:.4f}"
        )
    print(f"Heads           : {heads_path}")
    print(f"Report          : {args.output_dir / report_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
