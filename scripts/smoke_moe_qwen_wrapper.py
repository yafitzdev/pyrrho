"""Smoke-test the Qwen3-MoE seed pack with pyrrho governance heads.

Run from project root:
    python scripts/smoke_moe_qwen_wrapper.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import torch

from pyrrho.moe.data import MoEVocab
from pyrrho.moe.losses import MoELossWeights, multitask_loss
from pyrrho.moe.qwen_governance import QwenMoEForGovernance, QwenMoEGovernanceConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--seed-pack",
        type=Path,
        default=Path("outputs/moe/upcycling/qwen_alpha_seed_pack"),
        help="Local Qwen3-MoE seed pack directory",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/moe_v8"),
        help="Prepared MoE data directory",
    )
    p.add_argument("--split", choices=["train", "eval", "test"], default="test")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32", "auto"], default="bfloat16")
    p.add_argument("--trainable-trunk", action="store_true", help="Do not freeze trunk parameters")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/moe/upcycling/qwen_alpha_wrapper_smoke.json"),
        help="Output JSON smoke report",
    )
    return p.parse_args()


def choose_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def read_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    if len(rows) < limit:
        raise ValueError(f"wanted {limit} rows from {path}, got {len(rows)}")
    return rows


def scalar_tensors(
    rows: list[dict[str, Any]],
    scalar_fields: tuple[str, ...],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = []
    masks = []
    for row in rows:
        targets = row.get("scalar_targets") or {}
        row_values = []
        row_masks = []
        for field in scalar_fields:
            value = targets.get(field)
            if isinstance(value, int | float):
                row_values.append(float(value))
                row_masks.append(1.0)
            else:
                row_values.append(0.0)
                row_masks.append(0.0)
        values.append(row_values)
        masks.append(row_masks)
    return (
        torch.tensor(values, dtype=torch.float32, device=device),
        torch.tensor(masks, dtype=torch.float32, device=device),
    )


def main() -> int:
    args = parse_args()
    start = time.time()
    device = choose_device(args.device)
    vocab = MoEVocab.from_metadata(args.data_dir / "metadata.json")
    rows = read_rows(args.data_dir / f"{args.split}.jsonl", args.batch_size)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.seed_pack, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(
        [row["text"] for row in rows],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_length,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    model_cfg = QwenMoEGovernanceConfig(
        num_routes=len(vocab.route2id),
        num_taxonomy_patterns=len(vocab.taxonomy_pattern2id),
        num_scalar_targets=len(vocab.scalar_fields),
        freeze_trunk=not args.trainable_trunk,
    )
    model = QwenMoEForGovernance.from_seed_pack(
        args.seed_pack,
        model_cfg,
        dtype=args.dtype,
        local_files_only=True,
    ).to(device)
    model.eval()

    labels = torch.tensor([int(row["label_id"]) for row in rows], dtype=torch.long, device=device)
    route_ids = torch.tensor([int(row["route_id"]) for row in rows], dtype=torch.long, device=device)
    taxonomy_ids = torch.tensor(
        [int(row["taxonomy_pattern_id"]) for row in rows],
        dtype=torch.long,
        device=device,
    )
    scalar_targets, scalar_mask = scalar_tensors(rows, vocab.scalar_fields, device)

    with torch.no_grad():
        outputs = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            route_ids=route_ids,
        )
        _, parts = multitask_loss(
            outputs,
            labels=labels,
            route_ids=route_ids,
            taxonomy_ids=taxonomy_ids,
            scalar_targets=scalar_targets,
            scalar_mask=scalar_mask,
            weights=MoELossWeights(),
        )

    report = {
        "status": "pass",
        "seed_pack": str(args.seed_pack),
        "data_dir": str(args.data_dir),
        "split": args.split,
        "row_ids": [row["id"] for row in rows],
        "device": str(device),
        "dtype": args.dtype,
        "input_shape": list(encoded["input_ids"].shape),
        "outputs": {
            key: list(value.shape)
            for key, value in outputs.items()
            if isinstance(value, torch.Tensor) and value.ndim > 0
        },
        "selected_routes": [int(v) for v in outputs["selected_routes"].detach().cpu().tolist()],
        "loss_parts": parts,
        "elapsed_seconds": round(time.time() - start, 3),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Seed pack       : {args.seed_pack}")
    print(f"Device / dtype  : {device} / {args.dtype}")
    print(f"Input shape     : {report['input_shape']}")
    print(f"Governance logits: {report['outputs']['governance_logits']}")
    print(f"Route logits    : {report['outputs']['route_logits']}")
    print(f"Taxonomy logits : {report['outputs']['taxonomy_logits']}")
    print(f"Scalar preds    : {report['outputs']['scalar_preds']}")
    print(f"Loss            : {parts['loss']:.4f}")
    print(f"Output report   : {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
