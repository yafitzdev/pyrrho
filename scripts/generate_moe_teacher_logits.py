"""Generate governance-teacher logits for MoE Stage 1+ distillation.

The output is a sidecar JSONL keyed by case id. It intentionally does not
rewrite `data/moe_v8`; the trainer can opt into the sidecar with
`--teacher-logits-dir`.

Run from project root:
    python scripts/generate_moe_teacher_logits.py --max-samples 32 --splits train eval
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from pyrrho.data import ID2LABEL


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher", type=Path, default=Path("models/pyrrho-nano-g3"))
    p.add_argument("--data-dir", type=Path, default=Path("data/moe_v8"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/moe/teacher_logits/pyrrho_nano_g3"))
    p.add_argument("--splits", nargs="+", default=["train", "eval"])
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--sample-mode", choices=["prefix", "random"], default="prefix")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return p.parse_args()


def choose_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def load_rows(path: Path, *, limit: int | None, sample_mode: str, seed: int) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            if raw.strip():
                rows.append(json.loads(raw))
    if limit is not None and limit < len(rows):
        if sample_mode == "prefix":
            rows = rows[:limit]
        else:
            rng = random.Random(seed)
            indices = list(range(len(rows)))
            rng.shuffle(indices)
            rows = [rows[i] for i in sorted(indices[:limit])]
    return rows


def batched(rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]


def split_seed(seed: int, split: str) -> int:
    offsets = {"train": 0, "eval": 1, "validation": 1, "test": 2}
    return seed + offsets.get(split, 0)


def write_split(
    *,
    split: str,
    rows: list[dict[str, Any]],
    model: torch.nn.Module,
    tokenizer: Any,
    teacher: Path,
    output_path: Path,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for batch in tqdm(batched(rows, batch_size), desc=f"teacher-{split}", leave=False):
            enc = tokenizer(
                [row["text"] for row in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            with torch.no_grad():
                logits = model(**enc).logits.float().cpu()
            probs = torch.softmax(logits, dim=-1)
            for row, row_logits, row_probs in zip(batch, logits, probs, strict=True):
                pred_id = int(torch.argmax(row_probs).item())
                fh.write(
                    json.dumps(
                        {
                            "id": row["id"],
                            "split": split,
                            "teacher_model": str(teacher),
                            "logits": [float(v) for v in row_logits.tolist()],
                            "probabilities": [float(v) for v in row_probs.tolist()],
                            "prediction_id": pred_id,
                            "prediction_label": ID2LABEL[pred_id],
                            "gold_label": row.get("label"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def main() -> int:
    args = parse_args()
    device = choose_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    model = AutoModelForSequenceClassification.from_pretrained(args.teacher).to(device).eval()

    print(f"Teacher  : {args.teacher}")
    print(f"Data dir : {args.data_dir}")
    print(f"Output   : {args.output_dir}")
    print(f"Device   : {device}")
    for split in args.splits:
        rows = load_rows(
            args.data_dir / f"{split}.jsonl",
            limit=args.max_samples,
            sample_mode=args.sample_mode,
            seed=split_seed(args.seed, split),
        )
        print(f"{split:8s}: {len(rows)} rows")
        write_split(
            split=split,
            rows=rows,
            model=model,
            tokenizer=tokenizer,
            teacher=args.teacher,
            output_path=args.output_dir / f"{split}.jsonl",
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=device,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
