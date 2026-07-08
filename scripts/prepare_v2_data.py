"""Prepare fitz-gov-v2 accepted vault rows for pyrrho v2-alpha training.

Example:
    python scripts/prepare_v2_data.py ^
      --accepted-rows C:/Users/yanfi/PycharmProjects/fitz-gov-modern_generator/outputs/fitz_gov_v2_evidence_online_acceptance_loop/merged_vault/accepted.rows.jsonl ^
      --output data/v2_alpha
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict
from pyrrho.v2 import (
    EVIDENCE_KIND_KEYS,
    EVIDENCE_VERDICTS,
    FAILURE_MODES,
    RETRIEVAL_INTENT_KEYS,
    V2_FULL_LABEL_MASK,
    build_v2_full_text,
    build_v2_query_text,
    encode_v2_labels,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    query = row.get("query_text") or row.get("query") or ""
    contexts = row.get("contexts") or []
    if not isinstance(contexts, list) or not contexts:
        raise ValueError(f"{row.get('id')}: missing contexts")

    labels = encode_v2_labels(row)
    return {
        "id": row.get("id"),
        "text": build_v2_full_text(query, contexts),
        "query_only_text": build_v2_query_text(query),
        "labels": labels,
        "label_mask": list(V2_FULL_LABEL_MASK),
        "evidence_verdict": row.get("evidence_verdict"),
        "failure_mode": row.get("failure_mode"),
        "retrieval_intents": row.get("retrieval_intents") or {},
        "evidence_kinds": row.get("evidence_kinds") or {},
        "difficulty": row.get("difficulty"),
        "pack_shape": row.get("pack_shape"),
        "evidence_shape": row.get("evidence_shape"),
        "source_format": row.get("source_format"),
        "context_count": row.get("context_count"),
    }


def stratify_key(row: dict[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("evidence_verdict")),
            str(row.get("failure_mode")),
            str(row.get("difficulty")),
        ]
    )


def stable_split(
    rows: list[dict[str, Any]], *, train_ratio: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split by row id hash so adding new rows cannot reshuffle existing rows."""
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    cutoff = int(train_ratio * ((1 << 64) - 1))
    for row in rows:
        row_id = str(row.get("id") or "")
        digest = hashlib.blake2b(f"{seed}:{row_id}".encode(), digest_size=8).digest()
        score = int.from_bytes(digest, "big")
        if score <= cutoff:
            train_rows.append(row)
        else:
            eval_rows.append(row)
    return train_rows, eval_rows


def print_distribution(name: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n{name}: {len(rows):,}")
    for field in (
        "evidence_verdict",
        "failure_mode",
        "difficulty",
        "pack_shape",
        "evidence_shape",
        "source_format",
    ):
        counts = Counter(row.get(field) for row in rows)
        print(f"  {field}:")
        for key, value in counts.most_common():
            print(f"    {key}: {value:,} ({value / len(rows):.1%})")
    for field, keys in (
        ("retrieval_intents", RETRIEVAL_INTENT_KEYS),
        ("evidence_kinds", EVIDENCE_KIND_KEYS),
    ):
        print(f"  {field}:")
        for key in keys:
            value = sum(1 for row in rows if bool((row.get(field) or {}).get(key)))
            print(f"    {key}: {value:,} ({value / len(rows):.1%})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/v2_alpha"))
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-mode",
        choices=["stable-hash"],
        default="stable-hash",
        help="stable-hash keeps existing row ids in the same split when rows are added",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.accepted_rows.exists():
        print(f"ERROR: accepted rows not found: {args.accepted_rows}", file=sys.stderr)
        return 1
    if not 0.5 < args.train_ratio < 1.0:
        print("ERROR: --train-ratio must be between 0.5 and 1.0", file=sys.stderr)
        return 1

    raw_rows = read_jsonl(args.accepted_rows)
    rows = [normalize_row(row) for row in raw_rows]
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate row ids in accepted vault")

    expected_verdicts = set(EVIDENCE_VERDICTS)
    expected_failures = set(FAILURE_MODES)
    bad_verdicts = sorted({row["evidence_verdict"] for row in rows} - expected_verdicts)
    bad_failures = sorted({row["failure_mode"] for row in rows} - expected_failures)
    if bad_verdicts or bad_failures:
        raise ValueError(f"bad labels: verdicts={bad_verdicts} failures={bad_failures}")

    train_rows, eval_rows = stable_split(rows, train_ratio=args.train_ratio, seed=args.seed)
    if not train_rows or not eval_rows:
        raise ValueError("stable split produced an empty train or eval split")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "train.jsonl", train_rows)
    write_jsonl(output / "eval.jsonl", eval_rows)

    ds = DatasetDict(
        {
            "train": Dataset.from_list(train_rows),
            "eval": Dataset.from_list(eval_rows),
        }
    )
    ds.save_to_disk(str(output / "hf_dataset"))

    manifest = {
        "accepted_rows": str(args.accepted_rows.resolve()),
        "output": str(output),
        "seed": args.seed,
        "split_mode": args.split_mode,
        "train_ratio": args.train_ratio,
        "total": len(rows),
        "train": len(train_rows),
        "eval": len(eval_rows),
        "label_heads": {
            "evidence_verdict": list(EVIDENCE_VERDICTS),
            "failure_mode": list(FAILURE_MODES),
            "retrieval_intents": list(RETRIEVAL_INTENT_KEYS),
            "evidence_kinds": list(EVIDENCE_KIND_KEYS),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print_distribution("train", train_rows)
    print_distribution("eval", eval_rows)
    print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
