"""Merge clean multitask rows with stage-aware fitz-gov-sage rows.

The clean base corpus preserves broad governance behavior. The sage corpus adds
production-shaped evidence packs and explicit query-planning rows. Stage-aware
masking is expected to be applied before this merge.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from pyrrho.manifest import write_manifest


DEFAULT_BASE_DIR = Path("data/multitask_g5_1_v10_repaired")
DEFAULT_SAGE_DIR = Path("data/fitz_gov_sage_v1_1")
DEFAULT_OUTPUT_DIR = Path("data/multitask_sage_g1_1_v10_clean_plus_stage")
DEFAULT_CONFIG = Path("configs/encoder/modernbert_base_sage_g1_1_v10_clean_plus_stage.yaml")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    if not rows:
        raise ValueError(f"no rows loaded from {path}")
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "alpha_source_counts": dict(Counter(str(row.get("alpha_source") or "unknown") for row in rows)),
        "dataset_version_counts": dict(
            Counter(str(row.get("dataset_version") or row.get("version") or "unknown") for row in rows)
        ),
        "stage_counts": dict(Counter(str(row.get("stage") or "clean_base") for row in rows)),
        "label_counts": dict(Counter(str(row.get("label")) for row in rows if row.get("label"))),
        "retrieval_obligation_labeled_rows": sum(1 for row in rows if int(row.get("retrieval_obligation_id", -1)) >= 0),
        "retrieval_obligation_masked_rows": sum(1 for row in rows if int(row.get("retrieval_obligation_id", -1)) < 0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--sage-dir", type=Path, default=DEFAULT_SAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.time()
    base_dir = args.base_dir.resolve()
    sage_dir = args.sage_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_metadata = read_json(base_dir / "metadata.json")
    sage_metadata = read_json(sage_dir / "metadata.json")
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    seen_ids: set[str] = set()

    for split in ("train", "eval", "test"):
        merged: list[dict[str, Any]] = []
        for source_name, source_dir in (("clean_base", base_dir), ("sage_stage", sage_dir)):
            for row in read_jsonl(source_dir / f"{split}.jsonl"):
                row_id = str(row.get("id") or "")
                if not row_id:
                    raise ValueError(f"{source_dir / f'{split}.jsonl'}: row missing id")
                if row_id in seen_ids:
                    raise ValueError(f"duplicate row id after merge: {row_id}")
                seen_ids.add(row_id)
                copied = dict(row)
                copied.setdefault("mix_source", source_name)
                merged.append(copied)
        rows_by_split[split] = merged
        write_jsonl(output_dir / f"{split}.jsonl", merged)

    all_rows = [row for rows in rows_by_split.values() for row in rows]
    metadata = dict(base_metadata)
    metadata.update(
        {
            "source": {
                "base_dir": str(base_dir),
                "sage_dir": str(sage_dir),
                "base_source": base_metadata.get("source", {}),
                "sage_source": sage_metadata.get("source", {}),
            },
            "mixing": "clean_v10_base_plus_stage_aware_fitz_gov_sage_v1_1",
            "splits": {split: len(rows) for split, rows in rows_by_split.items()},
            "rows": len(all_rows),
            "source_rows": len({str(row.get("source_id") or row.get("id")) for row in all_rows}),
            **row_counts(all_rows),
        }
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_manifest(
        output_dir=output_dir,
        config_path=args.config,
        seed=0,
        pyrrho_repo=Path.cwd(),
        fitz_gov_repo=Path.cwd().parent / "fitz-gov",
        start_time=start,
        extra={
            "script": "prepare_sage_mixed_data.py",
            "base_dir": str(base_dir),
            "sage_dir": str(sage_dir),
        },
    )

    print(f"output : {output_dir}")
    print(f"splits : {metadata['splits']}")
    print(f"rows   : {metadata['rows']:,}")
    print(f"stages : {metadata['stage_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
