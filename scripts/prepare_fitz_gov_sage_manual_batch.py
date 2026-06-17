"""Prepare manual fitz-gov-sage messy-pack workpacks.

This script does not transform rows. It only selects source rows, writes
workpack JSON files, and records worker assignments so GPT-5.4 subagents can
manually write the JSONL output rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


SOURCE_SPLITS = ("train", "eval", "test")
DEFAULT_SOURCE_DATA_DIR = Path("data/multitask_g5_1_v10_repaired")
DEFAULT_PROMPT = Path("docs/prompts/FITZ_GOV_SAGE_MESSY_TRANSFORM_WORKER.md")


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            if not raw.strip():
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def source_id(row: dict[str, Any]) -> str:
    value = row.get("source_id") or row.get("id")
    if not value:
        raise ValueError("source row missing source_id/id")
    return str(value)


def load_excluded_ids(paths: Iterable[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            excluded.add(source_id(row))
    return excluded


def normalize_item(row: dict[str, Any], *, split: str | None = None) -> dict[str, Any]:
    if "labels" in row and isinstance(row["labels"], dict):
        labels = dict(row["labels"])
    else:
        labels = {
            "label": row.get("label"),
            "label_id": row.get("label_id"),
            "query_contract": row.get("query_contract"),
            "query_contract_id": row.get("query_contract_id"),
            "route": row.get("route"),
            "route_id": row.get("route_id"),
            "taxonomy_pattern": row.get("taxonomy_pattern"),
            "taxonomy_pattern_id": row.get("taxonomy_pattern_id"),
            "retrieval_action": row.get("retrieval_action"),
            "retrieval_action_id": row.get("retrieval_action_id"),
            "gap_type": row.get("gap_type"),
            "gap_type_id": row.get("gap_type_id"),
            "answerability_shape": row.get("answerability_shape"),
            "answerability_shape_id": row.get("answerability_shape_id"),
            "retrieval_modality": row.get("retrieval_modality"),
            "retrieval_modality_id": row.get("retrieval_modality_id"),
            "retrieval_obligation": row.get("retrieval_obligation"),
            "retrieval_obligation_id": row.get("retrieval_obligation_id"),
        }

    return {
        "source_id": source_id(row),
        "source_split": row.get("source_split") or row.get("split") or split,
        "source_dataset_version": row.get("source_dataset_version") or row.get("dataset_version") or row.get("version"),
        "source_alpha": row.get("source_alpha") or row.get("alpha_source"),
        "difficulty": row.get("difficulty"),
        "query": row.get("query"),
        "query_rewritten": row.get("query_rewritten"),
        "contexts": row.get("contexts") or [],
        "context_features": row.get("context_features") or [],
        "labels": labels,
        "scalar_targets": row.get("scalar_targets") or {},
        "evidence_chain": row.get("evidence_chain") or [],
        "grounding_targets": row.get("grounding_targets") or [],
    }


def load_source_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.source_selection:
        return [normalize_item(row) for row in read_jsonl(args.source_selection)]

    rows: list[dict[str, Any]] = []
    for split in SOURCE_SPLITS:
        path = args.source_data_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        rows.extend(normalize_item(row, split=split) for row in read_jsonl(path))
    if not rows:
        raise ValueError(f"no source rows loaded from {args.source_data_dir}")
    return rows


def stable_shuffle(rows: list[dict[str, Any]], seed: str) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(f"{seed}|{row['source_id']}".encode("utf-8")).hexdigest(),
    )


def balanced_rows(rows: list[dict[str, Any]], limit: int, seed: str) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        labels = row.get("labels") or {}
        key = (str(labels.get("retrieval_modality")), str(labels.get("label")))
        buckets[key].append(row)
    for key in list(buckets):
        buckets[key] = stable_shuffle(buckets[key], f"{seed}|{key}")
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    keys = sorted(buckets)
    while len(selected) < limit:
        moved = False
        for key in keys:
            bucket = buckets[key]
            while bucket:
                candidate = bucket.pop(0)
                sid = str(candidate["source_id"])
                if sid in seen:
                    continue
                selected.append(candidate)
                seen.add(sid)
                moved = True
                break
            if len(selected) >= limit:
                break
        if not moved:
            break
    return selected


def split_workpacks(rows: list[dict[str, Any]], workpack_size: int) -> list[list[dict[str, Any]]]:
    return [rows[idx : idx + workpack_size] for idx in range(0, len(rows), workpack_size)]


def worker_assignments(workpack_count: int, worker_count: int) -> list[dict[str, Any]]:
    assignments: list[dict[str, Any]] = []
    if workpack_count <= 0:
        return assignments
    worker_count = max(1, int(worker_count))
    per_worker = math.ceil(workpack_count / worker_count)
    for worker_idx in range(worker_count):
        start = worker_idx * per_worker
        end = min(workpack_count - 1, start + per_worker - 1)
        if start >= workpack_count:
            break
        assignments.append(
            {
                "worker": chr(ord("A") + worker_idx),
                "pack_start": f"pack_{start:04d}.json",
                "pack_end": f"pack_{end:04d}.json",
                "pack_indices": list(range(start, end + 1)),
                "output_files": [f"pack_{idx:04d}.jsonl" for idx in range(start, end + 1)],
            }
        )
    return assignments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-selection", type=Path, help="Existing source_selection.jsonl to reuse.")
    source.add_argument("--source-data-dir", type=Path, default=None, help="Prepared multitask data dir with train/eval/test.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exclude-source-selection", type=Path, action="append", default=[])
    parser.add_argument("--target-source-rows", type=int, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--selection-mode", choices=("sequential", "balanced", "stable-shuffle"), default="sequential")
    parser.add_argument("--seed", default="fitz-gov-sage-manual")
    parser.add_argument("--workpack-size", type=int, default=17)
    parser.add_argument("--worker-count", type=int, default=6)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--target-dataset", required=True)
    parser.add_argument("--target-model-line", default="pyrrho-sage-nano")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.source_data_dir is None:
        args.source_data_dir = DEFAULT_SOURCE_DATA_DIR

    output_dir = args.output_dir.resolve()
    workpack_dir = output_dir / "workpacks"
    subagent_dir = output_dir / "subagent_outputs"
    source_rows = load_source_rows(args)
    excluded = load_excluded_ids(args.exclude_source_selection)
    available = [row for row in source_rows if str(row["source_id"]) not in excluded]

    if args.selection_mode == "balanced":
        ordered = balanced_rows(available, len(available), args.seed)
    elif args.selection_mode == "stable-shuffle":
        ordered = stable_shuffle(available, args.seed)
    else:
        ordered = available

    start = max(0, int(args.start_index))
    selected = ordered[start : start + int(args.target_source_rows)]
    if len(selected) != int(args.target_source_rows):
        raise ValueError(f"selected {len(selected)} rows, expected {args.target_source_rows}")

    output_dir.mkdir(parents=True, exist_ok=True)
    workpack_dir.mkdir(parents=True, exist_ok=True)
    subagent_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(output_dir / "source_selection.jsonl", selected)
    packs = split_workpacks(selected, int(args.workpack_size))
    for idx, items in enumerate(packs):
        workpack = {
            "workpack_id": f"{args.target_dataset.replace('-', '_').replace('.', '_')}_pack_{idx:04d}",
            "prompt": str(args.prompt.resolve()),
            "target_dataset": args.target_dataset,
            "target_model_line": args.target_model_line,
            "expected_output_rows": len(items) * 2,
            "items": items,
        }
        write_json(workpack_dir / f"pack_{idx:04d}.json", workpack)

    assignments = worker_assignments(len(packs), int(args.worker_count))
    summary = {
        "source_selection": str(args.source_selection.resolve()) if args.source_selection else None,
        "source_data_dir": str(args.source_data_dir.resolve()) if args.source_data_dir else None,
        "output_dir": str(output_dir),
        "target_dataset": args.target_dataset,
        "target_model_line": args.target_model_line,
        "prompt": str(args.prompt.resolve()),
        "excluded_source_ids": len(excluded),
        "selection_mode": args.selection_mode,
        "seed": args.seed,
        "start_index": start,
        "selected_source_rows": len(selected),
        "expected_stage_rows": len(selected) * 2,
        "workpack_size": int(args.workpack_size),
        "workpacks": len(packs),
        "worker_count": int(args.worker_count),
        "worker_assignments": assignments,
        "label_counts": dict(Counter(str((row.get("labels") or {}).get("label")) for row in selected)),
        "modality_counts": dict(Counter(str((row.get("labels") or {}).get("retrieval_modality")) for row in selected)),
        "obligation_labeled_rows": sum(1 for row in selected if (row.get("labels") or {}).get("retrieval_obligation")),
    }
    write_json(output_dir / "batch_manifest.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

