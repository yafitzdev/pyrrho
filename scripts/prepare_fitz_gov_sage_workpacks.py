"""Prepare source workpacks for the fitz-gov-sage v1 transformation.

The script selects a balanced 10k-row source slice from the clean V10 repaired
multitask prep and writes compact JSON workpacks for subagents. It does not
create final training rows; it creates resumable transformation inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


DEFAULT_SOURCE_DATA_DIR = Path("data/multitask_g5_1_v10_repaired")
DEFAULT_OUTPUT_DIR = Path("data/fitz_gov_sage_v1_workpacks")
DEFAULT_PROMPT = Path("docs/prompts/FITZ_GOV_SAGE_TRANSFORM_SUBAGENT.md")
SOURCE_SPLITS = ("train", "eval", "test")
PRIMARY_DIMENSIONS = (
    "label",
    "route",
    "query_contract",
    "retrieval_modality",
    "retrieval_obligation",
    "answerability_shape",
    "taxonomy_pattern",
    "difficulty",
    "dataset_version",
)


def stable_int(value: str, *, seed: str) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            if raw.strip():
                yield json.loads(raw)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def dim_value(row: dict[str, Any], dim: str) -> str:
    if dim == "retrieval_obligation":
        value = row.get(dim)
        return str(value) if value else "<masked>"
    value = row.get(dim)
    return str(value) if value not in (None, "") else "<missing>"


def primary_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(dim_value(row, dim) for dim in PRIMARY_DIMENSIONS[:6])


def has_obligation(row: dict[str, Any]) -> bool:
    return int(row.get("retrieval_obligation_id", -1)) >= 0


def source_row(row: dict[str, Any]) -> dict[str, Any]:
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
        "source_id": row["id"],
        "source_split": row.get("split"),
        "source_dataset_version": row.get("dataset_version"),
        "source_alpha": row.get("alpha_source"),
        "difficulty": row.get("difficulty"),
        "query": row.get("query"),
        "query_rewritten": row.get("query_rewritten"),
        "contexts": row.get("contexts") or [],
        "context_features": row.get("context_features") or [],
        "labels": labels,
        "scalar_targets": row.get("scalar_targets") or {},
        "evidence_chain": row.get("evidence_chain"),
        "grounding_targets": row.get("grounding_targets") or [],
        "near_miss_reason": row.get("near_miss_reason"),
        "selection_key": dict(zip(PRIMARY_DIMENSIONS[:6], primary_key(row), strict=True)),
    }


def group_rows(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[primary_key(row)].append(row)
    return groups


def round_robin_select(
    rows: list[dict[str, Any]],
    *,
    limit: int,
    seed: str,
    excluded_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    excluded_ids = excluded_ids or set()
    groups = group_rows(row for row in rows if str(row["id"]) not in excluded_ids)
    for group_rows_ in groups.values():
        group_rows_.sort(key=lambda row: stable_int(str(row["id"]), seed=seed))
    keys = sorted(groups, key=lambda key: stable_int("|".join(key), seed=seed))

    selected: list[dict[str, Any]] = []
    indices: Counter[tuple[str, ...]] = Counter()
    while len(selected) < limit and keys:
        next_keys: list[tuple[str, ...]] = []
        for key in keys:
            idx = indices[key]
            values = groups[key]
            if idx < len(values):
                selected.append(values[idx])
                indices[key] += 1
                if indices[key] < len(values):
                    next_keys.append(key)
            if len(selected) >= limit:
                break
        keys = next_keys
    return selected


def balanced_fill_by_dimension(
    rows: list[dict[str, Any]],
    *,
    limit: int,
    dimension: str,
    starting_counts: Counter[str],
    seed: str,
    excluded_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    excluded_ids = excluded_ids or set()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row["id"]) not in excluded_ids:
            grouped[dim_value(row, dimension)].append(row)
    queues = {
        value: round_robin_select(values, limit=len(values), seed=f"{seed}:{value}")
        for value, values in grouped.items()
    }
    indices: Counter[str] = Counter()
    counts = Counter(starting_counts)
    selected: list[dict[str, Any]] = []
    while len(selected) < limit:
        available = [
            value for value, values in queues.items() if indices[value] < len(values)
        ]
        if not available:
            break
        value = min(
            available,
            key=lambda item: (counts[item], stable_int(item, seed=seed)),
        )
        row = queues[value][indices[value]]
        indices[value] += 1
        selected.append(row)
        counts[value] += 1
    return selected


def count_dimensions(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counters = {dim: Counter() for dim in PRIMARY_DIMENSIONS}
    for row in rows:
        for dim in PRIMARY_DIMENSIONS:
            counters[dim][dim_value(row, dim)] += 1
    return {dim: dict(sorted(counter.items())) for dim, counter in counters.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data-dir", type=Path, default=DEFAULT_SOURCE_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--target-source-rows", type=int, default=10_000)
    parser.add_argument("--workpack-size", type=int, default=50)
    parser.add_argument("--min-obligation-fraction", type=float, default=0.65)
    parser.add_argument("--seed", default="fitz-gov-sage-v1-20260616")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.time()
    source_dir = args.source_data_dir.resolve()
    output_dir = args.output_dir.resolve()
    prompt_path = args.prompt.resolve()
    metadata = read_json(source_dir / "metadata.json")

    rows: list[dict[str, Any]] = []
    for split in SOURCE_SPLITS:
        rows.extend(read_jsonl(source_dir / f"{split}.jsonl"))

    bad_versions = sorted(
        {
            str(row.get("dataset_version"))
            for row in rows
            if str(row.get("dataset_version")) not in {"v6", "v7", "v8", "v9", "v10"}
        }
    )
    if bad_versions:
        raise ValueError(f"source data contains non-V10-clean versions: {bad_versions}")

    target = min(int(args.target_source_rows), len(rows))
    obligation_target = min(
        int(math.floor(target * float(args.min_obligation_fraction))),
        sum(1 for row in rows if has_obligation(row)),
    )

    obligation_rows = [row for row in rows if has_obligation(row)]
    selected_obligation = round_robin_select(
        obligation_rows,
        limit=obligation_target,
        seed=f"{args.seed}:obligation",
    )
    selected_ids = {str(row["id"]) for row in selected_obligation}
    selected_broad = balanced_fill_by_dimension(
        [row for row in rows if not has_obligation(row)],
        limit=target - len(selected_obligation),
        dimension="answerability_shape",
        starting_counts=Counter(dim_value(row, "answerability_shape") for row in selected_obligation),
        seed=f"{args.seed}:broad-answerability",
        excluded_ids=selected_ids,
    )
    selected_ids.update(str(row["id"]) for row in selected_broad)
    if len(selected_obligation) + len(selected_broad) < target:
        selected_broad.extend(
            round_robin_select(
                rows,
                limit=target - len(selected_obligation) - len(selected_broad),
                seed=f"{args.seed}:broad-backfill",
                excluded_ids=selected_ids,
            )
        )
    selected = selected_obligation + selected_broad
    selected.sort(key=lambda row: stable_int(str(row["id"]), seed=f"{args.seed}:final"))

    source_selection_rows = [source_row(row) for row in selected]
    write_jsonl(output_dir / "source_selection.jsonl", source_selection_rows)

    workpack_dir = output_dir / "workpacks"
    workpack_count = math.ceil(len(source_selection_rows) / int(args.workpack_size))
    for pack_idx in range(workpack_count):
        start_idx = pack_idx * int(args.workpack_size)
        end_idx = start_idx + int(args.workpack_size)
        items = source_selection_rows[start_idx:end_idx]
        workpack = {
            "workpack_id": f"fitz_gov_sage_v1_pack_{pack_idx:04d}",
            "prompt": str(prompt_path),
            "source_data_dir": str(source_dir),
            "target_dataset": "fitz-gov-sage-v1",
            "target_model_line": "pyrrho-sage-nano-g1",
            "expected_output_rows": len(items) * 2,
            "items": items,
        }
        write_json(workpack_dir / f"pack_{pack_idx:04d}.json", workpack)

    selected_source_counts = Counter(str(row.get("alpha_source") or "unknown") for row in selected)
    selected_split_counts = Counter(str(row.get("split") or "unknown") for row in selected)
    selected_version_counts = Counter(str(row.get("dataset_version") or "unknown") for row in selected)
    summary = {
        "created_at_unix": time.time(),
        "elapsed_seconds": time.time() - start,
        "source_data_dir": str(source_dir),
        "source_metadata": {
            "splits": metadata.get("splits"),
            "dataset_version_counts": metadata.get("dataset_version_counts"),
            "retrieval_obligation_labeled_rows": metadata.get("retrieval_obligation_labeled_rows"),
            "retrieval_obligation_masked_rows": metadata.get("retrieval_obligation_masked_rows"),
        },
        "target_source_rows": target,
        "selected_source_rows": len(selected),
        "expected_stage_rows": len(selected) * 2,
        "workpack_size": int(args.workpack_size),
        "workpacks": workpack_count,
        "min_obligation_fraction": float(args.min_obligation_fraction),
        "selected_obligation_rows": sum(1 for row in selected if has_obligation(row)),
        "selected_masked_obligation_rows": sum(1 for row in selected if not has_obligation(row)),
        "selected_split_counts": dict(sorted(selected_split_counts.items())),
        "selected_dataset_version_counts": dict(sorted(selected_version_counts.items())),
        "selected_source_counts": dict(sorted(selected_source_counts.items())),
        "dimension_counts": count_dimensions(selected),
    }
    write_json(output_dir / "metadata.json", summary)

    summary_md = [
        "# fitz-gov-sage v1 Workpack Selection",
        "",
        f"- Source data: `{source_dir}`",
        f"- Selected source rows: **{len(selected):,}**",
        f"- Expected stage rows: **{len(selected) * 2:,}**",
        f"- Workpacks: **{workpack_count:,}** x {int(args.workpack_size)} source rows",
        f"- Obligation-labeled source rows: **{summary['selected_obligation_rows']:,}**",
        f"- Obligation-masked source rows: **{summary['selected_masked_obligation_rows']:,}**",
        "",
        "## Selected Dataset Versions",
        "",
    ]
    for key, value in sorted(selected_version_counts.items()):
        summary_md.append(f"- `{key}`: {value:,}")
    summary_md.extend(["", "## Selected Splits", ""])
    for key, value in sorted(selected_split_counts.items()):
        summary_md.append(f"- `{key}`: {value:,}")
    (output_dir / "summary.md").write_text("\n".join(summary_md) + "\n", encoding="utf-8")

    print(f"source rows   : {len(rows):,}")
    print(f"selected      : {len(selected):,}")
    print(f"stage rows    : {len(selected) * 2:,}")
    print(f"workpacks     : {workpack_count:,}")
    print(f"output        : {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
