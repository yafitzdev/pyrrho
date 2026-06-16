"""Aggregate pyrrho multitask encoder seed runs.

The multitask trainer writes nested `final_metrics.json` files. This script
collects seed_<N>/final_metrics.json runs, computes mean/std/min/max for every
numeric eval/test metric, and writes the summary shape expected by
`scripts/package_multitask_encoder.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any


if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--status", default="local_candidate_not_published_no_blind_label_qa")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--fitz-gov-base-version", required=True)
    parser.add_argument("--candidate-profile", required=True)
    parser.add_argument("--candidate-rows", type=int, required=True)
    parser.add_argument("--candidate-status", required=True)
    parser.add_argument("--blind-label-qa", default="not_run")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 7])
    parser.add_argument("--best-seed", type=int, default=None)
    parser.add_argument(
        "--selection-reason",
        default=(
            "Lowest held-out false-TRUSTWORTHY rate, then retrieval-obligation "
            "macro F1, then governance accuracy."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def flatten_numeric(prefix: str, value: Any, out: dict[str, float]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flatten_numeric(child_prefix, child, out)
        return
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        out[prefix] = float(value)


def stat(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(mean(values)),
        "std": float(stdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def metric(metrics: dict[str, Any], path: tuple[str, ...], default: float = 0.0) -> float:
    cursor: Any = metrics
    for part in path:
        if not isinstance(cursor, dict) or part not in cursor:
            return default
        cursor = cursor[part]
    return float(cursor)


def choose_best_seed(per_seed: dict[int, dict[str, Any]]) -> int:
    def key(seed: int) -> tuple[float, float, float]:
        metrics = per_seed[seed]
        return (
            metric(metrics, ("test", "governance_calibrated", "false_trustworthy_rate"), 1.0),
            -metric(metrics, ("test", "retrieval_obligation", "retrieval_obligation_macro_f1"), 0.0),
            -metric(metrics, ("test", "governance_calibrated", "accuracy"), 0.0),
        )

    return min(per_seed, key=key)


def source_counts_by_split(data_dir: Path) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for split in ("train", "eval", "test"):
        counts: Counter[str] = Counter()
        for row in read_jsonl(data_dir / f"{split}.jsonl"):
            counts[str(row.get("alpha_source") or "unknown")] += 1
        result[split] = dict(counts)
    return result


def main() -> int:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    data_dir = args.data_dir.resolve()
    metadata = read_json(data_dir / "metadata.json")

    seed_reports: dict[str, dict[str, Any]] = {}
    per_seed: list[dict[str, Any]] = []
    for seed in args.seeds:
        path = base_dir / f"seed_{seed}" / "final_metrics.json"
        if not path.exists():
            raise FileNotFoundError(path)
        report = read_json(path)
        seed_reports[str(seed)] = report
        per_seed.append({"seed": seed, "path": str(path), "metrics": report})

    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    for split in ("eval", "test"):
        flattened_by_seed: list[dict[str, float]] = []
        for row in per_seed:
            flat: dict[str, float] = {}
            flatten_numeric("", row["metrics"][split], flat)
            flattened_by_seed.append(flat)
        keys = sorted(set().union(*(flat.keys() for flat in flattened_by_seed)))
        aggregate[split] = {
            key: stat([flat[key] for flat in flattened_by_seed if key in flat])
            for key in keys
        }

    per_seed_by_id = {int(row["seed"]): row["metrics"] for row in per_seed}
    best_seed = args.best_seed if args.best_seed is not None else choose_best_seed(per_seed_by_id)
    summary = {
        "model": args.model,
        "status": args.status,
        "created_at": datetime.now(UTC).isoformat(),
        "config": str(args.config),
        "data_dir": str(args.data_dir),
        "base_model": args.base_model,
        "fitz_gov_base_version": args.fitz_gov_base_version,
        "candidate_profile": args.candidate_profile,
        "candidate_rows": args.candidate_rows,
        "candidate_status": args.candidate_status,
        "blind_label_qa": args.blind_label_qa,
        "splits": metadata.get("splits", {}),
        "dataset_version_counts": metadata.get("dataset_version_counts", {}),
        "source_counts_by_split": source_counts_by_split(data_dir),
        "answerability_shape_counts": metadata.get("answerability_shape_counts", {}),
        "retrieval_obligation_counts": metadata.get("retrieval_obligation_counts", {}),
        "retrieval_obligation_labeled_rows": metadata.get("retrieval_obligation_labeled_rows"),
        "retrieval_obligation_masked_rows": metadata.get("retrieval_obligation_masked_rows"),
        "seeds": args.seeds,
        "best_seed": best_seed,
        "best_seed_selection_reason": args.selection_reason,
        "aggregate": aggregate,
        "seed_reports": seed_reports,
        "per_seed": per_seed,
    }

    output = args.output or (base_dir / "summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    test = aggregate["test"]
    print(f"summary      : {output}")
    print(f"best_seed    : {best_seed}")
    for key in (
        "governance_calibrated.accuracy",
        "governance_calibrated.false_trustworthy_rate",
        "query_contract.query_contract_macro_f1",
        "taxonomy.taxonomy_accuracy",
        "scalars.scalar_mae",
        "retrieval_action.retrieval_action_macro_f1",
        "gap_type.gap_type_macro_f1",
        "answerability_shape.answerability_shape_macro_f1",
        "retrieval_modality.retrieval_modality_macro_f1",
        "retrieval_obligation.retrieval_obligation_macro_f1",
    ):
        row = test[key]
        scale = 1.0 if key == "scalars.scalar_mae" else 100.0
        print(f"{key:56s}: {row['mean'] * scale:.4f} +/- {row['std'] * scale:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
