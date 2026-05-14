"""sweep.py — Hyperparameter sweep orchestrator.

Reads a grid YAML, generates one config per combination, runs `train_encoder.py`
in a subprocess for each, then aggregates results into a single comparison table.

Two sweep modes:
- "grid": full cartesian product of all axes (use sparingly, combinatorial)
- "coordinate": one axis at a time from a `baseline` point (cheap, good for v1 exploration)

Run from project root:
    python scripts/sweep.py --grid configs/sweep_grids/encoder_v1.yaml
    python scripts/sweep.py --grid configs/sweep_grids/encoder_v1.yaml --mode grid

Grid YAML schema:
    base_config: configs/encoder/modernbert_base.yaml
    output_base: outputs/sweeps/encoder_v1
    mode: coordinate                # or "grid"
    seed: 42                        # single seed per cell; use run_seeds.py later for variance
    baseline:                       # only used in coordinate mode
      training.label_smoothing: 0.15
      training.class_weights: [2.3, 2.3, 1.0]
      training.num_train_epochs: 5
    axes:
      training.label_smoothing: [0.0, 0.05, 0.1, 0.15, 0.2]
      training.learning_rate:   [3.0e-5, 5.0e-5, 1.0e-4]
      training.num_train_epochs: [3, 5, 7]
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import yaml


def set_nested(cfg: dict, dotted_key: str, value: Any) -> None:
    """Set cfg["training"]["label_smoothing"] given key 'training.label_smoothing'."""
    parts = dotted_key.split(".")
    node = cfg
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def safe_token(value: Any) -> str:
    """Build a filename-safe token from a hyperparameter value."""
    if isinstance(value, list):
        return "_".join(safe_token(v) for v in value)
    s = str(value)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s[:32]


def cell_name(assignment: dict[str, Any]) -> str:
    """e.g. 'ls0.15__lr5e-05' for {'training.label_smoothing': 0.15, 'training.learning_rate': 5e-5}."""
    parts = []
    for k, v in assignment.items():
        leaf = k.split(".")[-1]
        # shorten common keys for readability
        short = {
            "label_smoothing": "ls",
            "learning_rate": "lr",
            "class_weights": "cw",
            "num_train_epochs": "ep",
            "warmup_ratio": "wu",
            "weight_decay": "wd",
            "per_device_train_batch_size": "bs",
        }.get(leaf, leaf)
        parts.append(f"{short}{safe_token(v)}")
    return "__".join(parts) or "default"


def coordinate_cells(baseline: dict[str, Any], axes: dict[str, list]) -> list[dict[str, Any]]:
    """One axis at a time. baseline cell first, then one cell per (axis, value != baseline)."""
    cells = [dict(baseline)]  # baseline included once
    for axis, values in axes.items():
        for v in values:
            if baseline.get(axis) == v:
                continue
            assignment = dict(baseline)
            assignment[axis] = v
            cells.append(assignment)
    # dedupe while preserving order
    seen = set()
    deduped = []
    for c in cells:
        key = json.dumps(c, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
    return deduped


def grid_cells(axes: dict[str, list]) -> list[dict[str, Any]]:
    """Full cartesian product. Use with care."""
    keys = list(axes.keys())
    values_lists = [axes[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values_lists)]


def materialize_config(base_config: dict, assignment: dict[str, Any]) -> dict:
    cfg = copy.deepcopy(base_config)
    for k, v in assignment.items():
        set_nested(cfg, k, v)
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid", type=Path, required=True, help="Path to grid YAML.")
    p.add_argument(
        "--mode",
        choices=("coordinate", "grid"),
        default=None,
        help="Override grid YAML 'mode'. coordinate=one axis at a time from baseline; grid=cartesian product.",
    )
    p.add_argument(
        "--python",
        type=str,
        default=str(Path(".venv/Scripts/python.exe").resolve()),
        help="Python interpreter for the training subprocess.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate cells + configs but do not call train_encoder.py.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    with args.grid.open("r", encoding="utf-8") as fh:
        grid_cfg = yaml.safe_load(fh)

    base_config_path = Path(grid_cfg["base_config"]).resolve()
    output_base = Path(grid_cfg["output_base"]).resolve()
    output_base.mkdir(parents=True, exist_ok=True)
    mode = args.mode or grid_cfg.get("mode", "coordinate")
    seed = int(grid_cfg.get("seed", 42))
    axes = grid_cfg["axes"]

    with base_config_path.open("r", encoding="utf-8") as fh:
        base_config = yaml.safe_load(fh)

    if mode == "coordinate":
        baseline = grid_cfg.get("baseline") or {}
        cells = coordinate_cells(baseline, axes)
    elif mode == "grid":
        cells = grid_cells(axes)
    else:
        print(f"Unknown mode: {mode}", file=sys.stderr)
        return 1

    print(f"Grid YAML       : {args.grid}")
    print(f"Base config     : {base_config_path}")
    print(f"Output base     : {output_base}")
    print(f"Mode            : {mode}")
    print(f"Cells           : {len(cells)}")
    print(f"Seed            : {seed}")
    print()
    for i, cell in enumerate(cells, 1):
        print(f"  [{i:3d}] {cell_name(cell):<40s} {cell}")
    if args.dry_run:
        print("\nDry run — exiting before any training.")
        return 0

    results = []
    for i, cell in enumerate(cells, 1):
        cell_dir = output_base / cell_name(cell)
        cell_dir.mkdir(parents=True, exist_ok=True)

        materialized = materialize_config(base_config, cell)
        cell_config_path = cell_dir / "config.yaml"
        with cell_config_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(materialized, fh, sort_keys=False)

        print(f"\n{'=' * 80}")
        print(f"[{i}/{len(cells)}] {cell_name(cell)} -> {cell_dir}")
        print(f"{'=' * 80}")

        cmd = [
            args.python,
            "scripts/train_encoder.py",
            "--config",
            str(cell_config_path),
            "--output-dir",
            str(cell_dir),
            "--seed",
            str(seed),
            "--no-wandb",
        ]
        proc = subprocess.run(cmd, cwd=Path.cwd())
        metrics_path = cell_dir / "final_metrics.json"
        if proc.returncode in (0, 2) and metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as fh:
                m = json.load(fh)
            results.append(
                {
                    "cell": cell,
                    "name": cell_name(cell),
                    "output_dir": str(cell_dir),
                    "eval_calibrated": m["eval_calibrated"],
                    "eval_uncalibrated": m["eval_uncalibrated"],
                    "tier0_calibrated": m.get("tier0_calibrated"),
                    "tier0_uncalibrated": m.get("tier0_uncalibrated"),
                    "threshold": m.get("threshold"),
                }
            )
        else:
            print(f"  CRASHED or no metrics. exit={proc.returncode}")
            results.append({"cell": cell, "name": cell_name(cell), "error": True})

    # Final table
    print(f"\n\n{'=' * 110}")
    print(f"SWEEP SUMMARY — {len(results)} cells")
    print(f"{'=' * 110}")
    print(
        f"{'cell':<40s} {'eval_acc':>9s} {'eval_FT':>8s} "
        f"{'t0_acc':>7s} {'t0_FT':>7s} {'tau':>6s}"
    )
    # Sort by eval calibrated accuracy descending so the winner is on top
    def sort_key(r):
        if r.get("error"):
            return -1
        return r["eval_calibrated"]["accuracy"]

    for r in sorted(results, key=sort_key, reverse=True):
        if r.get("error"):
            print(f"{r['name']:<40s}   CRASHED")
            continue
        ec = r["eval_calibrated"]
        t0 = r.get("tier0_calibrated") or {}
        print(
            f"{r['name']:<40s} {ec['accuracy'] * 100:>9.2f} {ec['false_trustworthy_rate'] * 100:>8.2f} "
            f"{t0.get('accuracy', 0) * 100:>7.2f} {t0.get('false_trustworthy_rate', 0) * 100:>7.2f} "
            f"{r['threshold']:>6.2f}"
        )

    summary_path = output_base / "sweep_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "grid_config": str(args.grid),
                "base_config": str(base_config_path),
                "mode": mode,
                "seed": seed,
                "results": results,
            },
            fh,
            indent=2,
        )
    print(f"\nWrote sweep summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
