"""run_seeds.py — Train encoder with N different seeds, aggregate metrics.

Validates whether the eval numbers are lucky-seed artifacts or genuine.
Reports mean +/- std across seeds for the key release-gate metrics.

Run from project root:
    python scripts/run_seeds.py
    python scripts/run_seeds.py --seeds 42 1337 7 99 --config configs/encoder/modernbert_base.yaml
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


METRIC_KEYS_EVAL = [
    "accuracy",
    "macro_f1",
    "false_trustworthy_rate",
    "recall_abstain",
    "recall_disputed",
    "recall_trustworthy",
    "precision_abstain",
    "precision_disputed",
    "precision_trustworthy",
]
METRIC_KEYS_TIER0 = ["accuracy", "false_trustworthy_rate"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=Path("configs/encoder/modernbert_base.yaml"))
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Processed dataset directory to pass through to train_encoder.py",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 7])
    p.add_argument(
        "--base-output-dir",
        type=Path,
        default=Path("outputs/multi_seed"),
        help="Each seed gets its own subdirectory under here",
    )
    p.add_argument(
        "--python",
        type=str,
        default=str(Path(".venv/Scripts/python.exe").resolve()),
        help="Python interpreter to use for the training subprocess",
    )
    return p.parse_args()


def fmt_mean_std(values: list[float], pct: bool = True) -> str:
    if len(values) < 2:
        return f"{values[0] * 100:.2f}" if pct and values else "n/a"
    m = mean(values)
    s = stdev(values)
    if pct:
        return f"{m * 100:.2f} +/- {s * 100:.2f}"
    return f"{m:.4f} +/- {s:.4f}"


def main() -> int:
    args = parse_args()
    args.base_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config       : {args.config}")
    print(f"Data dir     : {args.data_dir}")
    print(f"Seeds        : {args.seeds}")
    print(f"Output base  : {args.base_output_dir.resolve()}")
    print(f"Python       : {args.python}\n")

    per_seed_results = []
    for i, seed in enumerate(args.seeds, 1):
        out_dir = args.base_output_dir / f"seed_{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"=" * 80)
        print(f"[{i}/{len(args.seeds)}] seed={seed} -> {out_dir}")
        print(f"=" * 80)

        cmd = [
            args.python,
            "scripts/train_encoder.py",
            "--config",
            str(args.config),
            "--data-dir",
            str(args.data_dir),
            "--output-dir",
            str(out_dir),
            "--seed",
            str(seed),
            "--no-wandb",
        ]

        proc = subprocess.run(cmd, cwd=Path.cwd())
        # Exit 2 = "ran successfully, gates failed" — still collect metrics.
        if proc.returncode not in (0, 2):
            print(f"[{i}/{len(args.seeds)}] seed={seed} CRASHED with exit {proc.returncode}")
            continue

        metrics_path = out_dir / "final_metrics.json"
        if not metrics_path.exists():
            print(f"[{i}/{len(args.seeds)}] seed={seed} NO METRICS at {metrics_path}")
            continue
        with metrics_path.open("r", encoding="utf-8") as fh:
            metrics = json.load(fh)
        per_seed_results.append({"seed": seed, "metrics": metrics})
        print(f"\n[{i}/{len(args.seeds)}] seed={seed} done")
        gate_split = "test_calibrated" if "test_calibrated" in metrics else "eval_calibrated"
        print(
            f"  {gate_split.replace('_calibrated', '')} (cal): "
            f"acc={metrics[gate_split]['accuracy']:.4f}  "
            f"FT={metrics[gate_split]['false_trustworthy_rate']:.4f}  "
            f"tau={metrics['threshold']:.2f}"
        )
        if "tier0_calibrated" in metrics:
            print(
                f"  tier0 (cal): acc={metrics['tier0_calibrated']['accuracy']:.4f}  "
                f"FT={metrics['tier0_calibrated']['false_trustworthy_rate']:.4f}\n"
            )

    if not per_seed_results:
        print("\nNo successful runs. Aborting.")
        return 1

    # Aggregate
    print("\n" + "=" * 80)
    print(f"AGGREGATE ACROSS {len(per_seed_results)} SEED(S): {[r['seed'] for r in per_seed_results]}")
    print("=" * 80)

    split_specs = [
        ("eval uncalibrated", "eval_uncalibrated", METRIC_KEYS_EVAL),
        ("eval calibrated   ", "eval_calibrated", METRIC_KEYS_EVAL),
    ]
    if all("test_uncalibrated" in r["metrics"] for r in per_seed_results):
        split_specs.extend(
            [
                ("test uncalibrated", "test_uncalibrated", METRIC_KEYS_EVAL),
                ("test calibrated   ", "test_calibrated", METRIC_KEYS_EVAL),
            ]
        )
    if all("tier0_uncalibrated" in r["metrics"] for r in per_seed_results):
        split_specs.extend(
            [
                ("tier0 uncalibrated", "tier0_uncalibrated", METRIC_KEYS_TIER0),
                ("tier0 calibrated  ", "tier0_calibrated", METRIC_KEYS_TIER0),
            ]
        )

    for split_label, split_key, keys in split_specs:
        print(f"\n[{split_label}]")
        for k in keys:
            vals = [r["metrics"][split_key][k] for r in per_seed_results]
            print(f"  {k:30s}: {fmt_mean_std(vals)}")

    # Selected threshold per seed
    print(f"\n[threshold]")
    taus = [r["metrics"]["threshold"] for r in per_seed_results]
    print(f"  {'tau (selected)':30s}: " + ", ".join(f"{t:.2f}" for t in taus))

    # Save aggregate
    summary_path = args.base_output_dir / "summary.json"
    summary = {
        "config": str(args.config),
        "seeds": [r["seed"] for r in per_seed_results],
        "per_seed": per_seed_results,
        "aggregate": {
            split_key: {
                k: {
                    "mean": mean([r["metrics"][split_key][k] for r in per_seed_results]),
                    "std": stdev([r["metrics"][split_key][k] for r in per_seed_results])
                    if len(per_seed_results) > 1
                    else 0.0,
                }
                for k in keys
            }
            for split_key, keys in [
                *[
                    ("eval_uncalibrated", METRIC_KEYS_EVAL),
                    ("eval_calibrated", METRIC_KEYS_EVAL),
                ],
                *(
                    [
                        ("test_uncalibrated", METRIC_KEYS_EVAL),
                        ("test_calibrated", METRIC_KEYS_EVAL),
                    ]
                    if all("test_uncalibrated" in r["metrics"] for r in per_seed_results)
                    else []
                ),
                *(
                    [
                        ("tier0_uncalibrated", METRIC_KEYS_TIER0),
                        ("tier0_calibrated", METRIC_KEYS_TIER0),
                    ]
                    if all("tier0_uncalibrated" in r["metrics"] for r in per_seed_results)
                    else []
                ),
            ]
        },
    }
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nWrote summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
