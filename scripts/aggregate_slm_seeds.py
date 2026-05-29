"""aggregate_slm_seeds.py - Aggregate multi-seed SLM training results.

The encoder's run_seeds.py expects a different final_metrics.json schema
(eval_calibrated / eval_uncalibrated / tier0_calibrated / tier0_uncalibrated +
selected threshold). The SLM path doesn't have threshold calibration — labels
come straight out of the decoder — so this script reads the simpler
{eval, tier0_sanity, training} schema written by scripts/train_slm.py and
writes a summary.json suitable for build_model_card.py and the multi-seed
mean +/- std reporting.

Run from project root:
    python scripts/aggregate_slm_seeds.py --base-dir outputs/multi_seed_slm
    python scripts/aggregate_slm_seeds.py --seeds 42 1337 7
"""

from __future__ import annotations

import argparse
import json
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
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--base-dir",
        type=Path,
        default=Path("outputs/multi_seed_slm"),
        help="Directory containing seed_<N>/ subdirs (default: outputs/multi_seed_slm)",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 1337, 7],
        help="Which seed dirs to aggregate (default: 42 1337 7)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write summary.json (default: <base-dir>/summary.json)",
    )
    return p.parse_args()


def fmt_mean_std(values: list[float], pct: bool = True) -> str:
    if not values:
        return "n/a"
    if len(values) < 2:
        return f"{values[0] * 100:.2f}" if pct else f"{values[0]:.4f}"
    m = mean(values)
    s = stdev(values)
    if pct:
        return f"{m * 100:.2f} +/- {s * 100:.2f}"
    return f"{m:.4f} +/- {s:.4f}"


def main() -> int:
    args = parse_args()
    base = args.base_dir.resolve()
    if not base.exists():
        print(f"ERROR: base dir not found: {base}", file=sys.stderr)
        return 1

    per_seed = []
    missing = []
    for seed in args.seeds:
        metrics_path = base / f"seed_{seed}" / "final_metrics.json"
        if not metrics_path.exists():
            missing.append(seed)
            continue
        with metrics_path.open("r", encoding="utf-8") as fh:
            metrics = json.load(fh)
        per_seed.append({"seed": seed, "metrics": metrics})

    if missing:
        print(f"WARNING: missing final_metrics for seeds: {missing}", file=sys.stderr)
    if not per_seed:
        print("ERROR: no per-seed metrics found.", file=sys.stderr)
        return 1

    print(f"Aggregating {len(per_seed)} seed(s): {[r['seed'] for r in per_seed]}")
    print(f"Base         : {base}")

    # Aggregate
    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    for split_label, split_key, keys in (
        ("eval", "eval", METRIC_KEYS_EVAL),
        ("tier0_sanity", "tier0_sanity", METRIC_KEYS_TIER0),
    ):
        agg: dict[str, dict[str, float]] = {}
        print(f"\n[{split_label}]")
        for k in keys:
            vals = [
                r["metrics"][split_key][k]
                for r in per_seed
                if split_key in r["metrics"] and k in r["metrics"][split_key]
            ]
            if not vals:
                continue
            agg[k] = {
                "mean": float(mean(vals)),
                "std": float(stdev(vals)) if len(vals) > 1 else 0.0,
                "values": [float(v) for v in vals],
            }
            print(f"  {k:30s}: {fmt_mean_std(vals)}")
        aggregate[split_key] = agg

    # Decode-health sanity (fallback rate)
    print("\n[decode health]")
    for split_key in ("eval", "tier0_sanity"):
        vals = []
        for r in per_seed:
            health = r["metrics"].get(split_key, {}).get("decode_health", {})
            if "fraction_fallback" in health:
                vals.append(health["fraction_fallback"])
        if vals:
            print(f"  {split_key + ' fraction_fallback':30s}: {fmt_mean_std(vals, pct=False)}")

    # Training durations
    durations = [
        r["metrics"].get("training", {}).get("duration_seconds", 0)
        for r in per_seed
        if "training" in r["metrics"]
    ]
    if durations:
        print("\n[timing]")
        print(f"  training_seconds (mean)       : {mean(durations):.1f}")
        print(f"  training_seconds (each)       : {[round(d,1) for d in durations]}")

    summary = {
        "config_path": per_seed[0]["metrics"].get("config_path"),
        "seeds": [r["seed"] for r in per_seed],
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    out = args.output or (base / "summary.json")
    with out.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nWrote -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
