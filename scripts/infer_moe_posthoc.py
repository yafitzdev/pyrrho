"""Run Stage 0 MoE inference with an optional packaged post-hoc verifier.

Input JSONL accepts either prepared MoE rows or raw RAG rows:
    {"id": "...", "query": "...", "contexts": ["...", "..."]}

Run from project root:
    python scripts/infer_moe_posthoc.py \
      --package-dir outputs/moe/stage0_7_posthoc_verifier_g3_ft028_package \
      --seed 42 \
      --input data/moe_v8/test.jsonl \
      --max-samples 8 \
      --output outputs/moe/stage0_7_posthoc_verifier_g3_ft028_package/inference_smoke.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from pyrrho.moe.inference import (
    ENSEMBLE_POLICY_NAMES,
    MoEInferenceRuntime,
    combine_seed_prediction_rows,
    read_jsonl,
    write_jsonl,
)
from pyrrho.moe.posthoc_verifier import PosthocVerifierPackage

DEFAULT_PACKAGE_DIR = Path("outputs/moe/stage0_7_posthoc_verifier_g3_ft028_package")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="JSONL rows to score")
    parser.add_argument("--output", type=Path, default=None, help="Default: stdout JSONL")
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--seed", type=int, default=42, help="Packaged verifier seed to use")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Override package seed checkpoint")
    parser.add_argument("--config", type=Path, default=None, help="Override package seed config")
    parser.add_argument("--data-dir", type=Path, default=None, help="Dir containing metadata.json")
    parser.add_argument("--base-threshold", type=float, default=None)
    parser.add_argument("--verifier-threshold", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-verifier", action="store_true", help="Run base MoE thresholding only")
    parser.add_argument(
        "--policy",
        choices=("seed", *ENSEMBLE_POLICY_NAMES),
        default="seed",
        help="Use one seed or combine all packaged seeds with an ensemble policy",
    )
    parser.add_argument(
        "--skip-hash-check",
        action="store_true",
        help="Load package without verifier/report checksum validation",
    )
    return parser.parse_args()


def resolve_path(path: str | Path, root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (root / candidate).resolve()


def resolve_manifest_root(package: PosthocVerifierPackage) -> Path:
    raw = package.manifest.get("root")
    if raw is None:
        return package.package_dir
    root = Path(str(raw))
    if root.is_absolute():
        return root.resolve()
    return (package.package_dir / root).resolve()


def seed_entry(package: PosthocVerifierPackage, seed: int) -> dict[str, Any]:
    return package.seed(seed).manifest_entry


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def predict_for_seed(
    *,
    package: PosthocVerifierPackage | None,
    entry: dict[str, Any],
    manifest_root: Path,
    rows: list[dict[str, Any]],
    seed: int | None,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    checkpoint = args.checkpoint or resolve_path(entry["checkpoint"], manifest_root)
    config = args.config
    if config is None and entry.get("config"):
        config = resolve_path(entry["config"], manifest_root)
    data_dir = args.data_dir or resolve_path(entry["data_dir"], manifest_root)
    base_threshold = (
        float(args.base_threshold)
        if args.base_threshold is not None
        else float(entry.get("base_threshold", 0.34))
    )
    runtime = MoEInferenceRuntime.from_checkpoint(
        checkpoint=checkpoint,
        config_path=config,
        metadata_path=data_dir / "metadata.json",
        verifier_package=package,
        device=select_device(str(args.device)),
    )
    return runtime.predict_rows(
        rows,
        batch_size=args.batch_size,
        base_threshold=base_threshold,
        verifier_seed=seed,
        verifier_threshold=args.verifier_threshold,
    )


def main() -> int:
    args = parse_args()
    package = None
    manifest_root = Path.cwd()
    entry: dict[str, Any] = {}
    if not args.no_verifier:
        package = PosthocVerifierPackage.load(
            args.package_dir,
            verify_hashes=not args.skip_hash_check,
        )
        manifest_root = resolve_manifest_root(package)
        entry = seed_entry(package, args.seed)

    rows = read_jsonl(args.input, limit=args.max_samples)
    if args.no_verifier:
        if args.policy != "seed":
            raise ValueError("--policy must be seed when --no-verifier is set")
        if args.checkpoint is None or args.data_dir is None:
            raise ValueError("--checkpoint and --data-dir are required when --no-verifier is set")
        entry = {
            "checkpoint": str(args.checkpoint),
            "config": str(args.config) if args.config is not None else "",
            "data_dir": str(args.data_dir),
            "base_threshold": args.base_threshold if args.base_threshold is not None else 0.34,
        }
        predictions = predict_for_seed(
            package=None,
            entry=entry,
            manifest_root=Path.cwd(),
            rows=rows,
            seed=None,
            args=args,
        )
    elif args.policy == "seed":
        predictions = predict_for_seed(
            package=package,
            entry=entry,
            manifest_root=manifest_root,
            rows=rows,
            seed=int(args.seed),
            args=args,
        )
    else:
        assert package is not None
        seed_outputs: dict[int, list[dict[str, Any]]] = {}
        for seed in package.seed_ids:
            seed_outputs[int(seed)] = predict_for_seed(
                package=package,
                entry=seed_entry(package, int(seed)),
                manifest_root=manifest_root,
                rows=rows,
                seed=int(seed),
                args=args,
            )
        predictions = combine_seed_prediction_rows(seed_outputs, policy=args.policy)
    if args.output is not None:
        write_jsonl(args.output, predictions)
        print(f"Wrote predictions: {args.output}")
    else:
        for row in predictions:
            print(json.dumps(row, ensure_ascii=False))
    guarded_counts: dict[str, int] = {}
    rejected = 0
    for row in predictions:
        guarded_counts[str(row["classification"])] = guarded_counts.get(str(row["classification"]), 0) + 1
        if isinstance(row.get("seed_rejected"), dict):
            rejected += sum(1 for value in row["seed_rejected"].values() if bool(value))
        else:
            rejected += int(bool(row.get("verifier_rejected", False)))
    print(
        "Summary          : "
        f"rows={len(predictions)} policy={args.policy} counts={guarded_counts} rejected={rejected}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
