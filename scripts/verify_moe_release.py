"""Verify a packaged pyrrho MoE release directory offline.

The verifier checks the package manifest, artifact presence, recorded sizes,
recorded SHA-256 hashes, packaged post-hoc verifier loadability, and optionally
runs a small inference smoke from the package itself.

Run from project root:
    python scripts/verify_moe_release.py \
      --package-dir models/pyrrho-MoE-g3-alpha \
      --input data/moe_v8/test.jsonl \
      --max-samples 4 \
      --device cpu \
      --output models/pyrrho-MoE-g3-alpha/release_verify_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
from pyrrho.moe.posthoc_verifier import PosthocVerifierPackage, read_json, sha256_file

DEFAULT_PACKAGE_DIR = Path("models/pyrrho-MoE-g3-alpha")
DEFAULT_POLICY = "trustworthy_quorum_2_of_3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--input", type=Path, default=None, help="Optional JSONL rows for an inference smoke")
    parser.add_argument("--output", type=Path, default=None, help="Write verification report JSON")
    parser.add_argument("--predictions-output", type=Path, default=None, help="Write smoke predictions JSONL")
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--policy", choices=ENSEMBLE_POLICY_NAMES, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--skip-hash-check", action="store_true")
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def resolve_manifest_root(package_dir: Path, manifest: dict[str, Any]) -> Path:
    raw = manifest.get("root")
    if raw is None:
        return package_dir.resolve()
    root = Path(str(raw))
    if root.is_absolute():
        return root.resolve()
    return (package_dir.resolve() / root).resolve()


def resolve_path(path: str | Path, root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (root / candidate).resolve()


def check_file_artifact(
    *,
    label: str,
    path: Path,
    artifact: dict[str, Any],
    verify_hashes: bool,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "label": label,
        "path": str(path),
        "exists": path.exists(),
        "size_ok": None,
        "sha256_ok": None,
    }
    if not path.exists():
        row["ok"] = False
        row["error"] = "missing"
        return row
    actual_size = int(path.stat().st_size)
    row["bytes"] = actual_size
    expected_size = artifact.get("bytes")
    if expected_size is not None:
        row["expected_bytes"] = int(expected_size)
        row["size_ok"] = actual_size == int(expected_size)
    expected_hash = artifact.get("sha256")
    if verify_hashes and expected_hash:
        actual_hash = sha256_file(path)
        row["sha256"] = actual_hash
        row["expected_sha256"] = str(expected_hash)
        row["sha256_ok"] = actual_hash == str(expected_hash)
    checks = [
        value
        for value in (row["size_ok"], row["sha256_ok"])
        if value is not None
    ]
    row["ok"] = bool(row["exists"]) and all(bool(value) for value in checks)
    return row


def verify_release_structure(
    package_dir: Path,
    *,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    package_dir = package_dir.resolve()
    manifest_path = package_dir / "manifest.json"
    manifest = read_json(manifest_path)
    package = PosthocVerifierPackage.load(package_dir, verify_hashes=verify_hashes)
    manifest_root = resolve_manifest_root(package_dir, manifest)

    file_checks: list[dict[str, Any]] = []
    for relative in ["README.md", "manifest.json"]:
        path = package_dir / relative
        file_checks.append(
            {
                "label": relative,
                "path": str(path),
                "exists": path.exists(),
                "ok": path.exists(),
            }
        )

    config_path = resolve_path(str(manifest.get("config", "")), manifest_root)
    data_dir = resolve_path(str(manifest.get("data_dir", "")), manifest_root)
    file_checks.append(
        {
            "label": "metadata",
            "path": str(data_dir / "metadata.json"),
            "exists": (data_dir / "metadata.json").exists(),
            "ok": (data_dir / "metadata.json").exists(),
        }
    )

    seed_summaries: list[dict[str, Any]] = []
    for entry in manifest["seeds"]:
        seed = int(entry["seed"])
        artifacts = entry.get("artifacts", {})
        checkpoint = resolve_path(entry["checkpoint"], manifest_root)
        seed_config = resolve_path(entry.get("config") or manifest.get("config"), manifest_root)
        verifier = package_dir / entry["verifier_path"]
        report = package_dir / entry["report_path"]
        checks = [
            check_file_artifact(
                label=f"seed_{seed}/checkpoint",
                path=checkpoint,
                artifact=artifacts.get("checkpoint", {}),
                verify_hashes=verify_hashes,
            ),
            check_file_artifact(
                label=f"seed_{seed}/config",
                path=seed_config,
                artifact=artifacts.get("config", {}),
                verify_hashes=verify_hashes,
            ),
            check_file_artifact(
                label=f"seed_{seed}/verifier",
                path=verifier,
                artifact=artifacts.get("verifier", {}),
                verify_hashes=verify_hashes,
            ),
            check_file_artifact(
                label=f"seed_{seed}/report",
                path=report,
                artifact=artifacts.get("report", {}),
                verify_hashes=verify_hashes,
            ),
        ]
        file_checks.extend(checks)
        seed_summaries.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint),
                "config": str(seed_config),
                "base_threshold": float(entry["base_threshold"]),
                "verifier_threshold": float(entry["selected_threshold"]),
                "ok": all(bool(check["ok"]) for check in checks),
            }
        )

    ok = all(bool(check["ok"]) for check in file_checks)
    elapsed = time.perf_counter() - started
    return {
        "schema_version": "pyrrho_moe_release_verify_v1",
        "package_dir": str(package_dir),
        "manifest_schema_version": manifest.get("schema_version"),
        "release": manifest.get("release", {}),
        "manifest_root": str(manifest_root),
        "seed_ids": list(package.seed_ids),
        "feature_width": package.feature_width,
        "file_checks": file_checks,
        "seeds": seed_summaries,
        "elapsed_ms": elapsed * 1000.0,
        "ok": ok,
    }


def _predict_seed(
    *,
    package: PosthocVerifierPackage,
    entry: dict[str, Any],
    manifest_root: Path,
    rows: list[dict[str, Any]],
    seed: int,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    checkpoint = resolve_path(entry["checkpoint"], manifest_root)
    config = resolve_path(entry["config"], manifest_root)
    data_dir = resolve_path(entry["data_dir"], manifest_root)
    runtime = MoEInferenceRuntime.from_checkpoint(
        checkpoint=checkpoint,
        config_path=config,
        metadata_path=data_dir / "metadata.json",
        verifier_package=package,
        device=device,
    )
    return runtime.predict_rows(
        rows,
        batch_size=batch_size,
        base_threshold=float(entry["base_threshold"]),
        verifier_seed=seed,
    )


def run_inference_smoke(
    *,
    package_dir: Path,
    input_path: Path,
    max_samples: int,
    batch_size: int,
    policy: str,
    device_name: str,
    predictions_output: Path | None,
    verify_hashes: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    package = PosthocVerifierPackage.load(package_dir, verify_hashes=verify_hashes)
    manifest = package.manifest
    manifest_root = resolve_manifest_root(package_dir.resolve(), manifest)
    rows = read_jsonl(input_path, limit=max_samples)
    device = select_device(device_name)
    seed_outputs: dict[int, list[dict[str, Any]]] = {}
    for seed in package.seed_ids:
        entry = package.seed(seed).manifest_entry
        seed_outputs[int(seed)] = _predict_seed(
            package=package,
            entry=entry,
            manifest_root=manifest_root,
            rows=rows,
            seed=int(seed),
            device=device,
            batch_size=batch_size,
        )
    predictions = combine_seed_prediction_rows(seed_outputs, policy=policy)
    if predictions_output is not None:
        write_jsonl(predictions_output, predictions)
    counts: dict[str, int] = {}
    rejected = 0
    for row in predictions:
        counts[str(row["classification"])] = counts.get(str(row["classification"]), 0) + 1
        rejected += sum(1 for value in row["seed_rejected"].values() if bool(value))
    elapsed = time.perf_counter() - started
    return {
        "input": str(input_path),
        "rows": len(predictions),
        "policy": policy,
        "device": str(device),
        "counts": counts,
        "seed_level_rejections": rejected,
        "elapsed_ms": elapsed * 1000.0,
        "rows_per_second": float(len(predictions)) / max(elapsed, 1e-9),
        "predictions_output": str(predictions_output) if predictions_output is not None else None,
        "ok": len(predictions) == len(rows),
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    package_dir = args.package_dir.resolve()
    verify_hashes = not args.skip_hash_check
    report = verify_release_structure(package_dir, verify_hashes=verify_hashes)
    policy = args.policy or report.get("release", {}).get("default_policy") or DEFAULT_POLICY
    if args.input is not None:
        predictions_output = args.predictions_output
        if predictions_output is None and args.output is not None:
            predictions_output = args.output.with_suffix(".predictions.jsonl")
        report["inference_smoke"] = run_inference_smoke(
            package_dir=package_dir,
            input_path=args.input,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            policy=str(policy),
            device_name=args.device,
            predictions_output=predictions_output,
            verify_hashes=verify_hashes,
        )
        report["ok"] = bool(report["ok"]) and bool(report["inference_smoke"]["ok"])
    if args.output is not None:
        write_report(args.output, report)
        print(f"Wrote release verification report: {args.output}")
    print(
        "Release verification: "
        f"ok={report['ok']} package={package_dir} seeds={report['seed_ids']} "
        f"feature_width={report['feature_width']}"
    )
    if "inference_smoke" in report:
        smoke = report["inference_smoke"]
        print(
            "Inference smoke: "
            f"rows={smoke['rows']} policy={smoke['policy']} "
            f"counts={smoke['counts']} rejected={smoke['seed_level_rejections']}"
        )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
