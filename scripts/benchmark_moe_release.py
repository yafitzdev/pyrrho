"""Benchmark a packaged pyrrho MoE release directory locally.

This script is intentionally offline. It assumes the package already exists on
disk, loads the packaged seed checkpoints once, warms them up, then measures
repeated inference over a fixed JSONL sample.

Run from project root:
    python scripts/benchmark_moe_release.py \
      --package-dir models/pyrrho-MoE-g3-alpha \
      --input data/moe_v8/test.jsonl \
      --max-samples 32 \
      --batch-size 16 \
      --device cpu \
      --output models/pyrrho-MoE-g3-alpha/cpu_benchmark_32.json

Batch-size profile:
    python scripts/benchmark_moe_release.py \
      --package-dir models/pyrrho-MoE-g3-alpha \
      --input data/moe_v8/test.jsonl \
      --max-samples 32 \
      --batch-sizes 1 4 8 16 32 \
      --device cpu \
      --output models/pyrrho-MoE-g3-alpha/cpu_batch_profile_32.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import threading
import time
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import torch

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

try:
    import psutil
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    psutil = None

from pyrrho.moe.inference import (
    ENSEMBLE_POLICY_NAMES,
    MoEInferenceRuntime,
    combine_seed_prediction_rows,
    read_jsonl,
)
from pyrrho.moe.posthoc_verifier import PosthocVerifierPackage

DEFAULT_PACKAGE_DIR = Path("models/pyrrho-MoE-g3-alpha")
DEFAULT_POLICY = "trustworthy_quorum_2_of_3"


class RssSampler(AbstractContextManager["RssSampler"]):
    def __init__(self, interval_s: float = 0.02) -> None:
        self.interval_s = float(interval_s)
        self.available = psutil is not None
        self._process = psutil.Process(os.getpid()) if psutil is not None else None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[int] = []

    def current(self) -> int | None:
        if self._process is None:
            return None
        return int(self._process.memory_info().rss)

    def __enter__(self) -> "RssSampler":
        if self._process is None:
            return self
        first = self.current()
        if first is not None:
            self.samples.append(first)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            value = self.current()
            if value is not None:
                self.samples.append(value)

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        last = self.current()
        if last is not None:
            self.samples.append(last)

    @property
    def start(self) -> int | None:
        return self.samples[0] if self.samples else None

    @property
    def peak(self) -> int | None:
        return max(self.samples) if self.samples else None

    @property
    def end(self) -> int | None:
        return self.samples[-1] if self.samples else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Run a one-load batch-size profile; overrides --batch-size when set",
    )
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--policy", choices=("seed", *ENSEMBLE_POLICY_NAMES), default=None)
    parser.add_argument("--seed", type=int, default=42, help="Seed used when --policy seed")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--skip-hash-check", action="store_true")
    return parser.parse_args()


def select_device(name: str) -> torch.device:
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


def bytes_to_mib(value: int | None) -> float | None:
    if value is None:
        return None
    return float(value) / (1024.0 * 1024.0)


def normalize_batch_sizes(batch_size: int, batch_sizes: list[int] | None) -> list[int]:
    values = list(batch_sizes) if batch_sizes else [batch_size]
    normalized: list[int] = []
    seen: set[int] = set()
    for value in values:
        size = int(value)
        if size <= 0:
            raise ValueError(f"Batch size must be positive, got {size}")
        if size not in seen:
            normalized.append(size)
            seen.add(size)
    return normalized


def metric_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
        "median": float(statistics.median(values)),
    }


def prediction_summary(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    rejected = 0
    for row in predictions:
        counts[str(row["classification"])] = counts.get(str(row["classification"]), 0) + 1
        seed_rejected = row.get("seed_rejected")
        if isinstance(seed_rejected, dict):
            rejected += sum(1 for value in seed_rejected.values() if bool(value))
        else:
            rejected += int(bool(row.get("verifier_rejected", False)))
    return {
        "rows": len(predictions),
        "counts": counts,
        "seed_level_rejections": rejected,
    }


def fastest_profile(profiles: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not profiles:
        return None
    return min(
        profiles,
        key=lambda profile: (
            float(profile["aggregate"]["ms_per_row"]["mean"]),
            int(profile["batch_size"]),
        ),
    )


def load_runtimes(
    *,
    package: PosthocVerifierPackage,
    package_dir: Path,
    policy: str,
    seed: int,
    device: torch.device,
) -> tuple[dict[int, MoEInferenceRuntime], dict[int, dict[str, Any]]]:
    manifest_root = resolve_manifest_root(package_dir, package.manifest)
    seed_ids = [int(seed)] if policy == "seed" else [int(value) for value in package.seed_ids]
    runtimes: dict[int, MoEInferenceRuntime] = {}
    entries: dict[int, dict[str, Any]] = {}
    for seed_id in seed_ids:
        entry = package.seed(seed_id).manifest_entry
        checkpoint = resolve_path(entry["checkpoint"], manifest_root)
        config = resolve_path(entry["config"], manifest_root)
        data_dir = resolve_path(entry["data_dir"], manifest_root)
        runtimes[seed_id] = MoEInferenceRuntime.from_checkpoint(
            checkpoint=checkpoint,
            config_path=config,
            metadata_path=data_dir / "metadata.json",
            verifier_package=package,
            device=device,
        )
        entries[seed_id] = entry
    return runtimes, entries


def predict_with_runtimes(
    *,
    runtimes: dict[int, MoEInferenceRuntime],
    entries: dict[int, dict[str, Any]],
    rows: list[dict[str, Any]],
    policy: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    seed_outputs: dict[int, list[dict[str, Any]]] = {}
    for seed, runtime in runtimes.items():
        entry = entries[seed]
        seed_outputs[seed] = runtime.predict_rows(
            rows,
            batch_size=batch_size,
            base_threshold=float(entry["base_threshold"]),
            verifier_seed=seed,
        )
    if policy == "seed":
        return seed_outputs[next(iter(seed_outputs))]
    return combine_seed_prediction_rows(seed_outputs, policy=policy)


def measure_inference(
    *,
    runtimes: dict[int, MoEInferenceRuntime],
    entries: dict[int, dict[str, Any]],
    rows: list[dict[str, Any]],
    policy: str,
    batch_size: int,
    warmup_runs: int,
    repeats: int,
) -> dict[str, Any]:
    warmup_summaries = []
    for idx in range(max(0, int(warmup_runs))):
        warm_start = time.perf_counter()
        predictions = predict_with_runtimes(
            runtimes=runtimes,
            entries=entries,
            rows=rows,
            policy=policy,
            batch_size=batch_size,
        )
        warmup_summaries.append(
            {
                "iteration": idx,
                "elapsed_ms": (time.perf_counter() - warm_start) * 1000.0,
                **prediction_summary(predictions),
            }
        )

    measured = []
    for idx in range(max(1, int(repeats))):
        started = time.perf_counter()
        predictions = predict_with_runtimes(
            runtimes=runtimes,
            entries=entries,
            rows=rows,
            policy=policy,
            batch_size=batch_size,
        )
        elapsed = time.perf_counter() - started
        measured.append(
            {
                "iteration": idx,
                "elapsed_ms": elapsed * 1000.0,
                "ms_per_row": (elapsed * 1000.0) / max(len(predictions), 1),
                "rows_per_second": float(len(predictions)) / max(elapsed, 1e-9),
                **prediction_summary(predictions),
            }
        )

    elapsed_values = [float(row["elapsed_ms"]) for row in measured]
    ms_per_row_values = [float(row["ms_per_row"]) for row in measured]
    throughput_values = [float(row["rows_per_second"]) for row in measured]
    return {
        "batch_size": int(batch_size),
        "warmup": warmup_summaries,
        "runs": measured,
        "aggregate": {
            "elapsed_ms": metric_summary(elapsed_values),
            "ms_per_row": metric_summary(ms_per_row_values),
            "rows_per_second": metric_summary(throughput_values),
        },
    }


def memory_summary(rss: RssSampler, load_rss: int | None) -> dict[str, float | bool | None]:
    return {
        "rss_available": rss.available,
        "rss_start_mib": bytes_to_mib(rss.start),
        "rss_after_load_mib": bytes_to_mib(load_rss),
        "rss_peak_mib": bytes_to_mib(rss.peak),
        "rss_end_mib": bytes_to_mib(rss.end),
        "rss_load_delta_mib": bytes_to_mib(load_rss - rss.start) if load_rss is not None and rss.start is not None else None,
        "rss_peak_delta_mib": bytes_to_mib(rss.peak - rss.start) if rss.peak is not None and rss.start is not None else None,
    }


def run_benchmark(
    *,
    package_dir: Path,
    input_path: Path,
    max_samples: int,
    batch_size: int,
    warmup_runs: int,
    repeats: int,
    policy: str | None,
    seed: int,
    device_name: str,
    verify_hashes: bool,
) -> dict[str, Any]:
    package_dir = package_dir.resolve()
    rows = read_jsonl(input_path, limit=max_samples)
    device = select_device(device_name)
    gc.collect()
    with RssSampler() as rss:
        load_start = time.perf_counter()
        package = PosthocVerifierPackage.load(package_dir, verify_hashes=verify_hashes)
        active_policy = policy or package.manifest.get("release", {}).get("default_policy") or DEFAULT_POLICY
        runtimes, entries = load_runtimes(
            package=package,
            package_dir=package_dir,
            policy=str(active_policy),
            seed=seed,
            device=device,
        )
        load_elapsed = time.perf_counter() - load_start
        load_rss = rss.current()

        measurement = measure_inference(
            runtimes=runtimes,
            entries=entries,
            rows=rows,
            policy=str(active_policy),
            batch_size=batch_size,
            warmup_runs=warmup_runs,
            repeats=repeats,
        )

    memory = memory_summary(rss, load_rss)
    return {
        "schema_version": "pyrrho_moe_release_benchmark_v1",
        "package_dir": str(package_dir),
        "input": str(input_path),
        "rows": len(rows),
        "policy": str(active_policy),
        "seed_ids": sorted(runtimes),
        "device": str(device),
        "batch_size": int(batch_size),
        "warmup_runs": int(warmup_runs),
        "repeats": int(repeats),
        "load": {
            "elapsed_ms": load_elapsed * 1000.0,
        },
        "memory": memory,
        "warmup": measurement["warmup"],
        "runs": measurement["runs"],
        "aggregate": measurement["aggregate"],
    }


def run_batch_size_profile(
    *,
    package_dir: Path,
    input_path: Path,
    max_samples: int,
    batch_sizes: list[int],
    warmup_runs: int,
    repeats: int,
    policy: str | None,
    seed: int,
    device_name: str,
    verify_hashes: bool,
) -> dict[str, Any]:
    package_dir = package_dir.resolve()
    rows = read_jsonl(input_path, limit=max_samples)
    device = select_device(device_name)
    normalized_batch_sizes = normalize_batch_sizes(1, batch_sizes)
    gc.collect()
    with RssSampler() as rss:
        load_start = time.perf_counter()
        package = PosthocVerifierPackage.load(package_dir, verify_hashes=verify_hashes)
        active_policy = policy or package.manifest.get("release", {}).get("default_policy") or DEFAULT_POLICY
        runtimes, entries = load_runtimes(
            package=package,
            package_dir=package_dir,
            policy=str(active_policy),
            seed=seed,
            device=device,
        )
        load_elapsed = time.perf_counter() - load_start
        load_rss = rss.current()

        profiles = [
            measure_inference(
                runtimes=runtimes,
                entries=entries,
                rows=rows,
                policy=str(active_policy),
                batch_size=batch_size,
                warmup_runs=warmup_runs,
                repeats=repeats,
            )
            for batch_size in normalized_batch_sizes
        ]

    best = fastest_profile(profiles)
    return {
        "schema_version": "pyrrho_moe_release_batch_profile_v1",
        "package_dir": str(package_dir),
        "input": str(input_path),
        "rows": len(rows),
        "policy": str(active_policy),
        "seed_ids": sorted(runtimes),
        "device": str(device),
        "batch_sizes": normalized_batch_sizes,
        "warmup_runs": int(warmup_runs),
        "repeats": int(repeats),
        "load": {
            "elapsed_ms": load_elapsed * 1000.0,
        },
        "memory": memory_summary(rss, load_rss),
        "profiles": profiles,
        "fastest": {
            "batch_size": int(best["batch_size"]) if best is not None else None,
            "ms_per_row": float(best["aggregate"]["ms_per_row"]["mean"]) if best is not None else None,
            "rows_per_second": float(best["aggregate"]["rows_per_second"]["mean"]) if best is not None else None,
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.batch_sizes:
        report = run_batch_size_profile(
            package_dir=args.package_dir,
            input_path=args.input,
            max_samples=args.max_samples,
            batch_sizes=args.batch_sizes,
            warmup_runs=args.warmup_runs,
            repeats=args.repeats,
            policy=args.policy,
            seed=args.seed,
            device_name=args.device,
            verify_hashes=not args.skip_hash_check,
        )
    else:
        report = run_benchmark(
            package_dir=args.package_dir,
            input_path=args.input,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            warmup_runs=args.warmup_runs,
            repeats=args.repeats,
            policy=args.policy,
            seed=args.seed,
            device_name=args.device,
            verify_hashes=not args.skip_hash_check,
        )
    if args.output is not None:
        write_json(args.output, report)
        print(f"Wrote benchmark report: {args.output}")
    mem = report["memory"]
    if report["schema_version"] == "pyrrho_moe_release_batch_profile_v1":
        print(
            "Batch profile: "
            f"rows={report['rows']} policy={report['policy']} device={report['device']} "
            f"fastest_batch={report['fastest']['batch_size']} "
            f"ms_per_row={report['fastest']['ms_per_row']:.2f} "
            f"rows_s={report['fastest']['rows_per_second']:.2f}"
        )
        for profile in report["profiles"]:
            agg = profile["aggregate"]
            print(
                "  "
                f"batch={profile['batch_size']} "
                f"ms_per_row={agg['ms_per_row']['mean']:.2f}+/-{agg['ms_per_row']['std']:.2f} "
                f"rows_s={agg['rows_per_second']['mean']:.2f}"
            )
    else:
        agg = report["aggregate"]
        print(
            "Benchmark: "
            f"rows={report['rows']} policy={report['policy']} device={report['device']} "
            f"ms_per_row={agg['ms_per_row']['mean']:.2f}+/-{agg['ms_per_row']['std']:.2f} "
            f"rows_s={agg['rows_per_second']['mean']:.2f}"
        )
    if mem["rss_peak_mib"] is not None:
        print(
            "Memory: "
            f"start={mem['rss_start_mib']:.1f} MiB "
            f"after_load={mem['rss_after_load_mib']:.1f} MiB "
            f"peak={mem['rss_peak_mib']:.1f} MiB"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
