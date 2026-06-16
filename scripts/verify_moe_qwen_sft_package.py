"""Verify the local pyrrho Qwen generative MoE MVP package.

The package contract is intentionally different from the earlier posthoc MoE
release package: it contains a PEFT adapter and uses label-score thresholding as
the authoritative decision source.

Run from project root:
    python scripts/verify_moe_qwen_sft_package.py \
      --package-dir models/pyrrho-MoE-g3-mvp \
      --input data/moe_v8/test.jsonl
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_PACKAGE_DIR = Path("models/pyrrho-MoE-g3-mvp")
DEFAULT_INPUT = Path("data/moe_v8/test.jsonl")
DEFAULT_ACCURACY_GATE = 0.787
DEFAULT_FALSE_TRUSTWORTHY_GATE = 0.057

ADAPTER_REQUIRED_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "chat_template.jinja",
    "tokenizer_config.json",
    "tokenizer.json",
)

REPORT_PATH_KEYS = (
    "report",
    "threshold_sweep",
    "threshold_sweep_and_hf_agreement",
    "skip_generation_smoke",
    "full_generation_smoke",
    "full_test_report",
    "random512_report",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--seed-pack", type=Path, default=None, help="Override manifest base seed-pack path")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="JSONL rows for the package smoke")
    parser.add_argument("--output", type=Path, default=None, help="Write verification report JSON")
    parser.add_argument("--predictions-output", type=Path, default=None, help="Write smoke predictions JSONL")
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=None, help="Override manifest runtime threshold")
    parser.add_argument("--skip-smoke", action="store_true", help="Only verify package files and metrics")
    parser.add_argument("--accuracy-gate", type=float, default=DEFAULT_ACCURACY_GATE)
    parser.add_argument("--false-trustworthy-gate", type=float, default=DEFAULT_FALSE_TRUSTWORTHY_GATE)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def resolve_path(raw_path: str | Path, *, base: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def resolve_manifest_path(raw_path: str | Path, *, package_dir: Path, project_root: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    package_candidate = (package_dir / path).resolve()
    if package_candidate.exists():
        return package_candidate
    return (project_root / path).resolve()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024 * 8), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_file(
    label: str,
    path: Path,
    *,
    parse_json: bool = False,
    parse_jsonl: bool = False,
    min_bytes: int = 1,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "label": label,
        "path": str(path),
        "exists": path.exists(),
        "bytes": None,
        "expected_bytes": expected_bytes,
        "parse_ok": None,
        "sha256": None,
        "sha256_ok": None,
        "ok": False,
    }
    if not path.exists():
        row["error"] = "missing"
        return row
    if not path.is_file():
        row["error"] = "not_a_file"
        return row

    size = path.stat().st_size
    row["bytes"] = int(size)
    row["size_ok"] = size == expected_bytes if expected_bytes is not None else size >= min_bytes
    try:
        if parse_json:
            read_json(path)
            row["parse_ok"] = True
        elif parse_jsonl:
            rows = 0
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    json.loads(stripped)
                    rows += 1
            row["parse_ok"] = True
            row["rows"] = rows
        else:
            row["parse_ok"] = None
    except Exception as exc:  # pragma: no cover - report detail is more useful than exception type.
        row["parse_ok"] = False
        row["error"] = f"parse_failed: {exc}"

    if expected_sha256 and row.get("parse_ok") is not False:
        actual_sha256 = file_sha256(path)
        row["sha256"] = actual_sha256
        row["sha256_ok"] = actual_sha256.lower() == expected_sha256.lower()

    parse_ok = row["parse_ok"] is not False
    sha_ok = row["sha256_ok"] is not False
    row["ok"] = bool(row["exists"]) and bool(row["size_ok"]) and parse_ok and sha_ok
    return row


def collect_report_paths(manifest: dict[str, Any]) -> list[tuple[str, str]]:
    paths: list[tuple[str, str]] = []
    metrics = manifest.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    for parent_name, parent in (("metrics", metrics), ("verification", manifest.get("verification", {}))):
        if not isinstance(parent, dict):
            continue
        for section_name, section in parent.items():
            if not isinstance(section, dict):
                continue
            for key in REPORT_PATH_KEYS:
                raw_path = section.get(key)
                if raw_path:
                    paths.append((f"{parent_name}.{section_name}.{key}", str(raw_path)))
    return paths


def verify_metric_gates(
    manifest: dict[str, Any],
    *,
    accuracy_gate: float,
    false_trustworthy_gate: float,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    metrics = manifest.get("metrics", {})
    runtime = manifest.get("runtime", {})
    runtime_threshold = runtime.get("trustworthy_threshold")

    for section_name in (
        "full_eval_selected_label_score",
        "full_test_selected_label_score",
        "gguf_q4_sequence_full_test",
    ):
        section = metrics.get(section_name, {})
        accuracy = section.get("accuracy")
        false_trustworthy = section.get("false_trustworthy_rate")
        threshold = section.get("selected_threshold")
        ok = (
            isinstance(accuracy, int | float)
            and isinstance(false_trustworthy, int | float)
            and float(accuracy) >= accuracy_gate
            and float(false_trustworthy) <= false_trustworthy_gate
        )
        threshold_ok = runtime_threshold is None or threshold is None or float(threshold) == float(runtime_threshold)
        checks.append(
            {
                "label": section_name,
                "accuracy": accuracy,
                "false_trustworthy_rate": false_trustworthy,
                "threshold": threshold,
                "accuracy_gate": accuracy_gate,
                "false_trustworthy_gate": false_trustworthy_gate,
                "threshold_matches_runtime": threshold_ok,
                "ok": bool(ok and threshold_ok),
            }
        )

    bounded = metrics.get("bounded_512_full_generation", {})
    json_parse_rate = bounded.get("json_parse_rate")
    label_parse_rate = bounded.get("label_parse_rate")
    checks.append(
        {
            "label": "bounded_512_generation_parse",
            "json_parse_rate": json_parse_rate,
            "label_parse_rate": label_parse_rate,
            "ok": json_parse_rate == 1.0 and label_parse_rate == 1.0,
        }
    )
    checks.append(
        {
            "label": "raw_generation_false_trustworthy_observed",
            "false_trustworthy_rate": bounded.get("free_generation_false_trustworthy_rate"),
            "ok": True,
            "warning": "raw_generation is audit/debug only and is not the release decision source",
        }
    )
    return checks


def summarize_package_size(package_dir: Path) -> dict[str, Any]:
    file_count = 0
    total_bytes = 0
    for path in package_dir.rglob("*"):
        if path.is_file():
            file_count += 1
            total_bytes += path.stat().st_size
    return {"files": file_count, "bytes": total_bytes}


def run_smoke(
    *,
    project_root: Path,
    package_dir: Path,
    seed_pack: Path,
    input_path: Path,
    predictions_output: Path,
    max_samples: int,
    batch_size: int,
    threshold: float,
) -> dict[str, Any]:
    script = project_root / "scripts" / "infer_moe_qwen_sft.py"
    command = [
        sys.executable,
        str(script),
        "--seed-pack",
        str(seed_pack),
        "--adapter-path",
        str(package_dir / "adapter"),
        "--input",
        str(input_path),
        "--max-samples",
        str(max_samples),
        "--threshold",
        str(threshold),
        "--skip-generation",
        "--batch-size",
        str(batch_size),
        "--output",
        str(predictions_output),
    ]

    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.perf_counter() - started
    row: dict[str, Any] = {
        "command": command,
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "stdout_tail": (completed.stdout or "")[-4000:],
        "stderr_tail": (completed.stderr or "")[-4000:],
        "output": str(predictions_output),
        "ok": completed.returncode == 0 and predictions_output.exists(),
    }
    if not row["ok"]:
        return row

    counts: Counter[str] = Counter()
    rows = 0
    parse_ok = True
    with predictions_output.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows += 1
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                parse_ok = False
                continue
            selected = payload.get("selected_output") or {}
            label = selected.get("classification") or payload.get("classification")
            counts[str(label)] += 1

    row.update(
        {
            "rows": rows,
            "parse_ok": parse_ok,
            "classification_counts": dict(sorted(counts.items())),
            "ok": bool(row["ok"] and parse_ok and rows == max_samples),
        }
    )
    return row


def verify_package(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    project_root = args.project_root.resolve()
    package_dir = resolve_path(args.package_dir, base=project_root)
    manifest_path = package_dir / "manifest.json"

    file_checks: list[dict[str, Any]] = [
        check_file("manifest", manifest_path, parse_json=True),
        check_file("README", package_dir / "README.md"),
    ]
    manifest = read_json(manifest_path)

    adapter_dir = package_dir / manifest.get("adapter", {}).get("path", "adapter")
    for filename in ADAPTER_REQUIRED_FILES:
        file_checks.append(check_file(f"adapter.{filename}", adapter_dir / filename))

    metadata_path = package_dir / "metadata" / "metadata.json"
    file_checks.append(check_file("metadata.metadata_json", metadata_path, parse_json=True))

    for label, relative in collect_report_paths(manifest):
        path = package_dir / relative
        suffix = path.suffix.lower()
        file_checks.append(
            check_file(
                label,
                path,
                parse_json=suffix == ".json",
                parse_jsonl=suffix == ".jsonl",
            )
        )

    gguf_runtime = manifest.get("runtime", {}).get("gguf_low_memory", {})
    if isinstance(gguf_runtime, dict):
        model_path = gguf_runtime.get("model")
        if model_path:
            file_checks.append(
                check_file(
                    "runtime.gguf_low_memory.model",
                    resolve_manifest_path(model_path, package_dir=package_dir, project_root=project_root),
                    expected_bytes=gguf_runtime.get("model_bytes"),
                    expected_sha256=gguf_runtime.get("model_sha256"),
                )
            )
        patch_path = gguf_runtime.get("llama_cpp_patch")
        if patch_path:
            file_checks.append(
                check_file(
                    "runtime.gguf_low_memory.llama_cpp_patch",
                    resolve_manifest_path(patch_path, package_dir=package_dir, project_root=project_root),
                    expected_bytes=gguf_runtime.get("llama_cpp_patch_bytes"),
                    expected_sha256=gguf_runtime.get("llama_cpp_patch_sha256"),
                )
            )

    smoke_report = package_dir / "reports" / "package_inference_skipgen_smoke.jsonl"
    if smoke_report.exists():
        file_checks.append(check_file("reports.package_inference_skipgen_smoke", smoke_report, parse_jsonl=True))

    base_seed_pack = args.seed_pack or Path(str(manifest.get("base_seed_pack", {}).get("path", "")))
    seed_pack = resolve_path(base_seed_pack, base=project_root)
    seed_pack_check = {
        "label": "base_seed_pack",
        "path": str(seed_pack),
        "exists": seed_pack.exists(),
        "is_dir": seed_pack.is_dir(),
        "ok": seed_pack.exists() and seed_pack.is_dir(),
    }

    metric_checks = verify_metric_gates(
        manifest,
        accuracy_gate=args.accuracy_gate,
        false_trustworthy_gate=args.false_trustworthy_gate,
    )

    threshold = args.threshold
    if threshold is None:
        threshold = float(manifest.get("runtime", {}).get("trustworthy_threshold", 0.5))

    predictions_output = args.predictions_output
    if predictions_output is None:
        predictions_output = package_dir / "reports" / "package_verify_skipgen_smoke.jsonl"
    predictions_output = resolve_path(predictions_output, base=project_root)

    smoke: dict[str, Any] | None = None
    if not args.skip_smoke:
        smoke = run_smoke(
            project_root=project_root,
            package_dir=package_dir,
            seed_pack=seed_pack,
            input_path=resolve_path(args.input, base=project_root),
            predictions_output=predictions_output,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            threshold=float(threshold),
        )

    ok = (
        all(bool(check["ok"]) for check in file_checks)
        and bool(seed_pack_check["ok"])
        and all(bool(check["ok"]) for check in metric_checks)
        and (smoke is None or bool(smoke["ok"]))
    )
    elapsed = time.perf_counter() - started
    return {
        "schema_version": "pyrrho_moe_qwen_sft_package_verify_v1",
        "package_dir": str(package_dir),
        "project_root": str(project_root),
        "name": manifest.get("name"),
        "status": manifest.get("status"),
        "threshold": threshold,
        "package_size": summarize_package_size(package_dir),
        "file_checks": file_checks,
        "seed_pack_check": seed_pack_check,
        "metric_checks": metric_checks,
        "smoke": smoke,
        "elapsed_seconds": elapsed,
        "ok": ok,
    }


def main() -> None:
    args = parse_args()
    report = verify_package(args)
    output = args.output
    if output is None:
        output = resolve_path(args.package_dir, base=args.project_root.resolve()) / "release_verify_report.json"
    output = resolve_path(output, base=args.project_root.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(3):
        with output.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        final_package_size = summarize_package_size(Path(str(report["package_dir"])))
        if final_package_size == report["package_size"]:
            break
        report["package_size"] = final_package_size

    status = "PASS" if report["ok"] else "FAIL"
    print(f"{status} wrote {output}")
    print(
        "package files={files} bytes={bytes} smoke={smoke}".format(
            files=report["package_size"]["files"],
            bytes=report["package_size"]["bytes"],
            smoke="skipped" if report["smoke"] is None else report["smoke"]["ok"],
        )
    )
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
