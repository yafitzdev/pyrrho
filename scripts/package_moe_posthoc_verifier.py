"""Package and evaluate a Stage 0 MoE post-hoc verifier reranker.

The package is intentionally lightweight: it copies the verifier artifacts and
reports, records the frozen base-checkpoint/config paths, and can reload those
local checkpoints to prove the packaged reranker reproduces the validated
metrics.

Run from project root:
    python scripts/package_moe_posthoc_verifier.py create \
      --summary outputs/moe/stage0_7_posthoc_verifier_g3_3seed_ft028/summary.json \
      --output-dir outputs/moe/stage0_7_posthoc_verifier_g3_ft028_package \
      --evaluate

    python scripts/package_moe_posthoc_verifier.py evaluate \
      --package-dir outputs/moe/stage0_7_posthoc_verifier_g3_ft028_package
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch

from pyrrho.moe.data import MoEVocab
from pyrrho.moe.posthoc_verifier import (
    PACKAGE_SCHEMA_VERSION,
    PosthocVerifierPackage,
    feature_schema_from_config,
)


def _load_training_helpers():
    try:
        import train_moe_posthoc_verifier

        return train_moe_posthoc_verifier
    except ModuleNotFoundError:
        module_path = Path(__file__).resolve().parent / "train_moe_posthoc_verifier.py"
        spec = importlib.util.spec_from_file_location("train_moe_posthoc_verifier", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not load {module_path}") from None
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("train_moe_posthoc_verifier", module)
        spec.loader.exec_module(module)
        return module


_HELPERS = _load_training_helpers()
collect_split = _HELPERS.collect_split
evaluate_predictions = _HELPERS.evaluate_predictions
invert_mapping = _HELPERS.invert_mapping
load_config = _HELPERS.load_config
load_stage0_model = _HELPERS.load_stage0_model

DEFAULT_SUMMARY = Path("outputs/moe/stage0_7_posthoc_verifier_g3_3seed_ft028/summary.json")
DEFAULT_PACKAGE_DIR = Path("outputs/moe/stage0_7_posthoc_verifier_g3_ft028_package")
DEFAULT_RELEASE_DIR = Path("models/pyrrho-MoE-g3-alpha")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create", help="Create a lightweight verifier package")
    create.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    create.add_argument("--output-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    create.add_argument("--root", type=Path, default=Path.cwd(), help="Root for relative paths in reports")
    create.add_argument("--copy-predictions", action="store_true", help="Also copy test_predictions.jsonl")
    create.add_argument("--copy-checkpoints", action="store_true", help="Copy base MoE checkpoints into each seed directory")
    create.add_argument("--copy-config", action="store_true", help="Copy the base MoE YAML config into the package")
    create.add_argument("--copy-metadata", action="store_true", help="Copy data metadata.json into the package for raw inference")
    create.add_argument("--hash-checkpoints", action="store_true", help="Hash base checkpoints")
    create.add_argument("--manifest-root", default=None, help="Root value written to manifest; useful for portable release dirs")
    create.add_argument("--portable-provenance", action="store_true", help="Write provenance paths relative to --root when possible")
    create.add_argument("--release-name", default=None, help="Optional release name recorded in manifest")
    create.add_argument("--default-policy", default=None, help="Optional default ensemble policy recorded in manifest")
    create.add_argument("--policy-summary", type=Path, default=None, help="Optional policy-comparison summary to copy into reports/")
    create.add_argument("--evaluate", action="store_true", help="Evaluate package after creating it")
    create.add_argument("--eval-split", choices=["eval", "test", "both"], default="test")
    create.add_argument("--batch-size", type=int, default=None)
    create.add_argument("--max-samples", type=int, default=None)
    create.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")

    evaluate = sub.add_parser("evaluate", help="Reload and evaluate an existing package")
    evaluate.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    evaluate.add_argument("--root", type=Path, default=None, help="Override root for relative manifest paths")
    evaluate.add_argument("--data-dir", type=Path, default=None)
    evaluate.add_argument("--split", choices=["eval", "test", "both"], default="test")
    evaluate.add_argument("--batch-size", type=int, default=None)
    evaluate.add_argument("--max-samples", type=int, default=None)
    evaluate.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    evaluate.add_argument("--output", type=Path, default=None)

    return p.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_path(path: str | Path, root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (root / candidate).resolve()


def manifest_root(package_dir: Path, manifest: dict[str, Any], root_override: Path | None = None) -> Path:
    if root_override is not None:
        return root_override.resolve()
    raw = manifest.get("root")
    if raw is None:
        return package_dir.resolve()
    root = Path(str(raw))
    if root.is_absolute():
        return root.resolve()
    return (package_dir.resolve() / root).resolve()


def path_for_manifest(path: Path, root: Path, *, portable: bool) -> str:
    if not portable:
        return str(path)
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def package_relative(path: Path, package_dir: Path) -> str:
    return path.resolve().relative_to(package_dir.resolve()).as_posix()


def copy_artifact(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stat_file(path: Path, *, include_sha256: bool = False) -> dict[str, Any]:
    row: dict[str, Any] = {"bytes": int(path.stat().st_size)}
    if include_sha256:
        row["sha256"] = sha256_file(path)
    return row


def metric_snapshot(report: dict[str, Any], split: str) -> dict[str, Any]:
    baseline = report[split]["baseline"]["governance"]
    guarded = report[split]["guarded"]["governance"]
    return {
        "baseline_accuracy": float(baseline["accuracy"]),
        "baseline_false_trustworthy_rate": float(baseline["false_trustworthy_rate"]),
        "baseline_trustworthy_recall": float(baseline["recall_trustworthy"]),
        "guarded_accuracy": float(guarded["accuracy"]),
        "guarded_false_trustworthy_rate": float(guarded["false_trustworthy_rate"]),
        "guarded_trustworthy_recall": float(guarded["recall_trustworthy"]),
        "rejected_candidate_trustworthy": int(report[split]["rejected_candidate_trustworthy"]),
    }


def package_readme(manifest: dict[str, Any]) -> str:
    mean_std = manifest.get("summary", {}).get("mean_std", {})
    acc = mean_std.get("test_accuracy_guarded", {})
    ft = mean_std.get("test_false_trustworthy_guarded", {})
    release = manifest.get("release", {})
    lines = [
        f"# {release.get('name', 'Stage 0.7 Post-Hoc Verifier Package')}",
        "",
        "This directory packages the lightweight verifier/reranker artifacts for the",
        "frozen Stage 0.7 support-aggregation MoE baseline.",
        "",
        f"- Package schema: `{manifest['schema_version']}`",
        f"- Stage: `{manifest.get('stage')}`",
        f"- Base stage: `{manifest.get('base_stage')}`",
        f"- Verifier kind: `{manifest.get('verifier_kind')}`",
        f"- Target eval FT: `{manifest.get('target_ft')}`",
        f"- Feature width: `{manifest['feature_schema']['total_width']}`",
    ]
    if release.get("default_policy"):
        lines.append(f"- Default policy: `{release['default_policy']}`")
    if acc and ft:
        lines.extend(
            [
                f"- Guarded held-out accuracy: **{acc['mean'] * 100:.2f} +/- {acc['std'] * 100:.2f}%**",
                f"- Guarded held-out false-TRUSTWORTHY: **{ft['mean'] * 100:.2f} +/- {ft['std'] * 100:.2f}%**",
            ]
        )
    lines.extend(
        [
            "",
            "Reload check:",
            "",
            "```powershell",
            "python scripts/package_moe_posthoc_verifier.py evaluate --package-dir "
            f"{manifest['package_dir']}",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def create_package(
    *,
    summary_path: Path,
    output_dir: Path,
    root: Path,
    copy_predictions: bool,
    copy_checkpoints: bool = False,
    copy_config: bool = False,
    copy_metadata: bool = False,
    hash_checkpoints: bool,
    manifest_root_value: str | None = None,
    portable_provenance: bool = False,
    release_name: str | None = None,
    default_policy: str | None = None,
    policy_summary_path: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    summary_path = resolve_path(summary_path, root)
    summary = read_json(summary_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = output_dir.resolve()
    copy_artifact(summary_path, output_dir / "summary.json")

    runs = summary.get("runs", [])
    if not runs:
        raise ValueError(f"summary has no runs: {summary_path}")

    seed_entries: list[dict[str, Any]] = []
    first_model_config: dict[str, Any] | None = None
    first_model_kind: str | None = None
    first_data_dir: str | None = None
    first_config: str | None = None
    copied_config_by_source: dict[Path, Path] = {}
    copied_metadata_by_source: dict[Path, Path] = {}

    for run in runs:
        seed = int(run["seed"])
        report_path = resolve_path(str(run["path"]), root)
        report = read_json(report_path)
        run_dir = report_path.parent
        seed_dir = output_dir / "seeds" / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        verifier_src = run_dir / "verifier.joblib"
        report_md_src = run_dir / "verifier_report.md"
        predictions_src = run_dir / "test_predictions.jsonl"
        copy_artifact(verifier_src, seed_dir / "verifier.joblib")
        copy_artifact(report_path, seed_dir / "verifier_report.json")
        if report_md_src.exists():
            copy_artifact(report_md_src, seed_dir / "verifier_report.md")
        if copy_predictions and predictions_src.exists():
            copy_artifact(predictions_src, seed_dir / "test_predictions.jsonl")

        checkpoint = resolve_path(str(report["checkpoint"]), root)
        config = resolve_path(str(report["config"]), root)
        checkpoint_for_manifest = str(report["checkpoint"])
        checkpoint_for_stats = checkpoint
        if copy_checkpoints:
            checkpoint_dst = seed_dir / "model.pt"
            copy_artifact(checkpoint, checkpoint_dst)
            checkpoint_for_manifest = package_relative(checkpoint_dst, output_dir)
            checkpoint_for_stats = checkpoint_dst

        config_for_manifest = str(report["config"])
        config_for_stats = config
        if copy_config:
            config_dst = copied_config_by_source.get(config)
            if config_dst is None:
                config_dir = output_dir / "config"
                config_dst = config_dir / config.name
                copy_artifact(config, config_dst)
                copied_config_by_source[config] = config_dst
            config_for_manifest = package_relative(config_dst, output_dir)
            config_for_stats = config_dst

        data_dir = resolve_path(str(report["data_dir"]), root)
        data_dir_for_manifest = str(report["data_dir"])
        if copy_metadata:
            metadata_dst_dir = copied_metadata_by_source.get(data_dir)
            if metadata_dst_dir is None:
                metadata_src = data_dir / "metadata.json"
                metadata_dst_dir = output_dir / "metadata"
                copy_artifact(metadata_src, metadata_dst_dir / "metadata.json")
                copied_metadata_by_source[data_dir] = metadata_dst_dir
            data_dir_for_manifest = package_relative(metadata_dst_dir, output_dir)

        if first_model_config is None:
            payload = torch.load(checkpoint, map_location="cpu")
            first_model_config = dict(payload["config"])
            first_model_kind = str(payload["model_kind"])
            first_data_dir = data_dir_for_manifest
            first_config = config_for_manifest

        entry = {
            "seed": seed,
            "checkpoint": checkpoint_for_manifest,
            "config": config_for_manifest,
            "data_dir": data_dir_for_manifest,
            "base_threshold": float(report["base_threshold"]),
            "selected_threshold": float(report["selected_threshold"]),
            "selection_reason": str(report["selection_reason"]),
            "verifier_path": f"seeds/seed_{seed}/verifier.joblib",
            "report_path": f"seeds/seed_{seed}/verifier_report.json",
            "source_report": path_for_manifest(report_path, root, portable=portable_provenance),
            "source_run_dir": path_for_manifest(run_dir, root, portable=portable_provenance),
            "artifacts": {
                "verifier": stat_file(seed_dir / "verifier.joblib", include_sha256=True),
                "report": stat_file(seed_dir / "verifier_report.json", include_sha256=True),
                "checkpoint": stat_file(checkpoint_for_stats, include_sha256=hash_checkpoints),
                "config": stat_file(config_for_stats, include_sha256=True),
            },
            "eval": metric_snapshot(report, "eval"),
            "test": metric_snapshot(report, "test"),
        }
        if copy_predictions and (seed_dir / "test_predictions.jsonl").exists():
            entry["artifacts"]["test_predictions"] = stat_file(seed_dir / "test_predictions.jsonl")
            entry["test_predictions_path"] = f"seeds/seed_{seed}/test_predictions.jsonl"
        seed_entries.append(entry)

    if first_model_config is None or first_model_kind is None:
        raise ValueError("could not infer base model config from package runs")

    release_block: dict[str, Any] = {}
    if release_name:
        release_block["name"] = release_name
    if default_policy:
        release_block["default_policy"] = default_policy
    if copy_checkpoints:
        release_block["checkpoint_packaging"] = "copied_per_seed"
    if copy_metadata:
        release_block["metadata_packaging"] = "copied_metadata_json_only"
    if policy_summary_path is not None:
        policy_summary = resolve_path(policy_summary_path, root)
        reports_dir = output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        policy_summary_dst = reports_dir / "posthoc_policy_compare_summary.json"
        copy_artifact(policy_summary, policy_summary_dst)
        policy_report_src = policy_summary.with_name("report.md")
        policy_report_path = None
        if policy_report_src.exists():
            policy_report_dst = reports_dir / "posthoc_policy_compare_report.md"
            copy_artifact(policy_report_src, policy_report_dst)
            policy_report_path = package_relative(policy_report_dst, output_dir)
        release_block["policy_summary_path"] = package_relative(policy_summary_dst, output_dir)
        if policy_report_path is not None:
            release_block["policy_report_path"] = policy_report_path
        if default_policy:
            policy_payload = read_json(policy_summary)
            release_block["default_policy_metrics"] = extract_policy_metrics(
                policy_payload,
                policy_name=default_policy,
            )

    manifest = {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "package_dir": "." if manifest_root_value is not None and portable_provenance else str(output_dir),
        "root": manifest_root_value if manifest_root_value is not None else str(root),
        "source_summary": path_for_manifest(summary_path, root, portable=portable_provenance),
        "stage": summary.get("stage"),
        "base_stage": summary.get("base_stage"),
        "verifier_kind": summary.get("verifier_kind"),
        "target_ft": summary.get("target_ft"),
        "max_accuracy_drop": summary.get("max_accuracy_drop"),
        "model_kind": first_model_kind,
        "model_config": first_model_config,
        "config": first_config,
        "data_dir": first_data_dir,
        "feature_schema": feature_schema_from_config(first_model_config),
        "summary": {
            "seeds": summary.get("seeds"),
            "mean_std": summary.get("mean_std"),
            "key_test_slices": summary.get("key_test_slices"),
        },
        "seeds": seed_entries,
    }
    if release_block:
        manifest["release"] = release_block
    write_json(output_dir / "manifest.json", manifest)
    (output_dir / "README.md").write_text(package_readme(manifest), encoding="utf-8")
    return manifest


def extract_policy_metrics(policy_summary: dict[str, Any], *, policy_name: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for split_name, split_payload in (policy_summary.get("splits") or {}).items():
        for policy in split_payload.get("policies") or []:
            if policy.get("name") == policy_name:
                metrics[str(split_name)] = policy.get("metrics", {})
                break
    return metrics


def split_names(raw: str) -> list[str]:
    if raw == "both":
        return ["eval", "test"]
    return [raw]


def mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def metric_deltas(expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, float | int]:
    actual_gov = actual["guarded"]["governance"]
    actual_base = actual["baseline"]["governance"]
    return {
        "baseline_accuracy": float(actual_base["accuracy"]) - float(expected["baseline_accuracy"]),
        "baseline_false_trustworthy_rate": float(actual_base["false_trustworthy_rate"])
        - float(expected["baseline_false_trustworthy_rate"]),
        "guarded_accuracy": float(actual_gov["accuracy"]) - float(expected["guarded_accuracy"]),
        "guarded_false_trustworthy_rate": float(actual_gov["false_trustworthy_rate"])
        - float(expected["guarded_false_trustworthy_rate"]),
        "guarded_trustworthy_recall": float(actual_gov["recall_trustworthy"])
        - float(expected["guarded_trustworthy_recall"]),
        "rejected_candidate_trustworthy": int(actual["rejected_candidate_trustworthy"])
        - int(expected["rejected_candidate_trustworthy"]),
    }


def evaluate_package(
    *,
    package_dir: Path,
    root_override: Path | None,
    data_dir_override: Path | None,
    split: str,
    batch_size_override: int | None,
    max_samples: int | None,
    device_name: str = "auto",
    output: Path | None,
) -> dict[str, Any]:
    package_dir = package_dir.resolve()
    manifest = read_json(package_dir / "manifest.json")
    verifier_package = PosthocVerifierPackage.load(package_dir, verify_hashes=True)
    root = manifest_root(package_dir, manifest, root_override=root_override)
    device = select_device(device_name)
    data_dir = (data_dir_override or Path(manifest["data_dir"])).resolve()
    vocab = MoEVocab.from_metadata(data_dir / "metadata.json")
    route_names_by_id = invert_mapping(vocab.route2id)
    taxonomy_names_by_id = invert_mapping(vocab.taxonomy_pattern2id)

    rows: list[dict[str, Any]] = []
    for seed_entry in manifest["seeds"]:
        checkpoint = resolve_path(seed_entry["checkpoint"], root)
        config_path = resolve_path(seed_entry["config"], root)
        cfg = load_config(config_path)
        stage_cfg = cfg.get("stage0", {})
        data_cfg = cfg.get("data", {})
        payload = torch.load(checkpoint, map_location="cpu")
        model = load_stage0_model(payload)
        model_cfg = model.config
        batch_size = int(batch_size_override or stage_cfg.get("per_device_eval_batch_size", 128))
        max_length = int(stage_cfg.get("max_seq_length", data_cfg.get("max_seq_length", 768)))
        max_query_length = int(stage_cfg.get("max_query_length", 96))
        max_sources = int(stage_cfg.get("max_sources", 8))
        max_source_length = int(stage_cfg.get("max_source_length", 192))

        for split_name in split_names(split):
            frozen = collect_split(
                split=split_name,
                model=model,
                data_dir=data_dir,
                vocab=vocab,
                route_names_by_id=route_names_by_id,
                taxonomy_names_by_id=taxonomy_names_by_id,
                token_vocab_size=model_cfg.token_vocab_size,
                max_length=max_length,
                max_query_length=max_query_length,
                max_sources=max_sources,
                max_source_length=max_source_length,
                batch_size=batch_size,
                limit=max_samples,
                base_threshold=float(seed_entry["base_threshold"]),
                device=device,
            )
            guarded_result = verifier_package.apply_features(
                seed=int(seed_entry["seed"]),
                features=frozen.features,
                governance_logits=frozen.governance_logits,
            )
            row = {
                "seed": int(seed_entry["seed"]),
                "split": split_name,
                "rows": len(frozen.ids),
                "base_threshold": float(seed_entry["base_threshold"]),
                "selected_threshold": float(seed_entry["selected_threshold"]),
                "baseline": evaluate_predictions(frozen, frozen.base_preds),
                "guarded": evaluate_predictions(frozen, guarded_result.guarded_predictions),
                "rejected_candidate_trustworthy": guarded_result.rejected_count,
            }
            if max_samples is None and split_name in seed_entry:
                row["packaged_metric_deltas"] = metric_deltas(seed_entry[split_name], row)
            rows.append(row)

    aggregate: dict[str, Any] = {}
    for split_name in split_names(split):
        active = [row for row in rows if row["split"] == split_name]
        aggregate[split_name] = {
            "guarded_accuracy": mean_std(
                [float(row["guarded"]["governance"]["accuracy"]) for row in active]
            ),
            "guarded_false_trustworthy_rate": mean_std(
                [
                    float(row["guarded"]["governance"]["false_trustworthy_rate"])
                    for row in active
                ]
            ),
            "guarded_trustworthy_recall": mean_std(
                [float(row["guarded"]["governance"]["recall_trustworthy"]) for row in active]
            ),
            "baseline_accuracy": mean_std(
                [float(row["baseline"]["governance"]["accuracy"]) for row in active]
            ),
            "baseline_false_trustworthy_rate": mean_std(
                [
                    float(row["baseline"]["governance"]["false_trustworthy_rate"])
                    for row in active
                ]
            ),
        }

    report = {
        "schema_version": "pyrrho_moe_posthoc_verifier_package_eval_v1",
        "package_dir": str(package_dir),
        "manifest_schema_version": manifest["schema_version"],
        "device": str(device),
        "data_dir": str(data_dir),
        "split": split,
        "max_samples": max_samples,
        "aggregate": aggregate,
        "runs": rows,
    }
    output_path = output or (package_dir / "package_eval_report.json")
    write_json(output_path, report)
    markdown_path = output_path.with_suffix(".md")
    markdown_path.write_text(package_eval_markdown(report), encoding="utf-8")
    return report


def package_eval_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MoE Post-Hoc Verifier Package Eval",
        "",
        f"- Package: `{report['package_dir']}`",
        f"- Device: `{report['device']}`",
        f"- Data dir: `{report['data_dir']}`",
        f"- Split: `{report['split']}`",
        "",
        "| Split | Guarded Acc | Guarded FT | Guarded T Recall | Baseline Acc | Baseline FT |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split_name, metrics in report["aggregate"].items():
        lines.append(
            f"| {split_name} | "
            f"{metrics['guarded_accuracy']['mean'] * 100:.2f} +/- {metrics['guarded_accuracy']['std'] * 100:.2f}% | "
            f"{metrics['guarded_false_trustworthy_rate']['mean'] * 100:.2f} +/- {metrics['guarded_false_trustworthy_rate']['std'] * 100:.2f}% | "
            f"{metrics['guarded_trustworthy_recall']['mean'] * 100:.2f} +/- {metrics['guarded_trustworthy_recall']['std'] * 100:.2f}% | "
            f"{metrics['baseline_accuracy']['mean'] * 100:.2f} +/- {metrics['baseline_accuracy']['std'] * 100:.2f}% | "
            f"{metrics['baseline_false_trustworthy_rate']['mean'] * 100:.2f} +/- {metrics['baseline_false_trustworthy_rate']['std'] * 100:.2f}% |"
        )
    lines.extend(["", "## Per Seed", "", "| Seed | Split | Rows | Guarded Acc | Guarded FT | Rejected | Max Abs Delta |", "|---:|---|---:|---:|---:|---:|---:|"])
    for row in report["runs"]:
        deltas = row.get("packaged_metric_deltas", {})
        max_abs_delta = max((abs(float(v)) for v in deltas.values()), default=0.0)
        lines.append(
            f"| {row['seed']} | {row['split']} | {row['rows']} | "
            f"{row['guarded']['governance']['accuracy'] * 100:.2f}% | "
            f"{row['guarded']['governance']['false_trustworthy_rate'] * 100:.2f}% | "
            f"{row['rejected_candidate_trustworthy']} | {max_abs_delta:.6g} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.command == "create":
        manifest = create_package(
            summary_path=args.summary,
            output_dir=args.output_dir,
            root=args.root,
            copy_predictions=args.copy_predictions,
            copy_checkpoints=args.copy_checkpoints,
            copy_config=args.copy_config,
            copy_metadata=args.copy_metadata,
            hash_checkpoints=args.hash_checkpoints,
            manifest_root_value=args.manifest_root,
            portable_provenance=args.portable_provenance,
            release_name=args.release_name,
            default_policy=args.default_policy,
            policy_summary_path=args.policy_summary,
        )
        print(f"Wrote package   : {args.output_dir}")
        print(f"Wrote manifest  : {args.output_dir / 'manifest.json'}")
        print(
            "Packaged seeds  : "
            + ", ".join(str(seed["seed"]) for seed in manifest["seeds"])
        )
        if args.evaluate:
            report = evaluate_package(
                package_dir=args.output_dir,
                root_override=args.root,
                data_dir_override=None,
                split=args.eval_split,
                batch_size_override=args.batch_size,
                max_samples=args.max_samples,
                device_name=args.device,
                output=None,
            )
            for split_name, metrics in report["aggregate"].items():
                print(
                    f"{split_name:5s} guarded acc="
                    f"{metrics['guarded_accuracy']['mean']:.4f} "
                    f"ft={metrics['guarded_false_trustworthy_rate']['mean']:.4f}"
                )
        return 0

    report = evaluate_package(
        package_dir=args.package_dir,
        root_override=args.root,
        data_dir_override=args.data_dir,
        split=args.split,
        batch_size_override=args.batch_size,
        max_samples=args.max_samples,
        device_name=args.device,
        output=args.output,
    )
    print(f"Wrote eval report: {args.output or (args.package_dir / 'package_eval_report.json')}")
    for split_name, metrics in report["aggregate"].items():
        print(
            f"{split_name:5s} guarded acc={metrics['guarded_accuracy']['mean']:.4f} "
            f"ft={metrics['guarded_false_trustworthy_rate']['mean']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
