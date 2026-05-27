from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "package_moe_posthoc_verifier.py"
SPEC = importlib.util.spec_from_file_location("package_moe_posthoc_verifier", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
package_moe_posthoc_verifier = importlib.util.module_from_spec(SPEC)
sys.modules["package_moe_posthoc_verifier"] = package_moe_posthoc_verifier
SPEC.loader.exec_module(package_moe_posthoc_verifier)

PACKAGE_SCHEMA_VERSION = package_moe_posthoc_verifier.PACKAGE_SCHEMA_VERSION
create_package = package_moe_posthoc_verifier.create_package
feature_schema_from_config = package_moe_posthoc_verifier.feature_schema_from_config


def _metrics(accuracy: float, ft: float, recall_t: float) -> dict:
    return {
        "accuracy": accuracy,
        "false_trustworthy_rate": ft,
        "recall_trustworthy": recall_t,
    }


def _report(root: Path) -> dict:
    return {
        "checkpoint": "checkpoint.pt",
        "config": "config.yaml",
        "data_dir": str(root / "data"),
        "base_threshold": 0.34,
        "selected_threshold": 0.73,
        "selection_reason": "target_ft_and_accuracy_floor",
        "eval": {
            "baseline": {"governance": _metrics(0.89, 0.038, 0.84)},
            "guarded": {"governance": _metrics(0.88, 0.027, 0.81)},
            "rejected_candidate_trustworthy": 4,
        },
        "test": {
            "baseline": {"governance": _metrics(0.90, 0.032, 0.83)},
            "guarded": {"governance": _metrics(0.901, 0.021, 0.815)},
            "rejected_candidate_trustworthy": 6,
        },
    }


def test_feature_schema_width_matches_training_feature_builder() -> None:
    schema = feature_schema_from_config(
        {
            "num_labels": 3,
            "num_routes": 2,
            "num_taxonomy_patterns": 4,
            "num_scalar_targets": 5,
        }
    )

    assert schema["total_width"] == 35
    assert [block["name"] for block in schema["blocks"]][-2:] == [
        "route_pred_one_hot",
        "taxonomy_pred_one_hot",
    ]


def test_create_package_copies_verifier_and_writes_manifest(tmp_path: Path) -> None:
    checkpoint = {
        "model_kind": "support_aggregating_token",
        "config": {
            "num_labels": 3,
            "num_routes": 2,
            "num_taxonomy_patterns": 4,
            "num_scalar_targets": 5,
        },
    }
    torch.save(checkpoint, tmp_path / "checkpoint.pt")
    (tmp_path / "config.yaml").write_text("stage0: {}\n", encoding="utf-8")

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "verifier.joblib").write_bytes(b"dummy verifier")
    (run_dir / "verifier_report.md").write_text("# report\n", encoding="utf-8")
    report_path = run_dir / "verifier_report.json"
    report_path.write_text(json.dumps(_report(tmp_path)), encoding="utf-8")

    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "stage": "stage0_7_posthoc_verifier_ft028",
                "base_stage": "stage0_7_support_aggregation",
                "verifier_kind": "hgb",
                "target_ft": 0.028,
                "max_accuracy_drop": 0.015,
                "seeds": [42],
                "runs": [
                    {
                        "seed": 42,
                        "path": str(report_path.relative_to(tmp_path)),
                    }
                ],
                "mean_std": {
                    "test_accuracy_guarded": {"mean": 0.901, "std": 0.0},
                    "test_false_trustworthy_guarded": {"mean": 0.021, "std": 0.0},
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = create_package(
        summary_path=summary_path,
        output_dir=tmp_path / "package",
        root=tmp_path,
        copy_predictions=False,
        hash_checkpoints=False,
    )

    assert manifest["schema_version"] == PACKAGE_SCHEMA_VERSION
    assert manifest["feature_schema"]["total_width"] == 35
    assert manifest["seeds"][0]["seed"] == 42
    assert manifest["seeds"][0]["verifier_path"] == "seeds/seed_42/verifier.joblib"
    assert "sha256" in manifest["seeds"][0]["artifacts"]["verifier"]
    assert (tmp_path / "package" / "seeds" / "seed_42" / "verifier.joblib").exists()
    assert (tmp_path / "package" / "manifest.json").exists()
