from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

from pyrrho.moe.modeling import TinyMoEConfig, TinyMoEForGovernance
from pyrrho.moe.posthoc_verifier import (
    PACKAGE_SCHEMA_VERSION,
    feature_schema_from_config,
    sha256_file,
)

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "verify_moe_release.py"
SPEC = importlib.util.spec_from_file_location("verify_moe_release", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
verify_moe_release = importlib.util.module_from_spec(SPEC)
sys.modules["verify_moe_release"] = verify_moe_release
SPEC.loader.exec_module(verify_moe_release)


class FixedVerifier:
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        scores = np.full(features.shape[0], 0.9, dtype=np.float32)
        return np.column_stack([1.0 - scores, scores])


def _write_checkpoint(path: Path) -> None:
    cfg = TinyMoEConfig(
        token_vocab_size=128,
        hidden_size=8,
        expert_hidden_size=16,
        num_routes=2,
        num_taxonomy_patterns=3,
        num_scalar_targets=1,
        dropout=0.0,
    )
    model = TinyMoEForGovernance(cfg)
    torch.save(
        {
            "model_kind": "tiny",
            "config": cfg.__dict__,
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def _write_release_package(path: Path, *, bad_checkpoint_hash: bool = False) -> Path:
    path.mkdir()
    (path / "README.md").write_text("# release\n", encoding="utf-8")
    config_dir = path / "config"
    config_dir.mkdir()
    config_path = config_dir / "config.yaml"
    config_path.write_text("stage0:\n  max_seq_length: 32\n", encoding="utf-8")
    metadata_dir = path / "metadata"
    metadata_dir.mkdir()
    (metadata_dir / "metadata.json").write_text(
        json.dumps(
            {
                "route2id": {"general": 0, "technology": 1},
                "taxonomy_pattern2id": {
                    "direct_answer": 0,
                    "evidence_absent": 1,
                    "numerical_conflict": 2,
                },
                "scalar_fields": ["false_trustworthy_risk"],
            }
        ),
        encoding="utf-8",
    )
    seed_dir = path / "seeds" / "seed_42"
    seed_dir.mkdir(parents=True)
    checkpoint_path = seed_dir / "model.pt"
    verifier_path = seed_dir / "verifier.joblib"
    report_path = seed_dir / "verifier_report.json"
    _write_checkpoint(checkpoint_path)
    joblib.dump(FixedVerifier(), verifier_path)
    report_path.write_text("{}", encoding="utf-8")
    checkpoint_hash = sha256_file(checkpoint_path)
    if bad_checkpoint_hash:
        checkpoint_hash = "0" * 64
    manifest = {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "root": ".",
        "config": "config/config.yaml",
        "data_dir": "metadata",
        "feature_schema": feature_schema_from_config(
            {
                "num_labels": 3,
                "num_routes": 2,
                "num_taxonomy_patterns": 3,
                "num_scalar_targets": 1,
            }
        ),
        "release": {"default_policy": "trustworthy_quorum_2_of_3"},
        "seeds": [
            {
                "seed": 42,
                "checkpoint": "seeds/seed_42/model.pt",
                "config": "config/config.yaml",
                "data_dir": "metadata",
                "base_threshold": 0.34,
                "selected_threshold": 0.5,
                "verifier_path": "seeds/seed_42/verifier.joblib",
                "report_path": "seeds/seed_42/verifier_report.json",
                "artifacts": {
                    "checkpoint": {
                        "bytes": checkpoint_path.stat().st_size,
                        "sha256": checkpoint_hash,
                    },
                    "config": {
                        "bytes": config_path.stat().st_size,
                        "sha256": sha256_file(config_path),
                    },
                    "verifier": {
                        "bytes": verifier_path.stat().st_size,
                        "sha256": sha256_file(verifier_path),
                    },
                    "report": {
                        "bytes": report_path.stat().st_size,
                        "sha256": sha256_file(report_path),
                    },
                },
            }
        ],
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_verify_release_structure_checks_checkpoint_hashes(tmp_path: Path) -> None:
    package_dir = _write_release_package(tmp_path / "package")

    report = verify_moe_release.verify_release_structure(package_dir)

    assert report["ok"] is True
    assert report["seed_ids"] == [42]
    assert report["feature_width"] == 28
    labels = {check["label"]: check for check in report["file_checks"]}
    assert labels["seed_42/checkpoint"]["sha256_ok"] is True
    assert labels["seed_42/config"]["sha256_ok"] is True


def test_verify_release_structure_rejects_bad_checkpoint_hash(tmp_path: Path) -> None:
    package_dir = _write_release_package(tmp_path / "package", bad_checkpoint_hash=True)

    report = verify_moe_release.verify_release_structure(package_dir)

    assert report["ok"] is False
    labels = {check["label"]: check for check in report["file_checks"]}
    assert labels["seed_42/checkpoint"]["sha256_ok"] is False
