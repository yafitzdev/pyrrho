from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from pyrrho.moe.posthoc_verifier import (
    PACKAGE_SCHEMA_VERSION,
    PosthocVerifierPackage,
    build_posthoc_features,
    feature_schema_from_config,
    sha256_file,
)


class FixedVerifier:
    def __init__(self, scores: list[float]) -> None:
        self.scores = np.asarray(scores, dtype=np.float32)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if features.shape[0] != self.scores.shape[0]:
            raise ValueError("unexpected feature row count")
        return np.column_stack([1.0 - self.scores, self.scores])


def _write_package(tmp_path: Path, *, tamper_hash: bool = False) -> Path:
    package_dir = tmp_path / "package"
    seed_dir = package_dir / "seeds" / "seed_42"
    seed_dir.mkdir(parents=True)
    verifier_path = seed_dir / "verifier.joblib"
    report_path = seed_dir / "verifier_report.json"
    joblib.dump(FixedVerifier([0.9, 0.2, 0.1]), verifier_path)
    report_path.write_text("{}", encoding="utf-8")
    verifier_hash = sha256_file(verifier_path)
    if tamper_hash:
        verifier_hash = "0" * 64
    manifest = {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "feature_schema": feature_schema_from_config(
            {
                "num_labels": 3,
                "num_routes": 2,
                "num_taxonomy_patterns": 4,
                "num_scalar_targets": 5,
            }
        ),
        "seeds": [
            {
                "seed": 42,
                "base_threshold": 0.34,
                "selected_threshold": 0.5,
                "verifier_path": "seeds/seed_42/verifier.joblib",
                "report_path": "seeds/seed_42/verifier_report.json",
                "artifacts": {
                    "verifier": {
                        "bytes": verifier_path.stat().st_size,
                        "sha256": verifier_hash,
                    },
                    "report": {
                        "bytes": report_path.stat().st_size,
                        "sha256": sha256_file(report_path),
                    },
                },
            }
        ],
    }
    (package_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return package_dir


def _logits() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    governance = np.asarray(
        [
            [0.1, 0.2, 2.5],
            [0.1, 2.0, 2.1],
            [3.0, 0.2, 0.1],
        ],
        dtype=np.float32,
    )
    route = np.asarray([[2.0, 0.1], [0.2, 1.5], [1.2, 0.0]], dtype=np.float32)
    taxonomy = np.asarray(
        [
            [1.0, 0.1, 0.2, 0.3],
            [0.1, 1.0, 0.2, 0.3],
            [0.1, 0.2, 1.0, 0.3],
        ],
        dtype=np.float32,
    )
    scalars = np.zeros((3, 5), dtype=np.float32)
    return governance, route, taxonomy, scalars


def test_posthoc_verifier_package_applies_packaged_demotion_policy(tmp_path: Path) -> None:
    package = PosthocVerifierPackage.load(_write_package(tmp_path))
    governance, route, taxonomy, scalars = _logits()

    result = package.apply_logits(
        seed=42,
        governance_logits=governance,
        route_logits=route,
        taxonomy_logits=taxonomy,
        scalar_preds=scalars,
    )

    assert package.seed_ids == (42,)
    assert package.feature_width == 35
    assert result.base_predictions.tolist() == [2, 2, 0]
    assert result.guarded_predictions.tolist() == [2, 1, 0]
    assert result.rejected_mask.tolist() == [False, True, False]
    assert result.rejected_count == 1


def test_posthoc_feature_builder_matches_manifest_width(tmp_path: Path) -> None:
    package = PosthocVerifierPackage.load(_write_package(tmp_path))
    governance, route, taxonomy, scalars = _logits()
    features = build_posthoc_features(
        governance_logits=governance,
        route_logits=route,
        taxonomy_logits=taxonomy,
        scalar_preds=scalars,
    )

    assert features.shape == (3, package.feature_width)


def test_posthoc_verifier_package_rejects_hash_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sha256 mismatch"):
        PosthocVerifierPackage.load(_write_package(tmp_path, tamper_hash=True))
