from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import torch

from pyrrho.moe.data import MoEVocab
from pyrrho.moe.inference import (
    MoEInferenceLengths,
    MoEInferenceRuntime,
    combine_seed_prediction_rows,
    collate_inference_rows,
    prepare_inference_row,
)
from pyrrho.moe.modeling import TinyMoEConfig, TinyMoEForGovernance
from pyrrho.moe.posthoc_verifier import (
    PACKAGE_SCHEMA_VERSION,
    PosthocVerifierPackage,
    feature_schema_from_config,
    sha256_file,
)


class FixedVerifier:
    def __init__(self, scores: list[float]) -> None:
        self.scores = np.asarray(scores, dtype=np.float32)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return np.column_stack([1.0 - self.scores[: features.shape[0]], self.scores[: features.shape[0]]])


def _write_metadata(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "route2id": {"general": 0, "technology": 1},
                "taxonomy_pattern2id": {"direct_answer": 0, "evidence_absent": 1, "numerical_conflict": 2},
                "scalar_fields": ["false_trustworthy_risk"],
            }
        ),
        encoding="utf-8",
    )


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
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        model.governance_head.bias.copy_(torch.tensor([0.0, 0.2, 2.0]))
        model.taxonomy_head.bias.copy_(torch.tensor([1.0, 0.0, 0.0]))
    torch.save(
        {
            "model_kind": "tiny",
            "config": cfg.__dict__,
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def _write_package(path: Path, checkpoint: Path, metadata_dir: Path) -> None:
    seed_dir = path / "seeds" / "seed_42"
    seed_dir.mkdir(parents=True)
    verifier_path = seed_dir / "verifier.joblib"
    report_path = seed_dir / "verifier_report.json"
    joblib.dump(FixedVerifier([0.9, 0.1]), verifier_path)
    report_path.write_text("{}", encoding="utf-8")
    manifest = {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "root": str(path.parent),
        "feature_schema": feature_schema_from_config(
            {
                "num_labels": 3,
                "num_routes": 2,
                "num_taxonomy_patterns": 3,
                "num_scalar_targets": 1,
            }
        ),
        "seeds": [
            {
                "seed": 42,
                "checkpoint": str(checkpoint),
                "config": str(path.parent / "config.yaml"),
                "data_dir": str(metadata_dir),
                "base_threshold": 0.34,
                "selected_threshold": 0.5,
                "verifier_path": "seeds/seed_42/verifier.joblib",
                "report_path": "seeds/seed_42/verifier_report.json",
                "artifacts": {
                    "verifier": {"sha256": sha256_file(verifier_path)},
                    "report": {"sha256": sha256_file(report_path)},
                },
            }
        ],
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_prepare_inference_row_accepts_nested_fitz_gov_shape() -> None:
    prepared = prepare_inference_row(
        {
            "id": "case-1",
            "input": {
                "query": "What changed?",
                "contexts": [{"text": "The release changed the retry limit."}],
            },
        },
        index=0,
        token_vocab_size=128,
        lengths=MoEInferenceLengths(max_length=16, max_query_length=8, max_sources=2, max_source_length=8),
    )
    batch = collate_inference_rows([prepared])

    assert prepared.id == "case-1"
    assert prepared.query == "What changed?"
    assert prepared.contexts == ["The release changed the retry limit."]
    assert batch["source_valid_mask"].tolist() == [[1.0, 0.0]]


def test_moe_inference_runtime_applies_packaged_verifier(tmp_path: Path) -> None:
    metadata_dir = tmp_path / "data"
    metadata_dir.mkdir()
    _write_metadata(metadata_dir / "metadata.json")
    checkpoint = tmp_path / "model.pt"
    _write_checkpoint(checkpoint)
    (tmp_path / "config.yaml").write_text("stage0:\n  max_seq_length: 32\n", encoding="utf-8")
    package_dir = tmp_path / "package"
    _write_package(package_dir, checkpoint, metadata_dir)

    runtime = MoEInferenceRuntime.from_checkpoint(
        checkpoint=checkpoint,
        config_path=tmp_path / "config.yaml",
        metadata_path=metadata_dir / "metadata.json",
        verifier_package=PosthocVerifierPackage.load(package_dir),
        device=torch.device("cpu"),
    )
    predictions = runtime.predict_rows(
        [
            {"id": "row-1", "query": "Q1", "contexts": ["A"]},
            {"id": "row-2", "query": "Q2", "contexts": ["B"]},
        ],
        batch_size=2,
        verifier_seed=42,
    )

    assert [row["base_classification"] for row in predictions] == ["TRUSTWORTHY", "TRUSTWORTHY"]
    assert [row["classification"] for row in predictions] == ["TRUSTWORTHY", "DISPUTED"]
    assert [row["verifier_rejected"] for row in predictions] == [False, True]


def test_combine_seed_prediction_rows_applies_quorum_policy() -> None:
    def row(row_id: str, classification: str, rejected: bool = False) -> dict:
        return {
            "id": row_id,
            "classification": classification,
            "verifier_rejected": rejected,
            "governance_probabilities": {
                "ABSTAIN": 0.6 if classification == "ABSTAIN" else 0.05,
                "DISPUTED": 0.6 if classification == "DISPUTED" else 0.15,
                "TRUSTWORTHY": 0.8 if classification == "TRUSTWORTHY" else 0.2,
            },
        }

    combined = combine_seed_prediction_rows(
        {
            42: [row("a", "TRUSTWORTHY"), row("b", "TRUSTWORTHY")],
            7: [row("a", "TRUSTWORTHY"), row("b", "ABSTAIN", rejected=True)],
            1337: [row("a", "DISPUTED", rejected=True), row("b", "DISPUTED", rejected=True)],
        },
        policy="trustworthy_quorum_2_of_3",
    )

    assert [item["classification"] for item in combined] == ["TRUSTWORTHY", "DISPUTED"]
    assert combined[1]["seed_rejected"] == {"7": True, "42": False, "1337": True}
