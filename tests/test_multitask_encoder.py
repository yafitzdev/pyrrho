from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch

from pyrrho.data import QUERY_CONTRACT_LABEL2ID, build_query_contract_text
from pyrrho.multitask import PyrrhoMultiTaskConfig
from pyrrho.multitask_inference import class_prediction


def load_prepare_moe_data_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_moe_data.py"
    spec = importlib.util.spec_from_file_location("prepare_moe_data", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_case() -> dict:
    return {
        "id": "case_001",
        "version": "fitz-gov-8.0",
        "input": {
            "query": "Which quarterly report is most relevant?",
            "contexts": [{"id": "ctx_001", "text": "Q2 revenue increased by 8%."}],
        },
        "governance": {
            "classification": "TRUSTWORTHY",
            "evidence_sufficiency": 0.91,
            "query_evidence_alignment": 0.88,
            "answer_coverage": 0.86,
            "conflict_density": 0.08,
            "retrieval_retry_value": 0.12,
            "false_trustworthy_risk": 0.09,
        },
        "routing": {
            "expert_fired": "economics_finance",
            "query_contract": {"kind": "structured_lookup"},
        },
        "taxonomy": {
            "pattern": "direct_answer",
            "cell_id": "direct_answer__economics_finance__easy",
        },
        "meta": {"difficulty": "easy", "dataset_version": "v8"},
    }


def test_prepare_moe_flatten_preserves_query_contract_and_query_text():
    prepare_moe_data = load_prepare_moe_data_module()
    row = prepare_moe_data.flatten_case(
        make_case(),
        split="train",
        route2id={"economics_finance": 0},
        taxonomy2id={"direct_answer": 0},
        scalar_fields=("evidence_sufficiency", "retrieval_retry_value"),
        require_query_contract=True,
    )
    assert row["query_text"] == build_query_contract_text(
        "Which quarterly report is most relevant?"
    )
    assert row["query_contract"] == "structured_lookup"
    assert row["query_contract_id"] == QUERY_CONTRACT_LABEL2ID["structured_lookup"]
    assert row["scalar_targets"] == {
        "evidence_sufficiency": 0.91,
        "retrieval_retry_value": 0.12,
    }


def test_multitask_config_roundtrip(tmp_path):
    cfg = PyrrhoMultiTaskConfig(
        base_model="answerdotai/ModernBERT-base",
        num_governance_labels=3,
        num_query_contract_labels=6,
        num_routes=7,
        num_taxonomy_patterns=23,
        scalar_fields=("evidence_sufficiency", "retrieval_retry_value"),
        id2label={0: "ABSTAIN", 1: "DISPUTED", 2: "TRUSTWORTHY"},
        query_contract_id2label={0: "evidence_sufficiency"},
        route_id2label={0: "economics_finance"},
        taxonomy_id2label={0: "direct_answer"},
    )
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg.to_mapping()), encoding="utf-8")
    loaded = PyrrhoMultiTaskConfig.from_mapping(json.loads(path.read_text(encoding="utf-8")))
    assert loaded == cfg


def test_scalar_head_shape_without_backbone_download():
    hidden = 8
    scalar_head = torch.nn.Linear(hidden, len(("a", "b", "c")))
    out = torch.sigmoid(scalar_head(torch.zeros(2, hidden)))
    assert tuple(out.shape) == (2, 3)
    assert torch.all(out >= 0)
    assert torch.all(out <= 1)


def test_governance_threshold_fallback_metadata():
    result = class_prediction(
        [0.1, 0.2, 0.25],
        {0: "ABSTAIN", 1: "DISPUTED", 2: "TRUSTWORTHY"},
        trustworthy_threshold=0.5,
    )
    assert result["raw_label"] == "TRUSTWORTHY"
    assert result["final_label"] == "DISPUTED"
    assert result["used_threshold_fallback"] is True
    assert result["threshold"] == 0.5
    assert set(result["probabilities"]) == {"ABSTAIN", "DISPUTED", "TRUSTWORTHY"}
    assert result["runner_up_label"] == "DISPUTED"
    assert result["entropy"] > 0
