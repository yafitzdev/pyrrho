from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch

from pyrrho.data import (
    ANSWERABILITY_SHAPE_LABEL2ID,
    GAP_TYPE_LABEL2ID,
    QUERY_CONTRACT_LABEL2ID,
    RETRIEVAL_ACTION_LABEL2ID,
    RETRIEVAL_MODALITY_LABEL2ID,
    RETRIEVAL_OBLIGATION_LABEL2ID,
    build_query_contract_text,
)
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


def load_prepare_g4_alpha_data_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_g4_alpha_data.py"
    spec = importlib.util.spec_from_file_location("prepare_g4_alpha_data", path)
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
            "retrieval_control": {
                "retrieval_action": {"kind": "answer_now"},
                "gap_type": {"kind": "none"},
                "answerability_shape": {"kind": "exact_lookup"},
                "preferred_retrieval_modality": {"kind": "structured_table"},
                "retrieval_obligation": {"kind": "row_key_lookup"},
                "evidence_failure_severity": {"score": 0.07},
                "labeler": "codex_subagent_v8_2",
                "row_index": 1,
            },
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
        scalar_fields=(
            "evidence_sufficiency",
            "retrieval_retry_value",
            "evidence_failure_severity",
        ),
        require_query_contract=True,
        require_retrieval_control=True,
    )
    assert row["query_text"] == build_query_contract_text(
        "Which quarterly report is most relevant?"
    )
    assert row["query_contract"] == "structured_lookup"
    assert row["query_contract_id"] == QUERY_CONTRACT_LABEL2ID["structured_lookup"]
    assert row["retrieval_action"] == "answer_now"
    assert row["retrieval_action_id"] == RETRIEVAL_ACTION_LABEL2ID["answer_now"]
    assert row["gap_type"] == "none"
    assert row["gap_type_id"] == GAP_TYPE_LABEL2ID["none"]
    assert row["answerability_shape"] == "exact_lookup"
    assert row["answerability_shape_id"] == ANSWERABILITY_SHAPE_LABEL2ID["exact_lookup"]
    assert row["retrieval_modality"] == "structured_table"
    assert row["retrieval_modality_id"] == RETRIEVAL_MODALITY_LABEL2ID["structured_table"]
    assert row["retrieval_obligation"] == "row_key_lookup"
    assert row["retrieval_obligation_id"] == RETRIEVAL_OBLIGATION_LABEL2ID["row_key_lookup"]
    assert row["scalar_targets"] == {
        "evidence_sufficiency": 0.91,
        "retrieval_retry_value": 0.12,
        "evidence_failure_severity": 0.07,
    }


def test_g5_prep_normalizes_answer_action_aliases():
    prepare_g4_alpha_data = load_prepare_g4_alpha_data_module()
    assert prepare_g4_alpha_data.normalize_retrieval_action("answer_from_evidence") == "answer_now"
    assert prepare_g4_alpha_data.normalize_retrieval_action("return_answer") == "answer_now"
    assert prepare_g4_alpha_data.normalize_retrieval_action("use_retrieved_evidence") == "answer_now"


def test_g5_prep_forces_trustworthy_no_gap_to_answer_now():
    prepare_g4_alpha_data = load_prepare_g4_alpha_data_module()
    case = make_case()
    case["routing"]["retrieval_control"]["retrieval_action"]["kind"] = "join_by_correlation_id"
    metadata = {
        "route2id": {"economics_finance": 0},
        "taxonomy_pattern2id": {"direct_answer": 0},
        "query_contract2id": {
            label: idx for idx, label in enumerate(prepare_g4_alpha_data.CANONICAL_QUERY_CONTRACT_LABELS)
        },
        "retrieval_action2id": {
            label: idx for idx, label in enumerate(prepare_g4_alpha_data.CANONICAL_RETRIEVAL_ACTION_LABELS)
        },
        "gap_type2id": {
            label: idx for idx, label in enumerate(prepare_g4_alpha_data.CANONICAL_GAP_TYPE_LABELS)
        },
        "retrieval_modality2id": {
            label: idx for idx, label in enumerate(prepare_g4_alpha_data.CANONICAL_RETRIEVAL_MODALITY_LABELS)
        },
    }
    row = prepare_g4_alpha_data.flatten_sdgp_case(
        case,
        split="train",
        metadata=metadata,
        source_kind="unit_test",
    )
    assert row["label"] == "TRUSTWORTHY"
    assert row["gap_type"] == "none"
    assert row["retrieval_action_raw"] == "join_by_correlation_id"
    assert row["retrieval_action"] == "answer_now"


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
        retrieval_action_id2label={0: "answer_now"},
        gap_type_id2label={0: "none"},
        answerability_shape_id2label={0: "exact_lookup"},
        retrieval_modality_id2label={0: "structured_table"},
        retrieval_obligation_id2label={0: "row_key_lookup"},
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
