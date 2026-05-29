from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_prepare_data_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_data.py"
    spec = importlib.util.spec_from_file_location("prepare_data", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_case(
    case_id: str,
    *,
    query: str,
    label: str,
    modality: str = "structured",
    difficulty: str = "easy",
) -> dict:
    return {
        "id": case_id,
        "input": {
            "query": query,
            "contexts": [{"id": "ctx_001", "text": f"Evidence for {case_id}"}],
        },
        "governance": {"classification": label},
        "taxonomy": {
            "pattern": "direct_answer",
            "cell_id": f"direct_answer__technology_computing__{difficulty}",
        },
        "routing": {"expert_fired": "technology_computing"},
        "meta": {
            "modality": modality,
            "dataset_version": "v8",
            "difficulty": difficulty,
            "category": "trustworthy_direct" if label == "TRUSTWORTHY" else label.lower(),
        },
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_candidate_pack_append_filters_selection_and_preserves_query_groups(tmp_path):
    prepare_data = load_prepare_data_module()
    pack = tmp_path / "structured_pack"
    pack.mkdir()

    cases = [
        make_case("candidate_001", query="Shared query?", label="ABSTAIN"),
        make_case("candidate_002", query="Shared query?", label="ABSTAIN"),
        make_case("candidate_003", query="Unique disputed?", label="DISPUTED", difficulty="medium"),
        make_case("candidate_004", query="Unique trustworthy?", label="TRUSTWORTHY", difficulty="hard"),
        make_case("candidate_005", query="Unused row?", label="TRUSTWORTHY"),
    ]
    write_jsonl(pack / "cases.jsonl", cases)
    (pack / "manifest.json").write_text(
        json.dumps(
            {
                "modality": "structured",
                "rows": len(cases),
                "row_shape": "sdgp_v8",
                "version": "fitz-gov-modality-candidate-0.1",
            }
        ),
        encoding="utf-8",
    )

    selection = tmp_path / "selection.jsonl"
    write_jsonl(selection, [{"case_id": case["id"]} for case in cases[:4]])

    raw_splits = {
        "train": [make_case("base_001", query="Base query?", label="ABSTAIN", modality="unstructured")],
        "eval": [],
        "test": [],
    }
    summary = prepare_data.append_candidate_packs_to_splits(
        raw_splits,
        candidate_paths=[pack],
        selection_manifest_paths=[selection],
        eval_ratio=0.25,
        test_ratio=0.25,
        seed=7,
        split_key="query",
    )

    appended_ids = {
        case["id"]
        for split_name, rows in raw_splits.items()
        if split_name in {"train", "eval", "test"}
        for case in rows
    }
    assert "candidate_005" not in appended_ids
    assert summary["total"]["rows"] == 4
    assert summary["total"]["modalities"] == {"structured": 4}

    split_by_id = {
        case["id"]: split_name
        for split_name, rows in raw_splits.items()
        for case in rows
        if case["id"].startswith("candidate_")
    }
    assert split_by_id["candidate_001"] == split_by_id["candidate_002"]


def test_normalized_candidate_record_includes_modality():
    prepare_data = load_prepare_data_module()
    record = prepare_data.normalize_case(
        make_case("candidate_001", query="Which row?", label="TRUSTWORTHY"),
        num_classes=3,
    )
    assert record["modality"] == "structured"
    assert record["source_file"].startswith("structured:v8:")
