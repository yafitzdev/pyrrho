"""Shared data helpers — label constants, encoder text format, SLM prompt template, dataset loader.

The fitz-gov→splits conversion lives in `scripts/prepare_data.py`; this module is for code
that runs at train/eval/inference time and needs the same canonical formatting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from datasets import DatasetDict, load_from_disk


LABEL2ID: dict[str, int] = {"ABSTAIN": 0, "DISPUTED": 1, "TRUSTWORTHY": 2}
ID2LABEL: dict[int, str] = {v: k for k, v in LABEL2ID.items()}
LABEL_NAMES: tuple[str, ...] = ("ABSTAIN", "DISPUTED", "TRUSTWORTHY")

LABEL2ID_4CLASS: dict[str, int] = {
    "ABSTAIN": 0,
    "DISPUTED": 1,
    "TRUSTWORTHY_HEDGED": 2,
    "TRUSTWORTHY_DIRECT": 3,
}
ID2LABEL_4CLASS: dict[int, str] = {v: k for k, v in LABEL2ID_4CLASS.items()}

QUERY_CONTRACT_LABELS: tuple[str, ...] = (
    "evidence_sufficiency",
    "structured_lookup",
    "temporal_grounding",
    "exhaustive_coverage",
    "comparison_coverage",
    "representative_overview",
)
QUERY_CONTRACT_LABEL2ID: dict[str, int] = {
    label: idx for idx, label in enumerate(QUERY_CONTRACT_LABELS)
}
QUERY_CONTRACT_ID2LABEL: dict[int, str] = {
    idx: label for label, idx in QUERY_CONTRACT_LABEL2ID.items()
}

RETRIEVAL_ACTION_LABELS: tuple[str, ...] = (
    "answer_now",
    "retrieve_more",
    "broaden_search",
    "resolve_conflict",
    "ask_clarifying_question",
    "structured_lookup",
)
RETRIEVAL_ACTION_LABEL2ID: dict[str, int] = {
    label: idx for idx, label in enumerate(RETRIEVAL_ACTION_LABELS)
}
RETRIEVAL_ACTION_ID2LABEL: dict[int, str] = {
    idx: label for label, idx in RETRIEVAL_ACTION_LABEL2ID.items()
}

GAP_TYPE_LABELS: tuple[str, ...] = (
    "none",
    "missing_specific_fact",
    "missing_timeframe",
    "missing_comparison_side",
    "missing_source_authority",
    "conflicting_values",
    "wrong_entity",
    "wrong_version_or_scope",
    "too_broad",
    "incomplete_enumeration",
    "unsupported_inference",
    "ambiguous_query",
)
GAP_TYPE_LABEL2ID: dict[str, int] = {label: idx for idx, label in enumerate(GAP_TYPE_LABELS)}
GAP_TYPE_ID2LABEL: dict[int, str] = {idx: label for label, idx in GAP_TYPE_LABEL2ID.items()}

ANSWERABILITY_SHAPE_LABELS: tuple[str, ...] = (
    "single_fact",
    "explanation",
    "list",
    "exhaustive_list",
    "comparison",
    "timeline",
    "calculation",
    "yes_no",
    "summary",
    "citation_required",
    "exact_lookup",
)
ANSWERABILITY_SHAPE_LABEL2ID: dict[str, int] = {
    label: idx for idx, label in enumerate(ANSWERABILITY_SHAPE_LABELS)
}
ANSWERABILITY_SHAPE_ID2LABEL: dict[int, str] = {
    idx: label for label, idx in ANSWERABILITY_SHAPE_LABEL2ID.items()
}

RETRIEVAL_MODALITY_LABELS: tuple[str, ...] = (
    "unstructured_text",
    "structured_table",
    "code",
    "configuration",
    "log_trace",
    "pdf_layout",
    "mixed",
)
RETRIEVAL_MODALITY_LABEL2ID: dict[str, int] = {
    label: idx for idx, label in enumerate(RETRIEVAL_MODALITY_LABELS)
}
RETRIEVAL_MODALITY_ID2LABEL: dict[int, str] = {
    idx: label for label, idx in RETRIEVAL_MODALITY_LABEL2ID.items()
}

PYRRHO_G3_1_SCALAR_FIELDS: tuple[str, ...] = (
    "evidence_sufficiency",
    "query_evidence_alignment",
    "answer_coverage",
    "conflict_density",
    "retrieval_retry_value",
    "false_trustworthy_risk",
)


def collapse_4_to_3(label_id_4: int) -> int:
    """Map a 4-class label id to the 3-class space (HEDGED+DIRECT collapse to TRUSTWORTHY)."""
    return 2 if label_id_4 >= 2 else label_id_4


SLM_SYSTEM_PROMPT = (
    "You are a RAG governance classifier. Given a user question and retrieved sources, "
    "decide whether the sources support a confident answer.\n\n"
    "Output exactly one token: ABSTAIN, DISPUTED, or TRUSTWORTHY.\n"
    "- ABSTAIN: sources do not contain enough information to answer.\n"
    "- DISPUTED: sources contradict each other on the answer.\n"
    "- TRUSTWORTHY: sources consistently and sufficiently support an answer."
)


def build_encoder_text(query: str, contexts: Iterable[str]) -> str:
    """Encoder input format. Matches `scripts/prepare_data.py:build_text` exactly."""
    parts = [f"Question: {(query or '').strip()}", "", "Sources:"]
    for i, ctx in enumerate(contexts or [], start=1):
        parts.append(f"[{i}] {str(ctx).strip()}")
    return "\n".join(parts)


def build_query_contract_text(query: str) -> str:
    """Query-only input format for pre-retrieval query-contract classification."""
    return f"Question: {(query or '').strip()}"


def format_slm_messages(query: str, contexts: Iterable[str]) -> list[dict[str, str]]:
    """Build the chat-template message list for SLM training/inference. Single-turn."""
    numbered = "\n".join(f"[{i}] {str(c).strip()}" for i, c in enumerate(contexts or [], start=1))
    user_content = f"Question: {(query or '').strip()}\n\nSources:\n{numbered}"
    return [
        {"role": "system", "content": SLM_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def load_processed(data_dir: str | Path) -> DatasetDict:
    """Load the DatasetDict written by `prepare_data.py`.

    Required splits are `train` and `eval`. V7+ processed datasets also include
    `test`, which callers should use for final held-out reporting after
    checkpoint/threshold selection on `eval`. `tier0_sanity` is optional because
    the published V7 split contract already includes the legacy tier0 rows.
    """
    hf_dir = Path(data_dir) / "hf_dataset"
    if not hf_dir.exists():
        raise FileNotFoundError(
            f"Processed HF dataset not found at {hf_dir}. "
            f"Run: python scripts/prepare_data.py --output {data_dir}"
        )
    ds = load_from_disk(str(hf_dir))
    expected = {"train", "eval"}
    missing = expected - set(ds.keys())
    if missing:
        raise ValueError(f"Loaded dataset is missing splits: {sorted(missing)}")
    return ds
