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


def format_slm_messages(query: str, contexts: Iterable[str]) -> list[dict[str, str]]:
    """Build the chat-template message list for SLM training/inference. Single-turn."""
    numbered = "\n".join(f"[{i}] {str(c).strip()}" for i, c in enumerate(contexts or [], start=1))
    user_content = f"Question: {(query or '').strip()}\n\nSources:\n{numbered}"
    return [
        {"role": "system", "content": SLM_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def load_processed(data_dir: str | Path) -> DatasetDict:
    """Load the DatasetDict written by `prepare_data.py`. Expects splits: train / eval / tier0_sanity."""
    hf_dir = Path(data_dir) / "hf_dataset"
    if not hf_dir.exists():
        raise FileNotFoundError(
            f"Processed HF dataset not found at {hf_dir}. "
            f"Run: python scripts/prepare_data.py --output {data_dir}"
        )
    ds = load_from_disk(str(hf_dir))
    expected = {"train", "eval", "tier0_sanity"}
    missing = expected - set(ds.keys())
    if missing:
        raise ValueError(f"Loaded dataset is missing splits: {sorted(missing)}")
    return ds
