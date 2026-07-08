"""fitz-gov-v2 label constants and formatting helpers."""

from __future__ import annotations

from collections.abc import Iterable

EVIDENCE_VERDICTS: tuple[str, ...] = ("INSUFFICIENT", "DISPUTED", "SUFFICIENT")
EVIDENCE_VERDICT2ID: dict[str, int] = {name: i for i, name in enumerate(EVIDENCE_VERDICTS)}
ID2EVIDENCE_VERDICT: dict[int, str] = {i: name for name, i in EVIDENCE_VERDICT2ID.items()}

FAILURE_MODES: tuple[str, ...] = (
    "none",
    "unresolved_conflict",
    "missing_or_incomplete_evidence",
    "wrong_scope_or_version",
    "ambiguous_request",
)
FAILURE_MODE2ID: dict[str, int] = {name: i for i, name in enumerate(FAILURE_MODES)}
ID2FAILURE_MODE: dict[int, str] = {i: name for name, i in FAILURE_MODE2ID.items()}

RETRIEVAL_INTENT_KEYS: tuple[str, ...] = (
    "needs_lookup",
    "needs_temporal_resolution",
    "needs_comparison_or_set",
    "needs_broad_coverage",
)

EVIDENCE_KIND_KEYS: tuple[str, ...] = (
    "needs_text",
    "needs_table_or_record",
    "needs_code_or_symbol",
    "needs_config_or_setting",
    "needs_log_or_run_result",
    "needs_document_layout",
)

NUM_V2_LABELS = (
    len(EVIDENCE_VERDICTS)
    + len(FAILURE_MODES)
    + len(RETRIEVAL_INTENT_KEYS)
    + len(EVIDENCE_KIND_KEYS)
)

PYRRHO_PRE_TAG = "[PYRRHO_PRE]"
PYRRHO_POST_TAG = "[PYRRHO_POST]"

V2_FULL_LABEL_MASK: tuple[float, ...] = (1.0,) * NUM_V2_LABELS
V2_PRE_LABEL_MASK: tuple[float, ...] = (0.0,) * (len(EVIDENCE_VERDICTS) + len(FAILURE_MODES)) + (
    1.0,
) * (len(RETRIEVAL_INTENT_KEYS) + len(EVIDENCE_KIND_KEYS))


def build_v2_full_text(query: str, contexts: Iterable[dict[str, str] | str]) -> str:
    """Build the post-retrieval encoder input from query and retrieved contexts."""
    parts = [PYRRHO_POST_TAG, f"Question: {(query or '').strip()}", "", "Sources:"]
    for idx, ctx in enumerate(contexts or [], start=1):
        if isinstance(ctx, dict):
            source_id = str(ctx.get("source_id") or idx)
            text = str(ctx.get("text") or "").strip()
        else:
            source_id = str(idx)
            text = str(ctx).strip()
        parts.append(f"[{source_id}] {text}")
    return "\n".join(parts)


def build_v2_query_text(query: str) -> str:
    """Build the pre-retrieval encoder input from only the user query."""
    return f"{PYRRHO_PRE_TAG}\nQuestion: {(query or '').strip()}"


def encode_v2_labels(row: dict) -> list[float]:
    """Encode all active v2 heads into one 18-dimensional label vector."""
    labels: list[float] = []

    verdict = row.get("evidence_verdict")
    if verdict not in EVIDENCE_VERDICT2ID:
        raise ValueError(f"invalid evidence_verdict: {verdict!r}")
    labels.extend(
        1.0 if i == EVIDENCE_VERDICT2ID[verdict] else 0.0 for i in range(len(EVIDENCE_VERDICTS))
    )

    failure = row.get("failure_mode")
    if failure not in FAILURE_MODE2ID:
        raise ValueError(f"invalid failure_mode: {failure!r}")
    labels.extend(1.0 if i == FAILURE_MODE2ID[failure] else 0.0 for i in range(len(FAILURE_MODES)))

    intents = row.get("retrieval_intents") or {}
    labels.extend(1.0 if bool(intents.get(key)) else 0.0 for key in RETRIEVAL_INTENT_KEYS)

    kinds = row.get("evidence_kinds") or {}
    labels.extend(1.0 if bool(kinds.get(key)) else 0.0 for key in EVIDENCE_KIND_KEYS)

    if len(labels) != NUM_V2_LABELS:
        raise AssertionError(f"expected {NUM_V2_LABELS} labels, got {len(labels)}")
    return labels


def v2_label_names() -> list[str]:
    """Human-readable output names for the 18-logit alpha head."""
    return (
        [f"evidence_verdict.{name}" for name in EVIDENCE_VERDICTS]
        + [f"failure_mode.{name}" for name in FAILURE_MODES]
        + [f"retrieval_intents.{name}" for name in RETRIEVAL_INTENT_KEYS]
        + [f"evidence_kinds.{name}" for name in EVIDENCE_KIND_KEYS]
    )
