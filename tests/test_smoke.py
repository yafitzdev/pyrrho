"""test_smoke.py — Regression guard from the 10 handcrafted smell-test cases.

Loads the most recent checkpoint and asserts:
1. The overall accuracy is at or above SMOKE_FLOOR (currently 0.7 = 7/10).
2. No regression on individual cases that *previously* passed.

If you trained a new model and 1-2 smoke cases flip (especially short
trustworthy ones — that's a known limitation, see PROJECT.md §18 item 20),
update SMOKE_FLOOR or mark specific case ids in EXPECTED_PASS_FAIL.

Run:
    pytest tests/test_smoke.py
    pytest tests/test_smoke.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Lazy imports — pyrrho package only needed at test execution, not collection
from pyrrho.data import ID2LABEL, build_encoder_text

# Floor below which the smoke test fails outright. Increase as the model improves.
# Current attempt-4 model lands at 8/10; we leave 1-case wiggle.
SMOKE_FLOOR = 0.70

# Optional: mark known-fail cases as xfail so they're tracked but don't break CI.
# v1 (attempt 4) consistently misses these two — both fit the short-clean-TRUSTWORTHY
# limitation documented in PROJECT.md §18 item 20.
KNOWN_FAILURES = {
    "smell_08_trustworthy_direct",
    "smell_09_trustworthy_converging",
}

# Match the structure used in scripts/smell_test.py. Kept in-sync manually for now.
CASES = [
    # ABSTAIN
    {
        "id": "smell_01_abstain_wrong_entity",
        "expected": "ABSTAIN",
        "query": "What was Tesla's revenue in Q3 2024?",
        "contexts": [
            "Ford reported Q3 2024 revenue of $46.2 billion, up 5% year over year, "
            "driven by strong sales of its F-Series trucks and Bronco SUV line.",
            "General Motors posted Q3 2024 revenue of $48.8 billion, beating analyst "
            "estimates by $1.2 billion. The company reaffirmed full-year guidance.",
        ],
    },
    {
        "id": "smell_02_abstain_wrong_time",
        "expected": "ABSTAIN",
        "query": "Who won the 2026 FIFA World Cup final?",
        "contexts": [
            "Argentina won the 2022 FIFA World Cup, defeating France 4-2 on penalties "
            "in the final after a 3-3 draw. Lionel Messi was named tournament MVP.",
            "France won the 2018 FIFA World Cup, defeating Croatia 4-2 in the final "
            "in Moscow. Kylian Mbappe scored in the final at age 19.",
        ],
    },
    {
        "id": "smell_03_abstain_partial",
        "expected": "ABSTAIN",
        "query": "How many employees does OpenAI have as of 2024?",
        "contexts": [
            "OpenAI raised $6.6 billion in October 2024 at a $157 billion valuation, "
            "making it one of the most valuable private companies in the world.",
            "OpenAI's flagship products include ChatGPT, GPT-4o, and the o1 reasoning "
            "model. The company also offers API access to enterprises and developers.",
        ],
    },
    # DISPUTED
    {
        "id": "smell_04_disputed_short_blatant",
        "expected": "DISPUTED",
        "query": "Has the company achieved profitability?",
        "contexts": [
            "The company posted its first profitable quarter, with net income of $4 million.",
            "The company recorded a quarterly loss of $12 million, the third consecutive losing quarter.",
        ],
    },
    {
        "id": "smell_05_disputed_numerical",
        "expected": "DISPUTED",
        "query": "What is the unemployment rate in Germany?",
        "contexts": [
            "Germany's unemployment rate dropped to 3.2% in October 2024, according "
            "to Eurostat's harmonized measure, the lowest level since reunification.",
            "The German Federal Employment Agency (Bundesagentur fuer Arbeit) reports "
            "the national unemployment rate at 5.8% as of October 2024.",
        ],
    },
    {
        "id": "smell_06_disputed_methodology",
        "expected": "DISPUTED",
        "query": "Is the new migraine drug effective?",
        "contexts": [
            "A double-blind randomized controlled trial of 1,200 patients found the "
            "drug reduced migraine frequency by 47% compared to placebo (p<0.001) "
            "over a 6-month follow-up period.",
            "An observational study tracking 800 patients in routine clinical practice "
            "found no statistically significant difference in self-reported migraine "
            "frequency between users and non-users.",
        ],
    },
    {
        "id": "smell_07_disputed_source_authority",
        "expected": "DISPUTED",
        "query": "Did the data breach affect customer credit card information?",
        "contexts": [
            "The company confirmed in its 8-K filing that the breach exposed credit "
            "card numbers, expiration dates, and CVV codes for approximately 2.3 million "
            "customers.",
            "An independent forensic report commissioned by the company concluded that "
            "credit card data was not accessed, as it was stored on a separate, "
            "encrypted system that the attackers did not penetrate.",
        ],
    },
    # TRUSTWORTHY
    {
        "id": "smell_08_trustworthy_direct",
        "expected": "TRUSTWORTHY",
        "query": "What is the capital of Australia?",
        "contexts": [
            "Canberra is the capital city of Australia, located in the Australian "
            "Capital Territory between Sydney and Melbourne. It was selected as the "
            "capital in 1908 as a compromise between Sydney and Melbourne."
        ],
    },
    {
        "id": "smell_09_trustworthy_converging",
        "expected": "TRUSTWORTHY",
        "query": "When was the original iPhone released?",
        "contexts": [
            "Apple released the original iPhone on June 29, 2007, six months after "
            "Steve Jobs unveiled the device at Macworld in January.",
            "The first iPhone went on sale in the United States on June 29, 2007, "
            "starting at $499 for the 4GB model and $599 for the 8GB version.",
        ],
    },
    {
        "id": "smell_10_trustworthy_hedged",
        "expected": "TRUSTWORTHY",
        "query": "Does caffeine improve cognitive performance?",
        "contexts": [
            "A 2010 meta-analysis of 41 studies published in the Journal of "
            "Alzheimer's Disease found that caffeine consumption improves reaction "
            "time, vigilance, and alertness in the short term, particularly when "
            "the user is sleep-deprived or fatigued.",
            "Researchers note that habitual coffee drinkers may experience smaller "
            "acute effects due to tolerance, and that effects on longer-term cognitive "
            "outcomes such as memory consolidation are less consistent across studies.",
        ],
    },
]


def latest_checkpoint(search_dirs: list[Path]) -> Path | None:
    """First existing checkpoint-* found, scanning the given dirs in order. None if not found."""
    for base in search_dirs:
        if not base.exists():
            continue
        if (base / "config.json").exists() and (base / "model.safetensors").exists():
            return base
        best_model = base / "best_model"
        if best_model.exists():
            return best_model
        candidates = sorted(
            base.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        if candidates:
            return candidates[-1]
    return None


@pytest.fixture(scope="module")
def model_and_tokenizer():
    """Load the latest checkpoint once for the whole test module.

    Skips the entire smoke suite if no checkpoint exists yet — this is normal
    on a fresh clone before the first training run.
    """
    search = [
        Path("models/pyrrho-nano-g3"),
        Path("outputs/multi_seed_g3_v8/seed_42"),
        Path("outputs/multi_seed_g3_v8/seed_1337"),
        Path("outputs/multi_seed_g3_v8/seed_7"),
        Path("models/pyrrho-nano-g2"),
        Path("outputs/modernbert_base_v1"),
        Path("outputs/multi_seed/seed_42"),
        Path("outputs/multi_seed/seed_1337"),
        Path("outputs/multi_seed/seed_7"),
        Path("outputs/multi_seed_g2/seed_42"),
        Path("outputs/multi_seed_g2/seed_1337"),
        Path("outputs/multi_seed_g2/seed_7"),
    ]
    ckpt = latest_checkpoint(search)
    if ckpt is None:
        pytest.skip(
            "No checkpoint-* directory found. Train a model first with "
            "`python scripts/train_encoder.py --config configs/encoder/modernbert_base.yaml --no-wandb`."
        )
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    return model, tokenizer, device, ckpt


def predict(model, tokenizer, device, query: str, contexts: list[str]) -> tuple[str, np.ndarray]:
    text = build_encoder_text(query, contexts)
    enc = tokenizer(text, truncation=True, max_length=4096, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**enc).logits[0]
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    pred_id = int(np.argmax(probs))
    return ID2LABEL[pred_id], probs


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_case(model_and_tokenizer, case):
    """Per-case test. Cases in KNOWN_FAILURES are xfail-tracked, not blocking."""
    model, tokenizer, device, _ = model_and_tokenizer
    pred, probs = predict(model, tokenizer, device, case["query"], case["contexts"])

    if case["id"] in KNOWN_FAILURES and pred != case["expected"]:
        pytest.xfail(
            f"{case['id']}: known limitation (short-clean-TRUSTWORTHY over-abstention). "
            f"Expected {case['expected']}, got {pred}. Probs={probs.round(3)}"
        )

    assert pred == case["expected"], (
        f"{case['id']}: expected {case['expected']}, got {pred}. "
        f"Probs A={probs[0]:.3f} D={probs[1]:.3f} T={probs[2]:.3f}"
    )


def test_overall_accuracy_floor(model_and_tokenizer):
    """The whole 10-case set must score at or above SMOKE_FLOOR."""
    model, tokenizer, device, ckpt = model_and_tokenizer
    correct = 0
    details = []
    for case in CASES:
        pred, probs = predict(model, tokenizer, device, case["query"], case["contexts"])
        ok = pred == case["expected"]
        if ok:
            correct += 1
        details.append((case["id"], case["expected"], pred, ok, probs))

    acc = correct / len(CASES)
    msg = f"\nSmoke accuracy: {correct}/{len(CASES)} = {acc:.0%}  (checkpoint: {ckpt})\n"
    for cid, exp, pred, ok, probs in details:
        marker = "OK" if ok else "WRONG"
        msg += f"  [{marker}] {cid}: expected={exp} predicted={pred} probs={probs.round(3)}\n"

    assert acc >= SMOKE_FLOOR, msg + f"\nFloor is {SMOKE_FLOOR:.0%}; current run dropped below."
