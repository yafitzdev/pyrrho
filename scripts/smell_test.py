"""smell_test.py — 10 handcrafted cases to sanity-check pyrrho's eval numbers.

If the 85.8% calibrated accuracy is real, the model should get most of these.
If it doesn't, the eval numbers are masking a problem (e.g. some leakage,
distribution overfit, or threshold artifact).

Designed to cover:
  - 3 ABSTAIN cases (wrong entity, wrong time period, partially-relevant context)
  - 4 DISPUTED cases (incl. the "short direct contradiction" known blindspot)
  - 3 TRUSTWORTHY cases (direct factual, multiple-source agreement, hedged-but-answerable)

Run from project root:
    python scripts/smell_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from pyrrho.data import ID2LABEL, build_encoder_text


CASES = [
    # --- ABSTAIN ---
    {
        "id": "smell_01_abstain_wrong_entity",
        "expected": "ABSTAIN",
        "note": "Q is about Tesla, contexts only discuss Ford/GM",
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
        "note": "Q is about 2026, contexts only go through 2022",
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
        "note": "Topically about OpenAI but never mentions headcount",
        "query": "How many employees does OpenAI have as of 2024?",
        "contexts": [
            "OpenAI raised $6.6 billion in October 2024 at a $157 billion valuation, "
            "making it one of the most valuable private companies in the world.",
            "OpenAI's flagship products include ChatGPT, GPT-4o, and the o1 reasoning "
            "model. The company also offers API access to enterprises and developers.",
        ],
    },
    # --- DISPUTED ---
    {
        "id": "smell_04_disputed_short_blatant",
        "expected": "DISPUTED",
        "note": "Known blindspot: short direct yes/no contradiction",
        "query": "Has the company achieved profitability?",
        "contexts": [
            "The company posted its first profitable quarter, with net income of $4 million.",
            "The company recorded a quarterly loss of $12 million, the third consecutive losing quarter.",
        ],
    },
    {
        "id": "smell_05_disputed_numerical",
        "expected": "DISPUTED",
        "note": "Conflicting numerical claims about the same metric",
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
        "note": "Two studies disagree, methodological conflict (tier1-like)",
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
        "note": "Two authoritative-sounding sources disagree",
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
    # --- TRUSTWORTHY ---
    {
        "id": "smell_08_trustworthy_direct",
        "expected": "TRUSTWORTHY",
        "note": "Direct factual, single clear source",
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
        "note": "Multiple sources confirm the same fact",
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
        "note": "Answerable with appropriate hedging",
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


def latest_checkpoint(output_dir: Path) -> Path:
    candidates = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-* under {output_dir}")
    return candidates[-1]


def main() -> int:
    ckpt = latest_checkpoint(Path("outputs/modernbert_base_v1"))
    print(f"Checkpoint   : {ckpt}\n")

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    correct = 0
    n = len(CASES)
    print(f"{'id':<40s} {'expected':<13s} {'predicted':<13s} {'P(A)':>6s} {'P(D)':>6s} {'P(T)':>6s} {'verdict'}")
    print("-" * 110)
    results = []
    with torch.no_grad():
        for case in CASES:
            text = build_encoder_text(case["query"], case["contexts"])
            enc = tokenizer(text, truncation=True, max_length=4096, return_tensors="pt").to(device)
            logits = model(**enc).logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            pred_id = int(np.argmax(probs))
            pred_label = ID2LABEL[pred_id]
            expected = case["expected"]
            verdict = "OK" if pred_label == expected else "WRONG"
            if pred_label == expected:
                correct += 1
            results.append((case, pred_label, probs, verdict))
            print(
                f"{case['id']:<40s} {expected:<13s} {pred_label:<13s} "
                f"{probs[0]:>6.3f} {probs[1]:>6.3f} {probs[2]:>6.3f} {verdict}"
            )

    print("-" * 110)
    print(f"Score: {correct} / {n} = {correct / n:.0%}\n")

    # Show details for any wrong cases
    wrong = [(c, p, pr, v) for c, p, pr, v in results if v == "WRONG"]
    if wrong:
        print(f"{'=' * 100}")
        print(f"WRONG CASES — full detail")
        print(f"{'=' * 100}")
        for case, pred, probs, _ in wrong:
            print(f"\n[{case['id']}]  expected={case['expected']}  predicted={pred}")
            print(f"  P(A)={probs[0]:.3f}  P(D)={probs[1]:.3f}  P(T)={probs[2]:.3f}")
            print(f"  note   : {case['note']}")
            print(f"  query  : {case['query']}")
            for i, ctx in enumerate(case["contexts"], 1):
                print(f"  ctx[{i}]: {ctx[:200]}{'...' if len(ctx) > 200 else ''}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
