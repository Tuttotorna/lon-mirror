"""
Minimal OMNIA validator demo.

Goal:
- Build simple multi-base signature vectors
- Compute structural metrics (TruthΩ, Δ, κ, ε)
- Build ICE envelope (confidence / impossibility / flags)
- Print a stable JSON report

This example is intentionally simple:
it validates the pipeline, not the research.
"""

from __future__ import annotations
import json
from typing import Dict, List

from omnia import compute_metrics, build_ice


# -----------------------------
# Minimal multi-base signature
# -----------------------------
# This is a placeholder signature extractor.
# It maps a text into numeric features, then re-encodes those features across bases.
#
# In real OMNIA, signatures come from richer lenses (BASE/TIME/CAUSA/TOKEN/LCR),
# but the metrics/envelope pipeline is the same.

BASES = [2, 3, 4, 5, 7, 8, 10, 12, 16]


def _to_base_digits(n: int, base: int) -> List[int]:
    if n == 0:
        return [0]
    digits = []
    x = abs(n)
    while x > 0:
        digits.append(x % base)
        x //= base
    return list(reversed(digits))


def simple_text_signatures(text: str) -> Dict[int, List[float]]:
    """
    Produce comparable vectors across bases.
    Vector layout (fixed length = 6):
      [len, sum_char, sum_words, digit_sum(base), digit_entropy(base), last_digit(base)]
    """
    s = text.strip()
    length = len(s)
    words = s.split()
    n_words = len(words)

    # simple numeric projection of text
    sum_char = sum(ord(c) for c in s) if s else 0
    sum_words = sum(len(w) for w in words) if words else 0

    signatures: Dict[int, List[float]] = {}

    for b in BASES:
        digits = _to_base_digits(sum_char + 31 * n_words + 7 * length, b)
        digit_sum = sum(digits)
        # entropy-like measure (very rough)
        freq = {}
        for d in digits:
            freq[d] = freq.get(d, 0) + 1
        total = len(digits)
        ent = 0.0
        for c in freq.values():
            p = c / total
            ent -= p * (0.0 if p == 0 else __import__("math").log(p + 1e-12))

        last_digit = digits[-1] if digits else 0

        signatures[b] = [
            float(length),
            float(sum_char),
            float(sum_words),
            float(digit_sum),
            float(ent),
            float(last_digit),
        ]

    return signatures


def main() -> None:
    sample = "Socrates is a man. All men are mortal. Therefore Socrates is mortal."

    sigs = simple_text_signatures(sample)
    m = compute_metrics(sigs)
    ice = build_ice(m)

    report = {
        "input": sample,
        "bases_used": BASES,
        "metrics": {
            "truth_omega": m.truth_omega,
            "delta_coherence": m.delta_coherence,
            "kappa_alignment": m.kappa_alignment,
            "epsilon_drift": m.epsilon_drift,
        },
        "ice": ice.to_dict(),
    }

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()