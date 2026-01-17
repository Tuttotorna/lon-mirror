from __future__ import annotations

import random
import re
from typing import Callable, Sequence, Tuple

from omnia.lenses.redundancy_invariance import RedundancyInvariance

Transform = Callable[[str], str]


# -----------------------------
# Seeded / deterministic transforms
# -----------------------------

def t_identity(s: str) -> str:
    return s


def t_whitespace_collapse(s: str) -> str:
    s = s.replace("\r\n", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def t_shuffle_words(seed: int = 3) -> Transform:
    def _t(s: str) -> str:
        rng = random.Random(seed)
        parts = re.split(r"(\W+)", s)  # keep separators
        words = [p for p in parts if p.isalnum()]
        rng.shuffle(words)
        it = iter(words)
        out = []
        for p in parts:
            out.append(next(it) if p.isalnum() else p)
        return "".join(out)

    return _t


def t_drop_vowels(s: str) -> str:
    return re.sub(r"[aeiouAEIOUàèéìòùÀÈÉÌÒÙ]", "", s)


def t_reverse(s: str) -> str:
    return s[::-1]


def t_base_repr(seed: int = 7, base: int = 7) -> Transform:
    """
    Replace decimal integers with base-N representation (+ tiny seeded perturbation).
    Representation stress-test. No semantics.
    """
    if base < 2 or base > 36:
        raise ValueError("base must be 2..36")

    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"

    def to_base(n: int) -> str:
        if n == 0:
            return "0"
        neg = n < 0
        n = abs(n)
        out = []
        while n:
            out.append(alphabet[n % base])
            n //= base
        s2 = "".join(reversed(out))
        return "-" + s2 if neg else s2

    def _t(s: str) -> str:
        rng = random.Random(seed)

        def repl(m: re.Match) -> str:
            n = int(m.group(0))
            if rng.random() < 0.1:
                n += rng.choice([-1, 1])
            return to_base(n)

        return re.sub(r"\b\d+\b", repl, s)

    return _t


# -----------------------------
# Demo
# -----------------------------

def main() -> None:
    schedule: Sequence[Tuple[float, Transform]] = [
        (0.0, t_identity),
        (1.0, t_whitespace_collapse),
        (2.0, t_shuffle_words(seed=3)),
        (3.0, t_drop_vowels),
        (4.0, t_reverse),
        (5.0, t_base_repr(seed=7, base=7)),
    ]

    lens = RedundancyInvariance(schedule=schedule, omega_floor=0.15, late_fraction=0.7)

    # Two contrasting inputs: redundant vs minimal
    A_redundant = """
    Structure repeats. Structure repeats. Structure repeats.
    OMNIA measures structure only. 2026 2025 2024 12345.
    Structure repeats. Structure repeats. Structure repeats.
    """

    A_minimal = """
    One shot message. 2026.
    """

    for label, A in [("redundant", A_redundant), ("minimal", A_minimal)]:
        r = lens.measure(A)
        print()
        print("Redundancy Invariance (RIv-1.0) demo ->", label)
        print("Ω curve:", [round(v, 4) for v in r.omega_curve])
        print("collapse_cost:", r.collapse_cost)
        print("collapse_index:", r.collapse_index)
        print("is_redundant:", r.is_redundant)


if __name__ == "__main__":
    main()