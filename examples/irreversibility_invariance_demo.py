from __future__ import annotations

import random
import re
from typing import Callable

from omnia.lenses.irreversibility_invariance import IrreversibilityInvariance

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


def t_drop_vowels(s: str) -> str:
    return re.sub(r"[aeiouAEIOUàèéìòùÀÈÉÌÒÙ]", "", s)


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


# -----------------------------
# Demo
# -----------------------------

def main() -> None:
    A = """
    Irreversibility is structural hysteresis: A -> B -> A'.
    If the forward transform is lossy, A' cannot recover A.
    OMNIA measures only structure. No semantics. No narrative.
    2026 2025 2024 12345
    """

    cases = [
        ("lossy_vowels_then_id", t_drop_vowels, t_identity),
        ("lossy_vowels_then_ws", t_drop_vowels, t_whitespace_collapse),
        ("shuffle_then_shuffle", t_shuffle_words(seed=3), t_shuffle_words(seed=3)),
        ("ws_then_ws", t_whitespace_collapse, t_whitespace_collapse),
    ]

    print("Irreversibility Invariance (IIv-1.0) demo")
    for name, fwd, bwd in cases:
        lens = IrreversibilityInvariance(forward=fwd, backward=bwd, eps=1e-6)
        r = lens.measure(A)
        print()
        print("case:", name)
        print("  Ω(A,B):", round(r.omega_AB, 4))
        print("  Ω(A,A'):", round(r.omega_AAprime, 4))
        print("  IRI:", round(r.iri, 6))
        print("  is_irreversible:", r.is_irreversible)


if __name__ == "__main__":
    main()