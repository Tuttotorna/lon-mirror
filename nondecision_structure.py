from __future__ import annotations

import random
import re
from typing import Callable

from omnia.lenses.distribution_invariance import DistributionInvariance

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


def t_reverse(s: str) -> str:
    return s[::-1]


def t_drop_vowels(s: str) -> str:
    return re.sub(r"[aeiouAEIOUàèéìòùÀÈÉÌÒÙ]", "", s)


# -----------------------------
# Demo
# -----------------------------

def main() -> None:
    transforms = [
        ("id", t_identity),
        ("ws", t_whitespace_collapse),
        ("shuffle", t_shuffle_words(seed=3)),
        ("reverse", t_reverse),
        ("vow-", t_drop_vowels),
    ]

    lens = DistributionInvariance(transforms=transforms, threshold=0.8)

    A = """
    Distribution invariance: the structure exists as a global signature, not a path.
    Shuffle destroys syntax, but distributional shape can remain.
    OMNIA measures only structure. 2026 2025 2024 12345.
    """

    r = lens.measure(A)

    print("Distribution Invariance (DIv-1.0) demo")
    print("scores:", {k: round(v, 4) for k, v in r.scores.items()})
    print("min_score:", round(r.min_score, 4))
    print("mean_score:", round(r.mean_score, 4))
    print("is_invariant:", r.is_invariant)


if __name__ == "__main__":
    main()