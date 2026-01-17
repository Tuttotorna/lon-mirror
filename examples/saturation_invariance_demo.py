from __future__ import annotations

import random
import re
from typing import Callable, Sequence, Tuple

from omnia.features.meaning_blind import meaning_blind_features
from omnia.lenses.saturation_invariance import SaturationInvariance

Transform = Callable[[str], str]


# -----------------------------
# Seeded independent transforms
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
    Structural representation stress-test. No semantics.
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
    # Monotonic cost schedule: increasing “destructiveness”
    schedule: Sequence[Tuple[float, Transform]] = [
        (0.0, t_identity),
        (1.0, t_whitespace_collapse),
        (2.0, t_shuffle_words(seed=3)),
        (3.0, t_drop_vowels),
        (4.0, t_reverse),
        (5.0, t_base_repr(seed=7, base=7)),
    ]

    lens = SaturationInvariance(
        feature_fn=meaning_blind_features,
        schedule=schedule,
        sei_window=3,
        sei_eps=1e-3,
    )

    x = """
    Double-slit interference: 2 slits yield a stable fringe pattern.
    OMNIA measures structure only: length, repetition, compressibility, n-grams.
    2026 2025 2024 12345
    """

    r = lens.measure(x)

    print("Saturation Invariance (SIv-1.0)")
    print("Ω curve:", [round(v, 4) for v in r.omega_curve])
    print("SEI curve:", [round(v, 6) for v in r.sei_curve])
    print("Cost curve:", r.cost_curve)
    print("c* (saturation point):", r.c_star)
    print("is_saturated:", r.is_saturated)
    print("diagnostics:", {k: round(v, 6) for k, v in r.diagnostics.items() if isinstance(v, float)})


if __name__ == "__main__":
    main()