from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from omnia.features.meaning_blind import meaning_blind_features

Transform = Callable[[str], str]


def _feature_set(s: str) -> set:
    """
    Keys-only feature-set for aperspective invariance.

    Source features are meaning-blind (no semantics, no observer, no narrative),
    extracted by `meaning_blind_features`. We reduce to keys to keep the
    aperspective definition as pure set-intersection.
    """
    return set(meaning_blind_features(s).keys())


@dataclass(frozen=True)
class AperspectiveInvarianceResult:
    omega_score: float
    residue: List[str]
    per_transform_scores: Dict[str, float]


class AperspectiveInvariance:
    """
    Aperspective Invariance Lens (AIv-1.0)

    Measures invariants that persist under independent transformations without
    introducing any privileged point of view.

    - No semantics
    - No observer assumptions
    - No causality
    - No narrative framing

    Definition (keys-only):
      S0 = F(x)
      Si = F(t_i(x))
      R  = ⋂ Si
      Ω_ap = |R| / |S0|

    where F(.) is meaning-blind structural features (keys).
    """

    def __init__(self, transforms: List[Tuple[str, Transform]]):
        if not transforms:
            raise ValueError("Need at least one transform")
        self.transforms = transforms

    def measure(self, x: str) -> AperspectiveInvarianceResult:
        base_feats = _feature_set(x)
        inter = set(base_feats)

        per_scores: Dict[str, float] = {}

        for name, t in self.transforms:
            y = t(x)
            f = _feature_set(y)

            inter &= f

            denom = max(1, len(base_feats))
            per_scores[name] = len(base_feats & f) / denom

        omega = len(inter) / max(1, len(base_feats))
        residue_sorted = sorted(inter)

        return AperspectiveInvarianceResult(
            omega_score=omega,
            residue=residue_sorted[:200],
            per_transform_scores=per_scores,
        )


# -------------------------
# Example transform library
# -------------------------

def t_identity(s: str) -> str:
    return s


def t_whitespace_collapse(s: str) -> str:
    s = s.replace("\r\n", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def t_shuffle_words(seed: int = 1) -> Transform:
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


if __name__ == "__main__":
    transforms = [
        ("id", t_identity),
        ("ws", t_whitespace_collapse),
        ("rev", t_reverse),
        ("vow-", t_drop_vowels),
        ("shuf", t_shuffle_words(seed=3)),
        ("base7", t_base_repr(seed=7, base=7)),
    ]

    engine = AperspectiveInvariance(transforms)

    x = """
    Nel doppio slit, 2 fessure producono un pattern d'interferenza.
    Misuro solo struttura: lunghezze, ripetizioni, compressibilità, n-gram.
    2026 2025 2024
    """

    r = engine.measure(x)

    print("Ω-score (aperspective invariance):", round(r.omega_score, 4))
    print("Per-transform overlap:")
    for k, v in sorted(r.per_transform_scores.items()):
        print(" ", k, "->", round(v, 4))
    print("Residue sample (structural tokens):", r.residue[:20])