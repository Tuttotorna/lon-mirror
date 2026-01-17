from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from omnia.features.meaning_blind import meaning_blind_features

Transform = Callable[[str], str]


def _l1_distance(a: Dict[str, float], b: Dict[str, float]) -> float:
    """
    L1 distance over sparse feature maps (keys union).
    Meaning-blind, order-free.
    """
    keys = set(a.keys()) | set(b.keys())
    if not keys:
        return 0.0
    d = 0.0
    for k in keys:
        d += abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0)))
    return d


def _dist_invariance_score(base: Dict[str, float], other: Dict[str, float]) -> float:
    """
    Distribution Invariance score in [0,1].
    1 means identical distributions; lower means drift.
    """
    d = _l1_distance(base, other)
    # Normalize by total mass to keep scale stable
    mass = sum(float(v) for v in base.values()) + sum(float(v) for v in other.values())
    if mass <= 0.0:
        return 1.0
    # Map distance to similarity
    return max(0.0, min(1.0, 1.0 - (d / mass)))


@dataclass(frozen=True)
class DistributionInvarianceResult:
    """
    Distribution Invariance (DIv-1.0)

    Measures stability of the global distribution of structural features,
    independent of local ordering.

    - scores: per-transform distribution invariance scores
    - min_score: worst-case invariance
    - mean_score: average invariance
    - is_invariant: certificate based on threshold
    """
    scores: Dict[str, float]
    min_score: float
    mean_score: float
    is_invariant: bool
    diagnostics: Dict[str, float]


class DistributionInvariance:
    """
    Distribution Invariance Lens (DIv-1.0)

    Core idea:
    - Compare global feature distributions before/after transforms.
    - Ignore locality and order; measure only distributional drift.

    This captures structures that:
    - are nowhere locally
    - exist only as global statistical shape
    """

    def __init__(
        self,
        transforms: Sequence[Tuple[str, Transform]],
        *,
        threshold: float = 0.8,
    ):
        if not transforms:
            raise ValueError("Need at least one transform")
        if not (0.0 < threshold <= 1.0):
            raise ValueError("threshold must be in (0,1]")
        self.transforms = list(transforms)
        self.threshold = float(threshold)

    def measure(self, x: str) -> DistributionInvarianceResult:
        base = meaning_blind_features(x)

        scores: Dict[str, float] = {}
        for name, t in self.transforms:
            y = t(x)
            f = meaning_blind_features(y)
            scores[name] = _dist_invariance_score(base, f)

        vals = list(scores.values())
        min_score = min(vals) if vals else 0.0
        mean_score = sum(vals) / max(1, len(vals))

        is_invariant = min_score >= self.threshold

        diag = {
            "threshold": self.threshold,
            "num_transforms": float(len(self.transforms)),
        }

        return DistributionInvarianceResult(
            scores=scores,
            min_score=min_score,
            mean_score=mean_score,
            is_invariant=is_invariant,
            diagnostics=diag,
        )


# -------------------------
# Minimal demo
# -------------------------

if __name__ == "__main__":
    import random
    import re

    def t_identity(s: str) -> str:
        return s

    def t_shuffle_words(seed: int = 3) -> Transform:
        def _t(s: str) -> str:
            rng = random.Random(seed)
            parts = re.split(r"(\W+)", s)
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

    transforms = [
        ("id", t_identity),
        ("shuffle", t_shuffle_words(seed=3)),
        ("reverse", t_reverse),
    ]

    lens = DistributionInvariance(transforms, threshold=0.8)

    A = """
    Interference patterns exist as distributions, not local trajectories.
    OMNIA measures structure only. 2026 2025 2024 12345.
    """

    r = lens.measure(A)
    print("Distribution Invariance (DIv-1.0)")
    print("scores:", {k: round(v, 4) for k, v in r.scores.items()})
    print("min_score:", round(r.min_score, 4))
    print("mean_score:", round(r.mean_score, 4))
    print("is_invariant:", r.is_invariant)