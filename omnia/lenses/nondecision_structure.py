from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

from omnia.features.meaning_blind import meaning_blind_features

Transform = Callable[[str], str]


def _jaccard_like(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a and not b:
        return 1.0
    keys = set(a.keys()) | set(b.keys())
    if not keys:
        return 1.0
    num = 0.0
    den = 0.0
    for k in keys:
        av = float(a.get(k, 0.0))
        bv = float(b.get(k, 0.0))
        num += min(av, bv)
        den += max(av, bv)
    return 0.0 if den <= 0.0 else max(0.0, min(1.0, num / den))


@dataclass(frozen=True)
class NonDecisionResult:
    """
    Non-Decision Structure (NDv-1.0)

    Measures whether a structure remains stable across multiple,
    competing transformation paths without converging to a single
    preferred outcome.

    - path_scores: Ω(A, T_i(A)) per independent path
    - dispersion: variance-like measure of path_scores
    - mean_score: average structural stability
    - is_nondecisional: certificate (high mean, low dispersion)
    """
    path_scores: Dict[str, float]
    dispersion: float
    mean_score: float
    is_nondecisional: bool
    diagnostics: Dict[str, float]


class NonDecisionStructure:
    """
    Non-Decision Structure Lens (NDv-1.0)

    Core idea:
    - Apply multiple independent transformation paths.
    - Measure structural similarity to baseline for each path.
    - If structure stays stable *without one path dominating*,
      the system is non-decisional.

    This captures structures that:
    - do not optimize
    - do not select
    - do not converge
    yet remain structurally coherent.
    """

    def __init__(
        self,
        paths: Sequence[Tuple[str, Transform]],
        *,
        mean_floor: float = 0.7,
        dispersion_ceiling: float = 0.05,
    ):
        if not paths:
            raise ValueError("Need at least one path")
        self.paths = list(paths)
        self.mean_floor = float(mean_floor)
        self.dispersion_ceiling = float(dispersion_ceiling)

    def measure(self, x: str) -> NonDecisionResult:
        base = meaning_blind_features(x)

        scores: Dict[str, float] = {}
        for name, t in self.paths:
            y = t(x)
            f = meaning_blind_features(y)
            scores[name] = _jaccard_like(base, f)

        vals = list(scores.values())
        mean_score = sum(vals) / max(1, len(vals))

        # simple dispersion proxy: mean absolute deviation
        dispersion = (
            sum(abs(v - mean_score) for v in vals) / max(1, len(vals))
        )

        is_nondecisional = (
            mean_score >= self.mean_floor and dispersion <= self.dispersion_ceiling
        )

        diag = {
            "mean_floor": self.mean_floor,
            "dispersion_ceiling": self.dispersion_ceiling,
            "num_paths": float(len(self.paths)),
        }

        return NonDecisionResult(
            path_scores=scores,
            dispersion=dispersion,
            mean_score=mean_score,
            is_nondecisional=is_nondecisional,
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

    def t_shuffle_words(seed: int) -> Transform:
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

    paths = [
        ("id", t_identity),
        ("shuffle3", t_shuffle_words(seed=3)),
        ("shuffle7", t_shuffle_words(seed=7)),
        ("reverse", t_reverse),
    ]

    lens = NonDecisionStructure(paths, mean_floor=0.7, dispersion_ceiling=0.05)

    A = """
    Some structures persist without choosing a path.
    They do not optimize. They do not converge.
    OMNIA measures only structure. 2026 2025 2024.
    """

    r = lens.measure(A)
    print("Non-Decision Structure (NDv-1.0)")
    print("path_scores:", {k: round(v, 4) for k, v in r.path_scores.items()})
    print("mean_score:", round(r.mean_score, 4))
    print("dispersion:", round(r.dispersion, 6))
    print("is_nondecisional:", r.is_nondecisional)