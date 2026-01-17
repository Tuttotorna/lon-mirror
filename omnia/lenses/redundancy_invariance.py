from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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
class RedundancyResult:
    """
    Redundancy Invariance (RIv-1.0)

    Measures how much destructive transformation "pressure" is required
    before structural similarity collapses.

    - omega_curve: Ω(A, T_i(A)) along monotonic cost schedule
    - cost_curve: corresponding costs
    - collapse_cost: first cost where Ω <= omega_floor (if any)
    - collapse_index: index in schedule (if any)
    - is_redundant: True if collapse occurs late (beyond a fraction of schedule)
    """
    omega_curve: List[float]
    cost_curve: List[float]
    collapse_cost: Optional[float]
    collapse_index: Optional[int]
    is_redundant: bool
    diagnostics: Dict[str, float]


class RedundancyInvariance:
    """
    Redundancy Invariance Lens (RIv-1.0)

    Core idea:
    - Apply an increasing destructiveness schedule T with monotonic costs.
    - Measure Ω(A, T_i(A)) vs baseline A.
    - Define "collapse" when Ω drops below omega_floor.
    - A later collapse indicates deeper structural redundancy (tessitura).

    No semantics. Deterministic given seeds and schedule.
    """

    def __init__(
        self,
        schedule: Sequence[Tuple[float, Transform]],
        *,
        omega_floor: float = 0.15,
        late_fraction: float = 0.7,
    ):
        if not schedule:
            raise ValueError("schedule must be non-empty")
        costs = [float(c) for c, _ in schedule]
        if any(costs[i] < costs[i - 1] for i in range(1, len(costs))):
            raise ValueError("schedule costs must be monotonic non-decreasing")
        if not (0.0 < omega_floor < 1.0):
            raise ValueError("omega_floor must be in (0,1)")
        if not (0.0 < late_fraction <= 1.0):
            raise ValueError("late_fraction must be in (0,1]")

        self.schedule = [(float(c), t) for c, t in schedule]
        self.omega_floor = float(omega_floor)
        self.late_fraction = float(late_fraction)

    def measure(self, a: str) -> RedundancyResult:
        fA = meaning_blind_features(a)

        omega_curve: List[float] = []
        cost_curve: List[float] = []

        collapse_index: Optional[int] = None
        collapse_cost: Optional[float] = None

        for i, (c, t) in enumerate(self.schedule):
            b = t(a)
            fB = meaning_blind_features(b)
            omega = _jaccard_like(fA, fB)
            omega_curve.append(omega)
            cost_curve.append(c)

            if collapse_index is None and omega <= self.omega_floor:
                collapse_index = i
                collapse_cost = c

        # "Deep redundancy" if collapse happens late in the schedule (or never collapses)
        if collapse_index is None:
            is_redundant = True
        else:
            is_redundant = (collapse_index / max(1, len(self.schedule) - 1)) >= self.late_fraction

        diag = {
            "omega_floor": self.omega_floor,
            "late_fraction": self.late_fraction,
            "omega_min": min(omega_curve) if omega_curve else 0.0,
            "omega_max": max(omega_curve) if omega_curve else 0.0,
            "schedule_len": float(len(self.schedule)),
        }

        return RedundancyResult(
            omega_curve=omega_curve,
            cost_curve=cost_curve,
            collapse_cost=collapse_cost,
            collapse_index=collapse_index,
            is_redundant=is_redundant,
            diagnostics=diag,
        )


if __name__ == "__main__":
    import re
    import random

    def t_identity(s: str) -> str:
        return s

    def t_ws(s: str) -> str:
        s = s.replace("\r\n", "\n")
        s = re.sub(r"[ \t]+", " ", s)
        return s.strip()

    def t_drop_vowels(s: str) -> str:
        return re.sub(r"[aeiouAEIOUàèéìòùÀÈÉÌÒÙ]", "", s)

    def t_reverse(s: str) -> str:
        return s[::-1]

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

    schedule = [
        (0.0, t_identity),
        (1.0, t_ws),
        (2.0, t_shuffle_words(seed=3)),
        (3.0, t_drop_vowels),
        (4.0, t_reverse),
    ]

    lens = RedundancyInvariance(schedule=schedule, omega_floor=0.15, late_fraction=0.7)

    A = """
    A message with redundancy: repeated structure, repeated motifs, repeated forms.
    OMNIA measures structure only. 2026 2025 2024 12345.
    A message with redundancy: repeated structure, repeated motifs, repeated forms.
    """

    r = lens.measure(A)
    print("Redundancy Invariance (RIv-1.0)")
    print("Ω curve:", [round(v, 4) for v in r.omega_curve])
    print("collapse_cost:", r.collapse_cost)
    print("is_redundant:", r.is_redundant)