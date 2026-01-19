from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from omnia.lenses.aperspective_invariance import AperspectiveInvariance, Transform
from omnia.meta.total_collapse import (
    TotalCollapseOperator,
    t_identity,
    t_whitespace_collapse,
    t_reverse,
    t_drop_vowels,
    t_shuffle_words,
    t_chunk_shuffle,
    t_charset_remap,
    t_prefer_compressible,
    t_force_uniformity,
    t_drop_everything_but_class,
)

# CT = Total Collapsator: stronger schedule + fixed-point closure


@dataclass(frozen=True)
class CollapsatorTotalResult:
    """
    CT-1.0

    y_star:
        Final collapsed representation.

    omega0:
        Baseline aperspective omega of original x.

    omega_star:
        Aperspective omega of y_star.

    steps:
        How many collapse rounds were applied (fixed-point closure).

    collapsed:
        True if omega_star <= eps at any stage.

    notes:
        Minimal diagnostics only.
    """
    y_star: str
    omega0: float
    omega_star: float
    steps: int
    collapsed: bool
    notes: Dict[str, float]


class CollapsatorTotal:
    """
    Collapsator Total (CT-1.0)

    Idea:
      - Run a strong TotalCollapseOperator schedule.
      - Then re-run the schedule on its own output (closure),
        until reaching a fixed point (no further structural change)
        or full collapse.

    CT is NOT OMNIA.
    CT is maximal perturbation with deterministic closure.
    """

    def __init__(
        self,
        *,
        baseline_transforms: Sequence[Tuple[str, Transform]],
        collapse_schedule: Sequence[Tuple[float, str, Transform]],
        eps: float = 1e-4,
        max_rounds: int = 6,
        fixedpoint_tol: float = 1e-6,
    ):
        if not baseline_transforms:
            raise ValueError("baseline_transforms must be non-empty")
        if not collapse_schedule:
            raise ValueError("collapse_schedule must be non-empty")
        if eps <= 0:
            raise ValueError("eps must be > 0")
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")

        self.eps = float(eps)
        self.max_rounds = int(max_rounds)
        self.fixedpoint_tol = float(fixedpoint_tol)

        self.ap = AperspectiveInvariance(transforms=list(baseline_transforms))
        self.tco = TotalCollapseOperator(
            baseline_transforms=list(baseline_transforms),
            collapse_schedule=list(collapse_schedule),
            eps=float(eps),
            clamp=True,
        )

    def _omega_ap(self, s: str) -> float:
        return float(self.ap.measure(s).omega_score)

    def _apply_full_schedule(self, s: str) -> str:
        # Apply the collapse schedule sequentially (max perturbation path)
        y = s
        for _cost, _name, t in self.tco.collapse_schedule:
            y = t(y)
        return y

    def collapse(self, x: str) -> CollapsatorTotalResult:
        omega0 = self._omega_ap(x)

        y = x
        prev_omega: Optional[float] = None
        collapsed = False
        steps = 0

        for k in range(self.max_rounds):
            steps = k + 1
            y = self._apply_full_schedule(y)
            omega = self._omega_ap(y)

            if omega <= self.eps:
                collapsed = True
                prev_omega = omega
                break

            if prev_omega is not None:
                # fixed point: omega no longer decreases (within tolerance)
                if abs(prev_omega - omega) <= self.fixedpoint_tol:
                    prev_omega = omega
                    break

            prev_omega = omega

        omega_star = float(prev_omega if prev_omega is not None else self._omega_ap(y))

        notes: Dict[str, float] = {
            "omega0": float(omega0),
            "omega_star": float(omega_star),
            "eps": float(self.eps),
            "rounds": float(steps),
        }

        return CollapsatorTotalResult(
            y_star=y,
            omega0=float(omega0),
            omega_star=float(omega_star),
            steps=int(steps),
            collapsed=bool(collapsed),
            notes=notes,
        )


def default_ct() -> CollapsatorTotal:
    # Baseline aperspective transforms (no privileged view)
    baseline = [
        ("id", t_identity),
        ("ws", t_whitespace_collapse),
        ("rev", t_reverse),
        ("vow-", t_drop_vowels),
        ("shuf", t_shuffle_words(seed=3)),
    ]

    # Stronger schedule than TCO default (still deterministic, meaning-blind)
    # Cost is monotonic, but only used as ordering.
    schedule = [
        (1.0, "ws", t_whitespace_collapse),
        (2.0, "shuf_words_9", t_shuffle_words(seed=9)),
        (3.0, "chunk_shuffle_24", t_chunk_shuffle(seed=11, chunk=24)),
        (4.0, "charset_remap", t_charset_remap(seed=13)),
        (5.0, "prefer_compress", t_prefer_compressible(seed=21, keep_ratio=0.30, window=64)),
        (6.0, "force_uniform", t_force_uniformity(target_len=200)),
        (7.0, "class_only", t_drop_everything_but_class()),
        (8.0, "reverse", t_reverse),
        (9.0, "chunk_shuffle_16", t_chunk_shuffle(seed=17, chunk=16)),
        (10.0, "shuf_words_31", t_shuffle_words(seed=31)),
    ]

    return CollapsatorTotal(
        baseline_transforms=baseline,
        collapse_schedule=schedule,
        eps=1e-4,
        max_rounds=6,
        fixedpoint_tol=1e-6,
    )