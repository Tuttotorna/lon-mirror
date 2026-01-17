from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

Transform = Callable[[str], str]
FeatureFn = Callable[[str], Dict[str, float]]  # meaning-blind structural features


@dataclass(frozen=True)
class SaturationResult:
    """
    Saturation Invariance (SIv-1.0)

    Measures when marginal structural yield collapses under increasing
    transformation cost/strength, without semantics, observer, or narrative.

    - omega_curve: Ω values along a monotonic cost schedule
    - sei_curve: ΔΩ / ΔC along the schedule (finite differences)
    - c_star: estimated saturation point (first stable near-zero SEI window)
    - is_saturated: boolean certificate (schedule-local)
    """
    omega_curve: List[float]
    sei_curve: List[float]
    cost_curve: List[float]
    c_star: Optional[float]
    is_saturated: bool
    diagnostics: Dict[str, float]


def _jaccard_like(a: Dict[str, float], b: Dict[str, float]) -> float:
    """
    Meaning-blind similarity on sparse feature maps.
    Uses min/max overlap across keys (continuous Jaccard).
    """
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


def _finite_diff_sei(omega: List[float], cost: List[float]) -> List[float]:
    """
    SEI[i] = (Ω[i] - Ω[i-1]) / (C[i] - C[i-1]) for i>=1
    SEI[0] = 0 by definition (no previous point)
    """
    if len(omega) != len(cost):
        raise ValueError("omega and cost must have same length")
    if not omega:
        return []
    out = [0.0]
    for i in range(1, len(omega)):
        d_omega = omega[i] - omega[i - 1]
        d_cost = cost[i] - cost[i - 1]
        if d_cost == 0:
            out.append(0.0)
        else:
            out.append(d_omega / d_cost)
    return out


def _estimate_saturation_point(
    sei: List[float],
    cost: List[float],
    window: int = 3,
    eps: float = 1e-3,
) -> Optional[float]:
    """
    Finds first index i where a trailing window of |SEI| is <= eps.
    Returns corresponding cost C[i], else None.

    This is not a universal threshold; it's a local certificate relative
    to the chosen schedule and feature function.
    """
    if len(sei) != len(cost) or len(sei) == 0:
        return None
    if window < 2:
        window = 2

    for i in range(window - 1, len(sei)):
        w = sei[i - window + 1 : i + 1]
        if all(abs(x) <= eps for x in w):
            return cost[i]
    return None


class SaturationInvariance:
    """
    Saturation Invariance Lens (SIv-1.0)

    Core idea:
    - Build a monotonic "cost schedule" of transforms (increasing strength/cost)
    - Compute Ω along the schedule as structural similarity vs baseline
    - Compute SEI = ΔΩ/ΔC
    - Certify saturation when SEI remains near-zero for a window

    Notes:
    - No semantics: FeatureFn must be meaning-blind.
    - Deterministic: If transforms are stochastic, they must be seed-locked.
    """

    def __init__(
        self,
        feature_fn: FeatureFn,
        schedule: Sequence[Tuple[float, Transform]],
        *,
        sei_window: int = 3,
        sei_eps: float = 1e-3,
    ):
        if not schedule:
            raise ValueError("schedule must be non-empty")
        # Enforce monotonic non-decreasing cost
        costs = [float(c) for c, _ in schedule]
        if any(costs[i] < costs[i - 1] for i in range(1, len(costs))):
            raise ValueError("schedule costs must be monotonic non-decreasing")
        self.feature_fn = feature_fn
        self.schedule = [(float(c), t) for c, t in schedule]
        self.sei_window = int(sei_window)
        self.sei_eps = float(sei_eps)

    def measure(self, x: str) -> SaturationResult:
        base = self.feature_fn(x)

        omega_curve: List[float] = []
        cost_curve: List[float] = []

        # Ω here = similarity(base, features(transform(x)))
        # This is a structural yield curve under rising transformation cost.
        for c, t in self.schedule:
            y = t(x)
            f = self.feature_fn(y)
            omega = _jaccard_like(base, f)
            omega_curve.append(omega)
            cost_curve.append(c)

        sei_curve = _finite_diff_sei(omega_curve, cost_curve)
        c_star = _estimate_saturation_point(
            sei_curve, cost_curve, window=self.sei_window, eps=self.sei_eps
        )

        is_saturated = c_star is not None

        # Minimal diagnostics, purely structural
        diag = {
            "omega_min": min(omega_curve) if omega_curve else 0.0,
            "omega_max": max(omega_curve) if omega_curve else 0.0,
            "sei_min": min(sei_curve) if sei_curve else 0.0,
            "sei_max": max(sei_curve) if sei_curve else 0.0,
            "schedule_len": float(len(self.schedule)),
        }

        return SaturationResult(
            omega_curve=omega_curve,
            sei_curve=sei_curve,
            cost_curve=cost_curve,
            c_star=c_star,
            is_saturated=is_saturated,
            diagnostics=diag,
        )