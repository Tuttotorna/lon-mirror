from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple


# A "measurer" is anything that maps a representation -> scalar score Ω or Ω-like.
# It must be deterministic and semantics-free by design of OMNIA.
Measurer = Callable[[str], float]


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _safe_div(a: float, b: float, eps: float = 1e-12) -> float:
    return a / (b + eps)


@dataclass(frozen=True)
class ProjectionLossResult:
    """
    Structural Projection Loss measures how much structure is lost when a projected
    or privileged measurement regime is forced against an aperspective baseline.

    It is not interpretation.
    It is not semantic judgment.
    It is not decision.

    Definitions:
      omega_aperspective = aggregated aperspective score
      omega_projected    = aggregated projected score
      spl_abs            = max(0, omega_aperspective - omega_projected)
      spl_rel            = spl_abs / omega_aperspective
    """

    omega_aperspective: float
    omega_projected: float
    spl_abs: float
    spl_rel: float
    details: Dict[str, float]


class MeasurementProjectionLoss:
    """
    Minimal Structural Projection Loss operator.

    The operator compares two measurement regimes:

      1. Aperspective measurers:
         independent measurements that should not privilege a single basis.

      2. Projected measurers:
         constrained or basis-forced measurements.

    The result is a post-hoc structural delta.

    Boundary:
      measurement != inference != decision
    """

    def __init__(
        self,
        aperspective_measurers: Sequence[Tuple[str, Measurer]],
        projected_measurers: Sequence[Tuple[str, Measurer]],
        aggregator: str = "trimmed_mean",
        trim_q: float = 0.2,
    ) -> None:
        if not aperspective_measurers:
            raise ValueError("Need at least one aperspective measurer")
        if not projected_measurers:
            raise ValueError("Need at least one projected measurer")

        if aggregator not in ("mean", "median", "trimmed_mean"):
            raise ValueError("aggregator must be one of: mean, median, trimmed_mean")

        if not (0.0 <= trim_q < 0.5):
            raise ValueError("trim_q must be in [0, 0.5)")

        self.ap = list(aperspective_measurers)
        self.pr = list(projected_measurers)
        self.aggregator = aggregator
        self.trim_q = trim_q

    @staticmethod
    def _median(xs: List[float]) -> float:
        ys = sorted(xs)
        n = len(ys)

        if n == 0:
            return 0.0

        mid = n // 2

        if n % 2 == 1:
            return ys[mid]

        return 0.5 * (ys[mid - 1] + ys[mid])

    def _aggregate(self, xs: List[float]) -> float:
        if not xs:
            return 0.0

        ys = sorted(float(x) for x in xs)
        n = len(ys)

        if self.aggregator == "mean":
            return sum(ys) / n

        if self.aggregator == "median":
            return self._median(ys)

        # trimmed_mean
        #
        # Important:
        # int(n * trim_q) fails for small samples.
        #
        # Example:
        #   n = 3
        #   trim_q = 0.2
        #   int(3 * 0.2) = 0
        #
        # That means no trimming happens, so a single outlier such as 0.0
        # contaminates the aggregate.
        #
        # For a robust small-sample trimmed mean, any positive trim_q should
        # remove at least one item from each side when n >= 3 and the trim is
        # feasible.
        if self.trim_q <= 0.0 or n < 3:
            return sum(ys) / n

        k = math.ceil(n * self.trim_q)

        # Never trim away the entire sample.
        max_k = (n - 1) // 2
        k = min(k, max_k)

        if k <= 0:
            core = ys
        else:
            core = ys[k : n - k]

        if not core:
            core = ys

        return sum(core) / len(core)

    def measure(self, x: str) -> ProjectionLossResult:
        ap_scores: List[float] = []
        pr_scores: List[float] = []
        details: Dict[str, float] = {}

        for name, f in self.ap:
            v = float(f(x))
            ap_scores.append(v)
            details[f"ap::{name}"] = v

        for name, f in self.pr:
            v = float(f(x))
            pr_scores.append(v)
            details[f"pr::{name}"] = v

        omega_ap = _clamp01(self._aggregate(ap_scores))
        omega_pr = _clamp01(self._aggregate(pr_scores))

        spl_abs = max(0.0, omega_ap - omega_pr)
        spl_rel = _safe_div(spl_abs, max(1e-12, omega_ap))

        details["omega_ap"] = omega_ap
        details["omega_pr"] = omega_pr
        details["spl_abs"] = spl_abs
        details["spl_rel"] = spl_rel

        return ProjectionLossResult(
            omega_aperspective=omega_ap,
            omega_projected=omega_pr,
            spl_abs=spl_abs,
            spl_rel=spl_rel,
            details=details,
        )