from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple


# A "measurer" is anything that maps a representation -> scalar score Ω (or Ω-like).
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
    Structural Projection Loss (SPL) measures how much structure is lost when you force
    a "projection" (i.e., a privileged measurement basis / observer-induced constraint)
    compared to an aperspective baseline.

    It is NOT an interpretation.
    It is a measured delta between two measurement regimes.

    Definitions:
      - Ω_ap: aperspective baseline (no privileged basis)
      - Ω_proj: projected measurement (privileged basis enforced)
      - SPL_abs = max(0, Ω_ap - Ω_proj)
      - SPL_rel = SPL_abs / max(eps, Ω_ap)

    If SPL is high, your measurement regime is collapsing structure.
    """
    omega_aperspective: float
    omega_projected: float
    spl_abs: float
    spl_rel: float
    details: Dict[str, float]


class MeasurementProjectionLoss:
    """
    A minimal, closed operator that converts "observer/basis forcing" into a measurable cost.

    You provide:
      - aperspective_measurers: a set of independent measurers that should NOT privilege a basis
      - projected_measurers: a set of measurers that DO privilege a basis (projection), or
        implement a constrained view (selection, asymmetry, irreversible filtering)

    OMNIA logic:
      - aggregate each group into one Ω via robust mean (trimmed mean) or median
      - report the measured loss of structure when projection is introduced

    This is the missing bridge between:
      - aperspective invariance
      - observer perturbation
      - "collapse" as measurement artifact
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
        if self.aggregator == "mean":
            return sum(xs) / len(xs)
        if self.aggregator == "median":
            return self._median(xs)

        # trimmed_mean
        ys = sorted(xs)
        n = len(ys)
        k = int(n * self.trim_q)
        core = ys[k : n - k] if (n - 2 * k) > 0 else ys
        return sum(core) / len(core)

    def measure(self, x: str) -> ProjectionLossResult:
        # compute individual Ω scores
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

        omega_ap = self._aggregate(ap_scores)
        omega_pr = self._aggregate(pr_scores)

        # clamp (optional): OMNIA often uses normalized scores
        omega_ap = _clamp01(omega_ap)
        omega_pr = _clamp01(omega_pr)

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