from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from omnia.lenses.aperspective_invariance import (
    AperspectiveInvariance,
    AperspectiveInvarianceResult,
    Transform,
    t_identity,
    t_whitespace_collapse,
    t_reverse,
    t_drop_vowels,
    t_shuffle_words,
    t_base_repr,
)

# -------------------------
# Types
# -------------------------

Observer = Callable[[str], str]


@dataclass(frozen=True)
class ObserverPerturbationResult:
    """
    OPI-1.0: Observer Perturbation Index

    omega_aperspective: Ω_ap computed on the raw input under aperspective transforms
    omega_observed:     Ω_ap computed on the observed/annotated/optimized version under same transforms

    opi = max(0, omega_aperspective - omega_observed)
    ratio = opi / max(eps, omega_aperspective)

    Interpret only as structural loss under observer-introduced transformation.
    """
    omega_aperspective: float
    omega_observed: float
    opi: float
    ratio: float
    details_aperspective: AperspectiveInvarianceResult
    details_observed: AperspectiveInvarianceResult


class ObserverPerturbation:
    """
    ObserverPerturbation (OPI-1.0)

    Measures the structural cost of introducing an "observer transform" (e.g. explanation,
    formatting, optimization, narrative framing) relative to an aperspective baseline.

    - No semantics
    - No policy
    - No causality claims

    It simply measures: how much aperspective invariance drops after observer intervention.
    """

    def __init__(
        self,
        *,
        transforms: Optional[List[Tuple[str, Transform]]] = None,
        eps: float = 1e-9,
    ):
        self.transforms = transforms or self.default_transforms()
        self.eps = float(eps)

        # Reuse AperspectiveInvariance engine as the measurement core
        self._engine = AperspectiveInvariance(self.transforms)

    @staticmethod
    def default_transforms() -> List[Tuple[str, Transform]]:
        # Independent, meaning-blind stress transforms
        return [
            ("id", t_identity),
            ("ws", t_whitespace_collapse),
            ("rev", t_reverse),
            ("vow-", t_drop_vowels),
            ("shuf", t_shuffle_words(seed=3)),
            ("base7", t_base_repr(seed=7, base=7)),
        ]

    def measure(
        self,
        *,
        x: str,
        observer: Observer,
    ) -> ObserverPerturbationResult:
        # Baseline: aperspective invariance on raw x
        base = self._engine.measure(x)
        omega_ap = float(base.omega_score)

        # Observed: apply observer transform, then measure aperspective invariance again
        x_obs = observer(x)
        obs = self._engine.measure(x_obs)
        omega_obs = float(obs.omega_score)

        opi = max(0.0, omega_ap - omega_obs)
        ratio = opi / max(self.eps, omega_ap)

        return ObserverPerturbationResult(
            omega_aperspective=omega_ap,
            omega_observed=omega_obs,
            opi=opi,
            ratio=ratio,
            details_aperspective=base,
            details_observed=obs,
        )


# -------------------------
# Minimal "observer" library
# -------------------------

def o_identity(s: str) -> str:
    return s


def o_add_explanation(prefix: str = "Explanation: ") -> Observer:
    """
    A crude observer model: injects narrative/explanatory text.
    Meaning is not evaluated; only structural side-effects are measured.
    """
    def _o(s: str) -> str:
        return f"{prefix}{s}"
    return _o


def o_reformat_bullets() -> Observer:
    """
    Another observer model: imposes formatting structure (listification).
    """
    def _o(s: str) -> str:
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        if not lines:
            return s
        return "\n".join(f"- {ln}" for ln in lines)
    return _o


def o_optimize_for_clarity() -> Observer:
    """
    Lightweight rewrite proxy: normalize whitespace + add headings.
    Still no semantics, but introduces human-facing structure.
    """
    def _o(s: str) -> str:
        lines = [ln.strip() for ln in s.replace("\r\n", "\n").split("\n")]
        lines = [ln for ln in lines if ln]
        body = " ".join(lines)
        return "SUMMARY\n" + body + "\nEND"
    return _o


if __name__ == "__main__":
    lens = ObserverPerturbation()

    x = """
    Double-slit interference: 2 slits yield a stable fringe pattern.
    OMNIA measures structure only: length, repetition, compressibility, n-grams.
    2026 2025 2024 12345
    """

    for name, obs in [
        ("identity", o_identity),
        ("explanation", o_add_explanation("This means that: ")),
        ("bullets", o_reformat_bullets()),
        ("clarity", o_optimize_for_clarity()),
    ]:
        r = lens.measure(x=x, observer=obs)
        print("\nOPI-1.0:", name)
        print("  Ω_ap:", round(r.omega_aperspective, 4))
        print("  Ω_obs:", round(r.omega_observed, 4))
        print("  OPI:", round(r.opi, 6))
        print("  ratio:", round(r.ratio, 6))