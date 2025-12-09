"""
omnia.kernel — OMNIA_KERN v1.0
Unified kernel for structural lenses (Omniabase, Omniatempo, Omniacausa, future lenses).

Author: Massimiliano Brighindi (concepts) + MBX IA (formalization)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Callable, List, Optional, Mapping, Iterable


# =========================
# 1. DATA STRUCTURES
# =========================

@dataclass
class OmniaContext:
    """
    Generic context passed to lenses.

    Fields are intentionally generic:
    - n: integer target (e.g. for Omniabase / PBII)
    - series: main 1D time series (e.g. for Omniatempo)
    - series_dict: multivariate series (e.g. for Omniacausa)
    - extra: free-form dict for future lenses (e.g. token logs, metadata)
    """
    n: Optional[int] = None
    series: Optional[Iterable[float]] = None
    series_dict: Optional[Mapping[str, Iterable[float]]] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LensResult:
    """
    Canonical output of a single lens.

    scores:
        scalar metrics (e.g. 'base_instability', 'tempo_log_regime').
    metadata:
        any additional structured data (e.g. full OmniabaseSignature).
    """
    name: str
    scores: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class KernelResult:
    """
    Unified result of all registered lenses.

    overall_omega:
        fused scalar score (weighted sum of normalized components).
    components:
        per-lens contributions after weighting.
    lenses:
        raw LensResult objects for inspection.
    """
    overall_omega: float
    components: Dict[str, float]
    lenses: Dict[str, LensResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_omega": self.overall_omega,
            "components": dict(self.components),
            "lenses": {name: lr.to_dict() for name, lr in self.lenses.items()},
        }


# Type alias for lens function
LensFn = Callable[[OmniaContext], LensResult]


# =========================
# 2. KERNEL IMPLEMENTATION
# =========================

class OmniaKernel:
    """
    OMNIA_KERN v1.0

    Minimal design:
    - register_lens(name, fn, weight)
    - run(context) -> KernelResult
    - internal normalization of scores to avoid dominance by any single lens.

    Assumptions:
    - Each lens exposes at least ONE main scalar in LensResult.scores.
      By convention, if present, 'omega' is used as primary; otherwise
      the mean of all scores is used.
    """

    def __init__(self) -> None:
        self._lenses: Dict[str, LensFn] = {}
        self._weights: Dict[str, float] = {}

    # ---------- REGISTRATION ----------

    def register_lens(
        self,
        name: str,
        fn: LensFn,
        weight: float = 1.0,
    ) -> None:
        """
        Register a lens with a given name and weight.
        If called again with the same name, overwrites previous entry.
        """
        if weight < 0:
            raise ValueError("weight must be non-negative")
        self._lenses[name] = fn
        self._weights[name] = float(weight)

    def registered_lenses(self) -> List[str]:
        """Return sorted list of registered lens names."""
        return sorted(self._lenses.keys())

    # ---------- EXECUTION ----------

    def run(self, ctx: OmniaContext) -> KernelResult:
        """
        Execute all registered lenses on the given context
        and compute a fused Ω-like score.

        Fusion scheme:
        - For each lens:
            primary = scores['omega'] if present
                      else mean(scores.values())
        - Normalize primaries to [0,1] via min-max if range>0, else 0.
        - overall_omega = sum(weight_l * primary_norm_l).
        """
        if not self._lenses:
            # No lenses registered → trivial result
            return KernelResult(
                overall_omega=0.0,
                components={},
                lenses={},
            )

        # 1) Run all lenses
        lens_results: Dict[str, LensResult] = {}
        primaries: Dict[str, float] = {}
        for name, fn in self._lenses.items():
            lr = fn(ctx)
            lens_results[name] = lr

            if "omega" in lr.scores:
                primary = float(lr.scores["omega"])
            else:
                vals = list(lr.scores.values())
                primary = float(sum(vals) / len(vals)) if vals else 0.0
            primaries[name] = primary

        # 2) Normalize primaries to [0,1] via min-max
        vals = list(primaries.values())
        v_min = min(vals)
        v_max = max(vals)
        if v_max > v_min:
            norm = {k: (v - v_min) / (v_max - v_min) for k, v in primaries.items()}
        else:
            # All equal → map to 0.0
            norm = {k: 0.0 for k in primaries.keys()}

        # 3) Weighted fusion
        components: Dict[str, float] = {}
        overall = 0.0
        for name, p_norm in norm.items():
            w = self._weights.get(name, 1.0)
            contrib = w * p_norm
            components[name] = contrib
            overall += contrib

        return KernelResult(
            overall_omega=float(overall),
            components=components,
            lenses=lens_results,
        )