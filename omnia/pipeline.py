# MB-X.01 / OMNIA — Structural Pipeline Executor
# Massimiliano Brighindi
#
# Purpose:
#   Execute the structural measurement chain:
#   Ω → Ω̂ → SEI → IRI → OMNIA-LIMIT
#
# Properties:
#   - post-hoc
#   - deterministic
#   - model-agnostic
#   - no policy / no retry / no optimization

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from omnia.omega_set import omega_set, OmegaSetResult
from omnia.sei import sei_series, SEIResult
from omnia.iri import iri_cycle, IRIResult


@dataclass(frozen=True)
class PipelineResult:
    omega_series: List[float]
    omega_set: OmegaSetResult
    sei: SEIResult
    iri: Optional[IRIResult]
    limit_reached: bool
    note: str


def run_structural_pipeline(
    omega_series: Iterable[float],
    cost_series: Iterable[float],
    *,
    hysteresis_pair: Optional[tuple[float, float]] = None,
    smooth_window: int = 0,
    flatness_window: int = 0,
) -> PipelineResult:
    """
    Execute OMNIA structural pipeline.

    Inputs:
      omega_series: Ω values over steps
      cost_series: cumulative cost over steps (monotone)
      hysteresis_pair: optional (Ω(A), Ω(A')) for IRI
      smooth_window: optional smoothing for SEI
      flatness_window: window for SEI flatness

    Returns:
      PipelineResult with STOP (limit_reached) as epistemic declaration.
    """
    omega_list = list(omega_series)
    cost_list = list(cost_series)

    if len(omega_list) != len(cost_list):
        raise ValueError("omega_series and cost_series must have the same length")

    # Ω̂ (residue under transformations)
    omega_hat = omega_set(omega_list)

    # SEI (marginal yield)
    sei_res = sei_series(
        omega_list,
        cost_list,
        smooth_window=smooth_window,
        flatness_window=flatness_window,
    )

    # IRI (optional hysteresis)
    iri_res: Optional[IRIResult] = None
    if hysteresis_pair is not None:
        omega_a, omega_a_prime = hysteresis_pair
        iri_res = iri_cycle(omega_a=omega_a, omega_a_prime=omega_a_prime)

    # Epistemic STOP condition (no thresholds, only logical conjunction)
    sei_flat = (
        sei_res.flatness is not None and sei_res.flatness <= 0.0
    )
    iri_pos = iri_res is not None and iri_res.iri_abs > 0.0

    limit = sei_flat and iri_pos

    note = "OK"
    if limit:
        note = "OMNIA_LIMIT_REACHED"

    return PipelineResult(
        omega_series=omega_list,
        omega_set=omega_hat,
        sei=sei_res,
        iri=iri_res,
        limit_reached=limit,
        note=note,
    )