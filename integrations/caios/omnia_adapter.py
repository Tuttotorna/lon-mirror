"""
OMNIA → CAIOS/CPOL Adapter (pre-oscillation triage)

Goal:
- Provide a fast, model-agnostic "need_oscillation" gate for CPOL/Orchestrator.
- OMNIA computes a unified Ω-score from structural lenses:
  BASE (PBII), TIME (regime drift), CAUSA (lagged coupling), TOKEN (PBII-z),
  plus optional LCR external checks.

Usage:
- Call omnia_cpol_gate(...) before CPOL oscillation.
- If need_oscillation == False => skip CPOL and proceed normally.
- If True => enable CPOL (or a higher-precision mode) only when needed.

Author: Massimiliano Brighindi (MB-X.01 / OMNIA_TOTALE)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Mapping


# --- OMNIA imports (repo-local) ---
try:
    # preferred API (from your README)
    from omnia import omnia_totale_score
except Exception:
    # fallback: direct engine call if API name differs
    from omnia.engine import run_omnia_totale as omnia_totale_score  # type: ignore


@dataclass
class CPOLGateResult:
    need_oscillation: bool
    omega_total: float
    components: Dict[str, float]
    thresholds: Dict[str, float]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def omnia_cpol_gate(
    *,
    # optional numeric anchor (e.g. final answer integer, key intermediate)
    n: Optional[int] = None,
    # optional time series (drift/regime)
    series: Optional[Iterable[float]] = None,
    # optional multichannel series (causal/lag structure)
    series_dict: Optional[Mapping[str, Iterable[float]]] = None,
    # optional tokens for TOKEN lens
    tokens: Optional[List[str]] = None,
    token_numbers: Optional[List[int]] = None,
    # optional external checks (LCR) - pass precomputed score if available
    omega_ext: Optional[float] = None,
    # --- gating thresholds ---
    omega_gate: float = 0.65,
    token_gate: float = 0.55,
    lcr_gate: float = 0.55,
    # --- lens weights (keep simple; tune later) ---
    w_base: float = 1.0,
    w_tempo: float = 1.0,
    w_causa: float = 1.0,
    w_token: float = 1.0,
) -> CPOLGateResult:
    """
    Returns a gate decision for CPOL:
    - need_oscillation == True when Ω_total is high (instability/contradiction risk)
      OR TOKEN instability is high OR external LCR score is low.

    Notes:
    - Thresholds are intentionally conservative defaults.
    - You can tune them per-domain (physics/medicine => lower false positives).
    """

    extra: Dict[str, Any] = {}
    if tokens is not None:
        extra["tokens"] = tokens
    if token_numbers is not None:
        extra["token_numbers"] = token_numbers

    # Compute OMNIA Ω_total and components
    res = omnia_totale_score(
        n=n,
        series=series,
        series_dict=series_dict,
        w_base=w_base,
        w_tempo=w_tempo,
        w_causa=w_causa,
        w_token=w_token,
        extra=extra if extra else None,
    )

    # Normalize expected interface across versions
    omega_total = float(getattr(res, "omega_score", getattr(res, "omega_total", 0.0)))
    components = dict(getattr(res, "components", getattr(res, "lens_scores", {})) or {})
    metadata = dict(getattr(res, "metadata", getattr(res, "lens_metadata", {})) or {})

    # Extract token component name variants
    token_score = None
    for key in ("TOKEN", "token", "omniatoken"):
        if key in components:
            token_score = float(components[key])
            break

    # External check rule (LCR): low Ω_ext => inconsistency detected
    ext_trip = False
    if omega_ext is not None:
        ext_trip = float(omega_ext) < lcr_gate

    # Decision logic: triage before CPOL
    omega_trip = omega_total >= omega_gate
    token_trip = (token_score is not None) and (token_score >= token_gate)

    need_oscillation = bool(omega_trip or token_trip or ext_trip)

    return CPOLGateResult(
        need_oscillation=need_oscillation,
        omega_total=omega_total,
        components={k: float(v) for k, v in components.items()},
        thresholds={
            "omega_gate": float(omega_gate),
            "token_gate": float(token_gate),
            "lcr_gate": float(lcr_gate),
        },
        metadata={
            "omnia_metadata": metadata,
            "inputs_present": {
                "n": n is not None,
                "series": series is not None,
                "series_dict": series_dict is not None,
                "tokens": tokens is not None,
                "token_numbers": token_numbers is not None,
                "omega_ext": omega_ext is not None,
            },
            "trips": {
                "omega_trip": omega_trip,
                "token_trip": token_trip,
                "ext_trip": ext_trip,
            },
        },
    )