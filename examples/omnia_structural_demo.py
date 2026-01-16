"""
OMNIA — Structural Demo (SEI + IRI + Omega-hat)

Purpose
- Minimal executable artifact to demonstrate OMNIA's core claim:
  measure structural invariants and structural limits (not semantics, not decisions).

What it prints
- Ω̂ (omega-hat): invariant residue estimate under deterministic transformations
- SEI: Saturation / Exhaustion Index (marginal structural yield vs cost)
- IRI: Irreversibility / Hysteresis Index (loss of recoverability)

This is intentionally lightweight and dependency-minimal.
"""

from __future__ import annotations

import math
import statistics
import zlib
from dataclasses import dataclass
from typing import Dict, List, Tuple


# -----------------------------
# Utilities
# -----------------------------

def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default


def _normalize_text(s: str) -> str:
    # deterministic normalization, no language assumptions
    return " ".join(s.strip().split())


def _simple_distance(a: str, b: str) -> float:
    # lightweight character-level distance proxy (0..1)
    # not a semantic metric; purely representational drift
    if a == b:
        return 0.0
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 0.0
    # normalized length delta + mismatches in overlap
    m = min(la, lb)
    mism = sum(1 for i in range(m) if a[i] != b[i])
    return _clamp01((_safe_div(mism, max(m, 1)) * 0.7) + (_safe_div(abs(la - lb), max(la, lb, 1)) * 0.3))


def _compress_len(s: str) -> int:
    return len(zlib.compress(s.encode("utf-8"), level=9))


def _permute_deterministic(s: str) -> str:
    # deterministic permutation: reverse words, then reverse characters within words
    words = s.split()
    words = list(reversed(words))
    words = [w[::-1] for w in words]
    return " ".join(words)


def _constrain_ascii(s: str) -> str:
    # deterministic constraint: drop non-ascii
    return "".join(ch for ch in s if 32 <= ord(ch) <= 126)


def _tokenize_naive(s: str) -> List[str]:
    # deterministic split; avoids external tokenizers
    return s.split()


# -----------------------------
# Core signals (Ω̂, SEI, IRI)
# -----------------------------

@dataclass(frozen=True)
class OmegaHatResult:
    omega_hat: float
    residue_text: str
    per_lens_dist: Dict[str, float]
    per_lens_compress: Dict[str, int]


@dataclass(frozen=True)
class SEIResult:
    sei: float
    marginal_yield: float
    cost_units: float
    trend: str


@dataclass(frozen=True)
class IRIResult:
    iri: float
    recoverability: float
    forward_drift: float
    return_drift: float
    note: str


def omega_hat(text: str) -> OmegaHatResult:
    """
    Ω̂ = invariant residue estimate: what remains stable across representations.
    We build multiple deterministic representations and measure their mutual drift.
    Lower drift -> higher invariance -> higher Ω̂.

    Ω̂ is not "truth". It's a structural stability estimate under transformations.
    """
    base = _normalize_text(text)

    reps: Dict[str, str] = {
        "base": base,
        "compression_proxy": base,  # compression measured separately
        "permutation": _permute_deterministic(base),
        "constraint_ascii": _constrain_ascii(base),
        "tokenization_join": " ".join(_tokenize_naive(base)),
    }

    # pairwise drift from base
    per_dist: Dict[str, float] = {}
    per_comp: Dict[str, int] = {}
    for k, v in reps.items():
        per_dist[k] = _simple_distance(base, v)
        per_comp[k] = _compress_len(v)

    # invariance is inverse of average drift
    avg_drift = statistics.mean(per_dist.values()) if per_dist else 1.0
    omega = _clamp01(1.0 - avg_drift)

    # produce a minimal "residue" representation (most stable proxy)
    # choose the rep with minimal drift from base
    best_k = min(per_dist.keys(), key=lambda x: per_dist[x])
    residue = reps[best_k]

    return OmegaHatResult(
        omega_hat=omega,
        residue_text=residue,
        per_lens_dist=per_dist,
        per_lens_compress=per_comp,
    )


def sei_from_series(omega_series: List[float], cost_series: List[float]) -> SEIResult:
    """
    SEI measures marginal structural yield per unit cost.

    We interpret omega_series as "structural yield" over increasing computation steps,
    and cost_series as monotonic cost proxy (tokens, iterations, latency, etc.).

    SEI is a trend metric: it does NOT decide; it reports saturation tendency.
    """
    if len(omega_series) < 2 or len(cost_series) < 2 or len(omega_series) != len(cost_series):
        return SEIResult(sei=0.0, marginal_yield=0.0, cost_units=0.0, trend="invalid_series")

    # use last step marginal slope
    dy = omega_series[-1] - omega_series[-2]
    dx = cost_series[-1] - cost_series[-2]
    slope = _safe_div(dy, dx, default=0.0)

    # convert slope to a bounded saturation index:
    # high slope -> low saturation; low slope -> high saturation
    # SEI close to 1 means "flattening".
    sei = _clamp01(1.0 - _clamp01(slope * 10.0))

    # simple trend label using last 3 slopes if available
    trend = "flat"
    if len(omega_series) >= 3:
        slopes = []
        for i in range(1, len(omega_series)):
            dy_i = omega_series[i] - omega_series[i - 1]
            dx_i = cost_series[i] - cost_series[i - 1]
            slopes.append(_safe_div(dy_i, dx_i, default=0.0))
        last3 = slopes[-3:] if len(slopes) >= 3 else slopes
        m = statistics.mean(last3) if last3 else 0.0
        if m > 0.01:
            trend = "growing"
        elif m < 0.001:
            trend = "saturating"
        else:
            trend = "flat"

    return SEIResult(sei=sei, marginal_yield=slope, cost_units=dx, trend=trend)


def iri_hysteresis(text0: str, text1: str) -> IRIResult:
    """
    IRI detects irreversibility:
    - forward transformation from text0 -> text1
    - attempt to return to a simpler state does not restore the same structure

    Here we simulate:
    - forward drift: distance(text0, text1)
    - return drift: distance(text0, normalize(text1))  (a "simplification attempt")
    If return drift remains close to forward drift, recoverability is low -> higher IRI.
    """
    a = _normalize_text(text0)
    b = _normalize_text(text1)

    forward = _simple_distance(a, b)

    # "return" attempt: drop non-ascii + normalize spaces (structural simplification proxy)
    b_return = _normalize_text(_constrain_ascii(b))
    ret = _simple_distance(a, b_return)

    # recoverability: how much drift is removed by return attempt
    recover = _clamp01(1.0 - _safe_div(ret, max(forward, 1e-9), default=1.0))

    # IRI high when recoverability is low AND forward drift exists
    iri = _clamp01(forward * (1.0 - recover))

    note = "ok"
    if forward < 0.02:
        note = "no_change_detected"
    elif iri > 0.25:
        note = "irreversibility_detected"

    return IRIResult(
        iri=iri,
        recoverability=recover,
        forward_drift=forward,
        return_drift=ret,
        note=note,
    )


# -----------------------------
# Demo runner
# -----------------------------

def main() -> None:
    print("=== OMNIA STRUCTURAL DEMO (SEI + IRI + Ω̂) ===")

    # Example text0/text1: simulate "same answer, higher cost, lower structure" scenario
    text0 = """
    The result is 42.
    """
    text1 = """
    The result is 42.

    Step 1: restate the problem
    Step 2: restate the constraints
    Step 3: restate the same conclusion again
    Step 4: add redundant framing and extra tokens
    Final: 42
    """

    o0 = omega_hat(text0)
    o1 = omega_hat(text1)

    print("\n[Ω̂] Omega-hat (invariant residue estimate)")
    print("omega_hat(text0) =", round(o0.omega_hat, 6))
    print("omega_hat(text1) =", round(o1.omega_hat, 6))
    print("note: lower invariance under representation -> lower Ω̂")

    # SEI: treat "cost" as length proxy
    omega_series = [o0.omega_hat, o1.omega_hat]
    cost_series = [float(len(_normalize_text(text0))), float(len(_normalize_text(text1)))]
    s = sei_from_series(omega_series, cost_series)

    print("\n[SEI] Saturation / Exhaustion Index (marginal structural yield vs cost)")
    print("cost step =", s.cost_units)
    print("marginal_yield =", round(s.marginal_yield, 9))
    print("SEI =", round(s.sei, 6), "| trend =", s.trend)
    print("note: SEI is trend-only, not a stop rule")

    # IRI: irreversibility/hysteresis detection
    h = iri_hysteresis(text0, text1)
    print("\n[IRI] Irreversibility / Hysteresis Index")
    print("forward_drift =", round(h.forward_drift, 6))
    print("return_drift  =", round(h.return_drift, 6))
    print("recoverability =", round(h.recoverability, 6))
    print("IRI =", round(h.iri, 6), "| note =", h.note)

    # lens diagnostics (small, readable)
    print("\n[LENS] per-lens drift from base for text1")
    for k in sorted(o1.per_lens_dist.keys()):
        print(f"{k:>16}  drift={o1.per_lens_dist[k]:.6f}  comp_len={o1.per_lens_compress[k]}")

    print("\n=== END ===")


if __name__ == "__main__":
    main()