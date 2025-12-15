# omnia/metrics.py
# OMNIA — metrics (TruthΩ, Δ, κ, ε)
# MB-X.01 / OMNIA_TOTALE
#
# Design goal:
# - No heavy deps (no numpy)
# - Stable, deterministic, test-friendly
# - Provide the symbols that __init__.py / tests expect:
#   truth_omega, co_plus, score_plus, delta_coherence, kappa_alignment, epsilon_drift

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union


Number = Union[int, float]


# -----------------------------
# Helpers
# -----------------------------
def _clamp01(x: float) -> float:
    if x != x:  # NaN
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _safe_log(x: float, eps: float = 1e-12) -> float:
    return math.log(max(eps, x))


def _tokenize(text: str) -> List[str]:
    # conservative tokenizer: lowercase alnum words
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    inter = len(sa.intersection(sb))
    union = len(sa.union(sb))
    return 0.0 if union == 0 else inter / union


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _stdev(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(max(0.0, var))


# -----------------------------
# Core metrics
# -----------------------------
def truth_omega(
    x: Union[Number, str, Dict[str, Any]],
    y: Optional[Union[str, Dict[str, Any]]] = None,
    *,
    eps: float = 1e-12,
) -> float:
    """
    TruthΩ = -log( coherence )

    Accepts:
    - numeric coherence in [0..1]: truth_omega(0.8)
    - text similarity coherence via Jaccard overlap: truth_omega("a b", "a c")
    - dict with 'coherence' key: truth_omega({'coherence': 0.7})

    Returns:
    - 0 when coherence == 1
    - +inf-ish for coherence -> 0 (capped by eps)
    """
    coh: float

    # numeric coherence
    if isinstance(x, (int, float)) and y is None:
        coh = _clamp01(float(x))

    # dict coherence
    elif isinstance(x, dict) and y is None:
        if "coherence" not in x:
            raise KeyError("truth_omega(dict) requires key 'coherence'")
        coh = _clamp01(float(x["coherence"]))

    # text-to-text coherence (default)
    elif isinstance(x, str) and isinstance(y, str):
        coh = _clamp01(_jaccard(_tokenize(x), _tokenize(y)))

    # dict-to-dict or dict-to-text (fallbacks)
    elif isinstance(x, dict) and isinstance(y, dict):
        cx = _clamp01(float(x.get("coherence", 0.0)))
        cy = _clamp01(float(y.get("coherence", 0.0)))
        coh = _clamp01(1.0 - abs(cx - cy))  # alignment as coherence proxy
    elif isinstance(x, str) and isinstance(y, dict):
        # compare text signature to dict coherence (weak but defined)
        cy = _clamp01(float(y.get("coherence", 0.0)))
        coh = cy
    elif isinstance(x, dict) and isinstance(y, str):
        cx = _clamp01(float(x.get("coherence", 0.0)))
        coh = cx
    else:
        raise TypeError("truth_omega: unsupported input types")

    return -_safe_log(coh, eps=eps)


def co_plus(
    truth: Union[Number, Dict[str, Any]],
    *,
    cap: float = 1.0,
) -> float:
    """
    Co⁺ = exp(-TruthΩ), clamped to [0..cap]
    Accepts:
    - numeric TruthΩ
    - dict containing 'truth_omega'
    """
    if isinstance(truth, dict):
        if "truth_omega" not in truth:
            raise KeyError("co_plus(dict) requires key 'truth_omega'")
        t = float(truth["truth_omega"])
    else:
        t = float(truth)

    c = math.exp(-max(0.0, t))
    if cap <= 0:
        return c
    return min(cap, max(0.0, c))


def score_plus(
    coherence: Union[Number, Dict[str, Any]],
    *,
    info: float = 1.0,
    bias: float = 1.0,
    cap: float = 1.0,
    eps: float = 1e-12,
) -> float:
    """
    Score⁺: a stable composite.
    Default: Score⁺ = clamp01( Co⁺ * info / bias )

    Accepts:
    - numeric coherence in [0..1]
    - dict with 'coherence' or 'truth_omega' or 'co_plus'
    """
    if isinstance(coherence, dict):
        if "co_plus" in coherence:
            c = _clamp01(float(coherence["co_plus"]))
        elif "truth_omega" in coherence:
            c = co_plus(float(coherence["truth_omega"]), cap=cap)
        elif "coherence" in coherence:
            c = co_plus(truth_omega(float(coherence["coherence"]), eps=eps), cap=cap)
        else:
            raise KeyError("score_plus(dict) requires 'co_plus' or 'truth_omega' or 'coherence'")
    else:
        c = co_plus(truth_omega(float(coherence), eps=eps), cap=cap)

    denom = max(eps, float(bias))
    raw = c * float(info) / denom
    return _clamp01(raw if cap == 1.0 else min(cap, max(0.0, raw)))


def delta_coherence(values: Sequence[Number]) -> float:
    """
    Δ-coherence: dispersion measure of a set of coherence values.
    Returns stdev of clamped [0..1] values.
    """
    xs = [_clamp01(float(v)) for v in values]
    return _stdev(xs)


def kappa_alignment(a: Number, b: Number) -> float:
    """
    κ-alignment: 1 - |a-b| on clamped [0..1] scale.
    """
    aa = _clamp01(float(a))
    bb = _clamp01(float(b))
    return _clamp01(1.0 - abs(aa - bb))


def epsilon_drift(prev: Number, curr: Number) -> float:
    """
    ε-drift: absolute change between two scalar states (clamped [0..1]).
    """
    p = _clamp01(float(prev))
    c = _clamp01(float(curr))
    return abs(c - p)


# -----------------------------
# Optional structured result
# -----------------------------
@dataclass(frozen=True)
class Metrics:
    truth_omega: float
    co_plus: float
    score_plus: float
    delta: float = 0.0
    kappa: float = 0.0
    epsilon: float = 0.0


def compute_metrics(
    coherence: Number,
    *,
    cohort: Optional[Sequence[Number]] = None,
    prev: Optional[Number] = None,
    ref: Optional[Number] = None,
    info: float = 1.0,
    bias: float = 1.0,
    eps: float = 1e-12,
) -> Metrics:
    """
    Convenience wrapper used by demos/tests.
    """
    t = truth_omega(float(coherence), eps=eps)
    c = co_plus(t)
    s = score_plus(float(coherence), info=info, bias=bias, eps=eps)

    d = delta_coherence(cohort) if cohort else 0.0
    k = kappa_alignment(coherence, ref) if ref is not None else 0.0
    e = epsilon_drift(prev, coherence) if prev is not None else 0.0
    return Metrics(truth_omega=t, co_plus=c, score_plus=s, delta=d, kappa=k, epsilon=e)


__all__ = [
    "truth_omega",
    "co_plus",
    "score_plus",
    "delta_coherence",
    "kappa_alignment",
    "epsilon_drift",
    "Metrics",
    "compute_metrics",
]