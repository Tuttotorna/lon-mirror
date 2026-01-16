# MB-X.01 / OMNIA — SEI (Saturation / Exhaustion Index)
# Massimiliano Brighindi
#
# Purpose:
#   Measure marginal structural yield: ΔΩ / ΔC
#   - post-hoc
#   - deterministic
#   - model-agnostic
#   - no semantics, no policy, no stop rules
#
# SEI(k) = (Ω(k) - Ω(k-1)) / (C(k) - C(k-1) + eps)

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class SEIResult:
    sei: List[Optional[float]]
    sei_smooth: Optional[List[Optional[float]]]
    flatness: Optional[float]


def _moving_average(values: List[Optional[float]], window: int) -> List[Optional[float]]:
    if window <= 1:
        return values[:]
    out: List[Optional[float]] = [None] * len(values)
    buf: List[float] = []
    for i, v in enumerate(values):
        if v is None:
            out[i] = None
            continue
        buf.append(v)
        if len(buf) > window:
            buf.pop(0)
        out[i] = sum(buf) / len(buf)
    return out


def sei_series(
    omega: Iterable[float],
    cost: Iterable[float],
    eps: float = 1e-12,
    smooth_window: int = 0,
    flatness_window: int = 0,
) -> SEIResult:
    """
    Compute SEI over paired sequences (omega, cost).

    Inputs:
      omega[k]: structural score at step k (e.g., Ω-total)
      cost[k]: cumulative cost at step k (tokens, iterations, latency, etc.), must be non-decreasing

    Returns:
      SEIResult with:
        - sei: [None, sei(1), ..., sei(n-1)]
        - sei_smooth: optional moving average of sei (if smooth_window>1)
        - flatness: optional mean(|sei_smooth|) over last flatness_window valid points
    """
    o = list(omega)
    c = list(cost)

    if len(o) != len(c):
        raise ValueError(f"omega and cost must have the same length (got {len(o)} vs {len(c)})")
    if len(o) < 2:
        return SEIResult(sei=[None] * len(o), sei_smooth=None, flatness=None)

    sei: List[Optional[float]] = [None]
    for k in range(1, len(o)):
        dO = o[k] - o[k - 1]
        dC = c[k] - c[k - 1]
        if dC <= 0:
            # cost must be strictly increasing to interpret marginal yield;
            # if not, treat as 0 yield (or consider raising)
            sei.append(0.0)
        else:
            sei.append(dO / (dC + eps))

    sei_smooth: Optional[List[Optional[float]]] = None
    if smooth_window and smooth_window > 1:
        sei_smooth = _moving_average(sei, smooth_window)

    flatness: Optional[float] = None
    if flatness_window and flatness_window > 1:
        src = sei_smooth if sei_smooth is not None else sei
        tail: List[float] = []
        for v in reversed(src):
            if v is None:
                continue
            tail.append(abs(v))
            if len(tail) >= flatness_window:
                break
        if tail:
            flatness = sum(tail) / len(tail)

    return SEIResult(sei=sei, sei_smooth=sei_smooth, flatness=flatness)