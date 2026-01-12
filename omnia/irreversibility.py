from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import math
import statistics
import time


@dataclass
class IRIRecord:
    ts: float
    context: str
    step: int

    # Observed forward change (in state space)
    forward_distance: float

    # "Return loss" after inverse-projection (hysteresis residue)
    hysteresis_residue: float

    # Irreversibility Index in [0,1]
    iri: float

    # Diagnostics
    iri_z: Optional[float]
    trend: str  # "rising" | "flat" | "declining"
    hysteresis_detected: bool


def _safe_div(num: float, den: float, eps: float = 1e-12) -> float:
    return num / (den + eps)


def _euclid(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("State vectors must have same length")
    s = 0.0
    for x, y in zip(a, b):
        d = float(x) - float(y)
        s += d * d
    return math.sqrt(s)


def _robust_zscore(x: float, window: List[float]) -> Optional[float]:
    if len(window) < 5:
        return None
    med = statistics.median(window)
    mad = statistics.median([abs(v - med) for v in window])
    if mad < 1e-12:
        return None
    return 0.6745 * (x - med) / mad


def _trend_label(values: List[float], flat_eps: float = 0.03) -> str:
    if len(values) < 3:
        return "flat"
    last = values[-1]
    prev = values[:-1]
    m = statistics.median(prev)
    if m <= 0:
        if last > 1e-9:
            return "rising"
        return "flat"
    rel = (last - m) / m
    if rel > flat_eps:
        return "rising"
    if rel < -flat_eps:
        return "declining"
    return "flat"


class IrreversibilityEngine:
    """
    Layer-2: Irreversibility / Hysteresis Layer (IHL)

    Measures whether the *path* has become non-reversible in a structural sense.

    We treat each step as a state vector s_t in R^n.
    Forward move: s_t -> s_{t+1}
    If the system were perfectly reversible under the admissible transforms,
    an inverse-projection should map s_{t+1} back to s_t.

    We approximate inverse-projection using local linear regression
    over the last W states (path-based inverse estimate), and measure residue.

    IRI = residue / forward_distance, clamped to [0,1] via ratio bound.
    Trend-only. No thresholds required to "stop" anything.
    """

    def __init__(
        self,
        window: int = 7,
        flat_eps: float = 0.03,
        detect_iri: float = 0.25,
        detect_trend: str = "rising",
        eps: float = 1e-12,
    ):
        self.window = window
        self.flat_eps = flat_eps
        self.detect_iri = detect_iri
        self.detect_trend = detect_trend
        self.eps = eps

        self._states: List[List[float]] = []
        self._iri_hist: List[float] = []

    def _inverse_projection(self, t: int) -> List[float]:
        """
        Predict s_t from s_{t+1} using local affine map estimated on recent pairs.

        Fit per-dimension y ~= a*x + b from pairs (x = s_{k+1}[j], y = s_k[j])
        for k in recent window. This is a cheap, stable inverse estimate.

        Returns projected state s'_t.
        """
        # Need at least 3 transitions for stable regression
        start = max(0, t - self.window)
        end = t  # use pairs up to t-1 -> t
        if end - start < 3:
            # fallback: identity (no inverse knowledge)
            return list(self._states[t + 1])

        s_next = self._states[t + 1]
        dim = len(s_next)

        proj = []
        for j in range(dim):
            xs = []
            ys = []
            for k in range(start, end):
                xs.append(float(self._states[k + 1][j]))
                ys.append(float(self._states[k][j]))

            # simple least squares for y = a*x + b
            x_mean = sum(xs) / len(xs)
            y_mean = sum(ys) / len(ys)
            num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
            den = sum((x - x_mean) ** 2 for x in xs)

            if abs(den) < 1e-12:
                a = 0.0
            else:
                a = num / den
            b = y_mean - a * x_mean

            proj.append(a * float(s_next[j]) + b)

        return proj

    def add_state(self, context: str, state: Sequence[float]) -> Optional[IRIRecord]:
        """
        Add a new state. Returns an IRIRecord once we have at least 2 states.
        """
        st = [float(x) for x in state]
        if self._states and len(st) != len(self._states[0]):
            raise ValueError("All states must have identical dimension")

        self._states.append(st)
        step = len(self._states)

        # Need at least 2 states to compute forward distance
        if len(self._states) < 2:
            return None

        t = len(self._states) - 2  # compare s_t to s_{t+1}
        s_t = self._states[t]
        s_t1 = self._states[t + 1]

        forward_distance = _euclid(s_t, s_t1)

        # inverse projection from s_{t+1} back to t
        s_t_proj = self._inverse_projection(t)
        residue = _euclid(s_t, s_t_proj)

        iri = _safe_div(residue, forward_distance, eps=self.eps)
        # Bound to [0,1] as a diagnostic ratio (if residue > forward, treat as saturated)
        iri = max(0.0, min(1.0, iri))

        self._iri_hist.append(iri)
        if len(self._iri_hist) > max(self.window * 6, 60):
            self._iri_hist = self._iri_hist[-max(self.window * 6, 60):]

        recent = self._iri_hist[-self.window:]
        trend = _trend_label(recent, flat_eps=self.flat_eps)
        iri_z = _robust_zscore(iri, recent)

        hysteresis_detected = (iri >= self.detect_iri) and (trend == self.detect_trend)

        return IRIRecord(
            ts=time.time(),
            context=context,
            step=step,
            forward_distance=forward_distance,
            hysteresis_residue=residue,
            iri=iri,
            iri_z=iri_z,
            trend=trend,
            hysteresis_detected=hysteresis_detected,
        )

    def snapshot(self) -> Dict:
        last = self._iri_hist[-self.window:]
        return {
            "window": self.window,
            "last_window": last,
            "trend": _trend_label(last, flat_eps=self.flat_eps),
        }