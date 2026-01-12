from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import math
import statistics
import time


@dataclass
class SEIRecord:
    ts: float
    context: str

    # inputs (cost)
    tokens_in: int
    tokens_out: int
    latency_ms: float
    energy_joule: Optional[float]  # can be None; proxy allowed
    iterations: int

    # outputs (benefit)
    delta_quality: float
    delta_uncertainty: float

    # computed
    sei: float
    sei_z: Optional[float]
    trend: str  # "rising" | "flat" | "declining"


def _safe_div(num: float, den: float, eps: float = 1e-12) -> float:
    return num / (den + eps)


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _robust_zscore(x: float, window: List[float]) -> Optional[float]:
    """
    Robust z-score using MAD. Returns None if not enough variance.
    """
    if len(window) < 5:
        return None
    med = statistics.median(window)
    abs_devs = [abs(v - med) for v in window]
    mad = statistics.median(abs_devs)
    if mad < 1e-12:
        return None
    # 0.6745 makes MAD comparable to std under normal assumption
    return 0.6745 * (x - med) / mad


def _trend_label(sei_values: List[float], flat_eps: float = 0.03) -> str:
    """
    Compare last value vs median of previous values.
    flat_eps is relative tolerance.
    """
    if len(sei_values) < 3:
        return "flat"
    last = sei_values[-1]
    prev = sei_values[:-1]
    m = statistics.median(prev)
    if m <= 0:
        # if prior median is ~0, trend is determined by absolute last
        if last > 1e-9:
            return "rising"
        return "flat"

    rel = (last - m) / m
    if rel > flat_eps:
        return "rising"
    if rel < -flat_eps:
        return "declining"
    return "flat"


class SEIEngine:
    """
    Saturation / Exhaustion Index (SEI) engine.

    Measures marginal benefit per marginal cost over a rolling window.
    No decisions. No STOP. Only measurement.
    """

    def __init__(
        self,
        w_quality: float = 1.0,
        w_uncertainty: float = 1.0,
        v_energy: float = 1.0,
        v_tokens: float = 1.0,
        v_latency: float = 1.0,
        v_iterations: float = 1.0,
        window: int = 5,
        flat_eps: float = 0.03,
        eps: float = 1e-12,
    ):
        self.w_quality = w_quality
        self.w_uncertainty = w_uncertainty
        self.v_energy = v_energy
        self.v_tokens = v_tokens
        self.v_latency = v_latency
        self.v_iterations = v_iterations

        self.window = window
        self.flat_eps = flat_eps
        self.eps = eps

        self._sei_history: List[float] = []

    def compute_sei(
        self,
        delta_quality: float,
        delta_uncertainty: float,
        tokens_in: int,
        tokens_out: int,
        latency_ms: float,
        energy_joule: Optional[float],
        iterations: int,
    ) -> float:
        # benefit
        benefit = (self.w_quality * delta_quality) + (self.w_uncertainty * delta_uncertainty)

        # cost
        tokens = float(tokens_in + tokens_out)
        energy = float(energy_joule) if energy_joule is not None else 0.0  # proxy allowed
        cost = (self.v_energy * energy) + (self.v_tokens * tokens) + (self.v_latency * latency_ms) + (self.v_iterations * float(iterations))

        return _safe_div(benefit, cost, eps=self.eps)

    def add(
        self,
        context: str,
        delta_quality: float,
        delta_uncertainty: float,
        tokens_in: int,
        tokens_out: int,
        latency_ms: float,
        energy_joule: Optional[float],
        iterations: int,
    ) -> SEIRecord:
        sei = self.compute_sei(
            delta_quality=delta_quality,
            delta_uncertainty=delta_uncertainty,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            energy_joule=energy_joule,
            iterations=iterations,
        )

        # update rolling history
        self._sei_history.append(sei)
        if len(self._sei_history) > max(self.window * 4, 40):  # keep some tail for stats
            self._sei_history = self._sei_history[-max(self.window * 4, 40):]

        # window for zscore/trend
        recent = self._sei_history[-self.window:]
        trend = _trend_label(recent, flat_eps=self.flat_eps)

        sei_z = _robust_zscore(sei, recent)

        return SEIRecord(
            ts=time.time(),
            context=context,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            energy_joule=energy_joule,
            iterations=iterations,
            delta_quality=delta_quality,
            delta_uncertainty=delta_uncertainty,
            sei=sei,
            sei_z=sei_z,
            trend=trend,
        )

    def last_window(self) -> List[float]:
        return self._sei_history[-self.window:]

    def snapshot(self) -> Dict:
        return {
            "window": self.window,
            "last_window": self.last_window(),
            "trend": _trend_label(self.last_window(), flat_eps=self.flat_eps),
        }