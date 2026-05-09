# MB-X.01 / OMNIA — SEI (Saturation / Exhaustion Index)
# Massimiliano Brighindi
#
# Purpose:
#   Measure marginal structural yield: Delta Omega / Delta Cost.
#   - post-hoc
#   - deterministic
#   - model-agnostic
#   - semantics-free
#
# SEI(k) = (Omega(k) - Omega(k-1)) / (Cost(k) - Cost(k-1) + eps)
#
# Boundary:
#   measurement != inference != decision

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

    output: List[Optional[float]] = [None] * len(values)
    buffer: List[float] = []

    for i, value in enumerate(values):
        if value is None:
            output[i] = None
            continue

        buffer.append(float(value))

        if len(buffer) > window:
            buffer.pop(0)

        output[i] = sum(buffer) / len(buffer)

    return output


def sei_series(
    omega: Iterable[float],
    cost: Iterable[float],
    eps: float = 1e-12,
    smooth_window: int = 0,
    flatness_window: int = 0,
) -> SEIResult:
    """
    Compute SEI over paired sequences.

    Inputs:
      omega[k]: structural score at step k
      cost[k]: cumulative cost at step k

    Returns:
      SEIResult:
        - sei: [None, sei(1), ..., sei(n-1)]
        - sei_smooth: optional moving average
        - flatness: optional mean absolute SEI over final window
    """
    omega_values = [float(x) for x in omega]
    cost_values = [float(x) for x in cost]

    if len(omega_values) != len(cost_values):
        raise ValueError(
            f"omega and cost must have the same length "
            f"(got {len(omega_values)} vs {len(cost_values)})"
        )

    if len(omega_values) < 2:
        return SEIResult(
            sei=[None] * len(omega_values),
            sei_smooth=None,
            flatness=None,
        )

    sei: List[Optional[float]] = [None]

    for k in range(1, len(omega_values)):
        delta_omega = omega_values[k] - omega_values[k - 1]
        delta_cost = cost_values[k] - cost_values[k - 1]

        if delta_cost <= 0:
            sei.append(0.0)
        else:
            sei.append(delta_omega / (delta_cost + eps))

    sei_smooth: Optional[List[Optional[float]]] = None

    if smooth_window and smooth_window > 1:
        sei_smooth = _moving_average(sei, smooth_window)

    flatness: Optional[float] = None

    if flatness_window and flatness_window > 1:
        source = sei_smooth if sei_smooth is not None else sei
        tail: List[float] = []

        for value in reversed(source):
            if value is None:
                continue

            tail.append(abs(float(value)))

            if len(tail) >= flatness_window:
                break

        if tail:
            flatness = sum(tail) / len(tail)

    return SEIResult(
        sei=sei,
        sei_smooth=sei_smooth,
        flatness=flatness,
    )


class SEI:
    """
    Compatibility wrapper for legacy tests.

    Expected usage:
        from omnia.sei import SEI
        sei = SEI(window=1, eps=1e-12)
        sei.curve(omega_values, cost_values)

    Boundary:
        measurement != inference != decision
    """

    def __init__(self, window: int = 1, eps: float = 1e-12) -> None:
        self.window = max(1, int(window))
        self.eps = float(eps)

    def curve(self, omega_values: Iterable[float], cost_values: Iterable[float]) -> List[float]:
        result = sei_series(
            omega=omega_values,
            cost=cost_values,
            eps=self.eps,
            smooth_window=self.window,
            flatness_window=0,
        )

        source = result.sei_smooth if result.sei_smooth is not None else result.sei

        return [float(x) for x in source if x is not None]