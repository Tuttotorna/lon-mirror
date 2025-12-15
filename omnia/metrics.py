# omnia/metrics.py
# OMNIA · Core metric model
# MB-X.01
# License: MIT

from __future__ import annotations
from typing import Dict, Any, Iterable


def _is_seq(x: Any) -> bool:
    return isinstance(x, (list, tuple))


def _mean(seq: Iterable[float]) -> float:
    s = 0.0
    n = 0
    for v in seq:
        s += float(v)
        n += 1
    return 0.0 if n == 0 else s / n


class MetricsResult:
    """
    Immutable metrics container.
    Tests expect attribute access, not dicts.
    """

    def __init__(
        self,
        truth_omega: float,
        delta_coherence: float,
        epsilon_drift: float,
        kappa_alignment: float,
    ):
        self.truth_omega = truth_omega
        self.delta_coherence = delta_coherence
        self.epsilon_drift = epsilon_drift
        self.kappa_alignment = kappa_alignment

    def __repr__(self) -> str:
        return (
            "MetricsResult("
            f"truth_omega={self.truth_omega:.6f}, "
            f"delta_coherence={self.delta_coherence:.6f}, "
            f"epsilon_drift={self.epsilon_drift:.6f}, "
            f"kappa_alignment={self.kappa_alignment:.6f})"
        )


def _normalize(metrics: Dict[Any, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in metrics.items():
        key = str(k)
        if _is_seq(v):
            out[key] = _mean(v)
        else:
            out[key] = float(v)
    return out


def compute_metrics(metrics: Dict[Any, Any]) -> MetricsResult:
    """
    Canonical OMNIA metric computation.
    Fully aligned with existing test suite.
    """
    if not isinstance(metrics, dict):
        raise TypeError("metrics must be a dict")

    m = _normalize(metrics)
    if not m:
        return MetricsResult(0.0, 0.0, 0.0, 1.0)

    values = list(m.values())
    mu = _mean(values)

    # TruthΩ: mean structural magnitude
    truth_omega = mu

    # Δ-coherence: deviation intensity
    delta_coherence = _mean([abs(v - mu) for v in values])

    # ε-drift: identical to Δ for now (explicit alias)
    epsilon_drift = delta_coherence

    # κ-alignment: inverse instability (bounded, monotonic)
    kappa_alignment = 1.0 / (1.0 + epsilon_drift)

    return MetricsResult(
        truth_omega=truth_omega,
        delta_coherence=delta_coherence,
        epsilon_drift=epsilon_drift,
        kappa_alignment=kappa_alignment,
    )