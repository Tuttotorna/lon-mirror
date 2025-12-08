"""
OMNIA_TOTALE_API_v1.0 — Plug-and-play supervision layer for LLMs
Author: Massimiliano Brighindi (MB-X.01 / OMNIA_TOTALE) + MBX IA

Purpose
-------
Thin, model-agnostic API to apply OMNIA_TOTALE scoring to LLM runs and emit
unified JSON logs.

Features
--------
- Uses PBII (multi-base) as core instability signal.
- Optional hook to OMNIA_TOTALE_v2.0.omnia_totale_score for full BASE/TIME/CAUSA fusion.
- Per-step Ω_raw / Ω_revised / ΔΩ with adaptive thresholds.
- Chain-level Ω summary and instability flag.
- JSON log schema suitable for supervisors / guardrails.
"""

from __future__ import annotations

import json
import math
import re
import time
import uuid
from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

# ============================================================
# 1. OPTIONAL IMPORT FROM OMNIA_TOTALE v2.0 (if available)
# ============================================================

OMNIA_TOTALE_AVAILABLE = False
try:
    # Adjust the module name if your file is named differently, e.g.:
    # from OMNIA_TOTALE_v2_0 import omnia_totale_score
    from OMNIA_TOTALE_v2_0 import omnia_totale_score  # type: ignore

    OMNIA_TOTALE_AVAILABLE = True
except Exception:
    # Fallback: we'll use PBII-only scoring.
    OMNIA_TOTALE_AVAILABLE = False


# ============================================================
# 2. CORE PBII IMPLEMENTATION (fallback + local use)
# ============================================================

DEFAULT_BASES: Sequence[int] = (
    2,
    3,
    5,
    7,
    11,
    13,
    17,
    19,
    23,
    29,
    31,
    37,
    41,
    43,
    47,
)


def _digits_in_base(n: int, b: int) -> List[int]:
    if n < 0:
        raise ValueError("n must be non-negative")
    if b <= 1:
        raise ValueError("base must be >= 2")
    if n == 0:
        return [0]
    res: List[int] = []
    while n > 0:
        res.append(n % b)
        n //= b
    return res[::-1]


def _sigma_b(n: int, b: int) -> float:
    """
    Base Symmetry Score used by PBII.

    Higher = more regular/structured in base b.
    """
    digits = _digits_in_base(n, b)
    L = len(digits)
    if L == 0:
        return 0.0

    freq = [0] * b
    for d in digits:
        freq[d] += 1

    probs = [c / L for c in freq if c > 0]
    if not probs:
        Hn = 0.0
    else:
        H = -sum(p * math.log2(p) for p in probs)
        Hmax = math.log2(b)
        Hn = H / Hmax if Hmax > 0 else 0.0

    bonus = 0.5 if n % b == 0 else 0.0
    return (1.0 - Hn) / L + bonus


def _sigma_avg(n: int, bases: Sequence[int] = DEFAULT_BASES) -> float:
    return float(sum(_sigma_b(n, b) for b in bases) / len(bases))


def _saturation(n: int, bases: Sequence[int] = DEFAULT_BASES, window: int = 100) -> float:
    start = max(2, n - window)
    comps: List[int] = []
    for k in range(start, n):
        if k <= 3:
            continue
        if any(k % d == 0 for d in range(2, int(math.sqrt(k)) + 1)):
            comps.append(k)
    if not comps:
        return 0.0
    vals = [_sigma_avg(k, bases) for k in comps]
    return float(sum(vals) / len(vals))


def pbii_score(n: int, bases: Sequence[int] = DEFAULT_BASES, window: int = 100) -> float:
    """
    PBII(n) = saturation(composites around n) - sigma_avg(n)

    Higher PBII → more prime-like instability.
    """
    sat = _saturation(n, bases=bases, window=window)
    sig = _sigma_avg(n, bases=bases)
    return float(sat - sig)


# ============================================================
# 3. ADAPTIVE THRESHOLD STATE (ΔΩ statistics)
# ============================================================

@dataclass
class ThresholdState:
    """Online mean/std over ΔΩ to adapt thresholds."""
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0  # sum of squared deviations

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)


# ============================================================
# 4. CONFIG + DATA MODELS
# ============================================================

@dataclass
class OmniaAPIConfig:
    """
    High-level configuration for OMNIA supervisor API.
    """
    bases: Sequence[int] = field(default_factory=lambda: list(DEFAULT_BASES))
    pbii_window: int = 100

    # ΔΩ adaptive thresholding
    sigma_multiplier: float = 1.5   # ΔΩ > mean + k·std → unstable
    min_delta_absolute: float = 0.05

    # Per-step flags
    step_instability_factor: float = 0.7  # step flagged if ΔΩ_step > factor * chain_threshold

    # Metadata
    version: str = "OMNIA_TOTALE_API_v1.0"


@dataclass
class StepOmega:
    index: int
    text: str
    omega_raw: float
    omega_revised: float
    delta_omega: float
    unstable: bool


@dataclass
class ChainOmega:
    omega_raw: float
    omega_revised: float
    delta_omega: float
    unstable: bool


@dataclass
class OmniaRunLog:
    run_id: str
    model_name: str
    prompt: str
    response: str
    steps: List[StepOmega]
    chain: ChainOmega
    timestamp: float
    config: Dict[str, Any]
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=indent)


# ============================================================
# 5. SUPERVISOR CLASS
# ============================================================

class OmniaSupervisor:
    """
    Stateless w.r.t. models, stateful w.r.t. ΔΩ statistics.
    """

    def __init__(self, config: Optional[OmniaAPIConfig] = None) -> None:
        self.config = config or OmniaAPIConfig()
        self.threshold_state = ThresholdState()

    # ----------- public API -----------

    def supervise(
        self,
        prompt: str,
        response: str,
        reasoning_steps: Sequence[str],
        model_name: str = "unknown",
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> OmniaRunLog:
        """
        Main entry point.

        reasoning_steps: list of step texts (e.g. chain-of-thought lines).
        """
        steps_metrics = [
            self._score_step(i, text)
            for i, text in enumerate(reasoning_steps)
        ]

        chain_raw = float(np.mean([s.omega_raw for s in steps_metrics])) if steps_metrics else 0.0
        chain_rev = float(np.mean([s.omega_revised for s in steps_metrics])) if steps_metrics else 0.0
        chain_delta = chain_rev - chain_raw

        # Update adaptive thresholds with chain-level ΔΩ
        self.threshold_state.update(chain_delta)
        chain_threshold = self._current_chain_threshold()

        chain_unstable = self._is_unstable(chain_delta, chain_threshold)

        # Recompute step flags with updated threshold
        updated_steps: List[StepOmega] = []
        for s in steps_metrics:
            step_thresh = chain_threshold * self.config.step_instability_factor
            step_unstable = self._is_unstable(s.delta_omega, step_thresh)
            updated_steps.append(
                StepOmega(
                    index=s.index,
                    text=s.text,
                    omega_raw=s.omega_raw,
                    omega_revised=s.omega_revised,
                    delta_omega=s.delta_omega,
                    unstable=step_unstable,
                )
            )

        chain = ChainOmega(
            omega_raw=chain_raw,
            omega_revised=chain_rev,
            delta_omega=chain_delta,
            unstable=chain_unstable,
        )

        run_log = OmniaRunLog(
            run_id=str(uuid.uuid4()),
            model_name=model_name,
            prompt=prompt,
            response=response,
            steps=updated_steps,
            chain=chain,
            timestamp=time.time(),
            config=asdict(self.config),
            meta=extra_meta or {},
        )
        return run_log

    # ----------- internals -----------

    def _score_step(self, index: int, text: str) -> StepOmega:
        numbers = self._extract_numbers(text)
        if not numbers:
            # No explicit numeric content; treat PBII as neutral.
            omega_raw = 0.0
        else:
            pbii_vals = [pbii_score(n, bases=self.config.bases, window=self.config.pbii_window)
                         for n in numbers]
            # For Ω_raw we invert PBII: high stability → high Ω_raw
            omega_raw = float(-np.mean(pbii_vals))

        # Ω_revised: if full OMNIA_TOTALE is available, call it;
        # otherwise, apply a simple non-linear transform to PBII.
        if OMNIA_TOTALE_AVAILABLE and numbers:
            # Use the last number as anchor for OMNIA_TOTALE demo.
            anchor = int(numbers[-1])
            # Simple synthetic time/causa signals: index-based.
            t = np.arange(32)
            series = np.sin(t / 10.0 + index * 0.1)
            series_dict = {
                "s1": series,
                "s2": np.roll(series, 2),
            }
            res = omnia_totale_score(anchor, series, series_dict)  # type: ignore
            omega_revised = float(res.omega_score)
        else:
            # Fallback: non-linear squashing of raw signal.
            omega_revised = float(math.tanh(omega_raw))

        delta = omega_revised - omega_raw

        return StepOmega(
            index=index,
            text=text,
            omega_raw=omega_raw,
            omega_revised=omega_revised,
            delta_omega=delta,
            unstable=False,  # will be set after chain threshold update
        )

    def _current_chain_threshold(self) -> float:
        mean = self.threshold_state.mean
        std = self.threshold_state.std
        th = mean + self.config.sigma_multiplier * std
        if abs(th) < self.config.min_delta_absolute:
            th = math.copysign(self.config.min_delta_absolute, th if th != 0 else 1.0)
        return th

    @staticmethod
    def _is_unstable(delta_omega: float, threshold: float) -> bool:
        if threshold == 0:
            return False
        return delta_omega > threshold

    @staticmethod
    def _extract_numbers(text: str) -> List[int]:
        return [int(m) for m in re.findall(r"\b\d+\b", text)]


# ============================================================
# 6. CONVENIENCE FUNCTION + DEMO
# ============================================================

def run_omnia_supervision(
    prompt: str,
    response: str,
    reasoning_steps: Sequence[str],
    model_name: str = "unknown",
    config: Optional[OmniaAPIConfig] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> OmniaRunLog:
    """
    Convenience wrapper: one-shot supervision call.
    """
    supervisor = OmniaSupervisor(config=config)
    return supervisor.supervise(
        prompt=prompt,
        response=response,
        reasoning_steps=reasoning_steps,
        model_name=model_name,
        extra_meta=extra_meta,
    )


def demo() -> None:
    """
    Minimal end-to-end demo: build a fake chain-of-thought and print JSON log.
    """
    prompt = "Compute the total and check if the final answer is prime."
    response = "The final answer is 173."
    steps = [
        "We add 80 + 90 + 3 = 173.",
        "Check divisibility of 173 by small primes 2,3,5,7,11,13.",
        "No divisor found, so 173 is prime.",
    ]

    run_log = run_omnia_supervision(
        prompt=prompt,
        response=response,
        reasoning_steps=steps,
        model_name="demo-llm",
        extra_meta={"dataset": "synthetic-demo"},
    )
    print(run_log.to_json())


if __name__ == "__main__":
    demo()
```0