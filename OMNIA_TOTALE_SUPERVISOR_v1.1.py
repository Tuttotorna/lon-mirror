"""
OMNIA_TOTALE_SUPERVISOR v1.1
Adaptive ΔΩ thresholds + per-step causal traces + JSON logs

Author: Massimiliano Brighindi (concepts) + MBX IA (formalization)

Usage (demo):
    python OMNIA_TOTALE_SUPERVISOR_v1.1.py

This file is self-contained: it implements
- PBII-style instability scores on numeric content
- Ω_raw / Ω_adj / ΔΩ adaptive thresholds
- per-step causal traces
- structured JSON logs ready for LLM pipelines
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Dict, Optional

# =========================
# 1. PBII CORE (lightweight)
# =========================


def digits_in_base(n: int, b: int) -> List[int]:
    """Return digits of n in base b (MSB first)."""
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


def _entropy_norm(digits: List[int], base: int) -> float:
    """Normalized Shannon entropy of digits in given base."""
    L = len(digits)
    if L == 0:
        return 0.0
    freq = [0] * base
    for d in digits:
        freq[d] += 1
    probs = [c / L for c in freq if c > 0]
    if not probs:
        return 0.0
    H = -sum(p * math.log2(p) for p in probs)
    Hmax = math.log2(base)
    return H / Hmax if Hmax > 0 else 0.0


def sigma_b(
    n: int,
    b: int,
    length_weight: float = 1.0,
    length_exponent: float = 1.0,
    divisibility_bonus: float = 0.5,
) -> float:
    """
    Base Symmetry Score (PBII-style).

    sigma_b(n) = length_weight * (1 - H_norm) / L^length_exponent
                 + divisibility_bonus * I[n % b == 0]
    """
    digits = digits_in_base(n, b)
    L = len(digits)
    if L == 0:
        return 0.0

    Hn = _entropy_norm(digits, b)
    length_term = length_weight * (1.0 - Hn) / (L ** length_exponent)
    div_term = divisibility_bonus * (1.0 if n % b == 0 else 0.0)
    return float(length_term + div_term)


def sigma_avg(
    n: int,
    bases: Iterable[int] = (2, 3, 5, 7, 11, 13, 17, 19),
) -> float:
    """Average sigma_b over a set of bases."""
    bases = list(bases)
    vals = [sigma_b(n, b) for b in bases]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _is_composite(k: int) -> bool:
    if k <= 3:
        return False
    for d in range(2, int(math.sqrt(k)) + 1):
        if k % d == 0:
            return True
    return False


def saturation(
    n: int,
    bases: Iterable[int],
    window: int = 100,
) -> float:
    """
    Local saturation of sigma_avg on composites near n.
    Used as background level for PBII.
    """
    bases = list(bases)
    start = max(4, n - window)
    stop = max(start + 1, n)
    comps = [k for k in range(start, stop) if _is_composite(k)]
    if not comps:
        return 0.0
    vals = [sigma_avg(k, bases=bases) for k in comps]
    return float(sum(vals) / len(vals))


def pbii(
    n: int,
    bases: Iterable[int] = (2, 3, 5, 7, 11, 13, 17, 19),
    window: int = 100,
) -> float:
    """
    Prime Base Instability Index.

    PBII(n) = saturation(composites near n) - sigma_avg(n)
    Higher PBII ~ more prime-like instability.
    """
    bases = list(bases)
    sat = saturation(n, bases=bases, window=window)
    sig = sigma_avg(n, bases=bases)
    return float(sat - sig)


# =========================
# 2. SUPERVISOR DATA MODEL
# =========================


@dataclass
class StepTrace:
    step_index: int
    text: str
    numbers: List[int]
    omega_raw: float
    omega_adj: float
    delta_omega: float
    threshold: float
    unstable: bool
    reason: str

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ChainSummary:
    chain_id: str
    steps: int
    unstable_steps: int
    omega_mean: float
    omega_std: float
    global_flag: bool

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SupervisorResult:
    chain_id: str
    steps: List[StepTrace]
    summary: ChainSummary

    def to_dict(self) -> Dict:
        return {
            "chain_id": self.chain_id,
            "steps": [s.to_dict() for s in self.steps],
            "summary": self.summary.to_dict(),
        }

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


# =========================
# 3. Ω SCORE PER STEP
# =========================


_NUMBER_RE = re.compile(r"\b\d+\b")


def extract_numbers(text: str) -> List[int]:
    """Extract positive integers > 1 from a text step."""
    return [int(m.group(0)) for m in _NUMBER_RE.finditer(text) if int(m.group(0)) > 1]


def omega_from_numbers(
    nums: List[int],
    bases: Iterable[int] = (2, 3, 5, 7, 11, 13, 17, 19),
    window: int = 100,
) -> float:
    """
    Map a list of numbers in a step to a single Ω_raw score.

    Here Ω_raw is defined as the NEGATED mean PBII:
    - stable (composite-like) ≈ low PBII → higher Ω_raw
    - unstable (prime-like) ≈ high PBII → lower Ω_raw
    """
    if not nums:
        return 0.0
    vals = [pbii(n, bases=bases, window=window) for n in nums]
    mean_pbii = sum(vals) / len(vals)
    return float(-mean_pbii)


# =========================
# 4. ADAPTIVE ΔΩ THRESHOLDS
# =========================


def _ema_update(prev: float, value: float, alpha: float) -> float:
    """Exponential moving average update."""
    if math.isnan(prev):
        return value
    return (1.0 - alpha) * prev + alpha * value


def _var_update(prev_mean: float, prev_var: float, value: float, k: int) -> float:
    """
    Online variance update (Welford-like, population variance).
    k is the current sample count (1-based).
    """
    if k <= 1:
        return 0.0
    delta = value - prev_mean
    new_var = ((k - 2) / (k - 1)) * prev_var + (delta ** 2) / k
    return new_var


def _adaptive_threshold(
    base_threshold: float,
    local_std: float,
    k_var: float,
    min_factor: float = 1.0,
    max_factor: float = 4.0,
) -> float:
    """
    Compute adaptive threshold as:

        thr = base_threshold * clamp(1 + k_var * local_std, min_factor, max_factor)
    """
    factor = 1.0 + k_var * local_std
    factor = max(min_factor, min(max_factor, factor))
    return float(base_threshold * factor)


# =========================
# 5. SUPERVISOR CORE
# =========================


def analyze_chain(
    steps_text: List[str],
    chain_id: Optional[str] = None,
    bases: Iterable[int] = (2, 3, 5, 7, 11, 13, 17, 19),
    window: int = 100,
    base_threshold: float = 0.15,
    ema_alpha: float = 0.2,
    k_var: float = 1.5,
    json_log_dir: Optional[Path] = None,
) -> SupervisorResult:
    """
    Analyze a chain of LLM steps.

    - Computes Ω_raw per step from numeric content.
    - Maintains EMA and variance of Ω_raw across the chain.
    - Computes Ω_adj (centered) and ΔΩ per step.
    - Uses adaptive thresholds based on local_std to flag instability.
    - Produces per-step causal traces and global summary.

    If json_log_dir is provided, a JSON log is written there.
    """
    if chain_id is None:
        chain_id = datetime.utcnow().strftime("chain_%Y%m%dT%H%M%SZ")

    bases = list(bases)

    step_traces: List[StepTrace] = []

    ema = float("nan")
    var = 0.0
    count = 0

    omega_values: List[float] = []

    for idx, text in enumerate(steps_text, start=1):
        nums = extract_numbers(text)
        omega_raw = omega_from_numbers(nums, bases=bases, window=window)

        # Update running stats
        count += 1
        prev_ema = ema if not math.isnan(ema) else omega_raw
        ema = _ema_update(prev_ema, omega_raw, ema_alpha)
        var = _var_update(prev_ema, var, omega_raw, count)
        std = math.sqrt(var)

        # Centered Ω (adj) and ΔΩ
        omega_adj = omega_raw - ema
        delta_omega = abs(omega_adj)

        # Adaptive threshold
        thr = _adaptive_threshold(
            base_threshold=base_threshold,
            local_std=std,
            k_var=k_var,
        )

        unstable = delta_omega >= thr

        # Causal reason (compressed, human+machine readable)
        if not nums:
            reason = "no numeric content; Ω_raw=0.0, treated as stable"
        else:
            if unstable:
                if std > 0.0 and delta_omega > thr * 1.5:
                    reason = (
                        "large ΔΩ spike vs EMA; numeric pattern diverges sharply "
                        "from recent stability window"
                    )
                else:
                    reason = (
                        "ΔΩ above adaptive threshold; numeric content deviates "
                        "from chain baseline"
                    )
            else:
                reason = "ΔΩ below adaptive threshold; numeric content consistent with chain baseline"

        trace = StepTrace(
            step_index=idx,
            text=text,
            numbers=nums,
            omega_raw=omega_raw,
            omega_adj=omega_adj,
            delta_omega=delta_omega,
            threshold=thr,
            unstable=unstable,
            reason=reason,
        )
        step_traces.append(trace)
        omega_values.append(omega_raw)

    # Global summary
    if omega_values:
        mean_omega = float(sum(omega_values) / len(omega_values))
        var_omega = (
            float(sum((x - mean_omega) ** 2 for x in omega_values) / len(omega_values))
            if len(omega_values) > 1
            else 0.0
        )
        std_omega = math.sqrt(var_omega)
    else:
        mean_omega = 0.0
        std_omega = 0.0

    unstable_steps = sum(1 for s in step_traces if s.unstable)
    global_flag = unstable_steps > 0

    summary = ChainSummary(
        chain_id=chain_id,
        steps=len(step_traces),
        unstable_steps=unstable_steps,
        omega_mean=mean_omega,
        omega_std=std_omega,
        global_flag=global_flag,
    )

    result = SupervisorResult(chain_id=chain_id, steps=step_traces, summary=summary)

    # JSON log (optional)
    if json_log_dir is not None:
        json_log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        log_path = json_log_dir / f"{chain_id}_{ts}.json"
        result.to_json(log_path)

    return result


# =========================
# 6. DEMO (self-contained)
# =========================


DEMO_CHAIN = [
    # Stable-ish arithmetic
    "We add 24 apples and 36 apples to get 60 total apples after 3 days.",
    "Each of the 5 friends gets 12 apples, for a total of 60 again.",
    # Slight numeric drift
    "Later, 7 apples are lost, leaving 53 apples in the box.",
    # Strong instability (synthetic hallucination)
    "Then the box magically holds 997 apples after removing 3 and dividing by 2 twice.",
]


def demo() -> None:
    print("=== OMNIA_TOTALE_SUPERVISOR v1.1 demo ===")
    res = analyze_chain(
        DEMO_CHAIN,
        chain_id="demo_chain",
        json_log_dir=Path("./logs"),
    )

    print(f"Chain ID: {res.summary.chain_id}")
    print(f"Steps: {res.summary.steps}")
    print(f"Unstable steps: {res.summary.unstable_steps}")
    print(f"Ω_mean ≈ {res.summary.omega_mean:.4f}  Ω_std ≈ {res.summary.omega_std:.4f}")
    print(f"Global instability flag: {res.summary.global_flag}")
    print("\nPer-step causal traces:")
    for s in res.steps:
        flag = "UNSTABLE" if s.unstable else "stable"
        print(
            f"- step {s.step_index}: Ω_raw={s.omega_raw:.4f}, "
            f"ΔΩ={s.delta_omega:.4f}, thr={s.threshold:.4f} → {flag}"
        )
        print(f"  reason: {s.reason}")

    print("\nJSON log written to ./logs/ (filename includes chain_id and timestamp).")


if __name__ == "__main__":
    demo()
```0