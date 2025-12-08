"""
OMNIA_TOTALE_SUPERVISOR v1.0
Author: Massimiliano Brighindi (concepts) + MBX IA (formalization)

Purpose:
    Lightweight, model-agnostic supervisor for LLM reasoning chains.

Core idea:
    - Treat each reasoning step as an "event" in a structural space.
    - Extract integers from the text.
    - Compute PBII-style instability for each number.
    - Aggregate into a step-instability score.
    - Flag steps and chains whose instability exceeds configurable thresholds.

This file is self-contained:
    - No external LLM calls.
    - No plotting.
    - Only standard library + 'math' + 're'.

Intended usage in xAI / Grok:
    - Wrap this around internal chain-of-thought traces (or distilled logs).
    - Use 'supervise_chain(...)' as a "trust governor":
        * identify unstable segments
        * trigger re-generation / revision on those segments only
        * log structural drift over time for eval and safety.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Iterable, Optional
import math
import re

# =========================
# 1. OMNIABASE / PBII CORE
# =========================

def digits_in_base(n: int, b: int) -> List[int]:
    """
    Return digits of n in base b (MSB first).
    """
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


def normalized_entropy_base(n: int, b: int) -> float:
    """
    Normalized Shannon entropy of digits of n in base b.
    H_norm ∈ [0, 1].
    """
    digits = digits_in_base(n, b)
    L = len(digits)
    if L == 0:
        return 0.0
    # frequency of each digit
    freq = [0] * b
    for d in digits:
        freq[d] += 1
    probs = [c / L for c in freq if c > 0]
    if not probs:
        return 0.0
    H = -sum(p * math.log2(p) for p in probs)
    Hmax = math.log2(b)
    return H / Hmax if Hmax > 0 else 0.0


def sigma_b(
    n: int,
    b: int,
    length_weight: float = 1.0,
    length_exponent: float = 1.0,
    divisibility_bonus: float = 0.5,
) -> float:
    """
    Base Symmetry Score (same structure as OMNIA_LENSES v0.1/v0.2).

    sigma_b(n) = length_weight * (1 - H_norm) / L^length_exponent
                 + divisibility_bonus * I[n % b == 0]

    - High when representation is short, low-entropy, and divisible by base.
    - Primes tend to have lower sigma_b on average than composites.
    """
    digits = digits_in_base(n, b)
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

    length_term = length_weight * (1.0 - Hn) / (L ** length_exponent)
    div_term = divisibility_bonus * (1.0 if n % b == 0 else 0.0)
    return length_term + div_term


def sigma_avg(
    n: int,
    bases: Iterable[int],
    length_weight: float = 1.0,
    length_exponent: float = 1.0,
    divisibility_bonus: float = 0.5,
) -> float:
    """
    Mean sigma_b over the chosen bases.
    """
    bases = list(bases)
    if not bases:
        return 0.0
    vals = [
        sigma_b(
            n,
            b,
            length_weight=length_weight,
            length_exponent=length_exponent,
            divisibility_bonus=divisibility_bonus,
        )
        for b in bases
    ]
    return sum(vals) / len(vals)


def is_composite(k: int) -> bool:
    """
    Simple compositeness test (for saturation window).
    """
    if k <= 3:
        return False
    for d in range(2, int(math.sqrt(k)) + 1):
        if k % d == 0:
            return True
    return False


def saturation(
    n: int,
    bases: Iterable[int],
    W: int = 100,
    length_weight: float = 1.0,
    length_exponent: float = 1.0,
    divisibility_bonus: float = 0.5,
) -> float:
    """
    Saturation term Sat: average sigma over composites in window [n-W, n).
    """
    if W <= 0:
        return 0.0
    bases = list(bases)
    start = max(4, n - W)
    comps = [k for k in range(start, n) if is_composite(k)]
    if not comps:
        return 0.0
    vals = [
        sigma_avg(
            k,
            bases=bases,
            length_weight=length_weight,
            length_exponent=length_exponent,
            divisibility_bonus=divisibility_bonus,
        )
        for k in comps
    ]
    return sum(vals) / len(vals)


def pbii(
    n: int,
    bases: Optional[Iterable[int]] = None,
    W: int = 100,
    length_weight: float = 1.0,
    length_exponent: float = 1.0,
    divisibility_bonus: float = 0.5,
) -> float:
    """
    Prime Base Instability Index (PBII).

    PBII(n) = Sat(n) - sigma_avg(n)

    - High PBII: n is "more unstable" than its composite neighborhood (prime-like).
    - Low/negative PBII: n behaves like composites (saturated, structured).
    """
    if bases is None:
        bases = [2, 3, 5, 7, 11, 13, 17, 19]
    bases = list(bases)
    sat = saturation(
        n,
        bases=bases,
        W=W,
        length_weight=length_weight,
        length_exponent=length_exponent,
        divisibility_bonus=divisibility_bonus,
    )
    sig_n = sigma_avg(
        n,
        bases=bases,
        length_weight=length_weight,
        length_exponent=length_exponent,
        divisibility_bonus=divisibility_bonus,
    )
    return sat - sig_n


# =========================
# 2. STEP & CHAIN DATA MODEL
# =========================

@dataclass
class SupervisorConfig:
    """
    Configuration for OMNIA_TOTALE_SUPERVISOR.
    """
    # PBII parameters
    bases: List[int] = None
    window_size: int = 100
    length_weight: float = 1.0
    length_exponent: float = 1.0
    divisibility_bonus: float = 0.5

    # thresholds
    pbii_threshold: float = 0.10     # step flagged if avg PBII > this
    chain_instability_threshold: float = 0.08  # mean PBII across chain

    # misc
    min_numbers_per_step: int = 1    # require at least this many integers to score step

    def __post_init__(self):
        if self.bases is None:
            # Default: prime bases up to 19
            self.bases = [2, 3, 5, 7, 11, 13, 17, 19]


@dataclass
class StepObservation:
    """
    Structural observation for a single reasoning step.
    """
    index: int
    text: str
    numbers: List[int]
    pbii_values: List[float]
    pbii_mean: float
    flagged: bool

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ChainAssessment:
    """
    Global assessment for a reasoning chain.
    """
    steps: List[StepObservation]
    chain_pbii_mean: float
    flagged_steps_indices: List[int]
    chain_flagged: bool

    def to_dict(self) -> Dict:
        return {
            "chain_pbii_mean": self.chain_pbii_mean,
            "chain_flagged": self.chain_flagged,
            "flagged_steps_indices": self.flagged_steps_indices,
            "steps": [s.to_dict() for s in self.steps],
        }


# =========================
# 3. UTILITIES
# =========================

_num_regex = re.compile(r"\b\d+\b")


def extract_numbers(chain_text: str) -> List[int]:
    """
    Extract positive integers from a reasoning step.
    """
    nums = []
    for tok in _num_regex.findall(chain_text):
        try:
            v = int(tok)
            if v > 1:
                nums.append(v)
        except ValueError:
            continue
    return nums


# =========================
# 4. SUPERVISOR LOGIC
# =========================

def score_step(
    step_index: int,
    step_text: str,
    cfg: SupervisorConfig,
) -> StepObservation:
    """
    Compute PBII-based instability score for a single step.
    """
    nums = extract_numbers(step_text)
    if len(nums) < cfg.min_numbers_per_step:
        # Not enough structure -> treat as neutral, not flagged.
        return StepObservation(
            index=step_index,
            text=step_text,
            numbers=nums,
            pbii_values=[],
            pbii_mean=0.0,
            flagged=False,
        )

    vals: List[float] = []
    for n in nums:
        v = pbii(
            n,
            bases=cfg.bases,
            W=cfg.window_size,
            length_weight=cfg.length_weight,
            length_exponent=cfg.length_exponent,
            divisibility_bonus=cfg.divisibility_bonus,
        )
        vals.append(v)
    pbii_mean = sum(vals) / len(vals) if vals else 0.0
    flagged = pbii_mean > cfg.pbii_threshold

    return StepObservation(
        index=step_index,
        text=step_text,
        numbers=nums,
        pbii_values=vals,
        pbii_mean=pbii_mean,
        flagged=flagged,
    )


def supervise_chain(
    steps_text: List[str],
    cfg: Optional[SupervisorConfig] = None,
) -> ChainAssessment:
    """
    Main entry point.

    Input:
        - steps_text: list of reasoning steps (strings), in order.
        - cfg: SupervisorConfig (optional).

    Output:
        - ChainAssessment with:
            * per-step PBII-based instability
            * list of flagged steps
            * chain-level flag and mean PBII.
    """
    if cfg is None:
        cfg = SupervisorConfig()

    observations: List[StepObservation] = []
    for idx, txt in enumerate(steps_text):
        obs = score_step(idx, txt, cfg)
        observations.append(obs)

    pbii_means = [s.pbii_mean for s in observations if s.pbii_values]
    chain_pbii_mean = sum(pbii_means) / len(pbii_means) if pbii_means else 0.0

    flagged_indices = [s.index for s in observations if s.flagged]
    chain_flagged = chain_pbii_mean > cfg.chain_instability_threshold

    return ChainAssessment(
        steps=observations,
        chain_pbii_mean=chain_pbii_mean,
        flagged_steps_indices=flagged_indices,
        chain_flagged=chain_flagged,
    )


# =========================
# 5. DEMO (LOCAL TEST)
# =========================

def demo():
    """
    Minimal local demo.

    Run:
        python OMNIA_TOTALE_SUPERVISOR_v1.0.py

    to see output.
    """
    sample_chain = [
        # mostly stable arithmetic
        "We know that 12 + 8 = 20, and 5 * 4 = 20, so total apples = 20.",
        "Next, divide 100 by 5 to get 20 groups, and each has 10 items so 200 items total.",
        # inject some instability / hallucination-like noise
        "Assume now that 17 * 3 = 40 and 19 * 2 = 50, we keep 90 as approximate total.",
        "Using these approximations, if 90 / 3 ≈ 35, we can say each friend gets about 35.",
    ]

    cfg = SupervisorConfig(
        pbii_threshold=0.10,
        chain_instability_threshold=0.08,
    )

    assessment = supervise_chain(sample_chain, cfg)
    print("=== OMNIA_TOTALE_SUPERVISOR v1.0 demo ===")
    print(f"Chain PBII mean: {assessment.chain_pbii_mean:.4f}")
    print(f"Chain flagged: {assessment.chain_flagged}")
    print(f"Flagged steps: {assessment.flagged_steps_indices}")
    print("Per-step details:")
    for s in assessment.steps:
        print(f"- Step {s.index}: pbii_mean={s.pbii_mean:.4f}, flagged={s.flagged}, nums={s.numbers}")


if __name__ == "__main__":
    demo()
```0