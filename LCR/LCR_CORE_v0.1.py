"""
LCR_CORE_v0.1 — Logical Coherence Regressor (core module)
Author: Massimiliano Brighindi (concepts) + MBX IA (formalization)

Goal:
    Minimal, dependency-free core for LCR:
      - define a sample format (one model output with labels)
      - compute a fused LCR score from:
           * structural signal (e.g. OMNIA_TOTALE Ω_struct)
           * fact consistency
           * numeric consistency
      - provide a simple calibration routine to fit weights on a labeled dataset

This module does NOT:
    - load datasets
    - call external models
    - plot or log
Those parts stay in separate benchmark / notebook files.

Usage (example):

    from LCR_CORE_v0_1 import (
        LCRSample, LCRWeights,
        lcr_score, calibrate_lcr_weights
    )

    samples = [...]  # list[LCRSample]
    weights = calibrate_lcr_weights(samples)
    for s in samples:
        score = lcr_score(s, weights)

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import math
import statistics


# =========================
# 1. DATA MODEL
# =========================

@dataclass
class LCRSample:
    """
    One data point for LCR.

    Fields:
        id: optional identifier from dataset
        is_correct: True if the model's final answer is correct,
                    False if hallucinated / wrong.
        omega_struct: structural signal (e.g. OMNIA_TOTALE Ω or PBII-based)
        fact_consistency: [0,1] factual consistency score
        numeric_consistency: [0,1] numeric consistency score
    """

    id: Optional[str]
    is_correct: bool
    omega_struct: float
    fact_consistency: float
    numeric_consistency: float


@dataclass
class LCRWeights:
    """
    Linear fusion weights for LCR.

    LCR_score = sigmoid(
        w_struct * omega_struct
      + w_fact   * fact_consistency
      + w_num    * numeric_consistency
      + bias
    )
    """

    w_struct: float
    w_fact: float
    w_num: float
    bias: float


# =========================
# 2. CORE FUNCTIONS
# =========================

def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def lcr_score(sample: LCRSample, weights: LCRWeights) -> float:
    """
    Compute LCR score in [0,1] for a single sample.

    Interpretation:
        score ≈ probability that the answer is CORRECT
        (higher = more likely correct, lower = more likely hallucinated)
    """
    x = (
        weights.w_struct * sample.omega_struct
        + weights.w_fact * sample.fact_consistency
        + weights.w_num * sample.numeric_consistency
        + weights.bias
    )
    return _sigmoid(x)


def lcr_predict(sample: LCRSample, weights: LCRWeights, threshold: float = 0.5) -> bool:
    """
    Predict correctness from LCR score.

    Returns:
        True  → predicted correct
        False → predicted hallucinated
    """
    return lcr_score(sample, weights) >= threshold


# =========================
# 3. METRICS
# =========================

@dataclass
class LCRMetrics:
    accuracy: float
    recall_hallucinated: float
    precision_hallucinated: float
    false_positive_rate: float
    n_total: int
    n_hallucinated: int
    n_correct: int


def compute_metrics(
    samples: List[LCRSample],
    weights: LCRWeights,
    threshold: float = 0.5,
) -> LCRMetrics:
    """
    Compute basic metrics for LCR on a labeled dataset.

    We treat "hallucinated" as the positive class (is_correct == False).
    """
    tp = fp = tn = fn = 0
    n_total = len(samples)
    n_correct = 0
    n_hallucinated = 0

    for s in samples:
        pred_correct = lcr_predict(s, weights, threshold=threshold)
        true_correct = s.is_correct

        if true_correct:
            n_correct += 1
        else:
            n_hallucinated += 1

        if not true_correct and not pred_correct:
            # hallucinated and predicted hallucinated
            tp += 1
        elif true_correct and not pred_correct:
            # correct but predicted hallucinated
            fp += 1
        elif true_correct and pred_correct:
            # correct and predicted correct
            tn += 1
        else:
            # hallucinated but predicted correct
            fn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    recall_h = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision_h = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return LCRMetrics(
        accuracy=accuracy,
        recall_hallucinated=recall_h,
        precision_hallucinated=precision_h,
        false_positive_rate=fpr,
        n_total=n_total,
        n_hallucinated=n_hallucinated,
        n_correct=n_correct,
    )


# =========================
# 4. CALIBRATION (BRUTE-FORCE GRID)
# =========================

def calibrate_lcr_weights(
    samples: List[LCRSample],
    threshold: float = 0.5,
    w_struct_range: Tuple[float, float] = (-2.0, 2.0),
    w_fact_range: Tuple[float, float] = (0.0, 4.0),
    w_num_range: Tuple[float, float] = (0.0, 4.0),
    bias_range: Tuple[float, float] = (-2.0, 2.0),
    steps: int = 5,
) -> LCRWeights:
    """
    Very simple brute-force calibration:
      - scan a small grid of weights
      - pick the combination that maximizes accuracy on the provided samples

    This is intentionally naive but:
      - completely transparent
      - easily replaceable by xAI with more advanced optimization
    """
    if not samples:
        # Fallback: neutral weights
        return LCRWeights(w_struct=0.0, w_fact=1.0, w_num=1.0, bias=0.0)

    def lin_space(a: float, b: float, n: int) -> List[float]:
        if n <= 1:
            return [(a + b) / 2.0]
        step = (b - a) / (n - 1)
        return [a + i * step for i in range(n)]

    w_struct_vals = lin_space(w_struct_range[0], w_struct_range[1], steps)
    w_fact_vals = lin_space(w_fact_range[0], w_fact_range[1], steps)
    w_num_vals = lin_space(w_num_range[0], w_num_range[1], steps)
    bias_vals = lin_space(bias_range[0], bias_range[1], steps)

    best_weights = None
    best_accuracy = -1.0

    for ws in w_struct_vals:
        for wf in w_fact_vals:
            for wn in w_num_vals:
                for b in bias_vals:
                    w = LCRWeights(
                        w_struct=ws,
                        w_fact=wf,
                        w_num=wn,
                        bias=b,
                    )
                    metrics = compute_metrics(samples, w, threshold=threshold)
                    if metrics.accuracy > best_accuracy:
                        best_accuracy = metrics.accuracy
                        best_weights = w

    # best_weights cannot be None if samples is non-empty
    return best_weights  # type: ignore[return-value]


# =========================
# 5. SMALL SELF-TEST
# =========================

def _self_test():
    """
    Tiny sanity-check on synthetic data.
    This is not a full benchmark, just a smoke test.
    """
    # Synthetic pattern:
    #  - correct answers have high fact/numeric, moderate struct
    #  - hallucinated answers have low fact/numeric, noisy struct
    rng = [
        LCRSample(
            id=f"c{i}",
            is_correct=True,
            omega_struct=0.2 + 0.1 * math.sin(i),
            fact_consistency=0.8 + 0.1 * math.sin(i / 2.0),
            numeric_consistency=0.85 + 0.1 * math.cos(i / 3.0),
        )
        for i in range(20)
    ] + [
        LCRSample(
            id=f"h{i}",
            is_correct=False,
            omega_struct=0.5 + 0.5 * math.sin(i * 1.5),
            fact_consistency=0.3 + 0.2 * math.sin(i / 2.0),
            numeric_consistency=0.35 + 0.2 * math.cos(i / 3.0),
        )
        for i in range(20)
    ]

    weights = calibrate_lcr_weights(rng, threshold=0.5)
    metrics = compute_metrics(rng, weights, threshold=0.5)

    print("=== LCR_CORE_v0.1 self-test ===")
    print("Learned weights:")
    print(f"  w_struct = {weights.w_struct:.3f}")
    print(f"  w_fact   = {weights.w_fact:.3f}")
    print(f"  w_num    = {weights.w_num:.3f}")
    print(f"  bias     = {weights.bias:.3f}")
    print("Metrics on synthetic data:")
    print(f"  accuracy             = {metrics.accuracy:.3f}")
    print(f"  recall (halluc)      = {metrics.recall_hallucinated:.3f}")
    print(f"  precision (halluc)   = {metrics.precision_hallucinated:.3f}")
    print(f"  false positive rate  = {metrics.false_positive_rate:.3f}")
    print(f"  n_total              = {metrics.n_total}")
    print(f"  n_hallucinated       = {metrics.n_hallucinated}")
    print(f"  n_correct            = {metrics.n_correct}")


if __name__ == "__main__":
    _self_test()