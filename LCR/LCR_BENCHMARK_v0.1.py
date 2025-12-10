"""
LCR_BENCHMARK_v0.1
Author: Massimiliano Brighindi (concepts) + MBX IA (formalization)

Goal:
    Use LCR_CORE_v0.1 on a labeled dataset to:
      - calibrate LCR weights (w_struct, w_fact, w_num, bias)
      - compute detection metrics for hallucinations vs correct answers

Input file:
    data/lcr_samples.jsonl

Each line must be a JSON object like:

{
  "id": "sample_001",
  "is_correct": true,
  "omega_struct": 0.23,
  "fact_consistency": 0.91,
  "numeric_consistency": 0.88
}

Notes:
    - is_correct = true  → model answer is correct
    - is_correct = false → model answer is hallucinated / wrong
    - omega_struct can come from OMNIA_TOTALE Ω, PBII, or similar
    - fact_consistency, numeric_consistency can come from FACT_CHECK_ENGINE

Usage:
    python LCR_BENCHMARK_v0.1.py
"""

from __future__ import annotations
import json
import os
from typing import List, Dict, Any

from LCR_CORE_v0_1 import (
    LCRSample,
    LCRWeights,
    LCRMetrics,
    calibrate_lcr_weights,
    compute_metrics,
)

DATA_PATH = "data/lcr_samples.jsonl"


def load_lcr_samples(path: str) -> List[LCRSample]:
    """
    Load LCR samples from JSONL file.

    Each line must have:
        id: str
        is_correct: bool
        omega_struct: float
        fact_consistency: float
        numeric_consistency: float
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    samples: List[LCRSample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj: Dict[str, Any] = json.loads(line)
            except Exception as e:
                print(f"Skipping invalid JSON line: {e}")
                continue

            try:
                s = LCRSample(
                    id=obj.get("id"),
                    is_correct=bool(obj["is_correct"]),
                    omega_struct=float(obj["omega_struct"]),
                    fact_consistency=float(obj["fact_consistency"]),
                    numeric_consistency=float(obj["numeric_consistency"]),
                )
                samples.append(s)
            except KeyError as e:
                print(f"Skipping line missing field {e}: {obj}")
            except Exception as e:
                print(f"Skipping line (conversion error): {e}")
    return samples


def run_lcr_benchmark(
    samples: List[LCRSample],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Calibrate LCR weights and compute metrics on the given samples.
    """
    if not samples:
        raise ValueError("No samples provided to LCR benchmark.")

    print(f"Calibrating LCR weights on {len(samples)} samples...")
    weights: LCRWeights = calibrate_lcr_weights(
        samples,
        threshold=threshold,
    )

    print("Computing metrics with calibrated weights...")
    metrics: LCRMetrics = compute_metrics(
        samples,
        weights,
        threshold=threshold,
    )

    result = {
        "weights": {
            "w_struct": weights.w_struct,
            "w_fact": weights.w_fact,
            "w_num": weights.w_num,
            "bias": weights.bias,
        },
        "metrics": {
            "accuracy": metrics.accuracy,
            "recall_hallucinated": metrics.recall_hallucinated,
            "precision_hallucinated": metrics.precision_hallucinated,
            "false_positive_rate": metrics.false_positive_rate,
            "n_total": metrics.n_total,
            "n_hallucinated": metrics.n_hallucinated,
            "n_correct": metrics.n_correct,
        },
        "threshold": threshold,
    }
    return result


def main():
    print("=== LCR_BENCHMARK_v0.1 ===")
    print(f"Loading samples from: {DATA_PATH}")

    try:
        samples = load_lcr_samples(DATA_PATH)
    except FileNotFoundError as e:
        print(str(e))
        print("Please create data/lcr_samples.jsonl with LCRSample-format entries.")
        return

    if not samples:
        print("No valid samples found in data/lcr_samples.jsonl.")
        return

    # Default decision threshold
    threshold = 0.5

    result = run_lcr_benchmark(samples, threshold=threshold)

    w = result["weights"]
    m = result["metrics"]

    print("\n--- Learned LCR weights ---")
    print(f"w_struct = {w['w_struct']:.4f}")
    print(f"w_fact   = {w['w_fact']:.4f}")
    print(f"w_num    = {w['w_num']:.4f}")
    print(f"bias     = {w['bias']:.4f}")

    print("\n--- LCR metrics ---")
    print(f"accuracy               = {m['accuracy']:.3f}")
    print(f"recall (hallucinated)  = {m['recall_hallucinated']:.3f}")
    print(f"precision (hallucinated) = {m['precision_hallucinated']:.3f}")
    print(f"false positive rate    = {m['false_positive_rate']:.3f}")
    print(f"n_total                = {m['n_total']}")
    print(f"n_hallucinated         = {m['n_hallucinated']}")
    print(f"n_correct              = {m['n_correct']}")
    print(f"\nDecision threshold     = {result['threshold']}")


if __name__ == "__main__":
    main(