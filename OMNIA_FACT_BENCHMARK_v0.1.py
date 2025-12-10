"""
OMNIA_FACT_BENCHMARK v0.1
Author: Massimiliano Brighindi (concepts) + MBX IA (formalization)

Goal:
    Use FACT_CHECK_ENGINE on GSM8K-style model outputs to see:
      - how well factual/numeric checks separate correct vs. hallucinated answers
      - how much extra signal we get beyond structural Ω (OMNIA_TOTALE)

Input:
    data/gsm8k_model_outputs.jsonl

Each line must be a JSON object like:

{
  "id": "123",
  "question": "If John has 5 apples...",
  "gold_answer": "12",
  "model_chain": "Reasoning chain here...",
  "model_answer": "12"
}

Assumption:
    gold_answer is the ground-truth final answer from GSM8K (or similar).

Usage:
    python OMNIA_FACT_BENCHMARK_v0.1.py
"""

import json
import os
import math
import statistics
from typing import List, Dict, Any, Optional

# Import from your previous module.
# Make sure the file is named FACT_CHECK_ENGINE_v0_1.py in the same folder,
# or adjust the import name accordingly.
from FACT_CHECK_ENGINE_v0_1 import (
    fact_check_chain,
    gold_match_score,
    FactCheckSummary,
    fuse_omnia_with_factcheck,
    OmniaFactFusion,
)


DATA_PATH = "data/gsm8k_model_outputs.jsonl"


def load_gsm8k_outputs(path: str) -> List[Dict[str, Any]]:
    """Load GSM8K-style model outputs from JSONL."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                records.append(obj)
            except Exception as e:
                print(f"Skipping invalid JSON line: {e}")
                continue
    return records


def run_fact_benchmark(
    records: List[Dict[str, Any]],
    fact_threshold: float = 0.7,
    numeric_threshold: float = 0.75,
    omega_struct_default: float = 0.0,
) -> Dict[str, Any]:
    """
    Core benchmark:

      - For each record:
          * compute FactCheckSummary (no external backend)
          * compute gold_match (correct vs hallucinated)
          * fuse with a placeholder Ω_struct (can be replaced with OMNIA_TOTALE)
      - Build confusion matrix using a simple detection rule:
          "flagged" if fact_consistency < fact_threshold
                     OR numeric_consistency < numeric_threshold
    """
    summaries: List[FactCheckSummary] = []
    fusions: List[OmniaFactFusion] = []

    # Confusion matrix counters
    tp = fp = tn = fn = 0

    for rec in records:
        q = rec.get("question", "")
        gold_answer = rec.get("gold_answer")
        model_chain = rec.get("model_chain", "")
        model_answer = rec.get("model_answer", "")

        # Run fact-check chain (backend=None → only numeric + gold)
        summary = fact_check_chain(
            question=q,
            model_chain=model_chain,
            model_answer=model_answer,
            gold_answer=gold_answer,
            backend=None,       # xAI can plug their own backend here
            max_claims=10,
        )
        summaries.append(summary)

        # Structural Ω: here we use a placeholder (0.0).
        # xAI can replace with OMNIA_TOTALE Ω for full fusion.
        omega_struct = omega_struct_default

        fusion = fuse_omnia_with_factcheck(
            omega_struct=omega_struct,
            fact_summary=summary,
            w_fact=1.0,
            w_numeric=0.5,
            w_gold=1.0,
        )
        fusions.append(fusion)

        # Correct vs hallucinated (ground truth from gold_answer)
        gm = summary.gold_match
        is_correct: Optional[bool]
        if gm is None:
            # No gold → skip from detection stats
            is_correct = None
        else:
            is_correct = (gm == 1.0)

        # Detection rule: flag if low factual or numeric consistency
        flagged = (
            summary.fact_consistency < fact_threshold
            or summary.numeric_consistency < numeric_threshold
        )

        if is_correct is None:
            # No ground truth → not counted in confusion matrix
            continue

        if is_correct and flagged:
            fp += 1
        elif is_correct and not flagged:
            tn += 1
        elif (not is_correct) and flagged:
            tp += 1
        else:  # not correct and not flagged
            fn += 1

    # Compute metrics
    tp_fp = tp + fp
    tp_fn = tp + fn
    tn_fp = tn + fp
    total = tp + tn + fp + fn

    detection_rate = tp / tp_fn if tp_fn > 0 else 0.0       # recall on hallucinations
    precision = tp / tp_fp if tp_fp > 0 else 0.0
    false_positive_rate = fp / tn_fp if tn_fp > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0

    # Aggregate scores
    fact_scores = [s.fact_consistency for s in summaries] or [0.0]
    numeric_scores = [s.numeric_consistency for s in summaries] or [0.0]
    gold_scores = [s.gold_match for s in summaries if s.gold_match is not None] or [0.0]

    fused_scores = [f.fused_score for f in fusions] or [0.0]

    result = {
        "N_total": len(records),
        "N_with_gold": total,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "detection_rate": detection_rate,
        "precision": precision,
        "false_positive_rate": false_positive_rate,
        "accuracy": accuracy,
        "fact_consistency_mean": statistics.mean(fact_scores),
        "numeric_consistency_mean": statistics.mean(numeric_scores),
        "gold_match_mean": statistics.mean(gold_scores),
        "fused_score_mean": statistics.mean(fused_scores),
        "fact_threshold": fact_threshold,
        "numeric_threshold": numeric_threshold,
    }
    return result


def main():
    print("=== OMNIA_FACT_BENCHMARK v0.1 ===")
    print(f"Loading GSM8K-style outputs from: {DATA_PATH}")
    records = load_gsm8k_outputs(DATA_PATH)
    print(f"Loaded {len(records)} records.")

    if not records:
        print("No records found. Please populate data/gsm8k_model_outputs.jsonl first.")
        return

    # Example thresholds (can be tuned on validation set)
    fact_th = 0.7
    numeric_th = 0.75

    result = run_fact_benchmark(
        records,
        fact_threshold=fact_th,
        numeric_threshold=numeric_th,
        omega_struct_default=0.0,  # placeholder Ω
    )

    print("\n--- Detection metrics (fact + numeric lens only) ---")
    print(f"N_total (with gold):       {result['N_with_gold']}")
    print(f"TP (halluc flagged):       {result['TP']}")
    print(f"FN (halluc missed):        {result['FN']}")
    print(f"FP (correct flagged):      {result['FP']}")
    print(f"TN (correct unflagged):    {result['TN']}")
    print(f"Detection rate (recall):   {result['detection_rate']:.3f}")
    print(f"Precision:                 {result['precision']:.3f}")
    print(f"False positive rate (FPR): {result['false_positive_rate']:.3f}")
    print(f"Accuracy:                  {result['accuracy']:.3f}")

    print("\n--- Score statistics ---")
    print(f"Mean fact_consistency:     {result['fact_consistency_mean']:.3f}")
    print(f"Mean numeric_consistency:  {result['numeric_consistency_mean']:.3f}")
    print(f"Mean gold_match:           {result['gold_match_mean']:.3f}")
    print(f"Mean fused Ω_ext:          {result['fused_score_mean']:.3f}")

    print("\nThresholds used:")
    print(f"  fact_threshold    = {result['fact_threshold']}")
    print(f"  numeric_threshold = {result['numeric_threshold']}")


if __name__ == "__main__":
    main()