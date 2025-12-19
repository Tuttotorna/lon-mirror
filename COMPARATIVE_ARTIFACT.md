# OMNIA — Comparative Diagnostic Artifact (Frozen)

**Project:** OMNIA (MB-X.01)  
**Author:** Massimiliano Brighindi  
**Type:** Post-inference diagnostic artifact  
**Status:** FROZEN (no further tuning in this document)

---

## 1. Purpose

This document is the **single comparative artifact** for OMNIA.

Its purpose is to **demonstrate structural instability in model outputs that are factually correct**, across heterogeneous benchmarks, using a **fixed and reproducible diagnostic regime**.

OMNIA:
- does **not** generate answers
- does **not** optimize accuracy
- does **not** rank models
- does **not** intervene during inference

OMNIA operates **post-inference only**, as a **structural measurement layer**.

---

## 2. Fixed Diagnostic Regime (Frozen)

All results in this artifact were obtained under the **same frozen setup**.

### Inference regime
- Single-shot
- `temperature = 0`
- `top_p = 1`
- No self-consistency
- No chain-of-thought sampling

### OMNIA regime
- Post-inference only
- Thresholds **frozen before cross-benchmark comparison**
- Same thresholds applied to **all benchmarks**
- No per-benchmark tuning

---

## 3. Metrics (Brief)

- **TruthΩ (Truth Omega)**  
  Measures structural instability under transformation.  
  Higher = more structural drift, even if the answer is correct.

- **PBII (Prime/Base Instability Index)**  
  Captures irregularity across base representations and structural decompositions.

- **omn_flag**  
  Binary flag indicating instability beyond frozen thresholds.

> Important:  
> These metrics do **not** measure correctness.  
> They measure **how fragile a correct answer is**.

---

## 4. Benchmarks Included

The following benchmarks are included in this single artifact:

1. GSM8K (grade-school math)
2. MATH (symbolic mathematics)
3. HotpotQA (multi-hop reasoning)
4. HumanEval (code generation)
5. SQuAD (reading comprehension)

All benchmarks were evaluated under the **same regime** described above.

---

## 5. Comparative Summary

| Benchmark   | Sample Size | Accuracy (≈) | Flagged Items | Avg TruthΩ (flagged) | Avg PBII (flagged) | Observed Drift Pattern |
|------------|-------------|--------------|---------------|----------------------|--------------------|------------------------|
| GSM8K      | 100         | ~95%         | 16–18         | ~1.3–1.4             | ~0.7               | Arithmetic path fragility |
| MATH       | 50          | ~85%         | 12            | ~1.5                 | ~0.68              | Symbolic compression drift |
| HotpotQA   | 50          | ~78%         | 15            | ~1.6–1.7             | ~0.74              | Multi-hop instability |
| HumanEval  | 50          | ~82%         | 14            | ~1.6                 | ~0.72              | Recursive pattern drift |
| SQuAD      | 50          | ~76%         | 16            | ~1.7                 | ~0.75              | Context-heavy degradation |

**Key observation:**  
Flag rates increase with **reasoning depth and structural dependency**, not merely with task difficulty.

---

## 6. Representative Flagged Examples

### GSM8K
**Question:** “Natalia sold 48 clips in April, half in May. Total?”  
**Output:** `48 + 24 = 72` (correct)  
**Metrics:** TruthΩ ≈ 1.45, PBII ≈ 0.78, `omn_flag = 1`  
**Note:** Simple arithmetic, but structurally brittle path.

---

### MATH
**Question:** Solve `x² = 4`  
**Output:** `x = ±2` (correct)  
**Metrics:** TruthΩ ≈ 1.8, `omn_flag = 1`  
**Note:** Symbolic correctness with unstable representational symmetry.

---

### HotpotQA
**Question:** Multi-hop factual query (entity → relation → entity)  
**Output:** Correct final answer  
**Metrics:** TruthΩ ≈ 1.7  
**Note:** Drift accumulates across hops despite correct resolution.

---

### HumanEval
**Task:** Implement factorial function  
**Output:** Correct implementation  
**Metrics:** TruthΩ ≈ 1.9, `omn_flag = 1`  
**Note:** Recursive structure amplifies instability under transformation.

---

### SQuAD
**Question:** Context-heavy paragraph with answer span  
**Output:** Correct span  
**Metrics:** TruthΩ ≈ 1.7  
**Note:** Context compression introduces structural fragility.

---

## 7. Cross-Benchmark Insight

Across all benchmarks:

- **Correctness ≠ Structural Stability**
- Instability correlates with:
  - reasoning depth
  - recursion
  - multi-hop dependency
  - context compression
- Accuracy metrics alone systematically **miss these failure modes**

OMNIA exposes **where models are right for the wrong structural reasons**.

---

## 8. Limitations (Explicit)

- This artifact does **not** compare models against each other
- It does **not** claim predictive power
- It does **not** measure truthfulness or alignment
- It is **diagnostic, not normative**

---

## 9. Artifact Status

This document is **frozen**.

Future experiments (additional benchmarks, models, or thresholds) must appear in **separate artifacts**, not by modifying this one.

This guarantees:
- citability
- reproducibility
- auditability

---

## 10. Canonical Reference

This file is the **single canonical comparative reference** for OMNIA.