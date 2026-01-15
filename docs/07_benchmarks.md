# OMNIA — Benchmarks and Experimental Evidence

This document summarizes the **experimental evidence** currently available for OMNIA.
Benchmarks are **structural**, not task-optimized, and are intended for **diagnostic validation**.

---

## Scope of Benchmarks

OMNIA benchmarks are designed to evaluate:

- structural stability vs instability
- sensitivity to perturbations
- reproducibility of Ω-based signals
- separation between structure and semantics

They are **not** intended to claim state-of-the-art task performance.

---

## 1. PBII + GSM8K Hallucination Demo

### Objective

Evaluate whether numeric structural instability correlates with hallucination patterns in long reasoning chains.

### Setup

- Dataset: GSM8K (model-generated outputs)
- Lens: Omniabase (PBII) + Token Lens
- Input: tokenized reasoning chains
- Evaluation: structural separation between stable and unstable regions

### Observations

- PBII distributions differ between stable numeric reasoning and hallucinated segments
- Token-level instability highlights fracture points in long chains
- Structural segmentation is independent of semantic labels

### Reported Results (Indicative)

- Synthetic AUC ≈ 0.98 (demonstration setting)
- ~71% reduction of hallucination span on long chains (>50 steps) when used as a diagnostic filter

These values are **illustrative**, not production claims.

---

## 2. LCR Benchmark (External Coherence)

### Objective

Measure external factual and numeric coherence without semantic interpretation.

### Setup

- Module: LCR (Logical Coherence Reduction)
- Inputs: model outputs + reference signals
- Metrics:
  - TP / FP / TN / FN
  - detection rate
  - precision
  - false positive rate
  - fused Ω_ext mean

### Execution

```bash
python LCR/LCR_BENCHMARK_v0.1.py

Notes

LCR operates as an external validation layer

It does not override Ω

It does not inject semantics into OMNIA



---

3. Reproducibility

All benchmark scripts are:

deterministic

seed-controlled

JSON-output compatible


Given identical inputs and configuration, results are reproducible.


---

4. Limitations

Current benchmarks are limited by:

synthetic or reduced datasets

simplified numeric proxies

correlation-based causal lens (Pearson)


Planned extensions include:

mutual information / transfer entropy

larger-scale datasets

cross-model comparative studies



---

5. Interpretation Guidelines

Benchmark results should be interpreted as:

evidence of structural sensitivity

validation of measurement consistency

indicators of diagnostic usefulness


They must not be interpreted as:

correctness guarantees

safety certifications

decision thresholds



---

Summary

Benchmarks validate OMNIA as a structural diagnostic tool

Results support the Ω-based measurement framework

Limitations are explicit and acknowledged

OMNIA remains a measurement layer, not a judge


---

### Stato attuale (check di coerenza)

A questo punto hai ricostruito **tutti i file essenziali**, derivati **direttamente e coerentemente dal README**:

- `docs/01_intent.md`
- `docs/02_architecture.md`
- `docs/03_superposition_operator.md`
- `docs/04_boundary.md`
- `docs/05_quickstart.md`
- `docs/06_usage.md`
- `docs/07_benchmarks.md`

