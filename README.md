# OMNIA_TOTALE — Unified Structural Coherence Engine (MB-X.01)

Author: Massimiliano Brighindi (MB-X.01 / Omniabase±)  
Engine formalization: MBX-IA  
Package version: v2.0

---

## 1. Overview

OMNIA_TOTALE is a **unified structural-coherence engine** designed to evaluate
stability, predictability and internal consistency of:

- integers (multi-base analysis),
- time series (regime change detection),
- multi-channel signals (lagged causal structure),
- reasoning traces and tokens (instability mapping).

The engine is **model-agnostic**: it can wrap any LLM, numeric process, or dataset
and return a fused **Ω-score**, plus detailed per-lens diagnostics.

OMNIA is implemented as a modular Python package:

omnia/ ├─ init.py ├─ core/ │   ├─ init.py │   ├─ omniabase.py      # PBII + multi-base signatures │   ├─ omniatempo.py     # regime-change lens │   ├─ omniacausa.py     # lagged causal lens │   ├─ engine.py         # high-level fusion engine │   └─ kernel.py         # generic lens framework

A live demo benchmark (`gsm8k_benchmark_demo.py`) is also included for testing.

---

## 2. Lenses

OMNIA_TOTALE is composed of **three structural lenses**:

---

### 2.1 Omniabase — Multi-base numeric structure (PBII)

Captures structural regularity and instability in integers.

Components:
- digit entropy in multiple bases,
- σ-scores (base symmetry),
- PBII = Prime-Base Instability Index,
- multi-base signature.

Higher PBII ⇒ more "prime-like" instability.

---

### 2.2 Omniatempo — Regime-change time lens

Given a 1-D series:

- computes histograms in short/long windows,
- evaluates symmetric KL divergence,
- returns regime-change score,
- maps it to Ω via: `log(1 + score)`.

Detects shocks, transitions, volatility clusters.

---

### 2.3 Omniacausa — Lagged causal structure

Given multiple time-series channels:

- scans lags in `[-max_lag, +max_lag]`,
- computes Pearson correlation for each lag,
- selects strongest direction,
- emits causal edges when above threshold.

Fusion uses mean absolute correlation of accepted edges.

---

## 3. Fusion Engine (Ω-Supervisor)

OMNIA_TOTALE aggregates the lenses with weights:

Ω = w_base * BASE

w_tempo * TIME

w_causa * CAUSA


Returned object includes:
- Ω total score
- per-lens scores
- per-lens metadata (JSON-serializable)

The engine is accessed through:

from omnia.core.engine import run_omnia_totale

---

## 4. Installation

Requires only NumPy:

pip install numpy

Optional for benchmarks:

pip install matplotlib

---

## 5. Quick Start

Minimal example:

import numpy as np from omnia.core.engine import run_omnia_totale

n = 173  # test integer series = np.sin(np.arange(300) / 15.0) series_dict = {"s1": series, "s2": series * 0.7}

result = run_omnia_totale(n, series, series_dict)

print("Ω =", result.omega) print(result.component_scores)

---

## 6. Quick Test (NEW)

To verify that the OMNIA package is correctly installed and the lenses run end-to-end, execute:

python quick_omnia_test.py

This test script reports:

- PBII evaluation on sample integers,
- regime-shift detection on synthetic time series,
- causal-edge extraction on multi-channel sequences,
- full Ω-fusion output.

---

## 7. Benchmarks (demo only)

Included file:  
`gsm8k_benchmark_demo.py`

Demonstrates:

- synthetic hallucination detection on GSM8K-like reasoning chains,
- PBII-based prime/composite separation,
- histogram visualization.

**Note:** Benchmark metrics (e.g., “71% reduction on >50-step chains”, “AUC ~0.98”)  
are *demonstration placeholders*.  
Replace with real experimental results before publication.

---

## 8. Philosophy

OMNIA_TOTALE follows these principles:

- **Structural truth over narrative**  
- **Deterministic metrics**  
- **Model-agnostic safety signals**  
- **Transparent, reproducible algorithms**  
- **Minimal dependencies**  
- **Research-oriented modularity**

---

## 9. Citation

Massimiliano Brighindi (2025). OMNIA_TOTALE — Unified Structural Coherence Engine. https://github.com/Tuttotorna/lon-mirror

---

## 10. License

MIT License — free for research, auditing, integration.
