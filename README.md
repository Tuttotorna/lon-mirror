# OMNIA_TOTALE v2.0 — Unified Ω-fusion Engine

Author: **Massimiliano Brighindi (MB-X.01 / Omniabase±)**  
Engine formalization: **MBX IA**  
Repository: `Tuttotorna/lon-mirror`

---

## 1. Overview

**OMNIA_TOTALE v2.0** is a small, self-contained framework for **structural stability analysis** of:

- integers (multi-base numeric structure),
- time series (regime changes),
- multivariate series (lagged causal structure),
- and, experimentally, token-level traces from LLM reasoning.

It is designed as a **model-agnostic guardrail**: given numbers, sequences, and traces, it emits a set of **Ω-scores** that flag instability, regime shifts, and suspicious patterns, without depending on any particular model internals.

Core idea:  
Instead of trusting the narrative, OMNIA_TOTALE reads the **structure** of what a system is doing (digits, time, correlations) and checks if it is stable or "cracking" under its own logic.

---

## 2. Components

The framework is implemented as a small Python package under `omnia/` plus a set of scripts and demos.

### 2.1 `omnia.omniabase` — multi-base numeric lens (PBII)

File: `omnia/omniabase.py`

Provides:

- `digits_in_base_np(n, b)`  
  Convert integer `n` to digits in base `b` (NumPy, MSB first).

- `normalized_entropy_base(n, b)`  
  Normalized Shannon entropy of digits in base `b` (range `[0, 1]`).

- `sigma_b(n, b, ...)`  
  Base symmetry score:
  - low entropy + short representation → more structure,
  - bonus for `n % b == 0`.

- `OmniabaseSignature` (dataclass)  
  Holds per-base σ and entropy plus means over bases.

- `omniabase_signature(n, bases=...)`  
  Compute the multi-base signature for an integer.

- `pbii_index(n, composite_window=..., bases=...)`  
  **PBII — Prime Base Instability Index**:
  - `saturation` = mean σ over a window of composite neighbours,
  - `PBII(n) = saturation - sigma_mean(n)`,
  - higher PBII ≈ more "prime-like" instability.

This is the numeric heart of OMNIA_TOTALE.

---

### 2.2 `omnia.omniatempo` — temporal stability lens

File: `omnia/omniatempo.py`

Provides:

- `OmniatempoResult` (dataclass):
  - `global_mean`, `global_std`,
  - `short_mean`, `short_std`,
  - `long_mean`, `long_std`,
  - `regime_change_score`.

- `omniatempo_analyze(series, short_window=20, long_window=100, ...)`  
  Computes:
  - global statistics,
  - short/long window statistics,
  - symmetric KL-like divergence between recent-short and recent-long histograms (this is the **regime_change_score**).

In Ω-fusion, we typically use:
- `tempo_component = log(1 + regime_change_score)`.

---

### 2.3 `omnia.omniacausa` — lagged causal-structure lens

File: `omnia/omniacausa.py`

Provides:

- `OmniaEdge` (dataclass):  
  `source`, `target`, `lag`, `strength`.

- `OmniacausaResult` (dataclass):  
  list of `edges`.

- `omniacausa_analyze(series_dict, max_lag=5, strength_threshold=0.3)`:

  For each pair of channels `(src, tgt)`:
  - scans lags in `[-max_lag, ..., +max_lag]`,
  - computes lagged Pearson correlations,
  - keeps the lag with highest |corr|,
  - emits an edge if `|corr| ≥ strength_threshold`.

In Ω-fusion, we typically use:
- `causa_component = mean(|edge.strength|)` over accepted edges.

---

### 2.4 `omnia.kernel` — generic Ω-kernel

File: `omnia/kernel.py`

Defines small, generic kernel structures:

- `OmniaContext`  
  Holds input data:
  - `n` (integer),
  - `series` (main time series),
  - `series_dict` (multichannel),
  - `extra` (dict for future extensions).

- `LensResult`  
  Output of a single lens:
  - `name`,
  - `scores` (dict of scalar metrics),
  - `metadata` (free-form JSON-serializable dict).

- `KernelResult`  
  Global result:
  - `omega_total` (aggregated Ω),
  - per-lens `LensResult`,
  - full metadata snapshot.

- `OmniaKernel`  
  Lightweight engine:
  - register multiple lenses with weights,
  - run them on an `OmniaContext`,
  - combine their `scores["omega"]` into a single `omega_total`.

This kernel makes OMNIA_TOTALE **extensible**: new lenses can be plugged in without touching the core.

---

### 2.5 `omnia.engine` — OMNIA_TOTALE on top of the kernel

File: `omnia/engine.py`

Wraps the three lenses into the kernel:

- `_lens_omniabase(ctx)`  
  Uses `omniabase_signature` and `pbii_index`.  
  Exposes scores:

  - `"omega"` = PBII-style instability (higher ~ more prime-like),
  - `"sigma_mean"`,
  - `"entropy_mean"`.

- `_lens_omniatempo(ctx)`  
  Uses `omniatempo_analyze`.  
  Exposes scores:

  - `"omega"` = `log(1 + regime_change_score)`,
  - `"regime_change_score"`.

- `_lens_omniacausa(ctx)`  
  Uses `omniacausa_analyze`.  
  Exposes:

  - `"omega"` = mean |corr| across edges,
  - `"edge_count"`.

High-level helpers:

- `build_default_engine(w_base=1.0, w_tempo=1.0, w_causa=1.0) -> OmniaKernel`  
  Registers the three lenses with given fusion weights.

- `run_omnia_totale(n, series, series_dict, w_base=1.0, w_tempo=1.0, w_causa=1.0, extra=None) -> KernelResult`  
  Shortcut: construct context, build engine, run, return results.

This replaces older monolithic scripts and is the **canonical path** to use OMNIA_TOTALE in other projects.

---

## 3. Standalone script

File: `OMNIA_TOTALE_v2.0.py`

A self-contained script version of the fused Ω-engine (NumPy-accelerated).  
It mirrors the behaviour of the package, but is kept mainly for:

- compatibility with earlier experiments,
- quick inspection without importing the package.

New work should prefer the package (`omnia.*`) and `omnia.engine.run_omnia_totale`.

---

## 4. Benchmarks

### 4.1 GSM8K-like hallucination / primes demo

File: `benchmarks/gsm8k_benchmark_demo.py`

Contains two **synthetic demo benchmarks**:

1. **Hallucination detection demo**  
   - Builds small GSM8K-like chains (correct vs. altered).  
   - Extracts integers from each chain.  
   - Computes PBII for the numbers and flags chains with high average PBII as "instable".  
   - Reports:
     - false positive rate on correct chains,
     - detection rate on hallucinated chains.

   All numbers and metrics here are **placeholders** meant only for:
   - verifying that PBII behaves qualitatively as expected,
   - providing a starting point for real evaluations.

2. **Prime vs composite AUC demo**  
   - Samples random integers,
   - Labels them as prime/composite with a simple primality test,
   - Uses `-PBII(n)` as a score (low PBII for primes → high score),
   - Computes a simple ROC AUC estimate,
   - Plots the distribution of PBII for primes vs. composites and saves:
     - `pbii_distribution_demo.png`.

Again: this is **not** a formal benchmark; it is a reproducible demo for reviewers.

---

## 5. Repository layout

Current key layout (simplified):

```text
.
├── omnia/
│   ├── __init__.py          # package exports
│   ├── omniabase.py         # multi-base / PBII lens
│   ├── omniatempo.py        # temporal stability lens
│   ├── omniacausa.py        # causal-structure lens
│   ├── kernel.py            # generic Ω-kernel
│   └── engine.py            # OMNIA_TOTALE on top of the kernel
│
├── benchmarks/
│   └── gsm8k_benchmark_demo.py
│
├── OMNIA_TOTALE_v2.0.py     # legacy monolithic script (self-contained engine)
├── OMNIA_LENSES_v0.1.md     # conceptual notes on Omnia lenses
├── README.md                # this file
└── (other legacy prototypes / experiments)

Some additional files in the root directory are earlier prototypes and exploratory experiments. The canonical, maintained path is:

package: omnia/,

engine: omnia.engine,

demos: benchmarks/.



---

6. Installation

Requirements:

Python 3.9+

numpy

matplotlib (for demo plots only)


Install dependencies (local or virtualenv):

pip install numpy matplotlib

Clone the repository:

git clone https://github.com/Tuttotorna/lon-mirror.git
cd lon-mirror


---

7. Quickstart

7.1 Using the package

Example: run the fused Ω-engine on a synthetic prime vs composite test.

import numpy as np
from omnia.engine import run_omnia_totale

# Example integer
n = 173  # a prime

# Synthetic time series with a regime shift
t = np.arange(300)
series = np.sin(t / 15.0) + 0.1 * np.random.normal(size=t.size)
series[200:] += 0.8  # regime shift

# Multi-channel series for causal lens
s1 = np.sin(t / 10.0)
s2 = np.zeros_like(s1)
s2[2:] = 0.7 * s1[:-2] + 0.1 * np.random.normal(size=t.size - 2)
s3 = np.random.normal(size=t.size)

series_dict = {"s1": s1, "s2": s2, "s3": s3}

result = run_omnia_totale(
    n=n,
    series=series,
    series_dict=series_dict,
    w_base=1.0,
    w_tempo=1.0,
    w_causa=1.0,
)

print("Ω_total:", result.omega_total)
for lens_name, lens_res in result.lenses.items():
    print(lens_name, "→", lens_res.scores)

7.2 Running the benchmark demo

From the repo root:

python -m benchmarks.gsm8k_benchmark_demo

This will:

print hallucination detection stats,

print prime vs composite AUC estimate,

save pbii_distribution_demo.png in the current directory.



---

8. Status and caveats

This repository is work in progress.

All numerical claims like:

“71% hallucination reduction on >50-step GSM8K chains”,

“AUC ≈ 0.98 for prime vs composite separation” are currently synthetic placeholders, meant to illustrate how OMNIA_TOTALE could be evaluated.


For any formal claim, the benchmarks must be:

run on full-scale datasets,

documented,

independently reproducible.



OMNIA_TOTALE is offered as a transparent, inspectable structure:

small codebase,

no hidden dependencies beyond NumPy/Matplotlib,

clear separation between core lenses, kernel, and demos.



---

9. License and citation

License: MIT (see LICENSE if present; otherwise, default to MIT as declared here).

If you use OMNIA_TOTALE or Omniabase in academic or technical work, please cite:

> Massimiliano Brighindi (MB-X.01).
OMNIA_TOTALE: Unified Ω-fusion Engine for structural stability in numeric, temporal and causal domains.
GitHub: Tuttotorna/lon-mirror.



