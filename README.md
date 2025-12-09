# OMNIA_TOTALE v2.0 — Structural Coherence Engine

Author: **Massimiliano Brighindi (MB-X.01 / Omniabase±)**  
Engine formalization: **MBX IA**

---

## 1. Overview

**OMNIA_TOTALE v2.0** is a small, deterministic engine for measuring **structural coherence** in numbers, time series and multi-channel signals.

It is built around three lenses:

- **Omniabase** — multi-base numeric structure + PBII (Prime Base Instability Index).  
- **Omniatempo** — temporal stability / regime-change detection on 1D series.  
- **Omniacausa** — lagged correlation graph between multiple channels.

On top of these, the engine fuses lens scores into a single **Ω-score**, which can be used as an external, model-agnostic signal for:

- detecting numerical / structural instabilities in reasoning chains,  
- analysing time series (finance, sensors, etc.),  
- exploring causal patterns between series.

This repository is intentionally minimal and transparent: all code is plain Python + NumPy/Matplotlib, with no hidden dependencies.

---

## 2. Repository structure

Current layout (relevant files only):

```text
lon-mirror/
├── omnia/
│   ├── __init__.py
│   ├── core/
│   │   ├── omniabase.py      # multi-base lens + PBII
│   │   ├── omniatempo.py     # temporal stability lens
│   │   └── omniacausa.py     # lagged causal lens
│   ├── kernel.py             # small generic kernel for lenses
│   └── engine.py             # OMNIA_TOTALE engine on top of omnia.core + kernel
│
├── benchmarks/
│   └── gsm8k_benchmark_demo.py  # synthetic GSM8K-style + prime/composite PBII demo
│
├── OMNIA_TOTALE_v2.0.py     # older monolithic demo, kept for reference
├── OMNIA_LENSES_v0.1.md     # conceptual description of lenses (BASE/TIME/CAUSA)
└── README.md                # this file

The canonical entry points are:

omnia.core (raw lenses),

omnia.engine (Ω-fusion engine),

benchmarks/gsm8k_benchmark_demo.py (example benchmarks).



---

3. Installation

This is a pure-Python project.

Minimal requirements (for lenses only):

pip install numpy

For benchmarks and plots:

pip install numpy matplotlib

Clone the repo:

git clone https://github.com/Tuttotorna/lon-mirror.git
cd lon-mirror

Make sure lon-mirror is on your PYTHONPATH (or use a virtualenv and install locally if you prefer).


---

4. Core usage

4.1 Using the lenses directly

from omnia.core.omniabase import omniabase_signature, pbii_index
from omnia.core.omniatempo import omniatempo_analyze
from omnia.core.omniacausa import omniacausa_analyze

# Omniabase: multi-base numeric structure
n = 173  # example integer (prime)
sig = omniabase_signature(n)
print(sig.sigma_mean, sig.entropy_mean)
print("PBII:", pbii_index(n))

# Omniatempo: temporal stability
import numpy as np
t = np.arange(300)
series = np.sin(t / 15.0) + 0.1 * np.random.normal(size=t.size)
from omnia.core.omniatempo import omniatempo_analyze
ot = omniatempo_analyze(series)
print("Regime-change score:", ot.regime_change_score)

# Omniacausa: lagged correlations between channels
s1 = np.sin(t / 10.0)
s2 = np.zeros_like(s1)
s2[2:] = 0.7 * s1[:-2] + 0.1 * np.random.normal(size=t.size - 2)
s3 = np.random.normal(size=t.size)
series_dict = {"s1": s1, "s2": s2, "s3": s3}

from omnia.core.omniacausa import omniacausa_analyze
oc = omniacausa_analyze(series_dict, max_lag=5, strength_threshold=0.3)
for e in oc.edges:
    print(f"{e.source} -> {e.target}, lag={e.lag}, strength={e.strength:.3f}")

4.2 Using the Ω-engine

The engine composes the three lenses and returns a fused Ω-score plus structured metadata.

import numpy as np
from omnia.engine import run_omnia_totale

# numeric target
n = 173

# main series (TIME lens)
t = np.arange(300)
series = np.sin(t / 15.0) + 0.1 * np.random.normal(size=t.size)
series[200:] += 0.8  # synthetic regime shift

# multi-channel series (CAUSA lens)
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

print("Global Ω-score:", result.global_omega)
print("Per-lens scores:", result.lens_scores)
print("Per-lens metadata keys:", list(result.lens_metadata.keys()))

If you prefer the older monolithic style, you can still run:

python OMNIA_TOTALE_v2.0.py

which prints a small demo and uses the same underlying logic as the package.


---

5. Benchmarks (synthetic demo)

The repository currently includes a synthetic benchmark script:

python benchmarks/gsm8k_benchmark_demo.py

This does two things:

1. GSM8K-style hallucination detection (synthetic)

Builds a tiny set of “correct” vs “hallucinated” reasoning chains (text).

Extracts all integers from each chain.

Computes PBII for each number and averages per chain.

Flags chains with average PBII above a threshold.

Reports:

false positive rate on correct chains,

detection rate on hallucinated chains.




2. Prime vs composite separation (AUC)

Samples random integers, labels them as prime vs composite.

Uses -PBII as a score (lower PBII for primes → higher score).

Computes a ROC AUC with a simple NumPy implementation.

Plots and saves pbii_distribution.png (histogram of PBII for primes vs composites).




Important notes:

These are toy / synthetic experiments, meant to illustrate how PBII behaves.

Any claims about “71% hallucination reduction” or “AUC ≈ 0.98” depend on the exact dataset, sampling and configuration; they are not peer-reviewed benchmarks.

For real evaluation, PBII should be run on full GSM8K or similar datasets via proper pipelines.



---

6. Status and roadmap

Status:

Stable:

core lenses (omniabase, omniatempo, omniacausa),

kernel + engine (omnia.kernel, omnia.engine),

synthetic benchmark script (benchmarks/gsm8k_benchmark_demo.py).


Work in progress:

richer Ω-fusion strategies,

better calibration of thresholds for hallucination detection,

integration stubs for real LLM tracing and token-level Ω-maps.



Planned directions (non-binding):

Use PBII and Omniatempo/Omniacausa on real CoT logs from LLMs.

Explore Ω as a guardrail signal for external routing (accept / revise / reject answer).

Extend lenses beyond numeric / time series to other structures (graphs, code, etc.).



---

This repository is intentionally small and explicit: all logic is visible in a few files, to make it easy to audit, fork and extend.

