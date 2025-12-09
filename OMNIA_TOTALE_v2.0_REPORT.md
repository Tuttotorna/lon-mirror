# OMNIA_TOTALE v2.0 — Unified Ω-fusion Engine

Author: Massimiliano Brighindi (MB-X.01 / Omniabase±)  
Engine formalization: MBX IA

---

## 1. Goal

OMNIA_TOTALE v2.0 provides a **single Ω-score pipeline** that fuses three structural lenses:

- **BASE**: multi-base structure + PBII instability (Omniabase).
- **TIME**: regime-change detection on sequences (Omniatempo).
- **CAUSA**: lagged correlations between channels (Omniacausa).

The engine is designed as a **model-agnostic guardrail**: given integers, time series and multi-channel traces, it returns a fused Ω-score plus per-lens components, in a form that is easy to log, visualize or connect to external evaluation pipelines.

> Note: token-level analysis (TOKEN / TokenMap) exists as a separate experimental module, not yet part of the `omnia` core package. This README describes the stable v2.0 engine based on BASE/TIME/CAUSA.

---

## 2. Lenses

### 2.1 Omniabase / PBII (BASE)

Module: `omnia.omniabase`

Core ideas:

- Represent an integer `n` in multiple bases (2, 3, 5, 7, 11, …).
- Measure digit entropy and “compact” structure in each base.
- Compare the structure of `n` to nearby composites → PBII (Prime Base Instability Index).

Key functions and types:

- `digits_in_base_np(n, b)`:  
  Return digits of `n` in base `b` as a NumPy array (MSB first).

- `normalized_entropy_base(n, b)`:  
  Normalized Shannon entropy of digits of `n` in base `b` (range `[0, 1]`).

- `sigma_b(n, b, length_weight, length_exponent, divisibility_bonus)`:  
  Base Symmetry Score  
  \[
  \sigma_b(n) = \text{length\_weight} \cdot \frac{1 - H_{\text{norm}}}{L^{\text{length\_exponent}}}
  + \text{divisibility\_bonus} \cdot \mathbf{1}[n \bmod b = 0]
  \]

- `@dataclass OmniabaseSignature`:  
  Holds:
  - `n`
  - `bases: List[int]`
  - `sigmas: Dict[int, float]`
  - `entropy: Dict[int, float]`
  - `sigma_mean: float`
  - `entropy_mean: float`

- `omniabase_signature(n, bases=..., ...) -> OmniabaseSignature`:  
  Compute the multi-base signature for `n`.

- `pbii_index(n, composite_window=..., bases=...) -> float`:  
  Prime Base Instability Index:
  - `saturation` = mean σ on a fixed composite window,
  - `PBII(n) = saturation − sigma_mean(n)`,
  - Higher PBII ≈ more prime-like instability (weaker structure than nearby composites).

---

### 2.2 Omniatempo (TIME)

Module: `omnia.omniatempo`

Goal: detect **distributional shifts** and regime changes in a 1D time series.

Key types and functions:

- `@dataclass OmniatempoResult` with:
  - `global_mean`, `global_std`
  - `short_mean`, `short_std`
  - `long_mean`, `long_std`
  - `regime_change_score`: symmetric KL-like divergence between short vs long windows.

- `omniatempo_analyze(series, short_window=20, long_window=100, ...) -> OmniatempoResult`:

  Steps:
  1. Convert `series` to NumPy array.
  2. Take a short tail segment (e.g. last 20 points) and a longer tail segment (e.g. last 100).
  3. Build histograms for both segments.
  4. Compute a symmetric KL-like divergence:
     \[
     \text{regime} = \frac{1}{2} \big( KL(p \,\|\, q) + KL(q \,\|\, p) \big)
     \]
  5. Return stats and `regime_change_score`.

In fusion, the TIME component is:
\[
\text{tempo\_component} = \log(1 + \text{regime\_change\_score})
\]

---

### 2.3 Omniacausa (CAUSA)

Module: `omnia.omniacausa`

Goal: heuristic **directional dependencies** between channels using lagged correlations.

Key types and functions:

- `@dataclass OmniaEdge`:
  - `source: str`
  - `target: str`
  - `lag: int`
  - `strength: float` (Pearson correlation at that lag)

- `@dataclass OmniacausaResult`:
  - `edges: List[OmniaEdge]`

- `_lagged_corr_np(x, y, lag) -> float`:  
  Internal helper, Pearson correlation between `x` and `y` at given lag:
  - `lag > 0`: `x` leads `y` (x at t−lag → y at t)
  - `lag < 0`: `y` leads `x`
  - `lag = 0`: synchronous correlation

- `omniacausa_analyze(series_dict, max_lag=5, strength_threshold=0.3) -> OmniacausaResult`:

  For each pair `(src, tgt)`:
  1. Search lags in `[-max_lag, +max_lag]`.
  2. Keep the lag with the largest absolute correlation.
  3. If `|corr| ≥ strength_threshold`, emit an `OmniaEdge`.

In fusion, the CAUSA component is:
\[
\text{causa\_component} = \text{mean}(|\text{corr}|) \ \text{over all accepted edges}
\]

---

## 3. Fusion and API

Fusion lives in `omnia.engine` and provides a structured way to combine the three lenses.

### 3.1 Data structures

```python
from dataclasses import dataclass
from typing import Dict, List, Any
import numpy as np

@dataclass
class OmniaInput:
    n: int
    series: np.ndarray
    series_dict: Dict[str, np.ndarray]
    extra: Dict[str, Any] | None = None

Each lens returns its own result and an internal “ω-like” score:

@dataclass
class LensResult:
    name: str                 # "BASE", "TIME", "CAUSA"
    scores: Dict[str, float]  # e.g. {"omega": ..., "raw": ...}
    raw: Any                  # underlying dataclass (OmniabaseSignature, OmniatempoResult, OmniacausaResult)

The fused result:

@dataclass
class OmniaTotaleResult:
    omega: float               # fused Ω
    components: Dict[str, float]  # {"base_instability": ..., "tempo_log_regime": ..., "causa_mean_strength": ...}
    lens_results: List[LensResult]
    meta: Dict[str, Any]

API entry point:

from omnia.engine import run_omnia_totale

result = run_omnia_totale(
    n,
    series,
    series_dict,
    w_base=1.0,
    w_tempo=1.0,
    w_causa=1.0,
    extra={"label": "example"},
)


---

3.2 Fusion formula

Conceptually:

base_instability = pbii_index(n, ...)

tempo_log_regime = log(1 + regime_change_score)

causa_mean_strength = mean(|corr| over edges)


Then:

\Omega = w_{\text{base}} \cdot \text{base\_instability}
      + w_{\text{tempo}} \cdot \text{tempo\_log\_regime}
      + w_{\text{causa}} \cdot \text{causa\_mean\_strength}

Weights (w_base, w_tempo, w_causa) are hyperparameters; defaults are all 1.0.

The fusion does not assume any particular model or dataset. It simply compresses three structural views into a single scalar Ω plus interpretable components.


---

4. Files and layout

In this repository, the v2.0 engine is organized as:

omnia/__init__.py
Public exports for the package: omniabase, omniatempo, omniacausa, engine.

omnia/omniabase.py
Multi-base numeric lens (digits, entropy, σ, PBII).

omnia/omniatempo.py
Temporal stability lens (regime-change detection).

omnia/omniacausa.py
Lagged causal-structure lens (edges with correlation and lag).

omnia/engine.py
Fusion logic (run_omnia_totale, data classes, Ω-computation).

OMNIA_TOTALE_v2.0.py
Thin demo script: builds a synthetic example (prime vs composite, time series with regime shift, causal channels) and prints fused Ω and per-lens components.


Experimental / legacy files (e.g. token maps, self-review loops, earlier monolithic versions) are kept at the repository root and are not part of the stable omnia package API.


---

5. Quick start

Minimal example (from repository root):

pip install numpy
python OMNIA_TOTALE_v2.0.py

This runs the demo, computing Ω for a prime (n = 173) and a composite (n = 180), and prints:

fused Ω for each,

BASE/TIME/CAUSA components,

per-lens “ω-like” scores.


The engine is intended as a transparent, inspectable layer for structural coherence, not as a black-box predictor.