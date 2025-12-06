# OMNIA_TOTALE — Multi-Lens Structural Scoring for AI Coherence (v0.2)

**Author:** Massimiliano Brighindi (concepts) + MBX IA (formalization)  
**Status:** Experimental, research-grade  
**Core file:** `OMNIA_TOTALE_v0.2.py`

---

## 1. What OMNIA_TOTALE does

`OMNIA_TOTALE` is a *structural scoring module* designed to sit **around** a model, not inside it.

Given:
- an integer `n` (e.g. problem ID, hash, or structural token count),
- a 1D time series (e.g. per-step logprob, loss, or any scalar trace),
- a multivariate time series (e.g. multiple traces: logprob, entropy, contradiction score, etc.),

it returns:

- three lens-level analyses:
  - **Omniabase** (multi-base structure on `n`),
  - **Omniatempo** (temporal regime shifts),
  - **Omniacausa** (lagged dependencies between signals),
- a **fused scalar score** `Ω` that summarizes *instability / structure* across those views.

This `Ω` is meant to be used as:
- a **risk / instability score** for hallucinations,
- a **meta-signal** to decide when to re-ask, re-check, or revise a reasoning chain,
- a **feature** in downstream classifiers (e.g. “is this answer trustworthy?”), not as a replacement for the model itself.

---

## 2. Lenses overview

### 2.1 Omniabase (multi-base lens)

Goal: treat `n` as a “multi-base object” instead of a purely base-10 integer.

Key steps (see code for exact formulas):

- Convert `n` in several bases `b ∈ {2,3,5,7,11,13,17,19}`.
- For each base `b`:
  - compute **normalized entropy** of digit frequencies `H_norm(n,b) ∈ [0,1]`,
  - compute **Base Symmetry Score**:

    \[
    \sigma_b(n) = \text{length\_weight} \cdot \frac{1 - H_{\text{norm}}(n,b)}{L^{\text{length\_exponent}}}
                  + \text{divisibility\_bonus} \cdot \mathbf{1}[n \bmod b = 0]
    \]

  - where:
    - `L` = number of digits in base `b`,
    - low entropy + short length + divisibility by `b` ⇒ high structural score.

From there:

- `sigma_mean` = mean of all `σ_b(n)`  
- `entropy_mean` = mean of all `H_norm(n,b)`  

And a **PBII-style instability**:

\[
\text{PBII}(n) = \overline{\sigma}(\text{composites}) - \overline{\sigma}(n)
\]

Higher PBII ⇒ more “prime-like” instability (less saturated structure).

This is **not** a primality test: it is a **structural instability index** that happens to separate primes and composites well in aggregate, and becomes a useful component in the fused score.

---

### 2.2 Omniatempo (time lens)

Goal: measure how much a 1D time series has changed *recently*.

Input:
- `series`: list/array of floats (e.g. logprob per step, loss per step, etc.)

Outputs:
- global mean and std,
- mean/std over a **short** and **long** recent window,
- a **regime_change_score**:

  - build histograms of:
    - last `short_window` points,
    - last `long_window` points,
  - compute a **symmetrized KL-like divergence** between those distributions.

Intuition:
- small regime score ⇒ recent behaviour similar to long-term behaviour,
- large regime score ⇒ clear shift in distribution (e.g., a reasoning chain that starts to drift).

This is a simple, computable proxy for “has the model’s behaviour recently changed in a suspicious way?”.

---

### 2.3 Omniacausa (causal lens, heuristic)

Goal: detect directional dependencies between multiple signals (multivariate time series).

Input:
- `series_dict`: mapping name → list/array of floats, e.g.:

  ```python
  {
      "logprob": [...],
      "entropy": [...],
      "contradiction": [...],
  }