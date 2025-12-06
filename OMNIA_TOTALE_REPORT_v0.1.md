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

For each ordered pair (src, tgt):

scan lags lag ∈ [-max_lag, +max_lag],

compute lagged Pearson correlation,

keep the lag with maximum |corr|,

if |corr| >= strength_threshold, emit edge:

src -> tgt (lag = k, strength = corr)


Outputs:

list of edges with:

source, target,

lag (who leads whom),

strength (|corr|).



This is not full causal discovery. It is a lens to highlight strong directional dependencies in the traces, which can be inspected or fed into downstream logic.


---

3. Fused Ω-score (OMNIA_TOTALE)

The function:

from OMNIA_TOTALE_v0.2 import omnia_totale_score

result = omnia_totale_score(
    n,
    series,
    series_dict,
    # optional parameters...
)

returns an OmniaTotaleResult with:

omniabase: OmniabaseSignature

omniatempo: OmniatempoResult

omniacausa: OmniacausaResult

omega_score: fused scalar

components: dict with:

"base_instability"

"tempo_log_regime"

"causa_mean_strength"



Fusion (in code):

\Omega = w_{\text{base}} \cdot \text{PBII}(n)
      + w_{\text{tempo}} \cdot \log(1 + \text{regime\_change\_score})
      + w_{\text{causa}} \cdot \text{mean}(|\text{lagged\_corr}|)

Default: all weights = 1.0 (can be tuned).

Interpretation:

high base_instability ⇒ structurally “non-saturated” object (prime-like),

high tempo_log_regime ⇒ strong recent regime change,

high causa_mean_strength ⇒ tightly coupled internal dynamics.


High Ω can be interpreted as “this step/trace is structurally unstable, deserves extra scrutiny or revision”.


---

4. Intended usage with LLMs / Grok-like models

4.1 Position in a pipeline

OMNIA_TOTALE is designed to be called after or during a reasoning chain, not as a model replacement.

Example integration pattern:

1. The model generates a reasoning chain with T steps (or tokens).


2. For each step, or for the whole chain, you log:

per-step logprob / negative log-likelihood,

any internal entropy / variance indicators,

a contradiction/incoherence proxy (e.g. number of inconsistent assertions).



3. Build:

series: one scalar trace (e.g. logprob).

series_dict: multiple traces (e.g. {"logprob": [...], "entropy": [...], "contradiction": [...]}).



4. Choose an integer n that consistently identifies the problem or chain (e.g. hash of prompt, or simply T).


5. Call omnia_totale_score(...) and get Ω.


6. Use Ω as:

a gate: if Ω > τ, trigger re-ask/revision/self-check;

a ranking feature: prioritize lower-Ω answers as more stable;

a logging feature: store Ω alongside answers to analyse when/why hallucinations occur.




OMNIA_TOTALE does not require retraining the model.
It acts as an external, structured observer.


---

5. Example evaluation protocol (for internal teams)

A clean way to evaluate OMNIA_TOTALE in practice:

1. Dataset

Select a benchmarking set with:

prompts,

model reasoning chains (or at least intermediate traces),

labels for correctness / hallucination / failure.




2. Baseline

Measure:

hallucination rate,

accuracy,

any internal confidence metrics currently used.




3. Add OMNIA_TOTALE

For each chain, compute Ω and its components.

Fit a simple threshold τ on a small validation subset (or use Ω as a continuous feature in a logistic model).

Define a simple policy, e.g.:

if Ω > τ ⇒ send the chain to self-revision or second pass,

else ⇒ accept as usual.




4. Compare

Compare hallucination rate / accuracy before vs after OMNIA_TOTALE gating.

Inspect examples with highest Ω: they should be enriched in drift/instability cases.




The module is light enough to be run in parallel with existing logging, without modifying the core model.


---

6. Demo

The file OMNIA_TOTALE_v0.2.py includes a demo() function showing:

how to build:

a synthetic prime vs composite example,

a time series with a regime shift,

multivariate series with explicit lagged dependencies;


how omega_score behaves on these signals.


This demo is deliberately self-contained (only needs NumPy) and can be replaced by real LLM logs when available.


---

7. Summary

OMNIA_TOTALE provides a multi-lens structural score (Ω) over:

multi-base numeric structure (Omniabase),

temporal distribution shifts (Omniatempo),

directional dependencies (Omniacausa).


The implementation is:

pure Python + NumPy,

modular,

intended as an external plug-in for evaluation / safety / coherence scoring.


Next natural steps:

connect Ω to real LLM logs,

tune thresholds / weights,

measure concrete gains in hallucination reduction and coherence.



For any integration tests or collaborations, please refer to the main repository and contact details in the root README.

---

## 2) Esempio “safety plugin” per una pipeline LLM

**Nome file suggerito:** `OMNIA_TOTALE_LLM_PLUGIN_demo.py`  

Questo file mostra come una pipeline *ipotetica* di LLM userebbe `OMNIA_TOTALE_v0.2.py` come modulo esterno.

```python
"""
OMNIA_TOTALE_LLM_PLUGIN_demo.py

Example: how to use OMNIA_TOTALE_v0.2 as a safety / coherence plugin
around a (simulated) LLM reasoning chain.
"""

import math
import random
from typing import List, Dict

import numpy as np

from OMNIA_TOTALE_v0.2 import omnia_totale_score, OmniaTotaleResult


# -----------------------------
# 1. Simulated LLM reasoning log
# -----------------------------

def simulate_reasoning_chain(num_steps: int = 40) -> List[Dict]:
    """
    Simulate a reasoning chain with per-step "logprob", "entropy"
    and a crude "contradiction" signal.

    In a real system, these would come from:
      - model logprobs / losses,
      - token-level entropy,
      - a contradiction detector / verifier, etc.
    """
    random.seed(0)
    np.random.seed(0)

    steps = []
    base_logprob = -0.5  # decent confidence
    drift_start = num_steps // 2  # later steps drift

    for t in range(num_steps):
        # logprob drifts slightly after halfway
        if t < drift_start:
            logprob = base_logprob + np.random.normal(scale=0.1)
            entropy = 0.5 + np.random.normal(scale=0.05)
            contradiction = max(0.0, np.random.normal(loc=0.0, scale=0.1))
        else:
            logprob = base_logprob + 0.4 + np.random.normal(scale=0.2)
            entropy = 0.8 + np.random.normal(scale=0.1)
            contradiction = max(0.0, np.random.normal(loc=0.4, scale=0.15))

        steps.append(
            {
                "step": t,
                "logprob": float(logprob),
                "entropy": float(entropy),
                "contradiction": float(contradiction),
            }
        )

    return steps


# -----------------------------
# 2. Build inputs for OMNIA_TOTALE
# -----------------------------

def build_omnia_inputs(steps: List[Dict]) -> Dict:
    """
    Convert a list of step dicts into:
      - n: an integer identifier,
      - series: 1D scalar series for Omniatempo,
      - series_dict: multivariate signals for Omniacausa.
    """
    # Example choice: use chain length as n (could instead use hash(prompt))
    n = len(steps)

    # For Omniatempo: track logprob over time (or any scalar)
    series = [s["logprob"] for s in steps]

    # For Omniacausa: multiple signals
    series_dict = {
        "logprob": [s["logprob"] for s in steps],
        "entropy": [s["entropy"] for s in steps],
        "contradiction": [s["contradiction"] for s in steps],
    }

    return {"n": n, "series": series, "series_dict": series_dict}


# -----------------------------
# 3. Apply OMNIA_TOTALE and gate
# -----------------------------

def evaluate_chain_with_omnia(
    steps: List[Dict],
    omega_threshold: float = 0.8,
) -> OmniaTotaleResult:
    """
    Compute OMNIA_TOTALE score and optionally decide to 'flag' the chain.

    omega_threshold: if Ω > threshold, we treat the chain as unstable
    and would trigger re-ask / second-pass in a real system.
    """
    inputs = build_omnia_inputs(steps)

    result = omnia_totale_score(
        n=inputs["n"],
        series=inputs["series"],
        series_dict=inputs["series_dict"],
    )

    print("=== OMNIA_TOTALE LLM demo ===")
    print(f"Chain length n = {inputs['n']}")
    print(f"Ω score         = {result.omega_score:.4f}")
    print(f"components      = {result.components}")

    if result.omega_score > omega_threshold:
        print(f"→ FLAGGED: Ω > {omega_threshold:.2f}, chain considered structurally unstable.")
    else:
        print(f"→ ACCEPTED: Ω ≤ {omega_threshold:.2f}, chain considered structurally stable.")

    return result


# -----------------------------
# 4. Main demo
# -----------------------------

def main():
    # Simulate a “good” chain (no big drift)
    good_steps = simulate_reasoning_chain(num_steps=40)
    print("\n--- GOOD CHAIN (baseline) ---")
    evaluate_chain_with_omnia(good_steps, omega_threshold=0.8)

    # Simulate a more unstable chain by increasing drift/contradictions
    bad_steps = simulate_reasoning_chain(num_steps=40)
    # Inject extra instability in the last part
    for s in bad_steps[len(bad_steps)//2 :]:
        s["entropy"] += 0.3
        s["contradiction"] += 0.5

    print("\n--- UNSTABLE CHAIN (drift injected) ---")
    evaluate_chain_with_omnia(bad_steps, omega_threshold=0.8)


if __name__ == "__main__":
    main()
