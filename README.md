# OMNIA_TOTALE v2.0 — Unified Ω Engine

Author: **Massimiliano Brighindi (MB-X.01 / Omniabase±)**  
Engine formalization: **MBX IA**

---

## 1. What this repo is

This repository exposes **OMNIA_TOTALE v2.0** as a small, importable Python package:

- `omnia/` contains the **structural lenses**:
  - **Omniabase** → multi-base numeric structure + PBII (Prime Base Instability Index).
  - **Omniatempo** → temporal regime-change detection.
  - **Omniacausa** → lagged causal structure between time-series channels.
  - **kernel / engine** → generic fusion kernel + Ω-fusion on top of the lenses.
- `OMNIA_TOTALE_v2.0.py` is a **thin demo script** that runs the engine on synthetic data.

The goal is to provide a **model-agnostic coherence layer**: given numbers and time-series, the engine returns **per-lens scores** and a **fused Ω-score**, plus JSON-like metadata suitable for external integration (e.g. LLM safety / stability analysis).

---

## 2. Package layout

Minimal core layout:

```text
omnia/
  __init__.py        # public package surface
  omniabase.py       # multi-base numeric lens (PBII, signatures)
  omniatempo.py      # temporal stability / regime-change lens
  omniacausa.py      # lagged causal-structure lens
  kernel.py          # generic lens-kernel (context, LensResult, KernelResult)
  engine.py          # OMNIA_TOTALE engine built on top of the kernel

OMNIA_TOTALE_v2.0.py # demo script using omnia.engine.run_omnia_totale(...)

Other files in the repository (earlier prototypes, reports, experiments) are to be considered legacy or experimental unless explicitly documented here.


---

3. Lenses (high-level)

3.1 Omniabase — multi-base numeric lens (PBII)

Implemented in omnia/omniabase.py.

Core ideas:

Represent an integer n in multiple bases b (2, 3, 5, 7, 11, …).

Compute normalized digit entropy per base:

H_norm ∈ [0, 1], with 0 = fully structured, 1 = maximally random.


Define a base symmetry score sigma_b(n) that:

rewards low entropy,

softly penalizes long representations,

adds a divisibility bonus when n % b == 0.



OmniabaseSignature provides:

per-base σ scores and entropies,

mean σ and mean entropy across bases.


pbii_index(n) (Prime Base Instability Index):

computes a saturation value from a window of composite numbers,

returns PBII(n) = saturation − sigma_mean(n),

higher PBII ≈ more “prime-like” structural instability.


3.2 Omniatempo — temporal stability lens

Implemented in omnia/omniatempo.py.

Given a 1D time series:

computes global mean / std,

computes short-window vs long-window mean / std,

builds histograms on short vs long segments,

computes a symmetric KL-like divergence between these histograms.


The main output is regime_change_score:

0 → stable regime,

larger values → stronger distribution shift.


3.3 Omniacausa — lagged causal lens

Implemented in omnia/omniacausa.py.

Given a dict of named time-series:

for each pair (source, target):

scans lags in [-max_lag, +max_lag],

computes lagged Pearson correlations,

keeps the strongest (in absolute value),


emits an edge when |corr| >= strength_threshold.


Output:

list of OmniaEdge(source, target, lag, strength),

can be read as a lagged influence graph (heuristic, not full causal discovery).



---

4. Kernel and engine

4.1 omnia.kernel

Defines the generic lens kernel:

OmniaContext → holds inputs (n, series, series_dict, extra).

LensResult → name, scores: Dict[str, float], metadata: Dict.

KernelResult → fused Ω + per-lens results.

OmniaKernel → registry of lenses:

register_lens(name, fn, weight)

run(context) → executes all registered lenses, fuses their scores["omega"].



Fusion:

each lens returns a scalar omega component,

kernel computes a weighted sum:

fused_omega = Σ (weight_i * omega_i).



4.2 omnia.engine

Builds OMNIA_TOTALE on top of the kernel:

wraps the three lenses as kernel-compatible functions:

_lens_omniabase(ctx)

_lens_omniatempo(ctx)

_lens_omniacausa(ctx)


provides:

build_default_engine(w_base=1.0, w_tempo=1.0, w_causa=1.0)

run_omnia_totale(n, series, series_dict, ...) -> KernelResult



This is the preferred entrypoint for external systems.


---

5. Demo script (OMNIA_TOTALE_v2.0.py)

OMNIA_TOTALE_v2.0.py is a thin, executable demo:

builds synthetic inputs:

a prime integer n = 173 (you can toggle to a composite),

a time series with a late regime shift,

three channels with a known lagged dependency (s1 → s2, s3 = noise),


calls:


from omnia.engine import run_omnia_totale

result = run_omnia_totale(
    n=n,
    series=series,
    series_dict=series_dict,
    w_base=1.0,
    w_tempo=1.0,
    w_causa=1.0,
)

prints:

fused Ω,

per-lens omega and any additional scores.



To run:

pip install numpy
python OMNIA_TOTALE_v2.0.py


---

6. Status

The package core (omnia/ + OMNIA_TOTALE_v2.0.py) is executable and intended as the stable surface.

Other files in the repository are prototypes and experiments:

early monolithic versions of OMNIA_TOTALE,

PBII experiments,

reports and HTML visualisations.


Benchmarks (e.g. GSM8K-style hallucination detection, prime vs composite AUC) are currently work in progress and should not be treated as formal claims until dedicated benchmark scripts and datasets are fully published and documented here.



---

7. Intended use

This project is meant for:

researchers and engineers exploring structural coherence metrics,

experiments on LLM stability and reasoning drift,

numeric / time-series analysis where multi-base and temporal/causal lenses are informative.


It is explicitly designed to be:

transparent (simple Python, no hidden state),

deterministic given inputs,

model-agnostic (works with any system capable of exporting the required numeric traces).


