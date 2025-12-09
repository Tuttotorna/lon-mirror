OMNIA — Unified Structural Lenses (Omniabase ± Omniatempo ± Omniacausa ± Token)

Fused Ω-Score Engine · MB-X.01 / Massimiliano Brighindi

OMNIA is a model-agnostic structural analysis engine that detects instability, inconsistency and regime-shifts across:

Numbers (multi-base symmetry + PBII instability)

Time series (distribution drift / change-points)

Multi-channel systems (lagged causal structure)

Token sequences from LLMs (PBII → z-score instability)


All four lenses produce an Ω-component, and the fused engine
OMNIA_TOTALE outputs a single unified Ω-score with structured metadata.

The project is part of MB-X.01 research by Massimiliano Brighindi, designed for transparency, reproducibility and AI-safety evaluations.


---

1. Features

1.1 Omniabase (BASE)

Multi-base numeric structural lens.

Digit entropy across many bases

σ-symmetry, divisibility and length structure

PBII (Prime Base Instability Index): prime-like instability detector

Full multi-base signature with entropy/σ profiles


Use cases: integer analysis, anomaly detection, numeric hallucination scoring.


---

1.2 Omniatempo (TIME)

Temporal stability lens.

Global μ/σ

Short vs. long window distributions

Symmetric KL-divergence → regime_change_score

Highlights abrupt shifts, drifts, hidden transitions


Use cases: financial time series, sensor data, LLM reasoning drift.


---

1.3 Omniacausa (CAUSA)

Lagged causal-structure lens.

Computes correlations for all lags in [-max_lag, +max_lag]

Extracts strongest lag relationship for each pair

Emits edges when |corr| ≥ threshold


Use cases: multi-signal inference, hidden coupling, structured reasoning.


---

1.4 Token Lens (TOKEN)

PBII applied to text.

Maps each token to an integer proxy

Computes PBII for each token

Converts to z-scores to expose structural instability

TOKEN Ω-component = mean |z|


Use cases: LLM hallucination detection, chain-of-thought auditing.


---

1.5 Fused Engine (Ω-TOTAL)

omnia_totale_score or run_omnia_totale() combine:

Lens	Meaning	Contribution

BASE	numeric instability	PBII
TIME	temporal drift	log(1 + regime)
CAUSA	causal structure	mean
TOKEN	text instability	mean


Output:

Unified Ω-score

Individual components

Full metadata (JSON-safe)



---

2. Repository Structure

omnia/
    __init__.py
    omniabase.py
    omniatempo.py
    omniacausa.py
    omnia_totale.py
    engine.py
    kernel.py

quick_omnia_test.py      ← smoke test for the entire system
gsm8k_benchmark_demo.py  ← synthetic benchmark for PBII and hallucinations
README.md                ← this file
requirements.txt

All modules are standalone and import-safe.


---

3. Installation

Requires Python 3.9+ and:

pip install numpy matplotlib

Clone:

git clone https://github.com/Tuttotorna/lon-mirror
cd lon-mirror


---

4. Quick Start (Smoke Test)

To verify that OMNIA is correctly installed and all structural lenses are working:

python quick_omnia_test.py

This performs:

Omniabase test (PBII + σ/entropy)

Omniatempo test (regime-change detection)

Omniacausa test (lagged causal graph)

Token-lens test (PBII → z-scores)

Ω-fusion test


Expected output (example):

=== OMNIA SMOKE TEST ===
BASE: sigma_mean=..., entropy_mean=..., PBII=...
TIME: regime_change_score=...
CAUSA: edges found=...
TOKEN: z-mean=...
Ω_total = ...
components = {base: ..., tempo: ..., causa: ..., token: ...}

If this runs, the full framework is working.


---

5. Basic Usage

5.1 Omniabase

from omnia import omniabase_signature, pbii_index

sig = omniabase_signature(173)
pbii = pbii_index(173)

5.2 Omniatempo

from omnia import omniatempo_analyze
res = omniatempo_analyze(series)

5.3 Omniacausa

from omnia import omniacausa_analyze
res = omniacausa_analyze({"s1": s1, "s2": s2, "s3": s3})

5.4 Full Ω-Fusion

from omnia import omnia_totale_score

res = omnia_totale_score(
    n=173,
    series=series,
    series_dict={"s1": s1, "s2": s2},
)

print(res.omega_score)
print(res.components)


---

6. Benchmarks

GSM8K Hallucination Demo

python gsm8k_benchmark_demo.py

Shows PBII-based hallucination detection + synthetic AUC for primes vs composites.

All metrics are placeholders until validated on full-scale experiments.


---

7. Limitations

Benchmarks are synthetic; real evaluations require large datasets

PBII is sensitive to choice of bases

Causal lens uses Pearson only (future upgrade: MI / Granger)

Token lens currently requires integer proxy maps



---

8. Author / Research Lineage

Massimiliano Brighindi — MB-X.01
Designer of Omniabase±, OMNIA_TOTALE and the Ω-fusion framework.

This repository is a public, machine-readable mirror of the evolving MB-X research line.


---

9. License

MIT License.


---

10. Citation

Brighindi, M. (2025).
OMNIA Structural Lens Engine (v2.0).
GitHub: https://github.com/Tuttotorna/lon-mirror


