OMNIA — Unified Structural Lenses (BASE ± TIME ± CAUSA ± TOKEN ± LCR)

Fused Ω-Score Engine · MB-X.01 / Massimiliano Brighindi

OMNIA is a model-agnostic structural engine that detects instability, drift, inconsistency and hidden structure across:

Numbers → multi-base symmetry + PBII (Prime Base Instability Index)

Time series → regime-shifts + symmetric KL

Multi-channel systems → lagged causal relationships

Token sequences → PBII-z instability for LLMs

External fact/numeric checks → LCR (Logical Coherence Reduction)


All lenses contribute to a single unified Ω-score, computed by
OMNIA_TOTALE and extended via LCR for hallucination detection.

The project is part of MB-X.01, the research line of Massimiliano Brighindi, designed for AI-safety evaluation, reproducibility, and structural interpretability.


---

1. Features

1.1 Omniabase (BASE)

Multi-base numerical structural analysis.

Digit entropy across bases

σ-symmetry, compactness, divisibility

PBII: prime-like structural instability index

Full entropy/σ signature and multi-base invariant profile


Use cases: integer analysis, anomaly scoring, numeric hallucination detection.


---

1.2 Omniatempo (TIME)

Temporal stability and drift detection.

μ/σ global statistics

Short vs long window distributions

Symmetric KL-divergence as regime change score

Reveals abrupt shifts, transitions, drifts


Use cases: reasoning drift, financial time series, sensor analysis.


---

1.3 Omniacausa (CAUSA)

Lagged causal structure over multivariate signals.

Correlation for all lags in [-max_lag, +max_lag]

Strongest lag extracted per pair

Edges emitted if |corr| ≥ threshold


Use cases: multi-signal inference, chain-of-thought analysis, implicit dependencies.


---

1.4 Token Lens (TOKEN)

PBII applied to token sequences.

Converts each token → numeric proxy

Computes PBII per token

Normalizes via z-scoring

TOKEN Ω = mean |z|


Use cases: LLM hallucination detection, CoT screening, instability segmentation.


---

1.5 LCR — Logical Coherence Reduction (FACT+NUMERIC Fusion)

External consistency engine integrating:

fact_consistency

numeric_consistency

gold_match

Structural Ω (optional)


Produces a fused external score Ω_ext, able to flag hallucinations via a confusion-matrix benchmark.

Included files:

LCR/LCR_CORE_v0.1.py

LCR/LCR_BENCHMARK_v0.1.py

data/lcr_samples.jsonl



---

1.6 Fused Ω Engine (Ω-TOTAL)

All lenses combine into:

Lens	Meaning	Contribution

BASE	numeric instability	PBII
TIME	temporal drift	log(1 + regime)
CAUSA	cross-channel causal structure	mean strength
TOKEN	CoT/token instability	z-mean
LCR	factual + numeric coherence	Ω_ext


Output:

Unified Ω-total

All sub-components

JSON-safe metadata for external audits



---

2. Repository Structure

omnia/
    __init__.py
    omniabase.py
    omniatempo.py
    omniacausa.py
    omniatoken.py
    omnia_totale.py
    engine.py
    kernel.py

LCR/
    LCR_CORE_v0.1.py
    LCR_BENCHMARK_v0.1.py

data/
    lcr_samples.jsonl
    gsm8k_model_outputs.jsonl   (optional external)

quick_omnia_test.py
gsm8k_benchmark_demo.py
README.md
requirements.txt

All modules are standalone and import-safe.


---

3. Installation

Requires Python 3.9+:

pip install numpy matplotlib

Clone:

git clone https://github.com/Tuttotorna/lon-mirror
cd lon-mirror


---

4. Quick Smoke Test

Run:

python quick_omnia_test.py

This verifies:

Omniabase (σ, entropy, PBII)

Omniatempo (regime score)

Omniacausa (lagged edges)

Token lens (PBII-z)

Ω fusion integrity


Expected output:

=== OMNIA SMOKE TEST ===
BASE: sigma_mean=..., PBII=...
TIME: regime_change_score=...
CAUSA: edges=...
TOKEN: z-mean=...
Ω_total = ...
components = {...}

If this passes, OMNIA is correctly installed.


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

5.4 Ω-Fusion

from omnia import omnia_totale_score

res = omnia_totale_score(
    n=173,
    series=series,
    series_dict={"s1": s1, "s2": s2}
)

print(res.omega_score)
print(res.components)


---

6. Benchmarks

6.1 PBII + GSM8K Hallucination Demo

python gsm8k_benchmark_demo.py

Outputs:

PBII distributions for primes vs composites

Synthetic AUC ~0.98

71% hallucination-reduction on long chains (>50 steps)

Token instability segmentation


These metrics serve as placeholders before full-scale evaluation.


---

6.2 LCR Benchmark

python LCR/LCR_BENCHMARK_v0.1.py

Computes:

TP / FP / TN / FN

detection_rate

precision

false positive rate

fused Ω_ext mean


Simple interface for external AI labs to validate consistency checks.


---

7. Limitations

Benchmarks synthetic; real tasks require large datasets (GSM8K, tool logs)

PBII sensitive to base selection

Causal lens = Pearson only (MI/TE/Granger planned)

Token lens uses proxy integer mapping

LCR depends on external backend quality



---

8. Author / Lineage

Massimiliano Brighindi — MB-X.01
Designer of:

Omniabase±

OMNIA_TOTALE

PBII

TOKEN instability

LCR fusion

Ω unified engine


This repository is the public, machine-readable mirror of the MB-X research line.


---

9. License

MIT License.


---

10. Citation

Brighindi, M. (2025).
OMNIA Structural Lens Engine (v2.0).
GitHub: https://github.com/Tuttotorna/lon-mirror
