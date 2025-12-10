OMNIA — Unified Structural Lenses Engine (Ω)

MB-X.01 · Massimiliano Brighindi

OMNIA is a model-agnostic structural scoring engine that detects instability, drift, inconsistency and hidden structure across:

Numbers → multi-base symmetry + PBII instability

Time series → regime shifts via symmetric KL

Multi-channel systems → lagged causal relationships

Token sequences → PBII-z instability for LLM reasoning

External factual checks → LCR (Logical Coherence Reduction)


All lenses contribute to a single unified Ω-score, designed for:

hallucination detection

reasoning drift monitoring

chain-of-thought auditing

AI-safety evaluation

reproducible structural interpretability


OMNIA is part of MB-X.01, the research line of Massimiliano Brighindi.


---

1. Core Idea

LLMs hallucinate because they lack a structural metric that can independently evaluate:

numeric instability

temporal inconsistency

causal incoherence

token-level divergence

factual/numeric mismatch


OMNIA provides exactly that:
a zero-semantic, fully reproducible, model-agnostic scoring system.

The result is a single number: Ω
Higher Ω → more instability → higher hallucination probability.


---

2. Why OMNIA Works

Semantic-free

No embeddings, no language model required, no training, no fine-tuning.

Invariant

PBII (Prime Base Instability Index) reveals structure across numeric and token spaces.

Modular

Each lens can be used independently or fused.

Plug-and-play for LLMs

Outputs JSON-safe scores usable in any inference pipeline.


---

3. The Five Lenses

3.1 BASE — Omniabase

Multi-base digit entropy

σ-symmetry

Compactness

Divisibility structure

PBII instability
Use: numeric anomalies, hallucinated numbers, unstable reasoning jumps.


3.2 TIME — Omniatempo

Global vs local distribution comparison

Symmetric KL divergence

Drift and regime shift detection
Use: reasoning derailment, tool-use instability.


3.3 CAUSA — Omniacausa

Lagged correlation graph

Implicit dependency detection
Use: multi-signal reasoning, chain-of-thought structure.


3.4 TOKEN — PBII-Z Token Lens

Token → integer proxy

PBII per token

Z-scored instability
Use: CoT instability segmentation, hallucination hotspots.


3.5 LCR — Logical Coherence Reduction

fact_consistency

numeric_consistency

gold_match

optional Ω_struct
Produces a fused Ω_ext with confusion matrix benchmark.



---

4. Installation

pip install numpy matplotlib
git clone https://github.com/Tuttotorna/lon-mirror
cd lon-mirror


---

5. Quick Smoke Test (20 seconds)

python quick_omnia_test.py

This validates:

Omniabase (σ, entropy, PBII)

Omniatempo (regime change)

Omniacausa (edges)

TOKEN lens

Ω-total fusion


Expected:

=== OMNIA SMOKE TEST ===
BASE: sigma_mean=..., PBII=...
TIME: regime_change_score=...
CAUSA: edges=...
TOKEN: z-mean=...
Ω_total = ...
components = {...}


---

6. Benchmarks

6.1 PBII + GSM8K Synthetic Hallucination Detection

python gsm8k_benchmark_demo.py

Outputs:

PBII distributions for primes vs composites

Synthetic AUC ≈ 0.98

Hallucination detection on long GSM8K chains: ≈71%

Token instability segmentation


These results are placeholders for full-scale lab testing.


---

6.2 LCR Benchmark

python LCR/LCR_BENCHMARK_v0.1.py

Outputs:

TP / FP / TN / FN

detection_rate

precision

false positive rate

fused Ω_ext mean



---

7. Repository Structure

omnia/
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
    gsm8k_model_outputs.jsonl (optional)

quick_omnia_test.py
gsm8k_benchmark_demo.py
README.md
OVERVIEW.md
requirements.txt


---

8. Intended Applications

Safe AI deployment

Real-time hallucination suppression

CoT scoring & filtering

Structured reasoning audits

Dataset curation

Detecting drift in autonomous agents

Hybrid safety pipelines for labs (LLM-agnostic)



---

9. Limitations (current)

PBII sensitive to base selection

Causal lens limited to Pearson correlation

Token proxy mapping simplistic

Synthetic benchmarks only (no large-scale GSM8K yet)

LCR backend currently numeric + fact-only



---

10. Author

Massimiliano Brighindi
MB-X.01 · OMNIA Research Line
Designer of:

Omniabase±

PBII

OMNIA_TOTALE

TOKEN instability

LCR fusion


This repository is the machine-readable mirror of the MB-X lineage.


---

11. Citation

Brighindi, M. (2025).
OMNIA Structural Lens Engine (v2.0).
GitHub: https://github.com/Tuttotorna/lon-mirror
