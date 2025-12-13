# OMNIA — Unified Structural Lenses  
(BASE ± TIME ± CAUSA ± TOKEN ± LCR)

**Fused Ω-Score Engine · MB-X.01**  
**Author: Massimiliano Brighindi**

---

## Overview

**OMNIA** is a **model-agnostic structural measurement engine**.

It is designed to **detect, quantify, and expose structural instability** in numbers, sequences, time series, causal systems, and AI outputs using **independent structural lenses**.

OMNIA **does not decide**.  
OMNIA **does not optimize**.  
OMNIA **does not generate answers**.

OMNIA **measures structure**.

All outputs are **diagnostic**, **deterministic**, and **architecture-agnostic**.  
Any decision logic must remain **external**.

This repository is the public, machine-readable implementation of the **MB-X.01** research line by **Massimiliano Brighindi**.

---

## Core Principle

> **Truth is structural invariance.**  
> If a structure collapses under one independent lens, it is unstable.

OMNIA does not approximate truth.  
It measures **coherence under multiple independent constraints**.

---

## 1. Structural Lenses

### 1.1 Omniabase (BASE)

**Multi-base structural analysis of integers and numeric proxies.**

#### Core signals
- Digit entropy across multiple bases
- σ-symmetry and compactness
- Divisibility irregularities
- **PBII — Prime Base Instability Index**
- Multi-base invariant signature

#### Use cases
- Integer structure analysis
- Numeric anomaly detection
- Numeric hallucination sensing
- Prime-like structure discrimination

---

### 1.2 Omniatempo (TIME)

**Temporal stability and drift detection.**

#### Core signals
- Global μ / σ statistics
- Short-window vs long-window divergence
- Symmetric KL-divergence
- Regime-change score

#### Use cases
- LLM reasoning drift
- Time-series instability
- Sensor stream monitoring
- Long-chain reasoning degradation

---

### 1.3 Omniacausa (CAUSA)

**Lagged causal structure extraction over multivariate signals.**

#### Core signals
- Correlation across all lags
- Strongest lag per signal pair
- Edge emission when |corr| ≥ threshold

#### Use cases
- Hidden causal relationships
- Multi-signal dependency mapping
- Chain-of-thought causal inspection
- Structural inference (non-semantic)

---

### 1.4 Token Lens (TOKEN)

**Structural instability analysis applied to token sequences.**

#### Pipeline
- token → integer proxy
- PBII per token
- z-score normalization
- TOKEN Ω = mean(|z|)

#### Use cases
- Hallucination detection
- Token-level instability segmentation
- Chain-of-thought fracture localization

---

### 1.5 LCR — Logical Coherence Reduction  
*(FACT + NUMERIC fusion)*

**External coherence validation layer.**

#### Integrated signals
- Factual consistency
- Numeric consistency
- Gold-reference matching
- Optional structural Ω input

#### Output
- External fused score Ω_ext
- Confusion-matrix metrics

Included modules:

LCR/LCR_CORE_v0.1.py LCR/LCR_BENCHMARK_v0.1.py data/lcr_samples.jsonl

---

## 2. Fused Ω Engine (Ω-TOTAL)

All lenses combine into a unified **structural score**.

| Lens  | Meaning                    | Contribution                  |
|------|----------------------------|-------------------------------|
| BASE | Numeric instability        | PBII                          |
| TIME | Temporal drift             | log(1 + regime_score)         |
| CAUSA| Cross-channel structure    | mean edge strength            |
| TOKEN| Token instability          | mean |z|                      |
| LCR  | External coherence         | Ω_ext                         |

### Output
- Unified Ω-total score
- Full component breakdown
- JSON-safe metadata
- Deterministic reproducibility

---

## 3. Repository Structure

omnia/ init.py omniabase.py omniatempo.py omniacausa.py omniatoken.py omnia_totale.py engine/ kernel.py

adapters/ llm_output_adapter.py

LCR/ LCR_CORE_v0.1.py LCR_BENCHMARK_v0.1.py

data/ lcr_samples.jsonl gsm8k_model_outputs.jsonl   (optional)

examples/ omnia_gate_demo.py

quick_omnia_test.py gsm8k_benchmark_demo.py INTERFACE.md requirements.txt README.md

All modules are **standalone**, **deterministic**, and **import-safe**.

---

## 4. Installation

### Requirements
- Python ≥ 3.9

Install dependencies:
```bash
pip install numpy matplotlib

Clone repository:

git clone https://github.com/Tuttotorna/lon-mirror
cd lon-mirror


---

5. Quick Smoke Test

Run:

python quick_omnia_test.py

Validates:

Omniabase (σ, entropy, PBII)

Omniatempo (regime detection)

Omniacausa (lagged edges)

Token lens

Ω-fusion consistency


Expected output (example):

=== OMNIA SMOKE TEST ===
BASE: PBII=...
TIME: regime_change_score=...
CAUSA: edges=...
TOKEN: z_mean=...
Ω_total = ...
components = {...}


---

6. Basic Usage

Omniabase

from omnia import omniabase_signature, pbii_index

sig = omniabase_signature(173)
pbii = pbii_index(173)

Omniatempo

from omnia import omniatempo_analyze

res = omniatempo_analyze(series)

Omniacausa

from omnia import omniacausa_analyze

res = omniacausa_analyze({
    "s1": s1,
    "s2": s2,
    "s3": s3
})

Ω-Fusion

from omnia import omnia_totale_score

res = omnia_totale_score(
    n=173,
    series=series,
    series_dict={"s1": s1, "s2": s2}
)

print(res.omega_score)
print(res.components)


---

7. LLM Integration (Raw)

OMNIA can be used as a structural sensor on LLM outputs.

from adapters.llm_output_adapter import analyze_llm_output

report = analyze_llm_output(
    text=llm_output,
    tokens=token_ids,
)

print(report.omega_score, report.flags)

No framework dependency.
No policy logic.
Decision layers remain external.


---

8. Benchmarks

8.1 PBII + GSM8K Demo

python gsm8k_benchmark_demo.py

Outputs:

PBII distributions

Synthetic AUC ≈ 0.98

~71% hallucination reduction on long chains

Token instability segmentation


(Values are synthetic placeholders.)


---

8.2 LCR Benchmark

python LCR/LCR_BENCHMARK_v0.1.py

Outputs:

TP / FP / TN / FN

Precision, recall, FPR

Mean Ω_ext


Designed for external AI-lab validation.


---

9. Limitations

Benchmarks are synthetic

PBII sensitive to base selection

Causal lens uses Pearson correlation
(MI / TE / Granger planned)

Token lens requires numeric proxy

LCR depends on external backend quality



---

10. Author & Lineage

Massimiliano Brighindi — MB-X.01

Designer of:

Omniabase±

PBII

TOKEN instability lens

LCR fusion

OMNIA unified Ω engine


This repository is the public executable mirror of the MB-X research line.

External Logical Origin Node:
https://massimiliano.neocities.org/


---

11. License

MIT License.


---

12. Citation

Brighindi, M. (2025).
OMNIA — Unified Structural Lens Engine (v2.0)
GitHub: https://github.com/Tuttotorna/lon-mirror

