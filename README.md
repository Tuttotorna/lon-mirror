# OMNIA — Unified Structural Lenses  
**(BASE ± TIME ± CAUSA ± TOKEN ± LCR)**

**Fused Ω-Score Engine · MB-X.01**  
**Author: Massimiliano Brighindi**

---

## Overview

**OMNIA** is a **model-agnostic structural measurement engine** designed to detect:

- instability  
- drift  
- inconsistency  
- hidden structure  

across heterogeneous domains using **unified structural lenses**.

OMNIA does **not** decide.  
OMNIA does **not** optimize.  
OMNIA **measures structural coherence**.

All measurements are fused into a single **Ω-total score**, computed by the `OMNIA_TOTALE` engine and optionally extended by **LCR** for external factual and numeric coherence validation.

This repository is the public, machine-readable mirror of **MB-X.01**, the research line of **Massimiliano Brighindi**, focused on:

- structural interpretability  
- reproducibility  
- AI-safety diagnostics  
- architecture-agnostic evaluation  

---

## Quick Entry (Colab)

**Canonical · Reproducible run**

colab/OMNIA_REAL_RUN.ipynb

https://colab.research.google.com/github/Tuttotorna/lon-mirror/blob/main/colab/OMNIA_REAL_RUN.ipynb

**Exploratory · Inspection**

colab/OMNIA_DEMO_INSPECT.ipynb

https://colab.research.google.com/github/Tuttotorna/lon-mirror/blob/main/colab/OMNIA_DEMO_INSPECT.ipynb

> The demo notebook is not frozen and is not used for benchmarks.

---

## 1. Structural Lenses

### 1.1 Omniabase (BASE)

Multi-base numerical structure analysis.

**Signals**
- Digit entropy across bases  
- σ-symmetry, compactness, divisibility  
- PBII (Prime Base Instability Index)  
- Multi-base invariant profile  

**Use cases**
- Integer analysis  
- Numeric anomaly detection  
- Numeric hallucination detection  

---

### 1.2 Omniatempo (TIME)

Temporal stability and drift detection.

**Signals**
- Global μ / σ statistics  
- Short vs long window distributions  
- Symmetric KL-divergence  
- Regime change score  

**Use cases**
- LLM output drift  
- Time series analysis  
- Long-chain reasoning stability  

---

### 1.3 Omniacausa (CAUSA)

Lagged causal structure extraction over multivariate signals.

**Signals**
- Correlation across all lags  
- Strongest lag per signal pair  
- Edge emission if |corr| ≥ threshold  

**Use cases**
- Multi-signal inference  
- Hidden causal relationships  
- Cross-channel dependency mapping  

---

### 1.4 Token Lens (TOKEN)

PBII applied to token sequences.

**Pipeline**
- token → integer proxy  
- PBII per token  
- z-score normalization  
- TOKEN Ω = mean(|z|)  

**Use cases**
- Hallucination detection  
- Token-level instability segmentation  

---

### 1.5 LCR — Logical Coherence Reduction  
*(FACT + NUMERIC fusion)*

External coherence engine integrating:

- factual consistency  
- numeric consistency  
- optional structural Ω  

**Output**
- External Ω_ext score  
- Confusion-matrix metrics  

Modules:
- `LCR/LCR_CORE_v0.1.py`  
- `LCR/LCR_BENCHMARK_v0.1.py`  

---

### 1.6 Ω-TOTAL (Fused Engine)

All lenses combine into a single structural score.

| Lens  | Contribution              |
|------:|---------------------------|
| BASE  | PBII                      |
| TIME  | log(1 + regime_score)     |
| CAUSA | mean edge strength        |
| TOKEN | mean |z|                  |
| LCR   | Ω_ext                     |

**Output**
- Unified Ω-total  
- Full component breakdown  
- JSON-safe metadata  

---

## 2. Repository Structure

lon-mirror/ ├── omnia/ ├── adapters/ ├── LCR/ ├── data/ ├── examples/ ├── colab/ │   ├── OMNIA_REAL_RUN.ipynb │   └── OMNIA_DEMO_INSPECT.ipynb ├── quick_omnia_test.py ├── gsm8k_benchmark_demo.py ├── INTERFACE.md ├── README.md └── requirements.txt

All modules are **standalone**, **deterministic**, and **import-safe**.

---

## 3. Installation

**Requirements**
- Python ≥ 3.9

```bash
pip install numpy matplotlib
git clone https://github.com/Tuttotorna/lon-mirror
cd lon-mirror


---

4. Smoke Test

python quick_omnia_test.py

Validates:

BASE, TIME, CAUSA, TOKEN lenses

Ω fusion integrity



---

5. Usage (Core)

from omnia import omnia_totale_score

res = omnia_totale_score(
    n=173,
    series=series,
    series_dict={"s1": s1, "s2": s2}
)

print(res.omega_score)
print(res.components)


---

6. LLM Integration (Raw)

from adapters.llm_output_adapter import analyze_llm_output

report = analyze_llm_output(text, tokens)
print(report.omega_score, report.flags)

OMNIA is:

post-inference

semantics-free

decision-agnostic



---

7. Benchmarks

PBII + GSM8K (Demo)

python gsm8k_benchmark_demo.py

Synthetic results:

AUC ≈ 0.98

~71% hallucination reduction on long chains


LCR Benchmark

python LCR/LCR_BENCHMARK_v0.1.py

Designed for external audit.


---

8. Limitations

Benchmarks are synthetic

PBII sensitive to base choice

CAUSA uses Pearson correlation

TOKEN requires numeric proxy

LCR depends on external backends



---

9. Author / Lineage

Massimiliano Brighindi — MB-X.01

Designer of:

Omniabase±

OMNIA_TOTALE

PBII

TOKEN instability lens

LCR fusion


This repository is the authoritative public mirror of the MB-X research line.


---

10. License

MIT License.


---

11. Citation

Brighindi, M. (2025).
OMNIA Structural Lens Engine (v2.0)
GitHub: https://github.com/Tuttotorna/lon-mirror

