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

across heterogeneous domains using a system of **unified structural lenses**.

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

## 1. Structural Lenses

### 1.1 Omniabase (BASE)

Multi-base numerical structure analysis.

**Core signals**
- Digit entropy across bases  
- σ-symmetry, compactness, divisibility  
- PBII (Prime Base Instability Index)  
- Full entropy / σ signature  
- Multi-base invariant profile  

**Use cases**
- Integer analysis  
- Numeric anomaly detection  
- Numeric hallucination detection  
- Prime-like structure identification  

---

### 1.2 Omniatempo (TIME)

Temporal stability and drift detection.

**Core signals**
- Global μ / σ statistics  
- Short vs long window distributions  
- Symmetric KL-divergence  
- Regime change score  

Detects abrupt transitions, reasoning drift, and distribution shifts.

**Use cases**
- LLM output drift  
- Financial time series  
- Sensor streams  
- Long-chain reasoning stability  

---

### 1.3 Omniacausa (CAUSA)

Lagged causal structure extraction over multivariate signals.

**Core signals**
- Correlation across all lags  
- Strongest lag per signal pair  
- Edge emission if |corr| ≥ threshold  

**Use cases**
- Multi-signal inference  
- Hidden causal relationships  
- Chain-of-thought structure analysis  
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
- Chain-of-thought fracture detection  

---

### 1.5 LCR — Logical Coherence Reduction  
*(FACT + NUMERIC fusion)*

External coherence engine integrating:

- fact consistency  
- numeric consistency  
- optional structural Ω  

**Output**
- Fused external score Ω_ext  
- Confusion-matrix benchmark metrics  

**Included modules**
- `LCR/LCR_CORE_v0.1.py`  
- `LCR/LCR_BENCHMARK_v0.1.py`  
- `data/lcr_samples.jsonl`  

---

### 1.6 Ω-TOTAL (Fused Engine)

All lenses combine into a single structural score.

| Lens  | Meaning                     | Contribution              |
|------:|-----------------------------|---------------------------|
| BASE  | Numeric instability         | PBII                      |
| TIME  | Temporal drift              | log(1 + regime_score)     |
| CAUSA | Cross-channel structure     | mean edge strength        |
| TOKEN | Token instability           | mean |z|                  |
| LCR   | External coherence          | Ω_ext                     |

**Output**
- Unified Ω-total score  
- Full component breakdown  
- JSON-safe metadata for reproducibility  

---

## 2. Repository Structure

lon-mirror/ ├── omnia/ │   ├── init.py │   ├── omniabase.py │   ├── omniatempo.py │   ├── omniacausa.py │   ├── omniatoken.py │   ├── omnia_totale.py │   └── engine/ │       ├── engine.py │       └── kernel.py │ ├── adapters/ │   └── llm_output_adapter.py │ ├── LCR/ │   ├── LCR_CORE_v0.1.py │   └── LCR_BENCHMARK_v0.1.py │ ├── data/ │   ├── lcr_samples.jsonl │   └── gsm8k_model_outputs.jsonl   (optional) │ ├── examples/ │   └── omnia_gate_demo.py │ ├── colab/ │   ├── OMNIA_REAL_RUN.ipynb │   └── OMNIA_DEMO_INSPECT.ipynb │ ├── quick_omnia_test.py ├── gsm8k_benchmark_demo.py ├── INTERFACE.md ├── README.md └── requirements.txt

All modules are **standalone**, **deterministic**, and **import-safe**.

---

## 3. Installation

**Requirements**
- Python ≥ 3.9

Install dependencies:
```bash
pip install numpy matplotlib

Clone repository:

git clone https://github.com/Tuttotorna/lon-mirror
cd lon-mirror


---

4. Quick Smoke Test

Run:

python quick_omnia_test.py

Validates:

Omniabase (σ, entropy, PBII)

Omniatempo (regime score)

Omniacausa (lagged edges)

Token lens (PBII-z)

Ω-fusion correctness


If this runs without errors, OMNIA is correctly installed.


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

res = omniacausa_analyze({
    "s1": s1,
    "s2": s2,
    "s3": s3
})

5.4 Ω-TOTAL

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

7. Colab Notebooks (Official)

7.1 OMNIA_REAL_RUN.ipynb

Deterministic · Reproducible · Canonical

Path:

colab/OMNIA_REAL_RUN.ipynb

Purpose:

clean environment execution

reproducible runs

benchmark-safe entry point



---

7.2 OMNIA_DEMO_INSPECT.ipynb

Exploratory · Non-deterministic

Path:

colab/OMNIA_DEMO_INSPECT.ipynb

Purpose:

lightweight inspection

debugging and visualization

contrast with REAL_RUN


This notebook is not used for benchmarks or frozen runs.


---

8. Benchmarks

8.1 PBII + GSM8K Demo

Run:

python gsm8k_benchmark_demo.py

Outputs:

PBII distributions (primes vs composites)

Synthetic AUC ≈ 0.98

~71% hallucination reduction on long chains

Token instability segmentation


Results are synthetic placeholders pending full-scale evaluation.


---

8.2 LCR Benchmark

Run:

python LCR/LCR_BENCHMARK_v0.1.py

Outputs:

TP / FP / TN / FN

detection rate

precision

false positive rate

fused Ω_ext mean


Designed for external AI-lab validation.


---

9. Limitations

Benchmarks are synthetic

PBII sensitive to base selection

CAUSA currently uses Pearson correlation
(MI / TE / Granger planned)

Token lens requires numeric proxy

LCR depends on external backend quality



---

10. Author / Lineage

Massimiliano Brighindi — MB-X.01

Designer of:

Omniabase±

OMNIA_TOTALE

PBII

TOKEN instability lens

LCR fusion

Unified Ω engine


This repository is the authoritative public mirror of the MB-X research line.


---

11. License

MIT License.


---

12. Citation

Brighindi, M. (2025).
OMNIA Structural Lens Engine (v2.0)
GitHub: https://github.com/Tuttotorna/lon-mirror

