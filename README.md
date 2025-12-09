# OMNIA_TOTALE · Structural Coherence Lenses for LLMs

Author: **Massimiliano Brighindi (MB-X.01)**  
Core engine + formalization: **MBX IA**

OMNIA_TOTALE is a **model-agnostic coherence layer** for Large Language Models (LLMs).  
It does not replace a model. It observes its outputs and computes **structural stability signals**:

- `Omniabase` → multi-base numeric stability (PBII).
- `Omniatempo` → temporal regime change in sequences.
- `Omniacausa` → lagged correlation graph between signals.
- `OmniaTotale` → fused scalar Ω score combining all three.

The goal: provide **transparent, deterministic metrics** to flag instability, drift and hallucinations in long reasoning chains.

---

## 1. Repository layout

Current code is organized as:

```text
lon-mirror/
├─ omnia/
│  ├─ __init__.py
│  └─ core/
│     ├─ __init__.py
│     ├─ omniabase.py      # multi-base numeric lens (PBII, signatures)
│     ├─ omniatempo.py     # temporal stability lens
│     └─ omniacausa.py     # lagged causal-structure lens
├─ OMNIA_TOTALE_v2.0.py    # fused Ω score using omnia.core lenses
├─ gsm8k_benchmark_demo.py # (optional) synthetic GSM8K-style benchmark
├─ OMNIA_LENSES_v0.1.md    # conceptual overview of the three lenses
└─ other legacy files...   # earlier experiments and prototypes

Only the files listed above are meant as the current entry points for reviewers.


---

2. Installation

Requirements (minimal):

pip install numpy

Optional (for benchmarks and plots):

pip install matplotlib

No external ML frameworks are required to run the structural lenses.


---

3. Quick start

3.1 Run the fused demo (Ω score)

From the repository root:

python OMNIA_TOTALE_v2.0.py

This will:

Build synthetic time series with a regime shift.

Compute OmniaTotaleResult for:

a prime number n = 173

a composite number n = 180


Print:

fused omega_score

per-lens components:

base_instability (PBII-style)

tempo_log_regime

causa_mean_strength


causal edges discovered by omniacausa.



3.2 Use the lenses programmatically

Example (inside another Python script or notebook):

import numpy as np

from omnia.core import (
    omniabase_signature,
    pbii_index,
    omniatempo_analyze,
    omniacausa_analyze,
)
from OMNIA_TOTALE_v2_0 import omnia_totale_score  # if you rename the file to OMNIA_TOTALE_v2_0.py

# numeric lens
sig = omniabase_signature(173)
print(sig.sigma_mean, sig.entropy_mean)

# temporal lens
t = np.arange(300)
series = np.sin(t / 15.0) + 0.1 * np.random.normal(size=t.size)
ot = omniatempo_analyze(series)
print(ot.regime_change_score)

# causal lens
s1 = np.sin(t / 10.0)
s2 = np.zeros_like(s1)
s2[2:] = 0.7 * s1[:-2] + 0.1 * np.random.normal(size=t.size - 2)
s3 = np.random.normal(size=t.size)
oc = omniacausa_analyze({"s1": s1, "s2": s2, "s3": s3})

# fused Ω score
res = omnia_totale_score(
    n=173,
    series=series,
    series_dict={"s1": s1, "s2": s2, "s3": s3},
)
print(res.omega_score, res.components)

Note: Python cannot import from a file containing a dot in its name.
If you want to import the fused function, rename:

OMNIA_TOTALE_v2.0.py  →  OMNIA_TOTALE_v2_0.py

and adjust the import accordingly.


---

4. Components (technical summary)

4.1 Omniabase (numeric lens)

Defined in omnia/core/omniabase.py.

digits_in_base_np(n, b)
Integer → digit array in base b (NumPy, MSB first).

normalized_entropy_base(n, b)
Shannon entropy of digits, normalized to [0, 1].

sigma_b(n, b, ...)
Base symmetry score combining:

low entropy

length penalty

divisibility bonus (n % b == 0).


OmniabaseSignature
Dataclass with:

per-base sigmas and entropy

sigma_mean, entropy_mean.


omniabase_signature(n, bases=...)
Multi-base signature for integer n.

pbii_index(n, ...)
Prime Base Instability Index (PBII), measuring relative instability of n w.r.t. a local composite window.


4.2 Omniatempo (temporal lens)

Defined in omnia/core/omniatempo.py.

Global mean / std of the series.

Local mean / std over short and long windows.

Regime-change score based on symmetric KL-like divergence between histograms of recent short vs long segments.


4.3 Omniacausa (causal lens)

Defined in omnia/core/omniacausa.py.

Lagged Pearson-like correlations between all pairs of series.

For each pair, selects the lag with maximum |corr|.

Emits edges source → target when |corr| ≥ threshold, with associated lag and strength.


This is not a full causal discovery algorithm; it is a structural lens to highlight stable lead–lag patterns.

4.4 OmniaTotale (fused Ω)

Defined in OMNIA_TOTALE_v2.0.py.

OmniaTotaleResult:

integer n

OmniabaseSignature

OmniatempoResult

OmniacausaResult

scalar omega_score

components dict.


omnia_totale_score(...):

base_instability = pbii_index(n, ...)

tempo_log_regime = log(1 + regime_change_score)

causa_mean_strength = mean absolute strength of accepted causal edges.

omega_score = weighted sum of the three components.




---

5. Benchmarks (status)

Benchmarks are currently synthetic demos intended for reviewers, not formal claims.

gsm8k_benchmark_demo.py
Simulates GSM8K-style chains with:

“correct” vs “hallucinated” numeric reasoning.

PBII-based detection of unstable chains.

Simple AUC computation and PBII histograms (requires matplotlib).



Interpretation of any “71% hallucination reduction” or “AUC ≈ 0.98” should be:

Provisional.

Based on synthetic / internal experiments.

Meant as starting point for independent validation by external teams (e.g. xAI, research labs).



---

6. Repository status

This repository is:

Work-in-progress, but:

omnia/ lenses are self-contained and importable.

OMNIA_TOTALE_v2.0.py runs as a demo.

Benchmark demo is present and executable.


Older scripts and HTML files represent earlier experiments and can be ignored by reviewers focusing on OMNIA_TOTALE v2.0.


Future work (planned):

Clean directory structure: core/, benchmarks/, api/.

More systematic GSM8K-style evaluations.

Integration stubs for external LLMs (tool-calling, CoT auditing).



---

7. Author and citation

Core concepts and design:

Massimiliano Brighindi (MB-X.01) — independent watchmaker and AI conceptual architect.


If you reference this work in research or internal reports, you can cite:

GitHub: https://github.com/Tuttotorna/lon-mirror

(Optional) Zenodo / DOI entries linked from the author’s website.


This repository is published to allow external verification, critique and extension of the structural lenses approach (Omniabase, Omniatempo, Omniacausa, OmniaTotale).

