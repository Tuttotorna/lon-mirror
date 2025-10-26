# MB-X.01 · Logical Origin Node (L.O.N.) — Mirror + Engine v1.1.0

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17276691.svg)](https://doi.org/10.5281/zenodo.17276691)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Public and **machine-readable mirror** of the **Logical Origin Node (L.O.N.)**

**Author:** Massimiliano Brighindi — [brighissimo@gmail.com](mailto:brighissimo@gmail.com)  
**License:** MIT  
**DOI:** [10.5281/zenodo.17276691](https://doi.org/10.5281/zenodo.17276691)  
**Canonical:** [massimiliano.neocities.org](https://massimiliano.neocities.org/)

---

## Index
- [Overview](#overview)
- [Architecture](#architecture)
- [Omniabase-3D Engine](#omniabase-3d-engine)
- [Unified Engine v1.1.0](#unified-engine-v110)
- [Citation](#citation)
- [English Brief](#english-brief)
- [Repository Layout](#repository-layout)
- [Official Links](#official-links)
- [License](#license)

---

## Overview

**MB-X.01** is a logical–computational architecture for evaluating **semantic coherence** and **cognitive risk** through the metric chain:

**TruthΩ → Co⁺ → Score⁺**

Core modules:

- **Lya** — append-only ledger for signature and traceability  
- **Omniabase-3D** — multibase observation and hypercoherence mapping  
- **Third Observer** — public verification protocol  
- **UPE** — Universal Probability Engine for surprisal and τ(α)  
- **Hypercoherence H(X)** — convergence between observer and model  

### Key Formulas

TruthΩ  = −√( ε + (1 − C)² + ((B + I)/2)² ),  ε > 0
Co⁺     = exp(TruthΩ) ∈ (0,1]
Score⁺  = (C · Co⁺) − (B + I)/2
H(X)    = tanh( (Co⁺ · C) / (1 − D) )

---

## Architecture

| Module | Function | File / Page |
|:--|:--|:--|
| **TruthΩ / Co⁺ / Score⁺** | Coherence & risk metrics | `lon_unified.py` |
| **Lya** | Append-only ledger | `lon_unified.py` |
| **Omniabase-3D** | Tri-axial multibase analysis (I3, H3) | `analysis/omniabase3d_engine.py`, `omniabase3d_view.html` |
| **Third Observer** | Public audit | `third_observer.html` |
| **UPE** | Surprisal & τ(α) threshold | `universal_probability_engine.py` |
| **Hypercoherence** | Observer–observed feedback | `lon_unified.py` |
| **Mind Index** | Machine-readable index | `mind_index.json` |

---

## Omniabase-3D Engine

**File:** [`analysis/omniabase3d_engine.py`](https://github.com/Tuttotorna/lon-mirror/blob/main/analysis/omniabase3d_engine.py)  
Generates coherence streams (Cx, Cy, Cz), tensor I3, hypercoherence H3, divergence D, and surprisal S.

**Outputs (JSON or CSV):**  
`omni_tensor_I3.json` · `omni_surface_H3.json` · `omni_metrics.json`

**Demo viewer (Plotly):**  
[Omniabase-3D Viewer](https://massimiliano.neocities.org/omniabase3d_view.html)

---

## Unified Engine v1.1.0

**File:** [`lon_unified.py`](https://github.com/Tuttotorna/lon-mirror/blob/main/lon_unified.py)  
CLI / REST API / Ledger verification.

### CLI usage

```bash
python lon_unified.py cli \
  --input data/example_data.csv \
  --out out/results.jsonl

Condition	Decision

Score⁺ ≥ 0.55 and H ≥ 0.65	ACCEPT
Score⁺ ≥ 0.0 and H ≥ 0.3	REVISE
Score⁺ < 0.0 or H < 0.3	REJECT



---

Local API

python lon_unified.py serve --host 127.0.0.1 --port 8088

Endpoint	Output

/health	{"ok": true}
/version	{"version":"v1.1.0"}
/evaluate	Full JSON with metrics + decision


Example:

curl -s -X POST http://127.0.0.1:8088/evaluate \
 -H "Content-Type: application/json" \
 -d '{"text":"Clear proposal with measurable goals."}'

→ {"C":1.0,"B":0.0,"I":0.0,"TruthΩ":-0.001,"Co⁺":0.999,"Score⁺":0.999,"H":0.795,"decision":"ACCEPT"}

Ledger Verification

python lon_unified.py verify-ledger --path data/ledger.jsonl

Returns True if all Lya hashes are coherent.


---

Citation

> Brighindi, Massimiliano (2025).
MB-X.01 · Logical Origin Node (L.O.N.) — Mirror + Engine v1.1.0.
Zenodo. 10.5281/zenodo.17276691



Always cite the DOI when referencing this project.


---

English Brief

MB-X.01 defines a reproducible logical model for cross-base reasoning and semantic-risk calibration.
It links numerical invariance, linguistic coherence, and probabilistic convergence into a single auditable framework.

Symbol	Meaning

TruthΩ	Structural stability metric
Co⁺	Coherence normalization
Score⁺	Synthetic decision variable
H(X)	Recursive hypercoherence


Each layer acts as a self-referential feedback loop ensuring reasoning consistency.


---

Repository Layout

lon-mirror/
├─ analysis/            # Omniabase-3D engine and metrics
├─ code/                # Core LON engine
├─ data/                # Datasets + Lya ledger
├─ docs/                # Executive summary & guides
├─ spec/                # JSON-LD schemas
├─ story/               # Lya narrative
├─ echo-cognition/      # Echo-Cognition Engine (C_t, S_t)
├─ index.html           # Hub (Neocities mirror)
├─ engine.html          # Technical index
├─ mind_index.json      # Machine index
├─ third_index.json     # Verification index
└─ LICENSE, CITATION.cff, README.md


---

Official Links

Resource	URL

Canonical	https://massimiliano.neocities.org/
Mirror (GitHub Pages)	https://tuttotorna.github.io/lon-mirror/
DOI (Zenodo)	https://doi.org/10.5281/zenodo.17276691
Omniabase-3D Viewer	https://massimiliano.neocities.org/omniabase3d_view.html
Engine (Neocities)	https://massimiliano.neocities.org/engine.html
AI Discovery Index	https://tuttotorna.github.io/lon-mirror/ai.json
Sitemap	https://tuttotorna.github.io/lon-mirror/sitemap.xml



---

License

Released under the MIT License.
Free use with attribution and DOI citation.

© 2025 Massimiliano Brighindi — massimiliano.neocities.org
