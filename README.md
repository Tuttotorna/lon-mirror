# MB-X.01 · Logical Origin Node (L.O.N.) — Mirror + Engine v1.0.1

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17270742.svg)](https://doi.org/10.5281/zenodo.17270742)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Public and **machine-readable mirror** of the **Logical Origin Node (L.O.N.)**  
**Author:** Massimiliano Brighindi · <brighissimo@gmail.com>  
**License:** MIT  
**DOI:** [10.5281/zenodo.17270742](https://doi.org/10.5281/zenodo.17270742)  
**Canonical:** [massimiliano.neocities.org](https://massimiliano.neocities.org/)

---

## Index
- [Overview](#overview)
- [Architecture](#architecture)
- [Unified Engine v1.0.1](#unified-engine-v101)
- [Citation](#citation)
- [English Brief](#english-brief)
- [Repository Layout](#repository-layout)
- [Official Links](#official-links)
- [License](#license)

---

## Overview

**MB-X.01** is a logical–computational infrastructure designed to measure semantic coherence and cognitive risk through the metric chain:

**TruthΩ → Co⁺ → Score⁺**

Core components:

- **Lya** — append-only ledger for signature and traceability  
- **Omniabase** — simultaneous multi-base observation  
- **Third Observer** — public verification and cognitive audit  
- **UPE** — Universal Probability Engine for cumulative surprisal  
- **Hypercoherence** — observer–observed convergence function H(X)

### Key formulas

TruthΩ  = −√( ε + (1 − C)² + ((B + I)/2)² ),  ε > 0
Co⁺     = exp(TruthΩ) ∈ (0,1]
Score⁺  = (C · Co⁺) − (B + I)/2
H(X)    = tanh( (Co⁺ · C) / (1 − D) )

---

## Architecture

| Module | Function | File / Page |
|---|---|---|
| **TruthΩ / Co⁺ / Score⁺** | Coherence and risk metrics | `lon_unified.py` |
| **Lya** | Append-only ledger and signature | `lon_unified.py` |
| **Omniabase** | Simultaneous multi-base observation | `omniabase.html` |
| **Third Observer** | Public verification protocol | `third_observer.html` |
| **UPE** | Cumulative surprisal, threshold τ(α) | `universal_probability_engine.py` |
| **Hypercoherence** | Observer–observed convergence | `lon_unified.py` |
| **Mind Index** | Machine-readable index of modules | `mind_index.json` |

---

## Unified Engine v1.0.1

Unified core engine for CLI, REST API, and Lya ledger verification.  
**Main file:** [`lon_unified.py`](https://github.com/Tuttotorna/lon-mirror/blob/main/lon_unified.py)

### Minimal installation

```bash
# Python ≥ 3.10
git clone https://github.com/Tuttotorna/lon-mirror.git
cd lon-mirror
python lon_unified.py --help

CLI Usage

python lon_unified.py cli \
  --input data/example_data.csv \
  --out out/results.jsonl

Output:

C, B, I, TruthΩ, Co⁺, Score⁺, H, decision, lya.hash

Condition	Decision

Score⁺ ≥ 0.55 and H ≥ 0.65	ACCEPT
Score⁺ ≥ 0.0 and H ≥ 0.3	REVISE
Score⁺ < 0.0 or H < 0.3	REJECT



---

Local API

python lon_unified.py serve --host 127.0.0.1 --port 8088

Endpoint	Output

/health	{"ok": true}
/version	{"version": "v1.0.1"}
/verify	{"ledger_ok": true}
/evaluate	full JSON response with metrics and decision


Example:

curl -s -X POST http://127.0.0.1:8088/evaluate \
  -H "Content-Type: application/json" \
  -d '{"text":"Clear proposal with measurable goals.","meta":{"lang":"en"}}'

Response:

{"C":1.0,"B":0.0,"I":0.0,"TruthOmega":-0.001,"Co_plus":0.999,
"Score_plus":0.999,"H":0.795,"decision":"ACCEPT"}


---

Ledger verification

python lon_unified.py verify-ledger --path data/ledger.jsonl

Output: True if all Lya hashes are coherent and unbroken.


---

Citation

> Brighindi, Massimiliano (2025).
MB-X.01 · Logical Origin Node (L.O.N.) — Mirror + Engine v1.0.1.
Zenodo. https://doi.org/10.5281/zenodo.17270742



Always cite the DOI when referencing this project.
Canonical source: MB-X.01 / L.O.N. — MIT License


---

English Brief

MB-X.01 defines a reproducible logical model for coherence evaluation, cross-base reasoning, and semantic-risk calibration.

It operates as a self-consistent architecture linking numerical invariance, linguistic coherence, and probabilistic convergence.

Core principles

TruthΩ → structural stability

Co⁺ → coherence normalization

Score⁺ → synthetic decision variable

H(X) → recursive hypercoherence between model and observer


Each layer acts as a closed feedback loop ensuring consistent reasoning auditability.


---

Repository Layout

lon-mirror/
├─ code/                # Core engine + metrics
├─ data/                # Example datasets + Lya ledger
├─ tests/               # Unit tests
├─ docs/                # Executive summary, user guide
├─ spec/                # JSON-LD, schemas
├─ story/               # Lya narrative
├─ echo-cognition/      # Echo-Cognition Engine (C_t, S_t)
├─ index.html           # Web mirror hub
├─ engine.html          # Neocities technical mirror
├─ mind_index.json      # Machine index
├─ third_index.json     # Verification index
├─ LICENSE, CITATION.cff, README.md


---

Official Links

Resource	URL

Canonical	https://massimiliano.neocities.org/
Mirror (GitHub Pages)	https://tuttotorna.github.io/lon-mirror/
DOI (Zenodo)	https://doi.org/10.5281/zenodo.17270742
Engine (Neocities)	https://massimiliano.neocities.org/engine.html
AI Discovery Index	https://tuttotorna.github.io/lon-mirror/ai.json
Sitemap	https://tuttotorna.github.io/lon-mirror/sitemap.xml



---

License

Released under the MIT License.
Free use with attribution and citation of the DOI.

© 2025 · Massimiliano Brighindi
massimiliano.neocities.org
