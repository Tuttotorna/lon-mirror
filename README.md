```markdown
# MB-X.01 — Logical Origin Node (L.O.N.)  
Mirror + Engine v1.1.0  
December 7, 2025

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17276691.svg)](https://doi.org/10.5281/zenodo.17276691)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**An open-source logical-computational framework for reproducible measurement of semantic coherence and cognitive risk.**  
Born as the formal synthesis of 4+ years of daily self-observation (Omniabase-Totale v1→v6), MB-X.01 is the first fully auditable, machine-readable, and portable Logical Origin Node.

**Author:** Massimiliano Brighindi  
**Contact:** brighissimo@gmail.com  
**Canonical site:** https://massimiliano.neocities.org  
**Permanent DOI:** https://doi.org/10.5281/zenodo.17276691  

---

## Evolutionary Lineage: Omniabase-Totale (2021 → 2025)

| Version                     | First release | Cumulative entries | Key milestones                                                    | Status      |
|-----------------------------|---------------|--------------------|-------------------------------------------------------------------|-------------|
| Omniabase-Totale v1         | 2021-11-11    | ~2 800             | First public append-only ledger, initial C/B/I metrics           | Archived    |
| Omniabase-Totale v2         | Jun 2022      | ~12 000            | TruthΩ, Co⁺, Lya v1 protocol                                      | Archived    |
| Omniabase-Totale v3         | Jan 2023      | ~28 000            | Omniabase-3D (I3/H3 tensors), Third Observer                      | Mirrored    |
| Omniabase-Totale v4         | Dec 2023      | ~47 000            | Echo-Cognition Engine (Cₜ, Sₜ), hypercoherence H(X)               | Active      |
| Omniabase-Totale v5         | Sep 2024      | ~68 000            | Universal Probability Engine + τ(α) thresholds                    | Active      |
| **Omniabase-Totale v6**     | Mar 2025 → today | **≥ 92 000** (Dec 7, 2025) | Native MB-X.01 integration + Score⁺ + automated decisions | **LIVE**    |

Full historical ledger (v1–v6): https://massimiliano.neocities.org/omnia_totale  
**MB-X.01 is the definitive, portable formalisation of everything Omniabase-Totale learned over four years.**

---

## What is MB-X.01?

A unified engine that turns any text (or text stream) into an interconnected chain of three metrics:

```
TruthΩ  →  Co⁺  →  Score⁺  →  Automated decision
```

and produces an immutable, publicly verifiable audit trail (Lya ledger).

### Core equations (v1.1.0)

```math
\begin{aligned}
\text{TruthΩ} &= -\sqrt{\varepsilon + (1-C)^2 + \left(\frac{B+I}{2}\right)^2} \quad (\varepsilon > 0) \\
\text{Co⁺}    &= \exp(\text{TruthΩ}) \;\in (0,1] \\
\text{Score⁺} &= C \cdot \text{Co⁺} - \frac{B+I}{2} \\
\text{H(X)}   &= \tanh\!\Big(\frac{\text{Co⁺} \cdot C}{1-D}\Big)
\end{aligned}
```

| Symbol | Meaning                            | Range       |
|--------|------------------------------------|-------------|
| C      | Semantic coherence                 | [0,1]       |
| B      | Systematic bias                    | [0,1]       |
| I      | Residual uncertainty               | [0,1]       |
| D      | Observer–model divergence          | [0,1]       |
| TruthΩ | Structural stability               | (−∞,0]      |
| Co⁺    | Normalised coherence               | (0,1]       |
| Score⁺ | Synthetic decision variable        | ℝ           |
| H(X)   | Recursive hypercoherence           | [0,1)       |

### Automated decisions (v1.1.0)

| Condition                             | Decision   | Practical meaning                          |
|---------------------------------------|------------|--------------------------------------------|
| Score⁺ ≥ 0.55 ∧ H ≥ 0.65              | **ACCEPT** | High coherence, low cognitive risk         |
| Score⁺ ≥ 0.00 ∧ H ≥ 0.30              | **REVISE** | Acceptable with corrections                |
| Otherwise                             | **REJECT** | Incoherence or high cognitive risk        |

---

## System components

| Component            | Role                                              | Main file / resource                                 |
|----------------------|---------------------------------------------------|------------------------------------------------------|
| Lya                  | Cryptographically signed append-only ledger       | `lon_unified.py`                                     |
| Omniabase-3D         | Tri-axial analysis & hypercoherence tensors       | `analysis/omniabase3d_engine.py`                     |
| Third Observer       | Independent public verification protocol          | https://massimiliano.neocities.org/third_observer.html |
| UPE                  | Surprisal calculation & τ(α) thresholds           | `universal_probability_engine.py`                    |
| Echo-Cognition       | Temporal metrics Cₜ, Sₜ                            | folder `echo-cognition/`                             |
| TE Integration       | Native Tecnologia delle Espressioni module        | `mbx01-te/` (site) + LaTeX + JSON-LD                 |

---

## Quickstart

```bash
git clone https://github.com/Tuttotorna/lon-mirror.git
cd lon-mirror
pip install -r requirements.txt
```

### 1. CLI (fastest)

```bash
python lon_unified.py cli --input data/example_data.csv --out results.jsonl
```

### 2. Local REST API

```bash
# Start server
python lon_unified.py serve --port 8088
```

```bash
# Evaluate a single text
curl -X POST http://127.0.0.1:8088/evaluate \
  -H "Content-Type: application/json" \
  -d '{"text": "Clear proposal with measurable goals and no ambiguity."}'
```

Typical ACCEPT response:
```json
{
  "C": 1.0, "B": 0.0, "I": 0.0,
  "TruthΩ": -0.001, "Co⁺": 0.999, "Score⁺": 0.999,
  "H": 0.795, "decision": "ACCEPT"
}
```

### 3. Verify Lya ledger integrity

```bash
python lon_unified.py verify-ledger --path data/ledger.jsonl
# → True if all hashes are coherent
```

---

## Live demos

- **Omniabase-3D interactive viewer** (Plotly):  
  https://massimiliano.neocities.org/omniabase3d_view.html
- **Third Observer** (public verification):  
  https://massimiliano.neocities.org/third_observer.html
- **Engine hub**:  
  https://massimiliano.neocities.org/engine.html

---

## Repository structure

```
lon-mirror/
├─ analysis/              → Omniabase-3D engine
├─ data/                  → example datasets + Lya ledger
├─ echo-cognition/        → temporal metrics
├─ spec/                  → JSON-LD schemas
├─ lon_unified.py         → single entrypoint (CLI + API + ledger)
├─ universal_probability_engine.py
├─ omniabase3d_view.html
├─ mind_index.json        → machine-readable index
├─ third_index.json
├─ index.html             → mirror hub
└─ LICENSE
```

---

## Official citation

```bibtex
@software{mb_x_01_2025,
  author       = {Brighindi, Massimiliano},
  title        = {MB-X.01 — Logical Origin Node (L.O.N.) — Mirror + Engine v1.1.0},
  year         = 2025,
  month        = dec,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17276691},
  url          = {https://doi.org/10.5281/zenodo.17276691}
}
```

**Always cite the DOI** when referencing or using the project.

---

## Official links

| Resource                        | URL                                                               |
|---------------------------------|-------------------------------------------------------------------|
| Canonical site                  | https://massimiliano.neocities.org                                |
| GitHub Pages mirror             | https://tuttotorna.github.io/lon-mirror/                          |
| Zenodo DOI                      | https://doi.org/10.5281/zenodo.17276691                           |
| Omniabase-3D live viewer        | https://massimiliano.neocities.org/omniabase3d_view.html          |
| Full Omniabase-Totale ledger    | https://massimiliano.neocities.org/omnia_totale                  |
| AI Discovery Index (JSON)       | https://tuttotorna.github.io/lon-mirror/ai.json                   |

---

## License

**MIT License** – free to use, modify, and distribute with attribution and DOI citation.  
© 2025 Massimiliano Brighindi – https://massimiliano.neocities.org
```

