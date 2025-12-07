```markdown
# MB-X.01 — Logical Origin Node (L.O.N.)  
Mirror + Engine v1.1.0  
December 7, 2025

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17276691.svg)](https://doi.org/10.5281/zenodo.17276691)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**A brand-new, fully auditable logical-computational framework for measuring semantic coherence and cognitive risk in a reproducible way.**

Born on December 6, 2025, and released the following day, MB-X.01 is the first working implementation of the Logical Origin Node concept: a single, portable engine that combines structural metrics, probabilistic convergence, and public verifiability from day one.

**Author:** Massimiliano Brighindi  
**Contact:** brighissimo@gmail.com  
**Canonical site:** https://massimiliano.neocities.org  
**Permanent DOI:** https://doi.org/10.5281/zenodo.17276691  

---

## Core equations (v1.1.0)

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

### Automated decisions

| Condition                             | Decision   | Meaning                                    |
|---------------------------------------|------------|--------------------------------------------|
| Score⁺ ≥ 0.55 ∧ H ≥ 0.65              | **ACCEPT** | High coherence, negligible cognitive risk |
| Score⁺ ≥ 0.00 ∧ H ≥ 0.30              | **REVISE** | Acceptable with adjustments                |
| Otherwise                             | **REJECT** | Incoherence or high cognitive risk         |

---

## Key components (all functional on day 1)

| Component            | Role                                              | Main file / resource                                 |
|----------------------|---------------------------------------------------|------------------------------------------------------|
| Lya                  | Cryptographically signed append-only ledger       | `lon_unified.py`                                     |
| Omniabase-3D         | Tri-axial coherence mapping & visualisation       | `analysis/omniabase3d_engine.py` + viewer            |
| Third Observer       | Public verification protocol                      | https://massimiliano.neocities.org/third_observer.html |
| UPE                  | Surprisal & τ(α) thresholds                       | `universal_probability_engine.py`                    |
| Echo-Cognition       | Temporal coherence streams (Cₜ, Sₜ)               | folder `echo-cognition/`                             |
| TE Integration       | Native Tecnologia delle Espressioni module        | LaTeX + JSON-LD bundle                               |

---

## Quickstart

```bash
git clone https://github.com/Tuttotorna/lon-mirror.git
cd lon-mirror
pip install -r requirements.txt
```

### CLI
```bash
python lon_unified.py cli --input data/example_data.csv --out results.jsonl
```

### Local API
```bash
python lon_unified.py serve --port 8088
```

```bash
curl -X POST http://127.0.0.1:8088/evaluate \
  -H "Content-Type: application/json" \
  -d '{"text": "Clear, measurable, unambiguous proposal."}'
```

Example ACCEPT response:
```json
{
  "C": 1.0, "B": 0.0, "I": 0.0,
  "TruthΩ": -0.001, "Co⁺": 0.999, "Score⁺": 0.999,
  "H": 0.795, "decision": "ACCEPT"
}
```

### Verify ledger
```bash
python lon_unified.py verify-ledger --path data/ledger.jsonl
```

---

## Live demos (available since Dec 6, 2025)

- Omniabase-3D interactive viewer: https://massimiliano.neocities.org/omniabase3d_view.html  
- Third Observer public audit: https://massimiliano.neocities.org/third_observer.html  
- Engine hub: https://massimiliano.neocities.org/engine.html

---

## Repository structure

```
lon-mirror/
├─ analysis/              → Omniabase-3D engine
├─ data/                  → examples + Lya ledger
├─ echo-cognition/        → temporal metrics
├─ spec/                  → JSON-LD schemas
├─ lon_unified.py         → single entrypoint
├─ universal_probability_engine.py
├─ omniabase3d_view.html
├─ mind_index.json
├─ index.html
└─ LICENSE
```

---

## Citation

```bibtex
@software{mb_x_01_2025,
  author    = {Brighindi, Massimiliano},
  title     = {MB-X.01 — Logical Origin Node (L.O.N.) — Mirror + Engine v1.1.0},
  year      = 2025,
  month     = dec,
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.17276691},
  url       = {https://doi.org/10.5281/zenodo.17276691}
}
```

Please cite the DOI whenever you reference or use the project.

---

## Official links

| Resource                     | URL                                                               |
|------------------------------|-------------------------------------------------------------------|
| Canonical site               | https://massimiliano.neocities.org                                |
| GitHub Pages mirror          | https://tuttotorna.github.io/lon-mirror/                          |
| DOI                          | https://doi.org/10.5281/zenodo.17276691                           |
| Omniabase-3D viewer          | https://massimiliano.neocities.org/omniabase3d_view.html          |
| AI Discovery Index           | https://tuttotorna.github.io/lon-mirror/ai.json                   |

---

## License

MIT License — free to use, modify, and redistribute with attribution and DOI citation.  
© 2025 Massimiliano Brighindi
```

