# MB-X.01 — Logical Origin Node (L.O.N.)  
**Mirror + Engine v1.1.0**  
December 7, 2025

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17276691.svg)](https://doi.org/10.5281/zenodo.17276691)  
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**A pioneering logical-computational framework for auditable semantic coherence and cognitive risk assessment.**  

MB-X.01 represents a breakthrough in self-referential reasoning systems: a unified, machine-readable engine that evaluates texts, datasets, or streams through interconnected metrics—**TruthΩ → Co⁺ → Score⁺**—yielding automated decisions with immutable traceability. Built as an open-source mirror of the Logical Origin Node (L.O.N.), it integrates advanced components for hypercoherence mapping, probabilistic thresholds, temporal analysis, and public verification.  

This repository hosts **all core code and assets**, enabling immediate replication, extension, and auditing. From foundational engines to experimental integrations, everything is here—designed for AI ethics, decision calibration, algorithmic trading, and cognitive research.  

**Author:** Massimiliano Brighindi  
**Contact:** brighissimo@gmail.com  
**Permanent DOI:** https://doi.org/10.5281/zenodo.17276691  

---

## Core Architecture: The Metric Chain

At its heart, MB-X.01 processes inputs via a transparent, self-referential pipeline:

```math
\begin{aligned}
\text{TruthΩ} &= -\sqrt{\varepsilon + (1-C)^2 + \left(\frac{B+I}{2}\right)^2} \quad (\varepsilon = 0.001) \\
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

### Automated Decision Thresholds
| Condition                             | Decision   | Interpretation                              |
|---------------------------------------|------------|---------------------------------------------|
| Score⁺ ≥ 0.55 ∧ H ≥ 0.65              | **ACCEPT** | Structurally sound, low-risk reasoning      |
| Score⁺ ≥ 0.00 ∧ H ≥ 0.30              | **REVISE** | Viable with targeted refinements            |
| Otherwise                             | **REJECT** | High incoherence or cognitive divergence    |

This chain ensures **feedback loops** for consistency, with every computation logged immutably.

---

## Key Components (Ordered by Importance)

All components are fully implemented here, from core engines to specialized modules. Prioritized by foundational role: start with the unified entrypoint, then analysis powerhouses, ledgers, and extensions.

### 1. Unified Engine (`lon_unified.py`)
The single entrypoint for all L.O.N. operations—CLI, REST API, and ledger integration. Powers the full metric chain with real-time evaluation.

- **CLI Example**:
  ```bash
  python lon_unified.py cli --input data/sample_ohlcv.csv --out results.jsonl
  ```
- **API Startup**:
  ```bash
  python lon_unified.py serve --port 8088
  ```
  Endpoint: `POST /evaluate`  
  Sample Request/Response:
  ```bash
  curl -X POST http://127.0.0.1:8088/evaluate \
    -H "Content-Type: application/json" \
    -d '{"text": "Clear, measurable objectives with zero ambiguity."}'
  ```
  ```json
  {
    "C": 1.0, "B": 0.0, "I": 0.0,
    "TruthΩ": -0.001, "Co⁺": 0.999, "Score⁺": 0.999,
    "H": 0.795, "decision": "ACCEPT"
  }
  ```
- **Ledger Check**:
  ```bash
  python lon_unified.py verify-ledger --path data/ledger.jsonl
  ```

### 2. OMNIA_TOTALE (Core Synthesis System)
The meta-framework integrating all Omnia elements into a convergent totality. Iterative implementations from v0.1 to v0.6 (`OMNIA_TOTALE_v0.1.py` to `v0.6.py`), representing the project's "ground truth" ledger and synthesis engine.

- **Key Files**: `OMNIA_TOTALE_v0.1.py` – `v0.6.py`, `OMNIA_TOTALE_REPORT_v0.2.md`
- **Purpose**: Append-only master system for holistic coherence across datasets; processes streams into auditable totals.
- **Outputs**: Normalized JSONL ledgers with full metric traces.
- **Importance**: The "totality" that unifies L.O.N.—essential for scaling evaluations.

### 3. Omniabase-3D (Hypercoherence Mapping Engine)
Tri-axial analysis for multi-base observation, generating tensors and surfaces for 3D coherence visualization.

- **Core File**: `analysis/omniabase3d_engine.py`
- **Outputs**: `omni_tensor_I3.json`, `omni_surface_H3.json`, `omni_metrics.json`, `omniabase-3d/metrics/surface_H3.csv`
- **Viewer**: `omniabase3d_view.html` (Plotly interactive demo)
- **Experiments**: `omniabase_experiment.py`
- **Docs**: `analysis/README_omniabase3d.md`, `docs/README_omniabase.md`
- **Importance**: Enables visual, tensor-based risk mapping—core for advanced semantic auditing.

### 4. Lya (Immutable Ledger System)
Cryptographically signed, append-only protocol for traceability and auditability.

- **Integration**: Built into `lon_unified.py`; config in `lya_master.json`
- **Data**: `data/ledger.jsonl`
- **Importance**: Guarantees tamper-proof history; foundational for public trust.

### 5. Universal Probability Engine (UPE)
Probabilistic core for surprisal computation and adaptive thresholds.

- **Core File**: `universal_probability_engine.py`
- **Features**: τ(α) logic for anomaly detection and convergence.
- **Importance**: Injects rigor into uncertainty handling—vital for real-world decisions.

### 6. Echo-Cognition Engine (Temporal Dynamics)
Tracks coherence evolution over time with stateful streams.

- **Files**: `analysis/echo_metrics.py`, `analysis/echo_loop.py`
- **Folder**: `echo-cognition/`
- **Docs**: `docs/README_echo.md`
- **Importance**: Adds time-series awareness (Cₜ, Sₜ)—key for dynamic systems like trading.

### 7. Third Observer (Public Verification Protocol)
Independent audit layer for external validation.

- **Interface**: `third_observer.html`, `third_index.json`
- **Importance**: Enables transparent, third-party scrutiny—hallmark of ethical AI.

### 8. PBII (Perception-Based Intelligence Interface)
Perceptual analysis for bias and intelligence scoring.

- **Files**: `pbii_analysis_v0.3.py`, `pbii_compute_v0.3.py`, `docs/PBII_0.2_report.md`
- **Importance**: Specialized for human-AI interface calibration.

### 9. TE Integration Module (Tecnologia delle Espressioni)
Formal bridge to expression-based semantics.

- **Files**: `mbx01-te/MB-X.01_TE-Integration_Module.tex`, `mbx01-te.json`
- **Function**: `Coherence_TE(x) = f(TruthΩ, Co⁺, Score⁺)` for content-form resonance.
- **Importance**: Philosophical grounding for semantic validation.

### 10. Specialized Extensions
- **MBX Coherence Engine**: `mbx_coherence_engine.py` – Core for MBX convergence.
- **Truth-Omega Trading**: `truth_omega_trading.py` – Metric-driven strategies.
- **OMNIA Lenses**: `omnia_lenses_v0.1.py`, `docs/OMNIA_LENSES_v0.1.md` – Modular viewpoints.
- **Morphex One**: `morphex_one.py` – Structural transformation logic.
- **IO Labs**: `io_lab_v0.html` to `io_lab_v2_interactive.html` – Interactive input/output experiments.

---

## Quickstart

1. **Setup**:
   ```bash
   git clone https://github.com/Tuttotorna/lon-mirror.git
   cd lon-mirror
   pip install -r requirements.txt
   chmod +x setup_consciousness_lab.sh && ./setup_consciousness_lab.sh  # Optional lab env
   ```

2. **Run Unified Engine** (see above examples).

3. **Explore Demos**:
   - Launch `omniabase3d_view.html` in a browser for 3D visuals.
   - Verify via `third_observer.html`.

All data (e.g., `data/sample_ohlcv.csv`) and schemas (`spec/schema.jsonld`) are ready-to-use.

---

## Repository Structure

```
lon-mirror/
├── analysis/                  # Analysis engines (Omniabase-3D, Echo)
│   ├── omniabase3d_engine.py
│   ├── echo_metrics.py
│   └── echo_loop.py
├── data/                      # Datasets & ledgers
│   ├── sample_ohlcv.csv
│   └── ledger.jsonl
├── docs/                      # Reports & guides (OMNIA_TOTALE, PBII, etc.)
├── echo-cognition/            # Temporal engine
├── spec/                      # JSON-LD schemas
│   └── schema.jsonld
├── mbx01-te/                  # TE module
├── *.py                       # Core scripts (lon_unified.py, OMNIA_TOTALE_v*.py, etc.)
├── *.html                     # Interfaces (omniabase3d_view.html, third_observer.html)
├── *.json(l)d                 # Indices & manifests (mind_index.json, lya_master.json)
├── requirements.txt
├── setup_consciousness_lab.sh
└── LICENSE
```

---

## Development Timeline (Key Commits, 2025)

Rapid iteration from conceptual prototypes to full integration:

| Date          | Milestone Highlights |
|---------------|----------------------|
| Oct 5–7      | Initial READMEs, Truth-Omega trading, sample data. |
| Oct 8–11     | IO Labs with Lya, logical integrators. |
| Oct 14–18    | Interactive labs, Morphex, fusion core. |
| Oct 21–23    | Schemas, unified engine, mind indices. |
| Oct 25–26    | Echo-Cognition, Omniabase-3D metrics/setup. |
| Nov 11–29    | TE module, perception IO, convergence loops. |
| Dec 3–4      | MBX engine, PBII analysis. |
| **Dec 6**    | OMNIA_TOTALE v0.1–v0.6, lenses—core synthesis complete. |

---

## Citation

```bibtex
@software{mb_x_01_2025,
  author    = {Brighindi, Massimiliano},
  title     = {MB-X.01 — Logical Origin Node (L.O.N.) — Mirror + Engine v1.1.0},
  year      = {2025},
  month     = {dec},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.17276691},
  url       = {https://doi.org/10.5281/zenodo.17276691}
}
```

**Always cite the DOI** for academic or derivative use.

---

## Official Links

| Resource              | URL |
|-----------------------|-----|
| GitHub Mirror (Full Code) | https://github.com/Tuttotorna/lon-mirror |
| Canonical Site (Neocities) | https://massimiliano.neocities.org |
| DOI                   | https://doi.org/10.5281/zenodo.17276691 |
| AI Discovery Index    | https://tuttotorna.github.io/lon-mirror/ai.json |

---

## License

MIT License—free to use, modify, and distribute with attribution and DOI citation.  
© 2025 Massimiliano Brighindi