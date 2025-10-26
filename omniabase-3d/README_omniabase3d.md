# Omniabase 3D · Multibase Spatial Mathematics Engine  
**Module:** MB-X.01 / Logical Origin Node (L.O.N.) — Extended Mirror  
**Author:** Massimiliano Brighindi — [massimiliano.neocities.org](https://massimiliano.neocities.org/)  
**License:** MIT  
**DOI:** [10.5281/zenodo.17270742](https://doi.org/10.5281/zenodo.17270742)  

---

## Overview

**Omniabase 3D** expands the original Omniabase (multi-base numerical system) into a **spatial mathematical engine** operating on three orthogonal coherence axes:

Cx → base vector along X-axis
Cy → base vector along Y-axis
Cz → base vector along Z-axis

Each axis can operate in a distinct base (for example, base-8, base-12, base-16), allowing concurrent evaluation of logical or mathematical structures across multidimensional bases.

---

## Core Equations

At time *t*, for a signal or concept *Φ*:

Φₜ = f(Cxₜ, Cyₜ, Czₜ)

Coherence Tensor:      I₃ = ⟨Cx, Cy, Cz⟩ / ‖⟨Cx, Cy, Cz⟩‖ Hypercoherence Field:  H₃ = tanh( (Cₓ·Cᵧ·C_z) / (1 + |∇C|) ) Divergence Threshold:  Δ₃ = |∇·I₃| < τ(α)

**Definitions**

- **I₃** — multidimensional coherence tensor  
- **H₃** — hypercoherence surface (global stability field)  
- **Δ₃** — divergence metric under coherence threshold  

The field **H₃** expresses the geometric stability of coherence across the three bases, forming the core of *3D logical congruence*.

---

## Conceptual Model

| Layer | Function | Description |
|-------|-----------|-------------|
| **Input Layer** | Multi-base sampling | Converts scalar input into Cx, Cy, Cz base components |
| **Tensor Layer** | Coherence computation | Forms I₃ = normalized tri-base tensor |
| **Surface Layer** | Hypercoherence mapping | Projects I₃ into H₃ (stability visualization) |
| **Threshold Layer** | Decision metric | Evaluates Δ₃ < τ(α) for coherence acceptance |

---

## Implementation Notes

Prototype file: `omniabase3d_engine.py`  

Example usage:

```bash
python omniabase3d_engine.py --bases 8 12 16 --steps 1000 --alpha 0.005

Output files:

tensor_I3.csv — coherence tensor history

surface_H3.csv — hypercoherence field values

metrics.json — summary (mean coherence, divergence, surprisal)



---

Theoretical Context

Omniabase 3D represents a shift from linear mathematics (single-base progression) to volumetric mathematics,
where base transformations coexist instead of alternating.

This enables:

Spatial reasoning in logic and computation

Multi-perspective evaluation of coherence

Emergent 3D logical surfaces measurable across bases


Extensions

Variant	Description

Omniabase⁴⁺	Adds a temporal or contextual dimension to the tensor
Omniabaseᴛ	Integrates time-evolving base dynamics
Omniabase±	Models dual positive/negative coherence flows



---

Practical Applications

Cognitive architecture simulations

Prime number and pattern field prediction

AI reasoning verification (multi-base consistency)

High-dimensional logic visualization



---

Citation

If you use or extend this module, cite:

> Brighindi, Massimiliano (2025).
MB-X.01 · Logical Origin Node (L.O.N.) — Omniabase 3D Module.
Zenodo. DOI: 10.5281/zenodo.17270742
