# Measurement Projection Loss (SPL)

**SPL (Structural Projection Loss)** measures how much structure is lost when a measurement
regime forces a **privileged projection** (basis / observer constraint), compared to an
**aperspective** baseline.

This is not interpretation. It is a measured delta between two regimes.

---

## Definitions

Let:

- **Ω_ap** = aggregated aperspective measurement (no privileged basis)
- **Ω_proj** = aggregated projected measurement (privileged basis enforced)

Then:

- **SPL_abs = max(0, Ω_ap − Ω_proj)**
- **SPL_rel = SPL_abs / max(eps, Ω_ap)**

Interpretation:

- SPL ≈ 0  → projection is structurally neutral  
- SPL > 0  → projection destroys measurable structure  

SPL does not measure consciousness or intent.  
It measures the structural cost of forced viewpoint.

---

## Why this exists inside OMNIA

Many “collapse-like” effects can be reframed as:

> the measurement regime collapses the accessible structure into a single projection,
> not the world collapsing into a single state.

SPL provides a computable way to quantify that loss.

---

## Implementation

- Core operator: `omnia/meta/measurement_projection_loss.py`
- Minimal demo: `examples/measurement_projection_loss_demo.py`