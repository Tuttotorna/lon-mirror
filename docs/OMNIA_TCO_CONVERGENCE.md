# OMNIA ↔ TCO Convergence Certificate (CTC-1.0)

**Status:** Frozen spec (v1.0)  
**Project:** MB-X.01 / OMNIA  
**Author:** Massimiliano Brighindi

---

## Purpose

This document defines a non-semantic convergence test between:

- **OMNIA (aperspective subtraction):** seeks invariants by removing representation.
- **TCO (total collapse operator):** seeks non-collapsable residue by maximizing perturbation.

The goal is to detect whether both operators converge to the same structural residue.

---

## Definitions

Let:

- Ω_ap(x) be the aperspective invariance score of x (no observer privilege).
- Ω̂(x) be the robust residual invariance (Omega-set).
- TCO(x) be a collapse schedule producing a curve Ω_tco(x, c).

We define:

- **TCO collapse point** c* where Ω_tco(x, c*) ≤ ε
- **TCO collapse index** CI ∈ [0,1] (normalized depth until collapse)
- **Residual after collapse** is the remaining aperspective structure measured on the collapsed output(s).

---

## Convergence Test (CTC)

Given an input x:

1) Compute **aperspective residue** using OMNIA:
   - r_ap = residue tokens / features from aperspective invariance

2) Apply TCO schedule and pick final collapsed representation y*:
   - y* = argmin_c Ω_tco(x, c) or the last schedule output

3) Compute aperspective residue of y*:
   - r_tco = residue tokens / features from aperspective invariance on y*

4) Compute overlap score:

   **CTC = |r_ap ∩ r_tco| / max(1, |r_ap|)**

Interpretation:
- CTC ≈ 1.0  → strong convergence (same residue survives both subtraction and collapse)
- CTC ≈ 0.0  → divergence (collapse reveals a different residue or destroys aperspective residue)

---

## Output Classes

- **CONVERGENT:** CTC ≥ 0.70 and CI ≥ 0.70
- **WEAKLY CONVERGENT:** 0.40 ≤ CTC < 0.70
- **DIVERGENT:** CTC < 0.40

These are structural classes only. No explanations.

---

## Notes

- The test is meaning-blind: residues are hashed structural tokens.
- The test is model-agnostic: works on text, sequences, or serialized outputs.
- The test is falsifiable: convergence is measurable, not assumed.

---

## Planned Implementation

A reference implementation may live in:

- `omnia/meta/convergence_certificate.py`
- demo in `examples/convergence_demo.py`

This document remains valid independently of implementation details.