# OMNIA — Superposition Operator and Ω Residue

## Purpose

This document defines the **Superposition Operator** used in OMNIA and clarifies how the **Ω (Omega) structural residue** is obtained.

It replaces all legacy “Supdocs” material with a **single, canonical specification**.

---

## Superposition Principle

OMNIA evaluates structure by observing a representation under **controlled structural transformations**.

Let:
- R be a representation (text, sequence, numeric encoding, model output)
- T = {t₁, t₂, …, tₙ} be a set of admissible structural transformations

The **superposition** of R is the set of its transformed instances:

S(R) = { t(R) | t ∈ T }

Transformations are chosen to preserve admissibility while exposing instability.

---

## Admissible Transformations

A transformation is admissible if it:

- preserves the domain of representation
- does not inject semantic intent
- is deterministic
- is reversible or bounded

Examples (non-exhaustive):
- base changes (numeric)
- windowing / slicing (temporal)
- lag shifts (causal)
- token re-encoding (token lens)
- permutation under constraints

Transformations differ per lens but obey the same admissibility rule.

---

## Invariance and Residue

For each transformed instance t(R), a lens L produces a structural signal:

L(t(R)) → sᵢ

The **invariant component** is the part of the signal that remains stable across superposition.

The **Ω residue** is defined as:

Ω(R) = aggregate( stability(L(t(R))) ) for all t ∈ T and all lenses L

Ω is therefore obtained **by subtraction**, not construction:
- what changes is discarded
- what persists is retained

Ω is not meaning.
Ω is not correctness.
Ω is **structural persistence**.

---

## Multi-Lens Superposition

Each lens defines its own transformation family T_L.

Superposition is evaluated:
- within each lens
- then across lenses via Ω-TOTAL fusion

This produces:
- per-lens stability profiles
- cross-lens invariant alignment
- a unified Ω structural profile

---

## Structural Drift

When invariants degrade under superposition, OMNIA emits **drift signals**.

Drift indicates:
- loss of internal consistency
- sensitivity to perturbation
- approaching structural saturation

Drift is measurable without semantic interpretation.

---

## Saturation and Stop Condition

When additional transformations do not change the Ω profile beyond a margin, the system is considered **structurally saturated**.

At saturation:
- further analysis adds no information
- OMNIA must stop

This condition is formalized by boundary mechanisms (see boundary document).

---

## What the Operator Does Not Do

The Superposition Operator does not:
- optimize representations
- search for better answers
- rank alternatives by preference
- decide acceptance or rejection

It **exposes structure** and nothing more.

---

## Summary

- Superposition reveals invariants by controlled variation
- Ω is the residue that survives transformation
- Structure is measured by persistence, not interpretation
- All results are deterministic and reproducible

This operator is the **core unifying mechanism** of OMNIA.