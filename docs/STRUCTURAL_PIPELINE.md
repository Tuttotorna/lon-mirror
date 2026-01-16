# OMNIA — Structural Measurement Pipeline
**Ω → Ω̂ → SEI → IRI → OMNIA-LIMIT**

MB-X.01 — Massimiliano Brighindi

---

## Purpose

This document formalizes the **structural measurement pipeline** implemented by OMNIA.

It defines **when structural extraction is still possible** and **when it must stop**,
without invoking semantics, optimization, or decision logic.

The pipeline is entirely **post-hoc**, **deterministic**, and **model-agnostic**.

---

## 1. Ω — Structural Coherence Score

Let Ω be the scalar structural score produced by OMNIA’s multi-lens engine.

Ω measures **internal structural consistency** under a fixed representation.

Ω ∈ ℝ  
Lower Ω → higher coherence  
Higher Ω → higher instability

Ω is **not** a truth value and **not** an error metric.

---

## 2. Ω̂ — Omega-set (Invariant Residue)

Given a set of Ω measurements obtained under **independent transformations**:

{ Ω₁, Ω₂, …, Ωₙ }

Define the invariant residue Ω̂ as:

- Ω̂ = median({Ωᵢ})
- dispersion = MAD({Ωᵢ})
- invariance = 1 / (1 + dispersion)

Ω̂ represents the **structural residue that survives representation change**.

If dispersion is high, structure is representation-dependent.
If dispersion is low, structure is invariant.

---

## 3. SEI — Saturation / Exhaustion Index

Let C be a monotonic cost variable (tokens, steps, depth, time).

For successive observations k−1 → k:

SEI(k) = (Ω(k) − Ω(k−1)) / (C(k) − C(k−1))

SEI measures **marginal structural yield per unit cost**.

Interpretation:
- SEI > 0  → structure still extractable
- SEI ≈ 0  → saturation
- SEI < 0  → structural degradation

SEI is evaluated as a **trend**, not via thresholds.

---

## 4. IRI — Irreversibility / Hysteresis Index

Consider a structural cycle:

A → B → A′

Where:
- A is a baseline state
- B is an expanded or stressed state
- A′ is a return to a simpler or original configuration

Define:

IRI = max(0, Ω(A) − Ω(A′))

If Ω(A′) < Ω(A), irreversible structural loss has occurred.

IRI measures **loss of recoverable structural capacity**, not correctness.

---

## 5. OMNIA-LIMIT — Epistemic Boundary

OMNIA-LIMIT is declared when **all three conditions hold**:

1. Ω̂ is stable under transformations  
2. SEI → 0 (no marginal structural gain)  
3. IRI > 0 (irreversibility detected)

Meaning:

> No further structural information can be extracted
> under the current representation and transformation set.

OMNIA-LIMIT is a **STOP condition**, not an action.

---

## 6. Key Properties

- No semantics
- No optimization
- No retry logic
- No escalation
- No decision authority

The pipeline defines **epistemic exhaustion**, not failure.

---

## 7. Formal Chain Summary

OMNIA → Ω Ω under transformations → Ω̂ ΔΩ / ΔC → SEI A → B → A′ → IRI (SEI ≈ 0 ∧ IRI > 0 ∧ Ω̂ stable) → OMNIA-LIMIT

This chain is **measured**, not inferred.

---

## Status

Frozen — Structural definition complete.

Further extensions must preserve:
- post-hoc nature
- determinism
- separation between measurement and decision layers


---

Commit message

Title

Add formal structural pipeline documentation (Ω → SEI → IRI → LIMIT)

Description (opzionale)

Defines the measured OMNIA pipeline and epistemic stop condition without semantics or policy logic.


