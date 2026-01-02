# OMNIABASE-PURE
**Base-free Ω metric + null validation (z > 2)**

OMNIABASE-PURE defines a base-independent structural metric Ω.
No reference unit, no fixed base, no semantic assumptions.

---

## Definition

For each integer n ≥ 2, OMNIABASE-PURE returns a tuple:

(Ω, E0, Ω_min)

Where:
- Ω      : structural energy (base-free)
- E0     : equilibrium index
- Ω_min  : local minimum envelope (can be zero)

Ω is invariant under base choice and scale conventions.

---

## Properties

- Deterministic
- Base-free
- Scale-agnostic
- Tuple-valued (not scalar by design)
- Numeric stability verified (finite Ω)

---

## Sanity Check

Finite Ω validation:

Ω finite = True  
(abs(Ω) < 1e6, no NaN)

---

## Null Validation

A block-shuffle null model is used to estimate structural bias.

Statistic:
Δ = (P − C)_real − (P − C)_null

Validation rule:
z = (Δ_real − μ_null) / σ_null

Threshold:
z > 2 ⇒ structural signal detected

---

## Status

Validated in Colab.
Committed to MB-X.01 / OMNIA.