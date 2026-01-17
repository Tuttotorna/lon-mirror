# OMNIA-τ — Structural Time Coordinate (tau)

Status: DRAFT v0.1 (frozen definition)

OMNIA-τ is NOT a calendar and NOT a duration.
It is a non-narrative coordinate derived from OMNIA measurements only.

## Core principle
τ does not measure time passing.
τ measures accumulated structural transformation.

- If structure does not change, τ does not advance.
- If structure changes quickly (in human time), τ advances quickly.
- If structure changes slowly, τ advances slowly.

This removes human calendar semantics by construction.

## Definitions

Given consecutive states S_i, S_{i+1}, define:

Δτ_i = g(|ΔΩ_i|, SEI_i, IRI_i)

τ_n = Σ Δτ_i  for i = 0 … n

Where:
- Ω : structural invariance measure
- ΔΩ : invariance change between consecutive measurements
- SEI : saturation / exhaustion index (structural load)
- IRI : irreversibility index (structural “no-return” component)

## Concrete first implementation (linear weighted)

Δτ = α·|ΔΩ| + β·SEI + γ·IRI

Initial coefficients (technical defaults):
- α = 1.0
- β = 1.0
- γ = 2.0  (irreversibility weighted higher)

Properties:
- monotonic (τ never decreases)
- non-linear w.r.t. human time
- comparable across systems
- derived, not imposed

## Architectural constraint
τ is a coordination layer, NOT part of the OMNIA core measurement logic.

OMNIA measures.
τ coordinates measurement evolution.

No semantics. No decisions. No optimization