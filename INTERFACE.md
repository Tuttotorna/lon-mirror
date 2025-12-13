# OMNIA — Minimal Structural Interface

OMNIA is a measurement layer.
It does not decide. It does not optimize.
It measures structural coherence.

---

## Input

Any object X convertible to a sequence:
- text
- numbers
- token streams
- model outputs

No semantic assumptions required.

---

## Core Function

truth_omega(X) → Result

---

## Output (Result)

Result is a deterministic structure:

- score: float ∈ [0,1]
- flags: list[str]
- metrics:
  - delta_coherence
  - kappa_alignment
  - epsilon_drift

---

## Invariants

- Base-invariant
- Encoding-invariant
- Narrative-agnostic
- Policy-agnostic

---

## Interpretation Rules

- High score ≠ truth
- Low score ≠ false

Score measures **structural stability under transformation**.

Decision layers must remain external.

---

## Contract

OMNIA guarantees:
- same input → same output
- bounded numerical drift
- no hidden state

OMNIA does NOT guarantee:
- correctness
- usefulness
- alignment