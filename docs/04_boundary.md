# OMNIA — Boundary and Stop Conditions

## Purpose

This document defines the **non-negotiable boundaries** of OMNIA and the conditions under which **structural analysis must stop**.

Boundaries are not safeguards or policies.  
They are **epistemic limits** derived from structural saturation.

---

## Measurement Boundary

OMNIA is a **measurement layer only**.

It must not:
- interpret meaning
- infer intent
- rank answers by preference
- trigger actions
- enforce safety decisions

Any system that violates this separation is **no longer OMNIA**.

---

## Structural Saturation

Structural analysis proceeds by applying admissible transformations and observing invariants.

A system reaches **structural saturation** when:

- additional admissible transformations
- do not materially change
- the Ω profile beyond a defined margin

At saturation:
- information gain ≈ 0
- further processing is redundant
- analysis must stop

---

## Saturation Signals

Saturation may be indicated by:

- convergence of Ω across rounds
- bounded variance below threshold
- invariant alignment across lenses
- diminishing drift deltas

These signals are **measured**, not interpreted.

---

## Stop Condition (Formal)

Let Ωₖ be the Ω profile after iteration k.

A stop condition is met when:

|Ωₖ₊₁ − Ωₖ| ≤ ε   for all monitored components

Where:
- ε is a configured tolerance
- comparison is component-wise
- aggregation is deterministic

When the condition holds, OMNIA must halt analysis.

---

## Failure to Respect Boundaries

If OMNIA outputs are used to:
- decide acceptance/rejection
- replace reasoning
- enforce policy
- claim truth or correctness

then the violation occurs **outside** OMNIA.

Responsibility lies with the downstream system.

---

## Relationship to OMNIA-LIMIT / ICE / LCR

Boundary enforcement is supported by:

- OMNIA-LIMIT: formal declaration of non-reducibility
- ICE: impossibility and confidence envelopes
- LCR: external coherence signals (non-authoritative)

These mechanisms **signal limits**; they do not override them.

---

## Why Boundaries Matter

Without explicit boundaries:
- measurement collapses into judgment
- signals are misused as decisions
- reproducibility is lost
- trust degrades

Boundaries preserve:
- scientific validity
- architectural clarity
- long-term extensibility

---

## Summary

- OMNIA stops when structure saturates
- Stop conditions are measurable and explicit
- Boundaries are epistemic, not ethical
- Decisions always remain external

This boundary definition is **final** for the OMNIA measurement layer.