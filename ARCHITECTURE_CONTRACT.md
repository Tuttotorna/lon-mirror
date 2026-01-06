# OMNIA — Architecture Contract

This document defines the **non-negotiable architectural guarantees** of OMNIA.

OMNIA is a **diagnostic measurement layer**.
It does not generate, decide, optimize, or interpret.

---

## Guarantees

1. **Determinism**  
   Given the same inputs, transformations, and configuration, OMNIA produces the same output.

2. **Post-hoc operation**  
   OMNIA operates only on existing artifacts (numbers, sequences, model outputs).
   It never influences generation or decision processes.

3. **Structural focus**  
   All metrics measure structural invariance and instability under declared transformations.
   No semantic assumptions are introduced.

4. **Explicit transforms**  
   All transformations considered by OMNIA must be explicitly declared.
   There are no hidden normalizations or implicit preprocessing steps.

5. **Machine-readable output**  
   Results are emitted as structured, schema-valid JSON suitable for automated integration.

---

## Non-Guarantees (Explicit)

OMNIA does **not**:

- decide correctness or truth
- select actions or policies
- optimize models or objectives
- enforce constraints
- perform alignment
- provide explanations

Any system using OMNIA for these purposes must implement its own logic externally.

---

## Integration Boundary

OMNIA must remain **externally invoked**.

- It must not be embedded inside model training loops as a reward signal.
- It must not be used as a control mechanism at runtime.
- It must not be treated as a semantic validator.

OMNIA provides **measurement only**.

---

## Final Statement

If a system requires decisions, enforcement, or abstention,
those mechanisms must live **outside OMNIA**.

OMNIA measures whether structure is stable.
What to do with that information is not its concern.