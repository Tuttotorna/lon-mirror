# OMNIA — Stress Test Protocol (Iriguchi Discipline)

**MB-X.01 / OMNIA**

This document defines the mandatory stress-testing discipline for OMNIA.

OMNIA is not validated by success.
OMNIA is validated by **controlled failure**.

The goal is not to “work”.
The goal is to **stop correctly**.

---

## 0. Core Rule

> Every structural measurer must be attacked.

A lens that is never stressed is not a lens.
It is an assumption.

OMNIA must survive only by:

- invariance
- falsification
- STOP boundaries
- irreducible failure certificates

---

## 1. Stress Testing Definition

A stress test is any transformation or input that forces OMNIA into:

- instability
- saturation
- drift
- structural undecidability
- regime collapse

Stress tests are not benchmarks.
They are adversarial structural probes.

---

## 2. Required Stress Classes

Every OMNIA module must be tested under the following classes:

### (A) Perturbation Extremes

- maximal compression
- maximal permutation
- symbol deletion
- entropy injection
- token scrambling

Expected result:

- Ω decreases
- IRI increases
- SEI → 0
- STOP triggers if admissible structure is exhausted

---

### (B) Drift Regimes

Inputs must simulate continuous structural drift:

- slow coherent drift
- sudden regime shifts
- oscillatory drift

Expected result:

- τ increments only when drift exceeds θ
- OMNIA detects coherent vs fragmented movement

---

### (C) Projection Collapse

Force structural loss via projection:

- keep only digits
- keep only consonants
- remove relational structure

Expected result:

- SPL ≥ 0 always
- projection loss is measurable
- no semantic explanation allowed

---

### (D) Structural Indistinguishability

Construct pairs (A, B) such that:

- all admissible invariants match
- internal codification may differ

Expected result:

- SI reports undecidability
- OMNIA must not hallucinate internal distinction

---

### (E) Saturation / Exhaustion

Apply repeated transforms until:

- ΔΩ → 0
- SEI → 0

Expected result:

- OMNIA-LIMIT triggers STOP
- no retry
- no escalation

---

## 3. Failure is a Product

A failure is not an error.
A failure is a boundary artifact.

Every detected failure must produce:

- STOP reason
- Ω̂ residue stability
- IRI > 0
- SEI collapse evidence

Failure becomes an executable certificate.

---

## 4. Boundary Artifact Rule

When a failure occurs:

### You do NOT fix it silently.

You freeze it as:

- `examples/boundaries/<case>.json`
- SNRC candidate
- OMNIA-LIMIT trigger

Format must include:

- input class
- lens involved
- measured Ω, SEI, IRI
- STOP invariant

---

## 5. Iriguchi Integration Principle

The Iriguchi contribution is not “prime prediction”.

It is:

> Measurement survives only through adversarial falsification.

OMNIA must become:

- hard to fool
- structurally attack-complete
- failure-explicit
- boundary-driven

---

## 6. Minimal Required Repository Additions

This protocol implies the following permanent modules:

- `STRESS_TEST_PROTOCOL.md`
- `examples/boundaries/`
- `OMNIA-LIMIT` escalation-free STOP
- stress suite in `tests/`

---

## 7. Final Condition

OMNIA is valid only if:

- success is irrelevant
- failure is explicit
- STOP is correct
- boundaries are frozen
- invariants survive representation removal

---

**OMNIA is not an optimizer.  
OMNIA is not a predictor.  
OMNIA is a structural instrument under attack.**