# OMNIA — Stress Taxonomy (v1.0)

This document defines the **official stress taxonomy** for OMNIA / MB-X.01.

Stress here does **not** mean "harder benchmarks".
Stress means: **controlled conditions that reveal structural limits**.

OMNIA’s rule is:

- if a **real failure** emerges → it becomes a **boundary artifact**
- boundaries are **kept**, **versioned**, and **machine-readable**
- OMNIA does not “fight” failures — it **certifies** them

---

## Definitions

- **Signal**: the measurable structural residue produced by OMNIA (Ω, Ω̂, SEI, IRI, SI, etc.)
- **Stress**: a controlled perturbation / condition applied to expose instability
- **Failure**: a reproducible condition where the instrument’s admissible extraction ends
- **Boundary**: a formal STOP certificate derived from a failure
- **Artifact**: a stored, replayable record of the boundary (human + machine readable)

---

## Core Stress Classes

### STRESS-DRIFT
**Goal:** expose when structure begins to move directionally and stops being stationary.

**Typical indicators:**
- rising drift measures (e.g., T in PrimeState, or analogous drift proxies)
- unstable Ω under small perturbations
- increasing disagreement across lenses

**Expected outcome:**
- either: drift remains admissible and is measured
- or: drift triggers STOP via OMNIA-LIMIT / guardrails

---

### STRESS-SATURATION
**Goal:** detect exhaustion: additional processing yields no new admissible structure.

**Typical indicators:**
- SEI → 0 (ΔΩ / ΔC collapses)
- stable Ω̂ while costs rise
- repeated transforms yield no new residue

**Expected outcome:**
- OMNIA-LIMIT STOP certificate (saturation boundary)

---

### STRESS-IRREVERSIBILITY
**Goal:** detect hysteresis / irreversible structural loss in cycles.

**Mechanism:**
A → B → A′

**Typical indicators:**
- IRI > 0
- A′ fails to recover Ω(A) despite apparent similarity

**Expected outcome:**
- irreversible boundary artifact (IRI-certified)

---

### STRESS-INDISTINGUISHABILITY
**Goal:** test when distinct internal codifications remain structurally undecidable.

**Typical indicators:**
- SI ≈ high (or equivalent indistinguishability score)
- transformations preserve observable relations across candidates
- no admissible relation separates hypotheses

**Expected outcome:**
- “undecidable” boundary artifact (structural equivalence class)

---

### STRESS-OBSERVER-PERTURBATION
**Goal:** quantify damage introduced by enforced perspective / privileged basis.

**Typical indicators:**
- OPI = Ω_ap − Ω_obs > 0
- observer-aligned measurement collapses structure not collapsed in aperspective mode

**Expected outcome:**
- observer-perturbation boundary (measurement warning)
- or a guardrail upgrade if observer choice was invalid

---

### STRESS-PROJECTION-LOSS
**Goal:** quantify how structure is destroyed by projection (not by observation).

**Typical indicators:**
- SPL > 0 consistently under a defined projection
- projected measurers degrade residue vs aperspective baseline

**Expected outcome:**
- projection boundary artifact
- identifies which projection destroys admissible structure

---

## Stress Execution Principles

1. **Deterministic only**
   - no random seeds required
   - no stochastic sampling needed to reproduce a boundary

2. **Stress ≠ policy**
   - stress reveals limits, it does not decide what to do next

3. **Failure is a product**
   - a failure is not an embarrassment
   - it is a boundary to be frozen

4. **Boundaries are first-class**
   - every accepted failure becomes an artifact
   - artifacts are versioned and referenced from the README/docs

---

## Minimal Boundary Artifact Format (human-readable)

A boundary record must include:

- **id**: unique boundary id
- **module**: which OMNIA component triggered STOP
- **trigger**: which stress class (from this taxonomy)
- **inputs**: minimal reproducible input
- **metrics**: Ω, Ω̂, SEI, IRI, SI, confidence / guards
- **decision**: STOP (with reason)
- **reproduce**: exact command to replay

Machine-readable schema is defined separately under `schema/`.

---

## Status

- **Frozen:** yes (taxonomy v1.0)
- Next: add machine-readable boundary schema and a stress runner.

Author: Massimiliano Brighindi — MB-X.01