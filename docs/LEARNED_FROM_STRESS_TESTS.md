# LEARNED_FROM_STRESS_TESTS.md
## OMNIA — Structural Lessons Extracted from External Stress PRs (e.g. #4)

**Project:** MB-X.01 / OMNIA  
**Author:** Massimiliano Brighindi  
**Scope:** This document records what was *learned* from external stress-test pull requests  
without integrating governance, policy, or semantic decision layers.

OMNIA is a measurement instrument.  
External contributions are evaluated only by structural compatibility.

---

## 1. Vectorization Is an OMNIA-Safe Upgrade

A core lesson from stress PRs is that:

> Performance improvements are valid only when they do not alter measurement semantics.

Pure optimization is allowed when it is:

- deterministic  
- representation-invariant  
- lens-preserving  
- numerically stable  

### Rule

**Loop → Vectorization is acceptable.**  
**Interpretation → Adaptation is not.**

Example domains:

- TIME lens (sliding windows)
- CAUSA lens (lag structure)
- BASE lens (multi-mod arithmetic)

### Constraint

Vectorization must never introduce:

- learned parameters  
- adaptive weights  
- post-hoc correction  

Only faster execution of the same measurement.

---

## 2. OMNIA Tests Must Be Invariance Tests (Not Accuracy Tests)

OMNIA does not optimize for correctness.

It measures:

- invariance
- drift
- saturation
- boundary

Therefore the correct regression suite is:

### Invariance Regression

Input `x` must satisfy:

- Ω(x) ≈ Ω(T(x)) under admissible transforms  
- Ω remains bounded in [0,1]
- STOP triggers consistently when regimes degrade

### Limit Regression

If perturbation exceeds admissibility:

- OMNIA-LIMIT must trigger  
- No “repair” is allowed  
- Output must be STOP + reason

---

## 3. Benchmarks Must Measure Runtime + Drift + STOP-Rate

Stress PRs often focus only on speed.

OMNIA requires a structural benchmark contract:

### For every lens:

Measure:

- runtime (ms)
- ΔΩ drift vs reference
- STOP-rate under perturbation

A fast lens is meaningless if it changes Ω.

### Required benchmark tuple

(runtime, mean_drift, stop_fraction)

---

## 4. Boundary Reinforcement: STOP Is Allowed, Correction Is Not

Some external PR patterns introduce governance-like modules:

- adaptive weighting
- sovereign kernels
- attenuation controllers
- corrective blending

These violate OMNIA’s architectural boundary.

### OMNIA Rule

> OMNIA may declare STOP.  
> OMNIA may never correct.

OMNIA produces:

- Ω
- Ω̂
- SEI
- IRI
- STOP reason
- ICE / SNRC candidate

Anything that modifies outputs beyond measurement is an external decision layer.

---

## 5. Extracted Value from PR #4 (Without Integration)

From the stress-test pull request, OMNIA extracted only:

1. Vectorization as safe performance upgrade  
2. Regression focus on invariance, not prediction  
3. Benchmarking as drift-aware measurement  
4. Stronger boundary: STOP ≠ governance

No code was imported.

Only structural lessons were retained.

---

## Final Constraint

OMNIA remains:

- deterministic  
- non-semantic  
- model-agnostic  
- measurement-only  

External contributions are admissible only if they preserve:

> **Measurement invariance under representation removal.**

---

**MB-X.01 / OMNIA-LIMIT**  
Signal → Measure → Boundary → STOP

