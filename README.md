# OMNIA — Unified Structural Measurement Engine

Ω · Ω̂ · SEI · IRI · OMNIA-LIMIT · τ  
MB-X.01

**Author:** Massimiliano Brighindi

---

## Overview

**OMNIA** is a **post-hoc structural measurement engine**.

It measures **structural coherence, instability, and limits** of representations
under independent transformations.

OMNIA:

- does **not** interpret meaning
- does **not** decide
- does **not** optimize
- does **not** learn

OMNIA measures **what remains invariant when representation changes**.

---

## Core Principle

> **Structural truth is what survives the removal of representation.**

OMNIA evaluates outputs by applying independent structural lenses and measuring:

- invariance
- drift
- saturation
- irreversibility

The result is a **measured boundary**, not a judgment.

---

## The OMNIA Measurement Chain

OMNIA  
→ Ω  
→ Ω under transformations  
→ Ω̂ (Omega-set)  
→ ΔΩ / ΔC  
→ SEI (Saturation)  
→ A → B → A′  
→ IRI (Irreversibility)  
→ SEI ≈ 0 and IRI > 0  
→ **OMNIA-LIMIT (STOP)**

Each step is **measured**, not inferred.

---

## 1. Ω — Structural Coherence Score

Ω is the aggregated structural score produced by OMNIA’s lenses.

It reflects **internal consistency**, not correctness.

Ω can be computed over:

- numbers
- sequences
- time series
- token streams
- model outputs

Ω is **model-agnostic** and **semantics-free**.

---

## 2. Structural Lenses

### BASE — Omniabase

Multi-base numeric structure analysis.

Measures:
- digit entropy across bases
- σ-symmetry
- PBII (Prime Base Instability Index)
- base-invariant signatures

---

### TIME — Omniatempo

Temporal drift and regime instability.

Measures:
- distribution shifts
- short vs long window divergence
- regime change score

---

### CAUSA — Omniacausa

Lagged relational structure.

Measures:
- cross-signal correlations across lags
- dominant dependency edges

---

### TOKEN

Structural instability in token sequences.

Pipeline:
- token → integer proxy
- PBII per token
- z-score aggregation

Used for hallucination and chain-fracture detection.

---

### LCR — Logical Coherence Reduction

External coherence lens.

Combines:
- factual consistency
- numeric consistency
- optional Ω contribution

Produces **Ω_ext** for audit and benchmarking.

---

### APERSPECTIVE — Aperspective Invariance

**Aperspective Invariance** measures structural invariants that persist under
independent transformations **without introducing any privileged point of view**.

This lens operates without:
- observer assumptions
- semantics
- causality
- narrative framing

It computes:
- **Ω-score**: fraction of structure surviving across transformations
- **Residue**: intersection of invariants remaining after representation removal

This isolates structure that is **real but non-vivable for human cognition**.

Implementation:
- `omnia/lenses/aperspective_invariance.py`

---

## 3. Ω̂ — Omega-set (Residual Invariance)

Ω̂ formalizes the statement:

> **Ω is not assumed. Ω is deduced by subtraction.**

Given multiple Ω values under independent transformations:

- Ω̂ = robust center (median)
- dispersion = MAD
- invariance = 1 / (1 + MAD)

Ω̂ estimates the **structural residue** that survives representation change.

Implementation:
- `omnia/omega_set.py`

---

## 4. SEI — Saturation / Exhaustion Index

SEI measures **marginal structural yield**.

Definition:

SEI = ΔΩ / ΔC

Where:
- Ω = structural score
- C = monotonic cost (tokens, steps, depth, time)

Interpretation:
- SEI > 0 → structure still extractable
- SEI ≈ 0 → saturation
- SEI < 0 → structural degradation

SEI is a **trend**, not a threshold.

Implementation:
- `omnia/sei.py`

---

## 5. IRI — Irreversibility / Hysteresis Index

IRI measures **loss of recoverable structure**.

Cycle:

A → B → A′

Definition:

IRI = max(0, Ω(A) − Ω(A′))

If apparent output quality is similar but Ω drops after return,
**irreversible structural damage** has occurred.

IRI is **not** an error metric.

Implementation:
- `omnia/iri.py`

---

## 6. OMNIA-LIMIT — Epistemic Boundary

OMNIA-LIMIT declares a **STOP condition**, not a decision.

Triggered when:
- SEI → 0 (no marginal structural gain)
- IRI > 0 (irreversibility detected)
- Ω̂ stable (no invariant left to extract)

Meaning:

> **No further structure is extractable under current transformations.**

OMNIA-LIMIT does **not** retry, escalate, or optimize.

---

## 7. Structural Time (τ)

OMNIA can optionally expose a **structural time coordinate (τ)**.

τ is **not a calendar** and **not a duration**.
It is derived **only** from OMNIA measurements.

τ advances **only when structural transformation occurs**.
If structure does not change, τ does not advance.

This enables:
- comparison of non-synchronized runs
- tracking of structural drift and irreversibility
- coordination across non-human systems

τ is a **coordination layer**, not part of the OMNIA core.

Formal definition:
- `docs/OMNIA_TAU.md`

---

## 8. Colab Notebooks

### Canonical (Reproducible)

`colab/OMNIA_REAL_RUN.ipynb`

- deterministic
- seed-locked
- audit reference

### Exploratory

`colab/OMNIA_DEMO_INSPECT.ipynb`

- non-deterministic
- inspection only
- no claims

Tablet / no local shell:
run via **Google Colab**.

---

## 9. Repository Structure

omnia/ omniabase.py omniatempo.py omniacausa.py omniatoken.py omnia_totale.py sei.py iri.py omega_set.py lenses/ aperspective_invariance.py

LCR/ LCR_CORE_v0.1.py LCR_BENCHMARK_v0.1.py

colab/ OMNIA_REAL_RUN.ipynb OMNIA_DEMO_INSPECT.ipynb

All modules are:
- deterministic
- standalone
- import-safe

---

## 10. What OMNIA Is Not

- Not a model
- Not an evaluator
- Not a policy
- Not a decision system
- Not a truth oracle

OMNIA is a **measurement instrument**.

---

## 11. License

MIT License.

---

## 12. Citation

Brighindi, M.  
**OMNIA — Unified Structural Measurement Engine (MB-X.01)**  
GitHub: https://github.com/Tuttotorna/lon-mirror


