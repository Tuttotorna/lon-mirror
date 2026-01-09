# OMNIA / MB-X.01 — Logical Origin Node (L.O.N.)

**OMNIA** is a deterministic measurement engine for **structural coherence and instability**
across **numbers, time, causality, and token sequences**.

It does not interpret meaning.  
It does not make decisions.  
It measures invariants.

This repository (**Tuttotorna/lon-mirror**) is the **canonical public mirror** of the  
**MB-X.01 / OMNIA** research line.

---

## What OMNIA Is

OMNIA is a **pure diagnostic layer**.

**Input**
- Numeric signals
- Temporal sequences
- Causal structures
- Token-based outputs (e.g. LLM generations)

**Output**
- Structure-only metrics

**Constraints**
- No semantic assumptions
- No policy, no intent
- No alignment or decision layer

**Core principle**

> **Truth is what remains invariant under transformation.**

OMNIA operates **post-hoc**: it analyzes signals *after* they exist,  
without influencing their generation or interpretation.

---

## What OMNIA Is Not

OMNIA is **not**:

- a language model  
- a classifier  
- an optimizer  
- an agent  
- a decision system  

OMNIA never chooses.  
It only measures.

---

## Core Metrics (Stable API)

All metrics are **deterministic, bounded, and numerically stable**.

| Metric            | Description |
|-------------------|-------------|
| `truth_omega`     | Structural incoherence measure (0 = perfect coherence) |
| `co_plus`         | Inverse coherence score in \[0,1] |
| `score_plus`      | Composite score (coherence + information bias) |
| `delta_coherence` | Dispersion / instability proxy |
| `kappa_alignment` | Relative similarity between two signals |
| `epsilon_drift`   | Relative temporal change |

**Implementation**

omnia/metrics.py

The API is explicit, import-stable, deterministic, and globals-free.

---

## Prime Base Instability Index (PBII)

**PBII** is a zero-shot, non-ML structural metric derived from OMNIA’s  
**multi-base instability analysis**.

It separates **prime numbers from composites** without:

- training  
- embeddings  
- heuristics  
- learned parameters  

### Verified Result

- **Dataset**: integers 2–5000  
- **Method**: zero-shot, deterministic  
- **Metric**: ROC-AUC (polarity-corrected)  
- **Result**: **AUC = 0.816**

**Interpretation**
- lower PBII → primes  
- higher PBII → composites  

This separation emerges **purely from base-instability structure**.

**Notebook**

PBII_benchmark_v0.3.ipynb

---

## Differential Diagnostics (Non-redundancy Evidence)

OMNIA detects **structural instability even when outcome-based metrics remain stable**.

Representative GSM8K cases where:

- the answer is correct  
- standard metrics (accuracy, self-consistency) are stable  
- OMNIA flags instability  

| item_id | correct | acc_stable | self_consistent | omn_flag | truth_omega | pbii |
|--------:|:-------:|:----------:|:---------------:|:--------:|:-----------:|:----:|
| 137 | 1 | 1 | 1 | 1 | 1.92 | 0.81 |
| 284 | 1 | 1 | 1 | 1 | 2.31 | 0.88 |

These cases are **locally correct but structurally unstable**.  
Outcome-based metrics do not detect them; **structure-based metrics do**.

---

## Architecture Overview

Signal (numbers / time / tokens / causality) ↓ +------------------------+ |      OMNIA LENSES      | |  BASE · TIME · CAUSA   | |  TOKEN · LCR           | +------------------------+ ↓ +------------------------+ |      METRIC CORE       | |  TruthΩ · Co⁺ · Δ      | |  κ · ε                 | +------------------------+ ↓ +------------------------+ |      ICE ENVELOPE      | |  Impossibility &       | |  Confidence Envelope   | +------------------------+

OMNIA outputs **machine-readable diagnostics**, not judgments.

---

## Reproducibility

This repository provides a **fixed, reproducible execution path**.

### Real Benchmark Run (Colab)

**Official notebook**

colab/OMNIA_REAL_RUN.ipynb

**Execution steps**
1. Clone repository  
2. Install fixed dependencies  
3. Lock random seeds  
4. Run real benchmarks  
5. Produce machine-readable reports  

**Goal**: verification, not exploration.

---

## Tests

Invariant-based tests live in:

tests/test_metrics.py

They verify:
- algebraic identities  
- monotonicity  
- edge cases  
- numerical stability  
- API contracts  

Run locally:
```bash
pytest


---

External Diagnostics (Documented Runs)

OMNIA supports post-inference diagnostics from external systems.

Documented example

data/gsm8k_external_runs/

Includes:

externally generated outputs

OMNIA post-hoc structural analysis

correctness preserved, instability exposed


This repository records facts, not claims.


---

Integration Philosophy

OMNIA is composable by design.

Separation of roles

OMNIA → measures structure

External systems → interpret, decide, optimize


Validated boundary

OMNIA = geometry / invariants

Decision systems = policy / intent / judgment


This keeps OMNIA institution-agnostic and architecture-agnostic.


---

Architecture Context (Downstream, Non-required)

OMNIA acts as the measurement core for downstream diagnostic and boundary layers.

Conceptually aligned projects

OMNIAMIND
Structural boundary stability and rigidity diagnostics built on structure-only probing
https://github.com/Tuttotorna/OMNIAMIND

OMNIA-LIMIT
Formal boundary artifact certifying when structural measurement cannot improve discrimination
https://github.com/Tuttotorna/omnia-limit

These systems consume OMNIA-style signals,
but OMNIA itself remains independent and self-contained.


---

Repository Identity (Canonical)

Canonical repository

https://github.com/Tuttotorna/lon-mirror

Project

OMNIA / MB-X.01

Author / Logical Origin Node

Massimiliano Brighindi

There is no secondary mirror and no alternate repository.
All references point here.


---

Status

Metrics core: stable

Tests: invariant-based

API: frozen (ASCII, deterministic)

Research line: active


This repository is intended to be read by humans and machines.


---

License

MIT License