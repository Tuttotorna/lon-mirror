# OMNIA / MB-X.01 — Logical Origin Node (L.O.N.)

**OMNIA** is a deterministic measurement engine for **structural coherence and instability**
across numbers, time, causality, and token sequences.

It does **not** interpret meaning.  
It does **not** make decisions.  
It **measures invariants**.

This repository (`Tuttotorna/lon-mirror`) is the **canonical public mirror**
of the **MB-X.01 / OMNIA** research line.

---

## What OMNIA is

OMNIA is a **pure diagnostic layer**.

- **Input:** signals (numeric, temporal, causal, token-based)
- **Output:** structure-only metrics
- **No semantic assumptions**
- **No policy, no intent, no alignment layer**

Core principle:

> **Truth is what remains invariant under transformation.**

OMNIA operates *post-hoc*: it analyzes signals **after they exist**,
without influencing their generation.

---

## What OMNIA is NOT

- ❌ Not a language model  
- ❌ Not a classifier  
- ❌ Not an optimizer  
- ❌ Not an agent  
- ❌ Not a decision system  

OMNIA **never chooses**.  
It only **measures**.

---

## Core metrics (stable API)

All metrics are deterministic, bounded, and numerically stable.

| Metric              | Description |
|---------------------|-------------|
| `truth_omega`       | Structural incoherence measure (0 = perfect coherence) |
| `co_plus`           | Inverse coherence score in \[0,1] |
| `score_plus`        | Composite score (CoPlus + bias × info) |
| `delta_coherence`   | Dispersion / instability proxy |
| `kappa_alignment`   | Relative similarity between two signals |
| `epsilon_drift`     | Relative temporal change |

Implementation:

omnia/metrics.py

The API is explicit, import-stable, and globals-free.

---

## Differential diagnostics (non-redundancy evidence)

OMNIA is designed to detect **structural instability even when outcome-based
metrics remain stable**.

The table below shows representative GSM8K items where:
- the answer is correct,
- standard metrics (accuracy, self-consistency) remain stable,
- yet OMNIA flags structural instability.

| item_id | correct | acc_stable | self_consistent | omn_flag | truth_omega | pbii |
|--------:|:-------:|:----------:|:---------------:|:--------:|------------:|-----:|
| 137     | 1       | 1          | 1               | 1        | 1.92        | 0.81 |
| 284     | 1       | 1          | 1               | 1        | 2.31        | 0.88 |

These cases are **locally correct but structurally unstable**.
Outcome-based metrics do not detect them; structure-based metrics do.

---

## Architecture overview

Signal (numbers / time / tokens / causality) | v +------------------+ |   OMNIA LENSES   |   BASE / TIME / CAUSA / TOKEN / LCR +------------------+ | v +------------------+ |   METRIC CORE    |   TruthΩ, Co⁺, Δ, κ, ε +------------------+ | v +------------------+ |   ICE ENVELOPE   |   Impossibility & Confidence Envelope +------------------+

OMNIA outputs **machine-readable diagnostics**, not judgments.

---

## Reproducibility

This repository provides a **fixed, reproducible execution path**.

### Real benchmark run (Colab)

Official notebook:

colab/OMNIA_REAL_RUN.ipynb

What it does:

1. Clones this repository
2. Installs fixed dependencies
3. Locks random seeds
4. Runs real benchmarks
5. Produces machine-readable reports

**Goal:** verification, not exploration.

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

Integration philosophy

OMNIA is composable by design.

Typical separation of roles:

OMNIA → measures structure

External systems → interpret, decide, optimize


Validated separation:

OMNIA = geometry / invariants

Decision systems = policy / intent / judgment


This keeps OMNIA institution-agnostic and architecture-agnostic.


---

Repository identity (canonical)

Canonical repository: https://github.com/Tuttotorna/lon-mirror

Project name: OMNIA / MB-X.01

Author / Origin: Massimiliano Brighindi

There is no secondary mirror and no alternate “lon-mirror1”. All references point here.


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

