# OMNIA / MB-X.01 — Logical Origin Node (L.O.N.)

**OMNIA** is a deterministic measurement engine for **structural coherence and instability**  
across numbers, time, causality and token sequences.

It does **not** interpret meaning.  
It does **not** make decisions.  
It **measures invariants**.

This repository (`Tuttotorna/lon-mirror`) is the **canonical public mirror** of the MB-X.01 / OMNIA research line.

---

## What OMNIA is

OMNIA is a **pure diagnostic layer**:

- Input: signals (numeric, temporal, causal, token-based)
- Output: **structure-only metrics**
- No semantic assumptions
- No policy, no intent, no alignment layer

Core idea:  
> *Truth is what remains invariant across representations.*

---

## Core signals (stable API)

All metrics are deterministic, bounded, and numerically stable.

| Metric            | Meaning |
|------------------|---------|
| `TruthOmega`     | Structural incoherence measure (0 = perfect coherence) |
| `CoPlus`         | Inverse coherence score in \[0,1] |
| `ScorePlus`      | Composite score (CoPlus + bias × info) |
| `DeltaCoherence`| Dispersion / instability proxy |
| `KappaAlignment`| Relative similarity between two signals |
| `EpsilonDrift`  | Relative temporal change |

These metrics are implemented in:

omnia/metrics.py

and verified by invariant-based tests.

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

## Architecture overview

┌──────────┐ │  Signal  │  (numbers / time / tokens / causality) └────┬─────┘ │ ▼ ┌────────────┐ │ OMNIA LENS │  (BASE / TIME / CAUSA / TOKEN / LCR) └────┬───────┘ │ ▼ ┌──────────────┐ │ METRIC CORE  │  (TruthΩ, Co⁺, Δ, κ, ε) └────┬─────────┘ │ ▼ ┌──────────────┐ │ ICE ENVELOPE │  (Impossibility & Confidence Envelope) └──────────────┘

OMNIA outputs **machine-readable diagnostics**, not judgments.

---

## Reproducibility (important)

This repo contains a **fixed, reproducible execution path**.

### Real benchmark run (Colab)

Official notebook:

colab/OMNIA_REAL_RUN.ipynb

What it does:

1. Clones this repository
2. Installs fixed dependencies
3. Locks random seeds
4. Runs real benchmarks
5. Produces machine-readable reports

Target: **verification, not exploration**.

---

## Tests

Invariant-based tests live in:

tests/test_metrics.py

They verify:

- algebraic identities
- monotonicity
- edge cases
- numerical stability
- API contract

Run locally:

```bash
pytest


---

Integration philosophy

OMNIA is designed to be composable.

Typical usage:

OMNIA → measures structure

External system → interprets / decides


Validated separation (example):

OMNIA = geometry / invariants

Decision systems = policy / intent / judgment


This keeps OMNIA institution-agnostic and architecture-agnostic.


---

Repository identity (canonical)

Canonical repo:
https://github.com/Tuttotorna/lon-mirror

Project name:
OMNIA / MB-X.01

Author / Origin:
Massimiliano Brighindi


There is no secondary mirror and no alternate “lon-mirror1”.
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
