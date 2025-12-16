# OMNIA — Unified Structural Measurement Engine  
(BASE ± TIME ± CAUSA ± TOKEN ± LCR)

**Fused Ω-Score Engine · MB-X.01**  
**Author:** Massimiliano Brighindi

---

## Overview

**OMNIA** is a **model-agnostic structural measurement engine**.

It is designed to **measure, quantify, and expose structural instability**
in numbers, sequences, temporal processes, causal systems, and AI outputs
using **independent structural lenses**.

OMNIA **does not decide**.  
OMNIA **does not optimize**.  
OMNIA **does not generate answers**.

OMNIA **measures structural coherence**.

All outputs are:
- deterministic
- machine-readable
- reproducible
- architecture-agnostic

Any decision logic remains **external**.

This repository is the public, executable implementation of the **MB-X.01** research line.

---

## Architecture at a Glance

OMNIA is a **deterministic structural sensor**.

Input signals are observed through multiple **independent lenses**.  
Each lens produces numeric measurements.  
Measurements are fused into a single Ω-score and an optional confidence envelope.

OMNIA never interprets meaning.  
It measures **resistance to deformation**.

---

## Dependency Pipeline (End-to-End)

Input ↓ Ω-TOTAL Orchestrator ↓ Independent Structural Lenses ↓ Signal Fusion ↓ ICE Envelope (optional) ↓ Machine-Readable Report (JSON / CSV)

All entry points resolve to the same core engine and lenses.
The pipeline is deterministic and reproducible.

---

## Core Principle

> **Truth is structural invariance.**  
> A structure that collapses under an independent transformation is unstable.

OMNIA does not claim truth.  
It measures **structural resistance**.

The core scalar produced by OMNIA is:

**TruthΩ (Ω-score)**  
A continuous measure of invariance under independent structural lenses.

---

## Structural Lenses

Each lens observes the **same input** under a **different transformation**.  
All outputs are numeric and machine-verifiable.

---

### Omniabase (BASE)

**Multi-base structural analysis of numeric representations.**

**Core signals**
- multi-base digit entropy
- compactness / dispersion
- invariant signatures
- PBII — Prime Base Instability Index

**Use cases**
- numeric structure analysis
- anomaly detection
- numeric hallucination sensing
- structural discrimination (e.g. primes vs composites)

---

### Omniatempo (TIME)

**Temporal stability and drift detection.**

**Core signals**
- short vs long window divergence
- regime change detection
- drift accumulation

**Use cases**
- time-series monitoring
- reasoning drift in long chains
- sensor instability detection

---

### Omniacausa (CAUSA)

**Lagged causal structure extraction over multivariate signals.**

**Core signals**
- correlation across all lags
- strongest lag per signal pair
- edge emission above threshold

**Use cases**
- causal dependency mapping
- hidden interaction detection
- non-semantic causal inspection

---

### Token Lens (TOKEN)

**Structural instability analysis applied to token sequences.**

**Pipeline**
- token → numeric proxy
- structural dispersion analysis
- instability aggregation

**Use cases**
- hallucination localization
- chain-of-thought fracture detection
- token-level instability sensing

---

### LCR — Logical Coherence Reduction

**External coherence validation layer (FACT + NUMERIC fusion).**

**Integrated signals**
- factual consistency
- numeric consistency
- optional structural Ω input

**Outputs**
- Ω_ext (external coherence score)
- confusion-matrix metrics

LCR is **optional** and external-facing.  
OMNIA remains a **pure structural sensor**.

---

## Fused Ω Engine (Ω-TOTAL)

All lenses contribute to a unified structural score.

| Lens  | Meaning             | Contribution        |
|------|---------------------|---------------------|
| BASE | Numeric instability | PBII / invariants   |
| TIME | Temporal drift      | regime score        |
| CAUSA| Causal structure    | edge strength       |
| TOKEN| Token instability   | dispersion score    |
| LCR  | External coherence  | Ω_ext               |

**Outputs**
- Ω-total score
- per-lens component breakdown
- JSON-safe metadata
- deterministic reproducibility

---

## Repository Structure

omnia/ init.py omniabase.py omniatempo.py omniacausa.py omniatoken.py metrics.py        # TruthΩ, Δ-coherence, κ-alignment envelope.py       # ICE: confidence / impossibility ice.py            # legacy ICE gate (compatibility)

examples/ minimal_validator.py

tests/ test_metrics.py

ARCHITECTURE_BOUNDARY.md requirements.txt README.md

All modules are:
- import-safe
- deterministic
- standalone

---

## Installation

**Requirements**
- Python ≥ 3.9

Clone the repository:

```bash
git clone https://github.com/Tuttotorna/lon-mirror
cd lon-mirror

Install dependencies:

pip install -r requirements.txt


---

Quick Validation

Run the minimal validator:

python examples/minimal_validator.py

This executes:

multi-base signature generation

TruthΩ / Δ / κ computation

ICE envelope construction


Output is a stable, machine-readable diagnostic report.


---

Core API Usage

Structural Metrics

from omnia import compute_metrics

metrics = compute_metrics(signatures)
print(metrics.truth_omega)

ICE Envelope (Confidence / Impossibility)

from omnia import build_ice

ice = build_ice(metrics)
print(ice.confidence, ice.impossibility, ice.flags)

OMNIA does not gate by default.
The envelope is informational.


---

Design Boundaries

OMNIA:

does NOT generate content

does NOT enforce policies

does NOT decide correctness

does NOT replace reasoning engines


These constraints are enforced in
ARCHITECTURE_BOUNDARY.md.


---

Limitations

Structural metrics do not imply semantic truth

Multi-base analysis depends on chosen bases

Temporal and causal lenses are statistical, not ontological

LCR quality depends on external backend quality



---

Author & Lineage

Massimiliano Brighindi

Creator of:

Omniabase

TruthΩ

PBII

ICE Envelope

OMNIA Unified Structural Engine


This repository is the executable mirror of the MB-X.01 research line.

Logical Origin Node:
https://massimiliano.neocities.org/


---

License

MIT License


---

Citation

Brighindi, M. (2025).
OMNIA — Unified Structural Measurement Engine (MB-X.01)
GitHub: https://github.com/Tuttotorna/lon-mirror

