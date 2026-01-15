# OMNIA — Architecture

## Overview

OMNIA is a **layered structural measurement system** built from independent lenses that analyze different structural dimensions of a representation.

Each lens produces **deterministic signals**.  
All signals are fused into a unified **Ω-total structural profile**.

OMNIA does not interpret results and does not trigger actions.

---

## Architectural Separation

OMNIA enforces a strict separation of concerns:

- **Measurement layer** → OMNIA (this project)
- **Cognition layer** → external (models, reasoning systems)
- **Decision layer** → external (policies, actions, safety)

This separation is non-negotiable.

---

## Core Components

### 1. Lenses

Each lens measures a specific structural dimension.

#### BASE — Omniabase
Analyzes numeric structure across multiple bases.

Measures:
- digit entropy
- σ-symmetry and compactness
- divisibility patterns
- PBII (Prime Base Instability Index)
- multi-base invariant profiles

#### TIME — Omniatempo
Analyzes temporal stability and drift.

Measures:
- distribution shifts
- short vs long window divergence
- regime change signals
- temporal instability accumulation

#### CAUSA — Omniacausa
Extracts lagged causal structure across multivariate signals.

Measures:
- correlation across all lags
- strongest lag per signal pair
- edge emission above threshold

#### TOKEN — Token Lens
Applies numeric structural analysis to token sequences.

Pipeline:
- token → integer proxy
- PBII per token
- z-score normalization
- token-level instability aggregation

#### LCR — Logical Coherence Reduction
External coherence validation layer.

Measures:
- factual consistency
- numeric consistency
- gold-reference alignment
- optional structural Ω coupling

---

## Ω Fusion Engine (Ω-TOTAL)

All lens outputs are combined by the **Ω-TOTAL engine**.

Each lens contributes a normalized structural signal:

| Lens  | Structural Meaning          |
|------|-----------------------------|
| BASE | Numeric instability         |
| TIME | Temporal drift              |
| CAUSA| Cross-channel structure     |
| TOKEN| Token-level instability     |
| LCR  | External logical coherence  |

The fusion process produces:

- unified Ω-total score
- per-lens component scores
- JSON-safe metadata for reproducibility

Ω is not a truth value.  
Ω is a **structural residue**.

---

## Data Flow

1. Representation enters OMNIA
2. Each lens analyzes the representation independently
3. Lens outputs are normalized
4. Ω-TOTAL fuses signals
5. Structural profile is emitted

At no point does OMNIA:
- modify the representation
- feed results back into the model
- enforce thresholds as decisions

---

## Determinism

Given:
- identical input
- identical configuration
- identical lens set

OMNIA produces **identical outputs**.

Randomness is not permitted inside the measurement layer.

---

## Extensibility

New lenses may be added if they:
- are deterministic
- operate on structure, not meaning
- produce bounded, interpretable signals
- do not violate architectural separation

Fusion weights and aggregation strategies may evolve, but the separation principle must remain intact.