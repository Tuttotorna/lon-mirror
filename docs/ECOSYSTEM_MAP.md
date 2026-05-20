# MB-X.01 / OMNIA Ecosystem Map

This document defines the functional map of the MB-X.01 / OMNIA ecosystem.

The ecosystem should be read as one modular architecture, not as unrelated repositories.

    detect -> measure -> test invariance -> certify limit -> validate/falsify

---

## Canonical public chain

    lon-mirror
      ↓
    OMNIA-VALIDATION
      ↓
    OMNIA
      ↓
    OMNIABASE
      ↓
    OMNIA-RADAR
      ↓
    OMNIA-INVARIANCE
      ↓
    omnia-limit
      ↓
    OMNIA-CONSTANT

The public onboarding order is intentionally not the same as the theoretical dependency order.

For public adoption:

    evidence first
    engine second
    theory third

---

## Functional architecture

| Layer | Repository | Function |
|---|---|---|
| Public entry | lon-mirror | Canonical ecosystem hub |
| Evidence first | OMNIA-VALIDATION | Validation, falsification, artifacts, regression |
| Core engine | OMNIA | Post-hoc structural measurement |
| Foundation | OMNIABASE | Multi-base / multi-representation observation |
| Pre-measurement | OMNIA-RADAR | Signal, drift, persistence and anomaly detection |
| Transformation | OMNIA-INVARIANCE | Structural stability across transformations |
| Boundary | omnia-limit | Saturation, irreversibility and stop conditions |
| Post-analysis | OMNIA-CONSTANT | Falsification of stable Omega regions |
| Orchestration | OMNIAMIND | Structural cognition pipeline orchestration |
| Scientific case | OMNIA-THREE-BODY | Chaotic trajectory divergence stress test |
| Security vertical | OMNIA-SECURITY | Bounded structural diagnostics for security-like traces |
| Crypto vertical | OMNIA-CRYPTO | Bounded structural diagnostics for crypto-like behavior |

---

## Conceptual adjacency

| Repository | Receives from | Sends to | Boundary |
|---|---|---|---|
| lon-mirror | none | all repositories | hub only |
| OMNIA-VALIDATION | artifacts, outputs, reports, claims | reviewers, regressions, reproducibility checks | validates artifacts, not truth |
| OMNIA | traces, outputs, structures, representations | invariance, limit, validation | measures structure only |
| OMNIABASE | numbers, signals, representations | OMNIA, invariance | observes representation dependence |
| OMNIA-RADAR | raw candidates, traces | OMNIA, validation | detects signal, does not measure final admissibility |
| OMNIA-INVARIANCE | transformed variants | constant, validation | tests persistence |
| omnia-limit | OMNIA and invariance signals | external decision layer | certifies stop or continue, does not decide |
| OMNIA-CONSTANT | Omega-region candidates | validation | falsifies or weakens stability claims |
| OMNIAMIND | analytic workflows | OMNIA, limit | organizes, does not become consciousness |
| OMNIA-THREE-BODY | simulated trajectories | invariance, validation | stress test, not physics proof |
| OMNIA-SECURITY | security-like traces | validation, limit | diagnostic, not scanner |
| OMNIA-CRYPTO | crypto-like transformations | validation, limit | diagnostic, not cryptographic proof |

---

## Text diagram

    lon-mirror
      |
      +-- OMNIA-VALIDATION ----> external review
      |
      +-- OMNIABASE -----------> OMNIA
      |                            |
      +-- OMNIA-RADAR ----------> |
      |                            |
      +-- OMNIAMIND ------------> |
                                   |
                                   v
                              OMNIA-INVARIANCE
                                   |
                                   v
                              OMNIA-CONSTANT
                                   |
                                   v
                              OMNIA-VALIDATION

    OMNIA -> omnia-limit -> external decision

    OMNIA-THREE-BODY -> OMNIA-INVARIANCE -> OMNIA-VALIDATION
    OMNIA-SECURITY   -> OMNIA-VALIDATION
    OMNIA-CRYPTO     -> OMNIA-VALIDATION

---

## Public rule

A new visitor should never be forced to understand every repository before seeing value.

The correct first proof is:

    clone -> install -> run test -> inspect artifact -> read theory later

