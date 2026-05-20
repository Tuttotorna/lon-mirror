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

## Functional layers

| Layer | Repository | Role |
|---|---|---|
| Hub | lon-mirror | Public entry point and map |
| Evidence | OMNIA-VALIDATION | Reproducibility, validation, regression |
| Measurement | OMNIA | Core post-hoc structural measurement |
| Representation | OMNIABASE | Multi-base / multi-representation observation |
| Detection | OMNIA-RADAR | Structural signal pre-scan |
| Invariance | OMNIA-INVARIANCE | Persistence under transformation |
| Limit | omnia-limit | Boundary / stop condition |
| Post-analysis | OMNIA-CONSTANT | Falsification of Omega-region stability |
| Orchestration | OMNIAMIND | Structural cognition workflow organization |
| Scientific case | OMNIA-THREE-BODY | Chaotic trajectory divergence |
| Security vertical | OMNIA-SECURITY | Bounded security-relevant diagnostics |
| Crypto vertical | OMNIA-CRYPTO | Bounded crypto-like diagnostics |

---

## Conceptual adjacency

| Repository | Receives from | Sends to | Boundary |
|---|---|---|---|
| lon-mirror | none | all repos | hub only |
| OMNIA-VALIDATION | all repos | reviewers / regressions | validates artifacts, not truth |
| OMNIA | traces / outputs / representations | invariance, limit, validation | measures structure only |
| OMNIABASE | numbers / signals / representations | OMNIA / invariance | observes representation dependence |
| OMNIA-RADAR | raw candidates / traces | OMNIA / validation | detects signal, does not measure final admissibility |
| OMNIA-INVARIANCE | OMNIA outputs / transformed traces | constant / validation | tests persistence |
| omnia-limit | OMNIA / invariance signals | external decision layer | certifies stop/continue, does not decide |
| OMNIA-CONSTANT | Omega regions / invariance results | validation | falsifies stability claims |
| OMNIAMIND | analytic workflows | OMNIA / limit | organizes, does not become consciousness |
| OMNIA-THREE-BODY | simulated trajectories | invariance / validation | stress test, not physics proof |
| OMNIA-SECURITY | security-like traces | validation / limit | diagnostic, not scanner |
| OMNIA-CRYPTO | crypto-like transformations | validation / limit | diagnostic, not cryptographic proof |

---

## Public rule

A new visitor should never be forced to understand every repository before seeing value.

The correct first proof is:

    clone -> install -> run test -> inspect artifact -> read theory later

