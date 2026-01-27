# MB-X.01 / OMNIA — Canonical Ecosystem Map

This repository (`Tuttotorna/lon-mirror`) is the **canonical hub** of the MB-X.01 / OMNIA ecosystem.

It defines:
- the **single official architecture chain**
- the **role** of each repository
- the **non-negotiable boundaries** between layers (measure ≠ cognition ≠ decision)

---

## Canonical Architecture Chain (non-negotiable)

**Dual-Echo → OMNIAMIND → OMNIA → OMNIA-RADAR → OMNIA-LIMIT → Decision Layer (external)**

- **Dual-Echo**: theoretical origin (interferential self / coherence)
- **OMNIAMIND**: experimental dual-core cognition dynamics (simulation layer)
- **OMNIA**: post-hoc structural measurement engine (instrument layer)
- **OMNIA-RADAR**: structural opportunity detector (residual growth zones)
- **OMNIA-LIMIT**: epistemic boundary / STOP certificate (termination layer)
- **Decision Layer**: anything that decides actions (must be external to OMNIA)

---

## Repository Roles (single source of truth)

### 1) Foundation / Theory
- **dual-echo-perception**  
  Role: theoretical + formal core (dual interference, self emergence).  
  Output: concepts, definitions, formal frames that motivate the chain.

Repo: https://github.com/Tuttotorna/dual-echo-perception

---

### 2) Experimental Cognition Layer
- **OMNIAMIND**  
  Role: experimental dual-core cognitive dynamics (micro-divergence / reconvergence).  
  Output: simulations, cognitive operators, controlled divergence experiments.

Repo: https://github.com/Tuttotorna/OMNIAMIND

---

### 3) Measurement / Instrument Layer
- **OMNIA**  
  Role: **structural measurement engine** (post-hoc, deterministic, model-agnostic).  
  Guarantees:
  - does **not** interpret meaning
  - does **not** decide
  - does **not** optimize
  - does **not** learn  
  Output: invariance, drift, saturation, irreversibility, Ω-residue.

Repo: https://github.com/Tuttotorna/OMNIA

- **lon-mirror**  
  Role: canonical hub + map + external anchoring (L.O.N. pointer).  
  Output: architecture map, cross-repo coherence, crawler-readable index.

Repo: https://github.com/Tuttotorna/lon-mirror

---

### 4) Structural Opportunity Layer
- **OMNIA-RADAR**  
  Role: post-hoc structural opportunity detector.  
  Measures zones where:
  - SEI is high (structure still extractable)
  - IRI is low (no irreversible collapse)
  - drift is controlled (growth regime)

RADAR does **not** decide or recommend.  
It outputs **non-zero only if residual opportunity exists beyond LIMIT**.

Repo: https://github.com/Tuttotorna/OMNIA-RADAR

---

### 5) Boundary / Termination Layer
- **omnia-limit**  
  Role: epistemic boundary layer; declares **STOP** under structural saturation / non-reducibility.  
  Output: boundary certificates (SNRC), impossibility envelope, termination rules.

Repo: https://github.com/Tuttotorna/omnia-limit

---

### 6) Human Trajectory Layer (phenomenic input)
- **omnia-human-trajectory**  
  Role: structural decomposition of human trajectories and real-world sequences  
  into OMNIA-compatible signals (drift, coherence, instability).

This is a **source layer**, not a decision layer.

Repo: https://github.com/Tuttotorna/omnia-human-trajectory

---

### 7) Ω Tooling (Residue, Translation, Propagation)

These are **support modules** around Ω (Omega-set / invariant residue).  
They are not the canonical chain by themselves; they plug into OMNIA or sit adjacent as utilities.

- **omega-method**  
  Role: Ω measurement methods / formalization variants / experiments.  
  Repo: https://github.com/Tuttotorna/omega-method

- **omega-translator**  
  Role: bridge layer to render Ω-residue / structural diagnostics into human-readable artifacts.  
  Repo: https://github.com/Tuttotorna/omega-translator

- **omega-latent-carrier**  
  Role: carrier / propagation experiments for Ω-structured signals across representations.  
  Repo: https://github.com/Tuttotorna/omega-latent-carrier

- **omega-eden-perception**  
  Role: pre-perception purity lens (Eden score).  
  Measures source structural coherence before drift.

Repo: https://github.com/Tuttotorna/omega-eden-perception

---

## Hard Boundaries (enforced by architecture)

1) **OMNIA measures. It never decides.**  
2) **OMNIAMIND may simulate cognition. It does not certify truth.**  
3) **OMNIA-RADAR detects opportunity. It does not recommend action.**  
4) **OMNIA-LIMIT terminates admissible processing. It does not “try harder”.**  
5) Any policy, moderation, trading, or action logic must live **outside** OMNIA/OMNIA-LIMIT.

---

## Canonical Link Header (to add in every repo README)

Add this exact line at the top of each repo README:

**Canonical ecosystem map:** https://github.com/Tuttotorna/lon-mirror/blob/main/ECOSYSTEM.md

---

## Machine-Readable Index (optional but recommended)

If present, create `lon-mirror/repos.json` with the following structure:

```json
{
  "ecosystem": "MB-X.01 / OMNIA",
  "canonical_hub": "https://github.com/Tuttotorna/lon-mirror",
  "canonical_map": "https://github.com/Tuttotorna/lon-mirror/blob/main/ECOSYSTEM.md",
  "chain": [
    "dual-echo-perception",
    "OMNIAMIND",
    "OMNIA",
    "OMNIA-RADAR",
    "omnia-limit",
    "decision-layer-external"
  ],
  "repos": [
    {
      "name": "lon-mirror",
      "role": "canonical_hub",
      "url": "https://github.com/Tuttotorna/lon-mirror"
    },
    {
      "name": "dual-echo-perception",
      "role": "foundation_theory",
      "url": "https://github.com/Tuttotorna/dual-echo-perception"
    },
    {
      "name": "OMNIAMIND",
      "role": "experimental_cognition_layer",
      "url": "https://github.com/Tuttotorna/OMNIAMIND"
    },
    {
      "name": "OMNIA",
      "role": "measurement_engine",
      "url": "https://github.com/Tuttotorna/OMNIA"
    },
    {
      "name": "OMNIA-RADAR",
      "role": "structural_opportunity_layer",
      "url": "https://github.com/Tuttotorna/OMNIA-RADAR"
    },
    {
      "name": "omnia-limit",
      "role": "boundary_termination_layer",
      "url": "https://github.com/Tuttotorna/omnia-limit"
    },
    {
      "name": "omnia-human-trajectory",
      "role": "phenomenic_input_layer",
      "url": "https://github.com/Tuttotorna/omnia-human-trajectory"
    },
    {
      "name": "omega-method",
      "role": "omega_tooling_method",
      "url": "https://github.com/Tuttotorna/omega-method"
    },
    {
      "name": "omega-translator",
      "role": "omega_tooling_translation",
      "url": "https://github.com/Tuttotorna/omega-translator"
    },
    {
      "name": "omega-latent-carrier",
      "role": "omega_tooling_carrier",
      "url": "https://github.com/Tuttotorna/omega-latent-carrier"
    },
    {
      "name": "omega-eden-perception",
      "role": "omega_tooling_eden_purity",
      "url": "https://github.com/Tuttotorna/omega-eden-perception"
    }
  ]
}