# MB-X.01 / OMNIA

**Structural measurement beyond correctness.**

This repository is the public entry point for the MB-X.01 / OMNIA ecosystem.

OMNIA is not a reasoning engine, not a truth oracle, and not a decision system.

It is a post-hoc structural measurement ecosystem designed to test whether outputs, traces, systems, or representations preserve measurable structure under transformation, perturbation, and limit conditions.

    measurement != inference != decision

    Structural truth = invariance under transformation

---

## Start here in 60 seconds

If this is your first contact with the ecosystem, do not start from the theory.

Start from the smallest reproducible path:

    git clone https://github.com/Tuttotorna/OMNIA-VALIDATION.git
    cd OMNIA-VALIDATION
    python -m pip install -e .
    pytest

Then read:

1. [OMNIA-VALIDATION](https://github.com/Tuttotorna/OMNIA-VALIDATION) — evidence, artifacts, reproducibility.
2. [OMNIA](https://github.com/Tuttotorna/OMNIA) — core structural measurement engine.
3. [OMNIABASE](https://github.com/Tuttotorna/OMNIABASE) — multi-representation foundation.

The public path is:

    see evidence -> understand measurement -> understand representation

Not the reverse.

---

## What OMNIA does

OMNIA measures structural behavior.

It can be used to observe whether a generated output, trace, representation, transformation, or trajectory remains structurally admissible under independent perturbations and transformations.

At ecosystem level, the canonical pipeline is:

    detect -> measure -> test invariance -> certify limit -> validate/falsify

---

## What OMNIA does not do

OMNIA does **not**:

- infer semantic truth;
- decide what is correct;
- replace human judgment;
- claim consciousness;
- claim intelligence;
- perform security scanning;
- perform cryptographic attacks;
- recover keys;
- prove physical truth;
- convert structural stability into final meaning.

The boundary is intentional:

    measurement stops before inference
    inference stops before decision
    decision remains external

---

## Recommended onboarding order

| Step | Repository | Role | Layer |
|---:|---|---|---|
| 1 | [lon-mirror](https://github.com/Tuttotorna/lon-mirror) | Canonical public hub and ecosystem entry point | hub |
| 2 | [OMNIA-VALIDATION](https://github.com/Tuttotorna/OMNIA-VALIDATION) | Evidence, reproducibility, regression and artifact validation | evidence |
| 3 | [OMNIA](https://github.com/Tuttotorna/OMNIA) | Post-hoc structural measurement engine | core |
| 4 | [OMNIABASE](https://github.com/Tuttotorna/OMNIABASE) | Multi-representation and multi-base observation layer | foundation |
| 5 | [OMNIA-RADAR](https://github.com/Tuttotorna/OMNIA-RADAR) | Structural signal detection before measurement | detection |
| 6 | [OMNIA-INVARIANCE](https://github.com/Tuttotorna/OMNIA-INVARIANCE) | Cross-representation invariance under transformation | invariance |
| 7 | [omnia-limit](https://github.com/Tuttotorna/omnia-limit) | Boundary, stop condition and limit certification | boundary |
| 8 | [OMNIA-CONSTANT](https://github.com/Tuttotorna/OMNIA-CONSTANT) | Post-analysis and falsification of stable Omega regions | post-analysis |
| 9 | [OMNIAMIND](https://github.com/Tuttotorna/OMNIAMIND) | Structural cognition orchestration around OMNIA measurements | orchestration |
| 10 | [OMNIA-THREE-BODY](https://github.com/Tuttotorna/OMNIA-THREE-BODY) | Three-body structural divergence stress test | scientific-case |
| 11 | [OMNIA-SECURITY](https://github.com/Tuttotorna/OMNIA-SECURITY) | Bounded structural diagnostics for security-relevant traces | vertical |
| 12 | [OMNIA-CRYPTO](https://github.com/Tuttotorna/OMNIA-CRYPTO) | Bounded structural diagnostics for crypto-like behavior | vertical |

---

## Ecosystem map

    lon-mirror
      |
      |-- OMNIA-VALIDATION     evidence / reproducibility / regression
      |-- OMNIA                core structural measurement
      |-- OMNIABASE            multi-representation observation
      |-- OMNIA-RADAR          structural signal detection
      |-- OMNIA-INVARIANCE     invariance under transformation
      |-- omnia-limit          stop / continue boundary
      |-- OMNIA-CONSTANT       Omega-region falsification
      |-- OMNIAMIND            orchestration layer
      |-- OMNIA-THREE-BODY     dynamic stress test
      |-- OMNIA-SECURITY       bounded security diagnostics
      |-- OMNIA-CRYPTO         bounded crypto diagnostics

---

## Functional architecture

| Layer | Repository | Function |
|---|---|---|
| Public entry | [lon-mirror](https://github.com/Tuttotorna/lon-mirror) | Canonical ecosystem hub |
| Evidence first | [OMNIA-VALIDATION](https://github.com/Tuttotorna/OMNIA-VALIDATION) | Validation, falsification, artifacts, regression |
| Core engine | [OMNIA](https://github.com/Tuttotorna/OMNIA) | Post-hoc structural measurement |
| Foundation | [OMNIABASE](https://github.com/Tuttotorna/OMNIABASE) | Multi-base / multi-representation observation |
| Pre-measurement | [OMNIA-RADAR](https://github.com/Tuttotorna/OMNIA-RADAR) | Signal, drift, persistence and anomaly detection |
| Transformation | [OMNIA-INVARIANCE](https://github.com/Tuttotorna/OMNIA-INVARIANCE) | Structural stability across transformations |
| Boundary | [omnia-limit](https://github.com/Tuttotorna/omnia-limit) | Saturation, irreversibility and stop conditions |
| Post-analysis | [OMNIA-CONSTANT](https://github.com/Tuttotorna/OMNIA-CONSTANT) | Falsification of stable Omega regions |
| Orchestration | [OMNIAMIND](https://github.com/Tuttotorna/OMNIAMIND) | Structural cognition pipeline orchestration |
| Scientific case | [OMNIA-THREE-BODY](https://github.com/Tuttotorna/OMNIA-THREE-BODY) | Chaotic trajectory divergence stress test |
| Security vertical | [OMNIA-SECURITY](https://github.com/Tuttotorna/OMNIA-SECURITY) | Bounded structural diagnostics for security-like traces |
| Crypto vertical | [OMNIA-CRYPTO](https://github.com/Tuttotorna/OMNIA-CRYPTO) | Bounded structural diagnostics for crypto-like behavior |

---

## The shortest correct explanation

    MB-X.01 / OMNIA is an open-source research ecosystem for structural measurement beyond semantic interpretation.

    It detects, measures, validates, and bounds invariant structure across representations, perturbations, and domains.

    It does not infer meaning.
    It does not decide truth.
    It measures whether structure survives transformation.

---

## Why this exists

Modern systems often look correct at the surface while becoming structurally fragile under transformation.

OMNIA exists to make that fragility measurable.

The central question is not:

    Does this look right?

The central question is:

    What remains structurally stable when the representation changes?

---

## Canonical boundary

Every repository in this ecosystem should preserve the same boundary:

    measurement != inference != decision

That means:

- measurement may produce a structural signal;
- inference may interpret that signal;
- decision may act on that interpretation;
- OMNIA itself stops at measurement and boundary certification.

---

## Documentation

| Document | Purpose |
|---|---|
| [docs/ECOSYSTEM_MAP.md](docs/ECOSYSTEM_MAP.md) | Full map of repositories and conceptual adjacency |
| [docs/ONBOARDING_60_SECONDS.md](docs/ONBOARDING_60_SECONDS.md) | Minimal public onboarding path |
| [docs/COMPATIBILITY_MATRIX.md](docs/COMPATIBILITY_MATRIX.md) | Interface and compatibility matrix |
| [docs/ecosystem_inventory.json](docs/ecosystem_inventory.json) | Machine-readable ecosystem inventory |
| [archive/readme-history/](archive/readme-history/) | Previous README snapshots preserved during restructuring |

---

## Status

This hub is intentionally being reduced to a clear public entrance.

Material that is experimental, historical, or implementation-specific should live in:

    archive/
    lab/
    docs/
    examples/
    benchmarks/
    integrations/

The root README should remain a landing page, not a full theory dump.

---

## License

MIT.

