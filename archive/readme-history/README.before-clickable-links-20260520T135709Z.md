# MB-X.01 / OMNIA

## DOI

[![DOI](https://zenodo.org/badge/1067137742.svg)](https://zenodo.org/badge/latestdoi/1067137742)

Release DOI:

    10.5281/zenodo.20089284

Zenodo latest DOI link:

    https://doi.org/10.5281/zenodo.20089284

GitHub release:

    https://github.com/Tuttotorna/lon-mirror/releases/tag/v1.0.0

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

Canonical ecosystem pipeline:

    detect -> measure -> test invariance -> certify limit -> validate/falsify

---

## What OMNIA does not do

OMNIA does not:

- infer semantic truth;
- decide what is correct;
- replace human judgment;
- claim consciousness;
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

| Step | Repository | Role |
|---:|---|---|
| 1 | [lon-mirror](https://github.com/Tuttotorna/lon-mirror) | Canonical public hub and ecosystem entry point |
| 2 | [OMNIA-VALIDATION](https://github.com/Tuttotorna/OMNIA-VALIDATION) | Evidence, reproducibility, regression and artifact validation |
| 3 | [OMNIA](https://github.com/Tuttotorna/OMNIA) | Post-hoc structural measurement engine |
| 4 | [OMNIABASE](https://github.com/Tuttotorna/OMNIABASE) | Multi-representation and multi-base observation layer |
| 5 | [OMNIA-RADAR](https://github.com/Tuttotorna/OMNIA-RADAR) | Structural signal detection before measurement |
| 6 | [OMNIA-INVARIANCE](https://github.com/Tuttotorna/OMNIA-INVARIANCE) | Cross-representation invariance under transformation |
| 7 | [omnia-limit](https://github.com/Tuttotorna/omnia-limit) | Boundary, stop condition and limit certification |
| 8 | [OMNIA-CONSTANT](https://github.com/Tuttotorna/OMNIA-CONSTANT) | Post-analysis and falsification of stable Omega regions |
| 9 | [OMNIAMIND](https://github.com/Tuttotorna/OMNIAMIND) | Structural cognition orchestration around OMNIA measurements |
| 10 | [OMNIA-THREE-BODY](https://github.com/Tuttotorna/OMNIA-THREE-BODY) | Three-body structural divergence stress test |
| 11 | [OMNIA-SECURITY](https://github.com/Tuttotorna/OMNIA-SECURITY) | Bounded structural diagnostics for security-relevant traces |
| 12 | [OMNIA-CRYPTO](https://github.com/Tuttotorna/OMNIA-CRYPTO) | Bounded structural diagnostics for crypto-like behavior |

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

## Current ecosystem status

The public restructuring pass has been completed across the core MB-X.01 / OMNIA repositories.

    lon-mirror        = public entrance
    OMNIA-VALIDATION = showroom / proof
    OMNIA             = core measurement engine
    OMNIABASE         = representation foundation
    omnia-limit       = stop / continue boundary
    OMNIA-RADAR       = structural signal detection
    OMNIA-INVARIANCE  = structural invariance check
    OMNIA-CONSTANT    = structural constant candidate
    OMNIAMIND         = structural cognition orchestration
    OMNIA-THREE-BODY  = dynamic divergence stress test
    OMNIA-SECURITY    = bounded security diagnostics
    OMNIA-CRYPTO      = bounded crypto diagnostics

See [docs/ECOSYSTEM_STATUS.md](docs/ECOSYSTEM_STATUS.md) for the synchronized status table.

---

## Documentation

| Document | Purpose |
|---|---|
| [docs/ONBOARDING_60_SECONDS.md](docs/ONBOARDING_60_SECONDS.md) | Minimal public onboarding path |
| [docs/ECOSYSTEM_MAP.md](docs/ECOSYSTEM_MAP.md) | Full ecosystem map and conceptual adjacency |
| [docs/ECOSYSTEM_STATUS.md](docs/ECOSYSTEM_STATUS.md) | Current public restructuring status across repositories |
| [docs/PUBLIC_NAVIGATION.md](docs/PUBLIC_NAVIGATION.md) | Direct navigation by visitor need |
| [docs/FOUNDATION.md](docs/FOUNDATION.md) | Core principles, boundary and conceptual foundation |
| [docs/COMPATIBILITY_MATRIX.md](docs/COMPATIBILITY_MATRIX.md) | Interface and compatibility matrix |
| [docs/ecosystem_inventory.json](docs/ecosystem_inventory.json) | Machine-readable ecosystem inventory |
| [LANDING_SCOPE.md](LANDING_SCOPE.md) | Rule for keeping this repository readable |

---

## License

MIT.

