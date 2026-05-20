# Ecosystem Status

This document records the public restructuring status of the MB-X.01 / OMNIA repositories.

Canonical boundary:

    measurement != inference != decision

Public route:

    evidence first -> engine second -> theory third

Root principle:

    README = entrance
    docs = explanation
    repos = proof

---

## Current repository status

| Step | Repository | Public role | Pipeline | Test status |
|---:|---|---|---|---|
| 1 | [lon-mirror](https://github.com/Tuttotorna/lon-mirror) | Canonical public entry point | `visitor -> map -> evidence path -> repository route` | not applicable / hub repository |
| 2 | [OMNIA-VALIDATION](https://github.com/Tuttotorna/OMNIA-VALIDATION) | Public validation showroom | `test -> output -> artifact -> failure/fragility -> report` | 269 passed |
| 3 | [OMNIA](https://github.com/Tuttotorna/OMNIA) | Core structural measurement engine | `input -> transformation -> measurement -> output -> boundary` | 47 passed |
| 4 | [OMNIABASE](https://github.com/Tuttotorna/OMNIABASE) | Representation invariance foundation | `representation -> base shift -> observation -> invariant residue` | 5 passed |
| 5 | [omnia-limit](https://github.com/Tuttotorna/omnia-limit) | Stop / continue boundary layer | `measurement sequence -> saturation / irreversibility -> stop-or-continue boundary` | 11 passed |
| 6 | [OMNIA-RADAR](https://github.com/Tuttotorna/OMNIA-RADAR) | Structural signal detection layer | `candidate trace -> signal scan -> anomaly / persistence / drift -> measurement candidate` | 9 passed |
| 7 | [OMNIA-INVARIANCE](https://github.com/Tuttotorna/OMNIA-INVARIANCE) | Structural invariance layer | `source structure -> controlled transformation -> invariance check -> stability / collapse` | 5 passed |
| 8 | [OMNIA-CONSTANT](https://github.com/Tuttotorna/OMNIA-CONSTANT) | Structural constant candidate layer | `candidate invariant -> repeated validation -> stable region -> falsification pressure` | 5 passed |
| 9 | [OMNIAMIND](https://github.com/Tuttotorna/OMNIAMIND) | Structural cognition orchestration layer | `observation -> structural reasoning workflow -> OMNIA measurement -> limit boundary -> external decision` | no test suite currently present; pytest exit code 5 accepted as repository status |
| 10 | [OMNIA-THREE-BODY](https://github.com/Tuttotorna/OMNIA-THREE-BODY) | Dynamic divergence stress test | `initial state -> perturbation -> trajectory divergence -> structural instability` | 6 passed |
| 11 | [OMNIA-SECURITY](https://github.com/Tuttotorna/OMNIA-SECURITY) | Bounded structural security diagnostics | `security-relevant trace -> structural diagnostic -> risk signal -> external security decision` | 5 passed |
| 12 | [OMNIA-CRYPTO](https://github.com/Tuttotorna/OMNIA-CRYPTO) | Bounded structural crypto diagnostics | `crypto-relevant trace -> structural diagnostic -> entropy / invariance / drift signal -> external cryptographic review` | 5 passed |

---

## Structural map

    lon-mirror
      -> OMNIA-VALIDATION
      -> OMNIA
      -> OMNIABASE
      -> OMNIA-RADAR
      -> OMNIA-INVARIANCE
      -> omnia-limit
      -> OMNIA-CONSTANT
      -> OMNIAMIND
      -> OMNIA-THREE-BODY
      -> OMNIA-SECURITY
      -> OMNIA-CRYPTO

---

## Public interpretation

The ecosystem is now organized as a readable public architecture.

Each repository has:

- a narrow public role;
- explicit boundaries;
- dedicated documentation;
- a manifest file;
- at least one minimal example artifact;
- test status recorded where applicable.

---

## Boundary discipline

No repository should silently convert:

    measurement into inference
    inference into decision
    diagnostic signal into verdict
    stability into truth
    orchestration into consciousness
    crypto diagnostic into key recovery
    security diagnostic into vulnerability proof

Decision remains external.

