# Ecosystem Compatibility Matrix

This document defines the public interface expectation across the MB-X.01 / OMNIA repositories.

The matrix is intentionally simple.

It exists to prevent scope blur.

---

## Boundary invariant

Every repository must preserve the same boundary:

    measurement != inference != decision

No repository should silently convert structural measurement into semantic truth or final decision.

---

## Repository contracts

| Repository | Input | Output | Stable public role |
|---|---|---|---|
| lon-mirror | visitor attention | ecosystem map | public hub |
| OMNIA-VALIDATION | artifacts, results, claims | validation reports and regressions | evidence layer |
| OMNIA | output, trace, structure candidate | structural measurement | core engine |
| OMNIABASE | object, number, representation | cross-representation observation | foundation |
| OMNIA-RADAR | raw candidate or trace | detected structural signal | pre-measurement scan |
| OMNIA-INVARIANCE | transformed variants | invariance or instability evidence | transformation layer |
| omnia-limit | measurement sequence | stop or continue certificate | boundary layer |
| OMNIA-CONSTANT | stable Omega-region candidates | falsification, weakening, or persistence result | post-analysis |
| OMNIAMIND | analytic process | organized measurement workflow | orchestration |
| OMNIA-THREE-BODY | dynamic trajectory | divergence or stability artifacts | scientific stress test |
| OMNIA-SECURITY | security-like trace | bounded structural diagnostic | vertical |
| OMNIA-CRYPTO | crypto-like transformation | bounded structural diagnostic | vertical |

---

## Recommended public path

    lon-mirror
      -> OMNIA-VALIDATION
      -> OMNIA
      -> OMNIABASE
      -> OMNIA-RADAR
      -> OMNIA-INVARIANCE
      -> omnia-limit
      -> OMNIA-CONSTANT

Verticals:

    OMNIA-THREE-BODY
    OMNIA-SECURITY
    OMNIA-CRYPTO
    OMNIAMIND

---

## Minimum README schema for all repositories

Every repository should expose these sections near the top:

    1. What this is
    2. What this is not
    3. Start here
    4. Input
    5. Output
    6. Boundary
    7. Related repositories
    8. Reproducibility status

---

## Non-negotiable public distinction

    RADAR detects.
    OMNIA measures.
    INVARIANCE compares transformations.
    LIMIT certifies boundary.
    VALIDATION tests artifacts.
    External agents decide.

