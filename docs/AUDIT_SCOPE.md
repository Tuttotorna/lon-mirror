# Audit Scope

## Purpose

This document defines the public audit scope for `lon-mirror`.

It exists because the repository contains both:

```text
public normative review files
active executable code
tests
legacy artifacts
raw/generated experiment outputs
archive material
```

These categories must not be evaluated as if they had the same role.

The boundary remains:

```text
measurement != inference != decision
```

---

## Public normative surface

The public normative surface is the material a first external reviewer should read first.

Current public entrypoints:

- [`README.md`](../README.md)
- [`docs/REVIEWER_60_SECOND_START.md`](REVIEWER_60_SECOND_START.md)
- [`docs/ECOSYSTEM_ONE_PAGE.md`](ECOSYSTEM_ONE_PAGE.md)
- [`docs/PUBLIC_REVIEW_PATH.md`](PUBLIC_REVIEW_PATH.md)
- [`docs/PUBLIC_CLAIM_BOUNDARY_AUDIT.md`](PUBLIC_CLAIM_BOUNDARY_AUDIT.md)
- [`docs/ECOSYSTEM_LINK_AUDIT.md`](ECOSYSTEM_LINK_AUDIT.md)
- [`docs/AUDIT_SCOPE.md`](AUDIT_SCOPE.md)

These files define the public reading path.

They are the first files to audit for public clarity.

---

## Active executable surface

The active executable surface includes:

```text
tests/
core Python modules imported by tests
documented runnable examples
```

The minimum current executable health check is:

```bash
python -m pytest -q
```

If the tests pass, the active code surface is not automatically proven true.

It only means the current executable checks pass.

Boundary:

```text
tests pass != semantic truth
tests pass != final validation
tests pass != deployment approval
```

---

## Legacy / archive / raw artifact surface

The repository also contains legacy, archived, raw, generated, or exploratory artifacts.

These may include:

```text
archive/
raw experiment dumps
generated reports
old JSON or JSONL logs
historical validation material
non-normative scratch artifacts
```

These files are not the first public review surface.

They may contain old formatting, malformed raw rows, literal escaped newlines, mixed report text, or non-JSON payloads stored with `.json` / `.jsonl` extensions.

That does not automatically invalidate the public entrypoint.

Correct audit reading:

```text
legacy artifact issue != public claim failure
raw file parse issue != active code failure
archive noise != current normative boundary
```

---

## JSON / JSONL audit rule

A `.json` or `.jsonl` file should be treated as normative only if it is part of the current public review path or active validation path.

Raw, generated, archived, or historical files should be classified before scoring.

Correct classification:

```text
normative JSON/JSONL      = must parse cleanly
active validation JSONL   = must parse cleanly
legacy/raw/generated data = classify before scoring
archive material          = non-normative unless explicitly referenced
```

Incorrect classification:

```text
every historical .jsonl parse issue = public repo failure
every raw generated dump = active validation failure
archive formatting issue = current claim failure
```

---

## Claim audit rule

The public claim audit must distinguish between forbidden claims and negated forbidden claims.

For example, this is a forbidden claim:

```text
OMNIA is a truth oracle.
```

But this is a boundary statement:

```text
OMNIA is not a truth oracle.
```

And this is also a boundary statement:

```text
invariance != truth oracle
```

An audit must not score negated boundary language as if it were a positive claim.

Correct audit reading:

```text
"not a truth oracle" = boundary protection
"does not prove semantic truth" = boundary protection
"forbidden claim: proves semantic truth" = boundary protection
```

Incorrect audit reading:

```text
any occurrence of "truth oracle" = unbounded claim
any occurrence of "safety certificate" = unbounded claim
any occurrence of "absolute truth" = unbounded claim
```

---

## Current public claim

The strongest bounded public claim for this hub is:

```text
lon-mirror is the public hub for a bounded structural diagnostic and validation ecosystem.
```

This claim remains inside:

```text
measurement != inference != decision
```

---

## What this repository does not claim

This repository does not claim:

```text
truth oracle
semantic judge
final decision engine
safety certificate
deployment approval system
replacement for external review
replacement for domain expertise
replacement for scientific validation
```

---

## Correct audit order

A reviewer should audit in this order:

```text
1. README.md
2. docs/REVIEWER_60_SECOND_START.md
3. docs/ECOSYSTEM_ONE_PAGE.md
4. docs/PUBLIC_REVIEW_PATH.md
5. docs/PUBLIC_CLAIM_BOUNDARY_AUDIT.md
6. docs/ECOSYSTEM_LINK_AUDIT.md
7. docs/AUDIT_SCOPE.md
8. tests/
9. active examples
10. legacy/archive/raw artifacts only after classification
```

---

## Practical consequence

The next public step is not more architecture.

The next public step is:

```text
one concrete external-facing demonstration
Surface PASS -> Structural RISK
measurement != inference != decision
```

The repository already contains enough supporting infrastructure for that step.

---

## Final boundary

```text
public audit scope != scientific proof
traceability != truth
tests pass != semantic correctness
measurement != inference != decision
```
