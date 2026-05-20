# OMNIA — State Distance Patch v0.2.1

## Status

This document defines the v0.2.1 patch for OMNIA state distance.

The patch does not change the structural core of v0.2.

It introduces a pre-canonicalization layer before structural signature extraction.

Reason:

The v0.2 validation on the 36-pair synthetic set showed that the remaining residual failures are concentrated in input pre-ingestion and canonicalization, not primarily in the core state-distance logic.

Architectural boundary remains unchanged:

measurement != cognition != decision

OMNIA does not interpret meaning.
OMNIA does not decide.
OMNIA measures residual structural difference under admissible transformations.

---

## 1. Goal

Normalize superficial numeric and symbolic formatting before structural signature extraction, while preserving:

- order
- adjacency
- local structure
- token identity as far as possible

The patch must not introduce:

- permutation
- semantic reinterpretation
- structure-changing rewrite

---

## 2. New pipeline

The v0.2 pipeline implicitly allowed raw input to reach structural signature extraction.

The v0.2.1 pipeline becomes:

S_raw
-> precanonicalize_state(S_raw)
-> structural_signature_v0.2(S_clean)

This patch affects only pre-ingestion.

The Delta_Omega definition remains unchanged.

---

## 3. Pre-canonicalization principles

A valid pre-canonicalization must satisfy all of the following:

1. deterministic
2. reversible in intent when possible
3. structure-preserving
4. order-preserving
5. bounded
6. explicitly declared in protocol

If a preprocessing step changes structural regime, it is invalid.

---

## 4. Allowed pre-canonicalization operations

The following operations are allowed in v0.2.1.

### 4.1 Trim outer padding

Remove leading and trailing whitespace.

Example:

- "  text  " -> "text"

### 4.2 Normalize internal spacing

Collapse repeated spaces and linebreak-equivalent spacing when spacing is not itself the measured structure.

Examples:

- "line1  line2" -> "line1 line2"
- "line1\nline2" -> "line1 line2"

### 4.3 Normalize common neutral symbolic wrappers

Remove superficial wrappers that do not alter token order.

Examples:

- "{1,2,3}" -> "1,2,3"
- "(A,B,C)" -> "A,B,C"
- "[A][B][C]" must not be collapsed into reordered content

Wrapper removal is valid only if token order and adjacency remain intact.

### 4.4 Normalize list separators

Map equivalent superficial separators to a canonical separator.

Examples:

- ";" -> ","
- "|" -> ","

Only when separator normalization does not merge or reorder items.

### 4.5 Normalize case where case is not structurally relevant

Examples:

- "ID_045" -> "id_045"
- "ABC" -> "abc"

This is allowed only in protocols where case is not declared structurally meaningful.

### 4.6 Normalize common numeric surface forms

Allowed examples:

- "00123" -> "123"
- "1.000" -> "1"
- "100.50" -> "100.5"
- "$100.50" -> "100.5"

This applies only to token-local numeric formatting.

It must not change ordering or adjacency.

### 4.7 Normalize superficial connector variants

Examples:

- "_" and "-" may be mapped to one canonical connector
- only when connector identity is not itself the object of measurement

---

## 5. Forbidden pre-canonicalization operations

The following operations are forbidden:

- token reordering
- permutation
- sorting
- semantic expansion
- synonym substitution
- unit conversion
- numerical rescaling
- collapse of repeated tokens into counts
- deletion of interior content that changes adjacency

Examples of forbidden behavior:

- "1,2,3" -> "3,2,1"
- "ABABAB" -> "AAABBB"
- "100 cm" -> "1 m"
- "A B C" -> "ABC" if token boundary is structurally relevant

---

## 6. Minimal operational effect

The patch is expected to reduce residual failures caused by:

- leading zeros
- currency symbols
- superficial numeric formatting differences
- wrapper and separator noise
- non-structural padding

The patch is not intended to solve:

- true local swaps
- true regime changes
- true oscillatory versus uniform structure
- threshold calibration issues

---

## 7. Expected validation effect

After applying v0.2.1, the following cases are expected to improve first:

- eq_v1_001
- eq_v1_010

The following cases remain diagnostic and should not be automatically forced into pass:

- mv_v1_012
- br_v1_004
- br_v1_012

These are useful boundary cases for later calibration.

---

## 8. Minimal claim

v0.2.1 does not redefine dO.

It refines the ingestion layer so that dO is not distorted by superficial formatting artifacts before structural comparison begins.

This patch is an input-conditioning correction.
Not a new metric.
Not a new theory.
Not a threshold rewrite.