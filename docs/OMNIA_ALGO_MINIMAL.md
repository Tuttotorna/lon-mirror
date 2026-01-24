# OMNIA_ALGO_MINIMAL — The OMNIA Algorithm in One Page (Non-Semantic)

**OMNIA (MB-X.01)** is a post-hoc structural measurement engine.
This document defines the **minimal algorithmic kernel**: how OMNIA measures invariance, saturation, irreversibility, and STOP.

**Author:** Massimiliano Brighindi  
**Project:** OMNIA / MB-X.01  
**Repo:** https://github.com/Tuttotorna/lon-mirror

---

## 0) Purpose

Given a representation `x` (text, tokens, numbers, signals),
OMNIA measures **what survives independent, non-semantic transformations**.

Output is measurement, not explanation.

---

## 1) Minimal Inputs

- `x`: representation
- `T = {t1..tk}`: independent transformations (no semantics)
- `φ(.)`: deterministic structural feature extractor
- `d(.)`: distance on features (robust, deterministic)
- thresholds: `sei_eps`, `iri_eps`, `omega_spread_eps`

---

## 2) Transformations (Non-Semantic)

Transformations must change encoding while preserving intended structure.

Typical families:
- **Permutation**: reorder under constraints
- **Compression / quantization**: lossy but controlled
- **BASE / modular**: residue projections, multi-base structure (Omniabase)
- **Perturbation**: small noise, dropout, jitter
- **Slicing**: windowing, chunking, reassembly

Independence rule:
Each `ti` must be interpretable as a separate “observer lens”, not a variant of the same lens.

---

## 3) Feature Extraction φ(x)

`φ(x)` must be deterministic and non-semantic.

Examples:
- modular residue vectors, cycle signatures
- run-length / histogram signatures
- compression-length deltas (LZ-like proxies)
- autocorrelation / spectrum (signals)
- token shape statistics (counts, transition matrices)

`φ` must be:
- import-safe
- reproducible
- stable under repeated runs

---

## 4) Ω — Structural Coherence Score

Compute features on original and transformed representations:

- `f0 = φ(x)`
- `fi = φ(ti(x))` for i=1..k

Compute distances:
- `di = d(f0, fi)`

Define Ω (one robust option):
- `Ω = 1 - median(di)`  with Ω ∈ [0,1] by construction if d is normalized.

Interpretation:
- High Ω = structure persists under transformations.
- Low Ω = representation is fragile under re-encoding.

Ω is **not correctness** and **not semantic truth**.

---

## 5) Ω̂ — Residual Invariance (Omega-set Estimator)

Ω̂ is deduced as a robust residue across transformations:

- compute `Ωi` per lens (or per transform family)
- estimate:
  - `Ω̂ = median(Ωi)`
  - `spread = MAD(Ωi)` (or IQR)

Interpretation:
- Ω̂ = “what remains invariant after removing representation”
- spread = how observer-sensitive the measurement is

---

## 6) SEI — Saturation / Exhaustion Index

SEI measures marginal structural yield versus applied cost:

Let `C` be a monotone “cost” axis:
- number of transforms
- transform strength
- complexity of lens stack

For two successive configurations A → B:
- `ΔΩ̂ = Ω̂_B - Ω̂_A`
- `ΔC = C_B - C_A`

Define:
- `SEI = ΔΩ̂ / ΔC`

Saturation condition:
- `SEI → 0` means additional computation yields no new admissible structure.

---

## 7) IRI — Irreversibility / Hysteresis Index

IRI detects irreversible loss under a cycle:

Construct a cycle:
- `A = x`
- `B = u(A)` (a transform or lens application)
- `A' = v(B)` (attempt to return / re-encode back)

Compute:
- `IRI = d(φ(A), φ(A'))`

IRI ≥ 0 by construction.
- IRI ~ 0: reversible
- IRI > 0: hysteresis / irreversible structural loss

---

## 8) OMNIA-LIMIT — STOP Rule (Minimal)

STOP when structural extraction is saturated and boundary is stable:

Stop if all hold:
- `SEI <= sei_eps`
- `IRI >= iri_eps`
- `spread <= omega_spread_eps`

This is a **boundary declaration**, not a decision or policy.

OMNIA-LIMIT does not retry, optimize, escalate, or explain.
It outputs STOP + boundary certificate fields.

---

## 9) Minimal Output Schema (Human-Readable)

Return:

- `Ω`
- `Ω̂`
- `spread` (MAD/IQR)
- `SEI`
- `IRI`
- `STOP` boolean
- `reason` string (structural only, no narrative)
- optional: per-transform distances `di`

---

## 10) Toy Example A — Text (Non-Semantic)

- x: a generated string or model output
- transforms: token permutation within punctuation blocks, compression, slicing
- features: n-gram transition matrices, compression length proxy
- d: L1 distance over normalized feature vectors
- STOP: SEI≈0 + IRI>0 + stable Ω̂

Goal:
Measure whether the text’s structure is robust under re-encoding.

---

## 11) Toy Example B — Signals (PAM4 intuition)

- x: sampled waveform or symbol sequence
- transforms: quantization levels, jitter, resampling, phase rotation
- features: eye-opening proxies, spectrum shape, transition statistics
- d: distance on normalized feature descriptors

Interpretation:
As symbol density increases (multi-level signaling),
Ω̂ drops and IRI rises when recovery becomes structurally irreversible.

OMNIA is the measurement layer that detects the boundary.

---

## 12) Non-Negotiable Boundary

OMNIA measures. It does not decide.

Measurement ≠ cognition ≠ decision

---

## 13) Where This File Sits

This is documentation only.
It defines the minimal algorithmic kernel without adding modules to core.

Suggested location:
`docs/OMNIA_ALGO_MINIMAL.md`