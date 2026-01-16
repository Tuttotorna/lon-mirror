#OMNIA — Unified Structural Measurement Engine

Ω · SEI · IRI · Ω̂ · OMNIA-LIMIT
MB-X.01

Author: Massimiliano Brighindi


---

Overview

OMNIA is a post-hoc structural measurement engine.

It measures structural coherence and instability of representations under independent transformations.

OMNIA:

does not interpret meaning

does not decide

does not optimize

does not learn


OMNIA measures what remains invariant when representation changes.


---

Core Principle

> Structural truth is what survives the removal of representation.



OMNIA evaluates outputs by applying independent structural lenses and measuring:

invariance

drift

saturation

irreversibility


The result is a measured boundary, not a judgment.


---

The OMNIA Measurement Chain

OMNIA → Ω
Ω under transformations → Ω̂ (Omega-set)
ΔΩ / ΔC → SEI (Saturation)
A → B → A′ → IRI (Irreversibility)
SEI ≈ 0 and IRI > 0 → OMNIA-LIMIT (STOP)

Each step is measured, not inferred.


---

1. Ω — Structural Coherence Score

Ω is the aggregated structural score produced by OMNIA’s lenses.

It reflects internal consistency, not correctness.

Ω can be computed over:

numbers

sequences

time series

token streams

model outputs


Ω is model-agnostic and semantics-free.


---

2. Structural Lenses

BASE — Omniabase

Multi-base numeric structure analysis.

Measures:

digit entropy across bases

σ-symmetry

PBII (Prime Base Instability Index)

base-invariant signatures



---

TIME — Omniatempo

Temporal drift and regime instability.

Measures:

distribution shifts

short vs long window divergence

regime change score



---

CAUSA — Omniacausa

Lagged relational structure.

Measures:

cross-signal correlations across lags

dominant dependency edges



---

TOKEN

Structural instability in token sequences.

Pipeline:

token → integer proxy

PBII per token

z-score aggregation


Used for hallucination and chain fracture detection.


---

LCR — Logical Coherence Reduction

External coherence lens.

Combines:

factual consistency

numeric consistency

optional Ω contribution


Produces Ω_ext for audit and benchmarking.


---

3. Ω̂ — Omega-set (Residual Invariance)

Ω̂ formalizes the statement:

> “Ω is not assumed. Ω is deduced by subtraction.”



Given multiple Ω values obtained under independent transformations:

Ω̂ = robust center (median)

dispersion = MAD

invariance = 1 / (1 + MAD)


This estimates the structural residue that survives representation change.

Implemented in:

omnia/omega_set.py


---

4. SEI — Saturation / Exhaustion Index

SEI measures marginal structural yield.

Definition:

SEI(k) = ΔΩ / ΔC

Where:

Ω = structural score

C = monotonic cost (tokens, steps, depth, time)


Interpretation:

SEI > 0 → structure still extractable

SEI ≈ 0 → saturation

SEI < 0 → structural degradation


SEI is a trend, not a threshold.

Implemented in:

omnia/sei.py


---

5. IRI — Irreversibility / Hysteresis Index

IRI measures loss of recoverable structure.

Cycle:

A → B → A′

Definition:

IRI = max(0, Ω(A) − Ω(A′))

If output quality appears similar but Ω drops after returning,
irreversible structural damage has occurred.

IRI is not an error metric.

Implemented in:

omnia/iri.py


---

6. OMNIA-LIMIT — Epistemic Boundary

OMNIA-LIMIT declares a STOP condition, not a decision.

Triggered when:

SEI → 0 (no marginal structural gain)

IRI > 0 (irreversibility detected)

Ω̂ stable (no invariant left to extract)


Meaning:

> No further structure is extractable under current transformations.



OMNIA-LIMIT does not escalate, retry, or optimize.


---

7. Colab Notebooks

Canonical (Reproducible)

colab/OMNIA_REAL_RUN.ipynb

deterministic

seed-locked

audit reference


Exploratory

colab/OMNIA_DEMO_INSPECT.ipynb

non-deterministic

inspection only

no claims



---

8. Repository Structure

omnia/
  omniabase.py
  omniatempo.py
  omniacausa.py
  omniatoken.py
  omnia_totale.py
  sei.py
  iri.py
  omega_set.py

LCR/
  LCR_CORE_v0.1.py
  LCR_BENCHMARK_v0.1.py

colab/
  OMNIA_REAL_RUN.ipynb
  OMNIA_DEMO_INSPECT.ipynb

All modules are:

deterministic

standalone

import-safe



---

9. What OMNIA Is Not

Not a model

Not an evaluator

Not a policy

Not a decision system

Not a truth oracle


OMNIA is a measurement instrument.


---

10. License

MIT License.


---

11. Citation

Brighindi, M.
OMNIA — Unified Structural Measurement Engine (MB-X.01)
GitHub: https://github.com/Tuttotorna/lon-mirror


