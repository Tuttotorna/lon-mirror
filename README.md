OMNIA — Unified Structural Measurement Engine

Ω · Ω̂ · SEI · IRI · OMNIA-LIMIT · τ · SCI · CG
MB-X.01

Author: Massimiliano Brighindi


---

Overview

OMNIA is a post-hoc structural measurement engine.

It measures structural coherence, instability, compatibility, and limits of representations under independent transformations.

OMNIA:

does not interpret meaning

does not decide

does not optimize

does not learn

does not explain


OMNIA measures what remains invariant when representation changes,
and where continuation becomes structurally impossible.


---

Core Principle

> Structural truth is what survives the removal of representation.



OMNIA evaluates outputs by applying independent structural lenses and measuring:

invariance

drift

saturation

irreversibility

compatibility


The result is a measured boundary, not a judgment.


---

The OMNIA Measurement Chain

OMNIA
→ Ω
→ Ω under transformations
→ Ω̂ (Omega-set)
→ ΔΩ / ΔC
→ SEI (Saturation)
→ A → B → A′
→ IRI (Irreversibility)
→ SEI ≈ 0 and IRI > 0
→ OMNIA-LIMIT (STOP)
→ SCI (Structural Compatibility)
→ CG (Runtime STOP / CONTINUE)

Each step is measured, never inferred.


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


Used for hallucination and chain-fracture detection.


---

LCR — Logical Coherence Reduction

External coherence lens.

Combines:

factual consistency

numeric consistency

optional Ω contribution


Produces Ω_ext for audit and benchmarking.


---

APERSPECTIVE — Aperspective Invariance

Measures structural invariants that persist under independent transformations
without introducing any privileged point of view.

Operates without:

observer assumptions

semantics

causality

narrative framing


Computes:

Ω_ap: fraction of structure surviving transformations

Residue: intersection of invariants after representation removal


Isolates structure that is real but non-experiential for human cognition.

Implementation:

omnia/lenses/aperspective_invariance.py



---

SATURATION — Saturation Invariance

Measures when further transformations no longer yield new structure.

Captures the limit of extractability, not content.

Outputs:

Ω curve vs cost

SEI curve

saturation point (c*)


Implementation:

omnia/lenses/saturation_invariance.py



---

IRREVERSIBILITY — Irreversibility Invariance

Measures structural hysteresis in cycles:

A → B → A′

Detects irrecoverable loss even when surface similarity appears intact.

Outputs:

Ω(A,B)

Ω(A,A′)

IRI certificate


Implementation:

omnia/lenses/irreversibility_invariance.py



---

REDUNDANCY — Redundancy Invariance

Measures how much destructive pressure is required before collapse.

Captures deep structural tessitura, not efficiency.

Outputs:

collapse cost

redundancy certificate


Implementation:

omnia/lenses/redundancy_invariance.py



---

DISTRIBUTION — Distribution Invariance

Measures non-local structural stability.

Ignores ordering and locality; evaluates only global distribution shape.

Captures structures that:

are nowhere locally

exist only statistically


Implementation:

omnia/lenses/distribution_invariance.py



---

NON-DECISION — Non-Decision Structure

Measures structures that remain stable without converging to a choice.

Captures coherence without optimization or selection.

Outputs:

path stability

dispersion across paths

non-decision certificate


Implementation:

omnia/lenses/nondecision_structure.py



---

3. Ω̂ — Omega-set (Residual Invariance)

Ω̂ formalizes the statement:

> Ω is not assumed. Ω is deduced by subtraction.



Given multiple Ω values under independent transformations:

Ω̂ = robust center (median)

dispersion = MAD

invariance = 1 / (1 + MAD)


Ω̂ estimates the structural residue that survives representation change.

Implementation:

omnia/omega_set.py



---

4. SEI — Saturation / Exhaustion Index

SEI measures marginal structural yield.

Definition:

SEI = ΔΩ / ΔC

Interpretation:

SEI > 0 → structure still extractable

SEI ≈ 0 → saturation

SEI < 0 → degradation


SEI is a trend, not a threshold.

Implementation:

omnia/sei.py



---

5. IRI — Irreversibility / Hysteresis Index

IRI measures loss of recoverable structure.

Definition:

IRI = max(0, Ω(A) − Ω(A′))

IRI is not an error metric.

Implementation:

omnia/iri.py



---

6. OMNIA-LIMIT — Epistemic Boundary

Declares a STOP condition, not a decision.

Triggered when:

SEI → 0

IRI > 0

Ω̂ stable


Meaning:

> No further structure is extractable under current transformations.



OMNIA-LIMIT does not retry or optimize.


---

7. Structural Time (τ)

τ is a structural time coordinate.

not wall-clock

not duration

advances only when structure changes


Used for:

non-synchronized comparisons

drift tracking

non-human coordination


Definition:

docs/OMNIA_TAU.md



---

8. Structural Compatibility — SCI

OMNIA includes a meta-layer that measures the compatibility between structural measurements.

SCI does not operate on data.
It operates on outputs of OMNIA lenses.

SCI answers one question only:

> Can these measured structures coexist without contradiction or loss?



SCI produces:

a compatibility score (SCI ∈ [0,1])

a structural zone classification

a non-narrative admissibility certificate


Implementation:

omnia/meta/structural_compatibility.py



---

9. Structural Zones

SCI outputs are mapped to Structural Zones:

STABLE — continuation admissible

TENSE — continuation admissible (edge)

FRAGILE — STOP

IMPOSSIBLE — STOP (non-negotiable)


Zones are not explanations and not recommendations.

Definition:

docs/STRUCTURAL_ZONES.md



---

10. Compatibility Guard — Runtime STOP Layer

OMNIA provides a runtime guard that converts SCI into a strict
STOP / CONTINUE signal.

The guard:

introduces no policy

introduces no semantics

introduces no optimization

introduces no retries


It only enforces structural admissibility.

Implementation:

omnia/runtime/compatibility_guard.py



---

11. Repository Structure

omnia/
  omniabase.py
  omniatempo.py
  omniacausa.py
  omniatoken.py
  omega_set.py
  sei.py
  iri.py

omnia/lenses/
  aperspective_invariance.py
  saturation_invariance.py
  irreversibility_invariance.py
  redundancy_invariance.py
  distribution_invariance.py
  nondecision_structure.py

omnia/meta/
  structural_compatibility.py

omnia/runtime/
  compatibility_guard.py

docs/
  STRUCTURAL_ZONES.md
  OMNIA_TAU.md

LCR/
  LCR_CORE_v0.1.py
  LCR_BENCHMARK_v0.1.py

All modules are:

deterministic

standalone

import-safe



---

12. What OMNIA Is Not

Not a model

Not an evaluator

Not a policy

Not a decision system

Not a truth oracle

Not a narrative framework


OMNIA is a measurement instrument.


---

13. License

MIT License.


---

14. Citation

Brighindi, M.
OMNIA — Unified Structural Measurement Engine (MB-X.01)
GitHub: https://github.com/Tuttotorna/lon-mirror
