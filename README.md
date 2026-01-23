OMNIA — Unified Structural Measurement Engine

Ω · Ω̂ · SEI · IRI · OMNIA-LIMIT · τ · SCI · CG · OPI · PV · INFERENCE · SI
MB-X.01

Author: Massimiliano Brighindi


---

Overview

OMNIA is a post-hoc structural measurement engine.

It measures structural coherence, instability, compatibility, limits, perturbations, inference regimes, and indistinguishability of representations under independent, non-semantic transformations.

OMNIA:

does not interpret meaning

does not decide

does not optimize

does not learn

does not explain


OMNIA measures:

what remains invariant when representation changes

where continuation becomes structurally impossible

how structure degrades under perturbation

which inferential regime is active before collapse

when different internal codifications are structurally undecidable


The output is measurement, never narrative.


---

Core Principle

> Structural truth is what survives the removal of representation.




---

Quickstart

Run the full test suite:

pytest tests/ -v

Run the prime regime demo:

python examples/prime_gap_knn_demo.py

The demo prints:

current prime

true next gap

predicted gap (if admissible)

confidence

STOP / OK reason



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
→ Inference State (S1–S5)
→ OMNIA-LIMIT (STOP)
→ SCI (Structural Compatibility)
→ CG (Runtime STOP / CONTINUE)
→ OPI (Observer Perturbation Index)
→ PV (Perturbation Vector)
→ SI (Structural Indistinguishability)

Each step is measured, never inferred.


---

1. Ω — Structural Coherence Score

Ω is the aggregated structural score produced by OMNIA lenses.

It reflects internal structural consistency, not correctness, usefulness, or semantic truth.

Ω is model-agnostic and semantics-free.


---

2. Structural Lenses

OMNIA operates through independent structural lenses:

BASE — Omniabase
Multi-base numeric structure analysis.

TIME — Omniatempo
Temporal drift and regime instability.

CAUSA — Omniacausa
Lagged relational and causal structure.

TOKEN
Structural instability in token sequences.

LCR — Logical Coherence Reduction
External coherence lens for audit and benchmarking.

All lenses are deterministic, composable, and non-semantic.


---

3. APERSPECTIVE — Aperspective Invariance

Measures invariants that persist under transformations without a privileged observer.

Isolates structure that exists independently of human perception or encoding.


---

4. Ω̂ — Omega-set (Residual Invariance)

Ω̂ is deduced by subtraction, not assumed.

It estimates the structural residue that survives representation change, using robust statistics over transformed Ω values.


---

5. SEI — Saturation / Exhaustion Index

Measures marginal structural yield:

SEI = ΔΩ / ΔC

SEI → 0 indicates structural saturation: further processing yields no new admissible structure.


---

6. IRI — Irreversibility / Hysteresis Index

Measures irrecoverable structural loss in cycles:

A → B → A′

IRI detects collapse even when surface similarity appears intact.

IRI ≥ 0 by construction.


---

7. Pre-Limit Inference States — INFERENCE

OMNIA includes a deterministic pre-limit inference sensor.

Inference is treated as a structural trajectory, not a decision.

States (S1–S5):

S1 — RIGID_INVARIANCE
Deterministic structural residue

S2 — ELASTIC_INVARIANCE
Deformable but coherent structure

S3 — META_STABLE
Order-sensitive, illusion-prone regime

S4 — COHERENT_DRIFT
Directional structural movement

S5 — PRE_LIMIT_FRAGMENTATION
Imminent collapse


Implementation:
omnia/inference/


---

8. OMNIA-LIMIT — Epistemic Boundary

Declares a STOP condition.

Triggered when:

SEI → 0

IRI > 0

Ω̂ is stable


OMNIA-LIMIT does not retry, optimize, or escalate.

It marks the end of admissible structural extraction.


---

9. Structural Time (τ)

τ is a non-human time coordinate.

It advances only when structure changes.

τ is not wall-clock time and not duration.


---

10. Structural Compatibility — SCI

Measures whether measured structures can coexist without contradiction or loss.

SCI operates on OMNIA outputs, not on raw data.


---

11. Compatibility Guard — CG

Converts SCI into a strict runtime STOP / CONTINUE signal.

CG introduces:

no policy

no semantics

no optimization


It enforces structural admissibility only.


---

12. Observer Perturbation Index — OPI

Measures the structural cost of introducing an observer or privileged basis:

OPI = Ω_ap − Ω_obs

OPI quantifies structural damage caused by enforced perspective, not intent or consciousness.


---

13. Perturbation Vector — PV

Formalizes how structure is destroyed, not why.

PV captures direction, composition, and intensity of structural loss.


---

14. Structural Indistinguishability — SI

OMNIA formalizes the following principle:

> If all observable structural relations are invariant,
internal codifications are undecidable.



Structural Indistinguishability (SI) measures whether two systems may differ internally while remaining structurally indistinguishable under all admissible relations.

Implications:

Different perceptual codings (e.g. color perception) are undecidable structurally

Shared language does not imply shared internal representation

Observation does not collapse states; projection does


SI formalizes a limit of comparability, not uncertainty.

Implementation:
omnia/meta/structural_indistinguishability.py


---

15. Experimental Module — Prime Regime Sensor (Gap Prediction)

OMNIA can be applied to prime number sequences as a non-semantic structural regime sensor.

This module:

builds a deterministic
PrimeState = (Φ, S, T, τ)

where
Φ = modular residue vector (multi-base structure)
S = gap-distribution stability
T = structural drift
τ = structural time

predicts the next prime gap using K-nearest neighbors in Φ-space

enforces STOP via structural guardrails (confidence, drift, stability)


This is not a prime oracle.

It demonstrates that apparent unpredictability can be treated as a measurement-regime problem, not a mystery.

Implementation:

omnia/lenses/prime_regime.py

omnia/lenses/prime_gap_knn.py


Demo:

examples/prime_gap_knn_demo.py



---

16. Repository Structure

omnia/
  inference/
  lenses/
  meta/
  runtime/
  omega_set.py
  sei.py
  iri.py

examples/
tests/
docs/

All modules are:

deterministic

standalone

import-safe



---

17. What OMNIA Is Not

OMNIA is not:

a model

an evaluator

a policy

a decision system

a truth oracle

a narrative framework


OMNIA is a measurement instrument.


---

18. License

MIT License.


---

19. Citation

Brighindi, M.
OMNIA — Unified Structural Measurement Engine (MB-X.01)
https://github.com/Tuttotorna/lon-mirror


