OMNIA

Measuring Structural Invariance, Saturation, and Observer-Induced Perturbation

A Post-Hoc Diagnostic Framework for Scientific and Computational Systems

Massimiliano Brighindi
MB-X.01
GitHub: https://github.com/Tuttotorna/lon-mirror


---

Abstract

We introduce OMNIA, a post-hoc structural measurement framework designed to evaluate representations independently of semantics, models, or interpretation.
OMNIA measures what remains invariant under transformation, when further continuation becomes structurally impossible, and how much structure is lost when an observer is introduced.

Unlike predictive models or explanatory theories, OMNIA does not generate hypotheses, interpretations, or decisions.
It provides quantitative diagnostics for invariance, saturation, irreversibility, compatibility, and observer-induced perturbation.

The framework formalizes three recurring but previously unmeasured phenomena:

1. Structural saturation: the exhaustion of extractable invariants under continued transformation.


2. Irreversibility: loss of recoverable structure under cyclic operations.


3. Observer perturbation: degradation of invariants caused by interpretative or perspective-introducing transformations.



OMNIA is intended as a filter, not a theory: it classifies when and where continued reasoning, modeling, or interpretation is structurally admissible.


---

1. Motivation

Across multiple domains—physics, machine learning, neuroscience, and complex systems—research stagnation often occurs without explicit failure.

Typical symptoms include:

increasing theoretical elaboration without new constraints,

proliferation of interpretations without growth in invariants,

growing dependence on narrative coherence rather than structural robustness.


Existing methodologies lack a formal mechanism to answer a basic question:

> Is further work in this representational domain structurally admissible?



OMNIA addresses this gap by shifting focus from what a representation means to what survives when representation is changed.


---

2. Core Principle

> Structural truth is what survives the removal of representation.



OMNIA assumes that any claim, model, or output can be subjected to independent, meaning-blind transformations.
If structural invariants persist, the representation carries non-trivial structure.
If invariants collapse, continuation relies increasingly on interpretation.

No semantic ground truth is required.


---

3. Measurement Philosophy

OMNIA is explicitly post-hoc and model-agnostic.

It does not:

evaluate correctness,

validate truth,

optimize performance,

infer intent or meaning.


It only measures structural behavior under transformation.

All quantities are:

transformation-relative,

representation-level,

non-semantic.



---

4. Structural Quantities

4.1 Ω — Structural Coherence

Ω measures internal consistency of a representation under a given lens.

It applies to:

numeric sequences,

time series,

symbolic strings,

token streams,

model outputs.


Ω does not encode correctness—only structural regularity.


---

4.2 Ω̂ — Omega-set (Residual Invariance)

Given multiple Ω values computed under independent transformations, OMNIA defines:

Ω̂ as the robust center (median),

dispersion via MAD,

invariance as an inverse dispersion score.


Ω̂ represents the structural residue that survives representation change.


---

4.3 SEI — Saturation / Exhaustion Index

SEI measures marginal structural yield:

SEI = ΔΩ / ΔC

where C is a monotonic cost (depth, steps, tokens, transformations).

Interpretation:

SEI > 0 → structure still extractable,

SEI ≈ 0 → saturation,

SEI < 0 → structural degradation.


SEI identifies when continuation no longer produces new structure.


---

4.4 IRI — Irreversibility Index

IRI quantifies structural hysteresis:

A → B → A′
IRI = max(0, Ω(A) − Ω(A′))

IRI > 0 indicates irreversible structural loss, even if surface similarity appears intact.


---

4.5 OPI — Observer Perturbation Index

OPI measures the structural cost of introducing an observer.

An observer is defined operationally as any transformation that:

introduces asymmetry,

privileges a perspective,

reduces invariance.


OPI = Ω_ap − Ω_obs

where:

Ω_ap = aperspective invariance (no observer),

Ω_obs = invariance after observer-induced transformation.


OPI ≈ 0 indicates neutral observation.
OPI > 0 indicates structural damage due to interpretation.

OPI does not model consciousness or intent—only perturbation.


---

5. Aperspective Invariance

OMNIA introduces aperspective invariance: invariants that persist without any privileged point of view.

This isolates structures that:

exist prior to interpretation,

are non-experiential for human cognition,

remain stable across representation changes.


Aperspective invariance serves as the baseline against which observer effects are measured.


---

6. Structural Compatibility (SCI) and Zones

OMNIA evaluates whether multiple structural measurements can coexist without contradiction.

SCI operates on OMNIA outputs, not raw data.

SCI ∈ [0,1] is mapped to structural zones:

STABLE

TENSE

FRAGILE

IMPOSSIBLE


Zones are diagnostic, not prescriptive.


---

7. OMNIA-LIMIT and Runtime Guard

When:

SEI ≈ 0,

IRI > 0,

Ω̂ stabilizes,


OMNIA declares OMNIA-LIMIT.

This is not a failure condition.
It is a directional signal: continuation in the current representational domain is structurally inadmissible.

A runtime guard converts SCI into a strict STOP / CONTINUE signal, without policy or optimization.


---

8. Implications

8.1 Separation of Laws and Theories

Laws correspond to high Ω̂, low OPI, low IRI.

Theories may persist beyond saturation via narrative extension.


OMNIA provides a formal way to distinguish them.


---

8.2 Observer Cost in Physics and Computation

OMNIA makes explicit a notion implicit in quantum mechanics and relativity:
the observer is not neutral, and the cost of observation can be measured structurally.


---

8.3 Research Triage

OMNIA enables classification of research trajectories into:

structurally fertile,

saturated,

narrative-dependent.


This supports domain shifts rather than forced continuation.


---

9. Scope and Limits

OMNIA does not:

replace theories,

adjudicate truth,

resolve interpretation debates.


It provides a structural admissibility filter.


---

10. Conclusion

OMNIA formalizes a simple but previously unmeasured idea:

> Not everything that can be interpreted should be continued.
And not everything that exists is structured for comprehension.



By measuring invariance, saturation, irreversibility, compatibility, and observer perturbation, OMNIA offers a way to change direction without inventing narrative.


---

References

All implementations and artifacts:
https://github.com/Tuttotorna/lon-mirror
