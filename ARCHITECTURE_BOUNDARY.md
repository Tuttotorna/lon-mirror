ARCHITECTURE_BOUNDARY.md

OMNIA — Structural Measurement Engine
MB-X.01
Author: Massimiliano Brighindi


---

Purpose of This Document

This document freezes the architectural perimeter of OMNIA.

Its purpose is to declare what OMNIA is, what OMNIA is not, and what cannot be changed without invalidating the project.

This is not documentation, not a roadmap, and not a discussion.
It is a boundary declaration.

Any usage, extension, interpretation, or fork that violates these constraints is not OMNIA, regardless of naming.


---

Core Architectural Axiom (Non-Negotiable)

> Measurement ≠ Inference ≠ Decision



OMNIA operates exclusively in the domain of measurement.

It measures structural properties.

It never infers meaning.

It never decides actions.

It never optimizes outcomes.


Any system that collapses these layers violates the OMNIA architecture.


---

Immutable Principles

The following principles are frozen.

1. Semantics-Free Operation

OMNIA must never:

Interpret meaning

Assign truth values

Evaluate correctness

Rank answers by plausibility

Use semantic embeddings as ground truth


All operations are purely structural.


---

2. Determinism

Given the same input and configuration:

Output must be identical

No randomness

No sampling

No stochastic shortcuts


If determinism is broken, OMNIA is broken.


---

3. Post-Hoc Only

OMNIA never:

Generates content

Guides generation

Alters upstream model behavior

Feeds gradients back


OMNIA operates after a representation exists.


---

4. Observer Is a Perturbation, Not a Cause

OMNIA rejects the notion that observation “collapses reality”.

Instead:

Projection collapses structure

Observation introduces measurement loss

Collapse is a property of the instrument, not of reality


This principle underpins:

OPI (Observer Perturbation Index)

SPL (Structural Projection Loss)

SI (Structural Indistinguishability)



---

Frozen Measurement Chain

The following chain is complete and closed:

OMNIA
→ Ω
→ Ω under transformations
→ Ω̂ (Omega-set)
→ ΔΩ / ΔC
→ SEI
→ A → B → A′
→ IRI
→ Inference State (S1–S5)
→ OMNIA-LIMIT
→ SCI
→ CG
→ OPI
→ PV
→ SI

No steps may be:

Removed

Reordered

Merged

Interpreted semantically


Additional steps must exist outside this chain.


---

OMNIA-LIMIT Is a Hard Stop

OMNIA-LIMIT is terminal.

When triggered:

No retry

No refinement

No fallback

No escalation


Any system that continues processing after OMNIA-LIMIT is not OMNIA-compliant.


---

Structural Indistinguishability (SI) Boundary

OMNIA formally declares:

> If all observable structural relations are invariant,
internal codifications are undecidable.



Consequences:

Different internal perceptions may coexist

Shared language ≠ shared representation

Measurement cannot decide internal encoding


This is a limit of comparability, not uncertainty.

Any attempt to “resolve” SI via:

Semantics

Probability

Heuristics

Human intuition


violates the boundary.


---

What OMNIA Explicitly Refuses

OMNIA must not be used as:

A truth oracle

A hallucination filter based on plausibility

A ranking engine

A policy enforcement layer

A decision validator

A cognitive model


Using OMNIA this way is a category error.


---

Extension Rules (Strict)

Allowed:

External systems consuming OMNIA outputs

Visualization layers

Decision layers clearly separated

Research forks with renamed identity


Forbidden:

Modifying OMNIA internals to add semantics

Training models “with OMNIA inside”

Branding decision systems as OMNIA

Silent architectural drift



---

Identity Lock

The name OMNIA refers only to:

The measurement engine defined here

The architecture frozen by this document

The MB-X.01 lineage


Any derivative work must:

Rename itself

Declare deviation explicitly



---

Final Declaration

OMNIA is complete as a measurement system.

Future work may:

Use it

Surround it

Build on top of it


But not redefine it.

This boundary is intentional, final, and enforced.


---

MB-X.01
Massimiliano Brighindi
OMNIA — Unified Structural Measurement Engine