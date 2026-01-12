OMNIA / MB-X.01 — Logical Origin Node (L.O.N.)

OMNIA is a deterministic measurement engine for structural coherence, instability, and saturation
across numbers, time, causality, and token sequences.

It does not interpret meaning.
It does not make decisions.
It measures invariants and limits.

This repository (Tuttotorna/lon-mirror) is the canonical public mirror
of the MB-X.01 / OMNIA research line.


---

What OMNIA Is

OMNIA is a pure diagnostic layer.

Input

Numeric signals

Temporal series

Causal structures

Token sequences (e.g. model outputs)


Output

Structure-only, machine-readable metrics


Constraints

No semantic assumptions

No policy, intent, or alignment layer

Deterministic, bounded, reproducible

Post-hoc only (never in-loop)


Core principle

> Truth is what remains invariant under transformation.



OMNIA operates after signals exist, without influencing their generation,
selection, or interpretation.


---

What OMNIA Is Not

OMNIA is not:

a language model

a classifier

an optimizer

an agent

a decision system

a safety or alignment layer


OMNIA never chooses.
It only measures structure.


---

Core Metrics (Stable API)

All metrics are deterministic, bounded, and numerically stable.

Metric	Description

truth_omega	Structural incoherence measure (0 = perfect coherence)
co_plus	Inverse coherence score in 
score_plus	Composite score (coherence + information bias)
delta_coherence	Dispersion / instability proxy
kappa_alignment	Relative similarity between two signals
epsilon_drift	Relative temporal change


Implementation

omnia/metrics.py

The API is explicit, import-stable, deterministic, and globals-free.


---

Layer-1: Saturation / Exhaustion Index (SEI)

OMNIA includes a first-order diagnostic layer to measure marginal structural yield
under increasing computational cost.

SEI does not decide and does not stop execution.
It measures diminishing returns.

Purpose

SEI quantifies how much structural value is gained per unit of cost (tokens, latency, iterations, optional energy proxy).

It answers one question only:

> Is further computation still producing measurable structural benefit?



Conceptual Definition

SEI = (Δquality + Δuncertainty_reduction)
      ----------------------------------
      (tokens + latency + iterations + energy_proxy)

Properties

Rolling window (trend-based)

No fixed thresholds

No policy interpretation

Fully diagnostic


Implementation Artifacts

omnia/sei.py — SEI engine (Layer-1)

examples/sei_demo.py — synthetic demonstration

examples/sei_gsm8k_from_jsonl.py — real GSM8K run

examples/sei_gsm8k_uncertainty_from_jsonl.py — uncertainty-aware SEI


SEI prepares, but does not trigger, higher-order boundary reasoning.


---

Visual Diagnostic (SEI Trend)

A minimal visual diagnostic artifact is provided:

assets/diagnostics/sei_trend.png

The plot shows SEI vs iteration index, highlighting:

marginal structural yield

flattening or decline under increasing computation


There are:

no thresholds

no stop conditions

no decisions


The curve is evidence, not instruction.


---

Prime Base Instability Index (PBII)

PBII is a zero-shot, non-ML structural metric derived from OMNIA’s
multi-base instability analysis.

It separates prime numbers from composites without:

training

embeddings

heuristics

learned parameters


Verified Result

Dataset: integers 2–5000

Method: zero-shot, deterministic

Metric: ROC-AUC (polarity-corrected)

Result: AUC = 0.816


Interpretation

lower PBII → primes

higher PBII → composites


This separation emerges purely from base-instability structure.

Notebook:

PBII_benchmark_v0.3.ipynb


---

Differential Diagnostics (Non-redundancy Evidence)

OMNIA detects structural instability even when outcome-based metrics remain stable.

Representative GSM8K cases where:

the answer is correct

standard metrics (accuracy, self-consistency) are stable

OMNIA flags instability


item_id	correct	acc_stable	self_consistent	omn_flag	truth_omega	pbii

137	1	1	1	1	1.92	0.81
284	1	1	1	1	2.31	0.88


These cases are locally correct but structurally unstable.
Outcome-based metrics do not detect them; structure-based metrics do.


---

Architecture Overview

Signal (numbers / time / tokens / causality)
        ↓
+-------------------------------------------+
|              OMNIA LENSES                  |
|   BASE · TIME · CAUSA · TOKEN · LCR        |
+-------------------------------------------+
        ↓
+-------------------------------------------+
|              METRIC CORE                  |
|   TruthΩ · Co⁺ · Δ · κ · ε                |
+-------------------------------------------+
        ↓
+-------------------------------------------+
|          SEI (Layer-1, trend only)         |
|   Marginal Yield / Saturation Detection   |
+-------------------------------------------+
        ↓
+-------------------------------------------+
|              ICE ENVELOPE                 |
|   Impossibility & Confidence Envelope     |
+-------------------------------------------+

OMNIA outputs diagnostics, never judgments.


---

Reproducibility

This repository provides a fixed, reproducible execution path.

Real Benchmark Run (Colab)

Official notebook:

colab/OMNIA_REAL_RUN.ipynb

Execution steps:

1. Clone repository


2. Install fixed dependencies


3. Lock random seeds


4. Run real benchmarks


5. Produce machine-readable reports



Goal: verification, not exploration.


---

Recorded Benchmark Outputs (Closed Models)

Canonical benchmark outputs produced by fixed, reproducible runs:

results/closed_models/

Examples:

gpt4_metrics.jsonl

gpt4_metrics_omnia.jsonl


These files are machine-readable (.jsonl), deterministic,
and generated by OMNIA without semantic assumptions.


---

Tests

Invariant-based tests live in:

tests/test_metrics.py

They verify:

algebraic identities

monotonicity

edge cases

numerical stability

API contracts


Run locally:

pytest


---

Integration Philosophy

OMNIA is composable by design.

Separation of roles

OMNIA → measures structure

External systems → interpret, decide, optimize


Validated boundary:

OMNIA = geometry / invariants

Decision systems = policy / intent / judgment


This keeps OMNIA institution-agnostic and architecture-agnostic.


---

Architecture Context (Downstream, Non-required)

OMNIA acts as the measurement core for downstream boundary layers.

Aligned projects:

OMNIAMIND
Structural rigidity and boundary stability diagnostics
https://github.com/Tuttotorna/OMNIAMIND

OMNIA-LIMIT
Terminal boundary artifact certifying structural non-reducibility
https://github.com/Tuttotorna/omnia-limit

These systems consume OMNIA signals.
OMNIA itself remains independent and self-contained.


---

Repository Identity (Canonical)

Canonical repository:

https://github.com/Tuttotorna/lon-mirror

Project name:

OMNIA / MB-X.01

Author / Logical Origin Node:

Massimiliano Brighindi

There is no secondary mirror and no alternate repository.


---

Status

Metrics core: stable

SEI (Layer-1): active

Visual diagnostics: present

Tests: invariant-based

API: frozen

Research line: active


This repository is intended to be read by humans and machines.


---

License

MIT License
