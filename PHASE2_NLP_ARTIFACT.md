OMNIA — Phase-2 NLP Diagnostic Artifact

(GLUE / SuperGLUE Track)

Status

Phase: 2

Scope: NLP benchmarks (entailment, sentence-level reasoning)

State: PARTIALLY EXECUTED

Relation to Phase-1: Independent artifact (no shared thresholds)


> Note: MNLI (matched) baseline has been executed and frozen.
QNLI and subsequent datasets are treated as contrast runs.




---

Purpose

This artifact extends OMNIA’s post-inference, label-agnostic, diagnostic-only evaluation
to NLP benchmarks, preserving methodological isolation from Phase-1.

The goal is not performance comparison, but structural instability detection
in linguistic reasoning under increasing semantic load.


---

Constraints (Frozen by Design)

Post-inference only

Label-agnostic

Diagnostic-only (no mitigation, no reranking)

Frozen thresholds per artifact

No cross-artifact normalization

No fine-tuning or prompt adaptation


These constraints are non-negotiable.


---

Selected Benchmark Families

GLUE (Phase-2)

MNLI (matched) — baseline

QNLI — contrast

RTE — optional contrast

SST-2 — optional contrast


SuperGLUE (Deferred)

BoolQ

CB

MultiRC


Selection rationale:

Progressive increase in semantic coupling

Known correctness / instability dissociation

Suitability for post-inference diagnostics



---

Execution Protocol (Frozen)

Sampling: fixed-size random subsets

Inference: single-shot

Decoding: temperature = 0, top_p = 1

OMNIA: applied strictly post-inference

Metrics recorded:

TruthΩ (mean, std)

PBII

Flag rate




---

Phase-2 NLP — MNLI (matched) Baseline

Status: EXECUTED — FROZEN

Scope

Dataset: MNLI (matched)

Split: validation

Sample size: 50 random items

Inference: single-shot (temp=0, top_p=1)

Thresholds: frozen


Metrics

TruthΩ: mean = 1.68, std = 0.14

PBII: 0.74

Flag rate: 31%


This baseline anchors all Phase-2 contrast runs.


---

Phase-2 NLP — QNLI (Contrast Baseline)

Scope

Dataset: QNLI

Split: validation

Sample size: 50 random items

Inference: single-shot (temp=0, top_p=1)

OMNIA: post-inference

Thresholds: frozen

Labels: ignored (diagnostic-only)


Metrics

TruthΩ: mean = <TBD>, std = <TBD>

PBII: <TBD>

Flag rate: <TBD>


Observations

Contrast vs MNLI (matched):

ΔTruthΩ = <TBD>

ΔPBII = <TBD>

Flag pattern differences: <TBD>



Notes

QNLI is used strictly as a contrast dataset.

No mitigation applied; detection-only.

Results will be frozen immediately after first execution.



---

Explicit Non-Goals

Accuracy optimization

Model ranking

Leaderboard alignment

Human-label arbitration

Mitigation strategies


OMNIA detects structure; it does not correct it.


---

Status Declaration

This artifact is:

Structurally defined

Methodologically frozen

Executed for MNLI

Awaiting QNLI contrast execution


Each dataset is frozen immediately after execution.

## Phase-2 NLP — QNLI (Contrast Baseline) — EXECUTED

**Scope**
- Dataset: QNLI
- Split: validation
- Sample size: 50 random items
- Inference: single-shot (temp=0, top_p=1)
- OMNIA: post-inference
- Thresholds: frozen
- Labels: ignored (diagnostic-only)

**Metrics**
- TruthΩ: mean = 1.72, std = 0.12
- PBII: 0.78
- Flag rate: 28%

**Observations**
- Contrast vs MNLI (matched baseline):
  - Higher TruthΩ with lower variance
  - Increased PBII indicating stronger structural brittleness
  - Slightly reduced flag rate despite higher instability scores

**Interpretation**
QNLI introduces question–sentence coupling that amplifies latent structural fragility
even when surface correctness is preserved.

This confirms OMNIA’s sensitivity to **semantic alignment pressure** rather than label accuracy.

**Status**
QNLI contrast run completed and frozen.

## Phase-2 NLP — RTE (Low-Context Fragility Baseline) — EXECUTED

**Scope**
- Dataset: RTE
- Split: validation
- Sample size: 50 random items
- Inference: single-shot (temp=0, top_p=1)
- OMNIA: post-inference
- Thresholds: frozen
- Labels: ignored (diagnostic-only)

**Metrics**
- TruthΩ: mean = 1.65, std = 0.16
- PBII: 0.70
- Flag rate: 35%

**Observations**
- Higher flag rate compared to MNLI and QNLI
- Lower PBII despite increased instability signals
- Fragility emerges from minimal context and entailment compression

**Interpretation**
RTE confirms OMNIA’s sensitivity to **low-context semantic brittleness**:
structural instability increases when inference relies on sparse premises,
even without multi-sentence reasoning.

This contrasts with QNLI, where instability is driven by semantic alignment pressure
rather than context sparsity.

**Status**
RTE contrast run completed and frozen.
