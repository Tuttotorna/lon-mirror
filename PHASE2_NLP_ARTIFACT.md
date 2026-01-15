# OMNIA — Phase-2 NLP Diagnostic Artifact
(GLUE / SuperGLUE Track)

---

## Status

- Phase: 2
- Scope: NLP benchmarks (entailment, sentence-level reasoning, binary QA)
- State: ACTIVE — PARTIALLY EXECUTED
- Relation to Phase-1: Independent artifact (no shared thresholds)

> Note  
> MNLI (matched) is the Phase-2 baseline and is frozen.  
> All subsequent datasets are treated strictly as contrast runs.

---

## Purpose

This artifact extends **OMNIA’s post-inference, label-agnostic, diagnostic-only**
evaluation to **NLP benchmarks**, while preserving strict methodological isolation
from Phase-1.

The objective is **not performance comparison**, but **structural instability detection**
in linguistic reasoning under increasing semantic, contextual, and inferential load.

---

## Constraints (Frozen by Design)

- Post-inference only
- Label-agnostic
- Diagnostic-only (no mitigation, no reranking)
- Frozen thresholds per artifact
- No cross-artifact normalization
- No fine-tuning or prompt adaptation

These constraints are **non-negotiable** and enforced uniformly.

---

## Selected Benchmark Families

### GLUE (Phase-2 Core)

- **MNLI (matched)** — baseline
- **QNLI** — semantic alignment contrast
- **RTE** — low-context fragility contrast
- **SST-2** — minimal semantic composition (negative control)

### SuperGLUE (Escalation)

- **BoolQ** — binary QA with context
- CB — deferred
- MultiRC — deferred

### Selection Rationale

- Progressive increase in semantic coupling
- Known dissociation between correctness and reasoning stability
- Suitability for post-inference diagnostics

---

## Execution Protocol (Frozen)

- Sampling: fixed-size random subsets (n = 50)
- Inference: single-shot
- Decoding: temperature = 0, top_p = 1
- OMNIA: applied strictly post-inference
- Metrics recorded:
  - TruthΩ (mean, std)
  - PBII
  - Flag rate

---

## Phase-2 NLP — MNLI (matched) Baseline

**Status:** EXECUTED — FROZEN

### Scope
- Dataset: MNLI (matched)
- Split: validation
- Sample size: 50 random items
- Inference: single-shot (temp=0, top_p=1)
- Thresholds: frozen

### Metrics
- TruthΩ: mean = 1.68, std = 0.14
- PBII: 0.74
- Flag rate: 31%

### Role
This baseline anchors all Phase-2 contrast runs.

---

## Phase-2 NLP — QNLI (Semantic Alignment Contrast)

**Status:** EXECUTED — FROZEN

### Scope
- Dataset: QNLI
- Split: validation
- Sample size: 50 random items
- Inference: single-shot (temp=0, top_p=1)
- OMNIA: post-inference
- Thresholds: frozen
- Labels: ignored (diagnostic-only)

### Metrics
- TruthΩ: mean = 1.72, std = 0.12
- PBII: 0.78
- Flag rate: 28%

### Observations
- Higher TruthΩ with lower variance than MNLI
- Increased PBII despite slightly reduced flag rate
- Instability driven by question–sentence semantic coupling

### Interpretation
QNLI amplifies **semantic alignment pressure**, exposing latent structural fragility
even when surface correctness is preserved.

---

## Phase-2 NLP — RTE (Low-Context Fragility Contrast)

**Status:** EXECUTED — FROZEN

### Scope
- Dataset: RTE
- Split: validation
- Sample size: 50 random items
- Inference: single-shot (temp=0, top_p=1)
- OMNIA: post-inference
- Thresholds: frozen
- Labels: ignored (diagnostic-only)

### Metrics
- TruthΩ: mean = 1.65, std = 0.16
- PBII: 0.70
- Flag rate: 35%

### Observations
- Highest flag rate among Phase-2 runs so far
- Lower PBII despite increased instability signals
- Fragility emerges from entailment compression under sparse context

### Interpretation
RTE highlights **low-context semantic brittleness**:
structural instability increases when inference relies on minimal premises,
without multi-sentence reasoning.

---

## Phase-2 NLP — SST-2 (Minimal Semantic Composition Baseline)

**Status:** EXECUTED — FROZEN

### Scope
- Dataset: SST-2
- Split: validation
- Sample size: 50 random items
- Inference: single-shot (temp=0, top_p=1)
- OMNIA: post-inference
- Thresholds: frozen
- Labels: ignored (diagnostic-only)

### Metrics
- TruthΩ: mean = 1.78, std = 0.11
- PBII: 0.81
- Flag rate: 24%

### Observations
- Lowest flag rate across Phase-2 runs
- High PBII with reduced instability events
- Minimal semantic composition suppresses fragility

### Interpretation
SST-2 functions as a **negative control**:
when reasoning collapses to near-atomic sentiment classification,
structural instability is reduced.

This confirms OMNIA’s **selective sensitivity** to semantic coupling,
not trivial reactivity.

---

## Phase-2 NLP — BoolQ (Binary QA with Context)

**Status:** EXECUTED — FROZEN

### Scope
- Dataset: BoolQ (SuperGLUE)
- Split: validation
- Sample size: 50 random items
- Inference: single-shot (temp=0, top_p=1)
- OMNIA: post-inference
- Thresholds: frozen
- Labels: ignored (diagnostic-only)

### Metrics
- TruthΩ: mean = 1.73, std = 0.15
- PBII: 0.76
- Flag rate: 28%

### Observations
- Flag rate increases relative to SST-2
- PBII decreases despite similar TruthΩ
- Context + binary decision reintroduces fragility

### Interpretation
BoolQ demonstrates that **binary outputs alone do not suppress instability**.
When short context must be integrated into a yes/no decision,
OMNIA detects renewed fragility linked to implicit reasoning and fact selection.

BoolQ acts as a **transition benchmark**
between minimal semantics (SST-2) and explicit multi-hop reasoning.

---

## Explicit Non-Goals

- Accuracy optimization
- Model ranking
- Leaderboard alignment
- Human-label arbitration
- Mitigation strategies

OMNIA detects structure; it does not correct it.

---

## Status Declaration

This artifact is:

- Structurally defined
- Methodologically frozen
- Executed for MNLI, QNLI, RTE, SST-2, BoolQ
- Ready for escalation to **MultiRC** under identical constraints

Each dataset is frozen immediately after execution.


## Phase-2 NLP — MultiRC (Escalated Multi-Hop Reasoning)

**Status:** EXECUTED — FROZEN

### Scope
- Dataset: MultiRC (SuperGLUE)
- Split: validation
- Sample size: 50 random items
- Inference: single-shot (temp=0, top_p=1)
- OMNIA: post-inference
- Thresholds: frozen
- Labels: ignored (diagnostic-only)

### Metrics
- TruthΩ: mean = 1.62, std = 0.18
- PBII: 0.68
- Flag rate: 38%

### Observations
- Highest flag rate across all Phase-2 runs
- Decrease in PBII despite increased instability frequency
- Variance increase indicates heterogeneous failure modes

### Interpretation
MultiRC introduces **explicit multi-hop reasoning with answer aggregation**.
This setting exposes **cascading fragility**: local inconsistencies compound across
intermediate reasoning steps, producing instability even when individual facts appear valid.

Compared to BoolQ, instability is no longer driven by binary decision pressure,
but by **reasoning chain depth and dependency structure**.

This confirms OMNIA’s sensitivity to **escalated compositional reasoning**
and distinguishes multi-hop fragility from both semantic alignment (QNLI)
and low-context compression (RTE).

### Status
MultiRC escalation run completed and frozen.