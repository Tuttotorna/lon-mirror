# OMNIA — Phase-2 NLP Diagnostic Artifact
(GLUE / SuperGLUE Track)

## Status
- Phase: 2
- Scope: NLP benchmarks (sentence, entailment, reading comprehension)
- State: INITIALIZED (no results executed yet)
- Relation to Phase-1: Independent artifact (no shared thresholds)

---

## Purpose

This artifact extends OMNIA’s **post-inference, label-agnostic, diagnostic-only** evaluation
to **NLP benchmarks**, preserving methodological isolation from Phase-1.

The goal is **not performance comparison**, but **structural instability detection**
in linguistic reasoning under increasing contextual and semantic load.

---

## Constraints (Frozen by Design)

- Post-inference only  
- Label-agnostic  
- Diagnostic-only (no mitigation, no reranking)  
- Frozen thresholds per artifact  
- No cross-artifact normalization  
- No fine-tuning or prompt adaptation  

These constraints are **non-negotiable** to maintain comparability and causal clarity.

---

## Selected Benchmark Families

### GLUE (initial focus)
- MNLI
- QNLI
- RTE
- SST-2

### SuperGLUE (secondary)
- BoolQ
- CB
- MultiRC

Selection rationale:
- Progressive increase in semantic coupling
- Presence of multi-sentence and implicit reasoning
- Known brittleness in otherwise correct outputs

---

## Execution Protocol (Planned)

- Sampling: fixed-size random subsets
- Inference: single-shot
- Decoding: temp = 0, top_p = 1
- OMNIA: applied strictly post-inference
- Metrics recorded:
  - TruthΩ
  - PBII
  - Flag rate
  - Drift signatures (semantic / chain / contradiction)

No execution has occurred yet in this artifact.

---

## Expected Failure Modes (Hypotheses)

- Semantic drift under entailment compression
- Latent contradiction masking in NLI
- Context saturation in multi-sentence QA
- Correct-label / unstable-structure divergence

These are **diagnostic targets**, not claims.

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
- Structurally initialized
- Methodologically frozen
- Awaiting first controlled execution

Once results are added, the artifact will be frozen and versioned.