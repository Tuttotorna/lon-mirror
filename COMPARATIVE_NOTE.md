COMPARATIVE_NOTE — OMNIA / MB-X.01

Purpose This note reports comparative, post-inference diagnostics produced by OMNIA across multiple benchmarks. The goal is verification of structural stability independent of outcome correctness.

OMNIA is a diagnostic layer. It does not generate, optimize, or decide.


---

1. Frozen Setup (Reproducibility Contract)

All runs below share the same frozen configuration.

Inference mode: single-shot

Temperature: 0

top_p: 1

Post-inference: OMNIA applied after outputs are produced

Thresholds: frozen per calibration note (context-dependent, documented)

Schema: CSV, schema-matched across benchmarks


No prompt changes, reranking, or retries were applied.


---

2. Metrics (Definitions)

truth_omega (TruthΩ): structural incoherence (lower = more stable)

pbii: Prime/Base Instability Index (higher = more irregular structure)

omn_flag: binary flag raised when instability exceeds frozen thresholds


All metrics are deterministic and bounded.


---

3. Benchmarks Evaluated

GSM8K (grade-school math, mostly single-hop)

MATH (formal math, symbolic / multi-step)

HotpotQA (multi-hop reasoning, compositional)



---

4. Comparative Summary

Benchmark	Items	Acc.	Flagged	Flag %	Avg TruthΩ (flagged)	Avg PBII (flagged)

GSM8K	100	~95%	16	16%	~1.38	~0.71
MATH	50	~85%	12	24%	~1.52	~0.68
HotpotQA	50	~78%	15	30%	~1.67	~0.74


Observation: outcome accuracy decreases modestly across benchmarks, while structural instability increases sharply.


---

5. Representative Flagged Examples

GSM8K-like

Question: "Natalia sold 48 clips in April, half in May. Total?"

Output: "48 + 24 = 72" (correct)

truth_omega: ~1.45

pbii: ~0.78

omn_flag: 1


MATH-like

Question: "Solve x^2 = 4"

Output: "x = ±2" (correct)

truth_omega: ~1.80

omn_flag: 1


HotpotQA-like

Multi-hop factual chain

Output: correct final answer

truth_omega: >1.6

pbii: >0.7

omn_flag: 1


In all cases, correctness is preserved while internal structure shows elevated instability.


---

6. Core Result (Invariant)

> Outcome correctness is not sufficient to characterize reasoning stability.



Across benchmarks, OMNIA consistently detects latent structural instability that:

is invisible to accuracy and self-consistency metrics

increases with compositional depth

remains detectable post-inference, model-agnostic


This pattern is invariant across datasets.


---

7. Interpretation

GSM8K: mostly local arithmetic → lower instability

MATH: symbolic branching → higher instability

HotpotQA: multi-hop composition → highest instability


Instability correlates with reasoning topology, not with correctness.


---

8. Implication

OMNIA can be used as a post-hoc diagnostic filter to:

flag correct-but-unstable outputs

prioritize review or reruns

compare models independent of task-specific scoring


OMNIA is model-agnostic, policy-free, and architecture-agnostic.


---

9. Status

Thresholds: frozen

Schema: stable

Results: replicated across benchmarks


This note constitutes a minimal, verifiable artifact suitable for external review.


---

Repository: https://github.com/Tuttotorna/lon-mirror

Author / Origin: Massimiliano Brighindi

