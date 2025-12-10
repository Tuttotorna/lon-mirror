# LCR – Logical Coherence Reduction Benchmark (v0.1)

Author: Massimiliano Brighindi (MBX) + MBX IA  
Module type: add-on for OMNIA_TOTALE / Dual-Echo

---

## What this folder contains

- `LCR_CORE_v0.1.py`  
  Core utilities to fuse:
  - structural score (Ω_struct, e.g. OMNIA_TOTALE)
  - factual/numeric consistency (FACT_CHECK_ENGINE)
  into a single external metric Ω_ext for long reasoning chains.

- `LCR_BENCHMARK_v0.1.py`  
  Small, self-contained benchmark script that:
  - loads a JSONL file with model outputs
  - runs factual/numeric checks per sample
  - builds a confusion matrix (TP/FP/TN/FN)
  - reports detection rate, precision, FPR, accuracy

- `../data/lcr_samples.jsonl`  
  Tiny synthetic dataset (8 samples) to show how the pipeline works end-to-end.

---

## Expected JSONL format

`data/lcr_samples.jsonl` has **one JSON object per line** like:

```json
{"id": "s001", "is_correct": true,  "omega_struct": 0.62, "fact_consistency": 0.93, "numeric_consistency": 0.91}

Fields:

id: sample identifier

is_correct: ground-truth label (true = correct reasoning, false = hallucinated)

omega_struct: structural coherence score (e.g. OMNIA_TOTALE Ω)

fact_consistency: [0,1], factual consistency score

numeric_consistency: [0,1], numeric consistency score


xAI can replace the synthetic samples with real logs (e.g. GSM8K chains) without changing the code.


---

How to run

From the repo root:

python LCR/LCR_BENCHMARK_v0.1.py

Output (example):

TP, FP, TN, FN

detection_rate (recall on hallucinations)

precision

false_positive_rate

accuracy

mean fact_consistency / numeric_consistency / gold_match

mean fused Ω_ext


This makes LCR plug-and-play as a safety/quality layer on top of any model that exposes:

a structural score (Ω_struct)

factual/numeric checks

ground-truth labels for evaluation.
