# OMNIA — Post-Inference Diagnostic Integration

## Scope
OMNIA is applied strictly post-hoc, after model inference.
It does not modify outputs, prompts, or weights.

## Minimal Pipeline

1. Model generates outputs (any LLM, any decoding)
2. Outputs are frozen
3. OMNIA computes structural diagnostics
4. Items are flagged for review / rerun (no correction)

## GSM8K Example

Input:
- question
- model_output
- correctness label (optional)

Computed:
- truth_omega
- pbii
- omn_flag

Flag condition (example):
truth_omega > τ OR pbii > π

## Why this matters
Accuracy and self-consistency remain blind to structurally unstable
but locally correct outputs.

OMNIA detects this failure mode deterministically.

## Non-goals
- No re-ranking
- No reward shaping
- No alignment logic