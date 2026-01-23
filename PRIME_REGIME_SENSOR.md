# PRIME REGIME SENSOR (Experimental)

This module applies OMNIA as a **non-semantic regime sensor** to prime sequences.

It does **not** claim to “predict primes”.
It demonstrates that parts of *apparent unpredictability* can be treated as a **measurement-regime** problem with explicit STOP conditions.

## What it does

Builds a deterministic state:

**PrimeState = (Φ, S, T, τ)**

- **Φ**: modular residue vector (multi-base structure)  
- **S**: gap-distribution stability (entropy-derived, normalized)  
- **T**: structural drift between consecutive Φ states  
- **τ**: structural time (increments only when drift exceeds a threshold)

Predicts the next gap **ĝ** using **KNN in state space** and enforces guardrails:

- **STOP** if drift is too high (**T > T_max**)  
- **STOP** if stability is too low (**S < S_min**)  
- **STOP** if neighbor similarity is insufficient (**confidence < C_min**)  

Output is:

- predicted gap (only if admissible)
- confidence
- STOP/OK reason
- neighbor diagnostics

## Files

- `omnia/lenses/prime_regime.py` — builds `PrimeState`
- `omnia/lenses/prime_gap_knn.py` — deterministic KNN gap prediction + guardrails
- `examples/prime_gap_knn_demo.py` — runnable demo

## Run

```bash
python examples/prime_gap_knn_demo.py