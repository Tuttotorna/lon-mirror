# Prime Regime Sensor (Gap Prediction) — Contract

This module is a **structural regime sensor** applied to prime sequences.
It is **not** a prime oracle.

It demonstrates a principle consistent with OMNIA:
> apparent unpredictability can be reframed as a **measurement-regime** problem.

## What it does

Given a prime list `p_n`, it builds a deterministic state:

**PrimeState = (Φ, S, T, τ)**

- **Φ(p)**: modular residue vector of `p` under multiple moduli (multi-base proxy)
- **S**: stability of recent gap distribution (inverse normalized entropy)
- **T**: drift magnitude between consecutive Φ (L1 / |Φ|)
- **τ**: structural time (increments only when drift exceeds a threshold)

Then it performs a deterministic **KNN** lookup in (Φ, S, T)-space
to propose the next gap `g_{n+1}`.

## Hard guardrails (STOP is success, not failure)

The predictor MUST refuse to output a gap when the local regime is not admissible.

Current STOP conditions:

- `T > T_max`  → regime drift too high
- `S < S_min`  → regime stability too low
- `confidence < C_min` → nearest-neighbor similarity insufficient

This is aligned with OMNIA-LIMIT philosophy:
**no retry, no escalation, no narrative compensation.**

## Determinism

The module is deterministic:
- fixed `mods`, `window`, `drift_theta`, `K`, and thresholds
- no randomness
- no learned weights
- output is reproducible given the same prime list and parameters

## Known limits (declared)

- Φ uses modular residues: it is a proxy, not “the domain of primes”.
- KNN is local: it can only reuse observed structural neighborhoods.
- High STOP rate is expected when the regime is unstable. This is correct behavior.

## Files

- `omnia/lenses/prime_regime.py`
  - `PrimeState`
  - `prime_state_from_primes(...)`

- `omnia/lenses/prime_gap_knn.py`
  - `predict_next_gap_knn(...)`
  - STOP/OK + reason + neighbors (diagnostic output)

- `examples/prime_gap_knn_demo.py`
  - self-contained demo (generates primes, builds states, prints last predictions)

## How to run

```bash
python examples/prime_gap_knn_demo.py