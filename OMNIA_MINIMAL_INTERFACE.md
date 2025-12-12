# OMNIA — Minimal Interface (5-Minute Use)

OMNIA is **not a model**.  
OMNIA is a **structural diagnostic and gating engine**.

It measures instability, ambiguity, and impossibility across numbers, time, causality, language, and facts — and outputs a single actionable signal.

---

## What OMNIA does (in one sentence)

> Given an output (number / series / text), OMNIA tells you whether it is **structurally safe**, **ambiguous**, or **impossible**.

---

## Inputs (minimal)

OMNIA can be called with **any subset** of the following:

### 1. Numeric input
```python
n = 173

2. Time series

series = [ ... ]   # list of floats

3. Multi-channel signals

series_dict = {"s1": [...], "s2": [...], "s3": [...]}

4. Token / language input

tokens = ["The", "system", "deleted", "the", "user"]
token_numbers = [len(t) for t in tokens]  # or any proxy

5. External coherence (optional)

omega_ext = 0.72   # fact + numeric consistency (LCR)
ambiguity_score = 0.65


---

Core Call

from omnia.engine import run_omnia_totale
from ICE.OMNIA_ICE_v0_1 import ICEInput, ice_gate

result = run_omnia_totale(
    n=n,
    series=series,
    series_dict=series_dict,
    extra={
        "tokens": tokens,
        "token_numbers": token_numbers,
    }
)

ice = ice_gate(
    ICEInput(
        omega_total=result.omega_total,
        lens_scores=result.lens_scores,
        lens_metadata=result.lens_metadata,
        omega_ext=omega_ext,
        ambiguity_score=ambiguity_score,
    )
)


---

Output (what you actually use)

Structural score

result.omega_total

Per-lens contributions

result.lens_scores

Final gate decision

ice.status

Possible values:

PASS → structurally safe to output

ESCALATE → ambiguous, needs clarification or second pass

BLOCK → impossible (0% class), must not be output



---

Why this exists

OMNIA enforces one invariant:

> Zero-percent statements must never pass.



It does not decide truth.
It prevents impossible outputs and exposes ambiguity.


---

Typical use cases

LLM hallucination gating

Chain-of-thought auditing

Safety filters (pre- or post-generation)

Research diagnostics

Model-agnostic evaluation



---

Design principle

OMNIA is:

model-agnostic

non-narrative

composable

auditable

measurable


If you can produce an output, OMNIA can measure its structural integrity.


---

End of interface.