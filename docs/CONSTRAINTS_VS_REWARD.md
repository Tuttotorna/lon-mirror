# Constraints vs Reward: why hallucinations persist

## Core claim
Most "hallucinations" are not a pure knowledge failure.
They are a **control failure**: the system is allowed to take trajectories that should be **impossible**, then later we try to correct them with reward, sampling tricks, or post-hoc filters.

- **Reward / RL** = correction **after** a wrong trajectory exists.
- **Hard constraints** = removal of wrong trajectories **before** they can exist.

If the action exists in the state-space, the model will sometimes take it.

## Two different problems (often mixed)
### A) Uncertainty hallucination ("filling the gap")
The model does not know, but must output something.
Mitigations: abstention, calibration, verification, retrieval, self-consistency.

### B) Training-induced hallucination ("confident wrongness")
The model outputs a coherent but wrong answer because the objective rewards fluency, plausibility, and local likelihood.
Mitigations require **structural control**, not just more data.

These are different failure modes. Treating both as "lack of knowledge" is an error.

## Why "more memory" is not sufficient
Giving the system a bigger internal world model or longer context window does not remove invalid trajectories.
It increases capacity, but **capacity is not control**.

Without constraints, bigger models can hallucinate more convincingly.

## Where constraints should live (three layers)
A practical architecture separates ergonomics from enforcement:

1) **Language layer (ergonomics + early proof)**
   - Types, shape annotations, restricted constructs.
   - Helpful, but not a security boundary.

2) **Bytecode / IR layer (hard wall / contract)**
   - The program representation itself encodes invariants.
   - A verifier rejects invalid programs before execution.
   - This is the first *real* boundary.

3) **Runtime / VM layer (determinism + sandbox)**
   - Deterministic execution (no hidden nondeterminism).
   - No uncontrolled IO / syscalls.
   - Minimal trace emission for diagnostics.

Key point: **do not trust the compiler**. Trust the verifiable contract.

## Where OMNIA fits (and where it does NOT)
OMNIA is not a constraint system.
OMNIA is a **diagnostic measurement layer**.

- It does not decide.
- It does not enforce.
- It does not optimize.

OMNIA measures **structural instability** and **invariance breakdown** under transformation.
It is what tells you *where* constraints are needed and whether they worked.

In other words:

- **Constraints** remove bad trajectories.
- **OMNIA** measures residual instability and regressions, post-hoc, deterministically.

## Practical integration pattern
1) Generate outputs / traces.
2) Run OMNIA to compute invariance + instability metrics (Ω-total + lens breakdown).
3) Use the measured failure surface to place/adjust constraints (bytecode verifier / VM rules).
4) Re-run OMNIA to verify structural improvement under identical test transforms.

This prevents the common trap:
"Optimization accelerates failure if the objective is wrong."

## One-line takeaway
Reward can discourage bad moves.
Only constraints can make bad moves impossible.
OMNIA measures whether your constraints actually removed instability.