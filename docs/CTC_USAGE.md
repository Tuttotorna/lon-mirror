# CTC-1.0 — Usage and Operational Interpretation

**Project:** MB-X.01 / OMNIA  
**Module:** OMNIA ↔ TCO Convergence Certificate  
**Status:** Frozen (v1.0)

---

## Purpose

CTC-1.0 provides a **non-semantic convergence test** between:

- OMNIA (aperspective subtraction): what survives representation removal
- TCO (total collapse): what survives maximal perturbation

CTC does not explain results. It classifies **structural agreement** or **divergence**.

---

## Inputs

- A representation `x` (text, sequence, serialized output)
- A fixed aperspective baseline (no privileged observer)
- A fixed TCO collapse schedule

Consistency of baselines is mandatory.

---

## Outputs

CTC produces:

- **CTC score** ∈ [0,1]
- **Label** ∈ { CONVERGENT, WEAKLY_CONVERGENT, DIVERGENT }
- **CI (Collapse Index)** ∈ [0,1]
- **c\*** (collapse point, optional)
- Minimal diagnostics (counts and Ω values)

No semantic interpretation is included.

---

## Interpretation Rules (Operational)

### CONVERGENT
Conditions:
- CTC ≥ 0.70
- CI ≥ 0.70
- Sufficient aperspective residue size

Meaning:
- The same structural residue survives both subtraction and collapse.
- Indicates **robust, non-narrative structure**.

Action:
- Admissible to proceed structurally.
- No explanatory expansion required.

---

### WEAKLY_CONVERGENT
Conditions:
- 0.40 ≤ CTC < 0.70

Meaning:
- Partial overlap between residues.
- Structure may depend on representation or collapse path.

Action:
- Proceed only with representation change.
- Do not expand explanation.

---

### DIVERGENT
Conditions:
- CTC < 0.40

Meaning:
- Subtraction and collapse do not agree.
- Either:
  - the structure is representation-bound, or
  - collapse reveals a different residue class.

Action:
- STOP under current regime.
- Change domain or abandon line.

---

## What CTC Is Not

- Not a proof of truth
- Not a model selection criterion
- Not a physical law
- Not an explanation tool

CTC is a **structural admissibility test**.

---

## Notes on Reproducibility

- Baselines and schedules must be version-locked.
- Any change invalidates previous certificates.
- CTC results are comparable only under identical configurations.

---

## Minimal Example

See:
- `examples/convergence_demo.py`

---

## End