# Pre-Limit Inference Sensor — Frozen v1.0

MB-X.01 / OMNIA  
Author: Massimiliano Brighindi

## Scope
This repository contains a deterministic **pre-limit inference state sensor**.

It classifies the structural regime of an inference process **before** OMNIA-LIMIT saturation.

This module is a **sensor**:
- it does not interpret meaning
- it does not decide actions
- it does not learn
- it does not optimize

## Module location
`omnia/inference/`

## States (S1–S5)
- S1: RIGID_INVARIANCE
- S2: ELASTIC_INVARIANCE
- S3: META_STABLE
- S4: COHERENT_DRIFT
- S5: PRE_LIMIT_FRAGMENTATION

## Inputs (structural only)
The classifier consumes a `StructuralSignature`:
- omega
- omega_variance
- sei
- drift
- drift_vector
- order_sensitivity

## Output
Deterministic state classification via:
`classify_state(StructuralSignature) -> InferenceState`

Optional telemetry can be recorded through `InferenceTrajectory`.

## Boundary
OMNIA-LIMIT remains the termination layer.
This module does not override or modify OMNIA-LIMIT behavior.

## Status
Frozen v1.0 — interface stable.
Thresholds are deterministic and may be tuned in future versions, without changing the state space or module boundaries.