# OMNIA — Threshold Calibration (Diagnostic Regimes)

OMNIA flags structural instability, not correctness.

## Diagnostic regimes (current)

The following ranges are empirical and context-dependent.
They are frozen for reproducibility.

### TruthΩ
- < 0.8   : structurally stable
- 0.8–1.2 : mild instability
- 1.2–1.8 : elevated instability
- > 1.8   : high structural drift

### PBII
- < 0.5   : low base instability
- 0.5–0.7 : moderate
- > 0.7   : high base sensitivity

### omn_flag
omn_flag = 1 indicates *diagnostic attention*.
It does NOT imply incorrectness or failure.

## Notes
- Thresholds are not decision boundaries.
- No global constants are assumed.
- Calibration is dataset- and task-dependent.