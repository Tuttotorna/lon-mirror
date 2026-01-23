# OMNIA Boundaries (frozen failures)

This folder contains **frozen boundary artifacts**.

A boundary artifact is created when a **real failure mode** is observed and the correct response is **STOP**, not optimization.

Each boundary file is a minimal JSON record:

- input (or input hash + reproduction note)
- class (stress category)
- lens / module involved
- measured values (Ω / Ω̂ / SEI / IRI / SPL / SI when applicable)
- STOP decision + reason
- invariants (what must remain true in future versions)

Rule:
- Do not delete boundaries.
- Do not “fix” boundaries by hiding them.
- Boundaries define the perimeter of admissible extraction.

Next file to add:
`examples/boundaries/boundary_prime_knn_stop_drift.json`