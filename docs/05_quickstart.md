# OMNIA — Quickstart

This document provides the **minimal, canonical entry point** to OMNIA.

If this runs successfully, OMNIA is correctly installed and operational.

---

## Requirements

- Python 3.9 or higher
- No GPU required
- No external services required

OMNIA is fully local and deterministic.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Tuttotorna/lon-mirror
cd lon-mirror

Install in editable mode:

pip install -e .


---

Minimal Smoke Test

Run the minimal test script:

python quick_omnia_test.py


---

What This Test Validates

The smoke test verifies that all core lenses are operational:

BASE (Omniabase)
Numeric structure, entropy, PBII

TIME (Omniatempo)
Temporal statistics and regime score

CAUSA (Omniacausa)
Lagged correlation edges

TOKEN
Token-level PBII-z instability

Ω-TOTAL
Correct fusion of all components



---

Expected Output (Example)

=== OMNIA SMOKE TEST ===
BASE: sigma_mean=..., PBII=...
TIME: regime_change_score=...
CAUSA: edges=...
TOKEN: z_mean=...
Ω_total = ...
components = {...}

Exact numeric values may differ, but:

execution must complete without errors

all lenses must report outputs

Ω_total must be present



---

Failure Modes

If the test fails:

verify Python version

verify editable install succeeded

ensure no local modifications broke imports


If issues persist, OMNIA should be considered not operational.


---

Scope of This Test

This test does not:

validate semantic correctness

benchmark performance

certify safety


It only confirms structural measurement integrity.


---

Next Steps

After a successful quickstart:

see docs/06_usage.md for detailed usage

see docs/07_benchmarks.md for experimental evidence


