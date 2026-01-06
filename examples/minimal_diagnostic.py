"""
Minimal OMNIA diagnostic example.

Purpose:
- Demonstrate OMNIA as a post-hoc structural diagnostic layer
- No semantics, no decisions, no optimization
- Deterministic input -> deterministic output
"""

import json
from omnia.engine.omnia_totale import omnia_total_score

# Example inputs: two alternative outputs for the same task
# (strings, numbers, token sequences, etc.)
output_a = "The Eiffel Tower is 324 meters tall."
output_b = "The Eiffel Tower is 300 meters tall."

# Explicitly declared allowed transformations
# (no hidden assumptions)
transformations = {
    "case_normalization": True,
    "token_reordering": False,
    "numeric_scaling": False,
}

# Run OMNIA structural diagnostic
report = omnia_total_score(
    outputs=[output_a, output_b],
    transforms=transformations,
    return_breakdown=True,
)

# Emit machine-readable report (schema-compliant)
result = {
    "version": "v1.0",
    "omega_total": report["omega_total"],
    "lens_breakdown": report["breakdown"],
    "regime": report["regime"],  # OPEN / NEAR_BOUNDARY / SATURATED
    "meta": {
        "transforms": transformations
    }
}

print(json.dumps(result, indent=2, sort_keys=True))