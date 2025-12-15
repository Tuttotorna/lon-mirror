# examples/inevitability_demo.py
# Minimal demonstration of Ω-INEV
# OMNIA / MB-X.01

import numpy as np
from omnia.inevitability import omega_inevitability


# Reproducibility (demo-only)
np.random.seed(0)

# Base structure
signal = np.array([1, 2, 3, 4, 5], dtype=float)


# Structural signature (example)
def signature_fn(x: np.ndarray) -> float:
    return float(np.std(x))


# Independent perturbations
perturbations = [
    lambda x: x + np.random.normal(0, 0.01, size=len(x)),
    lambda x: x[::-1],
    lambda x: x * 1.01,
]


result = omega_inevitability(
    base_signal=signal,
    perturbations=perturbations,
    signature_fn=signature_fn,
    tolerance=1e-3,
)

print("Ω-INEV:", result.omega_inev)
print("Stability curve:", result.stability_curve)