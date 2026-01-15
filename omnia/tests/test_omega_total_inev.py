import numpy as np
from omnia.omega_total import omega_total


def test_omega_total_with_inevitability():
    metrics = {"a": 1.0, "b": 0.0}

    signal = np.array([1, 2, 3, 4, 5], dtype=float)

    def signature(x):
        return float(np.std(x))

    perturbations = [
        lambda x: x + 0.0001,
        lambda x: x[::-1],
    ]

    r = omega_total(
        metrics,
        include_inev=True,
        inevitability_input=signal,
        inevitability_signature=signature,
        inevitability_perturbations=perturbations,
        inevitability_tolerance=1e-3,
    )

    assert "omega_total" in r
    assert "omega_inev" in r
    assert 0.0 <= r["omega_inev"] <= 1.0