from __future__ import annotations

from typing import Dict
from omnia.meta.measurement_projection_loss import MeasurementProjectionLoss


class ABStructuralCycle:
    """
    Measures the A → B → A′ cycle using SPL / OPI logic.

    This class does NOT execute physics.
    It measures structural loss induced by projection.
    """

    def __init__(self, projection_operator: MeasurementProjectionLoss):
        self.op = projection_operator

    def measure(self, representation: str) -> Dict[str, float]:
        r = self.op.measure(representation)

        return {
            "Omega_A": r.omega_aperspective,
            "Omega_B": r.omega_projected,
            "SPL_abs": r.spl_abs,
            "SPL_rel": r.spl_rel,
            "OPI": r.spl_abs,  # equivalence
        }