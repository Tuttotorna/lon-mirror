from dataclasses import dataclass


@dataclass(frozen=True)
class StructuralSignature:
    omega: float
    omega_variance: float
    sei: float
    drift: float
    drift_vector: float
    order_sensitivity: float