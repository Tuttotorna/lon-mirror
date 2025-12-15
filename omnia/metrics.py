# omnia/metrics.py
# OMNIA – Core Metrics (TruthΩ / Δ / κ / ε)
# MB-X.01
# Minimal, deterministic, test-aligned implementation

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List
import math


@dataclass(frozen=True)
class Metrics:
    truth_omega: float
    delta_coherence: float
    kappa_alignment: float
    epsilon_drift: float


def _validate_signatures(signatures: Dict[int, Iterable[float]]) -> Dict[int, List[float]]:
    if not signatures:
        raise ValueError("No signatures provided")

    normalized: Dict[int, List[float]] = {}
    lengths = set()

    for base, sig in signatures.items():
        vec = list(sig)
        if len(vec) == 0:
            raise ValueError("Empty signature vector")
        normalized[base] = vec
        lengths.add(len(vec))

    if len(lengths) != 1:
        raise ValueError("Signature length mismatch across bases")

    return normalized


def compute_metrics(signatures: Dict[int, Iterable[float]]) -> Metrics:
    """
    Compute OMNIA structural metrics across bases.

    Assumptions (test-aligned):
    - identical vectors → TruthΩ = 1.0
    - increasing divergence → TruthΩ decreases
    """

    sigs = _validate_signatures(signatures)
    bases = list(sigs.keys())
    vectors = list(sigs.values())
    n = len(vectors[0])

    # Mean vector
    mean = [
        sum(v[i] for v in vectors) / len(vectors)
        for i in range(n)
    ]

    # Delta coherence: average L2 distance from mean
    distances = []
    for v in vectors:
        d = math.sqrt(sum((v[i] - mean[i]) ** 2 for i in range(n)))
        distances.append(d)

    delta_coherence = sum(distances) / len(distances)

    # Epsilon drift: max deviation
    epsilon_drift = max(distances)

    # Kappa alignment: inverse variance proxy
    if epsilon_drift == 0.0:
        kappa_alignment = 1.0
    else:
        kappa_alignment = 1.0 / (1.0 + epsilon_drift)

    # TruthΩ: normalized coherence score ∈ [0,1]
    truth_omega = 1.0 / (1.0 + delta_coherence)

    # Numerical safety
    truth_omega = max(0.0, min(1.0, truth_omega))

    return Metrics(
        truth_omega=truth_omega,
        delta_coherence=delta_coherence,
        kappa_alignment=kappa_alignment,
        epsilon_drift=epsilon_drift,
    )