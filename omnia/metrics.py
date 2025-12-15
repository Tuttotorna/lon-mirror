# omnia/metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import math
import numpy as np


@dataclass
class Metrics:
    truth_omega: float
    delta_coherence: float
    kappa_alignment: float
    length_penalty: float


def _flatten(signatures: Dict[int, List[float]]) -> np.ndarray:
    return np.array([v for vals in signatures.values() for v in vals], dtype=float)


def _base_means(signatures: Dict[int, List[float]]) -> np.ndarray:
    return np.array([np.mean(v) for v in signatures.values()], dtype=float)


def compute_metrics(signatures: Dict[int, List[float]]) -> Metrics:
    """
    Compute OMNIA core metrics.

    Conventions (IMPORTANT):
    - truth_omega ∈ [0,1]  (1 = perfectly invariant)
    - delta_coherence ≥ 0  (higher = more deformation)
    - kappa_alignment ∈ [0,1] (1 = aligned, 0 = divergent)
    - length_penalty ∈ [0,1]
    """

    # ---------- Δ-coherence (dispersion penalty) ----------
    flat = _flatten(signatures)
    delta_coherence = float(np.std(flat))

    # ---------- TruthΩ (inverse deformation, normalized) ----------
    truth_omega = float(1.0 / (1.0 + delta_coherence))

    # ---------- κ-alignment (base agreement) ----------
    means = _base_means(signatures)
    if len(means) <= 1:
        kappa_alignment = 1.0
    else:
        mean_std = float(np.std(means))
        # Normalize to (0,1], lower std → higher alignment
        kappa_alignment = float(1.0 / (1.0 + mean_std))

    # ---------- Length penalty ----------
    lengths = np.array([len(v) for v in signatures.values()], dtype=float)
    length_penalty = float(1.0 / (1.0 + np.std(lengths)))

    return Metrics(
        truth_omega=truth_omega,
        delta_coherence=delta_coherence,
        kappa_alignment=kappa_alignment,
        length_penalty=length_penalty,
    )