# OMNIA / MB-X.01
# Structural metrics core: TruthΩ, Δ-coherence, κ-alignment, ε-drift
# Model-agnostic. No semantics. Only structural invariance signals.

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math


@dataclass(frozen=True)
class OmegaMetrics:
    truth_omega: float       # [0,1] invariance / stability
    delta_coherence: float   # >=0 deformation energy (lower = better)
    kappa_alignment: float   # [0,1] alignment score (higher = better)
    epsilon_drift: float     # >=0 drift score (lower = better)


# -----------------------------
# Utility: safe normalization
# -----------------------------

def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0.0 else default


# -----------------------------
# Core metric computation
# -----------------------------
# INPUT CONTRACT:
# - signatures: mapping base -> numeric signature vector (list of floats)
#   Example: {2:[...], 3:[...], 10:[...]}
#
# These vectors must be comparable across bases (same length),
# representing the same object through different lenses.

def compute_metrics(signatures: Dict[int, List[float]]) -> OmegaMetrics:
    """
    Compute TruthΩ and related structural metrics from multi-base signature vectors.

    TruthΩ is defined here as an exponential transform of Δ-coherence:
      TruthΩ = exp(-Δ)

    Δ-coherence is mean pairwise normalized L2 distance between signature vectors.
    κ-alignment is 1 - normalized variance across bases (per-dimension), averaged.
    ε-drift is max pairwise normalized L2 distance (worst-case fragility).

    All metrics are representation-agnostic and purely structural.
    """
    bases = sorted(signatures.keys())
    if len(bases) < 2:
        # With 0/1 base, invariance cannot be evaluated; return neutral stable.
        return OmegaMetrics(
            truth_omega=1.0,
            delta_coherence=0.0,
            kappa_alignment=1.0,
            epsilon_drift=0.0,
        )

    # Validate vector lengths
    lengths = {len(signatures[b]) for b in bases}
    if len(lengths) != 1:
        raise ValueError(f"Signature vectors must have same length across bases. Got lengths={sorted(lengths)}")

    d = next(iter(lengths))

    # Prepare vectors
    vecs = [signatures[b] for b in bases]

    # Compute pairwise distances
    pair_dists: List[float] = []
    max_dist = 0.0

    # Normalization scale: average L2 norm of vectors (avoid dependence on absolute magnitude)
    norms = [math.sqrt(sum(x*x for x in v)) for v in vecs]
    scale = sum(norms) / len(norms)
    if scale == 0.0:
        scale = 1.0

    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            dist = math.sqrt(sum((vecs[i][k] - vecs[j][k]) ** 2 for k in range(d)))
            nd = dist / scale
            pair_dists.append(nd)
            if nd > max_dist:
                max_dist = nd

    # Δ-coherence: mean normalized deformation
    delta = sum(pair_dists) / len(pair_dists) if pair_dists else 0.0

    # ε-drift: worst-case deformation
    epsilon = max_dist

    # κ-alignment: 1 - normalized variance across bases
    # For each dimension k, compute variance across bases, normalize by (mean_abs + 1e-9),
    # convert to alignment by exp(-var_norm), then average.
    kappa_parts: List[float] = []
    for k in range(d):
        vals = [v[k] for v in vecs]
        mean = sum(vals) / len(vals)
        var = sum((x - mean) ** 2 for x in vals) / len(vals)
        mean_abs = sum(abs(x) for x in vals) / len(vals)
        var_norm = var / (mean_abs + 1e-9)
        kappa_parts.append(math.exp(-var_norm))

    kappa = sum(kappa_parts) / len(kappa_parts) if kappa_parts else 1.0
    kappa = _clamp01(kappa)

    # TruthΩ: bounded monotone transform of Δ
    truth_omega = math.exp(-delta)
    truth_omega = _clamp01(truth_omega)

    return OmegaMetrics(
        truth_omega=truth_omega,
        delta_coherence=delta,
        kappa_alignment=kappa,
        epsilon_drift=epsilon,
    )