from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple, Union, Optional

Number = Union[int, float]
Vec = List[float]
SignatureMap = Mapping[int, Sequence[Number]]  # base -> vector signature


def _to_vec(x: Sequence[Number]) -> Vec:
    return [float(v) for v in x]


def _mean(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    return sum(xs) / float(len(xs))


def _l1(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} != {len(b)}")
    return sum(abs(ai - bi) for ai, bi in zip(a, b))


def _l2(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} != {len(b)}")
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(sum(ai * ai for ai in a))


def _cosine_similarity(a: Sequence[float], b: Sequence[float], eps: float = 1e-12) -> float:
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} != {len(b)}")
    na = _norm(a)
    nb = _norm(b)
    if na < eps or nb < eps:
        # If one is near-zero, define similarity as 0 (non-informative alignment)
        return 0.0
    dot = sum(ai * bi for ai, bi in zip(a, b))
    return max(-1.0, min(1.0, dot / (na * nb)))


def delta_coherence(signatures: SignatureMap) -> float:
    """
    Δ-coherence: dispersion across bases.
    Returns a non-negative scalar. Lower is better (more invariant).
    Computation: mean pairwise L1 distance among base signatures.
    """
    items: List[Tuple[int, Vec]] = [(int(b), _to_vec(v)) for b, v in signatures.items()]
    if len(items) < 2:
        return 0.0

    # enforce same dimensionality
    dim = len(items[0][1])
    for _, v in items:
        if len(v) != dim:
            raise ValueError("All signatures must have same dimension.")

    dists: List[float] = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            dists.append(_l1(items[i][1], items[j][1]))

    return _mean(dists)


def truth_omega(signatures: SignatureMap) -> float:
    """
    TruthΩ: coherence-as-invariance score.
    Snapshot constraint: higher => more stable reasoning.
    Definition: TruthΩ = exp(-Δ), with Δ from delta_coherence().
    Range: (0, 1], where 1 means perfect invariance across bases.
    """
    d = delta_coherence(signatures)
    return math.exp(-max(0.0, d))


def kappa_alignment(signatures: SignatureMap) -> float:
    """
    κ-alignment: how strongly different bases agree in *direction* (not magnitude).
    Computation:
      1) build centroid vector c = mean(base_vectors)
      2) compute mean cosine similarity of each base vector vs centroid
    Range: [-1, 1], typically [0, 1] for non-adversarial signatures.
    Higher is better.
    """
    items: List[Vec] = [_to_vec(v) for v in signatures.values()]
    if not items:
        return 0.0
    if len(items) == 1:
        return 1.0

    dim = len(items[0])
    for v in items:
        if len(v) != dim:
            raise ValueError("All signatures must have same dimension.")

    centroid = [0.0] * dim
    for v in items:
        for k in range(dim):
            centroid[k] += v[k]
    centroid = [c / float(len(items)) for c in centroid]

    sims = [_cosine_similarity(v, centroid) for v in items]
    return _mean(sims)


def epsilon_drift(signatures_t0: SignatureMap, signatures_t1: SignatureMap) -> float:
    """
    ε-drift: change of structure across two states (e.g., time steps).
    Returns a non-negative scalar. Lower is better (less drift).
    Computation:
      - find common bases
      - average L2 distance per base between t0 and t1 signatures
    """
    common_bases = sorted(set(signatures_t0.keys()) & set(signatures_t1.keys()))
    if not common_bases:
        return 0.0

    dists: List[float] = []
    for b in common_bases:
        v0 = _to_vec(signatures_t0[b])
        v1 = _to_vec(signatures_t1[b])
        if len(v0) != len(v1):
            raise ValueError(f"Signature dimension mismatch for base {b}.")
        dists.append(_l2(v0, v1))

    return _mean(dists)