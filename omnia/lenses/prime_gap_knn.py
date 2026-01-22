from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

from omnia.lenses.prime_regime import PrimeState


EPS = 1e-12


def _l1(a: List[float], b: List[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))


def _median(xs: List[float]) -> float:
    ys = sorted(xs)
    if not ys:
        return 0.0
    k = len(ys)
    mid = k // 2
    if k % 2 == 1:
        return ys[mid]
    return 0.5 * (ys[mid - 1] + ys[mid])


@dataclass(frozen=True)
class GapPrediction:
    g_hat: Optional[int]
    confidence: float
    stop: bool
    reason: str
    neighbors: List[Tuple[int, float, int]]  # (abs_j, dist, g_{j+1})


def state_distance(
    a: PrimeState,
    b: PrimeState,
    lam_T: float = 0.25,
    lam_S: float = 0.25,
) -> float:
    if len(a.phi) != len(b.phi):
        raise ValueError("Phi length mismatch")
    d_phi = _l1(a.phi, b.phi) / (len(a.phi) + EPS)
    return d_phi + lam_T * abs(a.T - b.T) + lam_S * abs(a.S - b.S)


def predict_next_gap_knn(
    primes: List[int],
    states: List[PrimeState],
    n_idx: int,
    start_idx: int,
    K: int = 25,
    lam_T: float = 0.25,
    lam_S: float = 0.25,
    C_min: float = 0.80,
    T_max: float = 0.08,
    S_min: float = 0.35,
) -> GapPrediction:
    """
    Predict gap g_{n+1} using KNN over PrimeState space (Phi + S + T).
    Deterministic. No learning.
    STOP if target regime is unstable or neighbor similarity is insufficient.
    """
    i = n_idx - start_idx
    if i < 0 or i >= len(states):
        return GapPrediction(None, 0.0, True, "index_out_of_range", [])

    target = states[i]

    # Regime guardrails
    if target.T > T_max:
        return GapPrediction(None, 0.0, True, f"STOP: drift T={target.T:.4f} > {T_max}", [])
    if target.S < S_min:
        return GapPrediction(None, 0.0, True, f"STOP: stability S={target.S:.4f} < {S_min}", [])

    # KNN search
    dists: List[Tuple[int, float]] = []
    for j, st in enumerate(states):
        if j == i:
            continue
        d = state_distance(target, st, lam_T=lam_T, lam_S=lam_S)
        dists.append((j, d))
    dists.sort(key=lambda x: x[1])
    knn = dists[: max(1, min(K, len(dists)))]

    neighbors: List[Tuple[int, float, int]] = []
    for j, d in knn:
        abs_j = start_idx + j
        g_j1 = primes[abs_j + 1] - primes[abs_j]
        neighbors.append((abs_j, d, g_j1))

    med_d = _median([d for _, d in knn])
    confidence = max(0.0, min(1.0, 1.0 - med_d))

    if confidence < C_min:
        return GapPrediction(None, confidence, True, f"STOP: confidence={confidence:.3f} < {C_min}", neighbors)

    g_hat = int(round(_median([g for _, _, g in neighbors])))
    return GapPrediction(g_hat, confidence, False, "OK", neighbors)