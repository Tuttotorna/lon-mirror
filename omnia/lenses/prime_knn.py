from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import math

from prime_state import prime_state_from_primes, PrimeState  # importa dal file dove l'hai messo

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
    neighbors: List[Tuple[int, float, int]]  # (j, dist, g_{j+1})

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

def build_states(
    primes: List[int],
    mods: List[int],
    window: int = 512,
    drift_theta: float = 0.05,
    start_idx: int = 2,
    end_idx: Optional[int] = None,
) -> List[PrimeState]:
    """
    Build PrimeState list aligned to indices [start_idx..end_idx].
    Each state corresponds to index n (prime p_n).
    """
    if end_idx is None:
        end_idx = len(primes) - 2  # need j+1 for gap label

    states: List[PrimeState] = []
    prev_phi = None
    prev_tau = 0

    for n in range(start_idx, end_idx + 1):
        st = prime_state_from_primes(
            primes=primes,
            idx=n,
            mods=mods,
            window=window,
            drift_theta=drift_theta,
            prev_phi=prev_phi,
            prev_tau=prev_tau,
        )
        states.append(st)
        prev_phi = st.phi
        prev_tau = st.tau

    return states

def predict_next_gap_knn(
    primes: List[int],
    states: List[PrimeState],
    n_idx: int,
    K: int = 25,
    lam_T: float = 0.25,
    lam_S: float = 0.25,
    C_min: float = 0.80,
    T_max: float = 0.08,
    S_min: float = 0.35,
) -> GapPrediction:
    """
    Predict g_{n+1} using KNN in Phi-space.
    n_idx is the absolute prime index n (refers to p_n).
    states is built starting at start_idx; map accordingly.
    """
    # mapping: states[0] corresponds to start_idx
    # We infer start_idx from alignment assumption:
    start_idx = 2
    i = n_idx - start_idx
    if i < 0 or i >= len(states):
        return GapPrediction(None, 0.0, True, "index_out_of_range", [])

    target = states[i]

    # guardrails on target regime
    if target.T > T_max:
        return GapPrediction(None, 0.0, True, f"STOP: drift T={target.T:.4f} > {T_max}", [])
    if target.S < S_min:
        return GapPrediction(None, 0.0, True, f"STOP: stability S={target.S:.4f} < {S_min}", [])

    dists: List[Tuple[int, float]] = []
    for j, st in enumerate(states):
        if j == i:
            continue
        d = state_distance(target, st, lam_T=lam_T, lam_S=lam_S)
        dists.append((j, d))

    dists.sort(key=lambda x: x[1])
    knn = dists[: max(1, min(K, len(dists)))]

    # collect neighbor gap labels g_{j+1}
    neigh: List[Tuple[int, float, int]] = []
    for j, d in knn:
        abs_j = start_idx + j
        g_j1 = primes[abs_j + 1] - primes[abs_j]
        neigh.append((abs_j, d, g_j1))

    # confidence from median neighbor distance
    med_d = _median([d for _, d in knn])
    confidence = max(0.0, min(1.0, 1.0 - med_d))

    if confidence < C_min:
        return GapPrediction(None, confidence, True, f"STOP: confidence={confidence:.3f} < {C_min}", neigh)

    # robust prediction = median of neighbor gaps
    g_hat = int(round(_median([g for _, _, g in neigh])))

    return GapPrediction(g_hat, confidence, False, "OK", neigh)

def eval_knn(
    primes: List[int],
    mods: List[int],
    window: int = 512,
    drift_theta: float = 0.05,
    K: int = 25,
    eval_from: int = 2000,
    eval_to: int = 5000,
) -> dict:
    """
    Simple deterministic evaluation: MAE on gaps where not stopped.
    """
    states = build_states(primes, mods, window=window, drift_theta=drift_theta, start_idx=2, end_idx=eval_to)
    abs_errors = []
    stops = 0
    total = 0

    for n in range(eval_from, eval_to):
        pred = predict_next_gap_knn(primes, states, n_idx=n, K=K)
        true_g = primes[n + 1] - primes[n]
        total += 1
        if pred.stop or pred.g_hat is None:
            stops += 1
            continue
        abs_errors.append(abs(pred.g_hat - true_g))

    mae = sum(abs_errors) / len(abs_errors) if abs_errors else float("inf")
    return {
        "total": total,
        "predicted": len(abs_errors),
        "stopped": stops,
        "stop_rate": stops / total if total else 1.0,
        "mae_gap": mae,
        "K": K,
        "window": window,
        "mods": mods,
    }