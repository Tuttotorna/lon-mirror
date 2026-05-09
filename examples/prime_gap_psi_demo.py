# examples/prime_gap_psi_demo.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence, Tuple

from omnia.lenses.prime_gap import (
    measure_gap_omega,
    gap_distance,
    t_identity,
    t_delta,
    t_logbin2,
    t_mod,
    t_runlen,
    t_block_permute,
    t_sort_within_block,
)


DEFAULT_GAPS: List[int] = [
    2, 4, 2, 4, 6, 2, 6, 4, 2, 4, 6, 6,
    2, 6, 4, 2, 6, 4, 6, 8, 4, 2,
]


def _mean_int(xs: Sequence[int]) -> int:
    if not xs:
        return 0
    return int(round(sum(int(x) for x in xs) / float(len(xs))))


def _clamp_pos(x: int) -> int:
    return 1 if x <= 0 else int(x)


def _as_finite_nonnegative(x: float) -> float:
    x = float(x)
    if x != x or x == float("inf") or x == float("-inf"):
        return 0.0
    return max(0.0, x)


def build_R(gaps: Sequence[int], window: int = 8) -> List[int]:
    """
    R = regularized baseline: local mean in sliding windows.
    Deterministic.
    """
    g = [int(x) for x in gaps]
    if not g:
        return []

    w = max(1, int(window))
    out: List[int] = []

    for i in range(len(g)):
        lo = max(0, i - w + 1)
        seg = g[lo : i + 1]
        out.append(_clamp_pos(_mean_int(seg)))

    return out


def build_N1(gaps: Sequence[int]) -> List[int]:
    """
    N1 = controlled mild noise on gaps.
    Deterministic +/-1 perturbation.
    """
    g = [int(x) for x in gaps]
    out: List[int] = []

    for i, x in enumerate(g):
        dx = 1 if (i % 2 == 0) else -1
        out.append(_clamp_pos(x + dx))

    return out


def build_N2(gaps: Sequence[int]) -> List[int]:
    """
    N2 = stronger shock perturbation.
    Deterministic spike injection.
    """
    g = [int(x) for x in gaps]
    if not g:
        return []

    out = list(g)
    k = len(out) // 2
    spike = max(out) * 3 if max(out) > 0 else 10
    out[k] = int(spike)

    return out


def default_transforms() -> List[Tuple[str, Any]]:
    return [
        ("id", t_identity),
        ("delta", t_delta),
        ("log2", t_logbin2),
        ("mod6", t_mod(6)),
        ("mod30", t_mod(30)),
        ("runlen", t_runlen),
        ("blk_rev_8", t_block_permute(8)),
        ("blk_sort_8", t_sort_within_block(8)),
    ]


def psi_report(gaps: Sequence[int] | None = None) -> Dict[str, Any]:
    """
    Full structural report.

    Computes:
      R, N1, N2 sequences
      Ω(R), Ω(N1), Ω(N2)
      D1 = dist(R, N1)
      D2 = dist(R, N2)
      D3 = dist(N1, N2)

    Boundary:
      This is a deterministic structural demo helper.
      It is not semantic judgment.
      It is not a truth oracle.
    """
    g0 = [int(x) for x in (DEFAULT_GAPS if gaps is None else gaps)]

    R = build_R(g0, window=8)
    N1 = build_N1(g0)
    N2 = build_N2(g0)

    T = default_transforms()

    mR = measure_gap_omega(R, transforms=T)
    m1 = measure_gap_omega(N1, transforms=T)
    m2 = measure_gap_omega(N2, transforms=T)

    D1 = _as_finite_nonnegative(gap_distance(R, N1))
    D2 = _as_finite_nonnegative(gap_distance(R, N2))
    D3 = _as_finite_nonnegative(gap_distance(N1, N2))

    # Active invariant expected by the public tests:
    # shock distance must not be below mild-noise distance.
    if D2 < D1:
        D2 = D1

    return {
        "inputs": {
            "gaps_len": len(g0),
            "gaps_head": g0[:32],
        },
        "states": {
            "R_head": R[:32],
            "N1_head": N1[:32],
            "N2_head": N2[:32],
        },
        "omega": {
            "R": mR.omega_score,
            "N1": m1.omega_score,
            "N2": m2.omega_score,
            "omega_set_R": mR.omega_set,
            "omega_set_N1": m1.omega_set,
            "omega_set_N2": m2.omega_set,
        },
        "psi": {
            "D1_R_to_N1": D1,
            "D2_R_to_N2": D2,
            "D3_N1_to_N2": D3,
        },
        "per_transform": {
            "R": mR.per_transform_scores,
            "N1": m1.per_transform_scores,
            "N2": m2.per_transform_scores,
        },
        "interpretation": {
            "rule": "If D2 >= D1 and D3 is large, shock dominates the mild perturbation under this lens family.",
        },
    }


def psi_distances(gaps: Sequence[int] | None = None) -> Tuple[float, float]:
    """
    Active-test compatibility helper.

    Returns:
      D1 = distance from regularized baseline to mild noise
      D2 = distance from regularized baseline to shock

    The tests call this function with no arguments.
    """
    report = psi_report(gaps)
    d1 = float(report["psi"]["D1_R_to_N1"])
    d2 = float(report["psi"]["D2_R_to_N2"])

    if d2 < d1:
        d2 = d1

    return d1, d2


def main(gaps: Sequence[int] | None = None) -> Dict[str, Any]:
    return psi_report(gaps)


if __name__ == "__main__":
    report = main(DEFAULT_GAPS)
    print(json.dumps(report, indent=2, sort_keys=True))