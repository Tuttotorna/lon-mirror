# examples/prime_gap_psi_demo.py
from __future__ import annotations

import json
import math
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


def _mean_int(xs: Sequence[int]) -> int:
    if not xs:
        return 0
    return int(round(sum(int(x) for x in xs) / float(len(xs))))


def _clamp_pos(x: int) -> int:
    return 1 if x <= 0 else x


def build_R(gaps: Sequence[int], window: int = 8) -> List[int]:
    """
    R = "regularized" baseline: local mean in sliding windows.
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
    N1 = controlled noise on gaps (deterministic):
    add +/-1 pattern based on index parity, clamp to >= 1
    """
    g = [int(x) for x in gaps]
    out: List[int] = []
    for i, x in enumerate(g):
        dx = 1 if (i % 2 == 0) else -1
        out.append(_clamp_pos(x + dx))
    return out


def build_N2(gaps: Sequence[int]) -> List[int]:
    """
    N2 = shock: inject a large spike at 1/2 position (deterministic).
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


def psi_distances(gaps: Sequence[int]) -> Dict[str, Any]:
    """
    Computes:
      R, N1, N2 sequences
      Ω(R), Ω(N1), Ω(N2)
      D1 = dist(R, N1)
      D2 = dist(R, N2)
      D3 = dist(N1, N2)
    plus transform-wise overlaps.
    """
    g0 = [int(x) for x in gaps]

    R = build_R(g0, window=8)
    N1 = build_N1(g0)
    N2 = build_N2(g0)

    T = default_transforms()

    mR = measure_gap_omega(R, transforms=T)
    m1 = measure_gap_omega(N1, transforms=T)
    m2 = measure_gap_omega(N2, transforms=T)

    # Distances between sequences (structure-space)
    D1 = gap_distance(R, N1)
    D2 = gap_distance(R, N2)
    D3 = gap_distance(N1, N2)

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
            "rule": "If D2 >> D1 and D3 large, shock dominates; if D1~D2~D3 small, indistinguishable under this lens family.",
        },
    }


def main(gaps: Sequence[int]) -> Dict[str, Any]:
    return psi_distances(gaps)


if __name__ == "__main__":
    # Minimal deterministic sample (gap-like sequence)
    sample = [2, 4, 2, 4, 6, 2, 6, 4, 2, 4, 6, 6, 2, 6, 4, 2, 6, 4, 6, 8, 4, 2]
    report = main(sample)
    print(json.dumps(report, indent=2, sort_keys=True))