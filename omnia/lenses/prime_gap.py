# omnia/lenses/prime_gap.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple, Optional
import math
import re
import statistics


# ----------------------------
# Types
# ----------------------------
Gaps = List[int]
Transform = Callable[[Gaps], Gaps]


# ----------------------------
# Canonical encoders
# ----------------------------
def encode_gaps(gaps: Sequence[int], mode: str = "csv") -> str:
    """
    Deterministic encoding of integer gap sequences.
    mode:
      - "csv"   : "2,4,6,2,10"
      - "fixed" : fixed-width tokens separated by spaces, e.g. "0002 0004 ..."
    """
    if mode not in ("csv", "fixed"):
        raise ValueError(f"Unknown mode: {mode}")

    if not gaps:
        return ""

    if mode == "csv":
        return ",".join(str(int(x)) for x in gaps)

    # fixed-width
    m = max(int(x) for x in gaps)
    width = max(1, len(str(m)))
    return " ".join(f"{int(x):0{width}d}" for x in gaps)


# ----------------------------
# Transforms (aperspective, deterministic, non-semantic)
# ----------------------------
def t_identity(gaps: Gaps) -> Gaps:
    return list(gaps)


def t_delta(gaps: Gaps) -> Gaps:
    # First difference: g[i+1] - g[i]
    if len(gaps) < 2:
        return []
    out = []
    for i in range(len(gaps) - 1):
        out.append(int(gaps[i + 1]) - int(gaps[i]))
    return out


def t_logbin2(gaps: Gaps) -> Gaps:
    # floor(log2(g)) with protection
    out = []
    for g in gaps:
        gi = int(g)
        if gi <= 0:
            out.append(0)
        else:
            out.append(int(math.floor(math.log2(gi))))
    return out


def t_mod(m: int) -> Transform:
    if m <= 0:
        raise ValueError("m must be positive")

    def _t(gaps: Gaps) -> Gaps:
        return [int(g) % m for g in gaps]

    return _t


def t_runlen(gaps: Gaps) -> Gaps:
    """
    Run-length encoding flattened as [value1, count1, value2, count2, ...]
    Example: [2,2,2,6,6,4] -> [2,3,6,2,4,1]
    """
    if not gaps:
        return []
    out: List[int] = []
    cur = int(gaps[0])
    cnt = 1
    for g in gaps[1:]:
        gi = int(g)
        if gi == cur:
            cnt += 1
        else:
            out.extend([cur, cnt])
            cur = gi
            cnt = 1
    out.extend([cur, cnt])
    return out


def t_block_permute(block: int) -> Transform:
    if block <= 0:
        raise ValueError("block must be positive")

    def _t(gaps: Gaps) -> Gaps:
        if not gaps:
            return []
        blocks: List[List[int]] = []
        for i in range(0, len(gaps), block):
            blocks.append([int(x) for x in gaps[i : i + block]])
        blocks.reverse()
        out: List[int] = []
        for b in blocks:
            out.extend(b)
        return out

    return _t


def t_sort_within_block(block: int) -> Transform:
    if block <= 0:
        raise ValueError("block must be positive")

    def _t(gaps: Gaps) -> Gaps:
        if not gaps:
            return []
        out: List[int] = []
        for i in range(0, len(gaps), block):
            b = [int(x) for x in gaps[i : i + block]]
            b.sort()
            out.extend(b)
        return out

    return _t


# ----------------------------
# Distances (deterministic, non-semantic)
# ----------------------------
def gap_distance(a: Sequence[int], b: Sequence[int], kind: str = "normalized_L1") -> float:
    """
    Distance in [0, +inf). Default 'normalized_L1' returns value in [0, 1] when lengths match,
    otherwise adds a length mismatch penalty.
    """
    if kind != "normalized_L1":
        raise ValueError(f"Unknown distance kind: {kind}")

    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0

    na = len(a)
    nb = len(b)
    n = min(na, nb)

    s = 0.0
    for i in range(n):
        ai = float(a[i])
        bi = float(b[i])
        denom = max(abs(ai), abs(bi), 1.0)
        s += abs(ai - bi) / denom

    # average on overlap
    d = s / float(n)

    # length penalty (bounded)
    if na != nb:
        d += min(1.0, abs(na - nb) / float(max(na, nb)))

    return d


def overlap_from_distance(d: float) -> float:
    # Map to [0,1]
    if d <= 0:
        return 1.0
    if d >= 1.0:
        return 0.0
    return 1.0 - d


# ----------------------------
# Omega-set stats (robust)
# ----------------------------
def omega_set_stats(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"median": 0.0, "mad": 0.0}

    vals = [float(v) for v in values]
    med = statistics.median(vals)
    abs_dev = [abs(v - med) for v in vals]
    mad = statistics.median(abs_dev) if abs_dev else 0.0

    # "invariance" can be defined later; keep minimal stable stats now.
    return {"median": float(med), "mad": float(mad)}


# ----------------------------
# Measurement output
# ----------------------------
@dataclass(frozen=True)
class GapOmegaResult:
    omega_score: float
    per_transform_scores: Dict[str, float]
    omega_set: Dict[str, float]


def measure_gap_omega(
    gaps: Sequence[int],
    transforms: Optional[List[Tuple[str, Transform]]] = None,
    distance_kind: str = "normalized_L1",
) -> GapOmegaResult:
    """
    Compute Ω on gap sequences via transform overlaps.
    Deterministic, non-semantic, stable.
    """
    g0: List[int] = [int(x) for x in gaps]

    if transforms is None:
        transforms = [
            ("id", t_identity),
            ("delta", t_delta),
            ("log2", t_logbin2),
            ("mod6", t_mod(6)),
            ("mod30", t_mod(30)),
            ("runlen", t_runlen),
            ("blk_rev_8", t_block_permute(8)),
            ("blk_sort_8", t_sort_within_block(8)),
        ]

    per: Dict[str, float] = {}
    scores: List[float] = []

    for name, t in transforms:
        gt = t(g0)
        d = gap_distance(g0, gt, kind=distance_kind)
        s = overlap_from_distance(d)
        # enforce numeric bounds
        s = 0.0 if s < 0.0 else 1.0 if s > 1.0 else s
        per[name] = float(s)
        scores.append(float(s))

    stats = omega_set_stats(scores)
    omega = float(stats["median"])
    return GapOmegaResult(omega_score=omega, per_transform_scores=per, omega_set=stats)