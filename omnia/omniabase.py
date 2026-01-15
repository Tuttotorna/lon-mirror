from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Union

Number = Union[int, float]


# -----------------------------
# Base conversion utilities
# -----------------------------

def _to_base_digits(n: int, base: int) -> List[int]:
    if n == 0:
        return [0]
    if base < 2:
        raise ValueError("Base must be >= 2")
    digits: List[int] = []
    x = abs(n)
    while x > 0:
        digits.append(x % base)
        x //= base
    digits.reverse()
    return digits


# -----------------------------
# Signature construction
# -----------------------------

def omni_signature(
    value: int,
    bases: Iterable[int] = (2, 3, 4, 5, 7, 8, 10, 12, 16),
) -> Dict[int, Sequence[float]]:
    """
    Build a deterministic structural signature per base.

    Signature vector (fixed dimension = 5):
      [len(digits),
       sum(digits),
       mean(digits),
       max(digits),
       parity(sum(digits))]

    Notes:
    - No normalization across bases here (left to metrics).
    - Same vector shape across all bases (hard constraint).
    """
    sig: Dict[int, Sequence[float]] = {}

    for b in bases:
        d = _to_base_digits(int(value), int(b))
        L = float(len(d))
        S = float(sum(d))
        M = S / L if L > 0 else 0.0
        X = float(max(d)) if d else 0.0
        P = float(int(S) % 2)

        sig[int(b)] = [L, S, M, X, P]

    return sig


def omni_transform(
    values: Sequence[int],
    bases: Iterable[int] = (2, 3, 4, 5, 7, 8, 10, 12, 16),
) -> Dict[int, Sequence[float]]:
    """
    Aggregate signatures for a sequence of values.

    For each base, vectors are summed element-wise.
    Resulting vectors keep the same fixed dimension.
    """
    agg: Dict[int, List[float]] = {}

    for v in values:
        s = omni_signature(v, bases=bases)
        for b, vec in s.items():
            if b not in agg:
                agg[b] = [0.0] * len(vec)
            for i, x in enumerate(vec):
                agg[b][i] += float(x)

    return agg