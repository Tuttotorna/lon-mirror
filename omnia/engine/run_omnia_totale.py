from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, List

from omnia import (
    omni_signature,
    omni_transform,
    truth_omega,
    pbii_index,
)

# Optional lenses (if present in the package)
try:
    from omnia import omniatempo_analyze  # type: ignore
except Exception:
    omniatempo_analyze = None  # type: ignore

try:
    from omnia import omniacausa_analyze  # type: ignore
except Exception:
    omniacausa_analyze = None  # type: ignore


@dataclass(frozen=True)
class OmniaTotaleResult:
    omega_total: float
    lens_scores: Dict[str, float]
    lens_metadata: Dict[str, Any]


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _omega_from_positive_penalty(p: float) -> float:
    # deterministic mapping: penalty >= 0 -> omega in (0,1]
    p = max(0.0, _safe_float(p, 0.0))
    return 1.0 / (1.0 + p)


def _is_empty_series(series: Any) -> bool:
    """
    NumPy-safe emptiness check.
    - None -> empty
    - numpy.ndarray -> size == 0
    - list/tuple -> len == 0
    """
    if series is None:
        return True
    # numpy arrays: .size exists
    size = getattr(series, "size", None)
    if size is not None:
        try:
            return int(size) == 0
        except Exception:
            pass
    try:
        return len(series) == 0
    except Exception:
        return False


def _base_lens(n: int, bases: Iterable[int]) -> Tuple[float, Dict[str, Any]]:
    sig = omni_signature(n, bases=bases)
    omega = truth_omega(sig)
    pbii = pbii_index(n, bases=bases)

    # ω_base: stable if TruthΩ high and PBII low
    omega_base = float(omega * (1.0 / (1.0 + pbii)))

    meta = {
        "truth_omega": float(omega),
        "pbii": float(pbii),
        "bases": list(bases),
    }
    return omega_base, meta


def _tempo_lens(series: Any) -> Tuple[float, Dict[str, Any]]:
    """
    TIME lens:
    - if omniatempo_analyze exists, use it
    - else fallback deterministic drift proxy
    NumPy-safe.
    """
    if _is_empty_series(series):
        return 1.0, {"fallback": True, "penalty": 0.0}

    # If omniatempo_analyze exists, use it
    if omniatempo_analyze is not None:
        res = omniatempo_analyze(series)  # type: ignore
        rcs = _safe_float(getattr(res, "regime_change_score", 0.0))
        omega_tempo = _omega_from_positive_penalty(abs(rcs))
        meta = {
            "regime_change_score": float(rcs),
            "global_mean": _safe_float(getattr(res, "global_mean", 0.0)),
            "global_std": _safe_float(getattr(res, "global_std", 0.0)),
            "short_mean": _safe_float(getattr(res, "short_mean", 0.0)),
            "short_std": _safe_float(getattr(res, "short_std", 0.0)),
            "long_mean": _safe_float(getattr(res, "long_mean", 0.0)),
            "long_std": _safe_float(getattr(res, "long_std", 0.0)),
        }
        return float(omega_tempo), meta

    # Fallback: compare last vs first window means (works for list or numpy slice)
    n = len(series)
    w = max(5, n // 10)
    a = sum(series[:w]) / float(w)
    b = sum(series[-w:]) / float(w)
    penalty = abs(b - a)
    return _omega_from_positive_penalty(penalty), {"fallback": True, "penalty": float(penalty)}


def _causa_lens(
    series_dict: Mapping[str, Iterable[float]],
    max_lag: int = 5,
    strength_threshold: float = 0.3,
) -> Tuple[float, Dict[str, Any]]:
    # If omniacausa_analyze exists, use it; else fallback to empty graph
    if omniacausa_analyze is not None:
        res = omniacausa_analyze(series_dict, max_lag=max_lag, strength_threshold=strength_threshold)  # type: ignore
        edges = getattr(res, "edges", []) or []
        strengths: List[float] = []
        for e in edges:
            strengths.append(_safe_float(getattr(e, "strength", 0.0)))
        total_strength = sum(max(0.0, s) for s in strengths)
        omega_causa = _omega_from_positive_penalty(total_strength)

        meta = {
            "edges_count": int(len(edges)),
            "total_strength": float(total_strength),
            "max_lag": int(max_lag),
            "strength_threshold": float(strength_threshold),
        }
        return float(omega_causa), meta

    return 1.0, {"fallback": True, "edges_count": 0, "total_strength": 0.0}


def _token_lens(extra: Optional[Dict[str, Any]], bases: Iterable[int]) -> Tuple[float, Dict[str, Any]]:
    """
    TOKEN lens: converts token_numbers into a multi-base aggregate signature, then measures TruthΩ.
    Deterministic given token_numbers.
    """
    if not extra:
        return 1.0, {"tokens_present": False}

    token_numbers = extra.get("token_numbers", None)
    if not isinstance(token_numbers, list) or len(token_numbers) == 0:
        return 1.0, {"tokens_present": False}

    values: List[int] = []
    for x in token_numbers:
        try:
            values.append(int(x))
        except Exception:
            continue

    if len(values) == 0:
        return 1.0, {"tokens_present": False}

    sig = omni_transform(values, bases=bases)
    omega = truth_omega(sig)

    meta = {
        "tokens_present": True,
        "count": int(len(values)),
        "bases": list(bases),
    }
    return float(omega), meta


def run_omnia_totale(
    *,
    n: int,
    series: Optional[Sequence[float]] = None,
    series_dict: Optional[Mapping[str, Iterable[float]]] = None,
    bases: Iterable[int] = (2, 3, 4, 5, 7, 8, 10, 12, 16),
    w_base: float = 1.0,
    w_tempo: float = 1.0,
    w_causa: float = 1.0,
    w_token: float = 1.0,
    extra: Optional[Dict[str, Any]] = None,
) -> OmniaTotaleResult:
    """
    OMNIA_TOTALE: fused Ω-score from independent lenses.
    Measures only. No thresholds. No decision logic.

    Returns:
      - omega_total (0..1)
      - lens_scores (per-lens ω)
      - lens_metadata (per-lens diagnostics)
    """

    lens_scores: Dict[str, float] = {}
    lens_metadata: Dict[str, Any] = {}

    # BASE
    omega_base, meta_base = _base_lens(int(n), bases=bases)
    lens_scores["BASE"] = float(omega_base)
    lens_metadata["BASE"] = meta_base

    # TIME
    if series is None:
        lens_scores["TIME"] = 1.0
        lens_metadata["TIME"] = {"series_present": False}
    else:
        omega_time, meta_time = _tempo_lens(series)
        lens_scores["TIME"] = float(omega_time)
        lens_metadata["TIME"] = {"series_present": True, **meta_time}

    # CAUSA
    if series_dict is None:
        lens_scores["CAUSA"] = 1.0
        lens_metadata["CAUSA"] = {"series_dict_present": False}
    else:
        omega_causa, meta_causa = _causa_lens(series_dict)
        lens_scores["CAUSA"] = float(omega_causa)
        lens_metadata["CAUSA"] = {"series_dict_present": True, **meta_causa}

    # TOKEN
    omega_token, meta_token = _token_lens(extra, bases=bases)
    lens_scores["TOKEN"] = float(omega_token)
    lens_metadata["TOKEN"] = meta_token

    # Weighted fusion (normalized)
    weights = {
        "BASE": max(0.0, float(w_base)),
        "TIME": max(0.0, float(w_tempo)),
        "CAUSA": max(0.0, float(w_causa)),
        "TOKEN": max(0.0, float(w_token)),
    }

    num = (
        weights["BASE"] * lens_scores["BASE"]
        + weights["TIME"] * lens_scores["TIME"]
        + weights["CAUSA"] * lens_scores["CAUSA"]
        + weights["TOKEN"] * lens_scores["TOKEN"]
    )
    den = sum(weights.values())
    omega_total = float(num / den) if den > 0.0 else 0.0

    return OmniaTotaleResult(
        omega_total=omega_total,
        lens_scores=lens_scores,
        lens_metadata=lens_metadata,
    )