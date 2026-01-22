# examples/omnia_total_explainer.py
from __future__ import annotations

import json
import re
import zlib
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Minimal Ω-like measurers
# (semantics-free, deterministic)
# -----------------------------

def omega_compressibility(x: str) -> float:
    s = x.replace("\r\n", "\n")
    s = re.sub(r"[ \t]+", " ", s).strip()
    if not s:
        return 0.0
    comp = zlib.compress(s.encode("utf-8", errors="ignore"), level=9)
    ratio = len(comp) / max(1, len(s))
    return max(0.0, min(1.0, 1.0 - ratio))


def omega_digit_skeleton(x: str) -> float:
    digits = re.findall(r"\d+", x)
    if not digits:
        return 0.1
    total = sum(len(d) for d in digits)
    return max(0.0, min(1.0, 0.2 + (total / 200.0)))


def _project_keep_only_numbers(x: str) -> str:
    return re.sub(r"[^\d ]+", "", x)


def _project_keep_only_words(x: str) -> str:
    return re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ ]+", "", x)


def omega_projected_numbers(x: str) -> float:
    return omega_compressibility(_project_keep_only_numbers(x))


def omega_projected_words(x: str) -> float:
    return omega_compressibility(_project_keep_only_words(x))


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


# -----------------------------
# Canonical OMNIA report
# -----------------------------

def main(x: str, x_prime: Optional[str] = None) -> Dict[str, Any]:
    """
    Canonical, deterministic "total explainer" producing a single JSON report.

    Contract:
      - no semantics
      - deterministic
      - import-safe
      - returns a JSON-serializable dict

    This does NOT claim correctness of x.
    It measures structural behavior of x under aperspective transforms + meta-measures.
    """
    from omnia.lenses.aperspective_invariance import (
        AperspectiveInvariance,
        t_identity,
        t_whitespace_collapse,
        t_reverse,
        t_drop_vowels,
    )
    from omnia.omega_set import OmegaSet
    from omnia.sei import SEI
    from omnia.iri import IRI
    from omnia.meta.measurement_projection_loss import MeasurementProjectionLoss

    # 1) Aperspective invariance (Ω_ap)
    transforms: List[Tuple[str, Any]] = [
        ("id", t_identity),
        ("ws", t_whitespace_collapse),
        ("rev", t_reverse),
        ("vow-", t_drop_vowels),
    ]
    ap = AperspectiveInvariance(transforms=transforms)
    ap_r = ap.measure(x)

    per_t = dict(ap_r.per_transform_scores) if hasattr(ap_r, "per_transform_scores") else {}
    omega_ap = _safe_float(getattr(ap_r, "omega_score", 0.0))

    # 2) Ω̂ via OmegaSet over per-transform scores (robust stats)
    omega_samples = [_safe_float(v) for v in per_t.values()]
    os_est = OmegaSet(omega_samples).estimate() if omega_samples else {"median": omega_ap, "mad": 0.0}

    # 3) SEI from the ordered transform sequence (structural yield per cost step)
    # cost proxy: 1..N transforms applied (fixed order)
    omega_series = [_safe_float(per_t.get(name, 0.0)) for name, _ in transforms]
    cost_series = list(range(1, len(omega_series) + 1))
    sei = SEI(window=1, eps=1e-12)
    sei_curve = sei.curve(omega_series, cost_series)

    # 4) IRI if x_prime provided (A -> A')
    iri_block: Dict[str, Any] = {"enabled": False}
    if x_prime is not None:
        ap_r2 = ap.measure(x_prime)
        omega_ap2 = _safe_float(getattr(ap_r2, "omega_score", 0.0))
        iri = IRI()
        iri_val = _safe_float(iri.value(omega_ap, omega_ap2))
        iri_block = {
            "enabled": True,
            "omega_A": omega_ap,
            "omega_A_prime": omega_ap2,
            "iri": iri_val,
        }

    # 5) Observer Projection Loss (SPL / OPI-like)
    spl = MeasurementProjectionLoss(
        aperspective_measurers=[
            ("compressibility", omega_compressibility),
            ("digit_skeleton", omega_digit_skeleton),
        ],
        projected_measurers=[
            ("proj_numbers", omega_projected_numbers),
            ("proj_words", omega_projected_words),
        ],
        aggregator="trimmed_mean",
        trim_q=0.2,
    )
    spl_r = spl.measure(x)

    # 6) Optional INFERENCE state (only if module exists)
    inference_block: Dict[str, Any] = {"enabled": False}
    try:
        # Keep this optional to avoid breaking imports if not present.
        from omnia.inference.state_detector import InferenceStateDetector  # type: ignore

        det = InferenceStateDetector()
        s = det.classify(report_seed={"omega_ap": omega_ap, "omega_set": os_est, "sei": sei_curve, "iri": iri_block})
        inference_block = {"enabled": True, "state": str(s)}
    except Exception:
        inference_block = {"enabled": False}

    report: Dict[str, Any] = {
        "schema": "OMNIA_TOTAL_EXPLAINER_REPORT_v1",
        "input": {
            "x_len": len(x or ""),
            "x_prime_len": len(x_prime) if x_prime is not None else None,
            "x_preview": (x or "")[:180],
            "x_prime_preview": (x_prime or "")[:180] if x_prime is not None else None,
        },
        "measurements": {
            "aperspective": {
                "omega_ap": omega_ap,
                "per_transform_scores": per_t,
                "transforms": [name for name, _ in transforms],
            },
            "omega_set": os_est,
            "sei": {
                "omega_series": omega_series,
                "cost_series": cost_series,
                "sei_curve": sei_curve,
            },
            "iri": iri_block,
            "observer_projection": asdict(spl_r) if hasattr(spl_r, "__dataclass_fields__") else {
                "omega_aperspective": getattr(spl_r, "omega_aperspective", None),
                "omega_projected": getattr(spl_r, "omega_projected", None),
                "spl_abs": getattr(spl_r, "spl_abs", None),
                "spl_rel": getattr(spl_r, "spl_rel", None),
            },
            "inference": inference_block,
        },
        "notes": [
            "OMNIA measures structure, not semantic correctness.",
            "Deterministic report: same input must produce same output.",
        ],
    }

    return report


if __name__ == "__main__":
    demo_x = (
        "The sun does not erase stars; it saturates your detector.\n"
        "Double-slit interference: 2 slits yield a stable fringe pattern.\n"
        "2026 2025 2024 12345\n"
        "ABC ABC ABC ABC\n"
    )
    print(json.dumps(main(demo_x, x_prime=None), indent=2, ensure_ascii=False))