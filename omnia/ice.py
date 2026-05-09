"""
OMNIA ICE (Impossibility & Confidence Envelope) v0.2 — CANONICAL
MB-X.01 / OMNIA
Author: Massimiliano Brighindi

Goal:
    Turn "0% is false" into code:
      - detect IMPOSSIBLE cases (0% -> BLOCK)
      - detect BORDER cases (Omega near threshold -> BORDER -> ESCALATE)
      - detect AMBIGUOUS / LOW-QUALITY cases (-> ESCALATE)
      - allow STABLE cases (PASS)

Architecture boundary:
    - OMNIA = pure sensor: measures structure / geometry
    - ICE = thin, model-agnostic gate after OMNIA lenses plus optional LCR
    - Flags route attention, they do not override thresholds
    - Confidence = measurement quality / signal reliability, not outcome certainty
    - Border instability always escalates

Canonical convention:
    - omega_total is a coherence / health score: higher = better
    - structural risk is therefore inverted: risk_struct = 1 - omega_total

Boundary:
    measurement != inference != decision

ICE is not a truth oracle.
ICE is not a semantic judge.
ICE is not a decision engine.
Decision remains external.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class ICEStatus(str, Enum):
    PASS = "PASS"
    BORDER = "BORDER"
    ESCALATE = "ESCALATE"
    BLOCK = "BLOCK"


@dataclass
class ICEInput:
    omega_total: float
    lens_scores: Dict[str, float]
    lens_metadata: Dict[str, Dict[str, Any]]

    omega_ext: Optional[float] = None
    gold_match: Optional[float] = None

    ambiguity_score: float = 0.0
    notes: Optional[str] = None

    threshold: float = 0.70
    margin: float = 0.02

    signal_reliability: float = 1.0


@dataclass
class ICEResult:
    status: ICEStatus
    confidence: float
    impossibility: float
    ambiguity: float
    boundary_distance: float
    attention: List[str]
    reasons: Dict[str, Any]
    thresholds: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


def _clamp01(x: float) -> float:
    x = float(x)
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _derive_attention(
    lens_scores: Dict[str, float],
    lens_metadata: Dict[str, Dict[str, Any]],
    ambiguity: float,
    signal_reliability: float,
) -> List[str]:
    """
    Flags route attention only.

    Priority:
      1. low reliability / low quality lenses
      2. high problem-score lenses
      3. ambiguity hints
      4. low signal reliability hint
    """
    attention: List[str] = []

    for lens_name, meta in (lens_metadata or {}).items():
        quality = meta.get("reliability", meta.get("quality", None))
        if quality is not None and _safe_float(quality, 1.0) < 0.85:
            attention.append(lens_name)

    if lens_scores:
        ranked = sorted(
            lens_scores.items(),
            key=lambda item: _safe_float(item[1], 0.0),
            reverse=True,
        )
        for lens_name, _ in ranked:
            if lens_name not in attention:
                attention.append(lens_name)

    if ambiguity >= 0.55:
        for hint in ("TOKEN", "LCR", "TIME", "CAUSA", "BASE"):
            if hint in (lens_scores or {}) and hint not in attention:
                attention.insert(0, hint)

    if signal_reliability < 0.70 and "LCR" in (lens_scores or {}) and "LCR" not in attention:
        attention.insert(0, "LCR")

    seen = set()
    output: List[str] = []

    for item in attention:
        if item not in seen:
            seen.add(item)
            output.append(item)

    return output


def ice_gate(
    x: ICEInput,
    t_impossible: float = 0.85,
    t_ambiguous: float = 0.55,
    t_pass: float = 0.70,
    t_signal_min: float = 0.70,
    w_struct: float = 0.55,
    w_ext: float = 0.35,
    w_gold: float = 0.10,
) -> ICEResult:
    """
    Compute ICE gate status.

    Rule order:
      1. impossibility >= t_impossible -> BLOCK
      2. omega near threshold -> BORDER
      3. ambiguity >= t_ambiguous -> ESCALATE
      4. signal_reliability < t_signal_min -> ESCALATE
      5. confidence >= t_pass -> PASS
      6. otherwise -> ESCALATE
    """
    omega_total = _clamp01(_safe_float(x.omega_total, 0.0))
    ambiguity = _clamp01(_safe_float(x.ambiguity_score, 0.0))
    threshold = _clamp01(_safe_float(x.threshold, 0.70))
    margin = abs(_safe_float(x.margin, 0.02))
    signal_reliability = _clamp01(_safe_float(x.signal_reliability, 1.0))

    boundary_distance = abs(omega_total - threshold)
    is_boundary = boundary_distance <= margin

    risk_struct = _clamp01(1.0 - omega_total)

    if x.omega_ext is None:
        risk_ext = 0.5
        ext_available = False
    else:
        risk_ext = 1.0 - _clamp01(_safe_float(x.omega_ext, 0.5))
        ext_available = True

    if x.gold_match is None:
        risk_gold = 0.5
        gold_available = False
    else:
        risk_gold = 1.0 - _clamp01(_safe_float(x.gold_match, 0.5))
        gold_available = True

    impossibility = _clamp01(
        (w_struct * risk_struct)
        + (w_ext * risk_ext)
        + (w_gold * risk_gold)
    )

    base_confidence = 1.0 - impossibility
    confidence_after_ambiguity = _clamp01(base_confidence * (1.0 - 0.75 * ambiguity))
    confidence = _clamp01(confidence_after_ambiguity * (0.5 + 0.5 * signal_reliability))

    attention = _derive_attention(
        lens_scores=x.lens_scores or {},
        lens_metadata=x.lens_metadata or {},
        ambiguity=ambiguity,
        signal_reliability=signal_reliability,
    )

    if impossibility >= t_impossible:
        status = ICEStatus.BLOCK
    elif is_boundary:
        status = ICEStatus.BORDER
    elif ambiguity >= t_ambiguous:
        status = ICEStatus.ESCALATE
    elif signal_reliability < t_signal_min:
        status = ICEStatus.ESCALATE
    elif confidence >= t_pass:
        status = ICEStatus.PASS
    else:
        status = ICEStatus.ESCALATE

    reasons: Dict[str, Any] = {
        "omega_total": omega_total,
        "threshold": threshold,
        "margin": margin,
        "boundary_distance": boundary_distance,
        "risk_struct": risk_struct,
        "risk_ext": risk_ext,
        "risk_gold": risk_gold,
        "ext_available": ext_available,
        "gold_available": gold_available,
        "ambiguity": ambiguity,
        "signal_reliability": signal_reliability,
        "attention": attention,
        "notes": x.notes,
    }

    thresholds = {
        "t_impossible": float(t_impossible),
        "t_ambiguous": float(t_ambiguous),
        "t_pass": float(t_pass),
        "t_signal_min": float(t_signal_min),
        "w_struct": float(w_struct),
        "w_ext": float(w_ext),
        "w_gold": float(w_gold),
    }

    return ICEResult(
        status=status,
        confidence=confidence,
        impossibility=impossibility,
        ambiguity=ambiguity,
        boundary_distance=boundary_distance,
        attention=attention,
        reasons=reasons,
        thresholds=thresholds,
    )