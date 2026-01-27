"""
OMNIA ICE (Impossibility & Confidence Envelope) v0.2 — CANONICAL
MB-X.01 / OMNIA
Author: Massimiliano Brighindi

Goal:
    Turn "0% is false" into code:
      - detect IMPOSSIBLE cases (0% → BLOCK)
      - detect BORDER cases (Ω ≈ threshold → BORDER → ESCALATE)
      - detect AMBIGUOUS / LOW-QUALITY cases (→ ESCALATE)
      - allow STABLE cases (PASS)

Architecture boundary (critical):
    - OMNIA = pure sensor (measures structure / geometry)
    - ICE   = thin, model-agnostic gate *after* OMNIA lenses (+ optional LCR)
    - Flags route attention (which lens to inspect), they do NOT override thresholds
    - Confidence = measurement quality (signal reliability), NOT outcome certainty
    - Border instability (Ω near threshold) always escalates, never "targeted doubt"

Canonical semantic convention (CRITICAL):
    - omega_total (Ω) is a coherence/health score: higher = better
    - therefore structural risk must be inverted: risk_struct = 1 - Ω
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class ICEStatus(str, Enum):
    PASS = "PASS"          # allow to proceed (not a truth claim)
    BORDER = "BORDER"      # Ω near threshold -> escalation required
    ESCALATE = "ESCALATE"  # ambiguous / underdetermined / low measurement quality
    BLOCK = "BLOCK"        # impossible (0% class)


@dataclass
class ICEInput:
    # ----- primary measurement -----
    omega_total: float                          # Ω: coherence/health (0..1; higher = better)
    lens_scores: Dict[str, float]               # routing only; may represent "problem score" per lens
    lens_metadata: Dict[str, Dict[str, Any]]    # per-lens meta; may include reliability/quality flags

    # ----- optional external coherence (e.g., LCR / checks) -----
    omega_ext: Optional[float] = None           # 0..1 (higher = more coherent/better)
    gold_match: Optional[float] = None          # 0..1 (higher = better match to gold)

    # ----- ambiguity (underdetermination / multi-interpretation) -----
    ambiguity_score: float = 0.0                # 0..1 (higher = more ambiguous)
    notes: Optional[str] = None

    # ----- decision boundary (kept separate from flags) -----
    threshold: float = 0.70                     # τ (0..1)
    margin: float = 0.02                        # |Ω - τ| <= margin => BORDER -> ESCALATE

    # ----- measurement confidence (signal reliability), not outcome certainty -----
    signal_reliability: float = 1.0             # 0..1; low => ESCALATE


@dataclass
class ICEResult:
    status: ICEStatus
    confidence: float                           # 0..1 reliability of allowing PASS
    impossibility: float                        # 0..1 (1 => impossible / 0% class)
    ambiguity: float                            # 0..1
    boundary_distance: float                    # |Ω - τ|
    attention: List[str]                        # lenses to inspect first (routing only)
    reasons: Dict[str, Any]                     # machine-readable reasons
    thresholds: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


def _clamp01(x: float) -> float:
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
    Heuristics:
      1) prioritize lenses with low reliability/quality (meta)
      2) then lenses with high "problem score" (lens_scores, assumed higher=worse)
      3) if ambiguity high, prioritize TOKEN/LCR if present
      4) if signal reliability low, force LCR check if present
    """
    attention: List[str] = []

    # 1) low reliability/quality first
    for lname, meta in (lens_metadata or {}).items():
        q = meta.get("reliability", meta.get("quality", None))
        if q is not None and _safe_float(q, 1.0) < 0.85:
            attention.append(lname)

    # 2) high lens score next (routing only)
    if lens_scores:
        ranked = sorted(
            lens_scores.items(),
            key=lambda kv: _safe_float(kv[1], 0.0),
            reverse=True,
        )
        for lname, _ in ranked:
            if lname not in attention:
                attention.append(lname)

    # 3) ambiguity-driven routing hints (prepend)
    if ambiguity >= 0.55:
        for hint in ("TOKEN", "LCR", "TIME", "CAUSA", "BASE"):
            if hint in (lens_scores or {}) and hint not in attention:
                attention.insert(0, hint)

    # 4) low measurement reliability => check LCR if present
    if signal_reliability < 0.70 and "LCR" in (lens_scores or {}) and "LCR" not in attention:
        attention.insert(0, "LCR")

    # de-dup preserve order
    seen = set()
    out: List[str] = []
    for a in attention:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def ice_gate(
    x: ICEInput,
    # ----- thresholds -----
    t_impossible: float = 0.85,   # hard impossibility threshold (0..1): crossing => BLOCK
    t_ambiguous: float = 0.55,    # ambiguity threshold: crossing => ESCALATE
    t_pass: float = 0.70,         # PASS confidence threshold (only if not impossible, not ambiguous, not border, good signal)
    t_signal_min: float = 0.70,   # measurement reliability minimum: below => ESCALATE
    # ----- weights (tunable) -----
    w_struct: float = 0.55,
    w_ext: float = 0.35,
    w_gold: float = 0.10,
) -> ICEResult:
    """
    Compute:
      - impossibility: proxy for "0% class" (hard-block region)
      - ambiguity: underdetermination / multi-interpretation (escalate region)
      - confidence: reliability of allowing PASS (measurement quality, not outcome certainty)
      - BORDER: Ω near τ within margin -> escalate

    Rules order:
      1) if impossibility >= t_impossible  -> BLOCK
      2) else if BORDER (|Ω-τ|<=margin)    -> BORDER (escalate)
      3) else if ambiguity >= t_ambiguous  -> ESCALATE
      4) else if signal_reliability < min  -> ESCALATE
      5) else if confidence >= t_pass      -> PASS
      6) else                               ESCALATE
    """

    # ---- clamp primary inputs ----
    omega_total = _clamp01(_safe_float(x.omega_total, 0.0))
    ambiguity = _clamp01(_safe_float(x.ambiguity_score, 0.0))
    thr = _clamp01(_safe_float(x.threshold, 0.70))
    margin = abs(_safe_float(x.margin, 0.02))
    sigrel = _clamp01(_safe_float(x.signal_reliability, 1.0))

    # ---- BORDER instability (explicit) ----
    boundary_distance = abs(omega_total - thr)
    is_boundary = boundary_distance <= margin

    # ---- 1) Structural risk proxy ----
    # Ω is coherence/health (higher = better) -> risk must be inverted.
    risk_struct = _clamp01(1.0 - omega_total)

    # ---- 2) External component (omega_ext): higher = better -> lower risk ----
    if x.omega_ext is None:
        risk_ext = 0.5
        ext_available = False
    else:
        risk_ext = 1.0 - _clamp01(_safe_float(x.omega_ext, 0.5))
        ext_available = True

    # ---- 3) Gold component (if present): higher = better -> lower risk ----
    if x.gold_match is None:
        risk_gold = 0.5
        gold_available = False
    else:
        risk_gold = 1.0 - _clamp01(_safe_float(x.gold_match, 0.5))
        gold_available = True

    # ---- 4) Impossibility proxy (0% class detector) ----
    impossibility = _clamp01(
        (w_struct * risk_struct) +
        (w_ext * risk_ext) +
        (w_gold * risk_gold)
    )

    # ---- 5) Confidence (measurement quality), penalize ambiguity and low signal reliability ----
    base_conf = 1.0 - impossibility
    conf_after_ambiguity = _clamp01(base_conf * (1.0 - 0.75 * ambiguity))
    confidence = _clamp01(conf_after_ambiguity * (0.5 + 0.5 * sigrel))

    # ---- 6) Flags route attention (no threshold override) ----
    attention = _derive_attention(
        lens_scores=x.lens_scores or {},
        lens_metadata=x.lens_metadata or {},
        ambiguity=ambiguity,
        signal_reliability=sigrel,
    )

    # ---- 7) Decision ----
    if impossibility >= t_impossible:
        status = ICEStatus.BLOCK
    elif is_boundary:
        status = ICEStatus.BORDER
    elif ambiguity >= t_ambiguous:
        status = ICEStatus.ESCALATE
    elif sigrel < t_signal_min:
        status = ICEStatus.ESCALATE
    elif confidence >= t_pass:
        status = ICEStatus.PASS
    else:
        status = ICEStatus.ESCALATE

    reasons: Dict[str, Any] = {
        "omega_total": omega_total,
        "threshold": thr,
        "margin": margin,
        "boundary_distance": boundary_distance,

        # risk proxies (all in [0,1], higher = worse)
        "risk_struct": risk_struct,
        "risk_ext": risk_ext,
        "risk_gold": risk_gold,

        "ext_available": ext_available,
        "gold_available": gold_available,

        "ambiguity": ambiguity,
        "signal_reliability": sigrel,
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
```0