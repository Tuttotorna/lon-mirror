"""
OMNIA ICE (Impossibility & Confidence Envelope) v0.1
Author: Massimiliano Brighindi (MB-X.01 / OMNIA)

Goal:
    Turn "0% is false" into code:
      - detect IMPOSSIBLE cases (0% → block)
      - detect AMBIGUOUS cases (multi-interpretation → escalate)
      - allow STABLE cases (high confidence → pass)

This is a thin, model-agnostic gate that sits *after* OMNIA lenses
and optional LCR checks.

Design:
    - Not a "truth oracle"
    - It only enforces: never output structurally impossible statements.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, Any, Optional


class ICEStatus(str, Enum):
    PASS = "PASS"              # high confidence, not impossible
    ESCALATE = "ESCALATE"      # ambiguous / underdetermined
    BLOCK = "BLOCK"            # impossible (0%)


@dataclass
class ICEInput:
    # Unified structural scores
    omega_total: float
    lens_scores: Dict[str, float]          # e.g. {"BASE":..., "TIME":..., "CAUSA":..., "TOKEN":..., "LCR":...}
    lens_metadata: Dict[str, Dict[str, Any]]

    # Optional external coherence (LCR)
    omega_ext: Optional[float] = None      # fused fact+numeric score, if available
    gold_match: Optional[float] = None     # 1.0/0.0, if available

    # Optional ambiguity signals
    ambiguity_score: float = 0.0           # 0..1 (heuristic), higher = more ambiguous
    notes: Optional[str] = None


@dataclass
class ICEResult:
    status: ICEStatus
    confidence: float                      # 0..1 (reliability of allowing PASS)
    impossibility: float                   # 0..1 (0 means not impossible, 1 means impossible)
    ambiguity: float                       # 0..1
    reasons: Dict[str, Any]                # machine-readable reasons
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


def ice_gate(
    x: ICEInput,
    # ----- thresholds -----
    # "0%" = impossible if we cross these hard constraints:
    t_impossible: float = 0.85,     # hard impossibility threshold (0..1)
    # ambiguity threshold
    t_ambiguous: float = 0.55,      # above this → ESCALATE
    # confidence to PASS
    t_pass: float = 0.70,           # above this → PASS (if not impossible, not ambiguous)
    # weights (tunable)
    w_struct: float = 0.55,
    w_ext: float = 0.35,
    w_gold: float = 0.10,
) -> ICEResult:
    """
    Compute:
      - impossibility: how strongly the system believes the output is impossible (0% class)
      - ambiguity: how underdetermined it is
      - confidence: reliability of PASS decision

    Rules:
      - if impossibility >= t_impossible  -> BLOCK
      - else if ambiguity >= t_ambiguous  -> ESCALATE
      - else if confidence >= t_pass      -> PASS
      - else                               ESCALATE
    """

    # --- 1) Structural component: map Ω into a bounded "risk" proxy ---
    # We do not assume Ω scale is standardized. We use a soft squash.
    # Larger Ω -> higher instability -> higher risk.
    # risk_struct in [0,1]
    risk_struct = 1.0 - (1.0 / (1.0 + max(0.0, x.omega_total)))  # monotone, bounded
    risk_struct = _clamp01(risk_struct)

    # --- 2) External component (LCR): higher Ω_ext means more coherent (by your convention) ---
    # Convert to risk: low omega_ext => high risk
    if x.omega_ext is None:
        risk_ext = 0.5  # unknown
        ext_available = False
    else:
        # assume omega_ext already normalized 0..1
        risk_ext = 1.0 - _clamp01(float(x.omega_ext))
        ext_available = True

    # --- 3) Gold component (if present): gold_match=0 means impossible for that dataset ---
    if x.gold_match is None:
        risk_gold = 0.5
        gold_available = False
    else:
        # gold_match=1 -> risk 0, gold_match=0 -> risk 1
        risk_gold = 1.0 - _clamp01(float(x.gold_match))
        gold_available = True

    # --- 4) Combine into impossibility proxy ---
    # Impossibility is NOT the same as risk; we interpret high combined risk as "approaching 0%".
    impossibility = (
        w_struct * risk_struct
        + w_ext * risk_ext
        + w_gold * risk_gold
    )
    impossibility = _clamp01(impossibility)

    # --- 5) Ambiguity from input signal (can be replaced later by a proper ambiguity lens) ---
    ambiguity = _clamp01(float(x.ambiguity_score))

    # --- 6) Confidence is the complement of combined risk, penalized by ambiguity ---
    base_conf = 1.0 - impossibility
    confidence = _clamp01(base_conf * (1.0 - 0.75 * ambiguity))

    # --- 7) Decision ---
    if impossibility >= t_impossible:
        status = ICEStatus.BLOCK
    elif ambiguity >= t_ambiguous:
        status = ICEStatus.ESCALATE
    elif confidence >= t_pass:
        status = ICEStatus.PASS
    else:
        status = ICEStatus.ESCALATE

    reasons: Dict[str, Any] = {
        "risk_struct": risk_struct,
        "risk_ext": risk_ext,
        "risk_gold": risk_gold,
        "ext_available": ext_available,
        "gold_available": gold_available,
        "notes": x.notes,
    }

    thresholds = {
        "t_impossible": t_impossible,
        "t_ambiguous": t_ambiguous,
        "t_pass": t_pass,
        "w_struct": w_struct,
        "w_ext": w_ext,
        "w_gold": w_gold,
    }

    return ICEResult(
        status=status,
        confidence=confidence,
        impossibility=impossibility,
        ambiguity=ambiguity,
        reasons=reasons,
        thresholds=thresholds,
    )