# OMNIA / MB-X.01 — ICE / Δ-Policy (post-processor)
# Purpose: decide if a draft answer is ALLOW / BOUNDARY_ONLY / REFUSE
# based on structural instability, internal contradictions, and saturation signals.
#
# This module is model-agnostic: it consumes (prompt, draft, variants[])
# and outputs a DeltaDecision + final_text (if templates are used downstream).

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Dict
import re


class Boundary(str, Enum):
    OPEN = "open"
    NEAR_BOUNDARY = "near_boundary"
    SATURATED = "saturated"


class Loss(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    CRITICAL = "critical"


class Compressibility(str, Enum):
    YES = "yes"
    PARTIAL = "partial"
    NO = "no"


class Action(str, Enum):
    ALLOW = "ALLOW"
    BOUNDARY_ONLY = "BOUNDARY_ONLY"
    REFUSE = "REFUSE"


@dataclass(frozen=True)
class DeltaDecision:
    B: Boundary
    L: Loss
    C: Compressibility
    action: Action
    reasons: List[str]
    scores: Dict[str, float]


# -----------------------------
# Structural signal extractors
# -----------------------------

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _extract_numbers(text: str) -> List[str]:
    return _NUM_RE.findall(text)


def _tail_sentences(text: str, n: int = 2) -> str:
    # Cheap sentence split (deterministic). Good enough for structural signatures.
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(parts[-n:]) if parts else text.strip()


def conclusion_signature(text: str) -> Tuple[str, int, Tuple[str, ...]]:
    tail = _tail_sentences(text, 2).lower()
    neg = len(re.findall(r"\b(no|not|never|cannot|can't|impossible)\b", tail))
    nums = tuple(_extract_numbers(tail))
    return (tail, neg, nums)


def T_score(instances: List[str]) -> float:
    """
    Instability across variants: compares conclusion signatures.
    Returns [0..1] where 1 = highly unstable.
    """
    sigs = [conclusion_signature(x) for x in instances if x and x.strip()]
    if len(sigs) < 2:
        return 0.0

    # Divergence if tails differ substantially or numbers/negations mismatch
    tails = [s[0] for s in sigs]
    negs = [s[1] for s in sigs]
    nums = [s[2] for s in sigs]

    tail_div = 1.0 if len(set(tails)) > 1 else 0.0
    neg_div = 1.0 if len(set(negs)) > 1 else 0.0
    num_div = 1.0 if len(set(nums)) > 1 else 0.0

    # Conservative: any divergence raises instability
    return min(1.0, 0.5 * tail_div + 0.25 * neg_div + 0.25 * num_div)


def I_score(text: str) -> float:
    """
    Internal contradiction heuristic. Returns [0..1].
    Deterministic patterns:
      - conflicting absolute qualifiers (always/never) with exceptions (sometimes/may)
      - repeated numbers with conflicting contexts (weak heuristic)
    """
    t = (text or "").lower()
    if not t.strip():
        return 0.0

    abs_q = bool(re.search(r"\b(always|never|must|cannot|impossible)\b", t))
    exc_q = bool(re.search(r"\b(sometimes|may|might|can|often|in some cases)\b", t))
    qualifier_conflict = 1.0 if (abs_q and exc_q) else 0.0

    # Weak numeric inconsistency: many numbers in short answer often correlates with drift
    nums = _extract_numbers(t)
    numeric_density = min(1.0, len(nums) / 12.0)

    return min(1.0, 0.7 * qualifier_conflict + 0.3 * numeric_density)


def S_score(prompt: str, text: str) -> float:
    """
    Saturation/over-explanation heuristic. Returns [0..1].
    Signals:
      - answer far longer than prompt (verbosity mismatch)
      - deep causal chains without explicit assumptions (because -> because -> ...)
    """
    p = (prompt or "")
    a = (text or "")
    if not a.strip():
        return 0.0

    lp = max(1, len(p))
    la = len(a)
    length_ratio = min(1.0, la / (6.0 * lp))  # >6x prompt length saturates

    because_chain = len(re.findall(r"\b(because|therefore|thus|hence)\b", a.lower()))
    chain_score = min(1.0, because_chain / 8.0)

    return min(1.0, 0.6 * length_ratio + 0.4 * chain_score)


# -----------------------------
# Δ mapping (initial thresholds)
# -----------------------------

def delta_policy(
    prompt: str,
    draft: str,
    variants: List[str],
    t_hi: float = 0.70,
    t_mid: float = 0.40,
    i_hi: float = 0.60,
    s_hi: float = 0.60,
) -> DeltaDecision:
    """
    Post-processor decision using three structural signals.
    variants should contain alternative outputs under prompt reformatting.
    """
    instances = [draft] + list(variants or [])
    t = T_score(instances)
    i = I_score(draft)
    s = S_score(prompt, draft)

    reasons: List[str] = []
    if t >= t_hi:
        reasons.append("instability_high")
    elif t >= t_mid:
        reasons.append("instability_mid")

    if i >= i_hi:
        reasons.append("contradiction_high")

    if s >= s_hi:
        reasons.append("saturation_high")

    # Map to Δ(S)
    if i >= i_hi or t >= t_hi:
        B = Boundary.SATURATED
        L = Loss.CRITICAL
        C = Compressibility.NO
        action = Action.REFUSE
    elif s >= s_hi or t >= t_mid:
        B = Boundary.NEAR_BOUNDARY
        L = Loss.MEDIUM
        C = Compressibility.PARTIAL
        action = Action.BOUNDARY_ONLY
    else:
        B = Boundary.OPEN
        L = Loss.LOW
        C = Compressibility.YES
        action = Action.ALLOW

    return DeltaDecision(
        B=B, L=L, C=C, action=action,
        reasons=reasons,
        scores={"T": float(t), "I": float(i), "S": float(s)},
    )