"""
FACT_CHECK ENGINE v0.1 — OMNIA_TOTALE · fourth lens (facts)
Author: Massimiliano Brighindi (concepts) + MBX IA (formalization)

Goal:
    Provide a pluggable, model-agnostic factual consistency layer
    to sit on top of OMNIA_TOTALE (Omniabase + Omniatempo + Omniacausa).

    - PBII / OMNIA_TOTALE  → struttura, drift, instabilità.
    - FACT_CHECK ENGINE    → verità fattuale rispetto a dati esterni
                              o a "gold answers" (quando disponibili).

Dependencies:
    Standard library only (re, math, dataclasses, typing, statistics).
    Backends for real fact-checking must be plugged in separately.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Protocol, Any
import re
import math
import statistics


# ============================
# 1. DATA MODELS
# ============================

@dataclass
class FactEvidence:
    """
    One piece of evidence returned by a backend.
    label:
        "support" → evidence supports the claim
        "refute"  → evidence refutes the claim
        "unknown" → no clear stance
    weight:
        Relative strength / confidence of this evidence (0–1).
    """
    source: str
    label: str  # "support" | "refute" | "unknown"
    weight: float = 1.0
    meta: Optional[Dict[str, Any]] = None


@dataclass
class FactCheckClaim:
    """
    Single claim extracted from model output.
    text: raw text of the claim (short sentence or phrase)
    """
    text: str
    span_start: int
    span_end: int


@dataclass
class ClaimCheckResult:
    claim: FactCheckClaim
    evidence: List[FactEvidence]
    support_score: float   # in [0,1]
    refute_score: float    # in [0,1]
    uncertainty: float     # in [0,1]


@dataclass
class FactCheckSummary:
    """
    Global summary for one QA / chain-of-thought.
    """
    question: str
    model_answer: str
    gold_answer: Optional[str]
    claims: List[ClaimCheckResult]

    # Aggregate scores
    fact_consistency: float      # [0,1] factual agreement
    numeric_consistency: float   # [0,1] internal numeric coherence
    gold_match: Optional[float]  # 1.0 if exact, 0.0 if not, None if gold not provided

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


# ============================
# 2. BACKEND PROTOCOL
# ============================

class FactCheckBackend(Protocol):
    """
    Abstract interface for a factual backend.

    Implementations could:
        - call web APIs
        - query local knowledge bases
        - use dedicated retrieval+LLM checkers
    """
    def check_claim(self, claim_text: str) -> List[FactEvidence]:
        ...


# ============================
# 3. SIMPLE CLAIM EXTRACTION
# ============================

def split_into_sentences(text: str) -> List[str]:
    """
    Very simple sentence splitter, just for prototyping.
    For serious use, replace with proper NLP.
    """
    # Split on ., ?, ! while keeping it simple.
    parts = re.split(r'([\.!\?])', text)
    sentences: List[str] = []
    current = ""
    for piece in parts:
        if piece in ".!?":
            current += piece
            s = current.strip()
            if s:
                sentences.append(s)
            current = ""
        else:
            current += piece
    s = current.strip()
    if s:
        sentences.append(s)
    return [s for s in sentences if s]


def extract_claims(text: str, max_len: int = 200) -> List[FactCheckClaim]:
    """
    Naive claim extraction: split into sentences, keep non-trivial ones.
    """
    claims: List[FactCheckClaim] = []
    offset = 0
    for sent in split_into_sentences(text):
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) > max_len:
            # Long reasoning sentences: we still keep them as claims for now.
            pass
        start = text.find(sent, offset)
        if start == -1:
            start = offset
        end = start + len(sent)
        claims.append(FactCheckClaim(text=sent, span_start=start, span_end=end))
        offset = end
    return claims


# ============================
# 4. NUMERIC CONSISTENCY HELPERS
# ============================

_number_pattern = re.compile(r'[-+]?\d+(\.\d+)?')

def extract_numbers(text: str) -> List[float]:
    nums: List[float] = []
    for m in _number_pattern.finditer(text):
        try:
            nums.append(float(m.group()))
        except ValueError:
            continue
    return nums


def compare_numeric_strings(a: str, b: str) -> Optional[bool]:
    """
    Compare two strings as numbers if possible.
    Returns True / False / None (if non-numeric).
    """
    try:
        na = float(a.strip())
        nb = float(b.strip())
        return bool(math.isclose(na, nb, rel_tol=1e-9, abs_tol=1e-9))
    except ValueError:
        return None


def numeric_consistency_score(
    question: str,
    model_chain: str,
    model_answer: str,
) -> float:
    """
    Heuristic numeric consistency:
        - 1.0 if final answer is numeric and appears in chain numbers.
        - 0.5 if numeric but only appears in the answer.
        - 0.0 otherwise.

    This is intentionally simple; xAI team can replace with
    a full arithmetic checker.
    """
    nums_chain = extract_numbers(model_chain)
    cmp_res = compare_numeric_strings(model_answer, model_answer)
    # If answer is not numeric → no numeric consistency dimension.
    try:
        ans_val = float(model_answer.strip())
    except ValueError:
        return 0.0

    if any(math.isclose(ans_val, x, rel_tol=1e-9, abs_tol=1e-9) for x in nums_chain):
        return 1.0
    # At least numeric, but not obviously present in chain
    return 0.5


def gold_match_score(model_answer: str, gold_answer: Optional[str]) -> Optional[float]:
    """
    1.0 if model_answer == gold_answer (numeric or exact text match).
    0.0 if clearly different.
    None if gold_answer not provided.
    """
    if gold_answer is None:
        return None

    # Try numeric comparison first.
    num_cmp = compare_numeric_strings(model_answer, gold_answer)
    if num_cmp is True:
        return 1.0
    if num_cmp is False:
        return 0.0

    # Fallback: case-insensitive string equality.
    if model_answer.strip().lower() == gold_answer.strip().lower():
        return 1.0
    return 0.0


# ============================
# 5. MAIN FACT CHECK PIPELINE
# ============================

def aggregate_evidence(evidences: List[FactEvidence]) -> Dict[str, float]:
    """
    Aggregate evidence into support/refute/unknown scores in [0,1].
    """
    if not evidences:
        return {"support": 0.0, "refute": 0.0, "unknown": 1.0}

    total_weight = sum(ev.weight for ev in evidences) or 1.0
    support = sum(ev.weight for ev in evidences if ev.label == "support") / total_weight
    refute = sum(ev.weight for ev in evidences if ev.label == "refute") / total_weight
    unknown = sum(ev.weight for ev in evidences if ev.label == "unknown") / total_weight
    # Normalize again just to be safe
    s = support + refute + unknown
    if s <= 0:
        return {"support": 0.0, "refute": 0.0, "unknown": 1.0}
    return {"support": support / s, "refute": refute / s, "unknown": unknown / s}


def fact_check_chain(
    question: str,
    model_chain: str,
    model_answer: str,
    gold_answer: Optional[str] = None,
    backend: Optional[FactCheckBackend] = None,
    max_claims: int = 10,
) -> FactCheckSummary:
    """
    Core pipeline:

        1. Extract claims from chain + final answer.
        2. For each claim, ask backend (if provided) for evidence.
        3. Aggregate evidence into support/refute/uncertainty.
        4. Compute:
            - fact_consistency      (mean support_score)
            - numeric_consistency   (internal numeric coherence)
            - gold_match            (if gold_answer available)
    """
    # 1. Extract claims (chain + answer as last claim)
    full_text = model_chain.strip()
    claims = extract_claims(full_text)
    # Add final answer as explicit claim
    extra_claim = FactCheckClaim(
        text=f"Final answer: {model_answer.strip()}",
        span_start=len(full_text),
        span_end=len(full_text) + len(model_answer),
    )
    claims.append(extra_claim)

    # Limit claims for efficiency
    if len(claims) > max_claims:
        claims = claims[:max_claims]

    claim_results: List[ClaimCheckResult] = []

    # 2–3. Backend evidence + aggregation
    for cl in claims:
        evidences: List[FactEvidence] = []
        if backend is not None:
            try:
                evidences = backend.check_claim(cl.text)
            except Exception as e:
                # Backend failure → treat as unknown
                evidences = []
        agg = aggregate_evidence(evidences)
        claim_results.append(
            ClaimCheckResult(
                claim=cl,
                evidence=evidences,
                support_score=agg["support"],
                refute_score=agg["refute"],
                uncertainty=agg["unknown"],
            )
        )

    # 4a. fact_consistency = mean support over claims
    if claim_results:
        fact_consistency = statistics.mean(cr.support_score for cr in claim_results)
    else:
        fact_consistency = 0.0

    # 4b. numeric_consistency
    num_cons = numeric_consistency_score(question, model_chain, model_answer)

    # 4c. gold_match
    gm = gold_match_score(model_answer, gold_answer)

    return FactCheckSummary(
        question=question,
        model_answer=model_answer,
        gold_answer=gold_answer,
        claims=claim_results,
        fact_consistency=fact_consistency,
        numeric_consistency=num_cons,
        gold_match=gm,
    )


# ============================
# 6. INTEGRAZIONE CON OMNIA_TOTALE
# ============================

@dataclass
class OmniaFactFusion:
    """
    Minimal fusion layer for OMNIA_TOTALE + FACT_CHECK ENGINE.
    """
    omega_struct: float         # Ω from OMNIA_TOTALE (struttura)
    fact_consistency: float     # [0,1]
    numeric_consistency: float  # [0,1]
    gold_match: Optional[float]
    fused_score: float          # Ω_ext = Ω + w_f * (fact + numeric + gold...)


def fuse_omnia_with_factcheck(
    omega_struct: float,
    fact_summary: FactCheckSummary,
    w_fact: float = 1.0,
    w_numeric: float = 0.5,
    w_gold: float = 1.0,
) -> OmniaFactFusion:
    """
    Combine structural Ω with factual scores.

    Example fusion:
        Ω_ext = Ω_struct
                + w_fact    * fact_consistency
                + w_numeric * numeric_consistency
                + w_gold    * gold_match (if available)
    """
    extra = 0.0
    extra += w_fact * fact_summary.fact_consistency
    extra += w_numeric * fact_summary.numeric_consistency
    if fact_summary.gold_match is not None:
        extra += w_gold * fact_summary.gold_match

    fused = omega_struct + extra

    return OmniaFactFusion(
        omega_struct=omega_struct,
        fact_consistency=fact_summary.fact_consistency,
        numeric_consistency=fact_summary.numeric_consistency,
        gold_match=fact_summary.gold_match,
        fused_score=fused,
    )


# ============================
# 7. DEMO LOCALE (no backend)
# ============================

def demo():
    """
    Minimal demo without external backend:
        - Only numeric_consistency + gold_match.
    """
    question = "If John has 5 apples and buys 7 more, how many apples does he have?"
    model_chain = (
        "John starts with 5 apples. "
        "He buys 7 more apples, so we add 5 + 7 = 12. "
        "Therefore, he now has 12 apples."
    )
    model_answer = "12"
    gold_answer = "12"

    summary = fact_check_chain(
        question=question,
        model_chain=model_chain,
        model_answer=model_answer,
        gold_answer=gold_answer,
        backend=None,  # no external facts
    )

    omega_struct = 0.5  # esempio: Ω da OMNIA_TOTALE
    fused = fuse_omnia_with_factcheck(omega_struct, summary)

    print("=== FACT_CHECK ENGINE v0.1 demo ===")
    print("Fact consistency:", summary.fact_consistency)
    print("Numeric consistency:", summary.numeric_consistency)
    print("Gold match:", summary.gold_match)
    print("Ω_struct:", fused.omega_struct, "→ Ω_ext:", fused.fused_score)


if __name__ == "__main__":
    demo()