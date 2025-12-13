"""
OMNIA — LLM Output Adapter (raw)

Purpose:
- Measure structural stability of LLM outputs
- No policy, no decision, no prompting logic
"""

from typing import Any, Dict
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMOmniaReport:
    omega_score: float
    flags: list
    components: Dict[str, float]


def analyze_llm_output(
    text: str,
    *,
    tokens: list | None = None,
    series: list | None = None,
    series_dict: dict | None = None,
) -> LLMOmniaReport:
    """
    Raw adapter for LLM outputs.

    Inputs are optional and independent:
    - text: model final output
    - tokens: token ids or token proxies
    - series: scalar reasoning trace (optional)
    - series_dict: multi-channel traces (optional)
    """
    from omnia.engine.omnia_totale.kernel import truth_omega

    payload: Dict[str, Any] = {
        "text": text,
        "tokens": tokens,
        "series": series,
        "series_dict": series_dict,
    }

    r = truth_omega(payload)

    # Normalize output
    score = float(getattr(r, "score", getattr(r, "omega_score", 0.0)))
    flags = list(getattr(r, "flags", []))
    components = dict(getattr(r, "components", getattr(r, "metrics", {})))

    return LLMOmniaReport(
        omega_score=score,
        flags=flags,
        components=components,
    )