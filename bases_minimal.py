"""
Omniabase — Minimal Base Checkers (example)

Each base is an independent veto.
If any base returns False, Omniabase verdict becomes FALSE.

These are minimal placeholders to demonstrate the rigid Omniabase law.
Replace/extend them with stricter rules over time.
"""

from typing import Optional, Dict, Any


def check_base(statement: str, base: str, context: Optional[Dict[str, Any]] = None) -> bool:
    ctx = context or {}

    # Base 1: NON-CONTRADICTION / POSSIBILITY (hard veto)
    if base == "POSSIBILITY":
        # If explicitly impossible -> fail
        if ctx.get("explicit_impossibility") is True:
            return False
        # If depends on unobservable/inaccessible condition -> fail for strict truth
        if ctx.get("depends_on_unobservable") is True:
            return False
        return True

    # Base 2: CAUSALITY (hard veto if causal claim lacks mechanism)
    if base == "CAUSALITY":
        if ctx.get("causation_claim") is True and ctx.get("mechanism_provided") is not True:
            return False
        return True

    # Base 3: CONTEXT (hard veto if critical premise missing)
    if base == "CONTEXT":
        if ctx.get("missing_critical_premise") is True:
            return False
        return True

    # Unknown base -> conservative fail
    return False