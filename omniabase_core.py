"""
Omniabase — Rigid Truth Core
===========================

A statement is TRUE if and only if it holds across ALL bases.
If it fails in even one base, it is FALSE.

No probabilities.
No partial truth.
No averaging.

Omniabase = invariance across bases.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Any, Optional


@dataclass
class OmniabaseResult:
    statement: str
    bases_checked: Dict[str, bool]
    failed_bases: list
    verdict: str  # "TRUE" or "FALSE"


def omniabase_truth(
    statement: str,
    bases: Iterable[str],
    check_fn: Callable[[str, str, Optional[Dict[str, Any]]], bool],
    context: Optional[Dict[str, Any]] = None,
) -> OmniabaseResult:
    """
    Rigid Omniabase rule:

    TRUE  ⇔ statement holds in ALL bases
    FALSE ⇔ statement fails in AT LEAST ONE base
    """

    results: Dict[str, bool] = {}
    failed = []

    for base in bases:
        ok = bool(check_fn(statement, base, context))
        results[base] = ok
        if not ok:
            failed.append(base)

    verdict = "TRUE" if not failed else "FALSE"

    return OmniabaseResult(
        statement=statement,
        bases_checked=results,
        failed_bases=failed,
        verdict=verdict,
    )