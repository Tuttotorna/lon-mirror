from __future__ import annotations

from typing import Callable

Observer = Callable[[str], str]

def is_observer_transform(
    *,
    before: str,
    after: str,
) -> bool:
    """
    Observer model (OMNIA strict definition).

    A transformation is considered an 'observer' iff:
    - it is not bijective in practice
    - it introduces asymmetry or preference
    - it cannot be reversed without loss

    No semantics are inspected.
    This is a purely structural gate.
    """
    if before == after:
        return False

    # crude but strict structural signals
    if len(after) != len(before):
        return True

    if before in after or after in before:
        return True

    return True