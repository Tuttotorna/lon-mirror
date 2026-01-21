from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple
import math


# A representation is opaque: we never inspect "content"
Representation = Iterable[float]

# A relation extracts a scalar invariant from a representation
Relation = Callable[[Representation], float]


def _almost_equal(a: float, b: float, eps: float = 1e-9) -> bool:
    return abs(a - b) <= eps


@dataclass(frozen=True)
class IndistinguishabilityResult:
    indistinguishable: bool
    distance: float
    relations: Dict[str, Tuple[float, float]]


class StructuralIndistinguishability:
    """
    Determines whether two internal codifications are distinguishable
    given ONLY structural relations.

    Core OMNIA rule:
      If all observable relations are invariant,
      internal codifications are undecidable.

    No semantics.
    No interpretation.
    No access to internal meaning.
    """

    def __init__(self, relations: List[Tuple[str, Relation]]) -> None:
        if not relations:
            raise ValueError("At least one structural relation is required")
        self.relations = relations

    def compare(
        self,
        rep_a: Representation,
        rep_b: Representation,
    ) -> IndistinguishabilityResult:
        rel_values: Dict[str, Tuple[float, float]] = {}
        deltas: List[float] = []

        for name, rel in self.relations:
            va = float(rel(rep_a))
            vb = float(rel(rep_b))
            rel_values[name] = (va, vb)
            deltas.append(abs(va - vb))

        # Structural distance (L1 over relations)
        distance = sum(deltas)

        indistinguishable = all(_almost_equal(a, b) for a, b in rel_values.values())

        return IndistinguishabilityResult(
            indistinguishable=indistinguishable,
            distance=distance,
            relations=rel_values,
        )
# Example relations (semantics-free)
def mean(x): return sum(x) / len(x)
def variance(x):
    m = mean(x)
    return sum((v - m) ** 2 for v in x) / len(x)

relations = [
    ("mean", mean),
    ("variance", variance),
]

engine = StructuralIndistinguishability(relations)

# Due "codifiche" diverse ma strutturalmente isomorfe
rep_A = [1, 2, 3, 4]
rep_B = [101, 102, 103, 104]

result = engine.compare(rep_A, rep_B)

# result.indistinguishable == True