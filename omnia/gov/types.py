from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Literal


class Decision(str, Enum):
    ALLOW = "ALLOW"
    BOUNDARY_ONLY = "BOUNDARY_ONLY"
    REFUSE = "REFUSE"


class Boundary(str, Enum):
    OPEN = "open"
    NEAR = "near_boundary"
    SATURATED = "saturated"


@dataclass(frozen=True)
class OmniaMetrics:
    omega: float
    delta_omega: float
    sei: float
    iri: float
    omega_hat: List[str]
    sci: Optional[float] = None


@dataclass(frozen=True)
class ExternalConstraint:
    type: str
    payload: str


@dataclass(frozen=True)
class ActionBundle:
    intent: str
    plan: Any  # str or list[str]; opaque
    resources: Dict[str, Any]
    expected_effects: List[str]
    external_constraints: List[ExternalConstraint]


@dataclass(frozen=True)
class WorldProxy:
    irreversible_ops: int = 0
    rollback_cost: float = 0.0   # 0..1
    blast_radius: float = 0.0    # 0..1


@dataclass(frozen=True)
class Certificate:
    type: Literal["SNRC-ACT"]
    issued: bool
    notes: str = ""


@dataclass(frozen=True)
class GovDecision:
    decision: Decision
    boundary: Boundary
    reasons: List[str]
    scores: Dict[str, float]
    certificate: Optional[Certificate] = None