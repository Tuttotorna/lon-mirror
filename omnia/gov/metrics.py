from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import math

from .types import ActionBundle, OmniaMetrics, WorldProxy
from .invariants import extract_invariants, jaccard

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def compute_cvi(action: ActionBundle) -> float:
    plan_inv = extract_invariants(action.plan)
    cons_text = [c.payload for c in action.external_constraints]
    cons_inv = extract_invariants(cons_text)
    overlap = jaccard(plan_inv, cons_inv)
    return clamp(1.0 - overlap)

def compute_iri_act(world: Optional[WorldProxy],
                    w1: float = 0.45, w2: float = 0.30, w3: float = 0.25) -> float:
    if world is None:
        return 0.0
    # normalize irreversible ops with soft cap
    ops_norm = 1.0 - math.exp(-world.irreversible_ops / 5.0)
    return clamp(w1 * ops_norm + w2 * clamp(world.rollback_cost) + w3 * clamp(world.blast_radius))

def compute_hci(m: OmniaMetrics,
                sei_floor: float = 0.05,
                delta_floor: float = 0.02,
                iri_floor: float = 0.35,
                a: float = 6.0, b: float = 6.0, c: float = 4.0) -> float:
    # High when: sei small, delta small, iri not small
    x = a * (sei_floor - m.sei) + b * (delta_floor - m.delta_omega) + c * (m.iri - iri_floor)
    return clamp(sigmoid(x))