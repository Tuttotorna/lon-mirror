# omnia/meta/structural_compatibility.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _as_float(d: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    v = d.get(key, default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _normalize_score(x: float) -> float:
    """
    Accepts any real-valued metric and maps it to [0,1] with conservative clipping.
    If your upstream metrics are already in [0,1], this is identity.
    """
    return _clip01(float(x))


@dataclass(frozen=True)
class CompatibilityResult:
    """
    SCI-1.0 (Structural Compatibility Index)

    Goal:
    - Measure whether multiple lens-outputs can co-exist without producing
      internal contradictions (as a structural condition, not semantic).

    Inputs expected (minimal):
    - aperspective: omega_score in [0,1] (Ω_ap)
    - saturation:  is_saturated (bool), c_star (float), optional omega_curve, sei_curve
    - irreversibility: iri (float >= 0) or is_irreversible (bool)
    - redundancy: is_redundant (bool), collapse_cost (float)
    - distribution: min_score in [0,1] (or mean_score), is_invariant (bool)
    - nondecision: mean_score in [0,1], dispersion (>=0), is_nondecisional (bool)

    Output:
    - sci: compatibility in [0,1]
    - is_compatible: decisionless certificate with conservative thresholds
    - zone: coarse topology label (STABLE / TENSE / FRAGILE / IMPOSSIBLE)
    - tensions: contribution map to explain the SCI without narrative
    """
    sci: float
    is_compatible: bool
    zone: str
    tensions: Dict[str, float]
    diagnostics: Dict[str, float]


class StructuralCompatibility:
    """
    Structural Compatibility Index (SCI-1.0)

    This is not a new "lens on data".
    It is a lens on the *outputs of lenses*.

    It measures whether the ensemble of measured properties forms a coherent
    co-existing state, without privileging any observer narrative.

    Conservative logic:
    - Irreversibility is treated as a hard structural penalty.
    - Saturation + low aperspective invariance indicates "no stable residue".
    - High non-decision dispersion indicates internal instability across paths.
    - Distribution invariance helps stabilize non-local structure.
    - Redundancy buffers collapse but cannot override irreversibility.
    """

    def __init__(
        self,
        *,
        compatible_floor: float = 0.70,
        stable_floor: float = 0.80,
        tense_floor: float = 0.65,
        iri_hard: float = 0.15,
        dispersion_hard: float = 0.08,
    ):
        self.compatible_floor = float(compatible_floor)
        self.stable_floor = float(stable_floor)
        self.tense_floor = float(tense_floor)
        self.iri_hard = float(iri_hard)
        self.dispersion_hard = float(dispersion_hard)

    def measure(
        self,
        *,
        aperspective: Optional[Mapping[str, Any]] = None,
        saturation: Optional[Mapping[str, Any]] = None,
        irreversibility: Optional[Mapping[str, Any]] = None,
        redundancy: Optional[Mapping[str, Any]] = None,
        distribution: Optional[Mapping[str, Any]] = None,
        nondecision: Optional[Mapping[str, Any]] = None,
    ) -> CompatibilityResult:
        ap = aperspective or {}
        sa = saturation or {}
        ir = irreversibility or {}
        rd = redundancy or {}
        di = distribution or {}
        nd = nondecision or {}

        # --- pull normalized core signals ---
        omega_ap = _normalize_score(_as_float(ap, "omega_score", _as_float(ap, "omega_ap", 0.0)))
        di_min = _normalize_score(_as_float(di, "min_score", _as_float(di, "mean_score", 0.0)))
        nd_mean = _normalize_score(_as_float(nd, "mean_score", 0.0))
        nd_disp = max(0.0, _as_float(nd, "dispersion", 0.0))

        # irreversibility: accept either iri float or is_irreversible bool
        iri = max(0.0, _as_float(ir, "iri", 0.0))
        if bool(ir.get("is_irreversible", False)) and iri <= 0.0:
            iri = max(iri, self.iri_hard)

        # saturation: accept bool and c_star
        is_sat = bool(sa.get("is_saturated", False))
        c_star = _as_float(sa, "c_star", -1.0)

        # redundancy: accept is_redundant and collapse_cost
        is_red = bool(rd.get("is_redundant", False))
        collapse_cost = _as_float(rd, "collapse_cost", -1.0)

        # --- compute structural tensions (penalties) ---
        tensions: Dict[str, float] = {}

        # T1: irreversibility (hard penalty)
        # map iri to [0,1] penalty: 0 if iri=0, ->1 as iri grows beyond iri_hard*2
        t_iri = _clip01(iri / max(1e-9, (2.0 * self.iri_hard)))
        tensions["irreversibility"] = t_iri

        # T2: saturation with weak residue (saturated AND low omega_ap)
        t_sat_weak = 0.0
        if is_sat:
            # if saturated, demand residue: omega_ap should be high; low omega_ap => penalty
            t_sat_weak = _clip01((0.80 - omega_ap) / 0.80)
        tensions["saturation_weak_residue"] = t_sat_weak

        # T3: nondecision dispersion (paths disagree => instability)
        t_disp = _clip01(nd_disp / max(1e-9, self.dispersion_hard))
        tensions["path_dispersion"] = t_disp

        # T4: low distribution invariance (non-local instability)
        t_di = _clip01((0.75 - di_min) / 0.75)
        tensions["low_distribution_invariance"] = t_di

        # T5: low aperspective invariance (no representation-invariant core)
        t_ap = _clip01((0.70 - omega_ap) / 0.70)
        tensions["low_aperspective_invariance"] = t_ap

        # --- buffers (reduce penalties but never override irreversibility) ---
        # redundancy provides a buffer against collapse and local brittleness
        # it can reduce SAT/DISP/DI/AP penalties slightly, but not irreversibility.
        buffer = 0.0
        if is_red:
            # if collapse_cost exists and is high, stronger buffer; else modest
            # map collapse_cost into [0,0.15] buffer
            if collapse_cost > 0:
                buffer = _clip01(collapse_cost / 6.0) * 0.15
            else:
                buffer = 0.08

        # apply buffer to non-iri tensions
        for k in list(tensions.keys()):
            if k == "irreversibility":
                continue
            tensions[k] = _clip01(tensions[k] - buffer)

        # --- aggregate SCI ---
        # Weighted penalty sum (irreversibility dominates)
        w = {
            "irreversibility": 0.35,
            "saturation_weak_residue": 0.18,
            "path_dispersion": 0.17,
            "low_distribution_invariance": 0.15,
            "low_aperspective_invariance": 0.15,
        }
        penalty = 0.0
        for k, wk in w.items():
            penalty += wk * tensions.get(k, 0.0)

        sci = _clip01(1.0 - penalty)

        # --- zone classification ---
        # IMPOSSIBLE if irreversibility is strong AND saturated weak residue
        impossible = (iri >= self.iri_hard) and is_sat and (omega_ap < 0.40)
        if impossible:
            zone = "IMPOSSIBLE"
        elif sci >= self.stable_floor and iri < self.iri_hard and nd_disp <= self.dispersion_hard:
            zone = "STABLE"
        elif sci >= self.tense_floor:
            zone = "TENSE"
        else:
            zone = "FRAGILE"

        is_compatible = (zone in {"STABLE", "TENSE"}) and (sci >= self.compatible_floor) and (not impossible)

        diagnostics = {
            "omega_ap": omega_ap,
            "di_min": di_min,
            "nd_mean": nd_mean,
            "nd_disp": nd_disp,
            "iri": iri,
            "is_saturated": 1.0 if is_sat else 0.0,
            "c_star": c_star,
            "is_redundant": 1.0 if is_red else 0.0,
            "collapse_cost": collapse_cost,
            "buffer": buffer,
            "penalty": penalty,
            "compatible_floor": self.compatible_floor,
            "stable_floor": self.stable_floor,
            "tense_floor": self.tense_floor,
            "iri_hard": self.iri_hard,
            "dispersion_hard": self.dispersion_hard,
        }

        return CompatibilityResult(
            sci=sci,
            is_compatible=is_compatible,
            zone=zone,
            tensions=tensions,
            diagnostics=diagnostics,
        )


# -------------------------
# Minimal demo (standalone)
# -------------------------
if __name__ == "__main__":
    sci = StructuralCompatibility()

    # Example: stable coexistence (high aperspective, good distribution, low iri, low dispersion)
    r1 = sci.measure(
        aperspective={"omega_score": 0.82},
        saturation={"is_saturated": False, "c_star": 5.0},
        irreversibility={"iri": 0.0},
        redundancy={"is_redundant": True, "collapse_cost": 5.0},
        distribution={"min_score": 0.86},
        nondecision={"mean_score": 0.80, "dispersion": 0.03},
    )
    print("SCI-1.0 demo / case 1")
    print(" zone:", r1.zone, "sci:", round(r1.sci, 4), "compatible:", r1.is_compatible)
    print(" tensions:", {k: round(v, 4) for k, v in r1.tensions.items()})

    # Example: impossible zone (irreversibility + saturated + weak residue)
    r2 = sci.measure(
        aperspective={"omega_score": 0.22},
        saturation={"is_saturated": True, "c_star": 2.0},
        irreversibility={"iri": 0.22},
        redundancy={"is_redundant": False, "collapse_cost": -1.0},
        distribution={"min_score": 0.30},
        nondecision={"mean_score": 0.55, "dispersion": 0.12},
    )
    print("\nSCI-1.0 demo / case 2")
    print(" zone:", r2.zone, "sci:", round(r2.sci, 4), "compatible:", r2.is_compatible)
    print(" tensions:", {k: round(v, 4) for k, v in r2.tensions.items()})