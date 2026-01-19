from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from omnia.lenses.aperspective_invariance import AperspectiveInvariance, Transform
from omnia.meta.collapsator_total import CollapsatorTotal, default_ct


@dataclass(frozen=True)
class CycleResult:
    """
    OMNIA↔CT Cycle (OC-1.0)

    omega_curve:
        Ω_ap at each iteration state.

    ctc_curve:
        Residue overlap vs initial residue, per iteration.

    converged:
        True if residues stabilize (fixed point).

    collapsed:
        True if Ω_ap drops below eps.

    steps:
        Per-iteration minimal diagnostics.
    """
    omega_curve: List[float]
    ctc_curve: List[float]
    converged: bool
    collapsed: bool
    steps: List[Dict[str, float]]


class OMNIACTCycle:
    """
    OMNIA↔CT Cycle (OC-1.0)

    Iteration:
      s0 = x
      s1 = CT(s0)                # maximal collapse
      s2 = (aperspective residue of s1) measured on representation (still meaning-blind)
      s3 = CT(s2) ...
    In practice:
      we keep state as raw string, but measure residues each time and check stability.

    Convergence criterion:
      residue overlap between consecutive steps >= conv_thr
      and omega change <= omega_tol
    """

    def __init__(
        self,
        *,
        aperspective: AperspectiveInvariance,
        ct: CollapsatorTotal,
        eps: float = 1e-4,
        residue_cap: int = 200,
        conv_thr: float = 0.95,
        omega_tol: float = 1e-6,
        max_iters: int = 8,
    ):
        self.ap = aperspective
        self.ct = ct
        self.eps = float(eps)
        self.residue_cap = int(residue_cap)
        self.conv_thr = float(conv_thr)
        self.omega_tol = float(omega_tol)
        self.max_iters = int(max_iters)

    def _omega(self, s: str) -> float:
        return float(self.ap.measure(s).omega_score)

    def _residue(self, s: str) -> List[str]:
        r = self.ap.measure(s)
        return list(r.residue)[: self.residue_cap]

    @staticmethod
    def _overlap(a: Sequence[str], b: Sequence[str]) -> float:
        if not a:
            return 0.0
        sa = set(a)
        sb = set(b)
        return float(len(sa & sb) / max(1, len(sa)))

    def run(self, x: str) -> CycleResult:
        omega_curve: List[float] = []
        ctc_curve: List[float] = []
        steps: List[Dict[str, float]] = []

        r0 = self._residue(x)
        o0 = self._omega(x)

        omega_curve.append(o0)
        ctc_curve.append(1.0)

        prev_res = r0
        prev_omega = o0
        state = x

        converged = False
        collapsed = False

        for i in range(self.max_iters):
            # Apply CT
            ct_r = self.ct.collapse(state)
            state = ct_r.y_star

            omega = self._omega(state)
            res = self._residue(state)

            if omega <= self.eps:
                collapsed = True

            ov = self._overlap(r0, res)  # overlap vs initial residue (anchor)
            omega_curve.append(float(omega))
            ctc_curve.append(float(ov))

            # step diagnostics
            step = {
                "iter": float(i + 1),
                "omega": float(omega),
                "overlap_vs_initial": float(ov),
                "res_n": float(len(res)),
                "ct_steps": float(ct_r.steps),
                "ct_collapsed": 1.0 if ct_r.collapsed else 0.0,
            }
            steps.append(step)

            # Convergence test (consecutive)
            ov_prev = self._overlap(prev_res, res)
            if (ov_prev >= self.conv_thr) and (abs(prev_omega - omega) <= self.omega_tol):
                converged = True
                break

            prev_res = res
            prev_omega = omega

            if collapsed:
                break

        return CycleResult(
            omega_curve=omega_curve,
            ctc_curve=ctc_curve,
            converged=bool(converged),
            collapsed=bool(collapsed),
            steps=steps,
        )


def default_cycle() -> OMNIACTCycle:
    ct = default_ct()

    # Must match CT baseline
    baseline: List[Tuple[str, Transform]] = [
        ("id", ct.ap.transforms[0][1]),
        ("ws", ct.ap.transforms[1][1]),
        ("rev", ct.ap.transforms[2][1]),
        ("vow-", ct.ap.transforms[3][1]),
        ("shuf", ct.ap.transforms[4][1]),
    ]

    ap = AperspectiveInvariance(transforms=baseline)

    return OMNIACTCycle(
        aperspective=ap,
        ct=ct,
        eps=1e-4,
        residue_cap=200,
        conv_thr=0.95,
        omega_tol=1e-6,
        max_iters=8,
    )