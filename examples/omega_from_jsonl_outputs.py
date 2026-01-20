from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

from omnia.engine.superposition import SuperpositionKernel
from omnia.omega import OmegaEstimator
from omnia.lenses.compression import CompressionLens
from omnia.lenses.permutation import PermutationLens
from omnia.lenses.constraints import ConstraintLens

from omnia.inference import StructuralSignature, classify_state, InferenceTrajectory


def _load_jsonl(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _get_text(rec: Dict[str, Any], field: str) -> str:
    # dotted access: "a.b.c"
    cur: Any = rec
    for part in field.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return ""
    return "" if cur is None else str(cur)


def _safe_var(xs: List[float]) -> float:
    if not xs:
        return 0.0
    if len(xs) == 1:
        return 0.0
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def _order_sensitivity(omegas_window: List[float]) -> float:
    # proxy: variance of recent Ω values, scaled to [0,1]
    v = _safe_var(omegas_window)
    return min(1.0, v / 0.05)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="Path to JSONL outputs file")
    ap.add_argument(
        "--text-field",
        default="output",
        help='Field containing model output text. Supports dotted access, e.g. "response.text"',
    )
    ap.add_argument("--limit", type=int, default=None, help="Limit number of records")
    ap.add_argument(
        "--out-jsonl",
        default=None,
        help="Optional path to write telemetry JSONL (one line per record)",
    )
    args = ap.parse_args()

    records = _load_jsonl(args.jsonl, limit=args.limit)

    # Core OMNIA pieces already in repo
    lenses = [
        CompressionLens(),
        PermutationLens(),
        ConstraintLens(),
    ]
    kernel = SuperpositionKernel(lenses=lenses)
    estimator = OmegaEstimator(kernel=kernel)

    traj = InferenceTrajectory()
    omega_hist: List[float] = []
    telemetry: List[Dict[str, Any]] = []

    for i, rec in enumerate(records):
        text = _get_text(rec, args.text_field)

        # Compute Ω and related metrics via OmegaEstimator
        # IMPORTANT: this assumes OmegaEstimator exposes a method that returns omega + sei.
        # If your OmegaEstimator returns a dict, this code supports both patterns below.
        res = estimator.estimate(text)

        # Support both object-like and dict-like results
        if isinstance(res, dict):
            omega = float(res.get("omega", 0.0))
            sei = float(res.get("sei", 0.0))
            drift = float(res.get("drift", 0.0))
            drift_vector = float(res.get("drift_vector", 0.0))
        else:
            omega = float(getattr(res, "omega", 0.0))
            sei = float(getattr(res, "sei", 0.0))
            drift = float(getattr(res, "drift", 0.0))
            drift_vector = float(getattr(res, "drift_vector", 0.0))

        omega_hist.append(omega)

        sig = StructuralSignature(
            omega=omega,
            omega_variance=_safe_var(omega_hist[-5:]),
            sei=sei,
            drift=drift,
            drift_vector=drift_vector,
            order_sensitivity=_order_sensitivity(omega_hist[-5:]),
        )

        state = classify_state(sig)

        row = {
            "step": i,
            "omega": omega,
            "sei": sei,
            "drift": drift,
            "drift_vector": drift_vector,
            "omega_var_5": sig.omega_variance,
            "order_sens_5": sig.order_sensitivity,
            "state": state.name,
        }

        traj.append(state, record=row)
        telemetry.append(row)

    print("INFERENCE_TRAJECTORY:", [s.name for s in traj.states])

    if args.out_jsonl:
        os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
        with open(args.out_jsonl, "w", encoding="utf-8") as f:
            for r in telemetry:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())