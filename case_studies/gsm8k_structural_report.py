from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Import OMNIA core pieces (expected to exist in repo)
from omnia.omega import OmegaEstimator
from omnia.engine.superposition import SuperpositionKernel
from omnia.lenses.compression import CompressionLens
from omnia.lenses.permutation import PermutationLens
from omnia.lenses.constraints import ConstraintLens
from omnia.omega_set import omega_set_from_values  # if your repo uses different API, adjust import
from omnia.sei import sei_curve_from_omega          # idem: adjust to your actual function names if needed
from omnia.iri import iri_from_cycle                # idem: adjust to your actual function names if needed


@dataclass
class Record:
    id: str
    prompt: str
    output: str


def _load_jsonl(path: str, limit: Optional[int] = None) -> List[Record]:
    out: List[Record] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.append(
                Record(
                    id=str(obj.get("id", i)),
                    prompt=str(obj.get("prompt", "")),
                    output=str(obj.get("output", obj.get("completion", obj.get("text", "")))),
                )
            )
    return out


def _build_estimator() -> OmegaEstimator:
    # Minimal lens schedule: compression + permutation + constraints
    # Deterministic: no randomness, no semantics.
    kernel = SuperpositionKernel(distance_fn=None)  # repo-dependent; if distance_fn required, use simple_text_distance
    lenses = [
        CompressionLens(),
        PermutationLens(),
        ConstraintLens(),
    ]
    return OmegaEstimator(kernel=kernel, lenses=lenses)


def _measure_one(est: OmegaEstimator, text: str) -> Dict[str, Any]:
    # Expect estimator to return something like {"omega":..., "omega_values":[...], ...}
    r = est.measure(text)
    # Best-effort normalization over common shapes
    omega = getattr(r, "omega", None)
    if omega is None and isinstance(r, dict):
        omega = r.get("omega", r.get("Omega", 0.0))
    omega_values = getattr(r, "omega_values", None)
    if omega_values is None and isinstance(r, dict):
        omega_values = r.get("omega_values", r.get("omegas", []))
    return {"omega": float(omega or 0.0), "omega_values": list(omega_values or [])}


def main() -> None:
    here = os.path.dirname(__file__)
    # Use existing repo file if present; otherwise you can drop a small sample in case_studies/data/
    candidates = [
        os.path.join(here, "..", "data", "gsm8k_model_outputs.jsonl"),
        os.path.join(here, "data", "gsm8k_sample.jsonl"),
    ]
    path = None
    for p in candidates:
        if os.path.exists(p):
            path = p
            break
    if path is None:
        raise FileNotFoundError("No GSM8K JSONL found. Put one at data/gsm8k_model_outputs.jsonl or case_studies/data/gsm8k_sample.jsonl")

    rows = _load_jsonl(path, limit=25)

    est = _build_estimator()

    omegas: List[float] = []
    per: List[Tuple[str, float]] = []

    for rec in rows:
        m = _measure_one(est, rec.output)
        o = float(m["omega"])
        omegas.append(o)
        per.append((rec.id, o))

    # Omega-set (residual invariance) from aggregate omegas
    # NOTE: adjust this call to your actual omega_set API if needed.
    omega_hat = omega_set_from_values(omegas)

    # Basic SEI curve over omega list with unit cost (monotonic)
    sei = sei_curve_from_omega(omegas)

    print("OMNIA Case Study 01 — GSM8K Structural Report")
    print("Input:", os.path.normpath(path))
    print("N:", len(omegas))
    print()
    print("Top 10 Ω (most structurally coherent):")
    for rid, o in sorted(per, key=lambda x: x[1], reverse=True)[:10]:
        print(" ", rid, "->", round(o, 6))
    print()
    print("Bottom 10 Ω (most fragile):")
    for rid, o in sorted(per, key=lambda x: x[1])[:10]:
        print(" ", rid, "->", round(o, 6))
    print()
    print("Ω̂ (Omega-set / residual invariance):", omega_hat)
    print("SEI (trend sample):", [round(x, 6) for x in sei[: min(10, len(sei))]])
    print()
    print("Interpretation-free note:")
    print("If Ω̂ stabilizes while SEI → 0, additional processing yields no new structure.")
    print("This is a measurement boundary, not a semantic judgment.")


if __name__ == "__main__":
    main()