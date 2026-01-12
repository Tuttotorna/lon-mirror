from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from omnia.irreversibility import IrreversibilityEngine


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    # Prefer uncertainty-aware report; fallback to basic report
    p = Path("data/sei_gsm8k_uncertainty_report.jsonl")
    if not p.exists():
        p = Path("data/sei_gsm8k_report.jsonl")
    if not p.exists():
        raise FileNotFoundError("No SEI report found in data/")

    eng = IrreversibilityEngine(window=7, flat_eps=0.03, detect_iri=0.25)

    out_path = Path("data/iri_from_sei_report.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as out:
        for rec in _iter_jsonl(p):
            sei = float(rec.get("sei", 0.0))
            tokens = float(rec.get("tokens_in", 0)) + float(rec.get("tokens_out", 0))
            latency = float(rec.get("latency_ms", 0.0))
            acc = float(rec.get("acc", 0.0))

            state = [sei, tokens, latency, acc]
            iri_rec = eng.add_state(context="gsm8k_path_from_sei", state=state)
            if iri_rec is None:
                continue

            out.write(json.dumps({
                "step": iri_rec.step,
                "iri": iri_rec.iri,
                "iri_z": iri_rec.iri_z,
                "trend": iri_rec.trend,
                "forward_distance": iri_rec.forward_distance,
                "hysteresis_residue": iri_rec.hysteresis_residue,
                "hysteresis_detected": iri_rec.hysteresis_detected
            }, ensure_ascii=False) + "\n")

    print(f"Wrote: {out_path}")
    print("Snapshot:", eng.snapshot())


if __name__ == "__main__":
    main()