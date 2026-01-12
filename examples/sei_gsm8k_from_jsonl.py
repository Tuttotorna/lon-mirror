from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Iterator, Optional

from omnia.sei import SEIEngine


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _get_text(d: Dict[str, Any], *keys: str) -> str:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def _get_bool(d: Dict[str, Any], *keys: str) -> Optional[bool]:
    for k in keys:
        v = d.get(k)
        if isinstance(v, bool):
            return v
    return None


def _approx_tokens(text: str) -> int:
    # cheap proxy: ~1 token per ~0.75 words is typical, but we keep it simple:
    # words count is stable enough for trend tracking.
    if not text:
        return 0
    return max(1, len(text.split()))


def main():
    jsonl_path = Path("data/gsm8k_model_outputs.jsonl")
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Missing {jsonl_path}. Place your GSM8K outputs there.")

    sei = SEIEngine(
        # Layer-1: keep weights neutral. Energy/latency absent => tokens dominate cost.
        w_quality=1.0,
        w_uncertainty=0.0,
        v_energy=0.0,
        v_tokens=1.0,
        v_latency=0.0,
        v_iterations=0.0,
        window=5,
        flat_eps=0.03,
    )

    n = 0
    correct = 0
    last_acc = 0.0

    out_path = Path("data/sei_gsm8k_report.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as out:
        for rec in _iter_jsonl(jsonl_path):
            n += 1

            # Try common field names; adapt if your JSONL differs
            prompt = _get_text(rec, "prompt", "question", "input")
            output = _get_text(rec, "output", "model_output", "completion", "response")
            is_correct = _get_bool(rec, "is_correct", "correct", "passed")

            # If correctness is missing, we cannot compute a meaningful SEI.
            # Fail hard: Layer-1 must be measurable.
            if is_correct is None:
                raise KeyError(
                    "Missing correctness flag in JSONL record. "
                    "Expected one of: is_correct / correct / passed."
                )

            if is_correct:
                correct += 1

            acc = correct / n
            delta_quality = acc - last_acc
            last_acc = acc

            tokens_in = _approx_tokens(prompt)
            tokens_out = _approx_tokens(output)

            sei_rec = sei.add(
                context="gsm8k_jsonl_layer1",
                delta_quality=float(delta_quality),
                delta_uncertainty=0.0,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=0.0,
                energy_joule=None,
                iterations=1,
            )

            out.write(json.dumps({
                "n": n,
                "acc": acc,
                "delta_quality": delta_quality,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "sei": sei_rec.sei,
                "sei_z": sei_rec.sei_z,
                "trend": sei_rec.trend
            }, ensure_ascii=False) + "\n")

    print(f"Wrote: {out_path}")
    print(f"Final accuracy: {correct}/{n} = {correct/n:.4f}")
    print("Last window snapshot:", sei.snapshot())


if __name__ == "__main__":
    main()