from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

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


def _get_num(d: Dict[str, Any], *keys: str) -> Optional[float]:
    for k in keys:
        v = d.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _approx_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text.split()))


def _normalize_answer(text: str) -> str:
    # crude normalization for agreement proxy (GSM8K final numeric)
    t = text.strip().lower()
    # keep digits, minus, dot
    filtered = []
    for ch in t:
        if ch.isdigit() or ch in ".-":
            filtered.append(ch)
        elif filtered:
            # stop after first numeric blob
            break
    return "".join(filtered) if filtered else t[:80]


def _agreement_ratio(candidates: List[str]) -> float:
    """
    Returns ratio in [0,1] of the majority answer frequency.
    Higher ratio => lower uncertainty.
    """
    if not candidates:
        return 0.0
    norm = [_normalize_answer(c) for c in candidates if isinstance(c, str) and c.strip()]
    if not norm:
        return 0.0
    counts: Dict[str, int] = {}
    for a in norm:
        counts[a] = counts.get(a, 0) + 1
    top = max(counts.values())
    return top / len(norm)


def _uncertainty_proxy(rec: Dict[str, Any]) -> Optional[float]:
    """
    Returns uncertainty in [0,1] if derivable, else None.
    0 => very certain, 1 => very uncertain.
    Priority:
      1) verifier_scores list -> uncertainty = 1 - max(score)
      2) verifier_score scalar -> uncertainty = 1 - score
      3) candidates list -> uncertainty = 1 - agreement_ratio
      4) avg_logprob / logprobs -> map to [0,1] crudely
    """
    # (1) verifier_scores: list of floats
    vs = rec.get("verifier_scores")
    if isinstance(vs, list) and vs:
        vals = [float(x) for x in vs if isinstance(x, (int, float))]
        if vals:
            m = max(vals)
            return float(max(0.0, min(1.0, 1.0 - m)))

    # (2) verifier_score: scalar
    v = _get_num(rec, "verifier_score", "score", "verifier")
    if v is not None:
        # assume already in [0,1]
        return float(max(0.0, min(1.0, 1.0 - v)))

    # (3) candidates: list of strings
    cands = rec.get("candidates")
    if isinstance(cands, list) and cands:
        try:
            ar = _agreement_ratio([str(x) for x in cands])
            return float(max(0.0, min(1.0, 1.0 - ar)))
        except Exception:
            pass

    # (4) avg_logprob: scalar (usually negative). Higher => more confident.
    lp = _get_num(rec, "avg_logprob", "mean_logprob")
    if lp is not None:
        # crude mapping: clamp lp into [-10, 0] then invert -> [0,1]
        lp_clamped = max(-10.0, min(0.0, lp))
        conf = (lp_clamped - (-10.0)) / (0.0 - (-10.0))  # 0..1
        return float(1.0 - conf)

    return None


def main():
    jsonl_path = Path("data/gsm8k_model_outputs.jsonl")
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Missing {jsonl_path}")

    sei = SEIEngine(
        # Layer-1: neutral weights; tokens + latency cost if present
        w_quality=1.0,
        w_uncertainty=1.0,
        v_energy=0.0,
        v_tokens=1.0,
        v_latency=1.0,
        v_iterations=0.0,
        window=5,
        flat_eps=0.03,
    )

    n = 0
    correct = 0
    last_acc = 0.0
    last_unc = None  # uncertainty level in [0,1]

    out_path = Path("data/sei_gsm8k_uncertainty_report.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as out:
        for rec in _iter_jsonl(jsonl_path):
            n += 1

            prompt = _get_text(rec, "prompt", "question", "input")
            output = _get_text(rec, "output", "model_output", "completion", "response")

            is_correct = _get_bool(rec, "is_correct", "correct", "passed")
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

            # uncertainty proxy (0 certain .. 1 uncertain)
            unc = _uncertainty_proxy(rec)
            if unc is None:
                delta_uncertainty = 0.0
            else:
                if last_unc is None:
                    delta_uncertainty = 0.0
                else:
                    # if uncertainty decreases, that's beneficial => Δuncertainty positive
                    delta_uncertainty = float(max(-1.0, min(1.0, (last_unc - unc))))
                last_unc = unc

            tokens_in = _approx_tokens(prompt)
            tokens_out = _approx_tokens(output)

            latency_ms = _get_num(rec, "latency_ms", "latency", "time_ms") or 0.0

            sei_rec = sei.add(
                context="gsm8k_jsonl_layer1_uncertainty",
                delta_quality=float(delta_quality),
                delta_uncertainty=float(max(0.0, delta_uncertainty)),  # keep benefit-only
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=float(latency_ms),
                energy_joule=None,
                iterations=1,
            )

            out.write(json.dumps({
                "n": n,
                "acc": acc,
                "delta_quality": delta_quality,
                "uncertainty": unc,
                "delta_uncertainty": float(max(0.0, delta_uncertainty)),
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "latency_ms": latency_ms,
                "sei": sei_rec.sei,
                "sei_z": sei_rec.sei_z,
                "trend": sei_rec.trend
            }, ensure_ascii=False) + "\n")

    print(f"Wrote: {out_path}")
    print(f"Final accuracy: {correct}/{n} = {correct/n:.4f}")
    print("Last window snapshot:", sei.snapshot())


if __name__ == "__main__":
    main()