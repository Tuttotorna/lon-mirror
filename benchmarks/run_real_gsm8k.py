"""
benchmarks/run_real_gsm8k.py

OMNIA — Real GSM8K-style pipeline runner (v0.1)

Goal:
  Turn the repo from "demo" into "reproducible":
    - Load data/gsm8k_model_outputs.jsonl (real model outputs)
    - Compute OMNIA signals (BASE + TOKEN if available)
    - Compute LCR signals if available (optional)
    - Apply ICE gate (PASS / ESCALATE / BLOCK)
    - Emit a JSON report in reports/real_gsm8k_report.json
    - Emit a "worst cases" JSONL for debugging

Assumptions:
  Each JSONL line is like:
  {
    "id": "...",
    "question": "...",
    "gold_answer": "42",
    "model_chain": "...",
    "model_answer": "42"
  }

Requirements:
  - numpy installed
  - your repo already contains:
      omnia/ package (BASE/TOKEN)
      ICE/OMNIA_ICE_v0_1.py (ICEInput, ice_gate)
  - Optional: LCR modules (if present). This runner will not fail if LCR is missing.

Author: Massimiliano Brighindi (MB-X.01 / OMNIA)
"""

from __future__ import annotations

import os
import re
import json
import math
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# --- OMNIA imports (adapt if your paths differ) ---
try:
    from omnia.omniabase import pbii_index
except Exception as e:
    raise ImportError("Cannot import omnia.omniabase.pbii_index. Check your omnia package.") from e

# Token lens is optional; we implement a fallback PBII-z here.
# If you already have omnia.omniatoken, you can swap in your own.
# from omnia.omniatoken import token_instability_score

# --- ICE imports ---
try:
    from ICE.OMNIA_ICE_v0_1 import ICEInput, ice_gate
except Exception as e:
    raise ImportError("Cannot import ICE.OMNIA_ICE_v0_1 (ICEInput, ice_gate). Check ICE folder/file name.") from e


DATA_PATH = "data/gsm8k_model_outputs.jsonl"
REPORT_DIR = "reports"
REPORT_PATH = os.path.join(REPORT_DIR, "real_gsm8k_report.json")
WORST_JSONL_PATH = os.path.join(REPORT_DIR, "real_gsm8k_worst_cases.jsonl")


# =========================
# Helpers
# =========================

_INT_RE = re.compile(r"\b\d+\b")


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                # strict: skip broken lines
                continue
    return out


def extract_ints(text: str) -> List[int]:
    if not text:
        return []
    nums = []
    for m in _INT_RE.findall(text):
        try:
            v = int(m)
            if v > 1:
                nums.append(v)
        except Exception:
            pass
    return nums


def normalize_0_1(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.5
    v = (x - lo) / (hi - lo)
    return float(max(0.0, min(1.0, v)))


def token_proxy_numbers(text: str) -> List[int]:
    """
    Deterministic token->int proxy.
    Not semantic. Purely structural.
    """
    if not text:
        return []
    toks = re.findall(r"[A-Za-z0-9']+|[^\sA-Za-z0-9]", text)
    # Proxy: (len(token) + sum(char codes mod 97)) to get variety but deterministic.
    nums: List[int] = []
    for t in toks:
        s = 0
        for ch in t:
            s = (s + ord(ch)) % 97
        nums.append(max(2, len(t) + s))  # >=2 for pbii_index domain
    return nums


def pbii_z_mean(ints: List[int]) -> float:
    """
    TOKEN lens core:
      - PBII per token-proxy integer
      - z-score normalize
      - return mean(|z|)
    """
    if not ints:
        return 0.0
    scores = np.array([pbii_index(v) for v in ints], dtype=float)
    mu = float(scores.mean())
    sd = float(scores.std(ddof=0))
    if sd <= 1e-12:
        return 0.0
    z = (scores - mu) / sd
    return float(np.mean(np.abs(z)))


def gold_match_score(gold_answer: Optional[str], model_answer: str) -> Optional[float]:
    """
    Minimal GSM8K-like gold match:
      - extract last integer from gold_answer and model_answer
      - compare
    Returns: 1.0 / 0.0 / None if missing
    """
    if gold_answer is None:
        return None
    g_ints = extract_ints(str(gold_answer))
    m_ints = extract_ints(str(model_answer))
    if not g_ints or not m_ints:
        return None
    return 1.0 if g_ints[-1] == m_ints[-1] else 0.0


def ambiguity_heuristic(question: str, chain: str, answer: str) -> float:
    """
    Deterministic ambiguity proxy (0..1):
      - penalize ellipsis/polysemy patterns with very simple signals:
        * presence of strong polysemous verbs: kill/delete/remove/clear
        * short answers without explicit object
        * many quoted phrases or parentheses (idiom/meta)
    This is not NLP. It's a structural heuristic to feed ICE today.
    """
    text = f"{question} {chain} {answer}".lower()
    poly = 0.0
    for w in ["kill", "killed", "delete", "deleted", "remove", "removed", "clear", "cleared", "terminate"]:
        if w in text:
            poly += 0.15

    short_answer = 0.15 if len(answer.strip()) <= 6 else 0.0
    meta = 0.10 if ("(" in text or ")" in text or '"' in text or "'" in text) else 0.0

    # if chain is empty but answer has numbers, ambiguity rises (no justification)
    no_chain = 0.20 if (not chain.strip() and len(extract_ints(answer)) > 0) else 0.0

    a = poly + short_answer + meta + no_chain
    return float(max(0.0, min(1.0, a)))


# =========================
# Main runner
# =========================

def main() -> None:
    safe_mkdir(REPORT_DIR)

    records = load_jsonl(DATA_PATH)
    if not records:
        raise RuntimeError(f"No valid records found in {DATA_PATH}")

    # We will compute:
    # - base_instability = PBII(mean over chain integers) or PBII(answer integer) fallback
    # - token_instability = pbii_z_mean(token proxies from chain)
    # - omega_total = simple fusion for now (BASE + TOKEN only), then ICE gates it
    #
    # Later you can plug full OMNIA_TOTALE + LCR.

    rows: List[Dict[str, Any]] = []

    for rec in records:
        rid = str(rec.get("id", ""))
        q = str(rec.get("question", ""))
        gold = rec.get("gold_answer", None)
        chain = str(rec.get("model_chain", ""))
        ans = str(rec.get("model_answer", ""))

        # ---- BASE signal ----
        chain_ints = extract_ints(chain)
        ans_ints = extract_ints(ans)
        base_pbii_vals: List[float] = []
        for v in chain_ints:
            base_pbii_vals.append(pbii_index(v))
        if not base_pbii_vals and ans_ints:
            base_pbii_vals.append(pbii_index(ans_ints[-1]))
        base_instability = float(np.mean(base_pbii_vals)) if base_pbii_vals else 0.0

        # ---- TOKEN signal ----
        token_nums = token_proxy_numbers(chain if chain.strip() else ans)
        token_instability = pbii_z_mean(token_nums)

        # ---- Simple Ω_total fusion (real but minimal) ----
        # Both are unbounded-ish; squash to 0..1-ish, then sum with weights.
        base_s = 1.0 - (1.0 / (1.0 + max(0.0, base_instability)))
        tok_s = 1.0 - (1.0 / (1.0 + max(0.0, token_instability)))
        omega_total = 1.0 * base_s + 1.0 * tok_s

        # ---- LCR placeholder (optional) ----
        omega_ext = None  # leave None until you plug LCR real score

        # ---- gold match ----
        gm = gold_match_score(gold, ans)

        # ---- ambiguity ----
        amb = ambiguity_heuristic(q, chain, ans)

        # ---- ICE gate ----
        ice_in = ICEInput(
            omega_total=float(omega_total),
            lens_scores={
                "BASE": float(base_s),
                "TOKEN": float(tok_s),
                "TIME": 0.0,
                "CAUSA": 0.0,
                "LCR": 0.0,
            },
            lens_metadata={
                "BASE": {"base_instability_pbii_mean": base_instability},
                "TOKEN": {"token_instability_z_mean": token_instability},
                "LANG": {"ambiguity_heuristic": amb},
            },
            omega_ext=omega_ext,
            gold_match=gm,
            ambiguity_score=float(amb),
            notes=None,
        )
        ice_out = ice_gate(ice_in).to_dict()

        rows.append(
            {
                "id": rid,
                "gold_match": gm,
                "omega_total": float(omega_total),
                "base_instability_pbii_mean": base_instability,
                "token_instability_z_mean": token_instability,
                "ambiguity_score": float(amb),
                "ice": ice_out,
            }
        )

    # =========================
    # Metrics (real, using gold_match when present)
    # =========================
    with_gold = [r for r in rows if r["gold_match"] is not None]
    N = len(with_gold)

    # Define "hallucination" = gold_match==0 on this dataset
    # Define "flagged" = ICE status != PASS  (ESCALATE or BLOCK)
    tp = fp = tn = fn = 0
    for r in with_gold:
        is_halluc = (r["gold_match"] == 0.0)
        flagged = (r["ice"]["status"] != "PASS")
        if is_halluc and flagged:
            tp += 1
        elif (not is_halluc) and flagged:
            fp += 1
        elif (not is_halluc) and (not flagged):
            tn += 1
        else:
            fn += 1

    def _safe_div(a: float, b: float) -> float:
        return float(a / b) if b else 0.0

    detection_rate = _safe_div(tp, tp + fn)       # recall on hallucinations
    precision = _safe_div(tp, tp + fp)
    fpr = _safe_div(fp, fp + tn)
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)

    # Worst cases: highest omega_total among incorrect, to inspect
    wrong = [r for r in with_gold if r["gold_match"] == 0.0]
    worst = sorted(wrong, key=lambda x: x["omega_total"], reverse=True)[:50]

    # Save worst cases jsonl
    with open(WORST_JSONL_PATH, "w", encoding="utf-8") as f:
        for r in worst:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {
        "dataset": DATA_PATH,
        "N_total": len(rows),
        "N_with_gold": N,
        "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "metrics": {
            "detection_rate_recall": detection_rate,
            "precision": precision,
            "false_positive_rate": fpr,
            "accuracy": accuracy,
        },
        "ice_counts": {
            "PASS": sum(1 for r in rows if r["ice"]["status"] == "PASS"),
            "ESCALATE": sum(1 for r in rows if r["ice"]["status"] == "ESCALATE"),
            "BLOCK": sum(1 for r in rows if r["ice"]["status"] == "BLOCK"),
        },
        "notes": {
            "omega_total_is_minimal": True,
            "lenses_active": ["BASE", "TOKEN", "ICE"],
            "LCR_not_plugged": True,
            "Next_step": "Plug real LCR omega_ext and full OMNIA_TOTALE, then calibrate thresholds on validation split.",
        },
        "outputs": {
            "report_json": REPORT_PATH,
            "worst_cases_jsonl": WORST_JSONL_PATH,
        },
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=== OMNIA REAL RUN (v0.1) ===")
    print(f"Loaded records: {len(rows)} (with gold: {N})")
    print(f"TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"Recall(detection_rate)={detection_rate:.3f}  Precision={precision:.3f}  FPR={fpr:.3f}  Acc={accuracy:.3f}")
    print(f"Report: {REPORT_PATH}")
    print(f"Worst cases: {WORST_JSONL_PATH}")


if __name__ == "__main__":
    main()