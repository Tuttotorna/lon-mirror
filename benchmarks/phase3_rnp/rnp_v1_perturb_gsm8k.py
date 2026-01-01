# rnp_v1_perturb_gsm8k.py
# OMNIA–DELTA Phase-3 — GSM8K Structural Robustness Test
# Perturbation: RNP-v1 (Reordering + Neutral Padding + Formatting Noise)
#
# Output: JSONL with {id, split, original, perturbed, answer, meta}
#
# Usage example:
#   python rnp_v1_perturb_gsm8k.py --in data/gsm8k_test.jsonl --out data/gsm8k_rnp_v1.jsonl --limit 200 --seed 42
#
# Notes:
# - This script intentionally DOES NOT touch numbers or operations.
# - Reordering is conservative: it only performs safe swaps (adjacent) to avoid breaking dependencies.
# - Padding phrases are from a frozen whitelist.

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


PADDING_BANK = [
    "Consider the information carefully.",
    "All quantities are provided explicitly.",
    "No external assumptions are required.",
    "Proceed step by step.",
    "The question concerns basic arithmetic.",
]


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_NUM_TOKEN = re.compile(r"\d")


def load_jsonl(path: str, limit: Optional[int] = None) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def dump_jsonl(path: str, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def split_sentences(text: str) -> List[str]:
    # Keep it simple: GSM8K statements are usually short.
    parts = _SENT_SPLIT.split(text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    return parts


def safe_adjacent_shuffle(sentences: List[str], rng: random.Random, max_passes: int = 4) -> List[str]:
    """
    Conservative reordering:
    - Only adjacent swaps.
    - Never move the final question sentence away from the end.
    - Avoid swapping if it would likely break a dependency.
    Heuristic dependency guard:
    - If either sentence contains pronouns/refs ("it", "they", "this", "that", "then", "therefore"),
      do not swap.
    - If either sentence starts with a connective ("Then", "So", "Therefore", "After", "Before"), do not swap.
    """
    if len(sentences) <= 2:
        return sentences[:]

    def risky(s: str) -> bool:
        s0 = s.strip()
        if re.match(r"^(Then|So|Therefore|After|Before|Next|Finally|Thus|Hence)\b", s0, flags=re.IGNORECASE):
            return True
        if re.search(r"\b(it|they|this|that|those|these|therefore|then)\b", s0, flags=re.IGNORECASE):
            return True
        return False

    out = sentences[:]
    # Lock last sentence (often the question)
    locked_last = out[-1]

    work = out[:-1]
    n = len(work)

    for _ in range(max_passes):
        # one pass of potential adjacent swaps
        i = 0
        while i < n - 1:
            a, b = work[i], work[i + 1]
            if risky(a) or risky(b):
                i += 1
                continue
            # random decision to swap
            if rng.random() < 0.5:
                work[i], work[i + 1] = work[i + 1], work[i]
                i += 2
            else:
                i += 1

    return work + [locked_last]


def inject_neutral_padding(text: str, rng: random.Random, n_pad: int) -> str:
    pads = rng.sample(PADDING_BANK, k=n_pad)
    sents = split_sentences(text)
    if not sents:
        return " ".join(pads)

    # Insert pads at random positions except after the final question (keep question last)
    insert_positions = [rng.randrange(0, max(1, len(sents))) for _ in range(n_pad)]
    insert_positions.sort()

    out: List[str] = []
    pad_idx = 0
    for i, s in enumerate(sents):
        while pad_idx < n_pad and insert_positions[pad_idx] == i:
            out.append(pads[pad_idx])
            pad_idx += 1
        out.append(s)

    # Any remaining pads go at the start (should be rare)
    while pad_idx < n_pad:
        out.insert(0, pads[pad_idx])
        pad_idx += 1

    return " ".join(out)


def add_formatting_noise(text: str, rng: random.Random) -> str:
    """
    Formatting noise spec:
    - Insert one blank line.
    - Insert an empty numbered list "1.\n2.\n3." somewhere not at the very end.
    """
    lines = [text.strip()]

    # Create noise block (no content)
    noise_block = "1.\n2.\n3."

    # Choose insertion: start or middle
    if rng.random() < 0.5:
        combined = f"{noise_block}\n\n{lines[0]}"
    else:
        # try to split roughly in half by sentence boundary
        sents = split_sentences(lines[0])
        if len(sents) >= 2:
            mid = len(sents) // 2
            left = " ".join(sents[:mid])
            right = " ".join(sents[mid:])
            combined = f"{left}\n\n{noise_block}\n\n{right}"
        else:
            combined = f"{lines[0]}\n\n{noise_block}"
    return combined


@dataclass
class PerturbMeta:
    seed: int
    n_padding: int
    reorder_passes: int


def rnp_v1(text: str, rng: random.Random) -> Tuple[str, PerturbMeta]:
    # 1) Sentence split
    sents = split_sentences(text)

    # 2) Reorder (conservative)
    reorder_passes = 4
    sents2 = safe_adjacent_shuffle(sents, rng=rng, max_passes=reorder_passes)
    reordered = " ".join(sents2)

    # 3) Neutral padding (2–4 fixed phrases)
    n_padding = rng.randint(2, 4)
    padded = inject_neutral_padding(reordered, rng=rng, n_pad=n_padding)

    # 4) Formatting noise
    noised = add_formatting_noise(padded, rng=rng)

    meta = PerturbMeta(seed=0, n_padding=n_padding, reorder_passes=reorder_passes)
    return noised, meta


def extract_answer_field(row: Dict) -> str:
    """
    GSM8K typical schema:
      { "question": "...", "answer": ".... #### 42" }
    Keep answer as-is; downstream eval can parse.
    """
    return row.get("answer", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input GSM8K JSONL")
    ap.add_argument("--out", dest="out", required=True, help="Output JSONL with original+perturbed")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit")
    ap.add_argument("--seed", type=int, default=42, help="Global RNG seed")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = load_jsonl(args.inp, limit=args.limit)

    out_rows: List[Dict] = []
    for idx, r in enumerate(rows):
        q = r.get("question", "").strip()
        a = extract_answer_field(r)

        # Use per-item deterministic seed (stable across subsets)
        item_seed = (args.seed * 1_000_003) ^ idx
        item_rng = random.Random(item_seed)

        pert, meta = rnp_v1(q, rng=item_rng)
        meta.seed = item_seed

        out_rows.append(
            {
                "id": r.get("id", idx),
                "split": r.get("split", "test"),
                "question_original": q,
                "question_perturbed": pert,
                "answer": a,
                "meta": {
                    "perturbation": "RNP-v1",
                    "seed": meta.seed,
                    "n_padding": meta.n_padding,
                    "reorder_passes": meta.reorder_passes,
                },
            }
        )

    dump_jsonl(args.out, out_rows)
    print(f"Wrote {len(out_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()