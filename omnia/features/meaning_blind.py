from __future__ import annotations

import hashlib
import re
import zlib
from typing import Dict, List


def _sha(x: str) -> str:
    return hashlib.sha256(x.encode("utf-8", errors="ignore")).hexdigest()


def _ngrams(s: str, n: int = 4) -> List[str]:
    if len(s) < n:
        return [s]
    return [s[i : i + n] for i in range(len(s) - n + 1)]


def meaning_blind_features(s: str) -> Dict[str, float]:
    """
    Meaning-blind structural feature map.
    No embeddings, no semantics, no world model.

    Features:
    - length bucket
    - digit ratio bucket
    - compression ratio bucket
    - hashed 4-grams
    - hashed 64-char chunks
    """
    s2 = s.replace("\r\n", "\n")
    s2 = re.sub(r"[ \t]+", " ", s2).strip()

    length = len(s2)
    digits = sum(ch.isdigit() for ch in s2)
    digit_ratio = 0.0 if length == 0 else digits / max(1, length)

    comp = zlib.compress(s2.encode("utf-8", errors="ignore"), level=9)
    comp_ratio = 0.0 if length == 0 else len(comp) / max(1, length)

    feats: Dict[str, float] = {}
    feats[f"len:{min(10, length // 50)}"] = 1.0
    feats[f"dig:{int(digit_ratio * 10)}"] = 1.0
    feats[f"cmp:{int(comp_ratio * 10)}"] = 1.0

    for g in _ngrams(s2, 4):
        feats["ng:" + _sha(g)[:12]] = 1.0

    chunk = 64
    for i in range(0, len(s2), chunk):
        feats["ck:" + _sha(s2[i : i + chunk])[:12]] = 1.0

    return feats