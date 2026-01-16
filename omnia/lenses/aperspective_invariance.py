from __future__ import annotations

import hashlib
import random
import re
import zlib
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

Transform = Callable[[str], str]


def _sha(x: str) -> str:
    return hashlib.sha256(x.encode("utf-8", errors="ignore")).hexdigest()


def _ngrams(s: str, n: int = 4) -> List[str]:
    if len(s) < n:
        return [s]
    return [s[i : i + n] for i in range(len(s) - n + 1)]


def _feature_set(s: str) -> set:
    """
    Feature-set intentionally "meaning-blind":
    - character n-grams
    - length bucket
    - digit ratio bucket
    - compression ratio bucket
    - stable hashes of chunks

    This is not semantics. It's structural signatures.
    """
    s2 = s
    # Normalize only to reduce trivial variance; still not semantics.
    s2 = s2.replace("\r\n", "\n")
    s2 = re.sub(r"[ \t]+", " ", s2).strip()

    length = len(s2)
    digits = sum(ch.isdigit() for ch in s2)
    digit_ratio = 0.0 if length == 0 else digits / length

    comp = zlib.compress(s2.encode("utf-8", errors="ignore"), level=9)
    comp_ratio = 0.0 if length == 0 else len(comp) / max(1, length)

    # Bucketing avoids “overfitting” to one exact value.
    length_bucket = f"len:{min(10, length // 50)}"  # 0..10
    digit_bucket = f"dig:{int(digit_ratio * 10)}"   # 0..10
    comp_bucket = f"cmp:{int(comp_ratio * 10)}"     # 0..10

    feats = {length_bucket, digit_bucket, comp_bucket}

    # N-grams (structure of surface form)
    for g in _ngrams(s2, 4):
        feats.add("ng:" + _sha(g)[:12])

    # Chunk-hashes (coarse structure)
    chunk = 64
    for i in range(0, len(s2), chunk):
        feats.add("ck:" + _sha(s2[i : i + chunk])[:12])

    return feats


@dataclass
class AperspectiveInvarianceResult:
    omega_score: float
    residue: List[str]
    per_transform_scores: Dict[str, float]


class AperspectiveInvariance:
    """
    Invarianza Aperspettica (prototype):
    - no observer privileged
    - no semantics
    - invariance = intersection of structural features across independent transforms
    """

    def __init__(self, transforms: List[Tuple[str, Transform]]):
        if not transforms:
            raise ValueError("Need at least one transform")
        self.transforms = transforms

    def measure(self, x: str) -> AperspectiveInvarianceResult:
        base_feats = _feature_set(x)

        residues = []
        per_scores: Dict[str, float] = {}

        # compute intersection across all transforms (including identity)
        inter = set(base_feats)

        for name, t in self.transforms:
            y = t(x)
            f = _feature_set(y)
            inter &= f

            # per-transform overlap (how much survives under that transform)
            overlap = 0.0 if not base_feats else len(base_feats & f) / max(1, len(base_feats))
            per_scores[name] = overlap
            residues.append((name, y))

        omega = 0.0 if not base_feats else len(inter) / max(1, len(base_feats))
        # expose residue as stable tokens (still meaning-blind)
        residue_sorted = sorted(inter)

        return AperspectiveInvarianceResult(
            omega_score=omega,
            residue=residue_sorted[:200],  # cap for display
            per_transform_scores=per_scores,
        )


# -------------------------
# Example transform library
# -------------------------

def t_identity(s: str) -> str:
    return s

def t_whitespace_collapse(s: str) -> str:
    s = s.replace("\r\n", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()

def t_shuffle_words(seed: int = 1) -> Transform:
    def _t(s: str) -> str:
        rng = random.Random(seed)
        parts = re.split(r"(\W+)", s)  # keep separators
        words = [p for p in parts if p.isalnum()]
        rng.shuffle(words)
        it = iter(words)
        out = []
        for p in parts:
            out.append(next(it) if p.isalnum() else p)
        return "".join(out)
    return _t

def t_reverse(s: str) -> str:
    return s[::-1]

def t_drop_vowels(s: str) -> str:
    return re.sub(r"[aeiouAEIOUàèéìòùÀÈÉÌÒÙ]", "", s)

def t_base_repr(seed: int = 7, base: int = 7) -> Transform:
    """
    Turns digits into a different base string representation (structure shift).
    Not semantics: it's a representation transform.
    """
    if base < 2 or base > 36:
        raise ValueError("base must be 2..36")

    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"

    def to_base(n: int) -> str:
        if n == 0:
            return "0"
        neg = n < 0
        n = abs(n)
        out = []
        while n:
            out.append(alphabet[n % base])
            n //= base
        s2 = "".join(reversed(out))
        return "-" + s2 if neg else s2

    def _t(s: str) -> str:
        rng = random.Random(seed)
        # replace each decimal number with base-N representation
        def repl(m: re.Match) -> str:
            n = int(m.group(0))
            # small random perturbation to avoid trivial invariance
            if rng.random() < 0.1:
                n += rng.choice([-1, 1])
            return to_base(n)
        return re.sub(r"\b\d+\b", repl, s)
    return _t


if __name__ == "__main__":
    transforms = [
        ("id", t_identity),
        ("ws", t_whitespace_collapse),
        ("rev", t_reverse),
        ("vow-", t_drop_vowels),
        ("shuf", t_shuffle_words(seed=3)),
        ("base7", t_base_repr(seed=7, base=7)),
    ]

    engine = AperspectiveInvariance(transforms)

    x = """
    Nel doppio slit, 2 fessure producono un pattern d'interferenza.
    Misuro solo struttura: lunghezze, ripetizioni, compressibilità, n-gram.
    2026 2025 2024
    """

    r = engine.measure(x)

    print("Ω-score (aperspective invariance):", round(r.omega_score, 4))
    print("Per-transform overlap:")
    for k, v in sorted(r.per_transform_scores.items()):
        print(" ", k, "->", round(v, 4))
    print("Residue sample (hashed structural tokens):", r.residue[:20])