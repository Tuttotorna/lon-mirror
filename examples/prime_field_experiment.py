# examples/prime_field_experiment.py
# ============================================================
# Prime Field Reconstruction (Π-FIELD) — minimal, OMNIA-style
#
# Goal: build a structural signature F(p) for primes, define a
# structural distance DΠ, and cluster primes into Π-regions.
#
# This is NOT a prime predictor. It's a domain-shift probe.
#
# Run:
#   python examples/prime_field_experiment.py --max_n 20000 --k 8 --seed 1
# ============================================================

from __future__ import annotations

import argparse
import math
import random
import statistics
import zlib
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Tuple


# ----------------------------
# Prime generation (sieve)
# ----------------------------
def primes_upto(n: int) -> List[int]:
    if n < 2:
        return []
    sieve = bytearray(b"\x01") * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for p in range(2, int(n**0.5) + 1):
        if sieve[p]:
            step = p
            start = p * p
            sieve[start : n + 1 : step] = b"\x00" * (((n - start) // step) + 1)
    return [i for i in range(2, n + 1) if sieve[i]]


# ----------------------------
# Base encoding
# ----------------------------
_ALPH = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def to_base(n: int, b: int) -> str:
    if n == 0:
        return "0"
    if b < 2 or b > len(_ALPH):
        raise ValueError("base out of range")
    x = n
    out = []
    while x > 0:
        x, r = divmod(x, b)
        out.append(_ALPH[r])
    return "".join(reversed(out))


# ----------------------------
# Simple transforms (aperspective-ish)
# ----------------------------
def t_identity(s: str) -> str:
    return s


def t_reverse(s: str) -> str:
    return s[::-1]


def t_drop_vowels(s: str) -> str:
    return "".join(ch for ch in s if ch.upper() not in "AEIOU")


def t_whitespace_collapse(s: str) -> str:
    # for primes, whitespace absent; still keep for consistency
    return " ".join(s.split())


TRANSFORMS = [
    ("id", t_identity),
    ("rev", t_reverse),
    ("vow-", t_drop_vowels),
    ("ws", t_whitespace_collapse),
]


# ----------------------------
# Structural measurers (no semantics)
# ----------------------------
def sim_ratio(a: str, b: str) -> float:
    # Bounded [0,1], stable, deterministic
    if a == b:
        return 1.0
    return SequenceMatcher(a=a, b=b).ratio()


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq: Dict[str, int] = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(s)
    ent = 0.0
    for c in freq.values():
        p = c / n
        ent -= p * math.log2(p)
    return ent


def zlib_ratio(s: str) -> float:
    # compression ratio in (0,1] roughly, higher means less compressible
    raw = s.encode("utf-8", errors="ignore")
    if not raw:
        return 0.0
    comp = zlib.compress(raw, level=9)
    return min(1.0, len(comp) / len(raw))


def digit_run_score(s: str) -> float:
    # measures repeated runs; higher means more run-structure
    if not s:
        return 0.0
    runs = 1
    for i in range(1, len(s)):
        if s[i] != s[i - 1]:
            runs += 1
    # normalize: 1 run => 1.0, all alternating => ~0
    return 1.0 - (runs - 1) / max(1, len(s) - 1)


@dataclass(frozen=True)
class PrimeSignature:
    p: int
    vec: List[float]


def build_signature(p: int, bases: List[int]) -> PrimeSignature:
    # Represent p across bases; treat each representation as a "sensor slice"
    reps = [to_base(p, b) for b in bases]

    # Base-length profile (normalized)
    lens = [len(r) for r in reps]
    max_len = max(lens) if lens else 1
    len_norm = [l / max_len for l in lens]

    # Per-base entropy + compressibility + run structure
    ent = [shannon_entropy(r) for r in reps]
    ent_norm = [e / (math.log2(len(set(r))) if len(set(r)) > 1 else 1.0) for r, e in zip(reps, ent)]
    zcr = [zlib_ratio(r) for r in reps]
    run = [digit_run_score(r) for r in reps]

    # Aperspective overlap: compare each rep with its transforms
    # omega_like per base: mean similarity under transforms
    omega_like = []
    for r in reps:
        sims = []
        for _, t in TRANSFORMS:
            sims.append(sim_ratio(r, t(r)))
        omega_like.append(sum(sims) / len(sims))

    # Cross-base agreement: similarity between representations (string-level)
    # This is intentionally crude: we measure "shape overlap" after normalization by transforms.
    cross = []
    for i in range(len(reps)):
        for j in range(i + 1, len(reps)):
            a = reps[i]
            b = reps[j]
            cross.append(sim_ratio(a, b))
    cross_mean = sum(cross) / len(cross) if cross else 0.0
    cross_mad = statistics.median([abs(x - statistics.median(cross)) for x in cross]) if cross else 0.0

    # Build final vector:
    # - summaries across bases (mean/mad) to avoid base bias
    def mean(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    def mad(xs: List[float]) -> float:
        if not xs:
            return 0.0
        m = statistics.median(xs)
        return statistics.median([abs(x - m) for x in xs])

    vec = []
    vec += [mean(len_norm), mad(len_norm)]
    vec += [mean(ent_norm), mad(ent_norm)]
    vec += [mean(zcr), mad(zcr)]
    vec += [mean(run), mad(run)]
    vec += [mean(omega_like), mad(omega_like)]
    vec += [cross_mean, cross_mad]

    return PrimeSignature(p=p, vec=vec)


# ----------------------------
# DΠ distance + clustering
# ----------------------------
def l2(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def kmeans(signatures: List[PrimeSignature], k: int, seed: int, iters: int = 30) -> Tuple[List[int], List[List[float]]]:
    rng = random.Random(seed)
    vecs = [s.vec for s in signatures]
    n = len(vecs)
    if n == 0:
        return [], []
    k = max(1, min(k, n))

    # init centroids from random points
    centroids = [vecs[i][:] for i in rng.sample(range(n), k)]

    labels = [0] * n
    for _ in range(iters):
        changed = False

        # assign
        for i, v in enumerate(vecs):
            best = 0
            best_d = float("inf")
            for cidx, c in enumerate(centroids):
                d = l2(v, c)
                if d < best_d:
                    best_d = d
                    best = cidx
            if labels[i] != best:
                labels[i] = best
                changed = True

        # recompute
        new_centroids = [[0.0] * len(vecs[0]) for _ in range(k)]
        counts = [0] * k
        for lab, v in zip(labels, vecs):
            counts[lab] += 1
            for j in range(len(v)):
                new_centroids[lab][j] += v[j]

        for cidx in range(k):
            if counts[cidx] == 0:
                # re-seed empty cluster
                new_centroids[cidx] = vecs[rng.randrange(n)][:]  # deterministic under seed
            else:
                new_centroids[cidx] = [x / counts[cidx] for x in new_centroids[cidx]]

        centroids = new_centroids
        if not changed:
            break

    return labels, centroids


def summarize_clusters(signatures: List[PrimeSignature], labels: List[int], centroids: List[List[float]]) -> str:
    k = len(centroids)
    buckets: List[List[PrimeSignature]] = [[] for _ in range(k)]
    for s, lab in zip(signatures, labels):
        buckets[lab].append(s)

    lines = []
    lines.append("Π-FIELD RESULTS")
    lines.append("--------------")
    lines.append(f"items={len(signatures)} clusters={k}")
    lines.append("")
    for cidx, items in enumerate(buckets):
        items_sorted = sorted(items, key=lambda x: x.p)
        ps = [x.p for x in items_sorted]
        if not ps:
            lines.append(f"[Π-REGION {cidx}] EMPTY")
            continue

        # medoid-like example: closest to centroid
        centroid = centroids[cidx]
        best = min(items, key=lambda s: l2(s.vec, centroid))
        lines.append(f"[Π-REGION {cidx}] size={len(ps)} p_min={ps[0]} p_max={ps[-1]} exemplar={best.p}")
        # show first/last few
        head = ps[:8]
        tail = ps[-8:] if len(ps) > 8 else []
        lines.append(f"  head: {head}")
        if tail:
            lines.append(f"  tail: {tail}")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_n", type=int, default=20000, help="upper bound for primes")
    ap.add_argument("--k", type=int, default=8, help="number of Π-regions (clusters)")
    ap.add_argument("--seed", type=int, default=1, help="deterministic seed")
    ap.add_argument("--bases", type=str, default="2,3,5,8,10,16", help="comma-separated bases")
    args = ap.parse_args()

    bases = [int(x.strip()) for x in args.bases.split(",") if x.strip()]
    ps = primes_upto(args.max_n)

    sigs = [build_signature(p, bases=bases) for p in ps]

    labels, centroids = kmeans(sigs, k=args.k, seed=args.seed, iters=40)
    report = summarize_clusters(sigs, labels, centroids)
    print(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())