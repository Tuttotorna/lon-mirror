#!/usr/bin/env python3
"""
ΔΣ-PURE — Symbols (no semantics)
State S: multiset of anonymous token strings
Transformations T: destroy adjacency and local structure without adding info
Output: ΔΣ ∈ {INVARIANT, DEFORMED, COLLAPSED} + a reproducible threshold L*
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Tuple


# -------------------------
# State (structure-only)
# -------------------------

@dataclass(frozen=True)
class State:
    tokens: Tuple[str, ...]  # anonymous symbol strings


# -------------------------
# Structural fingerprints (non-semantic)
# -------------------------

def bigrams(s: State) -> List[Tuple[str, str]]:
    t = s.tokens
    return [(t[i], t[i + 1]) for i in range(len(t) - 1)]

def trigram_count(s: State) -> int:
    # count of length-3 runs with all distinct tokens (pure adjacency constraint)
    t = s.tokens
    c = 0
    for i in range(len(t) - 2):
        a, b, c3 = t[i], t[i + 1], t[i + 2]
        if a != b and b != c3 and a != c3:
            c += 1
    return c

def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    inter = len(set_a & set_b)
    uni = len(set_a | set_b)
    return inter / uni if uni else 0.0


# -------------------------
# ΔΣ classifier (no numbers-as-truth; only thresholds)
# -------------------------

def dsigma_classify(original: State, transformed: State) -> str:
    # Structural comparison uses only adjacency-derived constraints.
    # We do NOT compare token meanings, only relation sets.

    A = set(bigrams(original))
    B = set(bigrams(transformed))

    jac = jaccard(A, B)

    # Discrete outcomes:
    # - INVARIANT: adjacency relation essentially preserved
    # - DEFORMED: partial preservation
    # - COLLAPSED: adjacency relation lost

    # Fixed, domain-agnostic thresholds (coarse, non-optimized)
    if jac >= 0.80:
        return "INVARIANT"
    if jac >= 0.20:
        return "DEFORMED"
    return "COLLAPSED"


# -------------------------
# Transformations T (no new info)
# -------------------------

def T_block_shuffle(s: State, block: int, rng: random.Random) -> State:
    t = list(s.tokens)
    for i in range(0, len(t), block):
        chunk = t[i:i + block]
        rng.shuffle(chunk)
        t[i:i + block] = chunk
    return State(tokens=tuple(t))

def T_fragment(s: State, cut_prob: float, rng: random.Random) -> State:
    # randomly inserts "cuts" by permuting short local spans
    t = list(s.tokens)
    i = 0
    while i < len(t) - 3:
        if rng.random() < cut_prob:
            span = t[i:i + 4]
            rng.shuffle(span)
            t[i:i + 4] = span
            i += 4
        else:
            i += 1
    return State(tokens=tuple(t))


# -------------------------
# Threshold search: find L* where collapse becomes typical
# -------------------------

def find_L_star(
    s: State,
    blocks: List[int],
    trials: int = 60,
    seed: int = 0
) -> Tuple[int, List[Tuple[int, str]]]:
    rng = random.Random(seed)
    outcomes = []

    for b in blocks:
        collapsed = 0
        for _ in range(trials):
            x = T_block_shuffle(s, b, rng)
            out = dsigma_classify(s, x)
            if out == "COLLAPSED":
                collapsed += 1

        # majority vote outcome (discrete, not "probability")
        if collapsed >= (trials // 2 + 1):
            maj = "COLLAPSED"
        else:
            # if not collapsed-majority, decide between invariant/deformed by one sample
            maj = dsigma_classify(s, T_block_shuffle(s, b, rng))

        outcomes.append((b, maj))

    # L* = smallest block size producing COLLAPSED
    L_star = next((b for b, o in outcomes if o == "COLLAPSED"), -1)
    return L_star, outcomes


# -------------------------
# Demo state generator (anonymous symbols)
# -------------------------

def make_state(n: int = 160) -> State:
    # No semantics: repeating motifs create structure purely by adjacency.
    motif = ["A", "B", "C", "D", "E", "F", "G", "H"]
    tokens = []
    for i in range(n):
        tokens.append(motif[i % len(motif)])
    return State(tokens=tuple(tokens))


def main():
    s = make_state(160)

    blocks = [2, 3, 4, 6, 8, 12, 16, 24, 32]
    L_star, outcomes = find_L_star(s, blocks, trials=60, seed=0)

    print("ΔΣ-PURE / SYMBOLS")
    print("State length:", len(s.tokens))
    print("Fingerprint (trigram distinct count):", trigram_count(s))
    print("Block scan outcomes:")
    for b, o in outcomes:
        print(f"  block={b:>2} -> {o}")
    print("L* (first collapse block):", L_star)


if __name__ == "__main__":
    main()