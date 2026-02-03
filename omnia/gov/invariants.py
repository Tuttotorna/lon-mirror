from __future__ import annotations
import re
import hashlib
from typing import Iterable, Set, Union, List

_ASCII_RE = re.compile(r"[^\x20-\x7E]+")

def _norm_ascii(s: str) -> str:
    s = s.strip()
    s = _ASCII_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s)
    return s.lower()

def _tokenize(s: str) -> List[str]:
    s = _norm_ascii(s)
    # structural tokens only; no semantics
    return re.findall(r"[a-z0-9_]+", s)

def _hash(tok: str) -> str:
    return hashlib.sha256(tok.encode("utf-8")).hexdigest()[:16]

def extract_invariants(obj: Union[str, List[str]]) -> Set[str]:
    """
    Deterministic structural invariants:
    - unigram + bigram hashes
    - punctuation-free token stream
    """
    if isinstance(obj, list):
        s = " ".join(map(str, obj))
    else:
        s = str(obj)

    toks = _tokenize(s)
    inv: Set[str] = set()

    for t in toks:
        inv.add(_hash("u:" + t))

    for i in range(len(toks) - 1):
        inv.add(_hash("b:" + toks[i] + "|" + toks[i+1]))

    return inv

def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / (union + 1e-12)