from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from omnia.lenses.aperspective_invariance import AperspectiveInvariance

Transform = Callable[[str], str]


@dataclass(frozen=True)
class TotalCollapseResult:
    """
    TCO-1.0

    omega0:
        Baseline aperspective omega (Ω_ap) computed using baseline transforms.

    omega_curve:
        Ω after progressively applying collapse stages.

    cost_curve:
        Monotonic cost per stage (user-defined).

    c_star:
        First cost where Ω <= eps (collapse point). None if never collapses.

    collapse_index:
        Normalized collapse depth in [0,1]:
            0   -> collapses immediately
            1   -> never collapses within schedule

    notes:
        Minimal non-narrative diagnostics.
    """
    omega0: float
    omega_curve: List[float]
    cost_curve: List[float]
    c_star: Optional[float]
    collapse_index: float
    notes: Dict[str, float]


class TotalCollapseOperator:
    """
    Total Collapse Operator (TCO-1.0)

    Purpose:
        Apply progressively destructive (non-invariance-preserving) transforms
        to collapse all collapsable structure and measure what remains.

    This is intentionally dual to OMNIA’s aperspective extraction:
        OMNIA  -> subtract representation (seek invariants)
        TCO    -> maximize perturbation (seek non-collapsable residue)

    Output is measurement only.
    """

    def __init__(
        self,
        *,
        baseline_transforms: Sequence[Tuple[str, Transform]],
        collapse_schedule: Sequence[Tuple[float, str, Transform]],
        eps: float = 1e-4,
        clamp: bool = True,
    ):
        if not baseline_transforms:
            raise ValueError("baseline_transforms must be non-empty")
        if not collapse_schedule:
            raise ValueError("collapse_schedule must be non-empty")
        if eps <= 0:
            raise ValueError("eps must be > 0")

        self.baseline_transforms = list(baseline_transforms)
        self.collapse_schedule = list(collapse_schedule)
        self.eps = float(eps)
        self.clamp = bool(clamp)

        self._ap = AperspectiveInvariance(transforms=self.baseline_transforms)

    @staticmethod
    def _clamp01(x: float) -> float:
        if x < 0.0:
            return 0.0
        if x > 1.0:
            return 1.0
        return x

    def _omega_ap(self, x: str) -> float:
        r = self._ap.measure(x)
        w = float(r.omega_score)
        return self._clamp01(w) if self.clamp else w

    def _omega_after(self, x: str, t: Transform) -> float:
        y = t(x)
        r = self._ap.measure(y)
        w = float(r.omega_score)
        return self._clamp01(w) if self.clamp else w

    def measure(self, x: str) -> TotalCollapseResult:
        omega0 = self._omega_ap(x)

        omega_curve: List[float] = []
        cost_curve: List[float] = []
        c_star: Optional[float] = None

        for cost, _name, t in self.collapse_schedule:
            w = self._omega_after(x, t)
            omega_curve.append(w)
            cost_curve.append(float(cost))
            if c_star is None and w <= self.eps:
                c_star = float(cost)

        # collapse index: map where collapse happens in schedule depth
        if c_star is None:
            collapse_index = 1.0
        else:
            # normalize by last cost, robust to arbitrary costs
            last = max(1e-12, float(cost_curve[-1]))
            collapse_index = max(0.0, min(1.0, c_star / last))

        notes: Dict[str, float] = {
            "omega0": float(omega0),
            "omega_min": float(min(omega_curve)) if omega_curve else float(omega0),
            "omega_last": float(omega_curve[-1]) if omega_curve else float(omega0),
            "eps": float(self.eps),
        }

        return TotalCollapseResult(
            omega0=float(omega0),
            omega_curve=omega_curve,
            cost_curve=cost_curve,
            c_star=c_star,
            collapse_index=float(collapse_index),
            notes=notes,
        )


# -----------------------------
# Default collapse library (meaning-blind)
# -----------------------------

def t_identity(s: str) -> str:
    return s


def t_whitespace_collapse(s: str) -> str:
    import re
    s = s.replace("\r\n", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def t_drop_vowels(s: str) -> str:
    import re
    return re.sub(r"[aeiouAEIOUàèéìòùÀÈÉÌÒÙ]", "", s)


def t_reverse(s: str) -> str:
    return s[::-1]


def t_shuffle_words(seed: int = 1) -> Transform:
    import random, re

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


def t_chunk_shuffle(seed: int = 2, chunk: int = 32) -> Transform:
    import random

    def _t(s: str) -> str:
        rng = random.Random(seed)
        blocks = [s[i : i + chunk] for i in range(0, len(s), chunk)]
        rng.shuffle(blocks)
        return "".join(blocks)

    return _t


def t_charset_remap(seed: int = 13) -> Transform:
    import random, string

    rng = random.Random(seed)
    alpha = list(string.ascii_lowercase)
    perm = alpha[:]
    rng.shuffle(perm)
    m = {a: b for a, b in zip(alpha, perm)}
    mU = {a.upper(): b.upper() for a, b in zip(alpha, perm)}

    def _t(s: str) -> str:
        out = []
        for ch in s:
            if ch in m:
                out.append(m[ch])
            elif ch in mU:
                out.append(mU[ch])
            else:
                out.append(ch)
        return "".join(out)

    return _t


def t_prefer_compressible(seed: int = 21, keep_ratio: float = 0.4, window: int = 80) -> Transform:
    import random, zlib

    keep_ratio = max(0.1, min(1.0, float(keep_ratio)))
    window = max(16, int(window))

    def _score(w: str) -> float:
        raw = w.encode("utf-8", errors="ignore")
        comp = zlib.compress(raw, level=9)
        return len(comp) / max(1, len(raw))

    def _t(s: str) -> str:
        if len(s) <= window:
            return s
        rng = random.Random(seed)
        wins = []
        step = max(1, window // 2)
        for i in range(0, len(s) - window + 1, step):
            w = s[i : i + window]
            wins.append((i, w, _score(w)))
        if not wins:
            return s
        wins.sort(key=lambda x: x[2])  # most compressible first
        k = max(1, int(len(wins) * keep_ratio))
        chosen = wins[:k]
        chosen.sort(key=lambda x: x[0])
        return "\n".join(w for _, w, _ in chosen)

    return _t


def t_force_uniformity(target_len: int = 240) -> Transform:
    import re

    target_len = max(50, int(target_len))

    def _t(s: str) -> str:
        s2 = s.replace("\r\n", "\n")
        s2 = re.sub(r"[ \t]+", " ", s2)
        s2 = re.sub(r"[^\w\s\.\,\;\:\-\(\)\[\]\/]", "", s2)
        s2 = s2.strip()
        if len(s2) > target_len:
            s2 = s2[:target_len]
        else:
            s2 = s2.ljust(target_len)
        return s2

    return _t


def t_drop_everything_but_class() -> Transform:
    """
    Maximal representational collapse:
    letters -> 'a', digits -> '0', whitespace kept, punctuation dropped.
    Destroys lexical identity while preserving gross class pattern.
    """
    import re

    def _t(s: str) -> str:
        out = []
        for ch in s:
            if ch.isdigit():
                out.append("0")
            elif ch.isalpha():
                out.append("a")
            elif ch.isspace():
                out.append(ch)
            else:
                # drop punctuation/symbols entirely
                continue
        s2 = "".join(out)
        s2 = re.sub(r"[ \t]+", " ", s2).strip()
        return s2

    return _t


def default_tco() -> TotalCollapseOperator:
    baseline = [
        ("id", t_identity),
        ("ws", t_whitespace_collapse),
        ("rev", t_reverse),
        ("vow-", t_drop_vowels),
        ("shuf", t_shuffle_words(seed=3)),
    ]

    # cost must be monotonic (increasing destructiveness)
    collapse_schedule = [
        (1.0, "ws", t_whitespace_collapse),
        (2.0, "shuf_words", t_shuffle_words(seed=9)),
        (3.0, "chunk_shuffle", t_chunk_shuffle(seed=11, chunk=24)),
        (4.0, "charset_remap", t_charset_remap(seed=13)),
        (5.0, "prefer_compress", t_prefer_compressible(seed=21, keep_ratio=0.35, window=70)),
        (6.0, "force_uniform", t_force_uniformity(target_len=220)),
        (7.0, "class_only", t_drop_everything_but_class()),
        (8.0, "reverse", t_reverse),
    ]

    return TotalCollapseOperator(
        baseline_transforms=baseline,
        collapse_schedule=collapse_schedule,
        eps=1e-4,
        clamp=True,
    )