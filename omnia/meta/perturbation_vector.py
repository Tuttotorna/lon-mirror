from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# Reuse the meaning-blind aperspective lens as the baseline
from omnia.lenses.aperspective_invariance import AperspectiveInvariance

Transform = Callable[[str], str]


@dataclass(frozen=True)
class PerturbationVectorResult:
    """
    PV-1.0 (string/text baseline)

    omega_ap:
        Baseline aperspective invariance omega score.
        (No privileged observer, measured under neutral transforms)

    omega_by_type:
        Measured omega score after applying perturbation transforms of each type.

    pi_by_type:
        Perturbation indices per type:
            PI_type = omega_ap - omega_type

    dominant_type:
        The type with maximum PI (largest structural loss vs baseline).

    notes:
        Minimal non-narrative diagnostics.
    """
    omega_ap: float
    omega_by_type: Dict[str, float]
    pi_by_type: Dict[str, float]
    dominant_type: Optional[str]
    notes: Dict[str, float]


class PerturbationVector:
    """
    Perturbation Vector PV-1.0

    Goal: measure multiple non-semantic perturbation classes as losses of
    aperspective invariance (omega) under typed perturbation transforms.

    All indices are measured in the same coordinate:
        PI_k = Ω_ap - Ω_k

    Where:
        Ω_ap = baseline aperspective invariance (no privileged POV)
        Ω_k  = invariance after introducing a typed perturbation family

    Types (recommended):
        OBS  -> observer / perspective perturbation
        REP  -> representational / encoding perturbation
        TMP  -> temporalization / ordering perturbation
        GOAL -> goal/selection/optimization perturbation (structural, meaning-blind)
        COH  -> forced-coherence / over-normalization perturbation
    """

    def __init__(
        self,
        *,
        baseline_transforms: Sequence[Tuple[str, Transform]],
        typed_transforms: Dict[str, Sequence[Tuple[str, Transform]]],
        clamp: bool = True,
    ):
        if not baseline_transforms:
            raise ValueError("baseline_transforms must be non-empty")
        if not typed_transforms:
            raise ValueError("typed_transforms must be non-empty")

        self.baseline_transforms = list(baseline_transforms)
        self.typed_transforms: Dict[str, List[Tuple[str, Transform]]] = {
            k: list(v) for k, v in typed_transforms.items()
        }
        self.clamp = bool(clamp)

        # Baseline aperspective engine
        self._ap = AperspectiveInvariance(transforms=self.baseline_transforms)

    @staticmethod
    def _clamp01(x: float) -> float:
        if x < 0.0:
            return 0.0
        if x > 1.0:
            return 1.0
        return x

    def _measure_omega(self, x: str, transforms: Sequence[Tuple[str, Transform]]) -> float:
        # AperspectiveInvariance returns omega_score in [0,1] by construction
        eng = AperspectiveInvariance(transforms=list(transforms))
        r = eng.measure(x)
        omega = float(r.omega_score)
        return self._clamp01(omega) if self.clamp else omega

    def measure(self, x: str) -> PerturbationVectorResult:
        # 1) Baseline: aperspective omega under neutral transforms
        base = self._ap.measure(x)
        omega_ap = float(base.omega_score)
        omega_ap = self._clamp01(omega_ap) if self.clamp else omega_ap

        omega_by_type: Dict[str, float] = {}
        pi_by_type: Dict[str, float] = {}

        # 2) For each perturbation type, measure omega after applying that family
        for tname, tlist in self.typed_transforms.items():
            if not tlist:
                continue
            omega_k = self._measure_omega(x, transforms=tlist)
            omega_by_type[tname] = omega_k
            pi = omega_ap - omega_k
            if self.clamp:
                # PI is also bounded in [-1,1] but expected >= 0. Clamp negative to 0.
                pi = max(0.0, min(1.0, pi))
            pi_by_type[tname] = pi

        # 3) Dominant perturbation type
        dominant_type: Optional[str] = None
        if pi_by_type:
            dominant_type = max(pi_by_type.items(), key=lambda kv: kv[1])[0]

        notes: Dict[str, float] = {}
        notes["omega_ap"] = omega_ap
        if dominant_type is not None:
            notes["pi_max"] = float(pi_by_type[dominant_type])
        notes["pi_sum"] = float(sum(pi_by_type.values())) if pi_by_type else 0.0

        return PerturbationVectorResult(
            omega_ap=omega_ap,
            omega_by_type=omega_by_type,
            pi_by_type=pi_by_type,
            dominant_type=dominant_type,
            notes=notes,
        )


# -----------------------------
# Default transform library (meaning-blind)
# -----------------------------

def t_identity(s: str) -> str:
    return s


def t_whitespace_collapse(s: str) -> str:
    import re
    s = s.replace("\r\n", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def t_reverse(s: str) -> str:
    return s[::-1]


def t_drop_vowels(s: str) -> str:
    import re
    return re.sub(r"[aeiouAEIOUàèéìòùÀÈÉÌÒÙ]", "", s)


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
    """
    Temporalization-like: shuffles fixed-size chunks, destroys locality/order.
    """
    import random
    def _t(s: str) -> str:
        rng = random.Random(seed)
        blocks = [s[i:i+chunk] for i in range(0, len(s), chunk)]
        rng.shuffle(blocks)
        return "".join(blocks)
    return _t


def t_base_repr(seed: int = 7, base: int = 7) -> Transform:
    """
    Representation perturbation: replace decimal integers with base-N representation (+ tiny perturbation).
    """
    import random, re
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

        def repl(m: re.Match) -> str:
            n = int(m.group(0))
            if rng.random() < 0.1:
                n += rng.choice([-1, 1])
            return to_base(n)

        return re.sub(r"\b\d+\b", repl, s)

    return _t


def t_charset_remap(seed: int = 13) -> Transform:
    """
    Representation perturbation: remap letters deterministically (substitution cipher),
    preserving length and rough class structure, destroying human readability.
    """
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


def t_prefer_compressible(seed: int = 21, keep_ratio: float = 0.6, window: int = 80) -> Transform:
    """
    Goal/selection perturbation (meaning-blind):
    keep only the most compressible windows (introduces preference/selection).
    """
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
        # sample windows
        wins = []
        step = max(1, window // 2)
        for i in range(0, len(s) - window + 1, step):
            w = s[i:i+window]
            wins.append((i, w, _score(w)))
        if not wins:
            return s
        # lower ratio => more compressible
        wins.sort(key=lambda x: x[2])
        k = max(1, int(len(wins) * keep_ratio))
        chosen = wins[:k]
        # preserve original order of selected windows to avoid pure temporalization
        chosen.sort(key=lambda x: x[0])
        return "\n".join(w for _, w, _ in chosen)

    return _t


def t_force_uniformity(seed: int = 34, target_len: int = 200) -> Transform:
    """
    Forced-coherence perturbation:
    aggressively normalizes to uniform length and charset, removing outliers.
    """
    import re
    target_len = max(50, int(target_len))

    def _t(s: str) -> str:
        s2 = s.replace("\r\n", "\n")
        s2 = re.sub(r"[ \t]+", " ", s2)
        s2 = re.sub(r"[^\w\s\.\,\;\:\-\(\)\[\]\/]", "", s2)  # drop high-variance chars
        s2 = s2.strip()
        if len(s2) > target_len:
            s2 = s2[:target_len]
        else:
            s2 = s2.ljust(target_len)
        return s2

    return _t


def default_pv() -> PerturbationVector:
    """
    Sensible defaults for PV on text.
    Baseline transforms are "neutral" (aperspective) stressors.
    Typed transforms introduce different perturbation classes.
    """
    baseline = [
        ("id", t_identity),
        ("ws", t_whitespace_collapse),
        ("rev", t_reverse),
        ("vow-", t_drop_vowels),
        ("shuf", t_shuffle_words(seed=3)),
        ("base7", t_base_repr(seed=7, base=7)),
    ]

    typed = {
        # Observer-like: introduce asymmetry/perspective (strong normalization + selection)
        "OBS": [
            ("ws", t_whitespace_collapse),
            ("force_uniform", t_force_uniformity(seed=34, target_len=240)),
            ("prefer_compress", t_prefer_compressible(seed=21, keep_ratio=0.5, window=90)),
        ],
        # Representation: encoding-level remaps
        "REP": [
            ("base7", t_base_repr(seed=7, base=7)),
            ("charset_map", t_charset_remap(seed=13)),
        ],
        # Temporalization: destroy order/locality
        "TMP": [
            ("rev", t_reverse),
            ("chunk_shuffle", t_chunk_shuffle(seed=2, chunk=32)),
            ("shuf", t_shuffle_words(seed=3)),
        ],
        # Goal/selection: preference without semantics
        "GOAL": [
            ("prefer_compress", t_prefer_compressible(seed=21, keep_ratio=0.4, window=80)),
        ],
        # Forced coherence: over-normalization
        "COH": [
            ("force_uniform", t_force_uniformity(seed=34, target_len=240)),
        ],
    }

    return PerturbationVector(
        baseline_transforms=baseline,
        typed_transforms=typed,
        clamp=True,
    )