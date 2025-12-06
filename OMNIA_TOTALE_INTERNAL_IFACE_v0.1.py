from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Iterable, Callable, Any, Optional
import math

import numpy as np


"""
OMNIA_TOTALE_INTERNAL_IFACE_v0.1
Author: Massimiliano Brighindi (concepts) + MBX IA (formalization)

Goal:
    Minimal internal-interface stub showing how OMNIA (BASE/TIME/CAUSA)
    can plug into an LLM forward pass.

    - Token-level Ω-maps from token ids + logits.
    - Sequence-level Ω-score from BASE/TIME/CAUSA fused metrics.
    - Self-revision wrapper pattern for LLM calls.

    This is framework-agnostic: it does NOT require torch / TF.
    It expects generic callables and numpy arrays, so it can be adapted
    to real inference pipelines by xAI or other teams.
"""


# =========================
# 1. CORE DATA STRUCTURES
# =========================

@dataclass
class OmniaLensConfig:
    bases: Iterable[int] = (2, 3, 5, 7, 11, 13, 17, 19)
    length_weight: float = 1.0
    length_exponent: float = 1.0
    divisibility_bonus: float = 0.5
    short_window: int = 16
    long_window: int = 64
    hist_bins: int = 24
    max_lag: int = 4
    strength_threshold: float = 0.25
    w_base: float = 1.0
    w_time: float = 1.0
    w_causa: float = 1.0
    epsilon: float = 1e-9

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OmniaTokenOmega:
    index: int
    token_id: int
    omega_base: float
    omega_time: float
    omega_causa: float
    omega_total: float


@dataclass
class OmniaStepSnapshot:
    """
    Minimal view of an internal LLM step.

    token_ids: shape (T,)
    logits: shape (T, V) or (V,) for last token
    attn: optional attention tensor used as causal signal
          shape (H, T, T) or (T, T)
    """
    token_ids: np.ndarray             # int64 [T]
    logits: np.ndarray                # float32 [T, V] or [V]
    attn: Optional[np.ndarray] = None # float32 [H, T, T] or [T, T]


@dataclass
class OmniaEvalResult:
    omega_sequence: float
    omega_tokens: List[OmniaTokenOmega]
    components: Dict[str, float]


# =========================
# 2. BASE LENS (OMNIABASE)
# =========================

def _digits_in_base_np(n: int, b: int) -> np.ndarray:
    if n < 0:
        raise ValueError("n must be non-negative")
    if b <= 1:
        raise ValueError("base must be >= 2")
    if n == 0:
        return np.array([0], dtype=int)
    d = []
    while n > 0:
        d.append(n % b)
        n //= b
    return np.array(d[::-1], dtype=int)


def _normalized_entropy_base(n: int, b: int) -> float:
    digits = _digits_in_base_np(int(n), b)
    L = len(digits)
    if L == 0:
        return 0.0
    counts = np.bincount(digits, minlength=b).astype(float)
    probs = counts[counts > 0] / L
    if probs.size == 0:
        return 0.0
    H = -np.sum(probs * np.log2(probs))
    Hmax = math.log2(b)
    return float(H / Hmax) if Hmax > 0 else 0.0


def _sigma_b(
    n: int,
    b: int,
    length_weight: float,
    length_exponent: float,
    divisibility_bonus: float,
) -> float:
    digits = _digits_in_base_np(int(n), b)
    L = len(digits)
    if L == 0:
        return 0.0
    counts = np.bincount(digits, minlength=b).astype(float)
    probs = counts[counts > 0] / L
    if probs.size == 0:
        Hn = 0.0
    else:
        H = -np.sum(probs * np.log2(probs))
        Hmax = math.log2(b)
        Hn = float(H / Hmax) if Hmax > 0 else 0.0
    length_term = length_weight * (1.0 - Hn) / (L ** length_exponent)
    div_term = divisibility_bonus * (1.0 if n % b == 0 else 0.0)
    return float(length_term + div_term)


def _omniabase_scalar(
    n: int,
    cfg: OmniaLensConfig,
) -> float:
    bases = list(cfg.bases)
    sigmas = []
    for b in bases:
        sig = _sigma_b(
            n,
            b,
            length_weight=cfg.length_weight,
            length_exponent=cfg.length_exponent,
            divisibility_bonus=cfg.divisibility_bonus,
        )
        sigmas.append(sig)
    if not sigmas:
        return 0.0
    return float(np.mean(sigmas))


# =========================
# 3. TIME LENS (OMNIATEMPO)
# =========================

def _histogram_probs(x: np.ndarray, bins: int) -> np.ndarray:
    if x.size == 0:
        return np.zeros(bins, dtype=float)
    hist, _ = np.histogram(x, bins=bins, density=False)
    total = hist.sum()
    if total == 0:
        return np.zeros_like(hist, dtype=float)
    return hist.astype(float) / total


def _omniatempo_scalar(
    series: np.ndarray,
    cfg: OmniaLensConfig,
) -> float:
    x = np.asarray(series, dtype=float)
    if x.size == 0:
        return 0.0

    sw = min(cfg.short_window, x.size)
    lw = min(cfg.long_window, x.size)
    short_seg = x[-sw:]
    long_seg = x[-lw:]

    p = _histogram_probs(short_seg, bins=cfg.hist_bins) + cfg.epsilon
    q = _histogram_probs(long_seg, bins=cfg.hist_bins) + cfg.epsilon
    p /= p.sum()
    q /= q.sum()

    kl_pq = float(np.sum(p * np.log(p / q)))
    kl_qp = float(np.sum(q * np.log(q / p)))
    regime = 0.5 * (kl_pq + kl_qp)
    return float(math.log(1.0 + regime))


# =========================
# 4. CAUSAL LENS (OMNIACAUSA)
# =========================

def _lagged_corr_np(x: np.ndarray, y: np.ndarray, lag: int) -> float:
    if lag > 0:
        x_l = x[:-lag]
        y_l = y[lag:]
    elif lag < 0:
        lag_abs = -lag
        x_l = x[lag_abs:]
        y_l = y[:-lag_abs]
    else:
        x_l = x
        y_l = y

    if x_l.size < 2 or y_l.size < 2:
        return 0.0

    x_mean = x_l.mean()
    y_mean = y_l.mean()
    num = float(np.sum((x_l - x_mean) * (y_l - y_mean)))
    den = math.sqrt(
        float(np.sum((x_l - x_mean) ** 2) * np.sum((y_l - y_mean) ** 2))
    )
    if den == 0:
        return 0.0
    return float(num / den)


def _omniacausa_scalar(
    attn: Optional[np.ndarray],
    cfg: OmniaLensConfig,
) -> float:
    """
    attn: [H, T, T] or [T, T] or None
    We treat row-wise mean attention as time series per head.
    """
    if attn is None:
        return 0.0
    a = np.asarray(attn, dtype=float)
    if a.ndim == 2:
        a = a[None, :, :]  # [1, T, T]
    H, T, _ = a.shape
    if T < 3:
        return 0.0

    # For each head, build a 1D series: mean attention per step
    series = a.mean(axis=2)  # [H, T]
    keys = list(range(H))
    edges_strength: List[float] = []
    lags = list(range(-cfg.max_lag, cfg.max_lag + 1))
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            if i == j:
                continue
            x = series[i]
            y = series[j]
            best_corr = 0.0
            for lag in lags:
                c = _lagged_corr_np(x, y, lag)
                if abs(c) > abs(best_corr):
                    best_corr = c
            if abs(best_corr) >= cfg.strength_threshold:
                edges_strength.append(abs(best_corr))
    if not edges_strength:
        return 0.0
    return float(np.mean(edges_strength))


# =========================
# 5. TOKEN-LEVEL Ω MAP
# =========================

def compute_token_omega_map(
    snapshot: OmniaStepSnapshot,
    cfg: OmniaLensConfig,
) -> List[OmniaTokenOmega]:
    """
    Build token-level Ω map using:
        - BASE: omniabase on token id.
        - TIME: local logit max-prob dynamics across tokens.
        - CAUSA: same causal scalar for all tokens (from attn).
    """
    token_ids = np.asarray(snapshot.token_ids, dtype=int)
    if snapshot.logits.ndim == 1:
        logits = snapshot.logits[None, :]  # [1, V]
    else:
        logits = snapshot.logits  # [T, V]
    T = token_ids.shape[0]

    # Probabilities per token
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True) + cfg.epsilon
    max_probs = probs.max(axis=-1)  # [T]

    # Time lens: treat max_probs over sequence as time series
    # For per-token TIME, use sliding window around each token.
    time_series = max_probs
    causa_scalar = _omniacausa_scalar(snapshot.attn, cfg)

    tokens_omega: List[OmniaTokenOmega] = []
    for idx in range(T):
        tid = int(token_ids[idx])

        base_val = _omniabase_scalar(tid, cfg)

        # Local window around token for TIME
        left = max(0, idx - cfg.short_window // 2)
        right = min(T, idx + cfg.short_window // 2)
        local_series = time_series[left:right]
        time_val = _omniatempo_scalar(local_series, cfg)

        omega_total = (
            cfg.w_base * base_val
            + cfg.w_time * time_val
            + cfg.w_causa * causa_scalar
        )

        tokens_omega.append(
            OmniaTokenOmega(
                index=idx,
                token_id=tid,
                omega_base=base_val,
                omega_time=time_val,
                omega_causa=causa_scalar,
                omega_total=omega_total,
            )
        )
    return tokens_omega


# =========================
# 6. SEQUENCE-LEVEL Ω SCORE
# =========================

def evaluate_step_omega(
    snapshot: OmniaStepSnapshot,
    cfg: Optional[OmniaLensConfig] = None,
) -> OmniaEvalResult:
    """
    High-level wrapper:
        - builds token-level Ω map
        - aggregates into a sequence Ω score
    """
    if cfg is None:
        cfg = OmniaLensConfig()

    tokens_omega = compute_token_omega_map(snapshot, cfg)

    base_vals = np.array([t.omega_base for t in tokens_omega], dtype=float)
    time_vals = np.array([t.omega_time for t in tokens_omega], dtype=float)
    causa_vals = np.array([t.omega_causa for t in tokens_omega], dtype=float)

    base_comp = float(base_vals.mean()) if base_vals.size else 0.0
    time_comp = float(time_vals.mean()) if time_vals.size else 0.0
    causa_comp = float(causa_vals.mean()) if causa_vals.size else 0.0

    omega_seq = (
        cfg.w_base * base_comp
        + cfg.w_time * time_comp
        + cfg.w_causa * causa_comp
    )

    components = {
        "base_mean": base_comp,
        "time_mean": time_comp,
        "causa_mean": causa_comp,
    }

    return OmniaEvalResult(
        omega_sequence=omega_seq,
        omega_tokens=tokens_omega,
        components=components,
    )


# =========================
# 7. SELF-REVISION WRAPPER
# =========================

def omnia_self_revision_loop(
    prompt: str,
    llm_call: Callable[[str], Dict[str, Any]],
    encode_step: Callable[[Dict[str, Any]], OmniaStepSnapshot],
    cfg: Optional[OmniaLensConfig] = None,
    max_revisions: int = 2,
    omega_threshold: float = 0.0,
) -> Dict[str, Any]:
    """
    Generic Ω-driven self-revision loop.

    llm_call:
        prompt -> {"text": str, "token_ids": [...], "logits": np.ndarray,
                   "attn": np.ndarray | None, ...}

    encode_step:
        converts llm_call output to OmniaStepSnapshot.

    Logic:
        1. call LLM with prompt
        2. compute Ω-score
        3. if Ω-score < threshold and revisions_left > 0:
               build new prompt with diagnostic tag
               repeat
        4. return final {"text": ..., "omega": ..., "components": ..., "revisions": int}
    """
    if cfg is None:
        cfg = OmniaLensConfig()

    current_prompt = prompt
    best_result: Optional[Dict[str, Any]] = None

    for step in range(max_revisions + 1):
        raw = llm_call(current_prompt)
        snapshot = encode_step(raw)
        eval_res = evaluate_step_omega(snapshot, cfg)

        enriched = dict(raw)
        enriched["omega_sequence"] = eval_res.omega_sequence
        enriched["omega_components"] = eval_res.components
        enriched["omega_tokens"] = [asdict(t) for t in eval_res.omega_tokens]
        enriched["revision_step"] = step

        if best_result is None or eval_res.omega_sequence > best_result["omega_sequence"]:
            best_result = enriched

        if eval_res.omega_sequence >= omega_threshold or step == max_revisions:
            break

        # Build revision prompt (simple pattern, customizable by integrator)
        current_prompt = (
            f"{prompt}\n\n[Ω-revision #{step+1} – previous answer had low Ω={eval_res.omega_sequence:.3f}. "
            f"Please improve coherence, reduce contradictions, and tighten reasoning.]"
        )

    assert best_result is not None
    return best_result


# =========================
# 8. DEMO (FRAMEWORK-AGNOSTIC)
# =========================

def _fake_llm_call(prompt: str) -> Dict[str, Any]:
    """
    Minimal fake LLM:
        - encodes chars as token_ids
        - generates random logits and no real attention
    Only for demonstration of the interface.
    """
    vocab_size = 64
    token_ids = np.array([ord(c) % vocab_size for c in prompt], dtype=int)
    T = token_ids.shape[0] if token_ids.size > 0 else 1
    logits = np.random.normal(size=(T, vocab_size)).astype(float)
    attn = np.random.rand(2, T, T).astype(float)  # 2 fake heads

    return {
        "text": prompt.upper(),
        "token_ids": token_ids,
        "logits": logits,
        "attn": attn,
    }


def _encode_step_from_fake(raw: Dict[str, Any]) -> OmniaStepSnapshot:
    return OmniaStepSnapshot(
        token_ids=np.asarray(raw["token_ids"], dtype=int),
        logits=np.asarray(raw["logits"], dtype=float),
        attn=np.asarray(raw["attn"], dtype=float),
    )


def demo_internal_iface():
    """
    Run a minimal demo of:
        - token-level Ω map
        - sequence-level Ω score
        - Ω-driven self-revision loop
    """
    np.random.seed(0)
    cfg = OmniaLensConfig()

    prompt = "test prompt for OMNIA internal iface"
    raw = _fake_llm_call(prompt)
    snapshot = _encode_step_from_fake(raw)

    eval_res = evaluate_step_omega(snapshot, cfg)
    print("=== OMNIA_TOTALE_INTERNAL_IFACE v0.1 demo ===")
    print(f"Ω_sequence ≈ {eval_res.omega_sequence:.4f}")
    print("components:", eval_res.components)
    print("first 3 token Ω entries:")
    for t in eval_res.omega_tokens[:3]:
        print("  ", t)

    print("\n=== Ω self-revision demo ===")
    final = omnia_self_revision_loop(
        prompt,
        llm_call=_fake_llm_call,
        encode_step=_encode_step_from_fake,
        cfg=cfg,
        max_revisions=2,
        omega_threshold=0.5,
    )
    print("final text:", final["text"])
    print("final Ω:", final["omega_sequence"])
    print("revisions used:", final["revision_step"])


if __name__ == "__main__":
    demo_internal_iface()
```0