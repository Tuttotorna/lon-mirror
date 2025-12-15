"""
quick_omnia_test.py

Minimal end-to-end smoke test for the OMNIA package:

- Direct lens calls:
  - omniabase_signature / pbii_index
  - omniatempo_analyze
  - omniacausa_analyze

- Fused engine:
  - run_omnia_totale with BASE + TIME + CAUSA + TOKEN lenses

- ICE gate:
  - ice_gate over OMNIA_TOTALE outputs (adapter -> structural ICEInput)

Author: Massimiliano Brighindi (MB-X.01 / OMNIA)
"""

from __future__ import annotations

from typing import Dict, Iterable, Any, List
import numpy as np

from omnia import (
    # omniabase
    OmniabaseSignature,
    omniabase_signature,
    pbii_index,
    # omniatempo
    OmniatempoResult,
    omniatempo_analyze,
    # omniacausa
    OmniacausaResult,
    omniacausa_analyze,
    # ICE (structural)
    ice_gate,
)

from omnia.engine import run_omnia_totale
from omnia.adapters.ice_from_totale import ice_input_from_omnia_totale


# =========================
# 1. Direct lens tests
# =========================

def test_omniabase(n: int = 173) -> None:
    print("=== OMNIABASE test ===")
    bases = [2, 3, 5, 7, 11, 13, 17, 19]

    sig: OmniabaseSignature = omniabase_signature(n, bases=bases)
    pbii = pbii_index(n, bases=bases)

    print(f"n = {n}")
    print(f"bases = {bases}")
    print(f"sigma_mean = {sig.sigma_mean:.6f}")
    print(f"entropy_mean = {sig.entropy_mean:.6f}")
    print(f"PBII index = {pbii:.6f}")
    print()


def test_omniatempo() -> None:
    print("=== OMNIATEMPO test ===")
    t = np.arange(300)
    series = np.sin(t / 20.0) + 0.05 * np.random.normal(size=t.size)
    series[-100:] += 0.7  # regime shift

    res: OmniatempoResult = omniatempo_analyze(series)

    print(f"global_mean = {res.global_mean:.6f}")
    print(f"global_std  = {res.global_std:.6f}")
    print(f"short_mean  = {res.short_mean:.6f}")
    print(f"short_std   = {res.short_std:.6f}")
    print(f"long_mean   = {res.long_mean:.6f}")
    print(f"long_std    = {res.long_std:.6f}")
    print(f"regime_change_score = {res.regime_change_score:.6f}")
    print()


def test_omniacausa() -> None:
    print("=== OMNIACAUSA test ===")
    t = np.arange(300)
    s1 = np.sin(t / 15.0)
    s2 = np.zeros_like(s1)
    s2[2:] = 0.8 * s1[:-2] + 0.1 * np.random.normal(size=t.size - 2)
    s3 = np.random.normal(size=t.size)

    series_dict: Dict[str, Iterable[float]] = {
        "s1": s1,
        "s2": s2,
        "s3": s3,
    }

    res: OmniacausaResult = omniacausa_analyze(
        series_dict,
        max_lag=5,
        strength_threshold=0.3,
    )

    print(f"edges found = {len(res.edges)}")
    for e in res.edges:
        print(f"  {e.source} -> {e.target}  lag={e.lag:+d}  strength={e.strength:.3f}")
    print()


# =========================
# 2. Fused engine test (BASE + TIME + CAUSA + TOKEN)
# =========================

def test_omnia_engine() -> Any:
    print("=== OMNIA_TOTALE fused engine test ===")

    n = 173

    # time series for omniatempo
    t = np.arange(300)
    series = np.sin(t / 18.0) + 0.05 * np.random.normal(size=t.size)
    series[180:] += 0.6  # regime shift

    # multi-channel series for omniacausa
    s1 = np.sin(t / 12.0)
    s2 = np.zeros_like(s1)
    s2[3:] = 0.7 * s1[:-3] + 0.1 * np.random.normal(size=t.size - 3)
    s3 = np.random.normal(size=t.size)

    series_dict: Dict[str, Iterable[float]] = {"s1": s1, "s2": s2, "s3": s3}

    # token-level example for TOKEN lens
    tokens: List[str] = ["The", "final", "answer", "is", "173"]
    token_numbers: List[int] = [len(tok) for tok in tokens]  # placeholder proxy

    extra: Dict[str, Any] = {"tokens": tokens, "token_numbers": token_numbers}

    result = run_omnia_totale(
        n=n,
        series=series,
        series_dict=series_dict,
        w_base=1.0,
        w_tempo=1.0,
        w_causa=1.0,
        w_token=1.0,
        extra=extra,
    )

    omega_total = getattr(result, "omega_total", None)
    if omega_total is None:
        omega_total = getattr(result, "omega_score", 0.0)

    print(f"Ω_total = {float(omega_total):.6f}")
    print("Per-lens ω contributions:")
    for name, score in result.lens_scores.items():
        print(f"  {name}: {float(score):.6f}")

    print("\nAvailable lens metadata keys:")
    for name, meta in result.lens_metadata.items():
        try:
            keys = list(meta.keys())
        except Exception:
            keys = []
        print(f"  {name}: keys={keys}")

    print()
    return result


# =========================
# 3. ICE gate test (structural adapter)
# =========================

def test_ice_gate(omnia_result: Any) -> None:
    print("=== OMNIA ICE test ===")

    x = ice_input_from_omnia_totale(omnia_result)
    res = ice_gate(x)

    print(f"ICE status     = {res.status}")
    print(f"TruthΩ         = {res.truth_omega:.6f}")
    print(f"Δ              = {res.delta:.6f}")
    print(f"κ              = {res.kappa:.6f}")
    print(f"confidence     = {res.confidence:.6f}")
    print(f"reasons        = {list(res.reasons)}")
    print()


# =========================
# MAIN
# =========================

def main() -> None:
    np.random.seed(0)
    print("Running quick OMNIA smoke tests...\n")

    test_omniabase()
    test_omniatempo()
    test_omniacausa()

    omnia_result = test_omnia_engine()
    test_ice_gate(omnia_result)


if __name__ == "__main__":
    main()