from __future__ import annotations

from omnia.lenses.prime_regime import prime_state_from_primes
from omnia.lenses.prime_gap_knn import predict_next_gap_knn


def _first_primes(n: int) -> list[int]:
    primes = [2]
    x = 3
    while len(primes) < n:
        is_p = True
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                is_p = False
                break
        if is_p:
            primes.append(x)
        x += 2
    return primes


def _build_states(primes: list[int], mods: list[int], start_idx: int, end_idx: int) -> list:
    states = []
    prev_phi = None
    prev_tau = 0
    for n in range(start_idx, end_idx + 1):
        st = prime_state_from_primes(
            primes=primes,
            idx=n,
            mods=mods,
            window=256,
            drift_theta=0.05,
            prev_phi=prev_phi,
            prev_tau=prev_tau,
        )
        states.append(st)
        prev_phi = st.phi
        prev_tau = st.tau
    return states


def test_knn_is_deterministic():
    primes = _first_primes(3000)
    mods = [3, 5, 7, 11, 13]
    start_idx = 2
    end_idx = 2000
    states = _build_states(primes, mods, start_idx, end_idx)

    n = 1990
    p1 = predict_next_gap_knn(primes, states, n_idx=n, start_idx=start_idx)
    p2 = predict_next_gap_knn(primes, states, n_idx=n, start_idx=start_idx)

    assert p1 == p2


def test_prediction_contracts():
    primes = _first_primes(3000)
    mods = [3, 5, 7, 11, 13]
    start_idx = 2
    end_idx = 1500
    states = _build_states(primes, mods, start_idx, end_idx)

    n = 1490
    pred = predict_next_gap_knn(primes, states, n_idx=n, start_idx=start_idx)

    assert 0.0 <= pred.confidence <= 1.0
    if pred.stop:
        assert pred.g_hat is None
        assert isinstance(pred.reason, str) and len(pred.reason) > 0
    else:
        assert isinstance(pred.g_hat, int)
        assert pred.g_hat > 0