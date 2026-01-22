from __future__ import annotations

from omnia.lenses.prime_regime import prime_state_from_primes
from omnia.lenses.prime_gap_knn import predict_next_gap_knn


def _first_primes(n: int) -> list[int]:
    # deterministic naive prime generator (for demo only)
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


def main() -> None:
    primes = _first_primes(8000)  # demo scale
    mods = [3, 5, 7, 11, 13, 17, 19]
    window = 512
    drift_theta = 0.05

    start_idx = 2
    end_idx = 6000

    states = []
    prev_phi = None
    prev_tau = 0

    for n in range(start_idx, end_idx + 1):
        st = prime_state_from_primes(
            primes=primes,
            idx=n,
            mods=mods,
            window=window,
            drift_theta=drift_theta,
            prev_phi=prev_phi,
            prev_tau=prev_tau,
        )
        states.append(st)
        prev_phi = st.phi
        prev_tau = st.tau

    # print last 20 predictions with STOP reasons
    for n in range(end_idx - 20, end_idx):
        pred = predict_next_gap_knn(
            primes=primes,
            states=states,
            n_idx=n,
            start_idx=start_idx,
            K=25,
            C_min=0.80,
            T_max=0.08,
            S_min=0.35,
        )
        true_g = primes[n + 1] - primes[n]
        st = states[n - start_idx]
        print(
            f"n={n} p={primes[n]}  S={st.S:.3f} T={st.T:.3f} tau={st.tau:4d} | "
            f"true_g={true_g:3d}  pred={pred.g_hat}  conf={pred.confidence:.3f}  "
            f"{'STOP' if pred.stop else 'OK'}  {pred.reason}"
        )


if __name__ == "__main__":
    main()