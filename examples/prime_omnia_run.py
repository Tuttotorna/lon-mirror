# examples/prime_omnia_run.py
from __future__ import annotations

from omnia.engine.prime_kernel import PrimeKernel
from omnia.lenses.prime_field_lens import PrimeFieldLens


def main():
    primes = [
        2, 3, 5, 7, 11, 13, 17, 19,
        23, 29, 31, 37, 41, 43, 47, 53
    ]

    kernel = PrimeKernel(lenses=[
        ("prime_field", PrimeFieldLens(k=2)),
    ])

    report = kernel.measure(primes)

    print("Ω_prime =", round(report.omega_prime, 6))
    print("Lens:", list(report.per_lens.keys()))
    print("Ω_prime_field =", round(report.per_lens["prime_field"]["omega"], 6))


if __name__ == "__main__":
    main()