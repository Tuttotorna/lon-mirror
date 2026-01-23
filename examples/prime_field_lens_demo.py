# examples/prime_field_lens_demo.py
from __future__ import annotations

from omnia.lenses.prime_field_lens import PrimeFieldLens


def main():
    primes = [
        2, 3, 5, 7, 11, 13, 17, 19,
        23, 29, 31, 37, 41, 43, 47, 53
    ]

    lens = PrimeFieldLens(k=2)
    out = lens.measure(primes)

    print("Ω_prime_field =", round(out.omega_score, 6))
    # mostra qualche valore
    items = sorted(out.details["pld_by_prime"].items())[:10]
    for p, v in items:
        print(p, "PLD=", round(v, 6), "dk=", round(out.details["dk_by_prime"][p], 6))


if __name__ == "__main__":
    main()