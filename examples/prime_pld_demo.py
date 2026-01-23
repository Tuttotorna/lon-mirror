# examples/prime_pld_demo.py
from __future__ import annotations

from omnia.lenses.prime_field import compute_pld


def main():
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    res = compute_pld(primes, k=2)
    # stampa ordinata per primo
    for p in sorted(res.pld_by_prime):
        print(p, "PLD=", round(res.pld_by_prime[p], 6), "dk=", round(res.dk_by_prime[p], 6))


if __name__ == "__main__":
    main()