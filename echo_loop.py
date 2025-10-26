# Echo-Cognition Engine — MB-X.01 / L.O.N.
# Author: Massimiliano Brighindi
# License: MIT
# DOI: https://doi.org/10.5281/zenodo.17270742

import numpy as np
import argparse
import csv

def simulate_echo(T=5000, n=6, k=0.45, noise=0.02, alpha=0.005):
    """Simulates a minimal echo-cognition feedback loop."""
    A = np.eye(n)
    B = np.eye(n)
    K = k * np.eye(n)
    x = np.zeros((n, 1))
    x_hat = np.zeros((n, 1))

    log = []
    for t in range(T):
        u = np.random.randn(n, 1) * 0.05
        w = np.random.randn(n, 1) * noise
        x_next = A @ x + B @ u + w
        x_hat_next = A @ x_hat + B @ u + K @ (x_hat - x)

        e = x_hat - x
        C = np.exp(-np.linalg.norm(e))
        S = -np.log(C)

        log.append([t, float(C), float(S)])
        x, x_hat = x_next, x_hat_next

    return log


def save_csv(filename, data, headers):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Echo-Cognition Simulation")
    parser.add_argument("--T", type=int, default=5000)
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--k", type=float, default=0.45)
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--alpha", type=float, default=0.005)
    parser.add_argument("--out_csv", default="echo_log.csv")
    parser.add_argument("--out_metrics", default="echo_metrics.csv")
    args = parser.parse_args()

    log = simulate_echo(T=args.T, n=args.n, k=args.k, noise=args.noise, alpha=args.alpha)
    save_csv(args.out_csv, log, ["t", "C