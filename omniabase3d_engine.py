#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Omniabase 3D · Multibase Spatial Mathematics Engine (v1.0.1)
MB-X.01 / Logical Origin Node (L.O.N.) — Mirror

Computes tri-axis multibase coherence signals (Cx, Cy, Cz),
normalized coherence tensor I3, hypercoherence surface H3,
divergence proxy, surprisal, and summary metrics.

License: MIT
DOI: 10.5281/zenodo.17270742
Repo: https://github.com/Tuttotorna/lon-mirror/tree/main/omniabase-3d
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from statistics import mean
from typing import List, Tuple, Optional

EPS = 1e-12
VERSION = "1.0.1"


# ---------- math primitives ----------

def vdc(n: int, base: int) -> float:
    """Van der Corput sequence in given base ∈ [0,1)."""
    v, denom = 0.0, 1.0
    while n:
        n, rem = divmod(n, base)
        denom *= base
        v += rem / denom
    return v


def ewma(prev: float, new: float, alpha: float) -> float:
    """Exponential smoothing. alpha ∈ (0,1]."""
    return alpha * new + (1 - alpha) * prev


def norm3(x: float, y: float, z: float) -> float:
    return math.sqrt(x * x + y * y + z * z)


def safe_tanh(x: float) -> float:
    # numeric guard
    if x > 20:
        return 1.0
    if x < -20:
        return -1.0
    return math.tanh(x)


# ---------- core engine ----------

def generate_coherence_streams(
    steps: int,
    bases: Tuple[int, int, int],
    smooth: float,
    seed: Optional[int],
    jitter: float
) -> Tuple[List[float], List[float], List[float]]:
    """
    Build Cx, Cy, Cz in [0,1] using base-specific low-discrepancy signals,
    lightly smoothed to emulate continuous fields.
    """
    if seed is not None:
        random.seed(seed)

    bx, by, bz = bases
    a = max(0.001, min(1.0, smooth))

    cx = cy = cz = 0.5
    Cx: List[float] = []
    Cy: List[float] = []
    Cz: List[float] = []

    for t in range(1, steps + 1):
        rx = vdc(t, bx)
        ry = vdc(t, by)
        rz = vdc(t, bz)

        if jitter > 0:
            rx += (random.random() - 0.5) * jitter
            ry += (random.random() - 0.5) * jitter
            rz += (random.random() - 0.5) * jitter

        rx = min(1.0, max(0.0, rx))
        ry = min(1.0, max(0.0, ry))
        rz = min(1.0, max(0.0, rz))

        cx = ewma(cx, rx, a)
        cy = ewma(cy, ry, a)
        cz = ewma(cz, rz, a)

        Cx.append(cx)
        Cy.append(cy)
        Cz.append(cz)

    return Cx, Cy, Cz


def compute_fields(
    Cx: List[float],
    Cy: List[float],
    Cz: List[float],
    alpha: float
):
    """
    For each step:
      - I3 = <Cx,Cy,Cz> / ||<Cx,Cy,Cz>||
      - grad ≈ ||C_t - C_{t-1}||
      - div  ≈ |sum(I3_t - I3_{t-1})|
      - H3  = tanh( (Cx·Cy·Cz) / (1 + grad) )
      - surprisal s = -ln(max(ε, Cx·Cy·Cz))
    Returns (rows, metrics).
    """
    rows = []
    prev_c = None
    prev_i = None

    s_list, c_mean_list, div_list, h_list = [], [], [], []

    tau = -math.log(max(EPS, alpha))
    accept_count = 0

    for t, (cx, cy, cz) in enumerate(zip(Cx, Cy, Cz), start=1):
        n = max(EPS, norm3(cx, cy, cz))
        i3x, i3y, i3z = cx / n, cy / n, cz / n

        if prev_c is None:
            grad = 0.0
        else:
            dx = cx - prev_c[0]
            dy = cy - prev_c[1]
            dz = cz - prev_c[2]
            grad = norm3(dx, dy, dz)

        if prev_i is None:
            div = 0.0
        else:
            div = abs((i3x - prev_i[0]) + (i3y - prev_i[1]) + (i3z - prev_i[2]))

        prod = cx * cy * cz
        h3 = safe_tanh(prod / (1.0 + grad))

        s = -math.log(max(EPS, prod))
        c_bar = (cx + cy + cz) / 3.0

        if div < tau:
            accept_count += 1

        s_list.append(s)
        c_mean_list.append(c_bar)
        div_list.append(div)
        h_list.append(h3)

        rows.append({
            "t": t,
            "Cx": cx, "Cy": cy, "Cz": cz,
            "I3x": i3x, "I3y": i3y, "I3z": i3z,
            "grad": grad,
            "div": div,
            "H3": h3,
            "prod": prod,
            "surprisal": s
        })

        prev_c = (cx, cy, cz)
        prev_i = (i3x, i3y, i3z)

    metrics = {
        "version": VERSION,
        "steps": len(Cx),
        "alpha": alpha,
        "tau": tau,
        "mean_coherence": mean(c_mean_list),
        "mean_divergence": mean(div_list),
        "mean_surprisal": mean(s_list),
        "mean_H3": mean(h_list),
        "accept_ratio_divergence": accept_count / len(Cx)
    }
    return rows, metrics


# ---------- IO ----------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_tensor_csv(path: str, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "Cx", "Cy", "Cz", "I3x", "I3y", "I3z", "grad", "div", "H3", "prod", "surprisal"])
        for r in rows:
            w.writerow([r["t"], r["Cx"], r["Cy"], r["Cz"],
                        r["I3x"], r["I3y"], r["I3z"],
                        r["grad"], r["div"], r["H3"],
                        r["prod"], r["surprisal"]])


def write_surface_csv(path: str, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "H3"])
        for r in rows:
            w.writerow([r["t"], r["H3"]])


def write_metrics_json(path: str, metrics: dict, bases: Tuple[int, int, int], args: argparse.Namespace):
    payload = {
        "module": "Omniabase 3D · Multibase Spatial Mathematics Engine",
        "version": VERSION,
        "doi": "10.5281/zenodo.17270742",
        "license": "MIT",
        "bases": {"x": bases[0], "y": bases[1], "z": bases[2]},
        "cli": {
            "steps": args.steps,
            "alpha": args.alpha,
            "smooth": args.smooth,
            "seed": args.seed,
            "jitter": args.jitter
        },
        "metrics": metrics
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(description="Omniabase 3D · Multibase Spatial Mathematics Engine (v1.0.1)")
    p.add_argument("--bases", nargs=3, type=int, metavar=("BX", "BY", "BZ"),
                   required=True, help="three integer bases, e.g. 8 12 16 (each ≥ 2)")
    p.add_argument("--steps", type=int, default=1000, help="number of steps (default: 1000)")
    p.add_argument("--alpha", type=float, default=0.005, help="divergence significance level (default: 0.005)")
    p.add_argument("--smooth", type=float, default=0.15, help="EWMA smoothing factor in (0,1] (default: 0.15)")
    p.add_argument("--seed", type=int, default=None, help="PRNG seed for reproducibility")
    p.add_argument("--jitter", type=float, default=0.002, help="uniform jitter amplitude in raw signals (default: 0.002, 0 to disable)")
    p.add_argument("--outdir", type=str, default="omniabase-3d/metrics", help="output directory")
    return p.parse_args()


def main():
    args = parse_args()

    bx, by, bz = args.bases
    if any(b <= 1 for b in (bx, by, bz)):
        raise SystemExit("All bases must be ≥ 2.")
    if not (0 < args.smooth <= 1):
        raise SystemExit("smooth must be in (0,1].")
    if not (0 < args.alpha <= 1):
        raise SystemExit("alpha must be in (0,1].")
    if args.jitter < 0:
        raise SystemExit("jitter must be ≥ 0.")

    ensure_dir(args.outdir)

    # 1) signals
    Cx, Cy, Cz = generate_coherence_streams(
        steps=args.steps,
        bases=(bx, by, bz),
        smooth=args.smooth,
        seed=args.seed,
        jitter=args.jitter
    )

    # 2) fields + metrics
    rows, metrics = compute_fields(Cx, Cy, Cz, alpha=args.alpha)

    # 3) outputs
    tensor_path = os.path.join(args.outdir, "tensor_I3.csv")
    surface_path = os.path.join(args.outdir, "surface_H3.csv")
    metrics_path = os.path.join(args.outdir, "metrics.json")

    write_tensor_csv(tensor_path, rows)
    write_surface_csv(surface_path, rows)
    write_metrics_json(metrics_path, metrics, (bx, by, bz), args)

    # terse console summary
    print(f"[OK] v{VERSION} steps={args.steps} bases=({bx},{by},{bz}) outdir={args.outdir}")
    print(f" mean_coherence={metrics['mean_coherence']:.6f}"
          f" mean_divergence={metrics['mean_divergence']:.6f}"
          f" mean_surprisal={metrics['mean_surprisal']:.6f}"
          f" mean_H3={metrics['mean_H3']:.6f}"
          f" accept_ratio_divergence={metrics['accept_ratio_divergence']:.3f}")


if __name__ == "__main__":
    main()