#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Omniabase-3D · Multibase Spatial Mathematics Engine (all-in-one)
MB-X.01 / L.O.N. — v1.2 · MIT · DOI: 10.5281/zenodo.17270742

Funzioni:
- Genera Cx,Cy,Cz con Van der Corput + EWMA
- Calcola I3, H3, gradiente, divergenza, surprisal
- Produce: tensor_I3.csv, surface_H3.csv, metrics.json

Uso:
# Genera internamente
python omniabase3d_engine.py --bases 8 12 16 --steps 1000 --alpha 0.005 --smooth 0.15 --seed 42 --outdir omniabase-3d/metrics
# Oppure leggi segnali da CSV (colonne Cx,Cy,Cz)
python omniabase3d_engine.py --signals signals.csv --steps 800 --alpha 0.005 --outdir omniabase-3d/metrics
"""
from __future__ import annotations
import argparse, csv, json, math, os, random
from statistics import mean

EPS = 1e-12

# ------------------------- primitive -------------------------

def vdc(n: int, base: int) -> float:
    v, denom = 0.0, 1.0
    while n:
        n, rem = divmod(n, base)
        denom *= base
        v += rem / denom
    return v

def ewma(prev: float, new: float, alpha: float) -> float:
    return alpha * new + (1.0 - alpha) * prev

def norm3(x: float, y: float, z: float) -> float:
    return math.sqrt(x * x + y * y + z * z)

def stanh(x: float) -> float:
    if x > 20.0: return 1.0
    if x < -20.0: return -1.0
    e2 = math.exp(2.0 * x)
    return (e2 - 1.0) / (e2 + 1.0)

# --------------------- generazione segnali ---------------------

def generate_coherence_streams(
    steps: int,
    bases: tuple[int, int, int],
    smooth: float = 0.15,
    seed: int | None = None,
    jitter: float = 0.002
) -> tuple[list[float], list[float], list[float]]:
    if seed is not None:
        random.seed(seed)
    bx, by, bz = bases
    if any(b <= 1 for b in (bx, by, bz)):
        raise ValueError("Basi VdC devono essere ≥ 2.")
    a = max(0.001, min(1.0, float(smooth)))
    cx = cy = cz = 0.5
    Cx: list[float] = []; Cy: list[float] = []; Cz: list[float] = []
    for t in range(1, steps + 1):
        rx, ry, rz = vdc(t, bx), vdc(t, by), vdc(t, bz)
        jx, jy, jz = (random.random()-0.5)*jitter, (random.random()-0.5)*jitter, (random.random()-0.5)*jitter
        cx = ewma(cx, min(1.0, max(0.0, rx + jx)), a)
        cy = ewma(cy, min(1.0, max(0.0, ry + jy)), a)
        cz = ewma(cz, min(1.0, max(0.0, rz + jz)), a)
        Cx.append(cx); Cy.append(cy); Cz.append(cz)
    return Cx, Cy, Cz

def read_signals_csv(path: str, steps: int | None = None) -> tuple[list[float], list[float], list[float]]:
    Cx: list[float] = []; Cy: list[float] = []; Cz: list[float] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r, start=1):
            try:
                Cx.append(float(row["Cx"])); Cy.append(float(row["Cy"])); Cz.append(float(row["Cz"]))
            except Exception:
                continue
            if steps and i >= steps: break
    if not (Cx and Cy and Cz):
        raise ValueError("CSV vuoto o senza colonne Cx,Cy,Cz.")
    if steps and len(Cx) > steps:
        Cx, Cy, Cz = Cx[:steps], Cy[:steps], Cz[:steps]
    return Cx, Cy, Cz

# ----------------------- core analytics -----------------------

def compute_fields(
    Cx: list[float],
    Cy: list[float],
    Cz: list[float],
    alpha: float
) -> tuple[list[dict], dict]:
    tau = -math.log(max(EPS, alpha))
    rows: list[dict] = []
    prev_c = None
    prev_i = None
    mean_c: list[float] = []; mean_d: list[float] = []; mean_s: list[float] = []; mean_h: list[float] = []
    accept = 0
    for t, (cx, cy, cz) in enumerate(zip(Cx, Cy, Cz), start=1):
        n = max(EPS, norm3(cx, cy, cz))
        i3x, i3y, i3z = cx/n, cy/n, cz/n
        if prev_c is None: grad = 0.0
        else:
            dx, dy, dz = cx-prev_c[0], cy-prev_c[1], cz-prev_c[2]
            grad = norm3(dx, dy, dz)
        if prev_i is None: div = 0.0
        else:
            div = abs((i3x-prev_i[0]) + (i3y-prev_i[1]) + (i3z-prev_i[2]))
        prod = cx * cy * cz
        s = -math.log(max(EPS, abs(prod)))
        h3 = stanh(prod / (1.0 + grad))
        cbar = (cx + cy + cz) / 3.0
        if t > 1:
            if div < tau: accept += 1
            mean_c.append(cbar); mean_d.append(div); mean_s.append(s); mean_h.append(h3)
        rows.append({"t": t, "Cx": cx, "Cy": cy, "Cz": cz,
                     "I3x": i3x, "I3y": i3y, "I3z": i3z,
                     "grad": grad, "div": div, "H3": h3, "prod": prod, "surprisal": s})
        prev_c = (cx, cy, cz); prev_i = (i3x, i3y, i3z)
    total = len(Cx); analyzed = max(1, total-1)
    metrics = {
        "steps": total, "alpha": alpha, "tau": tau,
        "mean_coherence": mean(mean_c) if mean_c else 0.0,
        "mean_divergence": mean(mean_d) if mean_d else 0.0,
        "mean_surprisal": mean(mean_s) if mean_s else 0.0,
        "mean_H3": mean(mean_h) if mean_h else 0.0,
        "accept_ratio_divergence": accept / analyzed
    }
    return rows, metrics

# ----------------------------- IO -----------------------------

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def write_tensor_csv(path: str, rows: list[dict]) -> None:
    fields = ["t","Cx","Cy","Cz","I3x","I3y","I3z","grad","div","H3","prod","surprisal"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def write_surface_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["t","H3"]); w.writeheader()
        w.writerows([{"t": r["t"], "H3": r["H3"]} for r in rows])

def write_metrics_json(path: str, metrics: dict, bases: tuple[int,int,int] | None) -> None:
    payload = {
        "module": "Omniabase-3D · Multibase Spatial Mathematics Engine",
        "version": "1.2",
        "doi": "10.5281/zenodo.17270742",
        "license": "MIT",
        "bases": None if bases is None else {"x": bases[0], "y": bases[1], "z": bases[2]},
        "metrics": metrics
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

# ----------------------------- CLI -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Omniabase-3D all-in-one engine")
    p.add_argument("--signals", type=str, help="CSV input con Cx,Cy,Cz")
    p.add_argument("--bases", nargs=3, type=int, metavar=("BX","BY","BZ"), help="basi VdC es. 8 12 16")
    p.add_argument("--steps", type=int, default=1000, help="passi")
    p.add_argument("--alpha", type=float, default=0.005, help="significatività per τ")
    p.add_argument("--smooth", type=float, default=0.15, help="EWMA smoothing (0,1]")
    p.add_argument("--seed", type=int, default=None, help="seed PRNG")
    p.add_argument("--outdir", type=str, default="omniabase-3d/metrics", help="cartella output")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    if args.signals is None and args.bases is None:
        raise SystemExit("Specifica --signals CSV o --bases BX BY BZ.")
    ensure_dir(args.outdir)

    bases_tuple: tuple[int,int,int] | None = None
    try:
        if args.signals:
            Cx, Cy, Cz = read_signals_csv(args.signals, steps=args.steps)
            print(f"Lettura segnali: {args.signals} · passi={len(Cx)}")
        else:
            bx, by, bz = map(int, args.bases); bases_tuple = (bx, by, bz)
            Cx, Cy, Cz = generate_coherence_streams(args.steps, bases_tuple, args.smooth, args.seed)
            print(f"Generazione VdC basi={bases_tuple} · passi={len(Cx)} · EWMA={args.smooth} · seed={args.seed}")
    except Exception as e:
        raise SystemExit(f"Errore in I/O o generazione: {e}")

    rows, metrics = compute_fields(Cx, Cy, Cz, alpha=args.alpha)

    tensor_path  = os.path.join(args.outdir, "tensor_I3.csv")
    surface_path = os.path.join(args.outdir, "surface_H3.csv")
    metrics_path = os.path.join(args.outdir, "metrics.json")

    write_tensor_csv(tensor_path, rows)
    write_surface_csv(surface_path, rows)
    write_metrics_json(metrics_path, metrics, bases_tuple)

    print("\n--- Risultati ---")
    print(f"Output: {args.outdir}")
    if bases_tuple: print(f"Basi VdC: {bases_tuple}")
    print(f"τ(α): {metrics['tau']:.6f}  (α={args.alpha})")
    print(f"mean_C:  {metrics['mean_coherence']:.6f}")
    print(f"mean_div:{metrics['mean_divergence']:.6f}")
    print(f"mean_S*: {metrics['mean_surprisal']:.6f}")
    print(f"mean_H3: {metrics['mean_H3']:.6f}")
    print(f"acc_div: {metrics['accept_ratio_divergence']:.3f}")

if __name__ == "__main__":
    main()
```0