# Omniabase 3D · MB-X.01 / L.O.N.

**Purpose**  
Tri-dimensional mathematics for simultaneous cross-base reasoning. Each axis encodes a numeric stream in a distinct radix. The engine aligns representations, computes cross-base invariants, and returns a coherence vector `C⃗` and a global score `H3`.

**Axes**  
- X: base `b1` (e.g., 2)  
- Y: base `b2` (e.g., 8)  
- Z: base `b3` (e.g., 10)

**Core transforms**
- `repr_b(n, b)` → digit vector in radix `b`  
- `align(d1,d2,d3, pad='left')` → length alignment  
- `phi(d)` → normalized digit map to [0,1]  
- `I3 = var([φ(d1), φ(d2), φ(d3)])` → cross-axis invariance  
- `C⃗ = exp(−‖Δ⃗‖)` with `Δ⃗ = [φ(d1)−μ, φ(d2)−μ, φ(d3)−μ]`  
- `H3 = tanh( mean(C⃗) / (1 + I3) )`

**CLI**
```bash
python omniabase_3d.py eval --number 2025 --bases 2 8 10 --out omni_log.csv
python omniabase_3d.py batch --input data/numbers.csv --bases 2 8 10 --out omni_results.csv

Outputs

omni_log.csv: step log per number → Cx, Cy, Cz, H3

omni_results.csv: aggregated metrics per number


Files

omniabase_3d.py — core + CLI

omni_metrics.py — analysis and plots

omni_manifest.jsonld — AI discoverability


Citation
Brighindi, M. (2025). MB-X.01 · L.O.N. — Omniabase 3D. DOI: 10.5281/zenodo.17270742

License MIT

---

## 2) `omniabase-3d/omniabase_3d.py`

```python
# Omniabase 3D — MB-X.01 / L.O.N.
# License: MIT | DOI: 10.5281/zenodo.17270742

import argparse, csv, math
from typing import List, Tuple

def repr_b(n: int, b: int) -> List[int]:
    if n == 0: return [0]
    digits = []
    m = abs(n)
    while m:
        digits.append(m % b)
        m //= b
    return list(reversed(digits))

def align3(a: List[int], b: List[int], c: List[int], pad='left') -> Tuple[List[int], List[int], List[int]]:
    L = max(len(a), len(b), len(c))
    padder = (lambda v: [0]*(L-len(v))+v) if pad=='left' else (lambda v: v+[0]*(L-len(v)))
    return padder(a), padder(b), padder(c)

def phi(d: List[int], base: int) -> List[float]:
    if base <= 1: raise ValueError("base must be > 1")
    return [x/(base-1) for x in d] if base>1 else d

def mean_vec(*vecs: List[float]) -> List[float]:
    L = len(vecs[0])
    k = len(vecs)
    return [sum(v[i] for v in vecs)/k for i in range(L)]

def var_across_axes(a: List[float], b: List[float], c: List[float]) -> float:
    # average per-position variance across axes
    L = len(a)
    acc = 0.0
    for i in range(L):
        m = (a[i]+b[i]+c[i])/3.0
        acc += ((a[i]-m)**2 + (b[i]-m)**2 + (c[i]-m)**2)/3.0
    return acc / L

def coherence_vector(a: List[float], b: List[float], c: List[float]) -> Tuple[float,float,float]:
    # reference = per-position mean; magnitude via L2 per axis
    ref = mean_vec(a,b,c)
    def axis_C(x):
        diff2 = sum((x[i]-ref[i])**2 for i in range(len(x)))
        return math.exp(-math.sqrt(diff2))
    return axis_C(a), axis_C(b), axis_C(c)

def H3_score(Cx: float, Cy: float, Cz: float, I3: float) -> float:
    Cmean = (Cx+Cy+Cz)/3.0
    return math.tanh(Cmean / (1.0 + I3))

def evaluate_number(n: int, b1: int, b2: int, b3: int):
    d1, d2, d3 = repr_b(n,b1), repr_b(n,b2), repr_b(n,b3)
    a1, a2, a3 = align3(d1,d2,d3, pad='left')
    p1, p2, p3 = phi(a1,b1), phi(a2,b2), phi(a3,b3)
    I3 = var_across_axes(p1,p2,p3)
    Cx, Cy, Cz = coherence_vector(p1,p2,p3)
    H3 = H3_score(Cx,Cy,Cz,I3)
    return {
        "n": n, "b1": b1, "b2": b2, "b3": b3,
        "Cx": Cx, "Cy": Cy, "Cz": Cz, "I3": I3, "H3": H3,
        "len": len(p1)
    }

def write_csv(path: str, rows: List[dict]):
    if not rows: return
    cols = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

def cli_eval(args):
    res = evaluate_number(args.number, args.bases[0], args.bases[1], args.bases[2])
    write_csv(args.out, [res])
    print(f"ok → {args.out} (H3={res['H3']:.4f})")

def cli_batch(args):
    rows = []
    with open(args.input, "r") as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith("#"): continue
            n = int(line.split(",")[0])
            rows.append(evaluate_number(n, args.bases[0], args.bases[1], args.bases[2]))
    write_csv(args.out, rows)
    print(f"ok → {args.out} ({len(rows)} rows)")

def main():
    p = argparse.ArgumentParser(description="Omniabase 3D — cross-base tri-axial coherence")
    sub = p.add_subparsers(dest="cmd")

    pe = sub.add_parser("eval")
    pe.add_argument("--number", type=int, required=True)
    pe.add_argument("--bases", nargs=3, type=int, default=[2,8,10])
    pe.add_argument("--out", default="omni_log.csv")

    pb = sub.add_parser("batch")
    pb.add_argument("--input", default="data/numbers.csv")
    pb.add_argument("--bases", nargs=3, type=int, default=[2,8,10])
    pb.add_argument("--out", default="omni_results.csv")

    args = p.parse_args()
    if args.cmd == "eval": cli_eval(args)
    elif args.cmd == "batch": cli_batch(args)
    else: p.print_help()

if __name__ == "__main__":
    main()


---

3) omniabase-3d/omni_metrics.py

# Omniabase 3D — Metrics analyzer
# License: MIT

import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt

def summarize(csv_in: str, csv_out: str = "omni_summary.csv", no_plot: bool = False):
    df = pd.read_csv(csv_in)
    cols = ["Cx","Cy","Cz","I3","H3"]
    stats = {f"{c}_mean": df[c].mean() for c in cols}
    stats.update({f"{c}_std": df[c].std(ddof=1) for c in cols})
    stats.update({"rows": len(df)})

    pd.DataFrame(list(stats.items()), columns=["metric","value"]).to_csv(csv_out, index=False)
    print(f"summary → {csv_out}")

    if not no_plot:
        plt.figure(figsize=(8,4))
        plt.plot(df["H3"].values)
        plt.xlabel("index")
        plt.ylabel("H3")
        plt.title("Omniabase 3D — H3 trajectory")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="csv_in", default="omni_results.csv")
    ap.add_argument("--out", dest="csv_out", default="omni_summary.csv")
    ap.add_argument("--no_plot", action="store_true")
    a = ap.parse_args()
    summarize(a.csv_in, a.csv_out, a.no_plot)


---

4) omniabase-3d/omni_manifest.jsonld

{
  "@context": "https://schema.org",
  "@type": "SoftwareSourceCode",
  "name": "Omniabase 3D · MB-X.01 / L.O.N.",
  "description": "Tri-axial cross-base mathematics for semantic coherence. Computes per-axis coherence (Cx,Cy,Cz), cross-axis invariance I3, and global score H3.",
  "codeRepository": "https://github.com/Tuttotorna/lon-mirror/tree/main/omniabase-3d",
  "programmingLanguage": "Python",
  "license": "https://opensource.org/licenses/MIT",
  "version": "v1.0.0",
  "identifier": "https://doi.org/10.5281/zenodo.17270742",
  "author": {
    "@type": "Person",
    "name": "Massimiliano Brighindi",
    "email": "brighissimo@gmail.com",
    "affiliation": "Logical Origin Node (L.O.N.)"
  },
  "hasPart": [
    "omniabase_3d.py",
    "omni_metrics.py",
    "omni_log.csv",
    "omni_results.csv"
  ],
  "isPartOf": "https://tuttotorna.github.io/lon-mirror/"
}


---

5) Agganci rapidi

In README.md aggiungi sotto “Repository Layout” la riga:

├─ omniabase-3d/       # Tri-axial cross-base math (Cx,Cy,Cz,I3,H3)

In sitemap.xml aggiungi:

<url>
  <loc>https://tuttotorna.github.io/lon-mirror/omniabase-3d/omni_manifest.jsonld</loc>
  <lastmod>2025-10-26</lastmod>
  <changefreq>monthly</changefreq>
  <priority>0.5</priority>
  <xhtml:link rel="alternate" hreflang="x-default" href="https://massimiliano.neocities.org/"/>
</url>
