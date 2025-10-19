# ===============================================================
# MB-X.01 / L.O.N. — Universal Probability Engine
# Versione: 1.0 · Licenza: MIT
# Autore: Massimiliano Brighindi · brighissimo@gmail.com
# Canonico: https://massimiliano.neocities.org/
# DOI: https://doi.org/10.5281/zenodo.17270742
# ===============================================================
"""
Scopo:
Valutare quanto un insieme di eventi osservati (O) sia improbabile
in assenza di coordinamento, usando solo probabilità naturali conservative.

Formula base:
    P_nat(O)  = Π_i p_i
    S*(O)     = -ln(P_nat(O))
Decisione:
    flag = S*(O) >= τ(α) = -ln(α)

Dove:
  p_i   = probabilità naturale (bound superiore, stima conservativa)
  α     = livello di significatività (default 0.01 → 1 su 100)
  S*    = surprisal cumulativo (quanto è "strano" l'insieme)
"""

import math, csv, sys, argparse
from pathlib import Path

def surprisal(p): 
    if p <= 0: return float('inf')
    return -math.log(p)

def tau(alpha): 
    return -math.log(alpha)

def analyze(events, alpha=0.01):
    """Calcola P_nat, S*, flag per l'intera sequenza."""
    p_joint = 1.0
    notes = []
    for eid, p in events:
        p = max(1e-15, min(1.0, float(p)))  # sicurezza
        p_joint *= p
        notes.append(f"{eid}:{p:.3g}")
    S = surprisal(p_joint)
    return {
        "alpha": alpha,
        "tau": tau(alpha),
        "P_nat": p_joint,
        "S_star": S,
        "relevant": S >= tau(alpha),
        "log": "; ".join(notes)
    }

def main():
    ap = argparse.ArgumentParser(description="Universal Probability Engine — MB-X.01 / L.O.N.")
    ap.add_argument("--input", help="File CSV con colonne: id,p", default="events.csv")
    ap.add_argument("--alpha", type=float, default=0.01, help="Livello di significatività (default=0.01)")
    ap.add_argument("--output", help="File di output (default=results_universal.csv)", default="results_universal.csv")
    args = ap.parse_args()

    events = []
    with open(args.input, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                events.append((row["id"].strip(), float(row["p"])))
            except Exception:
                continue

    result = analyze(events, alpha=args.alpha)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["alpha","tau","P_nat","S_star","relevant","log"])
        w.writerow([result["alpha"], f"{result['tau']:.4f}", f"{result['P_nat']:.6g}",
                    f"{result['S_star']:.4f}", result["relevant"], result["log"]])

    print(f"\n--- Universal Probability Engine ---")
    print(f"Eventi analizzati: {len(events)}")
    print(f"P_nat(O): {result['P_nat']:.6g}")
    print(f"S*(O): {result['S_star']:.4f} nats")
    print(f"Soglia τ(α={args.alpha}): {result['tau']:.4f}")
    print(f"→ Rilevante: {result['relevant']}")
    print(f"Log: {result['log']}\n")

if __name__ == "__main__":
    main()