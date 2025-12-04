# pbii_analysis_v0.3.py
# MBX–Omniabase · PBII analysis engine (v0.3)
# Requires: pandas, numpy, matplotlib, scikit-learn
#
# Usage:
#   1) Run pbii_compute_v0.3.py to generate PBII_data_10M.csv (or smaller MAX_N).
#   2) Run this script:
#        python pbii_analysis_v0.3.py
#   3) It will:
#        - load the CSV
#        - compute stats for primes vs composites
#        - compute ROC + AUC
#        - save basic plots as PNG

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

CSV_FILE = "PBII_data_10M.csv"   # change if you used another name
MAX_N_ANALYSIS = 1_000_000       # you can lower this if the file is huge

def load_data(csv_path: str, max_n: int = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if max_n is not None:
        df = df[df["n"] <= max_n]
    return df

def basic_stats(df: pd.DataFrame):
    primes = df[df["is_prime"] == 1]
    comps = df[df["is_prime"] == 0]

    print("=== BASIC STATS ===")
    print(f"Total n: {len(df)}")
    print(f"Primes: {len(primes)}")
    print(f"Composites: {len(comps)}")
    print()

    print("PBII mean:")
    print(f"  E[PBII | prime]      = {primes['PBII'].mean():.6f}")
    print(f"  E[PBII | composite]  = {comps['PBII'].mean():.6f}")
    print()

    print("PBII std:")
    print(f"  std[PBII | prime]    = {primes['PBII'].std():.6f}")
    print(f"  std[PBII | composite]= {comps['PBII'].std():.6f}")
    print()

def plot_histograms(df: pd.DataFrame, out_prefix: str = "pbii_hist"):
    primes = df[df["is_prime"] == 1]["PBII"]
    comps = df[df["is_prime"] == 0]["PBII"]

    # Limit extreme tail to make hist readable
    p_max = np.quantile(primes, 0.99)
    c_max = np.quantile(comps, 0.99)
    xmax = max(p_max, c_max)

    plt.figure()
    plt.hist(comps[comps <= xmax], bins=100, alpha=0.5, density=True, label="composites")
    plt.hist(primes[primes <= xmax], bins=100, alpha=0.5, density=True, label="primes")
    plt.xlabel("PBII")
    plt.ylabel("Density")
    plt.title("PBII distribution: primes vs composites")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}.png", dpi=200)
    plt.close()

def compute_roc_auc(df: pd.DataFrame, out_prefix: str = "pbii_roc"):
    y_true = df["is_prime"].values
    scores = df["PBII"].values

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    print("=== ROC / AUC ===")
    print(f"AUC(PBII vs primality) = {roc_auc:.6f}")
    print()

    # Save ROC plot
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f"PBII (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], lw=1, linestyle="--", label="random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve: PBII as feature for primality")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}.png", dpi=200)
    plt.close()

def main():
    print("Loading data...")
    df = load_data(CSV_FILE, max_n=MAX_N_ANALYSIS)

    basic_stats(df)
    plot_histograms(df)
    compute_roc_auc(df)

    print("Done. Generated:")
    print("  - pbii_hist.png")
    print("  - pbii_roc.png")

if __name__ == "__main__":
    main()