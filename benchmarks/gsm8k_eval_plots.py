import o
import numpy as np
import matplotlib.pyplot as plt

from gsm8k_eval_full import evaluate, summarise


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def make_plots():
    records = evaluate()
    summary = summarise(records)

    out_dir = "benchmarks"
    ensure_dir(out_dir)

    # --- Scatter PBII instability vs correctness ---
    instabilities = [r["pbii_instability"] for r in records]
    correctness = [1 if r["correct"] else 0 for r in records]

    plt.figure(figsize=(8, 4))
    plt.scatter(instabilities, correctness, alpha=0.7, s=30)
    plt.yticks([0, 1], ["incorrect", "correct"])
    plt.xlabel("PBII chain instability")
    plt.ylabel("Answer correctness")
    plt.title("GSM8K – PBII instability vs correctness")
    plt.grid(True, alpha=0.3)
    scatter_path = os.path.join(out_dir, "gsm8k_pbii_scatter.png")
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=200)
    plt.close()

    # --- Histogram: correct vs incorrect PBII distribution ---
    corr_inst = [r["pbii_instability"] for r in records if r["correct"]]
    inc_inst = [r["pbii_instability"] for r in records if not r["correct"]]

    plt.figure(figsize=(8, 4))
    bins = 20
    plt.hist(corr_inst, bins=bins, alpha=0.6, label="correct")
    plt.hist(inc_inst, bins=bins, alpha=0.6, label="incorrect")
    plt.xlabel("PBII chain instability")
    plt.ylabel("Count")
    plt.title("GSM8K – PBII distribution (correct vs incorrect)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    hist_path = os.path.join(out_dir, "gsm8k_pbii_hist.png")
    plt.tight_layout()
    plt.savefig(hist_path, dpi=200)
    plt.close()

    print("Summary:", summary)
    print("Saved plots to:")
    print(" -", scatter_path)
    print(" -", hist_path)


if __name__ == "__main__":
    make_plots()