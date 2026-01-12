from __future__ import annotations

import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt


def load_sei_values(path: Path) -> List[float]:
    sei_vals = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            v = rec.get("sei")
            if isinstance(v, (int, float)):
                sei_vals.append(float(v))
    return sei_vals


def main():
    # choose one of the reports produced earlier
    report_path = Path("data/sei_gsm8k_uncertainty_report.jsonl")
    if not report_path.exists():
        report_path = Path("data/sei_gsm8k_report.jsonl")

    if not report_path.exists():
        raise FileNotFoundError("No SEI report found in data/")

    sei_vals = load_sei_values(report_path)
    if len(sei_vals) < 3:
        raise ValueError("Not enough SEI points to plot.")

    out_dir = Path("assets/diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sei_trend.png"

    x = list(range(1, len(sei_vals) + 1))

    plt.figure(figsize=(8, 4))
    plt.plot(x, sei_vals, marker="o", linewidth=1.5)
    plt.xlabel("Iteration")
    plt.ylabel("SEI (marginal yield)")
    plt.title("SEI Trend — Marginal Structural Yield")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()