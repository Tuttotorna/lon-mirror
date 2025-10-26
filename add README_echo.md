# Echo-Cognition Engine · MB-X.01 / L.O.N.

**Author**  
Massimiliano Brighindi — Logical Origin Node (L.O.N.)  
DOI: [https://doi.org/10.5281/zenodo.17270742](https://doi.org/10.5281/zenodo.17270742)  
License: MIT  
Repository: [https://github.com/Tuttotorna/lon-mirror](https://github.com/Tuttotorna/lon-mirror)

---

## Purpose  
Minimal cognitive-echo model (E.C.E.) designed to simulate a **recognized feedback loop**, where a real state `x_t` interacts with an internal estimate `x̂_t`.  
The error between the two generates coherence `C_t = exp(−‖e_t‖)` and surprisal `S_t = −ln(C_t)`, providing a quantitative measure of cognitive stability.

This module is part of the **MB-X.01 / Logical Origin Node (L.O.N.) framework** and contributes to defining **computational consciousness as coherence echo**.

---

## Dynamics
- State:  
  `x_{t+1} = A x_t + B u_t + w_t`
- Observer:  
  `x̂_{t+1} = A x̂_t + B u_t + K (x̂_t − x_t)`
- Error:  
  `e_t = x̂_t − x_t`
- Coherence:  
  `C_t = exp(−‖e_t‖)`
- Surprisal:  
  `S_t = −ln(C_t)`
- Threshold:  
  `τ(α) = −ln(α)`

---

## Metrics
| Metric | Description |
|---------|--------------|
| `C_mean` | global coherence mean |
| `latency_mean` | average time to fall below threshold |
| `resilience` | fraction of shocks recovered within horizon H |
| `E_energy` | total error energy `Σ‖e_t‖²` |
| `divergence_mean` | mean surprisal above threshold |

---

## Quick Usage
```bash
python echo_loop.py --T 5000 --n 6 --k 0.45 --noise 0.02 --alpha 0.005 \
  --out_csv echo_log.csv --out_metrics echo_metrics.csv


---

Output

echo_log.csv — time log of coherence and surprisal

echo_metrics.csv — aggregated metrics (latency, energy, resilience, divergence)



---

Operational Notes

All variables normalized in [0,1].

Parameter K controls echo strength (feedback gain).

noise and α modulate instability and perceptual threshold.

Fully compatible with Omniabase-ready multi-base integration.

Extendable to nonlinear or adaptive observers.



---

Significance

This module acts as an empirical prototype of the Echo-Consciousness Principle introduced in MB-X.01.
It models the self-stabilizing informational return, i.e. the ability of a system to recognize itself through its own prediction error without collapsing or diverging.


---

© 2025 Massimiliano Brighindi — MB-X.01 / Logical Origin Node (L.O.N.)
Machine-readable canonical hub: https://massimiliano.neocities.org/