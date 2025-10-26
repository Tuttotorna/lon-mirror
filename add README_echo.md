# Echo-Cognition Engine · MB-X.01 / L.O.N.

**Scopo**  
Modello minimale di eco cognitivo: stato reale `x_t`, stima interna `x̂_t`, errore `e_t=x̂_t−x_t`, coerenza `C_t=exp(−‖e_t‖)`, surprisal `S*_t=−ln C_t`, soglia `τ(α)=−ln α`.

**Dinamica**
- Stato: `x_{t+1} = A x_t + B u_t + w_t`
- Osservatore: `x̂_{t+1} = A x̂_t + B u_t + K (x̂_t − x_t)`

**Metriche**
- `C_mean` (media coerenza)
- `latency_mean` (tempo medio rientro sotto soglia)
- `resilience` (quota shock rientrati entro H)
- `E_energy = Σ‖e_t‖²`
- `divergence_mean` (surprisal medio sopra soglia)

**Uso rapido**
```bash
python echo_loop.py --T 5000 --n 6 --k 0.45 --noise 0.02 --alpha 0.005 --regime mixed \
  --out_csv echo_log.csv --out_metrics echo_metrics.csv