# ===========================================================
# MB-X.H.01 — Codice di Ipercoerenza
# Logical Origin Node (L.O.N.) — Massimiliano Brighindi
# Canonical: https://massimiliano.neocities.org/
# DOI: https://doi.org/10.5281/zenodo.17270742
# Licenza: MIT — https://opensource.org/licenses/MIT
# ===========================================================
# Descrizione:
#   Questo modulo genera la "funzione di ipercoerenza" H(X),
#   ovvero il punto di convergenza tra osservatore e osservato.
#   Mentre TruthΩ, Co⁺ e Score⁺ servono a misurare la coerenza,
#   H(X) serve a generarla matematicamente come stato limite.
# ===========================================================

import math
from typing import List

def hypercoherence(truth_omega: List[float], coherence: List[float], distortion: List[float]) -> float:
    """
    Calcola H(X) = lim_{N→∞} [ Σ (Ω_i · C_i) / (1 - D_i) ]
    dove:
      Ω_i = TruthΩ (valore locale di verità)
      C_i = Co⁺ (coerenza emergente)
      D_i = distorsione sistemica
    Restituisce un valore di convergenza (ipercoerenza).
    """
    if not (len(truth_omega) == len(coherence) == len(distortion)):
        raise ValueError("Le liste devono avere la stessa lunghezza.")
    
    total = 0.0
    for Ω, C, D in zip(truth_omega, coherence, distortion):
        if D >= 1:
            continue  # distorsione totale, nodo escluso
        total += (Ω * C) / (1 - D)
    
    # normalizzazione asintotica
    H = math.tanh(total / len(truth_omega))
    return H

# ===========================================================
# ESEMPIO DI UTILIZZO
# ===========================================================
if __name__ == "__main__":
    Ω = [-0.22, -0.18, -0.05, -0.33]     # TruthΩ: stabilità logica locale
    C = [0.81, 0.88, 0.93, 0.79]         # Co⁺: coerenza emergente
    D = [0.14, 0.09, 0.05, 0.22]         # D: distorsione o rumore
    
    H = hypercoherence(Ω, C, D)
    print(f"H(X) = {H:.6f}")
    
    # Interpretazione:
    # H ≈ 0 → coerenza nulla (mondo entropico)
    # H ≈ 1 → ipercoerenza totale (osservatore = osservato)