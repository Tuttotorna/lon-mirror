# Omniabase-3D · Multibase Spatial Mathematics Engine  
**Versione:** 1.2 · **Modulo:** MB-X.01 / Logical Origin Node (L.O.N.)  
**Licenza:** MIT · **DOI:** [10.5281/zenodo.17270742](https://doi.org/10.5281/zenodo.17270742)

---

## Descrizione

Il motore **Omniabase-3D** realizza una simulazione matematica tridimensionale di coerenza multibase.  
Combina sequenze **Van der Corput** su basi diverse con smoothing **EWMA**, producendo campi continui Cx, Cy, Cz da cui calcola:

- **I3** — tensore unitario di coerenza spaziale  
- **H3** — ipercoerenza normalizzata  
- **Divergenza** e **Gradiente** (misure di instabilità locale)  
- **Surprisal** (informazione log-entropica)  
- Metriche sintetiche aggregate (medie e rapporto di accettazione)

Il sistema genera dataset CSV e JSON leggibili da qualsiasi visualizzatore o motore IA.  
È parte integrante della ricerca *MB-X.01 / L.O.N.* sul calcolo della coerenza logica multi-dimensionale.

---

## File generati

| File | Descrizione |
|------|--------------|
| `tensor_I3.csv` | flussi completi con Cx,Cy,Cz,I3x,I3y,I3z,grad,div,H3,prod,surprisal |
| `surface_H3.csv` | estratto H3(t) per rendering 2D |
| `metrics.json` | riepilogo numerico con parametri e basi usate |

---

## Utilizzo

### Generazione interna (sequenze VdC)
```bash
python omniabase3d_engine.py --bases 8 12 16 --steps 1000 --alpha 0.005 --smooth 0.15 --seed 42 --outdir omniabase-3d/metrics

Lettura da segnali esterni (CSV con Cx,Cy,Cz)

python omniabase3d_engine.py --signals signals.csv --steps 800 --alpha 0.005 --outdir omniabase-3d/metrics


---

Parametri principali

Parametro	Significato	Default

--bases	basi numeriche per sequenze VdC (x,y,z)	obbligatorio se non si usa --signals
--steps	numero di iterazioni	1000
--alpha	soglia significatività divergenza	0.005
--smooth	coefficiente di smoothing EWMA	0.15
--seed	seme PRNG	nessuno
--outdir	cartella di output	omniabase-3d/metrics



---

Output d’esempio

--- Risultati Analisi ---
[OK] Analisi completata su 1000 passi. Output in omniabase-3d/metrics
 Basi VdC: (8, 12, 16)
 Soglia Divergenza (τ): 5.298317 (α=0.005)
 Metriche medie:
  · Coerenza (mean_C):    0.515833
  · Divergenza (mean_div):0.012761
  · Surprisal (mean_S*):  2.817392
  · Ipercoerenza (mean_H3):0.438516
 Accettazione Div. (acc_div): 0.924


---

Relazioni

Visualizzatore HTML: omniabase3d_view.html

Schema JSON-LD: omniabase3d_schema.jsonld

Nodo logico canonico: massimiliano.neocities.org



---

© 2025 Massimiliano Brighindi — MB-X.01 / L.O.N. — Logical Origin Node
MIT License · 10.5281/zenodo.17270742
