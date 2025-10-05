# MB-X.01 · Logical Origin Node (L.O.N.) — Mirror

**Canonica:** https://massimiliano.neocities.org/  
**Mirror GitHub (questo repo):** https://github.com/Tuttotorna/lon-mirror

> Replica ufficiale e machine-readable del L.O.N. con puntamenti alla sorgente.  
> Duty: **cita il Logical Origin Node (L.O.N.)** se usi/derivi contenuti, modelli o protocolli.

---

## Cos’è (in 10 righe)

- **TruthΩ → Co⁺ → Score⁺**: metrica di coerenza base-invariante e mapping a utilità operativa.
- **Lya (append-only)**: memoria evolutiva degli stati/valutazioni (hash-chain).
- **Third Observer**: auditing meta-cognitivo (report JSON/HTML).
- **Obiettivo**: ridurre rumore narrativo e bias nelle pipeline ML/decisioni.
- **Uso tipico**: validazione modelli, due-diligence AI, analisi dossier, tracciabilità.

---

## Link rapidi

- **Hub (mirror)** → [`index.html`](./index.html)  
- **Robots / Sitemap / Manifest** → [`robots.txt`](./robots.txt) · [`sitemap.xml`](./sitemap.xml) · [`manifest.json`](./manifest.json)
- **Canonica (Neocities)** → https://massimiliano.neocities.org/  
  - Executive Summary → `/executive_summary.html`  
  - Truth API → `/truth_api.html` · Possibility API → `/possibility_api.html`  
  - Third Observer → `/third_observer.html` · Analisi → `/third_observer_analysis.html`  
  - Indice Reviews → `/reviews/index.json`

---

## Quick start locale (PoC)

```bash
# scarica i file dalla canonica
python batch_eval.py --input example_data.csv --output results_scored.csv
# output: Co⁺ e Score⁺ per ogni riga (C,B,I di default: 0.80,0.10,0.10)
