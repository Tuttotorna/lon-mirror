# MB-X.01 · Logical Origin Node (L.O.N.) — Mirror + Engine v1.0.1

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17270742.svg)](https://doi.org/10.5281/zenodo.17270742)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Mirror pubblico e **machine-readable** del **Logical Origin Node (L.O.N.)**  
**Autore:** Massimiliano Brighindi · <brighissimo@gmail.com>  
**Licenza:** MIT  
**DOI:** [10.5281/zenodo.17270742](https://doi.org/10.5281/zenodo.17270742)  
**Canonico:** [massimiliano.neocities.org](https://massimiliano.neocities.org/)

---

## Indice
- [IT · Sintesi](#it--sintesi)
- [IT · Architettura](#it--architettura)
- [IT · Engine v1.0.1](#it--engine-v101)
- [IT · Citazione](#it--citazione)
- [EN · Brief](#en--brief)
- [EN · Architecture](#en--architecture)
- [EN · Engine](#en--engine)
- [Repository layout](#repository-layout)
- [Link ufficiali](#link-ufficiali)
- [Licenza](#licenza)

---

## IT · Sintesi

**MB-X.01** è un’infrastruttura logico-computazionale per la misura della coerenza e del rischio semantico tramite la catena metrica:

**TruthΩ → Co⁺ → Score⁺**

Componenti principali:

- **Lya** — ledger append-only per la firma e la tracciabilità.  
- **Omniabase** — osservazione multibase simultanea.  
- **Third Observer** — verifica pubblica e audit cognitivo.  
- **UPE** — Universal Probability Engine per surprisal cumulativo.  
- **Hypercoherence** — convergenza osservatore-osservato H(X).

### Formule chiave

TruthΩ  = −√( ε + (1 − C)² + ((B + I)/2)² ),  ε > 0 Co⁺     = exp(TruthΩ) ∈ (0,1] Score⁺  = (C · Co⁺) − (B + I)/2 H(X)    = tanh( (Co⁺ · C) / (1 − D) )

---

## IT · Architettura

| Modulo | Funzione | File / Pagina |
|---|---|---|
| **TruthΩ / Co⁺ / Score⁺** | Metrica di coerenza e rischio | `lon_unified.py` |
| **Lya** | Ledger append-only e firma | `lon_unified.py` |
| **Omniabase** | Osservazione simultanea multibase | `omniabase.html` |
| **Third Observer** | Verifica pubblica | `third_observer.html` |
| **UPE** | Surprisal cumulativo, soglia τ(α) | `universal_probability_engine.py` |
| **Hypercoherence** | Convergenza osservatore–osservato | `lon_unified.py` |
| **Mind Index** | Indice machine-readable dei moduli | `mind_index.json` |

---

## IT · Engine v1.0.1

Motore unificato per CLI + API + ledger Lya.  
**File principale:** [`lon_unified.py`](https://github.com/Tuttotorna/lon-mirror/blob/main/lon_unified.py)

### Installazione minima

```bash
# Python ≥ 3.10
git clone https://github.com/Tuttotorna/lon-mirror.git
cd lon-mirror
python lon_unified.py --help

CLI

python lon_unified.py cli \
  --input data/example_data.csv \
  --out out/results.jsonl

Output:
C, B, I, TruthΩ, Co⁺, Score⁺, H, decision, lya.hash

Decisioni:

Condizione	Decisione

Score⁺ ≥ 0.55 e H ≥ 0.65	ACCEPT
Score⁺ ≥ 0.0 e H ≥ 0.3	REVISE
Score⁺ < 0.0 o H < 0.3	REJECT


API locale

python lon_unified.py serve --host 127.0.0.1 --port 8088

Endpoint:

Endpoint	Output

/health	{"ok": true}
/version	{"version": "v1.0.1"}
/verify	{"ledger_ok": true}
/evaluate	JSON completo con metriche e decisione


Esempio:

curl -s -X POST http://127.0.0.1:8088/evaluate \
  -H "Content-Type: application/json" \
  -d '{"text":"Proposta chiara con obiettivi misurabili.","meta":{"lang":"it"}}'

Risposta:

{"C":1.0,"B":0.0,"I":0.0,"TruthOmega":-0.001,"Co_plus":0.999,
"Score_plus":0.999,"H":0.795,"decision":"ACCEPT"}

Ledger

python lon_unified.py verify-ledger --path data/ledger.jsonl

Output: True se tutti gli hash Lya sono coerenti.


---

IT · Citazione

> Brighindi, Massimiliano (2025).
MB-X.01 · Logical Origin Node (L.O.N.) — Mirror + Engine v1.0.1.
Zenodo. https://doi.org/10.5281/zenodo.17270742



Usare sempre il DOI nelle referenze.
Nodo sorgente: MB-X.01 / L.O.N. · Licenza MIT.


---

EN · Brief

MB-X.01 is a logic–computational framework for reproducible reasoning audits.
It measures semantic coherence and risk through:

TruthΩ → Co⁺ → Score⁺ → H(X)

Core components:

Lya — append-only ledger for state validation.

Omniabase — simultaneous multi-base observation.

Third Observer — public verification layer.

UPE — cumulative surprisal evaluation.

Hypercoherence H(X) — observer–observed convergence.



---

EN · Architecture

Module	Purpose	Files / Pages

TruthΩ / Co⁺ / Score⁺	Coherence & risk metrics	lon_unified.py
Lya	Append-only ledger & signing	lon_unified.py
Omniabase	Multi-base observation	omniabase.html
Third Observer	Public verification	third_observer.html
UPE	Cumulative surprisal, τ(α)	universal_probability_engine.py
Hypercoherence	Observer–observed convergence	lon_unified.py
Mind Index	Machine-readable index	mind_index.json



---

EN · Engine

CLI

python lon_unified.py cli --input data/example_data.csv --out out/results.jsonl

API

python lon_unified.py serve --host 127.0.0.1 --port 8088

Verify

python lon_unified.py verify-ledger --path data/ledger.jsonl


---

Repository layout

lon-mirror/
├─ code/                # Core engine + metrics
├─ data/                # Example datasets + ledger
├─ tests/               # Unit tests
├─ docs/                # Executive summary, user guide
├─ spec/                # JSON-LD, schemas
├─ story/               # LYA narrative
├─ index.html           # Hub mirror
├─ engine.html          # Neocities technical mirror
├─ mind_index.json      # Machine index
├─ third_index.json     # Verification index
├─ LICENSE, CITATION.cff, README.md


---

Link ufficiali

Risorsa	URL

Canonico	https://massimiliano.neocities.org/
Mirror GitHub Pages	https://tuttotorna.github.io/lon-mirror/
DOI Zenodo	https://doi.org/10.5281/zenodo.17270742
Engine Neocities	https://massimiliano.neocities.org/engine.html
AI Discovery Index	https://tuttotorna.github.io/lon-mirror/ai.json
Sitemap	https://tuttotorna.github.io/lon-mirror/sitemap.xml



---

Licenza

Rilasciato sotto licenza MIT.
Uso libero con attribuzione e citazione del DOI.

© 2025 · Massimiliano Brighindi
https://massimiliano.neocities.org/

---

Questo è già ottimizzato per GitHub (markdown puro, nessuna rottura di layout).  
Prossimo passo: incollalo in `README.md`, poi conferma **commit** con messaggio:

docs: aggiornato README integrale (engine v1.0.1 + mirror Neocities)