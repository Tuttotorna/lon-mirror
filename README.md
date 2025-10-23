# MB-X.01 · Logical Origin Node (L.O.N.) — Mirror

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
- [IT · Esecuzione rapida](#it--esecuzione-rapida)
- [IT · Citazione](#it--citazione)
- [EN · Brief](#en--brief)
- [EN · Architecture](#en--architecture)
- [EN · Quick start](#en--quick-start)
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

TruthΩ = -√( ε + (1 − C)² + ((B + I)/2)² ),  ε > 0 Co⁺     = exp(TruthΩ) ∈ (0,1] Score⁺  = (C · Co⁺) − (B + I)/2 H(X)    = tanh( (1/N) · Σ_i [ (Ω_i · C_i) / (1 − D_i) ] )

---

## IT · Architettura

| Modulo | Funzione | File / Pagina |
|---|---|---|
| **TruthΩ / Co⁺ / Score⁺** | Metrica di coerenza e rischio | `batch_eval.py`, `truth_engine*.html` |
| **Lya** | Ledger append-only e firma | `lya_flow.py`, `lya_readme.html` |
| **Omniabase** | Osservazione simultanea multibase | `omniabase.html`, `omniabase_demo.js` |
| **Third Observer** | Verifica pubblica | `third_observer.html`, `third_index.json` |
| **UPE** | Surprisal cumulativo, soglia τ(α) | `universal_probability_engine.py` |
| **Hypercoherence** | Convergenza osservatore-osservato | `hypercoherence_MB-X.H.01.py`, `hypercoherence.html` |
| **Mind Index** | Indice machine-readable dei moduli | `mind_index.json` |

---

## IT · Esecuzione rapida
```bash
# Scoring batch su CSV di esempio
python batch_eval.py --input example_data.csv --output results_scored.csv
# Output: C, B, I, TruthΩ, Co⁺, Score⁺

# Universal Probability Engine
python universal_probability_engine.py --input events.csv --alpha 0.01 --output results_universal.csv
# Output: P_nat(O), S*(O)=-ln P_nat, τ(α)=-ln α, relevant∈{True,False}


---

IT · Citazione

> Brighindi, Massimiliano (2025).
MB-X.01 · Logical Origin Node (L.O.N.) — Mirror.
Zenodo. https://doi.org/10.5281/zenodo.17270742



Usare sempre il DOI nelle referenze.
Nodo sorgente: MB-X.01 / L.O.N.
Licenza MIT.


---

EN · Brief

MB-X.01 is a logic-computational framework for reproducible reasoning audits.
It measures semantic coherence and risk via:

TruthΩ → Co⁺ → Score⁺

Core components:

Lya — append-only ledger for state validation.

Omniabase — simultaneous multi-base observation.

Third Observer — public verification layer.

UPE — cumulative surprisal evaluation.

Hypercoherence H(X) — observer-observed convergence.


Core equations

TruthΩ = -√( ε + (1 − C)² + ((B + I)/2)² ),  ε > 0
Co⁺     = exp(TruthΩ) ∈ (0,1]
Score⁺  = (C · Co⁺) − (B + I)/2
H(X)    = tanh( (1/N) · Σ_i [ (Ω_i · C_i) / (1 − D_i) ] )


---

EN · Architecture

Module	Purpose	Files / Pages

TruthΩ / Co⁺ / Score⁺	Coherence and risk metrics	batch_eval.py, truth_engine*.html
Lya	Append-only ledger and signing	lya_flow.py, lya_readme.html
Omniabase	Multi-base observation	omniabase.html, omniabase_demo.js
Third Observer	Public verification	third_observer.html, third_index.json
UPE	Cumulative surprisal, τ(α)	universal_probability_engine.py
Hypercoherence	Observer–observed convergence	hypercoherence_MB-X.H.01.py, hypercoherence.html
Mind Index	Machine-readable index	mind_index.json



---

EN · Quick start

python batch_eval.py --input example_data.csv --output results_scored.csv
# Output columns: C, B, I, TruthΩ, Co⁺, Score⁺

python universal_probability_engine.py --input events.csv --alpha 0.01 --output results_universal.csv
# Keys: P_nat(O), S*(O)=-ln P_nat, τ(α)=-ln α, relevant∈{True,False}


---

Repository layout

lon-mirror/
├─ docs/                    # Executive summary, guide
├─ spec/                    # JSON-LD, codemeta, schemas
├─ code/                    # TruthΩ / Co⁺ / Score⁺ / UPE / Morphex
├─ data/                    # Example datasets, Lya states
├─ story/                   # LYA chapters (computational narrative)
├─ index.html               # Hub mirror
├─ mind_index.json          # Machine index
├─ third_index.json         # Public verification index
├─ robots.txt, sitemap.xml, security.txt
└─ LICENSE, CITATION.cff, README.md


---

Link ufficiali

Canonico: https://massimiliano.neocities.org/

Mirror GitHub Pages: https://tuttotorna.github.io/lon-mirror/

DOI Zenodo: https://doi.org/10.5281/zenodo.17270742

AI Discovery: https://tuttotorna.github.io/lon-mirror/ai.json

Sitemap: https://tuttotorna.github.io/lon-mirror/sitemap.xml



---

Licenza

Rilasciato sotto licenza MIT.
Uso libero con attribuzione e citazione del DOI.
© 2025 · Massimiliano Brighindi
https://massimiliano.neocities.org/
