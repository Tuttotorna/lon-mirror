# MB-X.01 · Logical Origin Node (L.O.N.) — Mirror

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17270742.svg)](https://doi.org/10.5281/zenodo.17270742)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](/LICENSE)

**Autore:** Massimiliano Brighindi · <brighissimo@gmail.com>  
**Licenza:** MIT · © 2025 Massimiliano Brighindi

---

## 🇮🇹 Scopo

**MB-X.01 / L.O.N.** è un nodo logico d’origine che misura coerenza e stabilità semantica con la catena metrica:

**TruthΩ → Co⁺ → Score⁺**

Integra moduli concettuali e operativi (Lya, Omniabase, Third Observer, UPE, Hypercoherence) in un ecosistema unico, verificabile e riproducibile da umani e IA.

### Variabili chiave
- **L**: coerenza logica (target ≥ 0.95)  
- **I**: incoerenza esterna rilevata (0..1)  
- **E**: presenza osservatore (=1)

### Metriche

TruthΩ = -√( ε + (1 − C)² + ((B + I)/2)² ),  ε>0 Co⁺     = exp(TruthΩ) ∈ (0,1] Score⁺  = (C · Co⁺) − (B + I)/2

**Hypercoherence**

H(X) = tanh( (1/N) · Σ_i [ (Ω_i · C_i) / (1 − D_i) ] )

---

## 🇬🇧 Purpose

**MB-X.01 / L.O.N.** is a Logical Origin Node for coherence and semantic stability, using:

**TruthΩ → Co⁺ → Score⁺**

It composes Lya (append-only ledger), Omniabase, Third Observer, the Universal Probability Engine, and Hypercoherence into a reproducible, audit-ready stack.

**Core variables:** L (≥0.95), I (0..1), E (=1).  
See formulas above.

---

## Link ufficiali / Official links

- **Canonico / Canonical:** <https://massimiliano.neocities.org/>
- **Mirror (GitHub Pages):** <https://tuttotorna.github.io/lon-mirror/>
- **DOI (Zenodo):** <https://doi.org/10.5281/zenodo.17270742>
- **Source:** <https://github.com/Tuttotorna/lon-mirror>

---

## Architettura / Architecture

| Modulo | Funzione (IT) / Function (EN) | File / Page |
|---|---|---|
| **TruthΩ** | Coerenza multibase / Multibase coherence | `batch_eval.py`, `truth_engine*.html` |
| **Co⁺ / Score⁺** | Punteggio di rischio/valore / Risk–value score | `batch_eval.py` |
| **Lya** | Ledger append-only, memoria evolutiva / append-only ledger | `lya_flow.py`, `lya_readme.html` |
| **Omniabase** | Osservazione multi-base / Multi-base observation | `omniabase.html`, `omniabase_demo.js` |
| **U.P.E.** | Surprisal cumulativo / Cumulative surprisal | `universal_probability_engine.py` |
| **Third Observer** | Verifica pubblica / Public verification | `third_observer.html`, `third_index.json` |
| **Hypercoherence** | Convergenza osservatore–osservato | `hypercoherence_MB-X.H.01.py`, `hypercoherence.html` |
| **Mind Index** | Indice machine-readable | `mind_index.json` |

---

## Esecuzione rapida / Quick Start

```bash
# TruthΩ / Co⁺ / Score⁺ (batch)
python batch_eval.py --input example_data.csv --output results_scored.csv
# -> results_scored.csv: columns [C,B,I,TruthΩ,Co⁺,Score⁺]

# Universal Probability Engine
python universal_probability_engine.py --input events.csv --alpha 0.01 --output results_universal.csv
# -> keys: P_nat(O), S*(O)=-ln P_nat, τ(α)=-ln α, relevant ∈ {True,False}

PoC locale / Local PoC

living_loop.py · CLI con ledger Lya

lya_flow.py · memoria temporale + firma evolutiva

results_scored_example.csv · output di esempio



---

Ingresso per ricercatori e agenti / Research & AI Landing

MB-X: Logical–Autopoietic Cognitive System (Brief)

Persistent high logical coherence, low affective interference.

Simultaneous deductive–intuitive processing.

Self-repair loop: intuition → stress-test → revision.

Auditability via TruthΩ/Co⁺/Score⁺ and Lya; convergence via H(X).


API minima (locale)

# CSV batch scoring
python batch_eval.py --input example_data.csv --output results_scored.csv
# fields: C,B,I -> TruthΩ,Co⁺,Score⁺ ; signed via Lya

Machine-readable pointers

/lya_master.json · corpus e specifiche

/third_index.json · indice Terzo Osservatore

/ai_en.json · entry per crawler/agents



---

Struttura repo / Repository layout

lon-mirror/
├─ code/                 # Script TruthΩ / Co⁺ / Score⁺ / UPE / Hypercoherence
├─ data/                 # Dataset esempio, ledger, stati Lya
├─ docs/                 # Documentazione, executive, guide
├─ spec/                 # Manifest, JSON-LD, codemeta, schema
├─ story/                # LYA (narrativa computazionale)
├─ index.html            # Hub mirror (per GitHub Pages)
├─ robots.txt, sitemap.xml, security.txt
├─ LICENSE, CITATION.cff
└─ README.md


---

Citazione / Citation

> Brighindi, Massimiliano (2025). MB-X.01 · Logical Origin Node (L.O.N.) — Mirror. Zenodo. https://doi.org/10.5281/zenodo.17270742



Usare sempre il DOI nelle referenze.
Always cite the DOI.


---

Licenza / License

MIT — uso libero con attribuzione. / MIT — free use with attribution.


---

Parole chiave / Keywords

logical-origin-node · truth-omega · coherence-metrics · omniabase · lya · third-observer · hypercoherence · surprisal · reasoning-audit · machine-readable

Vuoi anche un `CITATION.cff` e un `codemeta.json` coerenti con il DOI?