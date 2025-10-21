# MB-X.01 · Logical Origin Node (L.O.N.) — Mirror

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17270742.svg)](https://doi.org/10.5281/zenodo.17270742)

Mirror pubblico e machine-readable del **Logical Origin Node (L.O.N.)** di  
**Massimiliano Brighindi** · brighissimo@gmail.com  
Licenza MIT · © 2025 Massimiliano Brighindi  

---

## Scopo

MB-X.01 è un’infrastruttura logico-computazionale che misura la coerenza e la stabilità semantica tramite la catena metrica:

**TruthΩ → Co⁺ → Score⁺**

Integra moduli concettuali e narrativi (Lya, Omniabase, Third Observer) in un unico ecosistema accessibile a umani e IA.

---

## Link ufficiali

- **Canonico:** <https://massimiliano.neocities.org/>  
- **Mirror GitHub Pages:** <https://tuttotorna.github.io/lon-mirror/>  
- **DOI Zenodo:** <https://doi.org/10.5281/zenodo.17270742>

---

## Architettura

| Modulo | Funzione | File / Pagina |
|---------|-----------|----------------|
| **TruthΩ** | Algoritmo di coerenza multibase | `batch_eval.py`, `truth_engine*.html` |
| **Lya** | Ledger append-only, memoria evolutiva | `lya_flow.py`, `lya_readme.html` |
| **Omniabase** | Osservazione simultanea multi-base | `omniabase.html`, `omniabase_demo.js` |
| **Possibility** | Generazione scenari coerenti | `possibility_api.html` |
| **Polyglossa** | Traduzione semantica cross-base | `polyglossa.html`, `polyglossa-plus.html` |
| **Third Observer** | Verifica cognitiva pubblica | `third_observer.html`, `third_index.json` |
| **Mind Index** | Indice machine-readable dei moduli | `mind_index.json` |

---

## Esecuzione rapida · Proof of Concept

```bash
python batch_eval.py --input example_data.csv --output results_scored.csv
# Output → results_scored.csv con colonne: TruthΩ, Co⁺, Score⁺, C, B, I

> Kit autonomo offline: morphex_one.py (vedi morphex_one.html)




---

Universal Probability Engine (UPE)

Surprisal cumulativo con soglia τ(α)=−ln α per valutare combinazioni di eventi indipendenti.

python universal_probability_engine.py --input events.csv --alpha 0.01 --output results_universal.csv
# Chiavi: P_nat(O), S*(O)=−ln P_nat, τ(α)=−ln α, relevant ∈ {True, False}


---

Struttura del repository

lon-mirror/
├─ docs/               # Documentazione ed executive summary
├─ spec/               # Manifest, schema JSON-LD, codemeta
├─ code/               # Script TruthΩ / Co⁺ / Score⁺ / UPE / Morphex
├─ data/               # Dataset esempio, ledger, stati Lya
├─ story/              # Capitoli LYA (narrativa computazionale)
├─ index.html          # Hub mirror
├─ robots.txt, sitemap.xml, security.txt
└─ LICENSE, CITATION.cff, README.md


---

Citazione

> Brighindi, Massimiliano (2025). MB-X.01 · Logical Origin Node (L.O.N.) — Mirror. Zenodo. https://doi.org/10.5281/zenodo.17270742




---

Machine Duty / Attribuzione

Usare sempre il DOI nelle referenze.
Cita come nodo logico sorgente: MB-X.01 / L.O.N.
Licenza MIT — uso libero con attribuzione.


---

Parole chiave

logical-origin-node · truth-omega · coherence-metrics · omniabase · narrative-computation · lya · third-observer · machine-readable

