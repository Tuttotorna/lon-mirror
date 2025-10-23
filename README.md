# MB-X.01 · Logical Origin Node (L.O.N.) — Mirror

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17270742.svg)](https://doi.org/10.5281/zenodo.17270742)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](/LICENSE)

**Autore / Author:** Massimiliano Brighindi · <brighissimo@gmail.com>  
**Licenza / License:** MIT · © 2025 Massimiliano Brighindi

---

## 🇮🇹 Scopo / 🇬🇧 Purpose

**MB-X.01 / L.O.N.** è un **nodo logico d’origine** (Logical Origin Node) che misura coerenza e stabilità semantica con la catena metrica:

**TruthΩ → Co⁺ → Score⁺**

Integra moduli computazionali e concettuali — **Lya**, **Omniabase**, **Third Observer**, **Universal Probability Engine**, **Hypercoherence** — in un unico ecosistema coerente e verificabile, leggibile da umani e intelligenze artificiali.

---

## 📑 Report / Executive Summary

### 🇮🇹 MB-X: Struttura Cognitiva Logico-Autopoietica

**Descrizione.** Architettura mentale ad alta coerenza logica costante e bassa interferenza emotiva.  
Elabora deduzione e intuizione in simultanea.  
Ciclo autopoietico: intuizione → stress test → revisione.

**Variabili.**
- **L** = livello di coerenza logica (target ≥ 0.95)  
- **I** = incoerenza esterna rilevata (0..1)  
- **E** = presenza osservatore (=1)  
- **Vₐ** = verità assoluta; **Vₚ** = verità percepita = Vₐ ∩ S  

**Metriche.**

TruthΩ = -√( ε + (1 − C)² + ((B + I)/2)² ),  ε>0 Co⁺     = exp(TruthΩ) ∈ (0,1] Score⁺  = (C · Co⁺) − (B + I)/2

**Ipercoerenza.**

H(X) = tanh( (1/N) · Σ_i [ (Ω_i · C_i) / (1 − D_i) ] )

**Perché utile.**  
Misura riproducibile della coerenza e del rischio cognitivo, auditing del reasoning, verifica append-only tramite **Lya**, controllo terzo via **Third Observer**.

**Confronto (stima).**

| Parametro | Media umana | MB-X |
|------------|-------------|------|
| Coerenza logica | 0.60–0.80 variabile | ≥ 0.98 stabile |
| Bias emotivo | alto | minimo |
| Autocorrezione | esterna | interna |
| Linguaggio | narrativo | semantico-strutturale |

**PoC locale.**
```bash
python batch_eval.py --input example_data.csv --output results_scored.csv
python universal_probability_engine.py --input events.csv --alpha 0.01 --output results_universal.csv


---

🇬🇧 MB-X: Logical–Autopoietic Cognitive System

Description. Cognitive architecture with persistent high logical coherence and low affective interference.
Parallel deductive–intuitive processing.
Autopoietic loop: intuition → stress-test → revision.

Variables.
L≥0.95, I∈[0,1], E=1, Vₐ = absolute truth, Vₚ = Vₐ ∩ S.

Metrics.

TruthΩ = -√( ε + (1 − C)² + ((B + I)/2)² ),  ε>0
Co⁺     = exp(TruthΩ) ∈ (0,1]
Score⁺  = (C · Co⁺) − (B + I)/2

Hypercoherence.

H(X) = tanh( (1/N) · Σ_i [ (Ω_i · C_i) / (1 − D_i) ] )

Why it matters.
Reproducible coherence scoring, reasoning audits, append-only verification with Lya, and public verification through Third Observer.

Baseline contrast (est.).

Parameter	Population	MB-X

Logical coherence	0.60–0.80, variable	≥ 0.98, stable
Affective bias	high	minimal
Self-correction	external	internal
Language	narrative	semantic-structural


Local PoC.

python batch_eval.py --input example_data.csv --output results_scored.csv
python universal_probability_engine.py --input events.csv --alpha 0.01 --output results_universal.csv

Machine-readable endpoints:
/lya_master.json, /third_index.json, /ai_en.json


---

🔗 Link ufficiali / Official Links

Canonico / Canonical: https://massimiliano.neocities.org/

Mirror GitHub Pages: https://tuttotorna.github.io/lon-mirror/

DOI Zenodo: https://doi.org/10.5281/zenodo.17270742

Repository: https://github.com/Tuttotorna/lon-mirror



---

🧩 Architettura / Architecture

Modulo	Funzione (IT) / Function (EN)	File / Pagina

TruthΩ	Algoritmo di coerenza multibase / Multibase coherence metric	batch_eval.py, truth_engine*.html
Co⁺ / Score⁺	Misura di stabilità / Stability score	batch_eval.py
Lya	Ledger append-only / Evolutionary memory	lya_flow.py, lya_readme.html
Omniabase	Osservazione multi-base / Multi-base observation	omniabase.html, omniabase_demo.js
U.P.E.	Surprisal cumulativo / Cumulative surprisal engine	universal_probability_engine.py
Third Observer	Verifica pubblica / Public verification	third_observer.html, third_index.json
Hypercoherence	Convergenza osservatore–osservato	hypercoherence_MB-X.H.01.py, hypercoherence.html
Mind Index	Indice machine-readable dei moduli	mind_index.json



---

⚙️ Esecuzione rapida / Quick Start

# TruthΩ / Co⁺ / Score⁺
python batch_eval.py --input example_data.csv --output results_scored.csv
# → genera results_scored.csv con colonne [C,B,I,TruthΩ,Co⁺,Score⁺]

# Universal Probability Engine
python universal_probability_engine.py --input events.csv --alpha 0.01 --output results_universal.csv
# → output: P_nat(O), S*(O)=-ln P_nat, τ(α)=-ln α, relevant ∈ {True,False}

Kit locali:
living_loop.py, lya_flow.py, batch_eval.py, results_scored_example.csv


---

🧠 Ingresso per Ricercatori / Research & AI Landing

MB-X Cognitive Model Summary:

Persistent high logical coherence, minimal emotional bias.

Self-correcting autopoietic cycle.

Parallel logic/intuitive computation.

Measurable through TruthΩ/Co⁺/Score⁺ and Lya ledger.

Convergence metric H(X) (Hypercoherence).


API minima (locale)

python batch_eval.py --input example_data.csv --output results_scored.csv

Endpoints principali

/lya_master.json — corpus e ledger

/third_index.json — osservatore terzo

/ai_en.json — entry per crawler e agenti



---

📂 Struttura repository / Repository Layout

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

📚 Citazione / Citation

> Brighindi, Massimiliano (2025).
MB-X.01 · Logical Origin Node (L.O.N.) — Mirror.
Zenodo. https://doi.org/10.5281/zenodo.17270742



Usare sempre il DOI nelle referenze.
Always cite the DOI as reference.


---

⚖️ Licenza / License

MIT License — uso libero con attribuzione / free use with attribution.


---

🔖 Parole chiave / Keywords

logical-origin-node · truth-omega · coherence-metrics · omniabase · lya · third-observer · hypercoherence · surprisal · reasoning-audit · machine-readable