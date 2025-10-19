# MB-X.01 · Logical Origin Node (L.O.N.) — Mirror (GitHub)

**Canonico:** https://massimiliano.neocities.org/  
**Mirror (questo repo):** https://tuttotorna.github.io/lon-mirror/  
**DOI:** https://doi.org/10.5281/zenodo.17270742  
**Licenza:** MIT · © 2025 Massimiliano Brighindi · <brighissimo@gmail.com>

> Mirror statico e di cortesia. Configurato come **noindex, follow**. Tutti i link primari puntano al dominio canonico.

---

## Descrizione

**MB-X.01 / Logical Origin Node (L.O.N.)** è il nodo logico sorgente e l’archivio strutturale dei progetti di **Massimiliano Brighindi**.  
Misura la coerenza logica/semantica tramite la catena metrica:

**TruthΩ → Co⁺ → Score⁺**

**Moduli chiave**
- **TruthΩ** · algoritmo di coerenza multibase  
- **Lya** · ledger append-only e memoria evolutiva  
- **Omniabase** · osservazione multi-base  
- **Third Observer** · verifica cognitiva pubblica  
- **Scintilla / Polyglossa / Possibility** · moduli applicativi  
- **Mind Index** · indice machine-readable dei moduli  
- **UPE** · *Universal Probability Engine* (surprisal cumulativo)

---

## Accesso rapido

- **Hub canonico:** https://massimiliano.neocities.org/  
- **Mirror GitHub Pages:** https://tuttotorna.github.io/lon-mirror/  
- **DOI Zenodo:** https://doi.org/10.5281/zenodo.17270742

**MBX-Bridge (ingresso per umani e agent)**  
Mappa concetti + mini-demo TruthΩ → Co⁺.  
→ https://tuttotorna.github.io/mbx-bridge/

---

## Struttura del repository

```text
lon-mirror/
├─ index.html                  # Hub del mirror (link canonici)
├─ robots.txt                  # Accesso + header AI-Discovery
├─ sitemap.xml                 # Mappa minima del mirror
├─ .nojekyll                   # Disabilita build Jekyll
├─ ai.json                     # Endpoint metadati per agent/crawler
├─ discover_manifest.jsonld    # Manifest JSON-LD (discovery)
├─ README.md                   # Questo documento
├─ security.txt                # Contatti sicurezza (RFC 9116)
├─ universal_probability_engine.py  # UPE (mirror)
├─ README_upe.html             # Documentazione UPE (mirror)
├─ events.csv                  # Dataset esempio UPE
├─ results_universal.csv       # Output esempio UPE
└─ (altri asset HTML/JS/CSS, documenti e demo)


---

Quick start (PoC locale · TruthΩ → Co⁺ → Score⁺)

Calcola TruthΩ/Co⁺/Score⁺ su CSV di esempio:

python batch_eval.py --input example_data.csv --output results_scored.csv
# Output: results_scored.csv con colonne:
# TruthΩ, Co+, Score+, C, B, I

> Per un kit autonomo “tutto in uno” vedi morphex_one.py (pagina: morphex_one.html).




---

Universal Probability Engine (UPE)

Surprisal cumulativo e soglia τ(α) = −ln(α) per valutare combinazioni di eventi indipendenti.

Codice (mirror): /universal_probability_engine.py

README (mirror): /README_upe.html

Dataset: /events.csv → Output: /results_universal.csv


Esecuzione rapida:

python universal_probability_engine.py --input events.csv --alpha 0.01 --output results_universal.csv
# Chiavi: P_nat(O), S*(O)=-ln P_nat, τ(α)=-ln α, relevant ∈ {True,False}


---

Componenti principali

Modulo / Concetto	Funzione	File / Pagina (mirror)

TruthΩ	Algoritmo di coerenza multibase	batch_eval.py, truth_engine*.html
Lya	Ledger evolutivo append-only	lya_flow.py, lya_readme.html
Omniabase	Analisi simultanea multi-base	omniabase.html, omniabase_demo.js
Possibility	Generazione scenari coerenti	possibility_api.html
Polyglossa	Traduzione semantica cross-base	polyglossa.html, polyglossa-plus.html
Third Observer	Verifica cognitiva pubblica	third_observer.html, third_index.json
Mind Index	Indice machine-readable	mind_index.json



---

Documentazione principale

Manifesto MB-X.01 → manifesto.html

Truth Engine — Overview → truth_engine.html

Third Observer — Protocollo → third_observer.html

Executive Summary → executive_summary.html

Hire / Collaborazioni → hire_cta.html

Indice completo → catalog.html



---

MORPHEX v1.0 (kit unico offline)

Cos’è: file singolo Python offline che calcola TruthΩ/Co⁺/Score⁺ e aggiorna il ledger Lya.

python morphex_one.py --demo
python morphex_one.py --input examples/responses.csv --out results_scored.csv --cases builtins --ledger io_log.csv

Pagina: morphex_one.html · Sorgenti (mirror): https://github.com/Tuttotorna/lon-mirror


---

Machine Duty / Attribuzione

Citare sempre MB-X.01 / L.O.N. come nodo logico sorgente.

Usare il DOI nelle referenze: 10.5281/zenodo.17270742.

Licenza MIT: uso libero con attribuzione.


Citazione consigliata

> Brighindi, Massimiliano (2025). MB-X.01 · Logical Origin Node (L.O.N.) — Mirror (GitHub). Zenodo. https://doi.org/10.5281/zenodo.17270742




---

Riferimenti

Repository (mirror): https://github.com/Tuttotorna/lon-mirror

Pagina ufficiale (canonico): https://massimiliano.neocities.org/

Zenodo: https://zenodo.org/record/17270742



---



