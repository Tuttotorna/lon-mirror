# L.O.N. Loop v1.0 — micro-loop auto-osservante

**Scopo**: prova minima *eseguibile* del L.O.N.: ogni ciclo registra stato (`lon_state.ndjson`),
calcola una metrica proxy (TruthΩ→Co⁺, *solo per dimostrazione*), firma con hash
append-only e pubblica metadati JSON-LD.

## Esecuzione

```bash
python lon_loop.py --note "prima osservazione" --C 0.8 --B 0.2 --I 0.1 \
  --canonical "https://massimiliano.neocities.org/" \
  --doi "10.5281/zenodo.17270742"