#!/usr/bin/env python3
# MB-X.01 · L.O.N. Loop v1.0 — micro loop auto-osservante
# Licenza: MIT

import argparse, hashlib, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(".")
DATA = ROOT / "lon_data"
NDJSON = DATA / "lon_state.ndjson"      # append-only (una riga JSON per stato)
INDEX = DATA / "lon_state.json"         # indice comodo (snapshot ultimi N)
RUNLD = DATA / "lon_run.jsonld"         # metadata JSON-LD dell'ultima osservazione
N_SNAPSHOT = 50

def utcnow():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def compute_truth_metrics(text: str, C: float, B: float, I: float):
    """
    Proxy minimale e deterministica (no RNG):
    - 'co' ~ penalizza la ripetitività: più caratteri unici -> meno incoerenza -> co meno negativo
    - 'co_plus' = exp(co) ricondotta in [0,1] tramite clamp
    - 'score_plus' = (C * co_plus) - (B+I)/2
    NB: è un POC di forma, NON la tua metrica completa.
    """
    if not text:
        text = "(empty)"
    unique_ratio = len(set(text)) / max(1, len(text))
    co = -1.0 * (1.0 - unique_ratio)              # ∈ [-1, 0]
    co_plus = max(0.0, min(1.0, pow(2.718281828, co) - 0.367879441))  # shift exp(-1)=0.3678…
    score_plus = (C * co_plus) - (B + I) / 2.0
    return round(co, 6), round(co_plus, 6), round(score_plus, 6)

def ensure_dirs():
    DATA.mkdir(parents=True, exist_ok=True)
    if not NDJSON.exists():
        NDJSON.touch()

def read_last():
    if not NDJSON.exists():
        return None
    last = None
    with NDJSON.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last = json.loads(line)
    return last

def append_ndjson(obj: dict):
    with NDJSON.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def write_index():
    # snapshot ultimi N_SNAPSHOT stati
    items = []
    with NDJSON.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    snap = items[-N_SNAPSHOT:]
    INDEX.write_text(json.dumps({
        "type": "LONStateIndex",
        "version": "1.0",
        "count": len(items),
        "snapshot_last": len(snap),
        "updated": utcnow(),
        "items": snap
    }, ensure_ascii=False, indent=2), encoding="utf-8")

def write_jsonld(last_obj: dict, canonical: str, doi: str|None):
    graph = {
      "@context": "https://schema.org",
      "@type": "Dataset",
      "@id": f"{canonical.rstrip('/')}/#lon-loop",
      "name": "MB-X.01 · L.O.N. Loop — run",
      "creator": {
        "@type": "Person",
        "name": "Massimiliano Brighindi",
        "email": "brighissimo@gmail.com"
      },
      "license": "https://opensource.org/licenses/MIT",
      "dateModified": last_obj["ts"],
      "isPartOf": canonical,
      "measurementTechnique": "TruthΩ→Co⁺ proxy (PoC); Lya-style append-only",
      "variableMeasured": [
        {"@type":"PropertyValue","name":"TruthΩ_proxy","value":last_obj["metrics"]["co"]},
        {"@type":"PropertyValue","name":"Co_plus","value":last_obj["metrics"]["co_plus"]},
        {"@type":"PropertyValue","name":"Score_plus","value":last_obj["metrics"]["score_plus"]}
      ],
      "distribution": [{
        "@type": "DataDownload",
        "name": "lon_state.ndjson",
        "contentUrl": f"{canonical.rstrip('/')}/lon_data/lon_state.ndjson",
        "encodingFormat": "application/x-ndjson"
      }]
    }
    if doi:
        graph["identifier"] = {"@type":"PropertyValue","name":"DOI","value":doi}
    RUNLD.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")

def main():
    parser = argparse.ArgumentParser(description="MB-X.01 · L.O.N. Loop v1.0")
    parser.add_argument("--note", type=str, default="", help="Osservazione/testo del ciclo")
    parser.add_argument("--C", type=float, default=0.8, help="Confidenza [0,1]")
    parser.add_argument("--B", type=float, default=0.2, help="Bias [0,1]")
    parser.add_argument("--I", type=float, default=0.1, help="Interesse [0,1]")
    parser.add_argument("--canonical", type=str, default="https://massimiliano.neocities.org/", help="URL canonica del L.O.N.")
    parser.add_argument("--doi", type=str, default="", help="DOI Zenodo (opzionale)")
    args = parser.parse_args()

    ensure_dirs()
    prev = read_last()
    seq = 1 if prev is None else int(prev["seq"]) + 1
    ts = utcnow()

    co, co_plus, score_plus = compute_truth_metrics(args.note, args.C, args.B, args.I)

    body_for_hash = json.dumps({
        "seq": seq, "ts": ts, "note": args.note, "C": args.C, "B": args.B, "I": args.I,
        "co": co, "co_plus": co_plus, "score_plus": score_plus,
        "prev_hash": (prev["hash"] if prev else "GENESIS")
    }, sort_keys=True, ensure_ascii=False)

    state = {
        "type": "LONState",
        "version": "1.0",
        "seq": seq,
        "ts": ts,
        "note": args.note,
        "metrics": {"co": co, "co_plus": co_plus, "score_plus": score_plus},
        "params": {"C": args.C, "B": args.B, "I": args.I},
        "prev": prev["hash"] if prev else None,
        "hash": sha256(body_for_hash)
    }

    append_ndjson(state)
    write_index()
    write_jsonld(state, args.canonical, args.doi or None)

    # Output umano
    print(f"[LON] seq={seq} ts={ts}")
    print(f"  note: {args.note}")
    print(f"  metrics: Co={co}  Co⁺={co_plus}  Score⁺={score_plus}")
    print(f"  hash: {state['hash']}")
    if prev: print(f"  prev: {prev['hash']}")
    print(f"  files: {NDJSON}  {INDEX}  {RUNLD}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)