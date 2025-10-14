#!/usr/bin/env python3
# ================================================================
# io_lab_v2_psi_o4.py — MB-X.01 / L.O.N.
# Io Reattivo + Ψ (salienza procedurale) + O₄ (osservatore-quadro)
#
# Compatibilità:
# - Scrive gli stessi file e lo stesso schema-base del v1:
#   * io_log.ndjson  (canali: T, R, O3)  [+ campi aggiuntivi safe]
#   * lya_flow.ndjson (kind: T/R/O3/CHK) [+ Ψ/O4 come record dedicati]
# - I vecchi checker (io_lab_check.py, io_universal_check.py) funzionano senza modifiche.
#
# Modalità input:
#   --input events.ndjson  (batch: NDJSON, una riga = evento)
#   --stdin                (stream NDJSON da pipe)
#   --auto                 (stimoli sintetici, fallback autonomo)
#
# Evento (schema minimo per --stdin/--input):
#   {"type":"USER|AGENT|STREAM","content":"testo o JSON","meta":{"source":"...", "id":"..."}}
#
# Opzioni principali:
#   --cycles N      (limite cicli; con input esterno si ferma a fine stream; con --auto genera N cicli)
#   --delay-ms MS   (ritmo minimo tra cicli)
#   --seed S        (seme RNG; default: 42)
#   --reset         (cancella i file di log prima di iniziare)
#   --pack-run      (genera pacchetto /runs/YYYY-MM-DD/ con sha256/report/manifest)
#   --run-date YYYY-MM-DD   --run-title "Titolo"
#
# MIT · Canonico/Mirror pointer inclusi nei metadata di ciascun record.
# ================================================================
CANONICAL = "https://massimiliano.neocities.org/io_lab_v2_psi_o4.py"
MIRROR    = "https://tuttotorna.github.io/lon-mirror/io_lab_v2_psi_o4.py"
VER_TAG   = "v2_psi_o4"

import sys, os, json, time, hashlib, random, argparse, datetime
from typing import Optional, Dict, Any, Iterable

# ---- Path configurabili via ENV (fallback ai default locali)
LOG_PATH = os.environ.get("IO_LOG_PATH", "io_log.ndjson")
LYA_PATH = os.environ.get("LYA_FLOW_PATH", "lya_flow.ndjson")

# ------------------------- Utilities base -------------------------
def now_iso() -> str:
    return datetime.datetime.utcnow().isoformat(timespec="milliseconds")+"Z"

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _hash(prev_hash: Optional[str], payload_dict: Dict[str, Any]) -> str:
    blob = (prev_hash or "") + json.dumps(payload_dict, sort_keys=True, ensure_ascii=False)
    return sha256_hex(blob)

def _append(path: str, entry: Dict[str,Any], prev_hash: Optional[str]) -> str:
    entry["ts"]   = now_iso()
    entry["prev"] = prev_hash
    entry["meta"] = {"canonical": CANONICAL, "mirror": MIRROR, "ver": VER_TAG}
    # Calcolo hash su TUTTO tranne "hash"
    payload = {k:entry[k] for k in entry if k!="hash"}
    entry["hash"] = _hash(prev_hash, payload)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False)+"\n")
    return entry["hash"]

def append_io(ch: str, data: Dict[str,Any], prev_hash: Optional[str]) -> str:
    return _append(LOG_PATH, {"ch": ch, "data": data}, prev_hash)

def append_lya(kind: str, data: Dict[str,Any], prev_hash: Optional[str]) -> str:
    return _append(LYA_PATH, {"kind": kind, "data": data}, prev_hash)

# ------------------------- Stato & Parametri -------------------------
INTENTS = ["ricerca","ordine","efficienza","verità"]

class IoState:
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.tick = 0
        self.intent_idx = 0
        # bias procedurali per stimolo
        self.bias = {"USER": 0.15, "AGENT": 0.05, "STREAM": 0.00}
        # homeostasi semplice (0..1)
        self.homeostasis = 0.75
        self.last_coherence = None

    def next_intent(self) -> str:
        intent = INTENTS[self.intent_idx % len(INTENTS)]
        self.intent_idx += 1
        return intent

# ------------------------- Trasformazioni core -------------------------
def T_from_event(st: IoState, event: Dict[str,Any]) -> Dict[str,Any]:
    etype = str(event.get("type","STREAM")).upper()
    content = event.get("content","")
    intent = st.next_intent()

    # energia base + bias + piccola dipendenza da lunghezza contenuto
    base = 0.70
    bias = st.bias.get(etype, 0.0)
    cont = 0.05 * min(1.0, len(str(content)) / 280.0)
    energy = max(0.0, min(1.0, round(base + bias + cont + (0.02*random.random()-0.01), 3)))

    # rumore ridotto per USER, medio per AGENT, più alto per STREAM
    noise_base = {"USER": 0.02, "AGENT": 0.03, "STREAM": 0.04}.get(etype, 0.04)
    noise = round(noise_base + 0.02*random.random(), 4)

    return {
        "tick": st.tick,
        "intent": intent,
        "energy": energy,
        "noise": noise,
        "stimulus": {"type": etype, "preview": str(content)[:160], "len": len(str(content))}
    }

def R_from_T(T: Dict[str,Any]) -> Dict[str,Any]:
    score = max(0.0, min(1.0, round(T["energy"] - T["noise"], 3)))
    return {"tick": T["tick"], "coherence_est": score, "intent_seen": T["intent"]}

def O3_from_TR(T: Dict[str,Any], R: Dict[str,Any]) -> Dict[str,Any]:
    coh = float(R["coherence_est"])
    ent = 1.0 if coh < 0.30 else (0.5 if coh < 0.70 else 0.2)
    cyc = 27 + (T["tick"] % 5)
    return {
        "tick": T["tick"],
        "cycle_ms": cyc,
        "entropy_bin": ent,
        "intent_match": int(T["intent"] == R["intent_seen"]),
    }

# Ψ: salienza procedurale (0..1) — funzione di coerenza, bias stimolo e novità contenuto
def PSI_from_event_R(event: Dict[str,Any], R: Dict[str,Any], bias_map: Dict[str,float]) -> Dict[str,Any]:
    etype = event.get("type","STREAM").upper()
    coh   = float(R["coherence_est"])
    bias  = bias_map.get(etype, 0.0)
    novelty = min(1.0, len(str(event.get("content",""))) / 280.0)
    # combinazione semplice
    psi = max(0.0, min(1.0, round(0.60*coh + 0.30*bias + 0.20*novelty, 3)))
    return {"tick": R["tick"], "psi": psi, "etype": etype, "novelty": round(novelty,3)}

# O₄: osservatore-quadro (gating contestuale + homeostasi)
def O4_from_TR_psi(st: IoState, T: Dict[str,Any], R: Dict[str,Any], PSI: Dict[str,Any]) -> Dict[str,Any]:
    coh = float(R["coherence_est"])
    psi = float(PSI["psi"])
    # rischio: alta salienza ma bassa coerenza
    risk = 1 if (psi >= 0.6 and coh < 0.4) else (0 if coh >= 0.6 else 0)
    # aggiorna homeostasi: si avvicina a coerenza
    st.homeostasis = max(0.0, min(1.0, round(0.85*st.homeostasis + 0.15*coh, 3)))
    return {
        "tick": T["tick"],
        "risk_flag": risk,
        "homeostasis": st.homeostasis,
        "gate": "throttle" if risk==1 else ("open" if coh>=0.6 else "caution")
    }

# ------------------------- Sorgenti eventi -------------------------
def events_from_stdin() -> Iterable[Dict[str,Any]]:
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        try:
            yield json.loads(line)
        except Exception:
            yield {"type":"STREAM","content":line}

def events_from_file(path: str) -> Iterable[Dict[str,Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                yield json.loads(line)
            except Exception:
                yield {"type":"STREAM","content":line}

def events_auto(cycles: int) -> Iterable[Dict[str,Any]]:
    # sequenza sintetica: USER, AGENT, STREAM ripetuta
    kinds = ["USER","AGENT","STREAM"]
    texts = {
        "USER":   "Nota vocale dell’utente: obiettivo chiaro, chiedi piano operativo.",
        "AGENT":  "Ping agente esterno: sincronizza stato.",
        "STREAM": "Telemetria: segnale moderato, nessuna anomalia."
    }
    for k in range(cycles):
        t = kinds[k % 3]
        yield {"type": t, "content": texts[t], "meta": {"auto": True, "k": k}}

# ------------------------- Pacchetto RUN -------------------------
def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def write_run_package(run_date: str, title: str) -> None:
    # cartella
    run_dir = os.path.join("runs", run_date)
    os.makedirs(run_dir, exist_ok=True)

    # sha256.txt
    sha_lines = []
    for p in [LOG_PATH, LYA_PATH]:
        if os.path.exists(p):
            sha_lines.append(f"{_sha256_file(p)}  {os.path.basename(p)}")
    with open(os.path.join(run_dir, "sha256.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(sha_lines)+"\n")

    # report.txt minimale
    summary = {
        "io_log_exists": os.path.exists(LOG_PATH),
        "lya_flow_exists": os.path.exists(LYA_PATH),
        "io_log_lines": sum(1 for _ in open(LOG_PATH, "r", encoding="utf-8")) if os.path.exists(LOG_PATH) else 0,
        "lya_flow_lines": sum(1 for _ in open(LYA_PATH, "r", encoding="utf-8")) if os.path.exists(LYA_PATH) else 0,
    }
    with open(os.path.join(run_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")

    # run_manifest.json
    manifest = {
        "run_date": run_date,
        "title": title,
        "created_utc": now_iso(),
        "artifacts": {
            "io_log": os.path.basename(LOG_PATH) if os.path.exists(LOG_PATH) else None,
            "lya_flow": os.path.basename(LYA_PATH) if os.path.exists(LYA_PATH) else None,
            "sha256": "sha256.txt",
            "report": "report.txt"
        },
        "meta": {"canonical": CANONICAL, "mirror": MIRROR, "ver": VER_TAG}
    }
    with open(os.path.join(run_dir, "run_manifest.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")

    # discover_manifest.jsonld (minimo)
    discover = {
        "@context":"https://schema.org",
        "@type":"Dataset",
        "name": f"MB-X.01 / L.O.N. — RUN {run_date}",
        "dateCreated": run_date,
        "creator":{"@type":"Person","name":"Massimiliano Brighindi"},
        "distribution":[
            {"@type":"DataDownload","name":"io_log.ndjson","contentUrl":os.path.basename(LOG_PATH)},
            {"@type":"DataDownload","name":"lya_flow.ndjson","contentUrl":os.path.basename(LYA_PATH)},
            {"@type":"DataDownload","name":"sha256.txt","contentUrl":"sha256.txt"},
            {"@type":"DataDownload","name":"report.txt","contentUrl":"report.txt"},
            {"@type":"DataDownload","name":"run_manifest.json","contentUrl":"run_manifest.json"}
        ]
    }
    with open(os.path.join(run_dir, "discover_manifest.jsonld"), "w", encoding="utf-8") as f:
        f.write(json.dumps(discover, ensure_ascii=False, indent=2)+"\n")

    # CHANGELOG.md
    with open(os.path.join(run_dir, "CHANGELOG.md"), "w", encoding="utf-8") as f:
        f.write(f"# RUN {run_date} — {title}\n\n- Pacchetto generato da {VER_TAG}\n")

# ------------------------- Motore principale -------------------------
def run_engine(mode: str, st: IoState, cycles: int, delay_ms: int, source: Optional[str]) -> None:
    # reset file se richiesto
    prev_io = None
    prev_lya = None

    if mode == "stdin":
        stream = events_from_stdin()
    elif mode == "file":
        stream = events_from_file(source)
    else:
        stream = events_auto(cycles)

    for k, event in enumerate(stream):
        if mode=="auto" and k >= cycles:
            break

        # T, R, O3
        T  = T_from_event(st, event)
        prev_io  = append_io("T",  T,  prev_io)
        prev_lya = append_lya("T",  T,  prev_lya)

        R  = R_from_T(T)
        prev_io  = append_io("R",  R,  prev_io)
        prev_lya = append_lya("R",  R,  prev_lya)

        O3 = O3_from_TR(T, R)
        prev_io  = append_io("O3", O3, prev_io)
        prev_lya = append_lya("O3", O3, prev_lya)

        # Ψ e O₄ (come record dedicati su lya_flow; su io_log solo dentro CHK->extras)
        PSI = PSI_from_event_R(event, R, st.bias)
        prev_lya = append_lya("PSI", PSI, prev_lya)

        O4  = O4_from_TR_psi(st, T, R, PSI)
        prev_lya = append_lya("O4",  O4,  prev_lya)

        # CHK di ciclo: sigilla puntando al tip di io_log
        chk = {
            "tick": st.tick,
            "log_hash_tip": prev_io,
            "extras": {"psi": PSI["psi"], "gate": O4["gate"], "risk": O4["risk_flag"]}
        }
        prev_lya = append_lya("CHK", chk, prev_lya)

        st.tick += 1
        # ritmo minimo tra cicli
        if delay_ms > 0:
            time.sleep(delay_ms/1000.0)

# ------------------------- CLI -------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="io_lab_v2 — Io Reattivo + Ψ + O₄")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--auto", action="store_true", help="stimoli sintetici")
    g.add_argument("--stdin", action="store_true", help="leggi eventi NDJSON da stdin")
    g.add_argument("--input", type=str, help="file NDJSON di eventi")

    ap.add_argument("--cycles", type=int, default=100, help="cicli (con --auto) / limite massimo")
    ap.add_argument("--delay-ms", type=int, default=30, help="ritmo minimo tra cicli")
    ap.add_argument("--seed", type=int, default=42, help="seme RNG")
    ap.add_argument("--reset", action="store_true", help="cancella i file di log prima di iniziare")

    ap.add_argument("--pack-run", action="store_true", help="genera pacchetto RUN in /runs/YYYY-MM-DD/")
    ap.add_argument("--run-date", type=str, default=None, help="YYYY-MM-DD (default: oggi UTC)")
    ap.add_argument("--run-title", type=str, default="MB-X.01 · v2_psi_o4 run", help="titolo breve per il pacchetto")

    return ap.parse_args()

def main():
    args = parse_args()

    if args.reset:
        for p in (LOG_PATH, LYA_PATH):
            if os.path.exists(p):
                os.remove(p)

    st = IoState(seed=args.seed)

    if args.auto:
        mode, source = "auto", None
    elif args.stdin:
        mode, source = "stdin", None
    else:
        mode, source = "file", args.input

    run_engine(mode=mode, st=st, cycles=args.cycles, delay_ms=args.delay_ms, source=source)

    if args.pack_run:
        run_date = args.run_date
        if not run_date:
            run_date = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        write_run_package(run_date, args.run_title)

if __name__ == "__main__":
    main()

# ================================================================
# Note operative:
# - Esempi:
#   python io_lab_v2_psi_o4.py --auto --cycles 50 --reset --pack-run --run-title "demo autonoma"
#   python io_lab_v2_psi_o4.py --input events.ndjson --pack-run --run-date 2025-10-11
#   tail -f /var/log/app.log | python io_lab_v2_psi_o4.py --stdin
#
# - Record extra:
#   * lya_flow: aggiunge "PSI" e "O4" (oltre a T/R/O3/CHK). I checker legacy li ignorano.
#   * io_log: resta focalizzato su T/R/O3 sequenziali; CHK in lya_flow punta al tip di io_log.
#
# - Compatibilità:
#   * Hash-chain invariata: prev + JSON(sort_keys=True) → sha256
#   * Validator web & script esistenti funzionano.
#
# - Sicurezza mentale (O₄):
#   * "risk_flag" segnala alta salienza ma bassa coerenza → gate "throttle".
#   * homeostasis converge gradualmente alla coerenza osservata.
# ================================================================
```0