#!/usr/bin/env python3
# lon_unified.py — MB-X.01 / L.O.N. single-file engine
from __future__ import annotations
import os, re, csv, json, math, time, argparse, hashlib
from typing import Any, Dict, List, Tuple, Optional
from http.server import BaseHTTPRequestHandler, HTTPServer

# --------- CONST ---------
VERSION = "v1.0.1"
DEFAULT_LEDGER = os.environ.get("LYA_LEDGER", "data/ledger.jsonl")
DEFAULT_PROFILES_PATH = os.environ.get("DIST_PROFILES", "data/distortion_profiles.json")
DEFAULT_PROFILES = {"cli":0.05,"api":0.08,"user":0.10,"doc":0.06,"web":0.15}
TAU_S_DEFAULT = 0.55
TAU_H_DEFAULT = 0.65

# --------- UTIL ---------
def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def round_record(rec: Dict[str, Any], decimals: int = 6) -> Dict[str, Any]:
    for k in ("C","B","I","TruthOmega","Co_plus","Score_plus","H"):
        if k in rec and isinstance(rec[k], (int,float)):
            rec[k] = round(float(rec[k]), decimals)
    return rec

# --------- INGEST ---------
def make_sample(text: str, meta: Optional[Dict[str,Any]]=None, *, source="user", conf=0.85, sid=None) -> Dict[str,Any]:
    return {
        "id": sid or f"evt-{int(time.time()*1000)}",
        "timestamp": now_iso(),
        "source": source,
        "confidence_source": float(conf),
        "text": text,
        "meta": meta or {}
    }

def load_distortion_profiles(path: str) -> Dict[str, float]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        profs = cfg.get("profiles", cfg)
        out = {k: float(v) for k,v in profs.items()}
        return out if out else DEFAULT_PROFILES
    except FileNotFoundError:
        return DEFAULT_PROFILES

# --------- PARSER (very lightweight PoC) ---------
def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

def extract_actors(text: str) -> List[str]:
    return sorted(set(re.findall(r"\b[A-Z][a-zA-Z]+\b", text)))

def parse_semantics(text: str, meta: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any], List[str], List[str]]:
    P = split_sentences(text)
    K = {"lang": meta.get("lang","it"), "topic": meta.get("topic",[]), "author": meta.get("author"), "subject": meta.get("subject")}
    A = extract_actors(text)
    G = meta.get("goals", [])
    return P, K, A, G

# --------- METRICS ---------
def _redundancy(P: List[str]) -> float:
    tokens = " ".join(P).lower().split()
    if not tokens: return 0.0
    uniq = len(set(tokens))
    return max(0.0, 1.0 - uniq/len(tokens))

def _signal_coherence(P: List[str], K: Dict[str, Any]) -> float:
    if not P: return 0.0
    lengths = [len(p) for p in P]
    mean_len = (sum(lengths)/len(lengths)) if lengths else 1.0
    var = (max(lengths) - min(lengths)) / (1.0 + mean_len)
    redundancy = _redundancy(P)
    c = max(0.0, min(1.0, 1.0 - 0.5*var - 0.3*redundancy))
    return float(c)

def _bias_penalty(text: str, P: List[str], K: Dict[str, Any], A: List[str]) -> float:
    hedges = len(re.findall(r"\b(maybe|perhaps|forse|probabilmente)\b", text.lower()))
    exclaims = text.count("!")
    author_ref = 1 if K.get("author")==K.get("subject")==True else 0
    b = max(0.0, min(1.0, 0.1*hedges + 0.05*exclaims + 0.2*author_ref))
    return float(b)

def _inconsistency_penalty(P: List[str], K: Dict[str, Any]) -> float:
    neg = sum(1 for p in P if re.search(r"\bnon\b", p.lower()))
    pos = len(P) - neg
    diff = abs(pos - neg) / (1 + len(P))
    contradictions = 0.2 if pos and neg and diff < 0.3 else 0.0
    return float(min(1.0, contradictions))

def compute_metrics(text: str, parsed: Tuple[List[str], Dict[str, Any], List[str], List[str]], eps: float=1e-6) -> Dict[str,float]:
    P, K, A, G = parsed
    C = _signal_coherence(P, K)
    B = _bias_penalty(text, P, K, A)
    I = _inconsistency_penalty(P, K)
    Omega = -math.sqrt(eps + (1 - C)**2 + ((B + I)/2)**2)
    Co = math.exp(Omega)                # Co⁺ ∈ (0,1]
    Score = (C * Co) - (B + I) / 2.0    # Score⁺
    return {"C":C, "B":B, "I":I, "TruthOmega":Omega, "Co_plus":Co, "Score_plus":Score}

# --------- HYPERCOHERENCE (fixed: uses Co⁺) ---------
def hypercoherence(omega: float, c: float, distortion: float=0.0) -> float:
    d = max(-0.999, min(0.999, float(distortion)))
    co = math.exp(float(omega))  # Co⁺
    val = (co * float(c)) / (1.0 - d)
    return math.tanh(val)

def distortion_from_profiles(source: str, profiles: Dict[str,float]) -> float:
    return float(profiles.get(source, profiles.get("default", 0.0)))

# --------- DECISION ---------
def decide(score_plus: float, H: float, tau_s: float=TAU_S_DEFAULT, tau_h: float=TAU_H_DEFAULT) -> str:
    if score_plus >= tau_s and H >= tau_h:
        return "ACCEPT"
    if score_plus < 0.0 or H < 0.3:
        return "REJECT"
    return "REVISE"

# --------- LYA LEDGER ---------
def _hash_record(prev_hash: str, payload: Dict[str, Any]) -> str:
    m = hashlib.sha256()
    m.update(prev_hash.encode("utf-8"))
    m.update(json.dumps(payload, sort_keys=True, separators=(",",":")).encode("utf-8"))
    return m.hexdigest()

def lya_append(record: Dict[str, Any], path: str = DEFAULT_LEDGER) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    prev = "GENESIS"
    last_line = None
    try:
        with open(path, "rb") as f:
            for line in f: last_line = line
        if last_line:
            last = json.loads(last_line.decode("utf-8"))
            prev = last.get("lya", {}).get("hash", "GENESIS")
    except FileNotFoundError:
        pass
    payload = {"ts": now_iso(), "record": record}
    h = _hash_record(prev, payload)
    out = {"prev_hash": prev, "hash": h}
    with open(path, "ab") as f:
        f.write(json.dumps({**payload, "lya": out}, ensure_ascii=False).encode("utf-8") + b"\n")
    return out

def lya_verify(path: str = DEFAULT_LEDGER) -> bool:
    prev = "GENESIS"
    try:
        with open(path, "rb") as f:
            for raw in f:
                obj = json.loads(raw.decode("utf-8"))
                expected = _hash_record(prev, {"ts": obj["ts"], "record": obj["record"]})
                if expected != obj["lya"]["hash"]:
                    return False
                prev = expected
        return True
    except FileNotFoundError:
        return True

# --------- API ---------
class Handler(BaseHTTPRequestHandler):
    profiles = load_distortion_profiles(DEFAULT_PROFILES_PATH)
    tau_s = TAU_S_DEFAULT
    tau_h = TAU_H_DEFAULT
    ledger_path = DEFAULT_LEDGER

    def _send(self, code: int, obj: Any):
        self.send_response(code)
        self.send_header("Content-Type","application/json")
        self.end_headers()
        self.wfile.write(json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"ok": True})
        if self.path == "/version":
            return self._send(200, {"version": VERSION})
        if self.path == "/verify":
            return self._send(200, {"ledger_ok": lya_verify(Handler.ledger_path)})
        return self._send(404, {"error":"not found"})

    def do_POST(self):
        if self.path != "/evaluate":
            return self._send(404, {"error":"not found"})
        try:
            length = int(self.headers.get("Content-Length","0"))
            req = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            return self._send(400, {"error":"invalid json"})

        text = req.get("text","")
        meta = req.get("meta",{}) or {}
        source = req.get("source","api")
        conf = float(req.get("confidence_source", 0.85))

        sample = make_sample(text, meta, source=source, conf=conf)
        P,K,A,G = parse_semantics(sample["text"], sample["meta"])
        met = compute_metrics(sample["text"], (P,K,A,G))
        D = distortion_from_profiles(sample["source"], Handler.profiles)
        H = hypercoherence(met["TruthOmega"], met["C"], D)
        decision = decide(met["Score_plus"], H, Handler.tau_s, Handler.tau_h)

        rec = {
            "id": sample["id"],
            "timestamp": sample["timestamp"],
            "source": sample["source"],
            "version": VERSION,
            "config": {"tau_s": Handler.tau_s, "tau_h": Handler.tau_h, "distortion": D},
            **met,
            "H": H,
            "decision": decision,
            "meta": sample["meta"]
        }
        rec = round_record(rec)
        rec["lya"] = lya_append(rec, Handler.ledger_path)
        return self._send(200, rec)

def run_api(host: str="127.0.0.1", port: int=8088, profiles_path: str=DEFAULT_PROFILES_PATH, ledger_path: str=DEFAULT_LEDGER):
    Handler.profiles = load_distortion_profiles(profiles_path)
    Handler.ledger_path = ledger_path
    server = HTTPServer((host, port), Handler)
    print(f"API listening on http://{host}:{port}")
    server.serve_forever()

# --------- CLI ---------
def run_cli(input_csv: str, out_jsonl: str, profiles_path: str, tau_s: float, tau_h: float, ledger_path: str):
    profiles = load_distortion_profiles(profiles_path)
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as out, open(input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("text","")
            meta = {"lang": row.get("lang","it")}
            src = row.get("source","cli")
            sample = make_sample(text, meta, source=src)
            P,K,A,G = parse_semantics(sample["text"], sample["meta"])
            met = compute_metrics(sample["text"], (P,K,A,G))
            # optional overrides from CSV
            for k in ("C","B","I"):
                if row.get(k):
                    met[k] = float(row[k])
                    # recompute derived if any override given
            # recompute if overrides touched C/B/I
            Omega = -math.sqrt(1e-6 + (1 - met["C"])**2 + ((met["B"] + met["I"])/2)**2)
            met["TruthOmega"] = Omega
            met["Co_plus"] = math.exp(Omega)
            met["Score_plus"] = (met["C"]*met["Co_plus"]) - (met["B"]+met["I"])/2.0

            D = distortion_from_profiles(sample["source"], profiles)
            H = hypercoherence(met["TruthOmega"], met["C"], D)
            decision = decide(met["Score_plus"], H, tau_s, tau_h)

            rec = {
                "id": sample["id"],
                "timestamp": sample["timestamp"],
                "source": sample["source"],
                "version": VERSION,
                "config": {"tau_s": tau_s, "tau_h": tau_h, "distortion": D},
                **met,
                "H": H,
                "decision": decision,
                "meta": sample["meta"]
            }
            rec = round_record(rec)
            rec["lya"] = lya_append(rec, ledger_path)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

# --------- MAIN ---------
def main():
    ap = argparse.ArgumentParser(description="MB-X.01 / L.O.N. unified engine")
    sub = ap.add_subparsers(dest="cmd")

    api_p = sub.add_parser("serve", help="Run HTTP API")
    api_p.add_argument("--host", default="127.0.0.1")
    api_p.add_argument("--port", type=int, default=8088)
    api_p.add_argument("--profiles", default=DEFAULT_PROFILES_PATH)
    api_p.add_argument("--ledger", default=DEFAULT_LEDGER)

    cli_p = sub.add_parser("cli", help="Run batch CLI on CSV")
    cli_p.add_argument("--input", required=True, help="CSV with columns: text[,C,B,I,source,lang]")
    cli_p.add_argument("--out", required=True, help="Output JSONL")
    cli_p.add_argument("--profiles", default=DEFAULT_PROFILES_PATH)
    cli_p.add_argument("--tau_s", type=float, default=TAU_S_DEFAULT)
    cli_p.add_argument("--tau_h", type=float, default=TAU_H_DEFAULT)
    cli_p.add_argument("--ledger", default=DEFAULT_LEDGER)

    ver_p = sub.add_parser("verify-ledger", help="Verify Lya ledger integrity")
    ver_p.add_argument("--ledger", default=DEFAULT_LEDGER)

    args = ap.parse_args()

    if args.cmd == "serve":
        run_api(args.host, args.port, args.profiles, args.ledger)
    elif args.cmd == "cli":
        run_cli(args.input, args.out, args.profiles, args.tau_s, args.tau_h, args.ledger)
    elif args.cmd == "verify-ledger":
        print(lya_verify(args.ledger))
    else:
        ap.print_help()

if __name__ == "__main__":
    main()

# CLI batch
python lon_unified.py cli --input data/example_data.csv --out out/results.jsonl

# Avvio API
python lon_unified.py serve --host 127.0.0.1 --port 8088

# Healthcheck
curl -s http://127.0.0.1:8088/health

# Valutazione
curl -s -X POST http://127.0.0.1:8088/evaluate \
  -H "Content-Type: application/json" \
  -d '{"text":"Proposta chiara con obiettivi misurabili.","meta":{"lang":"it"}}'

# Verifica ledger
python lon_unified.py verify-ledger
```0
