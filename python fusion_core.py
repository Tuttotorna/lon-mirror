#!/usr/bin/env python3
# fusion_core.py — MB-X.01 · L.O.N. — Human+Synthetic Fusion (definitivo)
# Uso rapido:
#   python fusion_core.py --text "Idea A" --text "Idea B"
#   python fusion_core.py --csv input.csv --text-col text
#   (opzionale una tantum) python fusion_core.py --finalize   # crea .fusion_lock e congela soglie
#
# Output: fusion_report.json (+ stdout riassunto)
#
# Principi:
# 1) Sufficiency-First: si ferma a “sufficiente coerente”, non ottimizza oltre.
# 2) Non-narrativo: misura segnali, non racconti.
# 3) Fusione: priorità alla coerenza interna (tua) + forma misurabile (mia).
# 4) Lock: .fusion_lock congela soglie/scelte per evitare cicli.

import argparse, csv, json, math, os, re, sys, datetime
from statistics import variance

LOCK_FILE = ".fusion_lock"  # se presente, congela soglie e parametri

# ------------------- PARAMETRI (congelabili) -------------------
DEFAULTS = {
    "HEDGE_BONUS_MAX": 0.05,      # premia incertezza sana
    "ASSERT_PENALTY_START": 0.02, # assertività oltre soglia penalizza
    "PAIR_WEIGHT": 4.0,           # contraddizioni dure
    "ASSERT_WEIGHT": 2.0,         # rigidità
    "REPET_WEIGHT": 1.5,          # monotonia lessicale
    "NEG_WEIGHT": 1.0,            # negazioni ricorrenti
    "INTEREST_WEIGHT": 3.0,       # bias d'interesse (B)
    "RIGIDITY_WEIGHT": 3.0,       # rigidità (I)
    "SUFFICIENCY_SCORE": 0.25,    # soglia “sufficiente coerente”
    "DELTA_MIN": 0.0              # niente “migliorie incrementaliste”
}

def load_params():
    if os.path.exists(LOCK_FILE):
        # Congela: non leggere variabili esterne, usa DEFAULTS inalterati.
        return DEFAULTS.copy()
    # Senza lock: permetti override via env minimi (ma restano stabili a run)
    p = DEFAULTS.copy()
    for k in list(p.keys()):
        v = os.environ.get(f"FUSION_{k}")
        if v is not None:
            try:
                p[k] = float(v) if "." in v or "e" in v.lower() else int(v)
            except:
                pass
    return p

P = load_params()

# ------------------- LESSICO DI SEGNALE -------------------
NEG = {"non","mai","no","nessuno","niente","nulla","senza"}
ASSERT = {"sempre","tutti","ovvio","perfetto","assoluto","indiscutibile","inevitabile"}
HEDGE = {"forse","circa","probabile","pare","sembra","ipotizzo","possibile"}
INTEREST = {"vendita","sconto","denaro","profitto","sponsor","click","engagement"}
CONFLICT_PAIRS = [("sempre","mai"),("vero","falso"),("si","no"),("ordine","caos"),("bene","male")]

# ------------------- UTILS -------------------
def norm_tokens(s):
    return re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", s.lower())

def features(text):
    toks = norm_tokens(text)
    T = len(toks) or 1
    neg = sum(t in NEG for t in toks)/T
    assertive = sum(t in ASSERT for t in toks)/T
    hedge = sum(t in HEDGE for t in toks)/T
    interest = sum(t in INTEREST for t in toks)/T
    pairs = sum(1 for a,b in CONFLICT_PAIRS if a in toks and b in toks)
    repet = 0.0
    if T>3:
        try:
            repet = 1/(1+variance([len(t) for t in toks]))
        except:
            repet = 0.0
    return dict(neg=neg,assertive=assertive,hedge=hedge,interest=interest,pairs=pairs,repet=repet,len=T)

# ------------------- FUSION METRICS -------------------
def truth_omega(feat):
    # Base-invariante proxy (non narrativa)
    k = 1.0
    raw = (
        P["PAIR_WEIGHT"]*feat["pairs"] +
        P["ASSERT_WEIGHT"]*max(0.0, feat["assertive"]-P["ASSERT_PENALTY_START"]) +
        P["REPET_WEIGHT"]*feat["repet"] +
        P["NEG_WEIGHT"]*feat["neg"]
    )
    bonus = 1.2*min(P["HEDGE_BONUS_MAX"], feat["hedge"])
    val = raw - bonus
    return -math.log(1 + (val/k))

def gate_scores(feat, co_plus):
    # C chiarezza; B interesse; I rigidità
    C = max(0.0, 1.0 - 2.5*feat["repet"] - 1.5*feat["pairs"])
    B = min(1.0, P["INTEREST_WEIGHT"]*feat["interest"])
    I = min(1.0, P["RIGIDITY_WEIGHT"]*max(0.0, feat["assertive"]-P["ASSERT_PENALTY_START"]))
    score = (C*co_plus) - (B+I)/2.0
    return C,B,I,score

def evaluate(text, idx):
    f = features(text)
    omega = truth_omega(f)
    co_plus = math.exp(omega)  # (0,1]
    C,B,I,score = gate_scores(f, co_plus)
    status = "SUFFICIENTE" if score >= P["SUFFICIENCY_SCORE"] else "FRAGILE"
    return {
        "id": idx,
        "text": text,
        "len": f["len"],
        "TruthΩ": round(omega,6),
        "Co_plus": round(co_plus,6),
        "C": round(C,4),
        "B": round(B,4),
        "I": round(I,4),
        "Score_plus": round(score,6),
        "status": status
    }

# ------------------- SUFFICIENCY & STOP -------------------
def should_stop(records):
    # Regola: se almeno il 60% è “SUFFICIENTE” e nessun elemento è “critico” (Score⁺ < 0),
    # fermarsi: sufficienza coerente raggiunta. Niente “ottimizzare”.
    if not records: return False
    ok = sum(1 for r in records if r["status"]=="SUFFICIENTE")
    crit = any(r["Score_plus"] < 0 for r in records)
    return (ok/len(records) >= 0.6) and (not crit)

# ------------------- IO -------------------
def load_csv(path, col):
    rows=[]
    with open(path, encoding="utf-8") as f:
        r=csv.DictReader(f)
        for i,row in enumerate(r, start=1):
            if col not in row: 
                raise SystemExit(f"Colonna '{col}' non trovata in {path}")
            rows.append((f"csv_{i}", row[col]))
    return rows

def dump_json(obj, path="fusion_report.json"):
    with open(path,"w",encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path

# ------------------- MAIN -------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--text", action="append", default=[], help="aggiungi testo da valutare (ripetibile)")
    ap.add_argument("--csv", help="file CSV in input")
    ap.add_argument("--text-col", default="text", help="colonna testo nel CSV")
    ap.add_argument("--finalize", action="store_true", help="crea .fusion_lock e congela parametri")
    args=ap.parse_args()

    if args.finalize:
        with open(LOCK_FILE,"w") as f: f.write("LOCKED "+datetime.datetime.utcnow().isoformat()+"Z")
        print("Lock creato: parametri congelati. (nessuna ottimizzazione futura)")
        return

    items=[]
    for i,t in enumerate(args.text, start=1):
        items.append((f"t_{i}", t))
    if args.csv:
        items += load_csv(args.csv, args.text_col)
    if not items:
        print("Nessun input. Usa --text o --csv.")
        sys.exit(1)

    recs=[evaluate(txt, idx) for idx,txt in items]
    stop = should_stop(recs)

    report = {
        "timestamp": datetime.datetime.utcnow().isoformat()+"Z",
        "version": "fusion-1.0-final",
        "locked": os.path.exists(LOCK_FILE),
        "params": P if not os.path.exists(LOCK_FILE) else "LOCKED",
        "counts": {
            "total": len(recs),
            "sufficient": sum(1 for r in recs if r["status"]=="SUFFICIENTE"),
            "critical": sum(1 for r in recs if r["Score_plus"]<0)
        },
        "stop_condition_met": stop,
        "records": recs
    }
    path = dump_json(report)
    # Stdout minimale (non narrativo)
    print(f"OK {path}  stop={stop}  sufficient={report['counts']['sufficient']}/{report['counts']['total']}  locked={report['locked']}")

if __name__ == "__main__":
    main()