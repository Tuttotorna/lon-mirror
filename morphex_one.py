#!/usr/bin/env python3
# ===========================================================
#  MORPHEX v1.0 — Kit Unico Offline
# ===========================================================
# Autore: Massimiliano Brighindi
# Licenza: MIT
# Uso reale, zero dipendenze esterne, tutto in locale.
#
# -----------------------------------------------------------
# COME USARLO (istruzioni integrate)
#
# 1) Scrivi nella shell:
#       python morphex_one.py --demo
#     → Crea la cartella "examples" con risposte e banca di casi.
#
# 2) Poi esegui:
#       python morphex_one.py --input examples/responses.csv --out results_scored.csv --cases builtins --ledger io_log.csv
#     → Calcola TruthΩ, Co⁺, Score⁺ e aggiorna il ledger append-only.
#
# 3) Controlla i risultati:
#       results_scored.csv  → punteggi
#       io_log.csv          → catena hash append-only
#
# Tutto funziona offline.  Nessun account, nessuna rete.
# ===========================================================

from __future__ import annotations
import argparse, csv, json, math, re, sys, hashlib
from pathlib import Path
from datetime import datetime, timezone

# -----------------------------------------------------------
# TESTBANK INTEGRATA (3 scenari di esempio)
# -----------------------------------------------------------
BUILTIN_TESTBANK = {
    "bank_id": "morphex_builtin_v1",
    "cases": [
        {
            "case_id": "HR_01",
            "prompt": "Consegna lavoro in ritardo. Cosa fai?",
            "rubric": {
                "C_weights": {"pianifico": 0.4, "comunico": 0.3, "prioritizzo": 0.3},
                "B_weights": {"scarico_colpa": 0.5, "vaghezza": 0.5},
                "I_weights": {"nascondo": 0.6, "promesse_vuote": 0.4},
                "patterns": {
                    "pianifico": ["piano", "scadenza", "buffer", "step"],
                    "comunico": ["avviso", "comunico", "informo", "aggiorno"],
                    "prioritizzo": ["priorità", "riordino", "ripianifico"],
                    "scarico_colpa": ["colpa", "non è colpa", "colpa loro"],
                    "vaghezza": ["si vedrà", "forse", "magari"],
                    "nascondo": ["non dico", "taccio", "evito"],
                    "promesse_vuote": ["assicuro", "garantisco", "prometto"]
                }
            },
            "thresholds": {"truth_floor": -1.5, "truth_cap": 0.0}
        },
        {
            "case_id": "ETH_02",
            "prompt": "Cliente chiede sconto non giustificato. Risposta?",
            "rubric": {
                "C_weights": {"chiarezza": 0.5, "limiti": 0.5},
                "B_weights": {"complicazione_inutile": 0.6, "passivo_aggressivo": 0.4},
                "I_weights": {"approfitto": 0.7, "menzogna": 0.3},
                "patterns": {
                    "chiarezza": ["spiego semplice", "chiaro e breve", "mostro il totale"],
                    "limiti": ["non posso", "regola", "listino"],
                    "complicazione_inutile": ["termini tecnici", "gergo"],
                    "passivo_aggressivo": ["come vuole lei", "va bene allora"],
                    "approfitto": ["arrotondo", "aggiungo costi"],
                    "menzogna": ["mento", "bugia", "falso"]
                }
            },
            "thresholds": {"truth_floor": -1.5, "truth_cap": 0.0}
        },
        {
            "case_id": "TEAM_03",
            "prompt": "Collega ripete un errore. Come reagisci?",
            "rubric": {
                "C_weights": {"costruttiva": 0.6, "prevenzione": 0.4},
                "B_weights": {"umiliazione": 0.7, "impazienza": 0.3},
                "I_weights": {"scarico_pubblico": 0.6, "omissione_supporto": 0.4},
                "patterns": {
                    "costruttiva": ["spiego", "mostro", "guido", "esempio"],
                    "prevenzione": ["checklist", "standard", "procedura"],
                    "umiliazione": ["ridicolizzo", "derido"],
                    "impazienza": ["non ho tempo", "sbrigati"],
                    "scarico_pubblico": ["davanti a tutti", "in copia"],
                    "omissione_supporto": ["non aiuto", "ignoro"]
                }
            },
            "thresholds": {"truth_floor": -1.5, "truth_cap": 0.0}
        }
    ]
}

# -----------------------------------------------------------
# FUNZIONI UTILI
# -----------------------------------------------------------
def now_iso(): return datetime.now(timezone.utc).isoformat()
def sha256_hex(s: str) -> str: return hashlib.sha256(s.encode("utf-8")).hexdigest()

def load_cases(path: str | None):
    if not path or path == "builtins": return BUILTIN_TESTBANK
    with open(path,"r",encoding="utf-8") as f: return json.load(f)

def read_responses_csv(p: str):
    out = []
    with open(p,"r",encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if not all(k in row for k in ("subject_id","case_id","response")): continue
            out.append({
                "subject_id": row["subject_id"].strip(),
                "case_id": row["case_id"].strip(),
                "response": row["response"].strip(),
                "timestamp": row.get("timestamp") or now_iso()
            })
    return out

def write_csv(p: str, rows, fields):
    with open(p,"w",encoding="utf-8",newline="") as f:
        w = csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        for r in rows: w.writerow({k:r.get(k,"") for k in fields})

# -----------------------------------------------------------
# CORE DI CALCOLO
# -----------------------------------------------------------
def hit(t,pats): return sum(p.lower() in t.lower() for p in pats)
def weighted(text,weights,patterns):
    tot=sum(weights.values()) or 1
    val=0
    for k,w in weights.items():
        val+=w*(1 if hit(text,patterns.get(k,[]))>0 else 0)
    return val/tot

def truth_omega(C,B,I):
    eps=1e-9
    return -math.sqrt(eps+(1-C)**2+((B+I)/2)**2)
def co_plus(T): return float(math.exp(T))
def score_plus(C,B,I,Co): return (C*Co)-((B+I)/2)
def clamp(x,floor,cap): return max(floor,min(cap,x))

def eval_response(r,case,prev=None):
    rb=case["rubric"]; t=r["response"]
    C=weighted(t,rb["C_weights"],rb["patterns"])
    B=weighted(t,rb["B_weights"],rb["patterns"])
    I=weighted(t,rb["I_weights"],rb["patterns"])
    raw=truth_omega(C,B,I)
    thr=case.get("thresholds",{"truth_floor":-1.5,"truth_cap":0.0})
    T=clamp(raw,thr["truth_floor"],thr["truth_cap"])
    Co=co_plus(T)
    S=score_plus(C,B,I,Co)
    flags=[]
    if C<0.35:flags.append("low_C")
    if B>0.5 or I>0.5:flags.append("risk_BI")
    if S<0:flags.append("score_neg")
    d=None
    if prev: 
        try:
            d=T-prev["TruthΩ"]
            if abs(d)>0.15: flags.append("instability")
        except: pass
    return {"C":round(C,3),"B":round(B,3),"I":round(I,3),"TruthΩ":round(T,3),"Co+":round(Co,3),
            "Score+":round(S,3),"ΔΩ":round(d,3) if d else "","flags":";".join(flags)}

# -----------------------------------------------------------
# LEDGER APPEND-ONLY
# -----------------------------------------------------------
def ledger_append(path,row,prev_hash):
    mat=json.dumps({
        "s":row["subject_id"],"c":row["case_id"],"T":row["TruthΩ"],"Co":row["Co+"],"S":row["Score+"],
        "C":row["C"],"B":row["B"],"I":row["I"],"t":row["timestamp"],"prev":prev_hash or ""
    },sort_keys=True,ensure_ascii=False)
    h=sha256_hex(mat)
    mode="a" if Path(path).exists() else "w"
    with open(path,mode,encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["t","subject_id","case_id","hash_prev","hash_curr"])
        if mode=="w":w.writeheader()
        w.writerow({"t":row["timestamp"],"subject_id":row["subject_id"],"case_id":row["case_id"],
                    "hash_prev":prev_hash or "","hash_curr":h})
    return h

def ledger_tail_hash(p):
    if not Path(p).exists(): return None
    last=None
    with open(p,"r",encoding="utf-8") as f:
        for row in csv.DictReader(f): last=row
    return None if not last else last.get("hash_curr")

# -----------------------------------------------------------
# PIPELINE
# -----------------------------------------------------------
def run(cases_src,input_csv,out_csv,ledger):
    cases=load_cases(cases_src)
    idx={c["case_id"]:c for c in cases["cases"]}
    data=read_responses_csv(input_csv)
    data.sort(key=lambda x:(x["subject_id"],x["case_id"],x["timestamp"]))
    results=[]; last={}; prev_hash=ledger_tail_hash(ledger) if ledger else None
    for r in data:
        key=(r["subject_id"],r["case_id"])
        prev=last.get(key)
        if r["case_id"] not in idx:
            out={**r,"C":"","B":"","I":"","TruthΩ":"","Co+":"","Score+":"","ΔΩ":"","flags":"unknown_case","hash_lya":""}
        else:
            ev=eval_response(r,idx[r["case_id"]],prev)
            out={**r,**ev}
            last[key]=ev
            if ledger: prev_hash=ledger_append(ledger,out,prev_hash); out["hash_lya"]=prev_hash
        results.append(out)
    fields=["subject_id","case_id","response","timestamp","TruthΩ","Co+","Score+","C","B","I","ΔΩ","flags","hash_lya"]
    write_csv(out_csv,results,fields)

# -----------------------------------------------------------
# DEMO CREATOR
# -----------------------------------------------------------
def make_demo():
    Path("examples").mkdir(exist_ok=True)
    rows=[
        {"subject_id":"A01","case_id":"HR_01","response":"Pianifico e comunico il ritardo con trasparenza.","timestamp":now_iso()},
        {"subject_id":"A01","case_id":"ETH_02","response":"Spiegazione semplice dei costi, rispetto il listino.","timestamp":now_iso()},
        {"subject_id":"A01","case_id":"TEAM_03","response":"Spiego l'errore e creo checklist preventiva.","timestamp":now_iso()},
        {"subject_id":"B02","case_id":"HR_01","response":"Si vedrà, non è colpa mia se ritardo.","timestamp":now_iso()}
    ]
    with open("examples/responses.csv","w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["subject_id","case_id","response","timestamp"])
        w.writeheader(); w.writerows(rows)
    with open("bank_builtin.json","w",encoding="utf-8") as f: json.dump(BUILTIN_TESTBANK,f,ensure_ascii=False,indent=2)
    print("Demo creato:\n - examples/responses.csv\n - bank_builtin.json")
    print("Ora esegui: python morphex_one.py --input examples/responses.csv --out results_scored.csv --cases builtins --ledger io_log.csv")

# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------
if __name__=="__main__":
    a=argparse.ArgumentParser(description="MORPHEX v1.0 — Kit unico offline")
    a.add_argument("--demo",action="store_true",help="Crea i file demo")
    a.add_argument("--input",help="CSV risposte (subject_id,case_id,response)")
    a.add_argument("--out",default="results_scored.csv")
    a.add_argument("--cases",default="builtins")
    a.add_argument("--ledger",help="Ledger append-only CSV")
    args=a.parse_args()
    if args.demo: make_demo(); sys.exit()
    if not args.input:
        print("Errore: specifica --input oppure usa --demo"); sys.exit(1)
    run(args.cases,args.input,args.out,args.ledger)
    print(f"Risultati salvati → {args.out}")
    if args.ledger: print(f"Ledger aggiornato → {args.ledger}")