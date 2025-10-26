#!/usr/bin/env bash
set -euo pipefail

ROOT="consciousness"
mkdir -p "$ROOT"/{protocols,sessions,datasets,analysis,web,out}

# 1) README
cat > "$ROOT/README_consciousness.md" <<'MD'
# Consciousness Lab · MB-X.01 + Omniabase 3D

Obiettivo: misurare marker della coscienza tramite coerenza multibase (Cx,Cy,Cz), tensore unitario (I3), ipercoerenza (H3), surprisal cumulativo e dinamiche di divergenza.

## Ipotesi
- H1: H̄3 ↑ in attenzione stabile vs riposo.
- H2: Latenza di recupero < τ(α) post-perturbazione è più breve nei soggetti ad alta coerenza.
- H3: Divergenza media ↓ su contenuto semanticamente coerente.
- H4: Surprisal cumulativo S* separa sequenze naturali vs sintetiche.

## Pipeline
1) Dati → 2) Normalizzazione → 3) run_omni3d.py → 4) compute_metrics.py → 5) viewer_consciousness.html

## Comandi
```bash
python analysis/run_omni3d.py --bases 8 12 16 --steps 200 --alpha 0.005 --smooth 0.15 \
  --signals datasets/example_signals.csv --outdir out/

python analysis/compute_metrics.py --tensor out/tensor_I3.csv --surface out/surface_H3.csv \
  --metrics out/metrics.json

Licenza codice: MIT. DOI: 10.5281/zenodo.17270742 MD

2) Preregistrazione

cat > "$ROOT/preregistration.md" <<'MD'

Preregistrazione

Disegno: within-subject. Condizioni: riposo, attenzione, n-back (2-back), percezione bistabile. Campioni: Pilot N=1, poi N≥10. Outcome primari: H̄3, latency_to_tau (tempo per rientrare sotto τ). Soglia: α=0.005 → τ=−ln(α). Analisi: test accoppiati o permutazioni, FDR per multipli. Esclusione: <80% trials validi. MD

3) Consenso/Etica

cat > "$ROOT/ethics_consent.md" <<'MD' Studio osservazionale non clinico. Nessun dato sanitario. Dati: timestamp, eventi, risposte, self-report. Anonimizzazione: ID casuale; niente PII. Condivisione: codice MIT; dati CC-BY. MD

4) Schema dati

cat > "$ROOT/data_schema.json" <<'JSON' { "$schema": "https://json-schema.org/draft/2020-12/schema", "title": "Consciousness Data Schema", "type": "object", "properties": { "session_id": {"type":"string"}, "subject_id": {"type":"string"}, "protocol": {"type":"string","enum":["P1_oddball","P2_bistable","P3_nback"]}, "alpha": {"type":"number"}, "bases": { "type":"object", "properties":{"x":{"type":"integer"},"y":{"type":"integer"},"z":{"type":"integer"}}, "required":["x","y","z"] }, "events": { "type":"array", "items": { "type":"object", "properties":{"t":{"type":"integer"},"type":{"type":"string"},"value":{"type":["number","string","boolean"]}}, "required":["t","type"] } } }, "required":["session_id","subject_id","protocol","alpha","bases","events"] } JSON

5) Protocolli

cat > "$ROOT/protocols/P1_oddball.md" <<'MD'

P1 · Oddball

Stimolo 600 ms, ISI 900 ms. Target p=0.2. ~6 min. Eventi (JSONL): {"t":123,"type":"stim","value":"standard"}, {"t":124,"type":"stim","value":"target"}, {"t":125,"type":"resp","value":true} Confronto: H̄3_target vs H̄3_standard; latency_to_tau post-target. MD

cat > "$ROOT/protocols/P2_bistable.md" <<'MD'

P2 · Percezione bistabile

Ambiguo visivo (Necker/plaids). Segna switch: {"t":..., "type":"switch","value":"dirA|dirB"}. Analisi: var(H3) e div̄ per fase; dwell-time correlato a H3. MD

cat > "$ROOT/protocols/P3_nback.md" <<'MD'

P3 · 2-back

Stimoli ogni 2s. Risposte true/false. Eventi: {"t":...,"type":"stim","value":"A"}, {"t":...,"type":"resp","value":true}. Analisi: H̄3 durante hit vs miss; div̄ nelle finestre di carico. MD

6) Sessione esempio

cat > "$ROOT/sessions/example_session.jsonl" <<'JSONL' {"session_id":"S001","subject_id":"anon01","protocol":"P1_oddball","alpha":0.005,"bases":{"x":8,"y":12,"z":16}} {"t":1,"type":"stim","value":"standard"} {"t":10,"type":"stim","value":"target"} {"t":11,"type":"resp","value":true} JSONL

7) Dati segnali esempio (brevi per demo)

cat > "$ROOT/datasets/example_signals.csv" <<'CSV' t,Cx,Cy,Cz 1,0.51,0.49,0.52 2,0.52,0.50,0.53 3,0.53,0.51,0.54 4,0.55,0.52,0.55 5,0.57,0.53,0.56 6,0.58,0.55,0.57 7,0.60,0.56,0.58 8,0.61,0.57,0.60 9,0.62,0.58,0.61 10,0.63,0.59,0.62 CSV

8) Analisi: run_omni3d.py (self-contained)

cat > "$ROOT/analysis/run_omni3d.py" <<'PY' #!/usr/bin/env python3 import argparse, math, json, os, csv from statistics import mean

EPS=1e-12 def tanh(x): if x>20: return 1.0 if x<-20: return -1.0 e2=math.exp(2*x); return (e2-1)/(e2+1)

def run_fields(Cx, Cy, Cz, alpha=0.005): rows=[]; prev_c=None; prev_i=None sL=[]; cL=[]; dL=[]; hL=[] tau = -math.log(max(EPS,alpha)); acc=0 for t,(cx,cy,cz) in enumerate(zip(Cx,Cy,Cz),start=1): n=max(EPS,math.sqrt(cxcx+cycy+czcz)) i3x,i3y,i3z=cx/n,cy/n,cz/n if prev_c is None: grad=0.0 else: dx,dy,dz=cx-prev_c[0],cy-prev_c[1],cz-prev_c[2] grad=math.sqrt(dxdx+dydy+dzdz) if prev_i is None: div=0.0 else: div=abs((i3x-prev_i[0])+(i3y-prev_i[1])+(i3z-prev_i[2])) prod=cxcycz h3=tanh(prod/(1.0+grad)) s=-math.log(max(EPS,prod)) cbar=(cx+cy+cz)/3.0 if div<tau: acc+=1 sL.append(s); cL.append(cbar); dL.append(div); hL.append(h3) rows.append({"t":t,"Cx":cx,"Cy":cy,"Cz":cz,"I3x":i3x,"I3y":i3y,"I3z":i3z,"grad":grad,"div":div,"H3":h3,"prod":prod,"surprisal":s}) prev_c=(cx,cy,cz); prev_i=(i3x,i3y,i3z) metrics={"steps":len(Cx),"alpha":alpha,"tau":tau,"mean_coherence":mean(cL),"mean_divergence":mean(dL),"mean_surprisal":mean(sL),"mean_H3":mean(hL),"accept_ratio_divergence":acc/len(Cx)} return rows, metrics

def read_signals(path, steps=None): Cx=[]; Cy=[]; Cz=[] with open(path, newline='', encoding='utf-8') as f: r=csv.DictReader(f) for i,row in enumerate(r, start=1): Cx.append(float(row["Cx"])); Cy.append(float(row["Cy"])); Cz.append(float(row["Cz"])) if steps and i>=steps: break return Cx,Cy,Cz

def main(): ap=argparse.ArgumentParser() ap.add_argument("--bases", nargs=3, type=int, required=True) ap.add_argument("--steps", type=int, default=200) ap.add_argument("--alpha", type=float, default=0.005) ap.add_argument("--smooth", type=float, default=0.15)  # compat placeholder ap.add_argument("--signals", required=True) ap.add_argument("--outdir", default="out") args=ap.parse_args() os.makedirs(args.outdir, exist_ok=True) Cx,Cy,Cz=read_signals(args.signals, steps=args.steps) rows,metrics=run_fields(Cx,Cy,Cz, alpha=args.alpha) with open(os.path.join(args.outdir,"tensor_I3.csv"),"w",newline="",encoding="utf-8") as f: w=csv.writer(f); w.writerow(["t","Cx","Cy","Cz","I3x","I3y","I3z","grad","div","H3","prod","surprisal"]) for r in rows: w.writerow([r[k] for k in ["t","Cx","Cy","Cz","I3x","I3y","I3z","grad","div","H3","prod","surprisal"]]) with open(os.path.join(args.outdir,"surface_H3.csv"),"w",newline="",encoding="utf-8") as f: w=csv.writer(f); w.writerow(["t","H3"]); [w.writerow([r["t"],r["H3"]]) for r in rows] payload={"module":"Omniabase 3D · Multibase Spatial Mathematics Engine","version":"1.0.1","doi":"10.5281/zenodo.17270742","license":"MIT","bases":{"x":args.bases[0],"y":args.bases[1],"z":args.bases[2]},"metrics":metrics} with open(os.path.join(args.outdir,"metrics.json"),"w",encoding="utf-8") as f: json.dump(payload,f,ensure_ascii=False,indent=2) print(f"OK steps={len(Cx)} bases={tuple(args.bases)} outdir={args.outdir}") print(" mean_C={:.6f} mean_div={:.6f} mean_S*={:.6f} mean_H3={:.6f} acc_div={:.3f}".format( metrics["mean_coherence"],metrics["mean_divergence"],metrics["mean_surprisal"],metrics["mean_H3"],metrics["accept_ratio_divergence"]))

if name=="main": main() PY chmod +x "$ROOT/analysis/run_omni3d.py"

9) Analisi: compute_metrics.py

cat > "$ROOT/analysis/compute_metrics.py" <<'PY' #!/usr/bin/env python3 import argparse, pandas as pd, json ap=argparse.ArgumentParser() ap.add_argument("--tensor", required=True) ap.add_argument("--surface", required=True) ap.add_argument("--metrics", required=True) args=ap.parse_args() T=pd.read_csv(args.tensor); H=pd.read_csv(args.surface) out={"n":int(len(T)),"C_mean":float((T["Cx"]+T["Cy"]+T["Cz"]).mean()/3),"div_mean":float(T["div"].mean()),"H3_mean":float(H["H3"].mean())} with open(args.metrics,"w") as f: json.dump(out,f,indent=2) print("OK") PY chmod +x "$ROOT/analysis/compute_metrics.py"

10) Viewer HTML (usa i file generati in /consciousness/out)

cat > "$ROOT/web/viewer_consciousness.html" <<'HTML' <!doctype html><html lang="it"><meta charset="utf-8">

<title>Consciousness Viewer · Omniabase-3D</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root{--bg:#0d1117;--card:#10151d;--fg:#e6edf3;--muted:#9aa4ad;--border:#212836}
body{margin:0;background:var(--bg);color:var(--fg);font:14px system-ui}
.wrap{max-width:1100px;margin:0 auto;padding:18px 12px}
.card{background:#10151d;border:1px solid var(--border);border-radius:14px;padding:12px;margin:12px 0}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
input{background:#0f141b;border:1px solid var(--border);color:var(--fg);border-radius:8px;padding:6px 8px;min-width:260px}
.btn{display:inline-block;padding:8px 12px;border:1px solid var(--border);border-radius:10px;background:#0f141b;color:#e6edf3;text-decoration:none}
.plot{width:100%;height:420px}
small{color:#9aa4ad}
</style>
<body><div class="wrap">
<div class="card">
  <h1>Consciousness Viewer · Omniabase-3D</h1>
  <div class="row">
    <input id="pT" value="/consciousness/out/tensor_I3.csv">
    <input id="pH" value="/consciousness/out/surface_H3.csv">
    <input id="pM" value="/consciousness/out/metrics.json">
    <a class="btn" href="javascript:void(0)" onclick="loadAll()">Carica</a>
  </div>
  <small>Genera prima i file con <code>run_omni3d.py</code>. Puoi sostituire con URL assoluti GitHub Pages.</small>
</div>
<div class="card" id="mWrap"><h2>Metriche</h2><div id="mBox" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:8px"></div></div>
<div class="card"><h2>Cx, Cy, Cz</h2><div id="plotC" class="plot"></div></div>
<div class="card"><h2>H3</h2><div id="plotH" class="plot"></div></div>
<div class="card"><h2>Tensore I3 (3D)</h2><div id="plotI3" class="plot" style="height:520px"></div></div>
</div>
<script>
async function fetchText(u){const r=await fetch(u,{cache:'no-store'});if(!r.ok)throw new Error(u+': '+r.status);return r.text();}
async function fetchJSON(u){const r=await fetch(u,{cache:'no-store'});if(!r.ok)throw new Error(u+': '+r.status);return r.json();}
function parseCSV(s){const [h,...rs]=s.trim().split(/\r?\n/);const ks=h.split(',');return rs.map(l=>{const c=l.split(',');const o={};ks.forEach((k,i)=>o[k.trim()]=parseFloat(c[i])||c[i]);return o;});}
function showMetrics(m){const b=m.bases||{};m=m.metrics?{bases:b,...m}:{bases:b,metrics:m};const box=document.getElementById('mBox');box.innerHTML='';
const items=[['Basi', (b.x!=null?`(${b.x},${b.y},${b.z})`:'—')],['Steps', m.metrics?.steps??'—'],['α', m.metrics?.alpha??'—'],
['τ(α)', m.metrics?.tau?.toFixed?m.metrics.tau.toFixed(3):'—'],['Mean C', fmt(m.metrics?.mean_coherence,6)],['Mean div',fmt(m.metrics?.mean_divergence,6)],
['Mean S*',fmt(m.metrics?.mean_surprisal,6)],['Mean H3',fmt(m.metrics?.mean_H3,6)],['Accept ratio',fmt(m.metrics?.accept_ratio_divergence,3)]];
for(const [k,v] of items){const d=document.createElement('div');d.style.border='1px dashed #212836';d.style.borderRadius='10px';d.style.padding='8px';d.innerHTML=`<div style="color:#9aa4ad">${k}</div><div>${v}</div>`;box.appendChild(d);}
}
function fmt(x,p){return (typeof x==='number')?x.toFixed(p):'—'}
function plotC(rows){const t=rows.map(r=>+r.t),Cx=rows.map(r=>+r.Cx),Cy=rows.map(r=>+r.Cy),Cz=rows.map(r=>+r.Cz);
Plotly.newPlot('plotC',[{x:t,y:Cx,mode:'lines',name:'Cx'},{x:t,y:Cy,mode:'lines',name:'Cy'},{x:t,y:Cz,mode:'lines',name:'Cz'}],
{paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:'#e6edf3'},margin:{l:50,r:15,t:10,b:40},yaxis:{range:[0,1]}},{displayModeBar:false});}
function plotH3(rows){const t=rows.map(r=>+r.t),H=rows.map(r=>+r.H3);
Plotly.newPlot('plotH',[{x:t,y:H,mode:'lines',name:'H3'}],
{paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:'#e6edf3'},margin:{l:50,r:15,t:10,b:40},yaxis:{rangemode:'tozero'}},{displayModeBar:false});}
function plotI3(rows){const I3x=rows.map(r=>+r.I3x),I3y=rows.map(r=>+r.I3y),I3z=rows.map(r=>+r.I3z),H=rows.map(r=>+r.H3);
Plotly.newPlot('plotI3',[{type:'scatter3d',mode:'markers',x:I3x,y:I3y,z=I3z,marker:{size:3,color:H,colorscale:'Viridis',showscale:true,colorbar:{title:'H3'}}}],
{scene:{xaxis:{title:'I3x'},yaxis:{title:'I3y'},zaxis:{title:'I3z'}},paper_bgcolor:'rgba(0,0,0,0)',font:{color:'#e6edf3'},margin:{l:0,r:0,t:0,b:0}},{displayModeBar:false});}
async function loadAll(){
  try{
    const [csvT,csvH,jM]=await Promise.all([fetchText(document.getElementById('pT').value),fetchText(document.getElementById('pH').value),fetchJSON(document.getElementById('pM').value)]);
    const rowsT=parseCSV(csvT),rowsH=parseCSV(csvH); showMetrics(jM); plotC(rowsT); plotH3(rowsH); plotI3(rowsT);
  }catch(e){alert("Errore: "+e.message);}
}
document.addEventListener('DOMContentLoaded',loadAll);
</script></html>
HTMLecho "Struttura creata in: $ROOT" echo "Prossimi passi:" echo "1) Esegui:  python $ROOT/analysis/run_omni3d.py --bases 8 12 16 --steps 200 --alpha 0.005 --smooth 0.15 --signals $ROOT/datasets/example_signals.csv --outdir $ROOT/out/" echo "2) Apri:    $ROOT/web/viewer_consciousness.html (su Neocities o in locale con un server statico)."