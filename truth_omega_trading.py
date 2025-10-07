#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TruthΩ Trading Simulator — MB-X.01 / L.O.N.
Confronta strategia basata su coerenza multibase (TruthΩ → Co⁺) vs. baseline (SMA crossover / buy&hold).
Input: CSV con colonne: timestamp, close  (timestamp in ISO8601 o 'YYYY-MM-DD', close float)
Se non fornisci un CSV, può generare una serie sintetica per demo (--demo).

MIT License © 2025 Massimiliano Brighindi
"""

import argparse, math, sys, json, os
from dataclasses import dataclass
import numpy as np
import pandas as pd

# ---------- util ----------

def read_price_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns or "close" not in df.columns:
        raise ValueError("CSV richiesto con colonne: timestamp, close")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def gen_demo(n=1500, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    # trend lento + regime shift + rumore
    base = 100 + 0.02*t + 2*np.sin(t/40.0) + 4*np.sin(t/200.0)
    regimes = (rng.standard_normal(n)*0.5).cumsum()*0.02
    price = base + regimes + rng.standard_normal(n)*0.8
    price = np.maximum(1.0, price)
    ts = pd.date_range("2015-01-01", periods=n, freq="D")
    return pd.DataFrame({"timestamp": ts, "close": price})

def pct_change(arr, period=1):
    x = pd.Series(arr).pct_change(periods=period).to_numpy()
    return x

# ---------- TruthΩ / Co⁺ / Score⁺ (simplificato, riproducibile) ----------

@dataclass
class TruthParams:
    bases: tuple = (1, 3, 5, 10, 20)  # scale in giorni (o barre)
    lookback_z: int = 60              # finestra per z-score per base
    eps: float = 1e-9

def truth_omega_series(close: np.ndarray, params: TruthParams) -> pd.Series:
    """
    Implementazione operativa coerente con la spec:
    - r_b(t) = ritorno percentuale a diverse basi b
    - φ_k: qui usiamo z-score di r_b su lookback per renderli comparabili
    - Var_b[φ_k(r_b(t))]: varianza cross-base al tempo t
    - TruthΩ(t) = -log(1 + mean_k Var_b[...]) ; qui k=1 (una feature principale)
    Risultato ≤ 0; più vicino a 0 = maggiore coerenza base-invariante.
    """
    df = pd.DataFrame({"close": close})
    mat = []
    for b in params.bases:
        r = df["close"].pct_change(b)
        z = (r - r.rolling(params.lookback_z).mean()) / (r.rolling(params.lookback_z).std() + params.eps)
        mat.append(z)
    M = pd.concat(mat, axis=1)
    M.columns = [f"z_b{b}" for b in params.bases]
    # varianza cross-base per riga
    var_cross = M.var(axis=1, skipna=True)
    # TruthΩ
    truth_omega = -np.log(1.0 + var_cross.fillna(var_cross.median()).clip(lower=0.0) + params.eps)
    return truth_omega

def co_plus(truth_omega: pd.Series) -> pd.Series:
    # Co⁺ = exp(TruthΩ) ∈ (0,1]
    return np.exp(truth_omega)

# ---------- Strategie ----------

@dataclass
class StrategyParams:
    co_threshold: float = 0.80        # soglia Co⁺ per abilitare operatività
    trend_fast: int = 10
    trend_slow: int = 50
    max_position: int = 1             # long-only: 0/1
    fees_bps: float = 1.0             # commissioni andata+ritorno in basis points totali (0.01%)
    slippage_bps: float = 1.0         # slippage medio stimato
    risk_stop: float = 0.15           # stop a -15% dal max equity

def truth_strategy(df: pd.DataFrame, sp: StrategyParams, tp: TruthParams) -> pd.DataFrame:
    out = df.copy()
    out["TruthΩ"] = truth_omega_series(out["close"].to_numpy(), tp)
    out["Co⁺"] = co_plus(out["TruthΩ"])

    # trend gating (direzione): mediana dei z_b>0? Usiamo filtro SMA fast/slow sul prezzo
    out["sma_fast"] = out["close"].rolling(sp.trend_fast).mean()
    out["sma_slow"] = out["close"].rolling(sp.trend_slow).mean()
    out["trend_up"] = (out["sma_fast"] > out["sma_slow"]).astype(int)

    # segnale: long se Co⁺ >= soglia e trend_up==1, altrimenti flat
    out["signal"] = ((out["Co⁺"] >= sp.co_threshold) & (out["trend_up"] == 1)).astype(int)

    # rendimenti log per stabilità numerica
    out["ret"] = np.log(out["close"]).diff().fillna(0.0)

    # commissioni/slippage quando cambia posizione
    out["pos"] = out["signal"].shift().fillna(0).astype(int)  # esegui apertura dalla barra successiva
    out["trade"] = (out["pos"].diff().abs().fillna(0) > 0).astype(int)
    fric = (sp.fees_bps + sp.slippage_bps) / 10000.0
    out["cost"] = - out["trade"] * fric

    # equity curve
    out["strategy_ret"] = out["pos"] * out["ret"] + out["cost"]
    out["equity"] = out["strategy_ret"].cumsum().apply(np.exp)

    # drawdown & risk stop ( opzionale: azzera pos quando drawdown oltre soglia )
    out["equity_peak"] = out["equity"].cummax()
    out["dd"] = (out["equity"] / out["equity_peak"] - 1.0)
    breach = (out["dd"] <= -sp.risk_stop).astype(int)
    # se breach, chiudi e resta flat finché dd non rientra (semplice: reset pos=0 quella barra)
    out.loc[breach == 1, "pos"] = 0
    out["strategy_ret"] = out["pos"] * out["ret"] + out["cost"]
    out["equity"] = out["strategy_ret"].cumsum().apply(np.exp)
    out["equity_peak"] = out["equity"].cummax()
    out["dd"] = (out["equity"] / out["equity_peak"] - 1.0)
    return out

def sma_baseline(df: pd.DataFrame, fast=10, slow=50, fees_bps=2.0) -> pd.DataFrame:
    out = df.copy()
    out["ret"] = np.log(out["close"]).diff().fillna(0.0)
    out["sma_fast"] = out["close"].rolling(fast).mean()
    out["sma_slow"] = out["close"].rolling(slow).mean()
    out["signal"] = (out["sma_fast"] > out["sma_slow"]).astype(int)
    out["pos"] = out["signal"].shift().fillna(0).astype(int)
    fric = (fees_bps)/10000.0
    out["trade"] = (out["pos"].diff().abs().fillna(0) > 0).astype(int)
    out["cost"] = - out["trade"] * fric
    out["strategy_ret"] = out["pos"] * out["ret"] + out["cost"]
    out["equity"] = out["strategy_ret"].cumsum().apply(np.exp)
    out["equity_peak"] = out["equity"].cummax()
    out["dd"] = (out["equity"] / out["equity_peak"] - 1.0)
    return out

def buy_hold(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret"] = np.log(out["close"]).diff().fillna(0.0)
    out["equity"] = out["ret"].cumsum().apply(np.exp)
    out["equity_peak"] = out["equity"].cummax()
    out["dd"] = (out["equity"] / out["equity_peak"] - 1.0)
    return out

# ---------- metriche ----------

def metrics(equity: pd.Series, rets: pd.Series, freq_per_year=252) -> dict:
    if len(equity) < 2:
        return {"CAGR": 0, "Vol": 0, "Sharpe": 0, "MaxDD": 0, "Final": float(equity.iloc[-1]) if len(equity)>0 else 1.0}
    total_return = float(equity.iloc[-1]) - 1.0
    years = max(1e-9, (len(equity) / freq_per_year))
    CAGR = (float(equity.iloc[-1]))**(1/years) - 1.0
    Vol = np.std(rets.dropna()) * math.sqrt(freq_per_year)
    Sharpe = (np.mean(rets.dropna()) * freq_per_year) / (Vol + 1e-9)
    peak = equity.cummax()
    dd = (equity / peak - 1.0).min()
    return {"CAGR": CAGR, "Vol": Vol, "Sharpe": Sharpe, "MaxDD": float(dd), "Final": float(equity.iloc[-1])}

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="TruthΩ Trading Simulator — MB-X.01 / L.O.N.")
    ap.add_argument("--csv", type=str, help="Path CSV con colonne: timestamp, close")
    ap.add_argument("--demo", action="store_true", help="Usa serie sintetica (demo)")
    ap.add_argument("--out", type=str, default="truth_omega_results.csv", help="File CSV risultati")
    ap.add_argument("--co_threshold", type=float, default=0.80, help="Soglia Co⁺")
    ap.add_argument("--fees_bps", type=float, default=1.0, help="Commissioni+slippage (bps) TruthΩ strat")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    # dati
    if args.csv:
        df = read_price_csv(args.csv)
        freq_year = 252  # assumiamo daily; se intraday, adegua
    elif args.demo:
        df = gen_demo(seed=args.seed)
        freq_year = 252
    else:
        print("Errore: specifica --csv <file.csv> oppure --demo", file=sys.stderr)
        sys.exit(1)

    df = df[["timestamp","close"]].dropna().reset_index(drop=True)

    # strategie
    tp = TruthParams()
    sp = StrategyParams(co_threshold=args.co_threshold, fees_bps=args.fees_bps, slippage_bps=0.0)
    truth_df = truth_strategy(df, sp, tp)
    base_df  = sma_baseline(df, fast=10, slow=50, fees_bps=2.0)
    bh_df    = buy_hold(df)

    # allinea index
    res = pd.DataFrame({
        "timestamp": df["timestamp"],
        "close": df["close"],
        "TruthOmega": truth_df["TruthΩ"],
        "CoPlus": truth_df["Co⁺"],
        "signal_truth": truth_df["pos"],
        "equity_truth": truth_df["equity"],
        "equity_sma": base_df["equity"],
        "equity_buyhold": bh_df["equity"],
        "dd_truth": truth_df["dd"],
    })

    # metriche
    m_truth = metrics(truth_df["equity"], truth_df["strategy_ret"], freq_per_year=freq_year)
    m_sma   = metrics(base_df["equity"],  base_df["strategy_ret"],  freq_per_year=freq_year)
    m_bh    = metrics(bh_df["equity"],    bh_df["ret"],             freq_per_year=freq_year)

    summary = {
        "TruthΩ_Strategy": m_truth,
        "SMA_Baseline": m_sma,
        "BuyAndHold": m_bh,
        "Params": {
            "TruthParams": tp.__dict__,
            "StrategyParams": sp.__dict__
        }
    }

    # output
    res.to_csv(args.out, index=False)
    with open(os.path.splitext(args.out)[0] + "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] Salvato: {args.out}")
    print(f"[OK] Salvato: {os.path.splitext(args.out)[0] + '_summary.json'}")
    print("\n== METRICHE ==")
    for k,v in summary.items():
        if k in ("Params",): continue
        print(k, {kk: round(vv,4) if isinstance(vv, (int,float)) else vv for kk,vv in v.items()})

if __name__ == "__main__":
    main()