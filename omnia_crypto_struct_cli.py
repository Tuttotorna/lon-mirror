# omnia_crypto_struct_cli.py
# OMNIA — Structural Reading of Encrypted Traffic (no decryption)
# CLI runner over JSONL metadata
# Author: Massimiliano Brighindi (MB-X.01 / OMNIA)

from __future__ import annotations
import json
import argparse
from dataclasses import asdict
from typing import List, Dict, Any, Iterable, Tuple
import math

from omnia_crypto_struct import (
    FlowEvent,
    OmniaConfig,
    omega_total,
)

# -----------------------------
# JSONL loader
# -----------------------------

def events_from_jsonl(path: str) -> List[FlowEvent]:
    out: List[FlowEvent] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            # required
            t = obj.get("t", None)
            size = obj.get("size", None)
            if t is None or size is None:
                continue

            # optional
            direction = int(obj.get("direction", 0))
            proto = str(obj.get("proto", "UNK"))

            try:
                ev = FlowEvent(t=float(t), size=int(size), direction=direction, proto=proto)
                if ev.size >= 0:
                    out.append(ev)
            except Exception:
                continue

    out.sort(key=lambda e: e.t)
    return out

# -----------------------------
# Windowing
# -----------------------------

def slice_windows(events: List[FlowEvent], window_s: float, step_s: float) -> List[Tuple[float, float, List[FlowEvent]]]:
    if not events:
        return []

    t0 = events[0].t
    t1 = events[-1].t
    windows: List[Tuple[float, float, List[FlowEvent]]] = []

    start = t0
    idx0 = 0  # moving start index
    n = len(events)

    while start <= t1:
        end = start + window_s

        # advance idx0 to first event >= start
        while idx0 < n and events[idx0].t < start:
            idx0 += 1

        # collect until < end
        j = idx0
        bucket: List[FlowEvent] = []
        while j < n and events[j].t < end:
            bucket.append(events[j])
            j += 1

        windows.append((start, end, bucket))
        start += step_s

    return windows

def group_by_proto(events: List[FlowEvent]) -> Dict[str, List[FlowEvent]]:
    d: Dict[str, List[FlowEvent]] = {}
    for e in events:
        d.setdefault(e.proto, []).append(e)
    return d

# -----------------------------
# Reporting
# -----------------------------

def analyze_windows(events: List[FlowEvent], cfg: OmniaConfig, window_s: float, step_s: float, per_proto: bool) -> Dict[str, Any]:
    windows = slice_windows(events, window_s=window_s, step_s=step_s)

    results: List[Dict[str, Any]] = []
    for (a, b, bucket) in windows:
        row: Dict[str, Any] = {
            "t_start": a,
            "t_end": b,
            "n_events": len(bucket),
        }

        if not bucket:
            row["Ω_total"] = 0.0
            row["Co+_total"] = 0.0
            results.append(row)
            continue

        if per_proto:
            proto_map = group_by_proto(bucket)
            proto_rows: Dict[str, Any] = {}
            for proto, evs in proto_map.items():
                proto_rows[proto] = omega_total(evs, cfg)
            row["by_proto"] = proto_rows

            # fused across protos: average Ω_total of each proto (weighted by count)
            total = 0.0
            wsum = 0
            for proto, evs in proto_map.items():
                w = len(evs)
                total += proto_rows[proto]["Ω_total"] * w
                wsum += w
            fused = total / max(1, wsum)
            row["Ω_total"] = fused
            row["Co+_total"] = math.exp(fused)
        else:
            row.update(omega_total(bucket, cfg))

        results.append(row)

    return {
        "config": {
            "window_s": window_s,
            "step_s": step_s,
            "bases": list(cfg.bases),
            "per_proto": per_proto,
        },
        "n_total_events": len(events),
        "t_min": events[0].t if events else None,
        "t_max": events[-1].t if events else None,
        "windows": results,
    }

# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="OMNIA structural reading over encrypted traffic metadata (JSONL). No decryption.")
    ap.add_argument("--in_jsonl", required=True, help="Input JSONL path with fields: t,size,direction,proto")
    ap.add_argument("--out_json", required=True, help="Output report JSON path")
    ap.add_argument("--window_s", type=float, default=60.0, help="Window length in seconds")
    ap.add_argument("--step_s", type=float, default=30.0, help="Window step in seconds")
    ap.add_argument("--per_proto", action="store_true", help="Compute Ω per proto and fuse")
    args = ap.parse_args()

    cfg = OmniaConfig()
    events = events_from_jsonl(args.in_jsonl)
    report = analyze_windows(events, cfg, window_s=args.window_s, step_s=args.step_s, per_proto=args.per_proto)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()