# benchmarks/run_closed_models.py

import json
import argparse
import os
import math

def read_jsonl(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            data[obj["id"]] = obj
    return data

def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def omega_score(text: str) -> float:
    if not text:
        return 0.0

    length = len(text)
    freq = {}

    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1

    entropy = 0.0
    for c in freq.values():
        p = c / length
        entropy -= p * math.log(p + 1e-12)

    return round(entropy / math.log(length + 2), 4)

def main(prompts_path, outputs_path, out_path):
    prompts = read_jsonl(prompts_path)
    outputs = read_jsonl(outputs_path)

    if set(prompts.keys()) != set(outputs.keys()):
        raise RuntimeError("ID mismatch between prompts and outputs")

    ensure_dir(out_path)

    with open(out_path, "w", encoding="utf-8") as f:
        for pid in sorted(prompts.keys()):
            p = prompts[pid]
            o = outputs[pid]

            text = o.get("output", "")
            omega = omega_score(text)

            row = {
                "id": pid,
                "category": p.get("category"),
                "model": o.get("model"),
                "omega_score": omega,
                "text_length": len(text),
            }

            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("DONE →", out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    main(args.prompts, args.outputs, args.out)