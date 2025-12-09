import json
import numpy as np
from omnia_totale import pbii, truth_omega  # o il path corretto
# Se il modulo è diverso, lo correggo dopo.

# ==== LOAD DATASET ====
def load_gsm8k_outputs(path="data/gsm8k_model_outputs.jsonl"):
    samples = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            samples.append(json.loads(line))
    return samples

# ==== PBII INSTABILITY SCORE ====
def pbii_chain_instability(numbers):
    return float(np.mean([pbii(n) for n in numbers])) if numbers else 0.0

# ==== EXTRACT NUMBERS FROM CHAIN ====
import re
def extract_numbers(text):
    return [int(x) for x in re.findall(r"\b\d+\b", text)]

# ==== MAIN EVAL ====
def evaluate():
    data = load_gsm8k_outputs()
    
    records = []
    for sample in data:
        nums = extract_numbers(sample["model_chain"])
        instability = pbii_chain_instability(nums)

        correct = (str(sample["model_answer"]).strip() == str(sample["gold_answer"]).strip())

        records.append({
            "id": sample["id"],
            "correct": correct,
            "pbii_instability": instability,
            "model_answer": sample["model_answer"],
            "gold_answer": sample["gold_answer"],
            "question": sample["question"]
        })
    
    return records

# ==== SUMMARY ====
def summarise(records):
    correct = [r for r in records if r["correct"]]
    incorrect = [r for r in records if not r["correct"]]

    avg_inst_correct = np.mean([r["pbii_instability"] for r in correct]) if correct else 0
    avg_inst_incorrect = np.mean([r["pbii_instability"] for r in incorrect]) if incorrect else 0

    return {
        "total": len(records),
        "correct": len(correct),
        "incorrect": len(incorrect),
        "avg_inst_correct": avg_inst_correct,
        "avg_inst_incorrect": avg_inst_incorrect,
        "instability_gap": avg_inst_incorrect - avg_inst_correct
    }

if __name__ == "__main__":
    records = evaluate()
    summary = summarise(records)

    print("====== GSM8K EVAL SUMMARY ======")
    print(summary)