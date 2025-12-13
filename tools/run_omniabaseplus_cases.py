
import json
from omniabaseplus_percent_v1 import evaluate

PATH = "cases/omniabaseplus_cases.jsonl"
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
def main():
    total = 0
    ok = 0
    mismatches = []

    with open(PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            case = json.loads(line)
            res = evaluate(case["statement"], case.get("context"))
            exp = case["expected"]

            pass_exp = exp["pass"]
            pass_got = {"B2": res.pass_b2, "B4": res.pass_b4, "B8": res.pass_b8}

            state_ok = (res.state == exp["state"])
            pass_ok = (pass_got == pass_exp)

            total += 1
            if state_ok and pass_ok:
                ok += 1
            else:
                mismatches.append({
                    "id": case["id"],
                    "expected_state": exp["state"],
                    "got_state": res.state,
                    "expected_pass": pass_exp,
                    "got_pass": pass_got,
                    "omegas": {"B2": res.omega2, "B4": res.omega4, "B8": res.omega8},
                })

    print(f"OK {ok}/{total}")
    if mismatches:
        print("\nMISMATCHES:")
        for m in mismatches:
            print(m)

if __name__ == "__main__":
    main()