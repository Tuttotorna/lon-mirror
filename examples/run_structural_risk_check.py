from pathlib import Path
from tools.structural_risk_check import structural_risk_check

def main() -> None:
    p = Path("examples/llm_output_bad.txt")
    text = p.read_text(encoding="utf-8")
    result = structural_risk_check(text)
    print(result)

if __name__ == "__main__":
    main()