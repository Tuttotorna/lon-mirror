import sys
from pathlib import Path

# Ensure repo root is on sys.path (works in Colab and locally)
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.structural_risk_check import structural_risk_check


def main() -> None:
    p = REPO_ROOT / "examples" / "llm_output_bad.txt"
    text = p.read_text(encoding="utf-8")
    result = structural_risk_check(text)
    print(result)


if __name__ == "__main__":
    main()