from omniabase_core import omniabase_truth
from bases_minimal import check_base

BASES = ["POSSIBILITY", "CAUSALITY", "CONTEXT"]

def run_case(name: str, statement: str, context: dict):
    r = omniabase_truth(statement=statement, bases=BASES, check_fn=check_base, context=context)
    print("\n==", name, "==")
    print("statement:", r.statement)
    print("bases_checked:", r.bases_checked)
    print("failed_bases:", r.failed_bases)
    print("verdict:", r.verdict)

def main():
    # Case A: explicit impossibility
    run_case(
        "A_impossible",
        "You can be in two cities at the same instant.",
        {"explicit_impossibility": True},
    )

    # Case B: unobservable dependency
    run_case(
        "B_unobservable",
        "Tomorrow I will make it rain with my mind.",
        {"depends_on_unobservable": True},
    )

    # Case C: truncated blame statement (missing premise + no mechanism)
    run_case(
        "C_truncated",
        "Tu non mi aiuti in casa.",
        {"missing_critical_premise": True, "causation_claim": True, "mechanism_provided": False},
    )

    # Case D: fully supported (should pass all three)
    run_case(
        "D_supported",
        "If the light switch is ON, the lamp turns on.",
        {"causation_claim": True, "mechanism_provided": True, "missing_critical_premise": False},
    )

if __name__ == "__main__":
    main()