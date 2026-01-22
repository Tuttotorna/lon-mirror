"""Integration tests for complete OMNIA measurement workflows."""

import pytest


def test_integration_if_available():
    pytest.importorskip("examples.omnia_total_explainer", reason="Integration module not found")
    from examples.omnia_total_explainer import main

    x = "Deterministic test input 123"

    report = main(x=x, x_prime=None)

    assert "measurements" in report
    assert "aperspective" in report["measurements"]

    report2 = main(x=x, x_prime=None)
    omega1 = report["measurements"]["aperspective"]["omega_ap"]
    omega2 = report2["measurements"]["aperspective"]["omega_ap"]

    assert omega1 == omega2