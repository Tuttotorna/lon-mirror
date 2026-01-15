from omnia.omega_total import omega_total


def test_omega_total_basic():
    metrics = {"a": 0.5, "b": 1.0}
    r = omega_total(metrics)
    assert "omega_total" in r
    assert abs(r["omega_total"] - 0.75) < 1e-6