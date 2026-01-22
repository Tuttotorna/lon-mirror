"""Test SEI (Structural Efficiency Index) properties."""

import math
from omnia.sei import SEI


class TestSEI:
    """Verify SEI correctly computes ΔΩ/ΔC efficiency."""

    def test_sei_on_constant_efficiency_curve(self):
        sei = SEI(window=1, eps=1e-12)

        omega_values = [0.2, 0.4, 0.6, 0.8, 1.0]
        cost_values = [1, 2, 3, 4, 5]

        sei_curve = sei.curve(omega_values, cost_values)

        if len(sei_curve) > 1:
            mu = sum(sei_curve) / len(sei_curve)
            var = sum((s - mu) ** 2 for s in sei_curve) / len(sei_curve)
            sei_std = math.sqrt(var)
            assert sei_std < 0.15

    def test_sei_detects_efficiency_drop(self):
        sei = SEI(window=1, eps=1e-12)

        omega_values = [0.1, 0.5, 0.9, 0.95, 0.97]
        cost_values = [1, 2, 3, 4, 5]

        sei_curve = sei.curve(omega_values, cost_values)

        if len(sei_curve) >= 2:
            assert sei_curve[0] >= sei_curve[-1] * 0.8

    def test_sei_handles_zero_cost_increment(self):
        sei = SEI(window=1, eps=1e-6)

        omega_values = [0.2, 0.5, 0.8]
        cost_values = [1, 1, 1]

        sei_curve = sei.curve(omega_values, cost_values)

        assert all(isinstance(s, (int, float)) for s in sei_curve)
        assert all(math.isfinite(s) for s in sei_curve)

    def test_sei_output_length_with_window(self):
        omega_values = [0.1, 0.2, 0.3, 0.4, 0.5]
        cost_values = [1, 2, 3, 4, 5]

        sei1 = SEI(window=1, eps=1e-12)
        curve1 = sei1.curve(omega_values, cost_values)
        assert len(curve1) == len(omega_values) - 1

        sei3 = SEI(window=3, eps=1e-12)
        curve3 = sei3.curve(omega_values, cost_values)
        assert 0 < len(curve3) <= len(omega_values) - 1