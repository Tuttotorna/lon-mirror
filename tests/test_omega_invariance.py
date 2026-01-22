"""Test aperspective invariance properties of Ω measurements."""

import pytest
from omnia.lenses.aperspective_invariance import AperspectiveInvariance


class TestAperspectiveInvariance:
    """Verify that Ω_ap satisfies theoretical invariance properties."""

    def test_identity_transform_is_perfect(self, tolerance):
        """Identity transform must yield overlap = 1.0."""
        from omnia.lenses.aperspective_invariance import t_identity

        x = "Test string with content"
        assert t_identity(x) == x

        ap = AperspectiveInvariance(transforms=[("id", t_identity)])
        result = ap.measure(x)

        assert abs(result.per_transform_scores["id"] - 1.0) < tolerance["strict"]

    def test_whitespace_invariance_isolated(self, transforms_whitespace_only, tolerance):
        """Ω should be invariant to whitespace-only changes."""
        x1 = "hello world"
        x2 = "hello    world"
        x3 = "hello\t\n  world"

        ap = AperspectiveInvariance(transforms=transforms_whitespace_only)

        omega1 = ap.measure(x1).omega_score
        omega2 = ap.measure(x2).omega_score
        omega3 = ap.measure(x3).omega_score

        assert abs(omega1 - omega2) < tolerance["loose"]
        assert abs(omega1 - omega3) < tolerance["loose"]

    def test_omega_bounded_01(self, sample_texts, transforms_full):
        """Ω must always be in [0, 1] interval."""
        ap = AperspectiveInvariance(transforms=transforms_full)

        for name, text in sample_texts.items():
            if text and text.strip():
                result = ap.measure(text)
                assert 0.0 <= result.omega_score <= 1.0, (
                    f"Ω out of bounds for '{name}': {result.omega_score}"
                )

    def test_repetition_monotonicity(self, transforms_whitespace_only):
        """Repetitive patterns should yield Ω >= non-repetitive (weak monotonicity)."""
        ap = AperspectiveInvariance(transforms=transforms_whitespace_only)

        structured = "ABCD " * 20
        low_structure = "QWERTYUIOPASDFGHJKLZXCVBNM"

        omega_structured = ap.measure(structured).omega_score
        omega_low = ap.measure(low_structure).omega_score

        assert omega_structured >= omega_low, (
            f"Repetition failed monotonicity: Ω_struct={omega_structured}, Ω_low={omega_low}"
        )

    @pytest.mark.benchmark
    def test_repetition_significant_gap(self, transforms_full, tolerance):
        """BENCHMARK: Repetitive patterns typically yield significantly higher Ω."""
        ap = AperspectiveInvariance(transforms=transforms_full)

        structured = "ABCD " * 20
        low_structure = "QWERTYUIOPASDFGHJKLZXCVBNM"

        omega_structured = ap.measure(structured).omega_score
        omega_low = ap.measure(low_structure).omega_score

        gap = omega_structured - omega_low
        assert gap > tolerance["structural"], f"Structural gap below expected: {gap}"

    def test_empty_input_handling(self, transforms_full):
        """Empty or whitespace-only inputs should not crash."""
        ap = AperspectiveInvariance(transforms=transforms_full)

        result_empty = ap.measure("")
        result_ws = ap.measure("   \n\t  ")

        assert isinstance(result_empty.omega_score, (int, float))
        assert isinstance(result_ws.omega_score, (int, float))
        assert 0.0 <= result_empty.omega_score <= 1.0
        assert 0.0 <= result_ws.omega_score <= 1.0