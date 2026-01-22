"""Test thermodynamic properties of IRI (Irreversibility Index)."""

from omnia.iri import IRI


class TestIRIThermodynamics:
    """Verify IRI satisfies second-law-like constraints."""

    def test_iri_always_nonnegative(self):
        iri = IRI()
        test_cases = [
            (0.9, 0.9),
            (0.8, 0.5),
            (0.5, 0.8),
            (1.0, 0.0),
            (0.0, 1.0),
            (0.7, 0.3),
        ]
        for omega_A, omega_A_prime in test_cases:
            iri_value = iri.value(omega_A, omega_A_prime)
            assert iri_value >= 0.0

    def test_iri_zero_when_no_loss(self):
        iri = IRI()
        assert iri.value(0.5, 0.5) == 0.0
        assert iri.value(0.5, 0.7) == 0.0
        assert iri.value(0.3, 1.0) == 0.0

    def test_iri_positive_when_omega_decreases(self):
        iri = IRI()
        assert iri.value(0.8, 0.5) > 0.0
        assert iri.value(1.0, 0.0) > 0.0
        assert iri.value(0.6, 0.2) > 0.0

    def test_iri_monotonic_with_loss(self):
        iri = IRI()
        iri_small = iri.value(0.8, 0.7)
        iri_large = iri.value(0.8, 0.2)
        assert iri_large >= iri_small

    def test_iri_boundary_conditions(self):
        iri = IRI()
        iri_max_loss = iri.value(1.0, 0.0)
        assert iri_max_loss > 0.0
        assert iri.value(0.0, 1.0) == 0.0
        assert iri.value(0.0, 0.0) == 0.0
        assert iri.value(1.0, 1.0) == 0.0