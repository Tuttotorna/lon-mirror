"""Test OmegaSet robust statistics."""

import math
from omnia.omega_set import OmegaSet


class TestOmegaSet:
    """Verify robust Ω-distribution estimation."""

    def test_median_robust_to_outliers(self):
        omega_clean = [0.5, 0.6, 0.55, 0.58, 0.52]
        omega_with_outlier = omega_clean + [0.0, 1.0]

        est_clean = OmegaSet(omega_clean).estimate()
        est_outlier = OmegaSet(omega_with_outlier).estimate()

        assert abs(est_clean["median"] - est_outlier["median"]) < 0.1

    def test_mad_measures_spread(self):
        tight = [0.50, 0.51, 0.49, 0.50, 0.51]
        spread = [0.1, 0.5, 0.9, 0.2, 0.8]

        mad_tight = OmegaSet(tight).estimate()["mad"]
        mad_spread = OmegaSet(spread).estimate()["mad"]

        assert mad_spread > mad_tight

    def test_invariance_score_finite_and_nonnegative(self):
        values = [0.3, 0.7, 0.5, 0.6, 0.4]
        est = OmegaSet(values).estimate()
        inv = est.get("invariance", est.get("inv"))

        if inv is not None:
            assert math.isfinite(inv)
            assert inv >= 0.0

    def test_single_value_degenerate_case(self):
        est = OmegaSet([0.7]).estimate()
        assert est["median"] == 0.7
        assert est["mad"] == 0.0