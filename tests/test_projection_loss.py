"""Test SPL (Structural Projection Loss) meta-measurement."""

import re
from omnia.meta.measurement_projection_loss import MeasurementProjectionLoss


class TestProjectionLoss:
    """Verify SPL properties under controlled measurers."""

    def test_spl_nonnegative_by_construction(self):
        def omega_len(x: str) -> float:
            s = x.strip()
            return min(1.0, len(s) / 100.0)

        def project_nums(x: str) -> str:
            return re.sub(r"[^\d]+", "", x)

        def omega_ap(x: str) -> float:
            return omega_len(x)

        def omega_proj(x: str) -> float:
            return omega_len(project_nums(x))

        spl = MeasurementProjectionLoss(
            aperspective_measurers=[("len", omega_ap)],
            projected_measurers=[("len_on_nums", omega_proj)],
            aggregator="mean",
        )

        for text in ["Hello world 123", "No numbers here", "987654321", "Mixed abc123xyz789"]:
            result = spl.measure(text)
            assert result.spl_abs >= -1e-12

    def test_spl_zero_when_lossless(self):
        def omega_identity(x):
            return min(1.0, len(x.strip()) / 50.0)

        spl = MeasurementProjectionLoss(
            aperspective_measurers=[("id1", omega_identity)],
            projected_measurers=[("id2", omega_identity)],
            aggregator="mean",
        )

        result = spl.measure("Test string")
        assert abs(result.spl_abs) < 1e-9

    def test_spl_detects_information_loss(self):
        def omega_full(x):
            return 0.8

        def omega_destructive(x):
            return 0.2

        spl = MeasurementProjectionLoss(
            aperspective_measurers=[("full", omega_full)],
            projected_measurers=[("dest", omega_destructive)],
            aggregator="mean",
        )

        result = spl.measure("Any text")
        assert result.spl_abs > 0.5

    def test_trimmed_mean_aggregator(self):
        def omega_normal_1(x): return 0.5
        def omega_normal_2(x): return 0.6
        def omega_outlier(x): return 0.0
        def omega_proj(x): return 0.4

        spl = MeasurementProjectionLoss(
            aperspective_measurers=[("n1", omega_normal_1), ("n2", omega_normal_2), ("out", omega_outlier)],
            projected_measurers=[("p", omega_proj)],
            aggregator="trimmed_mean",
            trim_q=0.3,
        )

        result = spl.measure("x")
        assert 0.5 <= result.omega_aperspective <= 0.6