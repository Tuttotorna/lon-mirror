import math
import pytest

from omnia.metrics import compute_metrics


def test_single_base_is_neutral_stable():
    m = compute_metrics({10: [1.0, 2.0, 3.0]})
    assert m.truth_omega == 1.0
    assert m.delta_coherence == 0.0
    assert m.kappa_alignment == 1.0
    assert m.epsilon_drift == 0.0


def test_identical_vectors_across_bases_gives_max_truth():
    sigs = {
        2: [1.0, 2.0, 3.0, 4.0],
        3: [1.0, 2.0, 3.0, 4.0],
        10: [1.0, 2.0, 3.0, 4.0],
    }
    m = compute_metrics(sigs)
    assert math.isclose(m.delta_coherence, 0.0, abs_tol=1e-12)
    assert math.isclose(m.epsilon_drift, 0.0, abs_tol=1e-12)
    assert math.isclose(m.truth_omega, 1.0, abs_tol=1e-12)
    assert 0.99 <= m.kappa_alignment <= 1.0


def test_length_mismatch_raises():
    sigs = {
        2: [1.0, 2.0, 3.0],
        10: [1.0, 2.0],
    }
    with pytest.raises(ValueError):
        compute_metrics(sigs)


def test_increasing_deformation_reduces_truthomega():
    sigs_low = {
        2: [1.0, 1.0, 1.0],
        10: [1.05, 0.95, 1.0],
    }
    sigs_high = {
        2: [1.0, 1.0, 1.0],
        10: [3.0, -2.0, 10.0],
    }

    m_low = compute_metrics(sigs_low)
    m_high = compute_metrics(sigs_high)

    assert m_high.delta_coherence > m_low.delta_coherence
    assert m_high.epsilon_drift > m_low.epsilon_drift
    assert m_high.truth_omega < m_low.truth_omega


def test_kappa_alignment_penalizes_dimension_variance():
    # Same mean magnitude, but base-10 diverges on one dimension
    sigs_stable = {
        2: [10.0, 10.0, 10.0],
        10: [10.0, 10.0, 10.0],
        16: [10.0, 10.0, 10.0],
    }
    sigs_unstable = {
        2: [10.0, 10.0, 10.0],
        10: [10.0, 999.0, 10.0],
        16: [10.0, 10.0, 10.0],
    }

    m_stable = compute_metrics(sigs_stable)
    m_unstable = compute_metrics(sigs_unstable)

    assert m_unstable.kappa_alignment < m_stable.kappa_alignment