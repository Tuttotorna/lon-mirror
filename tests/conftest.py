"""Shared fixtures and utilities for LON-MIRROR tests."""

import pytest


@pytest.fixture
def sample_texts():
    """Standard test texts with known structural properties.

    DETERMINISTIC: No random generation, fixed strings only.
    """
    return {
        "structured": "ABC ABC ABC ABC",
        "low_structure": "QWERTYUIOPASDFG",
        "numeric": "2026 2025 2024 12345 67890",
        "mixed": "The sun does not erase stars; it saturates your detector.",
        "empty": "",
        "whitespace": "   \n\t   ",
        "repeated": "A" * 100,
        "high_compressibility": "ABABABABABABABABABABAB",
    }


@pytest.fixture
def transforms_whitespace_only():
    """Only whitespace-invariant transforms for isolation testing."""
    from omnia.lenses.aperspective_invariance import t_identity, t_whitespace_collapse

    return [
        ("id", t_identity),
        ("ws", t_whitespace_collapse),
    ]


@pytest.fixture
def transforms_full():
    """Full aperspective transform set for integration testing."""
    from omnia.lenses.aperspective_invariance import (
        t_identity,
        t_whitespace_collapse,
        t_reverse,
        t_drop_vowels,
    )

    return [
        ("id", t_identity),
        ("ws", t_whitespace_collapse),
        ("rev", t_reverse),
        ("vow-", t_drop_vowels),
    ]


@pytest.fixture
def tolerance():
    """Numerical tolerance for floating point comparisons."""
    return {
        "strict": 1e-10,
        "normal": 1e-6,
        "loose": 1e-3,
        "structural": 0.15,
    }