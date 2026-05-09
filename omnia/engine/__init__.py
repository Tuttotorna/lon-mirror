"""
Compatibility namespace for omnia.engine.

This package exposes PrimeKernel through:

    from omnia.engine.prime_kernel import PrimeKernel

Boundary:
    measurement != inference != decision

This module is not a truth oracle.
This module is not a semantic judge.
This module is not a decision engine.
Decision remains external.
"""

from __future__ import annotations

from .prime_kernel import PrimeKernel

__all__ = [
    "PrimeKernel",
]