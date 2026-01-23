"""
Tests for the Sovereign Kernel.
"""
import pytest
import numpy as np
from omnia.sovereign import SovereignKernel, UNIVERSAL_STABILITY_LIMIT, MEMORY_SATURATION_THRESHOLD

def test_sovereign_initialization():
    kernel = SovereignKernel()
    assert kernel.state.memory_saturation == 0.5
    assert kernel.state.entropy == 0.0

def test_entropy_measurement():
    kernel = SovereignKernel()

    # Low entropy context (flat array)
    context_low = [1.0, 1.0, 1.0]
    e_low = kernel.measure_entropy(context_low)
    assert e_low == 0.0

    # High entropy context (random noise)
    context_high = [1.0, 10.0, -5.0, 100.0]
    e_high = kernel.measure_entropy(context_high)
    assert e_high > 10.0

def test_governance_stability_gate():
    kernel = SovereignKernel()

    # Create a context with extremely high variance to trigger Halt
    # Standard deviation of [0, 10] is 5.0 > 2.14
    context_chaos = [0.0, 10.0]

    result = kernel.govern(context_chaos, intent="TEST_CHAOS")

    assert result.decision == "HALT"
    assert "Universal Stability Limit Breached" in result.note
    assert f"$E" in result.s_lang_trace

def test_governance_memory_gate():
    kernel = SovereignKernel()

    # Context with entropy > 1.0 but < 2.14 (Stable but Complex)
    # std of [0, 3] is 1.5. 1.5 > 1.0.
    context_complex = [0.0, 3.0]

    # Default memory is 0.5 < 0.90
    result = kernel.govern(context_complex, intent="DECIDE_PARADOX")

    assert result.decision == "WAIT"
    assert "$Paradox" in result.s_lang_trace
    assert "Memory not saturated" in result.note

def test_governance_proceed():
    kernel = SovereignKernel()

    # Simple context
    context_simple = [1.0, 1.1]
    result = kernel.govern(context_simple, intent="ANALYSIS")

    assert result.decision == "PROCEED"
    assert "Gates_Passed" in result.s_lang_trace

def test_four_gate_verification():
    kernel = SovereignKernel()

    assert kernel.four_gate_verification("Sky is blue", evidence="Spectroscopy") == True
    assert kernel.four_gate_verification("Sky is green", evidence=None) == False
