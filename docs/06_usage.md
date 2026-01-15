# OMNIA — Usage

This document describes the **canonical usage patterns** of OMNIA.
All examples are structural, deterministic, and policy-free.

---

## General Principle

OMNIA is used as a **post-inference structural sensor**.

Input:
- numbers
- sequences
- time series
- token streams
- model outputs

Output:
- structural signals
- Ω-based profiles
- flags and metrics

OMNIA never alters the input and never feeds back into generation.

---

## 1. Omniabase (BASE)

Analyze numeric structure across multiple bases.

```python
from omnia import omniabase_signature, pbii_index

sig = omniabase_signature(173)
pbii = pbii_index(173)

print(sig)
print(pbii)