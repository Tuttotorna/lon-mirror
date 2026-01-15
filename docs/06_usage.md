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

Typical uses

numeric anomaly detection

prime-like structure analysis

invariant profiling



---

2. Omniatempo (TIME)

Analyze temporal stability and drift.

from omnia import omniatempo_analyze

res = omniatempo_analyze(series)

print(res.regime_score)
print(res.statistics)

Typical uses

reasoning drift detection

time series regime shifts

stability analysis over long outputs



---

3. Omniacausa (CAUSA)

Extract lagged causal structure from multivariate signals.

from omnia import omniacausa_analyze

res = omniacausa_analyze({
    "s1": s1,
    "s2": s2,
    "s3": s3,
})

print(res.edges)

Typical uses

dependency discovery

cross-channel structure mapping

chain-of-thought segmentation



---

4. Token Lens (TOKEN)

Apply numeric structural analysis to token streams.

from omnia import omniatoken_analyze

res = omniatoken_analyze(token_ids)

print(res.z_scores)
print(res.instability_segments)

Typical uses

hallucination detection

unstable region localization

long-chain fracture analysis



---

5. Ω Fusion (Ω-TOTAL)

Fuse all lens outputs into a unified structural profile.

from omnia import omnia_totale_score

res = omnia_totale_score(
    n=173,
    series=series,
    series_dict={"s1": s1, "s2": s2},
    tokens=token_ids,
)

print(res.omega_score)
print(res.components)

Ω is a structural residue, not a truth value.


---

6. LLM Output Analysis (Raw Integration)

OMNIA can be applied directly to raw LLM outputs.

from adapters.llm_output_adapter import analyze_llm_output

report = analyze_llm_output(
    text=llm_output,
    tokens=token_ids,
)

print(report.omega_score)
print(report.flags)

No framework integration is required. No policy logic is included.


---

7. Interpreting Results

OMNIA outputs must be interpreted carefully:

low Ω → stable structure

high Ω → structural instability

drift signals → sensitivity to perturbation

saturation → analysis should stop


Interpretation and decisions always remain external.


---

8. Anti-Patterns (Do Not Do This)

Do not:

treat Ω as correctness

threshold Ω to accept/reject answers

feed Ω back into generation

replace reasoning with Ω


These violate the OMNIA boundary.


---

Summary

OMNIA is a structural sensor

Usage is modular and deterministic

Outputs are signals, not decisions

Boundaries must be respected