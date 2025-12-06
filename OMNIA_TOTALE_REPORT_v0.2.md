# OMNIA_TOTALE v0.6 — Ω-FLOW ENGINE  
Author: Massimiliano Brighindi (concepts) + MBX IA (formalization)

This report summarizes the Ω-FLOW architecture implemented in `OMNIA_TOTALE_v0.6.py`, where three structural lenses — Omniabase, Omniatempo, and Omniacausa — are fused into a single coherence score with self-revision.

---

## 1. Goal

OMNIA_TOTALE v0.6 is not a model, but a **structural observer**:

- It takes:
  - an integer `n` (for multi-base structure),
  - a 1D time series `series` (for temporal behaviour),
  - a multivariate time series `series_dict` (for directional dependencies),
- And produces:
  - a fused **Ω_raw** score (structural coherence/instability),
  - a **self-revised Ω_revised** score using local token instabilities,
  - a full set of interpretable components.

The aim is to provide a plug-and-play engine that can sit around any reasoning process (LLM, solver, agent) and quantify how structurally “stable” or “drifting” the process is.

---

## 2. Lenses

### 2.1 Omniabase (multi-base structure)

For an integer `n` and a set of bases `B`:

- Convert `n` to each base `b ∈ B` and compute:
  - Normalized Shannon entropy `H_norm(n, b)` over digits,
  - Base Symmetry Score `σ_b(n)`:

\[
\sigma_b(n) = \text{length\_weight} \cdot \frac{1 - H_{\text{norm}}(n,b)}{L^{{\text{length\_exponent}}}} + \text{divisibility\_bonus} \cdot I[n \bmod b = 0]
\]

- Aggregate across bases:

\[
\Sigma_{\text{avg}}(n) = \mathbb{E}_{b \in B}[\sigma_b(n)]
\]

- PBII-like instability index:

\[
\text{PBII}(n) = \mathbb{E}_{c \in \text{composites}}[\Sigma_{\text{avg}}(c)] - \Sigma_{\text{avg}}(n)
\]

Higher PBII means “more prime-like instability” in the multi-base space.

### 2.2 Omniatempo (temporal structure)

Given a time series `series`:

- Compute global mean and std.
- Extract a **short** and **long** recent window.
- Build histograms for both and compute a symmetric KL-like divergence:

\[
\text{regime\_change} = \frac{1}{2}\left( KL(p \parallel q) + KL(q \parallel p) \right)
\]

- Then define:

\[
\text{tempo\_component} = \log(1 + \text{regime\_change})
\]

This captures how much the recent behaviour deviates from longer-term statistics.

### 2.3 Omniacausa (directional dependencies)

For a multivariate time series `series_dict: name → values`:

- For each pair `(src, tgt)` and lag `ℓ` in `[−L, …, +L]`, compute lagged Pearson correlation.
- Select the lag with maximum |corr|.
- If `|corr| ≥ threshold`, add an edge:

\[
\text{src} \rightarrow \text{tgt} \quad \text{with lag } \ell, \text{ strength } \rho
\]

- The **causal component** is:

\[
\text{causa\_mean\_strength} = \mathbb{E}[|\rho| \text{ over all edges}]
\]

It is a heuristic lens, not a full causal discovery algorithm, but provides a compact measure of directional structure.

---

## 3. Token-map (local instability)

On any 1D sequence `series` (e.g. scores, losses, logprobs), we compute a rolling z-score:

\[
z_t = \frac{|x_t - \mu_t|}{\sigma_t + \epsilon}
\]

where `μ_t, σ_t` are computed on a short trailing window.

The **token component** is:

\[
\text{token\_mean\_instability} = \mathbb{E}_t[z_t]
\]

This is a local anomaly measure: high when many steps deviate strongly from their local context.

---

## 4. Ω-FLOW: fused score + self-revision

Given all components for a run:

- Base instability (PBII-like): `base_instability`
- Temporal regime shift: `tempo_log_regime`
- Causal structure: `causa_mean_strength`
- Token instability: `token_mean_instability`

We define:

\[
\Omega_{\text{raw}} = w_{\text{base}} \cdot \text{base\_instability} +
                      w_{\text{tempo}} \cdot \text{tempo\_log\_regime} +
                      w_{\text{causa}} \cdot \text{causa\_mean\_strength}
\]

\[
\Omega_{\text{revised}} = \Omega_{\text{raw}} - w_{\text{token}} \cdot \text{token\_mean\_instability}
\]

\[
\Delta \Omega = \Omega_{\text{revised}} - \Omega_{\text{raw}}
\]

Interpretation:

- Ω_raw: how structurally “coherent/unstable” the object looks in multi-base, temporal, and causal lenses.
- Ω_revised: same, but penalized by local anomalies detected by the token-map.
- ΔΩ: how much the self-revision step corrects the original score (negative ΔΩ = downgrade).

---

## 5. Demo setup

The built-in `run_demo()` does:

- `n_prime = 173` (prime), `n_comp = 180` (composite),
- A sinusoidal time series with a regime shift at t=200,
- A toy causal system:
  - `s1`: sine wave,
  - `s2`: lagged copy of `s1` + noise,
  - `s3`: pure noise.

For each `n`:

- Runs `omnia_flow(n, series, series_dict)`,
- Prints Ω_raw, Ω_revised, ΔΩ and components,
- Prints all detected causal edges.

---

## 6. Outputs (example pattern)

Typical pattern observed in tests:

- Primes:  
  - Higher **base_instability** (PBII-like),  
  - Similar temporal and causal components (same series),  
  - Token penalty depends only on the shared series.

- Composites:  
  - Lower base_instability,  
  - Same temporal/causal components,  
  - Same token penalty.

So Ω_raw differs mainly by the base lens, while Ω_revised shows how much the token-map can “correct” the score based on local anomalies.

---

## 7. Intended usage

This Ω-FLOW engine is designed to be:

- **Model-agnostic**: can be wrapped around any system that exposes:
  - an integer identifier (or hash),
  - a 1D signal over time (loss, confidence, score),
  - a multivariate time log (internal features or traces).
- **Interpretable**: each component is inspectable and computable in a few lines.
- **Extensible**: bases, windows, lags, and weights are tunable; additional lenses can be added.

It is a concrete step toward using Omniabase + Omniatempo + Omniacausa as a unified structural observer for AI coherence and drift detection.