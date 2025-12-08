# OMNIA_TOTALE v2.0 — Unified Ω-fusion Engine

Author: Massimiliano Brighindi (MB-X.01 / Omniabase±)  
Engine formalization: MBX IA

---

## 1. Goal

OMNIA_TOTALE v2.0 provides a **single Ω-score pipeline** that fuses:

- **BASE**: multi-base structure + PBII instability (Omniabase).
- **TIME**: regime-change detection on sequences (Omniatempo).
- **CAUSA**: lagged correlations between channels (Omniacausa).
- **TOKEN**: token-level Ω-map for LLM text (PBII → z-scores).

The engine is designed as a **model-agnostic guardrail**: given numbers, time-series and token traces, it returns raw and revised Ω plus JSON logs suitable for xAI integration.

---

## 2. Lenses

### 2.1 Omniabase / PBII (BASE)

- `digits_in_base_np`, `normalized_entropy_base`, `sigma_b` implement multi-base digit entropy.
- `omniabase_signature(n, bases)` returns:
  - per-base σ scores and entropies,
  - mean σ / mean entropy across bases.
- `pbii_index(n, composite_window, bases)`:
  - saturation = mean σ on a fixed composite window,
  - PBII = saturation − σ(n),
  - higher PBII ≈ more prime-like instability.

### 2.2 Omniatempo (TIME)

`omniatempo_analyze(series)`:

- global mean/std,
- short/long-window mean/std,
- symmetric KL-like divergence between short vs long histograms.

The **regime_change_score** is passed to fusion as `log(1 + score)`.

### 2.3 Omniacausa (CAUSA)

`omniacausa_analyze(series_dict, max_lag, strength_threshold)`:

- for each pair of channels, scans lags in `[-max_lag, +max_lag]`,
- keeps the strongest Pearson correlation,
- emits an edge if `|corr| ≥ strength_threshold`.

The fusion component is the **mean |corr|** over accepted edges.

### 2.4 Token-level Ω map (TOKEN)

`token_level_omega_map(tokens, token_numbers)`:

- maps each token to an integer proxy (`token_numbers`),
- computes PBII per token,
- converts PBII to z-scores over the sequence.

The TOKEN component is the **mean |z|** (instability along the text).

---

## 3. Fusion and thresholds

### 3.1 Inputs

```python
@dataclass
class OmniaInput:
    n: int                       # target integer (e.g. answer or key state)
    series: np.ndarray           # main time series
    series_dict: Dict[str, np.ndarray]  # multi-channel series
    tokens: Optional[List[str]] = None
    token_numbers: Optional[List[int]] = None