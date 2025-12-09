# OMNIA_TOTALE_EVAL_REPORT v1.0  
Benchmark protocol for PBII + TruthΩ

Author: Massimiliano Brighindi (concepts) + MBX IA (formalization)  
Repo: https://github.com/Tuttotorna/lon-mirror  
Module: `OMNIA_TOTALE` (v0.6+)

---

## 1. Purpose

This document specifies how to evaluate the OMNIA_TOTALE family of lenses:

- **PBII / Omniabase** – multi-base numeric instability
- **Omniatempo** – regime change in time series
- **Omniacausa** – lagged causal edges
- **Ω-score** – fused structural coherence metric
- **TOKENMAP / SELFREV** – token-level Ω-maps and Ω-driven self-revision

Goal: move from **claims / placeholders** (e.g. “71% hallucination reduction”) to **verifiable metrics** that any lab can reproduce on its own infrastructure.

This report is intentionally explicit about what is:

- already implemented and reproducible on synthetic data, and
- still **placeholder** and requires real-world runs (e.g. full GSM8K).

---

## 2. Current status (honest summary)

1. The repo already contains:
   - `OMNIA_TOTALE_v0.6.py` – fused Ω engine (BASE/TIME/CAUSA + token z-scores).
   - `OMNIA_TOTALE_TOKENMAP_v0.1.py` – token-level Ω-maps.
   - `OMNIA_TOTALE_SELFREV_v0.1.py` – Ω-driven self-revision loop for LLMs.
   - `gsm8k_benchmark_demo.py` – **synthetic** demo for hallucination detection and AUC on primes vs composites.

2. The demo script shows:
   - Example **hallucination detection** on hand-crafted GSM8K-style chains.
   - Example **AUC** for prime vs composite separation using PBII.
   - Plots of PBII distributions.

3. Numbers like **“71% hallucination reduction on long chains”** are currently:
   - **Placeholders** based on small synthetic samples.
   - Not yet tied to a full GSM8K run or a real training/eval pipeline.

4. This report defines the protocol to:
   - Replace placeholders with real metrics.
   - Make it easy for xAI (or any lab) to plug OMNIA_TOTALE into their evaluation stack.

---

## 3. Evaluation artifacts

Planned evaluation files in the repo:

- `OMNIA_TOTALE_EVAL_REPORT_v1.0.md`  
  This document.

- `OMNIA_TOTALE_EVAL_v1.0.py`  
  Single script with three benchmark blocks:
  - **M1** – hallucination detection on long-chain GSM8K-like reasoning.
  - **M2** – numeric separation: primes vs composites via PBII.
  - **M3** – token-level Ω-map stability vs reasoning errors.

Optionally:

- `benchmarks/` folder with:
  - generated plots (PNG),
  - small JSON logs with metric summaries.

---

## 4. Metrics

### 4.1 M1 – Hallucination reduction on long-chain reasoning

**Objective**

Quantify how well PBII / Ω can flag **unstable chains** (hallucinated reasoning) before final answer emission.

**Setting**

- Task family: GSM8K-style math word problems or similar chain-of-thought datasets.
- Chains: only runs with **length ≥ 50 steps** (tokens or reasoning units, depending on internal logging granularity).
- Baseline: vanilla model without Ω-based filtering.
- Variant: same model with an Ω-based “instability gate” that can:
  - trigger self-revision, or
  - flag the chain as unreliable.

**Metric definitions**

On a test set of N problems with reference solutions:

- Let:
  - `FP_base` = hallucinations in baseline system,
  - `FP_omega` = hallucinations in Ω-gated system.

- Define:
  - **Hallucination reduction**  

    \[
    HR = \frac{FP_{\text{base}} - FP_{\text{omega}}}{FP_{\text{base}}}
    \]

  - **Coverage** (how often the Ω-gate triggers):

    \[
    C = \frac{\text{# chains where Ω triggers}}{N}
    \]

- Output:
  - `HR` in `[0, 1]` (e.g. 0.71 = 71% reduction),
  - `C` in `[0, 1]`,
  - confusion matrix: TP/FP/FN/TN for hallucination flags.

**Placeholder vs real**

- Current demo (`gsm8k_benchmark_demo.py`) uses:
  - a few hand-crafted “correct” vs “hallucinated” chains,
  - **hard-coded threshold** on PBII averages.
- Numbers from this demo are **not** to be cited as final results.
- Real evaluation requires:
  - full GSM8K (or comparable) run,
  - internal logging of chain correctness,
  - application of the above formula.

---

### 4.2 M2 – Prime vs composite separation (PBII AUC)

**Objective**

Check whether PBII, as a pure numeric lens, detects the structural instability of primes vs composites.

**Setting**

- Sample size: at least `N = 10 000` integers, uniform in a chosen range (e.g. `[2, 100 000]`), or stratified if preferred.
- Label:
  - `y = 1` for prime,
  - `y = 0` for composite.
- Score:
  - `s(n) = -PBII(n)` so that **higher score ⇒ more prime-like**.

**Metric**

- Compute **ROC AUC**:

  \[
  \text{AUC} = \Pr(s(n_\text{prime}) > s(n_\text{comp}))
  \]

- Implementation:
  - any standard library (`sklearn.metrics.roc_auc_score`) or the simple NumPy implementation already in the demo.

**Placeholder vs real**

- Demo currently uses:
  - `N = 100` numbers,
  - quick PBII implementation.
- This already yields high AUC in synthetic tests (≈0.98), but:
  - should be re-run with larger `N`,
  - seed and range documented in the script header.

---

### 4.3 M3 – Token-level Ω-maps vs reasoning errors

**Objective**

Link token-level Ω dynamics to **actual reasoning mistakes** inside LLM chains.

**Setting**

- Models: any LLM that can expose:
  - token stream,
  - intermediate logprobs, or
  - at least the textual chain of thought.
- For each chain:
  - compute Ω per token (or per reasoning step) using `OMNIA_TOTALE_TOKENMAP_v0.1`.
  - get ground-truth labels of **local errors**:
    - arithmetic mistakes,
    - logical contradictions,
    - unsupported jumps.

**Metrics**

1. **ΔΩ correlation with errors**

   - For each token / step t:
     - compute `ΔΩ_t = Ω_t - Ω_{t-1}`.
   - Label `e_t = 1` if a known error starts at or near t, else `0`.
   - Compute:
     - point-biserial correlation between `|ΔΩ_t|` and `e_t`,
     - or ROC AUC of `|ΔΩ_t|` as a “error score”.

2. **Peak-hit rate**

   - Define “peaks”: tokens with `|ΔΩ_t|` above a percentile (e.g. 95th).
   - Check how often peaks co-locate with error spans.
   - Metric:

     \[
     \text{HitRate} = \frac{\text{# error spans hit by at least one Ω peak}}{\text{# error spans total}}
     \]

**Placeholder vs real**

- Current token-map code is tested only on **synthetic** sequences and toy examples.
- Real evaluation requires:
  - instrumented runs on a labeled reasoning dataset,
  - annotation of internal mistakes or automatic extraction where possible.

---

## 5. Script structure: `OMNIA_TOTALE_EVAL_v1.0.py`

The evaluation script is intended to be:

- **Standalone**, importing only:
  - `numpy`,
  - `matplotlib`,
  - the OMNIA_TOTALE modules (`PBII`, Ω),
- **Modular**, with three main entry points:

```bash
# M1 – hallucination reduction (synthetic or real GSM8K hook)
python OMNIA_TOTALE_EVAL_v1.0.py --benchmark M1

# M2 – prime vs composite AUC
python OMNIA_TOTALE_EVAL_v1.0.py --benchmark M2

# M3 – token-level Ω vs errors (requires external logs)
python OMNIA_TOTALE_EVAL_v1.0.py --benchmark M3

Each benchmark should:

1. Print metric summary to stdout (HR, AUC, correlations, etc.).


2. Save plots under benchmarks/ (e.g. pbii_auc_primes.png, omega_delta_error_corr.png).


3. Optionally dump a small JSON with configuration + metrics.




---

6. Integration points for external labs (e.g. xAI)

To adapt OMNIA_TOTALE to an internal evaluation pipeline, the lab only needs to:

1. Numeric lens (PBII)

Import PBII from OMNIA_TOTALE.

Apply it to:

numeric outputs,

intermediate numeric states,

or hashes / IDs representing states.




2. Chain-of-thought lens (Ω over time)

Log chains (tokens, steps, or both).

Feed them to OMNIA_TOTALE as a time series.

Use M1 protocol to measure hallucination reduction.



3. Token-level Ω-maps

For each generated answer:

record token stream,

run TOKENMAP to compute Ω_raw, Ω_revised, and ΔΩ.


Use M3 protocol to relate Ω peaks to errors.



4. Self-revision

Wrap the model with SELFREV:

if Ω exceeds a threshold, trigger another pass or ask for self-correction.


Compare error rates with and without this wrapper.




All of the above can be done without modifying model weights, using OMNIA_TOTALE as an external, structural auditing layer.


---

7. Roadmap

Short-term:

Finalize OMNIA_TOTALE_EVAL_v1.0.py with:

clean CLI,

clear separation between synthetic demos and “real hooks”.


Replace all hard-coded placeholder numbers in README / comments with:

“synthetic demo only” labels, and

pointers to this evaluation report.



Mid-term:

Run full GSM8K (or similar) evaluations in a controlled environment.

Publish:

real values for M1 / M2 / M3,

plots and logs,

a concise public summary (1–2 pages).



Long-term:

Extend Ω-based evaluation to:

RL / robotics trajectories (Omniatempo + Omniacausa),

multimodal models (Omniabase on pixels, Ω on embeddings),

long-horizon planning agents (stability of policy updates over time).




---

8. Status of claims

To avoid ambiguity:

Any exact percentage (e.g. “71% reduction”) in previous code/comments must be treated as a placeholder until:

confirmed by a full evaluation run following this protocol, and

backed by logs / plots.



The current repo provides:

structural tools (PBII, Ω),

integration stubs (TOKENMAP, SELFREV),

synthetic demos and visualization.


This report defines the path from these tools to verifiable metrics suitable for internal xAI evaluation or any independent replication.