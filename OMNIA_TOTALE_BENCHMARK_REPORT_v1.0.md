# OMNIA_TOTALE · Benchmark Report v1.0

Author: **Massimiliano Brighindi** (concepts) + **MBX IA** (formalization)  
Code: `OMNIA_TOTALE_BENCHMARK_v1.0.py`  
Core engine: `OMNIA_TOTALE_v0.6.py`  

This report documents the current benchmark protocols and results for the OMNIA_TOTALE framework:

- PBII-based structural lens on integers (Omniabase).  
- Temporal regime-change lens (Omniatempo).  
- Causal lag lens (Omniacausa).  
- Fused Ω-score used for evaluation and anomaly detection.

The goals are:

1. Measure how well PBII separates primes from composites.  
2. Evaluate PBII as a detector of unstable / hallucinated reasoning chains on GSM8K-style math problems.  
3. Provide an optional demo of Ω-score correlation with task difficulty / error.

All numbers are produced by running `OMNIA_TOTALE_BENCHMARK_v1.0.py`.  
No metrics are hard-coded in the code; they are computed from data at evaluation time.

---

## 1. Setup

### 1.1. Environment

Minimal dependencies:

```bash
pip install numpy matplotlib
# optional: pip install datasets   # if a lab wants to auto-load GSM8K

Files:

OMNIA_TOTALE_v0.6.py – core Ω engine (BASE/TIME/CAUSA fusion).

OMNIA_TOTALE_BENCHMARK_v1.0.py – benchmark runner.

results/OMNIA_TOTALE_BENCHMARK_v1.0.json – metrics produced by a run.

figures/pbii_primes_vs_composites.png – PBII distribution plot.

Optional: gsm8k_generations.jsonl – real model generations on GSM8K.


1.2. Optional GSM8K generations format

For real hallucination evaluation the script expects a JSONL file:

{"id": "problem-id",
 "gold_answer": "...",
 "model_chain": "full chain-of-thought solution text",
 "is_hallucinated": true}

is_hallucinated = true if the chain contains a factual/numeric error.

The benchmark code does not decide labels; it uses the labels provided.


Path is configurable via environment variable:

export OMNIA_GSM8K_FILE=gsm8k_generations.jsonl

If the file is missing, only the synthetic benchmark is run.


---

2. PBII: Prime vs Composite AUC

2.1. Protocol

1. Sample N integers uniformly in [low, high] (default: N=500, 2..10,000).


2. Label them: 1 for prime, 0 for composite (deterministic primality check).


3. For each integer n, compute PBII(n) using the local implementation aligned with the OMNIA_TOTALE engine.


4. Convert to a prime score via score = -PBII(n) so that higher scores indicate “more prime-like”.


5. Compute ROC AUC using a numpy-only implementation (no sklearn).


6. Plot PBII distributions for primes vs composites into figures/pbii_primes_vs_composites.png.



2.2. Metrics

After running the benchmark script, the JSON file contains:

"prime_auc": {
  "auc": ...,
  "num_samples": ...,
  "primes": ...,
  "composites": ...
}

These values are entirely data-driven.
The AUC value gives a direct measure of how strongly PBII separates primes from composites.


---

3. PBII: Hallucination Detection on Chains

3.1. Heuristic detector

Given a reasoning chain (plain text), the detector:

1. Extracts all positive integers from the chain.


2. Computes PBII(n) for each integer.


3. Averages them into avg_pbii.


4. Flags the chain as “unstable” if avg_pbii > τ, with default threshold τ = 0.10.



High average PBII means the numbers in the chain are structurally more prime-like / unstable across bases, which is used as a proxy for potential hallucinations in long arithmetic chains.

3.2. Synthetic GSM8K-like benchmark

The script ships with two small sets:

SYNTH_CORRECT_CHAINS: simple, numerically consistent GSM8K-style solutions.

SYNTH_HALLUCINATED_CHAINS: the same problems with perturbed/incorrect numbers.


For these, the benchmark reports:

"hallucination_synthetic": {
  "threshold": 0.1,
  "false_positive_rate": ...,
  "true_positive_rate": ...,
  "num_correct": 5,
  "num_hallucinated": 5
}

false_positive_rate: fraction of correct chains incorrectly flagged as hallucinated.

true_positive_rate: fraction of hallucinated chains correctly flagged.


These numbers are small-scale but fully reproducible and serve as a sanity check of the PBII-based detector.

3.3. Real GSM8K evaluation (optional, for labs)

If gsm8k_generations.jsonl is provided, the script runs the same detector on real model outputs:

"hallucination_gsm8k_real": {
  "threshold": 0.1,
  "false_positive_rate": ...,
  "true_positive_rate": ...,
  "num_correct": ...,
  "num_hallucinated": ...
}

This is where verified hallucination metrics live.
Any lab can plug in their own GSM8K generations, label them, run the script, and get concrete numbers without changing the code.


---

4. Ω-score correlation demo (optional)

4.1. Protocol

If OMNIA_TOTALE_v0.6.py is present and importable, the benchmark script runs:

1. Build an “easy” time series: pure sinusoid (stable regime).


2. Build a “hard” time series: sinusoid + regime shift + noise.


3. Construct basic causal graphs for each (three series with lags).


4. For each regime, evaluate Ω for repeated (n, series, series_dict) triples using omnia_totale_score.


5. Label each sample as 0 = easy or 1 = hard.


6. Compute Pearson correlation between Ω and the difficulty label.



Output:

"omega_correlation_demo": {
  "pearson_r": ...,
  "num_points": ...
}

This is not a task benchmark, but a structural sanity check: Ω should correlate with regime complexity and causal instability.

If the core engine is not found, this section is skipped.


---

5. Running the full benchmark

python OMNIA_TOTALE_BENCHMARK_v1.0.py

Outputs:

results/OMNIA_TOTALE_BENCHMARK_v1.0.json

figures/pbii_primes_vs_composites.png


If gsm8k_generations.jsonl is present, real hallucination metrics are included.
Otherwise only the synthetic GSM8K-like benchmark is reported.


---

6. Notes on placeholders vs verified metrics

The code is fully executable and self-contained; it does not hard-code metrics.

Synthetic benchmarks (built-in chains, random primes/composites, Ω demo) are reproducible but limited in scope.

Verified hallucination reduction on GSM8K requires a real gsm8k_generations.jsonl file with model outputs and labels.

Labs (including xAI) can plug in their own generations and re-run the script to obtain full, non-placeholder metrics without touching the logic.



This makes OMNIA_TOTALE’s evaluation:

Transparent (all formulas and thresholds in open Python).

Reproducible (same script, different data).

Extensible (labs can swap in their own data and attach additional tasks while keeping PBII / Ω lenses intact).