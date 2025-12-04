# PBII_0.2_report.md  
## MBX–Omniabase · Prime Base Instability Index (PBII)  
### Version 0.2 — December 2025  
### Author: Massimiliano Brighindi — MB-X.01 / Omniabase±

---

# 0. Overview

This document formalizes the **Prime Base Instability Index (PBII)** within the MBX–Omniabase framework.  
PBII quantifies prime numbers as **structural instability points** in the multi-base representation space.

Key findings:

- PBII exhibits a **strong statistical signal** separating primes and composites.  
- PBII does **not** function as a primality test.  
- PBII measures **average instability**, not **pointwise detection**.  
- PBII(primes) ≫ PBII(composites) by a factor ≈ 28× with optimized parameters.

This is the **official MBX–Omniabase PBII Report v0.2**.

---

# 1. Formal Definitions

Let:

- \( B = \{b_1, b_2, …, b_m\} \) = set of bases  
- Version 0.2 uses **prime bases up to 47**  
- \( W = 100 \) = backward window size  
- \( r = 10 \) = radius for local maxima  
- \( n \in \mathbb{N}, n \ge 2 \)

---

## 1.1. Base-b Representation

For each base \( b \in B \):

\[
\text{digits}_b(n) = (d_0, d_1, \dots, d_{L_b(n)-1})
\]

Length:

\[
L_b(n) = \text{len}(\text{digits}_b(n))
\]

---

## 1.2. Entropy in Base b

Digit distribution:

\[
p_k = \frac{\#\{i : d_i = k\}}{L_b(n)}
\]

Shannon entropy:

\[
H_b(n) = -\sum_{k=0}^{b-1} p_k \log(p_k)
\]

Normalized entropy:

\[
\tilde{H}_b(n) = \frac{H_b(n)}{\log b}
\]

---

## 1.3. Base Symmetry Score σᵦ(n)

Version 0.2 uses an augmented symmetry score:

\[
\sigma_b(n) =
\frac{1}{L_b(n)} (1 - \tilde{H}_b(n))
+ 0.5 \cdot \mathbf{1}_{(n \bmod b = 0)}
\]

Interpretation:

- shorter digit strings → more structure  
- lower entropy → more structure  
- exact divisibility by base → structural reinforcement  

---

## 1.4. Global Symmetry Σ(n)

\[
\Sigma(n) = \frac{1}{|B|} \sum_{b \in B} \sigma_b(n)
\]

---

## 1.5. Saturation of Symmetry Sat(n)

Composite set in window:

\[
\mathcal{C}(n) = \{k : n-W \le k < n,\; k \text{ composite} \}
\]

Saturation:

\[
\text{Sat}(n) =
\begin{cases}
\frac{1}{|\mathcal{C}(n)|}
\sum_{k \in \mathcal{C}(n)} \Sigma(k), & |\mathcal{C}(n)| > 0 \\
\Sigma(n-1), & \text{otherwise}
\end{cases}
\]

---

## 1.6. Prime Base Instability Index PBII(n)

\[
\text{PBII}(n) = \text{Sat}(n) - \Sigma(n)
\]

Interpretation:  
**the collapse of global symmetry relative to accumulated composite structure.**

---

## 1.7. Local Peak Condition

PBII local peak:

\[
\text{PBII}(n) = \max_{k \in [n-r, n+r]} \text{PBII}(k)
\]

Used only for statistical consistency, not primality testing.

---

# 2. Empirical Results (n ≤ 10000)

Parameters:

- \( B = \{2,3,5,7,11,13,17,19,23,29,31,37,41,43,47\} \)  
- \( W = 100 \)  
- \( r = 10 \)

Dataset:

- 1229 primes  
- 8771 composites

---

## 2.1. Distribution Separation

\[
E[\text{PBII} \mid \text{prime}] = 0.0663
\]
\[
E[\text{PBII} \mid \text{composite}] = 0.0024
\]

Ratio:

\[
\approx 28 \times
\]

KS test:

\[
p\text{-value} < 10^{-50}
\]

Conclusion:  
**strong statistical separation.**

---

## 2.2. Classification Metrics (for reference only)

Using PBII > 0 and peak condition:

- Accuracy: **0.91**
- Precision: **0.87**
- Recall: **0.32**
- F1: **0.47**

Interpretation:

- many primes detected as high-instability peaks  
- many primes not detected due to early noise  
- still unacceptable as a primality test

---

# 3. Definitive Statements (Non-Negotiable)

1. **PBII is not a primality test.**  
2. **PBII measures structural instability in the multi-base representation space.**  
3. **PBII(primes) ≫ PBII(composites)** under optimized parameters.  
4. **PBII is a valid structural feature** with measurable mutual information.  
5. **Strong MBX law falsified; weak law confirmed.**

---

# 4. MBX–Omniabase Law (v0.2)

> Prime numbers are statistical maxima of multibase structural instability  
> relative to the accumulated symmetry of preceding composite numbers.  
>  
> This instability is measurable via PBII(n)  
> and separates the two PBII distributions with a ratio ≥ 20×  
> when parameters are optimized.

---

# 5. Roadmap to PBII v0.3

- Extend computation to \( n \le 10^7 \).  
- Expand bases to all primes ≤ 200.  
- Replace divisibility bonus with continuous modular distance.  
- Compute full AUC ROC curve.  
- Release reproducible code + PBII dataset.  
- Formalize mutual information analysis.

---

# 6. License

MIT License (same as MBX–AI Loop).

---

# 7. Author

**Massimiliano Brighindi**  
MB-X.01 / Omniabase±  
brighissimo@gmail.com

---

# End of PBII Report v0.2