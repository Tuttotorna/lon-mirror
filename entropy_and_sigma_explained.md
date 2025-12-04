# Entropy and Base Symmetry Score — Formal Technical Note
## MBX–Omniabase · Supplement to PBII Report v0.2
### Author: Massimiliano Brighindi — MB-X.01 / Omniabase±

---

# 1. Entropy in Base b ( \tilde{H}_b(n) )

Entropy measures the **digit-distribution uncertainty** of an integer \( n \) when expressed in base \( b \).
It comes directly from Shannon's Information Theory.

Given:
\[
\text{digits}_b(n) = (d_0, d_1, \dots, d_{L_b(n)-1}),\quad d_i \in \{0,1,\dots,b-1\}
\]

## 1.1. Digit Frequency Distribution

\[
p_k = \frac{\#\{i : d_i = k\}}{L_b(n)}, \quad k = 0,\dots,b-1
\]

## 1.2. Shannon Entropy (raw)

\[
H_b(n) = -\sum_{k=0}^{b-1} p_k \log(p_k)
\]

Convention:
\[
p_k = 0 \;\Rightarrow\; p_k \log(p_k) = 0
\]

## 1.3. Maximum Entropy in Base b

Occurs when digits are equiprobable:
\[
H_b^{\max} = \log(b)
\]

## 1.4. Normalized Entropy

\[
\tilde{H}_b(n) = \frac{H_b(n)}{H_b^{\max}}
\]

Properties:
- \( \tilde{H}_b(n) \in [0,1] \)
- \( \tilde{H}_b(n) \approx 1 \) → high randomness (low structure)
- \( \tilde{H}_b(n) \approx 0 \) → low randomness (high structure)

Entropy defines the **instability** component of a number in base \( b \).

---

# 2. Base Symmetry Score ( \sigma_b(n) )

This score measures how “structured” a number appears **in a specific base**.  
Version 0.2 uses two structural components and one divisibility reinforcement.

\[
\sigma_b(n) =
\frac{1}{L_b(n)} (1 - \tilde{H}_b(n))
+ 0.5 \cdot \mathbf{1}_{(n \bmod b = 0)}
\]

---

## 2.1. Component A — Anti-Instability Term

### (1) Anti-Entropy
\[
1 - \tilde{H}_b(n)
\]
High when the digit pattern is structured, low when it is random.

### (2) Length Penalty
\[
\frac{1}{L_b(n)}
\]
Shorter representations receive higher structure score.

Reason:
Short digit strings imply lower combinatorial freedom, thus greater implicit regularity.

---

## 2.2. Component B — Divisibility Bonus

\[
0.5 \cdot \mathbf{1}_{(n \bmod b = 0)}
\]

Interpretation:
- If \( n \) is exactly divisible by \( b \) → **structural reinforcement**.
- Example:  
  - even numbers in base 2  
  - multiples of 5 or 10 in base 10  
  - multiples of 3 in base 3  

This term is essential:  
it amplifies the structural advantage that composite numbers have over primes.

---

# 3. Why Composites Score High and Primes Score Low

**Composite numbers** often have:
- exact divisibility by many \( b \in B \)
- shorter representations in certain bases
- lower digit entropy (more repeated patterns)

**Prime numbers**:
- rarely divisible by any \( b \in B \) (except 1 and themselves)
- tend to have more irregular digit patterns across bases
- yield longer or less compressible representations

Hence:

\[
E[\sigma_b(n) \mid \text{prime}] < E[\sigma_b(n) \mid \text{composite}]
\]

and

\[
E[\text{PBII}(n) \mid \text{prime}] \gg E[\text{PBII}(n) \mid \text{composite}]
\]

This explains the **28× separation** observed in the PBII Report v0.2.

---

# 4. Role in PBII

PBII is defined as:

\[
\text{PBII}(n) = \text{Sat}(n) - \Sigma(n)
\]

where:
- \( \Sigma(n) \) is the **average** of \( \sigma_b(n) \) across bases
- \( \text{Sat}(n) \) is the **mean composite symmetry** in the backward window

The structure defined here (entropy + length + divisibility) is **exactly** what produces the measurable multibase instability of primes.

---

# End of Technical Note