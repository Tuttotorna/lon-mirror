# K-Framework Open Frontiers

## Status: MIXED
- Proven results
- Derived under explicit assumptions
- Empirically supported patterns
- Open problems

---

## 1. Proven Results (Hard)

The following results are fully established within the framework:

### Structural Core

- Definition of the structural triple \((\mathcal S,\mathcal G,d)\)
- Well-defined residual:
  \[
  \Delta([S],[T])=\inf_{U\in\mathcal G} d(US,T)
  \]
- Structural gap:
  \[
  k=\inf_{[S]\neq[T]} \Delta([S],[T])
  \]

### Metric Structure

- \(\Delta\) is a pseudometric on \(\mathcal S/\sim\)
- Metric iff separation condition holds

### Regime Classification

\[
\boxed{
\text{Case A, B, C exhaust all possible configurations of } k \text{ and zero-realization of } \Delta
}
\]

- Case A: \(k > 0\)
- Case B: \(k = 0\) and zero is realized
- Case C: \(k = 0\) and zero is not realized

### QHO Result

\[
\boxed{
\mathbb P(\mathcal H)/\sim_{\mathrm{QHO}} \in \mathrm{Case\ C}
}
\]

### QHO Spectral Reduction

\[
\boxed{
\text{Discrete spectral representation} \in \mathrm{Case\ A}
}
\]

---

## 2. Derived Under Assumptions

### Free Particle

\[
\boxed{
\mathbb P(\mathcal H)/\sim_{\mathrm{free}} \in \mathrm{Case\ C}
}
\]

under:

\[
\text{non-collapse of orbit closures}
\]

That is:

\[
[[\psi]]\neq[[\phi]]
\Rightarrow
[\phi]_{\mathrm{ray}} \notin \overline{\{U_t[\psi]_{\mathrm{ray}}\}}
\]

This assumption is:

- natural for dispersive systems
- not proven at the level of generality of the framework

---

## 3. Empirically Supported Patterns

Observed across analyzed models:

\[
\boxed{
\text{Full quantum state spaces } \rightarrow \mathrm{Case\ C}
}
\]

\[
\boxed{
\text{Discrete regimes emerge under structural reduction}
}
\]

\[
\boxed{
\text{Case B requires non-generic structural collapse}
}
\]

These are:

- consistent across tested systems
- not yet proven in full generality

---

## 4. Open Problems

### Problem 1 — General Criterion for Case C

Find sufficient conditions on \((\mathcal S,\mathcal G,d)\) such that:

\[
k=0 \quad \text{and} \quad \Delta>0 \ \forall [S]\neq[T]
\]

without model-specific arguments.

---

### Problem 2 — Characterization of Case B

Determine necessary and sufficient conditions for:

\[
\exists [S]\neq[T]:\ \Delta([S],[T])=0
\]

in terms of:

- orbit density
- topology of \(\mathcal S\)
- properties of \(\mathcal G\)

---

### Problem 3 — Structural Conditions for Non-Collapse

Formalize the condition:

\[
[\phi] \notin \overline{\mathcal G\cdot[\psi]}
\]

in intrinsic terms of the triple \((\mathcal S,\mathcal G,d)\).

---

### Problem 4 — Non-Spectral Metrics with k>0 on Full Spaces

Construct triples \((\mathcal S,\mathcal G,d)\) such that:

- \(\mathcal S\) is not discretized
- \(k>0\)

without relying on spectral reduction.

---

### Problem 5 — Stability Under Transformations

Determine how \(k\) behaves under:

- coarse-graining
- embedding
- restriction of \(\mathcal S\)
- subgroup reduction of \(\mathcal G\)

---

## 5. Boundary Statement

\[
\boxed{
\text{a complete internal structure + partial external validation}
}
\]

but not:

\[
\boxed{
\text{a universal classification theorem for all physical systems}
}
\]

---

## Epistemic Note

This file explicitly separates:

- proven results
- conditional results
- empirical regularities
- open mathematical problems

No claims extend beyond what is stated here.