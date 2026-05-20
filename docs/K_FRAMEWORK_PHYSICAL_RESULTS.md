# K-Framework Physical Results

## Status:
- QHO full projective quotient: PROVEN
- QHO spectral reduction: PROVEN
- Free particle full projective quotient: DERIVED UNDER ASSUMPTION

---

## Reference

All definitions and symbols are inherited from:

- `K_FRAMEWORK_CORE.md`
- `K_FRAMEWORK_FOUNDATIONS.md`

No redefinitions are introduced.

---

# 1. Theorem — QHO full projective quotient is Case C

Let the quantum harmonic oscillator be defined by

\[
H=\hbar\omega\left(a^\dagger a+\frac12\right)
\]

and consider the structural triple

\[
\mathfrak T_{\mathrm{QHO}}=
\left(
\mathbb P(\mathcal H),
\mathcal G=\{U_t=e^{-itH}\}_{t\in\mathbb R},
d_{FS}
\right).
\]

Define the orbit equivalence relation on \(\mathbb P(\mathcal H)\) by

\[
[\psi]_{\mathrm{ray}} \sim [\phi]_{\mathrm{ray}}
\iff
\exists t\in\mathbb R:\ U_t[\psi]_{\mathrm{ray}}=[\phi]_{\mathrm{ray}}.
\]

Let

\[
\mathcal Q_{\mathrm{QHO}}:=\mathbb P(\mathcal H)/\sim
\]

and denote orbit classes by

\[
[[\psi]]\in\mathcal Q_{\mathrm{QHO}}.
\]

Then:

\[
k(\mathcal Q_{\mathrm{QHO}})=0
\]

and for every pair of distinct orbit classes

\[
[[\psi]]\neq[[\phi]]
\]

one has

\[
\Delta([[\psi]],[[\phi]])>0.
\]

Therefore:

\[
\boxed{
\mathcal Q_{\mathrm{QHO}} \in \mathrm{Case\ C}
}
\]

## Proof

### Step 1 — Vanishing gap

Take

\[
[\psi]_{\mathrm{ray}}=[|0\rangle]_{\mathrm{ray}}
\]

and, for \(0<\varepsilon<1\),

\[
|\phi_\varepsilon\rangle
=
\sqrt{1-\varepsilon^2}\,|0\rangle+\varepsilon |1\rangle.
\]

Then

\[
[\phi_\varepsilon]_{\mathrm{ray}}\neq[\psi]_{\mathrm{ray}}
\qquad (\varepsilon>0)
\]

and

\[
d_{FS}([\psi]_{\mathrm{ray}},[\phi_\varepsilon]_{\mathrm{ray}})
=
\arccos\!\big(|\langle 0|\phi_\varepsilon\rangle|\big)
=
\arccos(\sqrt{1-\varepsilon^2})
\to 0
\qquad (\varepsilon\to 0).
\]

Moreover, \(U_t|0\rangle\) remains in the ray of \(|0\rangle\), while \([\phi_\varepsilon]_{\mathrm{ray}}\) has nonzero \(|1\rangle\)-component, hence

\[
[[0]]\neq[[\phi_\varepsilon]]
\qquad (\varepsilon>0).
\]

By Lemma 3,

\[
k(\mathcal Q_{\mathrm{QHO}})=0.
\]

### Step 2 — Strict positivity of the residual for distinct orbit classes

Fix

\[
[[\psi]]\neq[[\phi]].
\]

Define

\[
f_{\psi,\phi}(t)
=
d_{FS}(U_t[\psi]_{\mathrm{ray}},[\phi]_{\mathrm{ray}}).
\]

Since \(E_n=\hbar\omega(n+\tfrac12)\), all projective relative phases are integer multiples of \(\omega t\). Hence the induced flow on \(\mathbb P(\mathcal H)\) is periodic with period

\[
\frac{2\pi}{\omega}.
\]

Therefore \(f_{\psi,\phi}\) is continuous and periodic, so it attains a minimum on the compact interval \([0,2\pi/\omega]\). Thus there exists \(t_*\) such that

\[
\Delta([[\psi]],[[\phi]])
=
f_{\psi,\phi}(t_*).
\]

Assume by contradiction that

\[
\Delta([[\psi]],[[\phi]])=0.
\]

Then

\[
f_{\psi,\phi}(t_*)=0.
\]

Since \(d_{FS}\) is a metric on projective rays,

\[
U_{t_*}[\psi]_{\mathrm{ray}}=[\phi]_{\mathrm{ray}}.
\]

Hence \([\psi]_{\mathrm{ray}}\) and \([\phi]_{\mathrm{ray}}\) lie in the same orbit, i.e.

\[
[[\psi]]=[[ \phi ]],
\]

contradiction.

Therefore

\[
\Delta([[\psi]],[[\phi]])>0
\qquad
\forall [[\psi]]\neq[[\phi]].
\]

This proves

\[
\boxed{
\mathcal Q_{\mathrm{QHO}} \in \mathrm{Case\ C}
}
\]

---

# 2. Corollary — QHO spectral reduction is Case A

Consider the reduced spectral representation

\[
\mathcal S_{\mathrm{spec}}=\{|n\rangle\}_{n\in\mathbb N_0}
\]

with structural distance

\[
d_{\mathrm{spec}}(|n\rangle,|m\rangle)
=
|E_n-E_m|
=
\hbar\omega |n-m|.
\]

Then

\[
\Delta(|n\rangle,|m\rangle)=\hbar\omega |n-m|
\]

and therefore

\[
k=\inf_{n\neq m}\hbar\omega |n-m|=\hbar\omega>0.
\]

Hence

\[
\boxed{
\mathfrak T_{\mathrm{QHO,spec}}\in \mathrm{Case\ A}
}
\]

---

# 3. Proposition — Free particle full projective quotient is Case C (under stated condition)

Let the free particle be defined by

\[
H=\frac{P^2}{2m}
\]

and consider the structural triple

\[
\mathfrak T_{\mathrm{free}}=
\left(
\mathbb P(\mathcal H),
\mathcal G=\{U_t=e^{-itH}\}_{t\in\mathbb R},
d_{FS}
\right).
\]

Define the orbit quotient

\[
\mathcal Q_{\mathrm{free}}:=\mathbb P(\mathcal H)/\sim
\]

with orbit classes denoted by \( [[\psi]] \).

Assume the non-collapse condition:

\[
\boxed{
[[\psi]]\neq[[\phi]]
\ \Rightarrow\
[\phi]_{\mathrm{ray}} \notin \overline{\{U_t[\psi]_{\mathrm{ray}}:t\in\mathbb R\}}
}
\]

Then:

\[
k(\mathcal Q_{\mathrm{free}})=0
\]

and

\[
[[\psi]]\neq[[\phi]]
\ \Rightarrow\
\Delta([[\psi]],[[\phi]])>0.
\]

Therefore:

\[
\boxed{
\mathcal Q_{\mathrm{free}} \in \mathrm{Case\ C}
\quad\text{under the stated condition}
}
\]

## Proof

### Step 1 — Vanishing gap

Let \(|\psi\rangle\) be normalized and let \(|\eta\rangle\) be normalized with \(\langle\psi|\eta\rangle=0\). Define

\[
|\phi_\varepsilon\rangle
=
\sqrt{1-\varepsilon^2}\,|\psi\rangle+\varepsilon |\eta\rangle.
\]

Then

\[
[\phi_\varepsilon]_{\mathrm{ray}}\neq[\psi]_{\mathrm{ray}}
\qquad (\varepsilon>0)
\]

and

\[
d_{FS}([\psi]_{\mathrm{ray}},[\phi_\varepsilon]_{\mathrm{ray}})
=
\arccos(\sqrt{1-\varepsilon^2})
\to 0.
\]

Choosing \(|\eta\rangle\) outside the one-parameter orbit direction of \([\psi]_{\mathrm{ray}}\), the corresponding orbit classes remain distinct for sufficiently small \(\varepsilon\). Hence, by Lemma 3,

\[
k(\mathcal Q_{\mathrm{free}})=0.
\]

### Step 2 — Strict positivity under non-collapse

Fix distinct orbit classes

\[
[[\psi]]\neq[[\phi]].
\]

Define

\[
f_{\psi,\phi}(t)
=
d_{FS}(U_t[\psi]_{\mathrm{ray}},[\phi]_{\mathrm{ray}}).
\]

If

\[
\Delta([[\psi]],[[\phi]])=0,
\]

then there exists a sequence \(t_n\) such that

\[
f_{\psi,\phi}(t_n)\to 0,
\]

equivalently,

\[
U_{t_n}[\psi]_{\mathrm{ray}} \to [\phi]_{\mathrm{ray}}.
\]

Thus \([\phi]_{\mathrm{ray}}\) belongs to the orbit closure of \([\psi]_{\mathrm{ray}}\), contradicting the non-collapse assumption.

Therefore

\[
\Delta([[\psi]],[[\phi]])>0
\qquad
\forall [[\psi]]\neq[[\phi]].
\]

Hence

\[
\boxed{
\mathcal Q_{\mathrm{free}} \in \mathrm{Case\ C}
\quad\text{under the stated condition}
}
\]

---

# 4. Structural Consequence

\[
\boxed{
\text{Full quantum state spaces } \rightarrow \text{Case C}
}
\]

\[
\boxed{
\text{Case A emerges only under structural reduction}
}
\]

\[
\boxed{
\text{Discreteness is representation-dependent}
}
\]