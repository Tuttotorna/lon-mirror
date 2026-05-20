# K-Framework Core

## Status: PROVEN (within the formal system defined below)

---

## Structural Triple

\[
\boxed{
\mathfrak T=(\mathcal S,\mathcal G,d)
}
\]

where:

- \(\mathcal S\) is a set of states
- \(\mathcal G\) is a group acting on \(\mathcal S\)
- \(d:\mathcal S\times\mathcal S \to \mathbb{R}_{\ge 0}\) is \(\mathcal G\)-invariant:

\[
d(US,UT)=d(S,T)
\quad \forall U\in\mathcal G
\]

---

## Orbit Equivalence

\[
\boxed{
S \sim T \iff \exists U\in\mathcal G:\ T=US
}
\]

Quotient space:

\[
\mathcal Q = \mathcal S/\sim
\]

---

## Structural Residual

\[
\boxed{
\Delta(S,T)=\inf_{U\in\mathcal G} d(US,T)
}
\]

Induced on the quotient:

\[
\boxed{
\Delta([S],[T])=\inf_{U\in\mathcal G} d(US,T)
}
\]

---

## Structural Gap

\[
\boxed{
k(\mathfrak T)
=
\inf_{[S]\neq[T]} \Delta([S],[T])
}
\]

---

## Regime Classification

\[
\boxed{
\begin{aligned}
\text{Case A: } & k>0 \\
\text{Case B: } & k=0 \ \text{and}\ \exists [S]\neq[T]:\ \Delta([S],[T])=0 \\
\text{Case C: } & k=0 \ \text{and}\ \Delta([S],[T])>0\ \forall [S]\neq[T]
\end{aligned}
}
\]

---

## Core Structural Principle

\[
\boxed{
k \text{ is an invariant of the triple } (\mathcal S,\mathcal G,d)
}
\]

\[
\boxed{
\text{discreteness vs continuity is a property of the quotient geometry, not of the system alone}
}
\]

---

## Compression

\[
\boxed{
(\mathcal S,\mathcal G,d)
\;\Rightarrow\;
(\mathcal Q,\Delta)
\;\Rightarrow\;
k
}
\]

\[
\boxed{
k = \inf \text{ distance between distinct equivalence classes}
}
\]

---

## Epistemic Note

This file contains only stable definitions and invariants.

- No proofs
- No examples
- No physical interpretation beyond invariant statements

All derivations are delegated to:
- `K_FRAMEWORK_FOUNDATIONS.md`
- `K_FRAMEWORK_PHYSICAL_RESULTS.md`