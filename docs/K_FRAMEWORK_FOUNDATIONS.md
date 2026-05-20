# K-Framework Foundations

## Status: PROVEN (within the formal system defined in K_FRAMEWORK_CORE.md)

---

## Reference

All definitions and symbols are inherited from:

- `K_FRAMEWORK_CORE.md`

No redefinitions are introduced.

---

## Lemma 1 — Well-definedness of Δ on the quotient

\[
\boxed{
\Delta([S],[T])=\inf_{U\in\mathcal G} d(US,T)
\ \text{is well-defined}
}
\]

### Proof

Let \(S' = V S\), \(T' = W T\) with \(V,W \in \mathcal G\).

Then:

\[
\inf_{U\in\mathcal G} d(U S', T')
=
\inf_{U\in\mathcal G} d(U V S, W T)
\]

Using \(\mathcal G\)-invariance:

\[
d(U V S, W T)=d(W^{-1} U V S, T)
\]

Since \(\mathcal G\) is a group:

\[
W^{-1} U V \in \mathcal G
\]

Hence:

\[
\inf_{U\in\mathcal G} d(U S', T')
=
\inf_{U'\in\mathcal G} d(U' S, T)
\]

Therefore:

\[
\boxed{
\Delta([S],[T]) \ \text{is independent of representatives}
}
\]

---

## Lemma 2 — Metric structure on the quotient

\[
\boxed{
\Delta \text{ is a pseudometric on } \mathcal Q=\mathcal S/\sim
}
\]

That is:

- \(\Delta \ge 0\)
- \(\Delta([S],[S])=0\)
- \(\Delta([S],[T])=\Delta([T],[S])\)
- \(\Delta([S],[U]) \le \Delta([S],[T]) + \Delta([T],[U])\)

### Symmetry check

Using symmetry of \(d\), group inversion, and \(\mathcal G\)-invariance:

\[
\Delta([S],[T])
=
\inf_{U\in\mathcal G} d(US,T)
=
\inf_{U\in\mathcal G} d(T,US)
=
\inf_{U\in\mathcal G} d(U^{-1}T,S)
=
\Delta([T],[S])
\]

If additionally:

\[
\boxed{
\Delta([S],[T])=0 \Rightarrow [S]=[T]
}
\]

then:

\[
\boxed{
\Delta \text{ is a metric}
}
\]

---

## Lemma 3 — Characterization of k = 0

\[
\boxed{
k=0
\iff
\forall \varepsilon>0,\ \exists [S]\neq[T]\ \text{such that}\ \Delta([S],[T])<\varepsilon
}
\]

Equivalent formulation:

\[
\boxed{
k=0
\iff
\exists [S_n]\neq[T_n]\ \text{with}\ \Delta([S_n],[T_n]) \to 0
}
\]

---

## Lemma 4 — Separation between Case B and Case C

\[
\boxed{
\text{Case B}
\iff
\exists [S]\neq[T] \text{ such that } \Delta([S],[T])=0
}
\]

\[
\boxed{
\text{Case C}
\iff
\forall [S]\neq[T],\ \Delta([S],[T])>0
\ \text{and}\ k=0
}
\]

Thus:

- Case B → zero distance is realized
- Case C → zero distance is only asymptotic

---

## Structural Consequence

\[
\boxed{
k>0 \Rightarrow \text{uniform separation}
}
\]

\[
\boxed{
k=0 \Rightarrow \text{arbitrary closeness of distinct classes}
}
\]

---

## Compact Form

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
\Delta \text{ is a pseudometric (metric under separation)}
}
\]

\[
\boxed{
k=0 \iff \text{there exist arbitrarily close distinct classes}
}
\]

\[
\boxed{
\text{Case B vs C is determined by realization of } \Delta=0
}
\]

---

## Epistemic Note

This file contains:

- formal derivations
- no physical interpretation
- no examples beyond structural necessity

All applications are delegated to:

- `K_FRAMEWORK_PHYSICAL_RESULTS.md`