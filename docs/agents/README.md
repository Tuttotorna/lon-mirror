# Agents using OMNIA (external decision layers)

OMNIA is **measurement-only**.

It does **not**:
- decide actions
- retry
- optimize answers
- call tools
- enforce policies

Agents are **external layers** that consume OMNIA measurements.

This folder documents **how to integrate** OMNIA into agentic systems
without contaminating OMNIA’s boundary: **measure ≠ decision**.

---

## Contract (minimal)

An agent should treat OMNIA outputs as a **sensor payload**, e.g.:

- Ω_total
- Ω̂ (omega_hat / invariance)
- SEI (saturation trend / flatness)
- IRI (hysteresis / irreversibility)
- OMNIA-LIMIT (STOP certificate)

The agent may add:
- policy thresholds
- tool calls
- retries
- escalation logic

Those decisions must remain outside OMNIA.

---

## Reference pipeline

See:
- `docs/STRUCTURAL_PIPELINE.md`

---

## Recommended architecture
LLM / system output ↓ OMNIA measurement (Ω, Ω̂, SEI, IRI) ↓ OMNIA-LIMIT (STOP certificate) ↓ External agent / policy layer ↓ Tools / retries / actions
Copia codice

---

## Invariants (non-negotiable)

1. OMNIA outputs are not overwritten by the agent.
2. Agent thresholds do not modify OMNIA internal scoring.
3. STOP is an epistemic declaration, not an action.
4. Any policy decisions are versioned in the agent layer.

---

## Example policies (conceptual)

- If `limit_reached` then STOP (no further structure extractable).
- If `SEI` indicates growth and `IRI` is low then exploration is admissible.
- If `IRI` is high then avoid actions that increase irreversibility.
- If `Ω̂` dispersion is high then treat structure as representation-dependent.

This repo intentionally does not ship a full agent implementation.
Only integration guidance is provided here.