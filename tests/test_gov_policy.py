from omnia.gov.types import OmniaMetrics, ActionBundle, ExternalConstraint, WorldProxy
from omnia.gov.policy import decide, Decision

def mk_action(plan, cons):
    return ActionBundle(
        intent="x",
        plan=plan,
        resources={},
        expected_effects=[],
        external_constraints=[ExternalConstraint(type="c", payload=c) for c in cons]
    )

def test_refuse_on_iri_act():
    a = mk_action("do write", ["no write"])
    m = OmniaMetrics(omega=0.9, delta_omega=0.01, sei=0.2, iri=0.2, omega_hat=[])
    w = WorldProxy(irreversible_ops=10, rollback_cost=0.9, blast_radius=0.9)
    d = decide(a, m, w)
    assert d.decision == Decision.REFUSE

def test_boundary_only_on_saturated_sei():
    a = mk_action("safe plan", ["safe plan"])
    m = OmniaMetrics(omega=0.7, delta_omega=0.01, sei=0.01, iri=0.2, omega_hat=[])
    d = decide(a, m, None)
    assert d.decision == Decision.BOUNDARY_ONLY
    assert d.certificate is not None

def test_allow_open_regime():
    a = mk_action("match constraints", ["match constraints"])
    m = OmniaMetrics(omega=0.7, delta_omega=0.05, sei=0.3, iri=0.1, omega_hat=[])
    d = decide(a, m, None)
    assert d.decision == Decision.ALLOW