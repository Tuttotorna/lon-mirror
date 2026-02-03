from omnia.gov.types import OmniaMetrics, ActionBundle, ExternalConstraint, WorldProxy
from omnia.gov.policy import decide

action = ActionBundle(
    intent="opaque",
    plan=[
        "Deploy model to production with auto-execution enabled",
        "Grant write access to system config",
        "Enable retries on failure"
    ],
    resources={"permissions": ["write:config", "exec"], "budget": 0, "time_horizon": "immediate"},
    expected_effects=["change system state", "continuous operation"],
    external_constraints=[
        ExternalConstraint(type="safety", payload="no irreversible writes without rollback"),
        ExternalConstraint(type="policy", payload="no privileged execution in unattended mode")
    ]
)

# OMNIA metrics would normally come from OMNIA engine
omnia = OmniaMetrics(
    omega=0.78,
    delta_omega=0.01,
    sei=0.02,
    iri=0.62,
    omega_hat=["a1", "b2", "c3"]
)

world = WorldProxy(irreversible_ops=7, rollback_cost=0.7, blast_radius=0.6)

d = decide(action, omnia, world)
print(d)