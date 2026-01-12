from omnia.sei import SEIEngine

def run_demo():
    sei = SEIEngine(window=5, flat_eps=0.03)

    # Simulated iterations: cost grows, benefit shrinks
    samples = [
        # dq, du, tin, tout, latency, energy, iters
        (0.080, 0.040, 900, 450, 700,  18.0, 1),
        (0.050, 0.030, 1200, 600, 950,  22.0, 1),
        (0.030, 0.020, 1600, 800, 1300, 28.0, 2),
        (0.020, 0.015, 2200, 1100, 1900, 35.0, 2),
        (0.012, 0.010, 3100, 1500, 2800, 48.0, 3),
        (0.008, 0.006, 4200, 2100, 4100, 62.0, 4),
        (0.006, 0.004, 5600, 2800, 5600, 78.0, 5),
    ]

    for i, (dq, du, tin, tout, lat, ej, iters) in enumerate(samples, 1):
        rec = sei.add(
            context="llm_inference_demo",
            delta_quality=dq,
            delta_uncertainty=du,
            tokens_in=tin,
            tokens_out=tout,
            latency_ms=lat,
            energy_joule=ej,
            iterations=iters,
        )
        print(f"[{i}] SEI={rec.sei:.10f}  trend={rec.trend}  z={rec.sei_z}")

    print("snapshot:", sei.snapshot())

if __name__ == "__main__":
    run_demo()