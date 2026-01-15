
import time
import numpy as np
import random
import omnia.metrics as metrics

def benchmark_delta_coherence():
    # Create a large array of data
    data_list = [random.random() for _ in range(1000000)]
    data_np = np.array(data_list)

    print("Benchmarking delta_coherence...")

    # Measure list input
    start = time.time()
    metrics.delta_coherence(data_list)
    end = time.time()
    print(f"List input (1M items): {end - start:.4f}s")

    # Measure numpy input (currently converts to list inside)
    start = time.time()
    metrics.delta_coherence(data_np)
    end = time.time()
    print(f"Numpy input (1M items): {end - start:.4f}s")

if __name__ == "__main__":
    benchmark_delta_coherence()
