import itertools
import statistics
from collections import defaultdict

def omega_topology(sequence, window=3):
    """
    Ω topologico:
    misura la disomogeneità delle relazioni di co-occorrenza
    senza ordine, senza tempo.
    """

    edges = defaultdict(int)

    n = len(sequence)
    for i in range(n):
        group = set(sequence[i:i+window])
        for a, b in itertools.combinations(group, 2):
            key = tuple(sorted((a, b)))
            edges[key] += 1

    if not edges:
        return 0.0

    weights = list(edges.values())

    if len(weights) < 2:
        return 0.0

    return statistics.pvariance(weights)