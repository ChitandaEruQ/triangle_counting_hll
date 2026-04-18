"""DOULION approximate triangle counting via edge sampling."""

import time
import numpy as np
from typing import Set, Tuple

from .graph import orient_graph
from .exact import exact_triangle_count


def doulion(
    edges: Set[Tuple[int, int]],
    q: float,
    rng: np.random.Generator,
    r: float = 8.0,
) -> Tuple[float, float]:
    """
    DOULION: keep each edge with probability q, count triangles in the
    sample with the exact backend, then scale by 1/q^3.

    Parameters
    ----------
    edges : canonical undirected edge set
    q     : retention probability  (0 < q <= 1)
    rng   : numpy random generator
    r     : merge/bsearch threshold for exact backend

    Returns
    -------
    estimate : estimated global triangle count
    elapsed  : wall-clock time (sampling + orientation + counting)
    """
    t0 = time.perf_counter()

    edges_list = list(edges)
    mask = rng.random(len(edges_list)) < q
    sampled = {e for e, keep in zip(edges_list, mask) if keep}

    if not sampled:
        return 0.0, time.perf_counter() - t0

    forward_adj, _ = orient_graph(sampled)
    sample_count, _, _ = exact_triangle_count(forward_adj, r=r)
    elapsed = time.perf_counter() - t0

    return sample_count / (q ** 3), elapsed


def doulion_repeated(
    edges: Set[Tuple[int, int]],
    q: float,
    n_seeds: int = 10,
    r: float = 8.0,
    base_seed: int = 42,
) -> dict:
    """
    Run DOULION with n_seeds independent random seeds.

    Returns a dict with:
        q, n_seeds,
        mean_estimate, std_estimate,
        mean_time, std_time,
        estimates (list), times (list)
    """
    estimates = []
    times = []

    for seed in range(base_seed, base_seed + n_seeds):
        rng = np.random.default_rng(seed)
        est, elapsed = doulion(edges, q, rng, r=r)
        estimates.append(est)
        times.append(elapsed)

    est_arr = np.array(estimates, dtype=np.float64)
    time_arr = np.array(times, dtype=np.float64)

    return {
        'q': q,
        'n_seeds': n_seeds,
        'mean_estimate': float(np.mean(est_arr)),
        'std_estimate': float(np.std(est_arr)),
        'mean_time': float(np.mean(time_arr)),
        'std_time': float(np.std(time_arr)),
        'estimates': estimates,
        'times': times,
    }
