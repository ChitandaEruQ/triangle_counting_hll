"""Exact triangle counting: Forward-Exact-Adaptive."""

import time
import numpy as np
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Intersection kernels
# ---------------------------------------------------------------------------

def intersect_merge(a: np.ndarray, b: np.ndarray) -> int:
    """Sorted-merge intersection count.  O(|a| + |b|)."""
    return int(len(np.intersect1d(a, b, assume_unique=True)))


def intersect_bsearch(small: np.ndarray, large: np.ndarray) -> int:
    """
    Binary-search intersection count.
    Assumes |small| <= |large|.  O(|small| * log|large|).
    """
    if len(small) == 0:
        return 0
    idx = np.searchsorted(large, small)
    mask = idx < len(large)
    return int(np.sum(large[idx[mask]] == small[mask]))


def intersect_adaptive(a: np.ndarray, b: np.ndarray, r: float = 8.0) -> int:
    """
    Adaptive intersection: merge when size-ratio <= r, binary search otherwise.
    Always uses the smaller array as the probe set.
    """
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0
    if la > lb:
        a, b, la, lb = b, a, lb, la          # ensure a is the smaller array
    if lb / la <= r:
        return intersect_merge(a, b)
    return intersect_bsearch(a, b)


# ---------------------------------------------------------------------------
# Triangle counting
# ---------------------------------------------------------------------------

def exact_triangle_count(
    forward_adj: Dict[int, np.ndarray],
    r: float = 8.0,
    per_edge_timing: bool = False,
) -> Tuple[int, float, List]:
    """
    Count triangles exactly using the forward-orientation formula:

        T = Σ_{(u,v) ∈ E+}  |N+(u) ∩ N+(v)|

    Each triangle (a, b, c) — with a < b < c in orientation order — is counted
    exactly once, at the edge (a, b) whose common forward neighbour is c.

    Parameters
    ----------
    forward_adj     : dict  node -> sorted np.int64 array of forward neighbours
    r               : merge / binary-search threshold (use merge if |b|/|a| <= r)
    per_edge_timing : if True, record per-edge tuple
                      (u, v, deg_u, deg_v, support, time_s)

    Returns
    -------
    count      : exact global triangle count
    elapsed    : total counting wall-clock time (seconds)
    edge_data  : list of per-edge records (empty when per_edge_timing=False)
    """
    total: int = 0
    edge_data: List = []

    t0 = time.perf_counter()

    for u, nbrs_u in forward_adj.items():
        if len(nbrs_u) == 0:
            continue
        for v in nbrs_u:
            nbrs_v = forward_adj.get(int(v))
            if nbrs_v is None or len(nbrs_v) == 0:
                continue

            if per_edge_timing:
                te0 = time.perf_counter()
                s = intersect_adaptive(nbrs_u, nbrs_v, r)
                te1 = time.perf_counter()
                edge_data.append((u, int(v), len(nbrs_u), len(nbrs_v), s, te1 - te0))
            else:
                s = intersect_adaptive(nbrs_u, nbrs_v, r)

            total += s

    elapsed = time.perf_counter() - t0
    return total, elapsed, edge_data
