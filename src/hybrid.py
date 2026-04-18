"""Selective Hybrid HLL triangle counting."""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple

from .hll import HLL, union_estimate
from .exact import intersect_adaptive


# ---------------------------------------------------------------------------
# Sketch building
# ---------------------------------------------------------------------------

def build_sketches(
    forward_adj: Dict[int, np.ndarray],
    theta: int,
    p: int = 8,
) -> Dict[int, HLL]:
    """
    Build one HLL sketch per node whose forward degree >= theta.

    The sketch for node u encodes all elements of N+(u).

    Parameters
    ----------
    forward_adj : node -> sorted forward-neighbour array
    theta       : minimum forward degree to receive a sketch
    p           : HLL precision (2^p registers)

    Returns
    -------
    sketches : dict  node -> HLL  (only nodes with deg+(node) >= theta)
    """
    sketches: Dict[int, HLL] = {}
    for node, nbrs in forward_adj.items():
        if len(nbrs) >= theta:
            sk = HLL(p=p)
            sk.add_array(nbrs)
            sketches[node] = sk
    return sketches


# ---------------------------------------------------------------------------
# Hybrid counting
# ---------------------------------------------------------------------------

def hybrid_triangle_count(
    forward_adj: Dict[int, np.ndarray],
    sketches: Dict[int, HLL],
    gamma: int,
    r: float = 8.0,
    record_details: bool = False,
) -> Tuple[float, float, dict, Optional[List[dict]]]:
    """
    Selective Hybrid HLL global triangle count.

    Routing rule for oriented edge (u, v):
      proxy = min(deg+(u), deg+(v))
      → HLL path  if proxy >= gamma AND both u, v have sketches
      → Exact path otherwise

    HLL estimate of support:
      s_hat   = deg+(u) + deg+(v) - union_estimate(sk_u, sk_v)
      clamped = clip(s_hat, 0, min(deg+(u), deg+(v)))

    Global estimate:
      T_hat = Σ_{(u,v) ∈ E+}  support(u, v)

    Parameters
    ----------
    forward_adj    : node -> sorted forward-neighbour array
    sketches       : node -> HLL (built by build_sketches)
    gamma          : proxy threshold for HLL routing
    r              : merge/bsearch ratio for exact path
    record_details : if True, store per-edge dicts in returned list

    Returns
    -------
    total_estimate : float
    elapsed        : counting wall-clock time (seconds)
    stats          : dict with coverage, fallback_ratio, clamp_ratio, …
    edge_details   : list of per-edge dicts (None if record_details=False)
    """
    total: float = 0.0
    n_edges = 0
    n_approx = 0
    n_fallback = 0   # wanted HLL but one sketch was missing
    n_clamped = 0

    edge_details: Optional[List[dict]] = [] if record_details else None

    t0 = time.perf_counter()

    for u, nbrs_u in forward_adj.items():
        deg_u = len(nbrs_u)
        if deg_u == 0:
            continue
        sk_u = sketches.get(u)

        for v in nbrs_u:
            v_int = int(v)
            nbrs_v = forward_adj.get(v_int, np.array([], dtype=np.int64))
            deg_v = len(nbrs_v)
            n_edges += 1

            proxy = min(deg_u, deg_v)
            sk_v = sketches.get(v_int)

            wants_hll = proxy >= gamma
            has_both_sketches = (sk_u is not None) and (sk_v is not None)
            use_hll = wants_hll and has_both_sketches

            if wants_hll and not has_both_sketches:
                n_fallback += 1

            if use_hll:
                n_approx += 1
                union_est = union_estimate(sk_u, sk_v)
                s_hat = deg_u + deg_v - union_est
                upper = float(min(deg_u, deg_v))
                s_clamped = max(0.0, min(s_hat, upper))
                if s_hat < 0.0 or s_hat > upper:
                    n_clamped += 1
                total += s_clamped

                if record_details:
                    edge_details.append({
                        'u': u, 'v': v_int,
                        'deg_u': deg_u, 'deg_v': deg_v,
                        'proxy': proxy,
                        'method': 'hll',
                        's_hat': s_hat,
                        's_clamped': s_clamped,
                    })
            else:
                s = intersect_adaptive(nbrs_u, nbrs_v, r)
                total += s

                if record_details:
                    edge_details.append({
                        'u': u, 'v': v_int,
                        'deg_u': deg_u, 'deg_v': deg_v,
                        'proxy': proxy,
                        'method': 'exact',
                        's_hat': float(s),
                        's_clamped': float(s),
                    })

    elapsed = time.perf_counter() - t0

    stats = {
        'n_edges': n_edges,
        'n_approx': n_approx,
        'n_exact': n_edges - n_approx,
        'n_fallback': n_fallback,
        'n_clamped': n_clamped,
        'approx_coverage': n_approx / n_edges if n_edges > 0 else 0.0,
        'fallback_ratio': (
            n_fallback / (n_approx + n_fallback)
            if (n_approx + n_fallback) > 0 else 0.0
        ),
        'clamp_ratio': n_clamped / n_approx if n_approx > 0 else 0.0,
    }

    return total, elapsed, stats, edge_details
