"""Graph loading and orientation for triangle counting."""

import numpy as np
from collections import defaultdict
from typing import Dict, Set, Tuple


def load_edgelist(filepath: str, comment_chars: str = '#%') -> Set[Tuple[int, int]]:
    """
    Load undirected graph from edge list file.
    Returns canonical edge set (u < v for every edge).
    Skips comment lines and self-loops.
    """
    edges: Set[Tuple[int, int]] = set()
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line[0] in comment_chars:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            if u != v:
                edges.add((min(u, v), max(u, v)))
    return edges


def orient_graph(
    edges: Set[Tuple[int, int]],
) -> Tuple[Dict[int, np.ndarray], Dict[int, int]]:
    """
    Orient edges by (degree, node_id) total order.
    For edge (u,v): u -> v if (deg[u], u) < (deg[v], v).

    Returns
    -------
    forward_adj : dict  node -> sorted np.int64 array of forward neighbours
    degree      : dict  node -> undirected degree
    """
    degree: Dict[int, int] = defaultdict(int)
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1

    forward_raw: Dict[int, list] = defaultdict(list)
    nodes: Set[int] = set()
    for u, v in edges:
        nodes.add(u)
        nodes.add(v)
        if (degree[u], u) < (degree[v], v):
            forward_raw[u].append(v)
        else:
            forward_raw[v].append(u)

    # Every node gets an entry (empty array if no forward neighbours)
    forward_adj: Dict[int, np.ndarray] = {
        node: np.array(sorted(forward_raw[node]), dtype=np.int64)
        for node in nodes
    }
    return forward_adj, dict(degree)


def make_small_graph(triangles: list) -> Set[Tuple[int, int]]:
    """
    Build edge set from a list of triangle tuples (a, b, c).
    Useful for sanity-check graphs.
    """
    edges: Set[Tuple[int, int]] = set()
    for a, b, c in triangles:
        edges.add((min(a, b), max(a, b)))
        edges.add((min(b, c), max(b, c)))
        edges.add((min(a, c), max(a, c)))
    return edges
