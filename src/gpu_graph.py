"""Convert forward adjacency dict to CSR arrays on GPU (CuPy / Numba)."""

import numpy as np


def to_csr_numpy(forward_adj: dict):
    """
    Convert forward_adj to CSR format (CPU numpy arrays).

    Returns
    -------
    row_ptr  : int32 array, shape (n_nodes + 1,)
    col_idx  : int32 array, shape (n_oriented_edges,)
    degrees  : int32 array, shape (n_nodes,)  — forward degree per node
    node_ids : int32 array, shape (n_nodes,)  — sorted node IDs
    edge_u   : int32 array, shape (n_oriented_edges,)
    edge_v   : int32 array, shape (n_oriented_edges,)
    """
    node_ids = np.array(sorted(forward_adj.keys()), dtype=np.int32)
    n_nodes = len(node_ids)
    node_index = {nid: i for i, nid in enumerate(node_ids)}

    degrees = np.array([len(forward_adj[nid]) for nid in node_ids], dtype=np.int32)
    row_ptr = np.zeros(n_nodes + 1, dtype=np.int32)
    row_ptr[1:] = np.cumsum(degrees)
    n_edges = int(row_ptr[-1])

    col_idx = np.empty(n_edges, dtype=np.int32)
    edge_u  = np.empty(n_edges, dtype=np.int32)
    edge_v  = np.empty(n_edges, dtype=np.int32)

    for i, nid in enumerate(node_ids):
        start = row_ptr[i]
        nbrs  = forward_adj[nid]
        for k, v in enumerate(nbrs):
            j = node_index[int(v)]
            col_idx[start + k] = j
            edge_u[start + k]  = i
            edge_v[start + k]  = j

    return row_ptr, col_idx, degrees, node_ids, edge_u, edge_v


def to_csr_gpu(forward_adj: dict):
    """
    Same as to_csr_numpy but returns CuPy arrays on GPU.
    Requires cupy to be installed.
    """
    import cupy as cp
    row_ptr, col_idx, degrees, node_ids, edge_u, edge_v = to_csr_numpy(forward_adj)
    return (cp.asarray(row_ptr), cp.asarray(col_idx), cp.asarray(degrees),
            cp.asarray(node_ids), cp.asarray(edge_u), cp.asarray(edge_v))
