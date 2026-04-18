"""GPU HLL sketch building and union estimation via Numba CUDA."""

import math
import numpy as np
from numba import cuda, uint64, int32, uint8, float32


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

@cuda.jit(device=True, inline=True)
def _splitmix64(x: uint64) -> uint64:
    MASK = uint64(0xFFFFFFFFFFFFFFFF)
    x = (x ^ (x >> uint64(30))) * uint64(0xBF58476D1CE4E5B9) & MASK
    x = (x ^ (x >> uint64(27))) * uint64(0x94D049BB133111EB) & MASK
    return x ^ (x >> uint64(31))


@cuda.jit(device=True, inline=True)
def _alpha(m: int32) -> float32:
    if m == 16:
        return float32(0.673)
    if m == 32:
        return float32(0.697)
    if m == 64:
        return float32(0.709)
    return float32(0.7213) / (float32(1.0) + float32(1.079) / float32(m))


@cuda.jit(device=True, inline=True)
def _bit_length(w: uint64) -> int32:
    """Number of bits needed to represent w (equivalent to floor(log2(w))+1)."""
    if w == uint64(0):
        return int32(0)
    count = int32(0)
    while w > uint64(0):
        w >>= uint64(1)
        count += int32(1)
    return count


# ---------------------------------------------------------------------------
# Sketch building kernel  (one thread per node)
# ---------------------------------------------------------------------------

@cuda.jit
def build_sketches_kernel(
    row_ptr,      # int32 (n_nodes+1,)
    col_idx,      # int32 (n_edges,)
    n_nodes,      # int
    theta,        # int  — min degree to receive a sketch
    p,            # int  — HLL precision (m = 2^p registers)
    sketch_regs,  # uint8 (n_nodes * m,) — flattened register array
    has_sketch,   # uint8 (n_nodes,)    — 1 if node has sketch
):
    u = cuda.grid(1)
    if u >= n_nodes:
        return

    deg = row_ptr[u + 1] - row_ptr[u]
    if deg < theta:
        has_sketch[u] = uint8(0)
        return

    has_sketch[u] = uint8(1)
    m = int32(1) << int32(p)
    remaining = int32(64 - p)
    base = u * m

    for k in range(row_ptr[u], row_ptr[u + 1]):
        x = uint64(col_idx[k])
        h = _splitmix64(x)
        j = int32(h >> uint64(remaining))          # top-p bits → register index
        w = h & uint64((uint64(1) << uint64(remaining)) - uint64(1))
        if w == uint64(0):
            rho = uint8(remaining + 1)
        else:
            rho = uint8(remaining - _bit_length(w) + 1)
        if rho > sketch_regs[base + j]:
            sketch_regs[base + j] = rho


# ---------------------------------------------------------------------------
# Union estimation  (one thread per edge, called inside hybrid kernel)
# ---------------------------------------------------------------------------

@cuda.jit(device=True, inline=True)
def _union_estimate_device(
    u, v, p, m,
    sketch_regs,  # uint8 (n_nodes * m,)
) -> float32:
    """
    Estimate |N+(u) ∪ N+(v)| from merged registers.
    Mirrors CPU _estimate_from_registers: raw HLL + LinearCounting small-range
    correction + large-range correction.
    """
    alpha = _alpha(m)
    Z = float32(0.0)
    V = int32(0)          # zero registers in merged sketch
    base_u = u * m
    base_v = v * m
    for j in range(m):
        ru = sketch_regs[base_u + j]
        rv = sketch_regs[base_v + j]
        r  = ru if ru > rv else rv
        Z += float32(2.0) ** (-float32(r))
        if r == uint8(0):
            V += int32(1)
    E = alpha * float32(m) * float32(m) / Z

    # Small-range correction (LinearCounting) — mirrors hll.py:43-47
    if E <= float32(2.5) * float32(m) and V > int32(0):
        E = float32(m) * math.log(float32(m) / float32(V))

    # Large-range correction for the 64-bit hash domain used by _splitmix64.
    two64 = float32(18446744073709551616.0)
    if E > two64 / float32(30.0):
        E = -two64 * math.log(float32(1.0) - E / two64)

    return E


# ---------------------------------------------------------------------------
# Expose numpy helper for CPU-side allocation
# ---------------------------------------------------------------------------

def alloc_sketch_arrays(n_nodes: int, p: int):
    """Return zeroed CPU arrays for sketch_regs and has_sketch."""
    m = 1 << p
    sketch_regs = np.zeros(n_nodes * m, dtype=np.uint8)
    has_sketch  = np.zeros(n_nodes,     dtype=np.uint8)
    return sketch_regs, has_sketch
