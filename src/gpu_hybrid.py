"""GPU exact and selective Hybrid-HLL triangle counting kernels."""

import math
import time
import warnings

import numpy as np
from numba import cuda, int32, int64, uint8, float64

from .gpu_hll import _union_estimate_device


_THREADS = 256
# Shared reduction arrays in the CUDA kernels below are statically sized to
# 256 entries. Keep _THREADS fixed unless every shared array is changed too.
_FLOAT64_ATOMIC_WARNED = False


def _require_supported_block_size() -> None:
    if _THREADS != 256:
        raise ValueError(
            "gpu_hybrid kernels currently require _THREADS == 256 because "
            "their shared-memory reduction buffers are statically sized to 256."
        )


def _warn_if_float64_atomic_slow(cuda_module) -> None:
    global _FLOAT64_ATOMIC_WARNED
    if _FLOAT64_ATOMIC_WARNED:
        return
    cc = cuda_module.get_current_device().compute_capability
    if cc.major < 6:
        warnings.warn(
            "Hybrid kernels use float64 atomic add. On GPUs with compute "
            "capability < 6.0 this may be emulated and very slow.",
            RuntimeWarning,
            stacklevel=2,
        )
    _FLOAT64_ATOMIC_WARNED = True


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

@cuda.jit(device=True, inline=True)
def _bsearch_intersect(col_idx, u_start, u_end, v_start, v_end) -> int32:
    """Count the intersection of two sorted CSR ranges by binary search."""
    len_u = u_end - u_start
    len_v = v_end - v_start

    if len_u == 0 or len_v == 0:
        return int32(0)

    if len_u <= len_v:
        probe_s, probe_e = u_start, u_end
        tgt_s, tgt_e = v_start, v_end
    else:
        probe_s, probe_e = v_start, v_end
        tgt_s, tgt_e = u_start, u_end

    cnt = int32(0)
    for i in range(probe_s, probe_e):
        x = col_idx[i]
        lo = tgt_s
        hi = tgt_e
        while lo < hi:
            mid = (lo + hi) >> int32(1)
            val = col_idx[mid]
            if val < x:
                lo = mid + int32(1)
            elif val > x:
                hi = mid
            else:
                cnt += int32(1)
                break
    return cnt


@cuda.jit(device=True, inline=True)
def _use_hll_edge(u, v, degrees, has_sketch, gamma) -> bool:
    deg_u = degrees[u]
    deg_v = degrees[v]
    proxy = deg_u if deg_u < deg_v else deg_v
    return (
        (proxy >= gamma)
        and (has_sketch[u] == uint8(1))
        and (has_sketch[v] == uint8(1))
    )


# ---------------------------------------------------------------------------
# Exact baseline: one edge per thread, one global atomic per block
# ---------------------------------------------------------------------------

@cuda.jit
def exact_count_kernel(
    row_ptr,
    col_idx,
    edge_u,
    edge_v,
    n_edges,
    counts,    # int64 (1,)
):
    local_count = int64(0)

    gid = cuda.grid(1)
    if gid < n_edges:
        u = edge_u[gid]
        v = edge_v[gid]
        local_count = int64(_bsearch_intersect(
            col_idx,
            row_ptr[u], row_ptr[u + 1],
            row_ptr[v], row_ptr[v + 1],
        ))

    partial = cuda.shared.array(256, dtype=int64)
    lane = cuda.threadIdx.x
    partial[lane] = local_count
    cuda.syncthreads()

    stride = cuda.blockDim.x >> 1
    while stride > 0:
        if lane < stride:
            partial[lane] += partial[lane + stride]
        cuda.syncthreads()
        stride >>= 1

    if lane == 0:
        cuda.atomic.add(counts, 0, partial[0])


# ---------------------------------------------------------------------------
# Fused Hybrid kernel: retained for A/B tests, now with block reduction
# ---------------------------------------------------------------------------

@cuda.jit
def hybrid_count_kernel(
    row_ptr,
    col_idx,
    edge_u,
    edge_v,
    n_edges,
    degrees,
    sketch_regs,
    has_sketch,
    gamma,
    p,
    counts,       # float64 (1,)
    stats,        # int64 (3,) = [n_approx, n_exact, n_clamped]
):
    local_sum = float64(0.0)
    local_approx = int64(0)
    local_exact = int64(0)
    local_clamped = int64(0)

    gid = cuda.grid(1)
    if gid < n_edges:
        u = edge_u[gid]
        v = edge_v[gid]
        deg_u = degrees[u]
        deg_v = degrees[v]
        proxy = deg_u if deg_u < deg_v else deg_v

        if _use_hll_edge(u, v, degrees, has_sketch, gamma):
            m = int32(1) << int32(p)
            union_est = _union_estimate_device(u, v, p, m, sketch_regs)
            s = float64(deg_u) + float64(deg_v) - float64(union_est)
            upper = float64(proxy)
            if s < float64(0.0):
                s = float64(0.0)
                local_clamped = int64(1)
            elif s > upper:
                s = upper
                local_clamped = int64(1)
            local_sum = s
            local_approx = int64(1)
        else:
            local_sum = float64(_bsearch_intersect(
                col_idx,
                row_ptr[u], row_ptr[u + 1],
                row_ptr[v], row_ptr[v + 1],
            ))
            local_exact = int64(1)

    sum_s = cuda.shared.array(256, dtype=float64)
    approx_s = cuda.shared.array(256, dtype=int64)
    exact_s = cuda.shared.array(256, dtype=int64)
    clamped_s = cuda.shared.array(256, dtype=int64)
    lane = cuda.threadIdx.x
    sum_s[lane] = local_sum
    approx_s[lane] = local_approx
    exact_s[lane] = local_exact
    clamped_s[lane] = local_clamped
    cuda.syncthreads()

    stride = cuda.blockDim.x >> 1
    while stride > 0:
        if lane < stride:
            sum_s[lane] += sum_s[lane + stride]
            approx_s[lane] += approx_s[lane + stride]
            exact_s[lane] += exact_s[lane + stride]
            clamped_s[lane] += clamped_s[lane + stride]
        cuda.syncthreads()
        stride >>= 1

    if lane == 0:
        cuda.atomic.add(counts, 0, sum_s[0])
        cuda.atomic.add(stats, 0, approx_s[0])
        cuda.atomic.add(stats, 1, exact_s[0])
        cuda.atomic.add(stats, 2, clamped_s[0])


# ---------------------------------------------------------------------------
# Split Hybrid kernels: exact-routed and HLL-routed work launch separately
# ---------------------------------------------------------------------------

@cuda.jit
def hybrid_exact_part_kernel(
    row_ptr,
    col_idx,
    edge_u,
    edge_v,
    n_edges,
    degrees,
    has_sketch,
    gamma,
    counts,       # float64 (1,)
    stats,        # int64 (3,) = [n_approx, n_exact, n_clamped]
):
    local_sum = float64(0.0)
    local_exact = int64(0)

    gid = cuda.grid(1)
    if gid < n_edges:
        u = edge_u[gid]
        v = edge_v[gid]
        if not _use_hll_edge(u, v, degrees, has_sketch, gamma):
            local_sum = float64(_bsearch_intersect(
                col_idx,
                row_ptr[u], row_ptr[u + 1],
                row_ptr[v], row_ptr[v + 1],
            ))
            local_exact = int64(1)

    sum_s = cuda.shared.array(256, dtype=float64)
    exact_s = cuda.shared.array(256, dtype=int64)
    lane = cuda.threadIdx.x
    sum_s[lane] = local_sum
    exact_s[lane] = local_exact
    cuda.syncthreads()

    stride = cuda.blockDim.x >> 1
    while stride > 0:
        if lane < stride:
            sum_s[lane] += sum_s[lane + stride]
            exact_s[lane] += exact_s[lane + stride]
        cuda.syncthreads()
        stride >>= 1

    if lane == 0:
        cuda.atomic.add(counts, 0, sum_s[0])
        cuda.atomic.add(stats, 1, exact_s[0])


@cuda.jit
def hybrid_hll_part_kernel(
    edge_u,
    edge_v,
    n_edges,
    degrees,
    sketch_regs,
    has_sketch,
    gamma,
    p,
    counts,       # float64 (1,)
    stats,        # int64 (3,) = [n_approx, n_exact, n_clamped]
):
    local_sum = float64(0.0)
    local_approx = int64(0)
    local_clamped = int64(0)

    gid = cuda.grid(1)
    if gid < n_edges:
        u = edge_u[gid]
        v = edge_v[gid]
        if _use_hll_edge(u, v, degrees, has_sketch, gamma):
            deg_u = degrees[u]
            deg_v = degrees[v]
            proxy = deg_u if deg_u < deg_v else deg_v
            m = int32(1) << int32(p)
            union_est = _union_estimate_device(u, v, p, m, sketch_regs)
            s = float64(deg_u) + float64(deg_v) - float64(union_est)
            upper = float64(proxy)
            if s < float64(0.0):
                s = float64(0.0)
                local_clamped = int64(1)
            elif s > upper:
                s = upper
                local_clamped = int64(1)
            local_sum = s
            local_approx = int64(1)

    sum_s = cuda.shared.array(256, dtype=float64)
    approx_s = cuda.shared.array(256, dtype=int64)
    clamped_s = cuda.shared.array(256, dtype=int64)
    lane = cuda.threadIdx.x
    sum_s[lane] = local_sum
    approx_s[lane] = local_approx
    clamped_s[lane] = local_clamped
    cuda.syncthreads()

    stride = cuda.blockDim.x >> 1
    while stride > 0:
        if lane < stride:
            sum_s[lane] += sum_s[lane + stride]
            approx_s[lane] += approx_s[lane + stride]
            clamped_s[lane] += clamped_s[lane + stride]
        cuda.syncthreads()
        stride >>= 1

    if lane == 0:
        cuda.atomic.add(counts, 0, sum_s[0])
        cuda.atomic.add(stats, 0, approx_s[0])
        cuda.atomic.add(stats, 2, clamped_s[0])


# ---------------------------------------------------------------------------
# Python-level launchers
# ---------------------------------------------------------------------------

def gpu_exact_triangle_count(row_ptr_d, col_idx_d, edge_u_d, edge_v_d):
    """
    Launch exact_count_kernel and return (count, elapsed_seconds).

    The kernel uses block-level reduction, so global atomic traffic is one
    add per block rather than one add per edge.
    """
    from numba import cuda as _cuda

    _require_supported_block_size()
    n_edges = len(edge_u_d)
    counts_d = _cuda.to_device(np.zeros(1, dtype=np.int64))
    grid = math.ceil(n_edges / _THREADS)

    _cuda.synchronize()
    t0 = time.perf_counter()
    exact_count_kernel[grid, _THREADS](
        row_ptr_d, col_idx_d, edge_u_d, edge_v_d, n_edges, counts_d
    )
    _cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return int(counts_d.copy_to_host()[0]), elapsed


def gpu_hybrid_triangle_count(
    row_ptr_d,
    col_idx_d,
    edge_u_d,
    edge_v_d,
    degrees_d,
    sketch_regs_d,
    has_sketch_d,
    gamma: int,
    p: int,
    split_kernels: bool = True,
):
    """
    Launch Hybrid-HLL triangle counting.

    The default path launches two kernels: exact-routed work first, HLL-routed
    work second.  This keeps the heavy exact and HLL code paths out of the same
    runtime branch and uses block-level reduction in both kernels.

    Set split_kernels=False to run a fused kernel with the same block-level
    reduction but with the exact/HLL branch inside one kernel.

    Returns
    -------
    estimate : float
    elapsed_seconds : float
    stats : dict
    """
    from numba import cuda as _cuda

    _require_supported_block_size()
    _warn_if_float64_atomic_slow(_cuda)
    n_edges = len(edge_u_d)
    counts_d = _cuda.to_device(np.zeros(1, dtype=np.float64))
    stats_d = _cuda.to_device(np.zeros(3, dtype=np.int64))
    grid = math.ceil(n_edges / _THREADS)

    _cuda.synchronize()
    t0 = time.perf_counter()
    if split_kernels:
        hybrid_exact_part_kernel[grid, _THREADS](
            row_ptr_d, col_idx_d, edge_u_d, edge_v_d, n_edges,
            degrees_d, has_sketch_d, gamma, counts_d, stats_d,
        )
        hybrid_hll_part_kernel[grid, _THREADS](
            edge_u_d, edge_v_d, n_edges,
            degrees_d, sketch_regs_d, has_sketch_d,
            gamma, p, counts_d, stats_d,
        )
    else:
        hybrid_count_kernel[grid, _THREADS](
            row_ptr_d, col_idx_d, edge_u_d, edge_v_d, n_edges,
            degrees_d, sketch_regs_d, has_sketch_d,
            gamma, p, counts_d, stats_d,
        )
    _cuda.synchronize()
    elapsed = time.perf_counter() - t0

    estimate = float(counts_d.copy_to_host()[0])
    s = stats_d.copy_to_host()
    n_approx = int(s[0])
    n_exact = int(s[1])
    n_clamped = int(s[2])
    stats = {
        "n_approx": n_approx,
        "n_exact": n_exact,
        "n_clamped": n_clamped,
        "approx_coverage": n_approx / n_edges if n_edges > 0 else 0.0,
        "clamp_ratio": n_clamped / n_approx if n_approx > 0 else 0.0,
        "kernel_mode": (
            "split_block_reduction" if split_kernels
            else "fused_block_reduction"
        ),
    }
    return estimate, elapsed, stats
