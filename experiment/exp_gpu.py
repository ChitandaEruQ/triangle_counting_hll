"""
GPU Experiment — Complete and Comprehensive Measurement.

Methods
-------
1. [validate only] CPU Exact   exact ground truth on small graphs
2. GPU Exact (ours)            edge-parallel binary-search kernel
3. cuGraph Exact               NVIDIA cuGraph baseline (optional)
4. DOULION q=0.10              CPU approximate, 10 seeds
5. DOULION q=0.05              CPU approximate, more aggressive, 10 seeds
6. GPU Hybrid p=8  γ=64
7. GPU Hybrid p=8  γ=128
8. GPU Hybrid p=8  γ=256
9. GPU Hybrid p=10 γ=64
10. GPU Hybrid p=10 γ=128      higher HLL precision

Ground truth policy
-------------------
  For well-known benchmark graphs the exact triangle count is publicly
  available (e.g. SNAP lists com-Orkut = 627,584,181).  Pass it with
  --truth so no CPU Exact run is needed on the large graph.

  For the CPU speedup reference, pass --cpu-ref-time (seconds) from a
  published competitive CPU baseline (e.g. Shun et al. multicore C++).
  A Python/numpy CPU Exact is NOT a fair CPU baseline.

  Use --validate only on small graphs to confirm that GPU Exact matches
  the CPU Exact count before running the full benchmark.

  Truth source priority:
    1. --truth VALUE             (published / pre-verified count)
    2. --validate result         (CPU Exact on the target graph)
    3. GPU Exact (fallback)      (if neither above is provided)

Metrics (per method)
--------------------
  estimate, rel_error, bias,
  count_time_mean ± std  (GPU kernels / cuGraph: N_TIMING_RUNS repeats;
                          DOULION: N_SEEDS runs)
  e2e_time
  spdup_cnt / spdup_e2e  vs cpu_ref            — motivation: why GPU?
  spdup_vs_gpu_exact                           — GPU-to-GPU kernel comparison
  spdup_vs_cugraph                             — vs industry-standard GPU baseline
  approx_coverage (= HLL-routed oriented edges / total oriented edges)
  clamp_ratio     (= fraction of HLL estimates clipped to [0, min(deg_u,deg_v)])
  n_sketched      (Hybrid only)

E2E scope
---------
  E2E includes: device transfer (H2D) + method-specific GPU preprocessing
  (sketch allocation and build for Hybrid) + kernel execution.
  Excludes: file I/O, graph parsing, and orientation (shared preprocessing
  identical for all GPU methods).

Usage
-----
  # Large graph — use SNAP ground truth + literature CPU reference
  python experiment/exp_gpu.py datasets/com-orkut.ungraph.txt results/ \\
      --truth 627584181 --cpu-ref-time 30.0 --cpu-ref-label "Shun2015-multicore"

  # Small graph — run CPU Exact to validate GPU kernel, then benchmark
  python experiment/exp_gpu.py datasets/small.txt results/ --validate

  # Skip cuGraph if RAPIDS not installed
  python experiment/exp_gpu.py <graph> [outdir] --skip-cugraph
"""

import gc
import json
import math
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from src.exact import exact_triangle_count
from src.utils import relative_error, bias

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------

N_TIMING_RUNS = 3          # GPU kernel repetitions for stable timing
GAMMA_SWEEP   = [64, 128, 256]
P_SWEEP       = [8, 10]    # HLL precision; p=8 → 256 regs, p=10 → 1024 regs
DOULION_QS    = [0.10, 0.05]
DOULION_SEEDS = 10
COMMENT_CHARS = "#%"

# ---------------------------------------------------------------------------
# GPU imports (graceful fallback if numba not available)
# ---------------------------------------------------------------------------

_GPU_IMPORT_ERROR = None
try:
    from src.gpu_hybrid import gpu_exact_triangle_count, gpu_hybrid_triangle_count
    from src.gpu_hll import build_sketches_kernel, alloc_sketch_arrays
except ModuleNotFoundError as exc:
    if exc.name != "numba":
        raise
    _GPU_IMPORT_ERROR = exc
    gpu_exact_triangle_count  = None
    gpu_hybrid_triangle_count = None
    build_sketches_kernel     = None
    alloc_sketch_arrays       = None


def _require_gpu():
    if _GPU_IMPORT_ERROR is not None:
        raise RuntimeError(
            "GPU experiment requires numba with CUDA support."
        ) from _GPU_IMPORT_ERROR
    from numba import cuda
    if not cuda.is_available():
        raise RuntimeError(
            "Numba CUDA is not available. "
            "Set CUDA_HOME to the CUDA toolkit path and retry."
        )
    return cuda


# ---------------------------------------------------------------------------
# Streaming graph loader  (no Python Set[Tuple], memory-efficient)
# ---------------------------------------------------------------------------

def _iter_edges_raw(path: str):
    """Yield (u, v) for every non-comment, non-self-loop edge."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line[0] in COMMENT_CHARS:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            if u != v:
                yield u, v


def _load_graph_legacy(graph_path: str):
    """
    Two-pass streaming load.  Returns:
      row_ptr_h  : int32  (n_nodes+1,)
      col_idx_h  : int32  (n_oriented,)
      degrees_h  : int32  (n_nodes,)    forward degree
      edge_u_h   : int32  (n_oriented,)
      edge_v_h   : int32  (n_oriented,)
      forward_adj: dict[int, np.ndarray]  remapped indices, for CPU Exact
      edges_np   : int32  (n_edges, 2)   undirected canonical pairs (orig IDs)
                   kept for cuGraph — must not be freed before cuGraph runs
      node_ids   : int64  (n_nodes,)
      n_nodes    : int
      n_oriented : int
      t_load     : float  (seconds)

    Memory budget for Orkut (117 M edges):
      edges_buf  ≈ 117 M × 8 B ≈ 940 MB  (freed after dedup)
      edges_np   ≈ 940 MB  (kept for cuGraph)
      forward_adj col data ≈ 940 MB
      CSR col_idx ≈ 470 MB  (forward edges only)
      Peak ≈ 3–4 GB
    """
    print("\n[0] Pass 1/2 — reading edges, computing degrees ...")
    t0 = time.perf_counter()

    # Pre-allocate: one file scan to count
    n_raw = sum(1 for _ in _iter_edges_raw(graph_path))
    edges_buf = np.empty((n_raw, 2), dtype=np.int32)
    degree: dict = {}
    i = 0
    for u, v in _iter_edges_raw(graph_path):
        edges_buf[i, 0] = min(u, v)
        edges_buf[i, 1] = max(u, v)
        degree[u] = degree.get(u, 0) + 1
        degree[v] = degree.get(v, 0) + 1
        i += 1

    # Deduplicate (canonical u < v pairs)
    edges_np = np.unique(edges_buf[:i], axis=0)
    del edges_buf
    gc.collect()

    n_edges = len(edges_np)
    t1 = time.perf_counter()
    print(f"  RawLines={i:,}  UniqueEdges={n_edges:,}  t={t1-t0:.2f}s")

    # ── Build node index ──────────────────────────────────────────────────
    node_ids = np.union1d(edges_np[:, 0], edges_np[:, 1]).astype(np.int64)
    n_nodes  = len(node_ids)
    max_node = int(node_ids.max())
    # Direct-address map: nidx[original_id] = remapped_index
    nidx = np.full(max_node + 1, -1, dtype=np.int32)
    nidx[node_ids] = np.arange(n_nodes, dtype=np.int32)

    print(f"\n[0] Pass 2/2 — orienting and building CSR ...")

    # ── Orient all edges ──────────────────────────────────────────────────
    u_arr = edges_np[:, 0].astype(np.int64)
    v_arr = edges_np[:, 1].astype(np.int64)

    # Vectorised degree lookup via direct-address array
    max_node64 = int(node_ids.max())
    deg_arr = np.zeros(max_node64 + 1, dtype=np.int32)
    for nid, d in degree.items():
        deg_arr[nid] = d

    deg_u = deg_arr[u_arr]
    deg_v = deg_arr[v_arr]
    u_first = (deg_u < deg_v) | ((deg_u == deg_v) & (u_arr < v_arr))
    src_orig = np.where(u_first, u_arr, v_arr)
    dst_orig = np.where(u_first, v_arr, u_arr)
    src_i = nidx[src_orig].astype(np.int32)
    dst_i = nidx[dst_orig].astype(np.int32)

    del u_arr, v_arr, deg_u, deg_v, u_first, src_orig, dst_orig
    gc.collect()

    # ── Build CSR ─────────────────────────────────────────────────────────
    forward_deg = np.bincount(src_i, minlength=n_nodes).astype(np.int32)
    row_ptr_h   = np.zeros(n_nodes + 1, dtype=np.int32)
    row_ptr_h[1:] = np.cumsum(forward_deg)
    n_oriented  = int(row_ptr_h[-1])

    col_idx_h = np.empty(n_oriented, dtype=np.int32)
    edge_u_h  = np.empty(n_oriented, dtype=np.int32)
    edge_v_h  = np.empty(n_oriented, dtype=np.int32)

    # Fill CSR: sort neighbours per row for binary-search correctness
    order = np.lexsort((dst_i, src_i))
    src_sorted = src_i[order]
    dst_sorted = dst_i[order]
    col_idx_h[:] = dst_sorted
    edge_u_h[:]  = src_sorted
    edge_v_h[:]  = dst_sorted

    del src_i, dst_i, order, src_sorted, dst_sorted
    gc.collect()

    # ── Reconstruct forward_adj for CPU Exact ─────────────────────────────
    # forward_adj[i] = sorted int64 array of forward neighbours (remapped idx)
    forward_adj: dict = {}
    for i in range(n_nodes):
        s, e = int(row_ptr_h[i]), int(row_ptr_h[i + 1])
        forward_adj[i] = col_idx_h[s:e].astype(np.int64)

    t_load = time.perf_counter() - t0
    n_oriented_actual = sum(len(v) for v in forward_adj.values())
    print(f"  Nodes={n_nodes:,}  OrientedEdges={n_oriented_actual:,}  "
          f"total_load={t_load:.2f}s")

    del deg_arr, degree, nidx
    gc.collect()

    return (row_ptr_h, col_idx_h, forward_deg, edge_u_h, edge_v_h,
            forward_adj, edges_np, node_ids, n_nodes, n_oriented, t_load)


def _load_graph(graph_path: str,
                need_forward_adj: bool = False,
                keep_edges_np: bool = True,
                dedup_edges: bool = True):
    """
    Large-graph loader used by the GPU experiment.

    Expensive optional objects are built only when needed:
      - forward_adj is only for CPU Exact validation.
      - edges_np is only retained for cuGraph.

    The first pass still counts edges so the main edge array can be allocated
    once, but every long stage prints with flush=True so notebook runs show
    where time is being spent.
    """
    log_every = 10_000_000
    print("\n[0] Loading graph on CPU ...", flush=True)
    t0 = time.perf_counter()

    print("  counting raw edges ...", flush=True)
    n_raw = 0
    for _ in _iter_edges_raw(graph_path):
        n_raw += 1
        if n_raw % log_every == 0:
            print(f"    counted {n_raw:,} edges ...", flush=True)
    print(f"  raw_edges={n_raw:,}; allocating edge buffer ...", flush=True)

    edges_buf = np.empty((n_raw, 2), dtype=np.int32)

    print("  filling canonical edge buffer ...", flush=True)
    i = 0
    for u, v in _iter_edges_raw(graph_path):
        edges_buf[i, 0] = min(u, v)
        edges_buf[i, 1] = max(u, v)
        i += 1
        if i % log_every == 0:
            print(f"    filled {i:,}/{n_raw:,} edges ...", flush=True)

    if dedup_edges:
        print("  deduplicating canonical edges with np.unique ...", flush=True)
        edges_np = np.unique(edges_buf[:i], axis=0)
        del edges_buf
        gc.collect()
    else:
        print("  skipping dedup (--no-dedup); assuming unique input edges ...",
              flush=True)
        edges_np = edges_buf[:i]
        del edges_buf

    n_edges = len(edges_np)
    print(f"  RawEdges={i:,}  UsedEdges={n_edges:,}  t={time.perf_counter()-t0:.2f}s",
          flush=True)

    print("  building node index ...", flush=True)
    node_ids = np.union1d(edges_np[:, 0], edges_np[:, 1]).astype(np.int64)
    n_nodes = len(node_ids)
    max_node = int(node_ids.max())
    nidx = np.full(max_node + 1, -1, dtype=np.int32)
    nidx[node_ids] = np.arange(n_nodes, dtype=np.int32)

    print("  remapping endpoints and computing undirected degrees ...", flush=True)
    remap_u = nidx[edges_np[:, 0]]
    remap_v = nidx[edges_np[:, 1]]
    degree_undir = (
        np.bincount(remap_u, minlength=n_nodes)
        + np.bincount(remap_v, minlength=n_nodes)
    ).astype(np.int32)

    print("  orienting edges by (undirected degree, node id) ...", flush=True)
    deg_u = degree_undir[remap_u]
    deg_v = degree_undir[remap_v]
    u_first = (deg_u < deg_v) | (
        (deg_u == deg_v) & (edges_np[:, 0] < edges_np[:, 1])
    )
    src_i = np.where(u_first, remap_u, remap_v).astype(np.int32)
    dst_i = np.where(u_first, remap_v, remap_u).astype(np.int32)

    del remap_u, remap_v, degree_undir, deg_u, deg_v, u_first, nidx
    gc.collect()

    if keep_edges_np:
        edges_keep = edges_np
    else:
        print("  releasing undirected edge array (cuGraph skipped) ...", flush=True)
        edges_keep = None
        del edges_np
        gc.collect()

    print("  building CSR row pointers ...", flush=True)
    forward_deg = np.bincount(src_i, minlength=n_nodes).astype(np.int32)
    row_ptr_h = np.zeros(n_nodes + 1, dtype=np.int32)
    row_ptr_h[1:] = np.cumsum(forward_deg)
    n_oriented = int(row_ptr_h[-1])

    print("  sorting oriented edges by (src, dst) ...", flush=True)
    order = np.lexsort((dst_i, src_i))
    src_sorted = src_i[order]
    dst_sorted = dst_i[order]

    col_idx_h = dst_sorted.astype(np.int32, copy=False)
    edge_u_h  = src_sorted.astype(np.int32, copy=False)
    edge_v_h  = dst_sorted.astype(np.int32, copy=True)  # must be independent copy

    del src_i, dst_i, order, src_sorted, dst_sorted
    gc.collect()

    forward_adj = None
    if need_forward_adj:
        print("  reconstructing forward_adj for CPU Exact validation ...",
              flush=True)
        forward_adj = {}
        for i_node in range(n_nodes):
            s, e = int(row_ptr_h[i_node]), int(row_ptr_h[i_node + 1])
            forward_adj[i_node] = col_idx_h[s:e].astype(np.int64)

    t_load = time.perf_counter() - t0
    print(
        f"  Nodes={n_nodes:,}  OrientedEdges={n_oriented:,}  "
        f"forward_adj={'yes' if forward_adj is not None else 'no'}  "
        f"edges_for_cugraph={'yes' if edges_keep is not None else 'no'}  "
        f"total_load={t_load:.2f}s",
        flush=True,
    )

    return (row_ptr_h, col_idx_h, forward_deg, edge_u_h, edge_v_h,
            forward_adj, edges_keep, node_ids, n_nodes, n_oriented, t_load)


# ---------------------------------------------------------------------------
# DOULION — samples from forward edges (memory-efficient, mathematically equiv)
# ---------------------------------------------------------------------------

def _doulion_forward(edge_u_h: np.ndarray, edge_v_h: np.ndarray,
                     q: float, n_seeds: int = DOULION_SEEDS,
                     r: float = 8.0, base_seed: int = 42) -> dict:
    """
    DOULION using the already-oriented forward edge list.

    Each undirected edge appears exactly once in forward orientation, so
    sampling forward edges with prob q is identical to sampling undirected
    edges.  After sampling we can re-use the original orientation (it remains
    a valid DAG on the sub-graph), giving the same triangle count as
    re-orienting from scratch.

    Returns the same dict as doulion_repeated.
    """
    n_fwd = len(edge_u_h)
    estimates, times = [], []

    for seed in range(base_seed, base_seed + n_seeds):
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()

        mask = rng.random(n_fwd) < q
        s_u  = edge_u_h[mask]
        s_v  = edge_v_h[mask]

        if len(s_u) == 0:
            estimates.append(0.0)
            times.append(time.perf_counter() - t0)
            continue

        # Build forward_adj for the sample (numpy, no Python loops on full graph)
        order     = np.argsort(s_u, kind='stable')
        sorted_u  = s_u[order]
        sorted_v  = s_v[order]
        splits    = np.where(np.diff(sorted_u))[0] + 1
        unique_u  = sorted_u[np.concatenate([[0], splits])].tolist()
        grp_v     = np.split(sorted_v, splits)

        fwd = {int(u): np.sort(grp).astype(np.int64)
               for u, grp in zip(unique_u, grp_v)}

        count, _, _ = exact_triangle_count(fwd, r=r)
        estimates.append(count / (q ** 3))
        times.append(time.perf_counter() - t0)

    arr_e = np.array(estimates, dtype=np.float64)
    arr_t = np.array(times,     dtype=np.float64)
    return {
        'q': q, 'n_seeds': n_seeds,
        'mean_estimate': float(arr_e.mean()),
        'std_estimate':  float(arr_e.std()),
        'mean_time':     float(arr_t.mean()),
        'std_time':      float(arr_t.std()),
        'estimates': estimates, 'times': times,
    }


# ---------------------------------------------------------------------------
# cuGraph wrapper
# ---------------------------------------------------------------------------

def _cugraph_count(edges_np: np.ndarray) -> tuple:
    """
    Return (count, elapsed) using cuGraph (full e2e timing incl. graph build),
    or (None, None) if cuGraph is not installed.

    Parameters
    ----------
    edges_np : int32 array shape (n_edges, 2) — undirected canonical pairs
               with original node IDs.  cuGraph needs the full undirected
               graph; passing only forward-oriented edges would give wrong
               counts.
    """
    try:
        import cudf
        import cugraph

        # Build graph once (amortised), time the count kernel N_TIMING_RUNS times
        df = cudf.DataFrame({'src': edges_np[:, 0], 'dst': edges_np[:, 1]})
        G  = cugraph.Graph()
        t_build0 = time.perf_counter()
        G.from_cudf_edgelist(df, source='src', destination='dst')
        t_build  = time.perf_counter() - t_build0

        cnt_times = []
        count = None
        for _ in range(N_TIMING_RUNS):
            t0     = time.perf_counter()
            result = cugraph.triangle_count(G)
            cnt_times.append(time.perf_counter() - t0)
            count  = int(result['counts'].sum()) // 3

        cnt_mean = float(np.mean(cnt_times))
        cnt_std  = float(np.std(cnt_times))
        # e2e = graph build + mean count time (mirrors how GPU Exact e2e is computed)
        elapsed  = t_build + cnt_mean
        return count, elapsed, cnt_mean, cnt_std, t_build
    except ImportError:
        return None, None
    except Exception as exc:
        print(f"  cuGraph failed: {exc}")
        return None, None


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

def _gpu_warmup(cuda, row_ptr_d, col_idx_d, edge_u_d, edge_v_d,
                degrees_d, n_nodes: int,
                split_hybrid: bool = True) -> None:
    """
    Run each kernel once (discarded) to trigger Numba JIT compilation.
    Without warmup, the first timed run includes compilation overhead.
    """
    print("  [warmup] JIT compiling GPU kernels ...")
    t0 = time.perf_counter()
    gpu_exact_triangle_count(row_ptr_d, col_idx_d, edge_u_d, edge_v_d)
    # Warmup for hybrid: use smallest gamma/p so all branches compile
    sk_h, hs_h = alloc_sketch_arrays(n_nodes, p=8)
    sk_d = cuda.to_device(sk_h)
    hs_d = cuda.to_device(hs_h)
    threads = 256
    grid    = math.ceil(n_nodes / threads)
    cuda.synchronize()
    build_sketches_kernel[grid, threads](
        row_ptr_d, col_idx_d, n_nodes, 64, 8, sk_d, hs_d)
    cuda.synchronize()
    # Warm both Hybrid modes so custom in-process A/B runs do not pay JIT
    # compilation on the first measured iteration of the second mode.
    gpu_hybrid_triangle_count(
        row_ptr_d, col_idx_d, edge_u_d, edge_v_d,
        degrees_d, sk_d, hs_d, gamma=64, p=8,
        split_kernels=split_hybrid)
    gpu_hybrid_triangle_count(
        row_ptr_d, col_idx_d, edge_u_d, edge_v_d,
        degrees_d, sk_d, hs_d, gamma=64, p=8,
        split_kernels=not split_hybrid)
    del sk_d, hs_d
    cuda.synchronize()
    print(f"  [warmup] done in {time.perf_counter()-t0:.2f}s")


def _gpu_build_sketches(cuda, row_ptr_d, col_idx_d,
                        n_nodes: int, gamma: int, p: int) -> tuple:
    """
    Allocate + H2D + build HLL sketches on GPU.
    Returns (sketch_regs_d, has_sketch_d, build_time, n_sketched, alloc_time).
    """
    t_alloc0 = time.perf_counter()
    sk_h, hs_h = alloc_sketch_arrays(n_nodes, p)
    sk_d = cuda.to_device(sk_h)
    hs_d = cuda.to_device(hs_h)
    cuda.synchronize()
    alloc_time = time.perf_counter() - t_alloc0

    threads = 256
    grid    = math.ceil(n_nodes / threads)
    cuda.synchronize()
    t0 = time.perf_counter()
    build_sketches_kernel[grid, threads](
        row_ptr_d, col_idx_d, n_nodes, gamma, p, sk_d, hs_d)
    cuda.synchronize()
    build_time = time.perf_counter() - t0

    n_sketched = int(np.sum(hs_d.copy_to_host()))
    return sk_d, hs_d, build_time, n_sketched, alloc_time


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

def _fmt(val, fmt, na="N/A"):
    return format(val, fmt) if val is not None else na


def _print_table(rows: list, ref_label: str, ref_time: float,
                 gpu_exact_time: float = None,
                 cugraph_time: float = None) -> None:
    """
    Print summary table with three speedup reference columns:
      vs <ref_label>  — primary baseline (CPU Exact or GPU Exact)
      vs GPU Exact    — GPU-to-GPU comparison
      vs cuGraph      — vs industry-standard GPU baseline

    coverage  = HLL-routed oriented edges / total oriented edges
    clamp     = fraction of HLL estimates clipped to [0, min(deg_u, deg_v)]
    """
    def _spd(t_ref, t_method):
        if t_ref and t_method and t_method > 0:
            return f"{t_ref/t_method:8.2f}x"
        return f"{'N/A':>9}"

    hdr = (f"{'Method':<30}  {'Estimate':>14}  {'RelErr':>8}  {'Bias':>13}  "
           f"{'vs ' + (ref_label or 'ref'):>11}  "
           f"{'vs GPUExact':>11}  {'vs cuGraph':>10}  "
           f"{'CntMean':>8}  {'±Std':>6}  {'E2E':>8}  "
           f"{'Cov%':>6}  {'Clamp%':>7}")
    sep = "=" * len(hdr)
    ref_str = f"{ref_time:.3f}s" if ref_time else "N/A"
    gex_str = f"{gpu_exact_time:.3f}s" if gpu_exact_time else "N/A"
    cgx_str = f"{cugraph_time:.3f}s"   if cugraph_time   else "N/A"
    print(f"\n{sep}")
    print(f"SUMMARY  |  ref={ref_label}({ref_str})  "
          f"GPUExact({gex_str})  cuGraph({cgx_str})")
    print(sep)
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        est   = _fmt(r.get('estimate'),  "14,.0f")
        re    = _fmt(r.get('rel_error'), "8.5f")
        b     = _fmt(r.get('bias'),      "13,.0f")
        s_ref = _spd(ref_time,        r.get('time'))
        s_gex = _spd(gpu_exact_time,  r.get('time'))
        s_cgx = _spd(cugraph_time,    r.get('time'))
        ct    = (_fmt(r.get('time'),     "8.3f") + "s"
                 if r.get('time')    is not None else f"{'N/A':>9}")
        std   = (_fmt(r.get('time_std'), "6.3f") + "s"
                 if r.get('time_std') is not None else f"{'N/A':>7}")
        e2e   = (_fmt(r.get('e2e_time'), "8.3f") + "s"
                 if r.get('e2e_time') is not None else f"{'N/A':>9}")
        cov   = (f"{r['coverage']*100:6.2f}%"
                 if r.get('coverage', 0) > 0 else f"{'—':>7}")
        clamp = (f"{r['clamp_ratio']*100:7.2f}%"
                 if r.get('clamp_ratio') is not None else f"{'—':>8}")
        print(f"{r['method']:<30}  {est}  {re}  {b}  "
              f"{s_ref}  {s_gex}  {s_cgx}  "
              f"{ct}  {std}  {e2e}  {cov}  {clamp}")
    print(sep)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_exp_gpu(graph_path: str,
                output_dir: str = 'results',
                skip_cugraph: bool = False,
                known_truth: int = None,
                cpu_ref_time: float = None,
                cpu_ref_label: str = None,
                validate: bool = False,
                split_hybrid: bool = True,
                dedup_edges: bool = True) -> dict:
    """
    Parameters
    ----------
    known_truth   : pre-verified exact triangle count (e.g. from SNAP).
                    If provided, used directly as ground truth without
                    running CPU Exact on the target graph.
    cpu_ref_time  : CPU reference time in seconds from published literature.
                    Used as the denominator for spdup_cnt / spdup_e2e.
                    If None, GPU Exact time is used as the reference.
    cpu_ref_label : citation label for cpu_ref_time (e.g. "Shun2015-8core").
    validate      : if True, run CPU Exact on this graph and compare with
                    known_truth (or use as truth if known_truth is None).
                    Recommended only for small graphs.
    split_hybrid  : if True, run exact-routed and HLL-routed Hybrid work in
                    separate kernels. If False, use the fused Hybrid kernel.
    dedup_edges   : if True, deduplicate canonical undirected edges before
                    orientation. Disable only for trusted unique edge lists.
    """
    os.makedirs(output_dir, exist_ok=True)
    cuda = _require_gpu()

    print("=" * 70)
    print("GPU EXPERIMENT — Comprehensive Measurement")
    print(f"Graph      : {graph_path}")
    print(f"Timing runs: {N_TIMING_RUNS}  |  γ sweep: {GAMMA_SWEEP}"
          f"  |  p sweep: {P_SWEEP}")
    print(f"Hybrid mode: {'split kernels' if split_hybrid else 'fused kernel'}")
    print(f"Load opts  : dedup={dedup_edges}  keep_cugraph_edges={not skip_cugraph}")
    if known_truth is not None:
        print(f"Truth      : {known_truth:,} (pre-verified, no CPU Exact needed)")
    if cpu_ref_time is not None:
        lbl = cpu_ref_label or "literature"
        print(f"CPU ref    : {cpu_ref_time:.3f}s  [{lbl}]")
    print("=" * 70)

    # ── [0] Load ──────────────────────────────────────────────────────────
    (row_ptr_h, col_idx_h, degrees_h, edge_u_h, edge_v_h,
     forward_adj, edges_np, node_ids, n_nodes, n_oriented, t_load) = _load_graph(
         graph_path,
         need_forward_adj=(validate and known_truth is None),
         keep_edges_np=not skip_cugraph,
         dedup_edges=dedup_edges)

    # ── [0b] H2D ──────────────────────────────────────────────────────────
    print("\n[0b] H2D transfer ...")
    t_h2d0    = time.perf_counter()
    row_ptr_d = cuda.to_device(row_ptr_h)
    col_idx_d = cuda.to_device(col_idx_h)
    degrees_d = cuda.to_device(degrees_h)
    edge_u_d  = cuda.to_device(edge_u_h)
    edge_v_d  = cuda.to_device(edge_v_h)
    cuda.synchronize()
    t_h2d = time.perf_counter() - t_h2d0
    print(f"  H2D: {t_h2d:.3f}s")

    # ── Warmup ────────────────────────────────────────────────────────────
    print("\n[warmup] Triggering JIT ...")
    _gpu_warmup(cuda, row_ptr_d, col_idx_d, edge_u_d, edge_v_d,
                degrees_d, n_nodes, split_hybrid=split_hybrid)

    rows: list = []
    truth        = None
    truth_source = None
    ref_time     = None   # speedup denominator (CPU Exact count_time, or GPU Exact)
    ref_label    = None
    # These are filled once we know the actual times, used for cross-speedup
    _gpu_exact_cnt_time = None   # kernel-only time for GPU Exact
    _cugraph_cnt_time   = None   # kernel-only time for cuGraph

    def _add_row(method, estimate, rel_err, b,
                 count_time, count_time_std=None,
                 e2e_time=None, coverage=0.0, **kw):
        sc = ref_time / count_time if ref_time and count_time else None
        se = ref_time / e2e_time   if ref_time and e2e_time   else None
        rows.append({
            'method':       method,
            'estimate':     estimate,
            'rel_error':    rel_err,
            'bias':         b,
            'spdup_cnt':    sc,
            'spdup_e2e':    se,
            # cross-speedup placeholders — filled in post-processing
            'spdup_vs_gpu_exact': None,
            'spdup_vs_cugraph':   None,
            'time':         count_time,
            'time_std':     count_time_std,
            'e2e_time':     e2e_time if e2e_time is not None else count_time,
            'coverage':     coverage,
            **kw,
        })

    # ── [1] Ground Truth / CPU Exact ──────────────────────────────────────
    if known_truth is not None:
        truth        = known_truth
        truth_source = 'published (e.g. SNAP)'
        print(f"\n[1] Using pre-verified truth: {truth:,}  (skipping CPU Exact)")
        if validate:
            print("    NOTE: --validate ignored when --truth is supplied.")
    elif validate:
        print(f"\n[1] VALIDATE — CPU Exact (1 run) ...")
        cpu_count, cpu_time, _ = exact_triangle_count(forward_adj)
        print(f"  count={cpu_count:,}  time={cpu_time:.3f}s")
        truth        = cpu_count
        truth_source = 'CPU Exact (validation run)'
        _add_row('CPU Exact', cpu_count, 0.0, 0,
                 count_time=cpu_time, count_time_std=None)
    else:
        truth        = None   # will be set from GPU Exact below
        truth_source = 'GPU Exact'
        print("\n[1] No --truth / --validate — truth will be set from GPU Exact.")

    # Set speedup reference from literature CPU time if supplied
    if cpu_ref_time is not None:
        ref_time  = cpu_ref_time
        ref_label = cpu_ref_label or 'CPU literature'
        print(f"[1] CPU reference: {ref_time:.3f}s  [{ref_label}]")

    # ── [2] GPU Exact ─────────────────────────────────────────────────────
    print(f"\n[2] GPU Exact — our kernel ({N_TIMING_RUNS} runs) ...")
    gpu_ex_times = []
    gpu_ex_count = None
    for run in range(N_TIMING_RUNS):
        cnt, t = gpu_exact_triangle_count(row_ptr_d, col_idx_d,
                                          edge_u_d, edge_v_d)
        gpu_ex_times.append(t)
        gpu_ex_count = cnt
        print(f"  run {run+1}: count={cnt:,}  kernel={t:.3f}s")
    gpu_ex_mean = float(np.mean(gpu_ex_times))
    gpu_ex_std  = float(np.std(gpu_ex_times))
    _gpu_exact_cnt_time = gpu_ex_mean   # for cross-speedup post-processing

    if truth is None:
        # No --truth and no --validate: GPU Exact becomes ground truth
        truth        = gpu_ex_count
        truth_source = 'GPU Exact'

    if ref_time is None:
        # No --cpu-ref-time: GPU Exact kernel time is the speedup reference
        ref_time  = gpu_ex_mean
        ref_label = 'GPU Exact'

    re  = relative_error(gpu_ex_count, truth) if truth is not None else 0.0
    b   = bias(gpu_ex_count, truth)           if truth is not None else 0
    e2e = t_h2d + gpu_ex_mean
    spd_cnt = ref_time / gpu_ex_mean if ref_time else None
    spd_e2e = ref_time / e2e         if ref_time else None
    print(f"  → mean={gpu_ex_mean:.3f}s  std={gpu_ex_std:.3f}s  "
          f"relErr={re:.5f}  bias={b:+,d}  e2e={e2e:.3f}s"
          + (f"  spdup_cnt={spd_cnt:.2f}x  spdup_e2e={spd_e2e:.2f}x"
             if spd_cnt else ""))
    _add_row('GPU Exact (ours)', gpu_ex_count, re, b,
             count_time=gpu_ex_mean, count_time_std=gpu_ex_std,
             e2e_time=e2e)

    # ── [3] cuGraph ───────────────────────────────────────────────────────
    if skip_cugraph:
        print("\n[3] cuGraph skipped (--skip-cugraph).")
    else:
        print(f"\n[3] cuGraph Exact ({N_TIMING_RUNS} kernel runs, 1 graph build) ...")
        res_cg = _cugraph_count(edges_np)
        if res_cg[0] is not None:
            cg_count, cg_e2e, cg_cnt_mean, cg_cnt_std, cg_build = res_cg
            _cugraph_cnt_time = cg_cnt_mean
            re = relative_error(cg_count, truth)
            b  = bias(cg_count, truth)
            print(f"  count={cg_count:,}  relErr={re:.5f}  bias={b:+,d}  "
                  f"build={cg_build:.3f}s  "
                  f"cnt={cg_cnt_mean:.3f}s±{cg_cnt_std:.3f}s  "
                  f"e2e={cg_e2e:.3f}s"
                  + (f"  spdup_cnt={ref_time/cg_cnt_mean:.2f}x"
                     f"  spdup_e2e={ref_time/cg_e2e:.2f}x"
                     if ref_time else ""))
            _add_row('cuGraph Exact', cg_count, re, b,
                     count_time=cg_cnt_mean, count_time_std=cg_cnt_std,
                     e2e_time=cg_e2e)
        else:
            print("  cuGraph not available — skipping.")
            rows.append({'method': 'cuGraph Exact',
                         'estimate': None, 'rel_error': None, 'bias': None,
                         'spdup_cnt': None, 'spdup_e2e': None,
                         'time': None, 'time_std': None,
                         'e2e_time': None, 'coverage': 0.0})

    # ── [4–5] DOULION ─────────────────────────────────────────────────────
    for q in DOULION_QS:
        tag = f"DOULION q={q:.2f}"
        print(f"\n[DOULION] {tag} ({DOULION_SEEDS} seeds) ...")
        res = _doulion_forward(edge_u_h, edge_v_h, q=q,
                               n_seeds=DOULION_SEEDS)
        re = relative_error(res['mean_estimate'], truth)
        b  = bias(res['mean_estimate'], truth)
        print(f"  mean={res['mean_estimate']:,.0f}  std={res['std_estimate']:,.0f}"
              f"  relErr={re:.5f}  bias={b:+,.0f}"
              f"  time={res['mean_time']:.3f}s±{res['std_time']:.3f}s"
              + (f"  spdup={ref_time/res['mean_time']:.2f}x"
                 if ref_time and res['mean_time'] else ""))
        _add_row(tag, res['mean_estimate'], re, b,
                 count_time=res['mean_time'],
                 count_time_std=res['std_time'])

    # ── [6–9] GPU Hybrid ─────────────────────────────────────────────────
    print(f"\n[Hybrid] GPU Hybrid sweep (p×γ) ...")
    for p in P_SWEEP:
        for gamma in GAMMA_SWEEP:
            # p=10: skip γ=256 to limit memory (p=10,γ=256 sketch ≈ 3M×1024B = 3GB)
            if p == 10 and gamma == 256:
                continue

            tag = f"GPU Hybrid p={p} γ={gamma}"
            sk_d, hs_d, t_bld, n_sk, t_alloc = _gpu_build_sketches(
                cuda, row_ptr_d, col_idx_d, n_nodes, gamma, p)

            cnt_times = []
            est_last  = None
            stats_last = None
            for run in range(N_TIMING_RUNS):
                est, t_cnt, stats = gpu_hybrid_triangle_count(
                    row_ptr_d, col_idx_d, edge_u_d, edge_v_d,
                    degrees_d, sk_d, hs_d, gamma=gamma, p=p,
                    split_kernels=split_hybrid)
                cnt_times.append(t_cnt)
                est_last   = est
                stats_last = stats

            cnt_mean = float(np.mean(cnt_times))
            cnt_std  = float(np.std(cnt_times))
            re  = relative_error(est_last, truth)
            b   = bias(est_last, truth)
            # e2e = H2D + sketch alloc + sketch build + count (mean)
            e2e = t_h2d + t_alloc + t_bld + cnt_mean

            print(f"  {tag}: est={est_last:,.0f}  relErr={re:.5f}  bias={b:+,.0f}"
                  f"  cov={stats_last['approx_coverage']:.4f}"
                  f"  clamp={stats_last['clamp_ratio']:.3f}"
                  f"  mode={stats_last.get('kernel_mode', 'unknown')}"
                  f"  nSk={n_sk:,}  alloc={t_alloc:.3f}s  build={t_bld:.3f}s"
                  f"  cnt={cnt_mean:.3f}s±{cnt_std:.3f}s  e2e={e2e:.3f}s"
                  + (f"  spdup_cnt={ref_time/cnt_mean:.2f}x"
                     f"  spdup_e2e={ref_time/e2e:.2f}x"
                     if ref_time else ""))

            _add_row(tag, est_last, re, b,
                     count_time=cnt_mean, count_time_std=cnt_std,
                     e2e_time=e2e,
                     coverage=stats_last['approx_coverage'],
                     clamp_ratio=stats_last['clamp_ratio'],
                     kernel_mode=stats_last.get('kernel_mode'),
                     n_sketched=n_sk,
                     build_time=t_bld,
                     sketch_alloc_time=t_alloc)

            del sk_d, hs_d
            gc.collect()

    # ── Post-process: back-fill cross-speedup columns ─────────────────────
    for r in rows:
        t = r.get('time')
        if t and t > 0:
            if _gpu_exact_cnt_time:
                r['spdup_vs_gpu_exact'] = _gpu_exact_cnt_time / t
            if _cugraph_cnt_time:
                r['spdup_vs_cugraph'] = _cugraph_cnt_time / t

    # ── Summary table ─────────────────────────────────────────────────────
    _print_table(rows, ref_label, ref_time,
                 gpu_exact_time=_gpu_exact_cnt_time,
                 cugraph_time=_cugraph_cnt_time)

    # ── Save JSON ─────────────────────────────────────────────────────────
    result = {
        'graph':          graph_path,
        'truth':          truth,
        'truth_source':   truth_source,
        'reference':      ref_label,
        'reference_time': ref_time,
        'n_timing_runs':  N_TIMING_RUNS,
        'hybrid_mode':    'split_kernels' if split_hybrid else 'fused_kernel',
        'dedup_edges':    dedup_edges,
        'kept_cugraph_edges': not skip_cugraph,
        'built_forward_adj': forward_adj is not None,
        'h2d_time':       t_h2d,
        't_load':         t_load,
        'n_nodes':        n_nodes,
        'n_oriented':     n_oriented,
        'e2e_definition': (
            'H2D transfer + sketch_alloc + sketch_build + kernel_count. '
            'Excludes file I/O, graph parsing, and orientation '
            '(shared preprocessing identical for all GPU methods).'),
        'rows':           rows,
    }
    out_path = os.path.join(output_dir, 'exp_gpu_full.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, default=float)
    print(f"\nResults saved → {out_path}")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Comprehensive GPU triangle-counting experiment."
    )
    parser.add_argument("graph_path",
                        help="Path to edge-list file.")
    parser.add_argument("output_dir", nargs="?", default="results",
                        help="Directory for result JSON (default: results/).")
    parser.add_argument("--skip-cugraph", action="store_true",
                        help="Skip optional cuGraph baseline.")
    parser.add_argument("--truth", type=int, default=None,
                        metavar="N",
                        help="Pre-verified exact triangle count (e.g. from SNAP). "
                             "Skips CPU Exact entirely.")
    parser.add_argument("--cpu-ref-time", type=float, default=None,
                        metavar="SECS",
                        help="CPU reference time (seconds) from published literature, "
                             "used as speedup denominator. If omitted, GPU Exact time "
                             "is used instead.")
    parser.add_argument("--cpu-ref-label", type=str, default=None,
                        metavar="LABEL",
                        help="Citation label for --cpu-ref-time "
                             "(e.g. 'Shun2015-8core').")
    parser.add_argument("--validate", action="store_true",
                        help="Run CPU Exact on the graph and compare with --truth "
                             "(or use as truth if --truth not given). "
                             "Recommended only for small graphs.")
    parser.add_argument("--fused-hybrid", action="store_true",
                        help="Use the fused Hybrid kernel. Default is split "
                             "exact-routed and HLL-routed kernels.")
    parser.add_argument("--no-dedup", action="store_true",
                        help="Skip np.unique edge deduplication. Use only for "
                             "trusted datasets whose undirected edges are "
                             "already unique.")
    args = parser.parse_args()
    try:
        run_exp_gpu(args.graph_path, args.output_dir,
                    skip_cugraph=args.skip_cugraph,
                    known_truth=args.truth,
                    cpu_ref_time=args.cpu_ref_time,
                    cpu_ref_label=args.cpu_ref_label,
                    validate=args.validate,
                    split_hybrid=not args.fused_hybrid,
                    dedup_edges=not args.no_dedup)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
