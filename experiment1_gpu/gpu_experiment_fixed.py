"""
GPU Experiment: Real Wall-Clock Verification of Hybrid HLL
============================================================

USAGE
  python gpu_experiment_fixed.py --graph soc-LiveJournal1.txt \
    --gammas 32 64 128 256 --p 8 --warmup 3 --repeat 5

REQUIREMENTS
  - NVIDIA GPU with >= 6GB VRAM
  - pip install cupy-cuda12x  (or cupy-cuda11x)
"""

import argparse
import csv
import gc
import os
import sys
import time
from collections import defaultdict

import numpy as np
np.seterr(over='ignore')  # hash functions rely on uint32 overflow

try:
    import cupy as cp
except ImportError:
    print("ERROR: CuPy not installed. Run: pip install cupy-cuda12x")
    sys.exit(1)


# ============================================================================
# Graph loading (CPU) -> oriented CSR
# ============================================================================

def load_and_orient(path):
    print(f"  Loading graph from {path} ...")
    t0 = time.perf_counter()

    graph = defaultdict(set)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('%'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            if u == v:
                continue
            graph[u].add(v)
            graph[v].add(u)
    load_time = time.perf_counter() - t0
    print(f"  Loaded: {len(graph):,} nodes ({load_time:.1f}s)")

    print("  Orienting edges ...")
    t1 = time.perf_counter()
    degrees = {u: len(nbrs) for u, nbrs in graph.items()}
    forward = defaultdict(list)
    for u, nbrs in graph.items():
        for v in nbrs:
            if u >= v:
                continue
            if (degrees[u], u) < (degrees[v], v):
                forward[u].append(v)
            else:
                forward[v].append(u)
    for u in forward:
        forward[u].sort()
    orient_time = time.perf_counter() - t1
    print(f"  Oriented ({orient_time:.1f}s)")

    del graph
    gc.collect()

    print("  Building CSR ...")
    t2 = time.perf_counter()
    all_nodes = set()
    for u, nbrs in forward.items():
        all_nodes.add(u)
        for v in nbrs:
            all_nodes.add(v)
    node_list = sorted(all_nodes)
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    N = len(node_list)

    deg_plus = np.zeros(N, dtype=np.int32)
    edges_u = []
    edges_v = []
    col_idx_list = []
    row_ptr = np.zeros(N + 1, dtype=np.int64)

    for i, n in enumerate(node_list):
        nbrs = forward.get(n, [])
        mapped = sorted(node_to_idx[v] for v in nbrs)
        deg_plus[i] = len(mapped)
        col_idx_list.extend(mapped)
        row_ptr[i + 1] = row_ptr[i] + len(mapped)
        for v_idx in mapped:
            edges_u.append(i)
            edges_v.append(v_idx)

    col_idx = np.array(col_idx_list, dtype=np.int32)
    edges_u = np.array(edges_u, dtype=np.int32)
    edges_v = np.array(edges_v, dtype=np.int32)
    E = len(edges_u)

    csr_time = time.perf_counter() - t2
    print(f"  CSR built: N={N:,}, E={E:,} ({csr_time:.1f}s)")

    del forward, all_nodes, node_list, col_idx_list
    gc.collect()

    return N, E, row_ptr, col_idx, deg_plus, edges_u, edges_v


# ============================================================================
# CUDA Kernels
# ============================================================================

# Kernel: Build HLL sketches on GPU (node-parallel, no atomics needed)
build_sketch_kernel_code = r'''
extern "C" __global__
void build_sketches(
    const long long* row_ptr,
    const int* col_idx,
    unsigned char* sketches,
    int N,
    int p
) {
    int u = blockDim.x * blockIdx.x + threadIdx.x;
    if (u >= N) return;

    int m = 1 << p;
    int base = u * m;
    long long start = row_ptr[u];
    long long end = row_ptr[u + 1];

    for (long long j = start; j < end; j++) {
        unsigned int v = (unsigned int)col_idx[j];

        // MurmurHash3-like mixing
        unsigned int h = 0x9747B28Cu ^ v;
        h ^= h >> 16;
        h *= 0x45D9F3Bu;
        h ^= h >> 16;
        h *= 0x45D9F3Bu;
        h ^= h >> 16;

        int bucket = (h >> (32 - p)) & (m - 1);
        unsigned int remaining = (h << p) | 1u;

        // Count leading zeros of remaining bits
        int rho = 1;
        for (int bit = 31 - p; bit >= 0; bit--) {
            if (remaining & (1u << bit)) break;
            rho++;
        }

        int idx = base + bucket;
        if ((unsigned char)rho > sketches[idx]) {
            sketches[idx] = (unsigned char)rho;
        }
    }
}
'''

# Kernel: Exact set intersection (per-edge, merge-based)
exact_kernel_code = r'''
extern "C" __global__
void exact_intersection(
    const long long* row_ptr,
    const int* col_idx,
    const int* edges_u,
    const int* edges_v,
    float* results,
    int E
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= E) return;

    int u = edges_u[tid];
    int v = edges_v[tid];

    long long u_start = row_ptr[u], u_end = row_ptr[u + 1];
    long long v_start = row_ptr[v], v_end = row_ptr[v + 1];

    int count = 0;
    long long i = u_start, j = v_start;
    while (i < u_end && j < v_end) {
        int a = col_idx[i];
        int b = col_idx[j];
        if (a == b) {
            count++;
            i++; j++;
        } else if (a < b) {
            i++;
        } else {
            j++;
        }
    }
    results[tid] = (float)count;
}
'''

# Kernel: HLL union + inclusion-exclusion (per-edge)
hll_kernel_code = r'''
extern "C" __global__
void hll_union(
    const unsigned char* sketches,
    const int* edges_u,
    const int* edges_v,
    float* results,
    int E,
    int m,
    float alpha
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= E) return;

    int u = edges_u[tid];
    int v = edges_v[tid];
    int base_u = u * m;
    int base_v = v * m;

    float sum_u = 0.0f, sum_v = 0.0f, sum_union = 0.0f;
    int zeros_u = 0, zeros_v = 0, zeros_union = 0;

    for (int k = 0; k < m; k++) {
        unsigned char ru = sketches[base_u + k];
        unsigned char rv = sketches[base_v + k];
        unsigned char ru_union = ru > rv ? ru : rv;

        sum_u += exp2f(-(float)ru);
        sum_v += exp2f(-(float)rv);
        sum_union += exp2f(-(float)ru_union);

        if (ru == 0) zeros_u++;
        if (rv == 0) zeros_v++;
        if (ru_union == 0) zeros_union++;
    }

    float fm = (float)m;
    float est_u = alpha * fm * fm / sum_u;
    float est_v = alpha * fm * fm / sum_v;
    float est_union = alpha * fm * fm / sum_union;

    if (est_u <= 2.5f * fm && zeros_u > 0)
        est_u = fm * logf(fm / (float)zeros_u);
    if (est_v <= 2.5f * fm && zeros_v > 0)
        est_v = fm * logf(fm / (float)zeros_v);
    if (est_union <= 2.5f * fm && zeros_union > 0)
        est_union = fm * logf(fm / (float)zeros_union);

    float support = est_u + est_v - est_union;
    results[tid] = support > 0.0f ? support : 0.0f;
}
'''

# Kernel: Classify edges as light/heavy
classify_kernel_code = r'''
extern "C" __global__
void classify_edges(
    const int* deg_plus,
    const int* edges_u,
    const int* edges_v,
    int* is_light,
    int E,
    int gamma
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= E) return;

    int u = edges_u[tid];
    int v = edges_v[tid];
    int cost = deg_plus[u] < deg_plus[v] ? deg_plus[u] : deg_plus[v];
    is_light[tid] = (cost <= gamma) ? 1 : 0;
}
'''


# ============================================================================
# GPU sketch builder
# ============================================================================

def build_hll_sketches_gpu(d_row_ptr, d_col_idx, N, p=8):
    """Build HLL sketches entirely on GPU. Returns CuPy array on device."""
    m = 1 << p
    d_sketches = cp.zeros(N * m, dtype=cp.uint8)

    build_kernel = cp.RawKernel(build_sketch_kernel_code, 'build_sketches')
    block = 256
    grid = (N + block - 1) // block
    build_kernel((grid,), (block,), (
        d_row_ptr, d_col_idx, d_sketches, np.int32(N), np.int32(p)
    ))
    cp.cuda.Stream.null.synchronize()

    return d_sketches, m


# ============================================================================
# Helpers
# ============================================================================

def get_gpu_name(device_id=0):
    props = cp.cuda.runtime.getDeviceProperties(device_id)
    name = props["name"]
    if isinstance(name, bytes):
        name = name.decode()
    return name


def benchmark_kernel(kernel_fn, args, E, warmup=3, repeat=5, label=""):
    """Run one kernel multiple times, return median time in ms."""
    block = 256
    grid = (E + block - 1) // block

    for _ in range(warmup):
        kernel_fn((grid,), (block,), args)
    cp.cuda.Stream.null.synchronize()

    times = []
    for _ in range(repeat):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        kernel_fn((grid,), (block,), args)
        end.record()
        end.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, end))

    med = float(np.median(times))
    std = float(np.std(times))
    if label:
        print(f"  {label:<45} {med:>9.2f} ms  (std={std:.2f})")
    return med, std, times


# ============================================================================
# Main experiment
# ============================================================================

def run_gpu_experiment(
    N, E, row_ptr, col_idx, deg_plus, edges_u, edges_v,
    gammas, p=8, warmup=3, repeat=5, out_dir='results_gpu',
):
    print(f"\n{'=' * 75}")
    print(f"  GPU Experiment: N={N:,}, E={E:,}, p={p}")
    print(f"  Device: {get_gpu_name(0)}")
    mem = cp.cuda.Device(0).mem_info
    print(f"  VRAM: {mem[1] / 1e9:.1f} GB total, {mem[0] / 1e9:.1f} GB free")
    print(f"{'=' * 75}")

    # ── Transfer graph data to GPU ──
    print("\n  Transferring graph to GPU ...")
    t0 = time.perf_counter()
    d_row_ptr = cp.asarray(row_ptr)
    d_col_idx = cp.asarray(col_idx)
    d_deg_plus = cp.asarray(deg_plus)
    d_edges_u = cp.asarray(edges_u)
    d_edges_v = cp.asarray(edges_v)
    d_results = cp.zeros(E, dtype=cp.float32)
    transfer_time = time.perf_counter() - t0
    print(f"  Transferred ({transfer_time:.2f}s)")

    mem_after = cp.cuda.Device(0).mem_info
    print(f"  VRAM after graph transfer: {mem_after[0] / 1e9:.1f} GB free")

    # ── Build HLL sketches on GPU ──
    m = 1 << p
    alpha = 0.7213 / (1.0 + 1.079 / m) if m >= 128 else \
        {16: 0.673, 32: 0.697, 64: 0.709}.get(m, 0.7213)

    print(f"\n  Building HLL sketches on GPU (p={p}, m={m}) ...")
    t_sketch = time.perf_counter()
    d_sketches, m = build_hll_sketches_gpu(d_row_ptr, d_col_idx, N, p)
    sketch_time = time.perf_counter() - t_sketch
    sketch_mb = N * m / 1024 / 1024
    print(f"  Built: {sketch_mb:.1f} MB ({sketch_time:.1f}s)")

    mem_after2 = cp.cuda.Device(0).mem_info
    print(f"  VRAM after sketches: {mem_after2[0] / 1e9:.1f} GB free")

    # ── Compile kernels ──
    print("\n  Compiling CUDA kernels ...")
    exact_kernel = cp.RawKernel(exact_kernel_code, 'exact_intersection')
    hll_kernel = cp.RawKernel(hll_kernel_code, 'hll_union')
    classify_kernel = cp.RawKernel(classify_kernel_code, 'classify_edges')
    print("  Compiled")

    # ══════════════════════════════════════════════════════════════════════
    # Benchmark baselines
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 75}")
    print(f"  Benchmarks (warmup={warmup}, repeat={repeat})")
    print(f"{'─' * 75}")

    # B1: All-exact, input order
    exact_args = (d_row_ptr, d_col_idx, d_edges_u, d_edges_v, d_results, np.int32(E))
    exact_time, exact_std, _ = benchmark_kernel(
        exact_kernel, exact_args, E, warmup, repeat,
        "All-exact (input order kernel)")

    # B2: All-exact, sorted by cost
    costs_np = np.minimum(deg_plus[edges_u], deg_plus[edges_v])
    sort_order = np.argsort(costs_np)
    d_sorted_u = cp.asarray(edges_u[sort_order])
    d_sorted_v = cp.asarray(edges_v[sort_order])
    d_results_sorted = cp.zeros(E, dtype=cp.float32)

    sorted_args = (d_row_ptr, d_col_idx, d_sorted_u, d_sorted_v,
                   d_results_sorted, np.int32(E))
    sorted_time, sorted_std, _ = benchmark_kernel(
        exact_kernel, sorted_args, E, warmup, repeat,
        "All-exact (sorted-by-cost kernel)")

    del d_sorted_u, d_sorted_v, d_results_sorted, costs_np, sort_order
    cp.get_default_memory_pool().free_all_blocks()

    # B3: All-HLL
    d_results_hll = cp.zeros(E, dtype=cp.float32)
    hll_args = (d_sketches, d_edges_u, d_edges_v, d_results_hll,
                np.int32(E), np.int32(m), np.float32(alpha))
    hll_time, hll_std, _ = benchmark_kernel(
        hll_kernel, hll_args, E, warmup, repeat,
        "All-HLL (pure sketch kernel)")
    del d_results_hll
    cp.get_default_memory_pool().free_all_blocks()

    # ══════════════════════════════════════════════════════════════════════
    # Benchmark hybrid at each gamma
    # ══════════════════════════════════════════════════════════════════════
    hybrid_results = {}
    block = 256
    grid = (E + block - 1) // block

    for gamma in gammas:
        # Probe once to get light/heavy counts (not timed)
        d_is_light_probe = cp.zeros(E, dtype=cp.int32)
        cls_args_probe = (d_deg_plus, d_edges_u, d_edges_v,
                          d_is_light_probe, np.int32(E), np.int32(gamma))
        classify_kernel((grid,), (block,), cls_args_probe)
        cp.cuda.Stream.null.synchronize()

        light_mask_probe = d_is_light_probe.astype(cp.bool_)
        n_light = int(cp.sum(light_mask_probe))
        n_heavy = E - n_light

        del d_is_light_probe, light_mask_probe
        cp.get_default_memory_pool().free_all_blocks()

        # End-to-end timed hybrid run
        def run_hybrid_full_once():
            d_is_light = cp.zeros(E, dtype=cp.int32)
            cls_args = (d_deg_plus, d_edges_u, d_edges_v,
                        d_is_light, np.int32(E), np.int32(gamma))

            start = cp.cuda.Event()
            end = cp.cuda.Event()
            start.record()

            # Classify
            classify_kernel((grid,), (block,), cls_args)

            # Partition
            light_mask = d_is_light.astype(cp.bool_)
            heavy_mask = ~light_mask
            light_idx = cp.where(light_mask)[0]
            heavy_idx = cp.where(heavy_mask)[0]

            d_light_u = d_edges_u[light_idx]
            d_light_v = d_edges_v[light_idx]
            d_heavy_u = d_edges_u[heavy_idx]
            d_heavy_v = d_edges_v[heavy_idx]

            # Exact phase (light edges)
            light_time_ms = 0.0
            if d_light_u.size > 0:
                d_res_light = cp.zeros(d_light_u.size, dtype=cp.float32)
                light_args = (
                    d_row_ptr, d_col_idx, d_light_u, d_light_v,
                    d_res_light, np.int32(d_light_u.size),
                )
                exact_start = cp.cuda.Event()
                exact_end = cp.cuda.Event()
                exact_start.record()
                exact_kernel(
                    ((int(d_light_u.size) + block - 1) // block,),
                    (block,), light_args)
                exact_end.record()
                exact_end.synchronize()
                light_time_ms = cp.cuda.get_elapsed_time(exact_start, exact_end)
            else:
                d_res_light = None

            # HLL phase (heavy edges)
            heavy_time_ms = 0.0
            if d_heavy_u.size > 0:
                d_res_heavy = cp.zeros(d_heavy_u.size, dtype=cp.float32)
                heavy_args = (
                    d_sketches, d_heavy_u, d_heavy_v, d_res_heavy,
                    np.int32(d_heavy_u.size), np.int32(m), np.float32(alpha),
                )
                hll_start = cp.cuda.Event()
                hll_end = cp.cuda.Event()
                hll_start.record()
                hll_kernel(
                    ((int(d_heavy_u.size) + block - 1) // block,),
                    (block,), heavy_args)
                hll_end.record()
                hll_end.synchronize()
                heavy_time_ms = cp.cuda.get_elapsed_time(hll_start, hll_end)
            else:
                d_res_heavy = None

            end.record()
            end.synchronize()
            total_ms = cp.cuda.get_elapsed_time(start, end)

            # Cleanup
            del d_is_light, light_mask, heavy_mask
            del light_idx, heavy_idx
            del d_light_u, d_light_v, d_heavy_u, d_heavy_v
            if d_res_light is not None:
                del d_res_light
            if d_res_heavy is not None:
                del d_res_heavy
            cp.get_default_memory_pool().free_all_blocks()

            return total_ms, light_time_ms, heavy_time_ms

        # Warmup
        for _ in range(warmup):
            run_hybrid_full_once()

        # Timed runs
        total_times = []
        light_times = []
        heavy_times = []
        for _ in range(repeat):
            total_ms, light_ms, heavy_ms = run_hybrid_full_once()
            total_times.append(total_ms)
            light_times.append(light_ms)
            heavy_times.append(heavy_ms)

        total_hybrid = float(np.median(total_times))
        light_time = float(np.median(light_times))
        heavy_time = float(np.median(heavy_times))
        cls_time = max(total_hybrid - light_time - heavy_time, 0.0)

        speedup_vs_exact = exact_time / total_hybrid if total_hybrid > 0 else 0.0
        speedup_vs_hll = hll_time / total_hybrid if total_hybrid > 0 else 0.0

        hybrid_results[gamma] = {
            'n_light': n_light,
            'n_heavy': n_heavy,
            'light_pct': n_light / E * 100,
            'heavy_pct': n_heavy / E * 100,
            'cls_time': cls_time,
            'light_time': light_time,
            'heavy_time': heavy_time,
            'total_time': total_hybrid,
            'speedup_vs_exact': speedup_vs_exact,
            'speedup_vs_hll': speedup_vs_hll,
        }

        print(
            f"  Hybrid g={gamma:<5} "
            f"L={n_light/E*100:>5.1f}% H={n_heavy/E*100:>5.1f}%  "
            f"cls={cls_time:>7.2f} exact={light_time:>8.2f} "
            f"hll={heavy_time:>8.2f} total={total_hybrid:>9.2f} ms  "
            f"vs_exact={speedup_vs_exact:>5.2f}x"
        )

    # ══════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 75}")
    print("  SUMMARY")
    print(f"{'=' * 75}")

    tput_exact = E / exact_time * 1000.0
    tput_sorted = E / sorted_time * 1000.0
    tput_hll = E / hll_time * 1000.0

    print(f"\n  {'Strategy':<40} {'Time(ms)':>10} {'Tput(M e/s)':>12} {'vs Exact':>10}")
    print(f"  {'─' * 75}")
    print(f"  {'All-exact (input order)':<40} "
          f"{exact_time:>10.2f} {tput_exact/1e6:>12.2f} {'1.00x':>10}")
    print(f"  {'All-exact (sorted by cost)':<40} "
          f"{sorted_time:>10.2f} {tput_sorted/1e6:>12.2f} "
          f"{exact_time/sorted_time:>9.2f}x")
    print(f"  {'All-HLL (pure sketch)':<40} "
          f"{hll_time:>10.2f} {tput_hll/1e6:>12.2f} "
          f"{exact_time/hll_time:>9.2f}x")
    print()

    best_gamma = None
    best_time = float('inf')
    for gamma in gammas:
        h = hybrid_results[gamma]
        tput_h = E / h['total_time'] * 1000.0
        sp = exact_time / h['total_time']
        if h['total_time'] < best_time:
            best_time = h['total_time']
            best_gamma = gamma
        print(
            f"  Hybrid g={gamma:<5} "
            f"(L={h['light_pct']:>5.1f}% H={h['heavy_pct']:>5.1f}%)  "
            f"{h['total_time']:>10.2f} {tput_h/1e6:>12.2f} {sp:>9.2f}x"
        )

    best_h = hybrid_results[best_gamma]
    best_sp = exact_time / best_h['total_time']
    sorted_sp = exact_time / sorted_time if sorted_time > 0 else 0.0

    print(f"\n  Best hybrid: g={best_gamma}, "
          f"{best_h['total_time']:.2f} ms, {best_sp:.2f}x vs exact")

    # ── GO / NO-GO ──
    print(f"\n{'=' * 75}")
    print("  GPU TIMING VERDICT")
    print(f"{'=' * 75}")

    c1 = best_sp > 1.5
    c2 = best_sp > sorted_sp
    c3 = best_h['total_time'] < hll_time

    print(f"  [C1] Hybrid > 1.5x vs exact?       "
          f"{best_sp:.2f}x  ->  {'PASS' if c1 else 'FAIL'}")
    print(f"  [C2] Hybrid > sorted exact?         "
          f"{best_sp:.2f}x vs {sorted_sp:.2f}x  ->  {'PASS' if c2 else 'FAIL'}")
    print(f"  [C3] Hybrid faster than pure HLL?   "
          f"{best_h['total_time']:.2f} vs {hll_time:.2f} ms  ->  "
          f"{'PASS' if c3 else 'FAIL'}")

    n_pass = sum([c1, c2, c3])
    if n_pass == 3:
        print(f"\n  >>> DECISION: GO ({n_pass}/3)")
    elif n_pass >= 2:
        print(f"\n  >>> DECISION: MARGINAL ({n_pass}/3)")
    else:
        print(f"\n  >>> DECISION: NO-GO ({n_pass}/3)")

    # Time breakdown
    print(f"\n  Time breakdown for best hybrid (g={best_gamma}):")
    print(f"    Classification+partition: "
          f"{best_h['cls_time']:>8.2f} ms "
          f"({best_h['cls_time']/best_h['total_time']*100:>5.1f}%)")
    print(f"    Exact phase:             "
          f"{best_h['light_time']:>8.2f} ms "
          f"({best_h['light_time']/best_h['total_time']*100:>5.1f}%)")
    print(f"    HLL phase:               "
          f"{best_h['heavy_time']:>8.2f} ms "
          f"({best_h['heavy_time']/best_h['total_time']*100:>5.1f}%)")

    # Extra info
    print(f"\n  Sketch build time (GPU): {sketch_time:.2f}s")
    print(f"  Sketch memory: {sketch_mb:.1f} MB")
    print(f"  Graph transfer time: {transfer_time:.2f}s")

    # ── Save CSV ──
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'gpu_timing_results.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'strategy', 'gamma', 'time_ms', 'throughput_edges_per_sec',
            'speedup_vs_exact', 'light_pct', 'heavy_pct',
            'cls_ms', 'exact_ms', 'hll_ms',
        ])
        w.writerow([
            'all_exact_input', '', f'{exact_time:.4f}',
            f'{tput_exact:.0f}', '1.00', '100', '0', '', '', '',
        ])
        w.writerow([
            'all_exact_sorted', '', f'{sorted_time:.4f}',
            f'{tput_sorted:.0f}', f'{sorted_sp:.4f}', '100', '0', '', '', '',
        ])
        w.writerow([
            'all_hll', '', f'{hll_time:.4f}',
            f'{tput_hll:.0f}', f'{exact_time/hll_time:.4f}',
            '0', '100', '', '', '',
        ])
        for gamma in gammas:
            h = hybrid_results[gamma]
            tput_h = E / h['total_time'] * 1000.0
            w.writerow([
                f'hybrid_g{gamma}', gamma, f"{h['total_time']:.4f}",
                f'{tput_h:.0f}', f"{exact_time/h['total_time']:.4f}",
                f"{h['light_pct']:.2f}", f"{h['heavy_pct']:.2f}",
                f"{h['cls_time']:.4f}", f"{h['light_time']:.4f}",
                f"{h['heavy_time']:.4f}",
            ])
    print(f"\n  CSV saved to {csv_path}")

    return {
        'exact_time': exact_time,
        'sorted_time': sorted_time,
        'hll_time': hll_time,
        'hybrid_results': hybrid_results,
        'best_gamma': best_gamma,
        'transfer_time_s': transfer_time,
        'sketch_build_time_s': sketch_time,
        'sketch_mb': sketch_mb,
    }


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='GPU Experiment: Hybrid HLL timing verification')
    parser.add_argument('--graph', required=True, help='Path to edge-list file')
    parser.add_argument('--gammas', nargs='+', type=int,
                        default=[32, 64, 128, 256])
    parser.add_argument('--p', type=int, default=8,
                        help='HLL precision (default: 8)')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--out-dir', default='results_gpu')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    print(f"{'=' * 75}")
    print("  GPU Hybrid HLL Experiment")
    print(f"  Graph: {args.graph}")
    print(f"  Gammas: {args.gammas}")
    print(f"  HLL p={args.p} (m={1 << args.p})")
    print(f"{'=' * 75}")

    N, E, row_ptr, col_idx, deg_plus, edges_u, edges_v = \
        load_and_orient(args.graph)

    run_gpu_experiment(
        N, E, row_ptr, col_idx, deg_plus, edges_u, edges_v,
        gammas=args.gammas, p=args.p,
        warmup=args.warmup, repeat=args.repeat,
        out_dir=args.out_dir,
    )
