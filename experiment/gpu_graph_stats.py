"""
GPU graph structure and degree-distribution analysis.

The CPU side only parses the edge list and normalizes undirected edges.
Degree counting, summary statistics, exact degree histogram, and log2 degree
buckets are computed with CUDA kernels.

Example
-------
python experiment/gpu_graph_stats.py datasets/com-orkut.ungraph.txt results/graph_stats
python experiment/gpu_graph_stats.py datasets/soc-LiveJournal1.txt.gz results/lj_stats --dedup
"""

import argparse
import csv
import gzip
import json
import math
import os
import sys
import time
from typing import Iterator, Tuple

import numpy as np

try:
    from numba import cuda, int64
except ModuleNotFoundError as exc:
    raise RuntimeError("gpu_graph_stats.py requires numba with CUDA support.") from exc


COMMENT_CHARS = "#%"
THREADS = 256


def _open_text(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def _iter_edges(path: str, comment_chars: str = COMMENT_CHARS) -> Iterator[Tuple[int, int]]:
    with _open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line or line[0] in comment_chars:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u = int(parts[0])
            v = int(parts[1])
            if u == v:
                continue
            if u < 0 or v < 0:
                raise ValueError("This script expects non-negative node IDs.")
            yield (u, v)


def _load_edges(path: str, dedup: bool) -> tuple[np.ndarray, np.ndarray, float, int]:
    """
    Load an undirected edge list into canonical int32 src/dst arrays.

    Returns
    -------
    src, dst      : int32 arrays, canonical u < v
    elapsed       : load time in seconds
    raw_edges     : number of non-comment, non-self-loop edges before dedup
    """
    t0 = time.perf_counter()
    raw_edges = sum(1 for _ in _iter_edges(path))
    edges = np.empty((raw_edges, 2), dtype=np.int32)

    for i, (u, v) in enumerate(_iter_edges(path)):
        if u <= v:
            edges[i, 0] = u
            edges[i, 1] = v
        else:
            edges[i, 0] = v
            edges[i, 1] = u

    if dedup:
        edges = np.unique(edges, axis=0)

    elapsed = time.perf_counter() - t0
    return edges[:, 0].copy(), edges[:, 1].copy(), elapsed, raw_edges


@cuda.jit
def degree_count_kernel(src, dst, n_edges, degree):
    tid = cuda.grid(1)
    if tid >= n_edges:
        return
    u = src[tid]
    v = dst[tid]
    cuda.atomic.add(degree, u, 1)
    cuda.atomic.add(degree, v, 1)


@cuda.jit
def degree_stats_kernel(degree, n_slots, sums, minmax):
    """
    sums: int64[3] = [active_nodes, degree_sum, degree_sumsq]
    minmax: int32[2] = [max_degree, min_positive_degree]
    """
    tid = cuda.grid(1)
    if tid >= n_slots:
        return

    d = degree[tid]
    if d <= 0:
        return

    cuda.atomic.add(sums, 0, int64(1))
    cuda.atomic.add(sums, 1, int64(d))
    cuda.atomic.add(sums, 2, int64(d) * int64(d))
    cuda.atomic.max(minmax, 0, d)
    cuda.atomic.min(minmax, 1, d)


@cuda.jit
def degree_hist_kernel(degree, n_slots, hist, log_hist, n_log_buckets):
    tid = cuda.grid(1)
    if tid >= n_slots:
        return

    d = degree[tid]
    if d <= 0:
        return

    cuda.atomic.add(hist, d, int64(1))

    bucket = 0
    x = d
    while x > 1:
        x = x >> 1
        bucket += 1
    if bucket >= n_log_buckets:
        bucket = n_log_buckets - 1
    cuda.atomic.add(log_hist, bucket, int64(1))


def _require_cuda():
    if not cuda.is_available():
        raise RuntimeError(
            "Numba CUDA is not available. Check CUDA toolkit/driver and CUDA_HOME."
        )
    return cuda.get_current_device()


def _write_degree_csv(path: str, hist: np.ndarray, active_nodes: int) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["degree", "node_count", "fraction"])
        nonzero_degrees = np.nonzero(hist)[0]
        for d in nonzero_degrees:
            cnt = int(hist[d])
            writer.writerow([int(d), cnt, cnt / active_nodes if active_nodes else 0.0])


def _write_log_csv(path: str, log_hist: np.ndarray, active_nodes: int) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bucket", "degree_low", "degree_high", "node_count", "fraction"])
        for b, cnt in enumerate(log_hist):
            cnt_i = int(cnt)
            if cnt_i == 0:
                continue
            low = 1 << b
            high = (1 << (b + 1)) - 1
            writer.writerow([
                f"[{low},{high}]",
                low,
                high,
                cnt_i,
                cnt_i / active_nodes if active_nodes else 0.0,
            ])


def _top_degrees(degree_h: np.ndarray, top_k: int) -> list[dict]:
    active = np.nonzero(degree_h)[0]
    if len(active) == 0 or top_k <= 0:
        return []

    deg_active = degree_h[active]
    k = min(top_k, len(active))
    idx = np.argpartition(deg_active, -k)[-k:]
    order = idx[np.argsort(deg_active[idx])[::-1]]
    return [
        {"node": int(active[i]), "degree": int(deg_active[i])}
        for i in order
    ]


def analyze_graph_gpu(graph_path: str, output_dir: str, dedup: bool = False,
                      top_k: int = 20) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    device = _require_cuda()

    print("=" * 72)
    print("GPU GRAPH STRUCTURE / DEGREE DISTRIBUTION")
    print(f"Graph : {graph_path}")
    print(f"GPU   : {device.name.decode() if isinstance(device.name, bytes) else device.name}")
    print(f"Dedup : {dedup}")
    print("=" * 72)

    print("\n[1] Loading edge list on CPU ...")
    src_h, dst_h, load_time, raw_edges = _load_edges(graph_path, dedup=dedup)
    n_edges = int(len(src_h))
    if n_edges == 0:
        raise ValueError("No edges found.")
    max_node_id = int(max(src_h.max(), dst_h.max()))
    n_slots = max_node_id + 1
    print(
        f"  raw_edges={raw_edges:,}  used_edges={n_edges:,}  "
        f"max_node_id={max_node_id:,}  load={load_time:.3f}s"
    )

    print("\n[2] Copying edges to GPU ...")
    t0 = time.perf_counter()
    src_d = cuda.to_device(src_h)
    dst_d = cuda.to_device(dst_h)
    degree_d = cuda.to_device(np.zeros(n_slots, dtype=np.int32))
    cuda.synchronize()
    h2d_time = time.perf_counter() - t0
    print(f"  H2D + degree alloc: {h2d_time:.3f}s")

    print("\n[3] Counting degrees on GPU ...")
    grid_edges = math.ceil(n_edges / THREADS)
    cuda.synchronize()
    t0 = time.perf_counter()
    degree_count_kernel[grid_edges, THREADS](src_d, dst_d, n_edges, degree_d)
    cuda.synchronize()
    degree_time = time.perf_counter() - t0
    print(f"  degree kernel: {degree_time:.3f}s")

    print("\n[4] Computing summary stats on GPU ...")
    grid_nodes = math.ceil(n_slots / THREADS)
    sums_d = cuda.to_device(np.zeros(3, dtype=np.int64))
    minmax_d = cuda.to_device(np.array([0, np.iinfo(np.int32).max], dtype=np.int32))
    cuda.synchronize()
    t0 = time.perf_counter()
    degree_stats_kernel[grid_nodes, THREADS](degree_d, n_slots, sums_d, minmax_d)
    cuda.synchronize()
    stats_time = time.perf_counter() - t0

    sums_h = sums_d.copy_to_host()
    minmax_h = minmax_d.copy_to_host()
    active_nodes = int(sums_h[0])
    degree_sum = int(sums_h[1])
    degree_sumsq = int(sums_h[2])
    max_degree = int(minmax_h[0])
    min_degree = int(minmax_h[1]) if active_nodes else 0
    mean_degree = degree_sum / active_nodes if active_nodes else 0.0
    variance = (degree_sumsq / active_nodes - mean_degree * mean_degree) if active_nodes else 0.0
    std_degree = math.sqrt(max(0.0, variance))
    density = degree_sum / (active_nodes * (active_nodes - 1)) if active_nodes > 1 else 0.0
    print(
        f"  active_nodes={active_nodes:,}  min={min_degree:,}  "
        f"max={max_degree:,}  mean={mean_degree:.3f}  std={std_degree:.3f}  "
        f"stats={stats_time:.3f}s"
    )

    print("\n[5] Building degree histograms on GPU ...")
    hist_d = cuda.to_device(np.zeros(max_degree + 1, dtype=np.int64))
    n_log_buckets = max(1, int(math.floor(math.log2(max_degree))) + 1)
    log_hist_d = cuda.to_device(np.zeros(n_log_buckets, dtype=np.int64))
    cuda.synchronize()
    t0 = time.perf_counter()
    degree_hist_kernel[grid_nodes, THREADS](
        degree_d, n_slots, hist_d, log_hist_d, n_log_buckets
    )
    cuda.synchronize()
    hist_time = time.perf_counter() - t0
    hist_h = hist_d.copy_to_host()
    log_hist_h = log_hist_d.copy_to_host()
    print(f"  histogram kernel: {hist_time:.3f}s  degree_bins={len(hist_h):,}")

    print("\n[6] Copying compact outputs and writing files ...")
    t0 = time.perf_counter()
    degree_h = degree_d.copy_to_host()
    d2h_time = time.perf_counter() - t0

    top_nodes = _top_degrees(degree_h, top_k=top_k)
    degree_csv = os.path.join(output_dir, "degree_distribution.csv")
    log_csv = os.path.join(output_dir, "degree_log2_buckets.csv")
    json_path = os.path.join(output_dir, "graph_stats.json")
    _write_degree_csv(degree_csv, hist_h, active_nodes)
    _write_log_csv(log_csv, log_hist_h, active_nodes)

    result = {
        "graph": graph_path,
        "raw_edges": raw_edges,
        "used_edges": n_edges,
        "dedup": dedup,
        "active_nodes": active_nodes,
        "max_node_id": max_node_id,
        "degree_sum": degree_sum,
        "avg_degree": mean_degree,
        "min_degree": min_degree,
        "max_degree": max_degree,
        "std_degree": std_degree,
        "density_active_nodes": density,
        "degree_1_nodes": int(hist_h[1]) if len(hist_h) > 1 else 0,
        "top_degrees": top_nodes,
        "timing_seconds": {
            "load_cpu": load_time,
            "h2d_and_alloc": h2d_time,
            "degree_kernel": degree_time,
            "stats_kernel": stats_time,
            "histogram_kernel": hist_time,
            "degree_d2h_for_topk": d2h_time,
        },
        "outputs": {
            "degree_distribution_csv": degree_csv,
            "degree_log2_buckets_csv": log_csv,
            "summary_json": json_path,
        },
    }

    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  wrote: {json_path}")
    print(f"  wrote: {degree_csv}")
    print(f"  wrote: {log_csv}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute graph structure and degree distribution on GPU."
    )
    parser.add_argument("graph_path", help="Edge-list file path (.txt or .gz).")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=os.path.join("results", "graph_stats"),
        help="Directory for graph_stats.json and degree CSV files.",
    )
    parser.add_argument(
        "--dedup",
        action="store_true",
        help="Deduplicate canonical undirected edges before GPU counting. "
             "This costs extra CPU memory/time but is safer for raw edge lists.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of highest-degree nodes to include in the JSON summary.",
    )
    args = parser.parse_args(argv)

    analyze_graph_gpu(
        graph_path=args.graph_path,
        output_dir=args.output_dir,
        dedup=args.dedup,
        top_k=args.top_k,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
