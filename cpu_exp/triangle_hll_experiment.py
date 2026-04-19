#!/usr/bin/env python3
"""CPU experiments for HLL-based approximate triangle counting.

Methods implemented:
1. baseline: NetworKit TriangleEdgeScore, summed over edges / 3.
2. pure-hll: HLL estimates every forward-neighborhood intersection.
3. hybrid: exact intersections for cheap forward edges, HLL for expensive ones.

The approximate methods use the standard ordered/forward triangle decomposition.
Orient each undirected edge from lower (degree, node_id) to higher
(degree, node_id). For every oriented edge u -> v, count
|N+(u) intersection N+(v)|. Each triangle is counted exactly once in the
exact variant.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

try:
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "Missing dependency: numpy. Install dependencies with "
        "`python -m pip install -r requirements.txt`."
    ) from exc

try:  # Numba is optional, but strongly recommended for real experiments.
    import numba as nb

    HAVE_NUMBA = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    nb = None
    HAVE_NUMBA = False


MASK64 = (1 << 64) - 1
SM64_A = 0x9E3779B97F4A7C15
SM64_B = 0xBF58476D1CE4E5B9
SM64_C = 0x94D049BB133111EB


@dataclass(slots=True)
class CSRGraph:
    n: int
    m: int
    edges: np.ndarray
    offsets: np.ndarray
    adj: np.ndarray
    degree: np.ndarray
    labels: list[str] | None = None


@dataclass(slots=True)
class ForwardGraph:
    n: int
    m: int
    edges: np.ndarray
    offsets: np.ndarray
    adj: np.ndarray
    out_degree: np.ndarray


@dataclass(slots=True)
class MethodResult:
    method: str
    triangles: float
    total_seconds: float
    preprocess_seconds: float = 0.0
    query_seconds: float = 0.0
    details: dict[str, float | int | str] | None = None


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU triangle-counting experiment: NetworKit, HLL, hybrid."
    )
    parser.add_argument(
        "edge_list",
        nargs="?",
        type=Path,
        help="Text edge list with two node ids per line.",
    )
    parser.add_argument(
        "--run",
        default="baseline,pure-hll,hybrid",
        help="Comma-separated methods: baseline,pure-hll,hybrid,exact.",
    )
    parser.add_argument(
        "--hll-p",
        type=int,
        default=10,
        help="HLL precision p; registers per node = 2**p. Default: 10.",
    )
    parser.add_argument(
        "--hybrid-threshold",
        type=int,
        default=64,
        help=(
            "Use exact when min(out_degree[u], out_degree[v]) <= threshold; "
            "otherwise use HLL. Default: 64."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="64-bit hash seed for HLL sketches. Default: 1.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Thread count for NetworKit baseline when NetworKit is used.",
    )
    parser.add_argument(
        "--numba-threads",
        type=int,
        default=None,
        help="Thread count for Numba exact/HLL kernels. Default: Numba default.",
    )
    parser.add_argument(
        "--delimiter",
        default=None,
        help="Input delimiter. Default: any whitespace.",
    )
    parser.add_argument(
        "--comment-prefixes",
        default="#%",
        help="Lines beginning with any of these characters are skipped.",
    )
    parser.add_argument(
        "--no-remap",
        action="store_true",
        help="Treat ids as dense integers instead of remapping arbitrary labels.",
    )
    parser.add_argument(
        "--one-indexed",
        action="store_true",
        help="With --no-remap, subtract one from every input id.",
    )
    parser.add_argument(
        "--assume-simple",
        action="store_true",
        help="Skip duplicate-edge removal. Only use on simple undirected graphs.",
    )
    parser.add_argument(
        "--max-hll-gb",
        type=float,
        default=8.0,
        help="Abort before allocating HLL registers above this size. Default: 8.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path for machine-readable JSON results.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a tiny correctness test for exact and all-exact hybrid.",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Do not run Numba warm-up before method timing.",
    )
    return parser.parse_args(argv)


def read_edge_list(
    path: Path,
    *,
    delimiter: str | None,
    comment_prefixes: str,
    no_remap: bool,
    one_indexed: bool,
    assume_simple: bool,
) -> CSRGraph:
    src: list[int] = []
    dst: list[int] = []
    labels: list[str] | None = None

    if no_remap:
        with open_text(path) as handle:
            for line_no, raw in enumerate(handle, start=1):
                parsed = _parse_edge_line(raw, delimiter, comment_prefixes)
                if parsed is None:
                    continue
                a_raw, b_raw = parsed
                try:
                    a = int(a_raw)
                    b = int(b_raw)
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_no}: expected integer node ids") from exc
                if one_indexed:
                    a -= 1
                    b -= 1
                if a < 0 or b < 0:
                    raise ValueError(f"{path}:{line_no}: negative node id after parsing")
                src.append(a)
                dst.append(b)
    else:
        label_to_id: dict[str, int] = {}
        labels = []
        with open_text(path) as handle:
            for raw in handle:
                parsed = _parse_edge_line(raw, delimiter, comment_prefixes)
                if parsed is None:
                    continue
                a_label, b_label = parsed
                a = _intern_label(a_label, label_to_id, labels)
                b = _intern_label(b_label, label_to_id, labels)
                src.append(a)
                dst.append(b)

    if not src:
        return build_csr(0, np.empty((0, 2), dtype=np.int64), labels)

    u = np.asarray(src, dtype=np.int64)
    v = np.asarray(dst, dtype=np.int64)
    keep = u != v
    u = u[keep]
    v = v[keep]

    lo = np.minimum(u, v)
    hi = np.maximum(u, v)
    edges = np.column_stack((lo, hi)).astype(np.int64, copy=False)
    if not assume_simple and edges.size:
        edges = np.unique(edges, axis=0)

    n = len(labels) if labels is not None else int(edges.max(initial=-1)) + 1
    return build_csr(n, edges, labels)


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _parse_edge_line(
    raw: str, delimiter: str | None, comment_prefixes: str
) -> tuple[str, str] | None:
    line = raw.strip()
    if not line or line[0] in comment_prefixes:
        return None
    parts = line.split(delimiter) if delimiter is not None else line.split()
    if len(parts) < 2:
        return None
    return parts[0], parts[1]


def _intern_label(label: str, label_to_id: dict[str, int], labels: list[str]) -> int:
    node_id = label_to_id.get(label)
    if node_id is None:
        node_id = len(labels)
        label_to_id[label] = node_id
        labels.append(label)
    return node_id


def build_csr(n: int, edges: np.ndarray, labels: list[str] | None = None) -> CSRGraph:
    edges = np.asarray(edges, dtype=np.int64)
    if edges.size == 0:
        edges = np.empty((0, 2), dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must have shape (m, 2)")
    if edges.size and (edges.min() < 0 or edges.max() >= n):
        raise ValueError("edge endpoint outside node id range")

    degree = np.zeros(n, dtype=np.int64)
    if edges.size:
        np.add.at(degree, edges[:, 0], 1)
        np.add.at(degree, edges[:, 1], 1)

    offsets = np.empty(n + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(degree, out=offsets[1:])

    if edges.size:
        srcs = np.concatenate([edges[:, 0], edges[:, 1]])
        dsts = np.concatenate([edges[:, 1], edges[:, 0]])
        order = np.lexsort((dsts, srcs))
        adj = dsts[order]
    else:
        adj = np.empty(0, dtype=np.int64)

    return CSRGraph(
        n=n,
        m=int(edges.shape[0]),
        edges=edges,
        offsets=offsets,
        adj=adj,
        degree=degree,
        labels=labels,
    )


def build_forward_graph(csr: CSRGraph) -> ForwardGraph:
    if csr.m == 0:
        return ForwardGraph(
            n=csr.n,
            m=0,
            edges=np.empty((0, 2), dtype=np.int64),
            offsets=np.zeros(csr.n + 1, dtype=np.int64),
            adj=np.empty(0, dtype=np.int64),
            out_degree=np.zeros(csr.n, dtype=np.int64),
        )

    u = csr.edges[:, 0]
    v = csr.edges[:, 1]
    du = csr.degree[u]
    dv = csr.degree[v]
    u_to_v = (du < dv) | ((du == dv) & (u < v))
    src = np.where(u_to_v, u, v)
    dst = np.where(u_to_v, v, u)
    fwd_edges = np.column_stack((src, dst)).astype(np.int64, copy=False)

    out_degree = np.bincount(src, minlength=csr.n).astype(np.int64, copy=False)
    offsets = np.empty(csr.n + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(out_degree, out=offsets[1:])

    order = np.lexsort((dst, src))
    adj = dst[order]

    return ForwardGraph(
        n=csr.n,
        m=csr.m,
        edges=fwd_edges,
        offsets=offsets,
        adj=adj,
        out_degree=out_degree,
    )


def hll_alpha(register_count: int) -> float:
    if register_count == 16:
        return 0.673
    if register_count == 32:
        return 0.697
    if register_count == 64:
        return 0.709
    return 0.7213 / (1.0 + 1.079 / register_count)


def estimate_hll_cardinality(registers: np.ndarray) -> float:
    m = int(registers.size)
    inv_sum = float(np.sum(np.exp2(-registers.astype(np.float64))))
    raw = hll_alpha(m) * m * m / inv_sum
    zeros = int(np.count_nonzero(registers == 0))
    if raw <= 2.5 * m and zeros > 0:
        return m * math.log(m / zeros)
    if raw > 4294967296.0 / 30.0:
        return -4294967296.0 * math.log(1.0 - raw / 4294967296.0)
    return raw


def splitmix64_py(x: int) -> int:
    z = (x + SM64_A) & MASK64
    z = ((z ^ (z >> 30)) * SM64_B) & MASK64
    z = ((z ^ (z >> 27)) * SM64_C) & MASK64
    return (z ^ (z >> 31)) & MASK64


def rho_from_hash_py(hash_value: int, p: int) -> int:
    bits = 64 - p
    w = hash_value >> p
    if w == 0:
        return bits + 1
    return bits - w.bit_length() + 1


def build_hll_registers_py(fwd: ForwardGraph, p: int, seed: int) -> np.ndarray:
    register_count = 1 << p
    registers = np.zeros((fwd.n, register_count), dtype=np.uint8)
    seed &= MASK64
    mask = register_count - 1
    for u in range(fwd.n):
        start, end = int(fwd.offsets[u]), int(fwd.offsets[u + 1])
        row = registers[u]
        for pos in range(start, end):
            h = splitmix64_py((int(fwd.adj[pos]) + seed) & MASK64)
            idx = h & mask
            rho = rho_from_hash_py(h, p)
            if rho > row[idx]:
                row[idx] = rho
    return registers


def count_intersection_py(offsets: np.ndarray, adj: np.ndarray, a: int, b: int) -> int:
    a0, a1 = int(offsets[a]), int(offsets[a + 1])
    b0, b1 = int(offsets[b]), int(offsets[b + 1])
    len_a = a1 - a0
    len_b = b1 - b0
    if len_a == 0 or len_b == 0:
        return 0
    if len_a * 16 < len_b:
        return count_intersection_binary_py(adj, a0, a1, b0, b1)
    if len_b * 16 < len_a:
        return count_intersection_binary_py(adj, b0, b1, a0, a1)

    i, j = a0, b0
    count = 0
    while i < a1 and j < b1:
        av = int(adj[i])
        bv = int(adj[j])
        if av == bv:
            count += 1
            i += 1
            j += 1
        elif av < bv:
            i += 1
        else:
            j += 1
    return count


def count_intersection_binary_py(
    adj: np.ndarray, small0: int, small1: int, large0: int, large1: int
) -> int:
    count = 0
    for pos in range(small0, small1):
        x = int(adj[pos])
        lo, hi = large0, large1
        while lo < hi:
            mid = (lo + hi) // 2
            if int(adj[mid]) < x:
                lo = mid + 1
            else:
                hi = mid
        if lo < large1 and int(adj[lo]) == x:
            count += 1
    return count


def exact_forward_count_py(fwd: ForwardGraph) -> int:
    total = 0
    for u, v in fwd.edges:
        total += count_intersection_py(fwd.offsets, fwd.adj, int(u), int(v))
    return total


def hll_edge_intersection_py(
    registers: np.ndarray, out_degree: np.ndarray, u: int, v: int
) -> tuple[float, bool]:
    union_registers = np.maximum(registers[u], registers[v])
    union_estimate = estimate_hll_cardinality(union_registers)
    estimate = float(out_degree[u] + out_degree[v]) - union_estimate
    upper = float(min(out_degree[u], out_degree[v]))
    if estimate < 0.0:
        return 0.0, True
    if estimate > upper:
        return upper, True
    return estimate, False


def pure_hll_count_py(fwd: ForwardGraph, registers: np.ndarray) -> tuple[float, int]:
    total = 0.0
    clamp_count = 0
    for u, v in fwd.edges:
        est, clamped = hll_edge_intersection_py(registers, fwd.out_degree, int(u), int(v))
        total += est
        if clamped:
            clamp_count += 1
    return total, clamp_count


def hybrid_count_py(
    fwd: ForwardGraph, registers: np.ndarray, threshold: int
) -> tuple[float, int, int, float, float, int]:
    exact_sum = 0.0
    hll_sum = 0.0
    exact_edges = 0
    hll_edges = 0
    clamp_count = 0
    for u_raw, v_raw in fwd.edges:
        u = int(u_raw)
        v = int(v_raw)
        cost = int(min(fwd.out_degree[u], fwd.out_degree[v]))
        if cost <= threshold:
            exact_sum += count_intersection_py(fwd.offsets, fwd.adj, u, v)
            exact_edges += 1
        else:
            est, clamped = hll_edge_intersection_py(registers, fwd.out_degree, u, v)
            hll_sum += est
            hll_edges += 1
            if clamped:
                clamp_count += 1
    return exact_sum + hll_sum, exact_edges, hll_edges, exact_sum, hll_sum, clamp_count


if HAVE_NUMBA:

    @nb.njit(cache=True)
    def splitmix64_numba(x: np.uint64) -> np.uint64:
        z = x + np.uint64(SM64_A)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(SM64_B)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(SM64_C)
        return z ^ (z >> np.uint64(31))

    @nb.njit(cache=True)
    def rho_from_hash_numba(hash_value: np.uint64, p: int) -> int:
        bits = 64 - p
        w = hash_value >> np.uint64(p)
        if w == 0:
            return bits + 1
        rho = 1
        mask = np.uint64(1) << np.uint64(bits - 1)
        while (w & mask) == 0:
            rho += 1
            mask >>= np.uint64(1)
        return rho

    @nb.njit(cache=True, parallel=True)
    def build_hll_registers_numba(
        n: int, offsets: np.ndarray, adj: np.ndarray, p: int, seed: int
    ) -> np.ndarray:
        register_count = 1 << p
        registers = np.zeros((n, register_count), dtype=np.uint8)
        idx_mask = np.uint64(register_count - 1)
        seed64 = np.uint64(seed)
        for u in nb.prange(n):
            for pos in range(offsets[u], offsets[u + 1]):
                h = splitmix64_numba(np.uint64(adj[pos]) + seed64)
                idx = int(h & idx_mask)
                rho = rho_from_hash_numba(h, p)
                if rho > registers[u, idx]:
                    registers[u, idx] = rho
        return registers

    @nb.njit(cache=True)
    def count_intersection_binary_numba(
        adj: np.ndarray, small0: int, small1: int, large0: int, large1: int
    ) -> int:
        count = 0
        for pos in range(small0, small1):
            x = adj[pos]
            lo = large0
            hi = large1
            while lo < hi:
                mid = (lo + hi) // 2
                if adj[mid] < x:
                    lo = mid + 1
                else:
                    hi = mid
            if lo < large1 and adj[lo] == x:
                count += 1
        return count

    @nb.njit(cache=True)
    def count_intersection_numba(
        offsets: np.ndarray, adj: np.ndarray, a: int, b: int
    ) -> int:
        a0 = offsets[a]
        a1 = offsets[a + 1]
        b0 = offsets[b]
        b1 = offsets[b + 1]
        len_a = a1 - a0
        len_b = b1 - b0
        if len_a == 0 or len_b == 0:
            return 0
        if len_a * 16 < len_b:
            return count_intersection_binary_numba(adj, a0, a1, b0, b1)
        if len_b * 16 < len_a:
            return count_intersection_binary_numba(adj, b0, b1, a0, a1)

        i = a0
        j = b0
        count = 0
        while i < a1 and j < b1:
            av = adj[i]
            bv = adj[j]
            if av == bv:
                count += 1
                i += 1
                j += 1
            elif av < bv:
                i += 1
            else:
                j += 1
        return count

    @nb.njit(cache=True, parallel=True)
    def exact_forward_count_numba(
        edges: np.ndarray, offsets: np.ndarray, adj: np.ndarray
    ) -> int:
        total = 0
        for edge_id in nb.prange(edges.shape[0]):
            total += count_intersection_numba(
                offsets, adj, int(edges[edge_id, 0]), int(edges[edge_id, 1])
            )
        return total

    @nb.njit(cache=True)
    def hll_alpha_numba(register_count: int) -> float:
        if register_count == 16:
            return 0.673
        if register_count == 32:
            return 0.697
        if register_count == 64:
            return 0.709
        return 0.7213 / (1.0 + 1.079 / register_count)

    @nb.njit(cache=True)
    def hll_union_estimate_numba(
        registers: np.ndarray, u: int, v: int, register_count: int
    ) -> float:
        inv_sum = 0.0
        zeros = 0
        for idx in range(register_count):
            ru = registers[u, idx]
            rv = registers[v, idx]
            r = ru if ru >= rv else rv
            inv_sum += math.ldexp(1.0, -int(r))
            if r == 0:
                zeros += 1
        raw = hll_alpha_numba(register_count) * register_count * register_count / inv_sum
        if raw <= 2.5 * register_count and zeros > 0:
            return register_count * math.log(register_count / zeros)
        if raw > 4294967296.0 / 30.0:
            return -4294967296.0 * math.log(1.0 - raw / 4294967296.0)
        return raw

    @nb.njit(cache=True)
    def hll_edge_intersection_numba(
        registers: np.ndarray,
        out_degree: np.ndarray,
        u: int,
        v: int,
        register_count: int,
    ) -> float:
        union_estimate = hll_union_estimate_numba(registers, u, v, register_count)
        estimate = float(out_degree[u] + out_degree[v]) - union_estimate
        upper = float(out_degree[u] if out_degree[u] < out_degree[v] else out_degree[v])
        if estimate < 0.0:
            return 0.0
        if estimate > upper:
            return upper
        return estimate

    @nb.njit(cache=True, parallel=True)
    def pure_hll_count_numba(
        edges: np.ndarray,
        out_degree: np.ndarray,
        registers: np.ndarray,
        register_count: int,
    ) -> tuple[float, int]:
        total = 0.0
        clamp_count = 0
        for edge_id in nb.prange(edges.shape[0]):
            u = int(edges[edge_id, 0])
            v = int(edges[edge_id, 1])
            union_est = hll_union_estimate_numba(registers, u, v, register_count)
            estimate = float(out_degree[u] + out_degree[v]) - union_est
            upper = float(out_degree[u] if out_degree[u] < out_degree[v] else out_degree[v])
            if estimate < 0.0:
                clamp_count += 1
            elif estimate > upper:
                total += upper
                clamp_count += 1
            else:
                total += estimate
        return total, clamp_count

    @nb.njit(cache=True, parallel=True)
    def hybrid_count_numba(
        edges: np.ndarray,
        offsets: np.ndarray,
        adj: np.ndarray,
        out_degree: np.ndarray,
        registers: np.ndarray,
        register_count: int,
        threshold: int,
    ) -> tuple[float, int, int, float, float, int]:
        exact_sum = 0.0
        hll_sum = 0.0
        exact_edges = 0
        hll_edges = 0
        clamp_count = 0
        for edge_id in nb.prange(edges.shape[0]):
            u = int(edges[edge_id, 0])
            v = int(edges[edge_id, 1])
            cost = out_degree[u] if out_degree[u] < out_degree[v] else out_degree[v]
            if cost <= threshold:
                exact_sum += count_intersection_numba(offsets, adj, u, v)
                exact_edges += 1
            else:
                union_est = hll_union_estimate_numba(registers, u, v, register_count)
                estimate = float(out_degree[u] + out_degree[v]) - union_est
                upper = float(out_degree[u] if out_degree[u] < out_degree[v] else out_degree[v])
                if estimate < 0.0:
                    clamp_count += 1
                elif estimate > upper:
                    hll_sum += upper
                    clamp_count += 1
                else:
                    hll_sum += estimate
                hll_edges += 1
        return exact_sum + hll_sum, exact_edges, hll_edges, exact_sum, hll_sum, clamp_count

    @nb.njit(cache=True)
    def max_hybrid_exact_cost_numba(edges: np.ndarray, out_degree: np.ndarray) -> int:
        max_cost = 0
        for edge_id in range(edges.shape[0]):
            u = int(edges[edge_id, 0])
            v = int(edges[edge_id, 1])
            cost = out_degree[u] if out_degree[u] < out_degree[v] else out_degree[v]
            if cost > max_cost:
                max_cost = cost
        return max_cost


def build_hll_registers(fwd: ForwardGraph, p: int, seed: int) -> np.ndarray:
    if HAVE_NUMBA:
        return build_hll_registers_numba(
            fwd.n, fwd.offsets, fwd.adj, p, np.uint64(seed & MASK64)
        )
    return build_hll_registers_py(fwd, p, seed)


def exact_forward_count(fwd: ForwardGraph) -> int:
    if HAVE_NUMBA:
        return int(exact_forward_count_numba(fwd.edges, fwd.offsets, fwd.adj))
    return exact_forward_count_py(fwd)


def pure_hll_count(fwd: ForwardGraph, registers: np.ndarray) -> tuple[float, int]:
    register_count = registers.shape[1]
    if HAVE_NUMBA:
        result = pure_hll_count_numba(fwd.edges, fwd.out_degree, registers, int(register_count))
        return float(result[0]), int(result[1])
    return pure_hll_count_py(fwd, registers)


def hybrid_count(
    fwd: ForwardGraph, registers: np.ndarray, threshold: int
) -> tuple[float, int, int, float, float, int]:
    register_count = registers.shape[1]
    if HAVE_NUMBA:
        return hybrid_count_numba(
            fwd.edges,
            fwd.offsets,
            fwd.adj,
            fwd.out_degree,
            registers,
            int(register_count),
            threshold,
        )
    return hybrid_count_py(fwd, registers, threshold)


def run_networkit_baseline(csr: CSRGraph, threads: int | None) -> tuple[float, float, float]:
    try:
        import networkit as nk
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "NetworKit is required for the baseline. Install with "
            "`python -m pip install -r requirements.txt`, or omit baseline "
            "from --run."
        ) from exc

    if threads is not None:
        nk.setNumberOfThreads(int(threads))

    build_start = time.perf_counter()
    graph = nk.Graph(csr.n, weighted=False, directed=False, edgesIndexed=False)
    us = csr.edges[:, 0].tolist()
    vs = csr.edges[:, 1].tolist()
    for u, v in zip(us, vs):
        graph.addEdge(u, v)
    graph.indexEdges()
    build_seconds = time.perf_counter() - build_start

    run_start = time.perf_counter()
    scorer = nk.sparsification.TriangleEdgeScore(graph)
    scorer.run()
    run_seconds = time.perf_counter() - run_start
    return float(np.sum(scorer.scores()) / 3.0), build_seconds, run_seconds


def timed_result(method: str, fn) -> tuple[float, float]:
    start = time.perf_counter()
    value = fn()
    elapsed = time.perf_counter() - start
    return value, elapsed


def validate_hll_precision(p: int) -> None:
    if p < 4 or p > 16:
        raise ValueError("Use 4 <= --hll-p <= 16 for this implementation.")


def validate_args(args: argparse.Namespace) -> None:
    methods = {item.strip() for item in args.run.split(",") if item.strip()}
    if methods & {"pure-hll", "hybrid"}:
        validate_hll_precision(args.hll_p)
    if args.hybrid_threshold < 0:
        raise ValueError("--hybrid-threshold must be non-negative.")
    if args.max_hll_gb <= 0:
        raise ValueError("--max-hll-gb must be positive.")
    if args.threads is not None and args.threads <= 0:
        raise ValueError("--threads must be positive.")
    if args.numba_threads is not None and args.numba_threads <= 0:
        raise ValueError("--numba-threads must be positive.")


def maybe_build_hll(
    fwd: ForwardGraph, p: int, seed: int, max_hll_gb: float
) -> tuple[np.ndarray, float]:
    validate_hll_precision(p)
    register_count = 1 << p
    required_bytes = fwd.n * register_count
    limit_bytes = int(max_hll_gb * (1024**3))
    if required_bytes > limit_bytes:
        raise MemoryError(
            f"HLL register matrix would need {required_bytes / (1024**3):.2f} GiB "
            f"({fwd.n} nodes x {register_count} registers). Lower --hll-p or "
            "raise --max-hll-gb."
        )

    start = time.perf_counter()
    registers = build_hll_registers(fwd, p, seed)
    elapsed = time.perf_counter() - start
    return registers, elapsed


def max_hybrid_exact_cost(fwd: ForwardGraph) -> int:
    if fwd.m == 0:
        return 0
    if HAVE_NUMBA:
        return int(max_hybrid_exact_cost_numba(fwd.edges, fwd.out_degree))
    max_cost = 0
    for u_raw, v_raw in fwd.edges:
        u = int(u_raw)
        v = int(v_raw)
        cost = int(min(fwd.out_degree[u], fwd.out_degree[v]))
        if cost > max_cost:
            max_cost = cost
    return max_cost


def run_methods(
    csr: CSRGraph,
    fwd: ForwardGraph,
    methods: set[str],
    args: argparse.Namespace,
) -> list[MethodResult]:
    valid_methods = {"baseline", "pure-hll", "hybrid", "exact"}
    unknown = methods - valid_methods
    if unknown:
        raise ValueError(f"Unknown methods in --run: {', '.join(sorted(unknown))}")

    results: list[MethodResult] = []
    registers: np.ndarray | None = None
    hll_preprocess_seconds = 0.0

    if "baseline" in methods:
        triangles, build_seconds, run_seconds = run_networkit_baseline(
            csr, args.threads
        )
        elapsed = build_seconds + run_seconds
        results.append(
            MethodResult(
                method="baseline-networkit-triangle-edge-score",
                triangles=triangles,
                total_seconds=elapsed,
                preprocess_seconds=build_seconds,
                query_seconds=run_seconds,
                details={"threads": args.threads or "networkit-default"},
            )
        )

    if "exact" in methods:
        triangles, elapsed = timed_result("exact", lambda: exact_forward_count(fwd))
        results.append(
            MethodResult(
                method="exact-forward-degree-order",
                triangles=float(triangles),
                total_seconds=elapsed,
                query_seconds=elapsed,
            )
        )

    hybrid_needs_hll = "hybrid" in methods and (
        args.hybrid_threshold < max_hybrid_exact_cost(fwd)
    )
    need_hll = "pure-hll" in methods or hybrid_needs_hll
    if need_hll:
        registers, hll_preprocess_seconds = maybe_build_hll(
            fwd, args.hll_p, args.seed, args.max_hll_gb
        )

    if "pure-hll" in methods:
        assert registers is not None
        _t0 = time.perf_counter()
        triangles, clamp_count = pure_hll_count(fwd, registers)
        query_seconds = time.perf_counter() - _t0
        results.append(
            MethodResult(
                method="pure-hll-forward",
                triangles=triangles,
                total_seconds=hll_preprocess_seconds + query_seconds,
                preprocess_seconds=hll_preprocess_seconds,
                query_seconds=query_seconds,
                details={
                    "hll_p": args.hll_p,
                    "registers_per_node": 1 << args.hll_p,
                    "estimated_edges": fwd.m,
                    "clamp_count": clamp_count,
                    "clamp_pct": round(100.0 * clamp_count / fwd.m, 4) if fwd.m > 0 else 0.0,
                },
            )
        )

    if "hybrid" in methods:
        if registers is None:
            registers = np.empty((0, 0), dtype=np.uint8)
        start = time.perf_counter()
        triangles, exact_edges, hll_edges, exact_sum, hll_sum, clamp_count = hybrid_count(
            fwd, registers, args.hybrid_threshold
        )
        query_seconds = time.perf_counter() - start
        coverage = hll_edges / fwd.m if fwd.m > 0 else 0.0
        clamp_pct = 100.0 * clamp_count / hll_edges if hll_edges > 0 else 0.0
        results.append(
            MethodResult(
                method="hybrid-forward-exact-hll",
                triangles=triangles,
                total_seconds=hll_preprocess_seconds + query_seconds,
                preprocess_seconds=hll_preprocess_seconds,
                query_seconds=query_seconds,
                details={
                    "hll_p": args.hll_p,
                    "registers_per_node": 1 << args.hll_p,
                    "threshold": args.hybrid_threshold,
                    "exact_edges": exact_edges,
                    "hll_edges": hll_edges,
                    "exact_contribution": exact_sum,
                    "hll_contribution": hll_sum,
                    "coverage": round(coverage, 6),
                    "clamp_count": clamp_count,
                    "clamp_pct": round(clamp_pct, 4),
                },
            )
        )

    return results


def print_report(csr: CSRGraph, fwd: ForwardGraph, results: list[MethodResult]) -> None:
    print(f"graph: nodes={csr.n} edges={csr.m}")
    print(
        "forward: "
        f"oriented_edges={fwd.m} "
        f"max_out_degree={int(fwd.out_degree.max(initial=0))} "
        f"mean_out_degree={float(fwd.out_degree.mean()) if fwd.n else 0.0:.3f}"
    )
    print(f"numba: {'yes' if HAVE_NUMBA else 'no'}")
    print()

    exact_triangles = next(
        (r.triangles for r in results if r.method == "exact-forward-degree-order"), None
    )
    ref_triangles = exact_triangles or next(
        (r.triangles for r in results if r.method == "baseline-networkit-triangle-edge-score"),
        None,
    )
    exact_time = next(
        (r.total_seconds for r in results if r.method == "exact-forward-degree-order"), None
    )
    ref_label = "exact" if exact_triangles is not None else "nk-baseline"

    col_w = [38, 16, 10, 14, 10, 13, 11, 8]
    headers = ["method", "triangles", "rel_err", "|bias|", "e2e_s", f"spdup_{ref_label}", "coverage", "clamp%"]
    header = "  ".join(h.rjust(w) for h, w in zip(headers, col_w))
    print(header)
    print("-" * len(header))

    for result in results:
        rel_str = bias_str = spdup_str = cov_str = clamp_str = ""

        if ref_triangles is not None:
            abs_err = abs(result.triangles - ref_triangles)
            bias_str = f"{abs_err:.1f}"
            if ref_triangles != 0:
                rel_str = f"{abs_err / abs(ref_triangles):.4%}"
            else:
                rel_str = "0.00%" if result.triangles == 0 else "inf"

        if exact_time is not None:
            if result.method == "exact-forward-degree-order":
                spdup_str = "1.00x"
            elif result.total_seconds > 0:
                spdup_str = f"{exact_time / result.total_seconds:.2f}x"

        d = result.details or {}
        if result.method == "pure-hll-forward":
            cov_str = "100.0%"
            if "clamp_pct" in d:
                clamp_str = f"{d['clamp_pct']:.2f}%"
        elif result.method == "hybrid-forward-exact-hll":
            if "coverage" in d:
                cov_str = f"{d['coverage'] * 100:.1f}%"
            if "clamp_pct" in d:
                clamp_str = f"{d['clamp_pct']:.2f}%"

        row = [
            result.method,
            f"{result.triangles:.1f}",
            rel_str,
            bias_str,
            f"{result.total_seconds:.4f}",
            spdup_str,
            cov_str,
            clamp_str,
        ]
        print("  ".join(v.rjust(w) for v, w in zip(row, col_w)))

    print()
    for result in results:
        if result.details:
            print(f"{result.method} details:")
            for key, value in result.details.items():
                print(f"  {key}: {value}")
            print()


def result_payload(
    csr: CSRGraph, fwd: ForwardGraph, results: list[MethodResult]
) -> dict[str, object]:
    return {
        "graph": {
            "nodes": csr.n,
            "edges": csr.m,
            "forward_max_out_degree": int(fwd.out_degree.max(initial=0)),
            "forward_mean_out_degree": float(fwd.out_degree.mean()) if fwd.n else 0.0,
        },
        "numba": HAVE_NUMBA,
        "results": [asdict(result) for result in results],
    }


def make_self_test_graph() -> CSRGraph:
    # K4 has exactly 4 triangles. The path edge adds no triangle.
    edges = np.asarray(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 2],
            [1, 3],
            [2, 3],
            [3, 4],
            [4, 5],
        ],
        dtype=np.int64,
    )
    return build_csr(6, edges)


def run_self_test() -> None:
    csr = make_self_test_graph()
    fwd = build_forward_graph(csr)
    exact = exact_forward_count(fwd)
    if exact != 4:
        raise AssertionError(f"exact self-test failed: expected 4, got {exact}")
    registers, _ = maybe_build_hll(fwd, p=8, seed=1, max_hll_gb=1.0)
    hybrid, exact_edges, hll_edges, _, _, _ = hybrid_count(fwd, registers, threshold=10**9)
    if abs(hybrid - 4.0) > 1e-9 or hll_edges != 0:
        raise AssertionError(
            "all-exact hybrid self-test failed: "
            f"triangles={hybrid}, exact_edges={exact_edges}, hll_edges={hll_edges}"
        )
    print("self-test passed: exact=4, all-exact hybrid=4")


def warm_numba_kernels(p: int, seed: int) -> None:
    if not HAVE_NUMBA:
        return
    csr = make_self_test_graph()
    fwd = build_forward_graph(csr)
    registers = build_hll_registers(fwd, p, seed)
    max_hybrid_exact_cost(fwd)
    exact_forward_count(fwd)
    pure_hll_count(fwd, registers)
    hybrid_count(fwd, registers, threshold=1)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    validate_args(args)

    if HAVE_NUMBA and args.numba_threads is not None:
        nb.set_num_threads(int(args.numba_threads))

    if args.self_test:
        run_self_test()
        return 0

    if args.edge_list is None:
        raise SystemExit("edge_list is required unless --self-test is used.")
    if not args.edge_list.exists():
        raise SystemExit(f"edge list not found: {args.edge_list}")

    methods = {item.strip() for item in args.run.split(",") if item.strip()}
    if not args.no_warmup and methods & {"exact", "pure-hll", "hybrid"}:
        warm_numba_kernels(args.hll_p, args.seed)

    csr = read_edge_list(
        args.edge_list,
        delimiter=args.delimiter,
        comment_prefixes=args.comment_prefixes,
        no_remap=args.no_remap,
        one_indexed=args.one_indexed,
        assume_simple=args.assume_simple,
    )
    fwd = build_forward_graph(csr)
    results = run_methods(csr, fwd, methods, args)
    print_report(csr, fwd, results)

    if args.json_out is not None:
        payload = result_payload(csr, fwd, results)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote JSON results to {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
