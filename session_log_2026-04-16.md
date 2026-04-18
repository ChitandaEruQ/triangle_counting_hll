# 会话日志 — 2026-04-16

## 项目概述

GPU 三角形计数实验，目标图：**com-Orkut**（3M 节点，117M 边，已知三角形数 627,584,181）。

---

## 本次会话完成的工作

### 背景（上次会话摘要）

1. 下载并解压 com-orkut.ungraph 数据集
2. 解压 `src.ZIP`，分析并修复 `exp_gpu.py` 及相关文件中的 bug
3. 配置 CUDA/numba 环境（Windows 10，安装 `nvidia-cuda-nvcc-cu12`，设置 `CUDA_HOME`）
4. 设计并实现完整的 GPU 实验框架，比较多种三角形计数方法
5. 上次会话末尾：重新设计 ground truth 来源策略，改用 SNAP 公布值而非 CPU Exact

---

## 本次会话核心修改

### 文件：`experiment/exp_gpu.py`

#### 1. `[1]` Ground Truth / CPU Exact 块（替换旧的 `if no_cpu_exact`）

**旧逻辑**（已废弃）：
```python
if no_cpu_exact:
    print("\n[1] CPU Exact skipped (--no-cpu-exact).")
    ...
else:
    cpu_count, cpu_time, _ = exact_triangle_count(forward_adj)
    truth = cpu_count
    ref_time = cpu_time
```

**新逻辑**：
```python
if known_truth is not None:
    # 使用外部传入的已知值（如 SNAP 公布值）
    truth        = known_truth
    truth_source = 'published (e.g. SNAP)'
    print(f"\n[1] Using pre-verified truth: {truth:,}  (skipping CPU Exact)")
    if validate:
        print("    NOTE: --validate ignored when --truth is supplied.")
elif validate:
    # 只对小图运行 CPU Exact 进行验证
    cpu_count, cpu_time, _ = exact_triangle_count(forward_adj)
    truth        = cpu_count
    truth_source = 'CPU Exact (validation run)'
    _add_row('CPU Exact', cpu_count, 0.0, 0, count_time=cpu_time, ...)
else:
    # 既无 --truth 也无 --validate：ground truth 从 GPU Exact 获取
    truth        = None
    truth_source = 'GPU Exact'

if cpu_ref_time is not None:
    ref_time  = cpu_ref_time
    ref_label = cpu_ref_label or 'CPU literature'
```

#### 2. `[2]` GPU Exact 块（替换旧的第二个 `if no_cpu_exact`）

```python
if truth is None:
    truth        = gpu_ex_count
    truth_source = 'GPU Exact'

if ref_time is None:
    ref_time  = gpu_ex_mean
    ref_label = 'GPU Exact'
```

#### 3. 打印语句中 `ref_time` 空值保护

cuGraph、DOULION、GPU Hybrid 打印 speedup 时，加了 `if ref_time else ""` 防止在 `ref_time` 为 None 时崩溃：

```python
# DOULION
+ (f"  spdup={ref_time/res['mean_time']:.2f}x"
   if ref_time and res['mean_time'] else "")

# Hybrid
+ (f"  spdup_cnt={ref_time/cnt_mean:.2f}x"
   f"  spdup_e2e={ref_time/e2e:.2f}x"
   if ref_time else "")

# cuGraph
+ (f"  spdup_cnt={ref_time/cg_cnt_mean:.2f}x"
   f"  spdup_e2e={ref_time/cg_e2e:.2f}x"
   if ref_time else "")
```

#### 4. CLI 参数更新

**旧参数**（已删除）：
```
--no-cpu-exact
```

**新参数**：
```
--truth N              预验证的精确三角形数（如 SNAP 公布值），跳过 CPU Exact
--cpu-ref-time SECS    文献中 CPU 基准时间（秒），作为 speedup 分母
--cpu-ref-label LABEL  --cpu-ref-time 对应的引用标签（如 "Shun2015-8core"）
--validate             对当前图运行 CPU Exact 进行验证（仅建议小图使用）
```

**函数调用更新**：
```python
run_exp_gpu(args.graph_path, args.output_dir,
            skip_cugraph=args.skip_cugraph,
            known_truth=args.truth,
            cpu_ref_time=args.cpu_ref_time,
            cpu_ref_label=args.cpu_ref_label,
            validate=args.validate)
```

---

## 实验方案总结

### 方法列表

| # | 方法 | 类型 |
|---|------|------|
| 1 | CPU Exact（仅 validate 模式） | 精确，CPU |
| 2 | GPU Exact（ours） | 精确，GPU |
| 3 | cuGraph Exact | 精确，GPU（需要 RAPIDS） |
| 4 | DOULION q=0.10（10 seeds） | 近似，CPU |
| 5 | DOULION q=0.05（10 seeds） | 近似，CPU |
| 6 | GPU Hybrid p=8 γ=64 | 近似，GPU |
| 7 | GPU Hybrid p=8 γ=128 | 近似，GPU |
| 8 | GPU Hybrid p=8 γ=256 | 近似，GPU |
| 9 | GPU Hybrid p=10 γ=64 | 近似，GPU |
| 10 | GPU Hybrid p=10 γ=128 | 近似，GPU |

### Ground Truth 优先级

1. `--truth VALUE`（SNAP 等公布的已知值）——最优，无需计算
2. `--validate` 结果（CPU Exact，仅小图）
3. GPU Exact 结果（兜底）

### Speedup 参考基准

- **vs ref**：`--cpu-ref-time` 提供的文献 CPU 时间，或 GPU Exact 时间（若未提供）
- **vs GPU Exact**：与自身精确 GPU kernel 对比
- **vs cuGraph**：与 NVIDIA 官方 GPU 库对比

### E2E 定义

`H2D transfer + sketch_alloc + sketch_build + kernel_count`
不含文件 IO、图解析和方向化（所有方法共享的预处理）。

### 指标列表

| 指标 | 含义 |
|------|------|
| estimate | 估计值 |
| rel_error | 相对误差 |
| bias | 偏差（estimate - truth） |
| spdup_cnt | count kernel speedup vs ref |
| spdup_e2e | e2e speedup vs ref |
| spdup_vs_gpu_exact | count speedup vs GPU Exact |
| spdup_vs_cugraph | count speedup vs cuGraph |
| coverage | HLL 路由的有向边比例 |
| clamp_ratio | HLL 估计被截断的比例 |
| n_sketched | 建立 sketch 的节点数 |

---

## 用法示例

```bash
# 大图（com-Orkut）：使用 SNAP 已知值，跳过 CPU Exact
python experiment/exp_gpu.py datasets/com-orkut.ungraph.txt results/ \
    --truth 627584181 --skip-cugraph

# 大图 + 文献 CPU 基准时间
python experiment/exp_gpu.py datasets/com-orkut.ungraph.txt results/ \
    --truth 627584181 \
    --cpu-ref-time 30.0 --cpu-ref-label "Shun2015-multicore" \
    --skip-cugraph

# 小图：运行 CPU Exact 验证 GPU kernel 正确性
python experiment/exp_gpu.py datasets/small.txt results/ --validate
```

---

## 当前实验结果（gpu_only 模式，exp_gpu.json）

| 方法 | 估计值 | 相对误差 | Speedup(cnt) | 计数时间 |
|------|--------|---------|-------------|---------|
| GPU Exact (ours) | 627,584,181 | 0.000% | 1.00x | 1.808s |
| GPU Hybrid p=8 γ=64 | 698,993,714 | 11.38% | 1.79x | 1.011s |
| GPU Hybrid p=8 γ=128 | 657,905,231 | 4.83% | 1.94x | 0.933s |
| GPU Hybrid p=8 γ=256 | 628,956,903 | 0.22% | 1.40x | 1.293s |

Ground truth: 627,584,181（GPU Exact 验证值）

---

## 待办

- [ ] 安装 WSL2 + Ubuntu 22.04，在 Linux 环境下安装 cuGraph（RAPIDS 不支持 Windows）
- [ ] 运行完整实验（包含 p=10 系列和 DOULION），生成 exp_gpu_full.json
- [ ] 如有文献 CPU 基准时间，通过 `--cpu-ref-time` 补充对比
