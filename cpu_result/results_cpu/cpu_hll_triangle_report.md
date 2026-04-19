# 基于 CPU 的 HLL 近似三角计数实验报告

生成日期：2026-04-19  
数据来源：`results/*.json`，共 28 个结果文件，覆盖 9 个图数据集。

## 1. 实验目的

本实验评估 CPU 上三类三角计数方法的效果：

1. **Baseline**：NetworKit 标准实现 `TriangleEdgeScore`，对边三角数求和后除以 3。
2. **Pure-HLL**：对所有 forward edge 使用 HLL 估计 forward 邻居集合交集。
3. **Hybrid**：对低成本 forward edge 使用 exact intersection，对高成本 forward edge 使用 HLL。

其中 exact 和 hybrid 均基于 degree-ordered forward graph。每条无向边按 `(degree, node_id)` 从低 rank 指向高 rank，对每条有向边 `u -> v` 统计 `|N+(u) ∩ N+(v)|`，因此每个三角形只计一次。

## 2. 结果解读口径

本报告以 `exact-forward-degree-order` 作为 ground truth。若文件中存在 NetworKit baseline，则同时检查 NetworKit 与 exact 的三角数是否一致。

需要注意：从已有 JSON 的时间特征看，部分结果疑似包含 Numba 首次 JIT 编译开销。例如一些 all-exact hybrid 结果比单独 exact 快几十倍，这通常不是算法差异，而是计时口径问题。因此：

- **三角形数量、相对误差、HLL 边占比**可以作为主要结论依据。
- **运行时间和加速比**可用于观察趋势，但最终汇报或论文数据建议用已加入 warm-up 的当前脚本重跑。

## 3. 默认配置结果

默认配置指文件名中的 `p10_t64`，即 HLL precision `p=10`，hybrid threshold `t=64`。

| 数据集 | 点数 | 边数 | 三角形 exact | Exact(s) | Hybrid(s) | Hybrid 误差 | HLL 边占比 | Hybrid/Exact 加速 | Pure-HLL(s) | Pure 误差 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| as-skitter | 1,696,415 | 11,095,298 | 28,769,868 | 1.003 | 1.229 | 0.456% | 1.635% | 0.82x | 14.267 | 4.540% |
| ca-AstroPh | 18,772 | 198,050 | 1,351,441 | 0.533 | 0.013 | 0.000% | 0.000% | 42.10x | - | - |
| ca-HepTh | 9,877 | 25,973 | 28,339 | 0.233 | 0.019 | 0.000% | 0.000% | 12.33x | 0.048 | -0.252% |
| com-amazon.ungraph | 334,863 | 925,872 | 667,129 | 0.431 | 0.013 | 0.000% | 0.000% | 32.64x | - | - |
| com-orkut.ungraph | 3,072,441 | 117,185,083 | 627,584,181 | 26.919 | 95.293 | 1.901% | 23.021% | 0.28x | - | - |
| com-youtube.ungraph | 1,134,890 | 2,987,624 | 3,056,386 | 0.500 | 0.171 | -0.078% | 0.131% | 2.92x | - | - |
| facebook_combined | 4,039 | 88,234 | 1,612,010 | 0.524 | 0.069 | -0.760% | 8.154% | 7.59x | 0.958 | -2.431% |
| roadNet-CA | 1,965,206 | 2,766,607 | 120,676 | 0.361 | 0.295 | 0.000% | 0.000% | 1.22x | 3.535 | 5.322% |
| roadNet-TX | 1,379,917 | 1,921,660 | 82,869 | 0.318 | 0.213 | 0.000% | 0.000% | 1.49x | 2.500 | 5.418% |

主要观察：

- 在 9 个默认配置结果中，hybrid 的平均绝对相对误差约为 **0.355%**，中位数为 **0%**，最大误差来自 `com-orkut.ungraph`，约 **1.901%**。
- 当 HLL 边占比很低时，hybrid 误差通常很小。`com-youtube.ungraph` 在只有 **0.131%** 边走 HLL 时，误差约 **-0.078%**。
- 当大量边走 HLL 时，误差和时间都会恶化。`com-orkut.ungraph` 在 `t=64` 下有 **23.021%** 边走 HLL，误差达到 **1.901%**，且总时间慢于 exact。
- Pure-HLL 在已有结果中整体不占优。它对所有边都合并 HLL register，复杂度近似为 `O(m * 2^p)`，在 `p=10` 时每条边需要扫描 1024 个 register，导致查询时间较大；同时交集估计的误差也明显高于 hybrid。

## 4. NetworKit Baseline 校验

存在 baseline 的 5 个结果中，NetworKit 与 degree-ordered exact 的三角形数量完全一致。

| 数据集 | NetworKit 三角形 | Exact 三角形 | 差值 | NetworKit(s) | Exact(s) | NetworKit/Exact 时间比 |
|---|---:|---:|---:|---:|---:|---:|
| as-skitter | 28,769,868 | 28,769,868 | 0 | 22.609 | 1.003 | 22.55x |
| ca-HepTh | 28,339 | 28,339 | 0 | 0.068 | 0.233 | 0.29x |
| facebook_combined | 1,612,010 | 1,612,010 | 0 | 0.210 | 0.524 | 0.40x |
| roadNet-CA | 120,676 | 120,676 | 0 | 6.256 | 0.361 | 17.34x |
| roadNet-TX | 82,869 | 82,869 | 0 | 4.538 | 0.318 | 14.27x |

结论：

- exact forward 实现的逻辑正确性得到 NetworKit 校验。
- 时间上存在数据集差异。大图和 road network 上，本实现明显快于 NetworKit baseline；小图上 NetworKit 可能因 C++ 内部实现和 Python 调用路径不同而更快。
- 由于当前结果文件可能混入 JIT 编译时间，时间结论建议重跑后再作为最终汇报数据。

## 5. Hybrid Threshold 分析

以下为 `p=10` 时不同 threshold 的结果。threshold 越大，更多 edge 使用 exact，HLL 边占比下降，误差通常下降。

### ca-AstroPh

| threshold | Hybrid(s) | 相对误差 | HLL 边占比 | HLL 边数 |
|---:|---:|---:|---:|---:|
| 16 | 0.067 | -0.607% | 40.073% | 79,364 |
| 64 | 0.013 | 0.000% | 0.000% | 0 |
| 256 | 0.012 | 0.000% | 0.000% | 0 |
| 1024 | 0.011 | 0.000% | 0.000% | 0 |

### com-amazon.ungraph

| threshold | Hybrid(s) | 相对误差 | HLL 边占比 | HLL 边数 |
|---:|---:|---:|---:|---:|
| 16 | 0.013 | 0.000% | 0.000% | 0 |
| 64 | 0.013 | 0.000% | 0.000% | 0 |
| 256 | 0.029 | 0.000% | 0.000% | 0 |
| 1024 | 0.013 | 0.000% | 0.000% | 0 |

### com-youtube.ungraph

| threshold | Hybrid(s) | 相对误差 | HLL 边占比 | HLL 边数 |
|---:|---:|---:|---:|---:|
| 16 | 0.646 | 0.447% | 14.604% | 436,316 |
| 64 | 0.171 | -0.078% | 0.131% | 3,924 |
| 256 | 0.055 | 0.000% | 0.000% | 0 |
| 1024 | 0.055 | 0.000% | 0.000% | 0 |

### com-orkut.ungraph

| threshold | Hybrid(s) | 相对误差 | HLL 边占比 | HLL 边数 |
|---:|---:|---:|---:|---:|
| 16 | 177.574 | 6.325% | 89.444% | 104,815,013 |
| 64 | 95.293 | 1.901% | 23.021% | 26,977,276 |
| 256 | 26.812 | -0.111% | 0.679% | 795,253 |
| 1024 | 23.701 | 0.000% | 0.000% | 0 |

结论：

- threshold 是 hybrid 的关键参数。
- 对 `com-youtube.ungraph`，`t=64` 已经把 HLL 边占比压到 **0.131%**，误差低于 **0.1%**。
- 对 `com-orkut.ungraph`，`t=64` 仍有 **26,977,276** 条 HLL 边，占比 **23.021%**，这会导致 HLL merge 成为主要开销。`t=256` 将 HLL 边占比降到 **0.679%**，误差降到 **-0.111%**。
- 对低 forward out-degree 的图，较高 threshold 会退化为全 exact，误差为 0。

## 6. HLL Precision 分析

以下为 `threshold=64` 时不同 HLL precision 的结果。register 数为 `2^p`，当前实现使用 `uint8` dense matrix，因此 HLL 内存约为 `nodes * 2^p` bytes。

### ca-AstroPh

| p | registers/node | HLL 内存估计 | Hybrid(s) | 相对误差 |
|---:|---:|---:|---:|---:|
| 8 | 256 | 0.004 GiB | 0.036 | 0.000% |
| 10 | 1,024 | 0.018 GiB | 0.013 | 0.000% |
| 12 | 4,096 | 0.072 GiB | 0.037 | 0.000% |

### com-amazon.ungraph

| p | registers/node | HLL 内存估计 | Hybrid(s) | 相对误差 |
|---:|---:|---:|---:|---:|
| 8 | 256 | 0.080 GiB | 0.043 | 0.000% |
| 10 | 1,024 | 0.319 GiB | 0.013 | 0.000% |
| 12 | 4,096 | 1.277 GiB | 0.130 | 0.000% |

### com-youtube.ungraph

| p | registers/node | HLL 内存估计 | Hybrid(s) | 相对误差 |
|---:|---:|---:|---:|---:|
| 8 | 256 | 0.271 GiB | 0.105 | 0.057% |
| 10 | 1,024 | 1.082 GiB | 0.171 | -0.078% |
| 12 | 4,096 | 4.329 GiB | 0.687 | -0.042% |

### com-orkut.ungraph

| p | registers/node | HLL 内存估计 | Hybrid(s) | 相对误差 |
|---:|---:|---:|---:|---:|
| 8 | 256 | 0.733 GiB | 38.818 | 12.278% |
| 10 | 1,024 | 2.930 GiB | 95.293 | 1.901% |

结论：

- 提高 `p` 会线性增加每条 HLL merge 的 register 扫描成本，并以 `4x` 方式增加内存。
- 对 `com-youtube.ungraph`，`p=12` 相比 `p=10` 误差改善很小，但时间和内存显著增加。
- 对 `com-orkut.ungraph`，`p=8` 明显不够，误差达到 **12.278%**；`p=10` 将误差降到 **1.901%**，但时间也显著增加。
- 当前实现下，不建议盲目提高 `p`。更优先的策略是调高 threshold，减少 HLL 边占比。

## 7. 主要结论

1. **Exact forward 实现可靠。**  
   在有 NetworKit baseline 的数据集上，exact forward 与 NetworKit 的三角形数量完全一致。

2. **Pure-HLL 不适合作为主方法。**  
   Pure-HLL 对所有 forward edge 都执行 HLL union，成本为 `O(m * 2^p)`。在 `p=10` 下，Pure-HLL 通常比 exact 或 hybrid 更慢，且误差更高。

3. **Hybrid 的效果取决于 HLL 边占比。**  
   当 HLL 边占比低于约 1% 时，误差通常可以控制在 0.1% 量级；当 HLL 边占比达到 10% 以上，误差和时间都会明显恶化。

4. **`com-orkut.ungraph` 是主要压力场景。**  
   该图边数超过 1.17 亿，`t=64` 下仍有 23.021% 的边走 HLL，导致 hybrid 慢于 exact 且误差接近 2%。该图需要更高 threshold 或更节省的 sketch 策略。

5. **HLL dense matrix 内存是当前实现的重要瓶颈。**  
   `com-orkut.ungraph` 在 `p=10` 下仅 HLL register matrix 就约 **2.93 GiB**，`p=12` 会约 **11.72 GiB**。这解释了为什么大图上提高 `p` 的收益不一定划算。

## 8. 建议的最终实验配置

若以误差低于 0.5% 为目标，建议：

- 小图或低 forward out-degree 图：直接 exact 或 hybrid with high threshold，避免 HLL。
- `com-youtube.ungraph`：`p=10, threshold=64` 已可接受，误差约 **-0.078%**。
- `as-skitter`：`p=10, threshold=64` 误差约 **0.456%**，但速度不优；可尝试提高 threshold 到 128 或 256。
- `com-orkut.ungraph`：`p=10, threshold=256` 明显优于 `threshold=64`，误差约 **-0.111%**，HLL 边占比约 **0.679%**。

## 9. 后续改进建议

1. **重跑全部结果，统一计时口径。**  
   使用当前脚本默认 warm-up，固定 `--numba-threads` 和 NetworKit `--threads`，每个配置重复 3 到 5 次，报告 median 和 min。

2. **只为 HLL 相关端点构建 sketch。**  
   当前只要 hybrid 有 HLL 边，就为所有节点分配 `n * 2^p` 的 dense matrix。对 HLL 边很少的配置，应改为只为 HLL edge 涉及的端点构建 compact sketch。

3. **threshold 应按成本模型选择。**  
   HLL merge 成本约为 `2^p`，exact 成本约为 `min(out_degree[u], out_degree[v])`。因此 threshold 不应只手工指定，可根据 `2^p` 和机器性能自动选择。

4. **补充误差稳定性实验。**  
   对 HLL 使用多个 seed，报告平均误差、标准差和最大误差，避免单 seed 偶然性。

5. **考虑替代交集估计器。**  
   HLL 用 inclusion-exclusion 估计交集时，当真实交集远小于 union 时方差较高。可以对比 KMV/MinHash 或专门的 set intersection sketch。

## 10. 汇报用一句话总结

本实验表明，degree-ordered exact forward 实现可以作为可靠强基线；纯 HLL 因对所有边执行大规模 register merge 而不具竞争力；hybrid 只有在把 HLL 边占比压到很低时才同时具备较低误差和较好性能，关键参数是 threshold，其优先级高于盲目增大 HLL precision。
