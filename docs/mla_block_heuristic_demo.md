# MLA block 启发式函数 demo：从 649 个测试点拟合

日期：2026-11-18
数据：`benchmarks/softmax_attention/results/mla_tune`、`mla_tune_refine`、`mla_tune_remaining`（649 个 cell，0 个 error）
骨架：`flywheel_tpu/softmax_attention/kernels/mla_decode.py` 的 `default_mla_config`
完整测试报告：`benchmarks/softmax_attention/results/mla_tune_refine/final_report.md`
逐 cell 分档表：`docs/mla_block_heuristic_fit_table.md`
逐 cell 分档表 2（decode / verify 分支版）：`docs/mla_block_heuristic_fit_table2.md`
逐 cell 分档表 3（scenario × dtype 分支版）：`docs/mla_block_heuristic_fit_table3.md`

## 0. 先给结论

原来的 `default_mla_config` 在 v6e 上几乎总是输出 (512, 8) 或 (1024, 8)。数据表明，这组常量对现在的 kernel 太保守：649 个 cell 里最优 block 的众数是 4096；最优 step 的 token 数（block × items）大多落在 4096 到 16384，其中 4096 和 8192 最多（合计约 70%）。

我按原来的骨架改了一个 demo：

- `MAX_BLOCK_KV`（v6e）：1024 → **4096**
- `BOUND_DIVISOR`：4 → **1**（长上下文直接用 `next_pow2(S)`，不再除以 4）
- `TARGET_STEP_TOKENS`：16384 → **8192**
- `MIN_ITEMS` 的缩小规则：**取消**（设为 1）
- items 取整：新增一条 **items 取 2 的幂**（1、2、4、8）

用已有的 649 个 cell 做离线比较（没有跑新的 kernel 测试）：

| 口径 | 结果 |
|---|---|
| 全部 649 个 cell，短测（screening）时间 | 平均 regret 2.46%，中位数 0.60%，p90 7.00%，最大 38.22%；85.8% 在 5% 以内 |
| 357 个有完整复测（full-protocol）时间的 cell | 平均 regret 0.77%，中位数 0.15%，p90 2.43%，最大 15.61% |
| 334 个「demo 和现有线上配置都有完整复测时间」的 cell | 现有线上配置（旧表项或旧启发式）平均 6.36%，demo 平均 0.78%；demo 在 53% 的 cell 上快 1% 以上，45% 基本打平（±1%），只有 2% 慢 1% 以上，最慢慢 7% |

一句话：**大 batch 和 verify 已经很准，小 batch 的 decode 还有几个真正的“悬崖”拟合不了。** 这不是“每个 cell 都精确命中”，但作为默认启发式已经明显优于现在这版。

## 1. 你的理解：哪些对，哪些要补充

你的判断大方向是对的：大部分 cell 的最优 block 和次优 block 差距很小，确实可以不用查表，直接用一个公式算出来。我的补充有三点。

### 1.1 “每个 cell 的最优 block”不是硬标签

数据里的 `best_actual` 是“候选网格里、并且进入最终复测的 top-3”中的最优，不是理论全局最优。候选网格是：

- block：512、1024、2048、4096、8192、16384
- items：1、2、4、8

每个 cell 最多 24 个 target 组合，拟合后会有重复（例如 target 4096 在 seq 2560 上会被页对齐成 3072）。所以像 (3072, 5)、(5120, 1) 这些“奇怪”的最优值，其实是页对齐的副产物，不是独立的候选。

更关键的是，**很多 cell 的第一名和第二名只差 0.0% 到 1%**。例如：

- decode bf16 h32 b64 s4096：第一名 (4096, 2) 20.050 ms，第二名 (1024, 8) 20.056 ms，margin 0.03%。
- verify bf16 h64 b64 s4096 t8：第一名 (4096, 1) 3.522 ms，第二名 (4096, 2) 3.546 ms，margin 0.68%。

这些 cell 选第一名还是第二名，性能上几乎没区别。所以评价 demo 时，**exact match 不是合适的指标，regret 才是**。

### 1.2 光滑函数拟合不了小 batch 的悬崖

小 batch 的 decode 存在真正的突变。例如：

- decode fp8 h16 b8 s2560：最优 (512, 8) 3.562 ms，第二名 (2048, 2) 4.415 ms，margin 23.93%。
- decode bf16 h64 b8 s1536：最优 (2048, 1) 3.157 ms，第二名 (512, 8) 3.809 ms，margin 20.63%。

这种 cell 的耗时不是 block 的平滑函数，用“seq 和 batch 算一个数”很难同时照顾两端。demo 在这些点上会错，下面的残差分析会列出来。

### 1.3 结论：可以不用表，但要接受“低 regret”而不是“全对”

demo 的 exact config 只有 20.8%，exact block 只有 47.0%，但 85.8% 的 cell 在 5% 以内，全量平均 regret 2.46%，完整复测子集平均 0.77%。这已经足够替代 tune 表作为默认值；如果生产流量里确实有那几个悬崖 shape，再考虑加几条例外规则。

## 2. 现有启发式函数拆解

### 2.1 它输出什么

`default_mla_config` 返回 `MlaDecodeConfig(block_kv, items_per_step)`。

- `block_kv`：一个 **item** 一次覆盖多少 KV token。
- `items_per_step`：一个 **step** 同时处理多少个 item。
- 一个 **item** 是“一个 request 的一段连续 KV block”。一个 request 会被切成 `ceil(S ÷ block_kv)` 个 item，S 是这个 request 的 KV 长度。
- 一个 **step** 是 kernel 的一次循环，处理 `items_per_step` 个 item。所以一个 step 大约计算 `block_kv × items_per_step` 个 token 位置。

例子：`(block_kv=4096, items_per_step=2)` 表示一个 item 管 4096 个 KV token，一个 step 同时算 2 个 item，约 8192 个 token 位置。

### 2.2 输入参数

| 参数 | 含义 |
|---|---|
| `batch` (B) | 这次调用有多少条序列 |
| `max_seqlen_k` (S) | 这次调用里最长的 KV 长度（decode/verify 的最终长度） |
| `page_size` (P) | KV cache 的页大小；decode/verify 的 cell 里 P = 1024 |
| `pages_per_seq` | 每条序列最多占多少页，`ceil(S ÷ P)` |
| `query_tokens` (Q) | 每条序列这次调用写入几个 query token；decode 是 1，verify 是 `verify_tokens`（2、4、8） |
| `num_query_rows` (R) | `Q × heads`，一个 item 里 query 的行数 |
| `cache_dtype` | KV cache 的数据类型，bf16（2 字节）或 fp8（1 字节） |
| `return_lse` | 是否返回 log-sum-exp；这次测试都是 False |

`verify_tokens` 是投机解码 verify 阶段每个序列每步的 query token 数（`num_spec_tokens + 1`）。verify 的 cache 初始长度是 `S − verify_tokens`，每个 step 追加 `verify_tokens` 个 token，因果 attention 只看当前及之前的 token。decode 的 `verify_tokens = 1`。

### 2.3 常量表

这些常量大部分在 `mla_decode.py` 第 33 到 68 行；`MAX_VMEM_LIMIT_BYTES` 在 `softmax_attention/config.py`。

| 常量 | v6e 上的值 | 作用 |
|---|---|---|
| `WHOLE_REQUEST_TOKENS` | 2048 | 如果 `bound ≤ 2048`，block 直接等于 bound，让一个请求尽量落在一个 item 里，避免尾巴浪费 |
| `BOUND_DIVISOR` | 4 | 长上下文时 block = bound ÷ 4，把一个请求切成约 4 个 item，给流水线足够多的 item |
| `MIN_BLOCK_KV` | 512 | block 的下限 |
| `MIN_ITEMS` | 64 | 如果 `B × ceil(S ÷ block) < 64`，就把 block 减半，凑出更多 item，让 copy 和 compute 重叠 |
| `MAX_BLOCK_KV` | v6e 1024，v7 2048 | 长上下文时 target block 的上限 |
| `SINGLE_REQUEST_MANY_ROWS` | 64 | batch = 1 且 query rows ≥ 64 时，item 是 row-bound，给更大的 block |
| `SINGLE_REQUEST_LARGE_BLOCK_BOUND` | 4096 | 上面这条规则只在 `bound ≤ 4096` 时生效 |
| `SINGLE_REQUEST_MAX_BLOCK_KV` | 8192 | 上面这条规则的 block 上限 |
| `TARGET_STEP_TOKENS` | 16384 | 一个 step 的目标 token 数；items 会被 `16384 ÷ block` 截断 |
| `MAX_ITEMS_PER_STEP` | 8 | 一个 step 最多 8 个 item |
| `NUM_STAGES` | 2 | KV ring buffer 的 stage 数（双缓冲） |
| `NUM_GROUPS` | 2 | 一个 step 分成几个 group，让一组做 PV 时另一组做 softmax |
| `VMEM_BUDGET_FRACTION` | 0.75 | 内核 scratch 最多用 75% 的 VMEM |
| `MAX_VMEM_LIMIT_BYTES` | 100 MiB | v6e 每个核的 VMEM 预算；0.75 × 100 = 75 MiB |

另外几个和 block 计算相关的量：

- `LATENT_DIM = 512`、`ROPE_DIM = 64`、`CACHE_ROW_DIM = 640`：MLA 的 latent 是 512 维，rope 是 64 维，一行 cache 是 640 维。
- `NUM_LANES = 128`：TPU 的 lane 数。block 必须是 128 的整数倍（一个 lane tile 是 128 个 token）。
- `page_size = 1024`：decode/verify 的页大小，所以 block 实际上也是 1024 的整数倍。

### 2.4 决策流程

`default_mla_config`（第 230 到 337 行）的顺序是：

1. **先查 tune 表**：`get_tuned_mla_config(cache_dtype, num_query_rows, batch, max_seqlen_k)`。如果表里有这个 key，就用表里的 `(target_block_kv, max_items_per_step)`，跳过下面的默认规则。
2. 判断 chip：v6e 还是 v7；不在表里就报 `NotImplementedError`。
3. `bound = max(512, next_pow2(S))`。`next_pow2(S)` 是不小于 S 的最小 2 的幂。
4. **单请求多行规则**：如果 `B == 1`、`Q == 1`、`R ≥ 64`、`bound ≤ 4096`，则 `target = min(bound, 8192)`。
5. **否则**：
   - 如果 `bound ≤ 2048`，`target = bound`；
   - 否则 `target = min(max(512, bound ÷ 4), MAX_BLOCK_KV[chip])`；
   - 然后 `while target > 512 and B × ceil(S ÷ target) < 64: target ÷= 2`。
6. `max_items_per_step = 8`。
7. **页对齐**：`block = mla_block_kv(P, ceil(S ÷ P), target)`。这一步会把 target 变成页和 lane tile 的整数倍，可能比 S 大（因为页会补齐），也可能比 target 小（因为不能超过请求的页数）。
8. **定 items**：`items = max(1, min(8, 16384 ÷ block, B × ceil(S ÷ block)))`。
9. **VMEM 守卫**：如果 `mla_vmem_bytes(config) > 75 MiB`，先把 items 减半；如果还超，就把 target 减半、重新做页对齐，再重复。直到放得下；如果 target 减半后小于 128（一个 lane），或者页对齐后的 block 不再变小，就报错。

### 2.5 `mla_block_kv` 的页对齐

`mla_block_kv(page_size, pages_per_seq, target_block_kv)` 的规则（第 127 到 145 行）：

- 如果 `P ≥ target` 且 `P % target == 0`，直接返回 target。
- 否则令 `unit = lcm(P, 128)`（P = 1024 时 unit = 1024），`max_units = ceil(pages_per_seq × P ÷ unit)`，返回 `unit × max(1, min(target ÷ unit, max_units))`。

对 P = 1024 来说，block 一定是 1024 的整数倍，并且不超过这条请求补齐后的总长度。例如 S = 2560：

- `pages_per_seq = ceil(2560 ÷ 1024) = 3`
- target = 4096 时，`min(4096 ÷ 1024, 3) = 3`，所以 block = 3072。
- 3072 比 S = 2560 大，这是页补齐的结果。

### 2.6 VMEM 守卫在算什么

`mla_vmem_bytes`（第 167 到 227 行）把内核的 scratch 加起来，主要项是：

- KV ring：`2 × items × block × 640 × D`（D 是 cache 的字节数，bf16 = 2，fp8 = 1）
- Query ring：`2 × items × padded_rows × (640 + 128) × 2`
- 新 token、append tile、输出 ring、softmax 状态、可选的 LSE
- Score 临时量：`live_items × padded_rows × block × (4 + D)`

`padded_rows` 是把 `R` 向上取整到 8 的倍数（`BF16_TILE_ROWS`）。block 越大、items 越多、rows 越多，VMEM 越大。所以 demo 改了 block 之后，仍然要保留这个守卫，不能直接返回。

### 2.7 为什么现有函数 91% 输出 (512, 8) 或 (1024, 8)

在有 heuristic 记录的 582 个 cell 里：

| 输出 | 数量 |
|---|---|
| (1024, 8) | 339 |
| (512, 8) | 189 |
| (1024, 4) | 25 |
| (2048, 8) | 20 |
| (512, 4) | 8 |
| (512, 6) | 1 |

(512, 8) 和 (1024, 8) 合计 528 / 582 = 90.7%。

原因很直接：

- S > 2048 时，`bound ÷ 4` 再被 `MAX_BLOCK_KV = 1024` 截断，target 就是 1024。
- items = min(8, 16384 ÷ 1024 = 16, B × ceil(S ÷ 1024))。大多数 cell 的 batch 足够大，所以 items = 8。
- 小 batch 时 `MIN_ITEMS = 64` 的缩小规则把 target 减到 512，于是变成 (512, 8)。

而数据里最优 block 的众数是 4096，最优 step 的 token 数也集中在 4096 到 8192。**旧常量和现在的测量结果不一致**，这就是要重新拟合的原因。

## 3. 649 个测试点告诉我们什么

### 3.1 数据构成

- 432 个 main run + 146 个 refinement + 71 个 coverage = 649 个 cell。
- decode 459 个，verify 190 个。
- bf16 296 个，fp8 353 个。
- heads：16、32、64、128。
- batch：1 到 512。
- seq：512 到 262144。
- 每个 cell 的候选网格：block ∈ {512, 1024, 2048, 4096, 8192, 16384}，items ∈ {1, 2, 4, 8}，共 24 个 target 组合；拟合后重复的 actual 只测一次。
- 每个 cell 的 screening 前 3 名 + 现有基线，会用完整协议再复测一次。

### 3.2 最优 block 的分布

649 个 cell 的 `best_actual`：

| block | 数量 |
|---|---|
| 512 | 90 |
| 1024 | 113 |
| 2048 | 124 |
| 3072 | 10 |
| 4096 | 188 |
| 5120 | 2 |
| 6144 | 1 |
| 7168 | 1 |
| 8192 | 101 |
| 9216 | 1 |
| 10240 | 2 |
| 16384 | 16 |

4096 是众数。3072、5120、6144、7168、9216、10240 都是页对齐的副产物，不是独立的候选值。

按 scenario 分：

| scenario | 512 | 1024 | 2048 | 4096 | 8192 | 16384 | 其他 |
|---|---|---|---|---|---|---|---|
| decode（459） | 90 | 102 | 80 | 98 | 61 | 11 | 17 |
| verify（190） | 0 | 11 | 44 | 90 | 40 | 5 | 0 |

verify 的最优 block 集中在 2048 到 8192，512 很少见。这是因为 verify 的 `Q` 大于 1，query rows 更大，item 本身更“重”，不需要太小的 block。

### 3.3 最优 block × items 的分布

| block × items | 数量 |
|---|---|
| 4096 | 255 |
| 8192 | 200 |
| 16384 | 120 |
| 2048 | 36 |
| 1024 | 18 |
| 6144 | 5 |
| 512 | 3 |
| 10240 | 3 |
| 15360 | 3 |
| 其他 | 6 |

4096、8192、16384 三者合计 575 / 649 = 88.6%。也就是说，**一个 step 计算 4k 到 16k 个 token 位置是主流，8k 左右最密集**。

最优 items 的分布：

| items | 数量 |
|---|---|
| 1 | 171 |
| 2 | 215 |
| 4 | 142 |
| 8 | 118 |
| 5 | 3 |

items 只出现 1、2、4、8（5 只出现在 block 被页对齐成 3072 的 3 个 cell 里）。这支持“items 取 2 的幂”这条新规则。

### 3.4 三条可用的规律

**规律一：block 主要由 seq 决定，约等于 `min(next_pow2(S), 4096)`。**

- S ≤ 1024：block 512 或 1024；
- S 在 2048 到 4096：block 2048 或 4096；
- S ≥ 8192：最优 block 常见 4096 或 8192（S 在 8192 到 16384 时两者数量接近，S 更大时 8192 最多）。demo 为了中等 seq 的稳定性把 cap 定在 4096；长上下文用 4096/2 的 regret 通常在几个百分点以内，个别“整请求一个 item”的边界会到 20% 左右。

**规律二：items 由 step token 目标和 batch 共同决定。**

- block 4096 → 2 个 items（step 约 8192）；
- block 2048 → 4 个 items；
- block 1024 → 8 个 items；
- block 512 → 8 个 items（受 `MAX_ITEMS_PER_STEP = 8` 限制）。
- 小 batch 时，`B × ceil(S ÷ block)` 可能小于目标 items，此时 items 取 batch 能凑出的最大值，再向下取 2 的幂。

**规律三：dtype 和 heads 是二阶因素。**

- fp8 的 landscape 比 bf16 更“锯齿”，但同一个规则在两种 dtype 上的 full-protocol 平均 regret 分别是 0.98% 和 0.48%，都不需要 dtype 分支。
- heads（query rows）对最优 block 没有单调影响；它主要影响 VMEM，所以只在 VMEM 守卫里用到，不用来选 block。

### 3.5 现有线上配置 vs 数据

纯 `default_mla_config`（不带表项）在有记录的 582 个 cell 里，91% 选 (512, 8) 或 (1024, 8)，而数据里最优最多的是 4096 系。用完整复测时间比较，现有线上配置（没有表项时是纯启发式，有表项时是旧表项）在 593 个 cell 上的平均 regret 是 5.47%，中位数 1.52%，p90 16.23%，最大 37.08%。这就是 demo 要解决的主要问题。

## 4. demo 函数

### 4.1 参数表（旧值 → demo 值）

| 参数 | 旧值（v6e） | demo 值 | 依据 |
|---|---|---|---|
| `MAX_BLOCK_KV` / `BLOCK_CAP` | 1024 | **4096** | 4096 是最常见的最优 block（188 / 649）；cap 8192 会让中等 seq 选得太大，平均 regret 反而变差 |
| `BOUND_DIVISOR` | 4 | **1** | 数据里长上下文的 block 基本等于 `next_pow2(S)` 截到 4096，不是 `S ÷ 4` |
| `TARGET_STEP_TOKENS` | 16384 | **8192** | 最优 block × items 的众数是 4096 和 8192；8192 让 4096 → 2、2048 → 4、1024 → 8 |
| `MIN_ITEMS` | 64 | **1（取消缩小）** | 在 649 个点上，缩小规则整体没有降低 regret；小 batch 的悬崖不是靠“凑够 64 个 item”能解决的 |
| `MAX_ITEMS_PER_STEP` | 8 | **8** | 不变；实测最优 items 只出现 1、2、4、8 |
| `MIN_BLOCK_KV` | 512 | **512** | 不变 |
| `VMEM_BUDGET_FRACTION` | 0.75 | **0.75** | 不变；大 block 仍然要经过 VMEM 守卫 |
| items 取整 | 无 | **取 2 的幂** | 实测最优 items 没有 3、6、7；小 batch 时 `B × ceil(S ÷ block)` 可能是 3、6、7，向下取 2 的幂可以避免凑不齐一个 step |

### 4.2 公式

记：

- B = batch
- S = max_seqlen_k（这次调用里最长的 KV 长度）
- P = page_size（decode/verify 是 1024）
- Q = query_tokens（decode 是 1，verify 是 verify_tokens）
- R = Q × heads（VMEM 守卫用）
- D = cache dtype 的字节数（bf16 = 2，fp8 = 1）

demo 的步骤：

1. `bound = max(512, 2^⌈log₂ S⌉)`，即不小于 S 的最小 2 的幂，且不小于 512。
2. `target = min(bound, 4096)`。
3. `block = mla_block_kv(P, ⌈S ÷ P⌉, target)`，和现有函数一样做页/lane tile 对齐。
4. `A = B × ⌈S ÷ block⌉`，这个 cell 最多能凑出多少个 item。
5. `I₀ = min(8, ⌊8192 ÷ block⌋, A)`。
6. `I = 2^⌊log₂ I₀⌋`，即不大于 I₀ 的最大 2 的幂；如果 I₀ < 1，取 I = 1。
7. **VMEM 守卫**：如果 `mla_vmem_bytes(block, I) > 75 MiB`：
   - 先 `I ÷= 2`，直到 I = 1；
   - 如果还超，`target ÷= 2`，回到第 3 步重新做页对齐和 items 计算；
   - 如果 target 减半后小于 128（一个 lane），或者页对齐后的 block 不再变小，报错（和现有函数一样）。
8. 返回 `(block, I)`。

写成伪代码：

```text
demo_mla_config(B, S, P, Q, R, cache_dtype):
    bound  = max(512, next_pow2(S))
    target = min(bound, 4096)
    while True:
        block = mla_block_kv(P, ceil(S / P), target)
        A     = B * ceil(S / block)
        I     = largest_pow2_le(min(8, floor(8192 / block), A))
        while I > 1 and mla_vmem_bytes(block, I, R, Q, cache_dtype, False) > 75 MiB:
            I = I // 2
        if mla_vmem_bytes(block, I, R, Q, cache_dtype, False) <= 75 MiB:
            return (block, I)
        if target // 2 < 128:
            raise ValueError(...)
        target = target // 2
```

和现有骨架相比，只改了第 1、2、5、6 步和第 4.1 节的常量；第 3、7 步（页对齐、VMEM 守卫）原样保留。

### 4.3 五个例子

| 例子 | B | S | Q | bound | target | block | A | I₀ | demo 输出 | 数据最优 | 说明 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| decode bf16 h32 | 64 | 4096 | 1 | 4096 | 4096 | 4096 | 64 | 2 | (4096, 2) | (4096, 2) | 精确命中；第二名 (1024, 8) 只慢 0.03% |
| verify bf16 h64 | 64 | 4096 | 8 | 4096 | 4096 | 4096 | 64 | 2 | (4096, 2) | (4096, 1) | demo 是第二名，只慢 0.68% |
| decode bf16 h64 | 8 | 1536 | 1 | 2048 | 2048 | 2048 | 8 | 4 | (2048, 4) | (2048, 1) | 悬崖：最优比第二名快 20.63%，demo 不在这里 |
| decode fp8 h16 | 8 | 2560 | 1 | 4096 | 4096 | 3072 | 8 | 2 | (3072, 2) | (512, 8) | 悬崖：最优比第二名快 23.93%，demo 不在这里 |
| decode fp8 h128 | 4 | 8704 | 1 | 16384 | 4096 | 4096 | 12 | 2 | (4096, 2) | (9216, 1) | “整请求一个 item”的情况；demo 会多出一个 512 的尾巴 |

### 4.4 为什么是这几个数

- **为什么 cap 是 4096，不是 8192？**
  - 最优 block 的众数是 4096（188 / 649），8192 只有 101 个。
  - 如果 cap 设成 8192，中等 seq（4096 到 8192）会倾向选 8192，但这时 8192 往往只覆盖一个请求、items 变少，平均 regret 反而从 2.46% 升到 3.00%。
- **为什么 step 是 8192，不是 16384？**
  - 最优 block × items 的 88.6% 落在 4096 到 16384，最密集的是 4096 和 8192。
  - step = 8192 时，4096 的 block 配 2 个 items、2048 配 4 个、1024 配 8 个，和实测的最优 items 分布最吻合。
  - step = 16384 会让 4096 配 4 个 items，实测中 4096/4 只有 33 个 cell 最优，远少于 4096/2 的 72 个和 4096/1 的 83 个。
- **为什么 items 要取 2 的幂？**
  - 小 batch 时 `B × ceil(S ÷ block)` 可能是 3、6、7；如果不取整，kernel 会按 3、6、7 个 items 跑，最后一个 step 可能只有部分 item 有活，padding 浪费。
  - 实测最优 items 只有 1、2、4、8；向下取 2 的幂可以避免这些浪费，并且和候选网格一致。
- **为什么取消 MIN_ITEMS = 64 的缩小？**
  - 这条规则的目的是“凑够足够的 item 让 copy 和 compute 重叠”。但数据表明，小 batch 的最优 block 既可能是 512（很多 item），也可能是 2048/4096（很少 item），没有一个固定的 item 数阈值能同时照顾两端。
  - 在 649 个点上试了 4、8、16、32、64 几档阈值，取消缩小（阈值 1）的平均 regret 最低。
- **为什么单请求多行规则可以删掉？**
  - 旧规则在 `B = 1`、`Q = 1`、`R ≥ 64`、`bound ≤ 4096` 时给 `target = min(bound, 8192) = bound`。
  - demo 的 `target = min(bound, 4096) = bound`，在 `bound ≤ 4096` 时完全一样。所以这条规则被新的 cap 吸收了，不需要单独保留。

## 5. 离线拟合评估

这一节的所有数字都来自已有的 649 个 cell 的候选记录，**没有跑新的 kernel 测试**。只是把 demo 的预测配置和已有记录里的耗时对上，重新算 regret。

每个 cell 的分档明细（可用 / 基本可用 / 未确认 / 效果差，含最优、次优和 demo 配置）在配套文档 [`mla_block_heuristic_fit_table.md`](./mla_block_heuristic_fit_table.md) 中列出。按 scenario × dtype 拆分支后的最新分档见 [`mla_block_heuristic_fit_table3.md`](./mla_block_heuristic_fit_table3.md)。

### 5.1 两个口径

**regret 的定义**：对某个 cell，设 demo 选中的配置耗时是 t，这个 cell 里所有已测候选里最快的是 t_best，则

regret = t ÷ t_best − 1

例如 regret = 5% 表示比这个 cell 的最快候选慢 5%。

有两个时间口径：

- **screening（短测）**：每个候选在调优时先短测 3 轮 × 0.25 秒，取中位数。覆盖全部候选，但只有 3 个样本，噪声较大，尤其是在小 batch 的 cell 上。
- **full-protocol（完整复测）**：每个 cell 的 screening 前 3 名 + 现有基线，再用完整协议复测（预热 3 次、5 轮、取中位数）。更准，但只有 3 到 4 个候选有数据。

所以：

- 全部 649 个 cell 用 screening 口径；它覆盖全，但偏保守/有噪声。
- 只有 357 个 cell 的 demo 配置正好是复测候选，用 full-protocol 口径；它更可信，但样本是被 screening 选中的，略有选择偏差。

两个口径一起看，真实性能大概率在两者之间。

### 5.2 总体结果

| 子集 | n | exact | ≤2% | ≤5% | 平均 | 中位数 | p90 | 最大 | full-protocol 子集（n） |
|---|---|---|---|---|---|---|---|---|---|
| 全部 | 649 | 20.8% | 71.5% | 85.8% | 2.46% | 0.60% | 7.00% | 38.22% | 0.77% 平均，2.43% p90，15.61% 最大（357） |
| decode | 459 | 19.6% | 69.7% | 81.3% | 2.97% | 0.62% | 9.62% | 38.22% | 0.75% 平均，2.43% p90，15.61% 最大（226） |
| verify | 190 | 23.7% | 75.8% | 96.8% | 1.22% | 0.54% | 3.05% | 12.85% | 0.81% 平均，2.36% p90，7.55% 最大（131） |
| bf16 | 296 | 17.6% | 80.4% | 88.9% | 1.79% | 0.41% | 5.33% | 23.69% | 0.48% 平均，1.10% p90，7.55% 最大（146） |
| fp8 | 353 | 23.5% | 64.0% | 83.3% | 3.02% | 0.96% | 9.59% | 38.22% | 0.98% 平均，2.68% p90，15.61% 最大（211） |

解读：

- verify 很稳：96.8% 的 cell 在 5% 以内，full-protocol 平均 0.81%。
- bf16 比 fp8 稳：bf16 平均 1.79%，fp8 平均 3.02%；fp8 的锯齿更多。
- decode 的平均 2.97% 主要被小 batch 拉高；大 batch 的 decode 很好（见下表）。

### 5.3 按 batch 分组

| batch | n | exact | 平均 | 中位数 | p90 | 最大 | ≤5% |
|---|---|---|---|---|---|---|---|
| 1 | 6 | 50.0% | 6.45% | 2.42% | 10.21% | 23.58% | 66.7% |
| 2 到 4 | 90 | 16.7% | 7.23% | 5.46% | 19.20% | 33.72% | 48.9% |
| 5 到 8 | 143 | 13.3% | 3.97% | 2.12% | 9.99% | 38.22% | 76.2% |
| 9 到 16 | 10 | 30.0% | 4.85% | 0.40% | 12.41% | 22.39% | 60.0% |
| 17 到 32 | 118 | 24.6% | 1.05% | 0.47% | 2.76% | 9.38% | 95.8% |
| 33 到 64 | 130 | 23.8% | 0.60% | 0.30% | 1.42% | 4.39% | 100.0% |
| 65 到 128 | 120 | 21.7% | 0.56% | 0.29% | 1.32% | 3.04% | 100.0% |
| 129 到 512 | 32 | 28.1% | 0.62% | 0.26% | 1.05% | 5.00% | 96.9% |

**batch ≥ 17 的 cell，平均 regret 都在 1.05% 以内，p90 在 2.76% 以内，绝大多数在 5% 以内。** 小 batch（≤ 8）是主要误差来源。

### 5.4 按 seq 分组

| seq | n | exact | 平均 | 中位数 | p90 | 最大 | ≤5% |
|---|---|---|---|---|---|---|---|
| 512 到 1024 | 92 | 38.0% | 2.78% | 0.35% | 7.65% | 22.65% | 81.5% |
| 1025 到 2048 | 52 | 19.2% | 3.11% | 0.51% | 6.59% | 23.69% | 73.1% |
| 2049 到 4096 | 212 | 25.0% | 3.43% | 0.62% | 10.21% | 38.22% | 78.8% |
| 4097 到 8192 | 47 | 14.9% | 3.07% | 0.48% | 11.69% | 23.10% | 80.9% |
| 8193 到 16384 | 146 | 17.8% | 1.50% | 0.63% | 3.39% | 21.32% | 95.2% |
| 16385 到 65536 | 64 | 4.7% | 0.85% | 0.64% | 2.01% | 4.39% | 100.0% |
| 65537 到 262144 | 36 | 2.8% | 0.90% | 0.69% | 2.53% | 2.86% | 100.0% |

长上下文（≥ 16384）非常稳：100% 在 5% 以内。误差集中在 2049 到 8192 这一段，也就是小 batch 的悬崖区。

### 5.5 和现有线上配置对比（同一批 cell）

在 334 个「demo 和现有线上配置都有完整复测时间」的 cell 上：

| 配置 | 平均 regret | 中位数 | p90 | 最大 |
|---|---|---|---|---|
| 现有线上配置（旧表项/旧启发式） | 6.36% | 3.07% | 16.69% | 37.08% |
| demo | 0.78% | 0.15% | 2.36% | 15.61% |

demo 和现有线上配置的耗时比值（demo ÷ 现有）：

- 平均 0.952（平均快 4.8%）
- 中位数 0.979（中位数快 2.1%）
- 53% 的 cell 快 1% 以上
- 45% 的 cell 在 ±1% 以内（基本打平）
- 2% 的 cell 慢 1% 以上，最慢慢 7%

demo 不是每个 cell 都更快，但整体上明显更好，而且没有出现“平均好、尾部爆炸”的情况（最慢只慢 7%）。

### 5.6 候选排名

除了时间 regret，还可以看 demo 选中的配置在 cell 内的 screening 排名（每个 cell 大约 15 个候选）：

| 指标 | demo | 现有线上配置 |
|---|---|---|
| 中位数排名 | 第 3 名 | 第 7 名 |
| 第 1 名比例 | 19.9% | 8.6% |
| 前 2 名比例 | 38.8% | 16.5% |
| 前 3 名比例 | 53.8% | 22.7% |
| 前 5 名比例 | 75.5% | 40.5% |

这个指标不受时间噪声的绝对值影响，结论和 regret 一致：demo 明显更接近每个 cell 的最优候选。

### 5.7 exact match 为什么低，但性能不差

demo 的 exact config 只有 20.8%，exact block 47.0%，exact items 39.8%，block 在最优值的 2 倍以内有 80.1%。看起来“对上的比例”不高，但这不代表性能差，因为：

- 很多 cell 的前几名只差 0.1% 到 1%（前面举的 0.03%、0.68% 的例子）。
- block 换一档（例如 4096/2 和 2048/4）常常是同一档 step token 数，性能几乎一样。
- 真正重要的是 regret，不是 exact match。

verify bf16 h64 b64 s4096 t8 就是一个好例子：数据最优是 (4096, 1)，demo 选 (4096, 2)，但 (4096, 2) 是第二名，只慢 0.68%。

## 6. 残差：哪些 cell 会错

### 6.1 误差分布

按 screening 口径：

- regret > 2%：185 / 649
- > 5%：92 / 649
- > 10%：44 / 649
- > 15%：24 / 649
- > 20%：13 / 649
- > 30%：2 / 649

> 10% 的 44 个 cell 里，几乎全部是 decode、batch ≤ 8（少数是 batch 11 到 14 的补充点），并且 fp8 占多数。verify 只有 1 个 cell 超过 10%（screening 口径），full-protocol 最大 7.55%。

### 6.2 几个典型残差

| cell | demo | 数据最优 | 第一名 margin | screening regret | 性质 |
|---|---|---|---|---|---|
| fp8 h16 b8 s2560 | (3072, 2) | (512, 8) | 23.93% | 38.22% | 真悬崖：小 block 多 item 明显更好 |
| bf16 h64 b8 s1536 | (2048, 4) | (2048, 1) | 20.63% | 23.69% | 真悬崖：同一个 block，items 1 比 4 快 20% |
| fp8 h32 b11 s2048 | (2048, 4) | (512, 4) | 8.79% | 22.39% | 真悬崖：小 block 更好 |
| fp8 h128 b4 s8704 | (4096, 2) | (9216, 1) | 6.98% | 21.32% | “整请求一个 item”更省，demo 多一个 512 尾巴 |
| fp8 h16 b4 s5632 | (4096, 2) | (2048, 2) | 18.20% | 11.69% | 真悬崖：block 2048 比 4096 好 |
| bf16 h128 b1 s4096 | (4096, 1) | (512, 2) | 0.28% | 23.58% | 不是真错：前几名只差 0.28%，screening 噪声放大了 regret |

前 5 个是真正需要关注的；最后一个说明 screening 口径的尾部会被噪声夸大。用 full-protocol 口径时，demo 的最大 regret 是 15.61%（fp8 h64 b8 s512：demo 512/8 vs 最优 1024/4），没有 30% 以上的真实误差。

### 6.3 这些残差能靠公式修吗

试过几种简单的修正，都没有整体收益：

- 小 batch 加 `MIN_ITEMS` 缩小（4、8、16、32、64）：整体平均 regret 反而变差，因为这些悬崖的方向不一致。例如 s2560 喜欢 512/8，s2816 喜欢 2048/2，s4096 喜欢 4096/2。
- verify 按 rows 把 items 压到 1：verify 平均反而从 1.22% 升到 1.6% 到 2.4%。
- 长上下文 block cap 改成 8192：长上下文略好，但中等 seq 变差，整体从 2.46% 升到 3.00%。
- “整请求一个 item”规则（8192 < S ≤ 16384 且尾巴 ≤ 2048 时用 target 16384）：只修了少数 cell，整体平均从 2.46% 升到 2.74%。

所以 demo 没有再堆规则。这些残差要么接受，要么用少量例外表，要么补测确认。

## 7. 如果要落地

### 7.1 代码改动点

只需要改 `default_mla_config` 的“没有 tune 表条目”分支：

1. `MAX_BLOCK_KV["TPU v6e"]`：1024 → 4096。
2. `BOUND_DIVISOR`：4 → 1（或者直接写 `target = min(bound, 4096)`）。
3. `TARGET_STEP_TOKENS`：16384 → 8192。
4. `MIN_ITEMS`：64 → 1（等价于删除缩小循环）。
5. 在 items 计算之后，加一步“向下取 2 的幂”。
6. 删除 `SINGLE_REQUEST_MANY_ROWS` 分支（它被新的 cap 吸收了）。
7. 保留 `mla_block_kv`、`mla_vmem_bytes`、VMEM 守卫、tune 表查询路径不变。

### 7.2 验证方法

- 先用这份 649 点的离线评估作为回归基线：demo 的平均 regret、p90、最大 regret 不应比这里更差。
- 用 22 个剩余 ≥5% 的 sharp interval 做验证集：检查 demo 在区间两端的预测是否在“正确的一侧”，而不是只看单点。
- 上线前做 shadow 对比：同一批真实请求，同时跑 demo 和现有线上配置，比较 kernel ms；不要直接替换。

### 7.3 如果允许极少量补测

只补测两类点，不要再来一轮全局 bisection：

1. 小 batch（≤ 8）decode 里 demo 的 screening regret > 20% 的 cell，确认是噪声还是真悬崖。
2. “整请求一个 item”的边界：S 在 8192 到 16384 之间、尾巴 512 到 2048 的 shape。

如果确认是真悬崖，再加 5 到 10 条例外规则或一张很小的例外表；这比重新拟合一个更复杂的全局公式更划算。

## 8. 术语表

| 术语 | 解释 |
|---|---|
| 启发式函数（heuristic） | 不查表，用公式和规则直接算出一个配置的函数 |
| 拟合（fit） | 用已有数据调整公式里的常数，让预测尽量接近测量结果；这里没有跑新测试，只是离线比较 |
| cell | 一个具体的测试形状：scenario、heads、batch、seq、dtype、verify_tokens、decode_tokens |
| block_kv | 一个 item 一次覆盖多少 KV token |
| items_per_step | 一个 step 同时处理多少个 item |
| item | 一个（request，KV block）工作单元 |
| step | kernel 的一次循环，处理 items_per_step 个 item |
| KV cache | 推理时缓存历史 token 的 Key/Value，避免重复计算 |
| decode | 自回归生成，每次 1 个 query token |
| verify | 投机解码的验证阶段，每次 Q = verify_tokens 个 query token，做因果 attention |
| query rows | Q × heads，一个 item 里 query 的行数 |
| page | KV cache 的页；decode/verify 的 page_size = 1024 token |
| lane / lane tile | TPU 有 128 个 lane；一个 lane tile 是 128 个 token，block 必须是它的整数倍 |
| VMEM | TPU 核上的片上内存；v6e 约 100 MiB，内核的临时缓冲区放在这里 |
| next_pow2(S) | 不小于 S 的最小 2 的幂 |
| regret | 预测配置耗时 ÷ 该 cell 最快候选耗时 − 1 |
| margin | 同一 cell 内，第二名耗时 ÷ 第一名耗时 − 1 |
| screening | 调优时的短测：3 轮 × 0.25 秒，取中位数 |
| full-protocol | 完整复测：预热 3 次、5 轮、取中位数 |
| p90 | 90% 分位：把 649 个 regret 排序，第 90% 位置的值；表示“90% 的 cell 不会比这个更差” |
| 悬崖（cliff） | 相邻两个 block/items 的耗时突然差很多，例如 20% 以上 |
| tuned table | `tuned_configs.json` 里的 `mla\|...` 条目，按 dtype、query rows、batch、seq 查表 |

## 附录 A：统计口径

- 数据文件：`benchmarks/softmax_attention/results/mla_tune/summary.jsonl`（432）、`mla_tune_refine/summary.jsonl`（146）、`mla_tune_remaining/summary.jsonl`（71）。
- 每个 cell 的候选记录：同目录下的 `<cell>.jsonl`，`kind = candidate` 是短测，`kind = finalist` 是完整复测。
- 候选网格：block ∈ {512, 1024, 2048, 4096, 8192, 16384}，items ∈ {1, 2, 4, 8}。
- demo 的预测配置通过 `mla_block_kv` 和 VMEM 守卫得到；离线评估时，用每个 cell 的 target → actual 映射找到实际配置，再查它的耗时。
- 短测口径的 `t_best` 是该 cell 所有短测候选里的最小值；完整复测口径的 `t_best` 是该 cell 的 `best_ms`。
- 所有 cell 的 status 都是 ok，没有 run_error；本报告没有修改 tune 表，也没有跑新的性能测试。

## 附录 B：demo 一句话版本

如果只记一句话：

> block 取 `min(next_pow2(S), 4096)`，再按页对齐；items 取 `min(8, 8192 ÷ block, batch × ceil(S ÷ block))` 向下到 2 的幂；最后用原来的 VMEM 守卫收尾。

对 v6e 的 649 个点，这句话的平均 regret 是 2.46%，p90 7.00%；在最可靠的 357 个完整复测点上，平均 0.77%，p90 2.43%。大 batch 和 verify 已经足够好，小 batch 的少数悬崖是下一步要决定“接受、加例外、还是补测”的地方。
