# MLA decode 的块代价模型

`default_mla_config` 用这个模型选 `(block_kv, items_per_step)`。TPU v6e + bfloat16 cache 已启用，**优先于旧的 `tuned_configs.json` 调优表**；没有标定条目的 chip / dtype 先查旧调优表，再退回 `max_seqlen` 规则（并警告一次）。

## 公式

对一个候选 (B, ips)，在调用页宽 page 上：

C(B, ips, page) = Σᵢ wᵢ · [ e(B, page) · gᵢ(B, page) · stepsᵢ · ips · B + r_item · stepsᵢ · ips + r_step(page) · stepsᵢ ]

- i 是长度先验里的一支（共 12 支：3 个尺度 × 4 种形状），wᵢ 是它的权重；
- stepsᵢ = max(1, ceil(batch · Eᵢ[ceil(w / B)] / ips))，w 是一个请求的长度；
- e(B, page) 是满块时每列的效率，r_item / r_step(page) 是每个 item / 每个 step 的固定开销（标定取 r_item = 0）；
- gᵢ(B, page) 是一个 item 实际做了满块的百分之几。

g 的定义（x = item 的有效长度 / B）：

g(x, f) = f + (1 − f) · clamp(2x − 1, 0, 1)

f = f(B, page) 是地板比例：item 即使只有很少的有效 token，也要付 f 个满块的代价；剩下的部分在 block 的后半段线性涨上来。f = 1 时就是旧模型。

## 合法候选集

候选 B 不是 2 的幂，而是 kernel 真正接受的整数格：

- B 是 128（NUM_LANES）的倍数；
- B % page == 0 或 page % B == 0（块整页对齐，或块是页的整分片）。

最小的合法块 B_min = 128（page 是 128 的倍数时），否则 B_min = lcm(page, 128)；B_min > 2048 时该页没有候选。例如 page 512 的格是 128/256/512/1024/1536/2048，page 1024 是 128/256/512/1024/2048，page 128 是 128..2048 的全部 16 个 128 倍数。

ips 先枚举 1..max_items（max_items = min(8, 16384 // B)），再用实测剪枝：items 1/2 比 items 8 慢 56%/15%，items 3–7 只慢 2.6%/1.1%/0.5%/0.6%/0.2%，所以在 max_items ≥ 4 时跳过 1/2、保留 3..max_items；max_items < 4 时保留 1..max_items。放开 1/2 并不会改变 page 1024 的选择：在 2048/2 成为实测最优的 uniform/spiky 两组，items 2 也只比 items 8 快 0.4–0.6%；部署先验下 2048/2 的模型代价比选中的 512/8 高约 21%，比 256/8 高约 18%，问题在 block 选择而不是 items。只把 oracle 限制在 items ≥ 3，page 1024 的平均遗憾从 0.42% 降到 0.34%（0.08 个百分点），page 128 只降 0.01 个百分点。所以保留剪枝。VMEM/SMEM 预算过滤和平局规则（取更长 block）不变。未实测的 block 不再用 1.0 占位价参与比较：常数表非空时它不是候选；空表（dormant）仍保留全部格。

## 长度先验

先验的 12 支是 3 个尺度 × 4 种形状，尺度权重 0.40/0.45/0.15。v4 把三个尺度从 256/1024/4096 整体上移一倍到 512/2048/8192（形状和权重不变）。依据是两组与本次全格验证集无关的 TPU 实测（用 v4 常数重测，`results/mla_prior_v4`）：

- 5 个从旧先验抽样的 batch 64（长度均值 988–1613，页 128）：长先验选 384/8，平均落后实测最优 1.50%；旧先验选 384/7，平均落后 1.73%。实测最优都是 512/8。draw 2 上长先验反而更差（2.77% vs 2.08%），所以不是全面占优。
- 4 个 64k/128k、batch 32/64 的负载：长先验选 768/7 或 640/7，落后 0.13–0.37%；旧先验选 384/7，落后 0.79–1.06%。
- 9 组合计：长先验 mean 0.93% / max 2.77%，旧先验 mean 1.36% / max 2.08%。

在全格验证集上长先验全面不差：page 128/512/1024 的 mean regret 是 0.92%/0.46%/0.42%（旧先验 1.51%/0.46%/2.64%）。风险：8192 分支不是从真实线上流量拟合出来的，而是按“先验自采样 + 长上下文”两组独立测量整体上移；验证集恰好是平均长度 8192 的负载，验证上的提升会偏向长先验。若拿到真实流量分布，应重新评估尺度权重，而不是继续按验证集调参；draws 上的最坏遗憾也说明长先验并非处处更优。

## 实测（TPU v6e，page 128 基表）

| block | f(B, 128) | e(B, 128) |
|---|---|---|
| 128 | 0.923 | 1.000 |
| 256 | 0.882 | 0.759 |
| 384 | 0.877 | 0.702 |
| 512 | 0.809 | 0.714 |
| 640 | 0.780 | 0.701 |
| 768 | 0.731 | 0.696 |
| 896 | 0.734 | 0.695 |
| 1024 | 0.704 | 0.705 |
| 1152 | 0.693 | 0.707 |
| 1280 | 0.665 | 0.706 |
| 1408 | 0.670 | 0.709 |
| 1536 | 0.646 | 0.709 |
| 1664 | 0.647 | 0.711 |
| 1792 | 0.629 | 0.712 |
| 1920 | 0.635 | 0.714 |
| 2048 | 0.626 | 0.718 |

page 128 的 step 开销 r_step = 153.6。R² 0.963，MAPE 7.9%。

page 512 和 page 1024（部署页）各自单独标定，作为**逐页精确覆盖**存入 `page_efficiency` / `page_floor` / `page_r_step`，不做插值；没测过的页明确回退到 page 128 基表（`efficiency_with_source` 标记 "fallback"）。实测的页间差异不小：

- r_step：page 128/512/1024 = 153.6 / 249.6 / 257.4；
- e(page)/e(128)：page 512 的 256..2048 为 1.05–1.14，page 1024 的 512..2048 为 1.08–1.13；
- 地板差最大 +0.070（page 1024, block 256）；page 512 的 block 128 地板是 0.975，比基表高 0.052。

因此只覆盖 e、沿用基表 r_step/floor 会让 page 512 的验证遗憾从 0.46% 升到 2.66%，这三张表必须一起按页覆盖。

## 标定

```
PYTHONPATH=. python benchmarks/softmax_attention/calibrate_mla_decode_config.py \
    --out benchmarks/softmax_attention/results/mla_calibration_dense_p1024 \
    --page-size 1024 --floor-scan --validate
```

同样的命令分别跑 `--page-size 512` 和 `--page-size 128`。`--page-size` 默认 1024（部署页）；`--blocks` 默认取该页的完整合法格，长度会向下吸附到 block 的整数倍。`--floor-scan` 对每个 block 测 ≤ B/2 的固定长度和 B 本身，拟合 f(B)。

验证不再只在启发式自己的候选集上算：oracle 会实测该页每一个合法 (B, ips) cell，再拿启发式实际选择的配置去比，报告每组的最优/所选/遗憾以及 mean/max regret。oracle 包含被剪枝的 ips 1/2，所以遗憾里也含这条实测剪枝规则本身的代价。

不重新上 TPU、只用已保存的测量重拟合：

```
PYTHONPATH=. python benchmarks/softmax_attention/refit_mla_floor.py
```

## 当前状态

- `CALIBRATION` 已填入 TPU v6e / bfloat16 的 v4 常数（page 128 基表 + page 512/1024 覆盖），默认路径已切到新启发式。
- 用部署常数（含长先验）复评三个 page 的全格 oracle（每页 4 组、batch 64、平均长度 8192）：
  - page 128：mean 0.92%，max 1.58%，选 384/8 或 768/7（上一版 1.51%/1.63%）；
  - page 512：mean 0.46%，max 0.56%，选 512/8（与上一版相同，无退步）；
  - page 1024：mean 0.42%，max 0.61%，四组都选 512/8（上一版 2.64%/2.92%）。上一版这里四组都选 256/8，原因是默认长度先验比验证负载短；长先验改选 512/8。oracle 里 uniform/spiky 的 2048/2 仍属于被剪枝的 ips=2，但 512/8 实测只比它慢 ≤0.6%，遗憾因此从 2.6% 量级降到 0.4% 量级。
- 标定常数优先于 `tuned_configs.json` 的旧调优表；旧表只在没有标定条目的 dtype/chip 上使用（例如 fp8 cache）。
