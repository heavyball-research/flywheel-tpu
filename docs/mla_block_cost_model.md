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

ips 先枚举 1..max_items（max_items = min(8, 16384 // B)），再用实测剪枝：items 1/2 比 items 8 慢 56%/15%，items 3–7 只慢 2.6%/1.1%/0.5%/0.6%/0.2%，所以在 max_items ≥ 4 时跳过 1/2、保留 3..max_items；max_items < 4 时保留 1..max_items。VMEM/SMEM 预算过滤和平局规则（取更长 block）不变。未实测的 block 不再用 1.0 占位价参与比较：常数表非空时它不是候选；空表（dormant）仍保留全部格。

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
- 用部署常数复评三个 page 的全格 oracle（每页 4 组、batch 64、平均长度 8192）：
  - page 128：mean 1.51%，max 1.63%，选 384/8 或 384/7；
  - page 512：mean 0.46%，max 0.56%，选 512/8；
  - page 1024：mean 2.64%，max 2.92%，四组都选 256/8。uniform/spiky 的实测最优 2048/2 属于被剪枝的 ips=2；longtail/pareto 的最优 1024/4–6 在候选集里，但默认长度先验（均值 256/1024/4096 的三档）比验证负载（均值 8192）更短，仍然选 256/8。
- 标定常数优先于 `tuned_configs.json` 的旧调优表；旧表只在没有标定条目的 dtype/chip 上使用（例如 fp8 cache）。
