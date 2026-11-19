# MLA decode 的块代价模型

`default_mla_config` 用这个模型选 `(block_kv, items_per_step)`。它只在标定路径上使用；`CALIBRATION` 为空时走原来的 `max_seqlen` 规则。

## 公式

对一个候选 (B, ips)：

C(B, ips) = Σᵢ wᵢ · [ e(B) · gᵢ(B) · stepsᵢ · ips · B + r_item · stepsᵢ · ips + r_step · stepsᵢ ]

- i 是长度先验里的一支（共 12 支：3 个尺度 × 4 种形状），wᵢ 是它的权重；
- stepsᵢ = max(1, ceil(batch · Eᵢ[ceil(w / B)] / ips))，w 是一个请求的长度；
- e(B) 是满块时每列的效率，r_item / r_step 是每个 item / 每个 step 的固定开销（标定取 r_item = 0）；
- **gᵢ(B) 是新增的项**：一个 item 实际做了满块的百分之几。

g 的定义（x = item 的有效长度 / B）：

g(x) = f(B) + (1 − f(B)) · clamp(2x − 1, 0, 1)

f(B) 是地板比例：item 即使只有很少的有效 token，也要付 f(B) 个满块的代价；剩下的部分在 block 的后半段线性涨上来。f(B) = 1 时就是旧模型。

## 实测（TPU v6e，batch 64，ips 8，表宽 128 页）

| block | f(B) | e(B) |
|---|---|---|
| 128 | 0.921 | 1.000 |
| 256 | 0.881 | 0.787 |
| 512 | 0.803 | 0.756 |
| 1024 | 0.703 | 0.742 |
| 2048 | 0.627 | 0.756 |

每个 item 的地板约等于 0.414 µs + 0.630 ns/token × B。

旧模型假定每个 item 都按满块收费，所以在“短请求 + 宽表”时高估大 block 约 1.3–1.6 倍，默认先验会把选择压到 256/8（实测慢 3.65%）。加上 g 之后，同一个默认先验选 512/8，实测 0.00–0.20%。

## 标定

```
PYTHONPATH=. python benchmarks/softmax_attention/calibrate_mla_decode_config.py \
    --out benchmarks/softmax_attention/results/mla_calibration --floor-scan --validate
```

`--floor-scan` 会额外测 5 个 block × 8 个长度的网格（表宽固定 128 页），拟合 f(B)，并把每个 cell 的 slots 乘上它的 g 后再拟合 column、e(B)、r_step。

不重新上 TPU、只用已保存的测量重拟合：

```
PYTHONPATH=. python benchmarks/softmax_attention/refit_mla_floor.py
```

## 当前状态

- `CALIBRATION` 保持为空，路由没有改。
- 新的常数在 `benchmarks/softmax_attention/results/mla_calibration_v3/constants.py`。
- 验证结果：随机抽样、64k/128k 长上下文、四种验证 pattern 的 regret 都是 0–0.44%。
