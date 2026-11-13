# Benchmarks

The numbers on this page were measured on TPU v6e with `jax[tpu]==0.11.0` and
`libtpu==0.0.44`, the stack tpu-inference runs. The same benchmarks on v7x are
in [v7x.md](v7x.md). Run the commands below from a source checkout, in the venv
of the [README](../README.md#installation)'s `uv sync --extra tpu`.

## Softmax attention

Causal prefill on one v6e chip, head_dim 256, B = 1, for 32:32 (MHA)
and 32:4 (GQA) heads. The rows under each sequence length are our speedup
over [RPA v3](https://github.com/vllm-project/tpu-inference/tree/4420cae/tpu_inference/kernels/ragged_paged_attention/v3) (gray) and [Splash Attention](https://github.com/openxla/tokamax/tree/84e5f36/tokamax/_src/ops/experimental/tpu/splash_attention) (orange).

![Attention kernel throughput](../assets/softmax_attn_benchmark.png)

Every number in the figure comes from
[`benchmarks/softmax_attention/`](../benchmarks/softmax_attention/). Each cell
times a whole attention block under one `jax.jit`, from hidden states
x (B, T, d_model) to hidden states out, with d_model = heads × head_dim:

1. qkv projection: one einsum per q, k and v, each emitting the layout its
   kernel takes, as vLLM's attention layer does;
2. layout: whatever relayout XLA could not fold into the projection;
3. the attention kernel;
4. output projection.

| Backend | Kernel input | Kernel |
|---|---|---|
| flywheel | (B, heads, T, head_dim) | `flash_attn_func` |
| Splash Attention | (B · heads, T, head_dim) | `make_splash_mha_single_device` |
| RPA v3 | (B · T, heads, head_dim) and a paged KV cache | `ragged_paged_attention`, which also writes the cache |

Each cell yields two numbers:

- **Block level**: the median wall clock of the jitted block, from
  back-to-back calls with one sync per round
  ([`common/timing.py`](../benchmarks/common/timing.py)).
- **Kernel level**, which the figure plots: the cell reruns 3 calls under
  `jax.profiler`, and `segments.py` reads the device events of that same
  executable from the trace. Every leaf op goes into one segment by what it
  is: the Pallas call into `attn_kernel`, pure data movement (copy,
  transpose, reshape, slice, pad, dtype conversion, ...) into `layout`, and
  the rest into `qkv_proj` or `out_proj` by its `jax.named_scope`. Anything
  left is reported as `other`. The figure's throughput is attention FLOPs /
  `attn_kernel` time, with attention FLOPs = 4 · B · T² · heads · head_dim,
  halved for causal; the speedup rows are ratios of these throughputs.

Both baselines are loaded from their pinned submodules, with no install, and
each loaded file is checked against its SHA-256 at the pin:
tpu-inference's [RPA v3](https://github.com/vllm-project/tpu-inference/tree/4420cae/tpu_inference/kernels/ragged_paged_attention/v3)
at `4420cae` and Tokamax's [Splash Attention](https://github.com/openxla/tokamax/tree/84e5f36/tokamax/_src/ops/experimental/tpu/splash_attention)
at `84e5f36`. Their block configs come from `rpa_tuned_v6e.json` and
`splash_tuned_v6e.json` next to the script, searched earlier per
(heads, heads_k, head_dim, mask, T) at B = 1 on v6e, with tuners that are not
in this repo. On a v7x chip the script reads `rpa_tuned_v7x.json` and
`splash_tuned_v7x.json` instead, searched with
[`tune_blocks.py`](../benchmarks/softmax_attention/tune_blocks.py). No chip
stands in for another: on one with no tables of its own, the two baselines
are an error until `--rpa-table` and `--splash-table` name a table. The
tables cover every cell of the figure. A cell outside them
runs RPA's `get_default_block_sizes` or Tokamax's heuristic config instead,
and its JSONL record says so in `block_source` / `config_source`. flywheel
runs its own block-size lookup inside `flywheel_tpu`.

To reproduce the figure, sequentially on one v6e chip:

```bash
git submodule update --init third_party/tpu-inference third_party/tokamax
benchmarks/softmax_attention/attention_block.sh results/benchmark/softmax_attention
```

The defaults are the figure's grid: RPA v3, Splash Attention and flywheel,
32 q heads with 32 or 4 kv heads, head_dim 256, causal, B = 1, T from 1K to
128K. `IMPLS`, `HEADS`, `HEADS_K`, `HEAD_DIM`, `MASK` and `SEQLENS` override
them. The output directory gets `benchmark.jsonl`, `benchmark.log` and one
profiler trace per cell under `traces/`, which XProf opens. At the end the
script prints the kernel-level table the figure plots and the block-level
split, which shows the layout work RPA v3 does before its kernel runs and
flywheel and Splash Attention do not need. Rerunning into the same directory
measures every cell again, and the last ok record of each cell wins.

To time one cell, reprint the tables, or re-split a directory of traces:

```bash
PYTHONPATH=. uv run --no-sync python benchmarks/softmax_attention/attention_block.py \
    --impl flywheel --heads 32 --heads-k 4 --head-dim 256 --seq 8192 \
    --output results/benchmark/softmax_attention/benchmark.jsonl \
    --trace results/benchmark/softmax_attention/traces
PYTHONPATH=. uv run --no-sync python benchmarks/softmax_attention/report.py \
    results/benchmark/softmax_attention/benchmark.jsonl
PYTHONPATH=. uv run --no-sync python benchmarks/softmax_attention/segments.py \
    results/benchmark/softmax_attention/traces
```

Only fixed-shape prefill is benchmarked; the varlen and KV-cache APIs have
no benchmark entrypoint.

## MLA decode

Multi-head latent attention (DeepSeek-V3, Kimi K2) caches one row per token
that every query head shares: 512 latent values (`kv_lora_rank`) and a
64-wide RoPE key. Decode runs it in the absorbed form, MQA over that row:
QKᵀ spans all 576 values and PV the first 512. This section times
flywheel's `flash_mla_decode` against tpu-inference's
[`mla_ragged_paged_attention`](https://github.com/vllm-project/tpu-inference/tree/4420cae/tpu_inference/kernels/mla/v2)
(MLA v2) at `4420cae`, on one v6e chip, with `jax[tpu]==0.12.0` and
`libtpu==0.0.50`. MLA v2 is loaded from the pinned submodule by
[`common/mla.py`](../benchmarks/common/mla.py), each file checked against
its SHA-256.

The cells are the decode and verify cells of
[`kvcache_scenarios.py`](../benchmarks/softmax_attention/kvcache_scenarios.py)
for `--impl flywheel_mla` and `mla_v2`, on 1024-token pages shuffled across
sequences, the page tpu-inference serves MLA with for every cache dtype. Each
kernel reads its own cache layout and takes q in its own layout, both
prepared outside the clock:

- **MLA v2** (`mla_v2`) runs as tpu-inference's `mla_attention` layer calls
  it: on the cache its cache manager allocates by default (32 tokens per packed
  row, `MLA_KV_PACKING_SIZE`, each row padded to 640 values), with q
  quantized to the cache's dtype as serving does for an fp8 cache, with q,
  the new rows and the metadata padded to the next power of two of tokens
  and of sequences as its model runner pads a step, decode in its
  `batched_decode` case and speculative verify (MTP's K + 1 query tokens
  per sequence) in its `mixed` case, since serving sends only one-token
  decodes to the decode case. Its blocks come from its tuned table under the
  keys serving looks up, else its defaults. The table holds fp8 shapes only,
  each for one exact batch and pages per sequence, so every cell runs the
  defaults except the three in the last table.
- **flywheel** (`flywheel_mla`) runs `flash_mla_decode` on MLA v2's word
  layout at the dtype's packing, with bf16 q, causal over a verify window,
  and `max_seqlen_k` the longest sequence's final length. Its config comes
  from the `mla|` entries of `tuned_configs.json` when one matches, else its
  heuristic.

Each record's `block_source` says which blocks a cell ran. A timed call runs
64 decode steps or 8 verify steps; every step appends one latent row per new
token and attends to the whole cache. The tables give the median wall clock
per step, MLA v2's first and flywheel's second, and MLA v2's time over
flywheel's in parentheses. Each head count and cache dtype has a core cell,
64 sequences of 4K tokens, and each sweep moves one setting of it. 128 to 16
heads are DeepSeek-V3's 128 query heads at TP 1 to 8.

Decode, by sequences × tokens per sequence, µs per step:

| bf16 | 128 heads | 64 heads | 32 heads | 16 heads |
|---|---|---|---|---|
| 1 × 4K | 32.6 → 30.4 (1.07x) | 27.7 → 24.8 (1.12x) | 31.0 → 25.6 (1.21x) | 24.5 → 30.5 (0.81x) |
| 8 × 4K | 72.6 → 47.2 (1.54x) | 64.9 → 45.4 (1.43x) | 62.1 → 45.0 (1.38x) | 59.8 → 44.7 (1.34x) |
| 32 × 4K | 187.0 → 160.9 (1.16x) | 174.2 → 157.5 (1.11x) | 167.6 → 155.6 (1.08x) | 165.8 → 154.6 (1.07x) |
| 64 × 4K | 364.6 → 319.3 (1.14x) | 337.7 → 313.2 (1.08x) | 325.2 → 307.6 (1.06x) | 319.6 → 308.5 (1.04x) |
| 128 × 4K | 718.8 → 639.4 (1.12x) | 674.4 → 625.3 (1.08x) | 649.8 → 618.1 (1.05x) | 636.8 → 611.6 (1.04x) |
| 256 × 4K | 1442 → 1279 (1.13x) | 1341 → 1251 (1.07x) | 1298 → 1234 (1.05x) | 1273 → 1218 (1.05x) |
| 64 × 1K | 174.2 → 92.7 (1.88x) | 150.0 → 85.6 (1.75x) | 140.5 → 81.8 (1.72x) | 136.1 → 79.9 (1.70x) |
| 64 × 16K | 1269 → 1235 (1.03x) | 1246 → 1228 (1.01x) | 1234 → 1220 (1.01x) | 1232 → 1220 (1.01x) |
| 64 × 64K | 4918 → 4900 (1.00x) | 4895 → 4858 (1.01x) | 4878 → 4881 (1.00x) | 4893 → 4874 (1.00x) |
| 64 × 128K | 9802 → 9772 (1.00x) | 9776 → 9751 (1.00x) | 9765 → 9754 (1.00x) | 9752 → 9758 (1.00x) |

| fp8 | 128 heads | 64 heads | 32 heads | 16 heads |
|---|---|---|---|---|
| 1 × 4K | 50.7 → 29.6 (1.72x) | 31.1 → 30.6 (1.02x) | 28.7 → 23.8 (1.20x) | 30.5 → 23.5 (1.30x) |
| 8 × 4K | 97.4 → 52.2 (1.86x) | 56.0 → 39.9 (1.40x) | 45.1 → 34.9 (1.29x) | 41.2 → 33.1 (1.24x) |
| 32 × 4K | 218.4 → 138.5 (1.58x) | 127.7 → 91.5 (1.40x) | 100.8 → 80.2 (1.26x) | 95.7 → 79.2 (1.21x) |
| 64 × 4K | 403.6 → 265.1 (1.52x) | 236.2 → 171.7 (1.38x) | 182.5 → 159.5 (1.14x) | 173.4 → 156.9 (1.10x) |
| 128 × 4K | 784.7 → 518.3 (1.51x) | 461.4 → 330.9 (1.39x) | 358.4 → 317.7 (1.13x) | 335.9 → 313.4 (1.07x) |
| 256 × 4K | 1506 → 1024 (1.47x) | 898.1 → 658.5 (1.36x) | 718.5 → 633.5 (1.13x) | 649.7 → 626.1 (1.04x) |
| 64 × 1K | 236.2 → 89.1 (2.65x) | 134.6 → 61.0 (2.21x) | 99.8 → 52.5 (1.90x) | 85.7 → 49.8 (1.72x) |
| 64 × 16K | 1056 → 994.1 (1.06x) | 634.5 → 620.7 (1.02x) | 621.6 → 611.0 (1.02x) | 611.6 → 610.8 (1.00x) |
| 64 × 64K | 3666 → 3562 (1.03x) | 2466 → 2454 (1.00x) | 2431 → 2427 (1.00x) | 2428 → 2441 (0.99x) |
| 64 × 128K | 7088 → 7525 (0.94x) | 4904 → 4970 (0.99x) | 4870 → 4875 (1.00x) | 4865 → 4864 (1.00x) |

One sequence of 4K tokens, device time per step from the profiler trace, µs.
The one-sequence rows above are wall clock, most of which is the host
dispatching the step's call:

| 1 × 4K | 128 heads | 64 heads | 32 heads | 16 heads |
|---|---|---|---|---|
| bf16 | 30.5 → 9.3 (3.27x) | 25.7 → 8.3 (3.09x) | 23.3 → 8.2 (2.84x) | 21.4 → 8.1 (2.64x) |
| fp8 | 43.7 → 8.5 (5.14x) | 27.6 → 6.4 (4.31x) | 22.5 → 5.8 (3.88x) | 19.5 → 5.6 (3.47x) |

Uneven lengths, the core cell's tokens split across its 64 sequences another
way (`--pattern`, the lengths of [`common/lengths.py`](../benchmarks/common/lengths.py)):
spiky gives one sequence 32/95 of them, about 88K, and the rest the same
share; longtail falls off linearly from 8K to 126. Each kernel's step time
over its own uniform cell, and MLA v2's time over flywheel's:

| | 128, bf16 | 128, fp8 | 64, bf16 | 64, fp8 | 32, bf16 | 32, fp8 | 16, bf16 | 16, fp8 |
|---|---|---|---|---|---|---|---|---|
| spiky, MLA v2 | 1.14x | 1.30x | 1.13x | 1.31x | 1.14x | 1.25x | 1.13x | 1.18x |
| spiky, flywheel | 1.01x | 1.22x | 1.01x | 1.29x | 1.02x | 1.14x | 1.00x | 1.10x |
| spiky, MLA v2 / flywheel | 1.28x | 1.63x | 1.21x | 1.40x | 1.18x | 1.26x | 1.17x | 1.19x |
| longtail, MLA v2 | 0.99x | 0.95x | 1.00x | 0.95x | 1.00x | 0.95x | 0.99x | 0.95x |
| longtail, flywheel | 1.00x | 1.71x | 1.01x | 1.12x | 1.01x | 1.00x | 1.00x | 1.11x |
| longtail, MLA v2 / flywheel | 1.13x | 0.84x | 1.07x | 1.17x | 1.05x | 1.08x | 1.02x | 0.95x |

Verify, 64 sequences of 4K tokens and 1 to 4 query tokens each, µs per step:

| bf16 | 128 heads | 64 heads | 32 heads | 16 heads |
|---|---|---|---|---|
| 1 | 366.2 → 321.3 (1.14x) | 338.0 → 315.1 (1.07x) | 326.6 → 311.7 (1.05x) | 320.8 → 307.9 (1.04x) |
| 2 | 2129 → 338.6 (6.29x) | 1143 → 323.4 (3.53x) | 641.7 → 313.6 (2.05x) | 414.6 → 311.5 (1.33x) |
| 3 | 2183 → 415.7 (5.25x) | 1160 → 330.8 (3.51x) | 636.0 → 318.0 (2.00x) | 416.3 → 313.3 (1.33x) |
| 4 | 2181 → 535.4 (4.07x) | 1161 → 340.1 (3.41x) | 649.6 → 324.0 (2.00x) | 418.3 → 315.5 (1.33x) |

| fp8 | 128 heads | 64 heads | 32 heads | 16 heads |
|---|---|---|---|---|
| 1 | 403.5 → 265.1 (1.52x) | 236.2 → 171.7 (1.38x) | 182.6 → 160.0 (1.14x) | 173.5 → 157.8 (1.10x) |
| 2 | 3735 → 477.7 (7.82x) | 1907 → 276.0 (6.91x) | 1012 → 176.4 (5.74x) | 580.8 → 160.3 (3.62x) |
| 3 | 3764 → 687.5 (5.47x) | 1924 → 376.5 (5.11x) | 1015 → 225.3 (4.51x) | 570.2 → 163.8 (3.48x) |
| 4 | 3765 → 902.9 (4.17x) | 1925 → 472.2 (4.08x) | 1022 → 274.2 (3.73x) | 583.2 → 174.6 (3.34x) |

The three fp8 shapes MLA v2's tuned table holds, µs per step:

| fp8 | MLA v2 | flywheel | MLA v2 / flywheel |
|---|---|---|---|
| 128 heads, 8 × 3K, decode | 77.0 (tuned) | 43.9 (heuristic) | 1.75x |
| 64 heads, 64 × 9K, decode | 368.5 (tuned) | 408.0 (tuned) | 0.90x |
| 64 heads, 64 × 9K, verify 2 tokens | 15406 (tuned) | 614.6 (tuned) | 25.07x |

To reproduce the tables, on one v6e chip (131 cells for each of the two
kernels, about an hour):

```bash
git submodule update --init third_party/tpu-inference
IMPLS= MLA_IMPLS="flywheel_mla mla_v2" SCENARIOS="decode verify" \
    MLA_HEADS="128 64 32 16" \
    benchmarks/softmax_attention/kvcache_scenarios.sh results/benchmark/kvcache_scenarios_mla
```

`MLA_HEADS`, `MLA_KV_DTYPES`, `MLA_CORE_BATCH`, `MLA_CORE_SEQ`,
`MLA_DECODE_BATCHES`, `MLA_DECODE_SEQS`, `MLA_PATTERNS`, `MLA_VERIFY_TOKENS`
and `MLA_TUNED` override the grid, and an empty sweep is skipped. The output
directory gets `benchmark.jsonl`, `benchmark.log` and one profiler trace per
cell under `traces/`; each record's `trace_device_ms` is the device time of
one timed call. To time one cell:

```bash
PYTHONPATH=. uv run --no-sync python benchmarks/softmax_attention/kvcache_scenarios.py \
    --impl flywheel_mla --scenario verify --decode-tokens 8 --verify-tokens 2 \
    --heads 128 --kv-dtype float8_e4m3fn --batch 64 --seq 4096 \
    --output results/benchmark/kvcache_scenarios_mla/benchmark.jsonl \
    --trace results/benchmark/kvcache_scenarios_mla/traces
```

## Linear attention

GDN prefill on one v6e chip, n_kq = 16 and d_k = d_v = 128, with T tokens
packed into 1 or 8 sequences. The row under each sequence length is our
speedup over the vendored `gdn_v3` baseline (gray).

![GDN prefill kernel throughput](../assets/gdn_benchmark.png)

For GDN and KDA kernels, the vendored baselines live in
`benchmarks/linear_attention/baseline_kernel/`. The GDN benchmark compares
`gdn_v3` with our fused kernel by default; KDA's `--impl baseline` times the
chunked forward baseline, while `--pkgs` times fused kernels:

```bash
PYTHONPATH=. uv run --no-sync python benchmarks/linear_attention/benchmark/bench_gdn_fwd.py \
    --output results/gdn_fwd.json
PYTHONPATH=. uv run --no-sync python benchmarks/linear_attention/benchmark/bench_kda_fwd.py \
    --pkgs flywheel_tpu.linear_attention.kda --output results/kda_fwd.json
PYTHONPATH=. uv run --no-sync python benchmarks/linear_attention/benchmark/bench_kda_fwd.py \
    --impl baseline --output results/kda_fwd_baseline.json
```

