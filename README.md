<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-lockup-dark.svg">
    <img src="assets/logo-lockup.svg" alt="FlyWheel" width="70%">
  </picture>
</p>
<h1 align="center">
</h1>

<p align="center">
| <a href="docs/"><b>Documentation</b></a> | <a href="https://arxiv.org/abs/2611.01426"><b>Paper</b></a> | <a href="https://heavyball-research.github.io/blog/2026-09-29-attention-tpu/"><b>Blog</b></a> | <a href="https://github.com/heavyball-research/flywheel-tpu/releases/latest"><b>Latest Release</b></a> | <a href="https://pypi.org/project/flywheel-tpu/"><b>PyPI</b></a> |
</p>

[![CI](https://github.com/heavyball-research/flywheel-tpu/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/heavyball-research/flywheel-tpu/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/flywheel-tpu)](https://pypi.org/project/flywheel-tpu/)
[![License](https://img.shields.io/github/license/heavyball-research/flywheel-tpu)](LICENSE)

**FlyWheel** is an attention kernel library for TPU, written in JAX Pallas by
[heavyball-research](https://github.com/heavyball-research). It implements
softmax attention (MHA and MQA/GQA) behind flash-attn's API, and the fused
linear attention layers of hybrid models (GDN and KDA), for prefill, decode
and training on TPU v6e and v7x.

## Why FlyWheel?

- **Fast**: up to 2.32x over RPA v3 on chunked prefill over a paged KV cache
  ([v7x](docs/v7x.md#serving-with-a-kv-cache)), and 2.25x over RPA v3 and
  1.47x over Splash Attention on 16K sliding window prefill
  ([v6e](https://github.com/heavyball-research/flywheel-tpu/pull/12)). See the
  [benchmarks](docs/benchmarks.md).
- **flash-attn API**: `flash_attn_func`, `flash_attn_varlen_func` and
  `flash_attn_with_kvcache` keep flash-attn's names and arguments, and take
  and return JAX arrays.
- **Serving-ready**: paged bf16 and fp8 KV caches, chunked prefill, prefix
  caching and speculative verify windows.

## Core Features

### Softmax Attention
- **Prefill**: dense, varlen (packed sequences), and chunked prefill over a
  paged KV cache
- **Decode**: paged or contiguous KV cache, with the K/V append fused into the
  kernel; speculative verify windows run on the decode kernel
- **MLA decode**: DeepSeek-V2/V3/R1 and Kimi-K2, with fused cache append
  and bf16/fp8 paged latent caches
- **KV cache formats**: bf16, or an fp8 (e4m3, e5m2) paged cache with
  per-tensor descales
- **Attention variants**: causal, sliding window that skips fully masked
  tiles, softcap, attention sinks (`learnable_sink`), and fused rotary
  embedding
- **Backward**: `jax.grad` runs a Pallas backward kernel for MHA and MQA/GQA
  with causal or full masks

### Linear Attention
- **Fused GDN and KDA**: causal conv1d + silu and the chunked delta rule in
  one kernel, for Gated DeltaNet (e.g. Qwen3-Next) and Kimi Delta Attention
  (e.g. Kimi Linear)
- **Ragged serving batches**: decodes and prefills in one call, with
  per-request conv and recurrent states in slot caches, prefix caching and
  speculative verify windows
- **ReplaySSM decode**: decode from a state checkpoint plus the records of
  recent tokens, as in [ReplaySSM](https://arxiv.org/abs/2610.13778)

## News

Latest: [![GitHub Release](https://img.shields.io/github/v/release/heavyball-research/flywheel-tpu)](https://github.com/heavyball-research/flywheel-tpu/releases/latest)

- [2026/11] [v0.4.1](https://github.com/heavyball-research/flywheel-tpu/releases/tag/v0.4.1): optimized decode kernel and GDN conv pools
- [2026/11] Paper on arXiv: [FlyWheel: Attention Kernels Built for the TPU](https://arxiv.org/abs/2611.01426)
- [2026/10] [v0.4.0](https://github.com/heavyball-research/flywheel-tpu/releases/tag/v0.4.0): fp8 paged KV cache, speculative verify windows on the decode kernel, and attention sinks
- [2026/10] [tpu-inference](https://github.com/vllm-project/tpu-inference) adds FlyWheel as `--attention-backend flywheel` ([#3837](https://github.com/vllm-project/tpu-inference/pull/3837))
- [2026/10] [v0.3.0](https://github.com/heavyball-research/flywheel-tpu/releases/tag/v0.3.0): sliding window prefill that skips masked tiles, sliding window and softcap on the paged KV cache, and ReplaySSM decode for GDN/KDA
- [2026/10] [v0.2.0](https://github.com/heavyball-research/flywheel-tpu/releases/tag/v0.2.0): faster chunked prefill, prefix caching and decode over a paged KV cache
- [2026/10] [TPU v7x results](docs/v7x.md). Thanks to [lk-chen](https://github.com/lk-chen) for testing and contributing them!
- [2026/09] [v0.1.0](https://github.com/heavyball-research/flywheel-tpu/releases/tag/v0.1.0): flash-attn-style softmax attention (forward and backward, varlen, KV cache) and fused GDN/KDA linear attention

## Getting Started

### Installation

```bash
pip install flywheel-tpu            # CPU, Pallas interpret mode
pip install "flywheel-tpu[tpu]"     # TPU VM, adds jax[tpu]>=0.11.2
```

From source, with [uv](https://github.com/astral-sh/uv) (needed for the
benchmarks and the end-to-end evaluation):

```bash
git clone https://github.com/heavyball-research/flywheel-tpu.git && cd flywheel-tpu
uv sync --extra tpu     # plain `uv sync` off TPU
source .venv/bin/activate
```

### Quick Start

```python
import jax
import jax.numpy as jnp
from flywheel_tpu import flash_attn_func, flash_attn_with_kvcache

kq, kk, kv = jax.random.split(jax.random.key(0), 3)

# Causal GQA prefill: 32 query heads share 4 KV heads.
# Layout is head-major, (batch, heads, seqlen, head_dim).
q = jax.random.normal(kq, (1, 32, 4096, 128), jnp.bfloat16)
k = jax.random.normal(kk, (1, 4, 4096, 128), jnp.bfloat16)
v = jax.random.normal(kv, (1, 4, 4096, 128), jnp.bfloat16)
out = flash_attn_func(q, k, v, causal=True)  # (1, 32, 4096, 128)

# One decode step for 8 sequences: append the new K/V to the cache and attend,
# in one kernel. The cache is token-major, (batch, max_len, heads_k, head_dim),
# and is updated in place: rebind the returned caches.
k_cache = jax.random.normal(kk, (8, 8192, 4, 128), jnp.bfloat16)
v_cache = jax.random.normal(kv, (8, 8192, 4, 128), jnp.bfloat16)
cache_seqlens = jnp.array([4096, 100, 2000, 8000, 1, 512, 3000, 7000], jnp.int32)
q_new = jax.random.normal(kq, (8, 1, 32, 128), jnp.bfloat16)
k_new = jax.random.normal(kk, (8, 1, 4, 128), jnp.bfloat16)
v_new = jax.random.normal(kv, (8, 1, 4, 128), jnp.bfloat16)
out, k_cache, v_cache = flash_attn_with_kvcache(
    q_new, k_cache, v_cache, k_new, v_new, cache_seqlens=cache_seqlens, causal=True
)  # out: (8, 1, 32, 128)
```

Off TPU, pass `interpret=True` to run the Pallas TPU interpreter.

### Documentation

- [API reference](docs/api.md): every argument of the softmax and linear
  attention kernels, and where the layouts differ from flash-attn's
- [Benchmarks](docs/benchmarks.md) on v6e, and the [v7x results](docs/v7x.md)
- [End-to-end evaluation](docs/e2e_evaluation.md): reproduce the paper's
  RULER, BABILong and Video-MME results with vLLM

## Adoption

FlyWheel powers attention in:

- [tpu-inference](https://github.com/vllm-project/tpu-inference) (vLLM on
  TPU), as `--attention-backend flywheel`:

  ```bash
  VLLM_WORKER_MULTIPROC_METHOD=fork vllm serve Qwen/Qwen3.5-0.8B --attention-backend flywheel
  ```

## Acknowledgement

FlyWheel's API follows [FlashAttention](https://github.com/Dao-AILab/flash-attention).
Its GDN and KDA kernels start from tpu-inference's
[GDN v3 kernel](https://github.com/vllm-project/tpu-inference/tree/4420cae/tpu_inference/kernels/gdn/v3),
and its design draws on tpu-inference's
[RPA v3](https://github.com/vllm-project/tpu-inference/tree/4420cae/tpu_inference/kernels/ragged_paged_attention/v3),
Tokamax's [Splash Attention](https://github.com/openxla/tokamax/tree/84e5f36/tokamax/_src/ops/experimental/tpu/splash_attention)
and [ReplaySSM](https://arxiv.org/abs/2610.13778).

## Citation

If you find FlyWheel helpful in your project or research, please cite our
[paper](https://arxiv.org/abs/2611.01426):

```bibtex
@misc{bao2026flywheelattentionkernelsbuilt,
      title={FlyWheel: Attention Kernels Built for the TPU},
      author={Zishuo Bao and Zeshen Zhang and Linkun Chen and Yucheng Lu},
      year={2026},
      eprint={2611.01426},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2611.01426},
}
```
