#!/usr/bin/env python3
# Copyright 2026 heavyball-research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Decode steps of served models on one device: flywheel against the kernels
they are served with.

    PYTHONPATH=. python benchmarks/softmax_attention/decode_microbench.py \\
        --torchtpu-source ~/vllm-torchtpu \\
        --output results/benchmark/decode_microbench/flywheel.json
    python benchmarks/softmax_attention/decode_microbench.py --list
    python benchmarks/softmax_attention/decode_microbench.py \\
        --table results/benchmark/decode_microbench/*.json

A case is one decode step of a served model at its per-device attention
shape, a number of sequences and their cached lengths. A GQA case's shape is
its query heads, KV heads, head_dim and KV dtype, on 256-token pages and
max_model_len 16384, as vllm-torchtpu serves it. An MLA case attends in the
absorbed form, one KV head whose latent and rope are K and whose latent is V;
its shape is its query heads and KV dtype, on MLA v2's serving pages and
max_model_len 16384. CASES holds the cases: Qwen3.5-35B-A3B-FP8 at TP4 over
its serving runs' length mixes, batch sizes and fixed lengths; other served
models at TP4; the decode cells of ForgeMegakernel (arXiv 2609.12379, Table
3) on one device; DeepSeek-V3/R1 (all heads on a device, as tpu-inference
serves it with data-parallel attention, fp8 KV) over the same length mixes,
batch sizes and fixed lengths; and Kimi-K2 (fp8) and DeepSeek-V2-Lite (bf16)
at 1k/1k.

The step runs the model's attention layers back to back (at most
--max-layers of them), each with its own KV cache and all reading one step's
lengths and page table, so XLA merges the work built from those alone
(batched RPA's schedule, flash_mla_decode's) across the layers as it does in
serving. Each layer's query depends on the layer before, the step's metadata
passes an optimization barrier (so that work is not hoisted out of the timed
loop), and every kernel appends the step's token. The kernels, each given q
in the layout it takes:

  fw           flywheel. GQA: flash_attn_with_kvcache as vllm-torchtpu's
               flywheel backend calls it for decode. MLA: flash_mla_decode on
               token-major bf16 q with bf16 scores, its max_seqlen_k the
               step's longest sequence, as serving passes its bound
  rpa          GQA: batched RPA (load_batched_rpa, --torchtpu-source), as
               vllm-torchtpu's CUSTOM backend runs ragged_paged_attention:
               the decode request distribution and its own default block
               sizes
  mla_v2       MLA: mla_ragged_paged_attention (--mla-v2-source) as
               tpu-inference's mla_attention calls it: head-major q_nope (the
               layout its q_nope einsum emits), the block sizes its
               get_tuned_params gives the step's TuningKey, no
               vmem_limit_bytes; on fp8 KV w8a16, bf16 q with
               k_scale = v_scale

Times are device time per op from the profiler, in us per layer: flywheel's
kernel; batched RPA's decode kernel, its mixed kernel (launched in
decode-only steps too) and its schedule (once per step, spread over the
model's attention layers); MLA v2's kernels and the q and output transposes
its entry runs. loop_us is all device time of the timed loop per layer, the
query chaining and each kernel's ops outside its kernels included. The DMA
floor is the KV bytes a layer reads (the cache layout's padding included) at
--hbm-tbps. Each kernel's output is checked against an f32 reference over its
own cache, an MLA case's by FlashMLA's cosine diff, and its append byte for
byte; an MLA case's rows past a request's kv_lens are free, as MLA v2
rewrites the new token's whole packed word. To compare flywheel versions, run
once per checkout and join the result files with --table.
"""

# jax loads only where a TPU run needs it, so --list and --table never claim
# the TPU.
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import argparse
import collections
import dataclasses
import fnmatch
import functools
import glob
import json
import math
import os
import pathlib
import re
import zlib

import numpy as np

from benchmarks.common.mla import DEFAULT_MLA_V2_SOURCE, load_mla_v2

PAGE = 256  # a GQA case's page, vllm-torchtpu's
MAX_MODEL_LEN = 16384
MIN_MAX_SEQS = 128
NUM_REFERENCE_SEQS = 8
# Note (david): flash_mla_decode's cache row, kept here so that --list and
# --table never import jax: the latent, the rope, then zero padding lanes.
LATENT_DIM = 512
ROPE_DIM = 64
CACHE_ROW_DIM = 640
# MLA v2's serving page of each KV dtype, a fixed number of 32-bit words.
MLA_PAGE_SIZE = {"bf16": 512, "fp8": 1024}
# FlashMLA's benchmark scale for unit-variance q and cache.
SM_SCALE = (LATENT_DIM + ROPE_DIM) ** -0.5
# Note (david): an fp8 case quantizes its unit-normal cache and new tokens
# under one per-tensor scale, as a caller with a calibrated static scale
# does: a few sigma land on e4m3's largest finite value, and the rare value
# past it clips. The scale is no power of two, so a scale applied where it
# rounds shows in cos_diff.
FP8_SCALE = 6 / 448
# Note (david): cos_diff = |x - y|^2 / (|x|^2 + |y|^2), so a kernel that drops
# a fraction of each request's softmax mass scores about half that fraction on
# unit-normal operands. The bound catches a dropped page per request at every
# length the cases reach, and stays far above what rounding gives, fp8
# probabilities for P @ V included.
COS_DIFF_GROSS = 1e-3
# A GQA output's largest error, over the reference's largest value, that the
# checks flag.
GQA_REL_ERR_BOUND = 0.05
KERNELS = ("fw", "rpa", "mla_v2")
GQA_BASELINES = ("rpa",)
MLA_BASELINES = ("mla_v2",)
BASELINE_LABELS = {"rpa": "batched RPA", "mla_v2": "MLA v2"}
MLA_V2_CASE_CLASS = {
    "bd": "mla_v2_batched_decode",
    "d": "mla_v2_decode",
    "m": "mla_v2_mixed",
}
# The op classes a kernel's time sums; batched RPA's adds its schedule.
KERNEL_CLASSES = {
    "fw": ("fw_kernel",),
    "rpa": ("rpa_decode", "rpa_mixed", "rpa_other_kernel"),
    "mla_v2": (*MLA_V2_CASE_CLASS.values(), "mla_v2_xpose"),
}


@dataclasses.dataclass(frozen=True)
class Case:
    """One decode step: a per-device attention shape and its sequences."""

    name: str
    source: str  # model, sharding, and where the batch and lengths come from
    hq: int  # query heads per device
    hk: int  # KV heads per device (MLA: the one latent head)
    d: int  # head_dim (MLA: the latent and the rope)
    dtype: str  # KV cache dtype: "fp8" or "bf16"
    layers: int  # full-attention layers per step in the model
    b: int  # sequences
    # ("fixed", L) | ("uniform", lo, hi) | ("skew", long, lo, hi)
    # | ("lognormal", median, sigma), in cached tokens before the step
    lens: tuple
    mla: bool = False  # absorbed MLA: flash_mla_decode against MLA v2


QWEN35 = {"hq": 4, "hk": 1, "d": 256, "dtype": "fp8", "layers": 10}
Q35 = "Qwen3.5-35B-A3B-FP8 TP4"
U1K = ("uniform", 1024, 2048)  # 1k/1k decode, requests staggered
CASES = [
    # vllm-torchtpu's serving runs of Qwen3.5-35B-A3B-FP8: c64, 128 seqs max.
    Case(
        "prod_decode_trace",
        f"{Q35}; the decode-only trace: 64 prompts of 1024 started together,"
        " traced at ~1.2-1.5k cached tokens",
        **QWEN35,
        b=64,
        lens=("fixed", 1344),
    ),
    Case(
        "prod_1k1k",
        f"{Q35}; 1k/1k at c64, requests staggered",
        **QWEN35,
        b=64,
        lens=U1K,
    ),
    Case(
        "prod_8k1k",
        f"{Q35}; 8k/1k at c64 (input 7373 +-11%, output 922 +-11%)",
        **QWEN35,
        b=64,
        lens=("uniform", 6554, 9115),
    ),
    Case(
        "prod_wide",
        f"{Q35}; 1k/1k at range ratio 0.8",
        **QWEN35,
        b=64,
        lens=("uniform", 205, 3687),
    ),
    Case(
        "prod_skew",
        f"{Q35}; one 16k sequence among 63 short ones",
        **QWEN35,
        b=64,
        lens=("skew", 16000, 128, 1024),
    ),
    Case(
        "prod_lognormal",
        f"{Q35}; lognormal lengths, median 1024, sigma 1",
        **QWEN35,
        b=64,
        lens=("lognormal", 1024, 1.0),
    ),
]
CASES += [
    Case(
        f"batch_b{b}",
        f"{Q35}; {b} sequences at 1k/1k lengths",
        **QWEN35,
        b=b,
        lens=U1K,
    )
    for b in (1, 8, 16, 32, 128, 256)
]
CASES += [
    Case(
        f"fixed_L{length}",
        f"{Q35}; 64 sequences of {length} cached tokens",
        **QWEN35,
        b=64,
        lens=("fixed", length),
    )
    for length in (256, 4096, 16000)
]
# Other served models' per-device shapes, 64 sequences at 1k/1k lengths.
CASES += [
    Case(
        "q35_397b_tp4",
        "Qwen3.5-397B-A17B-FP8 TP4 (32/2 heads, KV replicated)",
        hq=8,
        hk=1,
        d=256,
        dtype="fp8",
        layers=15,
        b=64,
        lens=U1K,
    ),
    Case(
        "q35_27b_tp4",
        "Qwen3.5-27B TP4 (24/4 heads)",
        hq=6,
        hk=1,
        d=256,
        dtype="fp8",
        layers=16,
        b=64,
        lens=U1K,
    ),
    Case(
        "q3vl_30b_tp4",
        "Qwen3-VL-30B-A3B TP4 (32/4 heads)",
        hq=8,
        hk=1,
        d=128,
        dtype="bf16",
        layers=48,
        b=64,
        lens=U1K,
    ),
    Case(
        "llama2_13b_tp4",
        "Llama-2-13B TP4 (MHA, 40/40 heads)",
        hq=10,
        hk=10,
        d=128,
        dtype="bf16",
        layers=40,
        b=64,
        lens=U1K,
    ),
]
# ForgeMegakernel's decode cells (batch, context), one device, bf16 KV, each
# model's head counts from its config. Not included: its Llama-3.2-1B cell
# (b=1, S=96), head_dim 64, which vllm-torchtpu's flywheel backend does not
# decode.
CASES += [
    Case(
        f"paper_{tag}_b{b}_s{seq}",
        f"{model} TP1 ({hq}/{hk} heads); paper cell b={b}, S={seq}",
        hq=hq,
        hk=hk,
        d=128,
        dtype="bf16",
        layers=layers,
        b=b,
        lens=("fixed", seq),
    )
    for tag, model, hq, hk, layers, b, seq in [
        ("qwen3_0.6b", "Qwen3-0.6B", 16, 8, 28, 8, 2048),
        ("qwen3_0.6b", "Qwen3-0.6B", 16, 8, 28, 16, 1024),
        ("qwen3_0.6b", "Qwen3-0.6B", 16, 8, 28, 16, 4096),
        ("qwen3_4b_8b", "Qwen3-4B and Qwen3-8B", 32, 8, 36, 1, 128),
        ("qwen3_4b", "Qwen3-4B", 32, 8, 36, 8, 1024),
        ("qwen3_8b", "Qwen3-8B", 32, 8, 36, 1, 512),
        ("qwen3_8b", "Qwen3-8B", 32, 8, 36, 1, 2048),
        ("llama3.1_8b", "Llama-3.1-8B", 32, 8, 32, 4, 2048),
        ("llama2_13b", "Llama-2-13B-chat (MHA)", 40, 40, 40, 1, 128),
        ("minicpm5_1b", "MiniCPM5-1B", 16, 2, 24, 1, 128),
        (
            "minicpm4",
            "MiniCPM4 (MiniCPM4-8B's heads; the paper's 1B and 3B)",
            32,
            2,
            32,
            1,
            128,
        ),
    ]
]
# MLA models, absorbed: one KV head of the latent and the rope.
MLA = {"hk": 1, "d": LATENT_DIM + ROPE_DIM, "mla": True}
DSV3 = {"hq": 128, "dtype": "fp8", "layers": 61, **MLA}
D3 = "DeepSeek-V3/R1 FP8, data-parallel attention (128 heads a device)"
CASES += [
    Case(
        "dsv3_1k1k",
        f"{D3}; 1k/1k at c64, requests staggered",
        **DSV3,
        b=64,
        lens=U1K,
    ),
    Case(
        "dsv3_8k1k",
        f"{D3}; 8k/1k at c64 (input 7373 +-11%, output 922 +-11%)",
        **DSV3,
        b=64,
        lens=("uniform", 6554, 9115),
    ),
    Case(
        "dsv3_wide",
        f"{D3}; 1k/1k at range ratio 0.8",
        **DSV3,
        b=64,
        lens=("uniform", 205, 3687),
    ),
    Case(
        "dsv3_skew",
        f"{D3}; one 16k sequence among 63 short ones",
        **DSV3,
        b=64,
        lens=("skew", 16000, 128, 1024),
    ),
    Case(
        "dsv3_lognormal",
        f"{D3}; lognormal lengths, median 1024, sigma 1",
        **DSV3,
        b=64,
        lens=("lognormal", 1024, 1.0),
    ),
]
CASES += [
    Case(
        f"dsv3_batch_b{b}",
        f"{D3}; {b} sequences at 1k/1k lengths",
        **DSV3,
        b=b,
        lens=U1K,
    )
    for b in (1, 8, 16, 32, 128, 256)
]
CASES += [
    Case(
        f"dsv3_fixed_L{length}",
        f"{D3}; 64 sequences of {length} cached tokens",
        **DSV3,
        b=64,
        lens=("fixed", length),
    )
    for length in (256, 4096, 16000)
]
CASES += [
    Case(
        "kimi_k2_1k1k",
        "Kimi-K2 FP8, data-parallel attention (64 heads a device); 1k/1k"
        " at c64",
        hq=64,
        dtype="fp8",
        layers=61,
        b=64,
        lens=U1K,
        **MLA,
    ),
    Case(
        "dsv2_lite_1k1k",
        "DeepSeek-V2-Lite BF16 TP1 (16 heads); 1k/1k at c64",
        hq=16,
        dtype="bf16",
        layers=27,
        b=64,
        lens=U1K,
        **MLA,
    ),
]


def case_lens(case: Case) -> np.ndarray:
    """Cached tokens of each sequence before the step, seeded by the name."""
    rng = np.random.default_rng(zlib.crc32(case.name.encode()))
    kind, *a = case.lens
    if kind == "fixed":
        return np.full(case.b, a[0])
    if kind == "uniform":
        return rng.integers(a[0], a[1], case.b)
    if kind == "skew":
        lens = rng.integers(a[1], a[2], case.b)
        lens[rng.integers(case.b)] = a[0]
        return lens
    if kind == "lognormal":
        lens = np.round(rng.lognormal(math.log(a[0]), a[1], case.b))
        return np.clip(lens, 16, MAX_MODEL_LEN - 384).astype(int)
    raise ValueError(kind)


def lens_label(lens) -> str:
    kind, *a = lens
    if kind == "fixed":
        return f"{a[0]}"
    if kind == "uniform":
        return f"{a[0]}–{a[1] - 1}"
    if kind == "skew":
        return f"1×{a[0]} + {a[1]}–{a[2] - 1}"
    return f"lognormal, median {a[0]}"


def case_page(case: Case) -> int:
    """Tokens a page of the case's cache."""
    return MLA_PAGE_SIZE[case.dtype] if case.mla else PAGE


def token_bytes(case: Case) -> int:
    """Cache bytes of one token, the layout's padding included: an MLA row,
    or flywheel's merged rows, each KV head's K then its V."""
    if case.mla:
        return CACHE_ROW_DIM * (1 if case.dtype == "fp8" else 2)
    import jax.numpy as jnp

    from flywheel_tpu.softmax_attention.kv_layout import merged_cache_heads

    dtype = jnp.float8_e4m3fn if case.dtype == "fp8" else jnp.bfloat16
    rows = merged_cache_heads(case.hk, dtype)
    return rows * case.d * jnp.dtype(dtype).itemsize


def case_kernels(case: Case, kernels) -> list[str]:
    """The kernels a case runs: flywheel and its family's baselines."""
    family = ("fw",) + (MLA_BASELINES if case.mla else GQA_BASELINES)
    return [k for k in kernels if k in family]


def kv_rows(hk: int) -> tuple[list[int], list[int]]:
    """(K rows, V rows) of each KV head among one token's head rows: each
    head's K row then its V row, [k0, v0, k1, v1, ...], in batched RPA's cache
    and in flywheel's merged cache, bf16 and fp8 alike."""
    return [2 * h for h in range(hk)], [2 * h + 1 for h in range(hk)]


class Step:
    """One decode step of a case: its metadata, per-layer caches and q.
    GqaStep and MlaStep draw the family's operands and run its kernels, the
    ones in head_major_kernels on head-major q and output."""

    head_major_kernels = ()

    def __init__(self, case: Case, iters: int):
        import jax.numpy as jnp

        self.case, self.iters = case, iters
        self.page = case_page(case)
        self.dtype = jnp.float8_e4m3fn if case.dtype == "fp8" else jnp.bfloat16
        self.lens = case_lens(case)
        total = self.lens + 1
        self.max_seqlen_k = int(total.max())
        rng = np.random.default_rng(zlib.crc32(case.name.encode()) + 1)
        self.pages_per_seq = MAX_MODEL_LEN // self.page
        self.max_seqs = max(MIN_MAX_SEQS, case.b)
        need = -(-total // self.page)
        self.num_pages = int(need.sum()) + 1
        # pages scattered over the pool, as after a long serving run
        perm = rng.permutation(self.num_pages)
        table = np.zeros((self.max_seqs, self.pages_per_seq), np.int32)
        start = 0
        for i, n in enumerate(need):
            table[i, :n] = perm[start : start + n]
            start += n
        self.table = table
        kv_lens = np.zeros(self.max_seqs, np.int32)
        kv_lens[: case.b] = total  # serving's seq_lens count the step's token
        cu_q_lens = np.zeros(self.max_seqs + 1, np.int32)
        cu_q_lens[1 : case.b + 1] = np.arange(1, case.b + 1)
        cu_q_lens[case.b + 1 :] = case.b
        self.meta = (
            jnp.asarray(kv_lens),
            jnp.asarray(table.reshape(-1)),
            jnp.asarray(cu_q_lens),
            jnp.asarray([case.b, case.b, case.b], jnp.int32),
            jnp.asarray(case.b, jnp.int32),
        )
        # Drawn by the family's step: the layers' caches and layer 0's q.
        self.caches = ()
        self.q = None

    def loop(self, kernel):
        """The compiled timed loop: iters steps of every layer."""
        import jax
        import jax.numpy as jnp
        from jax import lax

        # flywheel_tpu jits internally, keyed on its arguments
        jax.clear_caches()
        head_major = kernel in self.head_major_kernels

        @functools.partial(jax.jit, donate_argnums=(0,))
        def run(caches, q):
            # Note (david): once a loop, so its iters * layers calls share it.
            if head_major:
                q = jnp.transpose(q, (1, 0, 2))

            def body(_, carry):
                caches, q, acc = carry
                # q varies per step, so the barrier's outputs do too: the
                # work built from the metadata alone runs every step, as in
                # serving, not once per loop.
                meta, q = lax.optimization_barrier((self.meta, q))
                new = []
                for cache in caches:
                    out, cache = self.layer(kernel, q, cache, meta)
                    new.append(cache)
                    q = q.astype(jnp.float32) + 1e-3 * out.astype(jnp.float32)
                    q = q.astype(jnp.bfloat16)
                return tuple(new), q, acc + out[0, 0, 0].astype(jnp.float32)

            caches, q, acc = lax.fori_loop(
                0, self.iters, body, (caches, q, jnp.float32(0))
            )
            if head_major:
                q = jnp.transpose(q, (1, 0, 2))
            return caches, q, acc

        return run.lower(self.caches, self.q).compile()

    def once(self, kernel):
        """One layer on a copy of layer 0's cache: (output (b, hq, head_dim
        of V), cache)."""
        import jax
        import jax.numpy as jnp

        jax.clear_caches()
        head_major = kernel in self.head_major_kernels
        q = jnp.transpose(self.q, (1, 0, 2)) if head_major else self.q
        out, cache = jax.jit(lambda c, q: self.layer(kernel, q, c, self.meta))(
            jnp.copy(self.caches[0]), q
        )
        return (jnp.transpose(out, (1, 0, 2)) if head_major else out), cache


class GqaStep(Step):
    """A GQA case's step: q, the step's k and v, and flywheel's merged caches,
    which batched RPA reads reshaped to its own layout."""

    def __init__(self, baselines: dict, case: Case, num_layers: int, iters):
        import jax
        import jax.numpy as jnp

        from flywheel_tpu.softmax_attention.kv_layout import merged_cache_heads

        super().__init__(case, iters)
        self.rpa = baselines.get("rpa")
        self.rows = merged_cache_heads(case.hk, self.dtype)
        self.fw_shape = (self.num_pages, PAGE, self.rows, case.d)
        if self.rpa is not None:
            self.rpa_shape = tuple(
                self.rpa.get_kv_cache_shape(
                    self.num_pages, PAGE, case.hk, case.d, self.dtype
                )
            )
            if math.prod(self.rpa_shape) != math.prod(self.fw_shape):
                raise ValueError(
                    f"cache sizes differ: {self.fw_shape} vs {self.rpa_shape}"
                )
        keys = jax.random.split(
            jax.random.key(zlib.crc32(case.name.encode())), 4
        )
        base = jax.random.normal(keys[0], self.fw_shape, jnp.bfloat16)
        base = base.astype(self.dtype)
        self.caches = tuple(jnp.copy(base) for _ in range(num_layers))
        del base
        self.q = jax.random.normal(
            keys[1], (case.b, case.hq, case.d), jnp.bfloat16
        )
        kv_shape = (case.b, case.hk, case.d)
        self.k = jax.random.normal(keys[2], kv_shape, jnp.bfloat16).astype(
            self.dtype
        )
        self.v = jax.random.normal(keys[3], kv_shape, jnp.bfloat16).astype(
            self.dtype
        )
        self.scale = case.d**-0.5
        self.descale = 1.0 if case.dtype == "fp8" else None

    def layer(self, kernel, q, cache, meta):
        """One attention layer's decode: (output [b, hq, d], cache)."""
        from flywheel_tpu import flash_attn_with_kvcache

        kv_lens, page_indices, cu_q_lens, distribution, num_active = meta
        b = self.case.b
        if kernel == "fw":
            # as vllm-torchtpu's flywheel backend decodes: the step's K/V
            # appended in-kernel at the length before it
            out, cache = flash_attn_with_kvcache(
                q[:, None],
                cache,
                None,
                self.k[:, None],
                self.v[:, None],
                cache_seqlens=kv_lens[:b] - 1,
                block_table=page_indices[: b * self.pages_per_seq].reshape(
                    b, self.pages_per_seq
                ),
                num_active=num_active,
                softmax_scale=self.scale,
                k_descale=self.descale,
                v_descale=self.descale,
            )
            return out[:, 0], cache
        out, cache = self.rpa.ragged_paged_attention(
            q,
            self.k,
            self.v,
            cache.reshape(self.rpa_shape),
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            sm_scale=self.scale,
            k_scale=self.descale,
            v_scale=self.descale,
        )
        return out[:b], cache.reshape(self.fw_shape)

    def fw_table_entry(self) -> list | None:
        return table_entry(self.case, self.dtype)

    def append_ok(self, cache) -> bool:
        """Only the step's token slots changed, holding the step's K and V."""
        import jax.numpy as jnp

        def bits(x):
            itemsize = jnp.dtype(x.dtype).itemsize
            return x.view(jnp.uint8 if itemsize == 1 else jnp.uint16)

        b = self.case.b
        changed = jnp.any(bits(cache) != bits(self.caches[0]), axis=(2, 3))
        pos = self.lens[:b]
        pages, slots = self.table[np.arange(b), pos // PAGE], pos % PAGE
        expect = np.zeros((self.num_pages, PAGE), bool)
        expect[pages, slots] = True
        new = cache[jnp.asarray(pages), jnp.asarray(slots)]
        k_rows, v_rows = kv_rows(self.case.hk)
        return (
            bool(jnp.array_equal(changed, jnp.asarray(expect)))
            and bool(jnp.array_equal(bits(new[:, k_rows]), bits(self.k)))
            and bool(jnp.array_equal(bits(new[:, v_rows]), bits(self.v)))
        )

    def reference(self, cache) -> np.ndarray:
        """f32 attention of the first sequences over a kernel's cache."""
        import jax
        import jax.numpy as jnp

        c = self.case
        rows = cache.reshape(self.num_pages * PAGE, self.rows, c.d)
        k_rows, v_rows = (jnp.asarray(r) for r in kv_rows(c.hk))
        outs = []
        for i in range(min(NUM_REFERENCE_SEQS, c.b)):
            n = int(self.lens[i]) + 1
            pages = self.table[i, : -(-n // PAGE)]
            index = (pages[:, None] * PAGE + np.arange(PAGE)).reshape(-1)[:n]
            kv = rows[jnp.asarray(index)].astype(jnp.float32)
            q = self.q[i].astype(jnp.float32).reshape(c.hk, c.hq // c.hk, c.d)
            s = jnp.einsum("hgd,nhd->hgn", q, kv[:, k_rows]) * self.scale
            p = jax.nn.softmax(s, axis=-1)
            o = jnp.einsum("hgn,nhd->hgd", p, kv[:, v_rows])
            outs.append(o.reshape(c.hq, c.d))
        return np.asarray(jnp.stack(outs))


class MlaStep(Step):
    """An MLA case's step: q_nope (q) and q_pe, the step's new latent tokens,
    and caches in MLA v2's word layout, which both kernels read."""

    head_major_kernels = MLA_BASELINES

    def __init__(self, baselines: dict, case: Case, num_layers: int, iters):
        import jax
        import jax.numpy as jnp

        from flywheel_tpu.softmax_attention.mla_decode import mla_cache_shape

        super().__init__(case, iters)
        self.fp8 = case.dtype == "fp8"
        self.cache_shape = mla_cache_shape(
            self.num_pages, self.page, self.dtype
        )
        seed = zlib.crc32(case.name.encode()) % 2**31
        keys = jax.random.split(jax.random.key(seed), 5)

        def draw(key, shape):
            return jax.random.normal(key, shape, jnp.bfloat16)

        def draw_kv(key, shape):
            # Note (david): as tpu-inference's static per-tensor quantize: the
            # division in f32, clipped to the largest finite fp8 value.
            rows = draw(key, shape)
            if self.fp8:
                bound = float(jnp.finfo(jnp.float8_e4m3fn).max)
                return jnp.clip(
                    rows.astype(jnp.float32) / FP8_SCALE, -bound, bound
                ).astype(jnp.float8_e4m3fn)
            else:
                return rows

        @functools.partial(jax.jit, static_argnums=(1,))
        def draw_cache(key, shape):
            rows = draw_kv(key, (*shape, LATENT_DIM + ROPE_DIM))
            pad = CACHE_ROW_DIM - LATENT_DIM - ROPE_DIM
            rows = jnp.pad(rows, ((0, 0), (0, 0), (0, pad)))
            return rows.reshape(self.cache_shape)

        base = draw_cache(keys[0], (self.num_pages, self.page))
        self.caches = tuple(jnp.copy(base) for _ in range(num_layers))
        del base
        self.q = draw(keys[1], (case.b, case.hq, LATENT_DIM))
        self.q_pe = draw(keys[2], (case.b, case.hq, ROPE_DIM))
        self.new_kv_c = draw_kv(keys[3], (case.b, LATENT_DIM))
        self.new_k_pe = draw_kv(keys[4], (case.b, ROPE_DIM))

        self.mla_v2 = baselines.get("mla_v2")
        if self.mla_v2 is not None:
            kernel, tuned = self.mla_v2
            v2_shape = tuple(
                kernel.get_kv_cache_shape(
                    self.num_pages, self.page, LATENT_DIM + ROPE_DIM, self.dtype
                )
            )
            if v2_shape != self.cache_shape:
                raise ValueError(
                    f"cache shapes differ: {self.cache_shape} vs {v2_shape}"
                )
            _, words, packing, _ = self.cache_shape
            fields = {
                "max_num_tokens": case.b,
                "actual_num_q_heads": case.hq,
                "actual_lkv_dim": LATENT_DIM,
                "actual_r_dim": ROPE_DIM,
                "kv_dtype": jnp.dtype(self.dtype).name,
                "q_dtype": jnp.dtype(jnp.bfloat16).name,
                "page_size_per_kv_packing": words,
                "kv_packing": packing,
                "max_num_seqs": self.max_seqs,
                "pages_per_seq": self.pages_per_seq,
            }
            decode = tuned.get_tuned_params(
                tuned.TuningKey(case="batched_decode", **fields)
            )
            mixed = tuned.get_tuned_params(
                tuned.TuningKey(case="mixed", **fields)
            )
            # Note (david): mla_attention runs prefill with the mixed params
            # and passes no vmem_limit_bytes, so the kernel takes the core's
            # VMEM.
            self.v2_params = {
                "num_kv_pages_per_block": (
                    decode.num_kv_pages_per_block,
                    mixed.num_kv_pages_per_block,
                    mixed.num_kv_pages_per_block,
                ),
                "num_queries_per_block": (
                    decode.num_queries_per_block,
                    mixed.num_queries_per_block,
                    mixed.num_queries_per_block,
                ),
                "decode_batch_size": decode.decode_batch_size,
                "mixed_q_split": mixed.q_split,
            }
            if self.fp8:
                self.v2_params.update(k_scale=FP8_SCALE, v_scale=FP8_SCALE)

    def layer(self, kernel, q, cache, meta):
        """One MLA layer's decode, q (q_nope) in the kernel's layout: (output,
        cache), the output in the same layout."""
        import jax.numpy as jnp

        from flywheel_tpu.softmax_attention.mla_decode import flash_mla_decode

        kv_lens, page_indices, cu_q_lens, distribution, _ = meta
        b = self.case.b
        if kernel == "fw":
            out, cache = flash_mla_decode(
                q[:, None],
                self.q_pe[:, None],
                self.new_kv_c[:, None],
                self.new_k_pe[:, None],
                cache,
                kv_lens[:b],
                page_indices[: b * self.pages_per_seq].reshape(
                    b, self.pages_per_seq
                ),
                sm_scale=SM_SCALE,
                kv_scale=(
                    jnp.asarray(FP8_SCALE, jnp.float32) if self.fp8 else None
                ),
                causal=True,
                s_dtype=jnp.bfloat16,
                max_seqlen_k=self.max_seqlen_k,
            )
            return out[:, 0], cache
        else:
            kernel_module, _ = self.mla_v2
            # Note (david): the step fixes the static arguments itself, so it
            # calls the kernel without its own jax.jit.
            return kernel_module.mla_ragged_paged_attention.__wrapped__(
                q,
                self.q_pe,
                self.new_kv_c,
                self.new_k_pe,
                cache,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                sm_scale=SM_SCALE,
                **self.v2_params,
            )

    def fw_table_entry(self) -> list | None:
        from flywheel_tpu.softmax_attention.dispatch import (
            get_tuned_mla_config,
        )

        entry = get_tuned_mla_config(
            cache_dtype=self.dtype,
            num_query_rows=self.case.hq,
            batch=self.case.b,
            max_seqlen_k=self.max_seqlen_k,
        )
        return None if entry is None else list(entry)

    def token_rows(self, cache):
        """(num_pages * page, CACHE_ROW_DIM): row t of page p is token
        p * page + t."""
        return cache.reshape(self.num_pages * self.page, CACHE_ROW_DIM)

    def append_ok(self, cache) -> bool:
        """The step's token rows hold its latent, rope and zero padding, and
        no row changed but those and the free rows a request's pages hold
        past its kv_lens."""
        import jax.numpy as jnp

        def bits(x):
            itemsize = jnp.dtype(x.dtype).itemsize
            return x.view(jnp.uint8 if itemsize == 1 else jnp.uint16)

        b = self.case.b
        rows = self.token_rows(cache)
        changed = jnp.any(
            bits(rows) != bits(self.token_rows(self.caches[0])), axis=1
        )
        pos = self.lens[:b]
        flat = self.table[np.arange(b), pos // self.page] * self.page + (
            pos % self.page
        )
        allowed = np.zeros(self.num_pages * self.page, bool)
        allowed[flat] = True
        # Note (david): MLA v2 writes the whole packed word of the new token,
        # so the free slots after it in that word change; they hold no token.
        for i in range(b):
            total = int(pos[i]) + 1
            pages = self.table[i, : -(-total // self.page)]
            index = (pages[:, None] * self.page + np.arange(self.page)).reshape(
                -1
            )
            allowed[index[total:]] = True
        pad = CACHE_ROW_DIM - LATENT_DIM - ROPE_DIM
        want = jnp.concatenate(
            [
                self.new_kv_c,
                self.new_k_pe,
                jnp.zeros((b, pad), self.new_kv_c.dtype),
            ],
            axis=1,
        )
        stray = jnp.any(changed & ~jnp.asarray(allowed))
        return (not bool(stray)) and bool(
            jnp.array_equal(bits(rows[jnp.asarray(flat)]), bits(want))
        )

    def reference(self, cache) -> np.ndarray:
        """f32 attention of the first sequences over a kernel's cache:
        (n, hq, LATENT_DIM)."""
        import jax.numpy as jnp

        q_nope = np.asarray(self.q.astype(jnp.float32))
        q_pe = np.asarray(self.q_pe.astype(jnp.float32))
        scale = np.float32(FP8_SCALE if self.fp8 else 1.0)
        rows = self.token_rows(cache)
        outs = []
        for i in range(min(NUM_REFERENCE_SEQS, self.case.b)):
            n = int(self.lens[i]) + 1
            pages = self.table[i, : -(-n // self.page)]
            index = (pages[:, None] * self.page + np.arange(self.page)).reshape(
                -1
            )[:n]
            kv = np.asarray(rows[jnp.asarray(index)].astype(jnp.float32))
            kv = kv * scale
            latent = kv[:, :LATENT_DIM]
            rope = kv[:, LATENT_DIM : LATENT_DIM + ROPE_DIM]
            s = (q_nope[i] @ latent.T + q_pe[i] @ rope.T) * np.float32(SM_SCALE)
            s = s - s.max(axis=-1, keepdims=True)
            p = np.exp(s)
            p = p / p.sum(axis=-1, keepdims=True)
            outs.append(p @ latent)
        return np.stack(outs)


def op_class(name: str) -> str | None:
    if name.startswith(("flash_attn_with_kvcache", "flash_mla_decode")):
        return "fw_kernel"
    if name.startswith("RPA"):
        return {"d": "rpa_decode", "m": "rpa_mixed"}.get(
            name[3:4], "rpa_other_kernel"
        )
    if name.startswith("rpa_metadata_schedule"):
        return "rpa_schedule"
    match = re.match(r"MLA[-_](bd|d|m)[-_]", name)
    if match:
        return MLA_V2_CASE_CLASS[match.group(1)]
    if name.startswith("xpose"):
        return "mla_v2_xpose"
    return None


def device_times(trace_dir: str):
    """Per op class on the first TPU: [total ns, count], and the timed loops'
    durations (ns) in run order."""
    import jax

    xplane = sorted(glob.glob(f"{trace_dir}/**/*.xplane.pb", recursive=True))[
        -1
    ]
    data = jax.profiler.ProfileData.from_file(xplane)
    plane = next(p for p in data.planes if p.name.startswith("/device:TPU:"))
    ops = collections.defaultdict(lambda: [0, 0])
    modules = []
    for line in plane.lines:
        if line.name == "XLA Ops":
            for event in line.events:
                # an op's event name is its HLO text: "%name.N = (...) ..."
                match = re.match(r"%?([^\s=]+)", event.name)
                cls = op_class(match.group(1)) if match else None
                if cls:
                    ops[cls][0] += event.duration_ns
                    ops[cls][1] += 1
        elif line.name == "XLA Modules":
            modules += [
                (event.start_ns, event.duration_ns)
                for event in line.events
                if event.name.startswith("jit_run")
            ]
    return ops, [duration for _, duration in sorted(modules)]


def table_entry(case: Case, dtype) -> list | None:
    """flywheel's tuned-table entry for a GQA case's decode call (None: its
    default config), from the same key fields the kernel looks up."""
    from flywheel_tpu.softmax_attention import dispatch

    config = dispatch.get_tuned_kvcache_config(
        dtype=dtype,
        num_query_heads=case.hq,
        num_kv_heads=case.hk,
        head_dim=case.d,
        capacity=MAX_MODEL_LEN,
        batch=case.b,
        append=True,
        sliding_window=None,
        return_lse=False,
    )
    return None if config is None else list(config)


def run_case(baselines: dict, case: Case, args, trace_root: str) -> dict:
    import jax
    import jax.numpy as jnp

    kernels = case_kernels(case, args.kernels)
    lens = case_lens(case)
    page = case_page(case)
    num_pages = int((-(-(lens + 1) // page)).sum()) + 1
    row_bytes = token_bytes(case)
    budget_layers = int(args.mem_gb * 2**30 // (num_pages * page * row_bytes))
    num_layers = max(1, min(case.layers, args.max_layers, budget_layers))
    step = (MlaStep if case.mla else GqaStep)(
        baselines, case, num_layers, args.iters
    )
    res = {
        "case": dataclasses.asdict(case),
        "device": jax.devices()[0].device_kind,
        "kernels": kernels,
        "lens_mean": float(lens.mean()),
        "layers_simulated": num_layers,
        "iters": args.iters,
        "dma_floor_us": (
            int((lens + 1).sum()) * row_bytes / (args.hbm_tbps * 1e6)
        ),
        "fw_table_entry": step.fw_table_entry(),
    }

    # Checks on layer 0, each kernel over its own copy of the cache.
    for kernel in kernels:
        try:
            out, cache = step.once(kernel)
        except Exception as e:  # pylint: disable=broad-exception-caught
            res[f"{kernel}_error"] = repr(e)[:400]
            continue
        ref = step.reference(cache)
        out = np.asarray(out[: len(ref)].astype(jnp.float32))
        if case.mla:
            # Note (david): FlashMLA's cosine diff, NaN and inf counted as 0.
            x = np.where(np.isfinite(out), out, 0.0).astype(np.float64)
            y = np.where(np.isfinite(ref), ref, 0.0).astype(np.float64)
            res[f"{kernel}_cos_diff"] = 1.0 - 2.0 * float(
                (x * y).sum()
            ) / float((x * x + y * y).sum())
        else:
            scale = float(np.max(np.abs(ref))) or 1.0
            res[f"{kernel}_rel_err"] = float(np.max(np.abs(out - ref))) / scale
        res[f"{kernel}_append_ok"] = step.append_ok(cache)
        del out, cache

    loops = {k: step.loop(k) for k in kernels if f"{k}_error" not in res}
    for run in loops.values():  # warm
        step.caches, step.q, acc = run(step.caches, step.q)
        acc.block_until_ready()
    trace_dir = os.path.join(trace_root, case.name)
    with jax.profiler.trace(trace_dir):
        for run in loops.values():
            step.caches, step.q, acc = run(step.caches, step.q)
            acc.block_until_ready()
    ops, modules = device_times(trace_dir)
    calls = args.iters * num_layers
    for i, kernel in enumerate(loops):
        per = {cls: ops[cls][0] / 1e3 / calls for cls in KERNEL_CLASSES[kernel]}
        per["attention_us"] = sum(per.values())
        if kernel == "rpa":
            per["rpa_schedule_per_step"] = (
                ops["rpa_schedule"][0] / 1e3 / args.iters
            )
            per["rpa_schedule_ops_per_step"] = (
                ops["rpa_schedule"][1] / args.iters
            )
            # once per step in serving, spread over the attention layers
            per["rpa_schedule"] = per["rpa_schedule_per_step"] / case.layers
            per["attention_us"] += per["rpa_schedule"]
        if i < len(modules):
            per["loop_us"] = modules[i] / 1e3 / calls
        res[kernel] = per
    for base in kernels:
        if base != "fw" and "fw" in res and base in res:
            res[f"fw_over_{base}"] = (
                res["fw"]["attention_us"] / res[base]["attention_us"]
            )
    return res


def check_failures(r: dict, kernel: str) -> list[str]:
    """A kernel's failed checks in a case's record."""
    failed = []
    if r.get(f"{kernel}_append_ok") is False:
        failed.append("append")
    if abs(r.get(f"{kernel}_cos_diff") or 0) > COS_DIFF_GROSS:
        failed.append(f"cos_diff {r[kernel + '_cos_diff']:.1e}")
    if (r.get(f"{kernel}_rel_err") or 0) > GQA_REL_ERR_BOUND:
        failed.append(f"rel_err {r[kernel + '_rel_err']:.2g}")
    return failed


def summary(name: str, r: dict) -> str:
    parts = [name + ":"]
    for kernel in r["kernels"]:
        if kernel in r:
            p = r[kernel]
            s = f"{kernel} {p['attention_us']:.1f} us/layer"
            if kernel == "rpa":
                s += (
                    f" (decode {p['rpa_decode']:.1f}, mixed"
                    f" {p['rpa_mixed']:.1f}, schedule"
                    f" {p['rpa_schedule_per_step']:.1f}/step in"
                    f" {p['rpa_schedule_ops_per_step']:.0f} ops)"
                )
            parts.append(s + ";")
        elif f"{kernel}_error" in r:
            parts.append(f"{kernel} ERROR {r[kernel + '_error'][:120]};")
        failed = check_failures(r, kernel)
        if failed:
            parts.append(f"{kernel} CHECK FAILED: {', '.join(failed)};")
    for base in r["kernels"]:
        if f"fw_over_{base}" in r:
            parts.append(f"fw/{base} {r[f'fw_over_{base}']:.2f};")
    parts.append(
        f"floor {r['dma_floor_us']:.1f}; table {r.get('fw_table_entry')};"
        f" layers {r['layers_simulated']}"
    )
    return " ".join(parts)


def print_table(paths: list[str]) -> None:
    """Markdown tables, one a family: the baselines from the first file,
    flywheel from each, over each baseline."""
    runs = [
        (pathlib.Path(p).stem, json.loads(pathlib.Path(p).read_text("utf-8")))
        for p in paths
    ]
    first = runs[0][1]
    for mla, baselines in ((False, GQA_BASELINES), (True, MLA_BASELINES)):
        names = [
            name
            for name, r in first.items()
            if bool(r["case"].get("mla")) == mla
        ]
        if not names:
            continue
        print(
            "| case | "
            + ("heads, KV" if mla else "heads (q/kv), head_dim, KV")
            + " | seqs | cached tokens | "
            + " | ".join(BASELINE_LABELS[b] for b in baselines)
            + " | "
            + " | ".join(f"flywheel {n}" for n, _ in runs)
            + " | DMA floor | notes |"
        )
        print(
            "|---|---|---:|---|"
            + "---:|" * (len(baselines) + len(runs) + 1)
            + "---|"
        )
        for name in names:
            r = first[name]
            c = r["case"]
            notes = []
            base_times = {}
            for b in baselines:
                base_times[b] = r.get(b, {}).get("attention_us")
                error = r.get(f"{b}_error")
                if error:
                    notes.append(
                        f"{BASELINE_LABELS[b]}: "
                        + ("out of VMEM" if "vmem" in error else error[:60])
                    )
                failed = check_failures(r, b)
                if failed:
                    notes.append(
                        f"{BASELINE_LABELS[b]} CHECK FAILED: "
                        + ", ".join(failed)
                    )
            cells = []
            for n, run in runs:
                rr = run.get(name, {})
                v = rr.get("fw", {}).get("attention_us")
                ratios = [
                    f"{v / base_times[b]:.2f}x"
                    for b in baselines
                    if v is not None and base_times[b]
                ]
                if v is None:
                    cells.append("—")
                elif ratios:
                    cells.append(f"{v:.1f} ({', '.join(ratios)})")
                else:
                    cells.append(f"{v:.1f}")
                if rr.get("fw_error"):
                    notes.append(f"flywheel {n}: {rr['fw_error'][:60]}")
                failed = check_failures(rr, "fw")
                if failed:
                    notes.append(
                        f"flywheel {n} CHECK FAILED: " + ", ".join(failed)
                    )
                if rr and rr.get("fw_table_entry") is None and "fw" in rr:
                    notes.append(f"flywheel {n}: default config")
            if mla:
                shape = f"{c['hq']}, {c['dtype']}"
            else:
                shape = f"{c['hq']}/{c['hk']}, {c['d']}, {c['dtype']}"
            base_cells = [
                "—" if base_times[b] is None else f"{base_times[b]:.1f}"
                for b in baselines
            ]
            print(
                f"| {name} | {shape} | {c['b']} | {lens_label(c['lens'])} | "
                + " | ".join(base_cells + cells)
                + f" | {r.get('dma_floor_us', 0):.1f} | {'; '.join(notes)} |"
            )
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", maxsplit=1)[0]
    )
    parser.add_argument("--list", action="store_true", help="list the cases")
    parser.add_argument(
        "--table", nargs="+", metavar="JSON", help="markdown table of results"
    )
    parser.add_argument(
        "--output", help="result file (JSON); cases already in it are skipped"
    )
    parser.add_argument(
        "--cases", default="*", help="comma-separated globs of case names"
    )
    parser.add_argument(
        "--kernels",
        default=",".join(KERNELS),
        help="comma-separated kernels; a case runs those of its family",
    )
    parser.add_argument(
        "--torchtpu-source",
        type=pathlib.Path,
        help="a vllm-torchtpu checkout for batched RPA (load_batched_rpa)",
    )
    parser.add_argument(
        "--mla-v2-source",
        type=pathlib.Path,
        default=DEFAULT_MLA_V2_SOURCE,
        help="a tpu-inference checkout at the pinned commit, for MLA v2",
    )
    parser.add_argument(
        "--iters", type=int, default=20, help="steps per timed loop"
    )
    parser.add_argument("--max-layers", type=int, default=10)
    parser.add_argument(
        "--mem-gb", type=float, default=24, help="KV cache budget of a step"
    )
    parser.add_argument(
        "--hbm-tbps",
        type=float,
        default=3.5,
        help="DMA rate of the floor (a v7x core's measured rate)",
    )
    parser.add_argument(
        "--trace", help="profiler trace directory (default: OUTPUT.traces)"
    )
    args = parser.parse_args()
    args.kernels = args.kernels.split(",")
    patterns = args.cases.split(",")
    cases = [
        c for c in CASES if any(fnmatch.fnmatch(c.name, p) for p in patterns)
    ]
    if args.list:
        for c in cases:
            print(
                f"{c.name:28s} {'mla' if c.mla else 'gqa'}"
                f" {c.hq:3d}/{c.hk:<2d} d{c.d:<3d} {c.dtype:4s} b{c.b:<3d}"
                f" {lens_label(c.lens):22s} {c.layers:2d} layers  {c.source}"
            )
        return
    if args.table:
        print_table(args.table)
        return
    if not args.output:
        parser.error("--output is required to run cases")
    baselines = {}
    if "rpa" in args.kernels and any(not c.mla for c in cases):
        if args.torchtpu_source is None:
            parser.error("batched RPA needs --torchtpu-source")
        from benchmarks.common.rpa import load_batched_rpa

        baselines["rpa"] = load_batched_rpa(args.torchtpu_source)
    if set(args.kernels) & set(MLA_BASELINES) and any(c.mla for c in cases):
        baselines["mla_v2"] = load_mla_v2(args.mla_v2_source)
    import jax

    print(jax.devices()[0].device_kind, jax.__version__, flush=True)
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(output.read_text("utf-8")) if output.exists() else {}
    trace_root = args.trace or f"{args.output}.traces"
    for case in cases:
        if case.name in results:
            continue
        try:
            r = run_case(baselines, case, args, trace_root)
        except Exception as e:  # pylint: disable=broad-exception-caught
            r = {"case": dataclasses.asdict(case), "error": repr(e)[:600]}
            print(f"{case.name}: ERROR {repr(e)[:600]}", flush=True)
        else:
            print(summary(case.name, r), flush=True)
        results[case.name] = r
        output.write_text(json.dumps(results, indent=1, default=str), "utf-8")
        jax.clear_caches()


if __name__ == "__main__":
    main()
