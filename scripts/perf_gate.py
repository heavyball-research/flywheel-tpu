#!/usr/bin/env python3
"""Perf gate: one realistic config per kernel path, timed through the public
API, and a comparison of two timed runs.

    PYTHONPATH=. uv run --no-sync python scripts/perf_gate.py run \\
        --out results/perf_gate/candidate.json [--only dense_fwd,decode_bf16]
    PYTHONPATH=. uv run --no-sync python scripts/perf_gate.py merge \\
        --runs run1.json run2.json run3.json --out results/perf_baseline.json
    PYTHONPATH=. uv run --no-sync python scripts/perf_gate.py compare \\
        --baseline results/perf_baseline.json \\
        --candidate results/perf_gate/candidate.json --threshold 2

run times every config (or the --only subset) in one process with
benchmarks/common/timing.py's time_call, --rounds times per config, and
writes one JSON; a config's median_ms is the median of its rounds' medians.
merge reduces runs from separate processes to the per-config median of their
medians and records each config's spread, (max - min) / min. compare prints
one row per config and exits 1 when any config of the candidate is more than
--threshold percent slower than the baseline.

On fa-hpc (v6e-1) three processes of main at 6a57d12 spread by at most 0.34%
per config (decode_bf16; every prefill and linear-attention config by at most
0.07%), and one full run takes about 7 minutes, so --threshold 2 leaves a
margin of over 4x. To confirm a failure, rerun the failing configs with
--only in two more processes and compare the merge of the three.

A comparison is valid on one machine and software stack only: every run
records its host, chip, jax / jaxlib / libtpu versions, git HEAD and a hash
of the flywheel_tpu sources it timed, and merge and compare refuse inputs
whose host, chip, versions or config parameters differ.

Every array is drawn once, outside the clock, and every call of a config does
the same work: KV-cache configs donate the cache and rewrite the same slots,
linear-attention configs chain their donated state buffers. Only the calls
are timed, the public entry points' XLA glue included. The last call's
outputs must be finite, so a kernel that breaks without crashing fails the
run instead of timing garbage.
"""

# jax loads only where a TPU run needs it, so compare and merge stay host-only.
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import argparse
import dataclasses
import datetime
import functools
import hashlib
import importlib
import importlib.metadata
import json
import pathlib
import socket
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
# Run metadata that must match between two runs before their times compare.
ENV_KEYS = ("hostname", "device_kind", "jax", "jaxlib", "libtpu")
PAGE_SIZE = 128
# write_kv_cache_pages is tpu-inference's prefill cache writer and is not
# re-exported at the package root; a refactor that moves it must update this
# module path in the same commit.
KV_WRITE_MODULE = "flywheel_tpu.softmax_attention.kernels.kv_cache_write"

# Llama-3-8B-like GQA attention: 32 query heads, 8 KV heads of 128.
HEADS = 32
HEADS_K = 8
HEAD_DIM = 128
DENSE_SEQ = 8192
# A ragged prefill batch of 8 prompts, 16384 tokens in all, none a multiple
# of the 128-token tile, so every tail path runs.
PREFILL_LENGTHS = (3989, 3127, 2577, 2049, 1531, 1283, 1011, 817)
# Decode rows hold a context drawn uniformly from [low, high).
DECODE_BATCH = 64
DECODE_CONTEXT = (6144, 10240)
VERIFY_BATCH = 32
VERIFY_TOKENS = 4
# Chunked prefill: (cached context, new tokens) per request.
CHUNKED_PREFILL = ((0, 2048), (4096, 2048), (2048, 1536), (7168, 512))
# The prefill cache fill: (cached context, new tokens) per request, some
# contexts mid-page so the writer splits page heads and tails.
KV_WRITE_REQUESTS = tuple(
    zip((0, 300, 1024, 0, 2048, 77, 513, 0), PREFILL_LENGTHS)
)
# Qwen3-Next's gated delta net and Kimi Linear's KDA layers.
GDN_DIMS = {"n_kq": 16, "n_v": 32, "d_k": 128, "d_v": 128, "kernel_size": 4}
KDA_DIMS = {"n_kq": 32, "n_v": 32, "d_k": 128, "d_v": 128, "kernel_size": 4}
LINEAR_PREFILL_LENGTHS = (3000, 2500, 1700, 992)
LINEAR_PARAM_SCALE = 0.1
REPLAY_BATCH = 128
REPLAY_SPEC_TOKENS = 3
REPLAY_WINDOW = 16
REPLAY_CONTEXT = 4096

Step = Callable[[Any], tuple[Any, Any]]


@dataclasses.dataclass(frozen=True)
class Config:
    """One gated kernel path: its parameters and the builder of its step."""

    name: str
    path: str
    params: dict[str, Any]
    build: Callable[[dict[str, Any]], tuple[Step, Any]]


def round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def merged_cache_heads(heads_k: int, kv_dtype: str) -> int:
    """Head rows of the paged merged cache: bf16 interleaves [k0, v0, k1, v1,
    ...]; fp8 stores head pairs, an odd count padded to a multiple of 4."""
    if kv_dtype == "bfloat16":
        return 2 * heads_k
    return round_up(2 * heads_k, 4)


@functools.cache
def normal_sampler(shape: tuple[int, ...], dtype_name: str) -> Callable:
    import jax
    import jax.numpy as jnp

    return jax.jit(
        lambda key: jax.random.normal(key, shape, jnp.float32).astype(
            dtype_name
        )
    )


def device_normal(seed: int, shape: tuple[int, ...], dtype: Any) -> Any:
    """Unit-normal values drawn on device, cast to dtype.

    The key is rbg's: on v6e with jax 0.11.2 a threefry normal took ~50 s to
    compile per shape, an rbg one ~2 s.
    """
    import jax
    import jax.numpy as jnp

    sampler = normal_sampler(tuple(shape), jnp.dtype(dtype).name)
    return jax.block_until_ready(sampler(jax.random.key(seed, impl="rbg")))


def page_table(
    rng: Any, tokens_per_seq: list[int], stride: int
) -> tuple[Any, int]:
    """A (num_seqs, stride) block table over a shuffled pool of exactly the
    pages each sequence's tokens need, and that pool's page count; a slot past
    a sequence's own pages repeats its last page, as a server's padded table
    does."""
    import numpy as np

    own_pages = [-(-tokens // PAGE_SIZE) for tokens in tokens_per_seq]
    if max(own_pages) > stride:
        raise ValueError(f"{own_pages=} exceed the table stride {stride}.")
    num_pages = sum(own_pages)
    pool = rng.permutation(num_pages).astype(np.int32)
    table = np.empty((len(own_pages), stride), np.int32)
    start = 0
    for row, count in enumerate(own_pages):
        table[row, :count] = pool[start : start + count]
        table[row, count:] = pool[start + count - 1]
        start += count
    return table, num_pages


def stateless(call: Callable[..., Any], *operands: Any) -> tuple[Step, None]:
    """A time_call step for a call with no state to carry."""
    return (lambda state: (call(*operands), state)), None


def build_dense_fwd(p: dict[str, Any]) -> tuple[Step, Any]:
    import jax
    import jax.numpy as jnp

    from flywheel_tpu import flash_attn_func

    if p["token_major"]:
        shapes = [
            (p["batch"], p["seq"], heads * p["head_dim"])
            for heads in (p["heads"], p["heads_k"], p["heads_k"])
        ]
    else:
        shapes = [
            (p["batch"], heads, p["seq"], p["head_dim"])
            for heads in (p["heads"], p["heads_k"], p["heads_k"])
        ]
    q, k, v = (
        device_normal(seed, shape, jnp.bfloat16)
        for seed, shape in enumerate(shapes)
    )
    window = p["window"]
    window_size = (-1, -1) if window is None else (window - 1, 0)
    call = jax.jit(
        functools.partial(
            flash_attn_func,
            causal=True,
            window_size=window_size,
            token_major=p["token_major"],
            head_dim=p["head_dim"] if p["token_major"] else None,
        )
    )
    return stateless(call, q, k, v)


def build_dense_bwd(p: dict[str, Any]) -> tuple[Step, Any]:
    """The backward kernel alone: the forward runs once, outside the clock,
    and the timed call applies its VJP to a fixed cotangent."""
    import jax
    import jax.numpy as jnp

    from flywheel_tpu import flash_attn_func

    q, k, v, dout = (
        device_normal(seed, (p["batch"], heads, p["seq"], p["head_dim"]), dtype)
        for seed, heads, dtype in (
            (0, p["heads"], jnp.bfloat16),
            (1, p["heads_k"], jnp.bfloat16),
            (2, p["heads_k"], jnp.bfloat16),
            (3, p["heads"], jnp.bfloat16),
        )
    )
    # jax.vjp returns a pytree Partial whose leaves are the residuals (q, k,
    # v, out, lse), so it leaves one jit and enters the next as an operand.
    vjp_fn = jax.jit(
        lambda q, k, v: jax.vjp(
            functools.partial(flash_attn_func, causal=True), q, k, v
        )[1]
    )(q, k, v)
    call = jax.jit(lambda vjp_fn, dout: vjp_fn(dout))
    return stateless(call, vjp_fn, dout)


def build_varlen_fwd(p: dict[str, Any]) -> tuple[Step, Any]:
    import jax
    import jax.numpy as jnp
    import numpy as np

    from flywheel_tpu import flash_attn_varlen_func

    lengths = p["lengths"]
    total = sum(lengths)
    max_seqlen = max(lengths)
    q, k, v = (
        device_normal(seed, (heads, total, p["head_dim"]), jnp.bfloat16)
        for seed, heads in enumerate((p["heads"], p["heads_k"], p["heads_k"]))
    )
    cu_seqlens = jnp.asarray(np.cumsum((0, *lengths)), jnp.int32)

    def call(q, k, v, cu_seqlens):
        return flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            causal=True,
        )

    return stateless(jax.jit(call), q, k, v, cu_seqlens)


def build_paged_prefill(p: dict[str, Any]) -> tuple[Step, Any]:
    """flash_attn_varlen_func's read-only paged kernel over a cache that
    already holds every prompt's K/V, as varlen_prefill.py times it."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from flywheel_tpu import flash_attn_varlen_func

    lengths = p["lengths"]
    max_seqlen = max(lengths)
    rng = np.random.default_rng(0)
    # A server sized for the longest prompt's power-of-two bucket allocates
    # every row that many page slots.
    stride = 1 << (max_seqlen - 1).bit_length()
    table, num_pages = page_table(rng, lengths, stride // PAGE_SIZE)
    kv_dtype = jnp.dtype(p["kv_dtype"])
    cache = device_normal(
        1,
        (
            num_pages,
            PAGE_SIZE,
            merged_cache_heads(p["heads_k"], p["kv_dtype"]),
            p["head_dim"],
        ),
        kv_dtype,
    )
    q = device_normal(
        0, (p["heads"], sum(lengths), p["head_dim"]), jnp.bfloat16
    )
    cu_seqlens = jnp.asarray(np.cumsum((0, *lengths)), jnp.int32)
    seqused_k = jnp.asarray(lengths, jnp.int32)
    descale = jnp.float32(1.0)

    def call(q, cache, cu_seqlens, table, seqused_k, k_descale, v_descale):
        return flash_attn_varlen_func(
            q,
            cache,
            None,
            cu_seqlens,
            None,
            max_seqlen,
            max_seqlen,
            causal=True,
            block_table=table,
            seqused_k=seqused_k,
            k_descale=k_descale if p["kv_dtype"] != "bfloat16" else None,
            v_descale=v_descale if p["kv_dtype"] != "bfloat16" else None,
            nheads_k=p["heads_k"],
        )

    return stateless(
        jax.jit(call),
        q,
        cache,
        cu_seqlens,
        jnp.asarray(table),
        seqused_k,
        descale,
        descale,
    )


def build_kvcache(p: dict[str, Any]) -> tuple[Step, Any]:
    """flash_attn_with_kvcache over the paged merged cache, appending each
    request's new tokens; every call rewrites the same slots, so each attends
    the same keys.

    query_tokens 1 is single-token decode, a larger count a speculative
    verify window (folded into the decode kernel), and requests given as
    (context, new) pairs a packed chunked prefill (append plus the paged
    varlen kernel).
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    from flywheel_tpu import flash_attn_with_kvcache

    rng = np.random.default_rng(0)
    is_packed = p["requests"] is not None
    if is_packed:
        contexts = [context for context, _ in p["requests"]]
        new_tokens = [new for _, new in p["requests"]]
        stride = -(-max(c + n for c, n in p["requests"]) // PAGE_SIZE)
    else:
        low, high = p["context"]
        contexts = rng.integers(low, high, p["batch"]).tolist()
        new_tokens = [p["query_tokens"]] * p["batch"]
        stride = -(-(high - 1 + p["query_tokens"]) // PAGE_SIZE)
    table, num_pages = page_table(
        rng, [c + n for c, n in zip(contexts, new_tokens)], stride
    )
    # The pool holds whole rows of max context, as a server's does, so pages
    # past a row's tokens stay allocated but unread.
    num_pages = max(num_pages, len(contexts) * stride)
    kv_dtype = jnp.dtype(p["kv_dtype"])
    cache = device_normal(
        3,
        (
            num_pages,
            PAGE_SIZE,
            merged_cache_heads(p["heads_k"], p["kv_dtype"]),
            p["head_dim"],
        ),
        kv_dtype,
    )
    if is_packed:
        token_shape = (sum(new_tokens),)
    else:
        token_shape = (p["batch"], p["query_tokens"])
    q = device_normal(
        0, (*token_shape, p["heads"], p["head_dim"]), jnp.bfloat16
    )
    k, v = (
        device_normal(
            seed, (*token_shape, p["heads_k"], p["head_dim"]), kv_dtype
        )
        for seed in (1, 2)
    )
    is_fp8 = p["kv_dtype"] != "bfloat16"
    packed_kwargs = {}
    if is_packed:
        packed_kwargs = {
            "cu_seqlens_q": jnp.asarray(np.cumsum((0, *new_tokens)), jnp.int32),
            "max_seqlen_q": max(new_tokens),
        }

    def call(cache, q, k, v, cache_seqlens, table, descale):
        return flash_attn_with_kvcache(
            q,
            cache,
            None,
            k,
            v,
            cache_seqlens=cache_seqlens,
            block_table=table,
            causal=True,
            k_descale=descale if is_fp8 else None,
            v_descale=descale if is_fp8 else None,
            nheads_k=p["heads_k"],
            **packed_kwargs,
        )

    kernel = jax.jit(call, donate_argnums=(0,))
    operands = (
        q,
        k,
        v,
        jnp.asarray(contexts, jnp.int32),
        jnp.asarray(table),
        jnp.float32(1.0),
    )
    return (lambda cache: kernel(cache, *operands)), cache


def build_decode_contiguous(p: dict[str, Any]) -> tuple[Step, Any]:
    """Single-token decode against FlashAttention's contiguous K/V pair."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from flywheel_tpu import flash_attn_with_kvcache

    low, high = p["context"]
    rng = np.random.default_rng(0)
    contexts = jnp.asarray(rng.integers(low, high, p["batch"]), jnp.int32)
    capacity = round_up(high, PAGE_SIZE)
    cache_shape = (p["batch"], capacity, p["heads_k"], p["head_dim"])
    caches = tuple(
        device_normal(seed, cache_shape, jnp.bfloat16) for seed in (3, 4)
    )
    q = device_normal(
        0, (p["batch"], 1, p["heads"], p["head_dim"]), jnp.bfloat16
    )
    k, v = (
        device_normal(
            seed, (p["batch"], 1, p["heads_k"], p["head_dim"]), jnp.bfloat16
        )
        for seed in (1, 2)
    )

    def call(caches, q, k, v, cache_seqlens):
        out, k_cache, v_cache = flash_attn_with_kvcache(
            q, *caches, k, v, cache_seqlens=cache_seqlens, causal=True
        )
        return out, (k_cache, v_cache)

    kernel = jax.jit(call, donate_argnums=(0,))
    return (lambda caches: kernel(caches, q, k, v, contexts)), caches


def kv_write_slices(requests: tuple[tuple[int, int], ...], table: Any) -> Any:
    """The i32[3, n] (cache_row_start, new_row_start, length) pieces that
    copy each request's new rows into its pages, each piece inside one page,
    as tpu-inference's build_kv_write_slices lays them out."""
    import numpy as np

    pieces = []
    new_row = 0
    for row, (context, new) in enumerate(requests):
        position = context
        while position < context + new:
            page_slot, offset = divmod(position, PAGE_SIZE)
            length = min(PAGE_SIZE - offset, context + new - position)
            pieces.append((
                int(table[row, page_slot]) * PAGE_SIZE + offset,
                new_row + position - context,
                length,
            ))
            position += length
        new_row += new
    return np.asarray(pieces, np.int32).T.copy()


def build_kv_cache_write(p: dict[str, Any]) -> tuple[Step, Any]:
    import jax.numpy as jnp
    import numpy as np

    write_kv_cache_pages = importlib.import_module(
        KV_WRITE_MODULE
    ).write_kv_cache_pages
    requests = p["requests"]
    rng = np.random.default_rng(0)
    stride = -(-max(c + n for c, n in requests) // PAGE_SIZE)
    table, _ = page_table(rng, [c + n for c, n in requests], stride)
    num_pages = len(requests) * stride
    kv_dtype = jnp.dtype(p["kv_dtype"])
    cache = device_normal(
        3,
        (
            num_pages,
            PAGE_SIZE,
            merged_cache_heads(p["heads_k"], p["kv_dtype"]),
            p["head_dim"],
        ),
        kv_dtype,
    )
    total = sum(new for _, new in requests)
    k, v = (
        device_normal(seed, (total, p["heads_k"], p["head_dim"]), kv_dtype)
        for seed in (1, 2)
    )
    slices = kv_write_slices(requests, table)
    slices_array = jnp.asarray(slices)
    num_slices = jnp.asarray([slices.shape[1]], jnp.int32)

    def step(cache):
        cache = write_kv_cache_pages(k, v, cache, slices_array, num_slices)
        return cache, cache

    return step, cache


def linear_inputs(
    p: dict[str, Any], query_lens: list[int], gate_width: int
) -> dict[str, Any]:
    """The per-token activations and weights shared by GDN and KDA calls.

    gate_width is the decay input's width per token, and dt_bias's width:
    n_v for GDN's a, n_v * d_k for KDA's per-key-channel g.
    """
    import jax.numpy as jnp
    import numpy as np

    dim = p["n_kq"] * p["d_k"] * 2 + p["n_v"] * p["d_v"]
    num_tokens = sum(query_lens)
    return {
        "qkv": device_normal(10, (num_tokens, dim), jnp.bfloat16),
        "b": device_normal(11, (num_tokens, p["n_v"]), jnp.bfloat16),
        "gate": device_normal(12, (num_tokens, gate_width), jnp.bfloat16),
        "conv_weight": (
            device_normal(13, (dim, 1, p["kernel_size"]), jnp.float32)
            * LINEAR_PARAM_SCALE
        ),
        "conv_bias": None,
        "a_log": (
            device_normal(14, (p["n_v"],), jnp.float32) * LINEAR_PARAM_SCALE
        ),
        "dt_bias": (
            device_normal(15, (gate_width,), jnp.float32) * LINEAR_PARAM_SCALE
        ),
        "query_start_loc": jnp.asarray(np.cumsum((0, *query_lens)), jnp.int32),
    }


def build_linear_prefill(p: dict[str, Any]) -> tuple[Step, Any]:
    """A fresh packed prefill (seq_lens == query lengths, zero initial state)
    through fused_conv1d_gdn / fused_conv1d_kda, the donated conv and
    recurrent state chained from call to call."""
    import jax.numpy as jnp

    from flywheel_tpu import linear_attention

    is_kda = p["kernel"] == "kda"
    lengths = list(p["lengths"])
    num_seqs = len(lengths)
    gate_width = p["n_v"] * p["d_k"] if is_kda else p["n_v"]
    inputs = linear_inputs(p, lengths, gate_width)
    dim = inputs["qkv"].shape[1]
    # Slot 0 is the null block; every sequence is a prefill.
    state_indices = jnp.arange(1, num_seqs + 1, dtype=jnp.int32)
    metadata = (
        inputs["query_start_loc"],
        state_indices,
        jnp.asarray([0, num_seqs, num_seqs], jnp.int32),
        jnp.asarray(lengths, jnp.int32),
        state_indices,
    )
    state = (
        jnp.zeros((num_seqs + 1, p["kernel_size"] - 1, dim), jnp.float32),
        jnp.zeros((num_seqs + 1, p["n_v"], p["d_k"], p["d_v"]), jnp.float32),
    )
    kernel = (
        linear_attention.fused_conv1d_kda
        if is_kda
        else linear_attention.fused_conv1d_gdn
    )
    dims = {
        name: p[name] for name in ("n_kq", "n_v", "d_k", "d_v", "kernel_size")
    }

    def step(state):
        state, out = kernel(
            inputs["qkv"],
            inputs["b"],
            inputs["gate"],
            *state,
            inputs["conv_weight"],
            inputs["conv_bias"],
            inputs["a_log"],
            inputs["dt_bias"],
            *metadata,
            **dims,
        )
        return out, state

    return step, state


def build_linear_replay(p: dict[str, Any]) -> tuple[Step, Any]:
    """BATCHED speculative-verify decode through fused_conv1d_*_replay, as
    bench_replay_decode.py runs it, at a fixed steady state: every window was
    fully accepted, so a row's committed records are a multiple of the window
    and the rows holding replay_window records flush this step. Every call
    sees the same replay_lens and does the same work."""
    import jax.numpy as jnp

    from flywheel_tpu import linear_attention
    from flywheel_tpu.linear_attention.gdn import wrapper as gdn_wrapper
    from flywheel_tpu.linear_attention.kda import wrapper as kda_wrapper

    is_kda = p["kernel"] == "kda"
    wrapper = kda_wrapper if is_kda else gdn_wrapper
    num_seqs = p["batch"]
    window = p["num_spec_tokens"] + 1
    replay_window = p["replay_window"]
    if replay_window % window:
        raise ValueError(f"{replay_window=} must be a multiple of {window=}.")
    gate_width = p["n_v"] * p["d_k"] if is_kda else p["n_v"]
    inputs = linear_inputs(p, [window] * num_seqs, gate_width)
    dim = inputs["qkv"].shape[1]
    # Each sequence owns window consecutive slots after the null block 0, the
    # layout the speculative path checkpoints into.
    num_slots = 1 + num_seqs * window
    state_indices = 1 + jnp.arange(num_seqs, dtype=jnp.int32) * window
    commits = replay_window // window
    replay_lens = window * (1 + jnp.arange(num_seqs, dtype=jnp.int32) % commits)
    metadata = (
        inputs["query_start_loc"],
        state_indices,
        jnp.asarray([num_seqs] * 3, jnp.int32),
        jnp.full((num_seqs,), p["context"] + window, jnp.int32),
        state_indices,
        replay_lens,
        jnp.full((num_seqs,), window - 1, jnp.int32),
    )
    layout = wrapper.replay_layout(p["n_kq"], p["n_v"], p["d_k"], p["d_v"])
    state = (
        device_normal(16, (num_slots, p["kernel_size"] - 1, dim), jnp.float32),
        wrapper.split_checkpoint(
            device_normal(
                17, (num_slots, p["n_v"], p["d_k"], p["d_v"]), jnp.float32
            )
            * LINEAR_PARAM_SCALE
        ),
        jnp.zeros(
            (num_slots, replay_window, layout.record_dim),
            jnp.dtype(p["record_dtype"]),
        ),
        jnp.zeros((num_slots, replay_window, layout.gate_dim), jnp.float32),
    )
    kernel = (
        linear_attention.fused_conv1d_kda_replay
        if is_kda
        else linear_attention.fused_conv1d_gdn_replay
    )
    dims = {
        name: p[name] for name in ("n_kq", "n_v", "d_k", "d_v", "kernel_size")
    }

    def step(state):
        state, out = kernel(
            inputs["qkv"],
            inputs["b"],
            inputs["gate"],
            *state,
            inputs["conv_weight"],
            inputs["conv_bias"],
            inputs["a_log"],
            inputs["dt_bias"],
            *metadata,
            num_spec_tokens=p["num_spec_tokens"],
            replay_window=replay_window,
            **dims,
        )
        return out, state

    return step, state


ATTN = {"heads": HEADS, "heads_k": HEADS_K, "head_dim": HEAD_DIM}
DENSE = {**ATTN, "batch": 1, "seq": DENSE_SEQ}
DECODE = {**ATTN, "batch": DECODE_BATCH, "context": DECODE_CONTEXT}
CONFIGS = (
    Config(
        "dense_fwd",
        "flash_attn_func: causal head-major forward",
        {**DENSE, "window": None, "token_major": False},
        build_dense_fwd,
    ),
    Config(
        "dense_fwd_token_major",
        "flash_attn_func: causal token-major forward",
        {**DENSE, "window": None, "token_major": True},
        build_dense_fwd,
    ),
    Config(
        "dense_fwd_window",
        "flash_attn_func: causal sliding window, the static band copies",
        {**DENSE, "window": 1024, "token_major": False},
        build_dense_fwd,
    ),
    Config(
        "dense_bwd",
        "flash_attn_func: causal backward (VJP only)",
        DENSE,
        build_dense_bwd,
    ),
    Config(
        "varlen_fwd",
        "flash_attn_varlen_func: packed K/V, causal per sequence",
        {**ATTN, "lengths": PREFILL_LENGTHS},
        build_varlen_fwd,
    ),
    Config(
        "paged_prefill_bf16",
        "flash_attn_varlen_func: paged merged bf16 cache, read-only",
        {**ATTN, "lengths": PREFILL_LENGTHS, "kv_dtype": "bfloat16"},
        build_paged_prefill,
    ),
    Config(
        "paged_prefill_fp8",
        "flash_attn_varlen_func: paged merged fp8 cache, read-only",
        {**ATTN, "lengths": PREFILL_LENGTHS, "kv_dtype": "float8_e4m3fn"},
        build_paged_prefill,
    ),
    Config(
        "decode_bf16",
        "flash_attn_with_kvcache: paged bf16 single-token decode with append",
        {**DECODE, "query_tokens": 1, "requests": None, "kv_dtype": "bfloat16"},
        build_kvcache,
    ),
    Config(
        "decode_fp8",
        "flash_attn_with_kvcache: paged fp8 single-token decode with append",
        {
            **DECODE,
            "query_tokens": 1,
            "requests": None,
            "kv_dtype": "float8_e4m3fn",
        },
        build_kvcache,
    ),
    Config(
        "decode_verify",
        "flash_attn_with_kvcache: paged bf16 verify window folded into decode",
        {
            **DECODE,
            "batch": VERIFY_BATCH,
            "query_tokens": VERIFY_TOKENS,
            "requests": None,
            "kv_dtype": "bfloat16",
        },
        build_kvcache,
    ),
    Config(
        "decode_contiguous",
        "flash_attn_with_kvcache: contiguous K/V pair single-token decode",
        DECODE,
        build_decode_contiguous,
    ),
    Config(
        "kvcache_varlen",
        "flash_attn_with_kvcache: packed chunked prefill, append + paged",
        {
            **ATTN,
            "requests": CHUNKED_PREFILL,
            "query_tokens": None,
            "kv_dtype": "bfloat16",
        },
        build_kvcache,
    ),
    Config(
        "kv_cache_write",
        "write_kv_cache_pages: bf16 prefill cache fill",
        {
            "heads_k": HEADS_K,
            "head_dim": HEAD_DIM,
            "requests": KV_WRITE_REQUESTS,
            "kv_dtype": "bfloat16",
        },
        build_kv_cache_write,
    ),
    Config(
        "gdn_prefill",
        "fused_conv1d_gdn: fresh packed prefill",
        {**GDN_DIMS, "kernel": "gdn", "lengths": LINEAR_PREFILL_LENGTHS},
        build_linear_prefill,
    ),
    Config(
        "gdn_replay_decode",
        "fused_conv1d_gdn_replay: batched verify decode",
        {
            **GDN_DIMS,
            "kernel": "gdn",
            "batch": REPLAY_BATCH,
            "num_spec_tokens": REPLAY_SPEC_TOKENS,
            "replay_window": REPLAY_WINDOW,
            "context": REPLAY_CONTEXT,
            "record_dtype": "float32",
        },
        build_linear_replay,
    ),
    Config(
        "kda_prefill",
        "fused_conv1d_kda: fresh packed prefill",
        {**KDA_DIMS, "kernel": "kda", "lengths": LINEAR_PREFILL_LENGTHS},
        build_linear_prefill,
    ),
    Config(
        "kda_replay_decode",
        "fused_conv1d_kda_replay: batched verify decode",
        {
            **KDA_DIMS,
            "kernel": "kda",
            "batch": REPLAY_BATCH,
            "num_spec_tokens": REPLAY_SPEC_TOKENS,
            "replay_window": REPLAY_WINDOW,
            "context": REPLAY_CONTEXT,
            "record_dtype": "float32",
        },
        build_linear_replay,
    ),
)
CONFIGS_BY_NAME = {config.name: config for config in CONFIGS}


def json_params(params: dict[str, Any]) -> dict[str, Any]:
    """params as they read back from JSON (tuples become lists)."""
    return json.loads(json.dumps(params))


def source_digest() -> str:
    """sha256 over the flywheel_tpu sources, path and bytes, so a run names
    the exact code it timed even when the tree is not committed."""
    digest = hashlib.sha256()
    package = REPO_ROOT / "flywheel_tpu"
    for path in sorted(package.rglob("*.py")):
        digest.update(str(path.relative_to(REPO_ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def run_metadata() -> dict[str, Any]:
    import jax

    import flywheel_tpu

    package_dir = pathlib.Path(flywheel_tpu.__file__).resolve().parent
    if package_dir != REPO_ROOT / "flywheel_tpu":
        raise RuntimeError(
            f"flywheel_tpu imports from {package_dir}, not this checkout's"
            f" {REPO_ROOT / 'flywheel_tpu'}; the gate would time other code."
        )
    return {
        "hostname": socket.gethostname(),
        "device_kind": jax.devices()[0].device_kind,
        "jax": jax.__version__,
        "jaxlib": importlib.metadata.version("jaxlib"),
        "libtpu": importlib.metadata.version("libtpu"),
        "git_head": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--", "flywheel_tpu")),
        "source_sha256": source_digest(),
    }


def check_finite(out: Any, name: str) -> None:
    import jax
    import jax.numpy as jnp

    for leaf in jax.tree_util.tree_leaves(out):
        if not bool(jnp.isfinite(leaf.astype(jnp.float32)).all()):
            raise FloatingPointError(f"{name}: non-finite output.")


def time_config(config: Config, rounds: int) -> dict[str, Any]:
    from benchmarks.common.timing import time_call

    start = time.perf_counter()
    step, state = config.build(config.params)
    # time_call keeps its last state to itself, and a donated state is dead
    # once a call consumed it, so every round resumes from the latest one.
    live = {"state": state}
    del state

    def tracked_step(state):
        out, live["state"] = step(state)
        return out, live["state"]

    round_records = []
    out = None
    for _ in range(rounds):
        timing, out = time_call(tracked_step, live["state"])
        round_records.append(timing)
    check_finite(out, config.name)
    medians = [record["median_ms"] for record in round_records]
    return {
        "path": config.path,
        "params": json_params(config.params),
        "median_ms": statistics.median(medians),
        "round_medians_ms": medians,
        "rounds": round_records,
        "wall_s": time.perf_counter() - start,
    }


def selected_configs(only: str | None) -> list[Config]:
    if only is None:
        return list(CONFIGS)
    names = [name for name in only.split(",") if name]
    unknown = [name for name in names if name not in CONFIGS_BY_NAME]
    if unknown or not names:
        raise SystemExit(
            f"--only names unknown configs {unknown}; known:"
            f" {', '.join(CONFIGS_BY_NAME)}."
        )
    return [CONFIGS_BY_NAME[name] for name in names]


def command_run(args: argparse.Namespace) -> int:
    import jax

    start = time.perf_counter()
    configs = selected_configs(args.only)
    if args.rounds < 1:
        raise SystemExit("--rounds must be at least 1.")
    if jax.default_backend() != "tpu":
        raise RuntimeError(f"requires TPU; got {jax.default_backend()!r}.")
    result = {
        "schema": SCHEMA_VERSION,
        "kind": "run",
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "note": args.note,
        **run_metadata(),
        "only": None if args.only is None else [c.name for c in configs],
        "rounds_per_config": args.rounds,
        "configs": {},
    }
    for config in configs:
        record = time_config(config, args.rounds)
        result["configs"][config.name] = record
        rounds = " ".join(f"{ms:.4f}" for ms in record["round_medians_ms"])
        print(
            f"{config.name:24s} {record['median_ms']:10.4f} ms"
            f"  rounds [{rounds}]  compile"
            f" {record['rounds'][0]['compile_s']:.1f} s  wall"
            f" {record['wall_s']:.1f} s",
            flush=True,
        )
    result["wall_s"] = time.perf_counter() - start
    print(f"total wall {result['wall_s']:.1f} s", flush=True)
    write_json(args.out, result)
    return 0


def write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path}", flush=True)


def load_json(path: pathlib.Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != SCHEMA_VERSION:
        raise SystemExit(
            f"{path}: schema {payload.get('schema')!r}, expected"
            f" {SCHEMA_VERSION}."
        )
    return payload


def check_same_env(
    first: dict[str, Any], other: dict[str, Any], label: str
) -> None:
    mismatched = {
        key: (first[key], other[key])
        for key in ENV_KEYS
        if first[key] != other[key]
    }
    if mismatched:
        raise SystemExit(
            f"{label}: runs from different machines or software stacks do not"
            f" compare: {mismatched}."
        )


def check_same_params(
    name: str, first: dict[str, Any], other: dict[str, Any], label: str
) -> None:
    if first["params"] != other["params"]:
        raise SystemExit(
            f"{label}: config {name} changed parameters,"
            f" {first['params']} vs {other['params']}; re-baseline."
        )


def spread_pct(values: list[float]) -> float:
    return (max(values) - min(values)) / min(values) * 100


def command_merge(args: argparse.Namespace) -> int:
    runs = [load_json(path) for path in args.runs]
    if len(runs) < 2:
        raise SystemExit("merge needs at least two runs.")
    first = runs[0]
    for path, run in zip(args.runs[1:], runs[1:]):
        check_same_env(first, run, str(path))
        if run["source_sha256"] != first["source_sha256"]:
            raise SystemExit(
                f"{path} timed different flywheel_tpu sources than"
                f" {args.runs[0]}."
            )
        if run["configs"].keys() != first["configs"].keys():
            raise SystemExit(f"{path} timed a different set of configs.")
    configs = {}
    for name, record in first["configs"].items():
        for path, run in zip(args.runs[1:], runs[1:]):
            check_same_params(name, record, run["configs"][name], str(path))
        medians = [run["configs"][name]["median_ms"] for run in runs]
        configs[name] = {
            "path": record["path"],
            "params": record["params"],
            "median_ms": statistics.median(medians),
            "run_medians_ms": medians,
            "spread_pct": spread_pct(medians),
        }
    merged = {
        "schema": SCHEMA_VERSION,
        "kind": "merged",
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "note": args.note,
        **{key: first[key] for key in ENV_KEYS},
        "git_head": first["git_head"],
        "git_dirty": first["git_dirty"],
        "source_sha256": first["source_sha256"],
        "only": first["only"],
        "rounds_per_config": first["rounds_per_config"],
        "runs": [
            {"path": str(path), "created_utc": run["created_utc"]}
            for path, run in zip(args.runs, runs)
        ],
        "max_spread_pct": max(c["spread_pct"] for c in configs.values()),
        "configs": configs,
    }
    print(f"{'config':24s} {'median ms':>10s} {'spread %':>9s}  runs (ms)")
    for name, record in configs.items():
        runs_ms = " ".join(f"{ms:.4f}" for ms in record["run_medians_ms"])
        print(
            f"{name:24s} {record['median_ms']:10.4f}"
            f" {record['spread_pct']:9.2f}  {runs_ms}"
        )
    write_json(args.out, merged)
    return 0


def command_compare(args: argparse.Namespace) -> int:
    baseline = load_json(args.baseline)
    candidate = load_json(args.candidate)
    check_same_env(baseline, candidate, str(args.candidate))
    unknown = candidate["configs"].keys() - baseline["configs"].keys()
    if unknown:
        raise SystemExit(f"the baseline has no {sorted(unknown)}; re-baseline.")
    missing = [
        name for name in baseline["configs"] if name not in candidate["configs"]
    ]
    if missing and candidate["only"] is None:
        raise SystemExit(f"the candidate is missing configs {missing}.")
    print(
        f"baseline  {args.baseline} ({baseline['git_head'][:12]}"
        f"{' dirty' if baseline['git_dirty'] else ''},"
        f" sources {baseline['source_sha256'][:12]})"
    )
    print(
        f"candidate {args.candidate} ({candidate['git_head'][:12]}"
        f"{' dirty' if candidate['git_dirty'] else ''},"
        f" sources {candidate['source_sha256'][:12]})"
    )
    print(
        f"host {baseline['hostname']} ({baseline['device_kind']}), rounds"
        f" {baseline['rounds_per_config']} / {candidate['rounds_per_config']},"
        f" threshold +{args.threshold:.2f}%"
    )
    header = (
        f"{'config':24s} {'base ms':>10s} {'cand ms':>10s} {'change %':>9s}"
        f" {'base spread %':>14s}  verdict"
    )
    print(header)
    print("-" * len(header))
    failed = []
    for name, base in baseline["configs"].items():
        if name not in candidate["configs"]:
            print(f"{name:24s} {'not run (--only)':>32s}")
            continue
        cand = candidate["configs"][name]
        check_same_params(name, base, cand, str(args.candidate))
        change = (cand["median_ms"] / base["median_ms"] - 1) * 100
        if change > args.threshold:
            verdict = "SLOWER"
            failed.append(name)
        elif change < -args.threshold:
            verdict = "faster"
        else:
            verdict = "ok"
        spread = base.get("spread_pct")
        spread_text = "-" if spread is None else f"{spread:.2f}"
        print(
            f"{name:24s} {base['median_ms']:10.4f} {cand['median_ms']:10.4f}"
            f" {change:+9.2f} {spread_text:>14s}  {verdict}"
        )
    if failed:
        print(
            f"FAIL: {len(failed)} config(s) slower than +{args.threshold}%:"
            f" {', '.join(failed)}"
        )
        return 1
    print("PASS")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="time the configs on this TPU")
    run.add_argument("--out", type=pathlib.Path, required=True)
    run.add_argument(
        "--only",
        help=f"comma-separated subset of: {', '.join(CONFIGS_BY_NAME)}",
    )
    run.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="time_call rounds per config; the median of their medians counts",
    )
    run.add_argument("--note", default="", help="free text kept in the JSON")
    merge = commands.add_parser(
        "merge", help="per-config median of runs from separate processes"
    )
    merge.add_argument("--runs", type=pathlib.Path, nargs="+", required=True)
    merge.add_argument("--out", type=pathlib.Path, required=True)
    merge.add_argument("--note", default="", help="free text kept in the JSON")
    compare = commands.add_parser(
        "compare", help="fail when a config slowed past the threshold"
    )
    compare.add_argument("--baseline", type=pathlib.Path, required=True)
    compare.add_argument("--candidate", type=pathlib.Path, required=True)
    compare.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="largest allowed slowdown, in percent",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "run":
        # The checkout's own flywheel_tpu and benchmarks must win over any
        # installed copy, whatever the working directory.
        sys.path.insert(0, str(REPO_ROOT))
        return command_run(args)
    elif args.command == "merge":
        return command_merge(args)
    return command_compare(args)


if __name__ == "__main__":
    sys.exit(main())
