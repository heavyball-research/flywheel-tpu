#!/usr/bin/env python3
"""Time softmax attention over a paged KV cache in two serving scenarios.

    PYTHONPATH=. uv run --no-sync python \\
        benchmarks/softmax_attention/kvcache_scenarios.py \\
        --impl flywheel --scenario chunked_prefill --heads 32 --heads-k 4 \\
        --output results/benchmark/kvcache_scenarios/benchmark.jsonl \\
        --trace results/benchmark/kvcache_scenarios/traces

chunked_prefill  One sequence of --seq tokens is prefilled in --chunk-token
                 chunks. Chunk i appends its K/V to the cache and attends
                 causally to the i chunks before it and to itself. One timed
                 call is the whole prefill, seq / chunk kernel calls.
decode           --batch sequences each hold seq - decode_tokens tokens of KV
                 cache, then run --decode-tokens decode steps: one query token
                 per sequence that appends its K/V and attends to the whole
                 cache. One timed call is all the steps.
verify           --batch sequences each hold seq - verify_tokens tokens of KV
                 cache, then run --decode-tokens speculative verify steps:
                 --verify-tokens query tokens per sequence that append their
                 K/V and attend causally (bottom-right) to the whole cache.
                 Every step rewrites the same slots, as a step whose drafts
                 are all rejected does, so each attends its whole length.

decode and verify cache batch * (seq - new tokens) tokens before the first
call, split across the sequences by --pattern (benchmarks/common/lengths.py):
uniform gives each sequence seq - new tokens, spiky and longtail hold the
same total in other shapes, so a pattern's cost over uniform is the cost of
the uneven lengths alone. Every sequence then appends the same new tokens.
The pool holds only the pages each sequence's length needs, shuffled across
sequences; a sequence's block table row runs to the longest sequence's page
count and repeats its own last page past its pages.

Both kernels read and write a paged cache of --page-size token pages (128,
the page flywheel's vLLM backend allocates), with pages shuffled across
sequences; flywheel's is the merged cache whose token rows interleave each kv
head's K and V rows. flywheel runs flash_attn_with_kvcache: its decode kernel
for decode, and for chunked prefill its ragged append plus the paged varlen
kernel. RPA v3 runs ragged_paged_attention with the request distribution
vLLM's TPU runner sends, mixed for chunked prefill and decode for decode, and
its own default block sizes unless --rpa-blocks names some.

On verify, flywheel's flash_attn_with_kvcache folds a window of at most
MAX_FOLDED_QUERY_ROWS query rows per KV head into the query heads of its
decode kernel, causal through the kernel's query_tokens, and runs a longer
one on the ragged append plus the paged varlen kernel. rpa sends verify
windows to the mixed case, as tpu-inference does (max_decode_tokens is 1).
batched_rpa runs vllm-torchtpu's batched RPA (--torchtpu-source) with every
window in its decode case and decode_query_size = --verify-tokens, as that
backend serves K + 1 verify tokens; its cache is RPA v3's, its blocks its own
default. With --verify-tokens 1 every impl runs its decode path.

flywheel_mla and mla_v2 run DeepSeek's absorbed MLA chunked prefill
(benchmarks/common/mla.py) on MLA v2's cache in its serving pages; an MLA
cell whose last output fails COS_DIFF_BOUND is recorded as check_failed.

On decode and verify they run the absorbed MLA decode: MQA over one cache row
per token, LATENT_DIM values plus ROPE_DIM of RoPE key, shared by all --heads
query heads. These cells run on MLA_DECODE_PAGE token pages for every cache
dtype, the page tpu-inference's MLA backend serves. Each impl keeps its cache
in its own layout and is given q in its own layout, both outside the clock.

mla_v2 runs mla_ragged_paged_attention as tpu-inference's mla_attention layer
calls it: on the cache layout its cache manager allocates by the
MLA_KV_PACKING_SIZE and MLA_TRANSPOSE_KV_CACHE environment variables (32
tokens per packed row and not transposed by default, read here with
tpu_inference/envs.py's defaults), with q in the cache's dtype, as
tpu-inference quantizes q for an fp8 cache unless
DISABLE_MLA_Q_ACTIVATION_QUANTIZATION is set, with q, the new rows and the
metadata padded to the next power of two of tokens and of requests, as its
model runner pads a step, with decode in its
batched_decode case and a verify window in its mixed case, and with the tuned
blocks of those cases, or its defaults when the tuned table has no entry for
the shape (block_source says which). Its wall clock includes the transposes
of q into and the output out of its kernel's token-major layout (xpose_* ops),
which kernel_ms leaves out.

flywheel_mla runs flash_mla_decode on its cache, MLA v2's word layout at the
dtype's packing, with q in bfloat16 (the only q dtype it takes), causal over
a verify window, and max_seqlen_k the longest sequence's final length, as
serving passes its bound. Its block comes from the mla| entries of
tuned_configs.json, else its heuristic (block_source says which).

Only the kernel calls are inside the clock: q, k and v are drawn once and
reused, so a step's inputs cost nothing. With --trace the scenario also runs
under jax.profiler, and every device op of one scenario is summed by name.
"""

# jax loads only where a TPU run needs it, so the host-only paths never
# claim the TPU.
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import pathlib
import sys

import numpy as np

from benchmarks.common.lengths import PATTERNS, pattern_weights, split_budget
from benchmarks.common.mla import (
    COS_DIFF_BOUND,
    DEFAULT_MLA_V2_SOURCE,
    KV_PACKING_SIZE_DEFAULT,
    LATENT_DIM,
    MLA_PREFILL_IMPLS,
    ROPE_DIM,
    SERVING_PAGE,
    SM_SCALE,
    build_mla_prefill,
    draw_normal,
    load_mla_v2,
    mla_cos_diff,
    mla_v2_blocks,
    mla_v2_padded_sizes,
    pad_mla_v2_meta,
    pad_mla_v2_tokens,
)
from benchmarks.common.records import recorded, write_record
from benchmarks.common.rpa import (
    DEFAULT_RPA_SOURCE,
    load_batched_rpa,
    load_rpa_v3,
    request_distribution,
    unjitted_kernel,
)
from benchmarks.common.timing import time_call
from benchmarks.softmax_attention import segments as segments_lib

IMPLS = ("flywheel", "rpa", "batched_rpa", *MLA_PREFILL_IMPLS)
SCENARIOS = ("chunked_prefill", "decode", "verify")
# Note (david): MLA v2's kernel is named MLA-<case>-..., flywheel's MLA
# kernels flash_mla_*.
KERNEL_MARKERS = ("flash_attn", "RPA", "ragged_paged", "MLA-", "flash_mla")
# tpu-inference's PallasMLAttentionBackend.get_page_size, the page of MLA
# decode and verify cells for every cache dtype.
MLA_DECODE_PAGE = 1024


@dataclasses.dataclass(frozen=True)
class Cell:
    """One benchmark point: an implementation on one KV cache scenario."""

    impl: str
    scenario: str
    heads: int
    heads_k: int
    head_dim: int
    seq: int
    chunk: int
    batch: int
    decode_tokens: int
    page_size: int
    sliding_window: int | None = None
    # verify: query tokens per sequence per step (num_spec_tokens + 1).
    verify_tokens: int = 1
    # q dtype; an MLA decode or verify cell draws q in kv_dtype, which mla_v2
    # takes as drawn and flywheel_mla as bfloat16. Every byte count below
    # follows the cache's itemsize.
    dtype: str = "bfloat16"
    # k, v and cache dtype. An fp8 cell keeps the descales at 1: they only
    # scale the scores and the output, not the work timed.
    kv_dtype: str = "bfloat16"
    # decode, verify: how the cached tokens split across the sequences.
    pattern: str = "uniform"

    @property
    def is_mla(self):
        return self.impl in MLA_PREFILL_IMPLS

    @property
    def key(self):
        if self.sliding_window is None:
            window_tag = ""
        else:
            window_tag = f"-w{self.sliding_window}"
        kv_tag = "" if self.kv_dtype == "bfloat16" else f"-kv{self.kv_dtype}"
        if self.is_mla:
            heads_tag = f"h{self.heads}"
        else:
            heads_tag = f"h{self.heads}-k{self.heads_k}-d{self.head_dim}"
        shape = (
            f"{self.impl}-{self.scenario}-{heads_tag}-s{self.seq}"
            f"-p{self.page_size}{window_tag}{kv_tag}"
        )
        if self.scenario == "chunked_prefill":
            return f"{shape}-c{self.chunk}"
        pattern_tag = "" if self.pattern == "uniform" else f"-{self.pattern}"
        batch_tag = f"-b{self.batch}{pattern_tag}"
        if self.scenario == "verify":
            return (
                f"{shape}{batch_tag}-t{self.verify_tokens}"
                f"-n{self.decode_tokens}"
            )
        return f"{shape}{batch_tag}-n{self.decode_tokens}"

    @property
    def new_tokens(self):
        """Tokens each sequence appends over the scenario."""
        if self.scenario == "chunked_prefill":
            return self.seq
        elif self.scenario == "verify":
            return self.verify_tokens
        return self.decode_tokens

    @property
    def contexts(self):
        """Cached tokens of each sequence before the first call, (num_seqs,):
        the pattern's split of num_seqs * (seq - new_tokens)."""
        context = self.seq - self.new_tokens
        if self.pattern == "uniform":
            return np.full(self.num_seqs, context, np.int64)
        weights = pattern_weights(self.pattern, self.num_seqs)
        return np.asarray(split_budget(weights, self.num_seqs * context))

    @property
    def seq_lens(self):
        """Each sequence's tokens after the last call; they sum to
        num_seqs * seq whatever the pattern."""
        return self.contexts + self.new_tokens

    @property
    def num_pages(self):
        """Pages in the pool: each sequence's own, and no more."""
        return int((-(-self.seq_lens // self.page_size)).sum())

    @property
    def num_calls(self):
        if self.scenario == "chunked_prefill":
            return self.seq // self.chunk
        return self.decode_tokens

    @property
    def num_seqs(self):
        return 1 if self.scenario == "chunked_prefill" else self.batch

    @property
    def query_tokens(self):
        """Query tokens of one kernel call."""
        if self.scenario == "chunked_prefill":
            return self.chunk
        elif self.scenario == "verify":
            return self.batch * self.verify_tokens
        return self.batch

    @property
    def itemsize(self):
        import jax.numpy as jnp

        return jnp.dtype(self.kv_dtype).itemsize

    @property
    def kv_token_bytes(self):
        """Cache bytes of one token: K and V of every KV head, or MLA's one
        cache row."""
        if self.is_mla:
            from flywheel_tpu.softmax_attention.kernels.mla_decode import (
                CACHE_ROW_DIM,
            )

            return CACHE_ROW_DIM * self.itemsize
        else:
            return 2 * self.heads_k * self.head_dim * self.itemsize

    @property
    def cache_bytes(self):
        return self.num_seqs * self.seq * self.kv_token_bytes


def attention_pairs(cell):
    """(query, key) pairs the whole scenario attends."""
    contexts = cell.contexts
    if cell.scenario == "verify":
        # Note (david): window token t sees the context and t + 1 new tokens.
        tokens = cell.verify_tokens
        per_step = tokens * contexts + tokens * (tokens + 1) // 2
        return cell.num_calls * int(per_step.sum())
    # (num_seqs, new_tokens): the keys each query token sees.
    causal_keys = contexts[:, None] + np.arange(
        1, cell.new_tokens + 1, dtype=np.int64
    )
    if cell.sliding_window is None:
        visible_keys = causal_keys
    else:
        visible_keys = np.minimum(causal_keys, cell.sliding_window)
    return int(visible_keys.sum())


def draw(generator, dtype, *shape):
    """Unit-normal values drawn on the host, in dtype."""
    import jax.numpy as jnp

    block = generator.standard_normal(shape, dtype=np.float32)
    return jnp.asarray(block.astype(jnp.dtype(dtype)))


def make_inputs(cell):
    """Per-call q, k, v (MLA decode and verify: q and the new cache rows, in
    two parts each) and the block table, shared by every impl."""
    import jax
    import jax.numpy as jnp

    generator = np.random.default_rng(cell.heads * 1000 + cell.heads_k)
    if cell.scenario == "chunked_prefill":
        q = draw(generator, cell.dtype, cell.seq, cell.heads, cell.head_dim)
        k = draw(
            generator, cell.kv_dtype, cell.seq, cell.heads_k, cell.head_dim
        )
        v = draw(
            generator, cell.kv_dtype, cell.seq, cell.heads_k, cell.head_dim
        )
        calls = [
            (
                q[i * cell.chunk : (i + 1) * cell.chunk],
                k[i * cell.chunk : (i + 1) * cell.chunk],
                v[i * cell.chunk : (i + 1) * cell.chunk],
            )
            for i in range(cell.num_calls)
        ]
    elif cell.is_mla:
        # Note: (ql_nope, q_pe, kv_c, k_pe), as tpu-inference's MLA layer
        # feeds its kernel: ql_nope, the absorbed q, head-major as the q_nope
        # projection emits it, q_pe token-major, and the new cache rows split
        # at LATENT_DIM.
        tokens = cell.query_tokens
        step = (
            draw(generator, cell.dtype, cell.heads, tokens, LATENT_DIM),
            draw(generator, cell.dtype, tokens, cell.heads, ROPE_DIM),
            draw(generator, cell.kv_dtype, tokens, LATENT_DIM),
            draw(generator, cell.kv_dtype, tokens, ROPE_DIM),
        )
        calls = [step] * cell.num_calls
    elif cell.scenario == "verify":

        def window_tokens(heads, dtype):
            return draw(
                generator, dtype, cell.query_tokens, heads, cell.head_dim
            )

        step = (
            window_tokens(cell.heads, cell.dtype),
            window_tokens(cell.heads_k, cell.kv_dtype),
            window_tokens(cell.heads_k, cell.kv_dtype),
        )
        calls = [step] * cell.num_calls
    else:

        def one_token(heads, dtype):
            return draw(generator, dtype, cell.batch, heads, cell.head_dim)

        step = (
            one_token(cell.heads, cell.dtype),
            one_token(cell.heads_k, cell.kv_dtype),
            one_token(cell.heads_k, cell.kv_dtype),
        )
        calls = [step] * cell.num_calls
    # Note: with uniform lengths this is a shuffled pool reshaped to
    # (num_seqs, pages_per_seq), the same table as before patterns existed.
    own_pages = -(-cell.seq_lens // cell.page_size)
    pool = generator.permutation(cell.num_pages).astype(np.int32)
    table = np.empty((cell.num_seqs, int(own_pages.max())), np.int32)
    first = 0
    for index, count in enumerate(own_pages):
        table[index, :count] = pool[first : first + count]
        table[index, count:] = pool[first + count - 1]
        first += count
    block_table = jnp.asarray(table)
    # Cached tokens per sequence before each call.
    if cell.scenario == "chunked_prefill":
        advance = cell.chunk
    elif cell.scenario == "verify":
        advance = 0
    else:
        advance = 1
    contexts = cell.contexts
    starts = [
        jnp.asarray((contexts + i * advance).astype(np.int32))
        for i in range(cell.num_calls)
    ]
    return jax.block_until_ready((calls, block_table, starts))


def fill_cache(shape, dtype, generator):
    """A cache of the given shape, tiled from random pages on device.

    Drawing the whole cache on the host does not fit at decode sizes, and the
    timing does not depend on the values.
    """
    import jax
    import jax.numpy as jnp

    base_pages = math.gcd(shape[0], 64)
    base = draw(generator, dtype, base_pages, *shape[1:])
    tile = jax.jit(
        lambda base: jnp.tile(
            base, (shape[0] // base_pages,) + (1,) * (len(shape) - 1)
        )
    )
    return jax.block_until_ready(tile(base))


def build_mla(cell, args):
    """(step, fresh_state, info, check) running an MLA prefill path on the
    chunked prefill; check(out) is the last call's cos_diff."""
    import jax
    import jax.numpy as jnp

    from flywheel_tpu.softmax_attention.mla_decode import mla_cache_shape

    keys = jax.random.split(jax.random.key(cell.heads * 1000 + cell.seq), 4)
    q_latent = draw_normal(
        keys[0], (cell.chunk, cell.heads, LATENT_DIM), jnp.bfloat16
    )
    q_pe = draw_normal(
        keys[1], (cell.chunk, cell.heads, ROPE_DIM), jnp.bfloat16
    )
    new_kv_c = draw_normal(keys[2], (cell.seq, LATENT_DIM), cell.kv_dtype)
    new_k_pe = draw_normal(keys[3], (cell.seq, ROPE_DIM), cell.kv_dtype)
    pages_per_seq = -(-cell.seq // cell.page_size)
    table = np.random.default_rng(cell.heads).permutation(pages_per_seq)
    table = table.astype(np.int32).reshape(1, pages_per_seq)
    cache_shape = mla_cache_shape(
        pages_per_seq, cell.page_size, jnp.dtype(cell.kv_dtype)
    )
    path = build_mla_prefill(
        cell.impl,
        heads=cell.heads,
        kv_dtype=cell.kv_dtype,
        cache_shape=cache_shape,
        num_tokens=cell.chunk,
        batch=1,
        pages_per_seq=pages_per_seq,
        max_seqlen_q=cell.chunk,
        max_seqlen_k=cell.seq,
        mla_v2_source=args.mla_v2_source,
    )
    queries = path.queries(q_latent, q_pe)
    chunks = [
        slice(i * cell.chunk, (i + 1) * cell.chunk)
        for i in range(cell.num_calls)
    ]
    tokens = [path.tokens(new_kv_c[chunk], new_k_pe[chunk]) for chunk in chunks]
    cu_q_lens = np.array([0, cell.chunk], np.int32)
    metas = [
        path.meta(cu_q_lens, np.array([chunk.stop], np.int32), table)
        for chunk in chunks
    ]
    jax.block_until_ready((queries, tokens, metas))
    kernel = jax.jit(path.attend, donate_argnums=(0,))

    def _step(cache):
        out = None
        for call_tokens, meta in zip(tokens, metas):
            out, cache = kernel(cache, *queries, *call_tokens, *meta)
        return out, cache

    return (
        _step,
        (lambda: jnp.zeros(cache_shape, cell.kv_dtype)),
        path.info,
        lambda out: mla_cos_diff(
            path.token_major(out),
            q_latent,
            q_pe,
            new_kv_c,
            new_k_pe,
            cell.kv_dtype,
        ),
    )


def build_flywheel(cell, inputs, args):
    """(step, fresh_state, info) running flash_attn_with_kvcache."""
    del args  # Unused.
    import jax

    from flywheel_tpu import flash_attn_with_kvcache
    from flywheel_tpu.softmax_attention.kv_layout import merged_cache_heads

    calls, block_table, starts = inputs
    # Note (david): the merged cache interleaves each kv head's K row and V
    # row (an fp8 cache pads an odd head count).
    shape = (
        cell.num_pages,
        cell.page_size,
        merged_cache_heads(cell.heads_k, cell.kv_dtype),
        cell.head_dim,
    )
    if cell.sliding_window is None:
        window_size = (-1, -1)
    else:
        window_size = (cell.sliding_window - 1, 0)

    def call(cache, q, k, v, start):
        if cell.scenario == "chunked_prefill":
            q, k, v = q[None], k[None], v[None]
        elif cell.scenario == "verify":
            q, k, v = (
                x.reshape(cell.batch, cell.verify_tokens, *x.shape[1:])
                for x in (q, k, v)
            )
        else:
            q, k, v = q[:, None], k[:, None], v[:, None]
        out, cache = flash_attn_with_kvcache(
            q,
            cache,
            None,
            k,
            v,
            cache_seqlens=start,
            block_table=block_table,
            causal=True,
            window_size=window_size,
            nheads_k=cell.heads_k,
        )
        return out, cache

    kernel = jax.jit(call, donate_argnums=(0,))

    def step(cache):
        out = None
        for (q, k, v), start in zip(calls, starts):
            out, cache = kernel(cache, q, k, v, start)
        return out, cache

    generator = np.random.default_rng(1)
    return step, (lambda: fill_cache(shape, cell.kv_dtype, generator)), {}


def build_rpa(cell, inputs, args):
    """(step, fresh_state, info) running ragged_paged_attention."""
    import jax
    import jax.numpy as jnp

    rpa = load_rpa_v3(args.rpa_source)
    calls, block_table, starts = inputs
    shape = rpa.get_kv_cache_shape(
        cell.num_pages,
        cell.page_size,
        cell.heads_k,
        cell.head_dim,
        jnp.dtype(cell.kv_dtype),
    )
    page_indices = block_table.reshape(-1)
    if cell.scenario == "chunked_prefill":
        cu_q_lens = jnp.array([0, cell.chunk], jnp.int32)
        distribution = request_distribution(num_decode=0, num_seqs=1)
        new_tokens = cell.chunk
        blocks_arg = "m_block_sizes"
    elif cell.scenario == "verify" and cell.verify_tokens > 1:
        # Note (david): tpu-inference's runner counts a request as decode only
        # at num_scheduled_tokens <= max_decode_tokens, which is 1, so a
        # verify window is mixed.
        cu_q_lens = (
            jnp.arange(cell.batch + 1, dtype=jnp.int32) * cell.verify_tokens
        )
        distribution = request_distribution(num_decode=0, num_seqs=cell.batch)
        new_tokens = cell.verify_tokens
        blocks_arg = "m_block_sizes"
    else:
        cu_q_lens = jnp.arange(cell.batch + 1, dtype=jnp.int32)
        distribution = request_distribution(
            num_decode=cell.batch, num_seqs=cell.batch
        )
        new_tokens = 1
        blocks_arg = "d_block_sizes"
    blocks = (
        tuple(int(b) for b in args.rpa_blocks.split(","))
        if args.rpa_blocks
        else None
    )
    # A decode call compiles RPA's mixed kernel too, whose default blocks can
    # run out of VMEM even when no request takes that path.
    mixed_blocks = (
        tuple(int(b) for b in args.rpa_mixed_blocks.split(","))
        if args.rpa_mixed_blocks
        else None
    )
    block_kwargs = {blocks_arg: blocks}
    if blocks_arg == "d_block_sizes":
        block_kwargs["m_block_sizes"] = mixed_blocks
    kernel_fn = unjitted_kernel(rpa)
    scale = 1.0 / math.sqrt(cell.head_dim)

    def call(cache, q, k, v, start):
        out, cache = kernel_fn(
            q,
            k,
            v,
            cache,
            start + new_tokens,
            page_indices,
            cu_q_lens,
            distribution,
            use_causal_mask=True,
            sm_scale=scale,
            sliding_window=cell.sliding_window,
            **block_kwargs,
        )
        return out, cache

    kernel = jax.jit(call, donate_argnums=(0,))

    def step(cache):
        out = None
        for (q, k, v), start in zip(calls, starts):
            out, cache = kernel(cache, q, k, v, start)
        return out, cache

    generator = np.random.default_rng(1)
    info = {
        "blocks": list(blocks) if blocks else None,
        "block_source": "--rpa-blocks" if blocks else "rpa default",
        "block_case": blocks_arg,
        "mixed_blocks": list(mixed_blocks) if mixed_blocks else None,
    }
    return step, (lambda: fill_cache(shape, cell.kv_dtype, generator)), info


def build_batched_rpa(cell, inputs, args):
    """(step, fresh_state, info) running vllm-torchtpu's batched RPA."""
    import jax
    import jax.numpy as jnp

    if args.torchtpu_source is None:
        raise ValueError("--impl batched_rpa needs --torchtpu-source.")
    wrapper = load_batched_rpa(args.torchtpu_source)
    calls, block_table, starts = inputs
    shape = wrapper.get_kv_cache_shape(
        cell.num_pages,
        cell.page_size,
        cell.heads_k,
        cell.head_dim,
        jnp.dtype(cell.kv_dtype),
    )
    page_indices = block_table.reshape(-1)
    if cell.scenario == "chunked_prefill":
        cu_q_lens = jnp.array([0, cell.chunk], jnp.int32)
        distribution = request_distribution(num_decode=0, num_seqs=1)
        new_tokens = cell.chunk
        decode_query_size = 1
    else:
        new_tokens = cell.verify_tokens if cell.scenario == "verify" else 1
        cu_q_lens = jnp.arange(cell.batch + 1, dtype=jnp.int32) * new_tokens
        distribution = request_distribution(
            num_decode=cell.batch, num_seqs=cell.batch
        )
        decode_query_size = new_tokens
    kernel_fn = unjitted_kernel(wrapper)
    scale = 1.0 / math.sqrt(cell.head_dim)

    def call(cache, q, k, v, start):
        out, cache = kernel_fn(
            q,
            k,
            v,
            cache,
            start + new_tokens,
            page_indices,
            cu_q_lens,
            distribution,
            sm_scale=scale,
            sliding_window=cell.sliding_window,
            decode_query_size=decode_query_size,
        )
        return out, cache

    kernel = jax.jit(call, donate_argnums=(0,))

    def step(cache):
        out = None
        for (q, k, v), start in zip(calls, starts):
            out, cache = kernel(cache, q, k, v, start)
        return out, cache

    generator = np.random.default_rng(1)
    return (
        step,
        (lambda: fill_cache(shape, cell.kv_dtype, generator)),
        {
            "block_source": "batched rpa default",
            "decode_query_size": decode_query_size,
        },
    )


def build_mla_v2_decode(cell, inputs, args):
    """(step, fresh_state, info) running tpu-inference's
    mla_ragged_paged_attention on decode or verify as its mla_attention layer
    calls it."""
    import jax
    import jax.numpy as jnp

    from flywheel_tpu.core.tpu import NUM_LANES

    kernel, tuned = load_mla_v2(args.mla_v2_source)
    calls, block_table, starts = inputs
    # Note: tpu-inference's cache manager lays the MLA cache out by these env
    # vars. Rows of kv_packing tokens (KV_PACKING_SIZE_DEFAULT, not the
    # dtype's packing), each the latent and its RoPE part padded to whole
    # lane tiles apiece; transposed, the RoPE part is not padded. load_mla_v2
    # stubs tpu_inference.envs, so they are read here as envs.py reads them.
    kv_packing = int(os.getenv("MLA_KV_PACKING_SIZE", KV_PACKING_SIZE_DEFAULT))
    transpose = os.getenv("MLA_TRANSPOSE_KV_CACHE", "False").lower() in (
        "1",
        "true",
    )
    if cell.page_size % kv_packing:
        raise ValueError(
            f"--page-size {cell.page_size} is not a multiple of"
            f" MLA_KV_PACKING_SIZE={kv_packing}."
        )
    rope_dim = ROPE_DIM if transpose else kernel.align_to(ROPE_DIM, NUM_LANES)
    shape = kernel.get_kv_cache_shape(
        cell.num_pages,
        cell.page_size,
        kernel.align_to(LATENT_DIM, NUM_LANES) + rope_dim,
        jnp.dtype(cell.kv_dtype),
        kv_packing,
        transpose_kv_cache=transpose,
    )
    new_tokens = cell.verify_tokens if cell.scenario == "verify" else 1
    # As in build_rpa: tpu-inference counts only a one-token request as
    # decode, so a longer verify window runs the mixed case.
    num_decode = cell.batch if new_tokens == 1 else 0
    # Note: q, the new rows and the metadata are padded as the model runner
    # pads a step, the way benchmarks/common/mla.py's prefill path is.
    padded_tokens, max_num_seqs = mla_v2_padded_sizes(
        cell.query_tokens, cell.batch
    )
    block_kwargs, cases = mla_v2_blocks(
        tuned,
        max_num_tokens=padded_tokens,
        actual_num_q_heads=cell.heads,
        actual_lkv_dim=LATENT_DIM,
        actual_r_dim=ROPE_DIM,
        kv_dtype=cell.kv_dtype,
        q_dtype=cell.dtype,
        page_size_per_kv_packing=shape[1],
        kv_packing=shape[2],
        max_num_seqs=max_num_seqs,
        pages_per_seq=block_table.shape[1],
    )
    cu_q_lens = np.arange(cell.batch + 1, dtype=np.int32) * new_tokens
    distribution = request_distribution(
        num_decode=num_decode, num_seqs=cell.batch
    )
    metas = [
        pad_mla_v2_meta(
            cu_q_lens,
            np.asarray(start) + new_tokens,
            block_table,
            distribution,
            max_num_seqs=max_num_seqs,
        )
        for start in starts
    ]
    # Note: only kv_lens changes from call to call; the rest of the metadata
    # is the jit's constant.
    kv_lens = [meta[0] for meta in metas]
    _, page_indices, cu_q_lens, distribution = metas[0]
    # Every call reuses one step's operands, so they are padded once:
    # ql_nope is head-major, the other three token-major.
    ql_nope, q_pe, kv_c, k_pe = calls[0]
    operands = jax.block_until_ready((
        pad_mla_v2_tokens(ql_nope, padded_tokens, axis=1),
        pad_mla_v2_tokens(q_pe, padded_tokens),
        pad_mla_v2_tokens(kv_c, padded_tokens),
        pad_mla_v2_tokens(k_pe, padded_tokens),
    ))
    # Note: without its own jax.jit, whose static arguments become constants
    # of the jit below; it donates only the cache either way.
    kernel_fn = kernel.mla_ragged_paged_attention.__wrapped__

    def call(cache, ql_nope, q_pe, kv_c, k_pe, call_kv_lens):
        out, cache = kernel_fn(
            ql_nope,
            q_pe,
            kv_c,
            k_pe,
            cache,
            call_kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            sm_scale=SM_SCALE,
            transpose_kv_cache=transpose,
            **block_kwargs,
        )
        return out, cache

    kernel_jit = jax.jit(call, donate_argnums=(0,))

    def step(cache):
        out = None
        for call_kv_lens in kv_lens:
            out, cache = kernel_jit(cache, *operands, call_kv_lens)
        return out, cache

    case = "batched_decode" if num_decode else "mixed"
    params, hit = cases[case]
    generator = np.random.default_rng(1)
    info = {
        "block_source": f"tpu-inference {'tuned' if hit else 'default'}",
        "block_case": case,
        "tuned_params": dataclasses.asdict(params),
        "q_dtype": cell.dtype,
        "kv_packing": kv_packing,
        "transpose_kv_cache": transpose,
        "xpose_n_tile_size": kernel.envs.MLA_XPOSE_N_TILE_SIZE,
        "padded_tokens": padded_tokens,
        "max_num_seqs": max_num_seqs,
    }
    return step, (lambda: fill_cache(shape, cell.kv_dtype, generator)), info


def build_flywheel_mla_decode(cell, inputs, args):
    """(step, fresh_state, info) running flash_mla_decode on decode or
    verify."""
    del args  # Unused.
    import jax
    import jax.numpy as jnp

    from flywheel_tpu.softmax_attention.dispatch import get_tuned_mla_config
    from flywheel_tpu.softmax_attention.mla_decode import (
        flash_mla_decode,
        mla_cache_shape,
    )

    calls, block_table, starts = inputs
    shape = mla_cache_shape(
        cell.num_pages, cell.page_size, jnp.dtype(cell.kv_dtype)
    )
    new_tokens = cell.verify_tokens if cell.scenario == "verify" else 1
    batch, heads = cell.batch, cell.heads

    @jax.jit
    def token_major(ql_nope, q_pe, kv_c, k_pe):
        """make_inputs' MLA operands in flash_mla_decode's layout and q
        dtype: (batch, new_tokens, ...), q in bfloat16."""
        q_nope = jnp.swapaxes(ql_nope, 0, 1).astype(jnp.bfloat16)
        return (
            q_nope.reshape(batch, new_tokens, heads, LATENT_DIM),
            q_pe.astype(jnp.bfloat16).reshape(
                batch, new_tokens, heads, ROPE_DIM
            ),
            kv_c.reshape(batch, new_tokens, LATENT_DIM),
            k_pe.reshape(batch, new_tokens, ROPE_DIM),
        )

    # Note: every call reuses one step's operands, so they are relaid once.
    operands = jax.block_until_ready(token_major(*calls[0]))
    fp8 = jnp.dtype(cell.kv_dtype) == jnp.float8_e4m3fn
    # The scale only scales the scores and the output, not the work timed.
    kv_scale = jnp.float32(1.0) if fp8 else None
    max_seqlen_k = int(cell.seq_lens.max())

    def call(cache, q_nope, q_pe, kv_c, k_pe, start):
        return flash_mla_decode(
            q_nope,
            q_pe,
            kv_c,
            k_pe,
            cache,
            start + new_tokens,
            block_table,
            sm_scale=SM_SCALE,
            kv_scale=kv_scale,
            causal=True,
            s_dtype=jnp.bfloat16,
            max_seqlen_k=max_seqlen_k,
        )

    kernel_jit = jax.jit(call, donate_argnums=(0,))

    def step(cache):
        out = None
        for start in starts:
            out, cache = kernel_jit(cache, *operands, start)
        return out, cache

    tuned = get_tuned_mla_config(
        cache_dtype=cell.kv_dtype,
        num_query_rows=new_tokens * heads,
        batch=batch,
        max_seqlen_k=max_seqlen_k,
    )
    generator = np.random.default_rng(1)
    info = {
        "block_source": f"flywheel {'default' if tuned is None else 'tuned'}",
        "tuned_params": None if tuned is None else list(tuned),
        "q_dtype": "bfloat16",
        "max_seqlen_k": max_seqlen_k,
    }
    return step, (lambda: fill_cache(shape, cell.kv_dtype, generator)), info


BUILDERS = {
    "flywheel": build_flywheel,
    "rpa": build_rpa,
    "batched_rpa": build_batched_rpa,
}
# An MLA impl's builder for decode and verify; build_mla runs its prefill.
MLA_DECODE_BUILDERS = {
    "flywheel_mla": build_flywheel_mla_decode,
    "mla_v2": build_mla_v2_decode,
}


def trace_scenario(cell, step, state, directory):
    """Device ms per op name over one scenario, from a jax.profiler trace."""
    import jax

    target = directory / cell.key
    target.mkdir(parents=True, exist_ok=True)
    out, state = step(state)
    jax.block_until_ready((out, state))
    with jax.profiler.trace(str(target)):
        out, state = step(state)
        jax.block_until_ready((out, state))
    trace = segments_lib.load(target)
    if trace is None:
        return None
    pids = segments_lib.device_pids(trace)
    per_op = {}
    for event in trace["traceEvents"]:
        if (
            event.get("ph") != "X"
            or "dur" not in event
            or event.get("pid") not in pids
            or event["name"].startswith("jit_")
        ):
            continue
        name = event["name"].split(".")[0]
        per_op[name] = per_op.get(name, 0.0) + event["dur"] / 1e3
    # Counter tracks (clock state, FIFO stats) show up as zero-length events.
    return {name: ms for name, ms in per_op.items() if ms >= 1e-3}


def run(cell, args):
    import jax

    pairs = attention_pairs(cell)
    if cell.is_mla:
        # Note (david): QK over the latent and the rope, P @ V over the latent.
        attn_flops = 2 * pairs * cell.heads * (2 * LATENT_DIM + ROPE_DIM)
    else:
        attn_flops = 4 * pairs * cell.heads * cell.head_dim
    # Every call reads each sequence's whole visible cache once.
    if cell.scenario == "decode":
        kv_tokens_read = pairs
    elif cell.scenario == "verify":
        kv_tokens_read = cell.num_calls * int(cell.seq_lens.sum())
    elif cell.sliding_window is None:
        kv_tokens_read = sum(
            (i + 1) * cell.chunk for i in range(cell.num_calls)
        )
    else:
        # Note (david): call i reads from its first query's window edge,
        # sliding_window - 1 keys back, not from token 0.
        kv_tokens_read = sum(
            (i + 1) * cell.chunk
            - max(0, i * cell.chunk - cell.sliding_window + 1)
            for i in range(cell.num_calls)
        )
    kv_bytes_read = kv_tokens_read * cell.kv_token_bytes
    record = {
        "key": cell.key,
        "cell": dataclasses.asdict(cell),
        "num_calls": cell.num_calls,
        "attention_pairs": pairs,
        "attn_flops": attn_flops,
        "kv_bytes_read": kv_bytes_read,
        "cache_bytes": cell.cache_bytes,
        "seq_len_min": int(cell.seq_lens.min()),
        "seq_len_max": int(cell.seq_lens.max()),
    }
    hbm = jax.devices()[0].memory_stats().get("bytes_limit")
    if hbm and cell.cache_bytes > 0.9 * hbm:
        record.update(
            status="error",
            error=(
                f"the KV cache needs {cell.cache_bytes / 2**30:.1f} GiB, more "
                f"than 90% of the device's {hbm / 2**30:.1f} GiB of HBM"
            ),
        )
        return record
    try:
        if cell.is_mla and cell.scenario == "chunked_prefill":
            step, fresh, info, check = build_mla(cell, args)
        else:
            builders = MLA_DECODE_BUILDERS if cell.is_mla else BUILDERS
            inputs = make_inputs(cell)
            step, fresh, info = builders[cell.impl](cell, inputs, args)
            check = None
        record.update(info)
        timing, out = time_call(step, fresh())
        if check is None:
            status = "ok"
        else:
            record["cos_diff"] = check(out)
            if record["cos_diff"] <= COS_DIFF_BOUND:
                status = "ok"
            else:
                status = "check_failed"
    except Exception as error:  # pylint: disable=broad-exception-caught
        record.update(
            status="error", error=f"{type(error).__name__}: {str(error)[:400]}"
        )
        return record

    ms = timing["median_ms"]
    record.update(
        status=status,
        **timing,
        ms_per_call=ms / cell.num_calls,
        tflops=attn_flops / (ms * 1e9),
        kv_read_gbps=kv_bytes_read / (ms * 1e6),
    )
    if cell.scenario in ("decode", "verify"):
        record["tokens_per_s"] = cell.num_calls * cell.query_tokens / (ms / 1e3)
    if args.trace:
        per_op = trace_scenario(cell, step, fresh(), args.trace)
        if per_op is None:
            record["trace_error"] = "no device events in the trace"
        else:
            kernel_ms = sum(
                t
                for name, t in per_op.items()
                if any(m in name for m in KERNEL_MARKERS)
            )
            record.update(
                trace_ops_ms=dict(
                    sorted(per_op.items(), key=lambda item: -item[1])[:8]
                ),
                trace_device_ms=sum(per_op.values()),
                kernel_ms=kernel_ms,
                kernel_tflops=attn_flops / (kernel_ms * 1e9)
                if kernel_ms
                else None,
            )
    return record


def main(argv=None):
    args = parse_args(argv)
    is_mla = args.impl in MLA_PREFILL_IMPLS
    mla_decode = is_mla and args.scenario != "chunked_prefill"
    if args.page_size is not None:
        page_size = args.page_size
    elif is_mla and args.scenario == "chunked_prefill":
        page_size = SERVING_PAGE[args.kv_dtype]
    elif is_mla:
        page_size = MLA_DECODE_PAGE
    else:
        page_size = 128
    cell = Cell(
        args.impl,
        args.scenario,
        args.heads,
        # Note (david): MLA's queries attend one latent head.
        1 if is_mla else args.heads_k,
        LATENT_DIM if is_mla else args.head_dim,
        args.seq,
        args.chunk,
        args.batch,
        args.decode_tokens,
        page_size,
        args.sliding_window,
        args.verify_tokens,
        dtype=args.kv_dtype if mla_decode else "bfloat16",
        kv_dtype=args.kv_dtype,
        pattern=args.pattern,
    )
    if args.scenario == "chunked_prefill" and args.seq % args.chunk:
        raise SystemExit("--seq must be a multiple of --chunk.")
    if cell.pattern != "uniform":
        try:
            _ = cell.contexts
        except ValueError as error:
            raise SystemExit(
                f"--pattern {cell.pattern}: {error}; the batch caches"
                f" {cell.batch} * (seq - {cell.new_tokens}) tokens."
            ) from None
    if args.output and not args.trace and recorded(args.output, cell.key):
        print(f"{cell.key}: already recorded in {args.output}")
        return 0
    record = run(cell, args)
    if args.output:
        write_record(args.output, record)
    print(json.dumps(record))
    return 0 if record["status"] == "ok" else 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--impl", choices=IMPLS, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--heads-k", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument(
        "--seq", type=int, default=16384, help="total tokens per sequence"
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=1024,
        help="chunked_prefill: query tokens per prefill call",
    )
    parser.add_argument(
        "--batch", type=int, default=256, help="decode: concurrent sequences"
    )
    parser.add_argument(
        "--decode-tokens",
        type=int,
        default=1024,
        help="decode: steps, the last tokens of --seq; verify: steps",
    )
    parser.add_argument(
        "--pattern",
        choices=PATTERNS,
        default="uniform",
        help=(
            "decode, verify: how batch * (seq - new tokens) cached tokens"
            " split across the sequences (benchmarks/common/lengths.py);"
            " every pattern caches the same total"
        ),
    )
    parser.add_argument(
        "--verify-tokens",
        type=int,
        default=1,
        help="verify: query tokens per sequence per step",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        help=(
            "default: 128; for an MLA impl, MLA v2's serving page of the dtype"
            f" on chunked_prefill and {MLA_DECODE_PAGE} on decode and verify"
        ),
    )
    parser.add_argument(
        "--kv-dtype",
        default="bfloat16",
        choices=("bfloat16", "float8_e4m3fn", "float8_e5m2"),
        help=(
            "k, v and KV cache dtype; q stays bfloat16, but takes it for"
            " mla_v2 on decode and verify"
        ),
    )
    parser.add_argument(
        "--sliding-window",
        type=int,
        help=(
            "keys each query attends, its own included "
            "(RPA's sliding_window; flywheel's window_size "
            "is (W - 1, 0)); default: full causal"
        ),
    )
    parser.add_argument(
        "--rpa-blocks",
        help=(
            "bq_sz,bkv_sz,bq_csz,bkv_csz for RPA's case; "
            "default: RPA's own formula"
        ),
    )
    parser.add_argument(
        "--rpa-mixed-blocks",
        help=(
            "decode: m_block_sizes for the mixed kernel RPA "
            "compiles alongside; default: RPA's formula"
        ),
    )
    parser.add_argument(
        "--rpa-source", type=pathlib.Path, default=DEFAULT_RPA_SOURCE
    )
    parser.add_argument(
        "--torchtpu-source",
        type=pathlib.Path,
        help="batched_rpa: a vllm-torchtpu checkout",
    )
    parser.add_argument(
        "--mla-v2-source",
        type=pathlib.Path,
        default=DEFAULT_MLA_V2_SOURCE,
        help="mla_v2: a tpu-inference checkout at the pinned commit",
    )
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument(
        "--trace",
        type=pathlib.Path,
        help="also trace one scenario into this directory",
    )
    args = parser.parse_args(argv)
    is_mla = args.impl in MLA_PREFILL_IMPLS
    if args.pattern != "uniform" and args.scenario == "chunked_prefill":
        parser.error(
            "--pattern splits decode and verify batches; chunked_prefill"
            " prefills one sequence"
        )
    elif is_mla and args.sliding_window is not None:
        parser.error(f"--impl {args.impl} takes no --sliding-window")
    elif is_mla and args.kv_dtype not in SERVING_PAGE:
        parser.error(f"--impl {args.impl} takes --kv-dtype {set(SERVING_PAGE)}")
    else:
        return args


if __name__ == "__main__":
    sys.exit(main())
