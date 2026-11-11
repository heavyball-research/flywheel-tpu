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
                 are all rejected does, so each attends seq keys.

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

flywheel_dsa, torchtpu_dsa and torchtpu_dsa_masked_dense run one DeepSeek
sparse attention layer step per call (benchmarks/common/dsa.py) on fp8 caches
of DSA_PAGE pages, on decode and chunked prefill. A DSA cell whose last call
fails COS_DIFF_BOUND or INDEX_DEFICIT_BOUND is recorded as check_failed.

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
import functools
import json
import math
import pathlib
import sys

import numpy as np

from benchmarks.common.dsa import (
    DSA_BASE_PAGES,
    DSA_CACHE_TOKEN_BYTES,
    DSA_GEOMETRIES,
    DSA_IMPLS,
    DSA_KV_DTYPE,
    DSA_PAGE,
    FLYWHEEL_DSA,
    TORCHTPU_DSA_IMPLS,
    DsaKeys,
    DsaRequests,
    DsaScenario,
    build_dsa_layer,
    draw_dsa_keys,
    draw_dsa_queries,
    dsa_args_error,
    dsa_work,
    edge_rows,
    filled_caches,
    measure_dsa,
    page_index_keys,
    prepare_torchtpu_process,
)
from benchmarks.common.mla import (
    COS_DIFF_BOUND,
    DEFAULT_MLA_V2_SOURCE,
    LATENT_DIM,
    MLA_PREFILL_IMPLS,
    ROPE_DIM,
    SERVING_PAGE,
    build_mla_prefill,
    draw_normal,
    mla_cos_diff,
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

IMPLS = ("flywheel", "rpa", "batched_rpa", *MLA_PREFILL_IMPLS, *DSA_IMPLS)
SCENARIOS = ("chunked_prefill", "decode", "verify")
DSA_SCENARIOS = ("chunked_prefill", "decode")
# Note (david): MLA v2's kernel is named MLA-<case>-..., flywheel's MLA
# kernels flash_mla_*.
KERNEL_MARKERS = ("flash_attn", "RPA", "ragged_paged", "MLA-", "flash_mla")


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
    # q dtype. Every byte count below follows the cache's itemsize.
    dtype: str = "bfloat16"
    # k, v and cache dtype. An fp8 cell keeps the descales at 1: they only
    # scale the scores and the output, not the work timed.
    kv_dtype: str = "bfloat16"
    # A DSA cell's DSA_GEOMETRIES entry.
    dsa_geometry: str | None = None
    # flywheel_dsa's pinned route; None picks it on device.
    dsa_route: str | None = None

    @property
    def is_mla(self):
        return self.impl in MLA_PREFILL_IMPLS

    @property
    def is_dsa(self):
        return self.impl in DSA_IMPLS

    @property
    def key(self):
        if self.sliding_window is None:
            window_tag = ""
        else:
            window_tag = f"-w{self.sliding_window}"
        kv_tag = "" if self.kv_dtype == "bfloat16" else f"-kv{self.kv_dtype}"
        if self.is_dsa and self.dsa_route is not None:
            heads_tag = f"{self.dsa_geometry}-{self.dsa_route}"
        elif self.is_dsa:
            heads_tag = self.dsa_geometry
        elif self.is_mla:
            heads_tag = f"h{self.heads}"
        else:
            heads_tag = f"h{self.heads}-k{self.heads_k}-d{self.head_dim}"
        shape = (
            f"{self.impl}-{self.scenario}-{heads_tag}-s{self.seq}"
            f"-p{self.page_size}{window_tag}{kv_tag}"
        )
        if self.scenario == "chunked_prefill":
            return f"{shape}-c{self.chunk}"
        elif self.scenario == "verify":
            return (
                f"{shape}-b{self.batch}-t{self.verify_tokens}"
                f"-n{self.decode_tokens}"
            )
        return f"{shape}-b{self.batch}-n{self.decode_tokens}"

    @property
    def context(self):
        """Cached tokens per sequence before the first call."""
        if self.scenario == "chunked_prefill":
            return 0
        elif self.scenario == "verify":
            return self.seq - self.verify_tokens
        return self.seq - self.decode_tokens

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
        """Cache bytes of one token: K and V of every KV head, MLA's one
        cache row, or DSA's latent and index key."""
        if self.is_dsa:
            return DSA_CACHE_TOKEN_BYTES
        elif self.is_mla:
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
    if cell.scenario == "verify":
        # Note (david): window token t sees the context and t + 1 new tokens.
        tokens = cell.verify_tokens
        per_step = tokens * cell.context + tokens * (tokens + 1) // 2
        return cell.num_calls * cell.batch * per_step
    if cell.scenario == "chunked_prefill":
        first_query, num_queries = 0, cell.seq
    else:
        first_query, num_queries = cell.context, cell.decode_tokens
    causal_keys = np.arange(
        first_query + 1, first_query + num_queries + 1, dtype=np.int64
    )
    if cell.sliding_window is None:
        visible_keys = causal_keys
    else:
        visible_keys = np.minimum(causal_keys, cell.sliding_window)
    return cell.num_seqs * int(visible_keys.sum())


def draw(generator, dtype, *shape):
    """Unit-normal values drawn on the host, in dtype."""
    import jax.numpy as jnp

    block = generator.standard_normal(shape, dtype=np.float32)
    return jnp.asarray(block.astype(jnp.dtype(dtype)))


def make_inputs(cell):
    """Per-call q, k, v and the block table, shared by both impls."""
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
    pages_per_seq = -(-cell.seq // cell.page_size)
    table = generator.permutation(cell.num_seqs * pages_per_seq)
    block_table = jnp.asarray(
        table.reshape(cell.num_seqs, pages_per_seq).astype(np.int32)
    )
    # Cached tokens per sequence before each call.
    if cell.scenario == "chunked_prefill":
        advance = cell.chunk
    elif cell.scenario == "verify":
        advance = 0
    else:
        advance = 1
    starts = [
        jnp.full((cell.num_seqs,), cell.context + i * advance, jnp.int32)
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


def dsa_block_table(cell, pages_per_seq):
    """Every sequence's pages, shuffled across sequences."""
    generator = np.random.default_rng(cell.heads * 1000 + cell.batch)
    pages = generator.permutation(cell.num_seqs * pages_per_seq)
    return pages.astype(np.int32).reshape(cell.num_seqs, pages_per_seq)


def dsa_decode_logical(cell, table, base_keys, index_key, step_keys, request):
    """A decode request's keys after the steps: its pages' own index keys
    and their base pages' latents, then the steps' tokens."""
    import jax.numpy as jnp

    page = cell.page_size
    base_pages = base_keys.index_keys.shape[0] // page
    positions = np.arange(cell.context)
    rows = (table[request, positions // page] % base_pages) * page
    cached = base_keys.rows(jnp.asarray(rows + positions % page))
    index_keys = page_index_keys(
        index_key, jnp.asarray(table[request]), base_keys.index_keys.shape[1]
    )
    cached = cached._replace(index_keys=index_keys[: cell.context])
    return DsaKeys(*(
        jnp.concatenate([
            values,
            jnp.broadcast_to(new[request], (cell.num_calls, *new.shape[1:])),
        ])
        for values, new in zip(cached, step_keys)
    ))


def build_dsa_decode(cell, args):
    """The decode steps on a filled cache (filled_caches); every step's
    tokens are the same, at the next position."""
    import jax

    geometry = DSA_GEOMETRIES[cell.dsa_geometry]
    pages_per_seq = -(-cell.seq // cell.page_size)
    num_pages = cell.batch * pages_per_seq
    table = dsa_block_table(cell, pages_per_seq)
    base_pages = math.gcd(num_pages, DSA_BASE_PAGES)
    keys = jax.random.split(jax.random.key(cell.heads * 1000 + cell.seq), 4)
    queries = draw_dsa_queries(keys[0], geometry, cell.batch)
    step_keys = draw_dsa_keys(keys[1], geometry, cell.batch)
    base_keys = draw_dsa_keys(keys[2], geometry, base_pages * cell.page_size)
    layer = build_dsa_layer(
        cell.impl,
        geometry,
        num_tokens=cell.batch,
        batch=cell.batch,
        max_seqlen_k=cell.seq,
        torchtpu_source=args.torchtpu_source,
        route=cell.dsa_route,
    )
    requests = [
        DsaRequests(
            np.arange(cell.batch + 1),
            np.full(cell.batch, cell.context + i + 1),
            table,
            cell.batch,
        )
        for i in range(cell.num_calls)
    ]
    layer_keys = layer.keys(step_keys)
    return DsaScenario(
        layer=layer,
        geometry=geometry,
        fresh=lambda: filled_caches(layer, base_keys, num_pages, keys[3]),
        queries=queries,
        layer_queries=layer.queries(queries),
        calls=[(layer_keys, layer.meta(step)) for step in requests],
        requests=requests,
        rows=edge_rows(0, cell.batch),
        logical=functools.partial(
            dsa_decode_logical, cell, table, base_keys, keys[3], step_keys
        ),
    )


def build_dsa_chunked_prefill(cell, args):
    """The chunked prefill from an empty cache; every chunk reuses one
    chunk's queries and writes its own keys."""
    import jax

    geometry = DSA_GEOMETRIES[cell.dsa_geometry]
    pages_per_seq = -(-cell.seq // cell.page_size)
    table = dsa_block_table(cell, pages_per_seq)
    keys = jax.random.split(jax.random.key(cell.heads * 1000 + cell.seq), 2)
    queries = draw_dsa_queries(keys[0], geometry, cell.chunk)
    sequence = draw_dsa_keys(keys[1], geometry, cell.seq)
    layer = build_dsa_layer(
        cell.impl,
        geometry,
        num_tokens=cell.chunk,
        batch=1,
        max_seqlen_k=cell.seq,
        torchtpu_source=args.torchtpu_source,
        route=cell.dsa_route,
    )
    requests = [
        DsaRequests(
            np.array([0, cell.chunk]),
            np.array([(i + 1) * cell.chunk]),
            table,
            0,
        )
        for i in range(cell.num_calls)
    ]
    chunks = [
        sequence.rows(slice(i * cell.chunk, (i + 1) * cell.chunk))
        for i in range(cell.num_calls)
    ]
    return DsaScenario(
        layer=layer,
        geometry=geometry,
        fresh=lambda: layer.empty_caches(pages_per_seq),
        queries=queries,
        layer_queries=layer.queries(queries),
        calls=[
            (layer.keys(chunk), layer.meta(call))
            for chunk, call in zip(chunks, requests)
        ],
        requests=requests,
        rows=edge_rows(0, cell.chunk),
        logical=lambda request: sequence,
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
        cell.num_seqs * block_table.shape[1],
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
        cell.num_seqs * block_table.shape[1],
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
        cell.num_seqs * block_table.shape[1],
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


BUILDERS = {
    "flywheel": build_flywheel,
    "rpa": build_rpa,
    "batched_rpa": build_batched_rpa,
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


def cache_hbm_error(cell):
    """Why the cell's KV cache cannot fit in HBM, or None."""
    import jax

    hbm = jax.devices()[0].memory_stats().get("bytes_limit")
    if hbm and cell.cache_bytes > 0.9 * hbm:
        return (
            f"the KV cache needs {cell.cache_bytes / 2**30:.1f} GiB, more "
            f"than 90% of the device's {hbm / 2**30:.1f} GiB of HBM"
        )
    return None


def dsa_cell_record(cell, build_scenario, trace_directory):
    """A DSA cell's work, timing and check, plus per-op trace times when a
    trace directory is given; or its error."""
    try:
        scenario = build_scenario()
        record = {
            **dsa_work(scenario.geometry, scenario.requests),
            **measure_dsa(scenario),
        }
    except Exception as error:  # pylint: disable=broad-exception-caught
        return {
            "status": "error",
            "error": f"{type(error).__name__}: {str(error)[:400]}",
        }
    if trace_directory is None:
        return record
    per_op = trace_scenario(
        cell, scenario.step, scenario.fresh(), trace_directory
    )
    if per_op is None:
        record["trace_error"] = "no device events in the trace"
    else:
        record["trace_ops_ms"] = dict(
            sorted(per_op.items(), key=lambda op: -op[1])
        )
    return record


def run_dsa(cell, args):
    record = {
        "key": cell.key,
        "cell": dataclasses.asdict(cell),
        "num_calls": cell.num_calls,
        "cache_bytes": cell.cache_bytes,
    }
    hbm_error = cache_hbm_error(cell)
    if hbm_error is not None:
        record.update(status="error", error=hbm_error)
        return record
    if cell.scenario == "decode":
        build = build_dsa_decode
    else:
        build = build_dsa_chunked_prefill
    record.update(
        dsa_cell_record(cell, functools.partial(build, cell, args), args.trace)
    )
    return record


def run(cell, args):
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
        kv_tokens_read = cell.num_calls * cell.batch * cell.seq
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
    }
    hbm_error = cache_hbm_error(cell)
    if hbm_error is not None:
        record.update(status="error", error=hbm_error)
        return record
    try:
        if cell.is_mla:
            step, fresh, info, check = build_mla(cell, args)
        else:
            inputs = make_inputs(cell)
            step, fresh, info = BUILDERS[cell.impl](cell, inputs, args)
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


def make_cell(args):
    is_mla = args.impl in MLA_PREFILL_IMPLS
    is_dsa = args.impl in DSA_IMPLS
    if args.page_size is not None:
        page_size = args.page_size
    elif is_dsa:
        page_size = DSA_PAGE
    elif is_mla:
        page_size = SERVING_PAGE[args.kv_dtype]
    else:
        page_size = 128
    if is_dsa:
        heads = DSA_GEOMETRIES[args.dsa_geometry].heads
    else:
        heads = args.heads
    return Cell(
        args.impl,
        args.scenario,
        heads,
        # Note (david): MLA's queries attend one latent head.
        1 if is_mla or is_dsa else args.heads_k,
        LATENT_DIM if is_mla or is_dsa else args.head_dim,
        args.seq,
        args.chunk,
        args.batch,
        args.decode_tokens,
        page_size,
        args.sliding_window,
        args.verify_tokens,
        kv_dtype=DSA_KV_DTYPE if is_dsa else args.kv_dtype,
        dsa_geometry=args.dsa_geometry,
        dsa_route=args.dsa_route,
    )


def main(argv=None):
    args = parse_args(argv)
    cell = make_cell(args)
    if args.scenario == "chunked_prefill" and args.seq % args.chunk:
        raise SystemExit("--seq must be a multiple of --chunk.")
    if args.output and not args.trace and recorded(args.output, cell.key):
        print(f"{cell.key}: already recorded in {args.output}")
        return 0
    if cell.impl in TORCHTPU_DSA_IMPLS:
        prepare_torchtpu_process()
    record = run_dsa(cell, args) if cell.is_dsa else run(cell, args)
    if args.output:
        write_record(args.output, record)
    print(json.dumps(record))
    return 0 if record["status"] == "ok" else 1


def scenario_dsa_args_error(args):
    """Why the DSA arguments do not fit together, or None."""
    if args.impl in DSA_IMPLS and args.scenario not in DSA_SCENARIOS:
        return f"--impl {args.impl} runs {' and '.join(DSA_SCENARIOS)} only"
    elif args.impl in DSA_IMPLS and args.sliding_window is not None:
        return f"--impl {args.impl} takes no --sliding-window"
    elif args.dsa_route is not None and args.impl != FLYWHEEL_DSA:
        return f"--impl {args.impl} takes no --dsa-route"
    return dsa_args_error(args.impl, args.dsa_geometry, args.page_size)


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
        "--verify-tokens",
        type=int,
        default=1,
        help="verify: query tokens per sequence per step",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        help=(
            "default: 128; for an MLA impl, MLA v2's serving page of the"
            f" dtype; for a DSA impl, {DSA_PAGE}"
        ),
    )
    parser.add_argument(
        "--kv-dtype",
        default="bfloat16",
        choices=("bfloat16", "float8_e4m3fn", "float8_e5m2"),
        help="k, v and KV cache dtype; q stays bfloat16",
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
        help="batched_rpa and the torchtpu DSA impls: a vllm-torchtpu checkout",
    )
    parser.add_argument(
        "--mla-v2-source",
        type=pathlib.Path,
        default=DEFAULT_MLA_V2_SOURCE,
        help="mla_v2: a tpu-inference checkout at the pinned commit",
    )
    parser.add_argument(
        "--dsa-geometry",
        choices=tuple(DSA_GEOMETRIES),
        help="a DSA impl's model geometry",
    )
    parser.add_argument(
        "--dsa-route",
        help=(
            "flywheel_dsa: pin dsa_attention's route (mixed or sparse);"
            " default: picked on device"
        ),
    )
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument(
        "--trace",
        type=pathlib.Path,
        help="also trace one scenario into this directory",
    )
    args = parser.parse_args(argv)
    is_mla = args.impl in MLA_PREFILL_IMPLS
    dsa_error = scenario_dsa_args_error(args)
    if dsa_error is not None:
        parser.error(dsa_error)
    elif is_mla and args.scenario != "chunked_prefill":
        parser.error(f"--impl {args.impl} runs chunked_prefill only")
    elif is_mla and args.sliding_window is not None:
        parser.error(f"--impl {args.impl} takes no --sliding-window")
    elif is_mla and args.kv_dtype not in SERVING_PAGE:
        parser.error(f"--impl {args.impl} takes --kv-dtype {set(SERVING_PAGE)}")
    else:
        return args


if __name__ == "__main__":
    sys.exit(main())
