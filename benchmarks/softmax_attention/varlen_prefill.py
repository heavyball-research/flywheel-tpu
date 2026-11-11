#!/usr/bin/env python3
"""Time one ragged prefill batch over a paged KV cache.

    PYTHONPATH=. uv run --no-sync python \\
        benchmarks/softmax_attention/varlen_prefill.py \\
        --impl flywheel --heads 32 --heads-k 8 --head-dim 128 \\
        --pattern spiky --batch 32 --budget 32768 \\
        --output results/benchmark/varlen_prefill/benchmark.jsonl \\
        --trace results/benchmark/varlen_prefill/traces

A cell's --budget tokens are split across --batch sequences by the pattern's
weights (FlashAttention's benchmarks/clc_bench.py), so every pattern at the
same (budget, batch) packs the same tokens and differs only in shape:

  uniform   every sequence the same length
  spiky     one sequence takes 32 / (batch + 31) of the budget, the rest
            split the remainder evenly
  longtail  lengths fall off linearly, the longest batch times the shortest

Every sequence is a whole prompt with no cached prefix, attended causally
over a paged cache of --page-size token pages. The page table gives every
sequence a fixed stride of next_pow2(max_seqlen) / page_size slots, as a
server sized for that length allocates it, over a pool that holds only the
pages the real lengths need, shuffled across sequences; a sequence's slots
past its own pages repeat its last page.

flywheel runs flash_attn_varlen_func's paged kernel, read-only as in
FlashAttention, over a cache that already holds the batch's K/V, with
max_seqlen the longest sequence and flywheel_tpu's own block sizes. Every
impl takes q and returns out token-major, (total, heads, head_dim), as a
server holds them; flywheel's paged kernel is head-major, so its call also
transposes q in and out back. rpa runs
RPA v3's ragged_paged_attention with every sequence mixed and its blocks from
rpa_tuned_<chip>.json at the next_pow2(max_seqlen) bucket, else its formula.
batched_rpa runs vllm-torchtpu's batched RPA (--torchtpu-source) with its
defaults. Both RPAs also write the new K/V into their own cache layout inside
the call, which flywheel's time does not include.

flywheel_mla and mla_v2 run DeepSeek's absorbed MLA prefill
(benchmarks/common/mla.py), writing the batch into an empty MLA v2 cache in
its serving pages; an MLA cell whose last sequence fails COS_DIFF_BOUND is
recorded as check_failed.

flywheel_dsa, torchtpu_dsa and torchtpu_dsa_masked_dense run one DeepSeek
sparse attention layer step (benchmarks/common/dsa.py): each impl writes the
batch's index keys and latents into empty fp8 caches of DSA_PAGE pages, then
scores, selects and attends. A DSA cell whose last sequence fails
COS_DIFF_BOUND or INDEX_DEFICIT_BOUND is recorded as check_failed.

Attention work is the causal (query, key) pairs, sum(s (s + 1) / 2), which
the patterns do not hold constant, so only TFLOP/s compares across patterns.
Only the call is inside the clock: q, k, v and the metadata are made once.
With --trace the cell also runs under jax.profiler, and every device op is
summed by name. A cell already recorded as ok in --output is skipped unless
--trace is given.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import math
import pathlib
import sys
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.common.dsa import (
    DSA_GEOMETRIES,
    DSA_IMPLS,
    DSA_KV_DTYPE,
    DSA_PAGE,
    TORCHTPU_DSA_IMPLS,
    DsaRequests,
    DsaScenario,
    build_dsa_layer,
    draw_dsa_keys,
    draw_dsa_queries,
    dsa_args_error,
    edge_rows,
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
from benchmarks.common.rpa import FIELDS as RPA_FIELDS
from benchmarks.common.timing import time_call
from benchmarks.softmax_attention.attention_block import tuned_table
from benchmarks.softmax_attention.kvcache_scenarios import (
    KERNEL_MARKERS,
    draw,
    dsa_cell_record,
    trace_scenario,
)
from flywheel_tpu import flash_attn_varlen_func
from flywheel_tpu.softmax_attention.config import next_pow2
from flywheel_tpu.softmax_attention.mla_decode import mla_cache_shape

IMPLS = ("flywheel", "rpa", "batched_rpa", *MLA_PREFILL_IMPLS, *DSA_IMPLS)
PATTERNS = ("uniform", "spiky", "longtail")

Step = Callable[[jax.Array], tuple[jax.Array, jax.Array]]
BuildInfo = dict[str, str | int | list[int]]


def pattern_weights(pattern: str, batch: int) -> list[float]:
    """Relative sequence lengths of a pattern."""
    if pattern == "uniform":
        return [1.0] * batch
    elif pattern == "spiky":
        return [32.0] + [1.0] * (batch - 1)
    elif pattern == "longtail":
        return [float(batch - index) for index in range(batch)]
    else:
        raise ValueError(f"unsupported pattern: {pattern}")


def split_budget(weights: list[float], total_tokens: int) -> list[int]:
    """Lengths in proportion to weights that sum exactly to total_tokens.

    Floor each share, then hand out (or take back) the remainder one token at a
    time in order of the dropped fraction.
    """
    if total_tokens < len(weights):
        raise ValueError(f"budget {total_tokens} below batch {len(weights)}")
    scaled = [w / sum(weights) * total_tokens for w in weights]
    lengths = [max(1, int(math.floor(value))) for value in scaled]
    delta = total_tokens - sum(lengths)
    order = sorted(
        range(len(weights)),
        key=lambda i: scaled[i] - math.floor(scaled[i]),
        reverse=delta > 0,
    )
    cursor = 0
    while delta != 0:
        index = order[cursor % len(order)]
        if delta > 0:
            lengths[index] += 1
            delta -= 1
        elif lengths[index] > 1:
            lengths[index] -= 1
            delta += 1
        cursor += 1
    return lengths


@dataclasses.dataclass(frozen=True)
class Cell:
    """One benchmark point: an implementation on one ragged prefill batch."""

    impl: str
    heads: int
    heads_k: int
    head_dim: int
    pattern: str
    batch: int
    budget: int
    page_size: int
    kv_dtype: str = "bfloat16"
    # A DSA cell's DSA_GEOMETRIES entry.
    dsa_geometry: str | None = None

    @property
    def is_mla(self) -> bool:
        return self.impl in MLA_PREFILL_IMPLS

    @property
    def is_dsa(self) -> bool:
        return self.impl in DSA_IMPLS

    @property
    def key(self) -> str:
        if self.is_dsa:
            heads_tag = self.dsa_geometry
        elif self.is_mla:
            heads_tag = f"h{self.heads}"
        else:
            heads_tag = f"h{self.heads}-k{self.heads_k}-d{self.head_dim}"
        kv_tag = "" if self.kv_dtype == "bfloat16" else f"-kv{self.kv_dtype}"
        return (
            f"{self.impl}-{heads_tag}-{self.pattern}-b{self.batch}"
            f"-t{self.budget}-p{self.page_size}{kv_tag}"
        )

    @property
    def seed(self) -> int:
        """The seed of the cell's draws; it leaves the impl out, so every impl
        sees the same numbers."""
        seed = 0
        for part in (
            self.heads,
            self.heads_k,
            self.head_dim,
            self.batch,
            self.budget,
            PATTERNS.index(self.pattern),
        ):
            seed = (seed * 1000003 + part) % (2**31)
        return seed

    @property
    def lengths(self) -> list[int]:
        return split_budget(
            pattern_weights(self.pattern, self.batch), self.budget
        )


@dataclasses.dataclass(frozen=True)
class Inputs:
    q: jax.Array
    k: jax.Array
    v: jax.Array
    kv_lens: jax.Array
    cu_q_lens: jax.Array
    block_table: jax.Array
    num_pages: int


def make_inputs(cell: Cell) -> Inputs:
    """Packed q, k, v, the ragged metadata and the page table, shared by every
    impl."""
    lengths = cell.lengths
    generator = np.random.default_rng(cell.seed)
    total = cell.budget
    q = draw(generator, "bfloat16", total, cell.heads, cell.head_dim)
    k = draw(generator, "bfloat16", total, cell.heads_k, cell.head_dim)
    v = draw(generator, "bfloat16", total, cell.heads_k, cell.head_dim)
    table, num_pages = page_table(generator, lengths, cell.page_size)
    cu_q_lens = np.zeros(cell.batch + 1, np.int32)
    cu_q_lens[1:] = np.cumsum(lengths)
    return jax.block_until_ready(
        Inputs(
            q,
            k,
            v,
            jnp.asarray(np.asarray(lengths, np.int32)),
            jnp.asarray(cu_q_lens),
            jnp.asarray(table),
            num_pages,
        )
    )


def page_table(
    generator: np.random.Generator, lengths: list[int], page_size: int
) -> tuple[np.ndarray, int]:
    """(table, num_pages): every sequence's next_pow2(max_seqlen) / page_size
    slots over a shuffled pool of the pages the lengths need, a sequence's
    slots past its own pages repeating its last page."""
    own_pages = [-(-length // page_size) for length in lengths]
    num_pages = sum(own_pages)
    pages_per_seq = -(-next_pow2(max(lengths)) // page_size)
    pool = generator.permutation(num_pages).astype(np.int32)
    table = np.empty((len(lengths), pages_per_seq), np.int32)
    start = 0
    for index, count in enumerate(own_pages):
        table[index, :count] = pool[start : start + count]
        table[index, count:] = pool[start + count - 1]
        start += count
    return table, num_pages


def build_mla(
    cell: Cell, args: argparse.Namespace
) -> tuple[
    Step, Callable[[], jax.Array], BuildInfo, Callable[[jax.Array], float]
]:
    """(step, fresh_state, info, check) running an MLA prefill path that
    writes the batch's latents into an empty cache and attends; check(out) is
    the last sequence's cos_diff."""
    lengths = cell.lengths
    keys = jax.random.split(jax.random.key(cell.seed), 4)
    q_latent = draw_normal(
        keys[0], (cell.budget, cell.heads, LATENT_DIM), jnp.bfloat16
    )
    q_pe = draw_normal(
        keys[1], (cell.budget, cell.heads, ROPE_DIM), jnp.bfloat16
    )
    new_kv_c = draw_normal(keys[2], (cell.budget, LATENT_DIM), cell.kv_dtype)
    new_k_pe = draw_normal(keys[3], (cell.budget, ROPE_DIM), cell.kv_dtype)
    table, num_pages = page_table(
        np.random.default_rng(cell.seed), lengths, cell.page_size
    )
    cu_q_lens = np.zeros(cell.batch + 1, np.int32)
    cu_q_lens[1:] = np.cumsum(lengths)
    cache_shape = mla_cache_shape(
        num_pages, cell.page_size, jnp.dtype(cell.kv_dtype)
    )
    path = build_mla_prefill(
        cell.impl,
        heads=cell.heads,
        kv_dtype=cell.kv_dtype,
        cache_shape=cache_shape,
        num_tokens=cell.budget,
        batch=cell.batch,
        pages_per_seq=table.shape[1],
        max_seqlen_q=max(lengths),
        max_seqlen_k=max(lengths),
        mla_v2_source=args.mla_v2_source,
    )
    operands = jax.block_until_ready((
        *path.queries(q_latent, q_pe),
        *path.tokens(new_kv_c, new_k_pe),
        *path.meta(cu_q_lens, np.asarray(lengths, np.int32), table),
    ))
    kernel = jax.jit(path.attend, donate_argnums=(0,))
    last = slice(int(cu_q_lens[-2]), int(cu_q_lens[-1]))
    return (
        (lambda cache: kernel(cache, *operands)),
        (lambda: jnp.zeros(cache_shape, cell.kv_dtype)),
        {
            **path.info,
            "pages_per_seq": int(table.shape[1]),
            "cache_pages": num_pages,
        },
        lambda out: mla_cos_diff(
            path.token_major(out)[last],
            q_latent[last],
            q_pe[last],
            new_kv_c[last],
            new_k_pe[last],
            cell.kv_dtype,
        ),
    )


def build_dsa(cell: Cell, args: argparse.Namespace) -> DsaScenario:
    """One DSA layer step that writes the batch into empty caches; the check
    takes the last sequence's rows."""
    geometry = DSA_GEOMETRIES[cell.dsa_geometry]
    lengths = cell.lengths
    keys = jax.random.split(jax.random.key(cell.seed), 2)
    queries = draw_dsa_queries(keys[0], geometry, cell.budget)
    tokens = draw_dsa_keys(keys[1], geometry, cell.budget)
    table, num_pages = page_table(
        np.random.default_rng(cell.seed), lengths, cell.page_size
    )
    cu_q_lens = np.zeros(cell.batch + 1, np.int32)
    cu_q_lens[1:] = np.cumsum(lengths)
    requests = DsaRequests(cu_q_lens, np.asarray(lengths, np.int32), table, 0)
    layer = build_dsa_layer(
        cell.impl,
        geometry,
        num_tokens=cell.budget,
        batch=cell.batch,
        max_seqlen_k=max(lengths),
        torchtpu_source=args.torchtpu_source,
    )
    return DsaScenario(
        layer=layer,
        geometry=geometry,
        fresh=lambda: layer.empty_caches(num_pages),
        queries=queries,
        layer_queries=layer.queries(queries),
        calls=[(layer.keys(tokens), layer.meta(requests))],
        requests=[requests],
        rows=edge_rows(int(cu_q_lens[-2]), int(cu_q_lens[-1])),
        logical=lambda request: tokens.rows(
            slice(int(cu_q_lens[request]), int(cu_q_lens[request + 1]))
        ),
    )


def run_dsa(cell: Cell, args: argparse.Namespace) -> dict[str, object]:
    record = {
        "key": cell.key,
        "cell": dataclasses.asdict(cell),
        "lengths": cell.lengths,
        "max_seqlen": max(cell.lengths),
    }
    record.update(
        dsa_cell_record(
            cell, functools.partial(build_dsa, cell, args), args.trace
        )
    )
    return record


def build_flywheel(
    cell: Cell, inputs: Inputs, args: argparse.Namespace
) -> tuple[Step, Callable[[], jax.Array], BuildInfo]:
    """(step, fresh_state, info) running flash_attn_varlen_func's paged
    kernel over a cache that holds the batch's K/V."""
    del args  # Unused.
    lengths = cell.lengths
    # Note (david): varlen callers pass the longest sequence as max_seqlen.
    max_seqlen = max(lengths)
    table = np.asarray(inputs.block_table)
    token_rows = np.concatenate([
        table[index, np.arange(length) // cell.page_size] * cell.page_size
        + np.arange(length) % cell.page_size
        for index, length in enumerate(lengths)
    ])

    @jax.jit
    def fill(k, v):
        # Note (david): a bf16 merged cache interleaves each kv head's K row
        # and V row, [k0, v0, k1, v1, ...].
        rows = jnp.stack([k, v], axis=2).reshape(
            cell.budget, 2 * cell.heads_k, cell.head_dim
        )
        pool = jnp.zeros(
            (
                inputs.num_pages * cell.page_size,
                2 * cell.heads_k,
                cell.head_dim,
            ),
            jnp.bfloat16,
        )
        return (
            pool.at[token_rows]
            .set(rows)
            .reshape(
                inputs.num_pages,
                cell.page_size,
                2 * cell.heads_k,
                cell.head_dim,
            )
        )

    cache = jax.block_until_ready(fill(inputs.k, inputs.v))

    def call(cache, q, cu_q_lens, block_table, kv_lens):
        # Note (david): a server holds q and out token-major, as both RPAs
        # take them, and the paged kernel is head-major only, so the two
        # relayouts are part of the call.
        out = flash_attn_varlen_func(
            jnp.transpose(q, (1, 0, 2)),
            cache,
            None,
            cu_q_lens,
            None,
            max_seqlen,
            max_seqlen,
            causal=True,
            block_table=block_table,
            seqused_k=kv_lens,
            nheads_k=cell.heads_k,
        )
        return jnp.transpose(out, (1, 0, 2))

    kernel = jax.jit(call)
    operands = (inputs.q, inputs.cu_q_lens, inputs.block_table, inputs.kv_lens)
    return (
        (lambda cache: (kernel(cache, *operands), cache)),
        (lambda: cache),
        {},
    )


def rpa_table_blocks(
    path: pathlib.Path, cell: Cell, bucket: int
) -> dict[str, int] | None:
    """The tuned RPA table's entry at this head config and bucket as
    {field: value}, or None. The table has no mask level."""
    if path.exists():
        table = json.loads(path.read_text())
        head = (
            f"q_head-{cell.heads}_kv_head-{cell.heads_k}_head-{cell.head_dim}"
        )
        values = table["configs"].get(head, {}).get(str(bucket))
        if values is None:
            return None
        else:
            return dict(zip(table["block_order"], values))
    else:
        return None


def build_rpa(
    cell: Cell, inputs: Inputs, args: argparse.Namespace
) -> tuple[Step, Callable[[], jax.Array], BuildInfo]:
    """(step, fresh_state, info) running ragged_paged_attention."""
    rpa = load_rpa_v3(args.rpa_source)
    shape = rpa.get_kv_cache_shape(
        inputs.num_pages,
        cell.page_size,
        cell.heads_k,
        cell.head_dim,
        jnp.dtype(jnp.bfloat16),
    )
    bucket = next_pow2(max(cell.lengths))
    blocks = rpa_table_blocks(args.rpa_table, cell, bucket)
    source = "table"
    if blocks is None:
        formula = rpa.get_default_block_sizes(
            jnp.bfloat16,
            jnp.bfloat16,
            cell.heads,
            cell.heads_k,
            cell.head_dim,
            cell.page_size,
            cell.budget,
            cell.batch,
            inputs.block_table.shape[1],
            case=rpa.RpaCase.MIXED,
        )
        blocks = {name: int(formula[name]) for name in RPA_FIELDS}
        source = "formula"
    block_sizes = tuple(int(blocks[name]) for name in RPA_FIELDS)
    distribution = request_distribution(num_decode=0, num_seqs=cell.batch)
    kernel_fn = unjitted_kernel(rpa)
    scale = 1.0 / math.sqrt(cell.head_dim)

    def call(cache, q, k, v, kv_lens, page_indices, cu_q_lens):
        out, cache = kernel_fn(
            q,
            k,
            v,
            cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            use_causal_mask=True,
            sm_scale=scale,
            m_block_sizes=block_sizes,
        )
        return out, cache

    kernel = jax.jit(call, donate_argnums=(0,))
    operands = (
        inputs.q,
        inputs.k,
        inputs.v,
        inputs.kv_lens,
        inputs.block_table.reshape(-1),
        inputs.cu_q_lens,
    )
    info = {
        "block_source": source,
        "blocks": list(block_sizes),
        "bucket": bucket,
    }
    return (
        (lambda cache: kernel(cache, *operands)),
        (lambda: jnp.zeros(shape, jnp.bfloat16)),
        info,
    )


def build_batched_rpa(
    cell: Cell, inputs: Inputs, args: argparse.Namespace
) -> tuple[Step, Callable[[], jax.Array], BuildInfo]:
    """(step, fresh_state, info) running vllm-torchtpu's batched RPA."""
    if args.torchtpu_source is None:
        raise ValueError("--impl batched_rpa needs --torchtpu-source.")
    wrapper = load_batched_rpa(args.torchtpu_source)
    shape = wrapper.get_kv_cache_shape(
        inputs.num_pages,
        cell.page_size,
        cell.heads_k,
        cell.head_dim,
        jnp.dtype(jnp.bfloat16),
    )
    distribution = request_distribution(num_decode=0, num_seqs=cell.batch)
    kernel_fn = unjitted_kernel(wrapper)
    scale = 1.0 / math.sqrt(cell.head_dim)

    def call(cache, q, k, v, kv_lens, page_indices, cu_q_lens):
        out, cache = kernel_fn(
            q,
            k,
            v,
            cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            sm_scale=scale,
            sliding_window=None,
        )
        return out, cache

    kernel = jax.jit(call, donate_argnums=(0,))
    operands = (
        inputs.q,
        inputs.k,
        inputs.v,
        inputs.kv_lens,
        inputs.block_table.reshape(-1),
        inputs.cu_q_lens,
    )
    return (
        (lambda cache: kernel(cache, *operands)),
        (lambda: jnp.zeros(shape, jnp.bfloat16)),
        {"block_source": "batched rpa default"},
    )


BUILDERS = {
    "flywheel": build_flywheel,
    "rpa": build_rpa,
    "batched_rpa": build_batched_rpa,
}


def run(cell: Cell, args: argparse.Namespace) -> dict[str, object]:
    lengths = cell.lengths
    pairs = sum(length * (length + 1) // 2 for length in lengths)
    if cell.is_mla:
        # Note (david): QK over the latent and the rope, P @ V over the latent.
        attn_flops = 2 * pairs * cell.heads * (2 * LATENT_DIM + ROPE_DIM)
    else:
        attn_flops = 4 * pairs * cell.heads * cell.head_dim
    record = {
        "key": cell.key,
        "cell": dataclasses.asdict(cell),
        "lengths": lengths,
        "max_seqlen": max(lengths),
        "attention_pairs": pairs,
        "attn_flops": attn_flops,
    }
    try:
        if cell.is_mla:
            step, fresh, info, check = build_mla(cell, args)
        else:
            inputs = make_inputs(cell)
            record.update(
                pages_per_seq=int(inputs.block_table.shape[1]),
                cache_pages=inputs.num_pages,
            )
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
    record.update(status=status, **timing, tflops=attn_flops / (ms * 1e9))
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


def make_cell(args: argparse.Namespace) -> Cell:
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
        heads,
        # Note (david): MLA's queries attend one latent head.
        1 if is_mla or is_dsa else args.heads_k,
        LATENT_DIM if is_mla or is_dsa else args.head_dim,
        args.pattern,
        args.batch,
        args.budget,
        page_size,
        DSA_KV_DTYPE if is_dsa else args.kv_dtype,
        args.dsa_geometry,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cell = make_cell(args)
    if args.output and not args.trace and recorded(args.output, cell.key):
        print(f"{cell.key}: already recorded in {args.output}")
        return 0
    if cell.impl in TORCHTPU_DSA_IMPLS:
        prepare_torchtpu_process()
    if cell.impl == "rpa" and args.rpa_table is None:
        args.rpa_table = tuned_table("rpa", jax.devices()[0].device_kind)
    record = run_dsa(cell, args) if cell.is_dsa else run(cell, args)
    if args.output:
        write_record(args.output, record)
    print(json.dumps(record))
    return 0 if record["status"] == "ok" else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--impl", choices=IMPLS, required=True)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--heads-k", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--pattern", choices=PATTERNS, required=True)
    parser.add_argument(
        "--batch",
        type=int,
        required=True,
        help="sequences the budget is split across",
    )
    parser.add_argument(
        "--budget",
        type=int,
        required=True,
        help="total packed tokens, split by the pattern",
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
        choices=tuple(SERVING_PAGE),
        help="cache dtype; fp8 for an MLA impl only (a DSA impl's is fp8)",
    )
    parser.add_argument(
        "--rpa-table",
        type=pathlib.Path,
        help=(
            "rpa: tuned block table; default: "
            "rpa_tuned_<chip>.json next to this file"
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
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument(
        "--trace",
        type=pathlib.Path,
        help="also trace the cell into this directory",
    )
    args = parser.parse_args(argv)
    dsa_error = dsa_args_error(args.impl, args.dsa_geometry, args.page_size)
    if dsa_error is not None:
        parser.error(dsa_error)
    elif args.impl not in MLA_PREFILL_IMPLS and args.kv_dtype != "bfloat16":
        parser.error(f"--impl {args.impl} takes a bfloat16 cache only")
    else:
        return args


if __name__ == "__main__":
    sys.exit(main())
