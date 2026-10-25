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
max_seqlen the longest sequence and flywheel_tpu's own block sizes. rpa runs
RPA v3's ragged_paged_attention with every sequence mixed and its blocks from
rpa_tuned_<chip>.json at the next_pow2(max_seqlen) bucket, else its formula.
batched_rpa runs vllm-torchtpu's batched RPA (--torchtpu-source) with its
defaults. Both RPAs also write the new K/V into their own cache layout inside
the call, which flywheel's time does not include.

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
import json
import math
import pathlib
import sys

import numpy as np

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
    trace_scenario,
)

IMPLS = ("flywheel", "rpa", "batched_rpa")
PATTERNS = ("uniform", "spiky", "longtail")


def pattern_weights(pattern, batch):
    """FlashAttention benchmarks/clc_bench.py pattern_weights."""
    if pattern == "uniform":
        return [1.0] * batch
    if pattern == "spiky":
        return [32.0] + [1.0] * (batch - 1)
    if pattern == "longtail":
        return [float(batch - index) for index in range(batch)]
    raise ValueError(f"unsupported pattern: {pattern}")


def split_budget(weights, total_tokens):
    """clc_bench.py normalize_lengths: weights -> lengths summing to the budget.

    Floor each share, then hand out (or take back) the remainder one token at a
    time in order of the dropped fraction, so the total is exact.
    """
    if total_tokens < len(weights):
        raise ValueError(f"budget {total_tokens} below batch {len(weights)}")
    scaled = [w / sum(weights) * total_tokens for w in weights]
    lengths = [max(1, int(math.floor(value))) for value in scaled]
    delta = total_tokens - sum(lengths)
    order = sorted(range(len(weights)),
                   key=lambda i: scaled[i] - math.floor(scaled[i]),
                   reverse=delta > 0)
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
    impl: str
    heads: int
    heads_k: int
    head_dim: int
    pattern: str
    batch: int
    budget: int
    page_size: int

    @property
    def key(self):
        return (f"{self.impl}-h{self.heads}-k{self.heads_k}-d{self.head_dim}"
                f"-{self.pattern}-b{self.batch}-t{self.budget}"
                f"-p{self.page_size}")

    @property
    def lengths(self):
        return split_budget(pattern_weights(self.pattern, self.batch),
                            self.budget)


@dataclasses.dataclass(frozen=True)
class Inputs:
    q: object
    k: object
    v: object
    kv_lens: object
    cu_q_lens: object
    block_table: object
    num_pages: int


def make_inputs(cell):
    """Packed q, k, v, the ragged metadata and the page table, shared by every
    impl: the seed leaves the impl out, so each sees the same numbers."""
    import jax
    import jax.numpy as jnp

    from flywheel_tpu.pallas.block_sizes import next_pow2

    lengths = cell.lengths
    seed = 0
    for part in (cell.heads, cell.heads_k, cell.head_dim, cell.batch,
                 cell.budget, PATTERNS.index(cell.pattern)):
        seed = (seed * 1000003 + part) % (2**31)
    generator = np.random.default_rng(seed)
    total = cell.budget
    q = draw(generator, "bfloat16", total, cell.heads, cell.head_dim)
    k = draw(generator, "bfloat16", total, cell.heads_k, cell.head_dim)
    v = draw(generator, "bfloat16", total, cell.heads_k, cell.head_dim)

    own_pages = [-(-length // cell.page_size) for length in lengths]
    num_pages = sum(own_pages)
    pages_per_seq = -(-next_pow2(max(lengths)) // cell.page_size)
    pool = generator.permutation(num_pages).astype(np.int32)
    table = np.empty((cell.batch, pages_per_seq), np.int32)
    start = 0
    for index, count in enumerate(own_pages):
        table[index, :count] = pool[start:start + count]
        table[index, count:] = pool[start + count - 1]
        start += count
    cu_q_lens = np.zeros(cell.batch + 1, np.int32)
    cu_q_lens[1:] = np.cumsum(lengths)
    return jax.block_until_ready(Inputs(
        q, k, v, jnp.asarray(np.asarray(lengths, np.int32)),
        jnp.asarray(cu_q_lens), jnp.asarray(table), num_pages))


def build_flywheel(cell, inputs, args):
    """(step, fresh_state, info) running flash_attn_varlen_func's paged
    kernel over a cache that holds the batch's K/V."""
    import jax
    import jax.numpy as jnp

    from flywheel_tpu import flash_attn_varlen_func

    lengths = cell.lengths
    # Note (david): flash_attn_with_kvcache sizes its tiles for a packed call
    # by max_seqlen_q = total_q, so this calls the paged kernel directly with
    # the longest sequence, as FlashAttention's varlen callers pass it.
    max_seqlen = max(lengths)
    table = np.asarray(inputs.block_table)
    token_rows = np.concatenate([
        table[index, np.arange(length) // cell.page_size] * cell.page_size
        + np.arange(length) % cell.page_size
        for index, length in enumerate(lengths)])

    @jax.jit
    def fill(k, v):
        # Note (david): a bf16 merged cache interleaves each kv head's K row
        # and V row, [k0, v0, k1, v1, ...].
        rows = jnp.stack([k, v], axis=2).reshape(
            cell.budget, 2 * cell.heads_k, cell.head_dim)
        pool = jnp.zeros((inputs.num_pages * cell.page_size,
                          2 * cell.heads_k, cell.head_dim), jnp.bfloat16)
        return pool.at[token_rows].set(rows).reshape(
            inputs.num_pages, cell.page_size, 2 * cell.heads_k,
            cell.head_dim)

    cache = jax.block_until_ready(fill(inputs.k, inputs.v))
    q_head_major = jax.block_until_ready(jnp.transpose(inputs.q, (1, 0, 2)))

    def call(cache, q, cu_q_lens, block_table, kv_lens):
        return flash_attn_varlen_func(
            q, cache, None, cu_q_lens, None, max_seqlen, max_seqlen,
            causal=True, block_table=block_table, seqused_k=kv_lens,
            nheads_k=cell.heads_k)

    kernel = jax.jit(call)
    operands = (q_head_major, inputs.cu_q_lens, inputs.block_table,
                inputs.kv_lens)
    return ((lambda cache: (kernel(cache, *operands), cache)),
            (lambda: cache), {})


def rpa_table_blocks(path, cell, bucket):
    """The tuned RPA table's entry at this head config and bucket as
    {field: value}, or None. The table has no mask level."""
    if not path.exists():
        return None
    table = json.loads(path.read_text())
    head = (f"q_head-{cell.heads}_kv_head-{cell.heads_k}"
            f"_head-{cell.head_dim}")
    values = table["configs"].get(head, {}).get(str(bucket))
    if values is None:
        return None
    return dict(zip(table["block_order"], values))


def build_rpa(cell, inputs, args):
    """(step, fresh_state, info) running ragged_paged_attention."""
    import jax
    import jax.numpy as jnp

    from flywheel_tpu.pallas.block_sizes import next_pow2

    rpa = load_rpa_v3(args.rpa_source)
    shape = rpa.get_kv_cache_shape(inputs.num_pages, cell.page_size,
                                   cell.heads_k, cell.head_dim,
                                   jnp.dtype(jnp.bfloat16))
    bucket = next_pow2(max(cell.lengths))
    blocks = rpa_table_blocks(args.rpa_table, cell, bucket)
    source = "table"
    if blocks is None:
        formula = rpa.get_default_block_sizes(
            jnp.bfloat16, jnp.bfloat16, cell.heads, cell.heads_k,
            cell.head_dim, cell.page_size, cell.budget, cell.batch,
            inputs.block_table.shape[1], case=rpa.RpaCase.MIXED)
        blocks = {name: int(formula[name]) for name in RPA_FIELDS}
        source = "formula"
    block_sizes = tuple(int(blocks[name]) for name in RPA_FIELDS)
    distribution = request_distribution(num_decode=0, num_seqs=cell.batch)
    kernel_fn = unjitted_kernel(rpa)
    scale = 1.0 / math.sqrt(cell.head_dim)

    def call(cache, q, k, v, kv_lens, page_indices, cu_q_lens):
        out, cache = kernel_fn(
            q, k, v, cache, kv_lens, page_indices, cu_q_lens, distribution,
            use_causal_mask=True, sm_scale=scale, m_block_sizes=block_sizes)
        return out, cache

    kernel = jax.jit(call, donate_argnums=(0,))
    operands = (inputs.q, inputs.k, inputs.v, inputs.kv_lens,
                inputs.block_table.reshape(-1), inputs.cu_q_lens)
    info = {"block_source": source, "blocks": list(block_sizes),
            "bucket": bucket}
    return ((lambda cache: kernel(cache, *operands)),
            (lambda: jnp.zeros(shape, jnp.bfloat16)), info)


def build_batched_rpa(cell, inputs, args):
    """(step, fresh_state, info) running vllm-torchtpu's batched RPA."""
    import jax
    import jax.numpy as jnp

    if args.torchtpu_source is None:
        raise ValueError("--impl batched_rpa needs --torchtpu-source.")
    wrapper = load_batched_rpa(args.torchtpu_source)
    shape = wrapper.get_kv_cache_shape(inputs.num_pages, cell.page_size,
                                       cell.heads_k, cell.head_dim,
                                       jnp.dtype(jnp.bfloat16))
    distribution = request_distribution(num_decode=0, num_seqs=cell.batch)
    kernel_fn = unjitted_kernel(wrapper)
    scale = 1.0 / math.sqrt(cell.head_dim)

    def call(cache, q, k, v, kv_lens, page_indices, cu_q_lens):
        out, cache = kernel_fn(
            q, k, v, cache, kv_lens, page_indices, cu_q_lens, distribution,
            sm_scale=scale, sliding_window=None)
        return out, cache

    kernel = jax.jit(call, donate_argnums=(0,))
    operands = (inputs.q, inputs.k, inputs.v, inputs.kv_lens,
                inputs.block_table.reshape(-1), inputs.cu_q_lens)
    return ((lambda cache: kernel(cache, *operands)),
            (lambda: jnp.zeros(shape, jnp.bfloat16)),
            {"block_source": "batched rpa default"})


BUILDERS = {"flywheel": build_flywheel, "rpa": build_rpa,
            "batched_rpa": build_batched_rpa}


def run(cell, args):
    lengths = cell.lengths
    pairs = sum(length * (length + 1) // 2 for length in lengths)
    attn_flops = 4 * pairs * cell.heads * cell.head_dim
    record = {"key": cell.key, "cell": dataclasses.asdict(cell),
              "lengths": lengths, "max_seqlen": max(lengths),
              "attention_pairs": pairs, "attn_flops": attn_flops}
    try:
        inputs = make_inputs(cell)
        record.update(pages_per_seq=int(inputs.block_table.shape[1]),
                      cache_pages=inputs.num_pages)
        step, fresh, info = BUILDERS[cell.impl](cell, inputs, args)
        record.update(info)
        timing, _ = time_call(step, fresh())
    except Exception as error:
        record.update(status="error",
                      error=f"{type(error).__name__}: {str(error)[:400]}")
        return record

    ms = timing["median_ms"]
    record.update(status="ok", **timing, tflops=attn_flops / (ms * 1e9))
    if args.trace:
        per_op = trace_scenario(cell, step, fresh(), args.trace)
        if per_op is None:
            record["trace_error"] = "no device events in the trace"
        else:
            kernel_ms = sum(t for name, t in per_op.items()
                            if any(m in name for m in KERNEL_MARKERS))
            record.update(
                trace_ops_ms=dict(sorted(per_op.items(),
                                         key=lambda item: -item[1])[:8]),
                trace_device_ms=sum(per_op.values()),
                kernel_ms=kernel_ms,
                kernel_tflops=attn_flops / (kernel_ms * 1e9) if kernel_ms
                else None)
    return record


def main(argv=None):
    args = parse_args(argv)
    cell = Cell(args.impl, args.heads, args.heads_k, args.head_dim,
                args.pattern, args.batch, args.budget, args.page_size)
    if args.output and not args.trace and recorded(args.output, cell.key):
        print(f"{cell.key}: already recorded in {args.output}")
        return 0
    if cell.impl == "rpa" and args.rpa_table is None:
        import jax

        args.rpa_table = tuned_table("rpa", jax.devices()[0].device_kind)
    record = run(cell, args)
    if args.output:
        write_record(args.output, record)
    print(json.dumps(record))
    return 0 if record["status"] == "ok" else 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--impl", choices=IMPLS, required=True)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--heads-k", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--pattern", choices=PATTERNS, required=True)
    parser.add_argument("--batch", type=int, required=True,
                        help="sequences the budget is split across")
    parser.add_argument("--budget", type=int, required=True,
                        help="total packed tokens, split by the pattern")
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--rpa-table", type=pathlib.Path,
                        help="rpa: tuned block table; default: "
                             "rpa_tuned_<chip>.json next to this file")
    parser.add_argument("--rpa-source", type=pathlib.Path,
                        default=DEFAULT_RPA_SOURCE)
    parser.add_argument("--torchtpu-source", type=pathlib.Path,
                        help="batched_rpa: a vllm-torchtpu checkout")
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--trace", type=pathlib.Path,
                        help="also trace the cell into this directory")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
