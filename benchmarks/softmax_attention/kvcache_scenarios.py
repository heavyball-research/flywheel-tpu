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
from benchmarks.common.timing import time_call
from benchmarks.softmax_attention import segments as segments_lib

IMPLS = ("flywheel", "rpa", "batched_rpa")
SCENARIOS = ("chunked_prefill", "decode", "verify")
KERNEL_MARKERS = ("flash_attn", "RPA", "ragged_paged")


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

    @property
    def key(self):
        if self.sliding_window is None:
            window_tag = ""
        else:
            window_tag = f"-w{self.sliding_window}"
        kv_tag = "" if self.kv_dtype == "bfloat16" else f"-kv{self.kv_dtype}"
        shape = (
            f"{self.impl}-{self.scenario}-h{self.heads}-k{self.heads_k}"
            f"-d{self.head_dim}-s{self.seq}-p{self.page_size}{window_tag}"
            f"{kv_tag}"
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
        """Cache bytes of one token: K and V of every KV head."""
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


def build_flywheel(cell, inputs, args):
    """(step, fresh_state, info) running flash_attn_with_kvcache."""
    del args  # Unused.
    import jax

    from flywheel_tpu import flash_attn_with_kvcache
    from flywheel_tpu.pallas.copy_utils import merged_cache_heads

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


def run(cell, args):
    import jax

    pairs = attention_pairs(cell)
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
        inputs = make_inputs(cell)
        step, fresh, info = BUILDERS[cell.impl](cell, inputs, args)
        record.update(info)
        timing, _ = time_call(step, fresh())
    except Exception as error:  # pylint: disable=broad-exception-caught
        record.update(
            status="error", error=f"{type(error).__name__}: {str(error)[:400]}"
        )
        return record

    ms = timing["median_ms"]
    record.update(
        status="ok",
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
    cell = Cell(
        args.impl,
        args.scenario,
        args.heads,
        args.heads_k,
        args.head_dim,
        args.seq,
        args.chunk,
        args.batch,
        args.decode_tokens,
        args.page_size,
        args.sliding_window,
        args.verify_tokens,
        kv_dtype=args.kv_dtype,
    )
    if args.scenario == "chunked_prefill" and args.seq % args.chunk:
        raise SystemExit("--seq must be a multiple of --chunk.")
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
        "--verify-tokens",
        type=int,
        default=1,
        help="verify: query tokens per sequence per step",
    )
    parser.add_argument("--page-size", type=int, default=128)
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
        help="batched_rpa: a vllm-torchtpu checkout",
    )
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument(
        "--trace",
        type=pathlib.Path,
        help="also trace one scenario into this directory",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
