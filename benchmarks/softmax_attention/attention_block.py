#!/usr/bin/env python3
"""Time a whole attention block on one prefill cell, hidden states in and out.

    PYTHONPATH=. uv run --no-sync python \\
        benchmarks/softmax_attention/attention_block.py \\
        --impl flywheel --heads 32 --heads-k 32 --head-dim 256 --seq 16384 \\
        --output results/benchmark/softmax_attention/benchmark.jsonl \\
        --trace results/benchmark/softmax_attention/traces

Every impl starts from x (batch, seq, d_model) and ends at (batch, seq,
d_model), with d_model = heads * head_dim, so the qkv projection, whatever
layout conversion the kernel's input needs, the kernel and the output
projection are all inside the clock:

  flywheel  einsum -> (batch, heads, seq, head_dim) -> flash_attn_func
  splash    einsum -> (batch, heads, seq, head_dim), heads folded into batch
            -> splash attention
  rpa       einsum -> (batch * seq, heads, head_dim) -> ragged_paged_attention,
            which also writes the paged KV cache

Each projection is one einsum against a (d_model, heads, head_dim) weight that
emits the shape its kernel takes, as vLLM's attention layer projects, so any
relayout left in the block is one XLA could not fold into the projection.

With --trace the cell also runs under jax.profiler, and segments.py splits the
device time of that same executable into the qkv projection, layout, the
attention kernel and the output projection. The kernel segment is the
kernel-level number (kernel_tflops); the wall clock is the block-level one.

Block sizes come from each impl's table for the chip the cell runs on: splash
and RPA from splash_tuned_<chip>.json and rpa_tuned_<chip>.json next to this
file (v6e or v7x; tune_blocks.py searches them), flywheel from flywheel_tpu's
own lookup. No chip stands in for another: on one with no tables, splash and
RPA cells are an error until --splash-table / --rpa-table name a table. A cell
its table does not hold runs the impl's own default: RPA's formula or Tokamax's
heuristic. A cell already recorded as ok in --output is skipped unless --trace
is given.
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
    load_rpa_v3,
    request_distribution,
    unjitted_kernel,
    vllm_page_size,
)
from benchmarks.common.rpa import FIELDS as RPA_FIELDS
from benchmarks.common.splash import (
    DEFAULT_SPLASH_SOURCE,
    HEURISTIC,
    load_splash,
)
from benchmarks.common.splash import FIELDS as SPLASH_FIELDS
from benchmarks.common.timing import time_call
from benchmarks.softmax_attention import segments as segments_lib

RPA_TABLE = pathlib.Path(__file__).with_name("rpa_tuned_v6e.json")
SPLASH_TABLE = pathlib.Path(__file__).with_name("splash_tuned_v6e.json")
# Chip, as flywheel_tpu names it, -> the suffix of the tables searched on it.
# There is no default: a chip missing here has no tables.
TABLE_TAGS = {"TPU v6e": "v6e", "TPU v7": "v7x"}
IMPLS = ("flywheel", "splash", "rpa")
TRACE_CALLS = 3


@dataclasses.dataclass(frozen=True)
class Cell:
    impl: str
    heads: int
    heads_k: int
    head_dim: int
    mask: str
    batch: int
    seq: int

    @property
    def causal(self):
        return self.mask == "causal"

    @property
    def d_model(self):
        return self.heads * self.head_dim

    @property
    def key(self):
        return (f"{self.impl}-h{self.heads}-k{self.heads_k}-d{self.head_dim}"
                f"-{self.mask}-b{self.batch}-s{self.seq}")


def tuned_table(impl, device_kind):
    """The path of impl's tuned table for the chip JAX calls device_kind."""
    from flywheel_tpu.tuned_block_sizes import get_device_variant_name

    chip = get_device_variant_name(device_kind)
    if chip not in TABLE_TAGS:
        raise ValueError(
            f"no tuned {impl} table for {device_kind!r} ({chip}): search one "
            f"with tune_blocks.py and add the chip to TABLE_TAGS, or name a "
            f"table with --{impl}-table.")
    return pathlib.Path(__file__).with_name(
        f"{impl}_tuned_{TABLE_TAGS[chip]}.json")


def table_entry(path, cell, per_mask):
    """The tuned table's entry for this cell as {field: value}, or None.

    The RPA table has no mask level, the splash table does.
    """
    if not path.exists():
        return None
    table = json.loads(path.read_text())
    head = (f"q_head-{cell.heads}_kv_head-{cell.heads_k}"
            f"_head-{cell.head_dim}")
    entry = table["configs"].get(head, {})
    if per_mask:
        entry = entry.get(cell.mask, {})
    values = entry.get(str(cell.seq))
    if values is None:
        return None
    order = table.get("config_order") or table["block_order"]
    return dict(zip(order, values))


def make_weights(cell):
    """x and the four projection weights, bf16, drawn on the host.

    Weights are scaled by 1/sqrt(d_model) so q, k and v come out near unit
    variance.
    """
    import jax
    import jax.numpy as jnp
    import ml_dtypes

    seed = 0
    for part in (cell.heads, cell.heads_k, cell.head_dim, cell.batch,
                 cell.seq):
        seed = (seed * 1000003 + part) % (2**31)
    generator = np.random.default_rng(seed)

    def draw(*shape, scale=1.0):
        block = generator.standard_normal(shape, dtype=np.float32) * scale
        return jnp.asarray(block.astype(ml_dtypes.bfloat16))

    fan = 1.0 / math.sqrt(cell.d_model)
    return jax.block_until_ready({
        "x": draw(cell.batch, cell.seq, cell.d_model),
        "wq": draw(cell.d_model, cell.heads, cell.head_dim, scale=fan),
        "wk": draw(cell.d_model, cell.heads_k, cell.head_dim, scale=fan),
        "wv": draw(cell.d_model, cell.heads_k, cell.head_dim, scale=fan),
        "wo": draw(cell.heads, cell.head_dim, cell.d_model, scale=fan),
    })


def build_flywheel(cell, w, args):
    """(step, fresh_state, info) for flash_attn_func on head-major q, k, v."""
    import jax
    import jax.numpy as jnp

    from flywheel_tpu import flash_attn_func

    def block(x, wq, wk, wv, wo):
        with jax.named_scope("qkv_proj"):
            q = jnp.einsum("btd,dnh->bnth", x, wq)
            k = jnp.einsum("btd,dnh->bnth", x, wk)
            v = jnp.einsum("btd,dnh->bnth", x, wv)
        with jax.named_scope("attn"):
            out = flash_attn_func(q, k, v, causal=cell.causal)
        with jax.named_scope("out_proj"):
            return jnp.einsum("bnth,nhd->btd", out, wo)

    call = jax.jit(block)
    operands = (w["x"], w["wq"], w["wk"], w["wv"], w["wo"])
    return (lambda state: (call(*operands), state)), (lambda: None), {}


def build_splash(cell, w, args):
    """(step, fresh_state, info) for splash attention on (batch * heads, ...)."""
    import jax
    import jax.numpy as jnp

    kernel, mask_lib = load_splash(args.splash_source)
    config = table_entry(args.splash_table, cell, per_mask=True)
    source = "table"
    if config is None:
        config, source = dict(HEURISTIC), "heuristic"

    layouts = {"h": kernel.QKVLayout.HEAD_DIM_MINOR,
               "s": kernel.QKVLayout.SEQ_MINOR}
    diag_grid = int(config["diag_grid"])
    splash_config = kernel.SplashConfig(
        block_q=int(config["block_q"]),
        block_kv=int(config["block_kv"]),
        block_kv_compute=int(config["block_kv_compute"]),
        num_stacked_q_heads=int(config["num_stacked_q_heads"]),
        q_layout=layouts[config["layout"][0]],
        k_layout=layouts[config["layout"][1]],
        v_layout=layouts[config["layout"][2]],
        use_experimental_scheduler=bool(config["scheduler"]),
        qk_diag_skip=diag_grid > 0,
        sv_diag_skip=diag_grid > 0,
        qk_diag_grid=diag_grid or 2)
    shape = (cell.seq, cell.seq)
    mask = (mask_lib.CausalMask(shape) if cell.causal
            else mask_lib.FullMask(shape))
    attention = kernel.make_splash_mha_single_device(mask, config=splash_config)

    # Splash takes no softmax scale, so it goes into wq, outside the clock.
    wq = (w["wq"].astype(jnp.float32)
          / math.sqrt(cell.head_dim)).astype(jnp.bfloat16)

    def block(x, wq, wk, wv, wo):
        with jax.named_scope("qkv_proj"):
            q = jnp.einsum("btd,dnh->bnth", x, wq)
            k = jnp.einsum("btd,dnh->bnth", x, wk)
            v = jnp.einsum("btd,dnh->bnth", x, wv)
        with jax.named_scope("attn"):
            out = attention(*(operand.reshape(-1, *operand.shape[2:])
                              for operand in (q, k, v))).reshape(q.shape)
        with jax.named_scope("out_proj"):
            return jnp.einsum("bnth,nhd->btd", out, wo)

    call = jax.jit(block)
    operands = (w["x"], wq, w["wk"], w["wv"], w["wo"])
    info = {"config_source": source,
            "config": [config[name] for name in SPLASH_FIELDS]}
    return (lambda state: (call(*operands), state)), (lambda: None), info


def build_rpa(cell, w, args):
    """(step, fresh_state, info) for ragged_paged_attention on a paged cache."""
    import jax
    import jax.numpy as jnp

    rpa = load_rpa_v3(args.rpa_source)
    total = cell.batch * cell.seq
    page_size = vllm_page_size(cell.seq, cell.batch)
    pages_per_seq = -(-cell.seq // page_size)
    num_pages = cell.batch * pages_per_seq
    cache_shape = rpa.get_kv_cache_shape(num_pages, page_size, cell.heads_k,
                                         cell.head_dim, jnp.bfloat16)
    metadata = (
        jnp.full((cell.batch,), cell.seq, jnp.int32),
        jnp.arange(num_pages, dtype=jnp.int32),
        jnp.arange(cell.batch + 1, dtype=jnp.int32) * cell.seq,
        request_distribution(num_decode=0, num_seqs=cell.batch),
    )

    blocks = table_entry(args.rpa_table, cell, per_mask=False)
    source = "table"
    if blocks is None:
        formula = rpa.get_default_block_sizes(
            jnp.bfloat16, jnp.bfloat16, cell.heads, cell.heads_k,
            cell.head_dim, page_size, total, cell.batch, pages_per_seq,
            case=rpa.RpaCase.MIXED)
        blocks = {name: int(formula[name]) for name in RPA_FIELDS}
        source = "formula"
    block_sizes = tuple(int(blocks[name]) for name in RPA_FIELDS)

    kernel = unjitted_kernel(rpa)
    scale = 1.0 / math.sqrt(cell.head_dim)

    def block(x, wq, wk, wv, wo, cache):
        with jax.named_scope("qkv_proj"):
            q = jnp.einsum("btd,dnh->btnh", x, wq).reshape(
                total, cell.heads, cell.head_dim)
            k = jnp.einsum("btd,dnh->btnh", x, wk).reshape(
                total, cell.heads_k, cell.head_dim)
            v = jnp.einsum("btd,dnh->btnh", x, wv).reshape(
                total, cell.heads_k, cell.head_dim)
        with jax.named_scope("attn"):
            out, cache = kernel(q, k, v, cache, *metadata,
                                use_causal_mask=cell.causal, sm_scale=scale,
                                m_block_sizes=block_sizes)
        with jax.named_scope("out_proj"):
            out = jnp.einsum(
                "btnh,nhd->btd",
                out.reshape(cell.batch, cell.seq, cell.heads, cell.head_dim),
                wo)
        return out, cache

    # The call donates the cache, so every timing run starts from a fresh one.
    call = jax.jit(block, donate_argnums=(5,))
    operands = (w["x"], w["wq"], w["wk"], w["wv"], w["wo"])
    info = {"block_source": source, "blocks": list(block_sizes),
            "page_size": page_size}
    return ((lambda state: call(*operands, state)),
            (lambda: jnp.zeros(cache_shape, jnp.bfloat16)), info)


BUILDERS = {"flywheel": build_flywheel, "splash": build_splash,
            "rpa": build_rpa}


def trace_cell(cell, step, state, directory):
    """Run a few calls under jax.profiler and split the trace, or None."""
    import jax

    target = directory / cell.key
    target.mkdir(parents=True, exist_ok=True)
    out, state = step(state)
    jax.block_until_ready(out)
    with jax.profiler.trace(str(target)):
        for _ in range(TRACE_CALLS):
            out, state = step(state)
        jax.block_until_ready(out)
    trace = segments_lib.load(target)
    return segments_lib.segments(trace) if trace is not None else None


def run(cell, args):
    import jax

    pair_count = cell.batch * cell.seq * cell.seq
    if cell.causal:
        pair_count //= 2
    attn_flops = 4 * pair_count * cell.heads * cell.head_dim
    proj_flops = (2 * cell.batch * cell.seq * cell.d_model
                  * (cell.heads + 2 * cell.heads_k) * cell.head_dim
                  + 2 * cell.batch * cell.seq * cell.heads * cell.head_dim
                  * cell.d_model)
    record = {"key": cell.key, "cell": dataclasses.asdict(cell),
              "d_model": cell.d_model, "attn_flops": attn_flops,
              "proj_flops": proj_flops}
    try:
        weights = make_weights(cell)
        step, fresh, info = BUILDERS[cell.impl](cell, weights, args)
        record.update(info)
        timing, _ = time_call(step, fresh())
    except Exception as error:
        record.update(status="error",
                      error=f"{type(error).__name__}: {str(error)[:300]}")
        return record

    ms = timing["median_ms"]
    record.update(status="ok", **timing,
                  block_tflops=(attn_flops + proj_flops) / (ms * 1e9))
    if args.trace:
        split = trace_cell(cell, step, fresh(), args.trace)
        if split is None:
            record["trace_error"] = "no device events in the trace"
        else:
            kernel_ms = split["ms"]["attn_kernel"]
            record.update(
                trace_module_ms=split["module_ms"],
                segments_ms=split["ms"],
                kernel_ms=kernel_ms,
                kernel_tflops=(attn_flops / (kernel_ms * 1e9)
                               if kernel_ms else None))
    return record


def main(argv=None):
    args = parse_args(argv)
    cell = Cell(args.impl, args.heads, args.heads_k, args.head_dim, args.mask,
                args.batch, args.seq)
    if not args.trace and recorded(args.output, cell.key):
        print(f"{cell.key}: already recorded, skipped", flush=True)
        return 0

    import jax

    if jax.default_backend() != "tpu":
        raise RuntimeError(f"requires TPU; got {jax.default_backend()!r}.")
    if (cell.impl in ("rpa", "splash")
            and getattr(args, f"{cell.impl}_table") is None):
        setattr(args, f"{cell.impl}_table",
                tuned_table(cell.impl, jax.devices()[0].device_kind))

    record = run(cell, args)
    write_record(args.output, record)
    if record["status"] != "ok":
        print(f"{cell.key}: {record['status']} {record.get('error')}",
              flush=True)
        return 1
    line = f"{cell.key}: block {record['median_ms']:.3f} ms"
    if "segments_ms" in record:
        parts = "  ".join(f"{name} {value:.3f}"
                          for name, value in record["segments_ms"].items())
        line += (f" | {parts} | kernel {record['kernel_tflops']:.1f}"
                 " TFLOP/s")
    print(line, flush=True)
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--impl", choices=IMPLS, required=True)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--heads-k", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--mask", choices=("causal", "full"), default="causal")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=16384)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--trace", type=pathlib.Path,
                        help="directory for the profiler traces, one "
                             "subdirectory per cell")
    parser.add_argument("--rpa-table", type=pathlib.Path,
                        help="default: rpa_tuned_<chip>.json next to this "
                             "file, for the chip the cell runs on")
    parser.add_argument("--splash-table", type=pathlib.Path,
                        help="default: splash_tuned_<chip>.json next to this "
                             "file, for the chip the cell runs on")
    parser.add_argument("--rpa-source", type=pathlib.Path,
                        default=DEFAULT_RPA_SOURCE)
    parser.add_argument("--splash-source", type=pathlib.Path,
                        default=DEFAULT_SPLASH_SOURCE)
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
