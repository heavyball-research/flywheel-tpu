"""Latency / throughput sweep of the fused conv1d + GDN kernel packages.

Times a fresh prefill through fused_conv1d_gdn, chaining the donated state
buffers from call to call as serving does. Defaults mirror bench_kda_fwd.py so
tokens/s is comparable. The first of --pkgs is the baseline.

    PYTHONPATH=. uv run --no-sync python benchmarks/linear_attention/benchmark/bench_gdn_fwd.py \
        --pkgs gdn_v3,flywheel_tpu.linear_attention.gdn \
        --output benchmarks/linear_attention/benchmark/results/gdn_v3_fwd_v6e.json
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp

from benchmarks.common.linear_attention import (
    BASELINE_KERNEL_DIR,
    PARAM_SCALE,
    kda_fwd_flops,
    print_speedup_table,
    report_throughput,
)
from benchmarks.common.timing import time_stateful

sys.path.insert(0, str(BASELINE_KERNEL_DIR))

KernelState = tuple[jax.Array, jax.Array]


def make_inputs(
    key: jax.Array,
    num_tokens: int,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    num_seqs: int,
    dtype: jnp.dtype,
) -> tuple[dict[str, jax.Array | None], KernelState]:
    """Fresh-prefill inputs of fused_conv1d_gdn and its zeroed conv / recurrent state."""
    dim = n_kq * d_k * 2 + n_v * d_v
    qkv_key, b_key, a_key, weight_key, a_log_key, dt_bias_key = (
        jax.random.split(key, 6))
    # Note (david): the activations stay raw because the kernel l2-normalizes q
    # and k and applies the sigmoid(b) / softplus gating on a itself.
    qkv = jax.random.normal(qkv_key, (num_tokens, dim), dtype)
    b = jax.random.normal(b_key, (num_tokens, n_v), dtype)
    a = jax.random.normal(a_key, (num_tokens, n_v), dtype)
    conv_weight = jax.random.normal(weight_key, (dim, 1, kernel_size),
                                    jnp.float32) * PARAM_SCALE
    a_log = jax.random.normal(a_log_key, (n_v, ), jnp.float32) * PARAM_SCALE
    dt_bias = jax.random.normal(dt_bias_key, (n_v, ),
                                jnp.float32) * PARAM_SCALE

    # Note (david): seq_lens == query_lens makes every sequence a fresh prefill
    # with a zero initial state, the distribution runs them all as PER_SEQ
    # prefill, and slot 0 is the null block.
    seq_len = num_tokens // num_seqs
    query_start_loc = jnp.arange(num_seqs + 1, dtype=jnp.int32) * seq_len
    seq_lens = jnp.full((num_seqs, ), seq_len, jnp.int32)
    state_indices = jnp.arange(1, num_seqs + 1, dtype=jnp.int32)
    distribution = jnp.array([0, num_seqs, num_seqs], jnp.int32)

    conv_state = jnp.zeros((num_seqs + 1, kernel_size - 1, dim), jnp.float32)
    recurrent_state = jnp.zeros((num_seqs + 1, n_v, d_k, d_v), jnp.float32)
    fixed_inputs = {
        "qkv": qkv, "b": b, "a": a, "conv_weight": conv_weight,
        "conv_bias": None, "a_log": a_log, "dt_bias": dt_bias,
        "query_start_loc": query_start_loc, "state_indices": state_indices,
        "distribution": distribution, "seq_lens": seq_lens,
        "read_state_indices": state_indices,
    }
    return fixed_inputs, (conv_state, recurrent_state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqlens", default="1024,2048,4096,8192,16384")
    parser.add_argument("--nseqs", default="1,8",
                        help="sequences per packed batch")
    parser.add_argument("--kq-heads", type=int, default=16)
    parser.add_argument("--v-heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--kernel-size", type=int, default=4)
    parser.add_argument("--mixed-tile-size", type=int, default=128)
    parser.add_argument("--compute-chunk-size", type=int, default=64,
                        help="chunked-GDN math step; only passed to packages "
                             "whose fused_conv1d_gdn takes it")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--pkgs", default="gdn_v3,flywheel_tpu.linear_attention.gdn",
                        help="kernel packages to time; the first is the "
                             "baseline")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    n_kq, n_v = args.kq_heads, args.v_heads
    d_k, d_v = args.head_dim, args.value_dim
    dtype = jnp.dtype(args.dtype)
    seqlens = [int(seqlen) for seqlen in args.seqlens.split(",")]
    nseqs_list = [int(num_seqs) for num_seqs in args.nseqs.split(",")]
    pkgs = args.pkgs.split(",")
    wrappers = []
    for pkg in pkgs:
        module = importlib.import_module(f"{pkg}.wrapper")
        wrappers.append((pkg, module, module.fused_conv1d_gdn))

    device = jax.devices()[0].device_kind
    print(f"device: {device}")
    print(f"config: n_kq={n_kq} n_v={n_v} d_k={d_k} d_v={d_v} "
          f"kernel_size={args.kernel_size} tile={args.mixed_tile_size} "
          f"compute_chunk={args.compute_chunk_size} dtype={args.dtype}")

    def _make_step(
        kernel: Callable[..., tuple[KernelState, jax.Array]],
        fixed_inputs: dict[str, jax.Array | None],
        mixed_tile_size: int,
        extra_kwargs: dict[str, int],
    ) -> Callable[..., tuple[KernelState, jax.Array]]:
        def _step(conv_state: jax.Array, recurrent_state: jax.Array
                  ) -> tuple[KernelState, jax.Array]:
            return kernel(
                fixed_inputs["qkv"], fixed_inputs["b"], fixed_inputs["a"],
                conv_state, recurrent_state, fixed_inputs["conv_weight"],
                fixed_inputs["conv_bias"], fixed_inputs["a_log"],
                fixed_inputs["dt_bias"], fixed_inputs["query_start_loc"],
                fixed_inputs["state_indices"], fixed_inputs["distribution"],
                fixed_inputs["seq_lens"], fixed_inputs["read_state_indices"],
                n_kq=n_kq, n_v=n_v, d_k=d_k, d_v=d_v,
                kernel_size=args.kernel_size,
                mixed_tile_size=mixed_tile_size, **extra_kwargs)
        return _step

    rows = []
    for num_tokens in seqlens:
        for num_seqs in nseqs_list:
            if num_tokens % num_seqs:
                continue
            fixed_inputs, zero_state = make_inputs(
                jax.random.PRNGKey(0), num_tokens, n_kq, n_v, d_k, d_v,
                args.kernel_size, num_seqs, dtype)

            # Note (david): the fused conv1d adds only ~2 * T * dim *
            # kernel_size FLOPs, and the chunked GDN cost scales with the math
            # chunk, not the DMA tile.
            flops = kda_fwd_flops(num_tokens, n_v, d_k, d_v,
                                  min(args.compute_chunk_size, num_tokens),
                                  nseqs=num_seqs)
            for pkg, _, kernel in wrappers:
                # Note (david): a package without the tile / compute-chunk
                # split has one knob, its DMA tile is its math chunk, so its
                # equivalent configuration is a tile of compute_chunk_size rows
                # (its [heads, tile, tile] matrices take O(tile**2) VMEM).
                if "compute_chunk_size" in inspect.signature(kernel).parameters:
                    tile_size = args.mixed_tile_size
                    extra_kwargs = {
                        "compute_chunk_size": args.compute_chunk_size
                    }
                else:
                    tile_size = args.compute_chunk_size
                    extra_kwargs = {}
                # Note (david): the buffers are donated, so every package needs
                # its own zeroed state, not the one the previous package
                # consumed.
                fresh_state = tuple(
                    jnp.zeros_like(state) for state in zero_state)
                median_ms, out = time_stateful(
                    _make_step(kernel, fixed_inputs, tile_size, extra_kwargs),
                    fresh_state, args.warmup, args.iters, args.repeats)
                assert bool(jnp.isfinite(out.astype(jnp.float32)).all()), (
                    f"non-finite output from {pkg} at T={num_tokens} "
                    f"nseqs={num_seqs}")
                rows.append({
                    "T": num_tokens, "nseqs": num_seqs, "pkg": pkg,
                    "median_ms": median_ms,
                    "mixed_tile_size": tile_size,
                    **report_throughput(num_tokens, num_seqs, median_ms, flops,
                                        f" {pkg:>16s}"),
                })

    if len(pkgs) > 1:
        print_speedup_table(rows, pkgs, seqlens, nseqs_list)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "device": device,
            "kernels": {pkg: f"{module.__name__}.{kernel.__name__}"
                        for pkg, module, kernel in wrappers},
            "config": {"kq_heads": n_kq, "v_heads": n_v, "head_dim": d_k,
                       "value_dim": d_v, "kernel_size": args.kernel_size,
                       "mixed_tile_size": args.mixed_tile_size,
                       "compute_chunk_size": args.compute_chunk_size,
                       "dtype": args.dtype},
            "timing": {"warmup": args.warmup, "iters": args.iters,
                       "repeats": args.repeats},
            "rows": rows,
        }, indent=2))
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
