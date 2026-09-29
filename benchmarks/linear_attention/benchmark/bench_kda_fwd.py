"""Latency / throughput sweep of the vendored KDA chunked forward baseline.

--impl times chunk_kda_fwd end to end, including the gather / scatter glue
around its four pallas_calls, since that glue is part of the baseline cost.
--pkgs times each package's fused_conv1d_kda (conv1d + silu + KDA) against the
first package.

    PYTHONPATH=. uv run --no-sync python benchmarks/linear_attention/benchmark/bench_kda_fwd.py \
        --pkgs flywheel_tpu.linear_attention.kda --compute-chunk-size 64 --mixed-tile-size 128

    PYTHONPATH=. uv run --no-sync python benchmarks/linear_attention/benchmark/bench_kda_fwd.py \
        --impl baseline \
        --output benchmarks/linear_attention/benchmark/results/kda_fwd_v6e.json
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import sys
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp

from benchmarks.common.linear_attention import (
    BASELINE_KERNEL_DIR,
    PARAM_SCALE,
    Header,
    KernelState,
    Row,
    kda_fwd_flops,
    print_speedup_table,
    report_throughput,
)
from benchmarks.common.timing import time_stateful

sys.path.insert(0, str(BASELINE_KERNEL_DIR))

IMPLS = {"baseline": "kda"}
RAW_GATE_SCALE = 0.05


def make_inputs(
    key: jax.Array,
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    value_dim: int,
    nseqs: int,
    dtype: jnp.dtype,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Packed q, k, v, g, beta and cu_seqlens of nseqs equal-length sequences."""
    assert num_tokens % nseqs == 0, f"T={num_tokens} not divisible by nseqs={nseqs}"
    q_key, k_key, v_key, g_key, beta_key = jax.random.split(key, 5)

    # Note (david): KDA l2-normalizes q and k outside this kernel
    # (use_qk_l2norm_in_kernel=False), and the delta-rule triangular solve is
    # only stable for ||k|| <= 1, so the inputs match that contract.
    def _l2_normalize(features: jax.Array) -> jax.Array:
        return features / jnp.linalg.norm(features.astype(jnp.float32), axis=-1, keepdims=True)

    q = _l2_normalize(
        jax.random.normal(q_key, (1, num_tokens, num_heads, head_dim), jnp.float32)
    ).astype(dtype)
    k = _l2_normalize(
        jax.random.normal(k_key, (1, num_tokens, num_heads, head_dim), jnp.float32)
    ).astype(dtype)
    v = jax.random.normal(v_key, (1, num_tokens, num_heads, value_dim), dtype)
    # Note (david): the gate cumsum expects raw per-key-dim log-decay gates,
    # which are small negatives.
    g = -RAW_GATE_SCALE * jnp.abs(
        jax.random.normal(g_key, (1, num_tokens, num_heads, head_dim), jnp.float32)
    )
    beta = jax.nn.sigmoid(jax.random.normal(beta_key, (1, num_tokens, num_heads), jnp.float32))
    cu_seqlens = jnp.arange(nseqs + 1, dtype=jnp.int32) * (num_tokens // nseqs)
    return q, k, v, g, beta, cu_seqlens


def make_fused_inputs(
    key: jax.Array,
    num_tokens: int,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    nseqs: int,
    dtype: jnp.dtype,
) -> tuple[dict[str, jax.Array | None], KernelState]:
    """Fresh-prefill inputs of fused_conv1d_kda and its zeroed conv / recurrent state."""
    assert num_tokens % nseqs == 0, f"T={num_tokens} not divisible by nseqs={nseqs}"
    conv_dim = n_kq * d_k * 2 + n_v * d_v
    qkv_key, b_key, g_key, conv_key, a_log_key, dt_bias_key = jax.random.split(key, 6)
    # Note (david): the kernel l2-normalizes q / k and applies sigmoid(b) and
    # the softplus-based gating on g itself, so raw activations are valid.
    qkv = jax.random.normal(qkv_key, (num_tokens, conv_dim), dtype)
    b = jax.random.normal(b_key, (num_tokens, n_v), dtype)
    # Note (david): KDA gates one value per key channel, so g and dt_bias are
    # d_k wide per value head.
    gate_dim = n_v * d_k
    g = jax.random.normal(g_key, (num_tokens, gate_dim), dtype)
    conv_weight = jax.random.normal(conv_key, (conv_dim, 1, kernel_size), jnp.float32) * PARAM_SCALE
    a_log = jax.random.normal(a_log_key, (n_v,), jnp.float32) * PARAM_SCALE
    dt_bias = jax.random.normal(dt_bias_key, (gate_dim,), jnp.float32) * PARAM_SCALE

    seq_len = num_tokens // nseqs
    query_start_loc = jnp.arange(nseqs + 1, dtype=jnp.int32) * seq_len
    # Note (david): a fresh prefill has seq_lens equal to the query lengths.
    seq_lens = jnp.full((nseqs,), seq_len, jnp.int32)
    # Note (david): state slot 0 is the null slot.
    state_indices = jnp.arange(1, nseqs + 1, dtype=jnp.int32)
    # Note (david): no decode or mixed sequences; every sequence is a prefill.
    distribution = jnp.array([0, nseqs, nseqs], jnp.int32)

    conv_state = jnp.zeros((nseqs + 1, kernel_size - 1, conv_dim), jnp.float32)
    recurrent_state = jnp.zeros((nseqs + 1, n_v, d_k, d_v), jnp.float32)
    fixed_inputs = {
        "qkv": qkv, "b": b, "g": g, "conv_weight": conv_weight, "conv_bias": None,
        "a_log": a_log, "dt_bias": dt_bias, "query_start_loc": query_start_loc,
        "state_indices": state_indices, "distribution": distribution,
        "seq_lens": seq_lens, "read_state_indices": state_indices
    }
    return fixed_inputs, (conv_state, recurrent_state)


def bench_fused(args: argparse.Namespace) -> tuple[Header, list[Row]]:
    n_kq, n_v, d_k, d_v = args.kq_heads, args.v_heads, args.head_dim, args.value_dim
    dtype = jnp.dtype(args.dtype)
    seqlens = [int(seqlen) for seqlen in args.seqlens.split(",")]
    nseqs_list = [int(nseqs) for nseqs in args.nseqs.split(",")]
    pkgs = args.pkgs.split(",")
    wrappers = []
    for pkg in pkgs:
        module = importlib.import_module(f"{pkg}.wrapper")
        wrappers.append((pkg, module, module.fused_conv1d_kda))

    print(f"config: n_kq={n_kq} n_v={n_v} d_k={d_k} d_v={d_v} "
          f"kernel_size={args.kernel_size} tile={args.mixed_tile_size} "
          f"compute_chunk={args.compute_chunk_size} dtype={args.dtype}")

    def _make_step(
        kernel: Callable[..., tuple[KernelState, jax.Array]],
        fixed_inputs: dict[str, jax.Array | None],
    ) -> Callable[..., tuple[KernelState, jax.Array]]:
        def _step(
            conv_state: jax.Array, recurrent_state: jax.Array
        ) -> tuple[KernelState, jax.Array]:
            return kernel(
                fixed_inputs["qkv"], fixed_inputs["b"], fixed_inputs["g"],
                conv_state, recurrent_state,
                fixed_inputs["conv_weight"], fixed_inputs["conv_bias"],
                fixed_inputs["a_log"], fixed_inputs["dt_bias"],
                fixed_inputs["query_start_loc"], fixed_inputs["state_indices"],
                fixed_inputs["distribution"], fixed_inputs["seq_lens"],
                fixed_inputs["read_state_indices"],
                n_kq=n_kq, n_v=n_v, d_k=d_k, d_v=d_v,
                kernel_size=args.kernel_size,
                mixed_tile_size=args.mixed_tile_size,
                compute_chunk_size=args.compute_chunk_size)
        return _step

    rows = []
    for num_tokens, nseqs in itertools.product(seqlens, nseqs_list):
        if num_tokens % nseqs:
            continue
        fixed_inputs, zero_state = make_fused_inputs(
            jax.random.PRNGKey(0), num_tokens, n_kq, n_v, d_k, d_v,
            args.kernel_size, nseqs, dtype)
        # Note (david): the fused conv1d adds only ~2 * T * dim * kernel_size
        # FLOPs, and the chunked-KDA cost scales with the math chunk, not the
        # DMA tile.
        flops = kda_fwd_flops(num_tokens, n_v, d_k, d_v,
                              min(args.compute_chunk_size, num_tokens), nseqs=nseqs)
        for pkg, _, kernel in wrappers:
            # Note (david): buffers are donated, so every package needs its own
            # zeroed state rather than the one the previous package consumed.
            fresh_state = tuple(jnp.zeros_like(buffer) for buffer in zero_state)
            median_ms, out = time_stateful(_make_step(kernel, fixed_inputs), fresh_state,
                                           args.warmup, args.iters, args.repeats)
            assert bool(jnp.isfinite(out.astype(jnp.float32)).all()), \
                f"non-finite output from {pkg} at T={num_tokens} nseqs={nseqs}"
            rows.append({
                "T": num_tokens, "nseqs": nseqs, "pkg": pkg, "median_ms": median_ms,
                "mixed_tile_size": args.mixed_tile_size,
                **report_throughput(num_tokens, nseqs, median_ms, flops, f" {pkg:>16s}"),
            })

    if len(pkgs) > 1:
        print_speedup_table(rows, pkgs, seqlens, nseqs_list)

    header = {
        "kernels": {pkg: f"{module.__name__}.{kernel.__name__}"
                    for pkg, module, kernel in wrappers},
        "config": {"kq_heads": n_kq, "v_heads": n_v, "head_dim": d_k,
                   "value_dim": d_v, "kernel_size": args.kernel_size,
                   "mixed_tile_size": args.mixed_tile_size,
                   "compute_chunk_size": args.compute_chunk_size,
                   "dtype": args.dtype},
    }
    return header, rows


def bench_chunked(args: argparse.Namespace) -> tuple[Header, list[Row]]:
    num_heads, head_dim, value_dim = args.heads, args.head_dim, args.value_dim
    chunk_size = args.chunk_size
    dtype = jnp.dtype(args.dtype)
    scale = 1.0 / head_dim**0.5
    seqlens = [int(seqlen) for seqlen in args.seqlens.split(",")]
    nseqs_list = [int(nseqs) for nseqs in args.nseqs.split(",")]
    chunk_kda_fwd = importlib.import_module(IMPLS[args.impl]).chunk_kda_fwd

    print(f"config: impl={args.impl} H={num_heads} K={head_dim} V={value_dim} BT={chunk_size} "
          f"dtype={args.dtype} scale={scale:.4f}")

    def _step(*inputs: jax.Array) -> tuple[KernelState, jax.Array]:
        q, k, v, g, beta, cu_seqlens = inputs
        # Note (david): only o is kept; the rest are training intermediates.
        o = chunk_kda_fwd(
            q, k, v, g, beta, scale,
            initial_state=None, output_final_state=False, cu_seqlens=cu_seqlens,
            chunk_size=chunk_size)[0]
        return inputs, o

    rows = []
    for num_tokens, nseqs in itertools.product(seqlens, nseqs_list):
        if num_tokens % nseqs or (num_tokens // nseqs) % chunk_size:
            continue
        inputs = make_inputs(jax.random.PRNGKey(0), num_tokens, num_heads, head_dim,
                             value_dim, nseqs, dtype)
        median_ms, o = time_stateful(_step, inputs, args.warmup, args.iters, args.repeats)
        assert bool(jnp.isfinite(o.astype(jnp.float32)).all()), \
            f"non-finite output at T={num_tokens} nseqs={nseqs}"
        flops = kda_fwd_flops(num_tokens, num_heads, head_dim, value_dim, chunk_size)
        rows.append({
            "T": num_tokens, "nseqs": nseqs, "median_ms": median_ms,
            **report_throughput(num_tokens, nseqs, median_ms, flops, ""),
        })

    header = {
        "config": {"impl": args.impl, "heads": num_heads, "head_dim": head_dim,
                   "value_dim": value_dim, "chunk_size": chunk_size, "dtype": args.dtype,
                   "scale": scale},
    }
    return header, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqlens", default="1024,2048,4096,8192,16384")
    parser.add_argument("--nseqs", default="1,8", help="sequences per packed batch")
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--dtype", default="bfloat16")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--impl", choices=sorted(IMPLS), default="baseline",
                      help="chunked KDA implementation (without conv1d)")
    mode.add_argument("--pkgs", help="fused KDA packages to time; the first is the baseline")
    parser.add_argument("--kq-heads", type=int, default=16, help="query/key heads for --pkgs")
    parser.add_argument("--v-heads", type=int, default=16, help="value heads for --pkgs")
    parser.add_argument("--kernel-size", type=int, default=4)
    parser.add_argument("--mixed-tile-size", type=int, default=128)
    parser.add_argument("--compute-chunk-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = jax.devices()[0].device_kind
    print(f"device: {device}")
    if args.pkgs is None:
        header, rows = bench_chunked(args)
    else:
        header, rows = bench_fused(args)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "device": device,
            **header,
            "timing": {"warmup": args.warmup, "iters": args.iters,
                       "repeats": args.repeats},
            "rows": rows,
        }, indent=2))
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
