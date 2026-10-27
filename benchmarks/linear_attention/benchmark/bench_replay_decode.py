"""Decode / speculative-verify latency of the fused GDN and KDA kernels,
conventional against ReplaySSM.

Every sequence continues an existing context, so the whole batch runs as
BATCHED decode: one token per sequence, or a verify window of
num_spec_tokens + 1 tokens. The conventional kernel reads the recurrent state
and writes one checkpoint per window position; ReplaySSM reads the checkpoint
and writes it only on a flush. The replay steps chain replay_lens as serving
would, every window fully accepted (the most frequent flushes), and the
sequences start at staggered record counts so their flushes spread over the
steps as under continuous batching.

    PYTHONPATH=. uv run --no-sync python \
        benchmarks/linear_attention/benchmark/bench_replay_decode.py \
        --kernel gdn
"""

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path

import jax
import jax.numpy as jnp

from benchmarks.common.linear_attention import PARAM_SCALE
from benchmarks.common.timing import time_stateful
from flywheel_tpu.linear_attention.gdn import wrapper as gdn_wrapper
from flywheel_tpu.linear_attention.kda import wrapper as kda_wrapper

F32_BYTES = 4
CONTEXT_LEN = 1024


def make_inputs(
    key: jax.Array,
    kernel: str,
    num_seqs: int,
    window: int,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
) -> dict[str, jax.Array | None]:
    dim = n_kq * d_k * 2 + n_v * d_v
    num_tokens = num_seqs * window
    keys = iter(jax.random.split(key, 8))
    # Note (david): each sequence owns window consecutive slots after the null
    # block 0, the layout the speculative path checkpoints into.
    num_slots = 1 + num_seqs * window
    state_indices = 1 + jnp.arange(num_seqs, dtype=jnp.int32) * window
    inputs = dict(
        qkv=jax.random.normal(next(keys), (num_tokens, dim), jnp.bfloat16),
        b=jax.random.normal(next(keys), (num_tokens, n_v), jnp.bfloat16),
        conv_weight=jax.random.normal(next(keys), (dim, 1, kernel_size))
        * PARAM_SCALE,
        conv_bias=None,
        a_log=jax.random.normal(next(keys), (n_v,)) * PARAM_SCALE,
        query_start_loc=jnp.arange(num_seqs + 1, dtype=jnp.int32) * window,
        state_indices=state_indices,
        distribution=jnp.array([num_seqs] * 3, jnp.int32),
        seq_lens=jnp.full((num_seqs,), CONTEXT_LEN + window, jnp.int32),
        read_state_indices=state_indices,
        conv_state=jax.random.normal(
            next(keys), (num_slots, kernel_size - 1, dim)
        ),
        recurrent_state=jax.random.normal(
            next(keys), (num_slots, n_v, d_k, d_v)
        )
        * 0.1,
    )
    if kernel == "gdn":
        inputs["a"] = jax.random.normal(
            next(keys), (num_tokens, n_v), jnp.bfloat16
        )
        inputs["dt_bias"] = jax.random.normal(next(keys), (n_v,)) * PARAM_SCALE
    else:
        inputs["g"] = jax.random.normal(
            next(keys), (num_tokens, n_v * d_k), jnp.bfloat16
        )
        inputs["dt_bias"] = (
            jax.random.normal(next(keys), (n_v * d_k,)) * PARAM_SCALE
        )
    return inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", choices=("gdn", "kda"), default="gdn")
    parser.add_argument("--batch-sizes", default="16,64,128,256")
    parser.add_argument(
        "--spec-tokens",
        default="0,3",
        help="num_spec_tokens values; 0 is plain decode",
    )
    parser.add_argument("--replay-window", type=int, default=16)
    parser.add_argument(
        "--record-dtype", choices=("float32", "bfloat16"), default="float32"
    )
    parser.add_argument("--kq-heads", type=int, default=16)
    parser.add_argument("--v-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--kernel-size", type=int, default=4)
    parser.add_argument("--decode-tile-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.kernel == "gdn":
        wrapper = gdn_wrapper
        conventional_fn = gdn_wrapper.fused_conv1d_gdn
        replay_fn = gdn_wrapper.fused_conv1d_gdn_replay
    else:
        wrapper = kda_wrapper
        conventional_fn = kda_wrapper.fused_conv1d_kda
        replay_fn = kda_wrapper.fused_conv1d_kda_replay
    n_kq, n_v = args.kq_heads, args.v_heads
    d_k = d_v = args.head_dim
    replay_window = args.replay_window
    record_dtype = jnp.dtype(args.record_dtype)
    state_bytes = n_v * d_k * d_v * F32_BYTES
    layout = wrapper.replay_layout(n_kq, n_v, d_k, d_v)
    print(f"device: {jax.devices()[0].device_kind} kernel: {args.kernel}")
    print(
        f"n_kq={n_kq} n_v={n_v} d={d_k} state/seq={state_bytes >> 20} MiB "
        f"replay_window={replay_window} "
        f"record={layout.record_dim * record_dtype.itemsize} B "
        f"{record_dtype} gate={layout.gate_dim * F32_BYTES} B float32"
    )
    model_dims = dict(
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=args.kernel_size,
        decode_tile_size=args.decode_tile_size,
    )

    rows = []
    for num_spec in [int(s) for s in args.spec_tokens.split(",")]:
        window = num_spec + 1
        for num_seqs in [int(b) for b in args.batch_sizes.split(",")]:
            inputs = make_inputs(
                jax.random.key(0),
                args.kernel,
                num_seqs,
                window,
                n_kq,
                n_v,
                d_k,
                d_v,
                args.kernel_size,
            )
            conv_state = inputs.pop("conv_state")
            recurrent_state = inputs.pop("recurrent_state")
            read_offsets = jnp.full((num_seqs,), window - 1, jnp.int32)

            def _conventional(
                conv,
                rec,
                inputs=inputs,
                read_offsets=read_offsets,
                num_spec=num_spec,
            ):
                return conventional_fn(
                    conv_state=conv,
                    recurrent_state=rec,
                    **inputs,
                    read_offsets=read_offsets,
                    num_spec_tokens=num_spec,
                    **model_dims,
                )

            base_ms, _ = time_stateful(
                _conventional,
                (conv_state.copy(), recurrent_state.copy()),
                args.warmup,
                args.iters,
                args.repeats,
            )

            num_slots = recurrent_state.shape[0]
            replay_state = jnp.zeros(
                (num_slots, replay_window, layout.record_dim), record_dtype
            )
            replay_gates = jnp.zeros(
                (num_slots, replay_window, layout.gate_dim), jnp.float32
            )
            replay_lens = jnp.arange(num_seqs, dtype=jnp.int32) % (
                replay_window - window + 1
            )
            num_accepted = jnp.full((num_seqs,), window, jnp.int32)

            # Note (david): one jit with the states donated, so the commit's
            # few ops add no eager dispatches and the kernel's in-place
            # aliasing survives the outer call.
            @functools.partial(jax.jit, donate_argnums=(0, 1, 2, 3, 4))
            def _replay(
                conv,
                rec,
                replay,
                gates,
                lens,
                inputs=inputs,
                read_offsets=read_offsets,
                num_spec=num_spec,
                num_accepted=num_accepted,
            ):
                (conv, rec, replay, gates), out = replay_fn(
                    conv_state=conv,
                    recurrent_state=rec,
                    replay_state=replay,
                    replay_gates=gates,
                    replay_lens=lens,
                    **inputs,
                    read_offsets=read_offsets,
                    num_spec_tokens=num_spec,
                    replay_window=replay_window,
                    **model_dims,
                )
                lens = wrapper.next_replay_lens(
                    lens,
                    inputs["query_start_loc"],
                    inputs["seq_lens"],
                    inputs["state_indices"],
                    inputs["read_state_indices"],
                    inputs["distribution"],
                    num_accepted,
                    replay_window,
                )
                return (conv, rec, replay, gates, lens), out

            replay_ms, _ = time_stateful(
                _replay,
                (
                    conv_state,
                    wrapper.split_checkpoint(recurrent_state),
                    replay_state,
                    replay_gates,
                    replay_lens,
                ),
                args.warmup,
                args.iters,
                args.repeats,
            )

            row = dict(
                kernel=args.kernel,
                batch=num_seqs,
                window=window,
                replay_window=replay_window,
                record_dtype=args.record_dtype,
                conventional_ms=base_ms,
                replay_ms=replay_ms,
                speedup=base_ms / replay_ms,
                conventional_state_gbps=num_seqs
                * state_bytes
                * (1 + window)
                / base_ms
                / 1e6,
            )
            rows.append(row)
            print(
                f"B={num_seqs:4d} window={window} conventional "
                f"{base_ms:7.3f} ms ({row['conventional_state_gbps']:6.0f} "
                f"GB/s)  replay {replay_ms:7.3f} ms  "
                f"speedup {row['speedup']:5.2f}x",
                flush=True,
            )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
