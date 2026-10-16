"""Long-run output drift of conventional vs ReplaySSM decode (f32 / bf16
records) against the f32 token-by-token reference.

Runs num_steps one-token decode steps per sequence and reports the max
absolute output error against the reference per block of steps, so error
that enters the recurrence at each flush shows up as growth over the run.

    PYTHONPATH=. uv run --no-sync python \
        benchmarks/linear_attention/benchmark/replay_drift.py --kernel gdn
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from flywheel_tpu.linear_attention.gdn import wrapper as gdn_wrapper
from flywheel_tpu.linear_attention.kda import wrapper as kda_wrapper
from tests.linear_attention import test_gdn_replay, test_kda_replay

MODES = ("conventional", "replay_f32", "replay_bf16")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", choices=("gdn", "kda"), default="gdn")
    parser.add_argument("--num-seqs", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=256)
    parser.add_argument("--replay-window", type=int, default=16)
    parser.add_argument("--gate-scale", type=float, default=1.0,
                        help="KDA raw gate scale, as in test_kda_replay")
    parser.add_argument("--block", type=int, default=32,
                        help="steps per reported error block")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.kernel == "gdn":
        wrapper = gdn_wrapper
        conventional_fn = gdn_wrapper.fused_conv1d_gdn
        replay_fn = gdn_wrapper.fused_conv1d_gdn_replay
        helpers = test_gdn_replay
        n_kq, n_v = helpers.N_KQ, helpers.N_V
        d_k, d_v = helpers.KQ_HEAD_DIM, helpers.V_HEAD_DIM
        dim = helpers.CONV_DIM
        model_dims = helpers.MODEL_DIMS
        ref_kwargs = model_dims
        gate_name = "a"
    else:
        wrapper = kda_wrapper
        conventional_fn = kda_wrapper.fused_conv1d_kda
        replay_fn = kda_wrapper.fused_conv1d_kda_replay
        helpers = test_kda_replay
        n_kq, n_v, d_k, d_v = (helpers.N_KQ, helpers.N_V, helpers.D_K,
                               helpers.D_V)
        dim = helpers.DIM
        model_dims = helpers.MODEL_DIMS
        ref_kwargs = dict(lower_bound=None)
        gate_name = "g"
    kernel_size = model_dims["kernel_size"]

    num_seqs, num_steps = args.num_seqs, args.num_steps
    keys = iter(jax.random.split(jax.random.key(0), 8 + 3 * num_steps))
    if args.kernel == "gdn":
        params = dict(
            conv_weight=jax.random.normal(next(keys),
                                          (dim, 1, kernel_size)),
            conv_bias=jax.random.normal(next(keys), (dim, )),
            a_log=jax.random.normal(next(keys), (n_v, )),
            dt_bias=jax.random.normal(next(keys), (n_v, )),
        )
    else:
        params = dict(
            conv_weight=jax.random.normal(next(keys),
                                          (dim, 1, kernel_size)),
            conv_bias=jax.random.normal(next(keys), (dim, )),
            a_log=jax.random.normal(next(keys), (n_v, )),
            dt_bias=jax.random.normal(next(keys), (n_v * d_k, )),
        )
    num_slots = 1 + num_seqs
    state_indices = jnp.arange(1, num_seqs + 1, dtype=jnp.int32)
    conv0 = jax.random.normal(next(keys), (num_slots, kernel_size - 1, dim))
    rec0 = 0.1 * jax.random.normal(next(keys), (num_slots, n_v, d_k, d_v))
    q_loc = jnp.arange(num_seqs + 1, dtype=jnp.int32)
    distribution = jnp.array([num_seqs] * 3, jnp.int32)
    record_dim = wrapper.replay_record_dim(n_kq, n_v, d_k, d_v)

    tokens = []
    for _ in range(num_steps):
        qkv = jax.random.normal(next(keys), (num_seqs, dim)).astype(
            jnp.bfloat16)
        b = jax.random.normal(next(keys), (num_seqs, n_v)).astype(jnp.bfloat16)
        if args.kernel == "gdn":
            gate = jax.random.normal(next(keys),
                                     (num_seqs, n_v)).astype(jnp.bfloat16)
        else:
            gate = (jax.random.normal(next(keys), (num_seqs, n_v * d_k)) *
                    args.gate_scale + args.gate_scale).astype(jnp.bfloat16)
        tokens.append((qkv, b, gate))

    ref_outs = []
    for seq in range(num_seqs):
        slot = int(state_indices[seq])
        seq_tokens = [jnp.stack([t[i][seq] for t in tokens]).astype(
            jnp.float32) for i in range(3)]
        _, _, out = helpers.sequence_ref(*seq_tokens, conv0[slot], rec0[slot],
                                         **params, **ref_kwargs)
        ref_outs.append(np.asarray(out, np.float32))
    ref_outs = np.stack(ref_outs, axis=1)

    results = {}
    for mode in MODES:
        conv, rec = conv0.copy(), rec0.copy()
        shadow = rec0.astype(jnp.bfloat16)
        replay_lens = jnp.zeros((num_seqs, ), jnp.int32)
        record_dtype = jnp.bfloat16 if mode == "replay_bf16" else jnp.float32
        replay = jnp.zeros((num_slots, args.replay_window, record_dim),
                           record_dtype)
        errors = []
        for step, (qkv, b, gate) in enumerate(tokens):
            seq_lens = jnp.full((num_seqs, ), 100 + step + 1, jnp.int32)
            common = dict(qkv=qkv, b=b, **{gate_name: gate}, **params,
                          query_start_loc=q_loc, state_indices=state_indices,
                          distribution=distribution, seq_lens=seq_lens,
                          read_state_indices=state_indices, **model_dims)
            if mode == "conventional":
                (conv, rec), out = conventional_fn(conv_state=conv,
                                                   recurrent_state=rec,
                                                   **common)
            else:
                (conv, rec, shadow, replay), out = replay_fn(
                    conv_state=conv, recurrent_state=rec,
                    recurrent_shadow=shadow, replay_state=replay,
                    replay_lens=replay_lens, replay_window=args.replay_window,
                    **common)
                replay_lens = wrapper.next_replay_lens(
                    replay_lens, q_loc, seq_lens, state_indices,
                    state_indices, distribution,
                    jnp.ones((num_seqs, ), jnp.int32), args.replay_window)
            errors.append(
                float(np.max(np.abs(np.asarray(out, np.float32) -
                                    ref_outs[step]))))
        blocks = [max(errors[i:i + args.block])
                  for i in range(0, num_steps, args.block)]
        results[mode] = blocks
        print(f"{mode:13s} max |out - ref| per {args.block} steps: " +
              " ".join(f"{e:.2e}" for e in blocks), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(vars(args), output=None,
                                               results=results), indent=2))


if __name__ == "__main__":
    main()
