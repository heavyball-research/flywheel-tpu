"""ReplaySSM decode of the fused GDN kernel against a token-by-token reference.

Each test runs many decode or speculative-verify steps, so the sequences cross
several flushes, and checks every step's outputs and, at the end, the state the
checkpoint and the committed records represent and that the bf16 shadow is
still the rounded checkpoint.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flywheel_tpu.linear_attention.gdn import wrapper
from tests.linear_attention.test_gdn import (
    CONV_DIM,
    KERNEL_SIZE,
    KQ_HEAD_DIM,
    MODEL_DIMS,
    N_KQ,
    N_V,
    STATIC,
    TOL,
    TPU_ONLY,
    V_HEAD_DIM,
    gdn_sequence_ref,
    random_weights,
)

pytestmark = TPU_ONLY
CONTEXT_LEN = 100


def _sequence_ref(*args, **kwargs):
    with jax.default_matmul_precision("highest"):
        return gdn_sequence_ref(*args, **kwargs)


sequence_ref = jax.jit(_sequence_ref, static_argnames=STATIC)


def materialize_state(checkpoint: np.ndarray, records: np.ndarray,
                      num_records: int) -> np.ndarray:
    """The state after folding the first num_records records into checkpoint,
    in f64."""
    num_lanes = records.shape[-1] - N_V * V_HEAD_DIM - N_KQ * KQ_HEAD_DIM
    assert num_lanes % 128 == 0, records.shape
    k_offset = N_V * V_HEAD_DIM
    g_offset = k_offset + N_KQ * KQ_HEAD_DIM
    state = checkpoint.astype(np.float64)
    for row in records[:num_records].astype(np.float64):
        u = row[:k_offset].reshape(N_V, V_HEAD_DIM)
        k = np.repeat(row[k_offset:g_offset].reshape(N_KQ, KQ_HEAD_DIM),
                      N_V // N_KQ, axis=0)
        decay = np.exp(row[g_offset:g_offset + N_V])
        state = state * decay[:, None, None] + k[:, :, None] * u[:, None, :]
    return state


def run_replay(
    num_seqs: int,
    num_spec_tokens: int,
    num_steps: int,
    replay_window: int,
    draw_lengths: Callable[[np.random.Generator], int],
    draw_accepted: Callable[[np.random.Generator, int], int],
    seed: int,
    record_dtype: jnp.dtype = jnp.float32,
    prefill_len: int = 0,
) -> None:
    window = num_spec_tokens + 1
    rng = np.random.default_rng(seed)
    keys = iter(jax.random.split(jax.random.key(seed), 11 + 3 * num_steps))
    weights = random_weights(keys)

    # Note (david): each sequence owns window consecutive slots after the null
    # block 0, as the conv state checkpoints one per window position; the
    # recurrent checkpoint and the records live in the base slot.
    num_slots = 1 + num_seqs * window
    state_indices = jnp.arange(num_seqs, dtype=jnp.int32) * window + 1
    conv_state = jax.random.normal(next(keys),
                                   (num_slots, KERNEL_SIZE - 1, CONV_DIM))
    recurrent_state = 0.1 * jax.random.normal(
        next(keys), (num_slots, N_V, KQ_HEAD_DIM, V_HEAD_DIM))
    recurrent_shadow = recurrent_state.astype(jnp.bfloat16)
    record_dim = wrapper.replay_record_dim(N_KQ, N_V, KQ_HEAD_DIM, V_HEAD_DIM)
    # Note (david): NaN records catch any read past the committed ones.
    replay_state = jnp.full((num_slots, replay_window, record_dim), jnp.nan,
                            record_dtype)
    replay_lens = jnp.zeros((num_seqs, ), jnp.int32)
    read_offsets = jnp.zeros((num_seqs, ), jnp.int32)

    initial_conv = [np.asarray(conv_state[slot]) for slot in state_indices]
    initial_rec = [np.asarray(recurrent_state[slot]) for slot in state_indices]
    history = [[] for _ in range(num_seqs)]
    max_len = prefill_len + (num_steps + 1) * window

    def _reference(seq: int, seq_tokens: list) -> tuple[jax.Array, jax.Array]:
        """Per-token states and outputs of seq_tokens from seq's initial
        state; zero tokens pad it to max_len so one compile serves all."""
        num_pad = max_len - len(seq_tokens)
        stacked = [
            jnp.pad(jnp.stack([token[i] for token in seq_tokens]),
                    ((0, num_pad), (0, 0))) for i in range(3)
        ]
        _, states, outs = sequence_ref(*stacked, initial_conv[seq],
                                       initial_rec[seq],
                                       weights["conv_weight"],
                                       weights["conv_bias"], weights["a_log"],
                                       weights["dt_bias"], **MODEL_DIMS)
        return states[:len(seq_tokens)], outs[:len(seq_tokens)]

    def _step(lengths: list[int], num_rows: int, num_decode: int,
              accept: Callable[[int], int]) -> None:
        """One replay call over sequences of lengths, the first num_decode
        of them decode (verify) windows and the rest prefills."""
        nonlocal conv_state, recurrent_state, recurrent_shadow
        nonlocal replay_state, replay_lens, read_offsets
        q_loc = jnp.array(np.concatenate([[0], np.cumsum(lengths)]),
                          jnp.int32)
        qkv = jax.random.normal(next(keys), (num_rows, CONV_DIM)).astype(
            jnp.bfloat16)
        b = jax.random.normal(next(keys), (num_rows, N_V)).astype(jnp.bfloat16)
        a = jax.random.normal(next(keys), (num_rows, N_V)).astype(jnp.bfloat16)
        seq_lens = jnp.array([CONTEXT_LEN + len(h) + n
                              for h, n in zip(history, lengths)], jnp.int32)
        distribution = jnp.array([num_decode, num_seqs, num_seqs], jnp.int32)

        (conv_state, recurrent_state, recurrent_shadow,
         replay_state), out = wrapper.fused_conv1d_gdn_replay(
            qkv, b, a, conv_state, recurrent_state, recurrent_shadow,
            replay_state, weights["conv_weight"], weights["conv_bias"],
            weights["a_log"], weights["dt_bias"], q_loc, state_indices,
            distribution, seq_lens, state_indices, replay_lens, read_offsets,
            replay_window=replay_window, num_spec_tokens=num_spec_tokens,
            **MODEL_DIMS)

        accepted = []
        for seq, length in enumerate(lengths):
            start = int(q_loc[seq])
            tokens = [(np.asarray(qkv[row], np.float32),
                       np.asarray(b[row], np.float32),
                       np.asarray(a[row], np.float32))
                      for row in range(start, start + length)]
            _, ref_out = _reference(seq, history[seq] + tokens)
            np.testing.assert_allclose(np.asarray(out[start:start + length],
                                                  np.float32),
                                       np.asarray(ref_out[-length:]),
                                       rtol=TOL,
                                       atol=TOL)
            num_accepted = accept(length)
            history[seq] += tokens[:num_accepted]
            accepted.append(num_accepted)

        accepted = jnp.array(accepted, jnp.int32)
        replay_lens = wrapper.next_replay_lens(replay_lens, q_loc, seq_lens,
                                               state_indices, state_indices,
                                               distribution, accepted,
                                               replay_window)
        # Note (david): only a verify window leaves one conv checkpoint per
        # position; a prefill leaves its last one at the base slot.
        is_decode = jnp.arange(num_seqs) < num_decode
        read_offsets = jnp.where(is_decode, accepted - 1, 0)

    if prefill_len:
        # Note (david): a prefill runs the PER_SEQ kernel, which must leave
        # the shadow of the checkpoint it writes for the decode steps after it.
        _step([prefill_len] * num_seqs, num_seqs * prefill_len, 0,
              lambda length: length)
    for _ in range(num_steps):
        # Note (david): a fixed token count keeps one compiled kernel; rows
        # past the last window are unused.
        _step([draw_lengths(rng) for _ in range(num_seqs)], num_seqs * window,
              num_seqs, lambda length: draw_accepted(rng, length))

    for seq in range(num_seqs):
        ref_states, _ = _reference(seq, history[seq])
        slot = int(state_indices[seq])
        state = materialize_state(np.asarray(recurrent_state[slot]),
                                  np.asarray(replay_state[slot],
                                             np.float32),
                                  int(replay_lens[seq]))
        np.testing.assert_allclose(state, np.asarray(ref_states[-1]),
                                   rtol=TOL, atol=TOL)
    np.testing.assert_array_equal(
        np.asarray(recurrent_shadow, np.float32),
        np.asarray(recurrent_state.astype(jnp.bfloat16), np.float32))


RECORD_DTYPES = pytest.mark.parametrize("record_dtype",
                                        [jnp.float32, jnp.bfloat16],
                                        ids=["f32", "bf16"])


@RECORD_DTYPES
@pytest.mark.parametrize("replay_window", [8, 16])
def test_replay_decode(replay_window: int, record_dtype: jnp.dtype) -> None:
    """One token per step for 40 steps: every sequence flushes several times;
    six sequences span two decode tiles."""
    run_replay(num_seqs=6, num_spec_tokens=0, num_steps=40,
               replay_window=replay_window,
               draw_lengths=lambda rng: 1,
               draw_accepted=lambda rng, length: length, seed=0,
               record_dtype=record_dtype)


@RECORD_DTYPES
@pytest.mark.parametrize("num_spec_tokens", [1, 3])
def test_replay_speculative(num_spec_tokens: int,
                            record_dtype: jnp.dtype) -> None:
    """Verify windows of random length with random acceptance: rejected
    records are dropped by moving replay_lens, and flushes fire on overflow."""
    run_replay(num_seqs=5, num_spec_tokens=num_spec_tokens, num_steps=20,
               replay_window=16,
               draw_lengths=lambda rng: int(
                   rng.integers(1, num_spec_tokens + 2)),
               draw_accepted=lambda rng, length: int(
                   rng.integers(1, length + 1)),
               seed=2 + num_spec_tokens, record_dtype=record_dtype)


@RECORD_DTYPES
def test_replay_after_prefill(record_dtype: jnp.dtype) -> None:
    """A prefill that continues the context, then decode: the PER_SEQ kernel
    writes the checkpoint and its shadow, which the decode steps read."""
    run_replay(num_seqs=3, num_spec_tokens=0, num_steps=20, replay_window=8,
               draw_lengths=lambda rng: 1,
               draw_accepted=lambda rng, length: length, seed=7,
               record_dtype=record_dtype, prefill_len=96)
