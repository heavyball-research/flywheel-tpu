"""Checks fused_conv1d_kda against an eager f32 token-by-token KDA reference.

The strong gate cases push the per-step decay to tens of nats, so the
intra-chunk Aqk / Akk matrices span the whole f32 exponent range.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flywheel_tpu.linear_attention.kda.compute_kda import (
    ChunkKDA,
    block_diag,
    pack_width,
    unpack_lanes,
)
from flywheel_tpu.linear_attention.kda.wrapper import fused_conv1d_kda

STATIC = ["n_kq", "n_v", "d_k", "d_v", "kernel_size", "num_spec_tokens"]
N_KQ, N_V, D_K, D_V, KERNEL_SIZE = 2, 8, 128, 128, 4
DIM = N_KQ * D_K * 2 + N_V * D_V
TOL = dict(rtol=2e-2, atol=2e-2)
OPERAND_KEY_DIM = 16
OPERAND_TOL = dict(rtol=2e-2, atol=2e-3)
L2_NORM_EPS = 1e-6
NUM_SPEC_TOKENS = 4
SPEC_CONTEXT_LEN = 16
# Note (david): fused_conv1d_kda sizes its tiles from pltpu.get_tpu_info() and
# has no interpret mode.
TPU_ONLY = pytest.mark.skipif(jax.default_backend() != "tpu",
                              reason="the fused KDA kernel runs on TPU only")

# Note (david): calling through an outer jit drops fused_conv1d_kda's buffer
# donation, so the reference can still read conv_state and recurrent_state.
run_fused_kda = jax.jit(fused_conv1d_kda, static_argnames=STATIC)


@pytest.mark.parametrize("chunk_size, heads", [(4, 1), (16, 8), (64, 4), (128, 1)])
@pytest.mark.parametrize("gate_scale", [0.0, 1.0, 20.0])
def test_qk_kk_operands_match_reference(chunk_size: int, heads: int,
                                        gate_scale: float) -> None:
    rng = np.random.default_rng(0)
    q, k = rng.normal(
        size=(2, heads, chunk_size, OPERAND_KEY_DIM)).astype(np.float32)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    k /= np.linalg.norm(k, axis=-1, keepdims=True)
    beta = rng.uniform(size=(heads, chunk_size, 1)).astype(np.float32)
    g_cumsum = -np.cumsum(
        rng.uniform(size=k.shape).astype(np.float32) * gate_scale, axis=1)

    # Note (david): the constructor's cumsum primitive is TPU-only, so the chunk
    # is assembled by hand to exercise operand preparation and its consumer on
    # CPU.
    chunk = ChunkKDA.__new__(ChunkKDA)
    chunk.chunk = chunk_size
    chunk.pack = pack_width(chunk_size, heads)
    chunk.groups = heads // chunk.pack
    chunk.k_t_weight = block_diag(
        chunk.transposed(jnp.asarray(k)).astype(jnp.bfloat16), chunk_size)
    chunk.qk_kk_operands = chunk.prepare_qk_kk_operands(
        jnp.asarray(q), jnp.asarray(k * beta), jnp.asarray(k),
        jnp.asarray(g_cumsum))
    chunk.akk_parts, chunk.aqk_parts = [], []
    chunk.num_kkt_groups_done = 0
    for q_decayed, k_beta_decayed, k_decayed_t, _ in chunk.qk_kk_operands:
        for operand in (q_decayed, k_beta_decayed, k_decayed_t):
            assert np.isfinite(np.asarray(operand, dtype=np.float32)).all()
    coverage = sum(np.asarray(pair_mask, dtype=np.int32)
                   for _, _, _, pair_mask in chunk.qk_kk_operands)
    causal = np.tri(chunk_size, dtype=bool)
    np.testing.assert_array_equal(
        coverage, np.tile(causal, (1, 1, chunk.pack)))

    # Note (david): run_pipelined splits the head groups across calls, and a
    # repeated call must be a no-op.
    chunk.kkt_qk(1)
    chunk.kkt_qk(1)
    chunk.kkt_qk()
    gate_diff = g_cumsum[:, :, None, :] - g_cumsum[:, None, :, :]
    decay = np.exp(np.where(causal[None, :, :, None], gate_diff, 0))
    for lhs, parts in ((q, chunk.aqk_parts), (k * beta, chunk.akk_parts)):
        expected = np.sum(lhs[:, :, None, :] * k[:, None, :, :] * decay,
                          axis=-1) * causal
        actual = unpack_lanes(jnp.concatenate(parts, axis=0), chunk.pack)
        np.testing.assert_allclose(actual, expected, **OPERAND_TOL)


def kda_reference(
    qkv: jax.Array,
    b: jax.Array,
    g: jax.Array,
    conv_state: jax.Array,
    recurrent_state: jax.Array,
    conv_weight: jax.Array,
    conv_bias: jax.Array | None,
    a_log: jax.Array,
    dt_bias: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    read_slots: jax.Array,
    seq_lens: jax.Array,
    seqs: range,
    window: int,
    lower_bound: float | None = None,
) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
    """Eager KDA over the sequences in seqs; f32 in, f32 out.

    Sequence s starts from slot read_slots[s] (zero if it has no context) and
    checkpoints to state_indices[s] (window == 1: state after its last token)
    or state_indices[s] + t for every window position t.
    """
    new_conv_state = jnp.array(conv_state)
    new_recurrent_state = jnp.array(recurrent_state)
    out = jnp.zeros((qkv.shape[0], N_V * D_V), jnp.float32)
    decay_scale = jnp.exp(a_log)[:, None]
    dt_bias = dt_bias.reshape(N_V, D_K)
    key_dim = N_KQ * D_K
    v_per_kq_head = N_V // N_KQ

    def _l2_normalize(heads):
        return heads / jnp.sqrt(
            jnp.sum(heads * heads, axis=-1, keepdims=True) + L2_NORM_EPS)

    def _step(state, token_inputs):
        q_t, k_t, v_t, beta_t, g_t = token_inputs
        # Note (david): KDA decays the [K, V] state row-wise, one gate per key
        # channel.
        decayed = state * jnp.exp(g_t)[:, :, None]
        v_new = beta_t[:, None] * (v_t - jnp.einsum("hd,hdm->hm", k_t, decayed))
        state = decayed + jnp.einsum("hd,hm->hdm", k_t, v_new)
        return state, (jnp.einsum("hd,hdm->hm", q_t, state) * D_K**-0.5, state)

    for seq_idx in seqs:
        start = int(query_start_loc[seq_idx])
        end = int(query_start_loc[seq_idx + 1])
        num_seq_tokens = end - start
        if num_seq_tokens <= 0:
            continue
        has_initial_state = int(seq_lens[seq_idx]) - num_seq_tokens > 0
        read_slot = int(read_slots[seq_idx])
        if has_initial_state:
            init_conv_state = conv_state[read_slot]
            init_recurrent_state = recurrent_state[read_slot]
        else:
            init_conv_state = jnp.zeros_like(conv_state[0])
            init_recurrent_state = jnp.zeros_like(recurrent_state[0])

        conv_input = jnp.concatenate([init_conv_state, qkv[start:end]], axis=0)
        tap_sum = sum(conv_input[tap:tap + num_seq_tokens] *
                      conv_weight[:, 0, tap][None]
                      for tap in range(KERNEL_SIZE))
        if conv_bias is None:
            conv_out = tap_sum
        else:
            conv_out = tap_sum + conv_bias[None]
        activation = jax.nn.silu(conv_out)
        q = _l2_normalize(activation[:, :key_dim].reshape(
            num_seq_tokens, N_KQ, D_K)).repeat(v_per_kq_head, 1)
        k = _l2_normalize(activation[:, key_dim:2 * key_dim].reshape(
            num_seq_tokens, N_KQ, D_K)).repeat(v_per_kq_head, 1)
        v = activation[:, 2 * key_dim:].reshape(num_seq_tokens, N_V, D_V)
        beta = jax.nn.sigmoid(b[start:end])
        shifted_gate = g[start:end].reshape(num_seq_tokens, N_V, D_K) + dt_bias
        if lower_bound is None:
            log_decay = -decay_scale * jax.nn.softplus(shifted_gate)
        else:
            log_decay = lower_bound * jax.nn.sigmoid(decay_scale * shifted_gate)
        _, (seq_out, seq_states) = jax.lax.scan(
            _step, init_recurrent_state, (q, k, v, beta, log_decay))
        out = out.at[start:end].set(seq_out.reshape(num_seq_tokens, -1))

        base_slot = int(state_indices[seq_idx])
        if window == 1:
            new_conv_state = new_conv_state.at[base_slot].set(
                conv_input[-(KERNEL_SIZE - 1):])
            new_recurrent_state = new_recurrent_state.at[base_slot].set(
                seq_states[-1])
        else:
            for token_idx in range(num_seq_tokens):
                new_conv_state = new_conv_state.at[base_slot + token_idx].set(
                    conv_input[token_idx + 1:token_idx + KERNEL_SIZE])
                new_recurrent_state = new_recurrent_state.at[
                    base_slot + token_idx].set(seq_states[token_idx])
    return (new_conv_state, new_recurrent_state), out


def make_inputs(
    key: jax.Array, num_tokens: int, gate_scale: float,
) -> tuple[dict[str, jax.Array], dict[str, jax.Array], dict[str, jax.Array]]:
    """Returns (bf16 kernel activations, their f32 copies, f32 weights)."""
    subkeys = iter(jax.random.split(key, 8))
    normal = jax.random.normal
    activations = dict(
        qkv=normal(next(subkeys), (num_tokens, DIM)),
        b=normal(next(subkeys), (num_tokens, N_V)),
        # Note (david): positive raw gates give softplus ~ gate_scale, so
        # strong gates decay channels by e^-10..e^-30 per token.
        g=normal(next(subkeys), (num_tokens, N_V * D_K)) * gate_scale +
        gate_scale,
    )
    kernel_activations = {
        name: act.astype(jnp.bfloat16) for name, act in activations.items()
    }
    reference_activations = {
        name: act.astype(jnp.float32)
        for name, act in kernel_activations.items()
    }
    params = dict(
        conv_weight=normal(next(subkeys), (DIM, 1, KERNEL_SIZE)),
        conv_bias=normal(next(subkeys), (DIM, )),
        a_log=normal(next(subkeys), (N_V, )),
        dt_bias=normal(next(subkeys), (N_V * D_K, )),
    )
    return kernel_activations, reference_activations, params


@TPU_ONLY
@pytest.mark.parametrize("gate_scale", [1.0, 8.0], ids=["gate", "strong"])
@pytest.mark.parametrize(
    "max_reqs, lengths, q_loc, distribution",
    [
        pytest.param(1, [1024], [0, 1024], [0, 0, 1], id="prefill"),
        pytest.param(3, [256, 128, 100], [0, 256, 384, 484], [0, 3, 3],
                     id="mixed"),
        pytest.param(64, [1] * 64, list(range(65)), [64, 64, 64],
                     id="decode_only"),
        pytest.param(11, [1] * 8 + [128, 128, 256],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8, 136, 264, 520], [8, 11, 11],
                     id="mixed_prefill_decode"),
        pytest.param(16, [128, 64, 32, 16, 8],
                     [0, 128, 192, 224, 240, 248] + [1] * 11, [0, 5, 5],
                     id="padded_mixed_prefill"),
    ],
)
def test_fused_matches_reference(max_reqs: int, lengths: list[int],
                                 q_loc: list[int], distribution: list[int],
                                 gate_scale: float) -> None:
    num_tokens = sum(lengths)
    query_start_loc = jnp.array(q_loc, dtype=jnp.int32)
    seq_distribution = jnp.array(distribution, dtype=jnp.int32)
    state_indices = jnp.arange(1, max_reqs + 1, dtype=jnp.int32)
    conv_state = jnp.zeros((max_reqs + 1, KERNEL_SIZE - 1, DIM))
    recurrent_state = jnp.zeros((max_reqs + 1, N_V, D_K, D_V))
    # Note (david): seq_lens equal to the new token counts means no sequence
    # has prior context.
    seq_lens = query_start_loc[1:max_reqs + 1] - query_start_loc[:max_reqs]

    kernel_acts, reference_acts, params = make_inputs(
        jax.random.key(0), num_tokens, gate_scale)
    (conv_out, rec_out), out = run_fused_kda(
        **kernel_acts, conv_state=conv_state, recurrent_state=recurrent_state,
        **params, query_start_loc=query_start_loc,
        state_indices=state_indices, distribution=seq_distribution,
        seq_lens=seq_lens, read_state_indices=state_indices, n_kq=N_KQ,
        n_v=N_V, d_k=D_K, d_v=D_V, kernel_size=KERNEL_SIZE)
    (conv_ref, rec_ref), out_ref = kda_reference(
        **reference_acts, conv_state=conv_state,
        recurrent_state=recurrent_state, **params,
        query_start_loc=query_start_loc, state_indices=state_indices,
        read_slots=state_indices, seq_lens=seq_lens,
        seqs=range(int(seq_distribution[2])), window=1)

    np.testing.assert_allclose(out.astype(jnp.float32), out_ref, **TOL)
    np.testing.assert_allclose(conv_out, conv_ref, **TOL)
    np.testing.assert_allclose(rec_out, rec_ref, **TOL)


@TPU_ONLY
@pytest.mark.parametrize(
    "spec_lengths, read_offsets, prefill_lengths",
    [
        pytest.param([5, 3, 1], [2, 0, 4], [], id="spec_windows"),
        pytest.param([5, 1, 3, 2, 4], [0, 1, 2, 3, 4], [],
                     id="spec_windows_padded"),
        pytest.param([5, 2], [3, 1], [64], id="spec_and_prefill"),
    ],
)
def test_spec_mode_checkpoints(spec_lengths: list[int], read_offsets: list[int],
                               prefill_lengths: list[int]) -> None:
    """Windows read base + read_offset and checkpoint each position to base + t.

    Prefills in the same batch keep the PER_SEQ behavior.
    """
    window = NUM_SPEC_TOKENS + 1
    lengths = list(spec_lengths) + list(prefill_lengths)
    num_seqs, num_spec = len(lengths), len(spec_lengths)
    num_tokens = sum(lengths)
    query_start_loc = jnp.array(np.concatenate([[0], np.cumsum(lengths)]),
                                jnp.int32)
    distribution = jnp.array([num_spec, num_spec, num_seqs], jnp.int32)
    state_indices = jnp.array(
        [1 + seq_idx * window for seq_idx in range(num_seqs)], jnp.int32)
    num_blocks = 1 + num_seqs * window
    state_read_offsets = jnp.array(
        list(read_offsets) + [0] * len(prefill_lengths), jnp.int32)
    # Note (david): spec windows continue a context; prefills start fresh.
    seq_lens = jnp.array(
        [SPEC_CONTEXT_LEN + window_len for window_len in spec_lengths] +
        list(prefill_lengths), jnp.int32)

    kernel_acts, reference_acts, params = make_inputs(
        jax.random.key(3), num_tokens, 1.0)
    state_keys = jax.random.split(jax.random.key(4), 2)
    # Note (david): distinct random states per slot catch reads and writes of
    # the wrong slot.
    conv_state = jax.random.normal(state_keys[0],
                                   (num_blocks, KERNEL_SIZE - 1, DIM))
    recurrent_state = jax.random.normal(state_keys[1],
                                        (num_blocks, N_V, D_K, D_V))

    (conv_out, rec_out), out = run_fused_kda(
        **kernel_acts, conv_state=conv_state, recurrent_state=recurrent_state,
        **params, query_start_loc=query_start_loc,
        state_indices=state_indices, distribution=distribution,
        seq_lens=seq_lens, read_state_indices=state_indices,
        read_offsets=state_read_offsets, n_kq=N_KQ, n_v=N_V, d_k=D_K,
        d_v=D_V, kernel_size=KERNEL_SIZE, num_spec_tokens=NUM_SPEC_TOKENS)

    reference_args = dict(**reference_acts, **params,
                          query_start_loc=query_start_loc,
                          state_indices=state_indices, seq_lens=seq_lens)
    (spec_conv_ref, spec_rec_ref), spec_out_ref = kda_reference(
        conv_state=conv_state, recurrent_state=recurrent_state,
        read_slots=state_indices + state_read_offsets, seqs=range(num_spec),
        window=window, **reference_args)
    if prefill_lengths:
        # Note (david): the prefills run on top of the spec sequences'
        # checkpoints, as in the kernel's BATCHED-then-PER_SEQ order.
        (conv_ref, rec_ref), prefill_out_ref = kda_reference(
            conv_state=spec_conv_ref, recurrent_state=spec_rec_ref,
            read_slots=state_indices, seqs=range(num_spec, num_seqs),
            window=1, **reference_args)
        prefill_start = int(query_start_loc[num_spec])
        out_ref = spec_out_ref.at[prefill_start:].set(
            prefill_out_ref[prefill_start:])
    else:
        conv_ref, rec_ref, out_ref = spec_conv_ref, spec_rec_ref, spec_out_ref

    np.testing.assert_allclose(out.astype(jnp.float32), out_ref, **TOL)
    written_slots = set()
    for seq_idx, num_window_tokens in enumerate(spec_lengths):
        base_slot = int(state_indices[seq_idx])
        written_slots.update(range(base_slot, base_slot + num_window_tokens))
    written_slots.update(int(state_indices[num_spec + prefill_idx])
                         for prefill_idx in range(len(prefill_lengths)))
    for slot in range(num_blocks):
        if slot in written_slots:
            expected_conv, expected_rec = conv_ref[slot], rec_ref[slot]
        else:
            expected_conv = conv_state[slot]
            expected_rec = recurrent_state[slot]
        np.testing.assert_allclose(conv_out[slot], expected_conv, **TOL,
                                   err_msg=f"conv slot {slot}")
        np.testing.assert_allclose(rec_out[slot], expected_rec, **TOL,
                                   err_msg=f"recurrent slot {slot}")
