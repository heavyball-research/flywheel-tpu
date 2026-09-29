"""Port of vLLM tpu-inference tests/kernels/gdn_attention_v3_test.py.

Upstream @ 4420cae36157, run against the vendored gdn_v3 baseline and
flywheel_tpu.linear_attention.gdn (PKGS). The latter only accepts bf16 qkv,
so the kernels get bf16-rounded activations while the eager reference sees the
same values in f32. Tolerances are upstream's.
"""

import importlib
import sys
from collections.abc import Callable, Iterator

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.common.linear_attention import BASELINE_KERNEL_DIR

sys.path.insert(0, str(BASELINE_KERNEL_DIR))

KernelResult = tuple[tuple[jax.Array, jax.Array], jax.Array]
FusedGdn = Callable[..., KernelResult]

PKGS = ("gdn_v3", "flywheel_tpu.linear_attention.gdn")
ACT_DTYPE = jnp.bfloat16
STATIC = ["n_kq", "n_v", "d_k", "d_v", "kernel_size"]
TOL = 2e-2
STALE_SLOT_TOL = 1e-5
L2_NORM_EPS = 1e-6
KQ_HEAD_DIM = 128
V_HEAD_DIM = 128
N_KQ = 2
N_V = 8
KERNEL_SIZE = 4
CONV_DIM = N_KQ * KQ_HEAD_DIM * 2 + N_V * V_HEAD_DIM
MODEL_DIMS = dict(n_kq=N_KQ,
                  n_v=N_V,
                  d_k=KQ_HEAD_DIM,
                  d_v=V_HEAD_DIM,
                  kernel_size=KERNEL_SIZE)
PREFILL_LEN = 64
# Note (david): both packages size their tiles from pltpu.get_tpu_info() and
# have no interpret mode, so every test here needs a TPU.
TPU_ONLY = pytest.mark.skipif(jax.default_backend() != "tpu",
                              reason="the fused GDN kernel runs on TPU only")
pytestmark = TPU_ONLY


@pytest.fixture(params=PKGS)
def fused_conv1d_gdn(request: pytest.FixtureRequest) -> FusedGdn:
    return importlib.import_module(f"{request.param}.wrapper").fused_conv1d_gdn


def act_pair(
        **acts: jax.Array) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    """bf16 copies for the kernel and the same values in f32 for the ref."""
    kernel_acts = {name: act.astype(ACT_DTYPE) for name, act in acts.items()}
    ref_acts = {
        name: act.astype(jnp.float32)
        for name, act in kernel_acts.items()
    }
    return kernel_acts, ref_acts


def random_activations(
        rngs: Iterator[jax.Array],
        num_tokens: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Draws query, key, value, b and a in that order; returns (qkv, b, a)."""
    query = jax.random.normal(next(rngs), (num_tokens, N_KQ * KQ_HEAD_DIM))
    key = jax.random.normal(next(rngs), (num_tokens, N_KQ * KQ_HEAD_DIM))
    value = jax.random.normal(next(rngs), (num_tokens, N_V * V_HEAD_DIM))
    b = jax.random.normal(next(rngs), (num_tokens, N_V))
    a = jax.random.normal(next(rngs), (num_tokens, N_V))
    return jnp.concatenate([query, key, value], axis=-1), b, a


def random_weights(rngs: Iterator[jax.Array]) -> dict[str, jax.Array]:
    """Draws conv_weight, conv_bias, a_log and dt_bias in that order."""
    return dict(
        conv_weight=jax.random.normal(next(rngs), (CONV_DIM, 1, KERNEL_SIZE)),
        conv_bias=jax.random.normal(next(rngs), (CONV_DIM, )),
        a_log=jax.random.normal(next(rngs), (N_V, )),
        dt_bias=jax.random.normal(next(rngs), (N_V, )),
    )


def gdn_sequence_ref(
    qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    conv_state: jax.Array,
    recurrent_state: jax.Array,
    conv_weight: jax.Array,
    conv_bias: jax.Array | None,
    a_log: jax.Array,
    dt_bias: jax.Array,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Conv1D + silu + gated delta rule over one sequence, token by token.

    conv_state and recurrent_state are the sequence's initial states. Returns
    the conv input window [kernel_size - 1 + query_len, dim], the recurrent
    state after every token [query_len, n_v, d_k, d_v] and the output
    [query_len, n_v * d_v].
    """
    query_len = qkv.shape[0]
    conv_window = jnp.concatenate([conv_state, qkv], axis=0)
    conv_acc = jnp.zeros((query_len, qkv.shape[-1]), dtype=jnp.float32)
    for tap in range(kernel_size):
        conv_acc += (conv_window[tap:tap + query_len].astype(jnp.float32) *
                     conv_weight[:, 0, tap].astype(jnp.float32)[None, :])
    if conv_bias is None:
        conv_pre_act = conv_acc
    else:
        conv_pre_act = conv_acc + conv_bias.astype(jnp.float32)[None, :]
    conv_out = jax.nn.silu(conv_pre_act.astype(qkv.dtype))

    def _l2_normalize(heads: jax.Array) -> jax.Array:
        heads_f32 = heads.astype(jnp.float32)
        norm = jnp.sqrt(
            jnp.sum(heads_f32 * heads_f32, axis=-1, keepdims=True) +
            L2_NORM_EPS)
        return (heads_f32 / norm).astype(heads.dtype)

    key_dim = n_kq * d_k
    v_per_kq_head = n_v // n_kq
    q_heads = conv_out[:, :key_dim].reshape(query_len, n_kq, d_k)
    k_heads = conv_out[:, key_dim:key_dim * 2].reshape(query_len, n_kq, d_k)
    v_heads = conv_out[:, key_dim * 2:].reshape(query_len, n_v, d_v)
    q = _l2_normalize(jnp.repeat(q_heads, v_per_kq_head, axis=1))
    k = _l2_normalize(jnp.repeat(k_heads, v_per_kq_head, axis=1))
    v = v_heads.astype(jnp.float32)

    beta = jax.nn.sigmoid(b.astype(jnp.float32))
    decay_rate = -jnp.exp(a_log.astype(jnp.float32))[None, :]
    log_decay = decay_rate * jax.nn.softplus(
        a.astype(jnp.float32) + dt_bias.astype(jnp.float32)[None, :])

    def _step(
        state: jax.Array, token_inputs: tuple[jax.Array, ...]
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        q_t, k_t, v_t, beta_t, log_decay_t = token_inputs
        q_t = q_t * (d_k**-0.5)
        decay = jnp.exp(log_decay_t)

        k_state = jnp.einsum("hd, hdm -> hm", k_t, state)
        v_diff = v_t - decay[:, None] * k_state
        v_new = beta_t[:, None] * v_diff

        q_state = jnp.einsum("hd, hdm -> hm", q_t, state)
        q_k = jnp.sum(q_t * k_t, axis=-1, keepdims=True)
        out_t = decay[:, None] * q_state + q_k * v_new

        k_v_new = jnp.einsum("hd, hm -> hdm", k_t, v_new)
        new_state = state * decay[:, None, None] + k_v_new
        new_state = new_state.astype(recurrent_state.dtype)

        return new_state, (out_t.astype(qkv.dtype), new_state)

    _, (out, token_states) = jax.lax.scan(_step, recurrent_state,
                                          (q, k, v, beta, log_decay))
    return conv_window, token_states, out.reshape(query_len, n_v * d_v)


def gdn_attention_ref(
    qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    conv_state: jax.Array,
    recurrent_state: jax.Array,
    conv_weight: jax.Array,
    conv_bias: jax.Array | None,
    a_log: jax.Array,
    dt_bias: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    distribution: jax.Array,
    seq_lens: jax.Array,
    read_state_indices: jax.Array,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    read_offsets: jax.Array | None = None,
) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
    """Eager reference of fused_conv1d_gdn, sequence by sequence.

    The first distribution[0] sequences are windows: sequence s reads slot
    read_state_indices[s] + read_offsets[s] and checkpoints the state after
    token t to state_indices[s] + t. The others read read_state_indices[s]
    and write their final state to state_indices[s].
    """
    num_windowed_seqs = int(distribution[0])
    new_conv_state = jnp.array(conv_state)
    new_recurrent_state = jnp.array(recurrent_state)
    output = jnp.zeros((qkv.shape[0], n_v * d_v), dtype=qkv.dtype)

    for req_idx in range(int(distribution[2])):
        start = int(query_start_loc[req_idx])
        end = int(query_start_loc[req_idx + 1])
        query_len = end - start
        if query_len <= 0:
            continue

        is_windowed = req_idx < num_windowed_seqs
        if is_windowed and read_offsets is not None:
            read_slot = (int(read_state_indices[req_idx]) +
                         int(read_offsets[req_idx]))
        else:
            read_slot = int(read_state_indices[req_idx])
        if (seq_lens[req_idx] - query_len) > 0:
            initial_conv_state = conv_state[read_slot]
            initial_recurrent_state = recurrent_state[read_slot]
        else:
            initial_conv_state = jnp.zeros_like(conv_state[read_slot])
            initial_recurrent_state = jnp.zeros_like(recurrent_state[read_slot])

        conv_window, token_states, out = gdn_sequence_ref(
            qkv[start:end],
            b[start:end],
            a[start:end],
            initial_conv_state,
            initial_recurrent_state,
            conv_weight,
            conv_bias,
            a_log,
            dt_bias,
            n_kq,
            n_v,
            d_k,
            d_v,
            kernel_size,
        )
        output = output.at[start:end].set(out)

        write_slot = int(state_indices[req_idx])
        if is_windowed:
            for token in range(query_len):
                new_conv_state = new_conv_state.at[write_slot + token].set(
                    conv_window[token + 1:token + kernel_size])
                new_recurrent_state = new_recurrent_state.at[
                    write_slot + token].set(token_states[token])
        else:
            new_conv_state = new_conv_state.at[write_slot].set(
                conv_window[-(kernel_size - 1):])
            new_recurrent_state = new_recurrent_state.at[write_slot].set(
                token_states[-1])

    return (new_conv_state, new_recurrent_state), output


def run_split_prefill(
    fused_conv1d_gdn: FusedGdn,
    seed: int,
    num_blocks: int,
    read_slot: int,
    write_slot: int,
) -> tuple[jax.Array, KernelResult, KernelResult]:
    """Runs one PREFILL_LEN-token prefill in one shot, then in two halves.

    The single shot and the first half start from zero states and write
    read_slot; the second half resumes from read_slot and writes write_slot.
    Returns the single-shot output and ((conv_state, recurrent_state), output)
    after each half.
    """
    half = PREFILL_LEN // 2
    rngs = iter(jax.random.split(jax.random.key(seed), 12))
    mixed_qkv, b, a = random_activations(rngs, PREFILL_LEN)
    weights = random_weights(rngs)
    kernel_acts, _ = act_pair(qkv=mixed_qkv, b=b, a=a)
    first_half = {name: act[:half] for name, act in kernel_acts.items()}
    second_half = {name: act[half:] for name, act in kernel_acts.items()}

    conv_state_zero = jnp.zeros((num_blocks, KERNEL_SIZE - 1, CONV_DIM))
    recurrent_state_zero = jnp.zeros(
        (num_blocks, N_V, KQ_HEAD_DIM, V_HEAD_DIM))

    run_jitted = jax.jit(fused_conv1d_gdn, static_argnames=STATIC)
    common_kwargs = dict(
        **weights,
        read_state_indices=jnp.array([read_slot]),
        distribution=jnp.array([0, 1, 1], dtype=jnp.int32),
        **MODEL_DIMS,
    )

    _, output_ref = run_jitted(
        **kernel_acts,
        conv_state=conv_state_zero,
        recurrent_state=recurrent_state_zero,
        query_start_loc=jnp.array([0, PREFILL_LEN]),
        state_indices=jnp.array([read_slot]),
        seq_lens=jnp.array([PREFILL_LEN], dtype=jnp.int32),
        **common_kwargs,
    )
    first_states, first_output = run_jitted(
        **first_half,
        conv_state=conv_state_zero,
        recurrent_state=recurrent_state_zero,
        query_start_loc=jnp.array([0, half]),
        state_indices=jnp.array([read_slot]),
        seq_lens=jnp.array([half], dtype=jnp.int32),
        **common_kwargs,
    )
    # Note (david): seq_lens of the full prefill against query_lens of a half
    # leaves half a prefill of context, so has_initial_state is set and the
    # second half continues from the first half's state.
    second_states, second_output = run_jitted(
        **second_half,
        conv_state=first_states[0],
        recurrent_state=first_states[1],
        query_start_loc=jnp.array([0, half]),
        state_indices=jnp.array([write_slot]),
        seq_lens=jnp.array([PREFILL_LEN], dtype=jnp.int32),
        **common_kwargs,
    )
    return (output_ref, (first_states, first_output),
            (second_states, second_output))


@pytest.mark.parametrize(
    "max_reqs, lengths, q_loc, distribution",
    [
        pytest.param(1, [8192], [0, 8192], [0, 0, 3], id="prefill"),
        pytest.param(3, [256, 128, 128], [0, 256, 384, 512], [0, 3, 3],
                     id="mixed"),
        pytest.param(64, [1] * 64, list(range(65)), [64, 64, 64],
                     id="decode_only"),
        pytest.param(11, [1] * 8 + [128, 128, 256],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8, 136, 264, 520], [8, 11, 11],
                     id="mixed_prefill_decode"),
        pytest.param(16, [128, 64, 32, 16, 8],
                     [0, 128, 192, 224, 240, 248] + [1] * 11, [0, 5, 5],
                     id="padded_mixed_prefill"),
        pytest.param(512, [1] * 64, list(range(65)) + [1] * 448, [64, 64, 64],
                     id="padded_decode_only"),
    ],
)
def test_run_jax_gdn_attention_local(fused_conv1d_gdn: FusedGdn,
                                     max_reqs: int, lengths: list[int],
                                     q_loc: list[int],
                                     distribution: list[int]) -> None:
    num_tokens = sum(lengths)
    query_start_loc = jnp.array(q_loc)

    # Note (david): slot 0 of both state caches is the null block for invalid
    # or padded tokens.
    state_indices = jnp.arange(1, max_reqs + 1)
    num_blocks = max_reqs + 1

    rngs = iter(jax.random.split(jax.random.key(0), 12))
    mixed_qkv, b, a = random_activations(rngs, num_tokens)

    conv_state = jnp.zeros((num_blocks, KERNEL_SIZE - 1, CONV_DIM))
    recurrent_state = jnp.zeros((num_blocks, N_V, KQ_HEAD_DIM, V_HEAD_DIM))

    qkv_widths = (N_KQ * KQ_HEAD_DIM, N_KQ * KQ_HEAD_DIM, N_V * V_HEAD_DIM)
    conv_weight_parts = [
        jax.random.normal(next(rngs), (width, 1, KERNEL_SIZE))
        for width in qkv_widths
    ]
    conv_bias_parts = [
        jax.random.normal(next(rngs), (width, )) for width in qkv_widths
    ]
    a_log = jax.random.normal(next(rngs), (N_V, ))
    dt_bias = jax.random.normal(jax.random.key(0), (N_V, ))

    # Note (david): seq_lens == query_lens means no prior context, so every
    # sequence starts from a zero state whatever its slot holds.
    seq_lens = jnp.asarray(
        query_start_loc[1:max_reqs + 1] - query_start_loc[:max_reqs],
        dtype=jnp.int32)

    kernel_acts, ref_acts = act_pair(qkv=mixed_qkv, b=b, a=a)
    common_kwargs = dict(
        conv_state=conv_state,
        recurrent_state=recurrent_state,
        conv_weight=jnp.concatenate(conv_weight_parts, axis=0),
        conv_bias=jnp.concatenate(conv_bias_parts, axis=-1),
        a_log=a_log,
        dt_bias=dt_bias,
        query_start_loc=query_start_loc,
        state_indices=state_indices,
        distribution=jnp.array(distribution, dtype=jnp.int32),
        seq_lens=seq_lens,
        read_state_indices=state_indices,
        **MODEL_DIMS,
    )

    new_states_ref, output_ref = gdn_attention_ref(**ref_acts, **common_kwargs)
    new_states, output = jax.jit(fused_conv1d_gdn, static_argnames=STATIC)(
        **kernel_acts, **common_kwargs)

    np.testing.assert_allclose(output.astype(jnp.float32),
                               output_ref,
                               rtol=TOL,
                               atol=TOL)
    np.testing.assert_allclose(new_states[0],
                               new_states_ref[0],
                               rtol=TOL,
                               atol=TOL)
    np.testing.assert_allclose(new_states[1],
                               new_states_ref[1],
                               rtol=TOL,
                               atol=TOL)


@pytest.mark.parametrize(
    "spec_lengths, read_offsets, prefill_lengths",
    [
        pytest.param([5, 3, 1], [2, 0, 4], [], id="spec_windows"),
        pytest.param([5, 1, 3, 2, 4], [0, 1, 2, 3, 4], [],
                     id="spec_windows_padded"),
        pytest.param([5, 2], [3, 1], [64], id="spec_and_prefill"),
    ],
)
def test_spec_mode_checkpoints(fused_conv1d_gdn: FusedGdn,
                               spec_lengths: list[int],
                               read_offsets: list[int],
                               prefill_lengths: list[int]) -> None:
    """Verify windows read base + read_offset and checkpoint to base + t.

    Prefill sequences in the same batch keep reading and writing their base
    slot, and slots nobody writes keep their contents.
    """
    num_spec_tokens = 4
    window_size = num_spec_tokens + 1
    context_len = 16

    lengths = list(spec_lengths) + list(prefill_lengths)
    num_seqs = len(lengths)
    num_spec_seqs = len(spec_lengths)
    num_tokens = sum(lengths)
    q_loc = jnp.array(np.concatenate([[0], np.cumsum(lengths)]),
                      dtype=jnp.int32)
    distribution = jnp.array([num_spec_seqs, num_spec_seqs, num_seqs],
                             dtype=jnp.int32)

    # Note (david): each sequence owns window_size consecutive slots after the
    # null block 0; spec_windows_padded has more windows than one decode tile.
    state_indices = jnp.array([1 + i * window_size for i in range(num_seqs)],
                              dtype=jnp.int32)
    num_blocks = 1 + num_seqs * window_size
    read_offsets_arr = jnp.array(list(read_offsets) +
                                 [0] * len(prefill_lengths),
                                 dtype=jnp.int32)

    rngs = iter(jax.random.split(jax.random.key(3), 12))
    mixed_qkv = jax.random.normal(next(rngs), (num_tokens, CONV_DIM))
    b = jax.random.normal(next(rngs), (num_tokens, N_V))
    a = jax.random.normal(next(rngs), (num_tokens, N_V))

    # Note (david): every slot holds a distinct random state, so reading or
    # writing the wrong slot is caught.
    conv_state = jax.random.normal(next(rngs),
                                   (num_blocks, KERNEL_SIZE - 1, CONV_DIM))
    recurrent_state = jax.random.normal(
        next(rngs), (num_blocks, N_V, KQ_HEAD_DIM, V_HEAD_DIM))
    weights = random_weights(rngs)

    # Note (david): the windows continue an existing context, so they have an
    # initial state; the prefills start fresh.
    seq_lens = jnp.array([context_len + length for length in spec_lengths] +
                         list(prefill_lengths),
                         dtype=jnp.int32)

    kernel_acts, ref_acts = act_pair(qkv=mixed_qkv, b=b, a=a)
    common_kwargs = dict(
        **weights,
        conv_state=conv_state,
        recurrent_state=recurrent_state,
        query_start_loc=q_loc,
        state_indices=state_indices,
        distribution=distribution,
        seq_lens=seq_lens,
        read_state_indices=state_indices,
        read_offsets=read_offsets_arr,
        **MODEL_DIMS,
    )

    run_jitted = jax.jit(fused_conv1d_gdn,
                         static_argnames=STATIC + ["num_spec_tokens"])
    (new_conv, new_rec), output = run_jitted(**kernel_acts,
                                             **common_kwargs,
                                             num_spec_tokens=num_spec_tokens)
    (ref_conv, ref_rec), ref_output = gdn_attention_ref(**ref_acts,
                                                        **common_kwargs)

    np.testing.assert_allclose(output.astype(jnp.float32),
                               ref_output,
                               rtol=TOL,
                               atol=TOL)
    # Note (david): the reference copies the slots it does not write, so the
    # whole-cache comparison also checks that untouched slots survive.
    np.testing.assert_allclose(new_conv, ref_conv, rtol=TOL, atol=TOL)
    np.testing.assert_allclose(new_rec, ref_rec, rtol=TOL, atol=TOL)


def test_has_initial_state_zeros_stale_slot(fused_conv1d_gdn: FusedGdn) -> None:
    """A new prefill ignores the stale state of a reused slot.

    vLLM's mamba pool reuses freed slots without clearing them, so the output
    and final state must match the same prefill on a zeroed slot.
    """
    max_reqs = 2
    lengths = [64, 64]
    q_loc = jnp.array([0, 64, 128])
    distribution = jnp.array([0, 2, 2], dtype=jnp.int32)
    num_tokens = sum(lengths)

    state_indices = jnp.arange(1, max_reqs + 1)
    num_blocks = max_reqs + 1

    rngs = iter(jax.random.split(jax.random.key(7), 12))
    mixed_qkv, b, a = random_activations(rngs, num_tokens)

    conv_state_fresh = jnp.zeros((num_blocks, KERNEL_SIZE - 1, CONV_DIM))
    recurrent_state_fresh = jnp.zeros(
        (num_blocks, N_V, KQ_HEAD_DIM, V_HEAD_DIM))
    stale_conv = jax.random.normal(next(rngs),
                                   (num_blocks, KERNEL_SIZE - 1, CONV_DIM))
    stale_recurrent = jax.random.normal(
        next(rngs), (num_blocks, N_V, KQ_HEAD_DIM, V_HEAD_DIM))
    # Note (david): slot 0 is the null block, so it stays zero.
    conv_state_stale = conv_state_fresh.at[1:].set(stale_conv[1:])
    recurrent_state_stale = recurrent_state_fresh.at[1:].set(
        stale_recurrent[1:])
    weights = random_weights(rngs)

    run_jitted = jax.jit(fused_conv1d_gdn, static_argnames=STATIC)
    kernel_acts, _ = act_pair(qkv=mixed_qkv, b=b, a=a)
    # Note (david): seq_lens == query_lens means no prior context, so
    # has_initial_state is False for both requests.
    common_kwargs = dict(
        **kernel_acts,
        **weights,
        query_start_loc=q_loc,
        state_indices=state_indices,
        distribution=distribution,
        seq_lens=jnp.asarray(lengths, dtype=jnp.int32),
        read_state_indices=state_indices,
        **MODEL_DIMS,
    )

    (new_conv_fresh, new_rec_fresh), output_fresh = run_jitted(
        conv_state=conv_state_fresh,
        recurrent_state=recurrent_state_fresh,
        **common_kwargs,
    )
    (new_conv_stale, new_rec_stale), output_stale = run_jitted(
        conv_state=conv_state_stale,
        recurrent_state=recurrent_state_stale,
        **common_kwargs,
    )

    np.testing.assert_allclose(output_fresh.astype(jnp.float32),
                               output_stale.astype(jnp.float32),
                               rtol=STALE_SLOT_TOL,
                               atol=STALE_SLOT_TOL)
    # Note (david): only the active slots are compared; the null slot 0 is
    # untouched in both runs.
    active_slots = slice(1, max_reqs + 1)
    np.testing.assert_allclose(
        new_conv_fresh[active_slots],
        new_conv_stale[active_slots],
        rtol=STALE_SLOT_TOL,
        atol=STALE_SLOT_TOL,
    )
    np.testing.assert_allclose(
        new_rec_fresh[active_slots],
        new_rec_stale[active_slots],
        rtol=STALE_SLOT_TOL,
        atol=STALE_SLOT_TOL,
    )


def test_has_initial_state_preserves_continuation(
        fused_conv1d_gdn: FusedGdn) -> None:
    """A prefill split in two halves matches the single-shot prefill.

    The second half has an initial state, so it must continue from the state
    the first half wrote (chunked prefill / prefix cache continuation).
    """
    half = PREFILL_LEN // 2
    output_ref, (_, first_output), (_, second_output) = run_split_prefill(
        fused_conv1d_gdn, seed=11, num_blocks=2, read_slot=1, write_slot=1)

    output_ref = output_ref.astype(jnp.float32)
    np.testing.assert_allclose(first_output.astype(jnp.float32),
                               output_ref[:half],
                               rtol=TOL,
                               atol=TOL)
    np.testing.assert_allclose(second_output.astype(jnp.float32),
                               output_ref[half:],
                               rtol=TOL,
                               atol=TOL)


def test_split_read_write_state_slots(fused_conv1d_gdn: FusedGdn) -> None:
    """Resume from one state slot while checkpointing into another.

    This is mamba prefix caching (align mode): a request continues from the
    state block cached at the last block boundary and checkpoints into the
    block of its current position. The numerics must match a single slot, and
    the read slot must stay intact so other requests can still hit it.
    """
    half = PREFILL_LEN // 2
    read_slot, write_slot = 1, 2
    output_ref, first_step, second_step = run_split_prefill(
        fused_conv1d_gdn,
        seed=23,
        num_blocks=3,
        read_slot=read_slot,
        write_slot=write_slot)
    (conv_after_first, rec_after_first), _ = first_step
    (conv_after_second, rec_after_second), second_output = second_step

    np.testing.assert_allclose(second_output.astype(jnp.float32),
                               output_ref[half:].astype(jnp.float32),
                               rtol=TOL,
                               atol=TOL)
    np.testing.assert_array_equal(conv_after_second[read_slot],
                                  conv_after_first[read_slot])
    np.testing.assert_array_equal(rec_after_second[read_slot],
                                  rec_after_first[read_slot])
    # Note (david): write_slot started out zeroed, so a nonzero state shows the
    # new checkpoint landed there.
    assert np.any(np.asarray(rec_after_second[write_slot]) != 0.0), (
        "write_slot should hold the step's final recurrent state")
