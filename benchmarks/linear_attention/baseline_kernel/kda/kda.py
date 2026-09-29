# Adapted from https://github.com/primatrix/pallas-kernel (rev 3c691ad3)
# Vendored to remove external dependency after the upstream repository went private.
#
# This file merges the following modules into a single file:
#   - tops/utils.py (cdiv, align_up, prepare_lens, prepare_chunk_indices, assert_shape, assert_shape_or_none)
#   - tops/ops/utils.py (exp, exp2, get_interpret)
#   - tops/ops/common/cumsum.py (chunk_local_cumsum_vector)
#   - tops/ops/kda/chunk_intra_fwd.py (solve_unit_lower_triangular, kda_fwd_intra)
#   - tops/ops/common/chunk_delta_h.py (chunk_gated_delta_rule_fwd_h)
#   - tops/ops/gla/chunk.py (chunk_kda_fwd_o_gk_varlen, here chunk_kda_fwd_o_gk)
#   - tops/ops/kda/gate.py (kda_gate_chunk_cumsum, pallas_kda_gate_cumsum)
#   - tops/ops/kda/chunk_fwd.py (chunk_kda_fwd)
"""KDA chunked forward pass for variable-length sequences (self-contained)."""

from __future__ import annotations

import functools
import math
import os
from collections.abc import Callable

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
from jax.experimental.pallas import dslice
from jax.experimental.pallas import tpu as pltpu

NUM_LANES = 128
NUM_SUBLANES = 8
VMEM_HW_LIMIT_BYTES = 30 * 1024 * 1024
F32_BYTES = 4
# Note (david): the cumsum kernel's input and output blocks are each
# double-buffered.
NUM_PIPELINED_BLOCKS = 4
TRI_SOLVE_BLOCK_SIZE = 16
MIN_NORMAL_F32_EXPONENT = -126.0
MAX_HEAD_DIM = 256
RCP_LN2 = 1.0 / math.log(2)
PAD_GATE_LOGIT = -1e4


def cdiv(x: int | jax.Array, y: int) -> int | jax.Array:
    return (x + y - 1) // y


def align_up(x: int | jax.Array, align: int) -> int | jax.Array:
    return cdiv(x, align) * align


def prepare_lens(cu_seqlens: jax.Array) -> jax.Array:
    return cu_seqlens[1:] - cu_seqlens[:-1]


def prepare_chunk_indices(
    cu_seqlens: jax.Array,
    chunk_size: int,
    max_T: int | None = None,
) -> jax.Array:
    """[max_T // chunk_size, 2] rows of (sequence id, chunk id within the sequence)."""
    chunks_per_seq = cdiv(prepare_lens(cu_seqlens), chunk_size)
    num_seqs = len(chunks_per_seq)
    num_chunks = max_T // chunk_size
    seq_ids = jnp.repeat(
        jnp.arange(num_seqs, dtype=jnp.int32), chunks_per_seq, total_repeat_length=num_chunks
    )
    first_chunk_of_seq = jnp.concatenate(
        [jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(chunks_per_seq)]
    )
    seq_first_chunks = jnp.repeat(
        first_chunk_of_seq[:-1], chunks_per_seq, total_repeat_length=num_chunks
    )
    local_chunk_ids = jnp.arange(num_chunks, dtype=jnp.int32) - seq_first_chunks
    return jnp.stack([seq_ids, local_chunk_ids], axis=1)


def assert_shape(x: jax.Array, expected_shape: tuple[int, ...], name: str = "tensor") -> None:
    assert x.shape == expected_shape, f"[{name}] Expected shape {expected_shape}, got {x.shape}"


def assert_shape_or_none(
    x: jax.Array | None, expected_shape: tuple[int, ...], name: str = "tensor"
) -> None:
    if x is not None:
        assert_shape(x, expected_shape, name)


def exp(x: jax.Array) -> jax.Array:
    return jnp.exp(x.astype(jnp.float32))


def exp2(x: jax.Array) -> jax.Array:
    return jnp.exp2(x.astype(jnp.float32))


def get_interpret() -> bool:
    return os.environ.get("PALLAS_INTERPRET", "").strip().lower() in ("1", "true")


def chunk_cumsum_kernel(
    g_ref: jax.Array,
    o_ref: jax.Array,
    *,
    chunk_size: int,
    is_reverse: bool,
    scale: float | None,
) -> None:
    cumsum = g_ref[:, 0, :, :].astype(jnp.float32)
    for step in range(int(math.log2(chunk_size))):
        stride = 1 << step
        if is_reverse:
            top_rows = cumsum[:, : chunk_size - stride, :] + cumsum[:, stride:, :]
            bottom_rows = cumsum[:, chunk_size - stride :, :]
        else:
            top_rows = cumsum[:, :stride, :]
            bottom_rows = cumsum[:, stride:, :] + cumsum[:, :-stride, :]
        cumsum = jnp.concatenate([top_rows, bottom_rows], axis=1)

    if scale is None:
        scaled_cumsum = cumsum
    else:
        scaled_cumsum = cumsum * scale
    o_ref[:, 0, dslice(0, chunk_size), :] = scaled_cumsum.astype(o_ref.dtype)


def chunk_local_cumsum_vector(
    g: jax.Array,
    chunk_size: int,
    reverse: bool = False,
    scale: float | None = None,
    cu_seqlens: jax.Array | None = None,
    head_first: bool = False,
    output_dtype: jnp.dtype | None = jnp.float32,
    chunk_indices: jax.Array | None = None,
) -> jax.Array:
    """Cumulative sum of g over the tokens of each chunk of each sequence.

    g is [B, T, H, S], or [B, H, T, S] when head_first; the result keeps the
    layout. chunk_indices defaults to prepare_chunk_indices over cu_seqlens.
    """
    assert g.ndim == 4, f"g must be 4-D, got {g.ndim}-D"
    assert chunk_size == 2 ** (chunk_size.bit_length() - 1), "chunk_size must be power of 2"
    assert cu_seqlens is not None, "This varlen-only module requires cu_seqlens"

    if head_first:
        batch_size, num_heads, num_tokens, feature_dim = g.shape
        rows = g.reshape(batch_size * num_heads, num_tokens, feature_dim)
    else:
        batch_size, num_tokens, num_heads, feature_dim = g.shape
        rows = jnp.transpose(g, (0, 2, 1, 3)).reshape(
            batch_size * num_heads, num_tokens, feature_dim
        )
    num_rows = batch_size * num_heads
    out_dtype = g.dtype if output_dtype is None else output_dtype

    padded_feature_dim = align_up(feature_dim, NUM_LANES)
    padded_num_rows = align_up(num_rows, NUM_SUBLANES)
    padded_rows = jnp.pad(
        rows,
        (
            (0, padded_num_rows - num_rows),
            (0, chunk_size),
            (0, padded_feature_dim - feature_dim),
        ),
    )

    cu_seqlens_i32 = cu_seqlens.astype(jnp.int32)
    if chunk_indices is None:
        max_num_tokens = num_tokens + (cu_seqlens.shape[0] - 1) * (chunk_size - 1)
        chunk_indices_i32 = prepare_chunk_indices(
            cu_seqlens, chunk_size, max_T=max_num_tokens
        ).astype(jnp.int32)
    else:
        chunk_indices_i32 = chunk_indices.astype(jnp.int32)
    num_chunks = len(chunk_indices_i32)
    num_seqs = cu_seqlens_i32.shape[0] - 1
    chunks_per_seq = (jnp.diff(cu_seqlens_i32) + chunk_size - 1) // chunk_size

    seq_ids = chunk_indices_i32[:, 0]
    local_chunk_ids = chunk_indices_i32[:, 1]
    safe_seq_ids = jnp.clip(seq_ids, 0, num_seqs - 1)
    is_valid_chunk = (
        (seq_ids == safe_seq_ids)
        & (local_chunk_ids >= 0)
        & (local_chunk_ids < chunks_per_seq[safe_seq_ids])
    )
    seq_starts = cu_seqlens_i32[safe_seq_ids]
    seq_ends = cu_seqlens_i32[safe_seq_ids + 1]
    chunk_starts = jnp.where(is_valid_chunk, seq_starts + local_chunk_ids * chunk_size, 0)
    positions = chunk_starts[:, None] + jnp.arange(chunk_size, dtype=jnp.int32)[None, :]
    is_valid_token = (
        is_valid_chunk[:, None] & (positions < seq_ends[:, None]) & (positions < num_tokens)
    )

    def _gather_chunk(start: jax.Array) -> jax.Array:
        return jax.lax.dynamic_slice(
            padded_rows,
            (0, start, 0),
            (padded_num_rows, chunk_size, padded_feature_dim),
        )

    g_chunks = jnp.where(
        is_valid_token[None, :, :, None],
        jax.vmap(_gather_chunk)(chunk_starts).transpose(1, 0, 2, 3),
        0,
    )

    row_block = NUM_SUBLANES
    while (
        row_block > 1
        and NUM_PIPELINED_BLOCKS * row_block * chunk_size * NUM_LANES * F32_BYTES
        > VMEM_HW_LIMIT_BYTES
    ):
        row_block //= 2
    block_shape = (row_block, 1, chunk_size, NUM_LANES)

    def _index_map(
        feature_block_idx: jax.Array, row_block_idx: jax.Array, chunk_idx: jax.Array
    ) -> tuple[jax.Array, jax.Array, int, jax.Array]:
        return (row_block_idx, chunk_idx, 0, feature_block_idx)

    o_chunks = pl.pallas_call(
        functools.partial(
            chunk_cumsum_kernel,
            chunk_size=chunk_size,
            is_reverse=reverse,
            scale=scale,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(padded_feature_dim // NUM_LANES, padded_num_rows // row_block, num_chunks),
            in_specs=[pl.BlockSpec(block_shape=block_shape, index_map=_index_map)],
            out_specs=pl.BlockSpec(block_shape=block_shape, index_map=_index_map),
        ),
        out_shape=jax.ShapeDtypeStruct(g_chunks.shape, out_dtype),
        interpret=get_interpret(),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel")
        ),
    )(g_chunks)

    o_chunks = jnp.where(is_valid_token[None, :, :, None], o_chunks, 0)
    sentinel = jnp.minimum(cu_seqlens_i32[-1], num_tokens)
    scatter_positions = jnp.where(is_valid_token, positions, sentinel).reshape(-1)
    o_rows = jnp.zeros(
        (padded_num_rows, num_tokens + chunk_size, padded_feature_dim), dtype=o_chunks.dtype
    )
    o_rows = o_rows.at[:, scatter_positions, :].add(
        o_chunks.reshape(padded_num_rows, num_chunks * chunk_size, padded_feature_dim)
    )
    o = o_rows[:num_rows, :num_tokens, :feature_dim].reshape(
        batch_size, num_heads, num_tokens, feature_dim
    )
    if head_first:
        return o
    else:
        return jnp.transpose(o, (0, 2, 1, 3))


def chunk_slot_layout(
    cu_seqlens: jax.Array,
    chunk_size: int,
    chunk_indices: jax.Array | None,
    num_tokens: int,
) -> tuple[jax.Array, jax.Array]:
    """Chunk slots of chunk-aligned packed sequences.

    Returns the start token of every slot [num_slots] and the scatter position
    of every slot row [num_slots * chunk_size]. Slots past the last real chunk
    read from token 0 and scatter to the padding row num_tokens + chunk_size - 1.
    """
    num_seqs = cu_seqlens.shape[0] - 1
    cu_seqlens_i32 = cu_seqlens.astype(jnp.int32)
    chunks_per_seq = (jnp.diff(cu_seqlens_i32) + chunk_size - 1) // chunk_size
    first_chunk_of_seq = jnp.pad(jnp.cumsum(chunks_per_seq), (1, 0))
    if chunk_indices is None:
        num_slots = num_tokens // chunk_size + num_seqs
    else:
        num_slots = len(chunk_indices)

    slot_ids = jnp.arange(num_slots, dtype=jnp.int32)
    is_real_chunk = slot_ids < first_chunk_of_seq[-1]
    seq_ids = jnp.minimum(
        jnp.searchsorted(first_chunk_of_seq[1:], slot_ids, side="right"), num_seqs - 1
    )
    local_chunk_ids = slot_ids - first_chunk_of_seq[seq_ids]
    # Note (david): every sequence is chunk-aligned, so every chunk is full and
    # needs no partial-chunk token mask.
    chunk_starts = jnp.where(
        is_real_chunk, cu_seqlens_i32[seq_ids] + local_chunk_ids * chunk_size, 0
    )
    row_positions = jnp.where(
        is_real_chunk[:, None],
        chunk_starts[:, None] + jnp.arange(chunk_size)[None, :],
        num_tokens + chunk_size - 1,
    )
    return chunk_starts, row_positions.reshape(-1)


def gather_chunks(packed: jax.Array, chunk_starts: jax.Array, chunk_size: int) -> jax.Array:
    """[1, T, H, D] packed tokens -> [H, num_slots, chunk_size, D] chunks."""
    _, _, num_heads, dim = packed.shape
    padded = jnp.pad(packed, ((0, 0), (0, chunk_size), (0, 0), (0, 0)))

    def _slice_chunk(start: jax.Array) -> jax.Array:
        return jax.lax.dynamic_slice(padded, (0, start, 0, 0), (1, chunk_size, num_heads, dim))[0]

    return jax.vmap(_slice_chunk)(chunk_starts).transpose(2, 0, 1, 3)


def scatter_chunks(chunks: jax.Array, row_positions: jax.Array, num_tokens: int) -> jax.Array:
    """[H, num_slots, chunk_size, D] chunks -> [1, T, H, D] packed tokens."""
    num_heads, _, chunk_size, dim = chunks.shape
    chunk_rows = chunks.transpose(1, 2, 0, 3).reshape(-1, num_heads, dim)
    packed = jnp.zeros((num_tokens + chunk_size, num_heads, dim), dtype=chunks.dtype)
    return packed.at[row_positions].add(chunk_rows)[:num_tokens][None]


def solve_unit_lower_triangular(strict_lower: jax.Array, rhs: jax.Array) -> jax.Array:
    """Solves (I + strict_lower) x = rhs in f32 by blocked forward substitution."""
    num_blocks = rhs.shape[0] // TRI_SOLVE_BLOCK_SIZE
    strict_lower_f32 = strict_lower.astype(jnp.float32)
    blocks = jnp.split(rhs.astype(jnp.float32), num_blocks, axis=0)

    for block_idx in range(num_blocks):
        start = block_idx * TRI_SOLVE_BLOCK_SIZE
        end = start + TRI_SOLVE_BLOCK_SIZE
        diag_block = strict_lower_f32[start:end, start:end]
        rows = [blocks[block_idx][row] for row in range(TRI_SOLVE_BLOCK_SIZE)]
        for row in range(1, TRI_SOLVE_BLOCK_SIZE):
            correction = jax.lax.dot_general(
                diag_block[row, :row][None, :],
                jnp.stack(rows[:row]),
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            ).squeeze(axis=0)
            rows[row] = rows[row] - correction
        solved_block = jnp.stack(rows)
        blocks[block_idx] = solved_block

        if block_idx < num_blocks - 1:
            unsolved = jnp.concatenate(blocks[block_idx + 1 :], axis=0)
            update = jax.lax.dot_general(
                strict_lower_f32[end:, start:end],
                solved_block,
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            blocks[block_idx + 1 :] = jnp.split(
                unsolved - update, num_blocks - 1 - block_idx, axis=0
            )

    return jnp.concatenate(blocks, axis=0)


def kda_fwd_intra_kernel(
    q_ref: jax.Array,
    k_ref: jax.Array,
    g_ref: jax.Array,
    beta_ref: jax.Array,
    v_ref: jax.Array,
    u_ref: jax.Array,
    w_ref: jax.Array,
    qg_ref: jax.Array,
    kg_ref: jax.Array,
    aqk_ref: jax.Array,
    akk_inv_ref: jax.Array,
    *,
    chunk_size: int,
    head_dim: int,
    value_dim: int,
    scale: float,
    disable_recompute: bool,
) -> None:
    q_chunk = q_ref[0, 0, 0]
    k_chunk = k_ref[0, 0, 0]
    g_chunk = g_ref[0, 0, 0]
    beta_chunk = beta_ref[0, 0, 0]
    v_chunk = v_ref[0, 0, 0]
    g = g_chunk.astype(jnp.float32)
    q = q_chunk.astype(jnp.float32)
    k = k_chunk.astype(jnp.float32)
    beta = beta_chunk.astype(jnp.float32)

    # Note (david): decays are exp2(g[i] - g[j]) rather than a split
    # normalization exp2(g - g_n), which overflows once per-step gate changes
    # exceed ~127. For causal i >= j the cumsum gives g[i] - g[j] <= 0, so the
    # decay stays in (0, 1].
    causal_mask = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.float32))
    strict_causal_mask = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.float32), k=-1)
    g_diff = g[:, None, :] - g[None, :, :]
    # Note (david): anti-causal entries would overflow exp2, so they are set to
    # the floor first; the causal masks zero them afterwards.
    g_diff = jnp.where(causal_mask[:, :, None] > 0, g_diff, MIN_NORMAL_F32_EXPONENT)
    decay = exp2(jnp.maximum(g_diff, MIN_NORMAL_F32_EXPONENT))

    aqk = (scale * jnp.sum(q[:, None, :] * decay * k[None, :, :], axis=-1) * causal_mask).astype(
        q_ref.dtype
    )
    akk_strict_lower = (
        jnp.sum(k[:, None, :] * decay * k[None, :, :], axis=-1) * beta * strict_causal_mask
    )

    rhs = jnp.concatenate(
        [
            v_chunk.astype(jnp.float32) * beta,
            k * exp2(g) * beta,
            jnp.eye(chunk_size, dtype=jnp.float32),
        ],
        axis=-1,
    )
    solution = solve_unit_lower_triangular(akk_strict_lower, rhs)
    u = solution[:, :value_dim]
    w = solution[:, value_dim : value_dim + head_dim]
    akk_inv = solution[:, value_dim + head_dim :]

    kg = k * exp2(g[chunk_size - 1 : chunk_size, :] - g)
    if disable_recompute:
        qg = q * exp2(g)
    else:
        qg = jnp.zeros_like(q)

    u_ref[0, 0, 0] = u.astype(u_ref.dtype)
    w_ref[0, 0, 0] = w.astype(w_ref.dtype)
    qg_ref[0, 0, 0] = qg.astype(qg_ref.dtype)
    kg_ref[0, 0, 0] = kg.astype(kg_ref.dtype)
    aqk_ref[0, 0, 0] = aqk.astype(aqk_ref.dtype)
    akk_inv_ref[0, 0, 0] = akk_inv.astype(akk_inv_ref.dtype)


@functools.partial(
    jax.jit,
    static_argnames=[
        "chunk_size",
        "scale",
        "safe_gate",
        "disable_recompute",
    ],
)
def kda_fwd_intra(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    gk: jax.Array,
    beta: jax.Array,
    scale: float,
    cu_seqlens: jax.Array,
    chunk_size: int = 64,
    chunk_indices: jax.Array | None = None,
    safe_gate: bool = True,
    disable_recompute: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array | None, jax.Array, jax.Array, jax.Array]:
    """Intra-chunk KDA solve over chunk-aligned packed sequences.

    q, k, gk [1, T, H, K], v [1, T, H, V], beta [1, T, H]. Returns w, u, qg
    (None unless disable_recompute), kg, Aqk and the inverted Akk, all [1, T,
    H, ...]. safe_gate is accepted for upstream parity and has no effect.
    """
    assert cu_seqlens is not None, "cu_seqlens must be provided for varlen"
    batch_size, num_tokens, num_heads, head_dim = q.shape
    value_dim = v.shape[-1]
    assert batch_size == 1, f"varlen requires B=1 (packed layout), got B={batch_size}"
    assert chunk_size >= TRI_SOLVE_BLOCK_SIZE and chunk_size % TRI_SOLVE_BLOCK_SIZE == 0
    assert_shape(k, (batch_size, num_tokens, num_heads, head_dim), "k")
    assert_shape(v, (batch_size, num_tokens, num_heads, value_dim), "v")
    assert_shape(gk, (batch_size, num_tokens, num_heads, head_dim), "gk")
    assert_shape(beta, (batch_size, num_tokens, num_heads), "beta")

    chunk_starts, row_positions = chunk_slot_layout(
        cu_seqlens, chunk_size, chunk_indices, num_tokens
    )
    num_slots = chunk_starts.shape[0]
    q_chunks, k_chunks, g_chunks, beta_chunks, v_chunks = (
        gather_chunks(packed, chunk_starts, chunk_size)[None]
        for packed in (q, k, gk, beta.reshape(batch_size, num_tokens, num_heads, 1), v)
    )

    def _block_spec(last_dim: int) -> pl.BlockSpec:
        return pl.BlockSpec(
            index_map=lambda batch_idx, head_idx, slot_idx: (batch_idx, head_idx, slot_idx, 0, 0),
            block_shape=(1, 1, 1, chunk_size, last_dim),
        )

    out_last_dims = (value_dim, head_dim, head_dim, head_dim, chunk_size, chunk_size)
    u_chunks, w_chunks, qg_chunks, kg_chunks, aqk_chunks, akk_inv_chunks = pl.pallas_call(
        functools.partial(
            kda_fwd_intra_kernel,
            chunk_size=chunk_size,
            head_dim=head_dim,
            value_dim=value_dim,
            scale=scale,
            disable_recompute=disable_recompute,
        ),
        interpret=get_interpret(),
        out_shape=[
            jax.ShapeDtypeStruct((batch_size, num_heads, num_slots, chunk_size, last_dim), q.dtype)
            for last_dim in out_last_dims
        ],
        in_specs=[
            _block_spec(last_dim) for last_dim in (head_dim, head_dim, head_dim, 1, value_dim)
        ],
        out_specs=[_block_spec(last_dim) for last_dim in out_last_dims],
        grid=(batch_size, num_heads, num_slots),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel")
        ),
    )(q_chunks, k_chunks, g_chunks, beta_chunks, v_chunks)

    if disable_recompute:
        qg = scatter_chunks(qg_chunks[0], row_positions, num_tokens)
    else:
        qg = None
    return (
        scatter_chunks(w_chunks[0], row_positions, num_tokens),
        scatter_chunks(u_chunks[0], row_positions, num_tokens),
        qg,
        scatter_chunks(kg_chunks[0], row_positions, num_tokens),
        scatter_chunks(aqk_chunks[0], row_positions, num_tokens),
        scatter_chunks(akk_inv_chunks[0], row_positions, num_tokens),
    )


def chunk_gated_delta_rule_fwd_kernel(
    cu_seqlens_ref: jax.Array,
    k_ref: jax.Array,
    v_ref: jax.Array,
    w_ref: jax.Array,
    g_ref: jax.Array | None,
    gk_ref: jax.Array | None,
    h0_ref: jax.Array | None,
    h_ref: jax.Array,
    v_new_ref: jax.Array | None,
    ht_ref: jax.Array | None,
    state_ref: jax.Array,
    *,
    exp_fn: Callable[[jax.Array], jax.Array],
) -> None:
    seq_idx = pl.program_id(0)
    chunk_idx = pl.program_id(2)

    seq_start = cu_seqlens_ref[seq_idx]
    seq_end = cu_seqlens_ref[seq_idx + 1]
    chunk_size = k_ref.shape[2]
    num_seq_chunks = (seq_end - seq_start) // chunk_size
    k_chunk = k_ref[0, 0]

    @pl.when(chunk_idx == 0)
    def _init_state() -> None:
        if h0_ref is None:
            state_ref[...] = jnp.zeros(state_ref.shape, dtype=jnp.float32)
        else:
            state_ref[...] = h0_ref[0, 0].astype(jnp.float32)

    @pl.when(chunk_idx < num_seq_chunks)
    def _advance_state() -> None:
        h_ref[0, 0, 0] = state_ref[...].astype(h_ref.dtype)

        w_state = jnp.dot(
            w_ref[0, 0].astype(jnp.float32),
            state_ref[...],
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        v_new = v_ref[0, 0].astype(jnp.float32) - w_state
        if v_new_ref is not None:
            v_new_ref[0, 0] = v_new.astype(v_new_ref.dtype)

        if g_ref is None:
            decayed_v_new = v_new
        else:
            g_chunk = g_ref[0, 0, :, 0]
            g_last = g_ref[0, 0, chunk_size - 1, 0].astype(jnp.float32)
            decayed_v_new = v_new * exp_fn(g_last - g_chunk)[:, None]
            g_last_decay = exp_fn(g_last)
            state_ref[...] *= g_last_decay
        if gk_ref is not None:
            gk_last = gk_ref[0, 0, chunk_size - 1].astype(jnp.float32)
            state_ref[...] *= exp_fn(gk_last)[:, None]

        state_ref[...] += jnp.dot(
            k_chunk.astype(jnp.float32).T,
            decayed_v_new.astype(jnp.float32),
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )

    if ht_ref is not None:

        @pl.when(chunk_idx == num_seq_chunks - 1)
        def _store_final_state() -> None:
            ht_ref[0, 0] = state_ref[...].astype(ht_ref.dtype)


def chunk_gated_delta_rule_fwd_h(
    k: jax.Array,
    w: jax.Array,
    u: jax.Array,
    g: jax.Array | None = None,
    gk: jax.Array | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    use_exp2: bool = True,
    cu_seqlens: jax.Array | None = None,
    chunk_indices: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array | None, jax.Array | None]:
    """Inter-chunk delta-rule state recurrence over chunk-aligned packed sequences.

    k, w, gk [1, T, H, K], u [1, T, H, V], g [1, T, H], initial_state [N, H,
    K, V]. Returns h [1, num_chunks, H, K, V], the state entering each chunk;
    v_new [1, T, H, V] (None unless save_new_value); and the final state [N, H,
    K, V] (None unless output_final_state).
    """
    batch_size, num_tokens, num_heads, head_dim = k.shape
    value_dim = u.shape[-1]
    assert cu_seqlens is not None, "This varlen-only module requires cu_seqlens"
    assert batch_size == 1, f"varlen mode requires B==1, got B={batch_size}"
    num_seqs = cu_seqlens.shape[-1] - 1
    assert_shape(w, (batch_size, num_tokens, num_heads, head_dim), "w")
    assert_shape(u, (batch_size, num_tokens, num_heads, value_dim), "u")
    assert_shape_or_none(g, (batch_size, num_tokens, num_heads), "g")
    assert_shape_or_none(gk, (batch_size, num_tokens, num_heads, head_dim), "gk")
    assert_shape_or_none(initial_state, (num_seqs, num_heads, head_dim, value_dim), "initial_state")
    assert head_dim <= MAX_HEAD_DIM, (
        f"current kernel does not support head dimension larger than {MAX_HEAD_DIM}."
    )
    assert chunk_indices is not None

    padded_head_dim = align_up(head_dim, NUM_LANES)
    padded_value_dim = align_up(value_dim, NUM_LANES)
    num_chunks = len(chunk_indices)

    def _to_head_major(packed: jax.Array, padded_dim: int) -> jax.Array:
        padded = jnp.pad(
            packed.astype(jnp.float32),
            ((0, 0), (0, chunk_size), (0, 0), (0, padded_dim - packed.shape[-1])),
        )
        return jnp.transpose(padded, (0, 2, 1, 3))

    def _token_index_map(
        seq_idx: jax.Array, head_idx: jax.Array, chunk_idx: jax.Array, cu_seqlens_ref: jax.Array
    ) -> tuple[int, jax.Array, jax.Array, int]:
        seq_start = pl.multiple_of(cu_seqlens_ref[seq_idx], chunk_size)
        block_idx = jnp.minimum(seq_start // chunk_size + chunk_idx, num_tokens // chunk_size)
        return (0, head_idx, block_idx, 0)

    def _h_index_map(
        seq_idx: jax.Array, head_idx: jax.Array, chunk_idx: jax.Array, cu_seqlens_ref: jax.Array
    ) -> tuple[int, jax.Array, jax.Array, int, int]:
        seq_start = pl.multiple_of(cu_seqlens_ref[seq_idx], chunk_size)
        h_chunk_idx = jnp.minimum(seq_start // chunk_size + chunk_idx, num_chunks - 1)
        return (0, h_chunk_idx, head_idx, 0, 0)

    def _seq_state_index_map(
        seq_idx: jax.Array, head_idx: jax.Array, chunk_idx: jax.Array, cu_seqlens_ref: jax.Array
    ) -> tuple[jax.Array, jax.Array, int, int]:
        return (seq_idx, head_idx, 0, 0)

    key_spec = pl.BlockSpec([1, 1, chunk_size, padded_head_dim], index_map=_token_index_map)
    value_spec = pl.BlockSpec([1, 1, chunk_size, padded_value_dim], index_map=_token_index_map)
    seq_state_spec = pl.BlockSpec(
        [1, 1, padded_head_dim, padded_value_dim], index_map=_seq_state_index_map
    )

    if g is None:
        g_head_major, g_spec = None, None
    else:
        g_head_major = _to_head_major(g.reshape(batch_size, num_tokens, num_heads, 1), NUM_LANES)
        g_spec = pl.BlockSpec([1, 1, chunk_size, NUM_LANES], index_map=_token_index_map)
    if gk is None:
        gk_head_major, gk_spec = None, None
    else:
        gk_head_major, gk_spec = _to_head_major(gk, padded_head_dim), key_spec
    if initial_state is None:
        h0, h0_spec = None, None
    else:
        h0 = jnp.pad(
            initial_state,
            ((0, 0), (0, 0), (0, padded_head_dim - head_dim), (0, padded_value_dim - value_dim)),
        )
        h0_spec = seq_state_spec
    if save_new_value:
        v_new_shape = jax.ShapeDtypeStruct(
            [batch_size, num_heads, num_tokens + chunk_size, padded_value_dim], jnp.float32
        )
        v_new_spec = value_spec
    else:
        v_new_shape, v_new_spec = None, None
    if output_final_state:
        final_state_shape = jax.ShapeDtypeStruct(
            [num_seqs, num_heads, padded_head_dim, padded_value_dim], jnp.float32
        )
        final_state_spec = seq_state_spec
    else:
        final_state_shape, final_state_spec = None, None

    h_padded, v_new_padded, final_state_padded = pl.pallas_call(
        functools.partial(chunk_gated_delta_rule_fwd_kernel, exp_fn=exp2 if use_exp2 else exp),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(num_seqs, num_heads, num_tokens // chunk_size),
            in_specs=[key_spec, value_spec, key_spec, g_spec, gk_spec, h0_spec],
            out_specs=[
                pl.BlockSpec(
                    [1, 1, 1, padded_head_dim, padded_value_dim], index_map=_h_index_map
                ),
                v_new_spec,
                final_state_spec,
            ],
            scratch_shapes=[pltpu.VMEM((padded_head_dim, padded_value_dim), jnp.float32)],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
        out_shape=[
            jax.ShapeDtypeStruct(
                [batch_size, num_chunks, num_heads, padded_head_dim, padded_value_dim],
                jnp.float32,
            ),
            v_new_shape,
            final_state_shape,
        ],
        interpret=get_interpret(),
    )(
        cu_seqlens,
        _to_head_major(k, padded_head_dim),
        _to_head_major(u, padded_value_dim),
        _to_head_major(w, padded_head_dim),
        g_head_major,
        gk_head_major,
        h0,
    )

    h = h_padded[:, :, :, :head_dim, :value_dim]
    if save_new_value:
        v_new = jnp.transpose(v_new_padded[:, :, :num_tokens, :value_dim], (0, 2, 1, 3))
    else:
        v_new = None
    if output_final_state:
        final_state = final_state_padded[:, :, :head_dim, :value_dim]
    else:
        final_state = None
    return h, v_new, final_state


def chunk_kda_fwd_o_gk_kernel(
    q_ref: jax.Array,
    v_ref: jax.Array,
    g_ref: jax.Array,
    h_ref: jax.Array,
    aqk_ref: jax.Array,
    o_ref: jax.Array,
    *,
    chunk_size: int,
    scale: float,
    exp_fn: Callable[[jax.Array], jax.Array],
) -> None:
    q_chunk = q_ref[0, 0]
    g_chunk = g_ref[0, 0]
    v_chunk = v_ref[0, 0]
    h_chunk = h_ref[0, 0]
    aqk_chunk = aqk_ref[0, 0]
    g = g_chunk.astype(jnp.float32)
    q = q_chunk.astype(jnp.float32)

    # Note (david): exp(g[t]) is split as exp(g[t] - g[0]) * exp(g[0]) with
    # exp(g[0]) folded into h. g is a decreasing cumsum, so g[t] - g[0] <= 0
    # and neither factor overflows.
    g_first = g[0:1, :]
    qg = q * exp_fn(jnp.maximum(g - g_first, MIN_NORMAL_F32_EXPONENT))
    h_scaled = (
        h_chunk.astype(jnp.float32)
        * exp_fn(jnp.maximum(g_first[0], MIN_NORMAL_F32_EXPONENT))[:, None]
    )
    o = jnp.dot(
        qg, h_scaled, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32
    )
    o = o * scale

    is_causal = jnp.arange(chunk_size)[:, None] >= jnp.arange(chunk_size)[None, :]
    aqk = jnp.where(is_causal, aqk_chunk, 0.0).astype(jnp.float32)
    o = o + jnp.dot(
        aqk,
        v_chunk.astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )

    o_ref[0, 0] = o.astype(o_ref.dtype)


def chunk_kda_fwd_o_gk(
    q: jax.Array,
    v: jax.Array,
    g: jax.Array,
    A: jax.Array,
    h: jax.Array,
    scale: float,
    *,
    cu_seqlens: jax.Array,
    chunk_indices: jax.Array | None = None,
    chunk_size: int = 64,
    use_exp2: bool = False,
) -> jax.Array:
    """KDA output over chunk-aligned packed sequences.

    q, g [1, T, H, K], v [1, T, H, V], A [1, T, H, chunk_size] (Aqk), h [1,
    num_chunks, H, K, V]. Returns o [1, T, H, V]: the gated q against h plus
    the causal A against v.
    """
    assert cu_seqlens is not None, "This varlen-only module requires cu_seqlens"
    batch_size, num_tokens, num_heads, head_dim = q.shape
    value_dim = v.shape[-1]
    assert batch_size == 1
    assert num_tokens % chunk_size == 0

    chunk_starts, row_positions = chunk_slot_layout(
        cu_seqlens, chunk_size, chunk_indices, num_tokens
    )
    num_slots = chunk_starts.shape[0]
    q_chunks, v_chunks, g_chunks, aqk_chunks = (
        gather_chunks(packed, chunk_starts, chunk_size) for packed in (q, v, g, A)
    )

    num_h_chunks = h.shape[1]
    h_head_major = h[0].transpose(1, 0, 2, 3)
    if num_slots > num_h_chunks:
        h_chunks = jnp.pad(h_head_major, ((0, 0), (0, num_slots - num_h_chunks), (0, 0), (0, 0)))
    elif num_slots < num_h_chunks:
        h_chunks = h_head_major[:, :num_slots]
    else:
        h_chunks = h_head_major

    def _chunk_index_map(
        head_idx: jax.Array, slot_idx: jax.Array
    ) -> tuple[jax.Array, jax.Array, int, int]:
        return (head_idx, slot_idx, 0, 0)

    key_spec = pl.BlockSpec([1, 1, chunk_size, head_dim], index_map=_chunk_index_map)
    value_spec = pl.BlockSpec([1, 1, chunk_size, value_dim], index_map=_chunk_index_map)
    o_chunks = pl.pallas_call(
        functools.partial(
            chunk_kda_fwd_o_gk_kernel,
            chunk_size=chunk_size,
            scale=scale,
            exp_fn=exp2 if use_exp2 else exp,
        ),
        grid=(num_heads, num_slots),
        out_shape=jax.ShapeDtypeStruct([num_heads, num_slots, chunk_size, value_dim], v.dtype),
        in_specs=[
            key_spec,
            value_spec,
            key_spec,
            pl.BlockSpec([1, 1, head_dim, value_dim], index_map=_chunk_index_map),
            pl.BlockSpec([1, 1, chunk_size, chunk_size], index_map=_chunk_index_map),
        ],
        out_specs=value_spec,
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
        interpret=get_interpret(),
    )(q_chunks, v_chunks, g_chunks, h_chunks, aqk_chunks)

    return scatter_chunks(o_chunks, row_positions, num_tokens)


def kda_gate_chunk_cumsum(
    g: jax.Array,
    A_log: jax.Array,
    chunk_size: int,
    scale: float | None = None,
    dt_bias: jax.Array | None = None,
    cu_seqlens: jax.Array | None = None,
    output_dtype: jnp.dtype | None = jnp.float32,
    chunk_indices: jax.Array | None = None,
    lower_bound: float | None = None,
) -> jax.Array:
    """KDA gate activation followed by chunk_local_cumsum_vector.

    g [B, T, H, K] is the raw gate, A_log [H], dt_bias [H * K]. The activation
    is -exp(A_log) * softplus(g + dt_bias), or lower_bound * sigmoid(exp(A_log)
    * (g + dt_bias)) when lower_bound is set.
    """
    _, _, num_heads, head_dim = g.shape
    assert A_log.shape == (num_heads,), f"A_log shape {A_log.shape} != ({num_heads},)"

    if dt_bias is None:
        gate = g.astype(jnp.float32)
    else:
        gate = g.astype(jnp.float32) + dt_bias.astype(jnp.float32).reshape(num_heads, head_dim)
    decay_rate = exp(A_log.astype(jnp.float32)).reshape(1, 1, num_heads, 1)
    if lower_bound is None:
        gate_activation = -decay_rate * jax.nn.softplus(gate)
    else:
        gate_activation = lower_bound * jax.nn.sigmoid(decay_rate * gate)

    return chunk_local_cumsum_vector(
        gate_activation,
        chunk_size=chunk_size,
        scale=scale,
        cu_seqlens=cu_seqlens,
        head_first=False,
        output_dtype=output_dtype,
        chunk_indices=chunk_indices,
    )


def pallas_kda_gate_cumsum(
    g: jax.Array,
    chunk_size: int,
    reverse: bool = False,
    scale: float | None = RCP_LN2,
    cu_seqlens: jax.Array | None = None,
    head_first: bool = False,
    output_dtype: jnp.dtype | None = jnp.float32,
    chunk_indices: jax.Array | None = None,
) -> jax.Array:
    """chunk_local_cumsum_vector for T divisible by chunk_size, scaled by 1 / ln 2 by default."""
    num_tokens = g.shape[2 if head_first else 1]
    assert num_tokens % chunk_size == 0, (
        f"T={num_tokens} must be divisible by chunk_size={chunk_size}"
    )

    return chunk_local_cumsum_vector(
        g,
        chunk_size=chunk_size,
        reverse=reverse,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        head_first=head_first,
        output_dtype=output_dtype,
    )


def segment_gather_index(
    dst_starts: jax.Array,
    src_starts: jax.Array,
    seg_lens: jax.Array,
    num_positions: int,
    fill_index: int,
) -> jax.Array:
    """Source index of every destination position when moving segments.

    Segment i covers seg_lens[i] positions from dst_starts[i] in the
    destination and from src_starts[i] in the source; positions outside every
    segment get fill_index.
    """

    def _place_segment(seg_idx: jax.Array, gather_idx: jax.Array) -> jax.Array:
        positions = jnp.arange(num_positions)
        dst_start = dst_starts[seg_idx]
        in_segment = (positions >= dst_start) & (positions < dst_start + seg_lens[seg_idx])
        return jnp.where(in_segment, src_starts[seg_idx] + (positions - dst_start), gather_idx)

    return jax.lax.fori_loop(
        0,
        seg_lens.shape[0],
        _place_segment,
        jnp.full(num_positions, fill_index, dtype=jnp.int32),
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "scale",
        "output_final_state",
        "use_qk_l2norm_in_kernel",
        "chunk_size",
        "safe_gate",
        "lower_bound",
        "use_gate_in_kernel",
        "disable_recompute",
        "return_intermediate_states",
        "cp_context",
        "transpose_state_layout",
    ),
)
def chunk_kda_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    scale: float,
    initial_state: jax.Array | None,
    output_final_state: bool,
    cu_seqlens: jax.Array,
    use_qk_l2norm_in_kernel: bool = False,
    chunk_indices: jax.Array | None = None,
    chunk_size: int = 64,
    safe_gate: bool = True,
    lower_bound: float | None = None,
    use_gate_in_kernel: bool = False,
    A_log: jax.Array | None = None,
    dt_bias: jax.Array | None = None,
    disable_recompute: bool = False,
    return_intermediate_states: bool = False,
    cp_context: None = None,
    transpose_state_layout: bool = False,
) -> tuple[jax.Array | None, ...]:
    """KDA chunked forward pass over packed variable-length sequences.

    q, k, g [1, T, H, K], v [1, T, H, V], beta [1, T, H], initial_state [N, H,
    K, V] or None, cu_seqlens [N + 1]. chunk_indices is rebuilt for the
    chunk-aligned layout. Returns the upstream 12-tuple (o, final_state,
    g_cumsum, Aqk, Akk, w, u, qg, kg, v_new, h, initial_state): o is [1, T, H,
    V]; g_cumsum (None when use_gate_in_kernel), Aqk and Akk are in the
    chunk-aligned token layout; w, u, qg, kg, v_new and h are None. The
    chunk_indices argument is ignored.
    """
    batch_size, num_tokens, num_heads, head_dim = q.shape
    value_dim = v.shape[-1]

    assert use_qk_l2norm_in_kernel is False
    assert cp_context is None
    assert not transpose_state_layout
    assert not return_intermediate_states
    assert not disable_recompute

    assert_shape(k, (batch_size, num_tokens, num_heads, head_dim), "k")
    assert_shape(v, (batch_size, num_tokens, num_heads, value_dim), "v")
    assert cu_seqlens is not None, "cu_seqlens must not be None for varlen path"
    assert batch_size == 1, f"varlen requires B=1 (packed layout), got B={batch_size}"

    num_seqs = cu_seqlens.shape[-1] - 1
    assert_shape(beta, (batch_size, num_tokens, num_heads), "beta")
    assert_shape_or_none(initial_state, (num_seqs, num_heads, head_dim, value_dim), "initial_state")

    seq_lens = prepare_lens(cu_seqlens)
    aligned_cu_seqlens = jnp.concatenate(
        [jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(align_up(seq_lens, chunk_size))]
    )
    num_aligned_tokens = align_up(num_tokens + num_seqs * (chunk_size - 1), chunk_size)
    align_gather_idx = segment_gather_index(
        dst_starts=aligned_cu_seqlens[:-1],
        src_starts=cu_seqlens[:-1],
        seg_lens=seq_lens,
        num_positions=num_aligned_tokens,
        fill_index=num_tokens,
    )
    q_aligned, k_aligned, v_aligned, g_aligned, beta_aligned = (
        jnp.pad(
            packed,
            ((0, 0), (0, num_aligned_tokens - num_tokens)) + ((0, 0),) * (packed.ndim - 2),
        )[:, align_gather_idx]
        for packed in (q, k, v, g, beta)
    )
    aligned_chunk_indices = prepare_chunk_indices(
        aligned_cu_seqlens, chunk_size, max_T=num_aligned_tokens
    )

    if use_gate_in_kernel:
        assert A_log is not None
        # Note (david): alignment pads g with 0, but softplus(0 + dt_bias) != 0
        # would give padding rows a non-zero gate that corrupts g_last and kg
        # in the state recurrence; a large negative logit makes it ~0.
        aligned_seq_starts = aligned_cu_seqlens[:-1]
        positions = jnp.arange(num_aligned_tokens)
        is_real_token = (
            (positions[None, :] >= aligned_seq_starts[:, None])
            & (positions[None, :] < (aligned_seq_starts + seq_lens)[:, None])
        ).any(axis=0)
        g_cumsum = kda_gate_chunk_cumsum(
            g=jnp.where(is_real_token[None, :, None, None], g_aligned, PAD_GATE_LOGIT),
            A_log=A_log,
            chunk_size=chunk_size,
            scale=RCP_LN2,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            cu_seqlens=aligned_cu_seqlens,
            chunk_indices=aligned_chunk_indices,
        )
    else:
        g_cumsum = pallas_kda_gate_cumsum(
            g=g_aligned,
            scale=RCP_LN2,
            chunk_size=chunk_size,
            cu_seqlens=aligned_cu_seqlens,
            chunk_indices=aligned_chunk_indices,
        )

    w, u, _, kg, aqk, akk = kda_fwd_intra(
        q=q_aligned,
        k=k_aligned,
        v=v_aligned,
        gk=g_cumsum,
        beta=beta_aligned,
        scale=scale,
        safe_gate=safe_gate,
        chunk_size=chunk_size,
        cu_seqlens=aligned_cu_seqlens,
        chunk_indices=aligned_chunk_indices,
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=kg,
        w=w,
        u=u,
        gk=g_cumsum,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        use_exp2=True,
        cu_seqlens=aligned_cu_seqlens,
        chunk_indices=aligned_chunk_indices,
    )
    o_aligned = chunk_kda_fwd_o_gk(
        q=q_aligned,
        v=v_new,
        g=g_cumsum,
        A=aqk,
        h=h,
        scale=scale,
        chunk_size=chunk_size,
        use_exp2=True,
        cu_seqlens=aligned_cu_seqlens,
        chunk_indices=aligned_chunk_indices,
    ).astype(q.dtype)

    unalign_gather_idx = segment_gather_index(
        dst_starts=cu_seqlens[:-1],
        src_starts=aligned_cu_seqlens[:-1],
        seg_lens=seq_lens,
        num_positions=num_tokens,
        fill_index=0,
    )
    o = o_aligned[:, unalign_gather_idx]

    if use_gate_in_kernel:
        returned_g_cumsum = None
    else:
        returned_g_cumsum = g_cumsum
    # Note (david): the training intermediates w, u, qg, kg, v_new and h are
    # returned as None so XLA can free them; the tuple keeps the upstream shape.
    return (
        o,
        final_state,
        returned_g_cumsum,
        aqk,
        akk,
        None,
        None,
        None,
        None,
        None,
        None,
        initial_state,
    )
