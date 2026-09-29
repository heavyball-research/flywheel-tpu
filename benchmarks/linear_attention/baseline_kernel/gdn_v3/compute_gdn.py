# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.numpy as jnp

from gdn_v3 import config

L2_NORM_EPS = 1e-6
INVERSE_BLOCK_SIZE = 16


def get_mask_dtype(dtype: jnp.dtype) -> jnp.dtype:
    """Integer dtype as wide as dtype, for iota masks over dtype data."""
    match jnp.dtype(dtype).itemsize:
        case 4:
            return jnp.int32
        case 2:
            return jnp.int16
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")


def invert_triangular_matrix(
    lower_tri: jax.Array,
    block_size: int = INVERSE_BLOCK_SIZE,
) -> jax.Array:
    """Inverse of unit lower triangular matrices [batch, n, n].

    Solved by blocked forward substitution.
    """
    out_dtype = lower_tri.dtype
    size = lower_tri.shape[-1]
    block_size = min(block_size, size)
    num_blocks = size // block_size

    def _forward_substitute(tri_block: jax.Array,
                            rhs_block: jax.Array) -> jax.Array:
        solved_rows = []
        for row in range(block_size):
            rhs_row = rhs_block[:, row, :]
            if row == 0:
                solved_row = rhs_row
            else:
                solved = jnp.stack(solved_rows, axis=1)
                prev_coeffs = tri_block[:, row, :row]
                prev_sum = jnp.sum(prev_coeffs[..., None] * solved, axis=1)
                solved_row = rhs_row - prev_sum
            solved_rows.append(solved_row)
        return jnp.stack(solved_rows, axis=1)

    solved_blocks = []
    row_iota = jax.lax.broadcasted_iota(jnp.int32, lower_tri.shape, 1)
    col_iota = jax.lax.broadcasted_iota(jnp.int32, lower_tri.shape, 2)
    identity = jnp.where(row_iota == col_iota, 1.0, 0.0)
    for block in range(num_blocks):
        start, end = block * block_size, (block + 1) * block_size
        identity_block = identity[:, start:end, :]

        if block == 0:
            rhs = identity_block
        else:
            prev_sum = jax.lax.dot(
                lower_tri[:, start:end, :start],
                jnp.concatenate(solved_blocks, axis=1),
                dimension_numbers=(((2, ), (1, )), ((0, ), (0, ))),
                preferred_element_type=jnp.float32,
            )
            rhs = identity_block - prev_sum

        # Note (david): the diagonal block is solved in fp32 to minimize the
        # cost of sublane rolling.
        diag_block = lower_tri[:, start:end, start:end].astype(jnp.float32)
        solved_blocks.append(
            _forward_substitute(diag_block, rhs).astype(out_dtype))

    return jnp.concatenate(solved_blocks, axis=1)


def fused_transpose_broadcast(values: jax.Array, src_dim: int,
                              dst_dim: int) -> jax.Array:
    """Moves axis src_dim of values into its size-1 axis dst_dim.

    Axis src_dim of the result has size 1.
    """
    assert values.shape[dst_dim] == 1

    dtype = values.dtype
    mask_dtype = get_mask_dtype(dtype)
    mask_shape = list(values.shape)
    mask_shape[dst_dim] = mask_shape[src_dim]
    src_iota = jax.lax.broadcasted_iota(mask_dtype, mask_shape, src_dim)
    dst_iota = jax.lax.broadcasted_iota(mask_dtype, mask_shape, dst_dim)
    is_same_index = src_iota == dst_iota
    return jnp.where(is_same_index, values, 0).sum(axis=src_dim,
                                                   keepdims=True,
                                                   dtype=dtype)


def prepare_gdn_inputs(
    real_sizes: jax.Array,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    beta_input: jax.Array,
    decay_input: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    cfg: config.GDNConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Masks rows past real_sizes, l2-normalizes q and k, derives the gates.

    q, k, v and the gate inputs b (beta_input) and a (decay_input) are [seq,
    heads, chunk, ...] in either layout. Returns q, k, v, beta and gating_log
    in cfg.dtypes.compute.
    """
    mask_dtype = get_mask_dtype(cfg.dtypes.compute)
    trailing_dims = (1, ) * (q.ndim - 3)
    token_iota = jax.lax.broadcasted_iota(
        mask_dtype, (cfg.seq_tile_size, 1, cfg.chunk_size) + trailing_dims, 2)
    is_real_token = token_iota < real_sizes.reshape(
        (-1, 1, 1) + trailing_dims).astype(mask_dtype)

    q = jnp.where(is_real_token, q.astype(cfg.dtypes.compute), 0)
    k = jnp.where(is_real_token, k.astype(cfg.dtypes.compute), 0)
    v = jnp.where(is_real_token, v.astype(cfg.dtypes.compute), 0)

    beta_input = beta_input.astype(cfg.dtypes.compute)
    decay_input = decay_input.astype(cfg.dtypes.compute)

    per_head_shape = (1, ) * (q.ndim - 1) + (-1, )
    a_log = a_log.reshape(per_head_shape).astype(cfg.dtypes.compute)
    dt_bias = dt_bias.reshape(per_head_shape).astype(cfg.dtypes.compute)

    # Note (david): elementwise work runs before the kq-to-v head repeat, so it
    # only touches num_kq_heads rows.
    q_norm = jnp.sqrt(
        jnp.sum(q * q, axis=-1, keepdims=True, dtype=q.dtype) + L2_NORM_EPS)
    q = q / q_norm * cfg.kq_head_dim**-0.5
    k_norm = jnp.sqrt(
        jnp.sum(k * k, axis=-1, keepdims=True, dtype=k.dtype) + L2_NORM_EPS)
    k = k / k_norm

    beta = jax.nn.sigmoid(beta_input)
    gating_log = -jnp.exp(a_log) * jax.nn.softplus(decay_input + dt_bias)
    beta = jnp.where(is_real_token, beta, 0)
    # Note (david): padded rows get gating_log 0, a decay of exp(0) = 1, so the
    # state passes through them unchanged.
    gating_log = jnp.where(is_real_token, gating_log, 0)

    return q, k, v, beta, gating_log


def chunked_gdn_per_seq(
    q_large: jax.Array,
    k_large: jax.Array,
    v_large: jax.Array,
    gating_log: jax.Array,
    beta: jax.Array,
    state_prev: jax.Array,
    cfg: config.GDNConfig,
) -> tuple[jax.Array, jax.Array]:
    """Chunked GDN over one sequence in the large layout.

    Takes q_large, k_large [num_kq_heads, chunk, kq_head_dim], v_large
    [num_v_heads, chunk, v_head_dim], gating_log and beta [1, chunk,
    aligned_num_v_heads] and state_prev [num_v_heads, kq_head_dim,
    v_head_dim]. Returns out [num_v_heads, chunk, v_head_dim] and the final
    state.
    """
    # Note (david): repeating along a non lane / sublane dim is free.
    q_repeat = jnp.repeat(q_large, cfg.v_per_kq_head, axis=0)
    k_repeat = jnp.repeat(k_large, cfg.v_per_kq_head, axis=0)

    g_cum_sum_list = [gating_log[:, :1]]
    for row in range(1, cfg.chunk_size):
        g_cum_sum_list.append(g_cum_sum_list[-1] + gating_log[:, row:row + 1])
    g_cum_sum_log = jnp.concat(g_cum_sum_list, axis=1)

    g_cum_sum_log = fused_transpose_broadcast(g_cum_sum_log,
                                              src_dim=2,
                                              dst_dim=0)
    g_cum_sum_log = g_cum_sum_log[:cfg.num_v_heads]
    beta = fused_transpose_broadcast(beta, src_dim=2, dst_dim=0)
    beta_large = beta[:cfg.num_v_heads]

    g_cum_sum_log_t = fused_transpose_broadcast(g_cum_sum_log,
                                                src_dim=1,
                                                dst_dim=2)
    g_cum_sum_diff_log = g_cum_sum_log - g_cum_sum_log_t
    gating_map = jnp.exp(g_cum_sum_diff_log)
    gating_backward = jnp.exp(-g_cum_sum_diff_log[..., -1:])
    gating_forward = jnp.exp(g_cum_sum_log)
    gating_last = gating_forward[:, -1:]

    mask_dtype = get_mask_dtype(cfg.dtypes.compute)
    row_iota = jax.lax.broadcasted_iota(mask_dtype, gating_map.shape, 1)
    col_iota = jax.lax.broadcasted_iota(mask_dtype, gating_map.shape, 2)
    is_diagonal = row_iota == col_iota
    is_strictly_lower = row_iota > col_iota
    is_lower = row_iota >= col_iota
    gating_map_masked = jnp.where(is_strictly_lower, gating_map, 0)

    k_beta_repeat = k_repeat * beta_large
    beta_k_k_t = jax.lax.dot(
        k_beta_repeat,
        k_repeat,
        dimension_numbers=(((2, ), (2, )), ((0, ), (0, ))),
        preferred_element_type=jnp.float32,
    ).astype(cfg.dtypes.compute)
    lower_tri = jnp.where(is_diagonal, 1, gating_map_masked * beta_k_k_t)
    lower_tri_inv = invert_triangular_matrix(lower_tri)

    v_beta_large = v_large * beta_large
    k_beta_gating = k_beta_repeat * gating_forward
    # Note (david): one matmul over [v, k] raises MXU utilization when
    # v_head_dim is below the MXU size, and the lane-dim concat / split is free
    # when v_head_dim is a multiple of the lane count.
    merged_v_k = jnp.concat([v_beta_large, k_beta_gating], axis=-1)
    merged_wy = jax.lax.dot(
        lower_tri_inv,
        merged_v_k,
        dimension_numbers=(((2, ), (1, )), ((0, ), (0, ))),
        preferred_element_type=jnp.float32,
    ).astype(cfg.dtypes.compute)
    wy_v, wy_k = jnp.split(merged_wy, [cfg.v_head_dim], axis=-1)

    q_large_gating = q_repeat * gating_forward
    # Note (david): stacking both lhs against the same rhs keeps the weights
    # stationary in the MXU.
    merged_wy_k_q = jnp.concat([wy_k, q_large_gating], axis=1)
    merged_state_products = jax.lax.dot(
        merged_wy_k_q,
        state_prev,
        dimension_numbers=(((2, ), (1, )), ((0, ), (0, ))),
        preferred_element_type=jnp.float32,
    )

    # Note (david): splitting along a non lane / sublane dim is free.
    wy_k_state, out_inter = jnp.split(merged_state_products, 2, axis=1)
    v_new = wy_v - wy_k_state.astype(cfg.dtypes.compute)

    k_repeat_gating = k_repeat * gating_backward
    state_delta = jax.lax.dot(
        k_repeat_gating,
        v_new,
        dimension_numbers=(((1, ), (1, )), ((0, ), (0, ))),
        preferred_element_type=jnp.float32,
    )
    state = state_prev * gating_last + state_delta

    attn_scores = jax.lax.dot(
        q_large,
        k_large,
        dimension_numbers=(((2, ), (2, )), ((0, ), (0, ))),
        preferred_element_type=jnp.float32,
    ).astype(cfg.dtypes.compute)
    # Note (david): repeat after the matmul so the matmul only runs on
    # num_kq_heads.
    attn_scores = jnp.repeat(attn_scores, cfg.v_per_kq_head, axis=0)
    attn_scores *= gating_map
    attn_scores = jnp.where(is_lower, attn_scores, 0)

    out_intra = jax.lax.dot(
        attn_scores,
        v_new,
        dimension_numbers=(((2, ), (1, )), ((0, ), (0, ))),
        preferred_element_type=jnp.float32,
    )

    return out_inter + out_intra, state


def chunked_gdn(
    real_sizes: jax.Array,
    q_large: jax.Array,
    k_large: jax.Array,
    v_large: jax.Array,
    b_large: jax.Array,
    a_large: jax.Array,
    state_prev: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    cfg: config.GDNConfig,
) -> tuple[jax.Array, jax.Array]:
    """Chunked GDN over a tile in the large layout [seq, heads, chunk, dim].

    Returns out [seq, chunk, num_v_heads, v_head_dim] and the state [seq, 1,
    num_v_heads, kq_head_dim, v_head_dim]. The chunked path only produces the
    final state, so it never runs when more than one checkpoint per sequence
    is kept.
    """
    assert not cfg.use_recurrent

    q_large, k_large, v_large, beta, gating_log = prepare_gdn_inputs(
        real_sizes, q_large, k_large, v_large, b_large, a_large, a_log,
        dt_bias, cfg)

    out_list = []
    state_list = []
    for idx in range(cfg.seq_tile_size):
        out, state = chunked_gdn_per_seq(
            q_large[idx],
            k_large[idx],
            v_large[idx],
            gating_log[idx],
            beta[idx],
            state_prev[idx],
            cfg,
        )
        out_list.append(out.swapaxes(0, 1))
        state_list.append(state)
    out = jnp.stack(out_list, axis=0)
    # Note (david): the single checkpoint gets the window axis of the recurrent
    # path's layout.
    state = jnp.stack(state_list, axis=0)[:, jnp.newaxis]
    return out, state


def recurrent_gdn_per_seq(
    q_compact: jax.Array,
    k_compact: jax.Array,
    k_compact_t: jax.Array,
    v_compact: jax.Array,
    gating: jax.Array,
    beta: jax.Array,
    state: jax.Array,
    cfg: config.GDNConfig,
) -> tuple[jax.Array, jax.Array]:
    """Token-recurrent GDN over one sequence in the compact layout.

    Takes q_compact, k_compact [num_kq_heads, chunk, 1, kq_head_dim],
    k_compact_t [num_kq_heads, chunk, kq_head_dim, 1], v_compact [num_v_heads,
    chunk, 1, v_head_dim], gating (the decay, not its log) and beta
    [num_v_heads, chunk, 1, 1] and state [num_v_heads, kq_head_dim,
    v_head_dim]. Returns out [chunk, num_v_heads, v_head_dim] and the states
    after the last window_size positions [window_size, num_v_heads,
    kq_head_dim, v_head_dim].
    """
    out_list = []
    state_list = []
    for c_idx in range(cfg.chunk_size):
        q_curr = jnp.repeat(q_compact[:, c_idx], cfg.v_per_kq_head, axis=0)
        k_curr = jnp.repeat(k_compact[:, c_idx], cfg.v_per_kq_head, axis=0)
        v_curr = v_compact[:, c_idx]
        k_curr_t = jnp.repeat(k_compact_t[:, c_idx], cfg.v_per_kq_head, axis=0)
        beta_curr = beta[:, c_idx]
        gating_curr = gating[:, c_idx]

        decayed_state = state * gating_curr
        v_pred = jax.lax.dot(
            k_curr,
            decayed_state,
            dimension_numbers=(((2, ), (1, )), ((0, ), (0, ))),
            preferred_element_type=jnp.float32,
        ).astype(cfg.dtypes.compute)
        v_new = beta_curr * (v_curr - v_pred)

        # Note (david): the multiply by k_curr_t is deferred as long as
        # possible because it expands the tensor by kq_head_dim.
        state = decayed_state + k_curr_t * v_new

        out = jax.lax.dot(
            q_curr,
            state,
            dimension_numbers=(((2, ), (1, )), ((0, ), (0, ))),
            preferred_element_type=jnp.float32,
        ).astype(cfg.dtypes.compute)

        out_list.append(out[:, 0, :])
        # Note (david): the caller masks rows past real_sizes, so the state
        # stops changing there and trailing checkpoints repeat the state of the
        # last real token.
        if c_idx >= cfg.chunk_size - cfg.window_size:
            state_list.append(state)

    return jnp.stack(out_list, axis=0), jnp.stack(state_list, axis=0)


def recurrent_gdn(
    real_sizes: jax.Array,
    q_compact: jax.Array,
    k_compact: jax.Array,
    v_compact: jax.Array,
    b_compact: jax.Array,
    a_compact: jax.Array,
    state_prev: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    cfg: config.GDNConfig,
) -> tuple[jax.Array, jax.Array]:
    """Token-recurrent GDN over a tile in the compact layout.

    Inputs are [seq, heads, chunk, 1, dim]. Returns out [seq, chunk,
    num_v_heads, v_head_dim] and one state checkpoint per window position
    [seq, window_size, num_v_heads, kq_head_dim, v_head_dim]; positions past
    real_sizes repeat the last real state and are never written back to HBM.
    """
    q_compact, k_compact, v_compact, beta, gating_log = prepare_gdn_inputs(
        real_sizes, q_compact, k_compact, v_compact, b_compact, a_compact,
        a_log, dt_bias, cfg)
    k_compact_t = fused_transpose_broadcast(k_compact, src_dim=4, dst_dim=3)
    gating = jnp.exp(gating_log)

    beta = fused_transpose_broadcast(beta, src_dim=4, dst_dim=1)
    beta = beta[:, :cfg.num_v_heads]
    gating = fused_transpose_broadcast(gating, src_dim=4, dst_dim=1)
    gating = gating[:, :cfg.num_v_heads]

    out_list = []
    state_list = []
    for idx in range(cfg.seq_tile_size):
        out, states = recurrent_gdn_per_seq(
            q_compact[idx],
            k_compact[idx],
            k_compact_t[idx],
            v_compact[idx],
            gating[idx],
            beta[idx],
            state_prev[idx],
            cfg,
        )
        out_list.append(out)
        state_list.append(states)

    return jnp.stack(out_list, axis=0), jnp.stack(state_list, axis=0)
