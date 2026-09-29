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
"""Causal depthwise conv1d over tokens in the native [*, rows, dim] layout."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from . import config

MXU_LANES = 256
NUM_LANES = 128
# Note (david): one bf16 sublane tile, whose last prev_kernel_size rows are the
# tokens preceding a chunk.
HALO_ROWS = 16


def token_rows(prev_tokens: jax.Array, chunk_tokens: jax.Array, start: int,
               count: int) -> jax.Array:
    """Rows [start, start + count) of concat([prev_tokens, chunk_tokens], 1).

    prev_tokens is [seq, prev_kernel_size, dim] and chunk_tokens [seq, chunk,
    dim]. start and count are static, so the split between the two operands
    resolves at trace time and the concatenation is never materialised.
    """
    num_prev = prev_tokens.shape[1]
    num_from_prev = min(max(num_prev - start, 0), count)
    num_from_chunk = count - num_from_prev
    row_slices = []
    if num_from_prev:
        row_slices.append(prev_tokens[:, start:start + num_from_prev])
    if num_from_chunk:
        chunk_start = start + num_from_prev - num_prev
        chunk_end = chunk_start + num_from_chunk
        row_slices.append(chunk_tokens[:, chunk_start:chunk_end])
    assert row_slices, (start, count)
    if len(row_slices) == 1:
        return row_slices[0]
    else:
        return jnp.concatenate(row_slices, axis=1)


def tap_select_matrix(chunk: int, kernel_size: int) -> jax.Array:
    """One-hot [chunk, kernel_size * chunk] bf16 that sums each token's taps.

    With p = kernel_size - 1, the dot's W stacks kernel_size copies of the
    chunk along K, copy k scaled by tap k. Output row t reads source token
    t + k - p from copy k; when that lies before the chunk (t + k < p) it is
    halo token j = t + k, which copy k keeps at row chunk - p + j: tap k never
    reads block rows past chunk - 1 + k - p, so its last p - k rows are free,
    and j >= k puts every halo token at the same row in every copy that needs
    it.
    """
    # Note (david): built from iotas because pallas_call rejects captured array
    # constants.
    prev_kernel = kernel_size - 1
    shape = (chunk, kernel_size * chunk)
    out_row = jax.lax.broadcasted_iota(jnp.int32, shape, 0)
    w_row = jax.lax.broadcasted_iota(jnp.int32, shape, 1)
    is_selected = None
    for tap in range(kernel_size):
        shifted_row = out_row + (tap - prev_kernel)
        src_row = jnp.where(shifted_row >= 0, shifted_row, shifted_row + chunk)
        is_tap_src = w_row == tap * chunk + src_row
        is_selected = (is_tap_src if is_selected is None else
                       jnp.logical_or(is_selected, is_tap_src))
    return jnp.where(is_selected, 1.0, 0.0).astype(jnp.bfloat16)


def causal_conv1d_static(
    qkv_view_ref: jax.Ref,
    halo: jax.Array,
    conv_weight: jax.Array,
    conv_bias: jax.Array | None,
    cfg: config.KDAConfig,
) -> jax.Array:
    """Causal conv1d of one compute chunk on the MXU, [1, chunk, dim] f32.

    qkv_view_ref is the bf16 [1, >= compute_chunk_size, dim] view of the block
    grid, halo the bf16 [HALO_ROWS, dim] tile whose last prev_kernel_size rows
    are the tokens before row 0, conv_weight [kernel_size, dim] and conv_bias
    [dim]. Rows before a sequence's first token are pad: their outputs are
    garbage but finite and masked downstream. Every row of the block and the
    halo must be finite, since the one-hot dot multiplies them by 0.
    """
    assert cfg.mode == config.KDAMode.PER_SEQ
    prev_kernel = cfg.prev_kernel_size
    chunk = cfg.compute_chunk_size
    dim = qkv_view_ref.shape[-1]
    assert halo.shape == (HALO_ROWS, dim), halo.shape
    lanes_per_dot = MXU_LANES if dim % MXU_LANES == 0 else NUM_LANES
    assert dim % lanes_per_dot == 0, dim
    tap_select = tap_select_matrix(chunk, cfg.kernel_size)
    tap_weights = conv_weight.astype(jnp.bfloat16)
    halo_row = jax.lax.broadcasted_iota(jnp.int32,
                                        (HALO_ROWS, lanes_per_dot), 0)

    # Note (david): per lane slab one dot tap_select @ W, W the kernel_size
    # tap-scaled copies of the chunk with each copy's unused tail rows holding
    # the tap-scaled halo, so K is exactly kernel_size * chunk (one 256-deep
    # MXU pass at chunk 64) and the f32 accumulator does the tap sum. Each tap
    # product is rounded to bf16 once before it is accumulated.
    slab_outs = []
    for slab in range(dim // lanes_per_dot):
        lanes = slice(slab * lanes_per_dot, (slab + 1) * lanes_per_dot)
        chunk_tokens = qkv_view_ref[0, :chunk, lanes]
        halo_tokens = halo[:, lanes]
        tap_copies = []
        for tap in range(cfg.kernel_size):
            tap_weight = tap_weights[tap, lanes][jnp.newaxis]
            scaled_chunk = tap_weight * chunk_tokens
            # Note (david): rows chunk - prev_kernel_size + j, j >= tap, of copy
            # tap carry halo token j.
            copy_tail = jnp.where(halo_row >= HALO_ROWS - prev_kernel + tap,
                                  tap_weight * halo_tokens,
                                  scaled_chunk[chunk - HALO_ROWS:])
            if chunk == HALO_ROWS:
                tap_copies.append(copy_tail)
            else:
                tap_copies.append(
                    jnp.concatenate(
                        [scaled_chunk[:chunk - HALO_ROWS], copy_tail], axis=0))
        slab_outs.append(
            jax.lax.dot(tap_select, jnp.concatenate(tap_copies, axis=0),
                        preferred_element_type=jnp.float32))
    conv_out = jnp.concatenate(slab_outs, axis=-1)
    if conv_bias is None:
        biased_out = conv_out
    else:
        biased_out = conv_out + conv_bias[jnp.newaxis]
    return biased_out[jnp.newaxis]


def make_halo(prev_conv: jax.Array, delta: jax.Array, block_head_ref: jax.Ref,
              cfg: config.KDAConfig) -> jax.Array:
    """Place a tile's carried conv state on the block grid.

    prev_conv is the [prev_kernel_size, dim] f32 run of tokens before the
    tile's first real token, which sits at row delta of the block, and
    block_head_ref the block's first HALO_ROWS rows. Returns the halo tile in
    the block's dtype.
    """
    prev_kernel = cfg.prev_kernel_size
    dim = prev_conv.shape[-1]
    halo_f32 = jnp.concatenate(
        [jnp.zeros((HALO_ROWS - prev_kernel, dim), jnp.float32), prev_conv],
        axis=0)
    halo = pltpu.roll(halo_f32, delta, 0).astype(block_head_ref.dtype)

    # Note (david): the carried tokens belong at rows [delta - prev_kernel_size,
    # delta) of concat(halo, block), so the ones the roll wraps past the halo
    # tile overwrite the block's pad rows [0, delta) in place.
    @pl.when(delta > 0)
    def _patch_pad_rows():
        row = jax.lax.broadcasted_iota(jnp.int32, (HALO_ROWS, dim), 0)
        block_head_ref[...] = jnp.where(row < delta, halo, block_head_ref[...])

    return halo


def conv_state_on_grid(qkv_view_ref: jax.Ref, halo: jax.Array,
                       end: jax.Array, cfg: config.KDAConfig) -> jax.Array:
    """Conv state after the chunk's last real token.

    Returns [prev_kernel_size, dim] f32. end is the block-grid row after the
    chunk's last real token. Pad rows before the first real token already hold
    the previous tokens (see make_halo), so the state is rows
    [end - prev_kernel_size, end) of concat(halo, block) wherever the real
    tokens start.
    """
    prev_kernel = cfg.prev_kernel_size
    chunk = cfg.compute_chunk_size
    chunk_tokens = qkv_view_ref[0, :chunk, :].astype(jnp.float32)
    prev_tokens = halo[HALO_ROWS - prev_kernel:].astype(jnp.float32)
    # Note (david): a partially filled chunk checkpoints after its last real
    # token, a full or empty one after the chunk.
    ckpt_row = jnp.where(jnp.logical_and(end > 0, end < chunk), end, chunk)
    # Note (david): for ckpt_row >= prev_kernel_size the state lies inside the
    # chunk and one roll brings it to the front, where the masking loop of
    # causal_conv1d would cost O(chunk) selects that each occupy a full sublane
    # tile in the native layout. pltpu.roll only accepts non-negative shifts.
    window_end = jnp.maximum(ckpt_row, prev_kernel)
    conv_state = pltpu.roll(chunk_tokens,
                            (chunk - (window_end - prev_kernel)) % chunk,
                            0)[:prev_kernel]
    # Note (david): ckpt_row < prev_kernel_size reaches back into the halo. Only
    # prev_kernel_size - 1 such cases exist and they are all static, so they
    # are plain selects.
    for short_end in range(1, prev_kernel):
        conv_state = jnp.where(
            short_end == end,
            token_rows(prev_tokens[jnp.newaxis], chunk_tokens[jnp.newaxis],
                       short_end, prev_kernel)[0],
            conv_state,
        )
    return conv_state


def causal_conv1d(
    real_sizes: jax.Array,
    chunk_tokens: jax.Array,
    prev_conv: jax.Array,
    conv_weight: jax.Array,
    conv_bias: jax.Array | None,
    cfg: config.KDAConfig,
) -> tuple[jax.Array, jax.Array]:
    """Causal conv1d of one compute chunk and its conv state checkpoints.

    real_sizes is [seq], chunk_tokens the chunk's raw tokens [seq, chunk,
    dim_size], prev_conv the [seq, prev_kernel_size, dim_size] tokens before
    them, conv_weight [kernel_size, dim_size] and conv_bias [dim_size]. Returns
    the [seq, compute_chunk_size, dim_size] output and the [seq, window_size,
    prev_kernel_size, dim_size] states: checkpoint w holds the last
    prev_kernel_size inputs ending at token compute_chunk_size - window_size +
    w, clamped to the last real token, i.e. the state a sequence resuming right
    after that token starts from.
    """
    assert cfg.mode == config.KDAMode.BATCHED
    assert chunk_tokens.ndim == 3 and prev_conv.ndim == 3
    assert prev_conv.shape[1] == cfg.prev_kernel_size

    conv_out = None
    for tap in range(cfg.kernel_size):
        # Note (david): out[c] = sum_k w[k] * lhs[c + k] with
        # lhs = concat(prev_conv, chunk_tokens), exactly upstream's indexing.
        tap_term = conv_weight[tap:tap + 1] * token_rows(
            prev_conv, chunk_tokens, tap, cfg.compute_chunk_size)
        conv_out = tap_term if conv_out is None else conv_out + tap_term
    if conv_bias is None:
        biased_out = conv_out
    else:
        biased_out = conv_out + conv_bias.reshape(1, 1, -1)

    # Note (david): decode and speculative windows keep chunk_size tiny
    # (== window_size), so a masking loop of a few selects beats the roll.
    real_sizes_col = real_sizes.reshape(-1, 1, 1)
    conv_ckpts = []
    for w_idx in range(cfg.window_size):
        last_row = 1 + cfg.compute_chunk_size - cfg.window_size + w_idx
        # Note (david): this is the checkpoint of sequences whose last real
        # token is at or past this window position; the selects below pick the
        # shorter ones.
        conv_ckpt = token_rows(prev_conv, chunk_tokens, last_row,
                               cfg.prev_kernel_size)
        for c_idx in range(1, last_row):
            conv_ckpt = jnp.where(
                c_idx == real_sizes_col,
                token_rows(prev_conv, chunk_tokens, c_idx,
                           cfg.prev_kernel_size),
                conv_ckpt,
            )
        conv_ckpts.append(conv_ckpt)

    return biased_out, jnp.stack(conv_ckpts, axis=1)
