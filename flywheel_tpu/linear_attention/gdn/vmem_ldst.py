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
"""VMEM loads that split a tile's native-layout activations per head."""

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from . import config, memory_ref


def split_qkv_heads(
    qkv: jax.Array,
    cfg: config.GDNConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Split qkv into per-head q, k [seq, num_kq_heads, *rest, kq_head_dim] and
    v [seq, num_v_heads, *rest, v_head_dim].

    qkv is [..., dim] with q, k and v concatenated along the last dimension. In
    the native layout a head is a lane-aligned slice of it, so this is plain
    vreg selection with no VMEM round trip or strided load.
    """
    k_offset = cfg.num_kq_heads * cfg.kq_head_dim
    v_offset = 2 * k_offset

    q_heads = []
    k_heads = []
    v_heads = []
    for kq_head in range(cfg.num_kq_heads):
        q_start = kq_head * cfg.kq_head_dim
        k_start = k_offset + q_start
        q_heads.append(qkv[..., q_start:q_start + cfg.kq_head_dim])
        k_heads.append(qkv[..., k_start:k_start + cfg.kq_head_dim])
    for v_head in range(cfg.num_v_heads):
        v_start = v_offset + v_head * cfg.v_head_dim
        v_heads.append(qkv[..., v_start:v_start + cfg.v_head_dim])

    return (jnp.stack(q_heads, axis=1), jnp.stack(k_heads, axis=1),
            jnp.stack(v_heads, axis=1))


def load_qkv_realigned(
    metadata_ref: memory_ref.MetadataRef,
    p_id: jax.Array,
    qkv_slot_ref: jax.Ref,
    cfg: config.GDNConfig,
) -> jax.Array:
    """Read one compute chunk's raw qkv as [seq, compute_chunk, dim] f32."""
    assert cfg.mode == config.GDNMode.BATCHED
    block_rows = qkv_slot_ref.shape[1]
    seq_chunks = []
    for idx in range(cfg.seq_tile_size):
        r_base = metadata_ref.get_record(p_id, idx).r_base
        delta = r_base & (config.SUBLANE_ALIGN - 1)
        block = qkv_slot_ref[idx].astype(jnp.float32)
        # Note (david): the aligned DMA landed the tokens delta rows down, and
        # pltpu.roll only accepts non-negative shifts, so rotate them up by
        # block_rows - delta.
        seq_chunks.append(
            pltpu.roll(block, block_rows - delta, 0)[:cfg.compute_chunk_size])
    return jnp.stack(seq_chunks, axis=0)


def load_squeeze_unit_axis(vmem_ref: jax.Ref) -> jax.Array:
    """Load a 32-bit [..., 1, heads] ref as [..., heads] with a strided load
    instead of a transpose."""
    assert vmem_ref.dtype.itemsize == 4
    assert vmem_ref.shape[-2] == 1
    num_cols = vmem_ref.shape[-1]
    squeezed_shape = vmem_ref.shape[:-2] + (num_cols,)
    num_lanes = pltpu.get_tpu_info().num_lanes

    flat_ref = vmem_ref.reshape(-1, num_cols)
    lane_slabs = []
    for col_start in range(0, num_cols, num_lanes):
        col_end = min(col_start + num_lanes, num_cols)
        lane_slabs.append(flat_ref[..., col_start:col_end])
    return jnp.concatenate(lane_slabs, axis=-1).reshape(squeezed_shape)


def load_and_select_states(
    metadata_ref: memory_ref.MetadataRef,
    p_id: jax.Array,
    conv_state_slot_ref: jax.Ref,
    recurrent_slot_ref: jax.Ref,
    carry_conv_scratch_ref: jax.Ref | None,
    carry_recurrent_scratch_ref: jax.Ref | None,
    cfg: config.GDNConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Pick each sequence's starting conv and recurrent state.

    A sequence's first tile starts from the state DMA'd from HBM, or zeros
    without an initial state; later tiles start from the previous tile's
    carry. conv_state_slot_ref is [seq_tile_size, 1, prev_kernel_size,
    dim_size] and recurrent_slot_ref [seq_tile_size, 1, num_v_heads,
    kq_head_dim, v_head_dim]; the carries drop the unit axis and are None in
    BATCHED mode. Returns real_sizes [seq_tile_size], the f32 conv state
    [seq_tile_size, prev_kernel_size, dim_size] and the recurrent state
    [seq_tile_size, num_v_heads, kq_head_dim, v_head_dim].
    """
    real_size_list = []
    conv_state_list = []
    recurrent_state_list = []

    for idx in range(cfg.seq_tile_size):
        record = metadata_ref.get_record(p_id, idx)
        r_size = record.r_size
        is_first_tile = record.is_first_tile
        has_initial_state = metadata_ref.s_idx_has_initial_state[record.s_idx]

        # Note (david): the initial state was DMA'd into window position 0. The
        # conv state is f32 because the halo it supplies is concatenated with
        # the f32-cast qkv block.
        hbm_conv_state = conv_state_slot_ref[idx, 0].astype(jnp.float32)
        initial_conv_state = jnp.where(has_initial_state, hbm_conv_state, 0)
        if carry_conv_scratch_ref is None:
            conv_state = initial_conv_state
        else:
            conv_state = jnp.where(is_first_tile, initial_conv_state,
                                   carry_conv_scratch_ref[idx])

        initial_recurrent_state = jnp.where(has_initial_state,
                                            recurrent_slot_ref[idx, 0], 0)
        if carry_recurrent_scratch_ref is None:
            recurrent_state = initial_recurrent_state
        else:
            recurrent_state = jnp.where(is_first_tile, initial_recurrent_state,
                                        carry_recurrent_scratch_ref[idx])

        real_size_list.append(r_size)
        conv_state_list.append(conv_state)
        recurrent_state_list.append(recurrent_state)

    return (jnp.stack(real_size_list, axis=0),
            jnp.stack(conv_state_list, axis=0),
            jnp.stack(recurrent_state_list, axis=0))


def load_activation_as_token(
    qkv_vreg: jax.Array,
    b_vmem_ref: jax.Ref,
    a_vmem_ref: jax.Ref,
    cfg: config.GDNConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Split native [seq, chunk, dim] activations into the recurrent path's
    [seq, heads, chunk, 1, head_dim] layout, one token per row."""
    # Note (david): only chunk_size 1 and speculative windows take this path. At
    # chunk_size 1 the extra unit axis is a no-op relayout (a one-row tile is
    # one-row tiled either way), so only multi-token verify windows pay for a
    # real relayout.
    qkv_token = jnp.expand_dims(qkv_vreg, axis=2)
    q_token, k_token, v_token = split_qkv_heads(qkv_token, cfg)
    b_token = jnp.expand_dims(b_vmem_ref[...], axis=1)
    a_token = jnp.expand_dims(a_vmem_ref[...], axis=1)
    return q_token, k_token, v_token, b_token, a_token


def load_activation_as_chunked(
    qkv_vreg: jax.Array,
    b_vmem_ref: jax.Ref,
    a_vmem_ref: jax.Ref,
    cfg: config.GDNConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Split native [seq, chunk, dim] activations into the chunked prefill
    path's per-head layout; b and a are widened from their compact layout."""
    q_chunked, k_chunked, v_chunked = split_qkv_heads(qkv_vreg, cfg)
    b_chunked = jnp.expand_dims(load_squeeze_unit_axis(b_vmem_ref), axis=1)
    a_chunked = jnp.expand_dims(load_squeeze_unit_axis(a_vmem_ref), axis=1)
    return q_chunked, k_chunked, v_chunked, b_chunked, a_chunked
