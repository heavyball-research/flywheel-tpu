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
from jax.experimental.pallas import tpu as pltpu

from gdn_v3 import config, memory_ref

WORD_BYTES = 4


def load_as_qkv_large(
        qkv_vmem_ref: jax.Ref,
        cfg: config.GDNConfig) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Splits one sequence's packed qkv into the large layout.

    qkv_vmem_ref is [chunk_size, 1, num_kq_heads * kq_head_dim * 2 +
    num_v_heads * v_head_dim] in the compact layout; each lane tile is fetched
    with one strided load. Returns q, k [num_kq_heads, chunk_size,
    kq_head_dim] and v [num_v_heads, chunk_size, v_head_dim].
    """
    num_lanes = pltpu.get_tpu_info().num_lanes
    lane_tiles_per_token = qkv_vmem_ref.shape[-1] // num_lanes
    lane_tiles_per_kq_head = cfg.kq_head_dim // num_lanes
    k_tile_offset = cfg.num_kq_heads * lane_tiles_per_kq_head

    q_heads = []
    k_heads = []
    v_heads = []

    qkv_rows_ref = qkv_vmem_ref.reshape(-1, num_lanes)
    for kq_head in range(cfg.num_kq_heads):
        q_head_tiles = []
        k_head_tiles = []
        for tile in range(lane_tiles_per_kq_head):
            q_tile = kq_head * lane_tiles_per_kq_head + tile
            k_tile = k_tile_offset + q_tile

            q_head_tiles.append(qkv_rows_ref[q_tile::lane_tiles_per_token])
            k_head_tiles.append(qkv_rows_ref[k_tile::lane_tiles_per_token])
        q_heads.append(jnp.concat(q_head_tiles, axis=-1))
        k_heads.append(jnp.concat(k_head_tiles, axis=-1))
    v_tile_offset = lane_tiles_per_kq_head * cfg.num_kq_heads * 2
    lane_tiles_per_v_head = cfg.v_head_dim // num_lanes
    for v_head in range(cfg.num_v_heads):
        v_head_tiles = []
        for tile in range(lane_tiles_per_v_head):
            v_tile = v_tile_offset + v_head * lane_tiles_per_v_head + tile
            v_head_tiles.append(qkv_rows_ref[v_tile::lane_tiles_per_token])
        v_heads.append(jnp.concat(v_head_tiles, axis=-1))

    q_large = jnp.stack(q_heads, axis=0)
    k_large = jnp.stack(k_heads, axis=0)
    v_large = jnp.stack(v_heads, axis=0)

    return q_large, k_large, v_large


def load_as_qkv_compact(
        qkv_vmem_ref: jax.Ref,
        cfg: config.GDNConfig) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Splits a tile's packed qkv into the compact layout, one load per head.

    qkv_vmem_ref is [seq_tile_size, chunk_size, 1, num_kq_heads * kq_head_dim
    * 2 + num_v_heads * v_head_dim]. Returns q, k [seq_tile_size,
    num_kq_heads, chunk_size, 1, kq_head_dim] and v [seq_tile_size,
    num_v_heads, chunk_size, 1, v_head_dim].
    """
    k_offset = cfg.num_kq_heads * cfg.kq_head_dim
    v_offset = cfg.num_kq_heads * 2 * cfg.kq_head_dim

    q_heads = []
    k_heads = []
    v_heads = []

    for kq_head in range(cfg.num_kq_heads):
        q_start = kq_head * cfg.kq_head_dim
        q_end = q_start + cfg.kq_head_dim
        k_start = k_offset + q_start
        k_end = k_start + cfg.kq_head_dim
        q_heads.append(qkv_vmem_ref[..., q_start:q_end])
        k_heads.append(qkv_vmem_ref[..., k_start:k_end])
    for v_head in range(cfg.num_v_heads):
        v_start = v_offset + v_head * cfg.v_head_dim
        v_end = v_start + cfg.v_head_dim
        v_heads.append(qkv_vmem_ref[..., v_start:v_end])

    q_compact = jnp.stack(q_heads, axis=1)
    k_compact = jnp.stack(k_heads, axis=1)
    v_compact = jnp.stack(v_heads, axis=1)

    return q_compact, k_compact, v_compact


def load_compact_to_large(vmem_ref: jax.Ref) -> jax.Array:
    """Loads a 32-bit [..., 1, cols] compact ref as [..., cols] large layout.

    Uses strided loads instead of a transpose.
    """
    assert vmem_ref.dtype.itemsize == WORD_BYTES
    assert vmem_ref.shape[-2] == 1
    num_cols = vmem_ref.shape[-1]
    large_shape = vmem_ref.shape[:-2] + (num_cols, )
    num_lanes = pltpu.get_tpu_info().num_lanes

    lane_tiles = []
    rows_ref = vmem_ref.reshape(-1, num_cols)
    for col_start in range(0, num_cols, num_lanes):
        col_end = min(col_start + num_lanes, num_cols)
        lane_tiles.append(rows_ref[..., col_start:col_end])
    return jnp.concat(lane_tiles, axis=-1).reshape(large_shape)


def load_and_select_states(
    metadata_ref: memory_ref.MetadataRef,
    p_id: jax.Array,
    conv_state_slot_ref: jax.Ref,
    recurrent_slot_ref: jax.Ref,
    carry_conv_scratch_ref: jax.Ref | None,
    carry_recurrent_scratch_ref: jax.Ref | None,
    cfg: config.GDNConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Selects the initial conv and recurrent state of each tile row.

    A sequence's first tile takes the state read from HBM, zeroed when the
    sequence has no initial state; a later tile takes the carry of the
    previous tile. The carry refs are None when tiles never split a sequence.

    Returns real_sizes [seq_tile_size], the conv state [seq_tile_size,
    prev_kernel_size, 1, dim_size] in fp32 and the recurrent state
    [seq_tile_size, num_v_heads, kq_head_dim, v_head_dim].
    """
    real_sizes_list = []
    prev_conv_state_list = []
    prev_recurrent_state_list = []

    for idx in range(cfg.seq_tile_size):
        record = metadata_ref.get_record(p_id, idx)
        s_idx = record.s_idx
        real_size = record.r_size
        is_first_tile = record.is_first_tile
        has_initial_state = metadata_ref.s_idx_has_initial_state[s_idx]

        # Note (david): the VMEM window holds one state per window position and
        # the initial state was DMA'd into position 0. Conv1D needs fp32
        # because it runs in the compact layout.
        hbm_conv_state = conv_state_slot_ref[idx, 0].astype(jnp.float32)
        initial_conv_state = jnp.where(has_initial_state, hbm_conv_state, 0)
        if carry_conv_scratch_ref is None:
            prev_conv_state = initial_conv_state
        else:
            prev_conv_state = jnp.where(is_first_tile, initial_conv_state,
                                        carry_conv_scratch_ref[idx])

        hbm_recurrent_state = recurrent_slot_ref[idx, 0]
        initial_recurrent_state = jnp.where(has_initial_state,
                                            hbm_recurrent_state, 0)
        if carry_recurrent_scratch_ref is None:
            prev_recurrent_state = initial_recurrent_state
        else:
            prev_recurrent_state = jnp.where(is_first_tile,
                                             initial_recurrent_state,
                                             carry_recurrent_scratch_ref[idx])

        real_sizes_list.append(real_size)
        prev_conv_state_list.append(prev_conv_state)
        prev_recurrent_state_list.append(prev_recurrent_state)

    real_sizes = jnp.stack(real_sizes_list, axis=0)
    prev_conv_state = jnp.stack(prev_conv_state_list, axis=0)
    prev_recurrent_state = jnp.stack(prev_recurrent_state_list, axis=0)

    return real_sizes, prev_conv_state, prev_recurrent_state


def load_activation_as_large(
    qkv_vmem_ref: jax.Ref,
    b_vmem_ref: jax.Ref,
    a_vmem_ref: jax.Ref,
    cfg: config.GDNConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Loads a tile's activations from VMEM in the large layout.

    Returns q, k [seq_tile_size, num_kq_heads, chunk_size, kq_head_dim], v
    [seq_tile_size, num_v_heads, chunk_size, v_head_dim] and b, a
    [seq_tile_size, 1, chunk_size, aligned_num_v_heads].
    """
    q_large_list = []
    k_large_list = []
    v_large_list = []
    for idx in range(cfg.seq_tile_size):
        q_large, k_large, v_large = load_as_qkv_large(qkv_vmem_ref.at[idx],
                                                      cfg)
        q_large_list.append(q_large)
        k_large_list.append(k_large)
        v_large_list.append(v_large)

    q_large = jnp.stack(q_large_list, axis=0)
    k_large = jnp.stack(k_large_list, axis=0)
    v_large = jnp.stack(v_large_list, axis=0)
    b_large = load_compact_to_large(b_vmem_ref)
    a_large = load_compact_to_large(a_vmem_ref)
    b_large = jnp.expand_dims(b_large, axis=1)
    a_large = jnp.expand_dims(a_large, axis=1)

    return q_large, k_large, v_large, b_large, a_large
