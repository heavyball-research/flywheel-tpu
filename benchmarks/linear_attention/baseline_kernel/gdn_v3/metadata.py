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
from jax.experimental import pallas as pl

from gdn_v3 import config, memory_ref


def compute_batched_seq_metadata(
    cfg: config.GDNConfig,
    seq_lens: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    read_offsets: jax.Array,
    end_seq: jax.Array,
    read_indices: jax.Array,
) -> memory_ref.MetadataRef:
    """Metadata for tiles holding several sequences, one row per sequence.

    A row holds all of a sequence's query tokens: a decode token or a
    speculative verify window of up to cfg.window_size tokens. The initial
    state is read from read_indices[s] + read_offsets[s] and window position t
    checkpoints to state_indices[s] + t.
    """
    all_seqs = jnp.arange(seq_lens.size)

    # Note (david): callers must guarantee query_lens[i] <= cfg.window_size for
    # i < end_seq; it is not checked here.
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    is_valid_seq = all_seqs < end_seq
    valid_seq_ids = jnp.where(is_valid_seq, all_seqs, 0)

    return memory_ref.MetadataRef.create(
        cfg=cfg,
        num_tiles=pl.cdiv(end_seq, cfg.tile_size),
        p_id_to_s_idx=valid_seq_ids,
        p_id_to_r_base=query_start_loc[valid_seq_ids],
        p_id_to_r_size=jnp.where(is_valid_seq, query_lens, 0),
        p_id_is_first_tile=is_valid_seq,
        p_id_is_last_tile=is_valid_seq,
        s_idx_has_initial_state=(seq_lens - query_lens) > 0,
        s_idx_to_state_indices=state_indices,
        s_idx_to_read_offset=read_offsets,
        s_idx_to_read_indices=read_indices,
    )


def compute_per_seq_metadata(
    cfg: config.GDNConfig,
    seq_lens: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    start_seq: jax.Array,
    end_seq: jax.Array,
    read_indices: jax.Array,
) -> memory_ref.MetadataRef:
    """Metadata for tiles holding one sequence each."""
    max_seqs = seq_lens.size
    # Note (david): the per-tile records live in SMEM (1 MiB on v6e), which one
    # record per token overflows from 128K tokens on, so they are sized by the
    # tile count plus one extra tile per sequence for the round up.
    max_tiles = pl.cdiv(cfg.batch_size, cfg.chunk_size) + max_seqs
    all_seqs = jnp.arange(max_seqs)
    all_p_ids = jnp.arange(max_tiles)

    query_start_loc = jnp.roll(query_start_loc, shift=-start_seq)
    seq_lens = jnp.roll(seq_lens, shift=-start_seq)
    state_indices = jnp.roll(state_indices, shift=-start_seq)
    read_indices = jnp.roll(read_indices, shift=-start_seq)

    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    # Note (david): only query_lens needs masking. It sets num_tiles, and the
    # other per-sequence arrays are never visited past it.
    num_seqs = end_seq - start_seq
    query_lens = jnp.where(all_seqs < num_seqs, query_lens, 0)

    s_idx_to_num_tiles = pl.cdiv(query_lens, cfg.chunk_size)
    s_idx_to_start_p_id = jnp.cumulative_sum(s_idx_to_num_tiles,
                                             include_initial=True)
    # Note (david): total_repeat_length makes jnp.repeat jit compilable. It pads
    # p_id_to_s_idx past num_tiles, which the kernel never reads.
    p_id_to_s_idx = jnp.repeat(all_seqs,
                               s_idx_to_num_tiles,
                               total_repeat_length=max_tiles)
    p_id_to_t_id = all_p_ids - s_idx_to_start_p_id[p_id_to_s_idx]
    p_id_to_r_base = (query_start_loc[p_id_to_s_idx] +
                      p_id_to_t_id * cfg.chunk_size)
    p_id_to_r_size = jnp.minimum(
        query_start_loc[p_id_to_s_idx + 1] - p_id_to_r_base,
        cfg.tile_size,
    )

    has_initial_state = (seq_lens - query_lens) > 0
    p_id_is_first_tile = p_id_to_t_id == 0
    p_id_is_last_tile = p_id_to_t_id == (s_idx_to_num_tiles[p_id_to_s_idx] - 1)

    # Note (david): query_lens is 0 past num_seqs, so the sum counts the tiles
    # of valid sequences only.
    num_tiles = s_idx_to_num_tiles.sum()

    return memory_ref.MetadataRef.create(
        cfg=cfg,
        num_tiles=num_tiles,
        p_id_to_s_idx=p_id_to_s_idx,
        p_id_to_r_base=p_id_to_r_base,
        p_id_to_r_size=p_id_to_r_size,
        p_id_is_first_tile=p_id_is_first_tile,
        p_id_is_last_tile=p_id_is_last_tile,
        s_idx_has_initial_state=has_initial_state,
        s_idx_to_state_indices=state_indices,
        # Note (david): prefill and mixed sequences always resume from the
        # base slot of their group.
        s_idx_to_read_offset=jnp.zeros_like(state_indices),
        s_idx_to_read_indices=read_indices,
    )
