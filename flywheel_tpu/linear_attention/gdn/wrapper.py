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

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from . import (
    compute_conv1d,
    compute_gdn,
    config,
    memory_ref,
    metadata,
    vmem_ldst,
)

F32_BYTES = 4
BF16_BYTES = 2
# Note (david): a verify window keeps one state checkpoint per position in
# VMEM, so large-head models (e.g. Qwen3.5-397B: 64 local v-heads, 21M per
# buffered window even at tile_size 1) do not fit the default budget; windowed
# kernels get a higher limit and the decode tile is sized against it.
DEFAULT_VMEM_FRACTION = 0.7
WINDOWED_VMEM_FRACTION = 0.9
# Note (david): pallas_call operands are the flattened metadata leaves, then
# qkv, b, a, conv_state, recurrent_state and the aliased output.
CONV_STATE_OPERAND = 3
RECURRENT_STATE_OPERAND = 4
ALIASED_OUT_OPERAND = 5


def chunk_qkv_view(qkv_slot_ref: jax.Array, chunk_idx: int | jax.Array,
                   cfg: config.GDNConfig) -> jax.Array:
    """[seq, compute_chunk_size, dim] view of chunk chunk_idx of a block-grid
    input; a static index is bounds-checked at trace time."""
    compute_chunk = cfg.compute_chunk_size
    if isinstance(chunk_idx, int):
        chunk_end = (chunk_idx + 1) * compute_chunk
        assert chunk_idx >= 0 and chunk_end <= cfg.qkv_block_rows, (
            chunk_idx, compute_chunk)
        chunk_start = chunk_idx * compute_chunk
    else:
        chunk_start = pl.multiple_of(chunk_idx * compute_chunk, compute_chunk)
    return qkv_slot_ref.at[:, pl.ds(chunk_start, compute_chunk)]


def store_out_rows(out_slot_ref: jax.Array, out: jax.Array,
                   chunk_idx: int | jax.Array, delta: jax.Array,
                   metadata_ref: memory_ref.MetadataRef, p_id: jax.Array,
                   cfg: config.GDNConfig) -> None:
    """Write one compute chunk's output into the tile's output stage.

    out is [seq, chunk, v_dim_size]. PER_SEQ writes chunk chunk_idx at its
    block-grid rows; BATCHED packs each sequence's real rows back to back from
    stage row delta.
    """
    out_dtype = out_slot_ref.dtype
    if cfg.mode == config.GDNMode.PER_SEQ:
        compute_chunk = cfg.compute_chunk_size
        if isinstance(chunk_idx, int):
            row_offset = chunk_idx * compute_chunk
        else:
            row_offset = pl.multiple_of(chunk_idx * compute_chunk,
                                        compute_chunk)
        out_slot_ref[pl.ds(row_offset, compute_chunk)] = out[0].astype(
            out_dtype)
    else:
        stage = out_slot_ref[...]
        stage_rows = lax.broadcasted_iota(jnp.int32, stage.shape, 0)
        seq_row_start = delta
        for seq_idx in range(cfg.seq_tile_size):
            num_seq_rows = metadata_ref.get_record(p_id, seq_idx).r_size
            for window_pos in range(cfg.window_size):
                token_out = jnp.broadcast_to(
                    out[seq_idx, window_pos:window_pos + 1].astype(out_dtype),
                    stage.shape)
                is_token_row = ((stage_rows == seq_row_start + window_pos) &
                                (window_pos < num_seq_rows))
                stage = jnp.where(is_token_row, token_out, stage)
            seq_row_start += num_seq_rows
        out_slot_ref[...] = stage


def inner_kernel(
    p_id: jax.Array,
    qkv_slot_ref: jax.Array,
    b_slot_ref: jax.Array,
    a_slot_ref: jax.Array,
    conv_in_ref: jax.Array,
    recurrent_in_ref: jax.Array,
    out_slot_ref: jax.Array,
    conv_out_ref: jax.Array,
    recurrent_out_ref: jax.Array,
    metadata_ref: memory_ref.MetadataRef,
    weights_ref: memory_ref.WeightRefs,
    carry_conv_scratch_ref: jax.Array | None,
    carry_recurrent_scratch_ref: jax.Array | None,
    *,
    cfg: config.GDNConfig,
) -> None:
    """Compute one tile from its VMEM ring slots and stage its outputs."""
    profile_scope = compute_gdn.profile_scope
    with profile_scope("load_states"):
        real_sizes, prev_conv, prev_recurrent = vmem_ldst.load_and_select_states(
            metadata_ref=metadata_ref,
            p_id=p_id,
            conv_state_slot_ref=conv_in_ref,
            recurrent_slot_ref=recurrent_in_ref,
            carry_conv_scratch_ref=carry_conv_scratch_ref,
            carry_recurrent_scratch_ref=carry_recurrent_scratch_ref,
            cfg=cfg,
        )

    conv_weight = weights_ref.conv.weight[...].astype(jnp.float32)
    if weights_ref.conv.bias is None:
        conv_bias = None
    else:
        conv_bias = weights_ref.conv.bias[...].astype(jnp.float32)
    num_v_padding = cfg.aligned_num_v_heads - cfg.num_v_heads
    a_log = jnp.pad(weights_ref.gdn.a_log[...], (0, num_v_padding))
    dt_bias = jnp.pad(weights_ref.gdn.dt_bias[...], (0, num_v_padding))
    r_base = metadata_ref.get_record(p_id, 0).r_base
    delta = r_base & (config.SUBLANE_ALIGN - 1)

    def _store_states(new_conv_state, new_recurrent_state):
        with profile_scope("state_store"):
            conv_out_ref[...] = new_conv_state
            recurrent_out_ref[...] = new_recurrent_state.astype(
                recurrent_out_ref.dtype)
            if carry_conv_scratch_ref is not None:
                carry_conv_scratch_ref[...] = new_conv_state[:, -1]
            if carry_recurrent_scratch_ref is not None:
                carry_recurrent_scratch_ref[...] = new_recurrent_state[:, -1]

    if cfg.mode == config.GDNMode.BATCHED:
        with profile_scope("load_qkv"):
            qkv_in = vmem_ldst.load_qkv_realigned(
                metadata_ref=metadata_ref, p_id=p_id,
                qkv_slot_ref=qkv_slot_ref, cfg=cfg)
        with profile_scope("conv1d"):
            qkv_out, conv_ckpt = compute_conv1d.causal_conv1d(
                real_sizes=real_sizes, chunk_tokens=qkv_in,
                prev_conv=prev_conv, conv_weight=conv_weight,
                conv_bias=conv_bias, cfg=cfg)
        with profile_scope("silu"):
            qkv_out = compute_gdn.silu(qkv_out)
        q, k, v, b, a = vmem_ldst.load_activation_as_token(
            qkv_vreg=qkv_out, b_vmem_ref=b_slot_ref,
            a_vmem_ref=a_slot_ref, cfg=cfg)
        out, recurrent_ckpt = compute_gdn.recurrent_gdn(
            real_sizes=real_sizes,
            q_token=q, k_token=k, v_token=v,
            b_token=b, a_token=a, state_prev=prev_recurrent,
            a_log=a_log, dt_bias=dt_bias, cfg=cfg)
        with profile_scope("store_out"):
            store_out_rows(out_slot_ref, out, 0, delta, metadata_ref, p_id,
                           cfg)
        _store_states(conv_ckpt, recurrent_ckpt)
        return

    assert cfg.mode == config.GDNMode.PER_SEQ
    halo_rows = compute_conv1d.HALO_ROWS
    compute_chunk = cfg.compute_chunk_size
    used_rows = delta + real_sizes[0]

    @pl.when(used_rows % halo_rows != 0)
    def _zero_tail():
        tail_start = pl.multiple_of(used_rows - used_rows % halo_rows,
                                    halo_rows)
        tail_ref = qkv_slot_ref.at[0, pl.ds(tail_start, halo_rows)]
        tail_rows = tail_start + lax.broadcasted_iota(
            jnp.int32, (halo_rows, cfg.dim_size), 0)
        tail_ref[...] = jnp.where(tail_rows < used_rows, tail_ref[...], 0)

    with profile_scope("halo"):
        initial_halo = compute_conv1d.make_halo(
            prev_conv[0], delta, qkv_slot_ref.at[0, pl.ds(0, halo_rows)], cfg)

    def _chunk_front(chunk_idx, halo):
        qkv_view = chunk_qkv_view(qkv_slot_ref, chunk_idx, cfg)
        row_lo = jnp.where(chunk_idx == 0, delta, 0)
        row_hi = jnp.clip(used_rows - chunk_idx * compute_chunk, 0,
                          compute_chunk)
        with profile_scope("conv1d"):
            qkv_out = compute_conv1d.causal_conv1d_static(
                qkv_view_ref=qkv_view, halo=halo,
                conv_weight=conv_weight, conv_bias=conv_bias, cfg=cfg)
        next_halo = qkv_view[0, compute_chunk - halo_rows:compute_chunk, :]
        with profile_scope("silu"):
            qkv_out = compute_gdn.silu(qkv_out)
        return row_lo, row_hi, qkv_out, next_halo

    def _load_act(chunk_idx, qkv_out):
        if cfg.num_compute_chunks == 1:
            b_chunk_ref = b_slot_ref
            a_chunk_ref = a_slot_ref
        else:
            # Note (david): the compact b / a token axis is untiled, so a traced
            # chunk offset needs no alignment hint.
            chunk_rows = pl.ds(chunk_idx * compute_chunk, compute_chunk)
            b_chunk_ref = b_slot_ref.at[:, chunk_rows]
            a_chunk_ref = a_slot_ref.at[:, chunk_rows]
        return vmem_ldst.load_activation_as_chunked(
            qkv_vreg=qkv_out, b_vmem_ref=b_chunk_ref, a_vmem_ref=a_chunk_ref,
            cfg=cfg)

    def _run_chunk(chunk_idx, halo, recurrent_state):
        row_lo, row_hi, qkv_out, next_halo = _chunk_front(chunk_idx, halo)
        with profile_scope("load_act"):
            q, k, v, b, a = _load_act(chunk_idx, qkv_out)
        out, recurrent_ckpt = compute_gdn.chunked_gdn(
            row_lo=row_lo, row_hi=row_hi,
            q_chunked=q, k_chunked=k, v_chunked=v, b_chunked=b, a_chunked=a,
            state_prev=recurrent_state, a_log=a_log, dt_bias=dt_bias, cfg=cfg)
        with profile_scope("store_out"):
            store_out_rows(out_slot_ref, out, chunk_idx, delta, metadata_ref,
                           p_id, cfg)
        with profile_scope("conv_state"):
            conv_ckpt = compute_conv1d.conv_state_on_grid(
                chunk_qkv_view(qkv_slot_ref, chunk_idx, cfg), halo, row_hi,
                cfg)[None, None]
        return conv_ckpt, next_halo, recurrent_ckpt

    if cfg.num_compute_chunks == 1:
        conv_ckpt, _, recurrent_ckpt = _run_chunk(0, initial_halo,
                                                  prev_recurrent)
        _store_states(conv_ckpt, recurrent_ckpt)
        return

    num_active_chunks = pl.cdiv(used_rows, compute_chunk)

    def _chunk_step(chunk_idx, carry):
        # Note (david): each chunk rebuilds the conv state from its block rows
        # and halo, so the carried one is only read after the loop.
        halo, _, recurrent_state = carry
        conv_ckpt, next_halo, recurrent_ckpt = _run_chunk(
            chunk_idx, halo, recurrent_state)
        return next_halo, conv_ckpt[:, -1], recurrent_ckpt[:, -1]

    def _full_tile():
        chunks = []
        halo = initial_halo
        for chunk_idx in range(cfg.num_compute_chunks):
            row_lo, row_hi, qkv_out, next_halo = _chunk_front(chunk_idx, halo)
            prev_halo, halo = halo, next_halo
            with profile_scope("load_act"):
                q, k, v, b, a = _load_act(chunk_idx, qkv_out)
            q, k, v, g, beta = compute_gdn.prepare_chunk(
                row_lo, row_hi, q, k, v, b, a, a_log, dt_bias, cfg)
            chunks.append(compute_gdn.ChunkGDN(
                q[0], k[0], v[0], g[0], beta[0], cfg))
        chunk_outs, recurrent_state = compute_gdn.run_pipelined(
            chunks, prev_recurrent[0])
        for chunk_idx, chunk_out in enumerate(chunk_outs):
            with profile_scope("store_out"):
                store_out_rows(out_slot_ref,
                               compute_gdn.pack_out(chunk_out, cfg)[None],
                               chunk_idx, delta, metadata_ref, p_id, cfg)

        def _static_conv_state():
            return halo[halo_rows - cfg.prev_kernel_size:].astype(jnp.float32)

        def _ragged_conv_state():
            return compute_conv1d.conv_state_on_grid(
                chunk_qkv_view(qkv_slot_ref, cfg.num_compute_chunks - 1, cfg),
                prev_halo, row_hi, cfg)

        with profile_scope("conv_state"):
            conv_ckpt = lax.cond(used_rows == cfg.chunk_size,
                                 _static_conv_state, _ragged_conv_state)
        _store_states(conv_ckpt[None, None], recurrent_state[None, None])

    def _tail_tile():
        _, conv_state, recurrent_state = lax.fori_loop(
            0, num_active_chunks, _chunk_step,
            (initial_halo, prev_conv, prev_recurrent))
        _store_states(conv_state[:, None], recurrent_state[:, None])

    # Note (david): a full tile unrolls its chunks so run_pipelined can
    # interleave them on the MXU; a tail tile loops over its active chunks only.
    lax.cond(num_active_chunks == cfg.num_compute_chunks, _full_tile,
             _tail_tile)


def outer_kernel(
    metadata_ref: memory_ref.MetadataRef,
    qkv_ref: jax.Array,
    b_ref: jax.Array,
    a_ref: jax.Array,
    conv_state_ref: jax.Array,
    recurrent_state_ref: jax.Array,
    aliased_out_ref: jax.Array | None,
    weights_ref: memory_ref.WeightRefs,
    pad_start_ref: jax.Array | None,
    out_ref: jax.Array,
    conv_state_out_ref: jax.Array,
    recurrent_state_out_ref: jax.Array,
    # Note (david): pallas_call passes the scratch shapes by keyword, so these
    # names are the keys of cfg.get_scratch_shape_dict.
    qkv_buf: jax.Array,
    b_buf: jax.Array,
    a_buf: jax.Array,
    conv_in_buf: jax.Array,
    rec_in_buf: jax.Array,
    conv_out_buf: jax.Array,
    rec_out_buf: jax.Array,
    out_buf: jax.Array,
    carry_conv_scratch_ref: jax.Array | None,
    carry_recurrent_scratch_ref: jax.Array | None,
    sems: jax.Array,
    *,
    cfg: config.GDNConfig,
) -> None:
    """Double-buffered tile pipeline over inner_kernel.

    Tile t uses ring slot t % 2. The inputs of tile t + 1 are prefetched before
    tile t computes and the outputs of tile t - 1 are waited only after tile
    t's outputs start, so both overlap tile t's compute. State inputs and
    outputs use separate rings, so no slot is read by one DMA while another
    writes it.
    """
    # Note (david): the tile loop indexes its VMEM rings as t % 2 and 1 - slot.
    assert cfg.num_buffers == config.NUM_BUFFERS, cfg.num_buffers
    # Note (david): aliased_out_ref aliases out_ref and the state outputs alias
    # their inputs; the DMAs go through the other ref of each pair.
    del aliased_out_ref, conv_state_out_ref, recurrent_state_out_ref

    align = config.SUBLANE_ALIGN
    slab = config.STAGE_SLAB_ROWS
    num_tiles = metadata_ref.num_tiles[...]
    first_window_base, _, _, _ = memory_ref.out_window(0, metadata_ref, cfg)
    profile_scope = compute_gdn.profile_scope

    def _start_in(p_id, slot):
        with profile_scope("start_in"):
            tile_args = (p_id, metadata_ref, cfg)
            memory_ref.start_qkv_in(qkv_ref, qkv_buf.at[slot],
                                    sems.at[config.STREAM_QKV, slot],
                                    *tile_args)
            memory_ref.start_compact_in(b_ref, b_buf.at[slot],
                                        sems.at[config.STREAM_B, slot],
                                        *tile_args)
            memory_ref.start_compact_in(a_ref, a_buf.at[slot],
                                        sems.at[config.STREAM_A, slot],
                                        *tile_args)
            memory_ref.start_state_in(conv_state_ref, conv_in_buf.at[slot],
                                      sems.at[config.STREAM_CONV_IN, slot],
                                      *tile_args)
            memory_ref.start_state_in(recurrent_state_ref,
                                      rec_in_buf.at[slot],
                                      sems.at[config.STREAM_REC_IN, slot],
                                      *tile_args)

    def _wait_in(p_id, slot):
        with profile_scope("wait_in"):
            tile_args = (p_id, metadata_ref, cfg)
            memory_ref.wait_qkv_in(qkv_ref, qkv_buf.at[slot],
                                   sems.at[config.STREAM_QKV, slot],
                                   *tile_args)
            memory_ref.wait_compact_in(b_buf.at[slot],
                                       sems.at[config.STREAM_B, slot],
                                       *tile_args)
            memory_ref.wait_compact_in(a_buf.at[slot],
                                       sems.at[config.STREAM_A, slot],
                                       *tile_args)
            memory_ref.wait_state_in(conv_in_buf.at[slot],
                                     sems.at[config.STREAM_CONV_IN, slot],
                                     *tile_args)
            memory_ref.wait_state_in(rec_in_buf.at[slot],
                                     sems.at[config.STREAM_REC_IN, slot],
                                     *tile_args)

    def _start_out(p_id, slot):
        with profile_scope("start_out"):
            tile_args = (p_id, metadata_ref, cfg)
            memory_ref.start_out(out_ref, out_buf.at[slot],
                                 sems.at[config.STREAM_OUT, slot], *tile_args)
            memory_ref.start_state_out(conv_state_ref, conv_out_buf.at[slot],
                                       sems.at[config.STREAM_CONV_OUT, slot],
                                       *tile_args)
            memory_ref.start_state_out(recurrent_state_ref,
                                       rec_out_buf.at[slot],
                                       sems.at[config.STREAM_REC_OUT, slot],
                                       *tile_args)

    def _wait_out(p_id, slot):
        with profile_scope("wait_out"):
            tile_args = (p_id, metadata_ref, cfg)
            memory_ref.wait_out(out_ref, out_buf.at[slot],
                                sems.at[config.STREAM_OUT, slot], *tile_args)
            memory_ref.wait_state_out(conv_out_buf.at[slot],
                                      sems.at[config.STREAM_CONV_OUT, slot],
                                      *tile_args)
            memory_ref.wait_state_out(rec_out_buf.at[slot],
                                      sems.at[config.STREAM_REC_OUT, slot],
                                      *tile_args)

    @pl.when(num_tiles > 0)
    def _prologue():
        # Note (david): scratch VMEM starts uninitialized, and the block-grid
        # conv's one-hot dot reads whole chunks of the qkv block, including
        # rows a ragged tile's DMA never writes; 0 * NaN = NaN, so the ring is
        # zeroed once before the first DMA.
        qkv_buf[...] = jnp.zeros(qkv_buf.shape, qkv_buf.dtype)
        # Note (david): tile 0's aligned output window starts below its first
        # row, on rows the BATCHED call already wrote (or on row 0 with delta
        # 0, where nothing is read). Seeding the other slot's head with them
        # lets the ordinary cross-tile blend restore them.
        seed_copy = pltpu.make_async_copy(
            out_ref.at[pl.ds(first_window_base, align)],
            out_buf.at[1, pl.ds(0, align)],
            sems.at[config.STREAM_OUT, 1])
        seed_copy.start()
        seed_copy.wait()
        _start_in(0, 0)

    def _tile(tile_idx, prev_window_base):
        slot = lax.rem(tile_idx, config.NUM_BUFFERS)
        other_slot = 1 - slot

        @pl.when(tile_idx + 1 < num_tiles)
        def _prefetch_next():
            _start_in(tile_idx + 1, other_slot)

        _wait_in(tile_idx, slot)
        inner_kernel(
            tile_idx,
            qkv_buf.at[slot],
            b_buf.at[slot],
            a_buf.at[slot],
            conv_in_buf.at[slot],
            rec_in_buf.at[slot],
            out_buf.at[slot],
            conv_out_buf.at[slot],
            rec_out_buf.at[slot],
            metadata_ref,
            weights_ref,
            carry_conv_scratch_ref,
            carry_recurrent_scratch_ref,
            cfg=cfg,
        )
        window_base, _, delta, used_rows = memory_ref.out_window(
            tile_idx, metadata_ref, cfg)

        with profile_scope("out_fixup"):
            # Note (david): rows past the tile's last real one hold stale stage
            # data (a chunk the tail path skipped, or the rotate's zero pad).
            # Only the slab holding used_rows can still fall inside the DMA
            # window; the next tile's window overwrites it, and the last tile's
            # lands in the padded tail, which the zero fill below expects zero.
            stage_ref = out_buf.at[slot]
            tail_slab_start = pl.multiple_of(
                used_rows - (used_rows & (slab - 1)), slab)
            tail_slab_rows = tail_slab_start + lax.broadcasted_iota(
                jnp.int32, (slab, cfg.v_dim_size), 0)
            stage_ref[pl.ds(tail_slab_start, slab)] = jnp.where(
                tail_slab_rows < used_rows,
                stage_ref[pl.ds(tail_slab_start, slab)], 0)

            # Note (david): the window's head rows belong to the previous tile,
            # whose stage is still resident since only its outbound DMA reads
            # it. Both windows sit on the SUBLANE_ALIGN grid, so the copy is
            # slab to slab and the previous window always covers row
            # window_base + delta - 1.
            @pl.when(delta > 0)
            def _blend_head_rows():
                prev_stage_row = pl.multiple_of(window_base - prev_window_base,
                                                align)
                # The previous stage is read a STAGE_SLAB_ROWS slab at a time,
                # and the head rows are that slab's lower or upper half.
                prev_slab_start = pl.multiple_of(
                    prev_stage_row - (prev_stage_row & (slab - 1)), slab)
                prev_slab = out_buf.at[other_slot][pl.ds(prev_slab_start,
                                                         slab)]
                prev_head = jnp.where(prev_stage_row == prev_slab_start,
                                      prev_slab[:align], prev_slab[align:])
                head_rows = lax.broadcasted_iota(
                    jnp.int32, (align, cfg.v_dim_size), 0)
                stage_ref[pl.ds(0, align)] = jnp.where(
                    head_rows < delta, prev_head, stage_ref[pl.ds(0, align)])

        _start_out(tile_idx, slot)

        # Note (david): tile t + 1 reuses other_slot, so tile t - 1's outputs
        # must leave it first; waiting after this tile's own start keeps two
        # output DMAs in flight.
        @pl.when(tile_idx >= 1)
        def _retire_prev():
            _wait_out(tile_idx - 1, other_slot)

        return window_base

    lax.fori_loop(0, num_tiles, _tile, first_window_base)

    @pl.when(num_tiles > 0)
    def _epilogue():
        last_tile = num_tiles - 1
        _wait_out(last_tile, lax.rem(last_tile, config.NUM_BUFFERS))

    if cfg.mode == config.GDNMode.PER_SEQ:
        # Note (david): no tile owns the output rows past the last real token.
        # Zeroing them here costs one DMA per stage of padded rows, where an
        # aliased zeros_like output would broadcast the whole output (about
        # 50 us at T = 8192) even without padding. The BATCHED call ran first
        # and only wrote rows below pad_start. The last tile's DMA already
        # zeroed up to the next SUBLANE_ALIGN row, so the fill starts on the
        # grid the token-major output requires.
        fill_start = pl.multiple_of(
            pl.cdiv(pad_start_ref[0], align) * align, align)
        rows_per_fill = cfg.out_window_rows - align
        fill_end = pl.cdiv(cfg.batch_size, align) * align

        @pl.when(fill_start < fill_end)
        def _zero_padded_rows():
            zero_stage_ref = out_buf.at[0]
            zero_stage_ref[...] = jnp.zeros(zero_stage_ref.shape,
                                            zero_stage_ref.dtype)

            def _zero_fill_step(fill_idx, carry):
                fill_base = fill_start + fill_idx * rows_per_fill
                fill_rows = pl.multiple_of(
                    jnp.minimum(rows_per_fill, fill_end - fill_base), align)
                zero_copy = pltpu.make_async_copy(
                    zero_stage_ref.at[pl.ds(0, fill_rows)],
                    out_ref.at[pl.ds(fill_base, fill_rows)],
                    sems.at[config.STREAM_OUT, 0])
                zero_copy.start()
                zero_copy.wait()
                return carry

            lax.fori_loop(0, pl.cdiv(fill_end - fill_start, rows_per_fill),
                          _zero_fill_step, None)


@jax.jit(
    donate_argnames=("conv_state", "recurrent_state"),
    static_argnames=(
        "n_kq",
        "n_v",
        "d_k",
        "d_v",
        "kernel_size",
        "num_spec_tokens",
        "decode_tile_size",
        "mixed_tile_size",
        "compute_chunk_size",
        "zero_initialize_out",
        "compute_precision",
    ),
)
def fused_conv1d_gdn(
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
    read_offsets: jax.Array | None = None,
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    num_spec_tokens: int = 0,
    zero_initialize_out: bool = True,
    compute_precision: jnp.dtype = jnp.float32.dtype,
    decode_tile_size: int = 4,
    # Note (david): 128 is the widest tile no slower than 64 at every measured
    # v6e shape; wider tiles win 3-9% on long sequences but lose on short
    # ones, whose tail tile runs its chunks without cross-chunk overlap.
    mixed_tile_size: int = 128,
    compute_chunk_size: int = 64,
) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
    """Conv1D + silu + gated delta rule in one fused kernel.

    Args:
        qkv: bf16 mixed query, key, value input [batch_size, dim_size], where
            dim_size = n_kq * d_k * 2 + n_v * d_v.
        b: Input to beta [batch_size, n_v].
        a: Input to the decay g [batch_size, n_v].
        conv_state: Convolution state cache [num_slots, kernel_size - 1,
            dim_size] holding the last kernel_size - 1 tokens of the previous
            invocation. Slot 0 is the null block for padded or invalid tokens.
            A slot may hold garbage on the first invocation of a sequence.
        recurrent_state: Recurrent state cache [num_slots, n_v, d_k, d_v],
            with the same null block and garbage caveat.
        conv_weight: Convolution weight [dim_size, 1, kernel_size].
        conv_bias: Optional convolution bias [dim_size].
        a_log: Per-head log decay scale [n_v]; the log decay is
            -exp(a_log) * softplus(a + dt_bias).
        dt_bias: Bias of the decay gate [n_v].
        query_start_loc: Start locations of the sequences [num_seqs + 1].
        state_indices: State cache slot of each sequence [num_seqs]. With
            speculative decoding these are the base slots of per-request
            groups of num_spec_tokens + 1 consecutive slots.
        distribution: int32 [3], [decode_end, prefill_end, mixed_end]. With
            num_spec_tokens > 0 the first segment holds speculative verify
            windows of up to num_spec_tokens + 1 tokens instead of 1-token
            decodes.
        seq_lens: Length of each sequence [num_seqs].
        read_state_indices: Slot each sequence reads its initial state from
            [num_seqs]. Equals state_indices unless mamba prefix caching
            (align mode) resumes from the cached state block of the previous
            block boundary.
        read_offsets: Optional int32 per-sequence state read offset
            [num_seqs] (num_accepted - 1 from the last verify step). Windowed
            sequences read their initial state from read_state_indices[s] +
            read_offsets[s] and write one checkpoint per window position to
            state_indices[s] + t. Required when num_spec_tokens > 0.
        n_kq: Number of key/query heads.
        n_v: Number of value heads.
        d_k: Key/query head dimension.
        d_v: Value head dimension.
        kernel_size: Convolution kernel size.
        num_spec_tokens: Number of speculative draft tokens; 0 gives plain
            1-token-per-sequence decode.
        zero_initialize_out: Whether to zero the output rows past the last
            real token, which no sequence owns. Real rows are always written.
        compute_precision: Computation dtype.
        decode_tile_size: Sequences per tile for decode sequences.
        mixed_tile_size: Rows one prefill / mixed DMA tile moves; a multiple
            of compute_chunk_size.
        compute_chunk_size: Tokens per chunked-GDN math step; a mixed tile
            unrolls mixed_tile_size // compute_chunk_size of them.

    Returns:
        (new_conv_state, new_recurrent_state): The updated state caches.
        out: The output [batch_size, n_v * d_v].
    """
    if qkv.dtype != jnp.bfloat16:
        raise TypeError(f"qkv must be bfloat16, got {qkv.dtype}")
    if qkv.ndim != 2 or qkv.shape[0] == 0:
        raise ValueError("qkv must be a nonempty [tokens, dim] array")
    # Note (david): the chunked math and the conv halo work on whole bf16
    # sublane tiles.
    sublane_rows = compute_gdn.BF16_SUBLANE_ROWS
    if compute_chunk_size <= 0 or compute_chunk_size % sublane_rows:
        raise ValueError("compute_chunk_size must be a positive multiple of "
                         f"{sublane_rows}")
    # Note (david): the triangular inverse combines diagonal blocks
    # BLOCKS_PER_LEVEL at a time (4 -> 16 -> 64 -> ...), so every level below
    # the chunk must tile it. Levels 4 and 16 divide any multiple of 16, so the
    # check starts one level above.
    inverse_block_size = sublane_rows * compute_gdn.BLOCKS_PER_LEVEL
    while inverse_block_size < compute_chunk_size:
        if compute_chunk_size % inverse_block_size:
            raise ValueError(
                "compute_chunk_size must be divisible by inverse block size "
                f"{inverse_block_size}")
        inverse_block_size *= compute_gdn.BLOCKS_PER_LEVEL
    if not 1 <= kernel_size - 1 <= compute_conv1d.HALO_ROWS:
        raise ValueError("kernel_size must be between 2 and "
                         f"{compute_conv1d.HALO_ROWS + 1}")
    if compute_chunk_size <= kernel_size - 1:
        raise ValueError(
            "compute_chunk_size must exceed the convolution history")
    if mixed_tile_size <= 0 or mixed_tile_size % compute_chunk_size:
        raise ValueError(
            "mixed_tile_size must be a positive multiple of compute_chunk_size")

    conv_out_dtype = conv_state.dtype
    recurrent_out_dtype = recurrent_state.dtype

    # Note (david): qkv enters the kernel in the caller's [batch, dim] layout
    # and dtype and is cast to f32 on-chip; an f32 [batch, 1, dim] reshape here
    # is a T(8,128) -> T(1,128) retile that XLA materializes in HBM, 0.497 ms
    # of gdn_v3's 1.306 ms forward at T = 8192, dim = 6144.
    b = b.astype(jnp.float32)
    a = a.astype(jnp.float32)
    conv_state = conv_state.astype(jnp.float32)

    num_seqs = state_indices.size
    batch_size, dim = qkv.shape
    assert conv_weight.shape == (dim, 1, kernel_size)
    if conv_bias is not None:
        assert conv_bias.shape == (dim, )
    assert query_start_loc.shape == (num_seqs + 1, )
    assert state_indices.shape == (num_seqs, )
    assert distribution.shape == (3, )
    if num_spec_tokens > 0:
        assert read_offsets is not None, (
            "read_offsets is required when num_spec_tokens > 0")
    if read_offsets is None:
        read_offsets = jnp.zeros((num_seqs, ), dtype=jnp.int32)
    else:
        read_offsets = read_offsets.astype(jnp.int32)
    assert read_offsets.shape == (num_seqs, )
    assert read_state_indices.shape == (num_seqs, )
    read_state_indices = read_state_indices.astype(state_indices.dtype)

    tpu_info = pltpu.get_tpu_info()
    num_lanes = tpu_info.num_lanes
    mixed_tile_size = min(
        mixed_tile_size,
        pl.cdiv(batch_size, compute_chunk_size) * compute_chunk_size)
    aligned_num_v_heads = pl.cdiv(n_v, num_lanes) * num_lanes

    if num_spec_tokens > 0:
        # Note (david): a verify window keeps one state checkpoint per position
        # and sequence in VMEM, multiplying the per-sequence footprint by the
        # window size. The tile shrinks so the double-buffered windows fit in
        # about half the scoped VMEM budget; weights, activation scratch and
        # compiler temporaries take the rest.
        verify_window_size = num_spec_tokens + 1
        recurrent_ckpt_bytes = n_v * d_k * d_v * F32_BYTES
        conv_ckpt_bytes = (kernel_size - 1) * dim * F32_BYTES
        activation_bytes = (dim * F32_BYTES +
                            2 * aligned_num_v_heads * F32_BYTES +
                            n_v * d_v * BF16_BYTES)
        bytes_per_seq = verify_window_size * (
            recurrent_ckpt_bytes + conv_ckpt_bytes + activation_bytes)
        vmem_budget = int(WINDOWED_VMEM_FRACTION *
                          tpu_info.vmem_capacity_bytes)
        spec_tile_budget = (vmem_budget // 2) // config.NUM_BUFFERS
        decode_tile_size = max(
            1,
            min(decode_tile_size, batch_size,
                spec_tile_budget // bytes_per_seq))
    else:
        decode_tile_size = min(decode_tile_size, batch_size)
        # Plain decode keeps that tile wherever its state rings fit VMEM: every
        # ring slot holds each sequence's initial state and its one checkpoint.
        # Where they cannot fit (v7x has half of v6e's VMEM: 4 sequences of 48
        # value heads at head_dim 128), the tile shrinks until they do with one
        # more state per sequence left over for the compiler's temporaries.
        state_bytes = n_v * d_k * d_v * recurrent_state.dtype.itemsize
        ring_bytes_per_seq = config.NUM_BUFFERS * 2 * (
            state_bytes + (kernel_size - 1) * dim * F32_BYTES)
        vmem_limit = int(DEFAULT_VMEM_FRACTION * tpu_info.vmem_capacity_bytes)
        if decode_tile_size * ring_bytes_per_seq > vmem_limit:
            decode_tile_size = max(
                1, vmem_limit // (ring_bytes_per_seq + state_bytes))

    # Note (david): b and a keep the compact [batch, 1, heads] layout: they are
    # tiny, and their untiled leading token axis keeps ragged DMAs offset-free.
    num_v_padding = aligned_num_v_heads - n_v
    b = jnp.pad(b, ((0, 0), (0, num_v_padding))).reshape(batch_size, 1, -1)
    a = jnp.pad(a, ((0, 0), (0, num_v_padding))).reshape(batch_size, 1, -1)

    conv_weight = conv_weight.swapaxes(0, 2).reshape(kernel_size,
                                                     dim).astype(jnp.float32)
    conv_bias = None if conv_bias is None else conv_bias.astype(jnp.float32)
    weights = memory_ref.WeightRefs(
        conv=memory_ref.ConvWeightsRef(weight=conv_weight, bias=conv_bias),
        gdn=memory_ref.GDNWeightsRef(a_log=a_log, dt_bias=dt_bias),
    )

    smem_spec = pl.BlockSpec(memory_space=pltpu.SMEM)
    vmem_spec = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    weights_spec = jax.tree.map(lambda _: vmem_spec, weights)

    def _call_kernel(
        in_conv_state: jax.Array,
        in_recurrent_state: jax.Array,
        aliased_out: jax.Array | None,
        mode: config.GDNMode,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        if mode == config.GDNMode.PER_SEQ:
            tile_size = mixed_tile_size
            window_size = 1
            kernel_compute_chunk = compute_chunk_size
        else:
            tile_size = decode_tile_size
            window_size = num_spec_tokens + 1
            # Note (david): a BATCHED tile holds each sequence's whole verify
            # window, which is one math chunk.
            kernel_compute_chunk = window_size

        cfg = config.GDNConfig(
            mode=mode,
            batch_size=batch_size,
            kernel_size=kernel_size,
            tile_size=tile_size,
            compute_chunk_size=kernel_compute_chunk,
            window_size=window_size,
            dim_size=dim,
            num_kq_heads=n_kq,
            num_v_heads=n_v,
            kq_head_dim=d_k,
            v_head_dim=d_v,
            dtypes=config.Dtypes(
                act_in=qkv.dtype,
                act_out=qkv.dtype,
                compute=compute_precision,
                recurrent_state=in_recurrent_state.dtype,
                conv_state=in_conv_state.dtype,
            ),
        )

        # Note (david): the metadata is rebuilt for every layer; the compiler
        # CSEs the copies.
        if mode == config.GDNMode.PER_SEQ:
            seq_metadata = metadata.compute_per_seq_metadata(
                cfg=cfg,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
                state_indices=state_indices,
                start_seq=distribution[0],
                end_seq=distribution[-1],
                read_indices=read_state_indices,
            )
        else:
            seq_metadata = metadata.compute_batched_seq_metadata(
                cfg=cfg,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
                state_indices=state_indices,
                read_offsets=read_offsets,
                end_seq=distribution[0],
                read_indices=read_state_indices,
            )
        metadata_spec = jax.tree.map(lambda _: smem_spec, seq_metadata)

        num_metadata_leaves = len(seq_metadata)
        state_aliases = {
            num_metadata_leaves + CONV_STATE_OPERAND: 1,
            num_metadata_leaves + RECURRENT_STATE_OPERAND: 2,
        }
        if aliased_out is None:
            out_shape = jax.ShapeDtypeStruct(
                (cfg.batch_size, cfg.v_dim_size), cfg.dtypes.act_out)
            aliased_out_spec = None
            input_output_aliases = state_aliases
        else:
            out_shape = aliased_out
            aliased_out_spec = hbm_spec
            input_output_aliases = state_aliases | {
                num_metadata_leaves + ALIASED_OUT_OPERAND: 0
            }

        # Note (david): the PER_SEQ kernel zero-fills [pad_start, batch_size),
        # the output rows no sequence owns; pad_start = batch_size turns the
        # fill off.
        if mode != config.GDNMode.PER_SEQ:
            pad_start = None
            pad_spec = None
        elif zero_initialize_out:
            pad_start = query_start_loc[distribution[-1]].reshape(1)
            pad_spec = smem_spec
        else:
            pad_start = jnp.full((1, ), batch_size, jnp.int32)
            pad_spec = smem_spec

        if window_size > 1:
            vmem_fraction = WINDOWED_VMEM_FRACTION
            # Note (david): windows of different sizes compile to different
            # kernels; the suffix keeps them apart in profiles.
            kernel_name_suffix = f"_w{window_size}"
        else:
            vmem_fraction = DEFAULT_VMEM_FRACTION
            kernel_name_suffix = ""

        kernel_metadata = {}
        for path, leaf in jax.tree_util.tree_leaves_with_path(
                dataclasses.asdict(cfg)):
            key = jax.tree_util.keystr(path, simple=True, separator=".")
            if isinstance(leaf, str | int | float):
                kernel_metadata[key] = leaf
            else:
                kernel_metadata[key] = str(leaf)

        return pl.pallas_call(
            functools.partial(outer_kernel, cfg=cfg),
            out_shape=(out_shape, in_conv_state, in_recurrent_state),
            in_specs=(
                metadata_spec,
                hbm_spec,
                hbm_spec,
                hbm_spec,
                hbm_spec,
                hbm_spec,
                aliased_out_spec,
                weights_spec,
                pad_spec,
            ),
            out_specs=(hbm_spec, hbm_spec, hbm_spec),
            scratch_shapes=cfg.get_scratch_shape_dict(),
            input_output_aliases=input_output_aliases,
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                vmem_limit_bytes=int(vmem_fraction *
                                     tpu_info.vmem_capacity_bytes),
            ),
            name=f"fused_conv1d_gdn_{mode.value}{kernel_name_suffix}",
            metadata=kernel_metadata,
        )(
            seq_metadata,
            qkv,
            b,
            a,
            in_conv_state,
            in_recurrent_state,
            aliased_out,
            weights,
            pad_start,
        )

    out, conv_state_out, recurrent_state_out = _call_kernel(
        conv_state, recurrent_state, None, config.GDNMode.BATCHED)
    out, conv_state_out, recurrent_state_out = _call_kernel(
        conv_state_out, recurrent_state_out, out, config.GDNMode.PER_SEQ)

    return (conv_state_out.astype(conv_out_dtype),
            recurrent_state_out.astype(recurrent_out_dtype)), out
