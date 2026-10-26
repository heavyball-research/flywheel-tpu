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
import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from . import (
    compute_conv1d,
    compute_kda,
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
# qkv, b, g, conv_state, recurrent_state and the aliased output.
CONV_STATE_OPERAND = 3
RECURRENT_STATE_OPERAND = 4
ALIASED_OUT_OPERAND = 5


def chunk_qkv_view(qkv_slot_ref: jax.Array, chunk_idx: int | jax.Array,
                   cfg: config.KDAConfig) -> jax.Array:
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
                   cfg: config.KDAConfig) -> None:
    """Write one compute chunk's output into the tile's output stage.

    out is [seq, chunk, v_dim_size]. PER_SEQ writes chunk chunk_idx at its
    block-grid rows; BATCHED packs each sequence's real rows back to back from
    stage row delta.
    """
    out_dtype = out_slot_ref.dtype
    if cfg.mode == config.KDAMode.PER_SEQ:
        compute_chunk = cfg.compute_chunk_size
        if isinstance(chunk_idx, int):
            row_offset = chunk_idx * compute_chunk
        else:
            # Note (david): out_window_rows makes the stage a multiple of
            # STAGE_TILE_ROWS, which v7x tiles 16 rows deep, so an 8-row hint
            # cannot prove this store aligned (E2003). The chunk grid can.
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
    g_slot_ref: jax.Array,
    conv_in_ref: jax.Array,
    recurrent_in_ref: jax.Array,
    out_slot_ref: jax.Array,
    conv_out_ref: jax.Array,
    recurrent_out_ref: jax.Array,
    metadata_ref: memory_ref.MetadataRef,
    weights_ref: memory_ref.WeightRefs,
    carry_conv_scratch_ref: jax.Array | None,
    carry_recurrent_scratch_ref: jax.Array | None,
    replay_in_ref: jax.Array | None,
    replay_out_ref: jax.Array | None,
    replay_gate_in_ref: jax.Array | None,
    replay_gate_out_ref: jax.Array | None,
    recurrent_lo_in_ref: jax.Array | None,
    wait_flush_in: Callable[[int], None] | None,
    start_flush_out: Callable[[int], None] | None,
    *,
    cfg: config.KDAConfig,
) -> None:
    """Compute one tile from its VMEM ring slots and stage its outputs.

    A replay tile reads its checkpoints' hi planes from recurrent_in_ref and
    owns its flushes' DMAs: wait_flush_in(idx) waits for record idx's lo plane
    read into recurrent_lo_in_ref and start_flush_out(idx) starts its
    checkpoint write.
    """
    profile_scope = compute_kda.profile_scope
    with profile_scope("load_states"):
        real_sizes, prev_conv, prev_recurrent = vmem_ldst.load_and_select_states(
            metadata_ref=metadata_ref,
            p_id=p_id,
            conv_state_slot_ref=conv_in_ref,
            recurrent_slot_ref=recurrent_in_ref,
            recurrent_lo_slot_ref=recurrent_lo_in_ref,
            carry_conv_scratch_ref=carry_conv_scratch_ref,
            carry_recurrent_scratch_ref=carry_recurrent_scratch_ref,
            cfg=cfg,
        )

    conv_weight = weights_ref.conv.weight[...].astype(jnp.float32)
    if weights_ref.conv.bias is None:
        conv_bias = None
    else:
        conv_bias = weights_ref.conv.bias[...].astype(jnp.float32)
    a_log = weights_ref.kda.a_log[...]
    dt_bias = weights_ref.kda.dt_bias[...]
    r_base = metadata_ref.get_record(p_id, 0).r_base
    delta = r_base & (config.SUBLANE_ALIGN - 1)

    def _store_states(new_conv_state, new_recurrent_state):
        with profile_scope("state_store"):
            conv_out_ref[...] = new_conv_state
            if cfg.split_checkpoint:
                recurrent_out_ref[...] = jnp.stack(
                    compute_kda.split_checkpoint_planes(
                        new_recurrent_state), axis=-4)
            else:
                recurrent_out_ref[...] = new_recurrent_state.astype(
                    recurrent_out_ref.dtype)
            if carry_conv_scratch_ref is not None:
                carry_conv_scratch_ref[...] = new_conv_state[:, -1]
            if carry_recurrent_scratch_ref is not None:
                carry_recurrent_scratch_ref[...] = new_recurrent_state[:, -1]

    if cfg.is_replay:
        with profile_scope("load_qkv"):
            qkv_in = vmem_ldst.load_qkv_realigned(
                metadata_ref=metadata_ref, p_id=p_id,
                qkv_slot_ref=qkv_slot_ref, cfg=cfg)
            g_in = vmem_ldst.load_qkv_realigned(
                metadata_ref=metadata_ref, p_id=p_id,
                qkv_slot_ref=g_slot_ref, cfg=cfg)
        with profile_scope("conv1d"):
            qkv_out, conv_ckpt = compute_conv1d.causal_conv1d(
                chunk_tokens=qkv_in, prev_conv=prev_conv,
                conv_weight=conv_weight, conv_bias=conv_bias, cfg=cfg)
        with profile_scope("silu"):
            qkv_out = compute_kda.silu(qkv_out)
        q, k, v, b, g = vmem_ldst.load_activation_as_chunked(
            qkv_vreg=qkv_out, g_vreg=g_in, b_vmem_ref=b_slot_ref, cfg=cfg)
        record_row_idx = lax.broadcasted_iota(
            jnp.int32, (cfg.replay_window, cfg.record_dim), 0)
        gate_row_idx = lax.broadcasted_iota(
            jnp.int32, (cfg.replay_window, cfg.gate_dim), 0)

        def _num_records(record):
            return jnp.where(record.is_first_tile,
                             metadata_ref.s_idx_to_replay_len[record.s_idx], 0)

        def _flush_pass():
            for idx in range(cfg.seq_tile_size):
                record = metadata_ref.get_record(p_id, idx)

                @pl.when(metadata_ref.s_idx_to_flush[record.s_idx] != 0)
                def _flush(idx=idx, record=record):
                    checkpoint = compute_kda.replay_flush(
                        records=replay_in_ref[idx].astype(jnp.float32),
                        gates=replay_gate_in_ref[idx],
                        num_records=_num_records(record),
                        checkpoint_hi_ref=recurrent_in_ref.at[idx, 0],
                        checkpoint_lo_ref=recurrent_lo_in_ref.at[idx, 0],
                        wait_checkpoint_lo=functools.partial(
                            wait_flush_in, idx),
                        cfg=cfg)
                    hi_plane, lo_plane = compute_kda.split_checkpoint_planes(
                        checkpoint)
                    recurrent_out_ref[idx, 0, 0] = hi_plane
                    recurrent_out_ref[idx, 0, 1] = lo_plane
                    start_flush_out(idx)

        def _merge_rows(row_idx, new_rows, rows_in, base, num_new_rows):
            """rows_in with new_rows' first num_new_rows rows written from row
            base on."""
            if cfg.replay_window == cfg.window_size:
                padded_rows = new_rows
            else:
                padded_rows = jnp.concatenate([
                    new_rows,
                    jnp.zeros((cfg.replay_window - cfg.window_size,
                               new_rows.shape[-1]), new_rows.dtype)
                ], axis=0)
            shifted = pltpu.roll(padded_rows, base, 0)
            is_new_row = (row_idx >= base) & (row_idx < base + num_new_rows)
            return jnp.where(is_new_row, shifted, rows_in)

        # Note (david): folding mid tile overlaps the flush's lo plane read
        # with the steps before it and its writes with the steps after. With
        # an f32 flush read in place of the lo plane (v6e, B=256, decode, tile
        # 4) this measured 494.4 us with the flush DMAs at the tile
        # boundaries and 416.0 us folding after 2 steps.
        flush_pass_pos = cfg.seq_tile_size // 2
        seq_outs = []
        for idx in range(cfg.seq_tile_size):
            if idx == flush_pass_pos:
                _flush_pass()
            record = metadata_ref.get_record(p_id, idx)
            s_idx = record.s_idx
            records_in = replay_in_ref[idx].astype(jnp.float32)
            gates_in = replay_gate_in_ref[idx]
            out, new_records, new_gates = compute_kda.replay_kda(
                q_chunked=q[idx], k_chunked=k[idx], v_chunked=v[idx],
                b_rows=b[idx], g_chunked=g[idx], num_tokens=record.r_size,
                num_records=_num_records(record), records=records_in,
                gates=gates_in, checkpoint_hi_ref=recurrent_in_ref.at[idx, 0],
                a_log=a_log, dt_bias=dt_bias, cfg=cfg)
            seq_outs.append(out)
            with profile_scope("replay_merge"):
                base = metadata_ref.replay_base(s_idx)
                # Note (david): the aligned window replay_out_copy writes also
                # covers committed rows, so the image keeps the rows it read.
                # The roll and the merge run on unpacked f32 rows; only the
                # merged image is cast to the storage dtype.
                replay_out_ref[idx] = _merge_rows(
                    record_row_idx, new_records, records_in, base,
                    record.r_size).astype(replay_out_ref.dtype)
                replay_gate_out_ref[idx] = _merge_rows(
                    gate_row_idx, new_gates, gates_in, base, record.r_size)

        with profile_scope("store_out"):
            store_out_rows(out_slot_ref, jnp.stack(seq_outs, axis=0), 0,
                           delta, metadata_ref, p_id, cfg)
        with profile_scope("state_store"):
            conv_out_ref[...] = conv_ckpt
        return

    if cfg.mode == config.KDAMode.BATCHED:
        with profile_scope("load_qkv"):
            qkv_in = vmem_ldst.load_qkv_realigned(
                metadata_ref=metadata_ref, p_id=p_id,
                qkv_slot_ref=qkv_slot_ref, cfg=cfg)
            g_in = vmem_ldst.load_qkv_realigned(
                metadata_ref=metadata_ref, p_id=p_id,
                qkv_slot_ref=g_slot_ref, cfg=cfg)
        with profile_scope("conv1d"):
            qkv_out, conv_ckpt = compute_conv1d.causal_conv1d(
                chunk_tokens=qkv_in, prev_conv=prev_conv,
                conv_weight=conv_weight, conv_bias=conv_bias, cfg=cfg)
        with profile_scope("silu"):
            qkv_out = compute_kda.silu(qkv_out)
        q, k, v, b, g = vmem_ldst.load_activation_as_token(
            qkv_vreg=qkv_out, g_vreg=g_in, b_vmem_ref=b_slot_ref, cfg=cfg)
        out, recurrent_ckpt = compute_kda.recurrent_kda(
            real_sizes=real_sizes,
            q_token=q, k_token=k, v_token=v,
            b_token=b, g_token=g, state_prev=prev_recurrent,
            a_log=a_log, dt_bias=dt_bias, cfg=cfg)
        with profile_scope("store_out"):
            store_out_rows(out_slot_ref, out, 0, delta, metadata_ref, p_id,
                           cfg)
        _store_states(conv_ckpt, recurrent_ckpt)
        return

    assert cfg.mode == config.KDAMode.PER_SEQ
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
            qkv_out = compute_kda.silu(qkv_out)
        return row_lo, row_hi, qkv_out, next_halo

    def _load_act(chunk_idx, qkv_out):
        # Note (david): the gate block shares the qkv block's grid, so its chunk
        # rows are the same slice, and its pad rows are masked like qkv's.
        g_chunk = chunk_qkv_view(g_slot_ref, chunk_idx, cfg)[...]
        if cfg.num_compute_chunks == 1:
            b_chunk_ref = b_slot_ref
        else:
            # Note (david): the compact b token axis is untiled, so a traced
            # chunk offset needs no alignment hint.
            chunk_rows = pl.ds(chunk_idx * compute_chunk, compute_chunk)
            b_chunk_ref = b_slot_ref.at[:, chunk_rows]
        return vmem_ldst.load_activation_as_chunked(
            qkv_vreg=qkv_out, g_vreg=g_chunk, b_vmem_ref=b_chunk_ref,
            cfg=cfg)

    def _run_chunk(chunk_idx, halo, recurrent_state):
        row_lo, row_hi, qkv_out, next_halo = _chunk_front(chunk_idx, halo)
        with profile_scope("load_act"):
            q, k, v, b, g = _load_act(chunk_idx, qkv_out)
        out, recurrent_ckpt = compute_kda.chunked_kda(
            row_lo=row_lo, row_hi=row_hi,
            q_chunked=q, k_chunked=k, v_chunked=v, b_chunked=b, g_chunked=g,
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
                q, k, v, b, g = _load_act(chunk_idx, qkv_out)
            q, k, v, g, beta = compute_kda.prepare_chunk(
                row_lo, row_hi, q, k, v, b, g, a_log, dt_bias, cfg)
            chunks.append(compute_kda.ChunkKDA(
                q[0], k[0], v[0], g[0], beta[0], cfg))
        chunk_outs, recurrent_state = compute_kda.run_pipelined(
            chunks, prev_recurrent[0])
        for chunk_idx, chunk_out in enumerate(chunk_outs):
            with profile_scope("store_out"):
                store_out_rows(out_slot_ref,
                               compute_kda.pack_out(chunk_out, cfg)[None],
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
    g_ref: jax.Array,
    conv_state_ref: jax.Array,
    recurrent_state_ref: jax.Array,
    aliased_out_ref: jax.Array | None,
    weights_ref: memory_ref.WeightRefs,
    pad_start_ref: jax.Array | None,
    replay_state_ref: jax.Array | None,
    replay_gates_ref: jax.Array | None,
    out_ref: jax.Array,
    conv_state_out_ref: jax.Array,
    recurrent_state_out_ref: jax.Array,
    *extra_state_out_refs: jax.Array,
    # Note (david): pallas_call passes the scratch shapes by keyword, so these
    # names are the keys of cfg.get_scratch_shape_dict.
    qkv_buf: jax.Array,
    b_buf: jax.Array,
    g_buf: jax.Array,
    conv_in_buf: jax.Array,
    rec_in_buf: jax.Array,
    rec_lo_in_buf: jax.Array | None,
    conv_out_buf: jax.Array,
    rec_out_buf: jax.Array,
    replay_in_buf: jax.Array | None,
    replay_out_buf: jax.Array | None,
    replay_gate_in_buf: jax.Array | None,
    replay_gate_out_buf: jax.Array | None,
    out_buf: jax.Array,
    carry_conv_scratch_ref: jax.Array | None,
    carry_recurrent_scratch_ref: jax.Array | None,
    sems: jax.Array,
    cfg: config.KDAConfig,
) -> None:
    """Double-buffered tile pipeline over inner_kernel.

    Tile t uses ring slot t % 2. The inputs of tile t + 1 are prefetched before
    tile t computes and the outputs of tile t - 1 are waited only after tile
    t's outputs start, so both overlap tile t's compute. State inputs and
    outputs use separate rings, so no slot is read by one DMA while another
    writes it. A replay tile's flushes wait for their checkpoints' lo plane
    reads and start their checkpoint writes inside its compute.
    """
    # Note (david): the tile loop indexes its VMEM rings as t % 2 and 1 - slot.
    assert cfg.num_buffers == config.NUM_BUFFERS, cfg.num_buffers
    # Note (david): aliased_out_ref aliases out_ref and the state outputs alias
    # their inputs (the replay records and gates, when present, follow the
    # recurrent one); the DMAs go through the other ref of each pair.
    del aliased_out_ref, conv_state_out_ref, recurrent_state_out_ref
    del extra_state_out_refs

    align = config.SUBLANE_ALIGN
    stage_tile = config.STAGE_TILE_ROWS
    # Note (david): the head blend picks one SUBLANE_ALIGN half of a stage tile.
    assert stage_tile == 2 * align, (stage_tile, align)
    num_tiles = metadata_ref.num_tiles[...]
    first_window_base, _, _, _ = memory_ref.out_window(0, metadata_ref, cfg)
    profile_scope = compute_kda.profile_scope

    def _start_in(p_id, slot):
        with profile_scope("start_in"):
            tile_args = (p_id, metadata_ref, cfg)
            memory_ref.start_qkv_in(qkv_ref, qkv_buf.at[slot],
                                    sems.at[config.STREAM_QKV, slot],
                                    *tile_args)
            memory_ref.start_compact_in(b_ref, b_buf.at[slot],
                                        sems.at[config.STREAM_B, slot],
                                        *tile_args)
            # Note (david): the gate stream is token-major like qkv, so it takes
            # the same widened DMA.
            memory_ref.start_qkv_in(g_ref, g_buf.at[slot],
                                    sems.at[config.STREAM_G, slot],
                                    *tile_args)
            memory_ref.start_conv_in(conv_state_ref, conv_in_buf.at[slot],
                                     sems.at[config.STREAM_CONV_IN, slot],
                                     *tile_args)
            if cfg.is_replay:
                # Note (david): a replay checkpoint has a single slot, so it
                # ignores the read offset. Only the flushes read the lo plane,
                # mid tile, so it queues behind the inputs the tile waits for
                # at its start.
                memory_ref.start_state_in(
                    recurrent_state_ref, rec_in_buf.at[slot],
                    sems.at[config.STREAM_REC_IN, slot], *tile_args,
                    apply_read_offset=False, plane=0)
                memory_ref.start_replay_in(
                    replay_state_ref, replay_in_buf.at[slot],
                    sems.at[config.STREAM_REPLAY_IN, slot], *tile_args)
                memory_ref.start_replay_in(
                    replay_gates_ref, replay_gate_in_buf.at[slot],
                    sems.at[config.STREAM_REPLAY_GATE_IN, slot], *tile_args)
                memory_ref.start_state_in(
                    recurrent_state_ref, rec_lo_in_buf.at[slot],
                    sems.at[config.STREAM_REC_LO_IN, slot], *tile_args,
                    apply_read_offset=False, only_flushing=True, plane=1)
            else:
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
            memory_ref.wait_qkv_in(g_ref, g_buf.at[slot],
                                   sems.at[config.STREAM_G, slot], *tile_args)
            memory_ref.wait_conv_in(conv_state_ref, conv_in_buf.at[slot],
                                    sems.at[config.STREAM_CONV_IN, slot],
                                    *tile_args)
            memory_ref.wait_state_in(rec_in_buf.at[slot],
                                     sems.at[config.STREAM_REC_IN, slot],
                                     *tile_args)
            if cfg.is_replay:
                memory_ref.wait_replay_in(
                    replay_state_ref, replay_in_buf.at[slot],
                    sems.at[config.STREAM_REPLAY_IN, slot], *tile_args)
                memory_ref.wait_replay_in(
                    replay_gates_ref, replay_gate_in_buf.at[slot],
                    sems.at[config.STREAM_REPLAY_GATE_IN, slot], *tile_args)

    def _start_out(p_id, slot):
        with profile_scope("start_out"):
            tile_args = (p_id, metadata_ref, cfg)
            memory_ref.start_out(out_ref, out_buf.at[slot],
                                 sems.at[config.STREAM_OUT, slot], *tile_args)
            memory_ref.start_conv_out(conv_state_ref, conv_out_buf.at[slot],
                                      sems.at[config.STREAM_CONV_OUT, slot],
                                      *tile_args)
            # Note (david): a replay tile starts its flushed checkpoint writes
            # in inner_kernel, right after each fold.
            if cfg.is_replay:
                memory_ref.start_replay_out(
                    replay_state_ref, replay_out_buf.at[slot],
                    sems.at[config.STREAM_REPLAY_OUT, slot], *tile_args)
                memory_ref.start_replay_out(
                    replay_gates_ref, replay_gate_out_buf.at[slot],
                    sems.at[config.STREAM_REPLAY_GATE_OUT, slot], *tile_args)
            else:
                memory_ref.start_state_out(
                    recurrent_state_ref, rec_out_buf.at[slot],
                    sems.at[config.STREAM_REC_OUT, slot], *tile_args)

    def _wait_out(p_id, slot):
        with profile_scope("wait_out"):
            tile_args = (p_id, metadata_ref, cfg)
            memory_ref.wait_out(out_ref, out_buf.at[slot],
                                sems.at[config.STREAM_OUT, slot], *tile_args)
            memory_ref.wait_conv_out(conv_state_ref, conv_out_buf.at[slot],
                                     sems.at[config.STREAM_CONV_OUT, slot],
                                     *tile_args)
            if cfg.is_replay:
                memory_ref.wait_ckpt_out(rec_out_buf.at[slot],
                                         sems.at[config.STREAM_REC_OUT, slot],
                                         *tile_args)
                memory_ref.wait_replay_out(
                    replay_state_ref, replay_out_buf.at[slot],
                    sems.at[config.STREAM_REPLAY_OUT, slot], *tile_args)
                memory_ref.wait_replay_out(
                    replay_gates_ref, replay_gate_out_buf.at[slot],
                    sems.at[config.STREAM_REPLAY_GATE_OUT, slot], *tile_args)
            else:
                memory_ref.wait_state_out(
                    rec_out_buf.at[slot],
                    sems.at[config.STREAM_REC_OUT, slot], *tile_args)

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

        def _wait_flush_in(idx):
            memory_ref.wait_flushed_ckpt_in(
                rec_lo_in_buf.at[slot], sems.at[config.STREAM_REC_LO_IN, slot],
                tile_idx, idx, metadata_ref, cfg)

        def _start_flush_out(idx):
            memory_ref.start_flushed_ckpt_out(
                recurrent_state_ref, rec_out_buf.at[slot],
                sems.at[config.STREAM_REC_OUT, slot], tile_idx, idx,
                metadata_ref)

        _wait_in(tile_idx, slot)
        inner_kernel(
            tile_idx,
            qkv_buf.at[slot],
            b_buf.at[slot],
            g_buf.at[slot],
            conv_in_buf.at[slot],
            rec_in_buf.at[slot],
            out_buf.at[slot],
            conv_out_buf.at[slot],
            rec_out_buf.at[slot],
            metadata_ref,
            weights_ref,
            carry_conv_scratch_ref,
            carry_recurrent_scratch_ref,
            None if replay_in_buf is None else replay_in_buf.at[slot],
            None if replay_out_buf is None else replay_out_buf.at[slot],
            None if replay_gate_in_buf is None else
            replay_gate_in_buf.at[slot],
            None if replay_gate_out_buf is None else
            replay_gate_out_buf.at[slot],
            None if rec_lo_in_buf is None else rec_lo_in_buf.at[slot],
            _wait_flush_in if cfg.is_replay else None,
            _start_flush_out if cfg.is_replay else None,
            cfg=cfg,
        )
        window_base, _, delta, used_rows = memory_ref.out_window(
            tile_idx, metadata_ref, cfg)

        with profile_scope("out_fixup"):
            # Note (david): rows past the tile's last real one hold stale stage
            # data (a chunk the tail path skipped, or the rotate's zero pad).
            # Only the stage tile holding used_rows can still fall inside the
            # DMA window; the next tile's window overwrites it, and the last
            # tile's lands in the padded tail, which the zero fill below
            # expects zero.
            stage_ref = out_buf.at[slot]
            tail_slab_start = pl.multiple_of(
                used_rows - (used_rows & (stage_tile - 1)), stage_tile)
            tail_slab_rows = tail_slab_start + lax.broadcasted_iota(
                jnp.int32, (stage_tile, cfg.v_dim_size), 0)
            stage_ref[pl.ds(tail_slab_start, stage_tile)] = jnp.where(
                tail_slab_rows < used_rows,
                stage_ref[pl.ds(tail_slab_start, stage_tile)], 0)

            # Note (david): the window's head rows belong to the previous tile,
            # whose stage is still resident since only its outbound DMA reads
            # it. Both windows sit on the SUBLANE_ALIGN grid, so the previous
            # stage holds them from row prev_stage_row, a multiple of 8 that
            # may sit mid stage tile, and the previous window always covers
            # row window_base + delta - 1. Both stage tiles are read whole and
            # the previous one's half is picked in f32, where 8 rows are one
            # whole (8, 128) tile.
            @pl.when(delta > 0)
            def _blend_head_rows():
                prev_stage_row = window_base - prev_window_base
                prev_slab_start = pl.multiple_of(
                    prev_stage_row - (prev_stage_row & (stage_tile - 1)),
                    stage_tile)
                prev_slab = out_buf.at[other_slot][
                    pl.ds(prev_slab_start, stage_tile)].astype(jnp.float32)
                own_slab = stage_ref[pl.ds(0, stage_tile)].astype(jnp.float32)
                head_rows = lax.broadcasted_iota(
                    jnp.int32, (align, cfg.v_dim_size), 0)
                prev_head = jnp.where(
                    head_rows + (prev_stage_row - prev_slab_start) < align,
                    prev_slab[:align], prev_slab[align:])
                own_head = jnp.where(head_rows < delta, prev_head,
                                     own_slab[:align])
                stage_ref[pl.ds(0, stage_tile)] = jnp.concatenate(
                    [own_head, own_slab[align:]]).astype(stage_ref.dtype)

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

    if cfg.mode == config.KDAMode.PER_SEQ:
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


STATIC_ARGNAMES = (
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
    "lower_bound",
    "beta_is_activated",
)


@dataclasses.dataclass(frozen=True)
class ReplayLayout:
    """One slot of the ReplaySSM state of fused_conv1d_kda_replay in HBM.

    record_dim: lanes of one replay record [u | k], the last axis of the
        replay state.
    gate_dim: lanes of one replay gate row, the last axis of the replay
        gates: the per-key-channel log decay per value head.
    checkpoint_shape: the int32 checkpoint, the hi and lo planes of an f32
        [n_v, d_k, d_v] checkpoint with two 16-bit values per word (rows i
        and i + d_k / 2).
    """
    record_dim: int
    gate_dim: int
    checkpoint_shape: tuple[int, int, int, int]


def replay_layout(n_kq: int, n_v: int, d_k: int, d_v: int) -> ReplayLayout:
    """The ReplayLayout of a model with n_kq kq heads of d_k and n_v value
    heads of d_v."""
    # Note (david): a decode unpacks the two row halves of a word onto whole
    # f32 sublane tiles.
    if d_k % (2 * config.SUBLANE_ALIGN):
        raise ValueError("a replay checkpoint needs d_k divisible by "
                         f"{2 * config.SUBLANE_ALIGN}, got {d_k}")
    return ReplayLayout(record_dim=n_v * d_v + n_kq * d_k,
                        gate_dim=n_v * d_k,
                        checkpoint_shape=(2, n_v, d_k // 2, d_v))


def split_checkpoint(recurrent_state: jax.Array) -> jax.Array:
    """The int32 replay checkpoint [..., 2, n_v, d_k // 2, d_v] of f32
    recurrent states [..., n_v, d_k, d_v]: plane 0 holds each value's bf16
    rounded half up on its bit pattern, plane 1 its low 16 bits."""
    if recurrent_state.dtype != jnp.float32:
        raise TypeError("split_checkpoint takes float32 states, got "
                        f"{recurrent_state.dtype}")
    return jnp.stack(compute_kda.split_checkpoint_planes(recurrent_state),
                     axis=-4)


def merge_checkpoint(checkpoint: jax.Array) -> jax.Array:
    """The f32 recurrent states [..., n_v, d_k, d_v] of an int32 replay
    checkpoint [..., 2, n_v, d_k // 2, d_v], bit for bit."""
    if checkpoint.dtype != jnp.int32:
        raise TypeError("merge_checkpoint takes int32 checkpoints, got "
                        f"{checkpoint.dtype}")
    if checkpoint.shape[-4] != 2:
        raise ValueError("merge_checkpoint takes [..., 2, n_v, d_k // 2, "
                         f"d_v], got {checkpoint.shape}")
    return compute_kda.merge_checkpoint_planes(
        checkpoint[..., 0, :, :, :], checkpoint[..., 1, :, :, :])


@jax.jit(donate_argnames=("conv_state", "recurrent_state"),
         static_argnames=STATIC_ARGNAMES)
def fused_conv1d_kda(
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
    lower_bound: float | None = None,
    beta_is_activated: bool = False,
) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
    """Conv1D + silu + KDA (Kimi Delta Attention) in one fused kernel.

    KDA is the gated delta rule with one decay gate per key channel: the log
    decay -exp(a_log) * softplus(g + dt_bias) is [n_v, d_k] per token, and the
    recurrent state [d_k, d_v] decays row-wise.

    Args:
        qkv: bf16 mixed query, key, value input [batch_size, dim_size], where
            dim_size = n_kq * d_k * 2 + n_v * d_v.
        b: Input to beta [batch_size, n_v].
        g: bf16 raw gate [batch_size, n_v * d_k], head-major like q / k.
        conv_state: Convolution state cache [num_slots, kernel_size - 1,
            dim_size] holding the last kernel_size - 1 tokens of the previous
            invocation. Slot 0 is the null block for padded or invalid tokens.
            A slot may hold garbage on the first invocation of a sequence.
        recurrent_state: Recurrent state cache [num_slots, n_v, d_k, d_v],
            with the same null block and garbage caveat.
        conv_weight: Convolution weight [dim_size, 1, kernel_size].
        conv_bias: Optional convolution bias [dim_size].
        a_log: Per-head log decay scale [n_v].
        dt_bias: Bias of the decay gate [n_v * d_k].
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
        compute_chunk_size: Tokens per chunked-KDA math step; a mixed tile
            unrolls mixed_tile_size // compute_chunk_size of them.
        lower_bound: Optional finite negative bound; when set, the log decay
            is lower_bound * sigmoid(exp(a_log) * (g + dt_bias)) instead.
        beta_is_activated: Whether b already holds beta, skipping the sigmoid.

    Returns:
        (new_conv_state, new_recurrent_state): The updated state caches.
        out: The output [batch_size, n_v * d_v].
    """
    (conv_state, recurrent_state, _, _), out = launch_fused_conv1d_kda(
        qkv, b, g, conv_state, recurrent_state, conv_weight, conv_bias, a_log,
        dt_bias, query_start_loc, state_indices, distribution, seq_lens,
        read_state_indices, read_offsets, None, None, None, n_kq=n_kq,
        n_v=n_v, d_k=d_k, d_v=d_v, kernel_size=kernel_size,
        num_spec_tokens=num_spec_tokens,
        zero_initialize_out=zero_initialize_out,
        compute_precision=compute_precision,
        decode_tile_size=decode_tile_size, mixed_tile_size=mixed_tile_size,
        compute_chunk_size=compute_chunk_size, lower_bound=lower_bound,
        beta_is_activated=beta_is_activated, replay_window=0)
    return (conv_state, recurrent_state), out


@jax.jit(donate_argnames=("conv_state", "recurrent_state", "replay_state",
                          "replay_gates"),
         static_argnames=STATIC_ARGNAMES + ("replay_window", ))
def fused_conv1d_kda_replay(
    qkv: jax.Array,
    b: jax.Array,
    g: jax.Array,
    conv_state: jax.Array,
    recurrent_state: jax.Array,
    replay_state: jax.Array,
    replay_gates: jax.Array,
    conv_weight: jax.Array,
    conv_bias: jax.Array | None,
    a_log: jax.Array,
    dt_bias: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    distribution: jax.Array,
    seq_lens: jax.Array,
    read_state_indices: jax.Array,
    replay_lens: jax.Array,
    read_offsets: jax.Array | None = None,
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    replay_window: int = 16,
    num_spec_tokens: int = 0,
    zero_initialize_out: bool = True,
    compute_precision: jnp.dtype = jnp.float32.dtype,
    decode_tile_size: int = 4,
    mixed_tile_size: int = 128,
    compute_chunk_size: int = 64,
    lower_bound: float | None = None,
    beta_is_activated: bool = False,
) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]:
    """fused_conv1d_kda whose decode segment runs ReplaySSM.

    The replay protocol is that of gdn.wrapper.fused_conv1d_gdn_replay: a
    decode sequence keeps the checkpoint in recurrent_state[slot] plus the
    first replay_lens[s] records of replay_state[slot] and
    replay_gates[slot], appends its tokens' records, and writes the
    checkpoint only when it flushes. A KDA gate row holds the
    per-key-channel log decay, n_v * d_k f32 lanes, where a GDN one holds one
    log decay per value head. The checkpoint's hi and lo planes are as there:
    a decode reads only the hi plane, a flush or prefill both.

    replay_layout(n_kq, n_v, d_k, d_v) gives checkpoint_shape, record_dim and
    gate_dim below.

    Args:
        recurrent_state: int32 checkpoint cache [num_slots,
            *checkpoint_shape], each f32 checkpoint as a hi and a lo plane;
            split_checkpoint and merge_checkpoint convert. All-zero planes
            are the zero state.
        replay_state: f32 or bf16 record buffer [num_slots, replay_window,
            record_dim].
        replay_gates: f32 gates of the records [num_slots, replay_window,
            gate_dim].
        replay_lens: int32 committed records of each sequence [num_seqs],
            0 <= replay_lens[s] <= replay_window.
        replay_window: Records a sequence keeps before it flushes; a multiple
            of 8 and at least num_spec_tokens + 1.
        The other arguments are those of fused_conv1d_kda.

    Returns:
        (new_conv_state, new_recurrent_state, new_replay_state,
        new_replay_gates) and out.
    """
    return launch_fused_conv1d_kda(
        qkv, b, g, conv_state, recurrent_state, conv_weight, conv_bias, a_log,
        dt_bias, query_start_loc, state_indices, distribution, seq_lens,
        read_state_indices, read_offsets, replay_state, replay_gates,
        replay_lens, n_kq=n_kq, n_v=n_v, d_k=d_k, d_v=d_v,
        kernel_size=kernel_size, num_spec_tokens=num_spec_tokens,
        zero_initialize_out=zero_initialize_out,
        compute_precision=compute_precision,
        decode_tile_size=decode_tile_size, mixed_tile_size=mixed_tile_size,
        compute_chunk_size=compute_chunk_size, lower_bound=lower_bound,
        beta_is_activated=beta_is_activated, replay_window=replay_window)


def next_replay_lens(
    replay_lens: jax.Array,
    query_start_loc: jax.Array,
    seq_lens: jax.Array,
    state_indices: jax.Array,
    read_state_indices: jax.Array,
    distribution: jax.Array,
    num_accepted: jax.Array,
    replay_window: int,
) -> jax.Array:
    """Committed records of each sequence after a fused_conv1d_kda_replay
    step that accepted num_accepted[s] of its tokens (query_len without
    speculation); prefill and mixed sequences restart from 0."""
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    replay_len, is_flush = metadata.plan_replay_step(
        replay_lens, query_lens, (seq_lens - query_lens) > 0,
        read_state_indices, state_indices, replay_window)
    base = jnp.where(is_flush, 0, replay_len)
    is_decode = jnp.arange(replay_lens.size) < distribution[0]
    return jnp.where(is_decode, base + num_accepted, 0).astype(jnp.int32)


def launch_fused_conv1d_kda(
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
    distribution: jax.Array,
    seq_lens: jax.Array,
    read_state_indices: jax.Array,
    read_offsets: jax.Array | None,
    replay_state: jax.Array | None,
    replay_gates: jax.Array | None,
    replay_lens: jax.Array | None,
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    num_spec_tokens: int,
    zero_initialize_out: bool,
    compute_precision: jnp.dtype,
    decode_tile_size: int,
    mixed_tile_size: int,
    compute_chunk_size: int,
    lower_bound: float | None,
    beta_is_activated: bool,
    replay_window: int,
) -> tuple[tuple[jax.Array, jax.Array, jax.Array | None, jax.Array | None],
           jax.Array]:
    """fused_conv1d_kda, with a ReplaySSM decode segment when
    replay_window > 0. Returns (conv, recurrent, replay, gates) states and
    out, the last two None without replay."""
    if lower_bound is not None and (not math.isfinite(lower_bound) or
                                    lower_bound >= 0):
        raise ValueError("lower_bound must be finite and negative")
    if qkv.dtype != jnp.bfloat16:
        raise TypeError(f"qkv must be bfloat16, got {qkv.dtype}")
    if g.dtype != jnp.bfloat16:
        raise TypeError(f"g must be bfloat16, got {g.dtype}")
    if qkv.ndim != 2 or qkv.shape[0] == 0:
        raise ValueError("qkv must be a nonempty [tokens, dim] array")
    # Note (david): the chunked math and the conv halo work on whole bf16
    # sublane tiles.
    sublane_rows = compute_kda.BF16_SUBLANE_ROWS
    if compute_chunk_size <= 0 or compute_chunk_size % sublane_rows:
        raise ValueError("compute_chunk_size must be a positive multiple of "
                         f"{sublane_rows}")
    # Note (david): the triangular inverse combines diagonal blocks
    # BLOCKS_PER_LEVEL at a time (4 -> 16 -> 64 -> ...), so every level below
    # the chunk must tile it. Levels 4 and 16 divide any multiple of 16, so the
    # check starts one level above.
    inverse_block_size = sublane_rows * compute_kda.BLOCKS_PER_LEVEL
    while inverse_block_size < compute_chunk_size:
        if compute_chunk_size % inverse_block_size:
            raise ValueError(
                "compute_chunk_size must be divisible by inverse block size "
                f"{inverse_block_size}")
        inverse_block_size *= compute_kda.BLOCKS_PER_LEVEL
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

    # The token-axis DMAs of qkv, g and out move whole SUBLANE_ALIGN-row
    # windows (memory_ref.aligned_window), which stay inside the allocation
    # only if XLA pads the token axis to that grid. It does from SUBLANE_ALIGN
    # tokens on; below, it tiles 1, 2 or 4 rows deep (bf16[2, dim] is
    # T(2,128)(2,1)) and the unchecked windows would read and write past the
    # buffers (on v7x the stray write halted a later program, issue #36). So
    # fewer tokens are padded up and the output is cut back.
    num_tokens = qkv.shape[0]
    token_padding = max(0, config.SUBLANE_ALIGN - num_tokens)
    if token_padding:
        qkv, b, g = (jnp.pad(x, ((0, token_padding), (0, 0)))
                     for x in (qkv, b, g))

    # Note (david): qkv enters the kernel in the caller's [batch, dim] layout
    # and dtype and is cast to f32 on-chip; an f32 [batch, 1, dim] reshape here
    # is a T(8,128) -> T(1,128) retile that XLA materializes in HBM, measured
    # on the gdn kernel at 0.497 ms of gdn_v3's 1.306 ms forward at T = 8192,
    # dim = 6144.
    b = b.astype(jnp.float32)
    # Note (david): XLA lays a serving-sized f32 conv cache out slot-minor
    # ({2,0,1:T(8,128)}), so its [kernel_size - 1, num_slots, dim] view is a
    # bitcast, while a slot-major kernel operand cost a transposing copy of the
    # whole cache around every call (v6e: 67 us at 257 slots, 373 us at 1025).
    conv_state = jnp.swapaxes(conv_state.astype(jnp.float32), 0, 1)

    num_seqs = state_indices.size
    batch_size, dim = qkv.shape
    assert conv_weight.shape == (dim, 1, kernel_size)
    if conv_bias is not None:
        assert conv_bias.shape == (dim, )
    assert g.shape == (batch_size, n_v * d_k), g.shape
    assert a_log.shape == (n_v, ), a_log.shape
    assert dt_bias.size == n_v * d_k, dt_bias.shape
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
    if replay_state is not None:
        # Note (david): the record buffer moves in whole sublane tiles, and a
        # flushed window writes its records from row 0.
        if replay_window <= 0 or replay_window % config.SUBLANE_ALIGN:
            raise ValueError("replay_window must be a positive multiple of "
                             f"{config.SUBLANE_ALIGN}")
        if num_spec_tokens + 1 > replay_window:
            raise ValueError("replay_window must hold a verify window of "
                             f"{num_spec_tokens + 1} tokens")
        layout = replay_layout(n_kq, n_v, d_k, d_v)
        num_slots = recurrent_state.shape[0]
        record_shape = (num_slots, replay_window, layout.record_dim)
        if replay_state.shape != record_shape:
            raise ValueError(f"replay_state must be {record_shape}, got "
                             f"{replay_state.shape}")
        if replay_state.dtype not in (jnp.float32, jnp.bfloat16):
            raise TypeError("replay_state must be float32 or bfloat16, got "
                            f"{replay_state.dtype}")
        gate_shape = (num_slots, replay_window, layout.gate_dim)
        if replay_gates.shape != gate_shape:
            raise ValueError(f"replay_gates must be {gate_shape}, got "
                             f"{replay_gates.shape}")
        if replay_gates.dtype != jnp.float32:
            raise TypeError("replay_gates must be float32, got "
                            f"{replay_gates.dtype}")
        if replay_lens.shape != state_indices.shape:
            raise ValueError("replay_lens must have one entry per sequence")
        checkpoint_shape = (num_slots, *layout.checkpoint_shape)
        if recurrent_state.shape != checkpoint_shape:
            raise ValueError(
                f"a replay checkpoint must be {checkpoint_shape}, got "
                f"{recurrent_state.shape}")
        if recurrent_state.dtype != jnp.int32:
            raise TypeError("a replay checkpoint must be int32, got "
                            f"{recurrent_state.dtype}")

    tpu_info = pltpu.get_tpu_info()
    num_lanes = tpu_info.num_lanes
    mixed_tile_size = min(
        mixed_tile_size,
        pl.cdiv(batch_size, compute_chunk_size) * compute_chunk_size)
    aligned_num_v_heads = pl.cdiv(n_v, num_lanes) * num_lanes

    if replay_window:
        # Note (david): a replay tile reads a checkpoint's hi plane, and its
        # lo plane on a flush, and writes at most one checkpoint per sequence
        # whatever its window: two f32 checkpoints' bytes per sequence and
        # ring slot, plus one for the compiler's temporaries. Only the conv
        # state keeps a checkpoint per window position.
        decode_tile_size = min(decode_tile_size, batch_size)
        state_bytes = n_v * d_k * d_v * F32_BYTES
        record_bytes = replay_window * (
            replay_state.shape[-1] * replay_state.dtype.itemsize +
            replay_gates.shape[-1] * F32_BYTES)
        conv_ckpt_bytes = (kernel_size - 1) * dim * F32_BYTES
        ring_bytes_per_seq = config.NUM_BUFFERS * (
            2 * state_bytes + (num_spec_tokens + 2) * conv_ckpt_bytes +
            2 * record_bytes)
        vmem_limit = int(DEFAULT_VMEM_FRACTION * tpu_info.vmem_capacity_bytes)
        vmem_bytes_per_seq = ring_bytes_per_seq + state_bytes
        if decode_tile_size * vmem_bytes_per_seq > vmem_limit:
            raise ValueError(
                f"decode_tile_size {decode_tile_size} needs "
                f"{decode_tile_size * vmem_bytes_per_seq / 2**20:.1f} MiB of "
                "VMEM for its replay decode tiles, over the "
                f"{vmem_limit / 2**20:.1f} MiB limit; the largest that fits "
                f"is {vmem_limit // vmem_bytes_per_seq}")
    elif num_spec_tokens > 0:
        # Note (david): a verify window keeps one state checkpoint per position
        # and sequence in VMEM, multiplying the per-sequence footprint by the
        # window size. The tile shrinks so the double-buffered windows fit in
        # about half the scoped VMEM budget; weights, activation scratch and
        # compiler temporaries take the rest.
        verify_window_size = num_spec_tokens + 1
        recurrent_ckpt_bytes = n_v * d_k * d_v * F32_BYTES
        conv_ckpt_bytes = (kernel_size - 1) * dim * F32_BYTES
        activation_bytes = (dim * F32_BYTES +
                            aligned_num_v_heads * F32_BYTES +
                            n_v * d_k * BF16_BYTES + n_v * d_v * BF16_BYTES)
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
        # Note (david): every state ring slot holds each sequence's initial
        # state and its one checkpoint. Where the rings cannot fit the scoped
        # VMEM limit (v7x has half of v6e's VMEM: 4 sequences of 48 value heads
        # at head_dim 128), the tile shrinks until they do, with one more state
        # per sequence left for the compiler's temporaries.
        state_bytes = n_v * d_k * d_v * recurrent_state.dtype.itemsize
        ring_bytes_per_seq = config.NUM_BUFFERS * 2 * (
            state_bytes + (kernel_size - 1) * dim * F32_BYTES)
        vmem_limit = int(DEFAULT_VMEM_FRACTION * tpu_info.vmem_capacity_bytes)
        if decode_tile_size * ring_bytes_per_seq > vmem_limit:
            decode_tile_size = max(
                1, vmem_limit // (ring_bytes_per_seq + state_bytes))

    # Note (david): b keeps the compact [batch, 1, heads] layout: it is tiny,
    # and its untiled leading token axis keeps ragged DMAs offset-free. g is as
    # wide as k and stays in the caller's [batch, n_v * d_k] layout like qkv.
    num_v_padding = aligned_num_v_heads - n_v
    b = jnp.pad(b, ((0, 0), (0, num_v_padding))).reshape(batch_size, 1, -1)

    conv_weight = conv_weight.swapaxes(0, 2).reshape(kernel_size,
                                                     dim).astype(jnp.float32)
    conv_bias = None if conv_bias is None else conv_bias.astype(jnp.float32)
    # Note (david): the gate parameters take the gate's [n_v, 1, d_k] layout,
    # one lane-dense row per head, so the kernel broadcasts them over tokens
    # for free.
    weights = memory_ref.WeightRefs(
        conv=memory_ref.ConvWeightsRef(weight=conv_weight, bias=conv_bias),
        kda=memory_ref.KDAWeightsRef(
            a_log=jnp.broadcast_to(
                a_log.astype(jnp.float32)[:, None, None], (n_v, 1, d_k)),
            dt_bias=dt_bias.astype(jnp.float32).reshape(n_v, 1, d_k)),
    )

    smem_spec = pl.BlockSpec(memory_space=pltpu.SMEM)
    vmem_spec = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    weights_spec = jax.tree.map(lambda _: vmem_spec, weights)

    def _call_kernel(
        in_conv_state: jax.Array,
        in_recurrent_state: jax.Array,
        aliased_out: jax.Array | None,
        mode: config.KDAMode,
        in_replay_state: jax.Array | None,
        in_replay_gates: jax.Array | None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array | None,
               jax.Array | None]:
        if mode == config.KDAMode.PER_SEQ:
            tile_size = mixed_tile_size
            window_size = 1
            kernel_compute_chunk = compute_chunk_size
            kernel_replay_window = 0
        else:
            tile_size = decode_tile_size
            window_size = num_spec_tokens + 1
            kernel_replay_window = replay_window
            # Note (david): a BATCHED tile holds each sequence's whole verify
            # window, which is one math chunk.
            kernel_compute_chunk = window_size

        cfg = config.KDAConfig(
            mode=mode,
            batch_size=batch_size,
            kernel_size=kernel_size,
            tile_size=tile_size,
            compute_chunk_size=kernel_compute_chunk,
            window_size=window_size,
            lower_bound=lower_bound,
            beta_is_activated=beta_is_activated,
            replay_window=kernel_replay_window,
            split_checkpoint=replay_window > 0,
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
                replay_record=(jnp.float32 if in_replay_state is None else
                               in_replay_state.dtype),
            ),
        )

        # Note (david): the metadata is rebuilt for every layer; the compiler
        # CSEs the copies.
        if mode == config.KDAMode.PER_SEQ:
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
                replay_lens=replay_lens,
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
        if mode != config.KDAMode.PER_SEQ:
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
        if cfg.is_replay:
            kernel_name_suffix += f"_replay{cfg.replay_window}"

        kernel_metadata = {}
        for path, leaf in jax.tree_util.tree_leaves_with_path(
                dataclasses.asdict(cfg)):
            key = jax.tree_util.keystr(path, simple=True, separator=".")
            if isinstance(leaf, str | int | float):
                kernel_metadata[key] = leaf
            else:
                kernel_metadata[key] = str(leaf)

        # Note (david): unpinned, XLA moves any state that fits its VMEM
        # budget there and back around each call, the whole buffer however
        # few slots the call touches (v6e, 65 slots, 32 decoding sequences:
        # 0.47 ms of a 9.55 ms step on the bf16 records of 24 GDN layers),
        # and a small conv state placed in VMEM makes the one-slot DMAs
        # through the slot-minor view move the wrong rows (jax 0.11.2).
        def _pin(state):
            return pltpu.with_memory_space_constraint(state, pltpu.HBM)

        operands = (seq_metadata, qkv, b, g, _pin(in_conv_state),
                    _pin(in_recurrent_state), aliased_out, weights, pad_start)
        in_specs = (metadata_spec, hbm_spec, hbm_spec, hbm_spec, hbm_spec,
                    hbm_spec, aliased_out_spec, weights_spec, pad_spec)
        out_shapes = (out_shape, in_conv_state, in_recurrent_state)
        out_specs = (hbm_spec, hbm_spec, hbm_spec)
        # Note (david): the replay records and gates follow every other
        # operand, so their flat input index counts the leaves before them.
        for extra_state in (in_replay_state, in_replay_gates):
            if extra_state is None:
                in_specs += (None, )
            else:
                input_output_aliases = input_output_aliases | {
                    len(jax.tree.leaves(operands)): len(out_shapes)
                }
                in_specs += (hbm_spec, )
                out_shapes += (extra_state, )
                out_specs += (hbm_spec, )
                extra_state = _pin(extra_state)
            operands += (extra_state, )

        results = pl.pallas_call(
            functools.partial(outer_kernel, cfg=cfg),
            out_shape=out_shapes,
            in_specs=in_specs,
            out_specs=out_specs,
            scratch_shapes=cfg.get_scratch_shape_dict(),
            input_output_aliases=input_output_aliases,
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True,
                vmem_limit_bytes=int(vmem_fraction *
                                     tpu_info.vmem_capacity_bytes),
            ),
            name=f"fused_conv1d_kda_{mode.value}{kernel_name_suffix}",
            metadata=kernel_metadata,
        )(*operands)
        if in_replay_state is None:
            return *results, None, None
        else:
            return tuple(results)

    (out, conv_state_out, recurrent_state_out, replay_state_out,
     replay_gates_out) = _call_kernel(conv_state, recurrent_state, None,
                                      config.KDAMode.BATCHED,
                                      replay_state, replay_gates)
    out, conv_state_out, recurrent_state_out, _, _ = _call_kernel(
        conv_state_out, recurrent_state_out, out, config.KDAMode.PER_SEQ,
        None, None)

    return (jnp.swapaxes(conv_state_out, 0, 1).astype(conv_out_dtype),
            recurrent_state_out.astype(recurrent_out_dtype), replay_state_out,
            replay_gates_out), out[:num_tokens]
