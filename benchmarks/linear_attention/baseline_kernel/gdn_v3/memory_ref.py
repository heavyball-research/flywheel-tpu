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
from typing import Self

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from gdn_v3 import config


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ConvWeightsRef:
    weight: jax.Array
    bias: jax.Array | None = None


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class GDNWeightsRef:
    a_log: jax.Array
    dt_bias: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class WeightRefs:
    conv: ConvWeightsRef
    gdn: GDNWeightsRef


@dataclasses.dataclass(frozen=True)
class PackedPIdRecord:
    """One p_id's record in the SMEM array of structs: [r_base, word].

    word packs is_first_tile (bit 0), is_last_tile (bit 1), r_size (bits 2-15)
    and s_idx (bits 16-31).
    """

    # Note (david): the four small fields share one word to save SMEM. Each
    # field masks after shifting, which also clears the sign bits that >>
    # extends on the signed int32 word.
    STRUCT_SIZE = 2
    WORD_BITS = 32
    FIRST_TILE_SHIFT = 0
    LAST_TILE_SHIFT = 1
    R_SIZE_SHIFT = 2
    S_IDX_SHIFT = 16
    FLAG_MASK = 1
    R_SIZE_MASK = (1 << (S_IDX_SHIFT - R_SIZE_SHIFT)) - 1
    S_IDX_MASK = (1 << (WORD_BITS - S_IDX_SHIFT)) - 1
    MAX_SEQS = S_IDX_MASK + 1

    data: jax.Array
    pos: jax.Array

    @property
    def r_base(self) -> jax.Array:
        return self.data[self.pos]

    @property
    def word(self) -> jax.Array:
        return self.data[self.pos + 1]

    @property
    def s_idx(self) -> jax.Array:
        return (self.word >> self.S_IDX_SHIFT) & self.S_IDX_MASK

    @property
    def r_size(self) -> jax.Array:
        return (self.word >> self.R_SIZE_SHIFT) & self.R_SIZE_MASK

    @property
    def is_first_tile(self) -> jax.Array:
        return (self.word & self.FLAG_MASK) != 0

    @property
    def is_last_tile(self) -> jax.Array:
        return ((self.word >> self.LAST_TILE_SHIFT) & self.FLAG_MASK) != 0

    @classmethod
    def pack(
        cls,
        s_idx: jax.Array,
        r_size: jax.Array,
        is_first_tile: jax.Array,
        is_last_tile: jax.Array,
    ) -> jax.Array:
        s_idx = s_idx.reshape(-1).astype(jnp.int32)
        # Note (david): PER_SEQ pads unused slots with large negative sizes;
        # shifting one of those left would set the high bits and overwrite
        # s_idx.
        r_size = jnp.maximum(r_size.reshape(-1).astype(jnp.int32), 0)
        is_first_tile = is_first_tile.reshape(-1).astype(jnp.int32)
        is_last_tile = is_last_tile.reshape(-1).astype(jnp.int32)
        word = s_idx << cls.S_IDX_SHIFT
        word |= r_size << cls.R_SIZE_SHIFT
        word |= is_last_tile << cls.LAST_TILE_SHIFT
        word |= is_first_tile << cls.FIRST_TILE_SHIFT
        return word


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MetadataRef:
    num_tiles: jax.Array
    records: jax.Array
    s_idx_has_initial_state: jax.Array
    s_idx_to_state_indices: jax.Array
    # Note (david): speculative decoding reads the initial state from
    # s_idx_to_read_indices[s] + s_idx_to_read_offset[s], the checkpoint of the
    # last accepted token. The offset is 0 without speculative decoding.
    s_idx_to_read_offset: jax.Array
    # Note (david): equals s_idx_to_state_indices unless mamba prefix caching
    # (align mode) resumes a sequence from the cached state block of the
    # previous block boundary while it checkpoints into a new block.
    s_idx_to_read_indices: jax.Array
    # Note (david): only the strides of shape are used, so its leading dim is
    # 1.
    shape: tuple[int, ...] = dataclasses.field(metadata=dict(static=True))

    def get_record(self, p_id: jax.Array, idx: int) -> PackedPIdRecord:
        row_stride, col_stride = pl.strides_from_shape(self.shape)
        record_idx = row_stride * p_id + col_stride * idx
        return PackedPIdRecord(self.records,
                               record_idx * PackedPIdRecord.STRUCT_SIZE)

    @classmethod
    def create(
        cls,
        cfg: config.GDNConfig,
        num_tiles: jax.Array,
        p_id_to_s_idx: jax.Array,
        p_id_to_r_base: jax.Array,
        p_id_to_r_size: jax.Array,
        p_id_is_first_tile: jax.Array,
        p_id_is_last_tile: jax.Array,
        s_idx_has_initial_state: jax.Array,
        s_idx_to_state_indices: jax.Array,
        s_idx_to_read_offset: jax.Array,
        s_idx_to_read_indices: jax.Array,
    ) -> Self:
        assert s_idx_has_initial_state.shape[0] <= PackedPIdRecord.MAX_SEQS, (
            f"Number of sequences ({s_idx_has_initial_state.shape[0]}) exceeds"
            f" PackedPIdRecord limit ({PackedPIdRecord.MAX_SEQS}).")
        assert cfg.tile_size <= PackedPIdRecord.R_SIZE_MASK, (
            f"Tile size ({cfg.tile_size}) exceeds PackedPIdRecord limit"
            f" ({PackedPIdRecord.R_SIZE_MASK}).")

        r_base = p_id_to_r_base.reshape(-1).astype(jnp.int32)
        word = PackedPIdRecord.pack(p_id_to_s_idx, p_id_to_r_size,
                                    p_id_is_first_tile, p_id_is_last_tile)
        # Note (david): every tile reads all seq_tile_size of its records, so
        # the last tile reads up to seq_tile_size - 1 records past the valid
        # ones. Zero padding makes those reads hit s_idx 0, r_size 0 and both
        # tile flags false, which issues no DMA, instead of whatever SMEM
        # follows.
        num_pad_records = -r_base.shape[0] % cfg.seq_tile_size
        r_base = jnp.pad(r_base, (0, num_pad_records))
        word = jnp.pad(word, (0, num_pad_records))
        records = jnp.stack([r_base, word], axis=-1).reshape(-1)

        return cls(
            num_tiles=num_tiles,
            records=records,
            s_idx_has_initial_state=s_idx_has_initial_state,
            s_idx_to_state_indices=s_idx_to_state_indices,
            s_idx_to_read_offset=s_idx_to_read_offset,
            s_idx_to_read_indices=s_idx_to_read_indices,
            shape=(1, cfg.seq_tile_size),
        )

    def __len__(self) -> int:
        return len(jax.tree_util.tree_leaves(self))


def wait_row_dma(sem: jax.Array, vmem_ref: jax.Ref,
                 num_rows: jax.Array) -> None:
    # Note (david): the self-copy descriptor is never executed, it only waits
    # for the bytes the matching start issued. With bounds checks disabled it
    # may nominally exceed the window row.
    wait_ref = vmem_ref.at[0, pl.ds(0, num_rows)]
    pltpu.make_async_copy(wait_ref, wait_ref, sem).wait()


@dataclasses.dataclass(frozen=True, kw_only=True)
class BaseBufferedRef(pltpu.BufferedRef):

    cfg: config.GDNConfig = dataclasses.field(metadata=dict(static=True))
    # Note (david): metadata_ref is static although it is a ref, because its
    # memory is allocated outside the kernel and this only points to it.
    metadata_ref: MetadataRef = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def create(
        cls,
        spec: pl.BlockSpec,
        dtype_or_type: jax.Array,
        buffer_type: pltpu.BufferType,
        buffer_count: int,
        use_lookahead: bool,
        cfg: config.GDNConfig,
        metadata_ref: MetadataRef,
    ) -> Self:
        standard_ref = pltpu.BufferedRef.create(
            spec=spec,
            dtype_or_type=dtype_or_type,
            buffer_type=buffer_type,
            buffer_count=buffer_count,
            grid_rank=1,
            use_lookahead=use_lookahead,
        )
        # Note (david): BufferedRef.create only builds the base class, so its
        # fields are copied over generically; listing them would hard-code
        # private JAX field names.
        return cls(
            cfg=cfg,
            metadata_ref=metadata_ref,
            **{
                field.name: getattr(standard_ref, field.name)
                for field in dataclasses.fields(pltpu.BufferedRef)
            },
        )

    def recv_slot_refs(self, slot: jax.Array) -> tuple[jax.Array, jax.Ref]:
        assert self.sem_recvs is not None
        assert self.window_ref is not None
        return self.sem_recvs.at[slot], self.window_ref.at[slot]

    def send_slot_refs(self, slot: jax.Array) -> tuple[jax.Array, jax.Ref]:
        assert self.sem_sends is not None
        assert self.window_ref is not None
        return self.sem_sends.at[slot], self.window_ref.at[slot]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class InBufferedRef(BaseBufferedRef):

    def copy_in(self, src_ref: jax.Ref,
                grid_indices: tuple[int | jax.Array, ...]) -> None:
        sem, vmem_ref = self.recv_slot_refs(self.current_copy_in_slot)
        for idx in range(self.cfg.seq_tile_size):
            record = self.metadata_ref.get_record(grid_indices[0], idx)
            r_base = record.r_base
            r_size = record.r_size
            pltpu.make_async_copy(
                src_ref.at[pl.ds(r_base, r_size)],
                vmem_ref.at[idx, pl.ds(0, r_size)],
                sem,
            ).start()

    def wait_in(self, src_ref: jax.Ref,
                grid_indices: tuple[int | jax.Array, ...]) -> None:
        sem, vmem_ref = self.recv_slot_refs(self.current_wait_in_slot)
        num_rows = 0
        for idx in range(self.cfg.seq_tile_size):
            num_rows += self.metadata_ref.get_record(grid_indices[0],
                                                     idx).r_size
        wait_row_dma(sem, vmem_ref, num_rows)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class OutBufferedRef(BaseBufferedRef):

    def copy_out(self, dst_ref: jax.Ref,
                 grid_indices: tuple[int | jax.Array, ...]) -> None:
        sem, vmem_ref = self.send_slot_refs(self.current_copy_out_slot)
        for idx in range(self.cfg.seq_tile_size):
            record = self.metadata_ref.get_record(grid_indices[0], idx)
            r_base = record.r_base
            r_size = record.r_size
            pltpu.make_async_copy(
                vmem_ref.at[idx, pl.ds(0, r_size)],
                dst_ref.at[pl.ds(r_base, r_size)],
                sem,
            ).start()

    def wait_out(self, dst_ref: jax.Ref,
                 grid_indices: tuple[int | jax.Array, ...]) -> None:
        sem, vmem_ref = self.send_slot_refs(self.current_wait_out_slot)
        num_rows = 0
        for idx in range(self.cfg.seq_tile_size):
            num_rows += self.metadata_ref.get_record(grid_indices[0],
                                                     idx).r_size
        wait_row_dma(sem, vmem_ref, num_rows)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class StateBufferedRef(BaseBufferedRef):
    """Input/output buffered ref for per-sequence conv / recurrent state.

    The VMEM window holds one state per window position, [seq_tile_size,
    window_size, *state_shape]. The initial state is read from
    read_indices[s] + read_offset[s] into position 0 of the sequence's row,
    and the first min(r_size, window_size) checkpoints are written back to
    state_indices[s] onwards.
    """

    def copy_in(self, src_ref: jax.Ref,
                grid_indices: tuple[int | jax.Array, ...]) -> None:
        sem, vmem_ref = self.recv_slot_refs(self.current_copy_in_slot)
        for idx in range(self.cfg.seq_tile_size):
            record = self.metadata_ref.get_record(grid_indices[0], idx)
            is_first_tile = record.is_first_tile
            s_idx = record.s_idx
            read_slot = self.metadata_ref.s_idx_to_read_indices[s_idx]
            has_initial_state = self.metadata_ref.s_idx_has_initial_state[
                s_idx]
            should_read = jnp.logical_and(is_first_tile, has_initial_state)
            num_states = jnp.where(should_read, 1, 0)
            # Note (david): resume from the checkpoint of the last accepted
            # token.
            read_slot += self.metadata_ref.s_idx_to_read_offset[s_idx]

            pltpu.make_async_copy(
                src_ref.at[pl.ds(read_slot, num_states)],
                vmem_ref.at[idx, pl.ds(0, num_states)],
                sem,
            ).start()

    def wait_in(self, src_ref: jax.Ref,
                grid_indices: tuple[int | jax.Array, ...]) -> None:
        sem, vmem_ref = self.recv_slot_refs(self.current_wait_in_slot)
        num_states = 0
        for idx in range(self.cfg.seq_tile_size):
            record = self.metadata_ref.get_record(grid_indices[0], idx)
            is_first_tile = record.is_first_tile
            has_initial_state = self.metadata_ref.s_idx_has_initial_state[
                record.s_idx]
            should_read = jnp.logical_and(is_first_tile, has_initial_state)
            num_states += jnp.where(should_read, 1, 0)
        wait_row_dma(sem, vmem_ref, num_states)

    def copy_out(self, dst_ref: jax.Ref,
                 grid_indices: tuple[int | jax.Array, ...]) -> None:
        sem, vmem_ref = self.send_slot_refs(self.current_copy_out_slot)
        for idx in range(self.cfg.seq_tile_size):
            record = self.metadata_ref.get_record(grid_indices[0], idx)
            is_last_tile = record.is_last_tile
            s_idx = record.s_idx
            r_size = record.r_size
            write_slot = self.metadata_ref.s_idx_to_state_indices[s_idx]
            # Note (david): r_size never exceeds window_size for windowed
            # sequences; the clamp is for PER_SEQ tiles, which hold many tokens
            # but keep only the final state.
            num_ckpts = jnp.minimum(r_size, self.cfg.window_size)
            num_states = jnp.where(is_last_tile, num_ckpts, 0)

            pltpu.make_async_copy(
                vmem_ref.at[idx, pl.ds(0, num_states)],
                dst_ref.at[pl.ds(write_slot, num_states)],
                sem,
            ).start()

    def wait_out(self, dst_ref: jax.Ref,
                 grid_indices: tuple[int | jax.Array, ...]) -> None:
        sem, vmem_ref = self.send_slot_refs(self.current_wait_out_slot)
        num_states = 0
        for idx in range(self.cfg.seq_tile_size):
            record = self.metadata_ref.get_record(grid_indices[0], idx)
            is_last_tile = record.is_last_tile
            num_ckpts = jnp.minimum(record.r_size, self.cfg.window_size)
            num_states += jnp.where(is_last_tile, num_ckpts, 0)
        wait_row_dma(sem, vmem_ref, num_states)


def create_allocs(
    metadata_ref: MetadataRef,
    qkv_ref: jax.Array,
    b_ref: jax.Array,
    a_ref: jax.Array,
    out_ref: jax.Array,
    conv_state_ref: jax.Array,
    recurrent_state_ref: jax.Array,
    cfg: config.GDNConfig,
) -> tuple[
        InBufferedRef,
        InBufferedRef,
        InBufferedRef,
        StateBufferedRef,
        StateBufferedRef,
        OutBufferedRef,
]:
    qkv_shape = (cfg.seq_tile_size, cfg.chunk_size, 1, cfg.dim_size)
    ba_shape = (cfg.seq_tile_size, cfg.chunk_size, 1, cfg.aligned_num_v_heads)
    out_shape = (
        cfg.seq_tile_size,
        cfg.chunk_size,
        cfg.num_v_heads,
        cfg.v_head_dim,
    )
    conv_shape = (cfg.seq_tile_size, cfg.window_size, cfg.prev_kernel_size, 1,
                  cfg.dim_size)
    recurrent_shape = (
        cfg.seq_tile_size,
        cfg.window_size,
        cfg.num_v_heads,
        cfg.kq_head_dim,
        cfg.v_head_dim,
    )

    pipeline_mode = pl.Buffered(buffer_count=cfg.num_buffers,
                                use_lookahead=False)

    block_spec_partial = functools.partial(
        pl.BlockSpec,
        memory_space=pltpu.VMEM,
        index_map=lambda i: (i, ),
        pipeline_mode=pipeline_mode,
    )

    qkv_spec = block_spec_partial(block_shape=qkv_shape)
    ba_spec = block_spec_partial(block_shape=ba_shape)
    in_buffered_partial = functools.partial(
        InBufferedRef.input,
        buffer_count=pipeline_mode.buffer_count,
        use_lookahead=pipeline_mode.use_lookahead,
        cfg=cfg,
        metadata_ref=metadata_ref,
    )
    qkv_alloc = in_buffered_partial(spec=qkv_spec, dtype_or_type=qkv_ref)
    b_alloc = in_buffered_partial(spec=ba_spec, dtype_or_type=b_ref)
    a_alloc = in_buffered_partial(spec=ba_spec, dtype_or_type=a_ref)

    out_alloc = OutBufferedRef.output(
        spec=block_spec_partial(block_shape=out_shape),
        dtype_or_type=out_ref,
        buffer_count=pipeline_mode.buffer_count,
        use_lookahead=pipeline_mode.use_lookahead,
        cfg=cfg,
        metadata_ref=metadata_ref,
    )

    conv_spec = block_spec_partial(block_shape=conv_shape)
    recurrent_spec = block_spec_partial(block_shape=recurrent_shape)
    state_buffered_partial = functools.partial(
        StateBufferedRef.input_output,
        buffer_count=pipeline_mode.buffer_count,
        use_lookahead=pipeline_mode.use_lookahead,
        cfg=cfg,
        metadata_ref=metadata_ref,
    )
    conv_alloc = state_buffered_partial(spec=conv_spec,
                                        dtype_or_type=conv_state_ref)
    recurrent_alloc = state_buffered_partial(spec=recurrent_spec,
                                             dtype_or_type=recurrent_state_ref)

    return qkv_alloc, b_alloc, a_alloc, conv_alloc, recurrent_alloc, out_alloc
