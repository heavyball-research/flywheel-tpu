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
"""Weight and metadata refs of the kernel and its HBM <-> VMEM DMA helpers."""

import dataclasses
from typing import Self

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from . import config


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ConvWeightsRef:
    weight: jax.Array
    bias: jax.Array | None = None


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class KDAWeightsRef:
    a_log: jax.Array
    dt_bias: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class WeightRefs:
    conv: ConvWeightsRef
    kda: KDAWeightsRef


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

    records: jax.Array
    offset: int | jax.Array

    # Note (david): each field reads one dynamically indexed word rather than a
    # slice, since JAX cannot slice a range at a traced index.
    @property
    def r_base(self) -> jax.Array:
        return self.records[self.offset]

    @property
    def word(self) -> jax.Array:
        return self.records[self.offset + 1]

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
    records_per_tile: int = dataclasses.field(metadata={"static": True})

    def get_record(self, p_id: int | jax.Array, idx: int) -> PackedPIdRecord:
        record_idx = p_id * self.records_per_tile + idx
        return PackedPIdRecord(
            records=self.records,
            offset=record_idx * PackedPIdRecord.STRUCT_SIZE)

    @classmethod
    def create(
        cls,
        cfg: config.KDAConfig,
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
        s_idx = p_id_to_s_idx.reshape(-1).astype(jnp.int32)
        # Note (david): PER_SEQ pads unused slots with large negative sizes;
        # shifting one of those left would set the high bits and overwrite
        # s_idx.
        r_size = jnp.maximum(p_id_to_r_size.reshape(-1).astype(jnp.int32), 0)
        is_first_tile = p_id_is_first_tile.reshape(-1).astype(jnp.int32)
        is_last_tile = p_id_is_last_tile.reshape(-1).astype(jnp.int32)
        word = ((s_idx << PackedPIdRecord.S_IDX_SHIFT)
                | (r_size << PackedPIdRecord.R_SIZE_SHIFT)
                | (is_last_tile << PackedPIdRecord.LAST_TILE_SHIFT)
                | (is_first_tile << PackedPIdRecord.FIRST_TILE_SHIFT))
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
            records_per_tile=cfg.seq_tile_size,
        )

    def __len__(self) -> int:
        return len(jax.tree_util.tree_leaves(self))


def aligned_window(r_base: jax.Array,
                   r_size: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Base and size of the smallest SUBLANE_ALIGN-aligned row window covering
    [r_base, r_base + r_size)."""
    # Note (david): the token axis of a native [batch, dim] array is its tiled
    # second-minor dimension, so Mosaic rejects any DMA whose offset or size it
    # cannot prove to be a sublane-tile multiple, which a ragged per-sequence
    # transfer never is. The window never leaves the tile-padded allocation: it
    # ends at ceil((r_base + r_size) / align) * align
    # <= ceil(batch / align) * align.
    align = config.SUBLANE_ALIGN
    delta = r_base & (align - 1)
    window_base = pl.multiple_of(r_base - delta, align)
    window_rows = pl.multiple_of(pl.cdiv(delta + r_size, align) * align, align)
    return window_base, window_rows


def wait_row_dma(buf_ref: jax.Ref, num_rows: jax.Array,
                 sem: jax.Ref) -> None:
    # Note (david): a DMA semaphore counts bytes, so one wait sized to the sum
    # of a tile's transfers covers all of them. The self-copy descriptor is
    # never executed; with bounds checks disabled it may nominally exceed the
    # window row.
    wait_ref = buf_ref.at[0, pl.ds(0, num_rows)]
    pltpu.make_async_copy(wait_ref, wait_ref, sem).wait()


def qkv_copy_refs(
    hbm_ref: jax.Ref, buf_ref: jax.Ref, p_id: int | jax.Array, idx: int,
    metadata_ref: MetadataRef
) -> tuple[jax.Ref, jax.Ref]:
    """Source and destination of the DMA that fetches record idx's rows of a
    native [batch, dim] input, widened to their aligned window."""
    record = metadata_ref.get_record(p_id, idx)
    window_base, window_rows = aligned_window(record.r_base, record.r_size)
    return (hbm_ref.at[pl.ds(window_base, window_rows)],
            buf_ref.at[idx, pl.ds(0, window_rows)])


def start_qkv_in(hbm_ref: jax.Ref, buf_ref: jax.Ref, sem: jax.Ref,
                 p_id: int | jax.Array, metadata_ref: MetadataRef,
                 cfg: config.KDAConfig) -> None:
    """Fetch each record's rows of a native [batch, dim] input, widened to
    their aligned window.

    A record's first row lands at row r_base % SUBLANE_ALIGN of its block
    rather than at row 0, and the rows past r_size are not the record's.
    """
    for idx in range(cfg.seq_tile_size):
        pltpu.make_async_copy(
            *qkv_copy_refs(hbm_ref, buf_ref, p_id, idx, metadata_ref),
            sem).start()


def wait_qkv_in(hbm_ref: jax.Ref, buf_ref: jax.Ref, sem: jax.Ref,
                p_id: int | jax.Array, metadata_ref: MetadataRef,
                cfg: config.KDAConfig) -> None:
    # Each copy is waited on itself. These rows are the block's tiled axis,
    # and where Mosaic tiles the block 16 rows deep (v7x, a verify window's
    # 16-row block) wait_row_dma's one self-copy of the summed rows left the
    # semaphore nonzero at kernel exit: three 8-row copies against a 24-row
    # self-copy. A copy's own descriptor takes back exactly what it added.
    for idx in range(cfg.seq_tile_size):
        pltpu.make_async_copy(
            *qkv_copy_refs(hbm_ref, buf_ref, p_id, idx, metadata_ref),
            sem).wait()


def start_compact_in(hbm_ref: jax.Ref, buf_ref: jax.Ref, sem: jax.Ref,
                     p_id: int | jax.Array, metadata_ref: MetadataRef,
                     cfg: config.KDAConfig) -> None:
    """Fetch each record's rows of a compact [batch, 1, heads] input (b)."""
    for idx in range(cfg.seq_tile_size):
        record = metadata_ref.get_record(p_id, idx)
        # Note (david): the compact token axis is untiled, so the window needs
        # no tail rounding and never leaves the batch. PER_SEQ starts it at the
        # qkv block's aligned row, so row r of b gates block row r.
        if cfg.mode == config.KDAMode.PER_SEQ:
            delta = record.r_base & (config.SUBLANE_ALIGN - 1)
            window_base = record.r_base - delta
            window_rows = delta + record.r_size
        else:
            window_base = record.r_base
            window_rows = record.r_size
        pltpu.make_async_copy(
            hbm_ref.at[pl.ds(window_base, window_rows)],
            buf_ref.at[idx, pl.ds(0, window_rows)],
            sem,
        ).start()


def wait_compact_in(buf_ref: jax.Ref, sem: jax.Ref, p_id: int | jax.Array,
                    metadata_ref: MetadataRef, cfg: config.KDAConfig) -> None:
    """Wait for a tile's compact transfers (b)."""
    num_rows = 0
    for idx in range(cfg.seq_tile_size):
        record = metadata_ref.get_record(p_id, idx)
        if cfg.mode == config.KDAMode.PER_SEQ:
            delta = record.r_base & (config.SUBLANE_ALIGN - 1)
            num_rows += delta + record.r_size
        else:
            num_rows += record.r_size
    wait_row_dma(buf_ref, num_rows, sem)


def out_window(
    p_id: int | jax.Array, metadata_ref: MetadataRef, cfg: config.KDAConfig
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """The tile's aligned output window: (hbm base, dma rows, delta, used rows).

    A tile's records are always consecutive batch rows (PER_SEQ: successive
    tiles of one sequence; BATCHED: successive sequences, padded records with
    r_size 0), so they form one window starting at the aligned first row, with
    the tile's own first row at delta. The head rows [base, base + delta)
    belong to the previous tile and are blended in from its still-resident ring
    slot; the rows past used are zeroed in the stage and the next tile's window
    overwrites them, since output DMAs issue in tile order on one stream.
    """
    first_row = metadata_ref.get_record(p_id, 0).r_base
    tile_rows = 0
    for idx in range(cfg.seq_tile_size):
        tile_rows += metadata_ref.get_record(p_id, idx).r_size
    window_base, window_rows = aligned_window(first_row, tile_rows)
    delta = first_row & (config.SUBLANE_ALIGN - 1)
    return window_base, window_rows, delta, delta + tile_rows


def out_copy_refs(
    hbm_ref: jax.Ref, buf_ref: jax.Ref, p_id: int | jax.Array,
    metadata_ref: MetadataRef, cfg: config.KDAConfig
) -> tuple[jax.Ref, jax.Ref]:
    """Source and destination of the one DMA that writes the tile's whole
    aligned output window."""
    window_base, window_rows, _, _ = out_window(p_id, metadata_ref, cfg)
    return (buf_ref.at[pl.ds(0, window_rows)],
            hbm_ref.at[pl.ds(window_base, window_rows)])


def start_out(hbm_ref: jax.Ref, buf_ref: jax.Ref, sem: jax.Ref,
              p_id: int | jax.Array, metadata_ref: MetadataRef,
              cfg: config.KDAConfig) -> None:
    pltpu.make_async_copy(
        *out_copy_refs(hbm_ref, buf_ref, p_id, metadata_ref, cfg),
        sem).start()


def wait_out(hbm_ref: jax.Ref, buf_ref: jax.Ref, sem: jax.Ref,
             p_id: int | jax.Array, metadata_ref: MetadataRef,
             cfg: config.KDAConfig) -> None:
    # The tile's one output DMA is waited on itself, for the reason
    # wait_qkv_in gives: the stage's rows are its tiled axis.
    pltpu.make_async_copy(
        *out_copy_refs(hbm_ref, buf_ref, p_id, metadata_ref, cfg),
        sem).wait()


def start_state_in(hbm_ref: jax.Ref, buf_ref: jax.Ref, sem: jax.Ref,
                   p_id: int | jax.Array, metadata_ref: MetadataRef,
                   cfg: config.KDAConfig) -> None:
    """Fetch each sequence's initial checkpoint into window position 0.

    Only a sequence's first tile reads, and only if it has an initial state.
    """
    for idx in range(cfg.seq_tile_size):
        record = metadata_ref.get_record(p_id, idx)
        s_idx = record.s_idx
        read_slot = (metadata_ref.s_idx_to_read_indices[s_idx] +
                     metadata_ref.s_idx_to_read_offset[s_idx])
        has_initial_state = metadata_ref.s_idx_has_initial_state[s_idx]
        should_read = jnp.logical_and(record.is_first_tile, has_initial_state)
        num_states = jnp.where(should_read, 1, 0)
        pltpu.make_async_copy(
            hbm_ref.at[pl.ds(read_slot, num_states)],
            buf_ref.at[idx, pl.ds(0, num_states)],
            sem,
        ).start()


def wait_state_in(buf_ref: jax.Ref, sem: jax.Ref, p_id: int | jax.Array,
                  metadata_ref: MetadataRef, cfg: config.KDAConfig) -> None:
    num_states = 0
    for idx in range(cfg.seq_tile_size):
        record = metadata_ref.get_record(p_id, idx)
        has_initial_state = metadata_ref.s_idx_has_initial_state[record.s_idx]
        should_read = jnp.logical_and(record.is_first_tile, has_initial_state)
        num_states += jnp.where(should_read, 1, 0)
    wait_row_dma(buf_ref, num_states, sem)


def start_state_out(hbm_ref: jax.Ref, buf_ref: jax.Ref, sem: jax.Ref,
                    p_id: int | jax.Array, metadata_ref: MetadataRef,
                    cfg: config.KDAConfig) -> None:
    """On a sequence's last tile, write one checkpoint per valid window
    position from its base slot state_indices[s] on."""
    for idx in range(cfg.seq_tile_size):
        record = metadata_ref.get_record(p_id, idx)
        write_slot = metadata_ref.s_idx_to_state_indices[record.s_idx]
        # Note (david): r_size never exceeds window_size for windowed
        # sequences; the clamp is for PER_SEQ tiles, which hold many tokens but
        # keep only the final state.
        num_ckpts = jnp.minimum(record.r_size, cfg.window_size)
        num_states = jnp.where(record.is_last_tile, num_ckpts, 0)
        pltpu.make_async_copy(
            buf_ref.at[idx, pl.ds(0, num_states)],
            hbm_ref.at[pl.ds(write_slot, num_states)],
            sem,
        ).start()


def wait_state_out(buf_ref: jax.Ref, sem: jax.Ref, p_id: int | jax.Array,
                   metadata_ref: MetadataRef, cfg: config.KDAConfig) -> None:
    num_states = 0
    for idx in range(cfg.seq_tile_size):
        record = metadata_ref.get_record(p_id, idx)
        num_ckpts = jnp.minimum(record.r_size, cfg.window_size)
        num_states += jnp.where(record.is_last_tile, num_ckpts, 0)
    wait_row_dma(buf_ref, num_states, sem)
