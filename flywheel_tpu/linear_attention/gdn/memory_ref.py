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
    s_idx_to_replay_len: jax.Array | None
    s_idx_to_flush: jax.Array | None
    records_per_tile: int = dataclasses.field(metadata={"static": True})

    def get_record(self, p_id: int | jax.Array, idx: int) -> PackedPIdRecord:
        record_idx = p_id * self.records_per_tile + idx
        return PackedPIdRecord(
            records=self.records,
            offset=record_idx * PackedPIdRecord.STRUCT_SIZE)

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
        s_idx_to_replay_len: jax.Array | None = None,
        s_idx_to_flush: jax.Array | None = None,
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
            s_idx_to_replay_len=s_idx_to_replay_len,
            s_idx_to_flush=s_idx_to_flush,
            records_per_tile=cfg.seq_tile_size,
        )

    def replay_base(self, s_idx: jax.Array) -> jax.Array:
        """Record row the sequence's new tokens are written from."""
        return jnp.where(self.s_idx_to_flush[s_idx] != 0, 0,
                         self.s_idx_to_replay_len[s_idx])

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


def qkv_copy(hbm_ref: jax.Ref, buf_ref: jax.Ref, sem: jax.Ref,
             p_id: int | jax.Array, idx: int, metadata_ref: MetadataRef):
    """The DMA that fetches record idx's rows of a native [batch, dim] input,
    widened to their aligned window."""
    record = metadata_ref.get_record(p_id, idx)
    window_base, window_rows = aligned_window(record.r_base, record.r_size)
    return pltpu.make_async_copy(
        hbm_ref.at[pl.ds(window_base, window_rows)],
        buf_ref.at[idx, pl.ds(0, window_rows)],
        sem,
    )


def start_qkv_in(hbm_ref: jax.Ref, buf_ref: jax.Ref, sem: jax.Ref,
                 p_id: int | jax.Array, metadata_ref: MetadataRef,
                 cfg: config.GDNConfig) -> None:
    """Fetch each record's rows of a native [batch, dim] input, widened to
    their aligned window.

    A record's first row lands at row r_base % SUBLANE_ALIGN of its block
    rather than at row 0, and the rows past r_size are not the record's.
    """
    for idx in range(cfg.seq_tile_size):
        qkv_copy(hbm_ref, buf_ref, sem, p_id, idx, metadata_ref).start()


def wait_qkv_in(hbm_ref: jax.Ref, buf_ref: jax.Ref, sem: jax.Ref,
                p_id: int | jax.Array, metadata_ref: MetadataRef,
                cfg: config.GDNConfig) -> None:
    # Note (david): each copy is waited on with its own descriptor, not one
    # wait_row_dma sized to the summed rows. These rows are the block's tiled
    # axis, and where Mosaic tiles the block 16 rows deep (v7x, e.g. a verify
    # window's 16-row block) the summed self-copy left the semaphore nonzero at
    # kernel exit and halted the core: three 8-row copies against one 24-row
    # wait. A copy's own descriptor takes back exactly what it added.
    for idx in range(cfg.seq_tile_size):
        qkv_copy(hbm_ref, buf_ref, sem, p_id, idx, metadata_ref).wait()


def start_compact_in(hbm_ref: jax.Ref, buf_ref: jax.Ref, sem: jax.Ref,
                     p_id: int | jax.Array, metadata_ref: MetadataRef,
                     cfg: config.GDNConfig) -> None:
    """Fetch each record's rows of a compact [batch, 1, heads] input (b, a)."""
    for idx in range(cfg.seq_tile_size):
        record = metadata_ref.get_record(p_id, idx)
        # Note (david): the compact token axis is untiled, so the window needs
        # no tail rounding and never leaves the batch. PER_SEQ starts it at the
        # qkv block's aligned row, so row r of b gates block row r.
        if cfg.mode == config.GDNMode.PER_SEQ:
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
                    metadata_ref: MetadataRef, cfg: config.GDNConfig) -> None:
    """Wait for a tile's compact transfers (b, a)."""
    num_rows = 0
    for idx in range(cfg.seq_tile_size):
        record = metadata_ref.get_record(p_id, idx)
        if cfg.mode == config.GDNMode.PER_SEQ:
            delta = record.r_base & (config.SUBLANE_ALIGN - 1)
            num_rows += delta + record.r_size
        else:
            num_rows += record.r_size
    wait_row_dma(buf_ref, num_rows, sem)


def out_window(
    p_id: int | jax.Array, metadata_ref: MetadataRef, cfg: config.GDNConfig
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


def out_copy(hbm_ref: jax.Ref, buf_ref: jax.Ref, sem: jax.Ref,
             p_id: int | jax.Array, metadata_ref: MetadataRef,
             cfg: config.GDNConfig):
    """The one DMA that writes the tile's whole aligned output window."""
    window_base, window_rows, _, _ = out_window(p_id, metadata_ref, cfg)
    return pltpu.make_async_copy(
        buf_ref.at[pl.ds(0, window_rows)],
        hbm_ref.at[pl.ds(window_base, window_rows)],
        sem,
    )


def start_out(hbm_ref: jax.Ref, buf_ref: jax.Ref, sem: jax.Ref,
              p_id: int | jax.Array, metadata_ref: MetadataRef,
              cfg: config.GDNConfig) -> None:
    out_copy(hbm_ref, buf_ref, sem, p_id, metadata_ref, cfg).start()


def wait_out(hbm_ref: jax.Ref, buf_ref: jax.Ref, sem: jax.Ref,
             p_id: int | jax.Array, metadata_ref: MetadataRef,
             cfg: config.GDNConfig) -> None:
    # Note (david): waited on with its own descriptor, for the reason
    # wait_qkv_in gives: the stage's rows are its tiled axis.
    out_copy(hbm_ref, buf_ref, sem, p_id, metadata_ref, cfg).wait()


def reads_state(record: PackedPIdRecord, metadata_ref: MetadataRef,
                only_flushing: bool) -> jax.Array:
    """Whether the record's sequence fetches its initial state: on its first
    tile, if it has one, and with only_flushing only if it flushes."""
    s_idx = record.s_idx
    should_read = jnp.logical_and(record.is_first_tile,
                                  metadata_ref.s_idx_has_initial_state[s_idx])
    if only_flushing:
        return jnp.logical_and(should_read,
                               metadata_ref.s_idx_to_flush[s_idx] != 0)
    else:
        return should_read


def start_state_in(hbm_ref: jax.Ref,
                   buf_ref: jax.Ref,
                   sem: jax.Ref,
                   p_id: int | jax.Array,
                   metadata_ref: MetadataRef,
                   cfg: config.GDNConfig,
                   apply_read_offset: bool = True,
                   only_flushing: bool = False) -> None:
    """Fetch each sequence's initial checkpoint into window position 0.

    Only a sequence's first tile reads, and only if it has an initial state.
    A replay checkpoint has a single slot, so it ignores the read offset; its
    f32 copy is read only by flushing sequences (only_flushing).
    """
    for idx in range(cfg.seq_tile_size):
        record = metadata_ref.get_record(p_id, idx)
        s_idx = record.s_idx
        if apply_read_offset:
            read_slot = (metadata_ref.s_idx_to_read_indices[s_idx] +
                         metadata_ref.s_idx_to_read_offset[s_idx])
        else:
            read_slot = metadata_ref.s_idx_to_read_indices[s_idx]
        num_states = jnp.where(
            reads_state(record, metadata_ref, only_flushing), 1, 0)
        pltpu.make_async_copy(
            hbm_ref.at[pl.ds(read_slot, num_states)],
            buf_ref.at[idx, pl.ds(0, num_states)],
            sem,
        ).start()


def wait_state_in(buf_ref: jax.Ref,
                  sem: jax.Ref,
                  p_id: int | jax.Array,
                  metadata_ref: MetadataRef,
                  cfg: config.GDNConfig,
                  only_flushing: bool = False) -> None:
    num_states = 0
    for idx in range(cfg.seq_tile_size):
        record = metadata_ref.get_record(p_id, idx)
        num_states += jnp.where(
            reads_state(record, metadata_ref, only_flushing), 1, 0)
    wait_row_dma(buf_ref, num_states, sem)


def start_state_out(hbm_ref: jax.Ref, buf_ref: jax.Ref, sem: jax.Ref,
                    p_id: int | jax.Array, metadata_ref: MetadataRef,
                    cfg: config.GDNConfig) -> None:
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
                   metadata_ref: MetadataRef, cfg: config.GDNConfig) -> None:
    num_states = 0
    for idx in range(cfg.seq_tile_size):
        record = metadata_ref.get_record(p_id, idx)
        num_ckpts = jnp.minimum(record.r_size, cfg.window_size)
        num_states += jnp.where(record.is_last_tile, num_ckpts, 0)
    wait_row_dma(buf_ref, num_states, sem)


# Note (david): the conv state cache arrives as its [prev_kernel_size,
# num_slots, dim_size] view, whose slot axis is tiled. Mosaic rejects a traced
# row count there, so each conv checkpoint moves as its own pl.when-guarded
# DMA, waited on with its own descriptor.


def conv_in_copies(hbm_ref: jax.Ref, buf_ref: jax.Ref, sem: jax.Ref,
                   p_id: int | jax.Array, metadata_ref: MetadataRef,
                   cfg: config.GDNConfig):
    """(is_read, DMA) per sequence: its initial conv state from
    read_indices[s] + read_offset[s] into window position 0, read on its first
    tile if it has an initial state, as start_state_in."""
    copies = []
    for idx in range(cfg.seq_tile_size):
        record = metadata_ref.get_record(p_id, idx)
        s_idx = record.s_idx
        read_slot = (metadata_ref.s_idx_to_read_indices[s_idx] +
                     metadata_ref.s_idx_to_read_offset[s_idx])
        is_read = jnp.logical_and(
            record.is_first_tile, metadata_ref.s_idx_has_initial_state[s_idx])
        copies.append((is_read,
                       pltpu.make_async_copy(hbm_ref.at[:, read_slot],
                                             buf_ref.at[idx, 0], sem)))
    return copies


def start_conv_in(hbm_ref: jax.Ref, buf_ref: jax.Ref, sem: jax.Ref,
                  p_id: int | jax.Array, metadata_ref: MetadataRef,
                  cfg: config.GDNConfig) -> None:
    for is_read, copy in conv_in_copies(hbm_ref, buf_ref, sem, p_id,
                                        metadata_ref, cfg):
        pl.when(is_read)(copy.start)


def wait_conv_in(hbm_ref: jax.Ref, buf_ref: jax.Ref, sem: jax.Ref,
                 p_id: int | jax.Array, metadata_ref: MetadataRef,
                 cfg: config.GDNConfig) -> None:
    for is_read, copy in conv_in_copies(hbm_ref, buf_ref, sem, p_id,
                                        metadata_ref, cfg):
        pl.when(is_read)(copy.wait)


def conv_out_copies(hbm_ref: jax.Ref, buf_ref: jax.Ref, sem: jax.Ref,
                    p_id: int | jax.Array, metadata_ref: MetadataRef,
                    cfg: config.GDNConfig):
    """(is_written, DMA) per sequence and window position: position t of a
    sequence's last tile goes to state_indices[s] + t, as start_state_out."""
    copies = []
    for idx in range(cfg.seq_tile_size):
        record = metadata_ref.get_record(p_id, idx)
        write_slot = metadata_ref.s_idx_to_state_indices[record.s_idx]
        num_ckpts = jnp.where(record.is_last_tile,
                              jnp.minimum(record.r_size, cfg.window_size), 0)
        for pos in range(cfg.window_size):
            copies.append((pos < num_ckpts,
                           pltpu.make_async_copy(
                               buf_ref.at[idx, pos],
                               hbm_ref.at[:, write_slot + pos], sem)))
    return copies


def start_conv_out(hbm_ref: jax.Ref, buf_ref: jax.Ref, sem: jax.Ref,
                   p_id: int | jax.Array, metadata_ref: MetadataRef,
                   cfg: config.GDNConfig) -> None:
    for is_written, copy in conv_out_copies(hbm_ref, buf_ref, sem, p_id,
                                            metadata_ref, cfg):
        pl.when(is_written)(copy.start)


def wait_conv_out(hbm_ref: jax.Ref, buf_ref: jax.Ref, sem: jax.Ref,
                  p_id: int | jax.Array, metadata_ref: MetadataRef,
                  cfg: config.GDNConfig) -> None:
    for is_written, copy in conv_out_copies(hbm_ref, buf_ref, sem, p_id,
                                            metadata_ref, cfg):
        pl.when(is_written)(copy.wait)


def replay_in_copy(hbm_ref: jax.Ref, buf_ref: jax.Ref, sem: jax.Ref,
                   p_id: int | jax.Array, idx: int,
                   metadata_ref: MetadataRef):
    """The DMA that fetches record idx's committed replay rows, rounded up to
    whole SUBLANE_ALIGN-row tiles; none on a padded record."""
    record = metadata_ref.get_record(p_id, idx)
    s_idx = record.s_idx
    align = config.SUBLANE_ALIGN
    num_rows = jnp.where(
        record.is_first_tile,
        pl.cdiv(metadata_ref.s_idx_to_replay_len[s_idx], align) * align, 0)
    num_rows = pl.multiple_of(num_rows, align)
    read_slot = metadata_ref.s_idx_to_read_indices[s_idx]
    return pltpu.make_async_copy(
        hbm_ref.at[read_slot, pl.ds(0, num_rows)],
        buf_ref.at[idx, pl.ds(0, num_rows)],
        sem,
    )


def start_replay_in(hbm_ref: jax.Ref, buf_ref: jax.Ref, sem: jax.Ref,
                    p_id: int | jax.Array, metadata_ref: MetadataRef,
                    cfg: config.GDNConfig) -> None:
    for idx in range(cfg.seq_tile_size):
        replay_in_copy(hbm_ref, buf_ref, sem, p_id, idx, metadata_ref).start()


def wait_replay_in(hbm_ref: jax.Ref, buf_ref: jax.Ref, sem: jax.Ref,
                   p_id: int | jax.Array, metadata_ref: MetadataRef,
                   cfg: config.GDNConfig) -> None:
    # Note (david): the record rows are the tiled axis, so each copy is waited
    # with its own descriptor, as in wait_qkv_in.
    for idx in range(cfg.seq_tile_size):
        replay_in_copy(hbm_ref, buf_ref, sem, p_id, idx, metadata_ref).wait()


def replay_out_copy(hbm_ref: jax.Ref, buf_ref: jax.Ref, sem: jax.Ref,
                    p_id: int | jax.Array, idx: int,
                    metadata_ref: MetadataRef):
    """The DMA that writes the aligned row window holding record idx's new
    records [base, base + r_size); none on a padded record.

    The window's rows before base are committed records the step read, and
    its rows past base + r_size are never read again.
    """
    record = metadata_ref.get_record(p_id, idx)
    base = metadata_ref.replay_base(record.s_idx)
    window_base, window_rows = aligned_window(base, record.r_size)
    window_rows = jnp.where(record.is_last_tile, window_rows, 0)
    write_slot = metadata_ref.s_idx_to_state_indices[record.s_idx]
    return pltpu.make_async_copy(
        buf_ref.at[idx, pl.ds(window_base, window_rows)],
        hbm_ref.at[write_slot, pl.ds(window_base, window_rows)],
        sem,
    )


def start_replay_out(hbm_ref: jax.Ref, buf_ref: jax.Ref, sem: jax.Ref,
                     p_id: int | jax.Array, metadata_ref: MetadataRef,
                     cfg: config.GDNConfig) -> None:
    for idx in range(cfg.seq_tile_size):
        replay_out_copy(hbm_ref, buf_ref, sem, p_id, idx, metadata_ref).start()


def wait_replay_out(hbm_ref: jax.Ref, buf_ref: jax.Ref, sem: jax.Ref,
                    p_id: int | jax.Array, metadata_ref: MetadataRef,
                    cfg: config.GDNConfig) -> None:
    for idx in range(cfg.seq_tile_size):
        replay_out_copy(hbm_ref, buf_ref, sem, p_id, idx, metadata_ref).wait()


def num_flushed_ckpts(record: PackedPIdRecord,
                      metadata_ref: MetadataRef) -> jax.Array:
    is_flush = metadata_ref.s_idx_to_flush[record.s_idx] != 0
    return jnp.where(jnp.logical_and(record.is_last_tile, is_flush), 1, 0)


def wait_flushed_ckpt_in(buf_ref: jax.Ref, sem: jax.Ref,
                         p_id: int | jax.Array, idx: int,
                         metadata_ref: MetadataRef,
                         cfg: config.GDNConfig) -> None:
    """Wait, before record idx's flush reads its f32 checkpoint, for the
    tile's checkpoint reads (start_state_in with only_flushing)."""
    # Note (david): the reads share one byte-counting semaphore, so a wait
    # sized to one record's read could return on another record's bytes; the
    # tile's first reading record waits for all of them.
    reads = [
        reads_state(metadata_ref.get_record(p_id, prev), metadata_ref,
                    only_flushing=True) for prev in range(idx + 1)
    ]
    is_first_read = reads[idx]
    for prev_reads in reads[:idx]:
        is_first_read = jnp.logical_and(is_first_read,
                                        jnp.logical_not(prev_reads))

    @pl.when(is_first_read)
    def _wait():
        wait_state_in(buf_ref, sem, p_id, metadata_ref, cfg,
                      only_flushing=True)


def start_flushed_ckpt_out(hbm_ref: jax.Ref, buf_ref: jax.Ref, sem: jax.Ref,
                           p_id: int | jax.Array, idx: int,
                           metadata_ref: MetadataRef) -> None:
    """Write record idx's new checkpoint to state_indices[s] if its sequence
    flushes; none on a padded record."""
    record = metadata_ref.get_record(p_id, idx)
    write_slot = metadata_ref.s_idx_to_state_indices[record.s_idx]
    num_states = num_flushed_ckpts(record, metadata_ref)
    pltpu.make_async_copy(
        buf_ref.at[idx, pl.ds(0, num_states)],
        hbm_ref.at[pl.ds(write_slot, num_states)],
        sem,
    ).start()


def wait_ckpt_out(buf_ref: jax.Ref, sem: jax.Ref, p_id: int | jax.Array,
                  metadata_ref: MetadataRef, cfg: config.GDNConfig) -> None:
    num_states = 0
    for idx in range(cfg.seq_tile_size):
        record = metadata_ref.get_record(p_id, idx)
        num_states += num_flushed_ckpts(record, metadata_ref)
    wait_row_dma(buf_ref, num_states, sem)
