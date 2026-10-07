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
"""Static configuration and VMEM scratch layout of the GDN kernel."""

import dataclasses
import enum

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Note (david): Mosaic requires every DMA offset and size along a tiled
# dimension to be provably a multiple of the sublane tile height, and the token
# axis of the native [batch, dim] layout is that tiled dimension.
SUBLANE_ALIGN = 8

# Note (david): Mosaic tiles a bf16 VMEM scratch ref whose row count is a
# multiple of 16 as (16, 128) (v7x does; v6e may keep (8, 128)), and a dynamic
# vector load or store must start on that tile. The bf16 output stage is
# therefore read and written in whole 16-row tiles, which is legal under either
# tiling; the HBM DMAs stay on SUBLANE_ALIGN.
STAGE_TILE_ROWS = 16

NUM_BUFFERS = 2

# Note (david): one DMA semaphore row per HBM <-> VMEM stream. The state streams
# keep their in and out rings on separate rows: during step t + 1 the prefetch
# of t + 2 (in ring) and the still-unwaited write-back of t (out ring) both sit
# on slot t % 2, and a DMA semaphore only counts bytes.
STREAM_QKV = 0
STREAM_B = 1
STREAM_A = 2
STREAM_CONV_IN = 3
STREAM_REC_IN = 4
STREAM_OUT = 5
STREAM_CONV_OUT = 6
STREAM_REC_OUT = 7
NUM_STREAMS = 8


class GDNMode(enum.StrEnum):
    """How a kernel call tiles its sequences.

    BATCHED packs several sequences per tile, each wholly inside one tile with
    at most window_size tokens (a decode token or a speculative verify window).
    PER_SEQ spreads one sequence over as many tiles as it needs.
    """

    BATCHED = enum.auto()
    PER_SEQ = enum.auto()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Dtypes:
    act_in: jnp.dtype
    act_out: jnp.dtype
    compute: jnp.dtype
    recurrent_state: jnp.dtype
    conv_state: jnp.dtype


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class GDNConfig:
    mode: GDNMode
    dtypes: Dtypes
    batch_size: int
    dim_size: int
    kernel_size: int
    tile_size: int
    num_kq_heads: int
    num_v_heads: int
    kq_head_dim: int
    v_head_dim: int
    # Note (david): tile_size is what one DMA moves, compute_chunk_size what one
    # delta-rule step sees. The [heads, chunk, chunk] triangular matrices cost
    # O(compute_chunk_size**2), so splitting the two lets a tile grow (fewer
    # DMA waits and metadata reads per token) while the math stays MXU-sized,
    # and the tile's chunks form a straight-line body whose state-independent
    # halves the LLO scheduler can overlap.
    compute_chunk_size: int
    num_buffers: int = NUM_BUFFERS
    # Note (david): max tokens per speculative verify window
    # (num_speculative_tokens + 1) and the number of state checkpoints kept per
    # sequence: rejected draft tokens roll back by checkpoint selection. It is 1
    # without speculative decoding, so shapes and loops size off it
    # unconditionally and the extra axis folds away.
    window_size: int = 1

    @property
    def chunk_size(self) -> int:
        if self.mode == GDNMode.PER_SEQ:
            return self.tile_size
        else:
            return self.window_size

    @property
    def num_compute_chunks(self) -> int:
        assert self.chunk_size % self.compute_chunk_size == 0, (
            f"chunk_size ({self.chunk_size}) must be a multiple of "
            f"compute_chunk_size ({self.compute_chunk_size})")
        return self.chunk_size // self.compute_chunk_size

    @property
    def seq_tile_size(self) -> int:
        if self.mode == GDNMode.PER_SEQ:
            return 1
        else:
            return self.tile_size

    @property
    def prev_kernel_size(self) -> int:
        return self.kernel_size - 1

    @property
    def qkv_block_rows(self) -> int:
        # Note (david): a sequence may start at any token, so its DMA widens to
        # the enclosing sublane-aligned window: up to SUBLANE_ALIGN - 1 rows
        # early and rounded up to whole tiles.
        widened_rows = self.chunk_size + SUBLANE_ALIGN - 1
        return pl.cdiv(widened_rows, SUBLANE_ALIGN) * SUBLANE_ALIGN

    @property
    def out_window_rows(self) -> int:
        # Note (david): the token-major output tiles its token axis, so a tile's
        # output DMA snaps to the SUBLANE_ALIGN grid and the tile's own rows
        # start r_base % SUBLANE_ALIGN into the window, so at most
        # tile_rows + SUBLANE_ALIGN - 1 rows are used. Rounding past that to
        # whole stage tiles keeps the 16-row tile holding the last used row,
        # and the DMA window, inside the stage.
        tile_rows = self.seq_tile_size * self.chunk_size
        return (pl.cdiv(tile_rows + SUBLANE_ALIGN, STAGE_TILE_ROWS) *
                STAGE_TILE_ROWS)

    @property
    def v_dim_size(self) -> int:
        return self.num_v_heads * self.v_head_dim

    @property
    def v_per_kq_head(self) -> int:
        return self.num_v_heads // self.num_kq_heads

    @property
    def aligned_num_v_heads(self) -> int:
        num_lanes = pltpu.get_tpu_info().num_lanes
        return pl.cdiv(self.num_v_heads, num_lanes) * num_lanes

    def get_scratch_shape_dict(self) -> dict[str, pl.MemoryRef | None]:
        """VMEM rings, inter-tile carries and DMA semaphores of the pipeline.

        Keys are the scratch keyword names of wrapper.outer_kernel. Every ring
        is num_buffers slots deep; state inputs hold the initial checkpoint at
        window position 0, state outputs one checkpoint per window position.
        """
        slots = self.num_buffers
        tile_seqs = self.seq_tile_size
        ba_shape = (slots, tile_seqs, self.chunk_size, 1,
                    self.aligned_num_v_heads)
        conv_state_shape = (self.prev_kernel_size, self.dim_size)
        rec_state_shape = (self.num_v_heads, self.kq_head_dim, self.v_head_dim)

        # Note (david): a BATCHED tile holds whole sequences, so only PER_SEQ
        # carries state from one tile to the next.
        if self.mode == GDNMode.PER_SEQ:
            carry_conv_scratch = pltpu.VMEM((tile_seqs, *conv_state_shape),
                                            jnp.float32)
            carry_recurrent_scratch = pltpu.VMEM(
                (tile_seqs, *rec_state_shape), jnp.float32)
        else:
            carry_conv_scratch = None
            carry_recurrent_scratch = None

        return {
            "qkv_buf": pltpu.VMEM(
                (slots, tile_seqs, self.qkv_block_rows, self.dim_size),
                self.dtypes.act_in),
            "b_buf": pltpu.VMEM(ba_shape, jnp.float32),
            "a_buf": pltpu.VMEM(ba_shape, jnp.float32),
            "conv_in_buf": pltpu.VMEM((slots, tile_seqs, 1, *conv_state_shape),
                                      self.dtypes.conv_state),
            "rec_in_buf": pltpu.VMEM((slots, tile_seqs, 1, *rec_state_shape),
                                     self.dtypes.recurrent_state),
            "conv_out_buf": pltpu.VMEM(
                (slots, tile_seqs, self.window_size, *conv_state_shape),
                self.dtypes.conv_state),
            "rec_out_buf": pltpu.VMEM(
                (slots, tile_seqs, self.window_size, *rec_state_shape),
                self.dtypes.recurrent_state),
            "out_buf": pltpu.VMEM(
                (slots, self.out_window_rows, self.v_dim_size),
                self.dtypes.act_out),
            "carry_conv_scratch_ref": carry_conv_scratch,
            "carry_recurrent_scratch_ref": carry_recurrent_scratch,
            "sems": pltpu.SemaphoreType.DMA((NUM_STREAMS, slots)),
        }
