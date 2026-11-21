# Copyright 2026 heavyball-research
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

# The metadata fill, the zero fill and the partial-sublane epilogue are adapted
# from megablox gmm_v2 in tpu-inference
# (tpu_inference/kernels/megablox/gmm_v2.py in
# https://github.com/vllm-project/tpu-inference), Copyright 2026 Google LLC,
# licensed under the Apache License, Version 2.0.

"""The fused MoE expert MLP: the dispatch gather, mm1, the gated activation and
mm2 in one kernel.

The kernel takes the unsorted activations in the packing-dim layout
[num_tokens, packing, hidden // packing], in which every token row is one
contiguous HBM block, plus token_indices, which maps each expert-sorted row to
its source token. It gathers the rows with one async DMA each instead of
reading a sorted copy from HBM. Per gm tile (a window of up to tile_m sorted
rows of one expert), mm1 against the gate and the up half of the expert's
merged weights w1 (two matmuls) collapses through the gated activation to a
bf16 intermediate tile that never leaves the chip, and mm2 against the
expert's down projection w2 gives the output. The epilogue multiplies each
row by its routing weight (row_scale), masks the rows of other experts and
carries the partial sublane block over to the next tile.

The intermediate is cast to bf16 where a two-kernel path (mm1, then mm2)
stores it to HBM. Rows of the tile window outside the current group carry
wrong-expert values; a matmul output row depends only on the same lhs row,
and those rows are masked after mm2.

The pipeline is handwritten (no emit_pipeline); PipelineDepths sets its
depths:

- lhs: one row DMA per sorted row of a gm tile up to its group's end. The
  whole-expert path issues tile c + lhs_ahead's DMAs inside tile c's compute
  block, as predicated straight-line code between mm1 and the w2 wait; the
  prologue, lhs_ahead = 0 and the weight-blocked path issue them from a loop
  over sublane groups at the top of the step.
- token_indices: a tile's window of token_indices into SMEM, a tile ahead of
  the gather that reads it as scalars.
- w1: one expert block per distinct group in visit order (ordinals from the
  metadata fill), waited on the first gm tile of each group only, so the gm
  tiles of a group reuse the buffered weights.
- w2: keyed by the same ordinals on its own slots and semaphores, waited
  just before mm2 so the load has the whole mm1 + activation window to land;
  with the in-block gather it issues right after the row DMAs, behind them in
  the DMA queue.
- row_scale: f32[size_m], each tile's window streamed into SMEM next to
  its token_indices window, on the same slots; the epilogue builds the
  per-row column from the scalars.
- out: a variable-size write-back DMA per gm tile, issued at step c, waited
  at step c + out_depth and drained after the loop.

When two whole experts of w1 and of w2 do not fit VMEM next to the other
buffers, the weight streams split the intermediate dim into blocks of
weight_block_columns columns instead: a block is the gate and up column
windows of w1, two DMAs into one [hidden, 2 * block_columns] slot, and the
[block_columns, hidden] rows of w2, carried through WEIGHT_BLOCK_DEPTH rings,
and mm2 accumulates across blocks in a resident f32 accumulator. Every gm
tile re-streams its expert's blocks; plan_weight_streams sizes the blocks
and the tiles.
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from flywheel_tpu.core.tpu import BF16_BYTES, F32_BYTES, NUM_LANES, NUM_SUBLANES
from flywheel_tpu.moe.config import (
    BF16_PACKING,
    INSTRUCTION_MEMORY_BUNDLES,
    MIN_TILE_M,
    ROW_ALIGNMENT,
    TILE_M_GRANULE,
    V6E_GENERATION,
    V7X_GENERATION,
    PipelineDepths,
    default_tail_bucket_m_values,
    default_tile_m,
    generation_and_mxu_columns,
    sublane_rows,
    vmem_limit_bytes,
)

ACTIVATIONS = {"silu": jax.nn.silu, "gelu": jax.nn.gelu}
# Note (david): the zero fill reuses one [tile_zero_m, packing, NUM_LANES]
# buffer for every lane chunk of a row block, so a long DMA needs no
# [tile_zero_m, hidden] buffer.
ZERO_BUFFER_BYTES = 2 * 1024 * 1024
# Note (david): the row-gather descriptor loop is unrolled because the scalar
# core's per-row issue cost dominates the gather; spreading the descriptors
# over several DMA queues or priorities changed nothing.
LHS_ISSUE_UNROLL = 16
# Note (david): only bucket bodies of at least
# tile_m // IN_BLOCK_GATHER_DIVISOR rows issue the next tile's row gather
# inside their compute. Each in-block copy of the gather is one predicated DMA
# per row of tile_m and costs compile time in every body that holds one; the
# smaller tail buckets are rare at the token counts where the gather is a
# large part of a tile, so they keep the loop at the top of the step.
IN_BLOCK_GATHER_DIVISOR = 2
# Note (david): the in-block gather guards this many row DMAs (a divisor of
# the sublane rows) with each pl.when: fewer predicated regions compile
# faster, and a region this small still becomes predicated DMAs inside the
# compute block.
IN_BLOCK_GATHER_ROWS_PER_WHEN = 4
# Note (david): the weight-blocked path re-streams an expert's blocks for
# every gm tile, so its weight stream bounds it below 256 rows, and at 512
# rows its unrolled blocks overflow the instruction memory, whose overlays
# then reload every step.
WEIGHT_BLOCK_TILE_M = 256
# Note (david): the weight-block rings hold one intermediate-dim block of w1
# and of w2 per slot. Step w issues block w + WEIGHT_BLOCK_DEPTH into the slot
# it just finished reading, so the DMA queue stays full at bandwidth-bound
# shapes while the buffers stay small enough for large hidden and
# intermediate sizes.
WEIGHT_BLOCK_DEPTH = 2
# Note (david): the register allocator's spill slots count against the scoped
# VMEM limit like the scratch buffers. Their bound is in f32 [tile_m, columns]
# tiles: on the whole-expert path the wider of the input and the output, and
# the intermediate; on the weight-blocked path the widest of the output, the
# input and a block's gate|up. The relayout of the output to the packing-dim
# layout spills UNALIGNED_OUTPUT_SPILL_TILES output tiles when the width is
# not a multiple of OUTPUT_SPILL_ALIGNMENT, and row segments spill a block of
# w1. Fitted per TPU generation to the VMEM the compiler reports and rounded
# up by SPILL_MARGIN_ROWS: a bound too low fails the compile, one too high
# only narrows a block.
OUTPUT_SPILL_ALIGNMENT = 2048
UNALIGNED_OUTPUT_SPILL_TILES = {V6E_GENERATION: 1, V7X_GENERATION: 2}
SPILL_MARGIN_ROWS = 32
DEFAULT_PIPELINE_DEPTHS = PipelineDepths()


@dataclasses.dataclass(frozen=True)
class BucketBodyCodeModel:
    """Static bundles of one whole-expert bucket body, linear in its rows and
    in the MXU weight tiles a row passes through (gate, up and down):
    fixed + weight_tiles * per_weight_tile + rows * (per_row + weight_tiles *
    per_row_weight_tile), plus per_in_block_gather_row for every row of
    tile_m when the body issues the in-block gather."""

    fixed: float
    per_weight_tile: float
    per_row: float
    per_row_weight_tile: float
    per_in_block_gather_row: float


# Note (david): fitted per TPU generation to the final LLO bundles of the
# bucket bodies, rounded up: an estimate too low lets the step loop outgrow
# the instruction memory, one too high only drops a tail bucket.
BUCKET_BODY_CODE_MODELS = {
    V6E_GENERATION: BucketBodyCodeModel(
        fixed=720,
        per_weight_tile=0,
        per_row=1.5,
        per_row_weight_tile=0.25,
        per_in_block_gather_row=3,
    ),
    V7X_GENERATION: BucketBodyCodeModel(
        fixed=0,
        per_weight_tile=16.5,
        per_row=5.7,
        per_row_weight_tile=0.095,
        per_in_block_gather_row=3,
    ),
}
# Note (david): the instruction memory the bucket bodies may fill; the
# overlay loader and the gm step's code around the bodies take the rest.
INSTRUCTION_MEMORY_RESERVE_BUNDLES = 2048


def index_window_capacity(bucket_m: int) -> int:
    """Elements of a 128-aligned token_indices window that covers bucket_m
    rows starting anywhere inside a 128-element block."""
    return pl.cdiv(bucket_m + ROW_ALIGNMENT, ROW_ALIGNMENT) * ROW_ALIGNMENT


def validate_tail_bucket_m_values(
    tail_bucket_m_values: tuple[int, ...], *, tile_m: int, sublane: int
) -> tuple[int, ...]:
    """The static tail bucket row counts, checked: sorted, unique, sublane
    multiples, ending with tile_m."""
    if not tail_bucket_m_values:
        raise ValueError("tail_bucket_m_values must be non-empty")
    values = tuple(int(bucket_m) for bucket_m in tail_bucket_m_values)
    if values != tuple(sorted(values)) or len(set(values)) != len(values):
        raise ValueError(
            f"tail_bucket_m_values must be sorted and unique, got {values}"
        )
    if values[-1] != tile_m:
        raise ValueError(
            f"tail_bucket_m_values must end with tile_m={tile_m}, got {values}"
        )
    for bucket_m in values:
        if bucket_m <= 0 or bucket_m % sublane != 0:
            raise ValueError(
                f"tail bucket {bucket_m} must be a positive multiple of"
                f" sublane={sublane}"
            )
    return values


def get_weight_block_gather_pipeline_schedule(
    tile_m: int, segment_rows: int, sublane: int, num_weight_blocks: int
) -> tuple[tuple[int, int, bool], ...]:
    """Static (row_start, row_count, is_last) segments of the next tile's
    gather descriptors, one per weight block of the current tile's compute;
    the next tile's weights follow the last."""
    if segment_rows <= 0 or segment_rows > tile_m:
        raise ValueError(
            f"segment_rows must be in [1, tile_m={tile_m}], got {segment_rows}"
        )
    if segment_rows % sublane != 0:
        raise ValueError(
            f"segment_rows must be a multiple of sublane={sublane}, got"
            f" {segment_rows}"
        )
    if tile_m % segment_rows != 0:
        raise ValueError(
            f"segment_rows={segment_rows} must divide tile_m={tile_m}"
        )
    num_segments = tile_m // segment_rows
    if num_segments != num_weight_blocks:
        raise ValueError(
            "the weight-block gather pipeline needs one row segment per"
            f" weight block, got {num_segments} segments and"
            f" {num_weight_blocks} blocks"
        )
    return tuple(
        (row_start, segment_rows, block == num_weight_blocks - 1)
        for block, row_start in enumerate(range(0, tile_m, segment_rows))
    )


@dataclasses.dataclass(frozen=True)
class Dimensions:
    """Problem sizes. size_m counts expert-sorted rows; size_n is w1's merged
    gate|up width and size_n_out the output width (hidden)."""

    size_m: int
    size_k: int
    size_n: int
    size_n_out: int
    num_experts: int
    # Rows of a bf16 HBM tile; gm tiles start and end on its multiples.
    sublane: int
    # bf16 values per 32-bit word: the packing-dim layout is
    # [rows, packing, cols // packing].
    packing: int


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MetadataRef:
    """SMEM tables the metadata fill writes and the pipeline reads."""

    gm_id_to_group_id: jax.Array
    gm_id_to_m_offset: jax.Array
    # The static row bucket of each gm tile: tile_m, or under tail_bucket
    # one of tail_bucket_m_values.
    gm_id_to_tile_m: jax.Array
    # Visit-order ordinal of each distinct non-empty group -> its group id.
    # Drives the weight streams: one block DMA per ordinal.
    ord_to_group_id: jax.Array
    # The [start, end) gm tile range of each ordinal.
    ord_to_gm_start: jax.Array
    ord_to_gm_end: jax.Array


@dataclasses.dataclass(frozen=True)
class FusedMlpConfig:
    """The static configuration of one fused MLP kernel."""

    dims: Dimensions
    tile_m: int
    activation: str
    # Tail tiles dispatch to static MXU row shapes; full tiles keep tile_m.
    tail_bucket: bool
    tail_bucket_m_values: tuple[int, ...]
    pipeline_depths: PipelineDepths
    # Splits the intermediate dim into blocks of weight_block_columns columns.
    weight_block_columns: int | None
    # Splits each gm tile into static weight_block_segment_rows-row segments for
    # every weight block's mm1 + activation + mm2.
    weight_block_segment_rows: int | None
    # Issues the next tile's gather descriptors in segments of this many
    # rows, between the current tile's compute segments.
    gather_pipeline_rows: int | None


def fill_metadata(
    group_sizes_ref: jax.Array,  # SMEM int32[num_experts]
    metadata_ref: MetadataRef,
    *,
    cfg: FusedMlpConfig,
) -> tuple[jax.Array, jax.Array]:
    """Tiles the expert-sorted rows into gm tiles and records the distinct
    groups in visit order.

    Returns:
        (num_gm, num_ord): the gm tile count and the count of distinct
        non-empty groups (the expert blocks the weight streams load).
    """
    sublane = cfg.dims.sublane
    metadata_ref.gm_id_to_m_offset[0] = 0

    @jax.named_scope("inner_tm_loop")
    def _inner_tm_loop(tm_id, curr_m_offset, *, end_m_offset, group_id):
        local_offset = curr_m_offset % sublane
        remaining = end_m_offset - curr_m_offset
        aligned_remaining = remaining + local_offset
        tail_bucket_m = cfg.tail_bucket_m_values[-1]
        for bucket_m in reversed(cfg.tail_bucket_m_values[:-1]):
            tail_bucket_m = jnp.where(
                aligned_remaining <= bucket_m, bucket_m, tail_bucket_m
            )
        tile_m = jnp.where(cfg.tail_bucket, tail_bucket_m, cfg.tile_m)
        tm_size = jnp.minimum(tile_m - local_offset, remaining)

        metadata_ref.gm_id_to_group_id[tm_id] = group_id
        metadata_ref.gm_id_to_tile_m[tm_id] = tile_m

        next_m_offset = curr_m_offset + tm_size
        metadata_ref.gm_id_to_m_offset[tm_id] = curr_m_offset
        metadata_ref.gm_id_to_m_offset[tm_id + 1] = next_m_offset
        return next_m_offset

    @jax.named_scope("outer_group_loop")
    def _outer_group_loop(group_id, carry):
        num_gm, start_m_offset, num_ord = carry
        group_size = group_sizes_ref[group_id]
        end_m_offset = start_m_offset + group_size

        # Note (david): a group that starts inside a sublane block spans that
        # block's leading rows too: with sublane = tile_m = 4, group_size 3
        # and start 7, | 0 0 0 0 | 0 0 0 1 | 1 1 0 0 | takes 2 tiles, not
        # cdiv(3, 4) = 1. An empty group would still count 1 when the offset
        # is not 0, so it is skipped explicitly.
        local_offset = start_m_offset % sublane
        curr_num_gm = pl.cdiv(group_size + local_offset, cfg.tile_m)
        should_process = group_size > 0
        curr_num_gm = jnp.where(should_process, curr_num_gm, 0)
        next_num_gm = num_gm + curr_num_gm

        @pl.when(should_process)
        def _():
            metadata_ref.ord_to_group_id[num_ord] = group_id
            metadata_ref.ord_to_gm_start[num_ord] = num_gm
            metadata_ref.ord_to_gm_end[num_ord] = next_num_gm

        next_num_ord = num_ord + should_process.astype(jnp.int32)
        lax.fori_loop(
            num_gm,
            next_num_gm,
            functools.partial(
                _inner_tm_loop, end_m_offset=end_m_offset, group_id=group_id
            ),
            start_m_offset,
        )
        return next_num_gm, end_m_offset, next_num_ord

    num_gm, _, num_ord = lax.fori_loop(
        0, cfg.dims.num_experts, _outer_group_loop, (0, 0, 0)
    )
    return num_gm, num_ord


def zero_out_start(
    out_ref: jax.Array,  # HBM [size_m, packing, size_n_out // packing]
    zero_ref: jax.Array,  # VMEM [tile_zero_m, packing, NUM_LANES]
    sem: jax.Array,  # DMA sem [1]
    metadata_ref: MetadataRef,
    num_gm: jax.Array,
    *,
    dims: Dimensions,
) -> jax.Array:
    """Starts zeroing the output rows past the last gm tile; returns the
    zeroed row count, whose bytes the kernel's last wait counts.

    The gm tiles cover rows [0, sum(group_sizes)), so these are the rows
    past it, which no group claims (expert ids outside [0, E)).
    """
    zero_ref[...] = jnp.zeros_like(zero_ref)
    sublane = dims.sublane
    blocks_per_dma = zero_ref.shape[0] // sublane
    num_blocks = dims.size_m // sublane
    start = pl.cdiv(metadata_ref.gm_id_to_m_offset[num_gm], sublane)

    def _fill_zero(i, zero_rows):
        dma_start = start + i * blocks_per_dma
        dma_blocks = jnp.minimum(blocks_per_dma, num_blocks - dma_start)
        row_start = pl.multiple_of(dma_start * sublane, sublane)
        rows = pl.multiple_of(dma_blocks * sublane, sublane)
        for col in range(0, dims.size_n_out // dims.packing, NUM_LANES):
            pltpu.make_async_copy(
                zero_ref.at[pl.ds(0, rows)],
                out_ref.at[pl.ds(row_start, rows), :, col : col + NUM_LANES],
                sem.at[0],
            ).start(priority=1)
        return zero_rows + rows

    return lax.fori_loop(
        0, pl.cdiv(num_blocks - start, blocks_per_dma), _fill_zero, 0
    )


def fused_mlp_kernel(
    # Scalar prefetch
    group_sizes_ref,  # SMEM int32[num_experts]
    # In (HBM)
    x_ref,  # [num_tokens, packing, size_k // packing]
    token_indices_ref,  # int32[size_m]
    w1_ref,  # [E, size_k, size_n]
    w2_ref,  # [E, size_n // 2, size_n_out]
    row_scale_ref,  # f32[size_m]
    # Out (HBM)
    out_ref,  # [size_m, packing, size_n_out // packing]
    # Scratch
    partial_out_ref,  # VMEM [sublane, size_n_out]
    metadata_ref,  # MetadataRef
    lhs_buf_ref,  # VMEM [lhs_depth, tile_m, packing, size_k // packing]
    lhs_sem,  # DMA sem [lhs_depth]
    w1_buf_ref,  # VMEM [w1_depth, size_k, size_n], or the w1 block ring
    w1_sem,  # DMA sem [w1_depth] or [WEIGHT_BLOCK_DEPTH]
    w2_buf_ref,  # VMEM [w2_depth, size_n // 2, n_out], or the w2 block ring
    w2_sem,  # DMA sem [w2_depth] or [WEIGHT_BLOCK_DEPTH]
    out_buf_ref,  # VMEM [out_depth, tile_m, packing, size_n_out // packing]
    out_sem,  # DMA sem [out_depth]
    scale_buf_ref,  # SMEM [lhs_depth * index_window_capacity(tile_m)] f32
    scale_sem,  # DMA sem [lhs_depth]
    zero_ref,  # VMEM [tile_zero_m, packing, NUM_LANES]
    zero_sem,  # DMA sem [1]
    idx_buf_ref,  # SMEM [lhs_depth * index_window_capacity(tile_m)] int32
    idx_sem,  # DMA sem [lhs_depth]
    mm2_acc_ref,  # VMEM [tile_m, size_n_out] f32, or None without blocks
    *,
    cfg: FusedMlpConfig,
):
    """The handwritten multi-stream pipeline over gm tiles.

    token_indices is an HBM input streamed per tile, not a scalar-prefetch
    operand: prefetching the whole int32[size_m] into the 1 MiB SMEM caps
    size_m at 262144 (T=32768 at top_k 8).
    """
    dims, depths, tile_m = cfg.dims, cfg.pipeline_depths, cfg.tile_m
    sublane = dims.sublane
    out_tile_tail = out_buf_ref.shape[2:]
    idx_capacity = index_window_capacity(tile_m)
    activation_fn = ACTIVATIONS[cfg.activation]

    num_gm, num_ord = fill_metadata(group_sizes_ref, metadata_ref, cfg=cfg)
    zero_rows = zero_out_start(
        out_ref, zero_ref, zero_sem, metadata_ref, num_gm, dims=dims
    )

    def _lhs_rows_for_bucket(chunk, bucket_m):
        """(base, rows) of gm tile chunk's sublane-aligned row window.

        rows < bucket_m only on the last window of the table; the buffer
        tail past rows is stale and masked out in the epilogue. Both values
        are sublane multiples (size_m and bucket_m are); pl.multiple_of
        carries that to Mosaic, which needs dynamic slices on tiled dims to
        be tile-aligned.
        """
        m_start = metadata_ref.gm_id_to_m_offset[chunk]
        base = m_start - m_start % sublane
        rows = jnp.minimum(bucket_m, dims.size_m - base)
        return pl.multiple_of(base, sublane), pl.multiple_of(rows, sublane)

    def _idx_window_for_bucket(chunk, bucket_m):
        """(window_start, window_length, row_offset): the 128-aligned slice
        of token_indices that covers gm tile chunk's window, clamped to
        size_m (a multiple of 128), and the offset of the window's first row
        inside it."""
        m_start = metadata_ref.gm_id_to_m_offset[chunk]
        base = m_start - m_start % sublane
        window_start = base - base % ROW_ALIGNMENT
        window_length = jnp.minimum(
            index_window_capacity(bucket_m), dims.size_m - window_start
        )
        return (
            pl.multiple_of(window_start, ROW_ALIGNMENT),
            pl.multiple_of(window_length, ROW_ALIGNMENT),
            base - window_start,
        )

    def _window_copies(chunk, slot):
        """Tile chunk's token_indices and row_scale window copies into slot
        of their flat SMEM rings."""
        if cfg.gather_pipeline_rows is not None and cfg.tail_bucket:
            bucket_m = tile_m
        else:
            bucket_m = metadata_ref.gm_id_to_tile_m[chunk]
        window_start, window_length, _ = _idx_window_for_bucket(chunk, bucket_m)
        ring_start = pl.multiple_of(slot * idx_capacity, ROW_ALIGNMENT)
        return tuple(
            pltpu.make_async_copy(
                src_ref.at[pl.ds(window_start, window_length)],
                ring_ref.at[pl.ds(ring_start, window_length)],
                sem.at[slot],
            )
            for src_ref, ring_ref, sem in (
                (token_indices_ref, idx_buf_ref, idx_sem),
                (row_scale_ref, scale_buf_ref, scale_sem),
            )
        )

    def _issue_idx(chunk, slot):
        """Streams tile chunk's token_indices and row_scale windows into
        SMEM a tile ahead of its row gather, so the DMA latency hides under
        the in-flight matmul instead of stalling the scalar core."""
        for copy in _window_copies(chunk, slot):
            copy.start()

    def _wait_idx(chunk, slot):
        for copy in _window_copies(chunk, slot):
            copy.wait()

    def _issue_gather_segment(
        chunk,
        slot,
        bucket_m: int,
        row_start: int,
        row_count: int,
        *,
        should_clamp_rows: bool,
        should_fully_unroll: bool,
    ):
        """Issues the row DMAs of one static row range of a gathered tile."""
        _, valid_rows = _lhs_rows_for_bucket(chunk, bucket_m)
        _, _, row_offset = _idx_window_for_bucket(chunk, bucket_m)
        idx_base = slot * idx_capacity + row_offset

        def _copy_row(local_row, carry):
            dst_row = row_start + local_row
            if should_clamp_rows:
                src_offset = jnp.minimum(dst_row, valid_rows - 1)
            else:
                src_offset = dst_row
            src_row = idx_buf_ref[idx_base + src_offset]
            pltpu.make_async_copy(
                x_ref.at[pl.ds(src_row, 1)],
                lhs_buf_ref.at[slot, pl.ds(dst_row, 1)],
                lhs_sem.at[slot],
            ).start()
            return carry

        lax.fori_loop(
            0,
            row_count,
            _copy_row,
            0,
            unroll=True if should_fully_unroll else LHS_ISSUE_UNROLL,
        )

    def _tile_gather_rows(chunk):
        """Rows of gm tile chunk's window up to its group's end, rounded up
        to a sublane multiple: still inside the window, as size_m and the
        buckets are sublane multiples."""
        m_start = metadata_ref.gm_id_to_m_offset[chunk]
        m_end = metadata_ref.gm_id_to_m_offset[chunk + 1]
        base = m_start - m_start % sublane
        rows = pl.cdiv(m_end - base, sublane) * sublane
        return pl.multiple_of(rows, sublane)

    def _row_gather_copy(slot, idx_offset, row):
        """The DMA of the token at idx_offset of the index ring into row of
        lhs slot."""
        return pltpu.make_async_copy(
            x_ref.at[pl.ds(idx_buf_ref[idx_offset], 1)],
            lhs_buf_ref.at[slot, pl.ds(row, 1)],
            lhs_sem.at[slot],
        )

    def _issue_tile_gather(chunk, slot, num_rows):
        """One row DMA per row below num_rows (a sublane multiple) of gm
        tile chunk, a sublane group per loop step: the flat index ring and
        the group's static row offsets keep the scalar work per descriptor
        minimal."""
        _, _, row_offset = _idx_window_for_bucket(chunk, tile_m)
        idx_base = slot * idx_capacity + row_offset

        def _gather_sublane_group(group, carry):
            group_start = pl.multiple_of(group * sublane, sublane)
            for row in range(sublane):
                _row_gather_copy(
                    slot, idx_base + group_start + row, group_start + row
                ).start()
            return carry

        lax.fori_loop(0, num_rows // sublane, _gather_sublane_group, 0)

    def _issue_ahead_gather_in_block(c):
        """Tile c + lhs_ahead's row DMAs as straight-line code inside tile
        c's compute, each under its sublane group's predicate (all off past
        the last tile).

        A pl.when that holds only a few DMA starts (see
        IN_BLOCK_GATHER_ROWS_PER_WHEN) becomes predicated DMAs in the
        enclosing block, so the descriptors stay in the compute block; the
        index loads stay outside the predicate (the ring is always in
        bounds) so that a row costs no more scalar work than an unguarded
        one. The DMA starts follow the lhs loads of this tile (the same
        buffer) and precede the w2 wait after them, so they issue between
        mm1's matmuls instead of in a scalar loop with the MXU idle.
        """
        chunk = c + depths.lhs_ahead
        last_chunk = jnp.minimum(chunk, num_gm - 1)
        num_rows = jnp.where(chunk < num_gm, _tile_gather_rows(last_chunk), 0)
        slot = lax.rem(chunk, depths.lhs_depth)
        _, _, row_offset = _idx_window_for_bucket(last_chunk, tile_m)
        idx_base = slot * idx_capacity + row_offset
        for group_start in range(0, tile_m, sublane):
            is_live = group_start < num_rows
            for first_row in range(
                group_start,
                group_start + sublane,
                IN_BLOCK_GATHER_ROWS_PER_WHEN,
            ):
                copies = [
                    _row_gather_copy(slot, idx_base + row, row)
                    for row in range(
                        first_row, first_row + IN_BLOCK_GATHER_ROWS_PER_WHEN
                    )
                ]

                @pl.when(is_live)
                def _(copies=copies):
                    for copy in copies:
                        copy.start()

    should_gather_per_bucket = (
        cfg.tail_bucket and cfg.gather_pipeline_rows is None
    )

    def _issue_lhs(chunk, slot):
        """Starts gm tile chunk's row gather into lhs slot.

        The weight-blocked path issues exactly bucket_m row DMAs, so the
        wait can use the bucket's byte count; rows past the tile's group (or
        past size_m, by the index clamp) are gathered but masked out in the
        epilogue.
        """

        def _issue_gather_rows(bucket_m: int):
            _, valid_rows = _lhs_rows_for_bucket(chunk, bucket_m)
            for should_clamp_rows, predicate in (
                (False, valid_rows == bucket_m),
                (True, valid_rows < bucket_m),
            ):

                @pl.when(predicate)
                def _(should_clamp_rows=should_clamp_rows):
                    _issue_gather_segment(
                        chunk,
                        slot,
                        bucket_m,
                        0,
                        bucket_m,
                        should_clamp_rows=should_clamp_rows,
                        should_fully_unroll=False,
                    )

        if cfg.weight_block_columns is None:
            _issue_tile_gather(chunk, slot, _tile_gather_rows(chunk))
        elif should_gather_per_bucket:
            bucket_m = metadata_ref.gm_id_to_tile_m[chunk]
            for bucket_m_static in cfg.tail_bucket_m_values:

                @pl.when(bucket_m == bucket_m_static)
                def _(bucket_m_static=bucket_m_static):
                    _issue_gather_rows(bucket_m_static)

        else:
            _issue_gather_rows(tile_m)

    def _wait_lhs(chunk, slot):
        def _wait_gather_rows(num_rows):
            buf = lhs_buf_ref.at[slot, pl.ds(0, num_rows)]
            pltpu.make_async_copy(buf, buf, lhs_sem.at[slot]).wait()

        if cfg.weight_block_columns is None:
            _wait_gather_rows(_tile_gather_rows(chunk))
        elif should_gather_per_bucket:
            bucket_m = metadata_ref.gm_id_to_tile_m[chunk]
            for bucket_m_static in cfg.tail_bucket_m_values:

                @pl.when(bucket_m == bucket_m_static)
                def _(bucket_m_static=bucket_m_static):
                    _wait_gather_rows(bucket_m_static)

        else:
            _wait_gather_rows(tile_m)

    def _issue_w1(ordinal):
        slot = lax.rem(ordinal, depths.w1_depth)
        group_id = metadata_ref.ord_to_group_id[ordinal]
        pltpu.make_async_copy(
            w1_ref.at[group_id], w1_buf_ref.at[slot], w1_sem.at[slot]
        ).start()

    def _issue_w2(ordinal):
        slot = lax.rem(ordinal, depths.w2_depth)
        group_id = metadata_ref.ord_to_group_id[ordinal]
        pltpu.make_async_copy(
            w2_ref.at[group_id], w2_buf_ref.at[slot], w2_sem.at[slot]
        ).start()

    def _wait_w1(slot):
        buf = w1_buf_ref.at[slot]
        pltpu.make_async_copy(buf, buf, w1_sem.at[slot]).wait()

    def _wait_w2(slot):
        buf = w2_buf_ref.at[slot]
        pltpu.make_async_copy(buf, buf, w2_sem.at[slot]).wait()

    # Note (david): one flattened block sequence w = gm_tile *
    # num_weight_blocks + block drives both the w1 and the w2 slices. Every gm
    # tile re-streams its expert's blocks, the price of weights larger than
    # VMEM.
    block_columns = cfg.weight_block_columns
    if block_columns is None:
        num_weight_blocks = total_blocks = None
    else:
        num_weight_blocks = dims.size_n // 2 // block_columns
        total_blocks = num_gm * num_weight_blocks

    def _issue_weight_block(block):
        """Starts weight block block's w1 and w2 loads: the gate and up
        column windows of the merged w1 into the two halves of one slot (one
        semaphore, waited by the slot's byte count), and the contiguous
        [block_columns, size_n_out] row slice of w2."""
        intermediate_size = dims.size_n // 2
        slot = lax.rem(block, WEIGHT_BLOCK_DEPTH)
        block_tile = block // num_weight_blocks
        block_in_tile = block - block_tile * num_weight_blocks
        group_id = metadata_ref.gm_id_to_group_id[block_tile]
        gate_col = block_in_tile * block_columns
        for half, col in enumerate((gate_col, gate_col + intermediate_size)):
            window = pl.ds(pl.multiple_of(col, NUM_LANES), block_columns)
            pltpu.make_async_copy(
                w1_ref.at[group_id, :, window],
                w1_buf_ref.at[
                    slot, :, pl.ds(half * block_columns, block_columns)
                ],
                w1_sem.at[slot],
            ).start()
        pltpu.make_async_copy(
            w2_ref.at[
                group_id, pl.ds(block_in_tile * block_columns, block_columns)
            ],
            w2_buf_ref.at[slot],
            w2_sem.at[slot],
        ).start()

    def _out_copy(chunk, out_slot):
        """(row_count, copy) of gm tile chunk's write-back.

        A tile other than the last stops at its last complete sublane block
        (the next tile rewrites the partial block after accumulating into
        it); the last tile rounds up. Issue and wait recompute the same byte
        count from the SMEM metadata.
        """
        m_start = metadata_ref.gm_id_to_m_offset[chunk]
        m_end = metadata_ref.gm_id_to_m_offset[chunk + 1]
        block_start = m_start // sublane
        block_end = jnp.where(
            chunk == num_gm - 1, pl.cdiv(m_end, sublane), m_end // sublane
        )
        row_start = pl.multiple_of(block_start * sublane, sublane)
        row_count = pl.multiple_of((block_end - block_start) * sublane, sublane)
        copy = pltpu.make_async_copy(
            out_buf_ref.at[out_slot, pl.ds(0, row_count)],
            out_ref.at[pl.ds(row_start, row_count)],
            out_sem.at[out_slot],
        )
        return row_count, copy

    def _wait_out(chunk, out_slot):
        row_count, copy = _out_copy(chunk, out_slot)

        @pl.when(row_count > 0)
        def _():
            copy.wait()

    for s in range(depths.lhs_ahead + 1):

        @pl.when(s < num_gm)
        def _(s=s):
            _issue_idx(s, s % depths.lhs_depth)

    for s in range(depths.lhs_ahead):

        @pl.when(s < num_gm)
        def _(s=s):
            _wait_idx(s, s % depths.lhs_depth)
            _issue_lhs(s, s)

    if block_columns is not None:
        for w in range(WEIGHT_BLOCK_DEPTH):

            @pl.when(w < total_blocks)
            def _(w=w):
                _issue_weight_block(w)

    else:
        for j in range(depths.w1_ahead):

            @pl.when(j < num_ord)
            def _(j=j):
                _issue_w1(j)

        for j in range(depths.w2_ahead):

            @pl.when(j < num_ord)
            def _(j=j):
                _issue_w2(j)

    # Note (david): the whole-expert path issues the next tile's gather inside
    # the compute of bucket bodies of at least in_block_min_rows rows (see
    # IN_BLOCK_GATHER_DIVISOR); with no lead the gather must land before its
    # own compute, so it stays at the top of the step.
    should_gather_in_block = (
        cfg.weight_block_columns is None and depths.lhs_ahead > 0
    )
    in_block_min_rows = tile_m // IN_BLOCK_GATHER_DIVISOR

    def _gm_step(c, cur_ord, *, should_pipeline_next: bool = False):
        lhs_slot = lax.rem(c, depths.lhs_depth)
        w1_slot = lax.rem(cur_ord, depths.w1_depth)
        w2_slot = lax.rem(cur_ord, depths.w2_depth)
        out_slot = lax.rem(c, depths.out_depth)

        group_id = metadata_ref.gm_id_to_group_id[c]
        prev_group_id = metadata_ref.gm_id_to_group_id[jnp.maximum(c - 1, 0)]
        is_first_of_group = jnp.logical_or(c == 0, group_id != prev_group_id)
        next_group_id = metadata_ref.gm_id_to_group_id[
            jnp.minimum(c + 1, num_gm - 1)
        ]
        is_last_of_group = jnp.logical_or(
            c == num_gm - 1, group_id != next_group_id
        )

        is_gather_pipelined = (
            cfg.gather_pipeline_rows is not None and should_pipeline_next
        )
        if is_gather_pipelined:
            next_chunk = c + 1
            next_slot = lax.rem(next_chunk, depths.lhs_depth)
        else:
            next_chunk = next_slot = None

        # Note (david): early issues, before any wait: the target slots' last
        # readers retired >= 2 steps ago, so the descriptor issue overlaps the
        # in-flight matmul. The index window of tile c + lhs_ahead + 1 goes a
        # tile ahead of its gather.
        @pl.when(c + depths.lhs_ahead + 1 < num_gm)
        def _():
            chunk = c + depths.lhs_ahead + 1
            _issue_idx(chunk, lax.rem(chunk, depths.lhs_depth))

        if not should_gather_in_block:
            should_issue_ahead_at_top = True
        elif cfg.tail_bucket:
            should_issue_ahead_at_top = (
                metadata_ref.gm_id_to_tile_m[c] < in_block_min_rows
            )
        else:
            should_issue_ahead_at_top = False

        if is_gather_pipelined:
            _wait_idx(next_chunk, next_slot)
        else:

            @pl.when(c + depths.lhs_ahead < num_gm)
            def _():
                chunk = c + depths.lhs_ahead
                _wait_idx(chunk, lax.rem(chunk, depths.lhs_depth))

                @pl.when(should_issue_ahead_at_top)
                def _():
                    _issue_lhs(chunk, lax.rem(chunk, depths.lhs_depth))

        def _issue_future_w2():
            @pl.when(
                jnp.logical_and(
                    is_first_of_group, cur_ord + depths.w2_ahead < num_ord
                )
            )
            def _():
                _issue_w2(cur_ord + depths.w2_ahead)

        if cfg.weight_block_columns is None:

            @pl.when(
                jnp.logical_and(
                    is_first_of_group, cur_ord + depths.w1_ahead < num_ord
                )
            )
            def _():
                _issue_w1(cur_ord + depths.w1_ahead)

            pl.when(should_issue_ahead_at_top)(_issue_future_w2)

        _wait_lhs(c, lhs_slot)

        if cfg.weight_block_columns is None:

            @pl.when(is_first_of_group)
            def _():
                _wait_w1(w1_slot)

        @pl.when(c >= depths.out_depth)
        def _():
            _wait_out(c - depths.out_depth, out_slot)

        def _compute_mlp_segment(row_count: int):
            lhs = lhs_buf_ref[lhs_slot, pl.ds(0, row_count)].reshape(
                -1, dims.size_k
            )
            # Note (david): gate and up as two matmuls, not one against the
            # merged w1. When the intermediate size is an odd multiple of
            # NUM_LANES, the merged result splits mid MXU tile and its
            # schedule stalls between mm1 and mm2; elsewhere the two take
            # the same MXU passes as one.
            intermediate_size = dims.size_n // 2
            gate, up = (
                jnp.matmul(
                    lhs,
                    w1_buf_ref[w1_slot, :, pl.ds(col, intermediate_size)],
                    preferred_element_type=jnp.float32,
                )
                for col in (0, intermediate_size)
            )
            # Note (david): the cast to bf16 is where a two-kernel path
            # stores the intermediate to HBM.
            intermediate = (activation_fn(gate) * up).astype(lhs_buf_ref.dtype)
            if should_gather_in_block and row_count >= in_block_min_rows:
                # Note (david): the DMA queue is first in, first out: the next
                # group's w2 goes after the rows, or the next step's lhs wait
                # would also wait out that w2 instead of overlapping it with
                # mm1.
                _issue_ahead_gather_in_block(c)
                _issue_future_w2()

            # Note (david): the deferred w2 wait gives the w2 DMA the whole
            # mm1 + activation window to land.
            @pl.when(is_first_of_group)
            def _():
                _wait_w2(w2_slot)

            return jnp.matmul(
                intermediate,
                w2_buf_ref[w2_slot],
                preferred_element_type=jnp.float32,
            )

        def _compute_mlp_blocked(
            bucket_m: int,
            *,
            next_chunk=None,
            next_slot=None,
            should_clamp_rows=False,
        ):
            """The weight-blocked gated MLP.

            Per block, mm1 against the [size_k, 2 * block_columns] gate|up
            slice collapses through the activation as in the single-block
            path (the K reduction stays whole, only mm1's n is narrower, so the
            intermediate values are unchanged), and mm2's
            [rows, block_columns] x [block_columns, size_n_out] partial
            accumulates into the resident f32 accumulator: written at
            block 0, added after, so it needs no zero fill.
            """
            if cfg.weight_block_segment_rows is None:
                segment_rows = bucket_m
            else:
                segment_rows = min(cfg.weight_block_segment_rows, bucket_m)
            base_block = c * num_weight_blocks
            if next_chunk is None:
                gather_schedule = None
            else:
                gather_schedule = get_weight_block_gather_pipeline_schedule(
                    tile_m, cfg.gather_pipeline_rows, sublane, num_weight_blocks
                )

            for j in range(num_weight_blocks):
                slot = lax.rem(base_block + j, WEIGHT_BLOCK_DEPTH)
                _wait_w1(slot)

                if gather_schedule is None:
                    is_last_segment = False
                else:
                    gather_row_start, gather_row_count, is_last_segment = (
                        gather_schedule[j]
                    )
                    _issue_gather_segment(
                        next_chunk,
                        next_slot,
                        tile_m,
                        gather_row_start,
                        gather_row_count,
                        should_clamp_rows=should_clamp_rows,
                        should_fully_unroll=True,
                    )
                    if is_last_segment:

                        @pl.when(base_block + num_weight_blocks < total_blocks)
                        def _():
                            _issue_weight_block(base_block + num_weight_blocks)

                w1_block = w1_buf_ref[slot]
                for segment, segment_start in enumerate(
                    range(0, bucket_m, segment_rows)
                ):
                    rows = min(segment_rows, bucket_m - segment_start)
                    lhs = lhs_buf_ref[
                        lhs_slot, pl.ds(segment_start, rows)
                    ].reshape(-1, dims.size_k)
                    gate_up = jnp.matmul(
                        lhs, w1_block, preferred_element_type=jnp.float32
                    )
                    gate = gate_up[:, :block_columns]
                    up = gate_up[:, block_columns:]
                    intermediate = (activation_fn(gate) * up).astype(
                        lhs_buf_ref.dtype
                    )
                    if segment == 0:
                        # Note (david): the deferred w2 wait gives the w2 DMA
                        # the whole first mm1 + activation window to land.
                        _wait_w2(slot)
                    partial = jnp.matmul(
                        intermediate,
                        w2_buf_ref[slot],
                        preferred_element_type=jnp.float32,
                    )
                    acc_slice = mm2_acc_ref.at[pl.ds(segment_start, rows)]
                    if j == 0:
                        acc_slice[...] = partial
                    else:
                        acc_slice[...] = acc_slice[...] + partial

                if gather_schedule is None:
                    # Note (david): block base_block + j's slot was just read,
                    # so block base_block + j + WEIGHT_BLOCK_DEPTH may land in
                    # it.
                    @pl.when(base_block + j + WEIGHT_BLOCK_DEPTH < total_blocks)
                    def _(j=j):
                        _issue_weight_block(base_block + j + WEIGHT_BLOCK_DEPTH)

                else:
                    # Note (david): the current tile's blocks keep their
                    # rolling lead. The next tile's blocks wait until every
                    # one of its gather descriptors is queued, which keeps the
                    # DMA FIFO order without starving its weight ring.
                    if j + WEIGHT_BLOCK_DEPTH < num_weight_blocks:

                        @pl.when(
                            base_block + j + WEIGHT_BLOCK_DEPTH < total_blocks
                        )
                        def _(j=j):
                            _issue_weight_block(
                                base_block + j + WEIGHT_BLOCK_DEPTH
                            )

                    if is_last_segment:

                        @pl.when(
                            base_block + num_weight_blocks + 1 < total_blocks
                        )
                        def _():
                            _issue_weight_block(
                                base_block + num_weight_blocks + 1
                            )

            return mm2_acc_ref[pl.ds(0, bucket_m)]

        def _scale_column(row_start: int, row_count: int):
            """Tile c's routing weights of rows [row_start, row_start +
            row_count) as a lane-replicated [row_count, NUM_LANES] f32
            column, built one f32 vreg of NUM_SUBLANES rows at a time from
            the SMEM window."""
            _, _, row_offset = _idx_window_for_bucket(c, tile_m)
            base = lhs_slot * idx_capacity + row_offset + row_start
            sublane_iota = lax.broadcasted_iota(
                jnp.int32, (NUM_SUBLANES, NUM_LANES), 0
            )
            groups = []
            for group_start in range(0, row_count, NUM_SUBLANES):
                column = jnp.full(
                    (NUM_SUBLANES, NUM_LANES),
                    scale_buf_ref[base + group_start],
                    jnp.float32,
                )
                for row in range(1, NUM_SUBLANES):
                    column = jnp.where(
                        sublane_iota == row,
                        scale_buf_ref[base + group_start + row],
                        column,
                    )
                groups.append(column)
            return jnp.concatenate(groups, axis=0)

        def _store_segment(row_start: int, row_count: int, acc):
            acc *= _scale_column(row_start, row_count)[:, :1]

            m_start = metadata_ref.gm_id_to_m_offset[c]
            m_end = metadata_ref.gm_id_to_m_offset[c + 1]
            m_offset = m_start - m_start % sublane
            iota = lax.broadcasted_iota(jnp.int32, acc.shape, 0) + row_start
            mask = jnp.logical_and(
                m_start - m_offset <= iota, iota < m_end - m_offset
            )
            # Note (david): cast before the packed-layout reshape: the f32 ->
            # bf16 rounding is the same either side, but relayouting in bf16
            # halves the vreg traffic of the lane-crossing pack.
            out_buf_ref[out_slot, pl.ds(row_start, row_count)] = (
                jnp.where(mask, acc, 0)
                .astype(out_buf_ref.dtype)
                .reshape(row_count, *out_tile_tail)
            )

        def _finish_store():
            """Adds the partial sublane block carried over from the previous
            tile, then stashes this tile's own partial rows.

            With sublane 4, | 0 0 1 2 | 2 2 2 2 | 3 3 4 4 | (group ids):
            group 1's tile does not fill its last block, which group 2's
            tile revisits, so the partial rows carry over; group 2 ends on a
            block boundary, so it clears the carry for group 3.
            """
            m_start = metadata_ref.gm_id_to_m_offset[c]
            m_end = metadata_ref.gm_id_to_m_offset[c + 1]
            m_end_local = m_end - (m_start - m_start % sublane)

            zeros = jnp.zeros_like(partial_out_ref)
            head = out_buf_ref.at[out_slot, pl.ds(0, sublane)]
            head[...] += jnp.where(c == 0, zeros, partial_out_ref[...]).reshape(
                sublane, *out_tile_tail
            )

            # Note (david): a tile that ends on a block boundary discards the
            # load below; the clamp keeps it inside the buffer when that
            # boundary is the buffer's end.
            last_block = jnp.minimum(
                m_end_local // sublane, tile_m // sublane - 1
            )
            last_rows = out_buf_ref[
                out_slot,
                pl.ds(pl.multiple_of(last_block * sublane, sublane), sublane),
            ]
            partial_out_ref[...] = jnp.where(
                m_end_local % sublane == 0,
                zeros,
                last_rows.reshape(partial_out_ref.shape),
            )

        def _compute_and_store(bucket_m: int):
            if cfg.weight_block_columns is None:
                acc = _compute_mlp_segment(bucket_m)
            else:
                acc = _compute_mlp_blocked(bucket_m)
            _store_segment(0, bucket_m, acc)

        if is_gather_pipelined:
            _, next_valid_rows = _lhs_rows_for_bucket(next_chunk, tile_m)
            for should_clamp_rows, predicate in (
                (False, next_valid_rows == tile_m),
                (True, next_valid_rows < tile_m),
            ):

                @pl.when(predicate)
                def _(should_clamp_rows=should_clamp_rows):
                    acc = _compute_mlp_blocked(
                        tile_m,
                        next_chunk=next_chunk,
                        next_slot=next_slot,
                        should_clamp_rows=should_clamp_rows,
                    )
                    _store_segment(0, tile_m, acc)

        elif cfg.tail_bucket:
            bucket_m = metadata_ref.gm_id_to_tile_m[c]
            for bucket_m_static in cfg.tail_bucket_m_values:

                @pl.when(bucket_m == bucket_m_static)
                def _(bucket_m_static=bucket_m_static):
                    _compute_and_store(bucket_m_static)

        else:
            _compute_and_store(tile_m)

        _finish_store()
        row_count, write_back = _out_copy(c, out_slot)

        @pl.when(row_count > 0)
        def _():
            write_back.start()

        return cur_ord + is_last_of_group.astype(jnp.int32)

    if cfg.gather_pipeline_rows is None:
        lax.fori_loop(0, num_gm, _gm_step, 0)
    elif cfg.tail_bucket:

        def _group_step(ordinal, cur_ord):
            group_start = metadata_ref.ord_to_gm_start[ordinal]
            group_end = metadata_ref.ord_to_gm_end[ordinal]
            # Note (david): a group's full tiles pipeline the next tile's
            # gather; its last tile when that is a smaller tail bucket, and
            # the table's last tile, run unpipelined.
            last_bucket_m = metadata_ref.gm_id_to_tile_m[group_end - 1]
            has_small_tail = last_bucket_m != tile_m
            full_end = group_end - jnp.where(has_small_tail, 1, 0)
            pipeline_end = jnp.minimum(full_end, jnp.maximum(num_gm - 1, 0))
            cur_ord = lax.fori_loop(
                group_start,
                pipeline_end,
                functools.partial(_gm_step, should_pipeline_next=True),
                cur_ord,
            )
            return lax.fori_loop(
                pipeline_end,
                group_end,
                functools.partial(_gm_step, should_pipeline_next=False),
                cur_ord,
            )

        lax.fori_loop(0, num_ord, _group_step, 0)
    else:
        cur_ord = lax.fori_loop(
            0,
            jnp.maximum(num_gm - 1, 0),
            functools.partial(_gm_step, should_pipeline_next=True),
            0,
        )

        @pl.when(num_gm > 0)
        def _():
            _gm_step(num_gm - 1, cur_ord, should_pipeline_next=False)

    for d in range(depths.out_depth, 0, -1):
        c = num_gm - d

        @pl.when(c >= 0)
        def _(c=c):
            _wait_out(c, lax.rem(c, depths.out_depth))

    # Note (david): one wait covers every lane chunk of the zero fill, by its
    # byte count.
    zeroed = out_ref.at[pl.ds(0, zero_rows)]
    pltpu.make_async_copy(zeroed, zeroed, zero_sem.at[0]).wait()


def zero_buffer_rows(dims: Dimensions) -> int:
    """Rows of the zero-fill buffer: ZERO_BUFFER_BYTES of [packing,
    NUM_LANES] bf16 rows, at most size_m."""
    row_bytes = dims.packing * NUM_LANES * BF16_BYTES
    return min(ZERO_BUFFER_BYTES // row_bytes, dims.size_m)


def spill_bytes(
    dims: Dimensions,
    *,
    tile_m: int,
    weight_block_columns: int | None,
    weight_block_segment_rows: int | None,
) -> int:
    """Upper bound of the register allocator's spill slots at one
    configuration (see SPILL_MARGIN_ROWS)."""
    rows = tile_m + SPILL_MARGIN_ROWS
    if weight_block_columns is None:
        columns = max(dims.size_k, dims.size_n_out) + dims.size_n // 2
        return rows * columns * F32_BYTES
    generation, _ = generation_and_mxu_columns()
    if generation not in UNALIGNED_OUTPUT_SPILL_TILES:
        raise ValueError(f"no spill model for TPU generation {generation}")
    if dims.size_n_out % OUTPUT_SPILL_ALIGNMENT == 0:
        output_tiles = 1
    else:
        output_tiles = UNALIGNED_OUTPUT_SPILL_TILES[generation]
    columns = max(
        output_tiles * dims.size_n_out, dims.size_k, 2 * weight_block_columns
    )
    spill = rows * columns * F32_BYTES
    if weight_block_segment_rows is not None:
        spill += dims.size_k * 2 * weight_block_columns * BF16_BYTES
    return spill


def vmem_bytes(
    dims: Dimensions,
    *,
    tile_m: int,
    pipeline_depths: PipelineDepths,
    weight_block_columns: int | None,
    weight_block_segment_rows: int | None,
) -> int:
    """The kernel's scoped VMEM at one configuration: its scratch buffers
    and the bound of the compiler's spill slots."""
    intermediate_size = dims.size_n // 2
    row_buffers = (
        tile_m
        * BF16_BYTES
        * (
            pipeline_depths.lhs_depth * dims.size_k
            + pipeline_depths.out_depth * dims.size_n_out
        )
    )
    fixed_buffers = BF16_BYTES * (
        dims.sublane * dims.size_n_out
        + zero_buffer_rows(dims) * dims.packing * NUM_LANES
    )
    if weight_block_columns is None:
        weights = BF16_BYTES * (
            pipeline_depths.w1_depth * dims.size_k * dims.size_n
            + pipeline_depths.w2_depth * intermediate_size * dims.size_n_out
        )
    else:
        weights = (
            WEIGHT_BLOCK_DEPTH
            * (2 * dims.size_k + dims.size_n_out)
            * weight_block_columns
            * BF16_BYTES
            + tile_m * dims.size_n_out * F32_BYTES
        )
    spill = spill_bytes(
        dims,
        tile_m=tile_m,
        weight_block_columns=weight_block_columns,
        weight_block_segment_rows=weight_block_segment_rows,
    )
    return row_buffers + fixed_buffers + weights + spill


def plan_weight_streams(
    dims: Dimensions,
    *,
    tile_m: int | None,
    pipeline_depths: PipelineDepths,
    weight_block_columns: int | None,
    weight_block_segment_rows: int | None,
) -> tuple[int | None, int]:
    """(weight_block_columns, tile_m) of the kernel; a given value is kept.

    Whole experts (weight_block_columns None) when two of w1 and of w2 fit VMEM
    at the default tile_m or a smaller one: every expert's weights are read
    once, prefetched a group ahead. Otherwise intermediate-dim blocks at the
    largest tile_m up to WEIGHT_BLOCK_TILE_M that fits, since every gm tile
    re-streams its expert's blocks, with the widest block that fits there,
    since every block adds a read-modify-write pass of the f32 accumulator
    over the tile.
    """
    limit = vmem_limit_bytes()
    config_vmem_bytes = functools.partial(
        vmem_bytes,
        dims,
        pipeline_depths=pipeline_depths,
        weight_block_segment_rows=weight_block_segment_rows,
    )

    def _fits(block_columns, candidate_tile_m):
        return limit >= config_vmem_bytes(
            tile_m=candidate_tile_m, weight_block_columns=block_columns
        )

    if tile_m is not None:
        whole_tiles = block_tiles = (tile_m,)
    else:
        largest_whole_tile = default_tile_m(dims.size_m, dims.num_experts)
        whole_tiles = tuple(
            range(largest_whole_tile, MIN_TILE_M - 1, -TILE_M_GRANULE)
        )
        block_tiles = tuple(
            range(WEIGHT_BLOCK_TILE_M, MIN_TILE_M - 1, -TILE_M_GRANULE)
        )
    if weight_block_columns is None:
        for candidate_tile_m in whole_tiles:
            if _fits(None, candidate_tile_m):
                return None, candidate_tile_m
        intermediate_size = dims.size_n // 2
        block_widths = tuple(
            block_columns
            for block_columns in range(intermediate_size, 0, -NUM_LANES)
            if intermediate_size % block_columns == 0
        )
    else:
        block_widths = (weight_block_columns,)
    for candidate_tile_m in block_tiles:
        for block_columns in block_widths:
            if _fits(block_columns, candidate_tile_m):
                return block_columns, candidate_tile_m
    smallest = config_vmem_bytes(
        tile_m=block_tiles[-1], weight_block_columns=block_widths[-1]
    )
    raise ValueError(
        f"no weight stream fits VMEM: the smallest, tile_m={block_tiles[-1]}"
        f" with weight_block_columns={block_widths[-1]}, takes {smallest} B,"
        f" over the limit {limit} B"
    )


def bucket_body_bundles(
    dims: Dimensions, bucket_m: int, *, tile_m: int, has_in_block_gather: bool
) -> float:
    """Estimated static bundles of the whole-expert body of bucket_m rows
    (see BucketBodyCodeModel)."""
    generation, mxu_columns = generation_and_mxu_columns()
    if generation not in BUCKET_BODY_CODE_MODELS:
        raise ValueError(
            f"no bucket body code model for TPU generation {generation}"
        )
    model = BUCKET_BODY_CODE_MODELS[generation]
    intermediate_tiles = pl.cdiv(dims.size_n // 2, mxu_columns)
    weight_tiles = intermediate_tiles * (
        2 * pl.cdiv(dims.size_k, mxu_columns)
        + pl.cdiv(dims.size_n_out, mxu_columns)
    )
    bundles = (
        model.fixed
        + weight_tiles * model.per_weight_tile
        + bucket_m * (model.per_row + weight_tiles * model.per_row_weight_tile)
    )
    if has_in_block_gather and bucket_m >= tile_m // IN_BLOCK_GATHER_DIVISOR:
        bundles += tile_m * model.per_in_block_gather_row
    return bundles


def fit_tail_buckets_to_instruction_memory(
    dims: Dimensions,
    candidate_m_values: tuple[int, ...],
    *,
    tile_m: int,
    has_in_block_gather: bool,
) -> tuple[int, ...]:
    """The candidate tail buckets, taken in order, whose bodies fit the
    instruction memory next to the tile_m body; sorted.

    Every step of the gm loop passes the branch of every bucket body, so the
    loop's overlays hold all of them: once they outgrow the instruction
    memory, every step reloads code from HBM.
    """
    body_bundles = functools.partial(
        bucket_body_bundles,
        dims,
        tile_m=tile_m,
        has_in_block_gather=has_in_block_gather,
    )
    free_bundles = (
        INSTRUCTION_MEMORY_BUNDLES
        - INSTRUCTION_MEMORY_RESERVE_BUNDLES
        - body_bundles(tile_m)
    )
    kept = []
    for bucket_m in candidate_m_values:
        bundles = body_bundles(bucket_m)
        if bundles <= free_bundles:
            kept.append(bucket_m)
            free_bundles -= bundles
    return tuple(sorted(kept))


def make_dimensions(
    x: jax.Array,
    token_indices: jax.Array,
    group_sizes: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    row_scale: jax.Array,
) -> Dimensions:
    """Checks the arrays against the kernel's envelope; returns their
    sizes."""
    if x.dtype != jnp.bfloat16 or x.ndim != 3:
        raise ValueError(
            "x must be bf16 in the packing-dim layout [num_tokens, 2,"
            f" hidden // 2], got {x.dtype}{list(x.shape)}"
        )
    if x.shape[1] != BF16_PACKING or x.shape[2] % NUM_LANES != 0:
        raise ValueError(
            f"x must be [num_tokens, {BF16_PACKING}, hidden // {BF16_PACKING}]"
            f" with hidden // {BF16_PACKING} a multiple of {NUM_LANES}, got"
            f" {list(x.shape)}"
        )
    size_k = BF16_PACKING * x.shape[2]

    if token_indices.ndim != 1 or token_indices.dtype != jnp.int32:
        raise ValueError(
            "token_indices must be int32[size_m], got"
            f" {token_indices.dtype}{list(token_indices.shape)}"
        )
    size_m = token_indices.shape[0]
    if size_m == 0 or size_m % ROW_ALIGNMENT != 0:
        raise ValueError(
            f"size_m must be a non-zero multiple of {ROW_ALIGNMENT} (the"
            f" token_indices window alignment), got {size_m}"
        )
    if row_scale.shape != (size_m,) or not jnp.issubdtype(
        row_scale.dtype, jnp.floating
    ):
        raise ValueError(
            f"row_scale must be float[size_m] = [{size_m}], got"
            f" {row_scale.dtype}{list(row_scale.shape)}"
        )

    if w1.dtype != jnp.bfloat16 or w2.dtype != jnp.bfloat16:
        raise ValueError(f"w1 and w2 must be bf16, got {w1.dtype}, {w2.dtype}")
    if w1.ndim != 3:
        raise ValueError(
            "w1 must be the merged gate|up weights [E, hidden, 2 *"
            f" intermediate], got {list(w1.shape)}"
        )
    num_experts, w1_k, size_n = w1.shape
    if w1_k != size_k:
        raise ValueError(f"w1's hidden dim {w1_k} != x's hidden {size_k}")
    if size_n % (2 * NUM_LANES) != 0:
        raise ValueError(
            f"the intermediate size must be a multiple of {NUM_LANES}, got"
            f" {size_n // 2}"
        )
    if w2.ndim != 3 or w2.shape[:2] != (num_experts, size_n // 2):
        raise ValueError(
            f"w2 must be [E, intermediate, hidden] = [{num_experts},"
            f" {size_n // 2}, ...], got {list(w2.shape)}"
        )
    size_n_out = w2.shape[2]
    if size_n_out % (BF16_PACKING * NUM_LANES) != 0:
        raise ValueError(
            "w2's output width must be a multiple of"
            f" {BF16_PACKING * NUM_LANES}, got {size_n_out}"
        )
    if group_sizes.shape != (num_experts,) or group_sizes.dtype != jnp.int32:
        raise ValueError(
            f"group_sizes must be int32[{num_experts}], got"
            f" {group_sizes.dtype}{list(group_sizes.shape)}"
        )
    return Dimensions(
        size_m=size_m,
        size_k=size_k,
        size_n=size_n,
        size_n_out=size_n_out,
        num_experts=num_experts,
        sublane=sublane_rows(),
        packing=BF16_PACKING,
    )


def make_config(
    dims: Dimensions,
    *,
    activation: str,
    tile_m: int | None,
    tail_bucket: bool,
    tail_bucket_m_values: tuple[int, ...] | None,
    pipeline_depths: PipelineDepths | None,
    weight_block_columns: int | None,
    weight_block_segment_rows: int | None,
    gather_pipeline_rows: int | None,
) -> FusedMlpConfig:
    """Checks the static knobs and resolves the planned ones: the weight
    stream, tile_m and the tail buckets."""
    if activation not in ACTIVATIONS:
        raise ValueError(
            f"activation must be one of {tuple(ACTIVATIONS)}, got"
            f" {activation!r}"
        )
    sublane = dims.sublane
    if tile_m is not None and (tile_m <= 0 or tile_m % sublane != 0):
        raise ValueError(
            f"tile_m={tile_m} must be a positive multiple of sublane={sublane}"
        )
    intermediate_size = dims.size_n // 2
    if weight_block_columns is not None and (
        weight_block_columns % NUM_LANES != 0
        or intermediate_size % weight_block_columns != 0
    ):
        raise ValueError(
            f"weight_block_columns={weight_block_columns} must be a multiple of"
            f" {NUM_LANES} that divides the intermediate size"
            f" {intermediate_size}"
        )
    if pipeline_depths is None:
        depths = DEFAULT_PIPELINE_DEPTHS
    else:
        depths = pipeline_depths
    planned_block_columns, planned_tile_m = plan_weight_streams(
        dims,
        tile_m=tile_m,
        pipeline_depths=depths,
        weight_block_columns=weight_block_columns,
        weight_block_segment_rows=weight_block_segment_rows,
    )
    if tail_bucket:
        if tail_bucket_m_values is not None:
            requested_m_values = tail_bucket_m_values
        else:
            candidate_m_values = default_tail_bucket_m_values(
                dims.size_m, dims.num_experts, planned_tile_m
            )
            if planned_block_columns is None:
                requested_m_values = fit_tail_buckets_to_instruction_memory(
                    dims,
                    candidate_m_values,
                    tile_m=planned_tile_m,
                    has_in_block_gather=depths.lhs_ahead > 0,
                )
            else:
                requested_m_values = tuple(sorted(candidate_m_values))
        bucket_m_values = validate_tail_bucket_m_values(
            tuple(v for v in requested_m_values if v < planned_tile_m)
            + (planned_tile_m,),
            tile_m=planned_tile_m,
            sublane=sublane,
        )
    else:
        bucket_m_values = (planned_tile_m,)
    for name, value in (
        ("weight_block_segment_rows", weight_block_segment_rows),
        ("gather_pipeline_rows", gather_pipeline_rows),
    ):
        if value is not None and planned_block_columns is None:
            raise ValueError(
                f"{name} requires the weight-blocked path, which these"
                " experts do not take"
            )
    if weight_block_segment_rows is not None and (
        weight_block_segment_rows <= 0
        or weight_block_segment_rows % sublane != 0
    ):
        raise ValueError(
            "weight_block_segment_rows must be a positive multiple of"
            f" sublane={sublane}, got {weight_block_segment_rows}"
        )
    if gather_pipeline_rows is not None:
        get_weight_block_gather_pipeline_schedule(
            planned_tile_m,
            gather_pipeline_rows,
            sublane,
            intermediate_size // planned_block_columns,
        )
        if depths.lhs_ahead != 1:
            raise ValueError(
                "gather_pipeline_rows requires pipeline_depths.lhs_ahead == 1"
            )

    return FusedMlpConfig(
        dims=dims,
        tile_m=planned_tile_m,
        activation=activation,
        tail_bucket=tail_bucket,
        tail_bucket_m_values=bucket_m_values,
        pipeline_depths=depths,
        weight_block_columns=planned_block_columns,
        weight_block_segment_rows=weight_block_segment_rows,
        gather_pipeline_rows=gather_pipeline_rows,
    )


def kernel_name(cfg: FusedMlpConfig) -> str:
    """The pallas_call name, which XProf shows: the shape and the knobs."""
    dims = cfg.dims
    name = (
        f"fused_moe_mlp-e_{dims.num_experts}-m_{dims.size_m}-k_{dims.size_k}"
        f"-n_{dims.size_n}-o_{dims.size_n_out}-tm_{cfg.tile_m}"
        f"-act_{cfg.activation}"
    )
    if cfg.tail_bucket:
        name += "-tail_bucket_" + "_".join(
            str(v) for v in cfg.tail_bucket_m_values
        )
    # Note (david): only non-default depths, so default kernel names stay
    # stable.
    if cfg.pipeline_depths != DEFAULT_PIPELINE_DEPTHS:
        name += f"-pipe_{cfg.pipeline_depths.short_name()}"
    if cfg.weight_block_columns is not None:
        name += f"-wb_{cfg.weight_block_columns}"
        if cfg.weight_block_segment_rows is not None:
            name += f"_segment_rows_{cfg.weight_block_segment_rows}"
    if cfg.gather_pipeline_rows is not None:
        name += f"-gp_{cfg.gather_pipeline_rows}"
    return name


def cost_estimate(cfg: FusedMlpConfig) -> pl.CostEstimate:
    dims = cfg.dims
    intermediate_size = dims.size_n // 2
    flops = (
        2
        * dims.size_m
        * (dims.size_k * dims.size_n + intermediate_size * dims.size_n_out)
    )
    # Note (david): the lhs stream reads size_m gathered rows, not the token
    # table.
    bytes_accessed = BF16_BYTES * (
        dims.size_m * dims.size_k
        + dims.num_experts
        * (dims.size_k * dims.size_n + intermediate_size * dims.size_n_out)
        + dims.size_m * dims.size_n_out
    )
    return pl.CostEstimate(
        flops=flops,
        bytes_accessed=bytes_accessed,
        transcendentals=dims.size_m * intermediate_size,
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "activation",
        "tile_m",
        "tail_bucket",
        "tail_bucket_m_values",
        "pipeline_depths",
        "weight_block_columns",
        "weight_block_segment_rows",
        "gather_pipeline_rows",
        "interpret",
    ),
)
def fused_mlp(
    x: jax.Array,
    token_indices: jax.Array,
    group_sizes: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    row_scale: jax.Array,
    *,
    activation: str = "silu",
    tile_m: int | None = None,
    tail_bucket: bool = False,
    tail_bucket_m_values: tuple[int, ...] | None = None,
    pipeline_depths: PipelineDepths | None = None,
    weight_block_columns: int | None = None,
    weight_block_segment_rows: int | None = None,
    gather_pipeline_rows: int | None = None,
    interpret: bool = False,
) -> jax.Array:
    """The gated expert MLP of expert-sorted rows gathered from x.

    Row i of the output is
    row_scale[i] * (act(x[t] @ gate) * (x[t] @ up)) @ w2[e], where
    t = token_indices[i], e is the group that row i falls in under
    group_sizes, and gate | up = w1[e]. Rows past sum(group_sizes) are zero.

    Args:
        x: bf16 activations in the packing-dim layout
            [num_tokens, 2, hidden // 2].
        token_indices: int32[size_m], the source token of each sorted row;
            size_m a multiple of 128.
        group_sizes: int32[E], the rows of each expert, in sorted order.
        w1: bf16 merged gate|up weights [E, hidden, 2 * intermediate].
        w2: bf16 down projection [E, intermediate, hidden_out].
        row_scale: float[size_m], the routing weight of each sorted row.
        activation: "silu" or "gelu" (tanh approximation).
        tile_m: Rows per gm tile; None lets plan_weight_streams pick it.
        tail_bucket: Run each expert's last tile at the smallest of
            tail_bucket_m_values that covers it, instead of at tile_m.
        tail_bucket_m_values: Sorted static row buckets; values at or
            above tile_m are dropped and tile_m is appended. None derives
            them from the rows per expert
            (config.default_tail_bucket_m_values), keeping on the
            whole-expert path those whose bodies fit the instruction memory.
        pipeline_depths: Stream depths and issue leads; None picks the
            defaults. Outputs do not depend on them.
        weight_block_columns: Force the weight-blocked path with blocks of this
            many intermediate columns (a multiple of 128 that divides
            intermediate). None lets plan_weight_streams pick whole experts
            or blocks from the VMEM they take.
        weight_block_segment_rows: With weight_block_columns, run each block
            over static segments of this many rows.
        gather_pipeline_rows: With weight_block_columns, issue the next tile's
            row gathers in segments of this many rows, one per weight block,
            between the current tile's block computes.
        interpret: Run the Pallas TPU interpreter, e.g. on CPU.

    Returns:
        bf16 [size_m, 2, hidden_out // 2], in the packing-dim layout.
    """
    dims = make_dimensions(x, token_indices, group_sizes, w1, w2, row_scale)
    cfg = make_config(
        dims,
        activation=activation,
        tile_m=tile_m,
        tail_bucket=tail_bucket,
        tail_bucket_m_values=tail_bucket_m_values,
        pipeline_depths=pipeline_depths,
        weight_block_columns=weight_block_columns,
        weight_block_segment_rows=weight_block_segment_rows,
        gather_pipeline_rows=gather_pipeline_rows,
    )
    depths, planned_tile_m = cfg.pipeline_depths, cfg.tile_m
    packed_out = (dims.packing, dims.size_n_out // dims.packing)

    row_scale_f32 = row_scale.astype(jnp.float32)

    max_num_gm = dims.num_experts + pl.cdiv(dims.size_m, planned_tile_m) - 1
    if cfg.weight_block_columns is not None:
        block_columns = cfg.weight_block_columns
        weight_scratch = [
            pltpu.VMEM(
                (WEIGHT_BLOCK_DEPTH, dims.size_k, 2 * block_columns), w1.dtype
            ),
            pltpu.SemaphoreType.DMA((WEIGHT_BLOCK_DEPTH,)),
            pltpu.VMEM(
                (WEIGHT_BLOCK_DEPTH, block_columns, dims.size_n_out), w2.dtype
            ),
            pltpu.SemaphoreType.DMA((WEIGHT_BLOCK_DEPTH,)),
        ]
        mm2_acc = pltpu.VMEM((planned_tile_m, dims.size_n_out), jnp.float32)
    else:
        weight_scratch = [
            pltpu.VMEM((depths.w1_depth, dims.size_k, dims.size_n), w1.dtype),
            pltpu.SemaphoreType.DMA((depths.w1_depth,)),
            pltpu.VMEM(
                (depths.w2_depth, dims.size_n // 2, dims.size_n_out), w2.dtype
            ),
            pltpu.SemaphoreType.DMA((depths.w2_depth,)),
        ]
        mm2_acc = None
    index_ring_size = depths.lhs_depth * index_window_capacity(planned_tile_m)
    scratch_shapes = [
        pltpu.VMEM((dims.sublane, dims.size_n_out), x.dtype),
        MetadataRef(
            gm_id_to_group_id=pltpu.SMEM((max_num_gm,), jnp.int32),
            gm_id_to_m_offset=pltpu.SMEM((max_num_gm + 1,), jnp.int32),
            gm_id_to_tile_m=pltpu.SMEM((max_num_gm,), jnp.int32),
            ord_to_group_id=pltpu.SMEM((dims.num_experts,), jnp.int32),
            ord_to_gm_start=pltpu.SMEM((dims.num_experts,), jnp.int32),
            ord_to_gm_end=pltpu.SMEM((dims.num_experts,), jnp.int32),
        ),
        pltpu.VMEM((depths.lhs_depth, planned_tile_m, *x.shape[1:]), x.dtype),
        pltpu.SemaphoreType.DMA((depths.lhs_depth,)),
        *weight_scratch,
        pltpu.VMEM((depths.out_depth, planned_tile_m, *packed_out), x.dtype),
        pltpu.SemaphoreType.DMA((depths.out_depth,)),
        pltpu.SMEM((index_ring_size,), jnp.float32),
        pltpu.SemaphoreType.DMA((depths.lhs_depth,)),
        pltpu.VMEM((zero_buffer_rows(dims), dims.packing, NUM_LANES), x.dtype),
        pltpu.SemaphoreType.DMA((1,)),
        # Note (david): a small per-tile SMEM ring instead of the whole
        # int32[size_m]; a window starts up to 127 elements before its tile.
        pltpu.SMEM((index_ring_size,), jnp.int32),
        pltpu.SemaphoreType.DMA((depths.lhs_depth,)),
        mm2_acc,
    ]

    inputs = (x, token_indices, w1, w2, row_scale_f32)
    return pl.pallas_call(
        functools.partial(fused_mlp_kernel, cfg=cfg),
        out_shape=jax.ShapeDtypeStruct((dims.size_m, *packed_out), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM) for _ in inputs],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_limit_bytes(), disable_bounds_checks=True
        ),
        interpret=pltpu.InterpretParams() if interpret else False,
        name=kernel_name(cfg),
        cost_estimate=cost_estimate(cfg),
    )(group_sizes, *inputs)
