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

"""Packed varlen forward main loop on the explicit MXU pipeline.

A build with traits.uses_mxu_pipeline runs varlen_mxu_fwd_body in place of
kernels/fwd_varlen.py's varlen_fwd_body. A pipeline head stacks the q heads
of one kv head, which share every K and V tile. A row runs its live kv tiles
in a start segment, then loop bodies; its last tile stays in flight into the
next row's start segment, so rows alternate between NUM_ROW_PARITIES row
states. Under a left window bound a row starts in its pivot tile, which
every kept row sees. The overflow guard checks a row by check units of
UNIT_TILES tiles and replays a flagged unit from its checkpoint.
"""

import dataclasses
import math
from collections.abc import Callable, Sequence
from functools import cached_property, partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from flywheel_tpu.core.tiling import round_up
from flywheel_tpu.core.tpu import BF16_TILE_ROWS, MXU_COLUMNS, NUM_LANES
from flywheel_tpu.softmax_attention.block_stream import (
    BlockDesc,
    RowVisits,
    VarlenBlockStream,
)
from flywheel_tpu.softmax_attention.config import BlockSizes, QKVLayout
from flywheel_tpu.softmax_attention.dma import drain
from flywheel_tpu.softmax_attention.fwd_blocks import (
    NUM_KV_STAGES,
    NUM_SEMS,
    KVRefs,
    OutRefs,
    QRefs,
    RowFinish,
    copy_row_lse,
    copy_row_out,
    fill_packed_tail,
    o_copy,
    row_out_slot,
    start_kv_block,
    start_q_row,
    wait_rotated_kv_block,
    wait_rotated_q_row,
    wait_row_out_slot,
)
from flywheel_tpu.softmax_attention.fwd_call import staged_shape
from flywheel_tpu.softmax_attention.fwd_mxu import (
    ENTRY_PARITY,
    LOOP_BODY_TILES,
    NUM_CHECKPOINTS,
    NUM_ROW_PARITIES,
    QUEUED_GROUPS,
    REPLAY_RESULT_SLOTS,
    ROW_START_TILES,
    UNIT_TILES,
    FlaggedTailOps,
    LargestSums,
    RowCheckpoints,
    TileBounds,
    UnitCheckpoint,
    accumulate_fragment,
    aligned_rows,
    first_final_unit,
    fold_fragment_scores,
    fold_replayed_fragment,
    fold_unit_end_scores,
    for_each_later_row,
    fragment_rows,
    group_capacity,
    guard_flag,
    masked_logits,
    mxu_pipeline_applies,
    per_seq_geometry,
    pipeline_shape,
    pivot_kept_rows,
    pivot_order,
    replay_matmul,
    row_finish_state,
    row_tiles,
    stage_row_finish,
    tile_bounds,
    zeroes_row_start,
)
from flywheel_tpu.softmax_attention.inputs import pad_axis
from flywheel_tpu.softmax_attention.mxu_pipeline import (
    FRAGMENT_ROWS,
    NUM_SLOTS,
    FragmentOps,
    MxuPipeline,
    PipelineShape,
    PipelineState,
    Tail,
    WorkItem,
)
from flywheel_tpu.softmax_attention.schedule import (
    FwdSchedule,
    MxuRowTiles,
    ScheduleRow,
    VarlenSchedule,
)
from flywheel_tpu.softmax_attention.traits import FwdTraits
from flywheel_tpu.softmax_attention.variant import Variant, VariantOperands

# The row finish normalizes at most this many rows a step.
FINISH_STEP_ROWS = 256
# A row group's later rows read their tiles from its first row's kv block, at
# offsets on the bf16 tile grid.
TILE_START_ROWS = BF16_TILE_ROWS
# The schedule checks each row of a group only against the row before it
# (schedule.varlen_schedule_table), so a group holds at most two rows.
MAX_VARLEN_GROUP_ROWS = 2
# A key index past every packed kv axis: a tile from it hides every key.
DEAD_KEY = 1 << 30


def varlen_uses_mxu_pipeline(
    *,
    block_sizes: BlockSizes,
    head_dim: int,
    head_dim_v: int,
    transposed_pv: bool,
) -> bool:
    """Whether a varlen build runs on the explicit MXU pipeline."""
    return mxu_pipeline_applies(
        block_sizes=block_sizes,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        transposed_pv=transposed_pv,
    )


def varlen_gqa_fold(q_heads_per_kv_head: int, is_token_major: bool) -> int:
    """The q heads a pipeline head stacks: those of one kv head, or one on a
    token-major Q, whose heads ride the lanes of one staged row."""
    return 1 if is_token_major else q_heads_per_kv_head


def varlen_group_rows(fragments_per_tile: int, shape: PipelineShape) -> int:
    """The most rows of a row group of an explicit varlen build on a
    pipeline of shape."""
    return min(group_capacity(fragments_per_tile, shape), MAX_VARLEN_GROUP_ROWS)


def varlen_pipeline_shape(traits: FwdTraits) -> PipelineShape:
    """The pipeline of the build's rows, whose start segment runs inside
    one kv block."""
    return pipeline_shape(
        traits.group_bq // FRAGMENT_ROWS, traits.num_k_compute_tiles
    )


def varlen_mxu_groups(traits: FwdTraits) -> tuple[MxuRowTiles, int]:
    """(row tiles, group capacity) of the build's rows, for its schedule."""
    fragments_per_tile = traits.group_bq // FRAGMENT_ROWS
    shape = varlen_pipeline_shape(traits)
    return (
        row_tiles(fragments_per_tile, shape),
        varlen_group_rows(fragments_per_tile, shape),
    )


def explicit_block_kv(block_kv: int, fragments_per_tile: int) -> int:
    """block_kv in whole MXU tiles, at least ROW_START_TILES, padded to
    whole loop bodies of the pipeline it takes (fwd_mxu.pipeline_shape)."""
    num_tiles = round_up(
        max(-(-block_kv // MXU_COLUMNS), ROW_START_TILES), LOOP_BODY_TILES
    )
    tiles = row_tiles(
        fragments_per_tile, pipeline_shape(fragments_per_tile, num_tiles)
    )
    return round_up(num_tiles, tiles.body_tiles) * MXU_COLUMNS


def varlen_mxu_block_sizes(
    block_sizes: BlockSizes,
    gqa_fold: int,
    window: tuple[int | None, int | None],
) -> BlockSizes:
    """The blocks of an explicit varlen build whose pipeline head stacks
    gqa_fold q heads; block_q counts tokens per head and, under a window
    bounded on both sides, follows the rows a block keeps. block_q_compute
    is the row finish's step."""
    block_q = round_up(block_sizes.block_q, FRAGMENT_ROWS)
    max_kept_rows = pivot_kept_rows(window)
    if max_kept_rows is not None:
        # The smallest per-head block whose folded rows fill whole laps.
        lap_rows = NUM_SLOTS // math.gcd(NUM_SLOTS, gqa_fold) * FRAGMENT_ROWS
        block_q = max(
            min(block_q, max_kept_rows // lap_rows * lap_rows), lap_rows
        )
    block_q_compute = FRAGMENT_ROWS
    while (
        2 * block_q_compute * gqa_fold <= FINISH_STEP_ROWS
        and block_q % (2 * block_q_compute) == 0
    ):
        block_q_compute *= 2
    fragments_per_tile = gqa_fold * block_q // FRAGMENT_ROWS
    block_kv = explicit_block_kv(block_sizes.block_kv, fragments_per_tile)
    shape = pipeline_shape(fragments_per_tile, block_kv // MXU_COLUMNS)
    return BlockSizes(
        block_q=block_q,
        block_kv=block_kv,
        block_kv_compute=MXU_COLUMNS,
        block_q_compute=block_q_compute,
        num_stages=QUEUED_GROUPS * varlen_group_rows(fragments_per_tile, shape),
        qkv_layout=QKVLayout.HEAD_DIM_MINOR,
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class FoldedVarlenTraits(FwdTraits):
    """A varlen build whose pipeline head stacks the q heads of one kv head."""

    gqa_fold: int = dataclasses.field()

    def __post_init__(self) -> None:
        assert self.head_fold == 1, self.head_fold
        assert self.num_q_heads % (self.num_kv_heads * self.gqa_fold) == 0
        super().__post_init__()


def fold_q_heads(traits: FwdTraits) -> FoldedVarlenTraits:
    """traits with each kv head's q heads stacked in one pipeline head."""
    fields = {
        field.name: getattr(traits, field.name)
        for field in dataclasses.fields(traits)
        if field.init
    }
    fields["head_fold"] = 1
    return FoldedVarlenTraits(
        **fields,
        gqa_fold=varlen_gqa_fold(
            traits.q_heads_per_kv_head, traits.token_major is not None
        ),
    )


def chunk_rows(
    traits: FwdTraits, chunk: jax.Array | int, head: jax.Array | int
) -> pl.Slice:
    """The row-state rows of chunk chunk of folded head head; rows run (q
    compute tile, folded head, token), as the row finish reads them."""
    token = chunk * FRAGMENT_ROWS
    q_tile = token // traits.bq_compute
    first_row = (
        q_tile * traits.frag_rows
        + head * traits.bq_compute
        + token % traits.bq_compute
    )
    return pl.ds(aligned_rows(first_row), FRAGMENT_ROWS)


class TileMask(NamedTuple):
    """What masks a kv tile: its kv block and its first packed key."""

    kv: jax.Array
    first_key: jax.Array


def row_tile_bounds(
    traits: FwdTraits, schedule: FwdSchedule, ctx: ScheduleRow, mask: TileMask
) -> TileBounds:
    """The bounds of a tile of the row of ctx, rows counted from the row's
    first token. Keys below the block's own first belong to the block before."""
    bounds = tile_bounds(
        per_seq_geometry(schedule, ctx),
        mask.first_key,
        owned_key=ctx.kv_base + mask.kv * traits.bkv,
    )
    return bounds.from_row(ctx.q_tok)


def staged_q_head(
    traits: FwdTraits, q_slot_ref: jax.Array, head: jax.Array | int
) -> jax.Array:
    """Folded head head's (bq, head_dim) rows of a Q staging slot."""
    if traits.group_heads == 1:
        return q_slot_ref
    return q_slot_ref.at[head]


def fill_padding_tiles(
    traits: FwdTraits,
    packed_q_end: jax.Array,
    stage_ref: jax.Array,
    finish: RowFinish,
    fill_value: float,
) -> None:
    """fwd_blocks.fill_packed_padding as a loop over the stage slot's tiles."""
    # Note (david): the compiler predicates fill_packed_padding's whole-slot
    # select into every row's finish; a loop of no step costs next to nothing.
    stage_slot = stage_ref.at[lax.rem(finish.out_count, traits.num_stages)]
    num_tiles = traits.bq // FRAGMENT_ROWS
    first_tile = jnp.clip(
        (packed_q_end - finish.q_tok) // FRAGMENT_ROWS, 0, num_tiles
    )
    leading = (slice(None),) * (len(stage_slot.shape) - 2)

    def _fill_tile(tile, carry):
        first_row = pl.multiple_of(tile * FRAGMENT_ROWS, FRAGMENT_ROWS)
        index = (*leading, pl.ds(first_row, FRAGMENT_ROWS), slice(None))
        staged = stage_slot[index]
        rows = (
            finish.q_tok
            + first_row
            + lax.broadcasted_iota(jnp.int32, staged.shape, staged.ndim - 2)
        )
        stage_slot[index] = jnp.where(
            rows < packed_q_end, staged, jnp.asarray(fill_value, staged.dtype)
        )
        return carry

    lax.fori_loop(first_tile, num_tiles, _fill_tile, None)


class TileRef(NamedTuple):
    """A segment's kv tile: its staging slot, first key row in it, and mask."""

    slot: jax.Array
    start: jax.Array
    mask: TileMask


class TileMember(NamedTuple):
    """The row of a tile key: its Q slot, schedule row and row state refs."""

    q_slot: jax.Array | int
    ctx: ScheduleRow
    row_max: jax.Array
    row_sum: jax.Array
    o: jax.Array


def tile_rows(start: jax.Array) -> pl.Slice:
    """The MXU_COLUMNS key rows of a staged block from start on."""
    return pl.ds(pl.multiple_of(start, TILE_START_ROWS), MXU_COLUMNS)


@dataclasses.dataclass(frozen=True, eq=False)
class VarlenFragmentOps:
    """The FragmentOps of a segment of the row of ctx. tiles places each tile
    key, and members names its row in a segment of several rows. row_starts
    says per tile key whether the tile starts its row; empty, none does."""

    traits: FwdTraits
    variant: Variant
    schedule: FwdSchedule
    ctx: ScheduleRow
    q_buf: jax.Array
    q_slot: jax.Array
    k_buf: jax.Array
    v_buf: jax.Array
    row_max: jax.Array
    row_sum: jax.Array
    o: jax.Array
    tiles: tuple[TileRef, ...]
    unit_end: UnitCheckpoint | None = None
    members: tuple[TileMember, ...] = ()
    row_starts: tuple[bool | jax.Array, ...] = ()

    def member(self, tile: int) -> TileMember:
        """The row of tile key tile."""
        if self.members:
            return self.members[tile]
        return TileMember(
            self.q_slot, self.ctx, self.row_max, self.row_sum, self.o
        )

    def is_row_start(self, tile: int) -> bool | jax.Array:
        return bool(self.row_starts) and self.row_starts[tile]

    def _tile_rows(self, tile: int) -> tuple[jax.Array, pl.Slice]:
        ref = self.tiles[tile]
        return ref.slot, tile_rows(ref.start)

    # A V handle: the tile's staging slot and first key row in it.
    v_handle_words = 2

    def v_handle(self, tile: int) -> tuple[jax.Array, ...]:
        ref = self.tiles[tile]
        return (ref.slot, ref.start)

    def v_tile_at(self, handle: tuple[jax.Array, ...]) -> jax.Array:
        slot, start = handle
        return self.v_buf[slot, tile_rows(start), :]

    def k_tile(self, tile: int) -> jax.Array:
        slot, rows = self._tile_rows(tile)
        return self.k_buf[slot, rows, :]

    def v_tile(self, tile: int) -> jax.Array:
        slot, rows = self._tile_rows(tile)
        return self.v_buf[slot, rows, :]

    def q_fragment(self, item: WorkItem) -> jax.Array:
        chunk, head = item.fragment
        q_slot = self.member(item.tile).q_slot
        q_ref = staged_q_head(self.traits, self.q_buf.at[q_slot], head)
        return q_ref[fragment_rows(chunk), :]

    @cached_property
    def masks(self) -> tuple[TileBounds, ...]:
        """Each tile's bounds; with one q head a pipeline head they count
        from each fragment row. The cache holds values of the trace that
        first reads it, so each branch of a cond builds its own ops."""
        bounds = tuple(
            row_tile_bounds(
                self.traits, self.schedule, self.member(key).ctx, tile.mask
            )
            for key, tile in enumerate(self.tiles)
        )
        if self.traits.gqa_fold > 1:
            return bounds
        return tuple(tile.from_fragment_rows() for tile in bounds)

    def rows(self, item: WorkItem) -> pl.Slice:
        """The row-state rows of item's fragment."""
        chunk, head = item.fragment
        return chunk_rows(self.traits, chunk, head)

    def masked_scores(self, item: WorkItem, scores: jax.Array) -> jax.Array:
        """item's raw scores as masked log2-unit logits."""
        chunk, _ = item.fragment
        return masked_logits(
            self.variant, scores, self.masks[item.tile], chunk * FRAGMENT_ROWS
        )

    def takes_anchor(self, item: WorkItem) -> None:
        """None: every row takes its anchor at its row start."""
        del item

    def is_unit_end(self, item: WorkItem) -> bool:
        return self.unit_end is not None and item.tile == self.unit_end.tile

    def softmax(self, item: WorkItem, scores: jax.Array) -> jax.Array:
        rows, masked = self.rows(item), self.masked_scores(item, scores)
        member = self.member(item.tile)
        if not self.is_unit_end(item):
            return fold_fragment_scores(
                member.row_max,
                member.row_sum,
                rows,
                masked,
                is_row_start=self.is_row_start(item.tile),
            )
        assert self.is_row_start(item.tile) is False, item
        return fold_unit_end_scores(
            member.row_max, member.row_sum, rows, masked, self.unit_end
        )

    def accumulate(self, item: WorkItem, weighted_values: jax.Array) -> None:
        accumulate_fragment(
            self.member(item.tile).o,
            self.rows(item),
            weighted_values,
            is_row_start=self.is_row_start(item.tile),
            checkpoint_o=self.unit_end.o if self.is_unit_end(item) else None,
        )


class KVStream(NamedTuple):
    """The kv ring while block j runs: the slots of blocks j, j + 1 and
    j + 2, whether j + 2 is in flight, and the descriptors of j + 1, j + 2."""

    slot: jax.Array
    next_slot: jax.Array
    ahead_slot: jax.Array
    is_ahead_fresh: jax.Array
    next_desc: BlockDesc
    ahead_desc: BlockDesc


# What a tile segment hands to the next: the pipeline's carried values, the kv
# stream and the masks of the segment's last unit of tiles.
TileCarry = tuple[tuple[jax.Array, ...], KVStream, tuple[TileMask, ...]]


class RowSpan(NamedTuple):
    """A row's live kv tiles in run order (VarlenMxuLoop.tile_place).
    pivot_block counts from lo, num_pivot_tiles of its tiles run first, and
    next_row_tile is the next row's first tile."""

    lo: jax.Array
    first_tile: jax.Array
    num_tiles: jax.Array
    pivot_block: jax.Array
    pivot_tile: jax.Array
    num_pivot_tiles: jax.Array
    next_row_tile: jax.Array


def pivot_block(
    traits: FwdTraits, lo: jax.Array, hi: jax.Array, ctx: ScheduleRow
) -> jax.Array:
    """The kv block a row visits first under a left window bound: the one
    with the first key its last kept row sees, which every kept row sees."""
    return jnp.clip((ctx.last_anchor_key - ctx.kv_base) // traits.bkv, lo, hi)


def pivot_visits(
    traits: FwdTraits, lo: jax.Array, hi: jax.Array, ctx: ScheduleRow
) -> RowVisits:
    """A row's blocks under a left window bound: pivot, down to lo, up to hi."""
    pivot = pivot_block(traits, lo, hi, ctx) - lo
    return RowVisits(hi - lo + 1, lambda visit: lo + pivot_order(pivot, visit))


class GridRow(NamedTuple):
    """A grid row: its group's first head, last kv block and first q token."""

    base_head: jax.Array
    hi: jax.Array
    ctx: ScheduleRow
    q_tok: jax.Array


class RowRun(NamedTuple):
    """A row while its tiles run: its Q slot, row-state index, live tiles
    and grid row, and the first two block descriptors of the next group."""

    q_slot: jax.Array
    state: jax.Array
    span: RowSpan
    row: GridRow
    ahead: tuple[BlockDesc, BlockDesc]


class FinishedGroup(NamedTuple):
    """A row group whose last tiles are in flight when the next group starts.
    is_live is 0 for the entry tile's rows; is_short, its rows share its
    start segment; prev_out_tok, the first token of the row written before."""

    is_live: jax.Array
    base_head: jax.Array
    first_si: jax.Array
    num_rows: jax.Array
    is_short: jax.Array
    parity: jax.Array
    q_count: jax.Array
    out_count: jax.Array
    prev_out_tok: jax.Array


class VarlenRowCarry(NamedTuple):
    """What a row group hands to the next. finished is the group in flight,
    last_mask the masks of a lone row's last unit of tiles, prev_out_tok the
    last row's first token and num_rows the output copies started."""

    q_count: jax.Array
    stream: KVStream
    probs: tuple[jax.Array, ...]
    finished: FinishedGroup
    last_mask: tuple[TileMask, ...]
    prev_out_tok: jax.Array
    num_rows: jax.Array


@dataclasses.dataclass(frozen=True, eq=False)
class ReplayRefs:
    """The guard replay's own staging: one Q row and one kv block."""

    q_refs: QRefs
    kv_refs: KVRefs


class ReplayedRow(NamedTuple):
    """A row as a replay recomputes it: its staged Q (slot q_slot of q_buf),
    row-state index, live tiles, schedule row and kv head."""

    q_buf: jax.Array
    q_slot: jax.Array | int
    state: jax.Array
    span: RowSpan
    ctx: ScheduleRow
    kv_head: jax.Array


@dataclasses.dataclass(frozen=True, eq=False)
class VarlenMxuLoop:
    """The explicit loop's static context and its steps. row_max, row_sum
    and o hold the row states of NUM_ROW_PARITIES parities; checkpoint_sum
    and checkpoint_o a check unit's entry state. In SMEM, guard is a row's
    overflow flag and unit_guard a check unit's."""

    traits: FwdTraits
    variant: Variant
    operands: VariantOperands
    schedule: FwdSchedule
    pipeline: MxuPipeline
    blocks: VarlenBlockStream
    q_refs: QRefs
    kv_refs: KVRefs
    out_refs: OutRefs
    row_max: jax.Array
    row_sum: jax.Array
    o: jax.Array
    checkpoint_sum: jax.Array
    checkpoint_o: jax.Array
    guard: jax.Array
    unit_guard: jax.Array
    replay: ReplayRefs

    @property
    def num_block_tiles(self) -> int:
        return self.traits.num_k_compute_tiles

    @property
    def num_chunks(self) -> int:
        """The chunks of FRAGMENT_ROWS tokens of a q block."""
        return self.traits.bq // FRAGMENT_ROWS

    @property
    def shape(self) -> PipelineShape:
        """The pipeline's delays and slot ring."""
        return varlen_pipeline_shape(self.traits)

    @property
    def tiles(self) -> MxuRowTiles:
        """How a row groups its tiles."""
        return row_tiles(self.traits.group_bq // FRAGMENT_ROWS, self.shape)

    @property
    def has_half_tiles(self) -> bool:
        """Whether a tile may run its second half's chunks alone: half its
        items still fill whole slot ring laps and a segment."""
        half_items = self.num_chunks // 2 * self.traits.gqa_fold
        return (
            self.num_chunks % 2 == 0
            and half_items % self.pipeline.num_slots == 0
            and half_items >= self.pipeline.delays.pv_pop
            and self.tiles.unit_tiles == 1
            and self.tiles.start_tiles == ROW_START_TILES
        )

    def stream_tiles(self, num_tiles: jax.Array) -> jax.Array:
        """The tiles a row of num_tiles live tiles runs: whole units, at
        least a start segment; those past its live tiles hide every key."""
        tiles = self.tiles
        unit = tiles.unit_tiles
        padded = (num_tiles + unit - 1) // unit * unit
        return jnp.maximum(padded, tiles.start_tiles)

    @property
    def has_left_window(self) -> bool:
        """Whether a row starts in its pivot tile."""
        return self.schedule.window[0] is not None

    def row_span(
        self, lo: jax.Array, hi: jax.Array, ctx: ScheduleRow
    ) -> tuple[jax.Array, ...]:
        """RowSpan's fields from first_tile to num_pivot_tiles of the row of
        ctx over kv blocks lo..hi. A pivot block that other blocks follow is
        padded to whole loop bodies, at least a start segment."""
        tiles, tiles_a_block = self.tiles, self.num_block_tiles
        first_tile = self.first_live_tile(lo, hi, ctx)
        num_tiles = (
            (hi - lo) * tiles_a_block
            + self.schedule.num_live_tiles(hi, ctx)
            - first_tile
        )
        if not self.has_left_window:
            return first_tile, num_tiles, jnp.int32(0), first_tile, num_tiles
        pivot_kv = pivot_block(self.traits, lo, hi, ctx)
        block_first = jnp.where(pivot_kv == lo, first_tile, 0)
        block_end = jnp.where(
            pivot_kv == hi, self.schedule.num_live_tiles(hi, ctx), tiles_a_block
        )
        key_tile = (
            ctx.last_anchor_key - self.schedule.kv_token(pivot_kv, ctx)
        ) // MXU_COLUMNS
        num_pivot_live = block_end - block_first
        body = tiles.body_tiles
        num_padded = jnp.maximum(
            (num_pivot_live + body - 1) // body * body, tiles.start_tiles
        )
        num_padding = jnp.where(
            jnp.logical_and(pivot_kv == hi, lo < hi),
            num_padded - num_pivot_live,
            0,
        )
        return (
            first_tile,
            num_tiles + num_padding,
            pivot_kv - lo,
            jnp.clip(key_tile, block_first, block_end - 1),
            num_pivot_live + num_padding,
        )

    def tile_place(
        self, span: RowSpan, index: jax.Array | int
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """(visit, kv block, tile in it) of tile index of the row of span in
        run order: the pivot block's tiles (fwd_mxu.pivot_order), the blocks
        below it down to lo, then those above it."""
        tiles_a_block = self.num_block_tiles
        if not self.has_left_window:
            position = span.first_tile + index
            block = position // tiles_a_block
            return block, span.lo + block, lax.rem(position, tiles_a_block)
        pivot = span.pivot_block
        pivot_first = jnp.where(pivot == 0, span.first_tile, 0)
        pivot_tile = pivot_first + pivot_order(
            span.pivot_tile - pivot_first, index
        )
        past = index - span.num_pivot_tiles
        num_lower_tiles = jnp.maximum(
            pivot * tiles_a_block - span.first_tile, 0
        )
        lower_block = pivot - 1 - past // tiles_a_block
        lower_tile = lax.rem(past, tiles_a_block) + jnp.where(
            lower_block == 0, span.first_tile, 0
        )
        upper_position = (pivot + 1) * tiles_a_block + past - num_lower_tiles
        upper_block = upper_position // tiles_a_block
        is_pivot = index < span.num_pivot_tiles
        is_lower = past < num_lower_tiles
        block = jnp.where(
            is_pivot, pivot, jnp.where(is_lower, lower_block, upper_block)
        )
        tile = jnp.where(
            is_pivot,
            pivot_tile,
            jnp.where(
                is_lower, lower_tile, lax.rem(upper_position, tiles_a_block)
            ),
        )
        visit = jnp.where(
            is_pivot,
            0,
            jnp.where(is_lower, pivot - lower_block, upper_block),
        )
        return visit, span.lo + block, tile

    def live_tile_place(
        self, span: RowSpan, index: jax.Array | int
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """tile_place of tile index, clamped to the row's last tile."""
        return self.tile_place(span, jnp.minimum(index, span.num_tiles - 1))

    def visit_start(self, span: RowSpan, visit: jax.Array) -> jax.Array:
        """The index of the first tile visit visit of the row of span runs."""
        tiles_a_block = self.num_block_tiles
        if not self.has_left_window:
            return jnp.maximum(visit * tiles_a_block - span.first_tile, 0)
        pivot = span.pivot_block
        num_lower_tiles = jnp.maximum(
            pivot * tiles_a_block - span.first_tile, 0
        )
        lower_start = span.num_pivot_tiles + (visit - 1) * tiles_a_block
        upper_start = (
            span.num_pivot_tiles
            + num_lower_tiles
            + (visit - pivot - 1) * tiles_a_block
        )
        return jnp.where(
            visit == 0,
            0,
            jnp.where(visit <= pivot, lower_start, upper_start),
        )

    def tile_items(
        self, tile: int, *, is_half: bool = False
    ) -> tuple[WorkItem, ...]:
        """One tile's work items, chunk-major, or its second half's when
        is_half; either ends on the same fragments, so all tails match."""
        first_chunk = self.num_chunks // 2 if is_half else 0
        return tuple(
            WorkItem(fragment=(chunk, head), tile=tile)
            for chunk in range(first_chunk, self.num_chunks)
            for head in range(self.traits.gqa_fold)
        )

    def tile_mask(
        self, ctx: ScheduleRow, kv: jax.Array, tile: jax.Array
    ) -> TileMask:
        """The mask of tile tile of kv block kv of the row of ctx."""
        first_key = self.schedule.kv_token(kv, ctx) + tile * MXU_COLUMNS
        return TileMask(kv=kv, first_key=first_key)

    @staticmethod
    def dead_mask() -> TileMask:
        """A mask that hides every key."""
        return TileMask(kv=jnp.int32(0), first_key=jnp.int32(DEAD_KEY))

    def is_last_tile_half(
        self, span: RowSpan, row: GridRow
    ) -> jax.Array | bool:
        """Whether no q row of the first half of the row's q block sees a key
        of the row's last tile, which then runs as a half tile."""
        if not self.has_half_tiles:
            return False
        traits, ctx = self.traits, row.ctx
        _, kv, tile = self.tile_place(span, span.num_tiles - 1)
        first_key = self.tile_mask(ctx, kv, tile).first_key
        # Keys below the block's own first one belong to the block before;
        # past it, a key's lowest kept q row only grows with the key.
        owned_key = jnp.maximum(ctx.kv_start, ctx.kv_base + kv * traits.bkv)
        kept_rows = per_seq_geometry(self.schedule, ctx)._replace(
            row_floor=ctx.kept_start
        )
        q_lower, _ = kept_rows.key_rows(jnp.maximum(first_key, owned_key))
        return q_lower - row.q_tok >= traits.bq // 2

    def ops(
        self,
        q_slot: jax.Array,
        state: jax.Array | int,
        ctx: ScheduleRow,
        tiles: tuple[TileRef, ...],
    ) -> VarlenFragmentOps:
        row_max, row_sum, o = self.state_refs(state)
        return VarlenFragmentOps(
            traits=self.traits,
            variant=self.variant,
            schedule=self.schedule,
            ctx=ctx,
            q_buf=self.q_refs.buf,
            q_slot=q_slot,
            k_buf=self.kv_refs.k_buf,
            v_buf=self.kv_refs.v_buf,
            row_max=row_max,
            row_sum=row_sum,
            o=o,
            tiles=tiles,
        )

    def tile_tail(
        self,
        q_slot: jax.Array,
        state: jax.Array | int,
        ctx: ScheduleRow,
        masks: tuple[TileMask, ...],
    ) -> Tail:
        """The tail every segment of a row leaves: one unit of tiles masked
        by masks, with the next segment's first K tile staged."""
        zero = jnp.int32(0)
        ops = self.ops(
            q_slot,
            state,
            ctx,
            tuple(TileRef(zero, zero, mask) for mask in masks),
        )
        return self.pipeline.tail_of(
            self.segment_items(len(masks), is_half=False),
            ops,
            has_staged_next_k=True,
        )

    def k_tile_at(self, slot: jax.Array, tile: jax.Array) -> jax.Array:
        start = pl.multiple_of(tile * MXU_COLUMNS, MXU_COLUMNS)
        return self.kv_refs.k_buf[slot, pl.ds(start, MXU_COLUMNS), :]

    def next_k_tile(
        self,
        stream: KVStream,
        span: RowSpan,
        index: jax.Array | int,
        visit: jax.Array,
    ) -> Callable[[], jax.Array]:
        """The K tile after a segment of visit visit: tile index of the row,
        or the next row's first tile past the last tile the row runs."""
        next_visit, _, next_tile = self.live_tile_place(span, index)
        is_row_end = index >= self.stream_tiles(span.num_tiles)
        is_in_block = jnp.logical_and(
            next_visit == visit, jnp.logical_not(is_row_end)
        )
        slot = jnp.where(is_in_block, stream.slot, stream.next_slot)
        tile = jnp.where(is_row_end, span.next_row_tile, next_tile)
        return lambda: self.k_tile_at(slot, tile)

    def step_block(
        self,
        stream: KVStream,
        is_step: jax.Array,
        run: RowRun,
        visit: jax.Array | int,
    ) -> KVStream:
        """Where is_step holds, wait for the next visit's block and start the
        block two visits on. Returns the stream while visit runs."""
        traits, blocks, row = self.traits, self.blocks, run.row
        ahead_desc = blocks.visit_ahead_desc(
            row.base_head,
            visit,
            blocks.visits(run.span.lo, row.hi, row.ctx),
            row.ctx,
            run.ahead,
        )
        is_ahead_fresh = blocks.is_fresh(ahead_desc, stream.ahead_desc)
        ahead_slot = jnp.where(
            is_ahead_fresh,
            lax.rem(stream.ahead_slot + 1, NUM_KV_STAGES),
            stream.ahead_slot,
        )

        @pl.when(jnp.logical_and(is_step, stream.is_ahead_fresh > 0))
        def _wait_next():
            wait_rotated_kv_block(traits, self.kv_refs, stream.ahead_slot)

        @pl.when(jnp.logical_and(is_step, is_ahead_fresh))
        def _start_ahead():
            blocks.start(ahead_desc, ahead_slot)

        stepped = KVStream(
            slot=stream.next_slot,
            next_slot=stream.ahead_slot,
            ahead_slot=ahead_slot,
            is_ahead_fresh=is_ahead_fresh.astype(jnp.int32),
            next_desc=stream.ahead_desc,
            ahead_desc=ahead_desc,
        )
        return jax.tree.map(
            lambda new, old: jnp.where(is_step, new, old), stepped, stream
        )

    @property
    def num_slots(self) -> int:
        """The tiles of a start segment."""
        return self.tiles.start_tiles

    @property
    def group_rows(self) -> int:
        """The most rows of a row group."""
        return varlen_group_rows(
            self.traits.group_bq // FRAGMENT_ROWS, self.shape
        )

    def row_state(
        self, parity: jax.Array | int, member: jax.Array | int
    ) -> jax.Array | int:
        """The row-state index of row member of a group at parity."""
        return parity * self.group_rows + member

    def state_refs(
        self, state: jax.Array | int
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """A row state's anchor, lane partial sums and accumulator."""
        return self.row_max.at[state], self.row_sum.at[state], self.o.at[state]

    def slot_member(self, slot: int) -> int:
        """The row state of start segment slot slot of a group of one-tile
        rows: its own row's, or the last row's past the group's capacity."""
        return min(slot, self.group_rows - 1)

    def q_slot(self, q_count: jax.Array) -> jax.Array:
        """The Q ring slot of the row at Q ring position q_count."""
        return lax.rem(q_count, self.traits.num_stages)

    def group_after(
        self, g: jax.Array, index: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """(head group, group index) of the row group after group index of
        head group g, and whether that one is the grid's last."""
        schedule = self.schedule
        is_group_end = index == schedule.num_groups - 1
        is_grid_end = jnp.logical_and(
            g == schedule.num_head_groups - 1, is_group_end
        )
        next_g = jnp.where(
            is_group_end, jnp.minimum(g + 1, schedule.num_head_groups - 1), g
        )
        next_index = jnp.where(
            is_grid_end, index, jnp.where(is_group_end, 0, index + 1)
        )
        return next_g, next_index, is_grid_end

    def start_group_q(
        self,
        g: jax.Array,
        q_count: jax.Array,
        first_si: jax.Array,
        num_rows: jax.Array,
    ) -> None:
        """Start the Q of the num_rows rows of head group g from schedule row
        first_si on, into the Q ring from position q_count on."""
        traits, schedule = self.traits, self.schedule

        def _start(member):
            _, _, ctx = schedule.row_interval(first_si + member)
            start_q_row(
                traits,
                self.q_refs,
                g * traits.group_heads,
                schedule.q_token(ctx),
                self.q_slot(q_count + member),
            )

        _start(0)
        for_each_later_row(num_rows, _start)

    def wait_group_q(self, q_count: jax.Array, num_rows: jax.Array) -> None:
        """Wait for the rotated, unscaled Q of num_rows rows from q_count on."""

        def _wait(member):
            wait_rotated_q_row(
                self.traits, self.q_refs, self.q_slot(q_count + member)
            )

        _wait(0)
        for_each_later_row(num_rows, _wait)

    def run_group(
        self, g: jax.Array, index: jax.Array, carry: VarlenRowCarry
    ) -> VarlenRowCarry:
        """Row group index of head group g: its start segment, with the
        finish of the group before staged behind it, then a lone row's other
        tiles. The group's last tiles stay in flight."""
        traits, schedule = self.traits, self.schedule
        finished = carry.finished
        parity = NUM_ROW_PARITIES - 1 - finished.parity
        first_si, num_rows = schedule.row_group(index)
        next_g, next_index, is_last = self.group_after(g, index)
        next_first_si, next_num_rows = schedule.row_group(next_index)
        span, row, ahead = self.group_plan(g, first_si, next_g, next_first_si)
        self.wait_group_q(carry.q_count, num_rows)

        @pl.when(jnp.logical_not(is_last))
        def _prefetch_q():
            self.start_group_q(
                next_g, carry.q_count + num_rows, next_first_si, next_num_rows
            )

        run = RowRun(
            self.q_slot(carry.q_count),
            self.row_state(parity, 0),
            span,
            row,
            ahead,
        )
        stream = self.step_block(carry.stream, jnp.bool_(True), run, 0)
        is_last_half = self.is_last_tile_half(span, row)
        first_finish = self.member_finish(finished, 0, finished.prev_out_tok)
        wait_row_out_slot(
            traits,
            self.out_refs,
            first_finish,
            row_out_slot(traits, first_finish),
        )
        slots = self.slot_tiles(first_si, num_rows, run, stream, carry.q_count)
        probs, last_mask = self.start_row(
            carry, run, stream, slots, num_rows > 1, is_last_half
        )
        self.complete_group(
            finished, self.tile_tail(run.q_slot, run.state, row.ctx, last_mask)
        )
        probs, stream, last_mask = self.run_tiles(
            (probs, stream, last_mask), run, is_last_half
        )
        _, _, last_ctx = schedule.row_interval(first_si + num_rows - 1)
        return VarlenRowCarry(
            q_count=carry.q_count + num_rows,
            stream=stream,
            probs=probs,
            finished=FinishedGroup(
                is_live=jnp.int32(1),
                base_head=row.base_head,
                first_si=first_si,
                num_rows=num_rows,
                is_short=(num_rows > 1).astype(jnp.int32),
                parity=jnp.int32(parity),
                q_count=carry.q_count,
                out_count=carry.num_rows,
                prev_out_tok=carry.prev_out_tok,
            ),
            last_mask=last_mask,
            prev_out_tok=schedule.q_token(last_ctx),
            num_rows=carry.num_rows + num_rows,
        )

    def group_plan(
        self,
        g: jax.Array,
        first_si: jax.Array,
        next_g: jax.Array,
        next_first_si: jax.Array,
    ) -> tuple[RowSpan, GridRow, tuple[BlockDesc, BlockDesc]]:
        """The live tiles and grid row of a group's first row, and the first
        two block descriptors of the group after it."""
        traits, schedule, blocks = self.traits, self.schedule, self.blocks
        lo, hi, ctx = schedule.row_interval(first_si)
        first_next_desc, second_next_desc, next_interval = (
            blocks.first_two_blocks(next_g, next_first_si)
        )
        next_row_tile = self.first_run_tile(*next_interval)
        span = RowSpan(lo, *self.row_span(lo, hi, ctx), next_row_tile)
        row = GridRow(g * traits.group_heads, hi, ctx, schedule.q_token(ctx))
        return span, row, (first_next_desc, second_next_desc)

    def first_run_tile(
        self, lo: jax.Array, hi: jax.Array, ctx: ScheduleRow
    ) -> jax.Array:
        """The tile the row of ctx runs first."""
        _, _, _, pivot_tile, _ = self.row_span(lo, hi, ctx)
        return pivot_tile

    def first_live_tile(
        self, lo: jax.Array, hi: jax.Array, ctx: ScheduleRow
    ) -> jax.Array:
        """The first tile of block lo the row runs. A row that spans blocks
        starts at a whole loop body that leaves a start segment in the
        block."""
        tiles = self.tiles
        first_tile = ctx.first_live_tile_lo
        aligned_tile = jnp.minimum(
            first_tile - first_tile % tiles.body_tiles,
            self.num_block_tiles - tiles.start_tiles,
        )
        return jnp.where(lo < hi, aligned_tile, first_tile)

    def member_tiles(
        self, first_si: jax.Array, num_rows: jax.Array
    ) -> tuple[tuple[ScheduleRow, ...], tuple[jax.Array, ...], tuple]:
        """Per start segment slot of a group of num_rows one-tile rows: its
        schedule row, its tile's first key in the group's kv block and its
        mask. A slot past the rows takes the last row and hides every key."""
        schedule = self.schedule
        lo, _, first_ctx = schedule.row_interval(first_si)
        block_tok = schedule.kv_token(lo, first_ctx)
        ctxs, starts, masks = [], [], []
        for slot in range(self.num_slots):
            lo, _, ctx = schedule.row_interval(
                first_si + jnp.minimum(slot, num_rows - 1)
            )
            mask = self.tile_mask(ctx, lo, ctx.first_live_tile_lo)
            ctxs.append(ctx)
            starts.append(mask.first_key - block_tok)
            masks.append(
                self.pick_mask(slot < num_rows, mask, self.dead_mask())
            )
        return tuple(ctxs), tuple(starts), tuple(masks)

    @staticmethod
    def pick_mask(
        predicate: jax.Array, true_mask: TileMask, false_mask: TileMask
    ) -> TileMask:
        return jax.tree.map(
            partial(jnp.where, predicate), true_mask, false_mask
        )

    def slot_tiles(
        self,
        first_si: jax.Array,
        num_rows: jax.Array,
        run: RowRun,
        stream: KVStream,
        q_count: jax.Array,
    ) -> tuple[tuple[TileRef, ...], tuple[TileMember, ...]]:
        """The tiles of a group's start segment, one a slot, and their rows:
        a lone row's first tiles in run order, or each one-tile row's tile in
        the group's kv block."""
        span, ctx = run.span, run.row.ctx
        is_short = num_rows > 1
        ctxs, starts, masks = self.member_tiles(first_si, num_rows)
        tiles, members = [], []
        for slot in range(self.num_slots):
            has_tile = span.num_tiles > slot
            _, kv, tile = self.tile_place(span, jnp.where(has_tile, slot, 0))
            long_mask = self.pick_mask(
                has_tile, self.tile_mask(ctx, kv, tile), self.dead_mask()
            )
            tiles.append(
                TileRef(
                    stream.slot,
                    jnp.where(is_short, starts[slot], tile * MXU_COLUMNS),
                    self.pick_mask(is_short, masks[slot], long_mask),
                )
            )
            members.append(
                TileMember(
                    self.q_slot(q_count + jnp.minimum(slot, num_rows - 1)),
                    ctxs[slot],
                    *self.state_refs(run.state),
                )
            )
        return tuple(tiles), tuple(members)

    def start_row(
        self,
        carry: VarlenRowCarry,
        run: RowRun,
        stream: KVStream,
        slots: tuple[tuple[TileRef, ...], tuple[TileMember, ...]],
        is_short: jax.Array,
        is_last_half: jax.Array | bool,
    ) -> tuple[tuple[jax.Array, ...], tuple[TileMask, ...]]:
        """The group's start segment, one copy a parity and kind of group.
        Returns the probabilities and the masks of its last unit of tiles."""
        num_tiles = run.span.num_tiles
        # A group of one-tile rows runs its start segment alone, as its
        # first row would.
        next_k_tile = self.next_k_tile(
            stream, run.span, jnp.int32(self.tiles.start_tiles), jnp.int32(0)
        )
        segment = partial(self.start_segment, carry, run, slots, next_k_tile)
        is_second_half = jnp.logical_and(
            self.has_half_tiles,
            jnp.logical_or(
                num_tiles == 1,
                jnp.logical_and(num_tiles == ROW_START_TILES, is_last_half),
            ),
        )

        def _at_parity(parity, probs):
            return lax.cond(
                is_short,
                partial(segment, parity, True, is_second_half),
                partial(segment, parity, False, is_second_half),
                probs,
            )

        probs = lax.cond(
            carry.finished.parity == ENTRY_PARITY,
            partial(_at_parity, NUM_ROW_PARITIES - 1 - ENTRY_PARITY),
            partial(_at_parity, ENTRY_PARITY),
            carry.probs,
        )
        tiles, _ = slots
        return probs, tuple(
            tile.mask for tile in tiles[-self.tiles.unit_tiles :]
        )

    def start_segment(
        self,
        carry: VarlenRowCarry,
        run: RowRun,
        slots: tuple[tuple[TileRef, ...], tuple[TileMember, ...]],
        next_k_tile: Callable[[], jax.Array],
        parity: int,
        is_short: bool,
        is_second_half: jax.Array,
        probs: tuple[jax.Array, ...],
    ) -> tuple[jax.Array, ...]:
        """start_row's segment at a static parity, which keeps the two
        groups' row states provably apart, so the finish of the group before
        overlaps it."""
        pipeline = self.pipeline
        tiles, _ = slots
        num_slots, unit = self.num_slots, self.tiles.unit_tiles
        prev_parity = NUM_ROW_PARITIES - 1 - parity
        if not is_short and zeroes_row_start(
            self.traits.group_bq // FRAGMENT_ROWS, self.shape
        ):
            self.o[self.row_state(parity, 0)] = jnp.zeros(
                self.o.shape[1:], self.o.dtype
            )

        def _segment(is_half, probs):
            state = pipeline.advance(
                PipelineState(
                    probs=probs,
                    tail=self.group_tail(
                        carry.finished, prev_parity, carry.last_mask
                    ),
                ),
                self.segment_items(num_slots, is_half=is_half),
                self.start_ops(run, slots, parity, is_short),
                next_k_tile=next_k_tile,
            )
            # The next segment is a loop body or the next group's start.
            for next_tail in (
                self.group_tail_of(state.tail.ops),
                self.tile_tail(
                    run.q_slot,
                    self.row_state(parity, 0),
                    run.row.ctx,
                    tuple(tile.mask for tile in tiles[-unit:]),
                ),
            ):
                pipeline.check_loop_tail(state.tail, next_tail)
            self.stage_row_finish(
                self.member_finish(
                    carry.finished, 0, carry.finished.prev_out_tok, prev_parity
                )
            )
            return state.probs

        if is_short or not self.has_half_tiles:
            return _segment(False, probs)
        return lax.cond(
            is_second_half,
            partial(_segment, True),
            partial(_segment, False),
            probs,
        )

    def start_ops(
        self,
        run: RowRun,
        slots: tuple[tuple[TileRef, ...], tuple[TileMember, ...]],
        parity: int,
        is_short: bool,
    ) -> VarlenFragmentOps:
        """The ops of start_segment's slots at parity: each slot of a group
        of one-tile rows starts a row, else only the first slot does."""
        tiles, members = slots
        num_slots = self.num_slots
        ops = self.ops(
            run.q_slot, self.row_state(parity, 0), run.row.ctx, tiles
        )
        if not is_short:
            return dataclasses.replace(
                ops, row_starts=(True,) + (False,) * (num_slots - 1)
            )
        return dataclasses.replace(
            ops,
            members=tuple(
                TileMember(
                    member.q_slot,
                    member.ctx,
                    *self.state_refs(
                        self.row_state(parity, self.slot_member(slot))
                    ),
                )
                for slot, member in enumerate(members)
            ),
            row_starts=tuple(
                slot < self.group_rows for slot in range(num_slots)
            ),
        )

    def group_tail_of(self, ops: VarlenFragmentOps) -> Tail:
        """The tail a start segment's worth of tiles of ops leaves, as the
        next group's start segment assumes it."""
        return self.pipeline.tail_of(
            self.segment_items(self.num_slots, is_half=False),
            ops,
            has_staged_next_k=True,
        )

    def group_tail(
        self,
        finished: FinishedGroup,
        parity: jax.Array | int,
        last_mask: tuple[TileMask, ...],
    ) -> Tail:
        """The tail finished leaves for the next group's start segment, its
        row states at parity; last_mask masks the last unit of a lone row's
        tiles."""
        schedule = self.schedule
        is_short = finished.is_short > 0
        _, _, first_ctx = schedule.row_interval(finished.first_si)
        ctxs, _, masks = self.member_tiles(finished.first_si, finished.num_rows)
        num_slots, unit = self.num_slots, self.tiles.unit_tiles
        long_masks = (self.dead_mask(),) * (num_slots - unit) + last_mask
        zero = jnp.int32(0)
        tiles, members = [], []
        for slot in range(num_slots):
            tiles.append(
                TileRef(
                    zero,
                    zero,
                    self.pick_mask(is_short, masks[slot], long_masks[slot]),
                )
            )
            members.append(
                TileMember(
                    zero,
                    jax.tree.map(
                        partial(jnp.where, is_short), ctxs[slot], first_ctx
                    ),
                    *self.state_refs(
                        self.row_state(
                            parity,
                            jnp.where(is_short, self.slot_member(slot), 0),
                        )
                    ),
                )
            )
        ops = dataclasses.replace(
            self.ops(zero, self.row_state(parity, 0), first_ctx, tuple(tiles)),
            members=tuple(members),
            row_starts=tuple(
                jnp.logical_and(is_short, slot < self.group_rows)
                for slot in range(num_slots)
            ),
        )
        return self.group_tail_of(ops)

    def member_finish(
        self,
        finished: FinishedGroup,
        member: jax.Array | int,
        prev_out_tok: jax.Array,
        parity: jax.Array | int | None = None,
    ) -> RowFinish:
        """The output finish of finished's row member; parity defaults to
        finished's own."""
        schedule = self.schedule
        si = finished.first_si + member
        _, _, ctx = schedule.row_interval(si)
        q_tok = schedule.q_token(ctx)
        return RowFinish(
            is_live=finished.is_live,
            si=si,
            q_tok=q_tok,
            base_head=finished.base_head,
            num_blend_rows=schedule.blend_start(ctx) - q_tok,
            prev_out_tok=prev_out_tok,
            out_count=finished.out_count + member,
            parity=self.row_state(
                finished.parity if parity is None else parity, member
            ),
        )

    def complete_group(self, finished: FinishedGroup, tail: Tail) -> None:
        """Complete finished's rows behind the next group's start segment,
        which left tail: the first is staged, the later ones stage in turn."""
        traits, schedule = self.traits, self.schedule
        self.complete_row(
            self.member_finish(finished, 0, finished.prev_out_tok), tail
        )

        def _complete_member(member):
            _, _, prev_ctx = schedule.row_interval(
                finished.first_si + member - 1
            )
            finish = self.member_finish(
                finished, member, schedule.q_token(prev_ctx)
            )
            out_slot = row_out_slot(traits, finish)
            wait_row_out_slot(traits, self.out_refs, finish, out_slot)
            self.stage_row_finish(finish)
            self.complete_row(finish, tail)

        for_each_later_row(finished.num_rows, _complete_member)

    def segment_items(
        self, num_tiles: int, *, is_half: bool
    ) -> tuple[WorkItem, ...]:
        """The items of num_tiles tiles, the last a half tile when is_half."""
        return sum(
            (
                self.tile_items(
                    offset, is_half=is_half and offset == num_tiles - 1
                )
                for offset in range(num_tiles)
            ),
            (),
        )

    def tile_segment(
        self,
        index: jax.Array,
        tile_carry: TileCarry,
        run: RowRun,
        *,
        num_segment_tiles: int,
        is_half: bool,
        steps_block: bool,
    ) -> TileCarry:
        """advance_tiles with the row's own ops, as a loop step."""
        state, stream, last_mask = self.advance_tiles(
            index,
            tile_carry,
            run,
            num_segment_tiles=num_segment_tiles,
            is_half=is_half,
            steps_block=steps_block,
        )
        return state.probs, stream, last_mask

    def advance_tiles(
        self,
        index: jax.Array,
        tile_carry: TileCarry,
        run: RowRun,
        *,
        num_segment_tiles: int,
        is_half: bool,
        steps_block: bool,
        ops_of: Callable[[VarlenFragmentOps], VarlenFragmentOps] | None = None,
        tail_ops_of: Callable[[VarlenFragmentOps], FragmentOps] | None = None,
    ) -> tuple[PipelineState, KVStream, tuple[TileMask, ...]]:
        """The segment of num_segment_tiles tiles of the row from tile index
        on, all in one visit; under steps_block it first steps the kv stream
        where the segment opens a visit. ops_of maps the segment's ops,
        tail_ops_of the ops that finish the segment before."""
        pipeline = self.pipeline
        span, ctx = run.span, run.row.ctx
        probs, stream, prev_masks = tile_carry
        visit, _, _ = self.live_tile_place(span, index)
        if steps_block:
            prev_visit, _, _ = self.tile_place(span, index - 1)
            stream = self.step_block(stream, visit != prev_visit, run, visit)
        tiles = ()
        for offset in range(num_segment_tiles):
            _, kv, tile = self.live_tile_place(span, index + offset)
            tiles += (
                TileRef(
                    stream.slot,
                    tile * MXU_COLUMNS,
                    self.pick_mask(
                        index + offset < span.num_tiles,
                        self.tile_mask(ctx, kv, tile),
                        self.dead_mask(),
                    ),
                ),
            )
        ops = self.ops(run.q_slot, run.state, ctx, tiles)
        assumed = self.tile_tail(run.q_slot, run.state, ctx, prev_masks)
        tail = assumed
        if tail_ops_of is not None:
            tail = assumed.with_ops(tail_ops_of(assumed.ops))
        state = pipeline.advance(
            PipelineState(probs=probs, tail=tail),
            self.segment_items(num_segment_tiles, is_half=is_half),
            ops if ops_of is None else ops_of(ops),
            next_k_tile=self.next_k_tile(
                stream, span, index + num_segment_tiles, visit
            ),
        )
        pipeline.check_loop_tail(state.tail, assumed)
        unit = self.tiles.unit_tiles
        return state, stream, tuple(ref.mask for ref in tiles[-unit:])

    def run_tiles(
        self,
        tile_carry: TileCarry,
        run: RowRun,
        is_last_half: jax.Array | bool,
    ) -> TileCarry:
        """The row's tiles after its start segment: check units, whole loop
        bodies, leftover units, then a half tile when is_last_half."""
        tiles = self.tiles
        num_tiles = run.span.num_tiles
        has_half_last = jnp.logical_and(
            is_last_half, num_tiles > tiles.start_tiles
        )
        whole_end = jnp.maximum(
            num_tiles - has_half_last.astype(jnp.int32), tiles.start_tiles
        )
        first_whole = tiles.start_tiles
        if self.has_checked_units:
            tile_carry, first_whole = self.checked_units(tile_carry, run)
        tile_carry = self.run_whole_tiles(
            tile_carry, run, first_whole, whole_end
        )
        first_left = (
            first_whole
            + (whole_end - first_whole) // tiles.body_tiles * tiles.body_tiles
        )
        tile_carry = self.run_units(
            tile_carry,
            run,
            first_left,
            self.stream_tiles(num_tiles) - has_half_last.astype(jnp.int32),
        )
        if self.has_half_tiles:
            tile_carry = self.run_last_half_tile(
                has_half_last, num_tiles - 1, tile_carry, run
            )
        return tile_carry

    def run_units(
        self,
        tile_carry: TileCarry,
        run: RowRun,
        first: jax.Array,
        end: jax.Array,
    ) -> TileCarry:
        """The row's tiles [first, end), fewer than a body, in rolled
        segments of one unit, inside the row's last block."""
        unit = self.tiles.unit_tiles
        return lax.fori_loop(
            0,
            jnp.maximum(end - first, 0) // unit,
            lambda step, step_carry: self.tile_segment(
                first + unit * step,
                step_carry,
                run,
                num_segment_tiles=unit,
                is_half=False,
                steps_block=False,
            ),
            tile_carry,
        )

    def run_whole_tiles(
        self,
        tile_carry: TileCarry,
        run: RowRun,
        first: jax.Array | int,
        whole_end: jax.Array,
    ) -> TileCarry:
        """The row's whole tiles from first up to whole_end, body_tiles a
        step: per visit its block step, then a rolled loop of segments. Every
        visit but the last holds whole bodies."""
        span = run.span
        body_tiles = self.tiles.body_tiles

        def _visit_of(index):
            visit, _, _ = self.tile_place(span, index)
            return visit

        def _run_visit(visit, tile_carry):
            # The segment that opened a visit before first has stepped it.
            probs, stream, prev_mask = tile_carry
            visit_first = self.visit_start(span, visit)
            stream = self.step_block(stream, visit_first >= first, run, visit)
            begin = jnp.maximum(visit_first, first)
            end = jnp.minimum(self.visit_start(span, visit + 1), whole_end)
            return lax.fori_loop(
                0,
                (end - begin) // body_tiles,
                lambda step, step_carry: self.tile_segment(
                    begin + body_tiles * step,
                    step_carry,
                    run,
                    num_segment_tiles=body_tiles,
                    is_half=False,
                    steps_block=False,
                ),
                (probs, stream, prev_mask),
            )

        first_visit = _visit_of(first)
        return lax.fori_loop(
            first_visit,
            jnp.where(
                whole_end > first, _visit_of(whole_end - 1) + 1, first_visit
            ),
            _run_visit,
            tile_carry,
        )

    def final_unit(self, num_tiles: jax.Array | int) -> jax.Array | int:
        """The first check unit of the final unit of a row of num_tiles live
        tiles; 0 when no unit is checked."""
        if not self.has_checked_units:
            return 0
        return first_final_unit(num_tiles // UNIT_TILES)

    @property
    def has_checked_units(self) -> bool:
        """Whether a row can hold a checked unit."""
        tiles = self.tiles
        if (
            tiles.body_tiles != LOOP_BODY_TILES
            or tiles.start_tiles != ROW_START_TILES
            or zeroes_row_start(
                self.traits.group_bq // FRAGMENT_ROWS, self.shape
            )
        ):
            return False
        max_row_tiles = self.schedule.num_kv_blocks * self.num_block_tiles
        return first_final_unit(max_row_tiles // UNIT_TILES) > 0

    def unit_end_ops(
        self,
        ops: VarlenFragmentOps,
        *,
        slot: jax.Array,
        largest: LargestSums,
        tile: int,
    ) -> VarlenFragmentOps:
        """ops whose items of tile key tile, a check unit's last tile, also
        write checkpoint slot slot and fold their row sums into largest."""
        unit_end = UnitCheckpoint(
            tile=tile,
            row_sum=self.checkpoint_sum.at[slot],
            o=self.checkpoint_o.at[slot],
            largest=largest,
        )
        return dataclasses.replace(ops, unit_end=unit_end)

    def checked_units(
        self, tile_carry: TileCarry, run: RowRun
    ) -> tuple[TileCarry, jax.Array]:
        """Run the row's check units after its start segment; returns the
        carry and the live tile after them."""
        final_unit = self.final_unit(run.span.num_tiles)
        num_units = jnp.where(final_unit > 0, final_unit + 1, 0)
        self.unit_guard[0] = jnp.int32(0)
        tile_carry = lax.fori_loop(
            0,
            num_units,
            partial(self.checked_unit, run=run, final_unit=final_unit),
            tile_carry,
        )
        return tile_carry, num_units * UNIT_TILES + LOOP_BODY_TILES

    def checked_unit(
        self,
        unit: jax.Array,
        tile_carry: TileCarry,
        *,
        run: RowRun,
        final_unit: jax.Array,
    ) -> TileCarry:
        """Check unit unit's last body, whose last tile writes the next
        unit's checkpoint, then the next unit's first body, which finishes
        that tile: its row sums flag unit, and a unit before final_unit
        replays. The last body adds nothing for the items a replay redid."""
        next_unit = unit + 1
        largest = LargestSums()
        unit_end_ops = partial(
            self.unit_end_ops,
            slot=lax.rem(next_unit, NUM_CHECKPOINTS),
            largest=largest,
        )
        state, stream, last_mask = self.advance_tiles(
            unit * UNIT_TILES + LOOP_BODY_TILES,
            tile_carry,
            run,
            num_segment_tiles=LOOP_BODY_TILES,
            is_half=False,
            # A last body opens a kv block only under a left window bound or
            # where a block's tiles are not whole check units.
            steps_block=(
                self.has_left_window or self.num_block_tiles % UNIT_TILES != 0
            ),
            ops_of=partial(unit_end_ops, tile=LOOP_BODY_TILES - 1),
            tail_ops_of=partial(
                FlaggedTailOps,
                is_flagged=self.unit_guard[0] > 0,
                is_row_bound=lambda mode: mode is not None,
            ),
        )
        state, stream, last_mask = self.advance_tiles(
            next_unit * UNIT_TILES,
            (state.probs, stream, last_mask),
            run,
            num_segment_tiles=LOOP_BODY_TILES,
            is_half=False,
            steps_block=True,
            tail_ops_of=partial(unit_end_ops, tile=0),
        )
        self.unit_guard[0] = jnp.where(
            unit < final_unit, guard_flag(self.traits, largest.value), 0
        )

        @pl.when(self.unit_guard[0] > 0)
        def _replay():
            self.replay_tiles(
                self.replayed_row(run),
                unit,
                next_unit * UNIT_TILES + LOOP_BODY_TILES,
                self.pipeline.free_slots(state.tail, REPLAY_RESULT_SLOTS),
                checkpoint_unit=next_unit,
            )

        return state.probs, stream, last_mask

    def run_last_half_tile(
        self,
        is_run: jax.Array,
        index: jax.Array,
        tile_carry: TileCarry,
        run: RowRun,
    ) -> TileCarry:
        """The row's last tile as a half tile where is_run holds."""
        return lax.cond(
            is_run,
            partial(
                self.tile_segment,
                index,
                run=run,
                num_segment_tiles=1,
                is_half=True,
                steps_block=True,
            ),
            lambda tile_carry: tile_carry,
            tile_carry,
        )

    def complete_row(self, finish: RowFinish, tail: Tail) -> None:
        """A row's finish after stage_row_finish: the replay of its final
        unit behind tail where the guard flagged it, then a live row's
        writes."""

        @pl.when(self.guard[0] > 0)
        def _replay():
            self.replay_row_end(
                finish, self.pipeline.free_slots(tail, REPLAY_RESULT_SLOTS)
            )
            self.stage_row_finish(finish)

        @pl.when(finish.is_live > 0)
        def _write_row():
            packed_q_end = self.schedule.packed_q_end
            out_slot = row_out_slot(self.traits, finish)
            copy_row_out(
                self.traits,
                self.out_refs,
                finish,
                out_slot,
                packed_q_end,
                fill_padding_tiles,
            )
            if self.variant.return_lse:
                copy_row_lse(
                    self.traits,
                    self.out_refs,
                    finish.base_head // self.traits.group_heads,
                    finish,
                    out_slot,
                    packed_q_end,
                    fill_padding_tiles,
                )

    def stage_row_finish(self, finish: RowFinish) -> None:
        """Stage the row's lse and output once its last softmax has run and
        its output stage slot is free; its lane partial sums reduce first,
        which sets its guard flag."""
        stage_row_finish(
            self.traits,
            self.variant,
            self.operands,
            row_finish_state(
                self.row_max, self.row_sum, self.o, self.guard, finish.parity
            ),
            self.out_refs,
            finish,
            reduces_sums=True,
        )

    def replay_row_end(
        self, finish: RowFinish, slots: tuple[jax.Array, ...]
    ) -> None:
        """Replay finish's final unit into its row state through QK result
        slots slots, between the next row's start segment and its next
        segment. The row's Q is staged again into the replay's own buffer."""
        traits, replay = self.traits, self.replay
        lo, hi, ctx = self.schedule.row_interval(finish.si)
        start_q_row(traits, replay.q_refs, finish.base_head, finish.q_tok, 0)
        wait_rotated_q_row(traits, replay.q_refs, 0)
        span = RowSpan(lo, *self.row_span(lo, hi, ctx), jnp.int32(0))
        row = ReplayedRow(
            q_buf=replay.q_refs.buf,
            q_slot=0,
            state=finish.parity,
            span=span,
            ctx=ctx,
            kv_head=finish.base_head // traits.q_heads_per_kv_head,
        )
        self.replay_tiles(
            row, self.final_unit(span.num_tiles), span.num_tiles, slots
        )

    def replayed_row(self, run: RowRun) -> ReplayedRow:
        """The running row as a replay between two of its segments
        recomputes it, from its Q in the Q stream."""
        return ReplayedRow(
            q_buf=self.q_refs.buf,
            q_slot=run.q_slot,
            state=run.state,
            span=run.span,
            ctx=run.row.ctx,
            kv_head=run.row.base_head // self.traits.q_heads_per_kv_head,
        )

    def replay_tiles(
        self,
        row: ReplayedRow,
        first_unit: jax.Array,
        end_tile: jax.Array,
        slots: tuple[jax.Array, ...],
        checkpoint_unit: jax.Array | None = None,
    ) -> None:
        """Recompute row's tiles from check unit first_unit's first to
        end_tile, in run order: from the row start at unit 0, else from the
        unit's checkpoint. checkpoint_unit's checkpoint is written again at
        its first tile and rebased at the end to the row's raised anchor."""
        traits, kv_refs, span = self.traits, self.replay.kv_refs, row.span
        first = first_unit * UNIT_TILES
        checkpoints = self.row_checkpoints(row.state)

        @pl.when(first_unit > 0)
        def _restore():
            checkpoints.copy_checkpoint(first_unit, is_restore=True)

        def _replay_visit(visit, carry):
            _, kv, _ = self.tile_place(span, self.visit_start(span, visit))
            kv_tok = self.schedule.kv_token(kv, row.ctx)
            start_kv_block(traits, kv_refs, row.kv_head, kv_tok, 0)
            wait_rotated_kv_block(traits, kv_refs, 0)

            def _replay_tile(index, tile_carry):
                if checkpoint_unit is not None:

                    @pl.when(index == checkpoint_unit * UNIT_TILES)
                    def _checkpoint():
                        checkpoints.copy_checkpoint(
                            checkpoint_unit, is_restore=False
                        )
                        checkpoints.save_anchor(first_unit)

                _, _, tile = self.tile_place(span, index)
                self.replay_tile(row, kv, tile, index == 0, slots)
                return tile_carry

            return lax.fori_loop(
                jnp.maximum(self.visit_start(span, visit), first),
                jnp.minimum(self.visit_start(span, visit + 1), end_tile),
                _replay_tile,
                carry,
            )

        first_visit, _, _ = self.tile_place(span, first)
        last_visit, _, _ = self.tile_place(span, end_tile - 1)
        lax.fori_loop(first_visit, last_visit + 1, _replay_visit, None)
        if checkpoint_unit is not None:
            checkpoints.rebase_checkpoint(checkpoint_unit, first_unit)

    def replay_tile(
        self,
        row: ReplayedRow,
        kv: jax.Array,
        tile: jax.Array,
        is_row_start: jax.Array,
        slots: tuple[jax.Array, ...],
    ) -> None:
        """Fold every fragment of row against tile tile of kv block kv,
        staged in the replay's buffer, rolled by guard_replay_unroll."""
        traits = self.traits
        bounds = row_tile_bounds(
            traits, self.schedule, row.ctx, self.tile_mask(row.ctx, kv, tile)
        )
        lax.fori_loop(
            0,
            traits.group_bq // FRAGMENT_ROWS,
            lambda index, carry: self.replay_fragment(
                row, index, (tile, bounds), is_row_start, slots
            ),
            None,
            unroll=traits.guard_replay_unroll,
        )

    def replay_fragment(
        self,
        row: ReplayedRow,
        index: jax.Array,
        tile_and_bounds: tuple[jax.Array, TileBounds],
        is_row_start: jax.Array,
        slots: tuple[jax.Array, ...],
    ) -> None:
        """Fold fragment index (chunk-major) of row against a tile of the
        replay's staged block, (its index in the block, its bounds)."""
        traits, kv_refs = self.traits, self.replay.kv_refs
        score_slot, value_slot = slots
        tile, bounds = tile_and_bounds
        chunk, head = index // traits.gqa_fold, index % traits.gqa_fold
        rows = chunk_rows(traits, chunk, head)
        keys = pl.ds(
            pl.multiple_of(tile * MXU_COLUMNS, MXU_COLUMNS), MXU_COLUMNS
        )
        q_ref = staged_q_head(traits, row.q_buf.at[row.q_slot], head)
        scores = replay_matmul(
            score_slot,
            q_ref[fragment_rows(chunk), :],
            kv_refs.k_buf[0, keys, :],
            transpose=True,
        )
        first_row = chunk * FRAGMENT_ROWS
        fold_replayed_fragment(
            self.row_max.at[row.state, rows],
            self.row_sum.at[row.state, rows],
            self.o.at[row.state, rows],
            masked_logits(self.variant, scores, bounds, first_row),
            is_row_start,
            value_slot,
            lambda: kv_refs.v_buf[0, keys, :],
        )

    def row_checkpoints(self, state: jax.Array) -> RowCheckpoints:
        """The check unit checkpoints of the row at row state state."""
        return RowCheckpoints(
            checkpoint_sum=self.checkpoint_sum,
            checkpoint_o=self.checkpoint_o,
            row_sum=self.row_sum,
            o=self.o,
            parity=state,
            row_anchor=lambda rows: self.row_max[state, rows, :],
        )


def varlen_mxu_fwd_body(
    traits: FwdTraits,
    variant: Variant,
    schedule: FwdSchedule,
    q_hbm: jax.Array,
    k_hbm: jax.Array,
    v_hbm: jax.Array,
    refs: Sequence[jax.Array],
    *,
    operands: VariantOperands,
) -> None:
    """Run the varlen forward on the explicit MXU pipeline; the arguments
    are varlen_fwd_body's, refs in varlen_mxu_scratch_shapes's order. A tile
    that nothing reads runs before the first row and after the last, so
    every start segment finishes a tail."""
    assert traits.is_per_seq and traits.uses_mxu_pipeline, traits
    assert variant.window == (None, None), variant.window
    # masked_logits scales the scores by the static Q scale.
    assert not variant.has_kv_scales, variant
    variant.check_operands(operands)
    loop, carry = make_mxu_loop(
        traits, variant, schedule, q_hbm, k_hbm, v_hbm, refs, operands
    )
    carry = run_entry_tile(loop, carry)
    carry = lax.fori_loop(
        0,
        schedule.num_head_groups,
        lambda g, head_carry: lax.fori_loop(
            0,
            schedule.num_groups,
            lambda index, group_carry: loop.run_group(g, index, group_carry),
            head_carry,
        ),
        carry,
    )
    finish_last_row(loop, carry)


def run_entry_tile(
    loop: VarlenMxuLoop, carry: VarlenRowCarry
) -> VarlenRowCarry:
    """The tile before the first group, on ENTRY_PARITY: it leaves the tail
    every start segment finishes, with the first row's first K tile staged."""
    pipeline, schedule = loop.pipeline, loop.schedule
    zero = jnp.int32(0)
    first_interval = schedule.row_interval(zero)
    _, _, first_ctx = first_interval
    entry_state = loop.row_state(ENTRY_PARITY, 0)
    entry_slot = carry.stream.next_slot
    num_entry_tiles = 1 if loop.has_half_tiles else loop.tiles.start_tiles
    entry = pipeline.advance(
        pipeline.idle_state(),
        loop.segment_items(num_entry_tiles, is_half=loop.has_half_tiles),
        loop.ops(
            zero,
            entry_state,
            first_ctx,
            (TileRef(entry_slot, zero, loop.dead_mask()),) * num_entry_tiles,
        ),
        next_k_tile=lambda: loop.k_tile_at(
            entry_slot, loop.first_run_tile(*first_interval)
        ),
    )
    pipeline.check_loop_tail(
        entry.tail,
        loop.group_tail(carry.finished, ENTRY_PARITY, carry.last_mask),
    )
    return carry._replace(probs=entry.probs)


def finish_last_row(loop: VarlenMxuLoop, carry: VarlenRowCarry) -> None:
    """After the last group: a tile on the free parity takes over its tail,
    the pipeline drains, the group finishes and every DMA in flight lands."""
    traits, pipeline = loop.traits, loop.pipeline
    stream, finished = carry.stream, carry.finished
    _, _, ctx = loop.schedule.row_interval(finished.first_si)
    exit_state = pipeline.advance(
        PipelineState(
            probs=carry.probs,
            tail=loop.group_tail(finished, finished.parity, carry.last_mask),
        ),
        loop.segment_items(1, is_half=loop.has_half_tiles),
        loop.ops(
            jnp.int32(0),
            loop.row_state(NUM_ROW_PARITIES - 1 - finished.parity, 0),
            ctx,
            (TileRef(stream.slot, jnp.int32(0), loop.dead_mask()),),
        ),
    )
    pipeline.drain(exit_state)
    last_finish = loop.member_finish(finished, 0, finished.prev_out_tok)
    wait_row_out_slot(
        traits, loop.out_refs, last_finish, row_out_slot(traits, last_finish)
    )
    loop.stage_row_finish(last_finish)
    loop.complete_group(finished, exit_state.tail)

    @pl.when(stream.is_ahead_fresh > 0)
    def _wait_ahead():
        loop.blocks.wait(stream.ahead_slot)

    drain(
        carry.num_rows,
        traits.num_stages,
        lambda slot: o_copy(traits, loop.out_refs, 0, 0, slot).wait(),
    )
    fill_packed_tail(
        traits,
        loop.variant,
        loop.out_refs,
        loop.schedule.packed_q_end,
        loop.schedule.num_head_groups,
    )


def make_mxu_loop(
    traits: FwdTraits,
    variant: Variant,
    schedule: FwdSchedule,
    q_hbm: jax.Array,
    k_hbm: jax.Array,
    v_hbm: jax.Array,
    refs: Sequence[jax.Array],
    operands: VariantOperands,
) -> tuple[VarlenMxuLoop, VarlenRowCarry]:
    """The explicit loop's context from the kernel's refs and its first
    carry, after the prologue."""
    q_refs, kv_refs, out_refs, state, replay = split_mxu_refs(
        traits, variant, (q_hbm, k_hbm, v_hbm), refs
    )
    (
        row_max,
        row_sum,
        o_scratch,
        checkpoint_sum,
        checkpoint_o,
        guard_smem,
        unit_guard,
    ) = state
    shape = varlen_pipeline_shape(traits)
    loop = VarlenMxuLoop(
        traits=traits,
        variant=variant,
        operands=operands,
        schedule=schedule,
        pipeline=MxuPipeline.allocate(
            shape,
            has_mode_free_accumulate=zeroes_row_start(
                traits.group_bq // FRAGMENT_ROWS, shape
            ),
        ),
        blocks=VarlenBlockStream(
            traits=traits,
            schedule=schedule,
            kv_refs=kv_refs,
            row_visits=(
                partial(pivot_visits, traits)
                if schedule.window[0] is not None
                else None
            ),
        ),
        q_refs=q_refs,
        kv_refs=kv_refs,
        out_refs=out_refs,
        row_max=row_max,
        row_sum=row_sum,
        o=o_scratch,
        checkpoint_sum=checkpoint_sum,
        checkpoint_o=checkpoint_o,
        guard=guard_smem,
        unit_guard=unit_guard,
        replay=replay,
    )
    if variant.return_lse:
        lse_stage = out_refs.lse_stage
        lse_stage[...] = jnp.zeros(lse_stage.shape, jnp.float32)
    # The first group finishes the entry tile as its previous group: from a
    # zero anchor and zero sums its masked scores raise no guard flag.
    for ref in (row_max, row_sum):
        ref[loop.row_state(ENTRY_PARITY, 0)] = jnp.zeros(
            ref.shape[1:], jnp.float32
        )
    return loop, first_row_carry(loop)


def split_stages(
    refs: Sequence[jax.Array], traits: FwdTraits
) -> tuple[tuple, list[jax.Array]]:
    """(The Q, K and V stages that lead refs with their Q and K coefficient
    stages, None without rotary or rotary_k; the refs after them)."""
    q_buf, k_buf, v_buf, *rest = refs
    rotary_q_buf = rest.pop(0) if traits.rotary_dim else None
    rotary_k_buf = (
        rest.pop(0) if traits.rotary_dim and traits.rotary_k else None
    )
    return (q_buf, k_buf, v_buf, rotary_q_buf, rotary_k_buf), rest


def split_mxu_refs(
    traits: FwdTraits,
    variant: Variant,
    hbm_refs: tuple[jax.Array, jax.Array, jax.Array],
    refs: Sequence[jax.Array],
) -> tuple[QRefs, KVRefs, OutRefs, tuple, ReplayRefs]:
    """The explicit loop's refs from the kernel's Q, K and V operands and its
    other refs (operands, then varlen_mxu_scratch_shapes's order); the row
    state refs come in VarlenMxuLoop's field order."""
    q_hbm, k_hbm, v_hbm = hbm_refs
    rest = list(refs)
    rotary_q_hbm = rest.pop(0) if traits.rotary_dim else None
    rotary_k_hbm = (
        rest.pop(0) if traits.rotary_dim and traits.rotary_k else None
    )
    o_hbm = rest.pop(0)
    lse_hbm = rest.pop(0) if variant.return_lse else None
    (q_buf, k_buf, v_buf, rotary_q_buf, rotary_k_buf), rest = split_stages(
        rest, traits
    )
    (
        o_stage,
        row_max,
        row_sum,
        o_scratch,
        checkpoint_sum,
        checkpoint_o,
        *rest,
    ) = rest
    replay_stages, rest = split_stages(rest, traits)
    guard_smem, unit_guard, sems, replay_sems, *rest = rest
    lse_stage = rest.pop(0) if variant.return_lse else None
    assert not rest, rest
    q_refs = QRefs(
        hbm=q_hbm,
        buf=q_buf,
        rotary_hbm=rotary_q_hbm,
        rotary_buf=rotary_q_buf,
        sems=sems,
    )
    kv_refs = KVRefs(
        k_hbm=k_hbm,
        v_hbm=v_hbm,
        k_buf=k_buf,
        v_buf=v_buf,
        rotary_hbm=rotary_k_hbm,
        rotary_buf=rotary_k_buf,
        sems=sems,
    )
    out_refs = OutRefs(
        o_hbm=o_hbm,
        o_stage=o_stage,
        lse_hbm=lse_hbm,
        lse_stage=lse_stage,
        sems=sems,
    )
    state = (
        row_max,
        row_sum,
        o_scratch,
        checkpoint_sum,
        checkpoint_o,
        guard_smem,
        unit_guard,
    )
    replay = replay_refs(q_refs, kv_refs, replay_stages, replay_sems)
    return q_refs, kv_refs, out_refs, state, replay


def replay_refs(
    q_refs: QRefs,
    kv_refs: KVRefs,
    stages: tuple[jax.Array | None, ...],
    sems: jax.Array,
) -> ReplayRefs:
    """The replay's refs: the operands of q_refs and kv_refs with the
    replay's own stages (split_stages's) and semaphores."""
    q_buf, k_buf, v_buf, rotary_q_buf, rotary_k_buf = stages
    return ReplayRefs(
        q_refs=dataclasses.replace(
            q_refs, buf=q_buf, rotary_buf=rotary_q_buf, sems=sems
        ),
        kv_refs=dataclasses.replace(
            kv_refs,
            k_buf=k_buf,
            v_buf=v_buf,
            rotary_buf=rotary_k_buf,
            sems=sems,
        ),
    )


def first_row_carry(loop: VarlenMxuLoop) -> VarlenRowCarry:
    """Start the first two kv blocks and the first group's Q, wait for the
    first block, and return the carry the first group starts from."""
    traits, blocks = loop.traits, loop.blocks
    zero = jnp.int32(0)
    first_desc, second_desc, _ = blocks.first_two_blocks(zero, zero)
    loop.start_group_q(zero, zero, *loop.schedule.row_group(zero))
    blocks.start(first_desc, 0)
    wait_rotated_kv_block(traits, loop.kv_refs, 0)
    is_second_fresh = blocks.is_fresh(second_desc, first_desc).astype(jnp.int32)
    second_slot = is_second_fresh

    @pl.when(is_second_fresh > 0)
    def _start_second():
        blocks.start(second_desc, second_slot)

    return VarlenRowCarry(
        q_count=zero,
        stream=KVStream(
            slot=zero,
            next_slot=zero,
            ahead_slot=second_slot,
            is_ahead_fresh=is_second_fresh,
            next_desc=first_desc,
            ahead_desc=second_desc,
        ),
        probs=loop.pipeline.idle_state().probs,
        # With no row at all, the epilogue finishes the entry tile's parity.
        finished=FinishedGroup(
            is_live=zero,
            base_head=zero,
            first_si=zero,
            num_rows=jnp.int32(1),
            is_short=zero,
            parity=jnp.int32(ENTRY_PARITY),
            q_count=zero,
            out_count=zero,
            prev_out_tok=zero,
        ),
        last_mask=(loop.dead_mask(),) * loop.tiles.unit_tiles,
        prev_out_tok=zero,
        num_rows=zero,
    )


def varlen_mxu_scratch_shapes(
    traits: FwdTraits,
    *,
    q_dtype: jnp.dtype,
    k_dtype: jnp.dtype,
    v_dtype: jnp.dtype,
    rotary_dtype: jnp.dtype | None,
    return_lse: bool,
) -> list[pl.MemoryRef]:
    """Every scratch buffer of one explicit varlen build, in the order
    split_mxu_refs unpacks them: the stream stages, the row states, the
    check unit checkpoints, the replay's stages, the guard flags, the
    semaphores and the lse stage."""
    bq, bkv = traits.bq, traits.bkv
    num_stages = traits.num_stages
    head_dim_qk, head_dim_v = traits.head_dim_qk, traits.head_dim_v
    group_bq = traits.group_bq
    num_states = NUM_ROW_PARITIES * varlen_group_rows(
        group_bq // FRAGMENT_ROWS, varlen_pipeline_shape(traits)
    )
    rotary_dim, rotary_k = traits.rotary_dim, traits.rotary_k
    num_rotary_streams = (1 + int(rotary_k)) if rotary_dim else 0
    layout = QKVLayout.HEAD_DIM_MINOR
    vmem = pltpu.VMEM

    def _rotary_buffers(q_stages, kv_stages):
        if not rotary_dim:
            return []
        streams = [(bq, q_stages)] + ([(bkv, kv_stages)] if rotary_k else [])
        return [
            vmem(
                (stages, traits.rotary_groups, block, rotary_dim), rotary_dtype
            )
            for block, stages in streams
        ]

    scratch_shapes = [
        vmem(
            staged_shape(traits, bq, head_dim_qk, layout, num_stages), q_dtype
        ),
        vmem((NUM_KV_STAGES, bkv, head_dim_qk), k_dtype),
        vmem((NUM_KV_STAGES, bkv, head_dim_v), v_dtype),
        *_rotary_buffers(num_stages, NUM_KV_STAGES),
        vmem(staged_shape(traits, bq, head_dim_v, layout, num_stages), q_dtype),
        vmem((num_states, group_bq, NUM_LANES), jnp.float32),
        vmem((num_states, group_bq, NUM_LANES), jnp.float32),
        vmem((num_states, group_bq, head_dim_v), jnp.float32),
        vmem((NUM_CHECKPOINTS, group_bq, NUM_LANES), jnp.float32),
        vmem((NUM_CHECKPOINTS, group_bq, head_dim_v), jnp.float32),
        vmem(staged_shape(traits, bq, head_dim_qk, layout, 1), q_dtype),
        vmem((1, bkv, head_dim_qk), k_dtype),
        vmem((1, bkv, head_dim_v), v_dtype),
        *_rotary_buffers(1, 1),
        pltpu.SMEM((traits.head_fold,), jnp.int32),
        pltpu.SMEM((1,), jnp.int32),
        pltpu.SemaphoreType.DMA(
            (NUM_SEMS + num_rotary_streams, max(num_stages, NUM_KV_STAGES))
        ),
        pltpu.SemaphoreType.DMA((NUM_SEMS + num_rotary_streams, 1)),
    ]
    if return_lse:
        scratch_shapes.append(vmem((num_stages, bq, NUM_LANES), jnp.float32))
    return scratch_shapes


def pad_to_blocks(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    rotary: tuple[jax.Array, jax.Array | None] | None,
    traits: FwdTraits,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    tuple[jax.Array, jax.Array | None] | None,
    VarlenSchedule,
]:
    """q, k, v and rotary with their packed token axes padded to at least
    one block of traits, and the schedule of the padded totals. Padding rows
    lie past cu_seqlens[-1], so no block reads them as keys."""
    token_axis = 1
    num_pad_q = max(traits.bq - q.shape[token_axis], 0)
    num_pad_kv = max(traits.bkv - k.shape[token_axis], 0)
    padded_q = pad_axis(q, token_axis, num_pad_q)
    padded_k, padded_v = (
        pad_axis(operand, token_axis, num_pad_kv) for operand in (k, v)
    )
    if rotary is None:
        padded_rotary = None
    else:
        q_coeff, k_coeff = rotary
        padded_rotary = (
            pad_axis(q_coeff, token_axis, num_pad_q),
            None
            if k_coeff is None
            else pad_axis(k_coeff, token_axis, num_pad_kv),
        )
    padded_sched = VarlenSchedule(
        padded_total_q=padded_q.shape[token_axis],
        padded_total_k=padded_k.shape[token_axis],
    )
    return padded_q, padded_k, padded_v, padded_rotary, padded_sched


def run_padded_to_blocks(
    run_kernel: Callable[..., jax.Array | tuple[jax.Array, jax.Array]],
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    rotary: tuple[jax.Array, jax.Array | None] | None,
    traits: FwdTraits,
    *,
    return_lse: bool,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Run an explicit build through run_kernel (fwd_varlen.py's
    run_varlen_kernel, its other arguments bound) with the q heads stacked
    and the operands padded to whole blocks; slice the outputs back."""
    folded_traits = fold_q_heads(traits)
    q_seq_len = q.shape[1]
    padded_q, padded_k, padded_v, padded_rotary, padded_sched = pad_to_blocks(
        q, k, v, rotary, folded_traits
    )
    outputs = run_kernel(
        padded_q,
        padded_k,
        padded_v,
        padded_rotary,
        varlen_sched=padded_sched,
        traits=folded_traits,
    )
    if not return_lse:
        return outputs[:, :q_seq_len]
    o, lse = outputs
    return o[:, :q_seq_len], lse[:, :q_seq_len]
