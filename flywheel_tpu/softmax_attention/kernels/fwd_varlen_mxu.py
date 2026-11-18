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

On a chip with MXU accumulators a head_dim 256 varlen build
(varlen_uses_mxu_pipeline) runs varlen_mxu_fwd_body in place of
kernels/fwd_varlen.py's varlen_fwd_body. Its pipeline head stacks the q heads
of one kv head, which share every K and V tile, and a row's segments run its
live kv tiles in order: the row's start segment (its first live tile and the
next), then LOOP_BODY_TILES tiles a loop step. Each row leaves its last tile
in flight to the next row's start segment, and finishes right after it; rows
alternate between NUM_ROW_PARITIES parities.

The overflow guard checks a row's tiles by check units of UNIT_TILES tiles,
two loop bodies that may cross kv blocks (VarlenMxuLoop.checked_unit). A
unit's last tile also writes the row state it leaves into a checkpoint, the
next unit's entry state, and collects the row sums it writes. The next
unit's first body finishes that tile's softmaxes in its first qk_pop ticks,
and at its end those row sums decide whether the unit replays: from its own
checkpoint, through the first body's tiles. The next body adds nothing for
the items the replay redid. A row's final unit, from the unit after its last
checked one to its last tile, replays when the row's own sums flag it,
before its output is written (VarlenMxuLoop.complete_row). The row state
stays in place at the row's parity.
"""

import dataclasses
import functools
import math
from collections.abc import Callable, Sequence
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from flywheel_tpu.core.tpu import MXU_COLUMNS, NUM_LANES
from flywheel_tpu.softmax_attention.block_stream import (
    BlockDesc,
    VarlenBlockStream,
)
from flywheel_tpu.softmax_attention.config import (
    DEFAULT_MASK_VALUE,
    BlockSizes,
    QKVLayout,
    TokenMajorInfo,
)
from flywheel_tpu.softmax_attention.dma import advance, drain
from flywheel_tpu.softmax_attention.fwd_blocks import (
    NUM_KV_STAGES,
    NUM_SEMS,
    KVRefs,
    OutRefs,
    QRefs,
    RowFinish,
    carried_finish,
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
    REPLAY_RESULT_SLOTS,
    ROW_START_TILES,
    UNIT_TILES,
    FlaggedTailOps,
    LargestSums,
    RowCheckpoints,
    UnitCheckpoint,
    accumulate_fragment,
    aligned_rows,
    first_final_unit,
    fold_fragment_scores,
    fold_replayed_fragment,
    fold_unit_end_scores,
    fragment_rows,
    guard_flag,
    mxu_pipeline_applies,
    replay_matmul,
    row_finish_state,
    scaled_logits,
    stage_row_finish,
)
from flywheel_tpu.softmax_attention.inputs import pad_axis
from flywheel_tpu.softmax_attention.mxu_pipeline import (
    FRAGMENT_ROWS,
    FragmentOps,
    MxuPipeline,
    PipelineState,
    Tail,
    WorkItem,
    slot_ring_rows,
)
from flywheel_tpu.softmax_attention.schedule import (
    FwdSchedule,
    ScheduleRow,
    VarlenSchedule,
)
from flywheel_tpu.softmax_attention.traits import FwdTraits
from flywheel_tpu.softmax_attention.variant import Variant, VariantOperands

# The q rows a pipeline head stages on the explicit path: long rows spread a
# row's start and finish over more work items, short ones pad less.
LONG_PIPELINE_ROWS = 2048
SHORT_PIPELINE_ROWS = 1024
# Dispatch's block_q follows the longest sequence; from this one up the build
# takes long rows.
LONG_ROWS_MIN_BLOCK_Q = 512
# The row finish normalizes at most this many rows a step.
FINISH_STEP_ROWS = 256
# Work item modes of the explicit loop: a row's first live tile takes its
# anchor and starts its sums, every other tile reads them.
ROW_START = "row_start"
ROW_TILE = "row_tile"
# A key index past every packed kv axis: a tile from it hides every key.
DEAD_KEY = 1 << 30


def varlen_uses_mxu_pipeline(
    *,
    variant: Variant,
    block_sizes: BlockSizes,
    head_dim: int,
    head_dim_v: int,
    head_fold: int,
    transposed_pv: bool,
    token_major: TokenMajorInfo | None,
) -> bool:
    """Whether a varlen build runs its fragments on the explicit MXU pipeline
    (fwd_mxu.mxu_pipeline_applies, plus what this loop supports).

    The loop stages one kv head's q heads head-major as one pipeline head, so
    it takes head_fold 1 and head-major q, k and v (a token-major fold rides
    the lanes of one head per K tile); varlen_mxu_block_sizes then sets its
    blocks, whatever dispatch picked.
    """
    return (
        head_fold == 1
        and token_major is None
        and mxu_pipeline_applies(
            variant=variant,
            block_sizes=BlockSizes(
                block_q=SHORT_PIPELINE_ROWS,
                block_kv=explicit_block_kv(block_sizes.block_kv),
                qkv_layout=block_sizes.qkv_layout,
            ),
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            transposed_pv=transposed_pv,
            group_rows=SHORT_PIPELINE_ROWS,
        )
    )


def explicit_block_kv(block_kv: int) -> int:
    """The kv block of an explicit build: dispatch's, rounded up to whole
    loop steps of MXU tiles and at least ROW_START_TILES tiles."""
    num_tiles = max(-(-block_kv // MXU_COLUMNS), ROW_START_TILES)
    return -(-num_tiles // LOOP_BODY_TILES) * LOOP_BODY_TILES * MXU_COLUMNS


def varlen_mxu_block_sizes(
    block_sizes: BlockSizes, q_heads_per_kv_head: int
) -> BlockSizes:
    """The blocks of an explicit varlen build from dispatch's block_sizes.

    A pipeline head stacks the q_heads_per_kv_head q heads of one kv head, so
    block_q counts tokens per head: a pipeline head holds LONG_PIPELINE_ROWS
    rows when dispatch's block_q (which follows the longest sequence) is at
    least LONG_ROWS_MIN_BLOCK_Q, else SHORT_PIPELINE_ROWS, rounded up so a
    tile's work items fill whole result slot rings. block_q_compute is the
    row finish's step, at most FINISH_STEP_ROWS rows of all folded heads.
    """
    ring_rows = slot_ring_rows()
    target_rows = (
        LONG_PIPELINE_ROWS
        if block_sizes.block_q >= LONG_ROWS_MIN_BLOCK_Q
        else SHORT_PIPELINE_ROWS
    )
    # The smallest per-head block whose folded rows fill whole rings.
    unit = max(
        ring_rows // math.gcd(ring_rows, q_heads_per_kv_head), FRAGMENT_ROWS
    )
    block_q = max(-(-target_rows // (q_heads_per_kv_head * unit)), 1) * unit
    block_q_compute = FRAGMENT_ROWS
    while (
        2 * block_q_compute * q_heads_per_kv_head <= FINISH_STEP_ROWS
        and block_q % (2 * block_q_compute) == 0
    ):
        block_q_compute *= 2
    return BlockSizes(
        block_q=block_q,
        block_kv=explicit_block_kv(block_sizes.block_kv),
        block_kv_compute=MXU_COLUMNS,
        block_q_compute=block_q_compute,
        num_stages=block_sizes.num_stages,
        qkv_layout=QKVLayout.HEAD_DIM_MINOR,
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class FoldedVarlenTraits(FwdTraits):
    """A packed varlen build on the explicit MXU pipeline. Its pipeline head
    stacks the gqa_fold q heads of one kv head, which share every K and V
    tile, so a tile serves gqa_fold * bq rows; block_q counts tokens per
    head."""

    gqa_fold: int = dataclasses.field()

    def __post_init__(self) -> None:
        assert self.head_fold == 1, self.head_fold
        assert self.num_q_heads == self.num_kv_heads * self.gqa_fold, self
        super().__post_init__()


def fold_q_heads(traits: FwdTraits) -> FoldedVarlenTraits:
    """traits with each kv head's q heads stacked in one pipeline head."""
    fields = {
        field.name: getattr(traits, field.name)
        for field in dataclasses.fields(traits)
        if field.init
    }
    return FoldedVarlenTraits(**fields, gqa_fold=traits.q_heads_per_kv_head)


def chunk_rows(
    traits: FwdTraits, chunk: jax.Array | int, head: jax.Array | int
) -> pl.Slice:
    """The row-state rows of q chunk chunk (FRAGMENT_ROWS tokens) of folded
    head head. Rows run (q compute tile, folded head, token), as the row
    finish reads them."""
    token = chunk * FRAGMENT_ROWS
    q_tile = token // traits.bq_compute
    first_row = (
        q_tile * traits.frag_rows
        + head * traits.bq_compute
        + token % traits.bq_compute
    )
    return pl.ds(aligned_rows(first_row), FRAGMENT_ROWS)


class TileMask(NamedTuple):
    """What masks a kv tile of a row: its block and its first key in the
    packed kv axis. A first key past every sequence's keys hides them all."""

    kv: jax.Array
    first_key: jax.Array


def tile_bounds(
    schedule: FwdSchedule, ctx: ScheduleRow, mask: TileMask
) -> jax.Array:
    """The (FRAGMENT_ROWS, MXU_COLUMNS) bounds of a tile of the row of ctx:
    per key, the first q row past the row's first token that sees it. They
    hold for every folded head."""
    key_ids = mask.first_key + lax.broadcasted_iota(
        jnp.int32, (1, MXU_COLUMNS), 1
    )
    q_lower, _ = schedule.q_bounds(ctx, mask.kv, key_ids)
    return jnp.broadcast_to(q_lower - ctx.q_tok, (FRAGMENT_ROWS, MXU_COLUMNS))


def fragment_row_ids() -> jax.Array:
    """Each fragment row's index, along the keys of a tile."""
    return lax.broadcasted_iota(jnp.int32, (FRAGMENT_ROWS, MXU_COLUMNS), 0)


def mask_fragment(
    variant: Variant,
    bounds: jax.Array,
    row_ids: jax.Array | None,
    chunk: jax.Array | int,
    scores: jax.Array,
) -> jax.Array:
    """q chunk chunk's (FRAGMENT_ROWS, MXU_COLUMNS) raw scores of unscaled Q
    in log2 units: the Q scale and softcap applied, and the mask value
    wherever tile_bounds's bounds hide a key from a q row. row_ids is
    fragment_row_ids's, or None for bounds that already count from each
    fragment row (VarlenFragmentOps.masks)."""
    first_row = chunk * FRAGMENT_ROWS
    if row_ids is None:
        is_visible = first_row >= bounds
    else:
        is_visible = row_ids + first_row >= bounds
    return jnp.where(
        is_visible, scaled_logits(variant, scores), DEFAULT_MASK_VALUE
    )


def staged_q_head(
    traits: FwdTraits, q_slot_ref: jax.Array, head: jax.Array | int
) -> jax.Array:
    """Folded head head's (bq, head_dim) rows of a Q staging slot; a build of
    one q head per pipeline head stages no fold axis."""
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
    """fwd_blocks.fill_packed_padding as a loop over the stage slot's tiles
    of FRAGMENT_ROWS rows at or past packed_q_end, of no step for a block
    that ends before it."""
    # Note (david): the compiler predicates fill_packed_padding's guarded
    # whole-slot select into every row's finish; a loop of no step costs a
    # row next to nothing.
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
    """Where a segment's kv tile sits, its K and V staging slot and first key
    row in the slot, and what masks it."""

    slot: jax.Array
    start: jax.Array
    mask: TileMask


@dataclasses.dataclass(frozen=True, eq=False)
class VarlenFragmentOps:
    """The FragmentOps of a segment of the row of ctx: q fragments from the
    row's Q slot, K and V tiles as tiles places the segment's tile keys, and
    the bounds-masked softmax into the row state of parity's rows; unit_end
    names a check unit's last tile (UnitCheckpoint)."""

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

    def _tile_rows(self, tile: int) -> tuple[jax.Array, pl.Slice]:
        ref = self.tiles[tile]
        start = pl.multiple_of(ref.start, MXU_COLUMNS)
        return ref.slot, pl.ds(start, MXU_COLUMNS)

    def k_tile(self, tile: int) -> jax.Array:
        slot, rows = self._tile_rows(tile)
        return self.k_buf[slot, rows, :]

    def v_tile(self, tile: int) -> jax.Array:
        slot, rows = self._tile_rows(tile)
        return self.v_buf[slot, rows, :]

    def q_fragment(self, item: WorkItem) -> jax.Array:
        chunk, head = item.fragment
        q_ref = staged_q_head(self.traits, self.q_buf.at[self.q_slot], head)
        return q_ref[fragment_rows(chunk), :]

    @functools.cached_property
    def masks(self) -> tuple[tuple[jax.Array, ...], jax.Array | None]:
        """Each tile's bounds and the fragment row ids, built once for all of
        the segment's items from the row's scalars. With one q head a
        pipeline head, no other head's item shares a chunk's compare, so the
        bounds count from each fragment row instead (row ids None)."""
        bounds = tuple(
            tile_bounds(self.schedule, self.ctx, tile.mask)
            for tile in self.tiles
        )
        row_ids = fragment_row_ids()
        if self.traits.gqa_fold > 1:
            return bounds, row_ids
        return tuple(tile - row_ids for tile in bounds), None

    def rows(self, item: WorkItem) -> pl.Slice:
        """The row-state rows of item's fragment."""
        chunk, head = item.fragment
        return chunk_rows(self.traits, chunk, head)

    def masked_scores(self, item: WorkItem, scores: jax.Array) -> jax.Array:
        """item's raw scores as masked log2-unit logits (mask_fragment)."""
        chunk, _ = item.fragment
        bounds, row_ids = self.masks
        return mask_fragment(
            self.variant, bounds[item.tile], row_ids, chunk, scores
        )

    def is_unit_end(self, item: WorkItem) -> bool:
        return self.unit_end is not None and item.tile == self.unit_end.tile

    def softmax(self, item: WorkItem, scores: jax.Array) -> jax.Array:
        rows, masked = self.rows(item), self.masked_scores(item, scores)
        if not self.is_unit_end(item):
            return fold_fragment_scores(
                self.row_max,
                self.row_sum,
                rows,
                masked,
                is_row_start=item.mode == ROW_START,
            )
        assert item.mode == ROW_TILE, item
        return fold_unit_end_scores(
            self.row_max, self.row_sum, rows, masked, self.unit_end
        )

    def accumulate(self, item: WorkItem, weighted_values: jax.Array) -> None:
        accumulate_fragment(
            self.o,
            self.rows(item),
            weighted_values,
            is_row_start=item.mode == ROW_START,
            checkpoint_o=self.unit_end.o if self.is_unit_end(item) else None,
        )


class KVStream(NamedTuple):
    """The kv ring while block j runs: block j's slot, block j + 1's (staged,
    so its first K tile can be pushed ahead) and block j + 2's, whether j + 2
    is in flight, and the descriptors of j + 1 and j + 2."""

    slot: jax.Array
    next_slot: jax.Array
    ahead_slot: jax.Array
    is_ahead_fresh: jax.Array
    next_desc: BlockDesc
    ahead_desc: BlockDesc


# What a tile segment hands to the next: the pipeline's carried probabilities,
# the kv stream and the mask of the segment's last tile.
TileCarry = tuple[tuple[jax.Array, ...], KVStream, TileMask]


class RowSpan(NamedTuple):
    """The live kv tiles of a row: its first kv block, its first live tile in
    that block, how many live tiles it runs, and the next row's first live
    tile in its own first block."""

    lo: jax.Array
    first_tile: jax.Array
    num_tiles: jax.Array
    next_row_tile: jax.Array


class GridRow(NamedTuple):
    """A row of the schedule's grid: its head group's first head, its last
    kv block, its schedule row and its first q token."""

    base_head: jax.Array
    hi: jax.Array
    ctx: ScheduleRow
    q_tok: jax.Array


class RowRun(NamedTuple):
    """A row while its tiles run: its Q slot, its row-state parity, its live
    tiles, the row, and the first two block descriptors of the row after it
    (VarlenBlockStream.ahead_desc's ahead)."""

    q_slot: jax.Array
    parity: jax.Array
    span: RowSpan
    row: GridRow
    ahead: tuple[BlockDesc, BlockDesc]


class VarlenRowCarry(NamedTuple):
    """What one row of the explicit loop hands to the next: the Q stream, the
    kv stream, the pipeline's carried probabilities, the row's schedule row
    and the mask of its last tile (whose items are still in flight), and the
    row's deferred finish."""

    q_loaded: jax.Array
    q_inflight: jax.Array
    q_slot: jax.Array
    stream: KVStream
    probs: tuple[jax.Array, ...]
    ctx: ScheduleRow
    last_mask: TileMask
    prev_out_tok: jax.Array
    num_rows: jax.Array
    finish: RowFinish


@dataclasses.dataclass(frozen=True, eq=False)
class ReplayRefs:
    """The guard replay's own staging: one Q row, one kv block and their
    semaphores, apart from the streams in flight."""

    q_refs: QRefs
    kv_refs: KVRefs


class ReplayedRow(NamedTuple):
    """A row as a replay recomputes it: its staged Q (slot q_slot of q_buf),
    its row-state parity, its first kv block and first live tile in it, its
    schedule row and its kv head."""

    q_buf: jax.Array
    q_slot: jax.Array | int
    parity: jax.Array
    lo: jax.Array
    first_tile: jax.Array
    ctx: ScheduleRow
    kv_head: jax.Array


@dataclasses.dataclass(frozen=True, eq=False)
class VarlenMxuLoop:
    """The explicit loop's static context: the build, the schedule and every
    ref the loop's steps read. The row state holds NUM_ROW_PARITIES slots of
    anchors, lane partial sums and output accumulators, and checkpoint_sum
    and checkpoint_o a check unit's entry state (UnitCheckpoint). In SMEM,
    guard holds the flag finish_row_sums sets for a row and unit_guard the
    flag of a check unit (checked_unit)."""

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
        """The q chunks of FRAGMENT_ROWS tokens a pipeline head's block
        holds per folded head."""
        return self.traits.bq // FRAGMENT_ROWS

    @property
    def has_half_tiles(self) -> bool:
        """Whether a tile may run its second half's chunks alone: half its
        items still fill whole result slot rings and a segment."""
        half_items = self.num_chunks // 2 * self.traits.gqa_fold
        return (
            self.num_chunks % 2 == 0
            and half_items % self.pipeline.num_slots == 0
            and half_items >= self.pipeline.delays.pv_pop
        )

    def tile_items(
        self, tile: int, mode: str, *, is_half: bool = False
    ) -> tuple[WorkItem, ...]:
        """One tile's work items, chunk-major, or its second half's alone
        when is_half: every tile, whole or half, ends on the same fragments,
        so all tails match."""
        first_chunk = self.num_chunks // 2 if is_half else 0
        return tuple(
            WorkItem(fragment=(chunk, head), tile=tile, mode=mode)
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
        of the row's last live tile, so that tile runs as a half tile. A
        causal row's last tile is mostly past its first half's diagonal."""
        if not self.has_half_tiles:
            return False
        traits, schedule = self.traits, self.schedule
        hi, ctx = row.hi, row.ctx
        tile = lax.rem(
            span.first_tile + span.num_tiles - 1, self.num_block_tiles
        )
        first_key = self.tile_mask(ctx, hi, tile).first_key
        # Every key below the block's own first one belongs to the block
        # before; past it, a key's lowest q row only grows with the key.
        owned_lo = jnp.maximum(ctx.kv_start, ctx.kv_base + hi * traits.bkv)
        q_lower, _ = schedule.q_bounds(
            ctx, hi, jnp.maximum(first_key, owned_lo)
        )
        return q_lower - row.q_tok >= traits.bq // 2

    def ops(
        self,
        q_slot: jax.Array,
        parity: jax.Array | int,
        ctx: ScheduleRow,
        tiles: tuple[TileRef, ...],
    ) -> VarlenFragmentOps:
        return VarlenFragmentOps(
            traits=self.traits,
            variant=self.variant,
            schedule=self.schedule,
            ctx=ctx,
            q_buf=self.q_refs.buf,
            q_slot=q_slot,
            k_buf=self.kv_refs.k_buf,
            v_buf=self.kv_refs.v_buf,
            row_max=self.row_max.at[parity],
            row_sum=self.row_sum.at[parity],
            o=self.o.at[parity],
            tiles=tiles,
        )

    def tile_tail(
        self,
        q_slot: jax.Array,
        parity: jax.Array | int,
        ctx: ScheduleRow,
        mask: TileMask,
    ) -> Tail:
        """The tail every segment leaves: one tile of the row of ctx at
        parity, masked by mask, with the next segment's first K tile staged.
        Its items' K and V pushes are all behind it."""
        zero = jnp.int32(0)
        ops = self.ops(q_slot, parity, ctx, (TileRef(zero, zero, mask),))
        return self.pipeline.tail_of(
            self.tile_items(0, ROW_TILE), ops, has_staged_next_k=True
        )

    def k_tile_at(self, slot: jax.Array, tile: jax.Array) -> jax.Array:
        start = pl.multiple_of(tile * MXU_COLUMNS, MXU_COLUMNS)
        return self.kv_refs.k_buf[slot, pl.ds(start, MXU_COLUMNS), :]

    def next_k_tile(
        self,
        stream: KVStream,
        next_tile: jax.Array,
        is_row_end: jax.Array,
        next_row_tile: jax.Array,
    ) -> Callable[[], jax.Array]:
        """The K tile after a segment: tile next_tile of the running block
        (one past its last means the next block's first), or the next row's
        first live tile, next_row_tile of the next block, at the row's end."""
        is_in_block = jnp.logical_and(
            next_tile < self.num_block_tiles, jnp.logical_not(is_row_end)
        )
        slot = jnp.where(is_in_block, stream.slot, stream.next_slot)
        tile = jnp.where(
            is_row_end, next_row_tile, lax.rem(next_tile, self.num_block_tiles)
        )
        return lambda: self.k_tile_at(slot, tile)

    def step_block(
        self,
        stream: KVStream,
        is_step: jax.Array,
        row: GridRow,
        kv: jax.Array,
        ahead: tuple[BlockDesc, BlockDesc],
    ) -> KVStream:
        """Block kv's step where is_step holds: wait for block kv + 1, whose
        first K tile the block's last segment stages, and start block
        kv + 2. Returns the stream while kv runs."""
        traits, blocks = self.traits, self.blocks
        ahead_desc = blocks.ahead_desc(
            row.base_head, kv, row.hi, row.ctx, ahead
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

    def run_row(
        self, g: jax.Array, si: jax.Array, carry: VarlenRowCarry
    ) -> VarlenRowCarry:
        """One row: its start segment with the previous row's finish behind
        it, then its other live tiles; the row's last tile stays in flight.
        Rows alternate between the NUM_ROW_PARITIES slots of the row state."""
        traits = self.traits
        parity = lax.rem(carry.num_rows, NUM_ROW_PARITIES)
        span, row, ahead, next_row = self.row_plan(g, si)
        q_loaded, q_inflight, q_slot = self.advance_q_row(
            g, si, row, next_row, carry
        )
        stream = self.step_block(
            carry.stream, jnp.bool_(True), row, span.lo, ahead
        )
        is_last_half = self.is_last_tile_half(span, row)
        prev_finish = carried_finish(carry.finish)._replace(parity=1 - parity)
        wait_row_out_slot(
            traits,
            self.out_refs,
            prev_finish,
            row_out_slot(traits, prev_finish),
        )
        run = RowRun(q_slot, parity, span, row, ahead)
        probs, last_mask = self.start_row(carry, run, stream, is_last_half)
        self.complete_row(
            prev_finish, self.tile_tail(q_slot, parity, row.ctx, last_mask)
        )
        probs, stream, last_mask = self.run_tiles(
            (probs, stream, last_mask), run, is_last_half
        )
        finish = RowFinish(
            is_live=jnp.int32(1),
            si=si,
            q_tok=row.q_tok,
            base_head=row.base_head,
            num_blend_rows=self.schedule.blend_start(row.ctx) - row.q_tok,
            prev_out_tok=carry.prev_out_tok,
            out_count=carry.num_rows,
            parity=parity,
        )
        return VarlenRowCarry(
            q_loaded=q_loaded,
            q_inflight=q_inflight,
            q_slot=q_slot,
            stream=stream,
            probs=probs,
            ctx=row.ctx,
            last_mask=last_mask,
            prev_out_tok=row.q_tok,
            num_rows=carry.num_rows + 1,
            finish=finish,
        )

    def advance_q_row(
        self,
        g: jax.Array,
        si: jax.Array,
        row: GridRow,
        next_row: tuple[jax.Array, jax.Array, jax.Array],
        carry: VarlenRowCarry,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Wait for the Q (rotated, unscaled) of row si of head group g and
        start the next row's; next_row is row_plan's. Returns the Q ring's
        (loaded, in flight, slot)."""
        traits, schedule = self.traits, self.schedule
        next_g, next_si, next_q_tok = next_row
        return advance(
            g * schedule.num_rows + si,
            next_g * schedule.num_rows + next_si,
            lambda slot: start_q_row(
                traits, self.q_refs, row.base_head, row.q_tok, slot
            ),
            lambda slot: start_q_row(
                traits,
                self.q_refs,
                next_g * traits.group_heads,
                next_q_tok,
                slot,
            ),
            partial(wait_rotated_q_row, traits, self.q_refs),
            carry.q_loaded,
            carry.q_inflight,
            carry.q_slot,
            traits.num_stages,
        )

    def row_plan(
        self, g: jax.Array, si: jax.Array
    ) -> tuple[RowSpan, GridRow, tuple[BlockDesc, BlockDesc], tuple]:
        """Row si of head group g: its live tiles, the row, the first two
        block descriptors of the row after it, and that row's (head group,
        row, q token)."""
        traits, schedule, blocks = self.traits, self.schedule, self.blocks
        lo, hi, ctx = schedule.row_interval(si)
        next_g, next_si = blocks.row_after(g, si)
        first_next_desc, second_next_desc, (next_lo, _, next_ctx) = (
            blocks.first_two_blocks(next_g, next_si)
        )
        span = RowSpan(
            lo,
            *self.live_tiles(lo, hi, ctx),
            schedule.first_live_tile(next_lo, next_ctx),
        )
        row = GridRow(g * traits.group_heads, hi, ctx, schedule.q_token(ctx))
        ahead = (first_next_desc, second_next_desc)
        return span, row, ahead, (next_g, next_si, schedule.q_token(next_ctx))

    def live_tiles(
        self, lo: jax.Array, hi: jax.Array, ctx: ScheduleRow
    ) -> tuple[jax.Array, jax.Array]:
        """(first live tile in block lo, live tiles) of the row of ctx over
        kv blocks lo..hi."""
        schedule = self.schedule
        num_tiles = (
            (hi - lo) * self.num_block_tiles
            + schedule.num_live_tiles(hi, ctx)
            - schedule.first_live_tile(hi, ctx)
        )
        return schedule.first_live_tile(lo, ctx), num_tiles

    def start_row(
        self,
        carry: VarlenRowCarry,
        run: RowRun,
        stream: KVStream,
        is_last_half: jax.Array | bool,
    ) -> tuple[tuple[jax.Array, ...], TileMask]:
        """The row's start segment, which finishes the previous row's last
        tile: the row's first live tile in row-start mode and the next one,
        a half tile when it is the row's last and is_last_half, or for a
        one-tile row the first tile again under a mask that hides every key
        (a half tile where the build has them). The previous row's finish is
        staged in the same straight run, which the segment's MXU stream
        overlaps. Returns the probabilities and the second tile's mask."""
        lo, first_tile, num_tiles, next_row_tile = run.span
        ctx = run.row.ctx
        has_second = num_tiles >= ROW_START_TILES
        second_tile = jnp.where(has_second, first_tile + 1, first_tile)
        second_mask = jax.tree.map(
            lambda live, dead: jnp.where(has_second, live, dead),
            self.tile_mask(ctx, lo, second_tile),
            self.dead_mask(),
        )
        tiles = (
            TileRef(
                stream.slot,
                first_tile * MXU_COLUMNS,
                self.tile_mask(ctx, lo, first_tile),
            ),
            TileRef(stream.slot, second_tile * MXU_COLUMNS, second_mask),
        )
        next_k_tile = self.next_k_tile(
            stream,
            first_tile + ROW_START_TILES,
            num_tiles <= ROW_START_TILES,
            next_row_tile,
        )
        segment = partial(self.start_segment, carry, run, tiles, next_k_tile)
        is_second_half = jnp.logical_and(
            self.has_half_tiles,
            jnp.logical_or(
                num_tiles == 1,
                jnp.logical_and(num_tiles == ROW_START_TILES, is_last_half),
            ),
        )
        probs = lax.cond(
            run.parity == 0,
            partial(segment, 0, is_second_half),
            partial(segment, 1, is_second_half),
            carry.probs,
        )
        return probs, second_mask

    def start_segment(
        self,
        carry: VarlenRowCarry,
        run: RowRun,
        tiles: tuple[TileRef, TileRef],
        next_k_tile: Callable[[], jax.Array],
        parity: int,
        is_second_half: jax.Array,
        probs: tuple[jax.Array, ...],
    ) -> tuple[jax.Array, ...]:
        """start_row's segment of tiles at a static parity, run's as a Python
        int, with the second tile a half tile where is_second_half holds: a
        static parity keeps the previous row's state and this row's provably
        apart, so the previous row's finish overlaps the segment."""
        pipeline = self.pipeline
        q_slot, ctx = run.q_slot, run.row.ctx
        prev_finish = carried_finish(carry.finish)._replace(parity=1 - parity)

        def _segment(is_half, probs):
            state = pipeline.advance(
                PipelineState(
                    probs=probs,
                    tail=self.tile_tail(
                        q_slot, 1 - parity, carry.ctx, carry.last_mask
                    ),
                ),
                self.tile_items(0, ROW_START)
                + self.tile_items(1, ROW_TILE, is_half=is_half),
                self.ops(q_slot, parity, ctx, tiles),
                next_k_tile=next_k_tile,
            )
            pipeline.check_loop_tail(
                state.tail, self.tile_tail(q_slot, parity, ctx, tiles[1].mask)
            )
            self.stage_row_finish(prev_finish)
            return state.probs

        if not self.has_half_tiles:
            return _segment(False, probs)
        return lax.cond(
            is_second_half,
            partial(_segment, True),
            partial(_segment, False),
            probs,
        )

    def segment_items(
        self, num_tiles: int, *, is_half: bool
    ) -> tuple[WorkItem, ...]:
        """The items of a segment of num_tiles tiles after a row's start
        segment, the last a half tile when is_half."""
        return sum(
            (
                self.tile_items(
                    offset,
                    ROW_TILE,
                    is_half=is_half and offset == num_tiles - 1,
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
        """advance_tiles's segment with the row's own ops, as a loop step."""
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
    ) -> tuple[PipelineState, KVStream, TileMask]:
        """The segment of num_segment_tiles live tiles of the row from tile
        index on, all in one block, the last a half tile when is_half; under
        steps_block it first steps the block when the segment opens it.
        ops_of maps the segment's ops, tail_ops_of the ops that finish the
        segment before. Returns the pipeline state, the kv stream and the
        mask of the segment's last tile."""
        pipeline = self.pipeline
        lo, first_tile, num_tiles, next_row_tile = run.span
        ctx = run.row.ctx
        probs, stream, prev_mask = tile_carry
        position = index + first_tile
        tile = lax.rem(position, self.num_block_tiles)
        kv = lo + position // self.num_block_tiles
        if steps_block:
            stream = self.step_block(stream, tile == 0, run.row, kv, run.ahead)
        tiles = tuple(
            TileRef(
                stream.slot,
                (tile + offset) * MXU_COLUMNS,
                self.tile_mask(ctx, kv, tile + offset),
            )
            for offset in range(num_segment_tiles)
        )
        ops = self.ops(run.q_slot, run.parity, ctx, tiles)
        assumed = self.tile_tail(run.q_slot, run.parity, ctx, prev_mask)
        tail = assumed
        if tail_ops_of is not None:
            tail = dataclasses.replace(assumed, ops=tail_ops_of(assumed.ops))
        state = pipeline.advance(
            PipelineState(probs=probs, tail=tail),
            self.segment_items(num_segment_tiles, is_half=is_half),
            ops if ops_of is None else ops_of(ops),
            next_k_tile=self.next_k_tile(
                stream,
                tile + num_segment_tiles,
                index + num_segment_tiles == num_tiles,
                next_row_tile,
            ),
        )
        pipeline.check_loop_tail(state.tail, assumed)
        return state, stream, tiles[-1].mask

    def run_tiles(
        self,
        tile_carry: TileCarry,
        run: RowRun,
        is_last_half: jax.Array | bool,
    ) -> TileCarry:
        """The row's live tiles after its start segment: its check units
        (checked_units), run_whole_tiles's from the tile after them, then the
        one whole tile they leave over, then the last tile as a half tile when
        is_last_half."""
        num_tiles = run.span.num_tiles
        has_half_last = jnp.logical_and(
            is_last_half, num_tiles > ROW_START_TILES
        )
        whole_end = jnp.maximum(
            num_tiles - has_half_last.astype(jnp.int32), ROW_START_TILES
        )
        first_whole = ROW_START_TILES
        if self.has_checked_units:
            tile_carry, first_whole = self.checked_units(tile_carry, run)
        tile_carry = self.run_whole_tiles(
            tile_carry, run, first_whole, whole_end
        )
        tile_carry = self.tile_segment_where(
            (whole_end - ROW_START_TILES) % LOOP_BODY_TILES > 0,
            whole_end - 1,
            tile_carry,
            run,
            is_half=False,
        )
        if self.has_half_tiles:
            tile_carry = self.tile_segment_where(
                has_half_last, num_tiles - 1, tile_carry, run, is_half=True
            )
        return tile_carry

    def run_whole_tiles(
        self,
        tile_carry: TileCarry,
        run: RowRun,
        first: jax.Array | int,
        whole_end: jax.Array,
    ) -> TileCarry:
        """The row's whole tiles from first up to whole_end, LOOP_BODY_TILES
        a step: per kv block its step, then an inner loop whose body is the
        segment alone. Every block but the last holds an even count of them
        (a row that spans blocks starts at its first block's tile 0, and a
        block holds an even count of tiles), so at most one is left over."""
        row, ahead = run.row, run.ahead
        lo, first_tile = run.span.lo, run.span.first_tile
        tiles_a_block = self.num_block_tiles

        def _block_of(index):
            return lo + (index + first_tile) // tiles_a_block

        def _run_block(kv, tile_carry):
            # A block that opens before first stepped with the segment that
            # opened it (the start segment or a check unit's).
            probs, stream, prev_mask = tile_carry
            block_first = (kv - lo) * tiles_a_block - first_tile
            stream = self.step_block(
                stream, block_first >= first, row, kv, ahead
            )
            begin = jnp.maximum(block_first, first)
            end = jnp.minimum(block_first + tiles_a_block, whole_end)
            return lax.fori_loop(
                0,
                (end - begin) // LOOP_BODY_TILES,
                lambda step, step_carry: self.tile_segment(
                    begin + LOOP_BODY_TILES * step,
                    step_carry,
                    run,
                    num_segment_tiles=LOOP_BODY_TILES,
                    is_half=False,
                    steps_block=False,
                ),
                (probs, stream, prev_mask),
            )

        return lax.fori_loop(
            _block_of(first),
            jnp.where(whole_end > first, _block_of(whole_end - 1) + 1, lo),
            _run_block,
            tile_carry,
        )

    def final_unit(self, num_tiles: jax.Array | int) -> jax.Array | int:
        """The first check unit of a row's final unit, given its live tiles:
        the units before it are checked (checked_unit), and the loop step of
        its own last body, whose first body is the row's last and checks
        nothing, ends at least two whole tiles before the row's last tile; 0
        when the row has no checked unit."""
        return first_final_unit(num_tiles // UNIT_TILES)

    @property
    def has_checked_units(self) -> bool:
        """Whether a row over the whole packed kv axis can hold a checked
        unit."""
        max_row_tiles = self.schedule.num_kv_blocks * self.num_block_tiles
        return self.final_unit(max_row_tiles) > 0

    def unit_end_ops(
        self,
        ops: VarlenFragmentOps,
        *,
        slot: jax.Array,
        largest: LargestSums,
        tile: int,
    ) -> VarlenFragmentOps:
        """ops whose items of tile key tile, a check unit's last tile, also
        write the row state they leave into checkpoint slot slot and fold
        their row sums into largest (UnitCheckpoint)."""
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
        """Run the row's check units after its start segment (checked_unit),
        up to the first body of the unit after its final unit; returns the
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
        """Run the row's check unit unit's last body, whose last tile also
        writes the next unit's entry state into that unit's checkpoint and
        collects its row sums (unit_end_ops), then the next unit's first
        body, which finishes that tile: its row sums flag unit, and a unit
        before final_unit replays at the first body's end (replay_tiles),
        through the first body's tiles. The last body adds nothing for the
        items of the first body before it that a replay redid
        (FlaggedTailOps). A first body may open a kv block; a last body opens
        one only when a block holds one body."""
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
            steps_block=self.num_block_tiles == LOOP_BODY_TILES,
            ops_of=partial(unit_end_ops, tile=LOOP_BODY_TILES - 1),
            tail_ops_of=partial(
                FlaggedTailOps,
                is_flagged=self.unit_guard[0] > 0,
                plain_mode=ROW_TILE,
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

    def tile_segment_where(
        self,
        is_run: jax.Array,
        index: jax.Array,
        tile_carry: TileCarry,
        run: RowRun,
        *,
        is_half: bool,
    ) -> TileCarry:
        """tile_segment's one tile at live tile index where is_run holds, a
        half tile when is_half; only a half tile, the row's last, may open a
        kv block."""
        return lax.cond(
            is_run,
            partial(
                self.tile_segment,
                index,
                run=run,
                num_segment_tiles=1,
                is_half=is_half,
                steps_block=is_half,
            ),
            lambda tile_carry: tile_carry,
            tile_carry,
        )

    def complete_row(self, finish: RowFinish, tail: Tail) -> None:
        """A row's finish after stage_row_finish: if the guard flagged the
        row, the replay of its final unit (replay_row_end) behind tail, the
        pipeline's tail, and a second staging; then the writes, if the row is
        one."""

        @pl.when(self.guard[0] > 0)
        def _replay():
            self.replay_row_end(
                finish, self.pipeline.free_slots(tail, REPLAY_RESULT_SLOTS)
            )
            self.stage_row_finish(finish)

        @pl.when(finish.is_live > 0)
        def _write_row():
            packed_q_end = self.schedule.packed_q_end
            copy_row_out(
                self.traits,
                self.out_refs,
                finish,
                row_out_slot(self.traits, finish),
                packed_q_end,
                fill_padding_tiles,
            )
            if self.variant.return_lse:
                copy_row_lse(
                    self.traits,
                    self.out_refs,
                    finish.base_head // self.traits.group_heads,
                    finish,
                    row_out_slot(self.traits, finish),
                    packed_q_end,
                    fill_padding_tiles,
                )

    def stage_row_finish(self, finish: RowFinish) -> None:
        """Stage the row's lse and normalized output from its row state at
        finish.parity, once its last softmax has run and its output stage slot
        is free (wait_row_out_slot): its lane partial sums reduce first,
        setting its guard flag."""
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
        """Replay finish's final unit (final_unit) into its row state at
        finish.parity from the unit's checkpoint, or from the row start,
        through QK result slots slots, between the next row's start segment
        and its next segment, which writes no checkpoint. The row's Q is
        staged again into the replay's own Q buffer."""
        traits, replay = self.traits, self.replay
        lo, hi, ctx = self.schedule.row_interval(finish.si)
        start_q_row(traits, replay.q_refs, finish.base_head, finish.q_tok, 0)
        wait_rotated_q_row(traits, replay.q_refs, 0)
        first_tile, num_tiles = self.live_tiles(lo, hi, ctx)
        row = ReplayedRow(
            q_buf=replay.q_refs.buf,
            q_slot=0,
            parity=finish.parity,
            lo=lo,
            first_tile=first_tile,
            ctx=ctx,
            kv_head=finish.base_head // traits.q_heads_per_kv_head,
        )
        self.replay_tiles(row, self.final_unit(num_tiles), num_tiles, slots)

    def replayed_row(self, run: RowRun) -> ReplayedRow:
        """The running row as a replay between two of its segments
        recomputes it, from its Q in the Q stream."""
        return ReplayedRow(
            q_buf=self.q_refs.buf,
            q_slot=run.q_slot,
            parity=run.parity,
            lo=run.span.lo,
            first_tile=run.span.first_tile,
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
        """Recompute row's live tiles from check unit first_unit's first to
        end_tile into its row state with the exact rebasing softmax, between
        two segments: from the row start at unit 0, else from the unit's
        checkpoint. Each kv block is staged again into the replay's buffer
        and every fragment meets each of its tiles (replay_tile). Check unit
        checkpoint_unit's checkpoint is written again as its first tile
        starts, and rebased at the end to the anchor the tiles after it
        raised (rebase_checkpoint), so a checkpoint always holds the row
        state on the row's current anchor."""
        traits, kv_refs = self.traits, self.replay.kv_refs
        tiles_a_block = self.num_block_tiles
        first = first_unit * UNIT_TILES + row.first_tile
        end = end_tile + row.first_tile
        checkpoints = self.row_checkpoints(row.parity)

        @pl.when(first_unit > 0)
        def _restore():
            checkpoints.copy_checkpoint(first_unit, is_restore=True)

        def _replay_block(block, carry):
            kv = row.lo + block
            kv_tok = self.schedule.kv_token(kv, row.ctx)
            start_kv_block(traits, kv_refs, row.kv_head, kv_tok, 0)
            wait_rotated_kv_block(traits, kv_refs, 0)
            block_first = block * tiles_a_block

            def _replay_tile(tile, tile_carry):
                tile_number = block_first + tile - row.first_tile
                if checkpoint_unit is not None:

                    @pl.when(tile_number == checkpoint_unit * UNIT_TILES)
                    def _checkpoint():
                        checkpoints.copy_checkpoint(
                            checkpoint_unit, is_restore=False
                        )
                        checkpoints.save_anchor(first_unit)

                self.replay_tile(row, kv, tile, tile_number == 0, slots)
                return tile_carry

            return lax.fori_loop(
                jnp.maximum(first - block_first, 0),
                jnp.minimum(end - block_first, tiles_a_block),
                _replay_tile,
                carry,
            )

        lax.fori_loop(
            first // tiles_a_block,
            (end - 1) // tiles_a_block + 1,
            _replay_block,
            None,
        )
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
        bounds = tile_bounds(
            self.schedule, row.ctx, self.tile_mask(row.ctx, kv, tile)
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
        tile_and_bounds: tuple[jax.Array, jax.Array],
        is_row_start: jax.Array,
        slots: tuple[jax.Array, ...],
    ) -> None:
        """Fold fragment index (chunk-major) of row against a tile of the
        replay's staged block, (its index in the block, its bounds), into
        the row state with the exact rebasing softmax
        (fwd_mxu.fold_replayed_fragment), both matmuls on QK_MXU through
        slots, the (scores, probs @ V) result slots."""
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
        fold_replayed_fragment(
            self.row_max.at[row.parity, rows],
            self.row_sum.at[row.parity, rows],
            self.o.at[row.parity, rows],
            mask_fragment(
                self.variant, bounds, fragment_row_ids(), chunk, scores
            ),
            is_row_start,
            value_slot,
            lambda: kv_refs.v_buf[0, keys, :],
        )

    def row_checkpoints(self, parity: jax.Array) -> RowCheckpoints:
        """The checkpoints of the check units of the row at parity."""
        return RowCheckpoints(
            checkpoint_sum=self.checkpoint_sum,
            checkpoint_o=self.checkpoint_o,
            row_sum=self.row_sum,
            o=self.o,
            parity=parity,
            row_anchor=lambda rows: self.row_max[parity, rows, :],
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
    """Run the varlen forward on the explicit MXU pipeline over every block
    the per-seq schedule yields; the arguments are varlen_fwd_body's, refs in
    varlen_mxu_scratch_shapes's order.

    A tile of fragments that nothing reads (a half tile where the build has
    them) runs before the first row (run_entry_tile) and after the last
    (finish_last_row), so every row's start segment finishes a tail.
    """
    assert traits.is_per_seq and traits.uses_mxu_pipeline, traits
    assert variant.window == (None, None), variant.window
    # mask_fragment scales the scores by the static Q scale.
    assert not variant.has_kv_scales, variant
    variant.check_operands(operands)
    loop, carry = make_mxu_loop(
        traits, variant, schedule, q_hbm, k_hbm, v_hbm, refs, operands
    )
    carry = run_entry_tile(loop, carry)
    carry = lax.fori_loop(
        0,
        schedule.num_head_groups,
        lambda g, group_carry: lax.fori_loop(
            0,
            schedule.num_rows,
            lambda si, row_carry: loop.run_row(g, si, row_carry),
            group_carry,
        ),
        carry,
    )
    finish_last_row(loop, carry)


def run_entry_tile(
    loop: VarlenMxuLoop, carry: VarlenRowCarry
) -> VarlenRowCarry:
    """The tile before the first row, on ENTRY_PARITY: it leaves the tail
    every row's start segment finishes, with the first row's first K tile
    staged. Returns the carry with its probabilities."""
    pipeline, schedule = loop.pipeline, loop.schedule
    zero = jnp.int32(0)
    first_lo, _, first_ctx = schedule.row_interval(zero)
    entry_parity = jnp.int32(ENTRY_PARITY)
    entry_slot = carry.stream.next_slot
    entry = pipeline.advance(
        pipeline.idle_state(),
        loop.tile_items(0, ROW_TILE, is_half=loop.has_half_tiles),
        loop.ops(
            zero,
            entry_parity,
            carry.ctx,
            (TileRef(entry_slot, zero, loop.dead_mask()),),
        ),
        next_k_tile=lambda: loop.k_tile_at(
            entry_slot, schedule.first_live_tile(first_lo, first_ctx)
        ),
    )
    pipeline.check_loop_tail(
        entry.tail,
        loop.tile_tail(zero, entry_parity, carry.ctx, loop.dead_mask()),
    )
    return carry._replace(probs=entry.probs)


def finish_last_row(loop: VarlenMxuLoop, carry: VarlenRowCarry) -> None:
    """After the last row: a tile on the parity no row holds any more takes
    over the last row's tail and the K tile it staged, the pipeline drains,
    the last row finishes, and every DMA in flight lands."""
    traits, pipeline = loop.traits, loop.pipeline
    stream = carry.stream
    exit_tail = loop.tile_tail(
        carry.q_slot,
        lax.rem(carry.num_rows + 1, NUM_ROW_PARITIES),
        carry.ctx,
        carry.last_mask,
    )
    exit_state = pipeline.advance(
        PipelineState(probs=carry.probs, tail=exit_tail),
        loop.tile_items(0, ROW_TILE, is_half=loop.has_half_tiles),
        loop.ops(
            carry.q_slot,
            lax.rem(carry.num_rows, NUM_ROW_PARITIES),
            carry.ctx,
            (TileRef(stream.slot, jnp.int32(0), loop.dead_mask()),),
        ),
    )
    pipeline.drain(exit_state)
    last_finish = carried_finish(carry.finish)
    wait_row_out_slot(
        traits, loop.out_refs, last_finish, row_out_slot(traits, last_finish)
    )
    loop.stage_row_finish(last_finish)
    loop.complete_row(last_finish, exit_state.tail)

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
    """The explicit loop's context from the kernel's refs, after its prologue
    (lse stage cleared, entry parity zeroed, first two kv blocks started and
    the first one landed), and the loop's first carry."""
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
    loop = VarlenMxuLoop(
        traits=traits,
        variant=variant,
        operands=operands,
        schedule=schedule,
        pipeline=MxuPipeline.allocate(),
        blocks=VarlenBlockStream(
            traits=traits, schedule=schedule, kv_refs=kv_refs
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
    # The tile before the first row runs on ENTRY_PARITY, and the first row
    # finishes it as its previous row: from a zero anchor and zero sums its
    # masked scores add nothing, so it raises no guard flag.
    for ref in (row_max, row_sum):
        ref[ENTRY_PARITY] = jnp.zeros(ref.shape[1:], jnp.float32)
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
    other refs (operands, then varlen_mxu_scratch_shapes's order): the Q and
    kv streams, the output refs, the row state (row max, lane partial sums,
    accumulator, check unit checkpoints of row sums and accumulator, the row's
    and the check unit's guard flags) and the replay's refs."""
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
    """Start the first two kv blocks of the flattened order, wait for the
    first, and return the carry the first row starts from."""
    traits, blocks = loop.traits, loop.blocks
    zero, minus_one = jnp.int32(0), jnp.int32(-1)
    first_desc, second_desc, (_, _, first_ctx) = blocks.first_two_blocks(
        zero, zero
    )
    blocks.start(first_desc, 0)
    wait_rotated_kv_block(traits, loop.kv_refs, 0)
    is_second_fresh = blocks.is_fresh(second_desc, first_desc).astype(jnp.int32)
    second_slot = is_second_fresh

    @pl.when(is_second_fresh > 0)
    def _start_second():
        blocks.start(second_desc, second_slot)

    return VarlenRowCarry(
        q_loaded=minus_one,
        q_inflight=minus_one,
        q_slot=zero,
        stream=KVStream(
            slot=zero,
            next_slot=zero,
            ahead_slot=second_slot,
            is_ahead_fresh=is_second_fresh,
            next_desc=first_desc,
            ahead_desc=second_desc,
        ),
        probs=loop.pipeline.idle_state().probs,
        ctx=first_ctx,
        last_mask=loop.dead_mask(),
        prev_out_tok=zero,
        num_rows=zero,
        # With no row at all, the epilogue finishes the entry tile's parity.
        finish=RowFinish(*([zero] * len(RowFinish._fields)))._replace(
            parity=jnp.int32(ENTRY_PARITY)
        ),
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
    split_mxu_refs unpacks them: Q (all folded heads), K and V stages, Q and
    (under rotary_k) K rotary coefficients, out stage, row max, lane partial
    sums and f32 accumulator per parity, the lane partial sums and
    accumulator of NUM_CHECKPOINTS check units, the replay's Q, K and V (and
    coefficients), the SMEM guard flags of a row and of a check unit, the DMA
    semaphores of the streams and of the replay, and the lse stage."""
    bq, bkv = traits.bq, traits.bkv
    num_stages = traits.num_stages
    head_dim_qk, head_dim_v = traits.head_dim_qk, traits.head_dim_v
    group_bq = traits.group_bq
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
        vmem((NUM_ROW_PARITIES, group_bq, NUM_LANES), jnp.float32),
        vmem((NUM_ROW_PARITIES, group_bq, NUM_LANES), jnp.float32),
        vmem((NUM_ROW_PARITIES, group_bq, head_dim_v), jnp.float32),
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
    """Head-major q, k, v and rotary with their packed token axes padded to
    at least one block of traits, and the schedule of the padded totals:
    an explicit build's blocks may exceed a short call's tokens. Padding
    rows lie past cu_seqlens[-1], so no block reads them as keys and their
    outputs are sliced off."""
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
    """Run an explicit build through run_kernel, fwd_varlen.py's
    run_varlen_kernel with its other arguments bound: with each kv head's q
    heads stacked (fold_q_heads) and the operands padded to whole blocks
    (pad_to_blocks), then slice the outputs back to q's tokens."""
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
