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

"""Paged forward main loop on the explicit MXU pipeline.

On a chip with MXU accumulators a paged build of head_dim 128 or 256 runs
paged_mxu_fwd_body in place of kernels/fwd_paged.py's paged_fwd_body.

The loop runs the schedule's row groups: a row alone, or rows of one live
tile each that share one start segment. Every segment stages the K tile of
the next one, so the pipeline never drains between groups: a group's last
items finish in the next group's start segment, and a drain group after the
grid's last group finishes that one. Groups alternate between
NUM_ROW_PARITIES parities of row states. kv streams block by block through a
ring of block slots: a row alone visits its kv blocks, each one block of the
stream, and a group of one-tile rows takes one block whose tile t is its row
t's tile. A row's start segment and each of its loop bodies run inside one
block. A block's DMAs, one a page, are waited for once, before the segment
whose next K tile is the block's first, and that segment starts every page
DMA of the block lead_blocks on inside its own basic block, predicated on
the wait. A segment's wait is set at the end of the segment before it. The
overflow guard
checks a row by check units of UNIT_TILES tiles, each replayed from its
checkpoint when its row sums flag it; a row's final unit replays as the row
is completed.
"""

import dataclasses
import math
from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from flywheel_tpu.core.tiling import round_up
from flywheel_tpu.core.tpu import MXU_COLUMNS, NUM_SUBLANES
from flywheel_tpu.softmax_attention.config import (
    DEFAULT_MASK_VALUE,
    BlockSizes,
)
from flywheel_tpu.softmax_attention.dma import drain
from flywheel_tpu.softmax_attention.fwd_blocks import (
    SEM_K,
    OutRefs,
    PagedKVRefs,
    PagedRefs,
    QRefs,
    RowFinish,
    copy_row_lse,
    copy_row_out,
    fill_packed_tail,
    o_copy,
    paged_buffer_tile,
    paged_kv_source,
    row_out_slot,
    start_q_row,
    wait_rotated_q_row,
    wait_row_out_slot,
)
from flywheel_tpu.softmax_attention.kv_layout import (
    kv_word_halves,
    load_staged_words,
)
from flywheel_tpu.softmax_attention.mxu.kernels.varlen import (
    pivot_block,
)
from flywheel_tpu.softmax_attention.mxu.pipeline import (
    FRAGMENT_ROWS,
    MxuPipeline,
    PipelineDelays,
    PipelineState,
    Tail,
    WorkItem,
    num_carried_v_pushes,
)
from flywheel_tpu.softmax_attention.mxu.row_state import (
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
    MaskGeometry,
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
    kept_rows_geometry,
    mxu_block_sizes,
    mxu_pipeline_applies,
    pivot_order,
    replay_matmul,
    row_finish_state,
    row_tiles,
    stage_row_finish,
    tile_bounds,
    zeroes_row_start,
)
from flywheel_tpu.softmax_attention.schedule import (
    FwdSchedule,
    MxuRowTiles,
    ScheduleRow,
)
from flywheel_tpu.softmax_attention.traits import PagedFwdTraits
from flywheel_tpu.softmax_attention.variant import Variant, VariantOperands

# A kv block's page DMAs start at least this many tiles before its first
# tile runs, and this many work items: a row of few fragments a tile runs
# through its tiles fast.
MXU_MIN_PREFETCH_TILES = 6
MXU_MIN_PREFETCH_ITEMS = 448
# A kv block's page DMAs start at least this many blocks before its wait.
MXU_MIN_LEAD_BLOCKS = 2
# A DMA start orders the VMEM loads after it, so the item that starts a
# segment's DMAs first loads the q fragments of this many items from it on.
MXU_PRELOADED_FRAGMENTS = 4
# A row's start segment starts its DMAs at this item, ahead of the finish
# staged behind its items; other segments start them mid first tile.
MXU_START_PREFETCH_ITEM = 4


class BlockCursor(NamedTuple):
    """The kv stream's cursor at the next block to issue: visit visit of the
    first row of row group index of head group group, flat grid group flat,
    or of the drain group once is_past."""

    flat: jax.Array
    group: jax.Array
    index: jax.Array
    is_past: jax.Array
    visit: jax.Array


class PagePiece(NamedTuple):
    """One page DMA of a kv block: page page from row page_row on, num_rows
    rows, into its slot from row slot_row on."""

    page: jax.Array
    page_row: jax.Array
    num_rows: jax.Array
    slot_row: jax.Array | int


class BlockDescriptor(NamedTuple):
    """The page DMAs of a kv block: kv head kv_head's word rows, num_rows
    rows in all; pieces of a row alone's block, or where is_short of a
    group of one-tile rows' block, and short_pieces (empty when pieces serve
    both kinds); is_read where a segment reads the block."""

    kv_head: jax.Array
    is_short: jax.Array
    num_rows: jax.Array
    pieces: tuple[PagePiece, ...]
    short_pieces: tuple[PagePiece, ...]
    is_read: jax.Array


# Note (david): the kv block stream lives in SMEM, out of the scalar registers
# of the loop bodies: the stream position of the next block to start, the
# cursor at that block, then per ring slot the rows its block's DMAs move.
STREAM_ISSUE_POSITION = 0
STREAM_CURSOR = 1
STREAM_WAIT_ROWS = STREAM_CURSOR + len(BlockCursor._fields)


class BlockRing(NamedTuple):
    """The kv block ring: num_slots block slots, plus one past them for a
    replay's tile. The segment that waits for a block starts every page DMA
    of the block lead_blocks blocks after it."""

    num_slots: int
    lead_blocks: int

    @property
    def replay_slot(self) -> int:
        return self.num_slots

    @property
    def num_staging_slots(self) -> int:
        return self.num_slots + 1

    @property
    def num_state_words(self) -> int:
        return STREAM_WAIT_ROWS + self.num_slots


class BlockWait(NamedTuple):
    """A segment's wait, before its first item: for the DMAs of num_rows rows
    into ring slot slot where is_due, which also starts the next block's."""

    is_due: jax.Array
    slot: jax.Array
    num_rows: jax.Array


def block_ring(group_rows: int, block_kv: int) -> BlockRing:
    """The kv block ring of rows of group_rows rows over kv blocks of
    block_kv keys. A block is waited for as the segment before its first
    tile starts, and the stream keeps lead_blocks blocks started past it:
    a block's first tile runs a loop body and lead_blocks whole blocks after
    its DMAs start, at least MXU_MIN_PREFETCH_TILES tiles and
    MXU_MIN_PREFETCH_ITEMS items. The slots hold the running block, the one
    waited for, the lead blocks and, while a pending V push may still read
    it, the block before the running one; the lead takes the slots a power
    of two leaves."""
    fragments_per_tile = group_rows // FRAGMENT_ROWS
    tiles = row_tiles(fragments_per_tile)
    block_tiles = block_kv // MXU_COLUMNS
    # Note (david): a tile's V is read at its V push, which may run in the
    # segment after the tile's; such tiles lie in the last segment, so in the
    # block before the running one.
    num_unread_tiles = num_carried_v_pushes(
        PipelineDelays(), fragments_per_tile
    )
    assert num_unread_tiles <= tiles.unit_tiles, (num_unread_tiles, tiles)
    num_unread_blocks = int(num_unread_tiles > 0)
    lead_tiles = max(
        MXU_MIN_PREFETCH_TILES,
        -(-MXU_MIN_PREFETCH_ITEMS // fragments_per_tile),
    )
    min_lead_blocks = max(
        -(-(lead_tiles - tiles.body_tiles) // block_tiles), MXU_MIN_LEAD_BLOCKS
    )
    num_slots = pl.next_power_of_2(2 + num_unread_blocks + min_lead_blocks)
    return BlockRing(
        num_slots=num_slots,
        lead_blocks=num_slots - 2 - num_unread_blocks,
    )


def paged_uses_mxu_pipeline(
    *,
    block_sizes: BlockSizes,
    head_dim: int,
) -> bool:
    """Whether a paged build runs on the explicit MXU pipeline."""
    return mxu_pipeline_applies(
        block_sizes=block_sizes,
        head_dim=head_dim,
        head_dim_v=head_dim,
        transposed_pv=False,
    )


def valid_v_end(ctx: ScheduleRow) -> jax.Array:
    """The end of the row's keys whose V rows its block's page DMAs load:
    staged rows past kv_load_end keep stale VMEM, which may hold NaN."""
    return jnp.minimum(ctx.kv_used_end, ctx.kv_load_end)


def explicit_paged_block_kv(block_kv: int, tiles: MxuRowTiles) -> int:
    """block_kv padded to whole loop bodies of MXU tiles of the default
    pipeline, at least a row's start segment, which runs inside one block."""
    num_tiles = max(-(-block_kv // MXU_COLUMNS), tiles.start_tiles)
    return round_up(num_tiles, tiles.body_tiles) * MXU_COLUMNS


def paged_mxu_block_sizes(block_sizes: BlockSizes, gqa_fold: int) -> BlockSizes:
    """block_sizes as an explicit paged build runs them: block_kv padded to
    whole loop bodies, at least a start segment, which runs inside one
    block; Q and output rings of QUEUED_GROUPS row groups."""
    blocks = mxu_block_sizes(block_sizes)
    fragments_per_tile = gqa_fold * blocks.block_q // FRAGMENT_ROWS
    return dataclasses.replace(
        blocks,
        block_kv=explicit_paged_block_kv(
            blocks.block_kv, row_tiles(fragments_per_tile)
        ),
        num_stages=QUEUED_GROUPS * group_capacity(fragments_per_tile),
    )


def floor_divide(value: jax.Array, divisor: int) -> jax.Array:
    """value // divisor for a static divisor, a shift when it is a power of
    two: a signed division expands into a sign fixup on the scalar unit."""
    if divisor & (divisor - 1) == 0:
        return jnp.right_shift(value, divisor.bit_length() - 1)
    return value // divisor


def floor_remainder(value: jax.Array, divisor: int) -> jax.Array:
    """value % divisor for a static divisor, a mask when it is a power of
    two."""
    if divisor & (divisor - 1) == 0:
        return jnp.bitwise_and(value, divisor - 1)
    return value % divisor


class TileSource(NamedTuple):
    """A staged kv tile of an explicit build: its first row in the staging
    ring, its kv head, and how many of its rows hold valid V (None: all of
    them)."""

    row: jax.Array
    kv_head: jax.Array
    num_valid_rows: jax.Array | None = None


def ring_words(pair_buf: jax.Array) -> jax.Array:
    """A (rows, head_dim) u32 view of every slot of the staging ring
    pair_buf, one word row a staged token."""
    num_slots, num_rows, word_stride, _, head_dim = pair_buf.shape
    return pair_buf.bitcast(jnp.uint32).reshape(
        num_slots * num_rows * word_stride, head_dim
    )


def paged_tile_rows(
    kv_refs: PagedKVRefs,
    source: TileSource,
    kv_part: int,
    scale: jax.Array | None = None,
) -> jax.Array:
    """The (MXU_COLUMNS, head_dim) bf16 K (kv_part 0) or V (kv_part 1) rows
    of a staged tile, times scale when given; V rows past num_valid_rows
    read as zero."""
    # Note (david): an MXU push takes its tile from registers, so the tile is
    # split from the staged words at its push and no plane is stored.
    words = load_staged_words(
        ring_words(kv_refs.pair_buf),
        pl.multiple_of(source.row, MXU_COLUMNS),
        MXU_COLUMNS,
        1,
    )
    if kv_part == 1 and source.num_valid_rows is not None:
        # Note (david): stale rows may hold NaN, and a zero probability times
        # a NaN value is NaN; masked keys need no select.
        row_ids = lax.broadcasted_iota(jnp.int32, words.shape, 0)
        words = jnp.where(
            row_ids < source.num_valid_rows, words, jnp.zeros_like(words)
        )
    rows = kv_word_halves(words, source.kv_head, kv_refs.pair_buf.dtype)
    if scale is None:
        return rows[kv_part].astype(jnp.bfloat16)
    return (rows[kv_part].astype(jnp.float32) * scale).astype(jnp.bfloat16)


def fragment_q_rows(
    traits: PagedFwdTraits, fragment: jax.Array | int
) -> tuple[jax.Array | int, jax.Array | int]:
    """(folded head, first token) of fragment fragment; a group's rows run
    (q tile, folded head, token)."""
    row = fragment * FRAGMENT_ROWS
    tile_row = row % traits.frag_rows
    head = tile_row // traits.bq_compute
    token = row // traits.frag_rows * traits.bq_compute
    return head, aligned_rows(token + tile_row % traits.bq_compute)


def paged_tile_bounds(
    geometry: MaskGeometry, q_tok: jax.Array, key_start: jax.Array
) -> TileBounds:
    """The bounds of the tile whose first key is key_start under a row's
    geometry, its rows counted from the row's first token q_tok."""
    # Note (david): without a left window bound, rows of an earlier sequence
    # in the block see the keys up to their own diagonal, which keeps their
    # anchor and scores on one scale; their output is overwritten.
    return tile_bounds(geometry, key_start).from_row(q_tok)


def chunk_major_fragments(traits: PagedFwdTraits) -> tuple[int, ...]:
    """A tile's fragments in item order: per q compute tile, each token
    chunk's folded heads in a row, so they share their scores' compare."""
    num_chunks = traits.bq_compute // FRAGMENT_ROWS
    return tuple(
        (q_tile * traits.gqa_fold + head) * num_chunks + chunk
        for q_tile in range(traits.num_q_compute_tiles)
        for chunk in range(num_chunks)
        for head in range(traits.gqa_fold)
    )


def row_tile_items(
    fragments: tuple[int, ...], num_tiles: int
) -> tuple[WorkItem, ...]:
    """The items of num_tiles kv tiles, each tile's in fragments' order."""
    return tuple(
        WorkItem(fragment=fragment, tile=tile)
        for tile in range(num_tiles)
        for fragment in fragments
    )


class TileMember(NamedTuple):
    """The row of a segment's tile key: its Q slot and row-state refs."""

    q_slot: jax.Array | int
    row_max: jax.Array
    row_sum: jax.Array
    o: jax.Array


@dataclasses.dataclass(frozen=True, eq=False)
class PagedFragmentOps:
    """The mxu.pipeline.FragmentOps of a segment of explicit paged rows.

    q_buf[q_slot] holds the row's staged Q, unscaled: the K tiles take the
    variant's Q scale. tiles maps a tile key to its staged tile, and bounds,
    one per tile key, masks the scores; None leaves the segment unmasked. A
    segment of several rows names each tile key's row in members. row_starts
    holds, per tile key, whether the tile starts its row; none does when
    empty. unit_end names a check unit's last tile.
    """

    traits: PagedFwdTraits
    variant: Variant
    kv_refs: PagedKVRefs
    operands: VariantOperands
    q_buf: jax.Array
    q_slot: jax.Array
    row_max: jax.Array
    row_sum: jax.Array
    o: jax.Array
    tiles: tuple[TileSource, ...] = ()
    bounds: tuple[TileBounds, ...] | None = None
    unit_end: UnitCheckpoint | None = None
    members: tuple[TileMember, ...] = ()
    row_starts: tuple[bool | jax.Array, ...] = ()
    masks: dict[tuple, jax.Array] = dataclasses.field(default_factory=dict)

    def member(self, tile: int) -> TileMember:
        """The row of tile key tile."""
        if self.members:
            return self.members[tile]
        return TileMember(self.q_slot, self.row_max, self.row_sum, self.o)

    def is_row_start(self, tile: int) -> bool | jax.Array:
        return bool(self.row_starts) and self.row_starts[tile]

    # A V handle: the tile's staging row, kv head and valid rows.
    v_handle_words = 3

    def v_handle(self, tile: int) -> tuple[jax.Array, ...]:
        source = self.tiles[tile]
        num_valid_rows = (
            MXU_COLUMNS
            if source.num_valid_rows is None
            else source.num_valid_rows
        )
        return tuple(
            jnp.asarray(word, jnp.int32)
            for word in (source.row, source.kv_head, num_valid_rows)
        )

    def v_tile_at(self, handle: tuple[jax.Array, ...]) -> jax.Array:
        return paged_tile_rows(self.kv_refs, TileSource(*handle), 1)

    def k_tile(self, tile: int) -> jax.Array:
        return paged_tile_rows(
            self.kv_refs, self.tiles[tile], 0, self.operands.q_scale
        )

    def v_tile(self, tile: int) -> jax.Array:
        return paged_tile_rows(self.kv_refs, self.tiles[tile], 1)

    def q_fragment(self, item: WorkItem) -> jax.Array:
        head, token = fragment_q_rows(self.traits, item.fragment)
        q_slot = self.member(item.tile).q_slot
        if self.traits.gqa_fold > 1:
            q_ref = self.q_buf.at[q_slot, head]
        else:
            q_ref = self.q_buf.at[q_slot]
        return q_ref[pl.ds(token, FRAGMENT_ROWS), :]

    def masked_scores(self, item: WorkItem, scores: jax.Array) -> jax.Array:
        """item's softcapped scores, masked under bounds; the folded heads of
        a static token chunk share one compare."""
        scores = self.variant.transform_logits(scores)
        if self.bounds is None:
            return scores
        _, token = fragment_q_rows(self.traits, item.fragment)
        bounds = self.bounds[item.tile]
        if not isinstance(token, int):
            return jnp.where(
                bounds.visible_keys(token), scores, DEFAULT_MASK_VALUE
            )
        key = (token, item.tile)
        if key not in self.masks:
            self.masks[key] = bounds.visible_keys(token)
        return jnp.where(self.masks[key], scores, DEFAULT_MASK_VALUE)

    def takes_anchor(self, item: WorkItem) -> None:
        """None: every row takes its anchor at its row start."""
        del item

    def rows(self, item: WorkItem) -> pl.Slice:
        """The row-state rows of item's fragment."""
        return fragment_rows(item.fragment)

    def is_unit_end(self, item: WorkItem) -> bool:
        return self.unit_end is not None and item.tile == self.unit_end.tile

    def softmax(self, item: WorkItem, scores: jax.Array) -> jax.Array:
        rows, member = self.rows(item), self.member(item.tile)
        scores = self.masked_scores(item, scores)
        if not self.is_unit_end(item):
            return fold_fragment_scores(
                member.row_max,
                member.row_sum,
                rows,
                scores,
                is_row_start=self.is_row_start(item.tile),
            )
        assert self.is_row_start(item.tile) is False, item
        return fold_unit_end_scores(
            member.row_max, member.row_sum, rows, scores, self.unit_end
        )

    def accumulate(self, item: WorkItem, weighted_values: jax.Array) -> None:
        accumulate_fragment(
            self.member(item.tile).o,
            self.rows(item),
            weighted_values,
            is_row_start=self.is_row_start(item.tile),
            checkpoint_o=self.unit_end.o if self.is_unit_end(item) else None,
        )


@dataclasses.dataclass(frozen=True, eq=False)
class PrefetchingOps:
    """ops whose item preload_keys[0] (fragment, tile) runs step as it
    issues its QK, inside the segment's own basic block. It first loads the
    q fragments of the items at preload_keys: the step's DMA starts order
    the VMEM loads after them."""

    ops: PagedFragmentOps
    preload_keys: tuple[tuple[int, int], ...]
    step: Callable[[], None]
    preloaded: dict[tuple[int, int], jax.Array] = dataclasses.field(
        default_factory=dict
    )

    @property
    def v_handle_words(self) -> int:
        return self.ops.v_handle_words

    def v_handle(self, tile: int) -> tuple[jax.Array, ...]:
        return self.ops.v_handle(tile)

    def v_tile_at(self, handle: tuple[jax.Array, ...]) -> jax.Array:
        return self.ops.v_tile_at(handle)

    def k_tile(self, tile: int) -> jax.Array:
        return self.ops.k_tile(tile)

    def v_tile(self, tile: int) -> jax.Array:
        return self.ops.v_tile(tile)

    def q_fragment(self, item: WorkItem) -> jax.Array:
        key = (item.fragment, item.tile)
        if key in self.preloaded:
            return self.preloaded.pop(key)
        if key not in self.preload_keys[:1]:
            return self.ops.q_fragment(item)
        for fragment, tile in self.preload_keys:
            self.preloaded[(fragment, tile)] = self.ops.q_fragment(
                item._replace(fragment=fragment, tile=tile)
            )
        self.step()
        return self.preloaded.pop(key)

    def softmax(self, item: WorkItem, scores: jax.Array) -> jax.Array:
        return self.ops.softmax(item, scores)

    def accumulate(self, item: WorkItem, weighted_values: jax.Array) -> None:
        self.ops.accumulate(item, weighted_values)


@dataclasses.dataclass(frozen=True, eq=False)
class DiscardedFragmentOps:
    """The FragmentOps of the items that fill the pipeline before the first
    row and empty it after the last: they read zeros and fold nothing."""

    head_dim: int
    v_handle_words = PagedFragmentOps.v_handle_words

    def v_handle(self, tile: int) -> tuple[jax.Array, ...]:
        del tile
        return (jnp.int32(0),) * self.v_handle_words

    def v_tile_at(self, handle: tuple[jax.Array, ...]) -> jax.Array:
        del handle
        return jnp.zeros((MXU_COLUMNS, self.head_dim), jnp.bfloat16)

    def k_tile(self, tile: int) -> jax.Array:
        del tile
        return jnp.zeros((MXU_COLUMNS, self.head_dim), jnp.bfloat16)

    def v_tile(self, tile: int) -> jax.Array:
        del tile
        return jnp.zeros((MXU_COLUMNS, self.head_dim), jnp.bfloat16)

    def q_fragment(self, item: WorkItem) -> jax.Array:
        del item
        return jnp.zeros((FRAGMENT_ROWS, self.head_dim), jnp.bfloat16)

    def softmax(self, item: WorkItem, scores: jax.Array) -> jax.Array:
        del item
        return jnp.zeros(scores.shape, jnp.bfloat16)

    def accumulate(self, item: WorkItem, weighted_values: jax.Array) -> None:
        del item, weighted_values


class FinishedGroup(NamedTuple):
    """A row group whose last tiles are still in flight when the next group
    starts. is_live is 0 for the fill's and the drain's rows; is_short marks
    rows of one live tile each that share the start segment. q_count and
    out_count are its first row's Q ring position and output count, and
    prev_out_tok the first token of the row written before it."""

    is_live: jax.Array
    group: jax.Array
    first_si: jax.Array
    num_rows: jax.Array
    is_short: jax.Array
    parity: jax.Array
    q_count: jax.Array
    out_count: jax.Array
    prev_out_tok: jax.Array
    num_tiles: jax.Array


class RowSpan(NamedTuple):
    """A row's live kv tiles in run order (PagedMxuLoop.tile_place): over
    kv blocks lo..hi from tile first_tile of block lo, num_tiles of them.
    pivot_block counts from lo, and num_pivot_tiles of its tiles run first,
    from pivot_tile."""

    lo: jax.Array
    hi: jax.Array
    first_tile: jax.Array
    num_tiles: jax.Array
    pivot_block: jax.Array
    pivot_tile: jax.Array
    num_pivot_tiles: jax.Array


class GroupRow(NamedTuple):
    """A row of a row group: schedule row si, its first token and its live
    tiles."""

    si: jax.Array
    ctx: ScheduleRow
    q_tok: jax.Array
    span: RowSpan


class SlotRow(NamedTuple):
    """The row of one tile slot of a start segment, or of the tail it
    leaves: its row-state index and Q slot, its first token, what masks its
    scores, the slot tile's first key and the end of the row's keys with
    valid V."""

    state: jax.Array
    q_slot: jax.Array
    q_tok: jax.Array
    kv_frontier: jax.Array
    bottom_right_offset: jax.Array
    kept_start: jax.Array
    kept_end: jax.Array
    key_start: jax.Array
    valid_v_end: jax.Array


class PagedRow(NamedTuple):
    """A row the loop runs: its schedule row, staged Q and row-state index,
    its live tiles, run in rolled bodies up to body_end and in units up to
    stream_end, the stream position of its first block, its and the next
    group's kv heads, and the staging row of the next group's first tile."""

    ctx: ScheduleRow
    q_slot: jax.Array
    q_tok: jax.Array
    state: jax.Array | int
    span: RowSpan
    body_end: jax.Array
    stream_end: jax.Array
    first_block: jax.Array
    kv_head: jax.Array
    next_kv_head: jax.Array
    next_tile_row: jax.Array


class PagedGroupCarry(NamedTuple):
    """The group loop's carry: the Q ring position of the group's first row,
    the stream position of its first block, its head group and group index,
    the pipeline's carried probs, the group before it, the first token of
    the last row written, and the output copies started."""

    q_count: jax.Array
    block: jax.Array
    group: jax.Array
    index: jax.Array
    probs: tuple[jax.Array, ...]
    finished: FinishedGroup
    prev_out_tok: jax.Array
    num_out_copies: jax.Array


@dataclasses.dataclass(frozen=True, eq=False)
class PagedMxuLoop:
    """The steps of paged_mxu_fwd_body's group loop, over the build's refs.

    row_max, row_sum and o hold the row states of each parity's group, and
    checkpoint_sum and checkpoint_o a check unit's entry state. In SMEM,
    stream_state holds the kv block stream (the STREAM_* words), guard a
    row's flag and unit_guard a check unit's.
    """

    traits: PagedFwdTraits
    variant: Variant
    operands: VariantOperands
    schedule: FwdSchedule
    pipeline: MxuPipeline
    q_refs: QRefs
    kv_refs: PagedKVRefs
    out_refs: OutRefs
    row_max: jax.Array
    row_sum: jax.Array
    o: jax.Array
    checkpoint_sum: jax.Array
    checkpoint_o: jax.Array
    stream_state: jax.Array
    guard: jax.Array
    unit_guard: jax.Array

    @property
    def is_causal(self) -> bool:
        return self.schedule.window[1] == 0

    @property
    def has_left_window(self) -> bool:
        """Whether rows start in their pivot tile and mask every tile."""
        return self.schedule.window[0] is not None

    @property
    def num_grid_groups(self) -> jax.Array:
        return self.schedule.num_head_groups * self.schedule.num_groups

    @property
    def num_slots(self) -> int:
        """The tiles of a start segment."""
        return self.tiles.start_tiles

    @property
    def group_rows(self) -> int:
        """The most rows of a group."""
        return group_capacity(self.traits.group_bq // FRAGMENT_ROWS)

    @property
    def num_block_tiles(self) -> int:
        return self.traits.num_k_compute_tiles

    def row_state(
        self, parity: jax.Array | int, member: jax.Array | int
    ) -> jax.Array | int:
        """The row-state index of row member of a group at parity."""
        return parity * self.group_rows + member

    def slot_member(self, slot: int) -> int:
        """The row of slot slot of a group of one-tile rows; a slot past the
        group's capacity hides every key and leaves the last row as it is."""
        return min(slot, self.group_rows - 1)

    def state_refs(
        self, state: jax.Array | int
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """The anchor, lane partial sums and accumulator of row state state."""
        return self.row_max.at[state], self.row_sum.at[state], self.o.at[state]

    @property
    def tiles(self) -> MxuRowTiles:
        """How a row groups its tiles."""
        return row_tiles(self.traits.group_bq // FRAGMENT_ROWS)

    @property
    def ring(self) -> BlockRing:
        return block_ring(self.traits.group_bq, self.traits.bkv)

    def tile_items(self, num_tiles: int = 1) -> tuple[WorkItem, ...]:
        return row_tile_items(chunk_major_fragments(self.traits), num_tiles)

    def start_items(self) -> tuple[WorkItem, ...]:
        """A group's start segment: one tile a slot."""
        return self.tile_items(self.num_slots)

    def next_grid_group(
        self, flat_group: jax.Array, group: jax.Array, index: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """(head group, group index) of the flat grid group after flat_group,
        clamped into the grid, from flat_group's, without a division."""
        is_group_end = index + 1 >= self.schedule.num_groups
        is_last = flat_group + 1 >= self.num_grid_groups
        return (
            jnp.where(jnp.logical_or(is_last, ~is_group_end), group, group + 1),
            jnp.where(is_last, index, jnp.where(is_group_end, 0, index + 1)),
        )

    def grid_group(
        self, index: jax.Array, is_drain: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """(first schedule row, rows) of group index, or of the drain group,
        the grid's last row alone."""
        first, num_rows = self.schedule.row_group(index)
        last = jnp.maximum(self.schedule.num_rows - 1, 0)
        return (
            jnp.where(is_drain, last, first),
            jnp.where(is_drain, 1, num_rows),
        )

    def group_row(self, si: jax.Array) -> GroupRow:
        lo, hi, ctx = self.schedule.row_interval(si)
        return GroupRow(
            si=si,
            ctx=ctx,
            q_tok=self.schedule.q_token(ctx),
            span=self.row_span(lo, hi, ctx),
        )

    def q_slot(self, q_count: jax.Array) -> jax.Array:
        """The Q ring slot of the row at Q ring position q_count."""
        return lax.rem(q_count, self.traits.num_stages)

    @property
    def replay_q_slot(self) -> int:
        """The Q slot a replay stages its row in, past the ring."""
        return self.traits.num_stages

    def kv_head(self, group: jax.Array) -> jax.Array:
        return floor_divide(
            group * self.traits.group_heads, self.schedule.q_heads_per_kv_head
        )

    def first_live_tile(
        self, lo: jax.Array, hi: jax.Array, ctx: ScheduleRow
    ) -> jax.Array:
        """The first tile of block lo the row runs. A row that spans blocks
        starts at a whole loop body that leaves a start segment in the
        block."""
        if not self.has_left_window:
            # Without a left window bound every row starts at key 0.
            return jnp.int32(0)
        tiles = self.tiles
        first_tile = ctx.first_live_tile_lo
        aligned_tile = jnp.minimum(
            first_tile - floor_remainder(first_tile, tiles.body_tiles),
            self.num_block_tiles - tiles.start_tiles,
        )
        return jnp.where(lo < hi, aligned_tile, first_tile)

    def row_span(
        self, lo: jax.Array, hi: jax.Array, ctx: ScheduleRow
    ) -> RowSpan:
        """The live tiles of the row of ctx over kv blocks lo..hi. A pivot
        block that other blocks follow is padded to whole loop bodies, at
        least a start segment."""
        tiles, tiles_a_block = self.tiles, self.num_block_tiles
        first_tile = self.first_live_tile(lo, hi, ctx)
        num_tiles = (
            (hi - lo) * tiles_a_block
            + self.schedule.num_live_tiles(hi, ctx)
            - first_tile
        )
        if not self.has_left_window:
            zero = jnp.int32(0)
            return RowSpan(
                lo, hi, first_tile, num_tiles, zero, first_tile, num_tiles
            )
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
            floor_divide(num_pivot_live + body - 1, body) * body,
            tiles.start_tiles,
        )
        num_padding = jnp.where(
            jnp.logical_and(pivot_kv == hi, lo < hi),
            num_padded - num_pivot_live,
            0,
        )
        return RowSpan(
            lo,
            hi,
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
        run order: the pivot block's tiles (mxu.row_state.pivot_order), the
        blocks below it down to lo, then those above it."""
        tiles_a_block = self.num_block_tiles
        if not self.has_left_window:
            position = span.first_tile + index
            visit = floor_divide(position, tiles_a_block)
            return (
                visit,
                span.lo + visit,
                floor_remainder(position, tiles_a_block),
            )
        pivot = span.pivot_block
        pivot_first = jnp.where(pivot == 0, span.first_tile, 0)
        pivot_tile = pivot_first + pivot_order(
            span.pivot_tile - pivot_first, index
        )
        if self.is_start_tile(index):
            return jnp.int32(0), span.lo + pivot, pivot_tile
        past = index - span.num_pivot_tiles
        num_lower_tiles = jnp.maximum(
            pivot * tiles_a_block - span.first_tile, 0
        )
        lower_block = pivot - 1 - floor_divide(past, tiles_a_block)
        lower_tile = floor_remainder(past, tiles_a_block) + jnp.where(
            lower_block == 0, span.first_tile, 0
        )
        upper_position = (pivot + 1) * tiles_a_block + past - num_lower_tiles
        upper_block = floor_divide(upper_position, tiles_a_block)
        is_pivot = index < span.num_pivot_tiles
        is_lower = past < num_lower_tiles
        block = jnp.where(
            is_pivot, pivot, jnp.where(is_lower, lower_block, upper_block)
        )
        tile = jnp.where(
            is_pivot,
            pivot_tile,
            jnp.where(
                is_lower,
                lower_tile,
                floor_remainder(upper_position, tiles_a_block),
            ),
        )
        visit = jnp.where(
            is_pivot,
            0,
            jnp.where(is_lower, pivot - lower_block, upper_block),
        )
        return visit, span.lo + block, tile

    def is_start_tile(self, index: jax.Array | int) -> bool:
        """Whether index is statically a start segment's tile, which lies in
        the row's pivot block: a pivot block other blocks follow holds a
        start segment."""
        return isinstance(index, int) and index < self.tiles.start_tiles

    def live_tile_place(
        self, span: RowSpan, index: jax.Array | int
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """tile_place of tile index, clamped to the row's last tile: a tile
        past it reads that one, under its own mask. A start segment's tile
        past a one-tile row reads another tile of its block, under that
        tile's own mask."""
        if self.has_left_window and self.is_start_tile(index):
            return self.tile_place(span, index)
        return self.tile_place(span, jnp.minimum(index, span.num_tiles - 1))

    def tile_visit(self, span: RowSpan, index: jax.Array | int) -> jax.Array:
        """The visit of tile index of the row of span in run order, clamped
        to the row's last tile: tile_place's visit, without its tile."""
        if self.is_start_tile(index):
            return jnp.int32(0)
        index = jnp.minimum(index, span.num_tiles - 1)
        tiles_a_block = self.num_block_tiles
        if not self.has_left_window:
            return floor_divide(span.first_tile + index, tiles_a_block)
        pivot = span.pivot_block
        past = index - span.num_pivot_tiles
        num_lower_tiles = jnp.maximum(
            pivot * tiles_a_block - span.first_tile, 0
        )
        return jnp.where(
            past < 0,
            0,
            jnp.where(
                past < num_lower_tiles,
                1 + floor_divide(past, tiles_a_block),
                pivot + 1 + floor_divide(past - num_lower_tiles, tiles_a_block),
            ),
        )

    def visit_block(self, span: RowSpan, visit: jax.Array) -> jax.Array:
        """The kv block the row of span visits at visit visit."""
        if not self.has_left_window:
            return span.lo + visit
        return span.lo + pivot_order(span.pivot_block, visit)

    @staticmethod
    def num_visits(span: RowSpan) -> jax.Array:
        return span.hi - span.lo + 1

    def key_start(self, span: RowSpan, index: jax.Array | int) -> jax.Array:
        """The first key of the tile the row of span runs as its tile index."""
        _, kv, tile = self.tile_place(span, index)
        return kv * self.traits.bkv + tile * MXU_COLUMNS

    def one_tile_key(self, lo: jax.Array, ctx: ScheduleRow) -> jax.Array:
        """The first key of the live tile of the one-tile row of ctx, whose
        kv blocks are lo..lo."""
        return lo * self.traits.bkv + ctx.first_live_tile_lo * MXU_COLUMNS

    def body_end(self, span: RowSpan, ctx: ScheduleRow) -> jax.Array:
        """The end of the row's rolled bodies of body_tiles tiles, whole
        ones from its start segment on: over all of its tiles under a left
        window bound, else over those before its first masked block."""
        tiles = self.tiles
        if self.has_left_window:
            end = span.num_tiles
        else:
            end = jnp.clip(
                jnp.clip(ctx.mask_from - span.lo, 0, span.hi - span.lo + 1)
                * self.num_block_tiles
                - span.first_tile,
                0,
                span.num_tiles,
            )
        num_bodies = jnp.maximum(end - tiles.start_tiles, 0) // tiles.body_tiles
        return tiles.start_tiles + num_bodies * tiles.body_tiles

    def final_unit(self, row: PagedRow) -> jax.Array | int:
        """The first check unit of row's final unit; 0 when none is checked."""
        if not self.has_checked_units:
            return 0
        return first_final_unit(row.body_end // UNIT_TILES)

    @property
    def has_checked_units(self) -> bool:
        """Whether a row of the build's capacity can hold a checked unit:
        none does unless bodies are LOOP_BODY_TILES tiles, start segments
        ROW_START_TILES tiles and rows do not start at zero."""
        tiles = self.tiles
        if (
            tiles.body_tiles != LOOP_BODY_TILES
            or tiles.start_tiles != ROW_START_TILES
            or zeroes_row_start(self.traits.group_bq // FRAGMENT_ROWS)
        ):
            return False
        paged = self.traits.paged
        max_row_tiles = paged.pages_per_seq * paged.page_size // MXU_COLUMNS
        return first_final_unit(max_row_tiles // UNIT_TILES) > 0

    def stream_tiles(self, num_tiles: jax.Array) -> jax.Array:
        """The tiles a row of num_tiles live tiles runs: at least its start
        segment, in whole units."""
        tiles = self.tiles
        num_tiles = jnp.maximum(num_tiles, tiles.start_tiles)
        unit = tiles.unit_tiles
        return floor_divide(num_tiles + unit - 1, unit) * unit

    def block_slot(self, position: jax.Array) -> jax.Array:
        """The ring slot of the block at stream position position."""
        return jnp.bitwise_and(position, self.ring.num_slots - 1)

    def tile_row(self, position: jax.Array, tile: jax.Array | int) -> jax.Array:
        """The staging row of tile tile of the block at stream position
        position."""
        return self.block_slot(position) * self.traits.bkv + tile * MXU_COLUMNS

    @property
    def tile_piece_rows(self) -> int:
        """The rows of one of a kv tile's DMAs, which lie in one page: the
        largest that divides both the tile and the page."""
        return math.gcd(self.traits.paged.page_size, MXU_COLUMNS)

    def tile_pieces(
        self, page_base: jax.Array, key_start: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array], ...]:
        """The (page, first row in it) of each DMA of the tile from key
        key_start of the sequence whose block table row starts at page_base,
        one a piece of tile_piece_rows rows, which lies in one page."""
        # Note (david): a page past the row's pages reads through the clamped
        # table entry; its keys are past seqused_k, so they are masked and
        # their V rows zeroed.
        page_size = self.traits.paged.page_size
        pieces = []
        for piece_key in range(0, MXU_COLUMNS, self.tile_piece_rows):
            key = key_start + piece_key
            page = jnp.clip(
                self.schedule.kv_page(
                    page_base, 0, floor_divide(key, page_size)
                ),
                0,
                self.kv_refs.num_cache_pages - 1,
            )
            pieces.append((page, floor_remainder(key, page_size)))
        return tuple(pieces)

    def start_pieces(
        self,
        kv_head: jax.Array,
        pieces: tuple[tuple[jax.Array, jax.Array], ...],
        slot: jax.Array | int,
        slot_row: jax.Array | int,
    ) -> None:
        """Start the DMAs of kv_head's word rows of a tile's pieces, each
        (page, first row in it), into staging slot slot from row slot_row
        on."""
        # Note (david): every DMA moves tile_piece_rows rows, so its size is
        # static.
        kv_refs, piece_rows = self.kv_refs, self.tile_piece_rows
        for index, (page, page_row) in enumerate(pieces):
            pltpu.make_async_copy(
                paged_kv_source(
                    kv_refs,
                    page,
                    piece_rows,
                    kv_head,
                    pl.multiple_of(page_row, piece_rows),
                ),
                paged_buffer_tile(
                    kv_refs,
                    slot,
                    pl.multiple_of(slot_row + index * piece_rows, piece_rows),
                    piece_rows,
                ),
                kv_refs.sems.at[SEM_K, slot],
            ).start()

    def wait_rows(self, slot: jax.Array | int, num_rows: jax.Array) -> None:
        """Wait for the DMAs into staging slot slot, num_rows rows in all."""
        # Note (david): every DMA into a slot signals its semaphore by its
        # size, and a wait only needs a destination of the awaited size, so
        # one wait covers a block's DMAs.
        window = paged_buffer_tile(
            self.kv_refs, slot, 0, pl.multiple_of(num_rows, NUM_SUBLANES)
        )
        pltpu.make_async_copy(
            window, window, self.kv_refs.sems.at[SEM_K, slot]
        ).wait()

    def slot_members(
        self, first_si: jax.Array, num_rows: jax.Array, is_short: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array, ScheduleRow], ...]:
        """Per row slot of the group of num_rows rows from schedule row
        first_si, (member, first kv block, schedule row) of its row: of a
        group of one-tile rows (is_short) row slot t's, or the last row's
        past its rows; else the group's one row. Tile slot t takes row slot
        min(t, group_rows - 1)."""
        members = []
        for slot in range(self.group_rows):
            member = jnp.where(is_short, jnp.minimum(slot, num_rows - 1), 0)
            lo, _, ctx = self.schedule.row_interval(first_si + member)
            members.append((member, lo, ctx))
        return tuple(members)

    def short_tile_key(
        self,
        members: tuple[tuple[jax.Array, jax.Array, ScheduleRow], ...],
        slot: int,
    ) -> tuple[jax.Array, ScheduleRow, jax.Array]:
        """(member, schedule row, first key) of tile slot slot of a group of
        one-tile rows with slot_members members: its row's tile, or past its
        rows a tile past its last row's, which hides every key."""
        member, lo, ctx = members[min(slot, self.group_rows - 1)]
        key = self.one_tile_key(lo, ctx) + (slot - member) * MXU_COLUMNS
        return member, ctx, key

    @property
    def block_piece_rows(self) -> int:
        """The rows of one of a kv block's page DMAs: the largest that
        divides both the block and the page, so a page the block holds whole
        is one DMA."""
        return math.gcd(self.traits.paged.page_size, self.traits.bkv)

    def page_piece(
        self,
        page_base: jax.Array,
        load_end: jax.Array,
        key: jax.Array,
        piece_rows: int,
        slot_row: int,
    ) -> PagePiece:
        """The page DMA of piece_rows keys from key of the sequence whose
        block table row starts at page_base, which lie in one page, into a
        slot from row slot_row on, cut at load_end but at least one sublane
        tile."""
        # Note (david): a page past the row's pages reads through the clamped
        # table entry; its keys are past seqused_k, so they are masked and
        # their V rows zeroed.
        page_size = self.traits.paged.page_size
        page = jnp.clip(
            self.schedule.kv_page(page_base, 0, floor_divide(key, page_size)),
            0,
            self.kv_refs.num_cache_pages - 1,
        )
        return PagePiece(
            page=page,
            page_row=floor_remainder(key, page_size),
            num_rows=pl.multiple_of(
                jnp.clip(load_end - key, NUM_SUBLANES, piece_rows),
                NUM_SUBLANES,
            ),
            slot_row=slot_row,
        )

    def page_pieces(
        self,
        ctx: ScheduleRow,
        key_start: jax.Array,
        num_keys: int,
        piece_rows: int,
        slot_row: int,
    ) -> tuple[PagePiece, ...]:
        """The page DMAs of keys [key_start, key_start + num_keys) of the
        sequence of ctx into a slot from row slot_row on, one a piece of
        piece_rows rows."""
        return tuple(
            self.page_piece(
                ctx.page_base,
                ctx.kv_load_end,
                key_start + offset,
                piece_rows,
                slot_row + offset,
            )
            for offset in range(0, num_keys, piece_rows)
        )

    def unified_pieces(
        self,
        row_ctx: ScheduleRow,
        block_key: jax.Array,
        members: tuple[tuple[jax.Array, jax.Array, ScheduleRow], ...],
        is_short: jax.Array,
    ) -> tuple[PagePiece, ...]:
        """The page DMAs of a block whose pieces have the same rows for a row
        alone, keys from block_key of the row of row_ctx, and for a group of
        one-tile rows (is_short) with slot_members members: piece rows r of
        the slot are row r's of the row's block, or of tile r // MXU_COLUMNS
        of the group's."""
        piece_rows = self.block_piece_rows
        pieces = []
        for offset in range(0, self.traits.bkv, piece_rows):
            _, ctx, short_key = self.short_tile_key(
                members, offset // MXU_COLUMNS
            )
            pieces.append(
                self.page_piece(
                    jnp.where(is_short, ctx.page_base, row_ctx.page_base),
                    jnp.where(is_short, ctx.kv_load_end, row_ctx.kv_load_end),
                    jnp.where(
                        is_short,
                        short_key + offset % MXU_COLUMNS,
                        block_key + offset,
                    ),
                    piece_rows,
                    offset,
                )
            )
        return tuple(pieces)

    def start_page_dmas(
        self,
        kv_head: jax.Array,
        pieces: tuple[PagePiece, ...],
        slot: jax.Array,
        piece_rows: int,
        is_started: jax.Array | bool,
    ) -> None:
        """Start the DMAs of kv_head's word rows of pieces into staging slot
        slot where is_started."""
        # Note (david): a DMA of a runtime size starts only when the size is
        # not zero, inside the segment's basic block; a predicated region
        # around the starts keeps the segment's vector work off their bundles.
        kv_refs = self.kv_refs
        for piece in pieces:
            num_rows = pl.multiple_of(
                jnp.where(is_started, piece.num_rows, 0), NUM_SUBLANES
            )
            pltpu.make_async_copy(
                paged_kv_source(
                    kv_refs,
                    piece.page,
                    num_rows,
                    kv_head,
                    pl.multiple_of(piece.page_row, piece_rows),
                ),
                paged_buffer_tile(kv_refs, slot, piece.slot_row, num_rows),
                kv_refs.sems.at[SEM_K, slot],
            ).start()

    def load_cursor(self) -> BlockCursor:
        return BlockCursor(*(
            self.stream_state[STREAM_CURSOR + index]
            for index in range(len(BlockCursor._fields))
        ))

    def store_cursor(self, cursor: BlockCursor) -> None:
        for index, value in enumerate(cursor):
            self.stream_state[STREAM_CURSOR + index] = jnp.int32(value)

    def step_cursor(
        self, cursor: BlockCursor, num_visits: jax.Array
    ) -> BlockCursor:
        """cursor one block on, from a group of num_visits blocks: its next
        visit, or the next grid group's first block."""
        group, index = self.next_grid_group(
            cursor.flat, cursor.group, cursor.index
        )
        next_group = BlockCursor(
            flat=cursor.flat + 1,
            group=group,
            index=index,
            is_past=jnp.logical_or(
                cursor.is_past > 0, cursor.flat + 1 >= self.num_grid_groups
            ).astype(jnp.int32),
            visit=jnp.int32(0),
        )
        return jax.tree.map(
            partial(jnp.where, cursor.visit + 1 >= num_visits),
            next_group,
            cursor._replace(visit=cursor.visit + 1),
        )

    def cursor_block(self) -> tuple[BlockDescriptor, BlockCursor]:
        """The page DMAs of the cursor's block, and the cursor one block on.
        The block is a visit of a row alone, its keys where its kv block
        holds them, or the block of a group of one-tile rows whose tile t is
        its row t's tile; the drain group's is its row's first visit."""
        cursor = self.load_cursor()
        is_past = cursor.is_past > 0
        first_si, num_rows = self.grid_group(cursor.index, is_past)
        first = self.group_row(first_si)
        is_short = first.span.num_tiles == 1
        kv_head = self.kv_head(
            jnp.where(is_past, self.schedule.num_head_groups - 1, cursor.group)
        )
        bkv, block_rows = self.traits.bkv, self.block_piece_rows
        block_key = self.visit_block(first.span, cursor.visit) * bkv
        members = self.slot_members(first_si, num_rows, is_short)
        if block_rows == self.tile_piece_rows:
            # One set of DMAs serves both kinds of block: a DMA start orders
            # the segment's VMEM loads around it, so each costs the segment.
            pieces = self.unified_pieces(
                first.ctx, block_key, members, is_short
            )
            short_pieces = ()
            num_moved_rows = sum(piece.num_rows for piece in pieces)
        else:
            pieces = self.page_pieces(first.ctx, block_key, bkv, block_rows, 0)
            short_pieces = ()
            for tile in range(self.num_slots):
                _, ctx, key = self.short_tile_key(members, tile)
                short_pieces += self.page_pieces(
                    ctx,
                    key,
                    MXU_COLUMNS,
                    self.tile_piece_rows,
                    tile * MXU_COLUMNS,
                )
            num_moved_rows = jnp.where(
                is_short,
                sum(piece.num_rows for piece in short_pieces),
                sum(piece.num_rows for piece in pieces),
            )
        descriptor = BlockDescriptor(
            kv_head=kv_head,
            is_short=is_short.astype(jnp.int32),
            num_rows=num_moved_rows,
            pieces=pieces,
            short_pieces=short_pieces,
            # The drain group waits for the block after its own, the last
            # one any segment reads.
            is_read=cursor.flat <= self.num_grid_groups + 1,
        )
        next_cursor = self.step_cursor(
            cursor,
            jnp.where(
                jnp.logical_or(is_short, is_past),
                1,
                self.num_visits(first.span),
            ),
        )
        return descriptor, next_cursor

    def start_block(self, is_due: jax.Array | bool) -> None:
        """Where is_due and a segment reads it, start every page DMA of the
        cursor's block, the one at the stream's issue position, record the
        rows they move for its wait and step the cursor one block on."""
        state = self.stream_state
        position = state[STREAM_ISSUE_POSITION]
        slot = self.block_slot(position)
        descriptor, next_cursor = self.cursor_block()
        is_due = jnp.logical_and(is_due, descriptor.is_read)
        if descriptor.short_pieces:
            is_short = descriptor.is_short > 0
            self.start_page_dmas(
                descriptor.kv_head,
                descriptor.pieces,
                slot,
                self.block_piece_rows,
                jnp.logical_and(is_due, jnp.logical_not(is_short)),
            )
            self.start_page_dmas(
                descriptor.kv_head,
                descriptor.short_pieces,
                slot,
                self.tile_piece_rows,
                jnp.logical_and(is_due, is_short),
            )
        else:
            self.start_page_dmas(
                descriptor.kv_head,
                descriptor.pieces,
                slot,
                self.block_piece_rows,
                is_due,
            )

        @pl.when(is_due)
        def _advance():
            state[STREAM_WAIT_ROWS + slot] = descriptor.num_rows
            state[STREAM_ISSUE_POSITION] = position + 1
            self.store_cursor(next_cursor)

    def wait_block(self, position: jax.Array | int) -> None:
        """Wait for the page DMAs of the block at stream position position."""
        slot = self.block_slot(position)
        self.wait_rows(slot, self.stream_state[STREAM_WAIT_ROWS + slot])

    def block_wait(
        self,
        row: PagedRow,
        tile_number: jax.Array | int,
        next_tile_number: jax.Array | int,
    ) -> BlockWait:
        """The wait of a segment from row's tile tile_number whose next K
        tile is row's tile next_tile_number, or the next group's first past
        the tiles row runs: due where that tile opens a block, for the block
        after the segment's."""
        span = row.span
        visit = self.tile_visit(span, tile_number)
        if self.has_left_window:
            opens_block = self.tile_visit(span, next_tile_number) != visit
        else:
            # Without a left window bound a row's live tiles run in key order,
            # so the next tile opens a block where it is one's first.
            opens_block = jnp.logical_and(
                next_tile_number < span.num_tiles,
                floor_remainder(next_tile_number, self.num_block_tiles) == 0,
            )
        slot = self.block_slot(row.first_block + visit + 1)
        return BlockWait(
            is_due=jnp.logical_or(
                next_tile_number >= row.stream_end, opens_block
            ),
            slot=slot,
            num_rows=self.stream_state[STREAM_WAIT_ROWS + slot],
        )

    def wait_due_block(self, wait: BlockWait) -> None:
        """Run wait's wait where it is due."""

        @pl.when(wait.is_due)
        def _wait():
            self.wait_rows(wait.slot, wait.num_rows)

    def with_prefetch(
        self,
        ops: PagedFragmentOps,
        items: tuple[WorkItem, ...],
        is_due: jax.Array | bool,
        prefetch_item: int | None = None,
    ) -> PrefetchingOps:
        """ops whose segment of items, where is_due, starts the next block's
        DMAs at its first tile's item prefetch_item (its middle one when
        None), inside the segment: a DMA start orders the segment's memory
        accesses around it."""
        first_tile = [item for item in items if item.tile == items[0].tile]
        if prefetch_item is None:
            first = len(first_tile) // 2
        else:
            first = prefetch_item
        preloaded = first_tile[first : first + MXU_PRELOADED_FRAGMENTS]
        return PrefetchingOps(
            ops=ops,
            preload_keys=tuple(
                (item.fragment, item.tile) for item in preloaded
            ),
            step=partial(self.start_block, is_due),
        )

    def first_tile_row(self, row: GroupRow, position: jax.Array) -> jax.Array:
        """The staging row of the first tile of the group whose first row is
        row, from its block at stream position position."""
        if not self.has_left_window:
            # Without a left window bound every row starts at tile 0 of its
            # first block.
            return self.tile_row(position, 0)
        _, _, tile = self.tile_place(row.span, 0)
        return self.tile_row(
            position, jnp.where(row.span.num_tiles == 1, 0, tile)
        )

    def row_tile(
        self, row: PagedRow, tile_number: jax.Array | int, *, is_masked: bool
    ) -> TileSource:
        """The source of the tile row runs as its tile tile_number;
        is_masked zeroes its V rows past the row's valid keys."""
        if is_masked:
            num_valid_rows = valid_v_end(row.ctx) - self.key_start(
                row.span, tile_number
            )
        else:
            num_valid_rows = None
        visit, _, tile = self.live_tile_place(row.span, tile_number)
        return TileSource(
            row=self.tile_row(row.first_block + visit, tile),
            kv_head=row.kv_head,
            num_valid_rows=num_valid_rows,
        )

    def next_tile(
        self, row: PagedRow, tile_number: jax.Array | int
    ) -> TileSource:
        """The source of row's tile tile_number, or of the next group's first
        tile past the tiles row runs."""
        is_row_end = tile_number >= row.stream_end
        visit, _, tile = self.live_tile_place(row.span, tile_number)
        return TileSource(
            row=jnp.where(
                is_row_end,
                row.next_tile_row,
                self.tile_row(row.first_block + visit, tile),
            ),
            kv_head=jnp.where(is_row_end, row.next_kv_head, row.kv_head),
        )

    def staged_k(self, source: TileSource) -> Callable[[], jax.Array]:
        """next_k_tile for the tile at source."""
        return lambda: paged_tile_rows(
            self.kv_refs, source, 0, self.operands.q_scale
        )

    def geometry(self, row: ScheduleRow | SlotRow) -> MaskGeometry:
        """The mask of a row from its schedule row, or from the SlotRow of
        one of its tile slots; a paged sequence's keys start at 0."""
        left, right = self.schedule.window
        geometry = MaskGeometry(
            shift=row.bottom_right_offset,
            left=left,
            right=right,
            key_end=row.kv_frontier,
        )
        return kept_rows_geometry(geometry, row.kept_start, row.kept_end)

    def ops(
        self,
        q_slot: jax.Array | int,
        state: jax.Array | int,
        tiles: tuple[TileSource, ...] = (),
        bounds: tuple[TileBounds, ...] | None = None,
    ) -> PagedFragmentOps:
        row_max, row_sum, o = self.state_refs(state)
        return PagedFragmentOps(
            traits=self.traits,
            variant=self.variant,
            kv_refs=self.kv_refs,
            operands=self.operands,
            q_buf=self.q_refs.buf,
            q_slot=q_slot,
            row_max=row_max,
            row_sum=row_sum,
            o=o,
            tiles=tiles,
            bounds=bounds,
        )

    def row_ops(
        self, row: PagedRow, tiles: tuple[TileSource, ...] = ()
    ) -> PagedFragmentOps:
        """The unmasked ops of row's items of tiles."""
        return self.ops(row.q_slot, row.state, tiles)

    def unit_end_ops(
        self, ops: PagedFragmentOps, slot: jax.Array, largest: LargestSums
    ) -> PagedFragmentOps:
        """ops whose items of a body's last tile key, a check unit's last
        tile, also write the row state they leave into checkpoint slot slot
        and fold their row sums into largest."""
        unit_end = UnitCheckpoint(
            tile=LOOP_BODY_TILES - 1,
            row_sum=self.checkpoint_sum.at[slot],
            o=self.checkpoint_o.at[slot],
            largest=largest,
        )
        return dataclasses.replace(ops, unit_end=unit_end)

    def masked_ops(
        self,
        row: PagedRow,
        tiles: tuple[TileSource, ...],
        tile_numbers: tuple[jax.Array | int, ...],
    ) -> PagedFragmentOps:
        """The ops of row's tiles tile_numbers (one per tile key), every score
        masked; tiles is empty for a tail's ops."""
        geometry = self.geometry(row.ctx)
        bounds = tuple(
            paged_tile_bounds(
                geometry, row.q_tok, self.key_start(row.span, tile_number)
            )
            for tile_number in tile_numbers
        )
        return self.ops(row.q_slot, row.state, tiles, bounds)

    def slot_rows(
        self,
        first_si: jax.Array,
        num_rows: jax.Array,
        is_short: jax.Array,
        parity: jax.Array | int,
        q_count: jax.Array,
        first_long_tile: jax.Array | int,
        span: RowSpan,
    ) -> tuple[SlotRow, ...]:
        """The rows of the tile slots of a group's start segment, or of the
        tail its last tiles leave. Slot t of a group of one-tile rows
        (is_short) is its row t's tile, and a slot past its rows a tile past
        its last row's, which hides every key. Every slot of a group of one
        row is the tile that row runs as its tile first_long_tile + t; span
        is the group's first row's live tiles."""
        members = self.slot_members(first_si, num_rows, is_short)
        slots = []
        for slot in range(self.num_slots):
            member, ctx, short_key = self.short_tile_key(members, slot)
            slots.append(
                SlotRow(
                    state=self.row_state(
                        parity, jnp.where(is_short, self.slot_member(slot), 0)
                    ),
                    q_slot=self.q_slot(q_count + member),
                    q_tok=self.schedule.q_token(ctx),
                    kv_frontier=ctx.kv_frontier,
                    bottom_right_offset=ctx.bottom_right_offset,
                    kept_start=ctx.kept_start,
                    kept_end=ctx.kept_end,
                    key_start=jnp.where(
                        is_short,
                        short_key,
                        self.key_start(span, first_long_tile + slot),
                    ),
                    valid_v_end=valid_v_end(ctx),
                )
            )
        return tuple(slots)

    def slot_ops(
        self,
        slots: tuple[SlotRow, ...],
        tiles: tuple[TileSource, ...],
        *,
        states: tuple[jax.Array | int, ...],
        row_starts: tuple[bool | jax.Array, ...],
    ) -> PagedFragmentOps:
        """The masked ops of a segment of one tile a slot, slot t's tile key
        t under its own row's mask on row state states[t], a row start where
        row_starts[t]; tiles is empty for a tail's ops."""
        first = slots[0]
        ops = self.ops(
            first.q_slot,
            states[0],
            tiles,
            tuple(
                paged_tile_bounds(
                    self.geometry(slot), slot.q_tok, slot.key_start
                )
                for slot in slots
            ),
        )
        members = tuple(
            TileMember(slot.q_slot, *self.state_refs(state))
            for slot, state in zip(slots, states)
        )
        return dataclasses.replace(ops, members=members, row_starts=row_starts)

    def finished_tail(
        self, finished: FinishedGroup, parity: jax.Array | int
    ) -> Tail:
        """The tail finished's last start segment's worth of tiles leaves
        for the next group's first segment, its row states at parity."""
        slots = self.slot_rows(
            finished.first_si,
            finished.num_rows,
            finished.is_short > 0,
            parity,
            finished.q_count,
            finished.num_tiles - self.num_slots,
            self.group_row(finished.first_si).span,
        )
        ops = self.slot_ops(
            slots,
            (),
            states=tuple(slot.state for slot in slots),
            row_starts=tuple(
                jnp.logical_and(finished.is_short > 0, slot < self.group_rows)
                for slot in range(self.num_slots)
            ),
        )
        return self.pipeline.tail_of(
            self.start_items(), ops, has_staged_next_k=True
        )

    def member_finish(
        self,
        finished: FinishedGroup,
        member: jax.Array | int,
        row: GroupRow,
        prev_out_tok: jax.Array,
        parity: jax.Array | int,
    ) -> RowFinish:
        """The output finish of finished's row member, row, whose row state
        is at parity; the row written before it starts at prev_out_tok."""
        return RowFinish(
            is_live=finished.is_live,
            si=row.si,
            q_tok=row.q_tok,
            base_head=finished.group * self.traits.group_heads,
            num_blend_rows=self.schedule.blend_start(row.ctx) - row.q_tok,
            prev_out_tok=prev_out_tok,
            out_count=finished.out_count + member,
            parity=self.row_state(parity, member),
        )

    def first_finish(
        self, finished: FinishedGroup, parity: jax.Array | int
    ) -> RowFinish:
        """The finish of finished's first row, whose row state is at parity."""
        return self.member_finish(
            finished,
            0,
            self.group_row(finished.first_si),
            finished.prev_out_tok,
            parity,
        )

    def stage_row_finish(self, finish: RowFinish) -> None:
        """Reduce the lane partial sums of finish's row, which sets its guard
        flag, then stage its lse and output; its output stage slot is free."""
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

    def start_segment(
        self,
        probs: tuple[jax.Array, ...],
        finished: FinishedGroup,
        slots: tuple[SlotRow, ...],
        tiles: tuple[TileSource, ...],
        next_k_tile: Callable[[], jax.Array],
        is_due: jax.Array,
        *,
        parity: int,
        is_short: bool,
    ) -> tuple[jax.Array, ...]:
        """A group's start segment at a static parity, which starts the next
        block's DMAs where is_due: each slot of a group of one-tile rows
        (is_short) starts a row on its own row state, else only the first.
        The finish of finished's first row is staged behind it: static
        parities keep the two groups' state provably apart."""
        num_slots = self.num_slots
        prev_parity = NUM_ROW_PARITIES - 1 - parity
        if is_short:
            states = tuple(
                self.row_state(parity, self.slot_member(slot))
                for slot in range(num_slots)
            )
            row_starts = tuple(
                slot < self.group_rows for slot in range(num_slots)
            )
        else:
            states = (self.row_state(parity, 0),) * num_slots
            row_starts = (True,) + (False,) * (num_slots - 1)
            if zeroes_row_start(self.traits.group_bq // FRAGMENT_ROWS):
                self.o[states[0]] = jnp.zeros(self.o.shape[1:], self.o.dtype)
        items = self.start_items()
        tail = self.finished_tail(finished, prev_parity)
        state = self.pipeline.advance(
            PipelineState(probs=probs, tail=tail),
            items,
            self.with_prefetch(
                self.slot_ops(
                    slots, tiles, states=states, row_starts=row_starts
                ),
                items,
                is_due,
                prefetch_item=MXU_START_PREFETCH_ITEM,
            ),
            next_k_tile=next_k_tile,
        )
        # The next segment is a loop body or the next group's start segment.
        for next_items in (items, self.tile_items(self.tiles.body_tiles)):
            self.pipeline.check_loop_tail(
                state.tail,
                self.pipeline.tail_of(
                    next_items, state.tail.ops, has_staged_next_k=True
                ),
            )
        self.stage_row_finish(self.first_finish(finished, prev_parity))
        return state.probs

    def start_group(
        self,
        probs: tuple[jax.Array, ...],
        finished: FinishedGroup,
        slots: tuple[SlotRow, ...],
        is_short: jax.Array,
        row: PagedRow,
    ) -> tuple[jax.Array, ...]:
        """Run the start segment of the group whose first row is row, one
        copy per parity and kind of group; stages finished's first row. Its
        tiles lie in the group's first block: a row's first ones, or tile t
        of a group of one-tile rows."""
        tiles = []
        for tile, slot in enumerate(slots):
            visit, _, long_tile = self.live_tile_place(row.span, tile)
            tiles.append(
                TileSource(
                    row=self.tile_row(
                        row.first_block + visit,
                        jnp.where(is_short, tile, long_tile),
                    ),
                    kv_head=row.kv_head,
                    num_valid_rows=slot.valid_v_end - slot.key_start,
                )
            )
        first = self.first_finish(finished, finished.parity)
        wait_row_out_slot(
            self.traits, self.out_refs, first, row_out_slot(self.traits, first)
        )
        wait = self.block_wait(row, 0, self.num_slots)
        self.wait_due_block(wait)
        segment = partial(
            self.start_segment,
            finished=finished,
            slots=slots,
            tiles=tuple(tiles),
            next_k_tile=self.staged_k(self.next_tile(row, self.num_slots)),
            is_due=wait.is_due,
        )

        def _at_parity(parity, probs):
            return lax.cond(
                is_short,
                partial(segment, parity=parity, is_short=True),
                partial(segment, parity=parity, is_short=False),
                probs,
            )

        return lax.cond(
            finished.parity == ENTRY_PARITY,
            partial(_at_parity, NUM_ROW_PARITIES - 1 - ENTRY_PARITY),
            partial(_at_parity, ENTRY_PARITY),
            probs,
        )

    def complete_group(self, finished: FinishedGroup) -> None:
        """Complete finished's rows once the next group's start segment ran:
        its first row, staged behind that segment, then each later row."""
        traits = self.traits
        self.complete_row(
            finished,
            self.group_row(finished.first_si),
            self.first_finish(finished, finished.parity),
        )

        def _complete_member(member):
            row = self.group_row(finished.first_si + member)
            prev_row = self.group_row(finished.first_si + member - 1)
            finish = self.member_finish(
                finished, member, row, prev_row.q_tok, finished.parity
            )
            out_slot = row_out_slot(traits, finish)
            wait_row_out_slot(traits, self.out_refs, finish, out_slot)
            self.stage_row_finish(finish)
            self.complete_row(finished, row, finish)

        for_each_later_row(finished.num_rows, _complete_member)

    def complete_row(
        self, finished: FinishedGroup, row: GroupRow, finish: RowFinish
    ) -> None:
        """Write out finished's row row, staged as finish; a row its row
        sums flag first replays its final unit and is staged again."""
        traits = self.traits

        @pl.when(finish.is_live > 0)
        def _complete():
            @pl.when(self.guard[0] > 0)
            def _replay():
                self.replay_row_end(finished.group, row, finish)
                self.stage_row_finish(finish)

            out_slot = row_out_slot(traits, finish)
            if self.variant.return_lse:
                copy_row_lse(
                    traits,
                    self.out_refs,
                    finished.group,
                    finish,
                    out_slot,
                    self.schedule.packed_q_end,
                )
            copy_row_out(
                traits,
                self.out_refs,
                finish,
                out_slot,
                self.schedule.packed_q_end,
            )

    def replay_row_end(
        self, group: jax.Array, row: GroupRow, finish: RowFinish
    ) -> None:
        """Replay row's final unit into its row state, between the next
        group's start segment and its next segment, which writes no
        checkpoint; the row's Q is staged again into the replay's Q slot."""
        traits, q_slot = self.traits, self.replay_q_slot
        start_q_row(traits, self.q_refs, finish.base_head, finish.q_tok, q_slot)
        wait_rotated_q_row(traits, self.q_refs, q_slot)
        replayed = PagedRow(
            ctx=row.ctx,
            q_slot=jnp.int32(q_slot),
            q_tok=row.q_tok,
            state=finish.parity,
            span=row.span,
            body_end=self.body_end(row.span, row.ctx),
            stream_end=self.stream_tiles(row.span.num_tiles),
            first_block=jnp.int32(0),
            kv_head=self.kv_head(group),
            next_kv_head=self.kv_head(group),
            next_tile_row=jnp.int32(0),
        )
        start_tail = self.pipeline.tail_of(
            self.start_items(),
            DiscardedFragmentOps(traits.head_dim_qk),
            has_staged_next_k=True,
        )
        self.replay_tiles(
            replayed,
            self.final_unit(replayed),
            jnp.maximum(row.span.num_tiles, ROW_START_TILES),
            self.pipeline.free_slots(start_tail, REPLAY_RESULT_SLOTS),
        )

    def stage_replay_tile(
        self, row: PagedRow, tile_number: jax.Array
    ) -> TileSource:
        """The source of a replayed row's tile tile_number, staged
        synchronously into the ring's replay slot."""
        replay_slot = self.ring.replay_slot
        pieces = self.tile_pieces(
            row.ctx.page_base, self.key_start(row.span, tile_number)
        )
        self.start_pieces(row.kv_head, pieces, replay_slot, 0)
        self.wait_rows(replay_slot, jnp.int32(MXU_COLUMNS))
        return self.row_tile(row, tile_number, is_masked=True)._replace(
            row=jnp.int32(replay_slot * self.traits.bkv)
        )

    def replay_tiles(
        self,
        row: PagedRow,
        first_unit: jax.Array | int,
        end_tile: jax.Array,
        slots: tuple[jax.Array, ...],
        checkpoint_unit: jax.Array | None = None,
    ) -> None:
        """Recompute row's tiles from check unit first_unit's first to
        end_tile into its row state with the exact rebasing softmax, between
        two segments: from the row start at unit 0, else from the unit's
        checkpoint. checkpoint_unit's checkpoint is written again as its
        first tile starts and rebased at the end to the row's anchor."""
        num_fragments = self.traits.group_bq // FRAGMENT_ROWS
        checkpoints = self.row_checkpoints(row.state)

        @pl.when(first_unit > 0)
        def _restore():
            checkpoints.copy_checkpoint(first_unit, is_restore=True)

        def _replay_tile(tile_number, carry):
            if checkpoint_unit is not None:

                @pl.when(tile_number == checkpoint_unit * UNIT_TILES)
                def _checkpoint():
                    checkpoints.copy_checkpoint(
                        checkpoint_unit, is_restore=False
                    )
                    checkpoints.save_anchor(first_unit)

            ops = self.masked_ops(
                row, (self.stage_replay_tile(row, tile_number),), (tile_number,)
            )

            def _replay_fragment(fragment, fragment_carry):
                self.replay_fragment(ops, fragment, tile_number == 0, slots)
                return fragment_carry

            return lax.fori_loop(0, num_fragments, _replay_fragment, carry)

        lax.fori_loop(first_unit * UNIT_TILES, end_tile, _replay_tile, None)
        if checkpoint_unit is not None:
            checkpoints.rebase_checkpoint(checkpoint_unit, first_unit)

    def row_checkpoints(self, state: jax.Array | int) -> RowCheckpoints:
        """The checkpoints of the row at row state state."""
        return RowCheckpoints(
            checkpoint_sum=self.checkpoint_sum,
            checkpoint_o=self.checkpoint_o,
            row_sum=self.row_sum,
            o=self.o,
            parity=state,
            row_anchor=lambda rows: self.row_max[state, rows, :],
        )

    def replay_fragment(
        self,
        ops: PagedFragmentOps,
        fragment: jax.Array,
        is_row_start: jax.Array,
        slots: tuple[jax.Array, ...],
    ) -> None:
        """Fold fragment fragment against the one tile of ops into the row
        state with the exact rebasing softmax, both matmuls on QK_MXU
        through slots, the (scores, probs @ V) result slots."""
        score_slot, value_slot = slots
        item = WorkItem(fragment=fragment, tile=0)
        scores = replay_matmul(
            score_slot, ops.q_fragment(item), ops.k_tile(0), transpose=True
        )
        rows = fragment_rows(fragment)
        fold_replayed_fragment(
            ops.row_max.at[rows],
            ops.row_sum.at[rows],
            ops.o.at[rows],
            ops.masked_scores(item, scores),
            is_row_start,
            value_slot,
            partial(ops.v_tile, 0),
        )

    def row_bodies(
        self, probs: tuple[jax.Array, ...], row: PagedRow
    ) -> tuple[jax.Array, ...]:
        """Run row's rolled bodies of body_tiles tiles up to body_end: its
        check units, then plain bodies; masked under a left window bound."""
        is_masked = self.has_left_window
        first_body = self.tiles.start_tiles
        body_tiles = self.tiles.body_tiles
        # Every later body's wait is set at the end of the body before it.
        wait = self.block_wait(row, first_body, first_body + body_tiles)
        if self.has_checked_units:
            final_unit = self.final_unit(row)
            num_units = jnp.where(final_unit > 0, final_unit + 1, 0)
            self.unit_guard[0] = jnp.int32(0)
            probs, wait = lax.fori_loop(
                0,
                num_units,
                partial(
                    self.checked_unit,
                    row=row,
                    final_unit=final_unit,
                    is_masked=is_masked,
                ),
                (probs, wait),
            )
            first_body = num_units * UNIT_TILES + ROW_START_TILES
        return self.plain_bodies(
            probs, row, first_body, row.body_end, wait, is_masked=is_masked
        )

    def plain_bodies(
        self,
        probs: tuple[jax.Array, ...],
        row: PagedRow,
        first_tile: jax.Array | int,
        end_tile: jax.Array | int,
        wait: BlockWait,
        *,
        is_masked: bool,
    ) -> tuple[jax.Array, ...]:
        """Run row's tiles [first_tile, end_tile) in rolled bodies of
        body_tiles tiles, masked when is_masked; wait is the first body's."""
        body_tiles = self.tiles.body_tiles
        num_bodies = jnp.maximum(end_tile - first_tile, 0) // body_tiles

        def _body(index, carry):
            probs, wait = carry
            first = first_tile + index * body_tiles
            ops_of, tail_ops = self.body_ops(row, first, is_masked=is_masked)
            state, wait = self.body(
                probs, row, first, ops_of, tail_ops, wait, is_masked=is_masked
            )
            return state.probs, wait

        probs, _ = lax.fori_loop(0, num_bodies, _body, (probs, wait))
        return probs

    def body_ops(
        self, row: PagedRow, first: jax.Array | int, *, is_masked: bool
    ) -> tuple[Callable[..., PagedFragmentOps], PagedFragmentOps]:
        """body's (ops_of, tail_ops) for row's body from tile first: when
        is_masked, masked at the body's tiles and at the body before."""
        if not is_masked:
            return self.row_ops, self.row_ops(row)
        body_tiles = self.tiles.body_tiles

        def _numbers(start):
            return tuple(start + tile for tile in range(body_tiles))

        return (
            lambda row, tiles: self.masked_ops(row, tiles, _numbers(first)),
            self.masked_ops(row, (), _numbers(first - body_tiles)),
        )

    def body(
        self,
        probs: tuple[jax.Array, ...],
        row: PagedRow,
        first: jax.Array | int,
        ops_of: Callable[..., PagedFragmentOps],
        tail_ops: FlaggedTailOps | PagedFragmentOps,
        wait: BlockWait,
        *,
        is_masked: bool,
    ) -> tuple[PipelineState, BlockWait]:
        """Run row's body_tiles tiles from first in one segment that stages
        the next tile's K, with ops_of(row, tiles)'s ops, after wait; tail_ops
        finishes the segment before. Returns the state and the next body's
        wait."""
        body_tiles = self.tiles.body_tiles
        items = self.tile_items(body_tiles)
        tiles = tuple(
            self.row_tile(row, first + tile, is_masked=is_masked)
            for tile in range(body_tiles)
        )
        self.wait_due_block(wait)
        ops = self.with_prefetch(ops_of(row, tiles), items, wait.is_due)
        assumed = self.pipeline.tail_of(items, tail_ops, has_staged_next_k=True)
        state = self.pipeline.advance(
            PipelineState(probs=probs, tail=assumed),
            items,
            ops,
            next_k_tile=self.staged_k(self.next_tile(row, first + body_tiles)),
        )
        self.pipeline.check_loop_tail(state.tail, assumed)
        next_first = first + body_tiles
        return state, self.block_wait(row, next_first, next_first + body_tiles)

    def checked_unit(
        self,
        unit: jax.Array,
        carry: tuple[tuple[jax.Array, ...], BlockWait],
        *,
        row: PagedRow,
        final_unit: jax.Array,
        is_masked: bool,
    ) -> tuple[tuple[jax.Array, ...], BlockWait]:
        """Run check unit unit's last body, whose last tile also writes the
        next unit's checkpoint and collects its row sums, then the next
        unit's first body, which finishes that tile. Those sums flag unit: a
        unit before final_unit then replays through the first body's tiles,
        and the last body adds nothing for the items a replay redid. carry
        holds the probs and the last body's wait."""
        probs, wait = carry
        next_unit = unit + 1
        largest = LargestSums()
        unit_end_ops = partial(
            self.unit_end_ops,
            slot=lax.rem(next_unit, NUM_CHECKPOINTS),
            largest=largest,
        )
        last_body = unit * UNIT_TILES + LOOP_BODY_TILES
        ops_of, tail_ops = self.body_ops(row, last_body, is_masked=is_masked)
        # A work item carries no mode: its ops know which tile starts a row,
        # and no tile of a loop body does.
        flagged_tail = FlaggedTailOps(
            ops=tail_ops,
            is_flagged=self.unit_guard[0] > 0,
            is_row_bound=lambda mode: mode is not None,
        )
        state, wait = self.body(
            probs,
            row,
            last_body,
            lambda row, tiles: unit_end_ops(ops_of(row, tiles)),
            flagged_tail,
            wait,
            is_masked=is_masked,
        )
        first_body = next_unit * UNIT_TILES
        ops_of, tail_ops = self.body_ops(row, first_body, is_masked=is_masked)
        state, wait = self.body(
            state.probs,
            row,
            first_body,
            ops_of,
            unit_end_ops(tail_ops),
            wait,
            is_masked=is_masked,
        )
        self.unit_guard[0] = jnp.where(
            unit < final_unit, guard_flag(self.traits, largest.value), 0
        )

        @pl.when(self.unit_guard[0] > 0)
        def _replay():
            self.replay_tiles(
                row,
                unit,
                next_unit * UNIT_TILES + LOOP_BODY_TILES,
                self.pipeline.free_slots(state.tail, REPLAY_RESULT_SLOTS),
                checkpoint_unit=next_unit,
            )

        return state.probs, wait

    @property
    def half_tile_items(self) -> tuple[WorkItem, ...]:
        """The items of a tile without its first q compute tile's fragments,
        if they keep the result slot ring's phase; else none."""
        num_fragments = self.traits.group_bq // FRAGMENT_ROWS
        first_fragment = self.traits.frag_rows // FRAGMENT_ROWS
        items = self.tile_items()[first_fragment:]
        pipeline = self.pipeline
        if (
            len(items) % pipeline.num_slots
            or len(items) < pipeline.delays.pv_pop
            or first_fragment >= num_fragments
        ):
            return ()
        return items

    def has_half_tile(self, row: PagedRow, first_tile: jax.Array) -> jax.Array:
        """Whether the last tile row runs is a masked one (from first_tile
        on) whose first q compute tile's rows see none of its keys."""
        last_tile = row.span.num_tiles - 1
        key_start = self.key_start(row.span, last_tile)
        sees_none = jnp.logical_or(
            key_start >= row.ctx.kv_frontier,
            row.q_tok + self.traits.bq_compute
            <= key_start - row.ctx.bottom_right_offset,
        )
        return jnp.logical_and(last_tile >= first_tile, sees_none)

    def half_tile(
        self,
        carry: tuple[tuple[jax.Array, ...], BlockWait],
        row: PagedRow,
    ) -> tuple[jax.Array, ...]:
        """Run row's last tile on half_tile_items; carry holds the probs and
        its wait."""
        probs, wait = carry
        tile_number = row.span.num_tiles - 1
        items = self.half_tile_items
        tile = self.row_tile(row, tile_number, is_masked=True)
        self.wait_due_block(wait)
        assumed = self.pipeline.tail_of(
            self.tile_items(),
            self.masked_ops(row, (), (tile_number - 1,)),
            has_staged_next_k=True,
        )
        state = self.pipeline.advance(
            PipelineState(probs=probs, tail=assumed),
            items,
            self.with_prefetch(
                self.masked_ops(row, (tile,), (tile_number,)),
                items,
                wait.is_due,
            ),
            next_k_tile=self.staged_k(self.next_tile(row, tile_number + 1)),
        )
        return state.probs

    def masked_tiles(
        self, probs: tuple[jax.Array, ...], row: PagedRow
    ) -> tuple[jax.Array, ...]:
        """Run row's stream tiles from body_end on, masked, in rolled bodies
        of one unit; a causal row's last tile skips the fragments of its
        first q compute tile when they see none of its keys."""
        unit = self.tiles.unit_tiles
        items = self.tile_items(unit)
        first_tile = row.body_end

        def _masked(first):
            tile_numbers = tuple(first + tile for tile in range(unit))
            tiles = tuple(
                self.row_tile(row, tile_number, is_masked=True)
                for tile_number in tile_numbers
            )
            return self.masked_ops(row, tiles, tile_numbers)

        def _body(index, carry):
            probs, wait = carry
            first = first_tile + index * unit
            self.wait_due_block(wait)
            assumed = self.pipeline.tail_of(
                items, _masked(first - unit), has_staged_next_k=True
            )
            state = self.pipeline.advance(
                PipelineState(probs=probs, tail=assumed),
                items,
                self.with_prefetch(_masked(first), items, wait.is_due),
                next_k_tile=self.staged_k(self.next_tile(row, first + unit)),
            )
            self.pipeline.check_loop_tail(state.tail, assumed)
            # The next unit's wait, set inside this one.
            return state.probs, self.block_wait(
                row, first + unit, first + 2 * unit
            )

        num_units = (row.stream_end - first_tile) // unit
        carry = (probs, self.block_wait(row, first_tile, first_tile + unit))
        if not (self.is_causal and self.half_tile_items and unit == 1):
            probs, _ = lax.fori_loop(0, num_units, _body, carry)
            return probs
        has_half_tile = self.has_half_tile(row, first_tile)
        carry = lax.fori_loop(
            0,
            num_units - has_half_tile.astype(jnp.int32),
            _body,
            carry,
        )
        return lax.cond(
            has_half_tile,
            partial(self.half_tile, row=row),
            lambda carry: carry[0],
            carry,
        )

    def start_stream(self) -> None:
        """Point the stream's cursor at the grid's first block and start the
        DMAs of its first lead_blocks + 1 blocks."""
        zero = jnp.int32(0)
        self.stream_state[STREAM_ISSUE_POSITION] = zero
        self.store_cursor(
            BlockCursor(
                flat=zero,
                group=zero,
                index=zero,
                is_past=(self.num_grid_groups == 0).astype(jnp.int32),
                visit=zero,
            )
        )

        def _start(_, carry):
            self.start_block(True)
            return carry

        lax.fori_loop(0, self.ring.lead_blocks + 1, _start, None)

    def start_group_q(
        self,
        group: jax.Array,
        q_count: jax.Array,
        first_si: jax.Array,
        num_rows: jax.Array,
    ) -> None:
        """Start the Q of the num_rows rows of head group group from schedule
        row first_si on, into the Q ring from position q_count on."""
        traits, schedule = self.traits, self.schedule

        def _start(member):
            _, _, ctx = schedule.row_interval(first_si + member)
            start_q_row(
                traits,
                self.q_refs,
                group * traits.group_heads,
                schedule.q_token(ctx),
                self.q_slot(q_count + member),
            )

        _start(0)
        for_each_later_row(num_rows, _start)

    def wait_group_q(self, q_count: jax.Array, num_rows: jax.Array) -> None:
        """Wait for the Q of num_rows rows from Q ring position q_count on;
        the K tiles take the Q scale, so the staged Q is only rotated."""

        def _wait(member):
            wait_rotated_q_row(
                self.traits, self.q_refs, self.q_slot(q_count + member)
            )

        _wait(0)
        for_each_later_row(num_rows, _wait)

    @property
    def fill_items(self) -> tuple[WorkItem, ...]:
        """The last items of a start segment, the fewest whole slot ring
        laps that hold pv_pop items: they leave a whole start segment's tail."""
        pipeline = self.pipeline
        laps = -(-pipeline.delays.pv_pop // pipeline.num_slots)
        return self.start_items()[-laps * pipeline.num_slots :]

    def fill(self) -> PagedGroupCarry:
        """Start the stream and the first group's Q and fill the pipeline;
        returns the group loop's initial carry. The fill's items read zeros
        at the parity the first group does not take, so that group's start
        segment finishes it as any group's last tiles."""
        zero = jnp.int32(0)
        self.start_stream()
        first_si, num_rows = self.grid_group(zero, self.num_grid_groups == 0)
        self.start_group_q(zero, zero, first_si, num_rows)
        self.wait_block(zero)
        pipeline = self.pipeline
        discarded = DiscardedFragmentOps(self.traits.head_dim_qk)
        first_tile = TileSource(
            row=self.first_tile_row(self.group_row(first_si), zero),
            kv_head=self.kv_head(zero),
        )
        fill = pipeline.advance(
            pipeline.idle_state(),
            self.fill_items,
            discarded,
            next_k_tile=self.staged_k(first_tile),
        )
        pipeline.check_loop_tail(
            fill.tail,
            pipeline.tail_of(
                self.start_items(), discarded, has_staged_next_k=True
            ),
        )
        finished = FinishedGroup(
            is_live=zero,
            group=zero,
            first_si=zero,
            num_rows=jnp.int32(1),
            is_short=zero,
            parity=jnp.int32(ENTRY_PARITY),
            q_count=zero,
            out_count=zero,
            prev_out_tok=zero,
            num_tiles=jnp.int32(self.num_slots),
        )
        return PagedGroupCarry(
            q_count=zero,
            block=zero,
            group=zero,
            index=zero,
            probs=fill.probs,
            finished=finished,
            prev_out_tok=zero,
            num_out_copies=zero,
        )

    def first_row(
        self,
        carry: PagedGroupCarry,
        first: GroupRow,
        parity: jax.Array,
        next_group: jax.Array,
        is_drain: jax.Array,
        next_tile_row: jax.Array,
    ) -> PagedRow:
        """The first row, first, of the group carry starts at parity; the
        drain group's row runs a start segment alone. next_tile_row is the
        staging row of the next group's first tile."""
        span = first.span._replace(
            num_tiles=jnp.where(is_drain, 1, first.span.num_tiles)
        )
        return PagedRow(
            ctx=first.ctx,
            q_slot=self.q_slot(carry.q_count),
            q_tok=first.q_tok,
            state=self.row_state(parity, 0),
            span=span,
            body_end=jnp.where(
                is_drain,
                self.tiles.start_tiles,
                self.body_end(first.span, first.ctx),
            ),
            stream_end=self.stream_tiles(span.num_tiles),
            first_block=carry.block,
            kv_head=self.kv_head(carry.group),
            next_kv_head=self.kv_head(next_group),
            next_tile_row=next_tile_row,
        )

    def group_step(
        self, index: jax.Array, carry: PagedGroupCarry
    ) -> PagedGroupCarry:
        """Run flat grid group index, the drain group past the grid's last:
        its start segment, which finishes the group before, then, for a
        group of one row, that row's other tiles."""
        is_drain = index == self.num_grid_groups
        next_group, next_index = self.next_grid_group(
            index, carry.group, carry.index
        )
        first_si, num_rows = self.grid_group(carry.index, is_drain)
        self.wait_group_q(carry.q_count, num_rows)

        @pl.when(jnp.logical_not(is_drain))
        def _prefetch_q():
            self.start_group_q(
                next_group,
                carry.q_count + num_rows,
                *self.grid_group(next_index, index + 1 == self.num_grid_groups),
            )

        first = self.group_row(first_si)
        is_short = first.span.num_tiles == 1
        num_visits = jnp.where(
            jnp.logical_or(is_drain, is_short), 1, self.num_visits(first.span)
        )
        parity = NUM_ROW_PARITIES - 1 - carry.finished.parity
        is_grid_row = jnp.logical_not(is_drain)
        next_first_si, _ = self.grid_group(
            next_index, index + 1 >= self.num_grid_groups
        )
        row = self.first_row(
            carry,
            first,
            parity,
            next_group,
            is_drain,
            self.first_tile_row(
                self.group_row(next_first_si), carry.block + num_visits
            ),
        )
        slots = self.slot_rows(
            first_si, num_rows, is_short, parity, carry.q_count, 0, first.span
        )
        probs = self.start_group(
            carry.probs, carry.finished, slots, is_short, row
        )
        self.complete_group(carry.finished)
        probs = self.row_bodies(probs, row)
        probs = self.masked_tiles(probs, row)
        is_live = is_grid_row.astype(jnp.int32)
        num_tiles = jnp.where(
            jnp.logical_or(is_drain, is_short),
            self.num_slots,
            self.stream_tiles(first.span.num_tiles),
        )
        _, _, last_ctx = self.schedule.row_interval(first_si + num_rows - 1)
        return PagedGroupCarry(
            q_count=carry.q_count + num_rows,
            block=carry.block + num_visits,
            group=next_group,
            index=next_index,
            probs=probs,
            finished=FinishedGroup(
                is_live=is_live,
                group=carry.group,
                first_si=first_si,
                num_rows=num_rows,
                is_short=is_short.astype(jnp.int32),
                parity=jnp.int32(parity),
                q_count=carry.q_count,
                out_count=carry.num_out_copies,
                prev_out_tok=carry.prev_out_tok,
                num_tiles=num_tiles,
            ),
            prev_out_tok=self.schedule.q_token(last_ctx),
            num_out_copies=carry.num_out_copies + is_live * num_rows,
        )

    def empty(
        self, probs: tuple[jax.Array, ...], finished: FinishedGroup
    ) -> None:
        """Empty the pipeline after the drain group, whose start segment
        waited for the last block the stream started."""
        state = self.pipeline.advance(
            PipelineState(
                probs=probs, tail=self.finished_tail(finished, finished.parity)
            ),
            self.tile_items()[: self.pipeline.delays.pv_pop],
            DiscardedFragmentOps(self.traits.head_dim_qk),
        )
        self.pipeline.drain(state)

    def finish_grid(self, last: PagedGroupCarry) -> None:
        """After the drain group, whose carry is last: empty the pipeline,
        wait out the output copies and fill the packed padding rows."""
        traits, schedule = self.traits, self.schedule
        self.empty(last.probs, last.finished)
        drain(
            last.num_out_copies,
            traits.num_stages,
            lambda slot: o_copy(traits, self.out_refs, 0, 0, slot).wait(),
        )
        fill_packed_tail(
            traits,
            self.variant,
            self.out_refs,
            schedule.packed_q_end,
            schedule.num_head_groups,
        )


def paged_mxu_fwd_body(
    traits: PagedFwdTraits,
    variant: Variant,
    schedule: FwdSchedule,
    paged_refs: PagedRefs,
    *,
    operands: VariantOperands,
) -> None:
    """paged_fwd_body on the explicit MXU pipeline: every row group of the
    grid, then a drain group. The arguments are paged_fwd_body's."""
    variant.check_operands(operands)
    assert schedule.q_token is not None and schedule.kv_page is not None
    assert not (traits.rotary_dim and traits.rotary_k), traits
    if variant.return_lse:
        lse_stage = paged_refs.out_refs.lse_stage
        lse_stage[...] = jnp.zeros(lse_stage.shape, jnp.float32)
    (
        row_max,
        row_sum,
        o,
        checkpoint_sum,
        checkpoint_o,
        stream_state,
        guard,
        unit_guard,
    ) = paged_refs.row_refs
    loop = PagedMxuLoop(
        traits=traits,
        variant=variant,
        operands=operands,
        schedule=schedule,
        pipeline=MxuPipeline.allocate(
            head_dim_v=traits.head_dim_v,
            has_mode_free_accumulate=zeroes_row_start(
                traits.group_bq // FRAGMENT_ROWS
            ),
        ),
        q_refs=paged_refs.q_refs,
        kv_refs=paged_refs.kv_refs,
        out_refs=paged_refs.out_refs,
        row_max=row_max,
        row_sum=row_sum,
        o=o,
        checkpoint_sum=checkpoint_sum,
        checkpoint_o=checkpoint_o,
        stream_state=stream_state,
        guard=guard,
        unit_guard=unit_guard,
    )
    last = lax.fori_loop(
        0, loop.num_grid_groups + 1, loop.group_step, loop.fill()
    )
    loop.finish_grid(last)
