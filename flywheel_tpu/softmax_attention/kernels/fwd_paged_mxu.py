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

On a chip with MXU accumulators a head_dim 256 paged build runs
paged_mxu_fwd_body in place of kernels/fwd_paged.py's paged_fwd_body.

The loop runs the schedule's row groups: a row alone, or rows of one live
tile each that share one start segment. Every segment stages the K tile of
the next one, so the pipeline never drains between groups: a group's last
items finish in the next group's start segment, and a drain group after the
grid's last group finishes that one. kv streams tile by tile, and groups
alternate between NUM_ROW_PARITIES parities of row states. The overflow guard
checks a row by check units of UNIT_TILES tiles, each replayed from its
checkpoint when its row sums flag it; a row's final unit replays as the row
is completed.
"""

import dataclasses
import math
from collections.abc import Callable, Sequence
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from flywheel_tpu.core.tpu import MXU_COLUMNS
from flywheel_tpu.softmax_attention.config import (
    DEFAULT_MASK_VALUE,
    PAGE_SIZE_MULTIPLE,
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
from flywheel_tpu.softmax_attention.kv_layout import (
    kv_word_halves,
    load_staged_words,
    staged_words,
)
from flywheel_tpu.softmax_attention.mxu_pipeline import (
    FRAGMENT_ROWS,
    MxuPipeline,
    PipelineState,
    Tail,
    WorkItem,
)
from flywheel_tpu.softmax_attention.schedule import (
    FwdSchedule,
    MxuRowTiles,
    ScheduleRow,
)
from flywheel_tpu.softmax_attention.traits import PagedFwdTraits
from flywheel_tpu.softmax_attention.variant import Variant, VariantOperands

# A tile's page DMAs start at least this many tiles before it runs.
MXU_MIN_PREFETCH_TILES = 6
# A DMA start orders the VMEM loads after it, so the item that starts a
# segment's DMAs first loads the q fragments of this many items from it on.
MXU_PRELOADED_FRAGMENTS = 4
# A row's start segment starts its DMAs at this item, ahead of the finish
# staged behind its items; other segments start them mid first tile.
MXU_START_PREFETCH_ITEM = 4


class TileStream(NamedTuple):
    """The kv tile stream: a ring of num_slots tile slots whose tiles' DMAs
    start prefetch_tiles tiles before they run, and one slot past it for a
    replay's tiles. A segment starts at most described_tiles tiles' DMAs."""

    num_slots: int
    prefetch_tiles: int
    described_tiles: int

    @property
    def replay_slot(self) -> int:
        return self.num_slots

    @property
    def num_staging_slots(self) -> int:
        return self.num_slots + 1

    @property
    def num_state_words(self) -> int:
        return STREAM_DESCRIPTORS + self.described_tiles * DESCRIPTOR_WORDS


def tile_stream(group_rows: int) -> TileStream:
    """The tile stream of rows of group_rows rows: room for its longest
    segment's tiles and the tiles in flight, more than a segment holds since
    a segment waits for the tile after it before it starts its prefetches."""
    tiles = row_tiles(group_rows // FRAGMENT_ROWS)
    segment_tiles = max(tiles.start_tiles, tiles.body_tiles, tiles.unit_tiles)
    num_slots = pl.next_power_of_2(
        segment_tiles + max(MXU_MIN_PREFETCH_TILES, segment_tiles + 1)
    )
    return TileStream(
        num_slots=num_slots,
        prefetch_tiles=num_slots - segment_tiles,
        described_tiles=pl.next_power_of_2(segment_tiles),
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


def paged_mxu_block_sizes(block_sizes: BlockSizes, gqa_fold: int) -> BlockSizes:
    """block_sizes as an explicit paged build runs them, with Q and output
    rings of QUEUED_GROUPS row groups."""
    blocks = mxu_block_sizes(block_sizes)
    capacity = group_capacity(gqa_fold * blocks.block_q // FRAGMENT_ROWS)
    return dataclasses.replace(blocks, num_stages=QUEUED_GROUPS * capacity)


class TileSource(NamedTuple):
    """A staged kv tile of an explicit build: its staging slot, its kv head,
    and how many of its rows hold valid V (None: all of them)."""

    slot: jax.Array | int
    kv_head: jax.Array
    num_valid_rows: jax.Array | None = None


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
        staged_words(kv_refs.pair_buf, source.slot), 0, MXU_COLUMNS, 1
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
    """The mxu_pipeline.FragmentOps of a segment of explicit paged rows.

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

    # A V handle: the tile's staging slot, kv head and valid rows.
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
            for word in (source.slot, source.kv_head, num_valid_rows)
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
    """ops whose item preload_keys[0] (fragment, tile) runs start_prefetch
    as it issues its QK, inside the segment's own basic block. It first
    loads the q fragments of the items at preload_keys: the DMA starts
    order the VMEM loads after them. No key starts none."""

    ops: PagedFragmentOps
    preload_keys: tuple[tuple[int, int], ...]
    start_prefetch: Callable[[], None]
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
        self.start_prefetch()
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


class TileCursor(NamedTuple):
    """The cursor of the kv tile stream: stream tile tile of count of row si
    of group, or of the drain row once is_past; pivot, first_key, page_base
    and kv_head are that row's."""

    group: jax.Array
    si: jax.Array
    is_past: jax.Array
    tile: jax.Array
    count: jax.Array
    pivot: jax.Array
    first_key: jax.Array
    page_base: jax.Array
    kv_head: jax.Array


# Note (david): the kv tile stream lives in SMEM, out of the scalar registers
# of the loop bodies: the stream position of the next tile to start, the
# cursor at the next tile to describe, then the DMA operands (kv head, then
# each piece's page and first row in it) of the described tiles. A page is
# whole PAGE_SIZE_MULTIPLE rows, which bounds a tile's page DMAs.
STREAM_START_POSITION = 0
STREAM_CURSOR = 1
STREAM_DESCRIPTORS = STREAM_CURSOR + len(TileCursor._fields)
DESCRIPTOR_KV_HEAD = 0
DESCRIPTOR_PIECES = 1
DESCRIPTOR_PIECE_WORDS = 2
DESCRIPTOR_WORDS = (
    DESCRIPTOR_PIECES
    + DESCRIPTOR_PIECE_WORDS * MXU_COLUMNS // PAGE_SIZE_MULTIPLE
)


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


class GroupRow(NamedTuple):
    """A row of a row group: schedule row si over kv blocks lo..hi, its first
    token, its live tiles from first_key and the one it starts in."""

    si: jax.Array
    lo: jax.Array
    hi: jax.Array
    ctx: ScheduleRow
    q_tok: jax.Array
    first_key: jax.Array
    num_tiles: jax.Array
    pivot: jax.Array


class SlotRow(NamedTuple):
    """The row of one tile slot of a start segment, or of the tail it
    leaves: its row-state index and Q slot, its first token, what masks its
    scores, the slot tile's first key and the row's valid keys."""

    state: jax.Array
    q_slot: jax.Array
    q_tok: jax.Array
    kv_frontier: jax.Array
    bottom_right_offset: jax.Array
    kept_start: jax.Array
    kept_end: jax.Array
    key_start: jax.Array
    kv_used_end: jax.Array


class PagedRow(NamedTuple):
    """A row the loop runs: its schedule row, staged Q and row-state index,
    its live tiles (num_tiles from first_key, run from its pivot tile, in
    rolled bodies up to body_end), the stream position of its first tile,
    and its and the next group's kv heads."""

    ctx: ScheduleRow
    q_slot: jax.Array
    q_tok: jax.Array
    state: jax.Array | int
    first_key: jax.Array
    num_tiles: jax.Array
    pivot: jax.Array
    body_end: jax.Array
    first_tile: jax.Array
    kv_head: jax.Array
    next_kv_head: jax.Array


class PagedGroupCarry(NamedTuple):
    """The group loop's carry: the Q ring position of the group's first row,
    the stream position of its first tile, its head group and group index,
    the pipeline's carried probs, the group before it, the first token of
    the last row written, and the output copies started."""

    q_count: jax.Array
    tile: jax.Array
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
    stream_state holds the kv tile stream (the STREAM_* words), guard a
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
    def num_grid_rows(self) -> jax.Array:
        return self.schedule.num_head_groups * self.schedule.num_rows

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
    def stream(self) -> TileStream:
        return tile_stream(self.traits.group_bq)

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
        num_tiles = self.num_row_tiles(lo, hi, ctx)
        return GroupRow(
            si=si,
            lo=lo,
            hi=hi,
            ctx=ctx,
            q_tok=self.schedule.q_token(ctx),
            first_key=self.row_first_key(lo, ctx),
            num_tiles=num_tiles,
            pivot=self.row_pivot(lo, ctx, num_tiles),
        )

    def q_slot(self, q_count: jax.Array) -> jax.Array:
        """The Q ring slot of the row at Q ring position q_count."""
        return lax.rem(q_count, self.traits.num_stages)

    @property
    def replay_q_slot(self) -> int:
        """The Q slot a replay stages its row in, past the ring."""
        return self.traits.num_stages

    def kv_head(self, group: jax.Array) -> jax.Array:
        return (
            group * self.traits.group_heads // self.schedule.q_heads_per_kv_head
        )

    def row_first_key(self, lo: jax.Array, ctx: ScheduleRow) -> jax.Array:
        """The first key of the first tile that holds a key a kept row of
        the row of ctx sees; lo is the row's first block."""
        return lo * self.traits.bkv + ctx.first_live_tile_lo * MXU_COLUMNS

    def num_row_tiles(
        self, lo: jax.Array, hi: jax.Array, ctx: ScheduleRow
    ) -> jax.Array:
        """The live kv tiles of the row of ctx over blocks lo..hi, as the
        schedule counts them for its row groups."""
        return (
            (hi - lo) * self.traits.num_k_compute_tiles
            + self.schedule.num_live_tiles(hi, ctx)
            - ctx.first_live_tile_lo
        )

    def row_pivot(
        self, lo: jax.Array, ctx: ScheduleRow, num_tiles: jax.Array
    ) -> jax.Array:
        """The live tile the row of ctx, of num_tiles live tiles, starts in:
        under a left window bound the one that holds the first key its last
        kept row sees, which every kept row sees; else its first."""
        if not self.has_left_window:
            return jnp.int32(0)
        first_key = self.row_first_key(lo, ctx)
        pivot = (ctx.last_anchor_key - first_key) // MXU_COLUMNS
        return jnp.clip(pivot, 0, jnp.maximum(num_tiles - 1, 0))

    def body_end(
        self, lo: jax.Array, hi: jax.Array, ctx: ScheduleRow
    ) -> jax.Array:
        """The end of the row's rolled bodies of body_tiles tiles, whole
        ones from its start segment on: over all of its tiles under a left
        window bound, else over those before its first masked block."""
        tiles = self.tiles
        num_tiles = self.num_row_tiles(lo, hi, ctx)
        if self.has_left_window:
            end = num_tiles
        else:
            end = jnp.clip(
                jnp.clip(ctx.mask_from - lo, 0, hi - lo + 1)
                * self.traits.num_k_compute_tiles
                - ctx.first_live_tile_lo,
                0,
                num_tiles,
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

    def tile_key(
        self,
        first_key: jax.Array,
        pivot: jax.Array | int,
        tile_number: jax.Array | int,
    ) -> jax.Array:
        """The first key of the tile a row whose live tiles start at
        first_key runs as its tile tile_number: in key order, or from its
        pivot tile pivot under a left window bound."""
        if not self.has_left_window:
            return first_key + tile_number * MXU_COLUMNS
        return first_key + pivot_order(pivot, tile_number) * MXU_COLUMNS

    def key_start(
        self, row: PagedRow, tile_number: jax.Array | int
    ) -> jax.Array:
        """The first key of the tile row runs as its tile tile_number."""
        return self.tile_key(row.first_key, row.pivot, tile_number)

    def stream_row(
        self, group: jax.Array, si: jax.Array, is_past: jax.Array
    ) -> tuple[jax.Array, ...]:
        """(stream tiles, pivot, first key, page_base, kv_head) of row si of
        group; is_past gives the drain row, the grid's last row with a start
        segment alone, from its first tile."""
        schedule = self.schedule
        group = jnp.where(is_past, schedule.num_head_groups - 1, group)
        si = jnp.where(is_past, jnp.maximum(schedule.num_rows - 1, 0), si)
        row = self.group_row(si)
        return (
            jnp.where(is_past, self.num_slots, row.ctx.stream_tiles),
            jnp.where(is_past, 0, row.pivot),
            row.first_key,
            row.ctx.page_base,
            self.kv_head(group),
        )

    def stream_tiles(self, num_tiles: jax.Array) -> jax.Array:
        """The stream tiles of a row of num_tiles live tiles: at least its
        start segment, in whole units."""
        tiles = self.tiles
        num_tiles = jnp.maximum(num_tiles, tiles.start_tiles)
        unit = tiles.unit_tiles
        return (num_tiles + unit - 1) // unit * unit

    def stream_slot(self, position: jax.Array) -> jax.Array:
        """The stream slot of the tile at stream position position."""
        return jnp.bitwise_and(position, self.stream.num_slots - 1)

    @property
    def tile_piece_rows(self) -> int:
        """The rows of one of a kv tile's DMAs, which lie in one page: the
        largest that divides both the tile and the page."""
        return math.gcd(self.traits.paged.page_size, MXU_COLUMNS)

    def tile_pages(
        self, page_base: jax.Array, key_start: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array], ...]:
        """The (page, first row in it) of each DMA of the tile whose first
        key is key_start, in the pages of the block table row at page_base."""
        # Note (david): a page past the row's pages reads through the clamped
        # table entry; its keys are past seqused_k, so they are masked and
        # their V rows zeroed.
        page_size = self.traits.paged.page_size
        pieces = []
        for piece_key in range(0, MXU_COLUMNS, self.tile_piece_rows):
            key = key_start + piece_key
            page = jnp.clip(
                self.schedule.kv_page(page_base, 0, key // page_size),
                0,
                self.kv_refs.num_cache_pages - 1,
            )
            pieces.append((page, key % page_size))
        return tuple(pieces)

    def start_tile_pieces(
        self,
        kv_head: jax.Array,
        pieces: Sequence[tuple[jax.Array, jax.Array]],
        slot: jax.Array | int,
    ) -> None:
        """Start the DMAs of kv_head's word rows of a tile's pieces, each
        (page, first row in it), into staging slot slot."""
        # Note (david): every DMA moves tile_piece_rows rows, so its size and
        # its wait are static.
        kv_refs, piece_rows = self.kv_refs, self.tile_piece_rows
        for piece, (page, page_row) in enumerate(pieces):
            pltpu.make_async_copy(
                paged_kv_source(
                    kv_refs,
                    page,
                    piece_rows,
                    kv_head,
                    pl.multiple_of(page_row, piece_rows),
                ),
                paged_buffer_tile(
                    kv_refs, slot, piece * piece_rows, piece_rows
                ),
                kv_refs.sems.at[SEM_K, slot],
            ).start()

    def wait_tile_dmas(self, slot: jax.Array | int) -> None:
        """Wait for the DMAs of a tile into staging slot slot."""
        # Note (david): every DMA of a tile signals the slot's semaphore by
        # its size, so one wait on the whole slot covers them.
        window = self.kv_refs.pair_buf.at[slot]
        pltpu.make_async_copy(
            window, window, self.kv_refs.sems.at[SEM_K, slot]
        ).wait()

    def store_cursor(self, cursor: TileCursor) -> None:
        for index, value in enumerate(cursor):
            self.stream_state[STREAM_CURSOR + index] = jnp.int32(value)

    def step_cursor(self, cursor: TileCursor) -> TileCursor:
        """cursor one stream tile on."""
        schedule = self.schedule
        is_group_end = cursor.si + 1 >= schedule.num_rows
        group = jnp.where(is_group_end, cursor.group + 1, cursor.group)
        si = jnp.where(is_group_end, 0, cursor.si + 1)
        is_past = jnp.logical_or(
            cursor.is_past > 0,
            jnp.logical_and(is_group_end, group >= schedule.num_head_groups),
        )
        count, pivot, first_key, page_base, kv_head = self.stream_row(
            group, si, is_past
        )
        next_row = TileCursor(
            group=group,
            si=si,
            is_past=is_past.astype(jnp.int32),
            tile=jnp.int32(0),
            count=count,
            pivot=pivot,
            first_key=first_key,
            page_base=page_base,
            kv_head=kv_head,
        )
        is_row_end = cursor.tile + 1 >= cursor.count
        return jax.tree.map(
            partial(jnp.where, is_row_end),
            next_row,
            cursor._replace(tile=cursor.tile + 1),
        )

    def descriptor(self, position: jax.Array | int) -> jax.Array:
        """The first SMEM word of the DMA operands of stream tile position."""
        return (
            STREAM_DESCRIPTORS
            + jnp.bitwise_and(position, self.stream.described_tiles - 1)
            * DESCRIPTOR_WORDS
        )

    def describe_tile(self, position: jax.Array | int) -> None:
        """Write the DMA operands of the cursor's tile, the one at stream
        position position, and step the cursor one stream tile on."""
        state = self.stream_state
        cursor = TileCursor(*(
            state[STREAM_CURSOR + index]
            for index in range(len(TileCursor._fields))
        ))
        pieces = self.tile_pages(
            cursor.page_base,
            self.tile_key(cursor.first_key, cursor.pivot, cursor.tile),
        )
        first_word = self.descriptor(position)
        state[first_word + DESCRIPTOR_KV_HEAD] = cursor.kv_head
        for piece, piece_words in enumerate(pieces):
            for offset, word in enumerate(piece_words):
                state[
                    first_word
                    + DESCRIPTOR_PIECES
                    + piece * DESCRIPTOR_PIECE_WORDS
                    + offset
                ] = word
        self.store_cursor(self.step_cursor(cursor))

    def start_tiles(self, num_tiles: int) -> None:
        """Start the DMAs of the next num_tiles stream tiles, described
        before, into their stream slots; then describe as many tiles on."""
        state = self.stream_state
        position = state[STREAM_START_POSITION]
        num_pieces = MXU_COLUMNS // self.tile_piece_rows
        for offset in range(num_tiles):
            first_word = self.descriptor(position + offset)
            self.start_tile_pieces(
                state[first_word + DESCRIPTOR_KV_HEAD],
                [
                    tuple(
                        state[
                            first_word
                            + DESCRIPTOR_PIECES
                            + piece * DESCRIPTOR_PIECE_WORDS
                            + word
                        ]
                        for word in range(DESCRIPTOR_PIECE_WORDS)
                    )
                    for piece in range(num_pieces)
                ],
                self.stream_slot(position + offset),
            )
        state[STREAM_START_POSITION] = position + num_tiles
        for offset in range(num_tiles):
            self.describe_tile(position + self.stream.described_tiles + offset)

    def wait_tile(self, position: jax.Array) -> None:
        """Wait for the DMAs of the stream tile at position."""
        self.wait_tile_dmas(self.stream_slot(position))

    def with_prefetch(
        self,
        ops: PagedFragmentOps,
        items: tuple[WorkItem, ...],
        num_tiles: int = 1,
        prefetch_item: int | None = None,
    ) -> PrefetchingOps:
        """ops whose segment of items starts the prefetches of num_tiles
        tiles at its first tile's item prefetch_item (its middle one when
        None): a DMA start orders the segment's memory accesses around it."""
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
            start_prefetch=partial(self.start_tiles, num_tiles),
        )

    def row_tile(
        self, row: PagedRow, tile_number: jax.Array | int, *, is_masked: bool
    ) -> TileSource:
        """The source of the tile row runs as its tile tile_number;
        is_masked zeroes its V rows past the row's valid keys."""
        if is_masked:
            num_valid_rows = row.ctx.kv_used_end - self.key_start(
                row, tile_number
            )
        else:
            num_valid_rows = None
        return TileSource(
            slot=self.stream_slot(row.first_tile + tile_number),
            kv_head=row.kv_head,
            num_valid_rows=num_valid_rows,
        )

    def next_row_tile(
        self, row: PagedRow, tile_number: jax.Array | int
    ) -> TileSource:
        """The source of row's tile tile_number, or of the next row's first
        tile past row's live tiles; waits for its DMAs."""
        self.wait_tile(row.first_tile + tile_number)
        return TileSource(
            slot=self.stream_slot(row.first_tile + tile_number),
            kv_head=jnp.where(
                tile_number < row.num_tiles, row.kv_head, row.next_kv_head
            ),
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
                geometry, row.q_tok, self.key_start(row, tile_number)
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
        starts_at_pivot: jax.Array,
    ) -> tuple[SlotRow, ...]:
        """The rows of the tile slots of a group's start segment, or of the
        tail its last tiles leave. Slot t of a group of one-tile rows
        (is_short) is its row t's tile, and a slot past its rows a tile past
        its last row's, which hides every key. Every slot of a group of one
        row is the tile that row runs as its tile first_long_tile + t, from
        its pivot tile where starts_at_pivot holds; the drain row runs from
        its first tile."""
        rows = [self.group_row(first_si)]
        for member in range(1, self.num_slots):
            rows.append(
                self.group_row(first_si + jnp.minimum(member, num_rows - 1))
            )
        first = rows[0]
        long_pivot = jnp.where(starts_at_pivot, first.pivot, 0)

        def _pick(short_value, long_value):
            return jnp.where(is_short, short_value, long_value)

        slots = []
        for slot, row in enumerate(rows):
            ctx = jax.tree.map(_pick, row.ctx, first.ctx)
            member = jnp.where(is_short, jnp.minimum(slot, num_rows - 1), 0)
            slots.append(
                SlotRow(
                    state=self.row_state(
                        parity, jnp.where(is_short, self.slot_member(slot), 0)
                    ),
                    q_slot=self.q_slot(q_count + member),
                    q_tok=_pick(row.q_tok, first.q_tok),
                    kv_frontier=ctx.kv_frontier,
                    bottom_right_offset=ctx.bottom_right_offset,
                    kept_start=ctx.kept_start,
                    kept_end=ctx.kept_end,
                    key_start=_pick(
                        row.first_key + (slot - member) * MXU_COLUMNS,
                        self.tile_key(
                            first.first_key, long_pivot, first_long_tile + slot
                        ),
                    ),
                    kv_used_end=ctx.kv_used_end,
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
            finished.is_live > 0,
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
        next_tile: TileSource,
        *,
        parity: int,
        is_short: bool,
    ) -> tuple[jax.Array, ...]:
        """A group's start segment at a static parity: each slot of a group
        of one-tile rows (is_short) starts a row on its own row state, else
        only the first. The finish of finished's first row is staged behind
        it: static parities keep the two groups' state provably apart."""
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
        state = self.pipeline.advance(
            PipelineState(
                probs=probs, tail=self.finished_tail(finished, prev_parity)
            ),
            items,
            self.with_prefetch(
                self.slot_ops(
                    slots, tiles, states=states, row_starts=row_starts
                ),
                items,
                num_tiles=num_slots,
                prefetch_item=MXU_START_PREFETCH_ITEM,
            ),
            next_k_tile=self.staged_k(next_tile),
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
        copy per parity and kind of group; stages finished's first row."""
        num_slots = self.num_slots
        tiles = tuple(
            TileSource(
                slot=self.stream_slot(row.first_tile + tile),
                kv_head=row.kv_head,
                num_valid_rows=slot.kv_used_end - slot.key_start,
            )
            for tile, slot in enumerate(slots)
        )
        first = self.first_finish(finished, finished.parity)
        wait_row_out_slot(
            self.traits, self.out_refs, first, row_out_slot(self.traits, first)
        )
        for tile in range(1, num_slots):
            self.wait_tile(row.first_tile + tile)
        segment = partial(
            self.start_segment,
            finished=finished,
            slots=slots,
            tiles=tiles,
            next_tile=self.next_row_tile(row, num_slots),
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
            first_key=row.first_key,
            num_tiles=row.num_tiles,
            pivot=row.pivot,
            body_end=self.body_end(row.lo, row.hi, row.ctx),
            first_tile=jnp.int32(0),
            kv_head=self.kv_head(group),
            next_kv_head=self.kv_head(group),
        )
        start_tail = self.pipeline.tail_of(
            self.start_items(),
            DiscardedFragmentOps(traits.head_dim_qk),
            has_staged_next_k=True,
        )
        self.replay_tiles(
            replayed,
            self.final_unit(replayed),
            jnp.maximum(row.num_tiles, ROW_START_TILES),
            self.pipeline.free_slots(start_tail, REPLAY_RESULT_SLOTS),
        )

    def stage_replay_tile(
        self, row: PagedRow, tile_number: jax.Array
    ) -> TileSource:
        """The source of a replayed row's tile tile_number, staged
        synchronously into the stream's replay slot."""
        pieces = self.tile_pages(
            row.ctx.page_base, self.key_start(row, tile_number)
        )
        self.start_tile_pieces(row.kv_head, pieces, self.stream.replay_slot)
        self.wait_tile_dmas(self.stream.replay_slot)
        return self.row_tile(row, tile_number, is_masked=True)._replace(
            slot=self.stream.replay_slot
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
        if self.has_checked_units:
            final_unit = self.final_unit(row)
            num_units = jnp.where(final_unit > 0, final_unit + 1, 0)
            self.unit_guard[0] = jnp.int32(0)
            probs = lax.fori_loop(
                0,
                num_units,
                partial(
                    self.checked_unit,
                    row=row,
                    final_unit=final_unit,
                    is_masked=is_masked,
                ),
                probs,
            )
            first_body = num_units * UNIT_TILES + ROW_START_TILES
        return self.plain_bodies(
            probs, row, first_body, row.body_end, is_masked=is_masked
        )

    def plain_bodies(
        self,
        probs: tuple[jax.Array, ...],
        row: PagedRow,
        first_tile: jax.Array | int,
        end_tile: jax.Array | int,
        *,
        is_masked: bool,
    ) -> tuple[jax.Array, ...]:
        """Run row's tiles [first_tile, end_tile) in rolled bodies of
        body_tiles tiles, masked when is_masked."""
        body_tiles = self.tiles.body_tiles
        num_bodies = jnp.maximum(end_tile - first_tile, 0) // body_tiles

        def _body(index, probs):
            first = first_tile + index * body_tiles
            ops_of, tail_ops = self.body_ops(row, first, is_masked=is_masked)
            return self.body(
                probs, row, first, ops_of, tail_ops, is_masked=is_masked
            ).probs

        return lax.fori_loop(0, num_bodies, _body, probs)

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
        *,
        is_masked: bool,
    ) -> PipelineState:
        """Run row's body_tiles tiles from first in one segment that stages
        the next tile's K, with ops_of(row, tiles)'s ops; tail_ops finishes
        the segment before."""
        body_tiles = self.tiles.body_tiles
        items = self.tile_items(body_tiles)
        tiles = tuple(
            self.row_tile(row, first + tile, is_masked=is_masked)
            for tile in range(body_tiles)
        )
        for tile in range(1, body_tiles):
            self.wait_tile(row.first_tile + first + tile)
        next_tile = self.next_row_tile(row, first + body_tiles)
        ops = ops_of(row, tiles)
        assumed = self.pipeline.tail_of(items, tail_ops, has_staged_next_k=True)
        state = self.pipeline.advance(
            PipelineState(probs=probs, tail=assumed),
            items,
            self.with_prefetch(ops, items, num_tiles=body_tiles),
            next_k_tile=self.staged_k(next_tile),
        )
        self.pipeline.check_loop_tail(state.tail, assumed)
        return state

    def checked_unit(
        self,
        unit: jax.Array,
        probs: tuple[jax.Array, ...],
        *,
        row: PagedRow,
        final_unit: jax.Array,
        is_masked: bool,
    ) -> tuple[jax.Array, ...]:
        """Run check unit unit's last body, whose last tile also writes the
        next unit's checkpoint and collects its row sums, then the next
        unit's first body, which finishes that tile. Those sums flag unit: a
        unit before final_unit then replays through the first body's tiles,
        and the last body adds nothing for the items a replay redid."""
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
        probs = self.body(
            probs,
            row,
            last_body,
            lambda row, tiles: unit_end_ops(ops_of(row, tiles)),
            flagged_tail,
            is_masked=is_masked,
        ).probs
        first_body = next_unit * UNIT_TILES
        ops_of, tail_ops = self.body_ops(row, first_body, is_masked=is_masked)
        state = self.body(
            probs,
            row,
            first_body,
            ops_of,
            unit_end_ops(tail_ops),
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

        return state.probs

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
        last_tile = row.num_tiles - 1
        key_start = self.key_start(row, last_tile)
        sees_none = jnp.logical_or(
            key_start >= row.ctx.kv_frontier,
            row.q_tok + self.traits.bq_compute
            <= key_start - row.ctx.bottom_right_offset,
        )
        return jnp.logical_and(last_tile >= first_tile, sees_none)

    def half_tile(
        self, probs: tuple[jax.Array, ...], row: PagedRow
    ) -> tuple[jax.Array, ...]:
        """Run row's last tile on half_tile_items."""
        tile_number = row.num_tiles - 1
        items = self.half_tile_items
        tile = self.row_tile(row, tile_number, is_masked=True)
        next_tile = self.next_row_tile(row, tile_number + 1)
        assumed = self.pipeline.tail_of(
            self.tile_items(),
            self.masked_ops(row, (), (tile_number - 1,)),
            has_staged_next_k=True,
        )
        state = self.pipeline.advance(
            PipelineState(probs=probs, tail=assumed),
            items,
            self.with_prefetch(
                self.masked_ops(row, (tile,), (tile_number,)), items
            ),
            next_k_tile=self.staged_k(next_tile),
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

        def _body(index, probs):
            first = first_tile + index * unit
            for tile in range(1, unit):
                self.wait_tile(row.first_tile + first + tile)
            next_tile = self.next_row_tile(row, first + unit)
            assumed = self.pipeline.tail_of(
                items, _masked(first - unit), has_staged_next_k=True
            )
            state = self.pipeline.advance(
                PipelineState(probs=probs, tail=assumed),
                items,
                self.with_prefetch(_masked(first), items, num_tiles=unit),
                next_k_tile=self.staged_k(next_tile),
            )
            self.pipeline.check_loop_tail(state.tail, assumed)
            return state.probs

        num_units = (self.stream_tiles(row.num_tiles) - first_tile) // unit
        if not (self.is_causal and self.half_tile_items and unit == 1):
            return lax.fori_loop(0, num_units, _body, probs)
        has_half_tile = self.has_half_tile(row, first_tile)
        probs = lax.fori_loop(
            0,
            num_units - has_half_tile.astype(jnp.int32),
            _body,
            probs,
        )
        return lax.cond(
            has_half_tile, partial(self.half_tile, row=row), tuple, probs
        )

    def start_stream(self) -> jax.Array:
        """Point the stream at the grid's first row, describe its first tiles
        and start their prefetches; returns that row's kv head."""
        zero = jnp.int32(0)
        is_past = self.num_grid_rows == 0
        count, pivot, first_key, page_base, kv_head = self.stream_row(
            zero, zero, is_past
        )
        self.stream_state[STREAM_START_POSITION] = zero
        self.store_cursor(
            TileCursor(
                group=zero,
                si=zero,
                is_past=is_past,
                tile=zero,
                count=count,
                pivot=pivot,
                first_key=first_key,
                page_base=page_base,
                kv_head=kv_head,
            )
        )
        stream = self.stream
        for position in range(stream.described_tiles):
            self.describe_tile(position)
        for _ in range(stream.prefetch_tiles):
            self.start_tiles(1)
        return kv_head

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
        kv_head = self.start_stream()
        first_si, num_rows = self.grid_group(zero, self.num_grid_groups == 0)
        self.start_group_q(zero, zero, first_si, num_rows)
        self.wait_tile(zero)
        pipeline = self.pipeline
        discarded = DiscardedFragmentOps(self.traits.head_dim_qk)
        fill = pipeline.advance(
            pipeline.idle_state(),
            self.fill_items,
            discarded,
            next_k_tile=self.staged_k(TileSource(slot=0, kv_head=kv_head)),
        )
        pipeline.check_loop_tail(
            fill.tail,
            pipeline.tail_of(
                self.start_items(), discarded, has_staged_next_k=True
            ),
        )
        # The fill's tiles read stream slot 0, which holds no row's tile yet.
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
            tile=zero,
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
    ) -> PagedRow:
        """The first row, first, of the group carry starts at parity; the
        drain group's row runs a start segment alone from its first tile."""
        return PagedRow(
            ctx=first.ctx,
            q_slot=self.q_slot(carry.q_count),
            q_tok=first.q_tok,
            state=self.row_state(parity, 0),
            first_key=first.first_key,
            num_tiles=jnp.where(is_drain, 1, first.num_tiles),
            pivot=jnp.where(is_drain, 0, first.pivot),
            body_end=jnp.where(
                is_drain,
                self.tiles.start_tiles,
                self.body_end(first.lo, first.hi, first.ctx),
            ),
            first_tile=carry.tile,
            kv_head=self.kv_head(carry.group),
            next_kv_head=self.kv_head(next_group),
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
        is_short = first.num_tiles == 1
        parity = NUM_ROW_PARITIES - 1 - carry.finished.parity
        is_grid_row = jnp.logical_not(is_drain)
        row = self.first_row(carry, first, parity, next_group, is_drain)
        slots = self.slot_rows(
            first_si, num_rows, is_short, parity, carry.q_count, 0, is_grid_row
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
            first.ctx.stream_tiles,
        )
        _, _, last_ctx = self.schedule.row_interval(first_si + num_rows - 1)
        return PagedGroupCarry(
            q_count=carry.q_count + num_rows,
            tile=carry.tile + num_tiles,
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
        self,
        tile: jax.Array,
        probs: tuple[jax.Array, ...],
        finished: FinishedGroup,
    ) -> None:
        """Empty the pipeline after the drain group, whose tiles end before
        stream position tile, and wait out the stream's last prefetches."""
        state = self.pipeline.advance(
            PipelineState(
                probs=probs, tail=self.finished_tail(finished, finished.parity)
            ),
            self.tile_items()[: self.pipeline.delays.pv_pop],
            DiscardedFragmentOps(self.traits.head_dim_qk),
        )
        self.pipeline.drain(state)
        for offset in range(1, self.stream.prefetch_tiles):
            self.wait_tile(tile + offset)

    def finish_grid(self, last: PagedGroupCarry) -> None:
        """After the drain group, whose carry is last: empty the pipeline,
        wait out the output copies and fill the packed padding rows."""
        traits, schedule = self.traits, self.schedule
        self.empty(last.tile, last.probs, last.finished)
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
            has_mode_free_accumulate=zeroes_row_start(
                traits.group_bq // FRAGMENT_ROWS
            )
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
