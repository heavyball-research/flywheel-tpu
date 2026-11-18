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

On a chip with MXU accumulators a head_dim 256 paged build
(paged_uses_mxu_pipeline) runs paged_mxu_fwd_body in place of
kernels/fwd_paged.py's paged_fwd_body.

A row's kv tiles run in segments of whole tiles: its first block's first
tile (the row start) and second tile, then rolled bodies of LOOP_BODY_TILES
tiles over its unmasked blocks and of one tile over its bounds-masked ones.
Every segment stages the K tile of the next one, the next row's first at a
row's end, so the pipeline never drains between rows: a row's last items
finish in the next row's first segment, which also stages that row's finish.
kv streams tile by tile (PagedMxuLoop.start_tiles), and rows alternate
between NUM_ROW_PARITIES parities. After the grid's last row, that row runs
once more as a drain row of two tiles whose own output is dropped, so the
last row finishes as every other row does.

The overflow guard checks a row's unmasked tiles by check units of
UNIT_TILES tiles, whole kv blocks of two loop bodies (PagedMxuLoop.
checked_unit). The items of a unit's last tile also write the row state they
leave into the next unit's checkpoint, that unit's entry state, and collect
their row sums; the next unit's first body finishes them in its first qk_pop
ticks, and those sums decide whether the unit replays, at the end of the
first body: from its checkpoint, through the first body's tiles. The next
body adds nothing for the items the replay redid. A row's final unit, from
its last checked unit on, replays when the row's own sums flag it, as the row
is completed in flight (PagedMxuLoop.complete_row). The row state stays in
place at the row's parity.
"""

import dataclasses
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
from flywheel_tpu.softmax_attention.dma import advance, drain
from flywheel_tpu.softmax_attention.fwd_blocks import (
    SEM_K,
    OutRefs,
    PagedKVRefs,
    PagedRefs,
    QRefs,
    RowFinish,
    carried_finish,
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
    stage_row_finish,
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
    slot_ring_rows,
)
from flywheel_tpu.softmax_attention.schedule import FwdSchedule, ScheduleRow
from flywheel_tpu.softmax_attention.traits import PagedFwdTraits
from flywheel_tpu.softmax_attention.variant import Variant, VariantOperands

# An explicit build streams kv tile by tile: a tile's page DMAs start
# MXU_PREFETCH_TILES tiles before it runs, into a ring of MXU_STREAM_SLOTS
# tile slots, a power of two so a slot is a mask of the stream position (a
# body of LOOP_BODY_TILES tiles starts that many tiles' DMAs while its first
# tile is still read), and a replay stages its tiles one at a time into
# MXU_REPLAY_SLOT, the slot past the ring.
MXU_STREAM_SLOTS = 8
MXU_PREFETCH_TILES = MXU_STREAM_SLOTS - LOOP_BODY_TILES
MXU_REPLAY_SLOT = MXU_STREAM_SLOTS
MXU_STAGING_SLOTS = MXU_REPLAY_SLOT + 1
# A segment starts at most MXU_DESCRIBED_TILES tiles' DMAs, whose operands
# were written before it; a power of two, so an operand set is a mask of the
# stream position.
MXU_DESCRIBED_TILES = LOOP_BODY_TILES
# A DMA start orders the VMEM loads after it, so the item that starts a
# segment's DMAs first loads the q fragments of this many items from it on.
MXU_PRELOADED_FRAGMENTS = 4
# A row's start segment starts its DMAs at this item, early, ahead of the
# finish staged behind its items; other segments start them mid first tile.
MXU_START_PREFETCH_ITEM = 4


def paged_uses_mxu_pipeline(
    *,
    variant: Variant,
    block_sizes: BlockSizes,
    head_dim: int,
    gqa_fold: int,
    page_size: int,
) -> bool:
    """Whether a paged build runs its fragments on the explicit MXU pipeline
    (fwd_mxu.mxu_pipeline_applies, plus what paged_mxu_fwd_body needs).

    A row starts with its first block's first ROW_START_TILES tiles, so a
    block holds at least that many; a rolled body of one tile keeps the result
    slot ring's phase, so a tile's items (gqa_fold * block_q / FRAGMENT_ROWS)
    fill whole rings; a fragment's rows lie in one q compute tile; and a kv
    tile's DMAs move whole pages or whole tiles of one page.
    """
    group_rows = gqa_fold * block_sizes.block_q
    if not mxu_pipeline_applies(
        variant=variant,
        block_sizes=block_sizes,
        head_dim=head_dim,
        head_dim_v=head_dim,
        transposed_pv=False,
        group_rows=group_rows,
    ):
        return False
    return (
        block_sizes.block_kv // MXU_COLUMNS >= ROW_START_TILES
        and group_rows % slot_ring_rows() == 0
        and block_sizes.block_q_compute % FRAGMENT_ROWS == 0
        and (MXU_COLUMNS % page_size == 0 or page_size % MXU_COLUMNS == 0)
    )


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
    of a staged tile, split from its words and multiplied by scale when given;
    an fp8 tile widens to bf16, and V rows past num_valid_rows read as
    zero."""
    # Note (david): an MXU push takes its tile from registers, so the tile is
    # split from the staged words at its push: no plane is stored and read
    # back, and the next tile's K needs only that tile's DMA.
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
    """(folded head, first token) of the FRAGMENT_ROWS q rows of explicit
    fragment fragment; a group's rows run (q tile, folded head, token)."""
    row = fragment * FRAGMENT_ROWS
    tile_row = row % traits.frag_rows
    head = tile_row // traits.bq_compute
    token = row // traits.frag_rows * traits.bq_compute
    return head, aligned_rows(token + tile_row % traits.bq_compute)


def paged_tile_lower_bounds(
    *,
    q_tok: jax.Array,
    kv_frontier: jax.Array,
    bottom_right_offset: jax.Array,
    key_start: jax.Array,
    block_q: int,
    is_causal: bool,
) -> jax.Array:
    """The (1, MXU_COLUMNS) first q row that sees each key of the tile whose
    first key is key_start; a key at or past the frontier gets q_tok +
    block_q, past every row of the block."""
    # Note (david): rows of an earlier sequence in the block's window see the
    # keys up to their own diagonal. Their output is overwritten, and while
    # the bounds rows of the dot_general loop hide every key from them, these
    # keep their anchor and their unmasked blocks' scores on one scale: a row
    # with an unmasked block sees all of that block, and so all of its row's
    # first tile.
    key_ids = key_start + lax.broadcasted_iota(jnp.int32, (1, MXU_COLUMNS), 1)
    if is_causal:
        first_rows = key_ids - bottom_right_offset
    else:
        first_rows = jnp.full(key_ids.shape, q_tok, jnp.int32)
    return jnp.where(key_ids < kv_frontier, first_rows, q_tok + block_q)


def chunk_major_fragments(traits: PagedFwdTraits) -> tuple[int, ...]:
    """A tile's fragments in item order: per q compute tile, its token
    chunks in order, each chunk's folded heads in a row, so the items of a
    chunk share their scores' bounds compare (fragment_q_rows)."""
    num_chunks = traits.bq_compute // FRAGMENT_ROWS
    return tuple(
        (q_tile * traits.gqa_fold + head) * num_chunks + chunk
        for q_tile in range(traits.num_q_compute_tiles)
        for chunk in range(num_chunks)
        for head in range(traits.gqa_fold)
    )


def row_tile_items(
    fragments: tuple[int, ...], num_tiles: int, *, is_row_start: bool
) -> tuple[WorkItem, ...]:
    """The items of num_tiles kv tiles of a row, every fragment per tile in
    the order of fragments."""
    return tuple(
        WorkItem(fragment=fragment, tile=tile, mode=is_row_start)
        for tile in range(num_tiles)
        for fragment in fragments
    )


@dataclasses.dataclass(frozen=True, eq=False)
class PagedFragmentOps:
    """The mxu_pipeline.FragmentOps of a segment of one explicit paged row.

    q_buf[q_slot] holds the row's staged Q, unscaled, and q_tok is its first
    token; the K tiles take the variant's Q scale (operands.q_scale), one
    scalar for every score. row_max, row_sum and o are the (group_bq, ...)
    row state of the row's parity. tiles maps a WorkItem's tile key to its
    staged tile, and lower_bounds (paged_tile_lower_bounds, one per tile
    key) masks the scores; None leaves the segment unmasked. A WorkItem's
    mode is whether its tile starts the row. unit_end names a check unit's
    last tile (UnitCheckpoint).
    """

    traits: PagedFwdTraits
    variant: Variant
    kv_refs: PagedKVRefs
    operands: VariantOperands
    q_buf: jax.Array
    q_slot: jax.Array
    q_tok: jax.Array
    row_max: jax.Array
    row_sum: jax.Array
    o: jax.Array
    tiles: tuple[TileSource, ...] = ()
    lower_bounds: tuple[jax.Array, ...] | None = None
    unit_end: UnitCheckpoint | None = None
    masks: dict[tuple, jax.Array] = dataclasses.field(default_factory=dict)

    def k_tile(self, tile: int) -> jax.Array:
        return paged_tile_rows(
            self.kv_refs, self.tiles[tile], 0, self.operands.q_scale
        )

    def v_tile(self, tile: int) -> jax.Array:
        return paged_tile_rows(self.kv_refs, self.tiles[tile], 1)

    def q_fragment(self, item: WorkItem) -> jax.Array:
        head, token = fragment_q_rows(self.traits, item.fragment)
        if self.traits.gqa_fold > 1:
            q_ref = self.q_buf.at[self.q_slot, head]
        else:
            q_ref = self.q_buf.at[self.q_slot]
        return q_ref[pl.ds(token, FRAGMENT_ROWS), :]

    def visible_keys(self, token: jax.Array | int, tile: int) -> jax.Array:
        """The (FRAGMENT_ROWS, MXU_COLUMNS) keys of tile key tile that the q
        rows from token on see, by lower_bounds."""
        q_ids = (
            self.q_tok
            + token
            + lax.broadcasted_iota(jnp.int32, (FRAGMENT_ROWS, MXU_COLUMNS), 0)
        )
        return q_ids >= self.lower_bounds[tile]

    def masked_scores(self, item: WorkItem, scores: jax.Array) -> jax.Array:
        """item's softcapped scores, bounds-masked under lower_bounds; the
        folded heads of a static token chunk share one compare (masks)."""
        scores = self.variant.transform_logits(scores)
        if self.lower_bounds is None:
            return scores
        _, token = fragment_q_rows(self.traits, item.fragment)
        if not isinstance(token, int):
            return jnp.where(
                self.visible_keys(token, item.tile), scores, DEFAULT_MASK_VALUE
            )
        key = (token, item.tile)
        if key not in self.masks:
            self.masks[key] = self.visible_keys(token, item.tile)
        return jnp.where(self.masks[key], scores, DEFAULT_MASK_VALUE)

    def rows(self, item: WorkItem) -> pl.Slice:
        """The row-state rows of item's fragment."""
        return fragment_rows(item.fragment)

    def is_unit_end(self, item: WorkItem) -> bool:
        return self.unit_end is not None and item.tile == self.unit_end.tile

    def softmax(self, item: WorkItem, scores: jax.Array) -> jax.Array:
        rows = self.rows(item)
        scores = self.masked_scores(item, scores)
        if not self.is_unit_end(item):
            return fold_fragment_scores(
                self.row_max, self.row_sum, rows, scores, is_row_start=item.mode
            )
        assert not item.mode, item
        return fold_unit_end_scores(
            self.row_max, self.row_sum, rows, scores, self.unit_end
        )

    def accumulate(self, item: WorkItem, weighted_values: jax.Array) -> None:
        accumulate_fragment(
            self.o,
            self.rows(item),
            weighted_values,
            is_row_start=item.mode,
            checkpoint_o=self.unit_end.o if self.is_unit_end(item) else None,
        )


@dataclasses.dataclass(frozen=True, eq=False)
class PrefetchingOps:
    """ops whose item preload_keys[0] (fragment, tile) runs start_prefetch
    as it issues its QK, inside the segment's own basic block, where the
    scheduler keeps the work near that item's tick: it first loads the q
    fragments of the items at preload_keys, itself and the ones after it,
    since the DMAs start_prefetch begins order the VMEM loads after them. No
    key starts none."""

    ops: PagedFragmentOps
    preload_keys: tuple[tuple[int, int], ...]
    start_prefetch: Callable[[], None]
    preloaded: dict[tuple[int, int], jax.Array] = dataclasses.field(
        default_factory=dict
    )

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
    """The FragmentOps of the items that fill the pipeline before a kernel's
    first row and empty it after its last: they read zeros and fold
    nothing."""

    head_dim: int

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
    """The cursor of an explicit build's kv tile stream: tile tile of row si
    of group, or of the drain row once is_past; a row of count stream tiles
    whose first kv block, block table row and kv head are lo, page_base and
    kv_head."""

    group: jax.Array
    si: jax.Array
    is_past: jax.Array
    tile: jax.Array
    count: jax.Array
    lo: jax.Array
    page_base: jax.Array
    kv_head: jax.Array


# Note (david): an explicit build keeps its kv tile stream in SMEM, out of the
# scalar registers of its bodies: the stream position of the next tile to
# start, the cursor at the next tile to describe, then the DMA operands (kv
# head, first row in its page, pages) of the next MXU_DESCRIBED_TILES tiles
# to start, by stream position. A page is whole PAGE_SIZE_MULTIPLE rows, so
# a tile takes at most MXU_COLUMNS // PAGE_SIZE_MULTIPLE page DMAs.
STREAM_START_POSITION = 0
STREAM_CURSOR = 1
STREAM_DESCRIPTORS = STREAM_CURSOR + len(TileCursor._fields)
DESCRIPTOR_KV_HEAD = 0
DESCRIPTOR_PAGE_ROW = 1
DESCRIPTOR_PAGES = 2
DESCRIPTOR_WORDS = DESCRIPTOR_PAGES + MXU_COLUMNS // PAGE_SIZE_MULTIPLE
STREAM_STATE_WORDS = STREAM_DESCRIPTORS + MXU_DESCRIBED_TILES * DESCRIPTOR_WORDS


class FinishedRow(NamedTuple):
    """An explicit row whose last tile is still in flight when the next row
    starts: its output finish, its group and Q slot (a replay reads them), and
    what masks its last tile's scores."""

    finish: RowFinish
    group: jax.Array
    q_slot: jax.Array
    kv_frontier: jax.Array
    bottom_right_offset: jax.Array
    last_key_start: jax.Array


class PagedRow(NamedTuple):
    """An explicit row: row si of group, its kv blocks lo..hi and schedule row
    ctx, its staged Q and row-state parity, its live tiles (num_tiles, the
    first num_unmasked of them in unmasked blocks), the stream position of
    its first tile, and its and the next row's kv heads."""

    group: jax.Array
    si: jax.Array
    lo: jax.Array
    hi: jax.Array
    ctx: ScheduleRow
    q_slot: jax.Array
    q_tok: jax.Array
    parity: jax.Array
    num_tiles: jax.Array
    num_unmasked: jax.Array
    first_tile: jax.Array
    kv_head: jax.Array
    next_kv_head: jax.Array


class QPrefetch(NamedTuple):
    """The Q stream after a row step's dma.advance: the row's Q slot and the
    flat grid row whose Q is prefetched (-1: none), its group and row."""

    q_slot: jax.Array
    inflight: jax.Array
    group: jax.Array
    si: jax.Array

    @property
    def is_started(self) -> jax.Array:
        return self.inflight >= 0

    def slot(self, num_stages: int) -> jax.Array:
        """The Q slot the prefetch fills."""
        return lax.rem(self.q_slot + 1, num_stages)


class PagedRowCarry(NamedTuple):
    """The explicit row loop's carry: the Q stream's advance state, the
    stream position of the row's first tile, the row's group and schedule
    row, the pipeline's carried probs, the row before it and that row's
    first token, and the output copies started so far."""

    q_loaded: jax.Array
    q_inflight: jax.Array
    q_slot: jax.Array
    tile: jax.Array
    group: jax.Array
    si: jax.Array
    probs: tuple[jax.Array, ...]
    finished: FinishedRow
    prev_out_tok: jax.Array
    num_out_copies: jax.Array


@dataclasses.dataclass(frozen=True, eq=False)
class PagedMxuLoop:
    """The steps of paged_mxu_fwd_body's row loop, over the build's refs.

    row_max, row_sum and o hold the row state of each of the
    NUM_ROW_PARITIES parities, and checkpoint_sum and checkpoint_o a check
    unit's entry state (UnitCheckpoint). In SMEM, stream_state holds the kv
    tile stream (the STREAM_* words), guard the flag finish_row_sums sets for
    a row and unit_guard the flag of a check unit (checked_unit).
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
    is_causal: bool

    @property
    def num_grid_rows(self) -> jax.Array:
        return self.schedule.num_head_groups * self.schedule.num_rows

    def tile_items(
        self, num_tiles: int = 1, *, is_row_start: bool = False
    ) -> tuple[WorkItem, ...]:
        return row_tile_items(
            chunk_major_fragments(self.traits),
            num_tiles,
            is_row_start=is_row_start,
        )

    def start_items(self) -> tuple[WorkItem, ...]:
        """A row's first two tiles: the row start, then a plain tile."""
        start_items = self.tile_items(is_row_start=True)
        return start_items + tuple(
            item._replace(tile=1) for item in self.tile_items()
        )

    def next_grid_row(
        self, flat_row: jax.Array, group: jax.Array, si: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """(group, row) of flat grid row flat_row + 1, clamped into the grid,
        from (group, si) of flat_row, without a division."""
        is_group_end = si + 1 >= self.schedule.num_rows
        is_last = flat_row + 1 >= self.num_grid_rows
        return (
            jnp.where(jnp.logical_or(is_last, ~is_group_end), group, group + 1),
            jnp.where(is_last, si, jnp.where(is_group_end, 0, si + 1)),
        )

    def kv_head(self, group: jax.Array) -> jax.Array:
        return (
            group * self.traits.group_heads // self.schedule.q_heads_per_kv_head
        )

    def num_row_tiles(
        self, lo: jax.Array, hi: jax.Array, ctx: ScheduleRow
    ) -> jax.Array:
        """The live kv tiles of the row of ctx over blocks lo..hi."""
        return (hi - lo) * self.traits.num_k_compute_tiles + (
            self.schedule.num_live_tiles(hi, ctx)
        )

    def num_unmasked_tiles(
        self, lo: jax.Array, hi: jax.Array, ctx: ScheduleRow
    ) -> jax.Array:
        """The tiles of the row of ctx in its unmasked blocks of lo..hi."""
        return (
            jnp.clip(ctx.mask_from - lo, 0, hi - lo + 1)
            * self.traits.num_k_compute_tiles
        )

    def final_unit(self, num_unmasked: jax.Array | int) -> jax.Array | int:
        """The first check unit of a row's final unit, given its tiles in
        unmasked blocks: the units before it are checked (checked_unit), and
        its first body ends the last first body, whose unit is fully unmasked
        too; 0 when the row has no checked unit."""
        return first_final_unit(num_unmasked // UNIT_TILES)

    @property
    def has_checked_units(self) -> bool:
        """Whether a check unit holds whole kv blocks and a row of the build's
        capacity can hold a checked unit."""
        paged = self.traits.paged
        max_row_tiles = paged.pages_per_seq * paged.page_size // MXU_COLUMNS
        return (
            UNIT_TILES % self.traits.num_k_compute_tiles == 0
            and self.final_unit(max_row_tiles) > 0
        )

    def key_start(self, row: PagedRow, tile_number: jax.Array) -> jax.Array:
        """The first key of row's live tile tile_number."""
        return row.lo * self.traits.bkv + tile_number * MXU_COLUMNS

    def stream_row(
        self, group: jax.Array, si: jax.Array, is_past: jax.Array
    ) -> tuple[jax.Array, ...]:
        """(stream tiles, lo, page_base, kv_head) of row si of group, whose
        stream tiles are its live tiles, at least ROW_START_TILES; is_past
        gives the drain row, the grid's last row with ROW_START_TILES
        tiles."""
        schedule = self.schedule
        group = jnp.where(is_past, schedule.num_head_groups - 1, group)
        si = jnp.where(is_past, jnp.maximum(schedule.num_rows - 1, 0), si)
        lo, hi, ctx = schedule.row_interval(si)
        num_tiles = jnp.where(
            is_past, ROW_START_TILES, self.num_row_tiles(lo, hi, ctx)
        )
        return (
            jnp.maximum(num_tiles, ROW_START_TILES),
            lo,
            ctx.page_base,
            self.kv_head(group),
        )

    def stream_slot(self, position: jax.Array) -> jax.Array:
        """The stream slot of the tile at stream position position."""
        return jnp.bitwise_and(position, MXU_STREAM_SLOTS - 1)

    @property
    def tile_piece_rows(self) -> int:
        """The rows of one of a kv tile's DMAs: a whole page, or a tile of a
        larger page."""
        return min(self.traits.paged.page_size, MXU_COLUMNS)

    def tile_pages(
        self, page_base: jax.Array, key_start: jax.Array
    ) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        """(first row in its page, pages of its DMAs) of the tile whose first
        key is key_start, in the pages the block table row at page_base
        names."""
        # Note (david): a page past the row's pages reads through the clamped
        # table entry, as start_paged_kv_block's do; its keys are past
        # seqused_k, so they are masked and their V rows zeroed
        # (paged_tile_rows).
        page_size = self.traits.paged.page_size
        pages = tuple(
            jnp.clip(
                self.schedule.kv_page(
                    page_base, 0, (key_start + piece_key) // page_size
                ),
                0,
                self.kv_refs.num_cache_pages - 1,
            )
            for piece_key in range(0, MXU_COLUMNS, self.tile_piece_rows)
        )
        if page_size > MXU_COLUMNS:
            return key_start % page_size, pages
        return jnp.int32(0), pages

    def start_tile_pieces(
        self,
        kv_head: jax.Array,
        page_row: jax.Array,
        pages: Sequence[jax.Array],
        slot: jax.Array | int,
    ) -> None:
        """Start the DMAs of kv_head's word rows of a tile, from row page_row
        of pages on, into staging slot slot."""
        # Note (david): every DMA moves whole pages or a whole tile, so its
        # size and its wait are static.
        kv_refs, piece_rows = self.kv_refs, self.tile_piece_rows
        if self.traits.paged.page_size > MXU_COLUMNS:
            first_row = pl.multiple_of(page_row, MXU_COLUMNS)
        else:
            first_row = 0
        for piece, page in enumerate(pages):
            pltpu.make_async_copy(
                paged_kv_source(kv_refs, page, piece_rows, kv_head, first_row),
                paged_buffer_tile(
                    kv_refs, slot, piece * piece_rows, piece_rows
                ),
                kv_refs.sems.at[SEM_K, slot],
            ).start()

    def start_tile_dmas(
        self,
        page_base: jax.Array,
        kv_head: jax.Array,
        key_start: jax.Array,
        slot: jax.Array | int,
    ) -> None:
        """Start the DMAs of kv_head's word rows of the tile whose first key
        is key_start, in the pages the block table row at page_base names,
        into staging slot slot."""
        page_row, pages = self.tile_pages(page_base, key_start)
        self.start_tile_pieces(kv_head, page_row, pages, slot)

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
        count, lo, page_base, kv_head = self.stream_row(group, si, is_past)
        next_row = TileCursor(
            group=group,
            si=si,
            is_past=is_past.astype(jnp.int32),
            tile=jnp.int32(0),
            count=count,
            lo=lo,
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
        """The first SMEM word of the DMA operands of the tile at stream
        position position."""
        return (
            STREAM_DESCRIPTORS
            + jnp.bitwise_and(position, MXU_DESCRIBED_TILES - 1)
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
        page_row, pages = self.tile_pages(
            cursor.page_base,
            cursor.lo * self.traits.bkv + cursor.tile * MXU_COLUMNS,
        )
        first_word = self.descriptor(position)
        state[first_word + DESCRIPTOR_KV_HEAD] = cursor.kv_head
        state[first_word + DESCRIPTOR_PAGE_ROW] = page_row
        for piece, page in enumerate(pages):
            state[first_word + DESCRIPTOR_PAGES + piece] = page
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
                state[first_word + DESCRIPTOR_PAGE_ROW],
                [
                    state[first_word + DESCRIPTOR_PAGES + piece]
                    for piece in range(num_pieces)
                ],
                self.stream_slot(position + offset),
            )
        state[STREAM_START_POSITION] = position + num_tiles
        for offset in range(num_tiles):
            self.describe_tile(position + MXU_DESCRIBED_TILES + offset)

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
        """ops whose segment of items, num_tiles tiles of them, starts one
        prefetch per tile, all at its first tile's item prefetch_item (its
        middle one when None): every DMA start orders the segment's memory
        accesses around it, so a segment has one such point
        (PrefetchingOps)."""
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
        """The source of row's live tile tile_number; is_masked zeroes its V
        rows past the row's valid keys."""
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
        """The source of row's live tile tile_number, or of the next row's
        first tile when tile_number passes row's live tiles; waits for its
        DMAs."""
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

    def lower_bounds(
        self,
        q_tok: jax.Array,
        kv_frontier: jax.Array,
        bottom_right_offset: jax.Array,
        key_start: jax.Array,
    ) -> jax.Array:
        return paged_tile_lower_bounds(
            q_tok=q_tok,
            kv_frontier=kv_frontier,
            bottom_right_offset=bottom_right_offset,
            key_start=key_start,
            block_q=self.traits.bq,
            is_causal=self.is_causal,
        )

    def ops(
        self,
        q_slot: jax.Array,
        q_tok: jax.Array,
        parity: jax.Array | int,
        tiles: tuple[TileSource, ...] = (),
        lower_bounds: tuple[jax.Array, ...] | None = None,
    ) -> PagedFragmentOps:
        return PagedFragmentOps(
            traits=self.traits,
            variant=self.variant,
            kv_refs=self.kv_refs,
            operands=self.operands,
            q_buf=self.q_refs.buf,
            q_slot=q_slot,
            q_tok=q_tok,
            row_max=self.row_max.at[parity],
            row_sum=self.row_sum.at[parity],
            o=self.o.at[parity],
            tiles=tiles,
            lower_bounds=lower_bounds,
        )

    def row_ops(
        self, row: PagedRow, tiles: tuple[TileSource, ...] = ()
    ) -> PagedFragmentOps:
        """The unmasked ops of row's items of tiles."""
        return self.ops(row.q_slot, row.q_tok, row.parity, tiles)

    def unit_end_ops(
        self, ops: PagedFragmentOps, slot: jax.Array, largest: LargestSums
    ) -> PagedFragmentOps:
        """ops whose items of a body's last tile key, a check unit's last
        tile, also write the row state they leave into checkpoint slot slot
        and fold their row sums into largest (UnitCheckpoint)."""
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
        """The ops of row's live tiles tile_numbers (one per tile key), every
        score bounds-masked; tiles is empty for a tail's ops."""
        bounds = tuple(
            self.lower_bounds(
                row.q_tok,
                row.ctx.kv_frontier,
                row.ctx.bottom_right_offset,
                self.key_start(row, tile_number),
            )
            for tile_number in tile_numbers
        )
        return self.ops(row.q_slot, row.q_tok, row.parity, tiles, bounds)

    def finished_row(
        self,
        row: PagedRow,
        is_live: jax.Array,
        prev_out_tok: jax.Array,
        out_count: jax.Array,
    ) -> FinishedRow:
        """row once its tiles ran: the last of them is tile 1 when row has a
        single live tile."""
        last_tile = jnp.maximum(row.num_tiles, ROW_START_TILES) - 1
        return FinishedRow(
            finish=RowFinish(
                is_live=is_live,
                si=row.si,
                q_tok=row.q_tok,
                base_head=row.group * self.traits.group_heads,
                num_blend_rows=self.schedule.blend_start(row.ctx) - row.q_tok,
                prev_out_tok=prev_out_tok,
                out_count=out_count,
                parity=row.parity,
            ),
            group=row.group,
            q_slot=row.q_slot,
            kv_frontier=row.ctx.kv_frontier,
            bottom_right_offset=row.ctx.bottom_right_offset,
            last_key_start=self.key_start(row, last_tile),
        )

    def finished_row_tail(self, finished: FinishedRow) -> Tail:
        """The tail finished's last tile leaves for the next row's first
        segment, masked as that tile."""
        finish = finished.finish
        bounds = self.lower_bounds(
            finish.q_tok,
            finished.kv_frontier,
            finished.bottom_right_offset,
            finished.last_key_start,
        )
        ops = self.ops(
            finished.q_slot, finish.q_tok, finish.parity, lower_bounds=(bounds,)
        )
        return self.pipeline.tail_of(
            self.tile_items(), ops, has_staged_next_k=True
        )

    def stage_row_finish(self, finish: RowFinish) -> None:
        """Stage the lse and normalized output of finish's row from its row
        state at finish.parity, a static parity, once its last softmax has
        run and its output stage slot is free (wait_row_out_slot): its lane
        partial sums reduce first, setting its guard flag."""
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
        finished: FinishedRow,
        row: PagedRow,
        tiles: tuple[TileSource, ...],
        next_tile: TileSource,
        *,
        parity: int,
    ) -> tuple[jax.Array, ...]:
        """row's first two tiles in one segment at a static parity, after
        finished's tail, with finished's finish staged behind it
        (stage_row_finish): static parities keep the two rows' state provably
        apart, so the finish overlaps the segment."""
        row = row._replace(parity=parity)
        finished = finished._replace(
            finish=finished.finish._replace(
                parity=NUM_ROW_PARITIES - 1 - parity
            )
        )
        items = self.start_items()
        state = self.pipeline.advance(
            PipelineState(probs=probs, tail=self.finished_row_tail(finished)),
            items,
            self.with_prefetch(
                self.masked_ops(row, tiles, tuple(range(ROW_START_TILES))),
                items,
                num_tiles=ROW_START_TILES,
                prefetch_item=MXU_START_PREFETCH_ITEM,
            ),
            next_k_tile=self.staged_k(next_tile),
        )
        self.pipeline.check_loop_tail(
            state.tail,
            self.pipeline.tail_of(
                self.tile_items(LOOP_BODY_TILES),
                state.tail.ops,
                has_staged_next_k=True,
            ),
        )
        self.stage_row_finish(finished.finish)
        return state.probs

    def start_row(
        self,
        probs: tuple[jax.Array, ...],
        finished: FinishedRow,
        row: PagedRow,
    ) -> tuple[jax.Array, ...]:
        """Run row's first two tiles (start_segment, one copy per parity),
        finishing finished's row."""
        tiles = tuple(
            self.row_tile(row, tile, is_masked=True)
            for tile in range(ROW_START_TILES)
        )
        wait_row_out_slot(
            self.traits,
            self.out_refs,
            finished.finish,
            row_out_slot(self.traits, finished.finish),
        )
        self.wait_tile(row.first_tile + 1)
        segment = partial(
            self.start_segment,
            finished=finished,
            row=row,
            tiles=tiles,
            next_tile=self.next_row_tile(row, ROW_START_TILES),
        )
        return lax.cond(
            row.parity == 0,
            partial(segment, parity=0),
            partial(segment, parity=1),
            probs,
        )

    def complete_row(self, finished: FinishedRow, prefetch: QPrefetch) -> None:
        """Write finished's row out once the next row's start segment staged
        it. A row its row sums flag first replays its final unit
        (replay_row_end) and is staged again; prefetch is the next row's Q
        stream."""
        traits = self.traits
        finish = finished.finish

        @pl.when(finish.is_live > 0)
        def _complete():
            @pl.when(self.guard[0] > 0)
            def _replay():
                self.replay_row_end(finished, prefetch)
                self.stage_row_finish(finish)

            if self.variant.return_lse:
                copy_row_lse(
                    traits,
                    self.out_refs,
                    finished.group,
                    finish,
                    row_out_slot(traits, finish),
                    self.schedule.packed_q_end,
                )
            copy_row_out(
                traits,
                self.out_refs,
                finish,
                row_out_slot(traits, finish),
                self.schedule.packed_q_end,
            )

    def replay_row_end(
        self, finished: FinishedRow, prefetch: QPrefetch
    ) -> None:
        """Replay finished's final unit (final_unit) into its parity's row
        state from the unit's checkpoint, or from the row start, between the
        next row's start segment and its next segment, which writes no
        checkpoint. The row's Q borrows the slot of the next row's Q prefetch
        (borrow_q_slot)."""
        q_slot = self.borrow_q_slot(finished, prefetch)
        row = self.replayed_row(finished, q_slot)
        start_tail = self.pipeline.tail_of(
            self.start_items(), DiscardedFragmentOps(self.traits.head_dim_qk)
        )
        self.replay_tiles(
            row,
            self.final_unit(row.num_unmasked),
            jnp.maximum(row.num_tiles, ROW_START_TILES),
            self.pipeline.free_slots(start_tail, REPLAY_RESULT_SLOTS),
        )
        self.return_q_slot(prefetch, q_slot)

    def borrow_q_slot(
        self, finished: FinishedRow, prefetch: QPrefetch
    ) -> jax.Array:
        """Stage finished's Q into the slot of the next row's Q prefetch, once
        that prefetch has landed, and return the slot: the prefetch fills the
        slot finished's Q was in."""
        traits, q_refs = self.traits, self.q_refs
        slot = prefetch.slot(traits.num_stages)

        @pl.when(prefetch.is_started)
        def _wait_prefetch():
            wait_rotated_q_row(traits, q_refs, slot)

        finish = finished.finish
        start_q_row(traits, q_refs, finish.base_head, finish.q_tok, slot)
        wait_rotated_q_row(traits, q_refs, slot)
        return slot

    def return_q_slot(self, prefetch: QPrefetch, slot: jax.Array) -> None:
        """Start the next row's Q prefetch again into slot, the one
        borrow_q_slot took, so dma.advance's next step waits for it as
        before."""
        traits, schedule = self.traits, self.schedule

        @pl.when(prefetch.is_started)
        def _restart_prefetch():
            _, _, ctx = schedule.row_interval(prefetch.si)
            start_q_row(
                traits,
                self.q_refs,
                prefetch.group * traits.group_heads,
                schedule.q_token(ctx),
                slot,
            )

    def replayed_row(
        self, finished: FinishedRow, q_slot: jax.Array
    ) -> PagedRow:
        """finished's row as a replay runs it, its Q in staging slot q_slot."""
        finish = finished.finish
        lo, hi, ctx = self.schedule.row_interval(finish.si)
        return PagedRow(
            group=finished.group,
            si=finish.si,
            lo=lo,
            hi=hi,
            ctx=ctx,
            q_slot=q_slot,
            q_tok=finish.q_tok,
            parity=finish.parity,
            num_tiles=self.num_row_tiles(lo, hi, ctx),
            num_unmasked=self.num_unmasked_tiles(lo, hi, ctx),
            first_tile=jnp.int32(0),
            kv_head=self.kv_head(finished.group),
            next_kv_head=self.kv_head(finished.group),
        )

    def stage_replay_tile(
        self, row: PagedRow, tile_number: jax.Array
    ) -> TileSource:
        """The source of a replayed row's tile tile_number, staged
        synchronously into MXU_REPLAY_SLOT, apart from the stream's
        prefetches in flight."""
        self.start_tile_dmas(
            row.ctx.page_base,
            row.kv_head,
            self.key_start(row, tile_number),
            MXU_REPLAY_SLOT,
        )
        self.wait_tile_dmas(MXU_REPLAY_SLOT)
        return self.row_tile(row, tile_number, is_masked=True)._replace(
            slot=MXU_REPLAY_SLOT
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
        checkpoint. Each tile is staged again (stage_replay_tile) and every
        fragment meets it bounds-masked on QK_MXU through slots
        (replay_fragment), leaving lane partial sums. Check unit
        checkpoint_unit's checkpoint is written again as its first tile
        starts, and rebased at the end to the anchor the tiles after it
        raised (rebase_checkpoint), so a checkpoint always holds the row
        state on the row's current anchor."""
        num_fragments = self.traits.group_bq // FRAGMENT_ROWS
        checkpoints = self.row_checkpoints(row.parity)

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

    def replay_fragment(
        self,
        ops: PagedFragmentOps,
        fragment: jax.Array,
        is_row_start: jax.Array,
        slots: tuple[jax.Array, ...],
    ) -> None:
        """Fold fragment fragment against the one tile of ops into its rows of
        the row state with the exact rebasing softmax
        (fwd_mxu.fold_replayed_fragment), both matmuls on QK_MXU through
        slots, the (scores, probs @ V) result slots."""
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

    def unmasked_tiles(
        self, probs: tuple[jax.Array, ...], row: PagedRow
    ) -> tuple[jax.Array, ...]:
        """Run row's unmasked tiles after its first two: its check units
        (checked_unit), up to the first body of the unit after its final
        unit, then rolled bodies of LOOP_BODY_TILES tiles."""
        first_tile = ROW_START_TILES
        if self.has_checked_units:
            final_unit = self.final_unit(row.num_unmasked)
            num_units = jnp.where(final_unit > 0, final_unit + 1, 0)
            self.unit_guard[0] = jnp.int32(0)
            probs = lax.fori_loop(
                0,
                num_units,
                partial(self.checked_unit, row=row, final_unit=final_unit),
                probs,
            )
            first_tile = num_units * UNIT_TILES + LOOP_BODY_TILES
        return self.plain_bodies(probs, row, first_tile, row.num_unmasked)

    def plain_bodies(
        self,
        probs: tuple[jax.Array, ...],
        row: PagedRow,
        first_tile: jax.Array | int,
        end_tile: jax.Array | int,
    ) -> tuple[jax.Array, ...]:
        """Run row's unmasked tiles [first_tile, end_tile) in rolled bodies of
        LOOP_BODY_TILES tiles."""
        num_bodies = jnp.maximum(end_tile - first_tile, 0) // LOOP_BODY_TILES

        def _body(index, probs):
            first = first_tile + index * LOOP_BODY_TILES
            return self.body(probs, row, first, self.row_ops).probs

        return lax.fori_loop(0, num_bodies, _body, probs)

    def body(
        self,
        probs: tuple[jax.Array, ...],
        row: PagedRow,
        first: jax.Array | int,
        ops_of: Callable[..., PagedFragmentOps],
        tail_ops: FlaggedTailOps | PagedFragmentOps | None = None,
    ) -> PipelineState:
        """Run row's LOOP_BODY_TILES unmasked tiles from first in one segment
        that stages the next tile's K, with ops_of(row, tiles)'s ops; tail_ops
        (the segment's own ops when None) finishes the segment before."""
        items = self.tile_items(LOOP_BODY_TILES)
        tiles = tuple(
            self.row_tile(row, first + tile, is_masked=False)
            for tile in range(LOOP_BODY_TILES)
        )
        for tile in range(1, LOOP_BODY_TILES):
            self.wait_tile(row.first_tile + first + tile)
        next_tile = self.next_row_tile(row, first + LOOP_BODY_TILES)
        ops = ops_of(row, tiles)
        assumed = self.pipeline.tail_of(
            items, ops if tail_ops is None else tail_ops, has_staged_next_k=True
        )
        state = self.pipeline.advance(
            PipelineState(probs=probs, tail=assumed),
            items,
            self.with_prefetch(ops, items, num_tiles=LOOP_BODY_TILES),
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
    ) -> tuple[jax.Array, ...]:
        """Run row's check unit unit's last body, whose last tile also writes
        the next unit's entry state into that unit's checkpoint and collects
        its row sums (unit_end_ops), then the next unit's first body, which
        finishes that tile: its row sums flag unit, and a unit before
        final_unit replays at the first body's end (replay_tiles), through
        the first body's tiles. The last body adds nothing for the items of
        the first body before it that a replay redid (FlaggedTailOps)."""
        next_unit = unit + 1
        largest = LargestSums()
        unit_end_ops = partial(
            self.unit_end_ops,
            slot=lax.rem(next_unit, NUM_CHECKPOINTS),
            largest=largest,
        )
        flagged_tail = FlaggedTailOps(
            ops=self.row_ops(row),
            is_flagged=self.unit_guard[0] > 0,
            plain_mode=False,
        )
        probs = self.body(
            probs,
            row,
            unit * UNIT_TILES + LOOP_BODY_TILES,
            lambda row, tiles: unit_end_ops(self.row_ops(row, tiles)),
            flagged_tail,
        ).probs
        state = self.body(
            probs,
            row,
            next_unit * UNIT_TILES,
            self.row_ops,
            unit_end_ops(self.row_ops(row)),
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
        when they make a segment that keeps the result slot ring's phase;
        else none."""
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
        """Whether row's last tile is a bounds-masked one (from first_tile on)
        whose first q compute tile's rows see none of its keys."""
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
        """Run row's bounds-masked tiles after its unmasked ones (and its
        first two) in rolled bodies of one tile; a causal row's last tile
        skips the fragments of its first q compute tile when they see none
        of its keys (half_tile)."""
        items = self.tile_items()

        def _body(tile_number, probs):
            tile = self.row_tile(row, tile_number, is_masked=True)
            next_tile = self.next_row_tile(row, tile_number + 1)
            assumed = self.pipeline.tail_of(
                items,
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
            self.pipeline.check_loop_tail(state.tail, assumed)
            return state.probs

        first_tile = jnp.maximum(ROW_START_TILES, row.num_unmasked)
        if not (self.is_causal and self.half_tile_items):
            return lax.fori_loop(first_tile, row.num_tiles, _body, probs)
        has_half_tile = self.has_half_tile(row, first_tile)
        probs = lax.fori_loop(
            first_tile,
            row.num_tiles - has_half_tile.astype(jnp.int32),
            _body,
            probs,
        )
        return lax.cond(
            has_half_tile, partial(self.half_tile, row=row), tuple, probs
        )

    def run_row(
        self,
        probs: tuple[jax.Array, ...],
        finished: FinishedRow,
        row: PagedRow,
        prefetch: QPrefetch,
    ) -> tuple[jax.Array, ...]:
        """Run row's tiles, finishing finished's row in row's first two and
        completing it after them (complete_row); prefetch is the Q stream
        after row's step."""
        probs = self.start_row(probs, finished, row)
        self.complete_row(finished, prefetch)
        probs = self.unmasked_tiles(probs, row)
        return self.masked_tiles(probs, row)

    def start_stream(self) -> jax.Array:
        """Point the stream at the grid's first row, describe its first
        MXU_DESCRIBED_TILES tiles and start its first MXU_PREFETCH_TILES;
        returns that row's kv head."""
        zero = jnp.int32(0)
        is_past = self.num_grid_rows == 0
        count, lo, page_base, kv_head = self.stream_row(zero, zero, is_past)
        self.stream_state[STREAM_START_POSITION] = zero
        self.store_cursor(
            TileCursor(
                group=zero,
                si=zero,
                is_past=is_past,
                tile=zero,
                count=count,
                lo=lo,
                page_base=page_base,
                kv_head=kv_head,
            )
        )
        for position in range(MXU_DESCRIBED_TILES):
            self.describe_tile(position)
        for _ in range(MXU_PREFETCH_TILES):
            self.start_tiles(1)
        return kv_head

    def fill(self) -> PagedRowCarry:
        """Start the stream and fill the pipeline before the grid's first
        row; returns the row loop's initial carry.

        The fill is one slot ring of a tile's items that read zeros, so the
        first row's first segment finishes it as it finishes any row's last
        tile; its parity is the one the first row does not take.
        """
        zero = jnp.int32(0)
        kv_head = self.start_stream()
        self.wait_tile(zero)
        pipeline = self.pipeline
        discarded = DiscardedFragmentOps(self.traits.head_dim_qk)
        fill = pipeline.advance(
            pipeline.idle_state(),
            self.tile_items()[-pipeline.num_slots :],
            discarded,
            next_k_tile=self.staged_k(TileSource(slot=0, kv_head=kv_head)),
        )
        pipeline.check_loop_tail(
            fill.tail,
            pipeline.tail_of(
                self.tile_items(), discarded, has_staged_next_k=True
            ),
        )
        finished = FinishedRow(
            finish=RowFinish(*([zero] * len(RowFinish._fields)))._replace(
                parity=jnp.int32(ENTRY_PARITY)
            ),
            group=zero,
            q_slot=zero,
            kv_frontier=zero,
            bottom_right_offset=zero,
            last_key_start=zero,
        )
        minus_one = jnp.int32(-1)
        return PagedRowCarry(
            q_loaded=minus_one,
            q_inflight=minus_one,
            q_slot=zero,
            tile=zero,
            group=zero,
            si=zero,
            probs=fill.probs,
            finished=finished,
            prev_out_tok=zero,
            num_out_copies=zero,
        )

    def advance_q_row(
        self,
        carry: PagedRowCarry,
        q_tok: jax.Array,
        next_group: jax.Array,
        next_si: jax.Array,
        next_ctx: ScheduleRow,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """dma.advance for the row of carry, whose first token is q_tok, and
        the next one: (q_loaded, q_inflight, q_slot)."""
        traits, schedule = self.traits, self.schedule
        group_heads, num_rows = traits.group_heads, schedule.num_rows
        return advance(
            carry.group * num_rows + carry.si,
            next_group * num_rows + next_si,
            lambda slot: start_q_row(
                traits, self.q_refs, carry.group * group_heads, q_tok, slot
            ),
            lambda slot: start_q_row(
                traits,
                self.q_refs,
                next_group * group_heads,
                schedule.q_token(next_ctx),
                slot,
            ),
            # The K tiles take the Q scale, so the staged Q is only rotated.
            partial(wait_rotated_q_row, traits, self.q_refs),
            carry.q_loaded,
            carry.q_inflight,
            carry.q_slot,
            traits.num_stages,
        )

    def grid_mxu_row(
        self,
        carry: PagedRowCarry,
        interval: tuple[jax.Array, jax.Array, ScheduleRow],
        q_tok: jax.Array,
        q_slot: jax.Array,
        is_drain_row: jax.Array,
        next_group: jax.Array,
    ) -> PagedRow:
        """The row of carry over its schedule row interval (lo, hi, ctx),
        the drain row's single tile past the grid's last."""
        lo, hi, ctx = interval
        num_unmasked = self.num_unmasked_tiles(lo, hi, ctx)
        return PagedRow(
            group=carry.group,
            si=carry.si,
            lo=lo,
            hi=hi,
            ctx=ctx,
            q_slot=q_slot,
            q_tok=q_tok,
            parity=1 - carry.finished.finish.parity,
            num_tiles=jnp.where(
                is_drain_row, 1, self.num_row_tiles(lo, hi, ctx)
            ),
            num_unmasked=jnp.where(is_drain_row, 0, num_unmasked),
            first_tile=carry.tile,
            kv_head=self.kv_head(carry.group),
            next_kv_head=self.kv_head(next_group),
        )

    def row_step(self, index: jax.Array, carry: PagedRowCarry) -> PagedRowCarry:
        """Run flat grid row index, the drain row past the grid's last."""
        carry = carry._replace(
            finished=carry.finished._replace(
                finish=carried_finish(carry.finished.finish)
            )
        )
        schedule = self.schedule
        is_drain_row = index == self.num_grid_rows
        next_group, next_si = self.next_grid_row(index, carry.group, carry.si)
        interval = schedule.row_interval(carry.si)
        _, _, next_ctx = schedule.row_interval(next_si)
        q_tok = schedule.q_token(interval[2])
        q_loaded, q_inflight, q_slot = self.advance_q_row(
            carry, q_tok, next_group, next_si, next_ctx
        )
        row = self.grid_mxu_row(
            carry, interval, q_tok, q_slot, is_drain_row, next_group
        )
        probs = self.run_row(
            carry.probs,
            carry.finished,
            row,
            QPrefetch(
                q_slot=q_slot, inflight=q_inflight, group=next_group, si=next_si
            ),
        )
        is_live = jnp.logical_not(is_drain_row).astype(jnp.int32)
        return PagedRowCarry(
            q_loaded=q_loaded,
            q_inflight=q_inflight,
            q_slot=q_slot,
            tile=carry.tile + jnp.maximum(row.num_tiles, ROW_START_TILES),
            group=next_group,
            si=next_si,
            probs=probs,
            finished=self.finished_row(
                row, is_live, carry.prev_out_tok, carry.num_out_copies
            ),
            prev_out_tok=q_tok,
            num_out_copies=carry.num_out_copies + is_live,
        )

    def empty(
        self,
        tile: jax.Array,
        probs: tuple[jax.Array, ...],
        finished: FinishedRow,
    ) -> None:
        """Empty the pipeline after the drain row, whose tiles end before
        stream position tile: items that read zeros load its staged K tile
        and finish its tail, then the pipeline drains and the stream's last
        prefetches land."""
        state = self.pipeline.advance(
            PipelineState(probs=probs, tail=self.finished_row_tail(finished)),
            self.tile_items()[: self.pipeline.delays.pv_pop],
            DiscardedFragmentOps(self.traits.head_dim_qk),
        )
        self.pipeline.drain(state)
        for offset in range(1, MXU_PREFETCH_TILES):
            self.wait_tile(tile + offset)

    def finish_grid(self, last: PagedRowCarry) -> None:
        """After the drain row, whose carry is last: empty the pipeline, wait
        out the output copies and fill the packed padding rows."""
        traits, schedule = self.traits, self.schedule
        self.empty(
            last.tile,
            last.probs,
            last.finished._replace(finish=carried_finish(last.finished.finish)),
        )
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
    is_causal: bool,
) -> None:
    """paged_fwd_body on the explicit MXU pipeline: every row of the grid,
    then a drain row. The arguments are paged_fwd_body's; is_causal is the
    schedule's right bound."""
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
        pipeline=MxuPipeline.allocate(),
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
        is_causal=is_causal,
    )
    last = lax.fori_loop(0, loop.num_grid_rows + 1, loop.row_step, loop.fill())
    loop.finish_grid(last)
