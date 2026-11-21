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

"""Dense forward rows on the explicit MXU pipeline.

A dense build with traits.uses_mxu_pipeline runs its rows through DenseMxuLoop:
a row is one pipeline over its kv blocks, from its first tile to its drain, and
keeps its row state at MXU_ROW_PARITY.

A row's blocks are the left edge (a left window bound hides keys), the interior
(every row sees every key) and the right edge. A build with a static mask
geometry lists each edge block's live items at compile time (static_row_plan);
any other runs them as masked loop bodies.

The overflow guard checks a row by check units of one kv block: a flagged unit
replays from its checkpoint, a row's final unit after the drain.
"""

import dataclasses
import functools
from collections.abc import Callable, Sequence
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl

from flywheel_tpu.core.tpu import MXU_COLUMNS
from flywheel_tpu.softmax_attention.block_stream import (
    BlockDesc,
    DenseBlockStream,
    RowVisits,
    select_desc,
)
from flywheel_tpu.softmax_attention.config import BlockSizes
from flywheel_tpu.softmax_attention.fwd_blocks import (
    NUM_KV_STAGES,
    KVRefs,
    OutRefs,
    QRefs,
    RowFinish,
    RowStateRefs,
    advance_dense_q_row,
    finish_row_lse,
    finish_row_lse_and_sinks,
    head_sinks,
    o_copy,
    row_out_slot,
    wait_rotated_kv_block,
    wait_rotated_q_row,
    wait_row_out_slot,
)
from flywheel_tpu.softmax_attention.fwd_mxu import (
    FINAL_UNIT_SPAN,
    LOOP_BODY_TILES,
    NUM_CHECKPOINTS,
    REPLAY_RESULT_SLOTS,
    UNBOUNDED_ROW,
    FlaggedTailOps,
    ItemClass,
    LargestSums,
    MaskGeometry,
    RowCheckpoints,
    TileBounds,
    UnitCheckpoint,
    accumulate_fragment,
    across_lanes,
    finish_row_sums,
    fold_fragment_scores,
    fold_replayed_fragment,
    fragment_rows,
    guard_flag,
    item_class,
    masked_logits,
    mxu_pipeline_applies,
    replay_matmul,
    scaled_logits,
    tile_bounds,
    zeroes_row_start,
)
from flywheel_tpu.softmax_attention.mxu_pipeline import (
    FRAGMENT_ROWS,
    MxuPipeline,
    PipelineDelays,
    PipelineState,
    Tail,
    WorkItem,
)
from flywheel_tpu.softmax_attention.schedule import FwdSchedule
from flywheel_tpu.softmax_attention.traits import FwdTraits
from flywheel_tpu.softmax_attention.variant import Variant, VariantOperands

# The row-state parity a row holds from its first tile to its finish.
MXU_ROW_PARITY = 0
# The body of a checked block that finishes its first body's items.
SECOND_BODY = 1
# The fewest rows and keys of a block under a mask that runs masked loop bodies
# or edge segments.
MIN_MASKED_BLOCK = 1024
# A segment before a block step ends on a tile of at least this many items: its
# V push then runs before the step's DMA may overwrite the block it reads.
MIN_LAST_TILE_ITEMS = (
    PipelineDelays().pv_issue - PipelineDelays().v_push_lead + 1
)


def dense_uses_mxu_pipeline(
    *,
    variant: Variant,
    block_sizes: BlockSizes,
    head_dim: int,
    head_dim_v: int,
    transposed_pv: bool,
) -> bool:
    """Whether a dense build runs on the explicit MXU pipeline. The loop scales
    its scores by the static Q scale, so it takes no kv descales."""
    return not variant.has_kv_scales and mxu_pipeline_applies(
        block_sizes=block_sizes,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        transposed_pv=transposed_pv,
    )


def dense_mxu_runs_blocks(variant: Variant, block_sizes: BlockSizes) -> bool:
    """Whether the explicit dense loop runs a build of block_sizes. Any mask but
    none or a static causal one of whole square blocks needs whole loop bodies
    per kv block and blocks of at least MIN_MASKED_BLOCK rows and keys."""
    block_q, block_kv = block_sizes.block_q, block_sizes.block_kv
    is_static_causal = (
        variant.window == (None, 0)
        and not variant.runtime_causal_offset
        and block_q == block_kv
        and variant.causal_offset % block_kv == 0
    )
    if variant.window == (None, None) or is_static_causal:
        return True
    return (
        min(block_q, block_kv) >= MIN_MASKED_BLOCK
        and block_kv % (LOOP_BODY_TILES * MXU_COLUMNS) == 0
    )


def dense_has_checked_units(num_kv_blocks: int, num_fragments: int) -> bool:
    """Whether rows of up to num_kv_blocks kv blocks and num_fragments fragments
    have a check unit before their final unit."""
    return num_kv_blocks > FINAL_UNIT_SPAN and not zeroes_row_start(
        num_fragments
    )


class FragmentMode(NamedTuple):
    """The static flags of a work item: a masked item's scores take the mask, a
    row-end item also finishes its fragment's rows."""

    is_row_start: bool = False
    is_masked: bool = False
    is_row_end: bool = False


PLAIN = FragmentMode()
ROW_START = FragmentMode(is_row_start=True)
ROW_END = FragmentMode(is_row_end=True)


def is_row_bound(mode: FragmentMode) -> bool:
    """Whether an item of mode starts or ends its fragment's rows."""
    return mode.is_row_start or mode.is_row_end


def tile_items(
    fragments: range, tile: int, mode: FragmentMode = PLAIN
) -> tuple[WorkItem, ...]:
    """fragments against kv tile tile, in order."""
    return tuple(
        WorkItem(fragment=fragment, tile=tile, mode=mode)
        for fragment in fragments
    )


def body_items(
    traits: FwdTraits,
    last_tile_mode: FragmentMode = PLAIN,
    masked_tiles: frozenset[int] = frozenset(),
) -> tuple[WorkItem, ...]:
    """A loop body's items: every fragment against LOOP_BODY_TILES tiles, the
    last tile's with last_tile_mode, those of masked_tiles masked."""
    fragments = range(traits.bq // FRAGMENT_ROWS)
    return sum(
        (
            tile_items(
                fragments,
                tile,
                (
                    last_tile_mode if tile == LOOP_BODY_TILES - 1 else PLAIN
                )._replace(is_masked=tile in masked_tiles),
            )
            for tile in range(LOOP_BODY_TILES)
        ),
        (),
    )


def single_block_items(
    traits: FwdTraits, masked_tiles: frozenset[int] = frozenset()
) -> tuple[WorkItem, ...]:
    """A row of one kv block as one segment that starts and ends it."""
    fragments = range(traits.bq // FRAGMENT_ROWS)
    num_tiles = traits.num_k_compute_tiles
    return sum(
        (
            tile_items(
                fragments,
                tile,
                FragmentMode(
                    is_row_start=tile == 0,
                    is_masked=tile in masked_tiles,
                    is_row_end=tile == num_tiles - 1,
                ),
            )
            for tile in range(num_tiles)
        ),
        (),
    )


def row_start_items(traits: FwdTraits) -> tuple[WorkItem, ...]:
    """A row's first segment: the tile that starts the row, then the plain tile
    a loop body needs before it."""
    fragments = range(traits.bq // FRAGMENT_ROWS)
    return tile_items(fragments, 0, ROW_START) + tile_items(fragments, 1)


class BlockSegment(NamedTuple):
    """One edge kv block of a row as a segment of static items. block counts
    from the row's frame, kv block qi * block_q / block_kv for q block qi. The
    first first_body items run before the check of the unit before."""

    block: int
    items: tuple[WorkItem, ...]
    first_body: int = 0


def split_first_body(
    segment: BlockSegment, fragments: set[int], delays: PipelineDelays
) -> BlockSegment:
    """segment with its first body split off and padded, with masked items of no
    live key on fragments, to pv_pop items and a last tile of
    MIN_LAST_TILE_ITEMS."""
    first = [item for item in segment.items if item.tile < LOOP_BODY_TILES]
    rest = [item for item in segment.items if item.tile >= LOOP_BODY_TILES]
    assert segment.items == (*first, *rest), segment
    assert not any(is_row_bound(item.mode) for item in first), segment
    last_tile = LOOP_BODY_TILES - 1
    present = {(item.tile, item.fragment) for item in first}
    for fragment in sorted(fragments):
        num_last = sum(1 for item in first if item.tile == last_tile)
        if len(first) >= delays.pv_pop and num_last >= MIN_LAST_TILE_ITEMS:
            break
        if (last_tile, fragment) not in present:
            first.append(
                WorkItem(
                    fragment=fragment,
                    tile=last_tile,
                    mode=PLAIN._replace(is_masked=True),
                )
            )
    assert len(rest) >= delays.pv_pop, segment
    return BlockSegment(segment.block, (*first, *rest), len(first))


class StaticRowPlan(NamedTuple):
    """The row plan of a build whose rows all meet their edges at the same
    offsets of their frame: left edge segments, interior blocks as loop bodies,
    right edge segments. A row that visits no interior block starts at
    right_row_start in place of the first right segment. interior_first is None
    when the interior starts at kv block 0."""

    left: tuple[BlockSegment, ...]
    right: tuple[BlockSegment, ...]
    right_row_start: BlockSegment | None
    interior_first: int | None


def frame_geometry(variant: Variant, causal_offset: int) -> MaskGeometry:
    """The mask of a q block of a build with a static offset, rows counted from
    the block's first row and keys from that row's frame key."""
    left, right = variant.window
    return MaskGeometry(
        shift=causal_offset,
        left=left,
        right=right,
        key_floor=UNBOUNDED_ROW,
    )


def block_classes(
    geometry: MaskGeometry, traits: FwdTraits, block: int
) -> dict[tuple[int, int], ItemClass]:
    """{(tile, fragment): class} of the frame's block block's live items."""
    classes = {}
    for tile in range(traits.num_k_compute_tiles):
        first_key = block * traits.bkv + tile * MXU_COLUMNS
        for fragment in range(traits.bq // FRAGMENT_ROWS):
            item = item_class(geometry, fragment * FRAGMENT_ROWS, first_key)
            if item.is_live:
                classes[(tile, fragment)] = item
    return classes


def is_interior_block(
    geometry: MaskGeometry,
    traits: FwdTraits,
    block: int,
    *,
    anchor_tiles: int = 0,
) -> bool:
    """Whether every row sees every key of the frame's block block and none
    takes its anchor there past its first anchor_tiles tiles."""
    classes = block_classes(geometry, traits, block)
    num_items = traits.num_k_compute_tiles * traits.bq // FRAGMENT_ROWS
    return len(classes) == num_items and all(
        item.is_interior and (tile < anchor_tiles or not item.has_anchor)
        for (tile, _), item in classes.items()
    )


class PlanRows(NamedTuple):
    """The rows a static plan serves: q blocks first_row..last_row, the kv
    blocks a q block's frame holds, the kv blocks of the axis, and whether the
    mask shows a padded key to a row."""

    first_row: int
    last_row: int
    blocks_per_row: int
    num_kv_blocks: int
    has_padded_keys: bool = False

    def may_leave_axis(self, block: int) -> bool:
        """Whether block block of some row's frame lies outside the kv axis."""
        return any(
            not 0 <= row * self.blocks_per_row + block < self.num_kv_blocks
            for row in range(self.first_row, self.last_row + 1)
        )

    def first_key_blocks(self, first_block: int) -> set[int]:
        """The frame's blocks that hold kv block 0 for a row whose frame's block
        first_block lies before the kv axis."""
        return {
            -row * self.blocks_per_row
            for row in range(self.first_row, self.last_row + 1)
            if row * self.blocks_per_row + first_block < 0
        }


def edge_regions(
    geometry: MaskGeometry, traits: FwdTraits, rows: PlanRows
) -> tuple[range, range, int | None] | None:
    """(left blocks, right blocks, first interior block) of the frame; every
    block is in the left run when none is interior. None when the rows share no
    static plan: no right bound, or a right edge that starts before kv block 0
    without a left bound."""
    bq, bkv = traits.bq, traits.bkv
    left, right = geometry.left, geometry.right
    if right is None:
        return None
    last = (bq - 1 + geometry.shift + right) // bkv
    if left is None:
        first_right = last
        while not is_interior_block(geometry, traits, first_right - 1):
            first_right -= 1
        if rows.first_row * rows.blocks_per_row + first_right < 0:
            return None
        return range(0), range(first_right, last + 1), None
    first = (geometry.shift - left) // bkv
    # The first interior body after the left blocks is masked, so the first
    # interior block may hold anchors on its first LOOP_BODY_TILES tiles.
    interior = []
    for block in range(first + 1, last + 1):
        anchor_tiles = 0 if interior else LOOP_BODY_TILES
        if not is_interior_block(
            geometry, traits, block, anchor_tiles=anchor_tiles
        ):
            break
        interior.append(block)
    if not interior:
        return range(first, last + 1), range(0), None
    right_first = interior[-1] + 1
    assert not any(
        is_interior_block(geometry, traits, block)
        for block in range(right_first, last + 1)
    ), (interior, last)
    return range(first, interior[0]), range(right_first, last + 1), interior[0]


def region_items(
    geometry: MaskGeometry,
    traits: FwdTraits,
    rows: PlanRows,
    blocks: range,
    first_key_blocks: set[int],
) -> dict[int, dict[tuple[int, int], bool]]:
    """{block: {(tile, fragment): is_masked}} of the live items of blocks:
    masked where a row sees part of the tile or may take its anchor there, on
    tile 0 of first_key_blocks, and in a block that may leave the kv axis."""
    region = {}
    for block in blocks:
        leaves_axis = rows.may_leave_axis(block)
        region[block] = {
            (tile, fragment): (
                not item.is_interior
                or item.has_anchor
                or leaves_axis
                or (tile == 0 and block in first_key_blocks)
            )
            for (tile, fragment), item in block_classes(
                geometry, traits, block
            ).items()
        }
    return region


def pad_block(
    items: dict[tuple[int, int], bool],
    traits: FwdTraits,
    started: set[int],
    *,
    min_items: int,
    min_last_tile_items: int,
) -> None:
    """Add masked items of no live key to items, one block's, until it holds
    min_items and its last tile min_last_tile_items, on the fragments of
    started, from its last tile back."""
    num_fragments = traits.bq // FRAGMENT_ROWS
    last_tile = max(tile for tile, _ in items)
    for tile in range(last_tile, -1, -1):
        need_tile = tile == last_tile
        for fragment in range(num_fragments):
            is_short = len(items) < min_items or (
                need_tile
                and sum(1 for t, _ in items if t == last_tile)
                < min_last_tile_items
            )
            if not is_short:
                return
            if fragment in started and (tile, fragment) not in items:
                items[(tile, fragment)] = True
    raise ValueError(
        f"an edge block of {len(items)} items has too few started fragments"
        " to fill a segment."
    )


def interleaved_tile(
    entries: list[tuple[int, bool]],
) -> list[tuple[int, bool]]:
    """One tile's (fragment, is_masked) entries, masked ones alternating with
    plain ones: a masked item holds its scores through the compare, which a run
    of them cannot hide."""
    masked = [entry for entry in entries if entry[1]]
    plain = [entry for entry in entries if not entry[1]]
    paired = [entry for pair in zip(masked, plain) for entry in pair]
    return paired + masked[len(plain) :] + plain[len(masked) :]


def block_segments(
    region: dict[int, dict[tuple[int, int], bool]],
    starts: set[int],
    ends: set[int],
    *,
    is_row_start_free: bool,
) -> tuple[BlockSegment, ...]:
    """The segments of region, each block's items tile by tile. A fragment of
    starts starts its row at its first item unless is_row_start_free; one of
    ends ends it at its last."""
    order = [
        (block, tile, fragment, region[block][(tile, fragment)])
        for block in sorted(region)
        for tile in sorted({tile for tile, _ in region[block]})
        for fragment, _ in interleaved_tile(
            sorted(
                (fragment, is_masked)
                for (t, fragment), is_masked in region[block].items()
                if t == tile
            )
        )
    ]
    first_of = {}
    last_of = {}
    for position, (_, _, fragment, _) in enumerate(order):
        first_of.setdefault(fragment, position)
        last_of[fragment] = position
    segments = {}
    for position, (block, tile, fragment, is_masked) in enumerate(order):
        mode = FragmentMode(
            is_row_start=(
                not is_row_start_free
                and fragment in starts
                and first_of[fragment] == position
            ),
            is_masked=is_masked,
            is_row_end=fragment in ends and last_of[fragment] == position,
        )
        segments.setdefault(block, []).append(
            WorkItem(fragment=fragment, tile=tile, mode=mode)
        )
    return tuple(
        BlockSegment(block, tuple(items)) for block, items in segments.items()
    )


def add_missing_items(
    region: dict[int, dict[tuple[int, int], bool]],
    fragments: range,
    *,
    block: int,
    tile: int | None,
) -> None:
    """Give every fragment without an item in region a masked one on tile tile
    (None: the last) of block block, where its row starts or ends."""
    present = {fragment for items in region.values() for _, fragment in items}
    items = region[block]
    target = max(t for t, _ in items) if tile is None else tile
    for fragment in fragments:
        if fragment not in present:
            items[(target, fragment)] = True


def pad_region(
    region: dict[int, dict[tuple[int, int], bool]],
    traits: FwdTraits,
    started: set[int],
    *,
    ends_row: bool,
    delays: PipelineDelays,
) -> None:
    """pad_block every block of region to a segment; all but the row's last end
    on a tile of MIN_LAST_TILE_ITEMS."""
    blocks = sorted(region)
    num_block_items = traits.num_k_compute_tiles * (traits.bq // FRAGMENT_ROWS)
    for block in blocks:
        is_row_end = ends_row and block == blocks[-1]
        pad_block(
            region[block],
            traits,
            started,
            min_items=min(delays.pv_pop, num_block_items),
            min_last_tile_items=1 if is_row_end else MIN_LAST_TILE_ITEMS,
        )


def edge_items(
    geometry: MaskGeometry,
    traits: FwdTraits,
    rows: PlanRows,
    regions: tuple[range, range],
    delays: PipelineDelays,
) -> tuple[dict, dict]:
    """The items of the left and right edge blocks of regions: every fragment
    has an item in each run, and each block fills a segment."""
    left_blocks, right_blocks = regions
    first_key_blocks = set()
    if geometry.left is not None:
        first_key_blocks = rows.first_key_blocks(left_blocks.start)
    left_region, right_region = (
        region_items(geometry, traits, rows, blocks, first_key_blocks)
        for blocks in (left_blocks, right_blocks)
    )
    fragments = range(traits.bq // FRAGMENT_ROWS)
    # Every fragment ends each left block on its last tile, which writes the
    # next unit's checkpoint.
    last_tile = traits.num_k_compute_tiles - 1
    for items in left_region.values():
        for fragment in fragments:
            items.setdefault((last_tile, fragment), True)
    if right_region:
        add_missing_items(
            right_region, fragments, block=right_blocks.start, tile=0
        )
        # The block before the right edge stages its first tile.
        right_region[right_blocks.start].setdefault((0, 0), True)
    for region, ends_row in (
        (left_region, not right_region),
        (right_region, True),
    ):
        pad_region(
            region, traits, set(fragments), ends_row=ends_row, delays=delays
        )
    return left_region, right_region


def static_row_plan(
    traits: FwdTraits,
    variant: Variant,
    causal_offset: int,
    rows: PlanRows,
    delays: PipelineDelays,
) -> StaticRowPlan | None:
    """The static row plan of a build whose rows all meet the mask the same way;
    None when they need the runtime plan's masked loop bodies. Under a left
    bound every fragment starts in the left blocks and ends in the right ones;
    without it the rows start on the first interior block, or on the right edge
    when they visit none."""
    if variant.window == (None, None):
        return StaticRowPlan((), (), None, None)
    if traits.bq % traits.bkv or rows.has_padded_keys:
        return None
    geometry = frame_geometry(variant, causal_offset)
    regions = edge_regions(geometry, traits, rows)
    if regions is None:
        return None
    left_region, right_region = edge_items(
        geometry, traits, rows, regions[:2], delays
    )
    fragments = set(range(traits.bq // FRAGMENT_ROWS))
    right_segments = block_segments(
        right_region, set(), fragments, is_row_start_free=True
    )
    right_row_start = None
    if variant.window[0] is None and right_region:
        starting = block_segments(
            right_region, fragments, fragments, is_row_start_free=False
        )
        assert starting[1:] == right_segments[1:], starting
        right_row_start = starting[0]
    left_segments = block_segments(
        left_region,
        fragments,
        set() if right_region else fragments,
        is_row_start_free=False,
    )
    if dense_has_checked_units(rows.num_kv_blocks, len(fragments)):
        left_segments = left_segments[:1] + tuple(
            split_first_body(segment, fragments, delays)
            for segment in left_segments[1:]
        )
    return StaticRowPlan(
        left_segments, right_segments, right_row_start, regions[2]
    )


class CheckRefs(NamedTuple):
    """A build's check unit scratch: the (NUM_CHECKPOINTS, bq, ...) entry row
    sums and accumulators, and the SMEM (1,) flag of the unit a first body
    checks."""

    checkpoint_sum: jax.Array
    checkpoint_o: jax.Array
    unit_guard: jax.Array


@dataclasses.dataclass(eq=False)
class RowEnd:
    """What a row's last items finish into: the output stage slot and the head's
    sink logit (None without sinks). largest_sum collects the finished row sums;
    the drain stores their maximum into largest_sum_ref for the guard."""

    out_slot: jax.Array
    sink: jax.Array | None
    largest_sum_ref: jax.Array
    largest_sum: LargestSums = dataclasses.field(default_factory=LargestSums)

    def store_largest_sum(self) -> None:
        """Store the largest row sum, after the row's drain."""
        self.largest_sum_ref[...] = self.largest_sum.value
        self.largest_sum.value = None


class StagedBlocks(NamedTuple):
    """The kv ring slots and first keys of a visit's block and the next's."""

    slot: jax.Array
    next_slot: jax.Array
    key: jax.Array
    next_key: jax.Array


@dataclasses.dataclass(frozen=True, eq=False)
class DenseFragmentOps:
    """The FragmentOps of a dense row: q fragments from staging slot q_slot, K
    and V tiles of the kv block at kv_slot (first key block_key) counted from
    first_tile, and the row state at MXU_ROW_PARITY. A masked item takes
    geometry's mask, its rows counted from q_first. A row-end item writes its
    normalized output rows into o_stage at row_end's slot. unit_end is a check
    unit's last tile; largest collects the row sums of tile key 0."""

    traits: FwdTraits
    variant: Variant
    operands: VariantOperands
    q_refs: QRefs
    kv_refs: KVRefs
    state: RowStateRefs
    o_stage: jax.Array
    q_slot: jax.Array
    geometry: MaskGeometry
    q_first: jax.Array
    kv_slot: jax.Array | None = None
    block_key: jax.Array | int = 0
    first_tile: jax.Array | int = 0
    masks_all: bool = False
    row_end: RowEnd | None = None
    unit_end: UnitCheckpoint | None = None
    largest: LargestSums | None = None

    def with_tiles(
        self,
        kv_slot: jax.Array,
        block_key: jax.Array | int,
        first_tile: jax.Array | int,
        row_end: RowEnd | None = None,
    ) -> "DenseFragmentOps":
        return dataclasses.replace(
            self,
            kv_slot=kv_slot,
            block_key=block_key,
            first_tile=first_tile,
            row_end=row_end,
        )

    def masked(self) -> "DenseFragmentOps":
        """These ops with every item masked."""
        return dataclasses.replace(self, masks_all=True)

    def _tile(self, buf: jax.Array, tile: int) -> jax.Array:
        assert self.kv_slot is not None, "a tail's ops push no kv tile."
        tile_start = (self.first_tile + tile) * MXU_COLUMNS
        if isinstance(tile_start, int):
            keys = pl.ds(tile_start, MXU_COLUMNS)
        else:
            keys = pl.ds(pl.multiple_of(tile_start, MXU_COLUMNS), MXU_COLUMNS)
        return buf[self.kv_slot, keys, :]

    # A V handle: the block's staging slot and the tile's first key in it.
    v_handle_words = 2

    def v_handle(self, tile: int) -> tuple[jax.Array, ...]:
        assert self.kv_slot is not None, "a tail's ops push no kv tile."
        return (
            jnp.asarray(self.kv_slot, jnp.int32),
            jnp.asarray((self.first_tile + tile) * MXU_COLUMNS, jnp.int32),
        )

    def v_tile_at(self, handle: tuple[jax.Array, ...]) -> jax.Array:
        slot, start = handle
        keys = pl.ds(pl.multiple_of(start, MXU_COLUMNS), MXU_COLUMNS)
        return self.kv_refs.v_buf[slot, keys, :]

    def k_tile(self, tile: int) -> jax.Array:
        return self._tile(self.kv_refs.k_buf, tile)

    def v_tile(self, tile: int) -> jax.Array:
        return self._tile(self.kv_refs.v_buf, tile)

    def q_fragment(self, item: WorkItem) -> jax.Array:
        return self.q_refs.buf[self.q_slot, fragment_rows(item.fragment), :]

    @property
    def row_max(self) -> jax.Array:
        return self.state.row_max

    @property
    def row_sum(self) -> jax.Array:
        return self.state.row_sum.at[MXU_ROW_PARITY]

    def rows(self, item: WorkItem) -> pl.Slice:
        """The row-state rows of item's fragment."""
        return fragment_rows(item.fragment)

    def is_masked(self, item: WorkItem) -> bool:
        return self.masks_all or item.mode.is_masked

    @functools.cached_property
    def bounds(self) -> dict[int, TileBounds]:
        """tile_bounds's cache by tile key, one per copy of the ops."""
        return {}

    def tile_bounds(self, tile: int) -> TileBounds:
        """Tile key tile's bounds, counted from each fragment row."""
        if tile not in self.bounds:
            first_key = self.block_key + (self.first_tile + tile) * MXU_COLUMNS
            self.bounds[tile] = (
                tile_bounds(self.geometry, first_key)
                .from_row(self.q_first)
                .from_fragment_rows()
            )
        return self.bounds[tile]

    def masked_scores(self, item: WorkItem, scores: jax.Array) -> jax.Array:
        """item's raw scores as log2-unit logits, a masked item's masked."""
        if not self.is_masked(item):
            return scaled_logits(self.variant, scores)
        return masked_logits(
            self.variant,
            scores,
            self.tile_bounds(item.tile),
            item.fragment * FRAGMENT_ROWS,
        )

    def takes_anchor(self, item: WorkItem) -> jax.Array | None:
        """The rows of item's fragment that take their anchor in its tile; None
        where they all take it at their row start."""
        if not (self.geometry.has_deferred_anchors and self.is_masked(item)):
            return None
        return self.tile_bounds(item.tile).takes_anchor(
            item.fragment * FRAGMENT_ROWS
        )

    def is_unit_end(self, item: WorkItem) -> bool:
        return self.unit_end is not None and item.tile == self.unit_end.tile

    def softmax(self, item: WorkItem, scores: jax.Array) -> jax.Array:
        is_unit_end = self.is_unit_end(item)
        largest = self.largest if item.tile == 0 and not is_unit_end else None
        probs = fold_fragment_scores(
            self.row_max,
            self.row_sum,
            self.rows(item),
            self.masked_scores(item, scores),
            is_row_start=item.mode.is_row_start,
            takes_anchor=self.takes_anchor(item),
            largest=largest,
            checkpoint_sum=self.unit_end.row_sum if is_unit_end else None,
        )
        if item.mode.is_row_end and not is_unit_end:
            self.finish_row_sums(item)
        return probs

    def finish_row_sums(self, item: WorkItem) -> None:
        """Reduce a row-end item's lane partial sums to row sums and fold them
        into row_end's maximum."""
        rows, row_sum_ref = self.rows(item), self.row_sum
        row_sum = across_lanes(row_sum_ref[rows, :].sum(axis=-1))
        row_sum_ref[rows, :] = row_sum
        self.row_end.largest_sum.fold(row_sum)

    def accumulate(self, item: WorkItem, weighted_values: jax.Array) -> None:
        is_unit_end = self.is_unit_end(item)
        if is_unit_end or not item.mode.is_row_end:
            accumulate_fragment(
                self.state.o.at[MXU_ROW_PARITY],
                self.rows(item),
                weighted_values,
                is_row_start=item.mode.is_row_start,
                checkpoint_o=self.unit_end.o if is_unit_end else None,
            )
            return
        # A row-end item's output rows are final: they go to the stage
        # normalized, sink folded, instead of back to the accumulator.
        rows = fragment_rows(item.fragment)
        out = weighted_values
        if not item.mode.is_row_start:
            out += self.state.o[MXU_ROW_PARITY, rows, :]
        row_sum = self.state.row_sum[MXU_ROW_PARITY, rows, :]
        if self.row_end.sink is not None:
            _, _, out_scale = self.variant.fold_sink(
                self.state.row_max[rows, :], row_sum, self.row_end.sink
            )
            row_sum = 1.0 / out_scale
        scale = jnp.tile(
            self.variant.output_scale(row_sum, self.operands),
            (1, self.traits.head_dim_v_repeats),
        )
        self.o_stage[self.row_end.out_slot, rows, :] = (out * scale).astype(
            self.o_stage.dtype
        )


class DenseRow(NamedTuple):
    """A row while it runs. Its interior loop runs visits interior_start up to
    interior_end after start_state, and num_masked masked visits follow."""

    visits: RowVisits
    base_head: jax.Array
    ops: DenseFragmentOps
    kv_carry: tuple
    interior_start: jax.Array | int
    interior_end: jax.Array
    num_masked: jax.Array | int
    row_end: RowEnd
    start_state: PipelineState


class LoopEntry(NamedTuple):
    """Where a row's rolled loop takes over: the pipeline state its first
    segments leave, the loop's first visit and the bodies of it they ran."""

    state: PipelineState
    first_visit: jax.Array | int
    first_body: int


@dataclasses.dataclass(frozen=True)
class DenseRowLayout:
    """Which kv blocks a dense row visits and how: by the static plan when the
    build has one, else blocks lo..hi with the edge blocks as masked loop
    bodies. geometry is the mask in absolute positions."""

    traits: FwdTraits
    schedule: FwdSchedule
    geometry: MaskGeometry
    plan: StaticRowPlan | None

    @property
    def blocks_per_row(self) -> int:
        return self.traits.bq // self.traits.bkv

    def q_first(self, si: jax.Array) -> jax.Array:
        return (self.schedule.first_row + si) * self.traits.bq

    def visits(self, si: jax.Array) -> RowVisits:
        """The blocks row si visits."""
        if self.plan is None:
            lo, hi, _ = self.schedule.row_interval(si)
            return RowVisits(hi - lo + 1, lambda visit: lo + visit)
        plan = self.plan
        frame = (self.schedule.first_row + si) * self.blocks_per_row
        interior_lo, num_interior = self.interior_blocks(si)
        num_left = len(plan.left)
        right_first = frame + (plan.right[0].block if plan.right else 0)
        left_first = frame + (plan.left[0].block if plan.left else 0)

        def _block_of(visit):
            return jnp.where(
                visit < num_left,
                left_first + visit,
                jnp.where(
                    visit < num_left + num_interior,
                    interior_lo + visit - num_left,
                    right_first + visit - num_left - num_interior,
                ),
            )

        return RowVisits(num_left + num_interior + len(plan.right), _block_of)

    def interior_range(
        self,
        frame: jax.Array | int,
        minimum: Callable = jnp.minimum,
        maximum: Callable = jnp.maximum,
    ) -> tuple[jax.Array | int, jax.Array | int]:
        """(first block, count) of the interior blocks of a static-plan row
        whose frame starts at kv block frame; Python's min and max for a static
        frame."""
        plan, num_kv_blocks = self.plan, self.schedule.num_kv_blocks
        if not plan.right:
            return 0, 0 if plan.left else num_kv_blocks
        end = minimum(frame + plan.right[0].block, num_kv_blocks)
        first = 0
        if plan.interior_first is not None:
            first = maximum(frame + plan.interior_first, 0)
        return first, maximum(end - first, 0)

    def interior_blocks(self, si: jax.Array) -> tuple[jax.Array, jax.Array]:
        """interior_range of row si."""
        frame = (self.schedule.first_row + si) * self.blocks_per_row
        first, count = self.interior_range(frame)
        return jnp.asarray(first, jnp.int32), jnp.asarray(count, jnp.int32)

    @property
    def never_visits_interior(self) -> bool:
        """Whether no row of the static plan visits an interior block."""
        schedule = self.schedule
        rows = range(schedule.first_row, schedule.first_row + schedule.num_rows)
        return not any(
            self.interior_range(row * self.blocks_per_row, min, max)[1]
            for row in rows
        )

    def interior_visits(
        self, si: jax.Array, visits: RowVisits, *, has_masked_interior: bool
    ) -> tuple[jax.Array | int, jax.Array, jax.Array | int]:
        """(first, end, masked): row si's interior loop runs visits first up to
        end, then masked visits follow. Under the runtime plan a left bound
        leaves every visit to the masked ones unless has_masked_interior."""
        if self.plan is not None:
            _, num_interior = self.interior_blocks(si)
            num_left = len(self.plan.left)
            return num_left, num_left + num_interior, len(self.plan.right)
        num_visits = visits.num_visits
        if self.geometry.left is not None and not has_masked_interior:
            end = jnp.int32(0)
        elif self.geometry.right is None:
            end = num_visits
        else:
            lo, _, _ = self.schedule.row_interval(si)
            first_right = (
                self.q_first(si) + self.geometry.shift + self.geometry.right + 1
            ) // self.traits.bkv
            end = jnp.clip(first_right - lo, 0, num_visits)
        return 0, end, num_visits - end


def dense_row_layout(
    traits: FwdTraits,
    variant: Variant,
    schedule: FwdSchedule,
    causal_offset: jax.Array | int,
    delays: PipelineDelays,
    num_valid_keys: int | None = None,
) -> DenseRowLayout:
    """The row layout of a dense build: the static plan where its offset is
    static, else the runtime one. num_valid_keys ends the mask's kv axis."""
    left, right = variant.window
    num_axis_keys = schedule.num_kv_blocks * traits.bkv
    geometry = MaskGeometry(
        shift=causal_offset,
        left=left,
        right=right,
        key_end=num_axis_keys if num_valid_keys is None else num_valid_keys,
    )
    plan = None
    if isinstance(causal_offset, int):
        rows = PlanRows(
            first_row=schedule.first_row,
            last_row=schedule.first_row + schedule.num_rows - 1,
            blocks_per_row=traits.bq // traits.bkv,
            num_kv_blocks=schedule.num_kv_blocks,
            has_padded_keys=num_valid_keys is not None,
        )
        plan = static_row_plan(traits, variant, causal_offset, rows, delays)
    return DenseRowLayout(traits, schedule, geometry, plan)


@dataclasses.dataclass(frozen=True, eq=False)
class DenseMxuLoop:
    """The rows of a dense build on the explicit MXU pipeline, over
    dense_fwd_body's refs. row_end_sum holds the largest finished row sum of a
    drained row; check_refs the check units' scratch, None in a build without
    checked units. num_valid_keys is the keys of a kv axis padded to whole
    blocks where the mask shows a padded key to a row, else None."""

    traits: FwdTraits
    variant: Variant
    operands: VariantOperands
    schedule: FwdSchedule
    pipeline: MxuPipeline
    blocks: DenseBlockStream
    layout: DenseRowLayout
    q_refs: QRefs
    kv_refs: KVRefs
    out_refs: OutRefs
    state: RowStateRefs
    row_end_sum: jax.Array
    check_refs: CheckRefs | None
    num_valid_keys: int | None = None

    def __post_init__(self) -> None:
        traits, delays = self.traits, self.pipeline.delays
        assert traits.head_fold == 1, traits.head_fold
        assert (self.check_refs is not None) == self.has_checked_units
        # A first body's tail holds none of its first tile's items.
        assert not self.has_checked_units or (
            self.num_fragments >= delays.pv_pop
        )
        if not self.runs_loop_bodies:
            return
        # A row's first body precedes the body that ends it, and a body
        # finishes items of the body before it alone.
        assert traits.num_k_compute_tiles % LOOP_BODY_TILES == 0, traits
        assert (
            min(self.bodies_per_block, self.last_block_bodies) > SECOND_BODY
        ), (self.bodies_per_block, self.last_block_bodies)
        assert LOOP_BODY_TILES * self.num_fragments >= delays.pv_pop, traits

    @property
    def bodies_per_block(self) -> int:
        return self.traits.num_k_compute_tiles // LOOP_BODY_TILES

    @property
    def last_block_keys(self) -> int | None:
        """The valid keys of the kv axis's last block; None when the mask shows
        no padded key."""
        if self.num_valid_keys is None:
            return None
        return (
            self.num_valid_keys
            - (self.schedule.num_kv_blocks - 1) * self.traits.bkv
        )

    @property
    def last_block_tiles(self) -> int:
        """The tiles of the kv axis's last block that hold a valid key."""
        if self.last_block_keys is None:
            return self.traits.num_k_compute_tiles
        return -(-self.last_block_keys // MXU_COLUMNS)

    @property
    def last_block_bodies(self) -> int:
        """The bodies of the kv axis's last block that hold a valid key."""
        return -(-self.last_block_tiles // LOOP_BODY_TILES)

    def last_visit_bodies(self, row: DenseRow) -> jax.Array | int:
        """The bodies of row's last visit: last_block_bodies when it is the kv
        axis's last block."""
        if self.last_block_bodies == self.bodies_per_block:
            return self.bodies_per_block
        if self.geometry.right is None:
            return self.last_block_bodies
        visits = row.visits
        return jnp.where(
            visits.block_of(visits.num_visits - 1)
            == self.schedule.num_kv_blocks - 1,
            self.last_block_bodies,
            self.bodies_per_block,
        )

    def padded_tiles(self, first_tile: int, num_tiles: int) -> frozenset[int]:
        """Which of num_tiles tiles from first_tile of the kv axis's last block,
        counted from first_tile, hold a padded key."""
        if self.last_block_keys is None:
            return frozenset()
        return frozenset(
            tile
            for tile in range(num_tiles)
            if (first_tile + tile + 1) * MXU_COLUMNS > self.last_block_keys
        )

    @property
    def num_fragments(self) -> int:
        return self.traits.bq // FRAGMENT_ROWS

    @property
    def has_checked_units(self) -> bool:
        return dense_has_checked_units(
            self.schedule.num_kv_blocks, self.num_fragments
        )

    @property
    def is_single_block_row(self) -> bool:
        """Whether a row is one segment: no mask, a kv axis of one block."""
        return (
            self.variant.window == (None, None)
            and self.schedule.num_kv_blocks == 1
        )

    @property
    def runs_loop_bodies(self) -> bool:
        """Whether a row may run rolled loop bodies: not when every row is one
        segment or static segments alone."""
        if self.is_single_block_row:
            return False
        layout = self.layout
        return layout.plan is None or not layout.never_visits_interior

    @property
    def geometry(self) -> MaskGeometry:
        return self.layout.geometry

    @property
    def has_masked_interior(self) -> bool:
        """Whether a runtime-plan row's interior loop masks its items: under a
        left bound in a build with checked units."""
        return (
            self.layout.plan is None
            and self.geometry.left is not None
            and self.has_checked_units
        )

    def final_unit(self, row: DenseRow) -> jax.Array | int:
        """The first visit of row's final unit, its masked visits and at least
        its last FINAL_UNIT_SPAN; 0 when no unit is checked."""
        if not self.has_checked_units:
            return 0
        span = jnp.maximum(row.num_masked + 1, FINAL_UNIT_SPAN)
        return jnp.maximum(row.visits.num_visits - span, 0)

    @functools.cached_property
    def loop_items(self) -> tuple[WorkItem, ...]:
        return body_items(self.traits)

    def step_block(
        self,
        visit: jax.Array | int,
        visits: RowVisits,
        ahead: tuple[BlockDesc, BlockDesc],
        base_head: jax.Array,
        kv_carry: tuple,
    ) -> tuple[tuple, StagedBlocks]:
        """Step the kv ring at the top of visit visit: wait for the next visit's
        block and start the block two visits ahead; ahead is the next row's
        first two descriptors."""
        kv_slot, next_slot = kv_carry[:2]
        ahead_block = self.ahead_block(
            visit, visits, ahead, base_head, kv_carry
        )
        self.wait_next_block(kv_carry)
        return self.start_ahead_block(
            kv_carry, ahead_block
        ), self.staged_blocks(visits, visit, kv_slot, next_slot)

    def staged_blocks(
        self,
        visits: RowVisits,
        visit: jax.Array | int,
        slot: jax.Array,
        next_slot: jax.Array,
    ) -> StagedBlocks:
        """Visit visit's block at slot and the next visit's at next_slot."""
        bkv = self.traits.bkv
        return StagedBlocks(
            slot=slot,
            next_slot=next_slot,
            key=visits.block_of(visit) * bkv,
            next_key=visits.block_of(visit + 1) * bkv,
        )

    def ahead_block(
        self,
        visit: jax.Array | int,
        visits: RowVisits,
        ahead: tuple[BlockDesc, BlockDesc],
        base_head: jax.Array,
        kv_carry: tuple,
    ) -> tuple[BlockDesc, jax.Array, jax.Array]:
        """The descriptor, slot and freshness of the block two visits ahead."""
        blocks = self.blocks
        _, next_slot, _, _, _, next_desc = kv_carry
        first_next_row_desc, second_next_row_desc = ahead
        ahead_desc = select_desc(
            visit + 2 < visits.num_visits,
            blocks.desc(base_head, visits.block_of(visit + 2)),
            select_desc(
                visit + 2 == visits.num_visits,
                first_next_row_desc,
                second_next_row_desc,
            ),
        )
        is_ahead_fresh = blocks.is_fresh(ahead_desc, next_desc).astype(
            jnp.int32
        )
        ahead_slot = jnp.where(
            is_ahead_fresh > 0, lax.rem(next_slot + 1, NUM_KV_STAGES), next_slot
        )
        return ahead_desc, ahead_slot, is_ahead_fresh

    def wait_next_block(self, kv_carry: tuple) -> None:
        """Wait for the next visit's block, when it is fresh."""
        _, next_slot, _, is_next_fresh, _, _ = kv_carry

        @pl.when(is_next_fresh > 0)
        def _wait_next_block():
            wait_rotated_kv_block(self.traits, self.kv_refs, next_slot)

    def start_ahead_block(
        self,
        kv_carry: tuple,
        ahead_block: tuple[BlockDesc, jax.Array, jax.Array],
    ) -> tuple:
        """Start ahead_block when fresh; returns the kv carry one visit on."""
        _, next_slot, _, is_next_fresh, _, next_desc = kv_carry
        ahead_desc, ahead_slot, is_ahead_fresh = ahead_block

        @pl.when(is_ahead_fresh > 0)
        def _start_ahead_block():
            self.blocks.start(ahead_desc, ahead_slot)

        return (
            next_slot,
            ahead_slot,
            is_next_fresh,
            is_ahead_fresh,
            next_desc,
            ahead_desc,
        )

    def run_body(
        self,
        ops: DenseFragmentOps,
        assumed: Tail,
        staged: StagedBlocks,
        body: jax.Array,
        probs: tuple[jax.Array, ...],
    ) -> tuple[jax.Array, ...]:
        """One loop body: LOOP_BODY_TILES tiles from tile body * LOOP_BODY_TILES
        of staged's block. It stages the next body's first K tile, the next
        block's after the block's last body."""
        pipeline = self.pipeline
        first_tile = body * LOOP_BODY_TILES
        is_block_end = body == self.bodies_per_block - 1
        next_ops = ops.with_tiles(
            jnp.where(is_block_end, staged.next_slot, staged.slot),
            jnp.where(is_block_end, staged.next_key, staged.key),
            jnp.where(is_block_end, 0, first_tile + LOOP_BODY_TILES),
        )
        body_state = pipeline.advance(
            PipelineState(probs=probs, tail=assumed),
            self.loop_items,
            ops.with_tiles(staged.slot, staged.key, first_tile),
            next_k_tile=lambda: next_ops.k_tile(0),
        )
        pipeline.check_loop_tail(body_state.tail, assumed)
        return body_state.probs

    def loop_tail(
        self, ops: DenseFragmentOps, state_before: PipelineState
    ) -> Tail:
        """The tail a loop body assumes after the segment before it."""
        pipeline = self.pipeline
        return pipeline.tail_of(
            self.loop_items,
            ops,
            slot_base=state_before.tail.end_slot(pipeline.num_slots),
            has_staged_next_k=True,
        )

    def visit_bodies(
        self, row: DenseRow, entry: LoopEntry, visit: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """(first, end) of visit visit's loop bodies; the row's last visit
        leaves its last body to the row's end."""
        return (
            jnp.where(visit == entry.first_visit, entry.first_body, 0),
            jnp.where(
                visit == row.visits.num_visits - 1,
                self.last_visit_bodies(row) - 1,
                self.bodies_per_block,
            ),
        )

    def run_interior(
        self,
        row: DenseRow,
        ahead: tuple[BlockDesc, BlockDesc],
        entry: LoopEntry,
        staged: StagedBlocks,
    ) -> tuple[tuple, tuple[jax.Array, ...], StagedBlocks]:
        """Row's rolled interior visits from entry's to row.interior_end; staged
        holds the block its first segments ran."""
        loop_ops = row.ops.masked() if self.has_masked_interior else row.ops
        assumed = self.loop_tail(loop_ops, entry.state)
        self.pipeline.check_loop_tail(entry.state.tail, assumed)
        if self.has_checked_units:
            run_visit = self.run_checked_visit
        else:
            run_visit = self.run_visit
        kv_carry, probs, staged = lax.fori_loop(
            entry.first_visit,
            row.interior_end,
            partial(run_visit, row, ahead, entry, assumed),
            (row.kv_carry, entry.state.probs, staged),
        )
        return kv_carry, probs, staged

    def run_visit(
        self,
        row: DenseRow,
        ahead: tuple[BlockDesc, BlockDesc],
        entry: LoopEntry,
        assumed: Tail,
        visit: jax.Array,
        carry: tuple,
    ) -> tuple:
        """A visit of a build without checked units: the block step, then its
        bodies. carry is (kv carry, probs, blocks of the visit before)."""
        kv_carry, probs, _ = carry
        kv_carry, staged = self.step_block(
            visit, row.visits, ahead, row.base_head, kv_carry
        )
        probs = lax.fori_loop(
            *self.visit_bodies(row, entry, visit),
            partial(self.run_body, assumed.ops, assumed, staged),
            probs,
        )
        return kv_carry, probs, staged

    def body_tail(
        self,
        assumed: Tail,
        blocks: tuple[StagedBlocks, StagedBlocks],
        body: jax.Array | int,
    ) -> Tail:
        """assumed, a loop body's tail, at the tiles of the body before body
        body; blocks are the visit before's and this one's. A masked tail's
        masks depend on where its tiles lie."""
        prev, staged = blocks
        is_first = body == 0
        ops = assumed.ops.with_tiles(
            jnp.where(is_first, prev.slot, staged.slot),
            jnp.where(is_first, prev.key, staged.key),
            jnp.where(
                is_first,
                self.traits.num_k_compute_tiles - LOOP_BODY_TILES,
                (body - 1) * LOOP_BODY_TILES,
            ),
        )
        return assumed.with_ops(ops)

    def run_masked_visits(
        self,
        row: DenseRow,
        ahead: tuple[BlockDesc, BlockDesc],
        entry: LoopEntry,
        carry: tuple,
    ) -> tuple:
        """Row's masked visits, from row.interior_end on, as masked loop bodies;
        each finishes the body before with masks at that body's tiles."""
        masked_ops = row.ops.masked()
        assumed = self.loop_tail(masked_ops, entry.state)

        def _visit(visit, carry):
            kv_carry, probs, prev = carry
            kv_carry, staged = self.step_block(
                visit, row.visits, ahead, row.base_head, kv_carry
            )

            def _body(body, probs):
                tail = self.body_tail(assumed, (prev, staged), body)
                return self.run_body(masked_ops, tail, staged, body, probs)

            probs = lax.fori_loop(
                *self.visit_bodies(row, entry, visit), _body, probs
            )
            return kv_carry, probs, staged

        return lax.fori_loop(
            row.interior_end, row.visits.num_visits, _visit, carry
        )

    def run_checked_visit(
        self,
        row: DenseRow,
        ahead: tuple[BlockDesc, BlockDesc],
        entry: LoopEntry,
        assumed: Tail,
        visit: jax.Array,
        carry: tuple,
    ) -> tuple:
        """A visit of a build with checked units, as run_visit. Its first body
        checks the unit before; the block two visits ahead starts after it, once
        that unit's block can no longer replay from its slot."""
        kv_carry, probs, prev = carry
        staged = self.staged_blocks(row.visits, visit, *kv_carry[:2])
        self.wait_next_block(kv_carry)
        blocks = (prev, staged)
        probs = self.run_first_body(row, entry, assumed, visit, blocks, probs)
        kv_carry = self.start_ahead_block(
            kv_carry,
            self.ahead_block(visit, row.visits, ahead, row.base_head, kv_carry),
        )
        probs = self.run_later_bodies(row, entry, assumed, visit, blocks, probs)
        return kv_carry, probs, staged

    def run_first_body(
        self,
        row: DenseRow,
        entry: LoopEntry,
        assumed: Tail,
        visit: jax.Array,
        blocks: tuple[StagedBlocks, StagedBlocks],
        probs: tuple[jax.Array, ...],
    ) -> tuple[jax.Array, ...]:
        """A checked visit's first body (checked_body), unless the row's first
        segments ran it. It finishes the last tile of the unit before, which
        writes the visit's checkpoint."""
        unit_end_tail = self.loop_body_tail(assumed, blocks, 0)
        unit_end_tail = unit_end_tail.with_ops(
            self.unit_end_ops(unit_end_tail.ops, visit, LOOP_BODY_TILES - 1)
        )

        def _checked(probs):
            state = self.checked_body(
                row,
                visit,
                blocks,
                PipelineState(probs=probs, tail=unit_end_tail),
                assumed.ops,
            )
            self.pipeline.check_loop_tail(state.tail, assumed)
            return state.probs

        if entry.first_body == 0:
            return _checked(probs)
        return lax.cond(
            visit == entry.first_visit, lambda probs: probs, _checked, probs
        )

    def run_later_bodies(
        self,
        row: DenseRow,
        entry: LoopEntry,
        assumed: Tail,
        visit: jax.Array,
        blocks: tuple[StagedBlocks, StagedBlocks],
        probs: tuple[jax.Array, ...],
    ) -> tuple[jax.Array, ...]:
        """A checked visit's bodies after its first. The second finishes the
        first one's items (run_second_body). The bodies after it, or the second
        in a block of two bodies, write the next unit's checkpoint on their
        last tile."""
        _, staged = blocks
        loop_ops = assumed.ops
        is_last_visit = visit == row.visits.num_visits - 1
        # The last visit writes over the checkpoint of its own unit, which the
        # row's final unit does not read.
        writing_ops = self.unit_end_ops(
            loop_ops,
            visit + jnp.logical_not(is_last_visit).astype(jnp.int32),
            LOOP_BODY_TILES - 1,
        )
        second_body = partial(self.run_second_body, assumed, staged)

        def _body(body, probs):
            tail = self.loop_body_tail(assumed, blocks, body)
            return self.run_body(writing_ops, tail, staged, body, probs)

        _, body_end = self.visit_bodies(row, entry, visit)
        # Note (david): the second body is its own segment, not a loop body,
        # so the ahead block's DMA start joins its basic block.
        is_block_end = self.bodies_per_block == SECOND_BODY + 1
        second_ops = writing_ops if is_block_end else loop_ops
        if is_block_end or self.last_block_bodies == SECOND_BODY + 1:
            # The row's end runs the second body of a last visit of two bodies.
            probs = lax.cond(
                body_end > SECOND_BODY,
                partial(second_body, second_ops),
                lambda probs: probs,
                probs,
            )
        else:
            probs = second_body(second_ops, probs)
        if is_block_end:
            return probs
        return lax.fori_loop(SECOND_BODY + 1, body_end, _body, probs)

    def loop_body_tail(
        self,
        assumed: Tail,
        blocks: tuple[StagedBlocks, StagedBlocks],
        body: jax.Array | int,
    ) -> Tail:
        """The tail a checked loop body finishes: assumed, at its tiles when the
        loop's items are masked."""
        if not assumed.ops.masks_all:
            return assumed
        return self.body_tail(assumed, blocks, body)

    def run_second_body(
        self,
        assumed: Tail,
        staged: StagedBlocks,
        body_ops: DenseFragmentOps,
        probs: tuple[jax.Array, ...],
    ) -> tuple[jax.Array, ...]:
        """Body SECOND_BODY of a checked visit, with body_ops. It finishes the
        first body's items with FlaggedTailOps: they add nothing once a replay
        redid them."""
        first_body_ops = assumed.ops
        if self.geometry.has_deferred_anchors:
            # The first body may be a masked row-start segment whose tiles
            # hold anchors.
            first_body_ops = self.body_tail(
                assumed.with_ops(assumed.ops.masked()),
                (staged, staged),
                SECOND_BODY,
            ).ops
        tail = assumed.with_ops(
            FlaggedTailOps(
                first_body_ops,
                self.check_refs.unit_guard[0] > 0,
                is_row_bound=is_row_bound,
            )
        )
        return self.run_body(body_ops, tail, staged, SECOND_BODY, probs)

    def unit_end_ops(
        self, ops: DenseFragmentOps, unit: jax.Array | int, tile: int
    ) -> DenseFragmentOps:
        """ops whose items of tile key tile, check unit unit - 1's last tile,
        also write the row state they leave into unit's checkpoint."""
        index = lax.rem(unit, NUM_CHECKPOINTS)
        unit_end = UnitCheckpoint(
            tile=tile,
            row_sum=self.check_refs.checkpoint_sum.at[index],
            o=self.check_refs.checkpoint_o.at[index],
        )
        return dataclasses.replace(ops, unit_end=unit_end)

    def checked_body(
        self,
        row: DenseRow,
        visit: jax.Array | int,
        blocks: tuple[StagedBlocks, StagedBlocks],
        state_before: PipelineState,
        loop_ops: DenseFragmentOps,
    ) -> PipelineState:
        """Visit visit's first body with loop_ops after state_before. Its first
        tile's row sums bound those of check unit visit - 1, which replays at
        the body's end when they flag it."""
        _, staged = blocks
        largest = LargestSums()
        ops = loop_ops.with_tiles(staged.slot, staged.key, 0)
        next_ops = loop_ops.with_tiles(staged.slot, staged.key, LOOP_BODY_TILES)
        state = self.pipeline.advance(
            state_before,
            self.loop_items,
            dataclasses.replace(ops, largest=largest),
            next_k_tile=lambda: next_ops.k_tile(0),
        )
        self.check_unit(row, visit, blocks, largest, state)
        return state

    def run_segments(
        self,
        row: DenseRow,
        ahead: tuple[BlockDesc, BlockDesc],
        segments: Sequence[BlockSegment],
        first_visit: jax.Array | int,
        carry: tuple[tuple, PipelineState],
        *,
        stages_next: bool,
        writes_checkpoints: bool = False,
    ) -> tuple[tuple, PipelineState, StagedBlocks]:
        """Run static segments, one a visit from first_visit on, each after its
        block's step; carry is (kv carry, pipeline state). The last stages the
        next visit's first K tile when stages_next. writes_checkpoints makes
        each a check unit in a build with checked units."""
        kv_carry, state = carry
        prev = None
        for index, segment in enumerate(segments):
            visit = first_visit + index
            is_checked = self.has_checked_units and segment.first_body > 0
            ahead_block = self.ahead_block(
                visit, row.visits, ahead, row.base_head, kv_carry
            )
            staged = self.staged_blocks(row.visits, visit, *kv_carry[:2])
            self.wait_next_block(kv_carry)
            ops = row.ops.with_tiles(
                staged.slot, staged.key, 0, row_end=row.row_end
            )
            if is_checked:
                # The block before stays in its slot until its unit is
                # checked, so the block two visits ahead starts after.
                state = self.run_checked_segment(
                    row, visit, (prev, staged), ops, segment, state
                )
                segment = segment._replace(
                    items=segment.items[segment.first_body :]
                )
            kv_carry = self.start_ahead_block(kv_carry, ahead_block)
            has_next_segment = index + 1 < len(segments)
            ends_row = not (has_next_segment or stages_next)
            if self.has_checked_units and writes_checkpoints and not ends_row:
                # A left block's last tile writes the next unit's checkpoint;
                # the row's last block ends the row there instead.
                ops = self.unit_end_ops(
                    ops, visit + 1, self.traits.num_k_compute_tiles - 1
                )
            prev = staged
            if ends_row:
                next_k_tile = None
            else:
                next_ops = row.ops.with_tiles(
                    staged.next_slot, staged.next_key, 0
                )
                next_tile = 0
                if has_next_segment:
                    next_tile = segments[index + 1].items[0].tile
                next_k_tile = partial(next_ops.k_tile, next_tile)
            state = self.pipeline.advance(
                state, segment.items, ops, next_k_tile=next_k_tile
            )
        return kv_carry, state, staged

    def run_checked_segment(
        self,
        row: DenseRow,
        visit: jax.Array | int,
        blocks: tuple[StagedBlocks, StagedBlocks],
        ops: DenseFragmentOps,
        segment: BlockSegment,
        state: PipelineState,
    ) -> PipelineState:
        """Run a left segment's first body after state, then check unit visit -
        1. The rest of the segment finishes the returned state's tail with
        FlaggedTailOps."""
        first_body = segment.items[: segment.first_body]
        next_tile = segment.items[segment.first_body].tile
        largest = LargestSums()
        state = self.pipeline.advance(
            state,
            first_body,
            dataclasses.replace(ops, largest=largest),
            next_k_tile=partial(ops.k_tile, next_tile),
        )
        self.check_unit(row, visit, blocks, largest, state)
        tail_ops = FlaggedTailOps(
            state.tail.ops,
            self.check_refs.unit_guard[0] > 0,
            is_row_bound=is_row_bound,
        )
        return state._replace(tail=state.tail.with_ops(tail_ops))

    def check_unit(
        self,
        row: DenseRow,
        visit: jax.Array | int,
        blocks: tuple[StagedBlocks, StagedBlocks],
        largest: LargestSums,
        state: PipelineState,
    ) -> None:
        """Flag check unit visit - 1 when largest's row sums could pass the
        guard and it is not in the row's final unit, and replay it at once;
        state is what visit visit's first body left."""
        check_refs = self.check_refs
        check_refs.unit_guard[0] = jnp.where(
            visit - 1 < self.final_unit(row),
            guard_flag(self.traits, largest.value),
            0,
        )

        @pl.when(check_refs.unit_guard[0] > 0)
        def _replay():
            self.replay_unit(
                row.ops,
                visit,
                blocks,
                self.pipeline.free_slots(state.tail, REPLAY_RESULT_SLOTS),
            )

    def run_row_start(
        self,
        row: DenseRow,
        *,
        is_masked: bool,
        prev: StagedBlocks | None = None,
    ) -> tuple[LoopEntry, StagedBlocks]:
        """A row's first segment, the first body of its interior loop's first
        visit, in the kv ring's current slot. At visit 0 it starts the row;
        after left segments, whose last block prev holds, it checks their last
        unit."""
        slot, visit = row.kv_carry[0], row.interior_start
        staged = self.staged_blocks(row.visits, visit, slot, slot)
        ops = row.ops.with_tiles(slot, staged.key, 0)
        if is_masked:
            ops = ops.masked()
        if isinstance(visit, int) and visit == 0:
            assert row.start_state.tail is None, row.start_state.tail
            items = row_start_items(self.traits)
        else:
            items = self.loop_items
        is_checked = self.has_checked_units and prev is not None
        largest = LargestSums()
        if is_checked:
            ops = dataclasses.replace(ops, largest=largest)
        next_ops = row.ops.with_tiles(slot, staged.key, LOOP_BODY_TILES)
        start_state = self.pipeline.advance(
            row.start_state,
            items,
            ops,
            next_k_tile=lambda: next_ops.k_tile(0),
        )
        if is_checked:
            self.check_unit(row, visit, (prev, staged), largest, start_state)
        elif self.has_checked_units:
            # A row-start segment at the row's first visit checks nothing.
            self.check_refs.unit_guard[0] = jnp.int32(0)
        return LoopEntry(start_state, visit, 1), staged

    def run_row_end(
        self,
        row: DenseRow,
        probs: tuple[jax.Array, ...],
        assumed: Tail,
        staged: StagedBlocks,
    ) -> None:
        """A row's last body, which ends its fragments, then the drain; staged
        is the last visit's block, assumed the loop body's tail. Masked under
        the runtime layout; under the static one only the tiles that hold a
        padded key."""
        last_body = self.last_visit_bodies(row) - 1
        first_tile = last_body * LOOP_BODY_TILES
        end_ops = row.ops.with_tiles(
            staged.slot, staged.key, first_tile, row.row_end
        )
        masked_tiles = frozenset()
        if self.layout.plan is None:
            end_ops = end_ops.masked()
        else:
            masked_tiles = self.padded_tiles(first_tile, LOOP_BODY_TILES)
        last_state = self.pipeline.advance(
            PipelineState(
                probs=probs,
                tail=self.body_tail(assumed, (staged, staged), last_body),
            ),
            body_items(self.traits, ROW_END, masked_tiles),
            end_ops,
        )
        self.drain_row(row, last_state)

    def drain_row(self, row: DenseRow, state: PipelineState) -> None:
        """Drain a row's pipeline, then store its largest finished row sum."""
        self.pipeline.drain(state)
        row.row_end.store_largest_sum()

    def run_single_block_row(
        self, row: DenseRow, ahead: tuple[BlockDesc, BlockDesc]
    ) -> tuple[tuple, StagedBlocks]:
        """A row without a mask over a kv axis of one block: one segment, the
        tiles that hold a padded key masked, then the drain."""
        traits = self.traits
        kv_carry, staged = self.step_block(
            0, row.visits, ahead, row.base_head, row.kv_carry
        )
        items = single_block_items(
            traits, self.padded_tiles(0, traits.num_k_compute_tiles)
        )
        state = self.pipeline.advance(
            row.start_state,
            items,
            row.ops.with_tiles(staged.slot, staged.key, 0, row.row_end),
        )
        self.drain_row(row, state)
        return kv_carry, staged

    def run_runtime_row(
        self, row: DenseRow, ahead: tuple[BlockDesc, BlockDesc]
    ) -> tuple[tuple, StagedBlocks]:
        """A row under the runtime layout: masked row-start segment, interior
        visits, masked visits, masked last body."""
        entry, staged = self.run_row_start(row, is_masked=True)
        kv_carry, probs, staged = self.run_interior(row, ahead, entry, staged)
        kv_carry, probs, staged = self.run_masked_visits(
            row, ahead, entry, (kv_carry, probs, staged)
        )
        assumed = self.loop_tail(row.ops.masked(), entry.state)
        self.run_row_end(row, probs, assumed, staged)
        return kv_carry, staged

    def run_full_row(
        self, row: DenseRow, ahead: tuple[BlockDesc, BlockDesc]
    ) -> tuple[tuple, StagedBlocks]:
        """A row without a mask: row-start segment, visits, last body."""
        entry, staged = self.run_row_start(row, is_masked=False)
        kv_carry, probs, staged = self.run_interior(row, ahead, entry, staged)
        self.run_row_end(
            row, probs, self.loop_tail(row.ops, entry.state), staged
        )
        return kv_carry, staged

    def run_static_row(
        self, row: DenseRow, ahead: tuple[BlockDesc, BlockDesc]
    ) -> tuple[tuple, StagedBlocks]:
        """A row under the static plan: left segments, then interior visits and
        right segments up to the drain."""
        plan = self.layout.plan
        if not (plan.left or plan.right):
            return self.run_full_row(row, ahead)
        kv_carry, state = row.kv_carry, row.start_state
        staged = None
        if plan.left:
            kv_carry, state, staged = self.run_segments(
                row,
                ahead,
                plan.left,
                0,
                (kv_carry, state),
                stages_next=bool(plan.right),
                writes_checkpoints=True,
            )
        if not plan.right:
            self.drain_row(row, state)
            return kv_carry, staged
        return self.run_interior_and_right(
            row._replace(kv_carry=kv_carry, start_state=state),
            ahead,
            staged,
        )

    def run_interior_and_right(
        self,
        row: DenseRow,
        ahead: tuple[BlockDesc, BlockDesc],
        left_blocks: StagedBlocks | None,
    ) -> tuple[tuple, StagedBlocks]:
        """A static row after its left segments, which left row.start_state and
        the blocks left_blocks (None without them): interior visits, right
        segments, drain. A row that visits no interior block starts on the right
        segments; both copies run to the drain, so each leaves the MXUs' staging
        registers empty."""
        plan = self.layout.plan

        def _right(kv_carry, state, first_segment):
            kv_carry, state, staged = self.run_segments(
                row,
                ahead,
                (first_segment, *plan.right[1:]),
                row.interior_end,
                (kv_carry, state),
                stages_next=False,
            )
            self.drain_row(row, state)
            return kv_carry, staged

        def _with_interior(kv_carry):
            interior_row = row._replace(kv_carry=kv_carry)
            entry, staged = self.run_row_start(
                interior_row, is_masked=bool(plan.left), prev=left_blocks
            )
            kv_carry, probs, _ = self.run_interior(
                interior_row, ahead, entry, staged
            )
            state = PipelineState(
                probs=probs, tail=self.loop_tail(row.ops, entry.state)
            )
            return _right(kv_carry, state, plan.right[0])

        def _without_interior(kv_carry):
            segment = plan.right_row_start or plan.right[0]
            return _right(kv_carry, row.start_state, segment)

        if self.always_visits_interior:
            return _with_interior(row.kv_carry)
        if self.layout.never_visits_interior:
            return _without_interior(row.kv_carry)
        return lax.cond(
            row.interior_end > row.interior_start,
            _with_interior,
            _without_interior,
            row.kv_carry,
        )

    @property
    def always_visits_interior(self) -> bool:
        """Whether every row of a static plan without a left edge visits an
        interior block before its right edge."""
        plan, schedule = self.layout.plan, self.schedule
        if plan.left:
            return False
        frame = schedule.first_row * self.layout.blocks_per_row
        return frame + plan.right[0].block > 0

    def replay_unit(
        self,
        ops: DenseFragmentOps,
        visit: jax.Array | int,
        blocks: tuple[StagedBlocks, StagedBlocks],
        result_slots: tuple[jax.Array, ...],
    ) -> None:
        """Recompute check unit visit - 1 and visit visit's first body, in
        blocks's blocks, with the exact rebasing softmax, from the unit's
        checkpoint or the row start. Visit visit's checkpoint is rewritten and
        rebased to the anchor the first body raised."""
        unit = visit - 1
        prev, staged = blocks
        checkpoints = self.row_checkpoints

        @pl.when(unit > 0)
        def _restore():
            checkpoints.copy_checkpoint(unit, is_restore=True)

        self.replay_block(
            ops,
            (prev.slot, prev.key),
            self.traits.num_k_compute_tiles,
            unit == 0,
            result_slots,
        )
        checkpoints.copy_checkpoint(visit, is_restore=False)
        checkpoints.save_anchor(unit)
        self.replay_block(
            ops, (staged.slot, staged.key), LOOP_BODY_TILES, False, result_slots
        )
        checkpoints.rebase_checkpoint(visit, unit)

    def replay_block(
        self,
        ops: DenseFragmentOps,
        block: tuple[jax.Array, jax.Array],
        num_tiles: jax.Array | int,
        starts_row: jax.Array | bool,
        result_slots: tuple[jax.Array, ...],
    ) -> None:
        """Recompute the first num_tiles tiles of the kv block at block's (slot,
        first key) into the row state, masked, leaving lane partial sums.
        starts_row: tile 0 starts every fragment's row."""
        kv_slot, block_key = block

        def _replay_tile(tile, carry):
            tile_ops = ops.with_tiles(kv_slot, block_key, tile)
            is_row_start = jnp.logical_and(starts_row, tile == 0)
            bounds = tile_bounds(
                self.geometry, block_key + tile * MXU_COLUMNS
            ).from_row(ops.q_first)

            def _replay_fragment(fragment, fragment_carry):
                self.replay_fragment(
                    tile_ops, fragment, is_row_start, bounds, result_slots
                )
                return fragment_carry

            return lax.fori_loop(0, self.num_fragments, _replay_fragment, carry)

        lax.fori_loop(0, num_tiles, _replay_tile, None)

    def replay_fragment(
        self,
        ops: DenseFragmentOps,
        fragment: jax.Array,
        is_row_start: jax.Array,
        bounds: TileBounds,
        result_slots: tuple[jax.Array, ...],
    ) -> None:
        """Fold fragment fragment against tile key 0 of ops into the row state
        with the exact rebasing softmax, both matmuls on QK_MXU through
        result_slots; bounds are the tile's."""
        state = self.state
        score_slot, value_slot = result_slots
        item = WorkItem(fragment=fragment, tile=0)
        first_row = fragment * FRAGMENT_ROWS
        scores = replay_matmul(
            score_slot, ops.q_fragment(item), ops.k_tile(0), transpose=True
        )
        rows = fragment_rows(fragment)
        fold_replayed_fragment(
            state.row_max.at[rows],
            state.row_sum.at[MXU_ROW_PARITY, rows],
            state.o.at[MXU_ROW_PARITY, rows],
            masked_logits(self.variant, scores, bounds, first_row),
            is_row_start,
            value_slot,
            partial(ops.v_tile, 0),
            takes_anchor=(
                bounds.takes_anchor(first_row)
                if self.geometry.has_deferred_anchors
                else None
            ),
        )

    @property
    def row_checkpoints(self) -> RowCheckpoints:
        """The checkpoints of a row's check units."""
        return RowCheckpoints(
            checkpoint_sum=self.check_refs.checkpoint_sum,
            checkpoint_o=self.check_refs.checkpoint_o,
            row_sum=self.state.row_sum,
            o=self.state.o,
            parity=MXU_ROW_PARITY,
            row_anchor=lambda rows: self.state.row_max[rows, :],
        )

    def replay_row_end(
        self, row: DenseRow, finish: RowFinish, slot: jax.Array
    ) -> None:
        """Recompute row's final unit after its drain, from its checkpoint or
        the row start. Each block is staged again into slot, the row's last
        block's, which the last visit leaves as it found it."""
        final_visit = self.final_unit(row)
        visits = row.visits
        # A drained pipeline holds no result.
        result_slots = self.pipeline.qk_slots[:REPLAY_RESULT_SLOTS]
        if self.has_checked_units:

            @pl.when(final_visit > 0)
            def _restore():
                self.row_checkpoints.copy_checkpoint(
                    final_visit, is_restore=True
                )

        def _replay_visit(visit, carry):
            block = visits.block_of(visit)
            self.blocks.start(self.blocks.desc(finish.base_head, block), slot)
            wait_rotated_kv_block(self.traits, self.kv_refs, slot)
            num_tiles = self.traits.num_k_compute_tiles
            if self.last_block_keys is not None:
                num_tiles = jnp.where(
                    block == self.schedule.num_kv_blocks - 1,
                    self.last_block_tiles,
                    num_tiles,
                )
            self.replay_block(
                row.ops,
                (slot, block * self.traits.bkv),
                num_tiles,
                visit == 0,
                result_slots,
            )
            return carry

        lax.fori_loop(final_visit, visits.num_visits, _replay_visit, None)

    def normalize_row_out(self, out_slot: jax.Array) -> None:
        """Normalize a replayed row's output into its stage slot, which the row
        already waited for."""
        traits, variant, state = self.traits, self.variant, self.state
        bq_compute, o_stage = traits.bq_compute, self.out_refs.o_stage

        def _normalize_tile(q_tile, carry):
            rows = pl.ds(
                pl.multiple_of(q_tile * bq_compute, bq_compute), bq_compute
            )
            scale = jnp.tile(
                variant.output_scale(
                    state.row_sum[MXU_ROW_PARITY, rows, :], self.operands
                ),
                (1, traits.head_dim_v_repeats),
            )
            o_stage[out_slot, rows, :] = (
                state.o[MXU_ROW_PARITY, rows, :] * scale
            ).astype(o_stage.dtype)
            return carry

        lax.fori_loop(0, traits.num_q_compute_tiles, _normalize_tile, None)

    def advance_q_row(
        self,
        row: tuple[jax.Array, jax.Array],
        next_row: tuple[jax.Array, jax.Array],
        q_carry: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[tuple[jax.Array, jax.Array], tuple]:
        """advance_dense_q_row on the loop's Q ring. The staged Q stays
        unscaled: the scores take the Q scale."""
        return advance_dense_q_row(
            self.traits,
            self.schedule,
            self.q_refs,
            partial(wait_rotated_q_row, self.traits, self.q_refs),
            row,
            next_row,
            q_carry,
        )

    def run_row(self, g: jax.Array, si: jax.Array, carry: tuple) -> tuple:
        """A row: one pipeline from its first tile to its drain, then the guard
        replay, the lse and the output copy. carry is dense_fwd_body's row
        carry."""
        traits, blocks = self.traits, self.blocks
        q_loaded, q_inflight, q_slot, kv_carry, num_out_copies = carry
        next_g, next_si = blocks.row_after(g, si)
        first_next_desc, second_next_desc, _ = blocks.first_two_blocks(
            next_g, next_si
        )
        (base_head, q_tok), (q_loaded, q_inflight, q_slot) = self.advance_q_row(
            (g, si), (next_g, next_si), (q_loaded, q_inflight, q_slot)
        )
        finish = RowFinish(
            is_live=jnp.int32(1),
            si=si,
            q_tok=q_tok,
            base_head=base_head,
            num_blend_rows=jnp.int32(0),
            prev_out_tok=jnp.int32(0),
            out_count=num_out_copies,
            parity=MXU_ROW_PARITY,
        )
        out_slot = row_out_slot(traits, finish)
        # The row's last items write its stage slot.
        wait_row_out_slot(traits, self.out_refs, finish, out_slot)
        sink = self.row_sink(base_head)
        if zeroes_row_start(self.num_fragments):
            o_ref = self.state.o.at[MXU_ROW_PARITY]
            o_ref[...] = jnp.zeros(o_ref.shape, o_ref.dtype)
        row = self.dense_row(si, base_head, q_slot, kv_carry, out_slot, sink)
        if self.is_single_block_row:
            run = self.run_single_block_row
        elif self.layout.plan is None:
            run = self.run_runtime_row
        else:
            run = self.run_static_row
        kv_carry, last = run(row, (first_next_desc, second_next_desc))
        self.finish_row(g, row, finish, sink, last.slot)
        return q_loaded, q_inflight, q_slot, kv_carry, num_out_copies + 1

    def dense_row(
        self,
        si: jax.Array,
        base_head: jax.Array,
        q_slot: jax.Array,
        kv_carry: tuple,
        out_slot: jax.Array,
        sink: jax.Array | None,
    ) -> DenseRow:
        """Row si of the head at base_head as it starts, its Q at q_slot."""
        visits = self.layout.visits(si)
        interior_start, interior_end, num_masked = self.layout.interior_visits(
            si, visits, has_masked_interior=self.has_masked_interior
        )
        return DenseRow(
            visits=visits,
            base_head=base_head,
            ops=self.row_ops(q_slot, self.layout.q_first(si)),
            kv_carry=kv_carry,
            interior_start=interior_start,
            interior_end=interior_end,
            num_masked=num_masked,
            row_end=RowEnd(out_slot, sink, self.row_end_sum),
            start_state=self.pipeline.idle_state(),
        )

    def row_sink(self, base_head: jax.Array) -> jax.Array | None:
        """The sink logit of the head at base_head; None without sinks."""
        if not self.variant.has_sinks:
            return None
        return head_sinks(
            self.traits,
            self.operands.sinks_ref,
            self.schedule.heads_outer_batch,
            base_head,
        )[0]

    def row_ops(
        self, q_slot: jax.Array, q_first: jax.Array
    ) -> DenseFragmentOps:
        """The ops of the row whose Q, from row q_first, is staged at q_slot."""
        return DenseFragmentOps(
            self.traits,
            self.variant,
            self.operands,
            self.q_refs,
            self.kv_refs,
            self.state,
            self.out_refs.o_stage,
            q_slot,
            self.geometry,
            q_first,
        )

    def finish_row(
        self,
        g: jax.Array,
        row: DenseRow,
        finish: RowFinish,
        sink: jax.Array | None,
        last_slot: jax.Array,
    ) -> None:
        """A row's lse, the replay of its final unit when its finished row sums
        flag the guard, and its output copy. last_slot holds its last block."""
        traits, variant = self.traits, self.variant
        out_slot = row_out_slot(traits, finish)
        row_guard = guard_flag(traits, self.row_end_sum[...])
        if variant.return_lse:
            finish_row_lse(
                traits,
                variant,
                self.state,
                self.out_refs,
                g,
                finish,
                None if sink is None else [sink],
                None,
            )

        @pl.when(row_guard > 0)
        def _replay():
            self.replay_row_end(row, finish, last_slot)
            finish_row_sums(traits, self.state, MXU_ROW_PARITY)
            finish_row_lse_and_sinks(
                traits,
                variant,
                self.operands,
                self.schedule,
                self.state,
                self.out_refs,
                g,
                finish,
            )
            self.normalize_row_out(out_slot)

        o_copy(
            traits, self.out_refs, finish.base_head, finish.q_tok, out_slot
        ).start()


def make_dense_mxu_loop(
    *,
    traits: FwdTraits,
    variant: Variant,
    operands: VariantOperands,
    schedule: FwdSchedule,
    kv_refs: KVRefs,
    loop_refs: tuple,
    num_valid_keys: int | None = None,
) -> tuple[DenseMxuLoop, DenseBlockStream]:
    """The explicit loop of a dense build and its kv block stream. loop_refs is
    (q_refs, out_refs, state, row_end_sum, check_refs)."""
    q_refs, out_refs, state, row_end_sum, check_refs = loop_refs
    pipeline = MxuPipeline.allocate(
        has_mode_free_accumulate=zeroes_row_start(traits.bq // FRAGMENT_ROWS)
    )
    layout = dense_row_layout(
        traits,
        variant,
        schedule,
        variant.mask_offset(operands),
        pipeline.delays,
        num_valid_keys,
    )
    blocks = DenseBlockStream(
        traits=traits,
        schedule=schedule,
        kv_refs=kv_refs,
        row_visits=layout.visits,
    )
    loop = DenseMxuLoop(
        traits=traits,
        variant=variant,
        operands=operands,
        schedule=schedule,
        pipeline=pipeline,
        blocks=blocks,
        layout=layout,
        q_refs=q_refs,
        kv_refs=kv_refs,
        out_refs=out_refs,
        state=state,
        row_end_sum=row_end_sum,
        check_refs=check_refs,
        num_valid_keys=num_valid_keys,
    )
    return loop, blocks
