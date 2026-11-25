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

"""Row-state blocks of a forward main loop on the explicit MXU pipeline.

A build with traits.uses_mxu_pipeline runs its fragments through
mxu_pipeline.py: a work item is FRAGMENT_ROWS q rows of one pipeline head
against one kv tile of MXU_COLUMNS keys. Its loop (kernels/fwd_dense_mxu.py,
fwd_varlen_mxu.py or fwd_paged_mxu.py) keeps fwd_blocks.RowStateRefs, with
three differences from the dot_general pipeline while a row runs:

- A row holds one parity of row_sum and o from its first tile to its finish.
- row_sum holds lane partial sums: lane l of a row sums its probabilities at
  the keys l mod NUM_LANES of every tile. finish_row_sums reduces them once
  per row, before the row finish reads them.
- The overflow guard is checked once per row and once per check unit of the
  tiles before its final unit; a flagged unit is replayed from the checkpoint
  of its entry state with the exact rebasing softmax.

MaskGeometry says which q rows see which keys. Under a left window bound a
row holds UNANCHORED_ROW_MAX until the tile of its first visible key, where
it takes its anchor.
"""

import dataclasses
import functools
import operator
from collections.abc import Callable, Hashable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from flywheel_tpu.core.tiling import round_up
from flywheel_tpu.core.tpu import MXU_COLUMNS, NUM_LANES, NUM_SUBLANES
from flywheel_tpu.softmax_attention.config import (
    DEFAULT_MASK_VALUE,
    BlockSizes,
    QKVLayout,
)
from flywheel_tpu.softmax_attention.fwd_blocks import (
    OutRefs,
    RowFinish,
    RowStateRefs,
    fold_out_sinks,
    head_sinks,
    row_out_slot,
    stage_row_lse,
    stage_row_out,
)
from flywheel_tpu.softmax_attention.mxu_pipeline import (
    FRAGMENT_ROWS,
    NUM_SLOTS,
    QK_MXU,
    STAGING_REGISTER,
    FragmentOps,
    PipelineDelays,
    PipelineShape,
    WorkItem,
)
from flywheel_tpu.softmax_attention.schedule import (
    FwdSchedule,
    MxuRowTiles,
    ScheduleRow,
    mxu_row_tiles,
)
from flywheel_tpu.softmax_attention.softmax import (
    UNANCHORED_ROW_BOUND,
    UNANCHORED_ROW_MAX,
)
from flywheel_tpu.softmax_attention.traits import FwdTraits
from flywheel_tpu.softmax_attention.variant import Variant, VariantOperands

# The kv tiles of one rolled loop body: a loop back edge stalls the pipeline's
# last latency chains, so a longer body spreads it thinner.
LOOP_BODY_TILES = 2
# A row's start segment holds its first tile and the next one: a loop body
# finishes the items in flight as plain ones, so a plain tile precedes it.
ROW_START_TILES = 2
# The rows of a per-seq loop alternate between two row states: the previous
# row finishes after the next one has started.
NUM_ROW_PARITIES = 2
# The first row takes parity 0; what runs before it takes the other one.
ENTRY_PARITY = NUM_ROW_PARITIES - 1
# The most rows of row state a row group of several rows holds.
MAX_GROUP_STATE_ROWS = 4096
# The row groups whose rows a loop's Q and output rings hold: the running
# group's and the next one's.
QUEUED_GROUPS = 2
# The rows finish_row_sums reduces a step.
ROW_SUM_CHUNK_ROWS = NUM_LANES
# A replay between two segments pushes into the staging register the pipeline
# leaves empty there.
REPLAY_STAGING_REGISTER = 1 - STAGING_REGISTER
# A replay's QK result slots: the scores, then probs @ V.
REPLAY_RESULT_SLOTS = 2
# The checkpoints of a unit and the unit after it.
NUM_CHECKPOINTS = 2
# The shallowest pipeline: a QK latency, a carried fragment and a PV latency
# of one tick each.
MIN_PV_POP = 3
# A check unit: a first body, whose end checks the unit before, and a last
# body. Its first body may be a row's start segment.
UNIT_TILES = 2 * LOOP_BODY_TILES
assert ROW_START_TILES == LOOP_BODY_TILES
# The smaller slot rings pipeline_shape tries are the powers of two below it.
assert NUM_SLOTS & (NUM_SLOTS - 1) == 0, NUM_SLOTS
# A row's final unit holds its last FINAL_UNIT_SPAN whole check units and the
# tiles after them.
FINAL_UNIT_SPAN = 2
# A row index past every q row a build addresses, and one before them all.
HIDDEN_ROW = 1 << 30
UNBOUNDED_ROW = -HIDDEN_ROW


def mxu_pipeline_applies(
    *,
    block_sizes: BlockSizes,
    head_dim: int,
    head_dim_v: int,
    transposed_pv: bool,
) -> bool:
    """Whether the chip and the shapes allow the explicit MXU pipeline: MXU
    result accumulators, K, V and a tile of scores as wide as an MXU, and
    tokens on sublanes. A kernel adds its own conditions."""
    if jax.default_backend() != "tpu":
        return False
    tpu_info = pltpu.get_tpu_info()
    if not tpu_info.num_accumulators or tpu_info.mxu_column_size != MXU_COLUMNS:
        return False
    return (
        head_dim == head_dim_v == MXU_COLUMNS
        and block_sizes.qkv_layout == QKVLayout.HEAD_DIM_MINOR
        and not transposed_pv
    )


def _maximum(a: jax.Array | int, b: jax.Array | int) -> jax.Array | int:
    if isinstance(a, int) and isinstance(b, int):
        return max(a, b)
    return jnp.maximum(a, b)


def _minimum(a: jax.Array | int, b: jax.Array | int) -> jax.Array | int:
    if isinstance(a, int) and isinstance(b, int):
        return min(a, b)
    return jnp.minimum(a, b)


def _select(
    predicate: jax.Array | bool, a: jax.Array | int, b: jax.Array | int
) -> jax.Array | int:
    if isinstance(predicate, bool):
        return a if predicate else b
    return jnp.where(predicate, a, b)


class MaskGeometry(NamedTuple):
    """Which q rows see which keys: row i sees key j iff row_floor <= i <=
    row_ceiling, key_floor <= j < key_end and i + shift - left <= j <= i +
    shift + right, a None bound unbounded. Rows and keys count in one
    coordinate system; row_ceiling takes a left bound. A field is a Python
    int when the build fixes it, else an int32 scalar."""

    shift: jax.Array | int
    left: int | None
    right: int | None
    key_floor: jax.Array | int = 0
    key_end: jax.Array | int | None = None
    row_floor: jax.Array | int | None = None
    row_ceiling: jax.Array | int | None = None

    @property
    def has_deferred_anchors(self) -> bool:
        """Whether a row's first visible key depends on the row, so its first
        tiles may hide every key from it."""
        return self.left is not None

    def first_visible_key(self, row: jax.Array | int) -> jax.Array | int:
        """The first key row sees, if it sees one."""
        if self.left is None:
            return self.key_floor
        return _maximum(row + self.shift - self.left, self.key_floor)

    def last_visible_key(self, row: jax.Array | int) -> jax.Array | int:
        """The last key row sees, if it sees one; HIDDEN_ROW unbounded."""
        last = HIDDEN_ROW
        if self.right is not None:
            last = row + self.shift + self.right
        if self.key_end is not None:
            last = _minimum(last, self.key_end - 1)
        return last

    def sees_keys(
        self, row: jax.Array | int, first_key: jax.Array | int, last_key: int
    ) -> jax.Array | bool:
        """Whether row sees a key of first_key..last_key."""
        first = _maximum(self.first_visible_key(row), first_key)
        last = _minimum(self.last_visible_key(row), last_key)
        sees = first <= last
        if self.row_floor is not None:
            sees = sees & (row >= self.row_floor)
        if self.row_ceiling is not None:
            sees = sees & (row <= self.row_ceiling)
        return sees

    def key_rows(
        self, keys: jax.Array, owned_key: jax.Array | int | None = None
    ) -> tuple[jax.Array, jax.Array | None]:
        """Per key of keys, the first and the last q row that sees it (None:
        no left bound). A key below owned_key, which an earlier tile counted,
        is hidden."""
        if self.right is None:
            first_row = (
                UNBOUNDED_ROW if self.row_floor is None else self.row_floor
            )
            lower = jnp.full(keys.shape, first_row, jnp.int32)
        else:
            lower = keys - self.shift - self.right
            if self.row_floor is not None:
                lower = jnp.maximum(lower, self.row_floor)
        key_floor = self.key_floor
        if owned_key is not None:
            key_floor = _maximum(key_floor, owned_key)
        is_key = keys >= key_floor
        if self.key_end is not None:
            is_key = jnp.logical_and(is_key, keys < self.key_end)
        lower = jnp.where(is_key, lower, HIDDEN_ROW)
        if self.left is None:
            assert self.row_ceiling is None, self
            return lower, None
        upper = keys - self.shift + self.left
        if self.row_ceiling is not None:
            upper = jnp.minimum(upper, self.row_ceiling)
        return lower, upper

    def anchor_rows(
        self,
        first_key: jax.Array | int,
        owned_key: jax.Array | int | None = None,
    ) -> tuple[jax.Array | int, jax.Array | int]:
        """(first, last): the q rows whose first visible key lies in the
        tile of MXU_COLUMNS keys from first_key, from owned_key on when
        given; empty when last < first."""
        last_key = first_key + MXU_COLUMNS - 1
        if owned_key is not None:
            first_key = _maximum(first_key, owned_key)
        floor_in_tile = (self.key_floor >= first_key) & (
            self.key_floor <= last_key
        )
        if self.left is None:
            first = _select(floor_in_tile, UNBOUNDED_ROW, HIDDEN_ROW)
            last = HIDDEN_ROW
        else:
            first = _select(
                self.key_floor < first_key,
                first_key - self.shift + self.left,
                UNBOUNDED_ROW,
            )
            last = _select(
                self.key_floor <= last_key,
                last_key - self.shift + self.left,
                UNBOUNDED_ROW,
            )
            if self.key_end is not None:
                last = _minimum(last, self.key_end - 1 - self.shift + self.left)
        if self.right is not None:
            first = _maximum(first, self.key_floor - self.shift - self.right)
        if self.row_floor is not None:
            first = _maximum(first, self.row_floor)
        if self.row_ceiling is not None:
            last = _minimum(last, self.row_ceiling)
        return first, last


class TileBounds(NamedTuple):
    """One kv tile's mask: per key, (1, MXU_COLUMNS), the first and last q
    row that see it (upper None: no left bound), and the rows anchor_first..
    anchor_last that take their anchor in the tile."""

    lower: jax.Array
    upper: jax.Array | None
    anchor_first: jax.Array | int
    anchor_last: jax.Array | int

    def from_row(self, first_row: jax.Array | int) -> "TileBounds":
        """The bounds with rows counted from first_row, a q block's first
        row, so a fragment's first row is static."""
        return TileBounds(
            lower=self.lower - first_row,
            upper=None if self.upper is None else self.upper - first_row,
            anchor_first=self.anchor_first - first_row,
            anchor_last=self.anchor_last - first_row,
        )

    def from_fragment_rows(self) -> "TileBounds":
        """The bounds counted from each fragment row, (FRAGMENT_ROWS,
        MXU_COLUMNS): visible_keys then compares a fragment's first row."""
        row_ids = lax.broadcasted_iota(
            jnp.int32, (FRAGMENT_ROWS, MXU_COLUMNS), 0
        )
        return self._replace(
            lower=self.lower - row_ids,
            upper=None if self.upper is None else self.upper - row_ids,
        )

    def visible_keys(self, first_row: jax.Array | int) -> jax.Array:
        """The (FRAGMENT_ROWS, MXU_COLUMNS) keys the fragment of rows from
        first_row sees."""
        if self.lower.shape[0] == FRAGMENT_ROWS:
            rows = first_row
        else:
            rows = first_row + lax.broadcasted_iota(
                jnp.int32, (FRAGMENT_ROWS, MXU_COLUMNS), 0
            )
        is_visible = rows >= self.lower
        if self.upper is not None:
            is_visible = jnp.logical_and(is_visible, rows <= self.upper)
        return is_visible

    def takes_anchor(self, first_row: jax.Array | int) -> jax.Array:
        """(FRAGMENT_ROWS, NUM_LANES): which rows of the fragment from
        first_row take their anchor in the tile, every lane a row's flag."""
        rows = first_row + lax.broadcasted_iota(
            jnp.int32, (FRAGMENT_ROWS, NUM_LANES), 0
        )
        return jnp.logical_and(
            rows >= self.anchor_first, rows <= self.anchor_last
        )


def pivot_kept_rows(window: tuple[int | None, int | None]) -> int | None:
    """The most q rows a per-seq block of an explicit build keeps under
    window, so that every kept row sees the first key the last kept row
    sees; None keeps whole blocks."""
    left, right = window
    if left is None or right is None:
        return None
    return left + right + 1


def pivot_order(pivot: jax.Array | int, step: jax.Array | int) -> jax.Array:
    """The unit, counted from a row's first, that the row runs at step step:
    down from its pivot unit to the first, then up from the pivot to the
    last."""
    return jnp.where(step <= pivot, pivot - step, step)


def per_seq_geometry(schedule: FwdSchedule, ctx: ScheduleRow) -> MaskGeometry:
    """The mask of the per-seq schedule row ctx, in packed coordinates: its
    sequence's window around the bottom-right diagonal, over the sequence's
    keys up to the q block's frontier. Under a left window bound only the
    block's kept rows see keys."""
    left, right = schedule.window
    geometry = MaskGeometry(
        shift=ctx.bottom_right_offset,
        left=left,
        right=right,
        key_floor=ctx.kv_start,
        key_end=ctx.kv_frontier,
    )
    return kept_rows_geometry(geometry, ctx.kept_start, ctx.kept_end)


def kept_rows_geometry(
    geometry: MaskGeometry,
    kept_start: jax.Array | int,
    kept_end: jax.Array | int,
) -> MaskGeometry:
    """geometry with only rows [kept_start, kept_end) seeing keys under a
    left window bound; unchanged without one."""
    if geometry.left is None:
        return geometry
    return geometry._replace(row_floor=kept_start, row_ceiling=kept_end - 1)


def tile_bounds(
    geometry: MaskGeometry,
    first_key: jax.Array | int,
    owned_key: jax.Array | int | None = None,
) -> TileBounds:
    """The mask of the kv tile of MXU_COLUMNS keys from first_key under
    geometry; keys below owned_key are hidden (MaskGeometry.key_rows)."""
    keys = first_key + lax.broadcasted_iota(jnp.int32, (1, MXU_COLUMNS), 1)
    lower, upper = geometry.key_rows(keys, owned_key)
    anchor_first, anchor_last = geometry.anchor_rows(first_key, owned_key)
    return TileBounds(lower, upper, anchor_first, anchor_last)


def masked_logits(
    variant: Variant,
    scores: jax.Array,
    bounds: TileBounds,
    first_row: jax.Array | int,
) -> jax.Array:
    """One fragment's raw f32 scores as log2-unit logits, the mask value
    wherever bounds hide a key from a row of the fragment from first_row."""
    return jnp.where(
        bounds.visible_keys(first_row),
        scaled_logits(variant, scores),
        DEFAULT_MASK_VALUE,
    )


class ItemClass(NamedTuple):
    """A static (fragment, kv tile) pair's class under a static geometry:
    is_live, a row sees a key; is_interior, every row sees every key; and
    has_anchor, a row takes its anchor in the tile."""

    is_live: bool
    is_interior: bool
    has_anchor: bool


def item_class(
    geometry: MaskGeometry, first_row: int, first_key: int
) -> ItemClass:
    """The class of the fragment of FRAGMENT_ROWS rows from first_row
    against the tile of MXU_COLUMNS keys from first_key, every field of
    geometry a Python int."""
    last_key = first_key + MXU_COLUMNS - 1
    rows = range(first_row, first_row + FRAGMENT_ROWS)
    sees = [geometry.sees_keys(row, first_key, last_key) for row in rows]
    is_interior = all(
        geometry.sees_keys(row, first_key, first_key)
        and geometry.sees_keys(row, last_key, last_key)
        and geometry.first_visible_key(row) <= first_key
        for row in rows
    )
    anchor_first, anchor_last = geometry.anchor_rows(first_key)
    has_anchor = max(anchor_first, first_row) <= min(
        anchor_last, first_row + FRAGMENT_ROWS - 1
    )
    return ItemClass(any(sees), is_interior, has_anchor)


def mxu_block_sizes(block_sizes: BlockSizes) -> BlockSizes:
    """block_sizes as an explicit MXU build runs them: block_q padded to
    whole fragments, block_kv to whole kv tiles, the kv compute tile one MXU
    tile, and the q compute tile whole fragments that divide block_q."""
    block_q = round_up(block_sizes.block_q, FRAGMENT_ROWS)
    block_q_compute = round_up(block_sizes.block_q_compute, FRAGMENT_ROWS)
    while block_q % block_q_compute:
        block_q_compute -= FRAGMENT_ROWS
    return BlockSizes(
        block_q=block_q,
        block_kv=round_up(block_sizes.block_kv, MXU_COLUMNS),
        block_kv_compute=MXU_COLUMNS,
        block_q_compute=block_q_compute,
        num_stages=block_sizes.num_stages,
        qkv_layout=block_sizes.qkv_layout,
    )


def row_tiles(
    fragments_per_tile: int, shape: PipelineShape = PipelineShape()
) -> MxuRowTiles:
    """How a loop on a pipeline of shape groups the kv tiles of a row of
    fragments_per_tile fragments a tile (schedule.mxu_row_tiles)."""
    return mxu_row_tiles(
        fragments_per_tile,
        num_slots=shape.num_slots,
        qk_pop=shape.delays.qk_pop,
        pv_pop=shape.delays.pv_pop,
        min_start_tiles=ROW_START_TILES,
        min_body_tiles=LOOP_BODY_TILES,
    )


def _scaled_delays(pv_pop: int) -> PipelineDelays:
    """The default delays scaled down to pv_pop: carried fragments in the
    default ratio, at least one, equal QK and PV latencies around them, and
    the default push leads, V's at most pv_issue."""
    default = PipelineDelays()
    num_carried = max(pv_pop * default.num_carried_probs // default.pv_pop, 1)
    qk_pop = (pv_pop - num_carried) // 2
    pv_issue = qk_pop + num_carried
    return PipelineDelays(
        qk_pop=qk_pop,
        pv_issue=pv_issue,
        pv_pop=pv_pop,
        k_push_lead=default.k_push_lead,
        v_push_lead=min(default.v_push_lead, pv_issue),
    )


def _fits_block(
    shape: PipelineShape, fragments_per_tile: int, block_tiles: int
) -> bool:
    """Whether shape's ring holds every result plus a replay's, its start
    segment fits in a block of block_tiles tiles and its bodies tile it."""
    tiles = row_tiles(fragments_per_tile, shape)
    return (
        shape.has_enough_slots
        and shape.delays.qk_pop + REPLAY_RESULT_SLOTS <= shape.num_slots
        and tiles.start_tiles <= block_tiles
        and block_tiles % tiles.body_tiles == 0
    )


@functools.cache
def pipeline_shape(fragments_per_tile: int, block_tiles: int) -> PipelineShape:
    """The pipeline of rows of fragments_per_tile fragments a tile whose
    start segment runs inside one kv block of block_tiles tiles. A block of
    at least default pv_pop items takes the default pipeline, and the caller
    pads it to whole default bodies. A smaller block takes the deepest
    shallower pipeline that fits it, on the smallest ring that holds it."""
    default = PipelineShape()
    if block_tiles * fragments_per_tile >= default.delays.pv_pop:
        return default
    ring_sizes = [1 << power for power in range(NUM_SLOTS.bit_length())]
    for pv_pop in range(block_tiles * fragments_per_tile, MIN_PV_POP - 1, -1):
        for num_slots in ring_sizes:
            shape = PipelineShape(_scaled_delays(pv_pop), num_slots)
            if _fits_block(shape, fragments_per_tile, block_tiles):
                return shape
    raise ValueError(
        f"no explicit MXU pipeline runs {fragments_per_tile} fragments a tile"
        f" in kv blocks of {block_tiles} tiles."
    )


def group_capacity(
    fragments_per_tile: int, shape: PipelineShape = PipelineShape()
) -> int:
    """The most rows of a row group: one per tile of a start segment, or one
    when their row states would hold more than MAX_GROUP_STATE_ROWS rows."""
    start_tiles = row_tiles(fragments_per_tile, shape).start_tiles
    if start_tiles * fragments_per_tile * FRAGMENT_ROWS > MAX_GROUP_STATE_ROWS:
        return 1
    return start_tiles


def mxu_row_layout(fragments_per_tile: int) -> tuple[MxuRowTiles, int]:
    """(row tiles, group capacity) of rows of fragments_per_tile fragments
    a tile, as a schedule table groups them (schedule.mxu_row_groups)."""
    return row_tiles(fragments_per_tile), group_capacity(fragments_per_tile)


def num_row_states(fragments_per_tile: int) -> int:
    """The row states of a loop whose row groups alternate between
    NUM_ROW_PARITIES parities."""
    return NUM_ROW_PARITIES * group_capacity(fragments_per_tile)


def for_each_later_row(
    num_rows: jax.Array, row_fn: Callable[[jax.Array], None]
) -> None:
    """row_fn(member) for each row of a row group of num_rows rows past its
    first, in order; a row alone pays one branch."""

    @pl.when(num_rows > 1)
    def _later_rows():
        def _row(member, carry):
            row_fn(member)
            return carry

        lax.fori_loop(1, num_rows, _row, None)


def zeroes_row_start(
    fragments_per_tile: int, shape: PipelineShape = PipelineShape()
) -> bool:
    """Whether a row starts its accumulator at zero: a tile of fewer items
    than pv_pop leaves first-tile items in flight past the start segment,
    where a loop body finishes them as later ones."""
    return fragments_per_tile < shape.delays.pv_pop


def aligned_rows(row: jax.Array | int) -> jax.Array | int:
    """row, a multiple of FRAGMENT_ROWS, marked as one when traced."""
    if isinstance(row, int):
        return row
    return pl.multiple_of(row, FRAGMENT_ROWS)


def fragment_rows(fragment: jax.Array | int) -> pl.Slice:
    """The rows of the fragment at index fragment."""
    return pl.ds(aligned_rows(fragment * FRAGMENT_ROWS), FRAGMENT_ROWS)


def scaled_logits(variant: Variant, scores: jax.Array) -> jax.Array:
    """The log2-unit logits of raw f32 scores of an unscaled Q."""
    # Note (david): scaling each fragment's scores, in spare VALU slots,
    # saves each row a pass over its staged Q.
    return variant.transform_logits(scores * variant.q_scale)


def across_lanes(row_values: jax.Array) -> jax.Array:
    """row_values, one value a row, as a (rows, NUM_LANES) row state: every
    lane of a row holds the row's value."""
    return lax.broadcast_in_dim(
        row_values, (row_values.shape[0], NUM_LANES), (0,)
    )


def anchored_probs(scores: jax.Array, anchor: jax.Array) -> jax.Array:
    """The f32 probabilities of one fragment's log2-unit scores against
    anchor, the (FRAGMENT_ROWS, NUM_LANES) row maximum of its rows."""
    num_lane_tiles = scores.shape[-1] // NUM_LANES
    return jnp.exp2(scores - jnp.tile(anchor, (1, num_lane_tiles)))


def guard_flag(traits: FwdTraits, largest_sums: jax.Array) -> jax.Array:
    """1 if a row could hold a score past the overflow guard, given the
    largest row sums of its rows; a NaN or inf sum fails the compare too."""
    return jnp.any(
        jnp.logical_not(largest_sums <= traits.guard_detect_bound)
    ).astype(jnp.int32)


def first_final_unit(num_units: jax.Array | int) -> jax.Array | int:
    """The first check unit of the final unit of a row of num_units whole
    check units (FINAL_UNIT_SPAN); 0 when the row has no checked unit."""
    if isinstance(num_units, int):
        return max(num_units - FINAL_UNIT_SPAN, 0)
    return jnp.maximum(num_units - FINAL_UNIT_SPAN, 0)


def lane_partial_sums(probs: jax.Array) -> jax.Array:
    """The (rows, NUM_LANES) lane partial sums of (rows, keys)
    probabilities: lane l sums the keys l mod NUM_LANES."""
    num_lane_tiles = probs.shape[-1] // NUM_LANES
    return functools.reduce(
        operator.add,
        (
            probs[:, tile * NUM_LANES : (tile + 1) * NUM_LANES]
            for tile in range(num_lane_tiles)
        ),
    )


@dataclasses.dataclass(eq=False)
class LargestSums:
    """The (NUM_SUBLANES, NUM_LANES) running maximum of the lane partial
    sums a run of softmaxes wrote; guard_flag reads it after them."""

    value: jax.Array | None = None

    def fold(self, sums: jax.Array) -> None:
        largest = sums.reshape(-1, NUM_SUBLANES, NUM_LANES).max(axis=0)
        if self.value is None:
            self.value = largest
        else:
            self.value = jnp.maximum(self.value, largest)


class UnitCheckpoint(NamedTuple):
    """A check unit's last tile, tile key tile: its items also write the row
    sums and accumulator rows they leave, the next unit's entry state, into
    the checkpoint buffers row_sum and o, and fold the row sums into largest
    when given."""

    tile: int
    row_sum: jax.Array
    o: jax.Array
    largest: LargestSums | None = None


def select_row_start(
    is_row_start: bool | jax.Array,
    at_start: Callable[[], jax.Array],
    after_start: Callable[[], jax.Array],
) -> jax.Array:
    """at_start() for a fragment's first tile, after_start() for a later
    one; a traced is_row_start computes both and selects."""
    if isinstance(is_row_start, bool):
        return at_start() if is_row_start else after_start()
    return jnp.where(is_row_start, at_start(), after_start())


def fragment_anchor(
    row_max_ref: jax.Array,
    rows: pl.Slice,
    scores: jax.Array,
    *,
    is_row_start: bool | jax.Array,
    takes_anchor: jax.Array | None,
) -> jax.Array:
    """The anchor of one fragment's rows for its masked log2-unit scores,
    written back to row_max_ref when it changes.

    Without takes_anchor a row start takes its row maximum and a later tile
    reads the anchor. With takes_anchor, (FRAGMENT_ROWS, NUM_LANES), a row
    takes its row maximum where its flag is set and otherwise holds
    UNANCHORED_ROW_MAX from its row start on, so its probabilities, sums and
    accumulator stay exactly 0 until it anchors."""
    if takes_anchor is None and is_row_start is False:
        return row_max_ref[rows, :]
    row_maximum = across_lanes(scores.max(axis=-1))
    if takes_anchor is None:
        anchor = select_row_start(
            is_row_start, lambda: row_maximum, lambda: row_max_ref[rows, :]
        )
    else:
        held = select_row_start(
            is_row_start,
            lambda: UNANCHORED_ROW_MAX,
            lambda: row_max_ref[rows, :],
        )
        anchor = jnp.where(takes_anchor, row_maximum, held)
    row_max_ref[rows, :] = anchor
    return anchor


def fold_fragment_scores(
    row_max_ref: jax.Array,
    row_sum_ref: jax.Array,
    rows: pl.Slice,
    scores: jax.Array,
    *,
    is_row_start: bool | jax.Array,
    takes_anchor: jax.Array | None = None,
    largest: LargestSums | None = None,
    checkpoint_sum: jax.Array | None = None,
) -> jax.Array:
    """Fold one fragment's masked (FRAGMENT_ROWS, MXU_COLUMNS) log2-unit
    scores into its rows of a pipeline head's anchor and lane partial sums;
    returns its bf16 probabilities. At a row start (static or traced) its
    lane sums start the row sums. checkpoint_sum also takes the sums
    written, and largest collects them."""
    anchor = fragment_anchor(
        row_max_ref,
        rows,
        scores,
        is_row_start=is_row_start,
        takes_anchor=takes_anchor,
    )
    probs = anchored_probs(scores, anchor)
    lane_sums = lane_partial_sums(probs)
    new_sums = select_row_start(
        is_row_start,
        lambda: lane_sums,
        lambda: row_sum_ref[rows, :] + lane_sums,
    )
    row_sum_ref[rows, :] = new_sums
    if checkpoint_sum is not None:
        checkpoint_sum[rows, :] = new_sums
    if largest is not None:
        largest.fold(new_sums)
    return probs.astype(jnp.bfloat16)


def fold_unit_end_scores(
    row_max_ref: jax.Array,
    row_sum_ref: jax.Array,
    rows: pl.Slice,
    scores: jax.Array,
    unit_end: UnitCheckpoint,
    takes_anchor: jax.Array | None = None,
) -> jax.Array:
    """fold_fragment_scores past a row's start for an item of unit_end's
    tile, loading the row sums before it sums the lanes."""
    anchor = fragment_anchor(
        row_max_ref,
        rows,
        scores,
        is_row_start=False,
        takes_anchor=takes_anchor,
    )
    probs = anchored_probs(scores, anchor)
    sums = row_sum_ref[rows, :] + lane_partial_sums(probs)
    row_sum_ref[rows, :] = sums
    unit_end.row_sum[rows, :] = sums
    unit_end.largest.fold(sums)
    return probs.astype(jnp.bfloat16)


def accumulate_fragment(
    o_ref: jax.Array,
    rows: pl.Slice,
    weighted_values: jax.Array,
    *,
    is_row_start: bool | jax.Array,
    checkpoint_o: jax.Array | None = None,
) -> None:
    """Add one fragment's f32 probs @ V to its rows of a pipeline head's
    accumulator; checkpoint_o also takes the rows written."""
    values = select_row_start(
        is_row_start,
        lambda: weighted_values,
        lambda: o_ref[rows, :] + weighted_values,
    )
    o_ref[rows, :] = values
    if checkpoint_o is not None:
        checkpoint_o[rows, :] = values


@dataclasses.dataclass(frozen=True, eq=False)
class FlaggedTailOps:
    """The ops with which the body after a checked first body finishes it.
    Once is_flagged replayed that body's tiles, its pending softmaxes add
    nothing to the row sums and give zero probabilities, and its other
    pending accumulates add nothing (a select: the flagged values may be
    inf). A row takes its anchor in a pending tile only when not flagged: a
    replay already took it. zeroed holds the (fragment, tile) keys of the
    softmaxes run; no pending softmax starts or ends a row (is_row_bound).
    """

    ops: FragmentOps
    is_flagged: jax.Array
    is_row_bound: Callable[[Hashable], bool]
    zeroed: set = dataclasses.field(default_factory=set)

    @property
    def v_handle_words(self) -> int:
        return self.ops.v_handle_words

    def v_handle(self, tile: Hashable) -> tuple[jax.Array, ...]:
        return self.ops.v_handle(tile)

    def v_tile_at(self, handle: tuple[jax.Array, ...]) -> jax.Array:
        return self.ops.v_tile_at(handle)

    def k_tile(self, tile: Hashable) -> jax.Array:
        return self.ops.k_tile(tile)

    def v_tile(self, tile: Hashable) -> jax.Array:
        return self.ops.v_tile(tile)

    def q_fragment(self, item: WorkItem) -> jax.Array:
        return self.ops.q_fragment(item)

    def softmax(self, item: WorkItem, scores: jax.Array) -> jax.Array:
        assert not self.is_row_bound(item.mode), item
        self.zeroed.add((item.fragment, item.tile))
        ops, rows = self.ops, self.ops.rows(item)
        logits = ops.masked_scores(item, scores)
        anchor = ops.row_max[rows, :]
        takes_anchor = ops.takes_anchor(item)
        if takes_anchor is not None:
            anchor = jnp.where(
                jnp.logical_and(takes_anchor, jnp.logical_not(self.is_flagged)),
                across_lanes(logits.max(axis=-1)),
                anchor,
            )
            ops.row_max[rows, :] = anchor
        probs = anchored_probs(logits, anchor)
        row_sum_ref = ops.row_sum
        sums = row_sum_ref[rows, :]
        row_sum_ref[rows, :] = jnp.where(
            self.is_flagged, sums, sums + lane_partial_sums(probs)
        )
        return jnp.where(self.is_flagged, jnp.zeros_like(probs), probs).astype(
            jnp.bfloat16
        )

    def accumulate(self, item: WorkItem, weighted_values: jax.Array) -> None:
        if (item.fragment, item.tile) not in self.zeroed:
            weighted_values = jnp.where(
                self.is_flagged,
                jnp.zeros_like(weighted_values),
                weighted_values,
            )
        self.ops.accumulate(item, weighted_values)


def replay_matmul(
    slot: jax.Array, lhs: jax.Array, rhs: jax.Array, *, transpose: bool = False
) -> jax.Array:
    """One fragment's lhs @ rhs (rhs^T under transpose) on QK_MXU between two
    segments: rhs goes through REPLAY_STAGING_REGISTER and the f32 result
    through slot, a free QK result slot (MxuPipeline.free_slots)."""
    pltpu.matmul_push_rhs(
        rhs,
        staging_register=REPLAY_STAGING_REGISTER,
        mxu_index=QK_MXU,
        transpose=transpose,
    )
    pltpu.matmul_acc_lhs(slot, lhs, REPLAY_STAGING_REGISTER)
    return pltpu.matmul_pop(slot)


def unanchored_rescale(
    prev_anchor: jax.Array, new_anchor: jax.Array
) -> jax.Array:
    """The factor that moves a row's sums from prev_anchor to new_anchor: 0
    for a row unanchored at prev_anchor (UNANCHORED_ROW_MAX), whose state is
    0."""
    return jnp.where(
        prev_anchor >= UNANCHORED_ROW_BOUND,
        0.0,
        jnp.exp2(prev_anchor - new_anchor),
    )


def rebase_fragment(
    row_max_ref: jax.Array,
    row_sum_ref: jax.Array,
    scores: jax.Array,
    is_row_start: jax.Array,
    takes_anchor: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """The exact rebasing softmax of one fragment's masked scores, for a
    replay: raise the rows' anchor to cover them (or start it at a row
    start), rescale the lane partial sums to it and add the fragment's. The
    refs hold the fragment's rows; takes_anchor as in fragment_anchor.
    Returns the f32 probabilities, the accumulator's rescale and the rows
    whose state starts."""
    tile_max = across_lanes(scores.max(axis=-1))
    prev_max = row_max_ref[...]
    raised_max = jnp.maximum(prev_max, tile_max)
    if takes_anchor is None:
        new_max = jnp.where(is_row_start, tile_max, raised_max)
        starts = jnp.broadcast_to(is_row_start, tile_max.shape)
    else:
        new_max = jnp.where(
            takes_anchor,
            tile_max,
            jnp.where(is_row_start, UNANCHORED_ROW_MAX, raised_max),
        )
        starts = jnp.logical_or(is_row_start, takes_anchor)
    probs = anchored_probs(scores, new_max)
    rescale = unanchored_rescale(prev_max, new_max)
    lane_sums = lane_partial_sums(probs)
    row_sum_ref[...] = jnp.where(
        starts, lane_sums, rescale * row_sum_ref[...] + lane_sums
    )
    row_max_ref[...] = new_max
    return probs, rescale, starts


def accumulate_rebased(
    o_ref: jax.Array,
    weighted_values: jax.Array,
    rescale: jax.Array,
    starts: jax.Array,
) -> None:
    """Add a replayed fragment's f32 probs @ V to its rows of o_ref under
    rebase_fragment's rescale; a row whose state starts takes them alone."""
    lane_tiles = weighted_values.shape[-1] // NUM_LANES
    o_ref[...] = jnp.where(
        jnp.tile(starts, (1, lane_tiles)),
        weighted_values,
        jnp.tile(rescale, (1, lane_tiles)) * o_ref[...] + weighted_values,
    )


def fold_replayed_fragment(
    row_max_ref: jax.Array,
    row_sum_ref: jax.Array,
    o_ref: jax.Array,
    scores: jax.Array,
    is_row_start: jax.Array,
    value_slot: jax.Array,
    v_tile: Callable[[], jax.Array],
    takes_anchor: jax.Array | None = None,
) -> None:
    """Fold a replayed fragment's masked log2-unit scores into its rows of
    the row state (rebase_fragment), then add its probs @ V, the matmul on
    QK_MXU through value_slot against the tile v_tile loads."""
    probs, rescale, starts = rebase_fragment(
        row_max_ref, row_sum_ref, scores, is_row_start, takes_anchor
    )
    weighted_values = replay_matmul(
        value_slot, probs.astype(jnp.bfloat16), v_tile()
    )
    accumulate_rebased(o_ref, weighted_values, rescale, starts)


@dataclasses.dataclass(frozen=True, eq=False)
class RowCheckpoints:
    """The checkpoints of a row's check units: checkpoint_sum and
    checkpoint_o, (NUM_CHECKPOINTS, rows, ...), hold the lane partial sums
    and accumulator a unit enters with; row_sum and o hold the row's at
    parity, and row_anchor loads the anchor of a fragment's rows."""

    checkpoint_sum: jax.Array
    checkpoint_o: jax.Array
    row_sum: jax.Array
    o: jax.Array
    parity: jax.Array | int
    row_anchor: Callable[[pl.Slice], jax.Array]

    @property
    def num_fragments(self) -> int:
        return self.checkpoint_sum.shape[1] // FRAGMENT_ROWS

    def for_each_fragment(
        self, fragment_fn: Callable[[pl.Slice], None]
    ) -> None:
        """Run fragment_fn on the rows of each fragment, rolled."""

        def _fragment(fragment, carry):
            fragment_fn(fragment_rows(fragment))
            return carry

        lax.fori_loop(0, self.num_fragments, _fragment, None)

    def copy_checkpoint(
        self, unit: jax.Array | int, *, is_restore: bool
    ) -> None:
        """Copy the row's lane partial sums and accumulator into check unit
        unit's checkpoint, or back from it (is_restore)."""
        slot = lax.rem(unit, NUM_CHECKPOINTS)
        pairs = (
            (self.row_sum.at[self.parity], self.checkpoint_sum.at[slot]),
            (self.o.at[self.parity], self.checkpoint_o.at[slot]),
        )

        def _copy(rows):
            for row_state, checkpoint in pairs:
                if is_restore:
                    row_state[rows, :] = checkpoint[rows, :]
                else:
                    checkpoint[rows, :] = row_state[rows, :]

        self.for_each_fragment(_copy)

    def save_anchor(self, unit: jax.Array | int) -> None:
        """Copy the row's anchor into check unit unit's checkpoint row sums,
        which its replay has read."""
        slot = lax.rem(unit, NUM_CHECKPOINTS)

        def _save(rows):
            self.checkpoint_sum[slot, rows, :] = self.row_anchor(rows)

        self.for_each_fragment(_save)

    def rebase_checkpoint(
        self, unit: jax.Array | int, anchor_unit: jax.Array | int
    ) -> None:
        """Rescale check unit unit's checkpoint from the anchor save_anchor
        left in check unit anchor_unit's checkpoint row sums to the row's
        anchor, which a replay may have raised since."""
        slot = lax.rem(unit, NUM_CHECKPOINTS)
        anchor_slot = lax.rem(anchor_unit, NUM_CHECKPOINTS)
        lane_tiles = self.checkpoint_o.shape[-1] // NUM_LANES

        def _rebase(rows):
            saved_anchor = self.checkpoint_sum[anchor_slot, rows, :]
            rescale = unanchored_rescale(saved_anchor, self.row_anchor(rows))
            self.checkpoint_sum[slot, rows, :] *= rescale
            self.checkpoint_o[slot, rows, :] *= jnp.tile(
                rescale, (1, lane_tiles)
            )

        self.for_each_fragment(_rebase)


def finish_row_sums(
    traits: FwdTraits, state: RowStateRefs, parity: jax.Array | int
) -> None:
    """Reduce the lane partial sums of the row at parity to row sums, every
    lane a row's value, and set each pipeline head's guard flag. Runs after
    the row's last softmax and before its finish."""
    group_bq = traits.group_bq
    for g in range(traits.head_fold):
        row_sum_ref = state.row_sum.at[parity, pl.ds(g * group_bq, group_bq)]
        # Note (david): unrolled, with one vreg of running maxima, the chunks'
        # lane sums overlap on the XLUs.
        max_sum = jnp.zeros((NUM_SUBLANES, NUM_LANES), jnp.float32)
        for first_row in range(0, group_bq, ROW_SUM_CHUNK_ROWS):
            rows = pl.ds(
                first_row, min(ROW_SUM_CHUNK_ROWS, group_bq - first_row)
            )
            row_sum = across_lanes(row_sum_ref[rows, :].sum(axis=-1))
            row_sum_ref[rows, :] = row_sum
            max_sum = jnp.maximum(
                max_sum,
                row_sum.reshape(-1, NUM_SUBLANES, NUM_LANES).max(axis=0),
            )
        state.guard[g] = guard_flag(traits, max_sum)


def row_finish_state(
    row_max: jax.Array,
    row_sum: jax.Array,
    o: jax.Array,
    guard: jax.Array,
    parity: jax.Array | int,
) -> RowStateRefs:
    """The state of the row at parity as the row finish of fwd_blocks reads
    it; row_max, row_sum and o lead with a NUM_ROW_PARITIES axis."""
    return RowStateRefs(
        row_max=row_max.at[parity],
        row_sum=row_sum,
        o=o,
        # A row keeps one parity from start to finish, so the row finish
        # reads no block parity.
        parity=None,
        guard=guard,
    )


def stage_row_finish(
    traits: FwdTraits,
    variant: Variant,
    operands: VariantOperands,
    state: RowStateRefs,
    out_refs: OutRefs,
    finish: RowFinish,
    *,
    reduces_sums: bool,
) -> None:
    """Stage the lse and the normalized output of finish's row from state,
    once its last softmax has run and its output stage slot is free.
    reduces_sums first reduces its lane partial sums (finish_row_sums); a
    replay leaves whole row sums."""
    if reduces_sums:
        finish_row_sums(traits, state, finish.parity)
    sinks = (
        head_sinks(traits, operands.sinks_ref, None, finish.base_head)
        if variant.has_sinks
        else None
    )
    if variant.return_lse:
        stage_row_lse(
            traits,
            variant,
            state,
            out_refs,
            finish,
            row_out_slot(traits, finish),
            sinks,
        )
    if sinks is not None:
        fold_out_sinks(traits, variant, state, finish, sinks)
    stage_row_out(
        traits,
        variant,
        operands,
        state,
        out_refs,
        finish,
        row_out_slot(traits, finish),
    )
