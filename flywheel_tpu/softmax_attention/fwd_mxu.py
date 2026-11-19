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

- A row holds one parity of row_sum and o from its first tile to its finish;
  nothing ping-pongs between kv blocks.
- row_sum holds lane partial sums: lane l of a row is the sum of its
  probabilities at the keys l mod NUM_LANES of every tile. finish_row_sums
  reduces them once per row, after the row's last softmax, to the value the
  row finish reads (fwd_blocks.finish_row_lse, fold_out_sinks,
  finish_row_out).
- The overflow guard is checked there too, once per row: a flagged pipeline
  head replays its row's final unit. Each loop also checks a row's tiles
  before it by check units (LargestSums), each from a checkpoint of its
  entry state.

fold_fragment_scores and accumulate_fragment are the bodies of a
FragmentOps.softmax and .accumulate; the kernel masks the scores first and
names the rows. A check unit's last tile also writes the next unit's
checkpoint (UnitCheckpoint, RowCheckpoints), and the body after a checked
first body finishes it with FlaggedTailOps. fold_replayed_fragment replays
one fragment between two segments with the exact rebasing softmax.
stage_row_finish is the row finish of a per-seq loop, whose rows alternate
between NUM_ROW_PARITIES parities.
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

from flywheel_tpu.core.tpu import MXU_COLUMNS, NUM_LANES, NUM_SUBLANES
from flywheel_tpu.softmax_attention.config import BlockSizes, QKVLayout
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
    QK_MXU,
    STAGING_REGISTER,
    FragmentOps,
    PipelineDelays,
    WorkItem,
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
# The first row takes parity 0, so what runs before it takes the other one,
# which the second row overwrites before reading it.
ENTRY_PARITY = NUM_ROW_PARITIES - 1
# The rows finish_row_sums reduces a step.
ROW_SUM_CHUNK_ROWS = NUM_LANES
# A replay between two segments pushes into the staging register the pipeline
# leaves empty there.
REPLAY_STAGING_REGISTER = 1 - STAGING_REGISTER
# A replay's QK result slots: the scores, then probs @ V.
REPLAY_RESULT_SLOTS = 2
# The checkpoints of a unit and the unit after it.
NUM_CHECKPOINTS = 2
# A check unit of a loop over a row's kv tiles: a first body, whose end checks
# the unit before, and a last body.
UNIT_TILES = 2 * LOOP_BODY_TILES
# A check unit's first body, a row's start segment or a loop body, covers as
# many tiles as a loop body.
assert ROW_START_TILES == LOOP_BODY_TILES
# A row's final unit holds its last FINAL_UNIT_SPAN whole check units and the
# tiles after them: the first body that would check the first of them is the
# row's last first body, which checks nothing.
FINAL_UNIT_SPAN = 2


def mxu_pipeline_applies(
    *,
    variant: Variant,
    block_sizes: BlockSizes,
    head_dim: int,
    head_dim_v: int,
    transposed_pv: bool,
    group_rows: int,
) -> bool:
    """Whether a forward build can run its fragments on the explicit MXU
    pipeline: the chip and the shapes every such loop needs. A kernel adds
    its own conditions.

    The chip's MXUs have result accumulators; K, V and a tile of scores are as
    wide as an MXU; tokens are on sublanes; the mask is full or causal (a left
    window bound keeps the band copies); and one kv block's items make a
    segment. group_rows is the q rows that share a K tile: block_q times the
    q heads stacked into a fragment's rows.
    """
    if jax.default_backend() != "tpu":
        return False
    info = pltpu.get_tpu_info()
    if not info.num_accumulators or info.mxu_column_size != MXU_COLUMNS:
        return False
    num_block_items = (
        group_rows // FRAGMENT_ROWS * (block_sizes.block_kv // MXU_COLUMNS)
    )
    return (
        head_dim == head_dim_v == MXU_COLUMNS
        and block_sizes.qkv_layout == QKVLayout.HEAD_DIM_MINOR
        and not transposed_pv
        and variant.window[0] is None
        and variant.window[1] in (None, 0)
        and block_sizes.block_kv % MXU_COLUMNS == 0
        and group_rows % FRAGMENT_ROWS == 0
        and num_block_items >= PipelineDelays().pv_pop
    )


def mxu_block_sizes(block_sizes: BlockSizes) -> BlockSizes:
    """block_sizes with the kv compute tile an explicit MXU build runs."""
    return BlockSizes(
        block_q=block_sizes.block_q,
        block_kv=block_sizes.block_kv,
        block_kv_compute=MXU_COLUMNS,
        block_q_compute=block_sizes.block_q_compute,
        num_stages=block_sizes.num_stages,
        qkv_layout=block_sizes.qkv_layout,
    )


def aligned_rows(row: jax.Array | int) -> jax.Array | int:
    """row, a multiple of FRAGMENT_ROWS, marked as one when traced."""
    if isinstance(row, int):
        return row
    return pl.multiple_of(row, FRAGMENT_ROWS)


def fragment_rows(fragment: jax.Array | int) -> pl.Slice:
    """The rows of fragment fragment in a run of fragments of FRAGMENT_ROWS
    rows."""
    return pl.ds(aligned_rows(fragment * FRAGMENT_ROWS), FRAGMENT_ROWS)


def scaled_logits(variant: Variant, scores: jax.Array) -> jax.Array:
    """The log2-unit logits of raw f32 scores of an unscaled Q: the scores
    take the static Q scale, then the softcap."""
    # Note (david): a loop that leaves its staged Q unscaled saves each row a
    # pass over its Q block; the spare VALU slots of its body absorb the
    # scaling of each fragment's scores.
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
    largest row sums of its rows: a row sum bounds each of its scores from
    above, and a NaN or inf sum carries through the max and fails the compare
    as well."""
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
    """The largest lane partial sums a run of softmaxes wrote, one
    (NUM_SUBLANES, NUM_LANES) running maximum while their segments trace;
    guard_flag reads it once, after them."""

    value: jax.Array | None = None

    def fold(self, sums: jax.Array) -> None:
        largest = sums.reshape(-1, NUM_SUBLANES, NUM_LANES).max(axis=0)
        if self.value is None:
            self.value = largest
        else:
            self.value = jnp.maximum(self.value, largest)


class UnitCheckpoint(NamedTuple):
    """A check unit's last tile, tile key tile, whose items also write the
    row sums and accumulator rows they leave, the next unit's entry state,
    into the (rows, ...) checkpoint buffers row_sum and o, and fold the row
    sums into largest when given."""

    tile: int
    row_sum: jax.Array
    o: jax.Array
    largest: LargestSums | None = None


def fold_fragment_scores(
    row_max_ref: jax.Array,
    row_sum_ref: jax.Array,
    rows: pl.Slice,
    scores: jax.Array,
    *,
    is_row_start: bool,
    largest: LargestSums | None = None,
    checkpoint_sum: jax.Array | None = None,
) -> jax.Array:
    """Fold one fragment's masked (FRAGMENT_ROWS, MXU_COLUMNS) log2-unit
    scores into its rows of row_max_ref and row_sum_ref, a pipeline head's
    (group rows, NUM_LANES) anchor and lane partial sums; returns the
    fragment's bf16 probabilities.

    is_row_start (static) is the fragment's first tile: its row maximum
    becomes the rows' anchor and its lane sums start their row sums. Every
    later tile reads the anchor as it is. checkpoint_sum, a unit's
    checkpoint (UnitCheckpoint), takes the sums written from the registers
    that store them, and largest collects them.
    """
    if is_row_start:
        anchor = across_lanes(scores.max(axis=-1))
        row_max_ref[rows, :] = anchor
    else:
        anchor = row_max_ref[rows, :]
    probs = anchored_probs(scores, anchor)
    lane_sums = lane_partial_sums(probs)
    if is_row_start:
        new_sums = lane_sums
    else:
        new_sums = row_sum_ref[rows, :] + lane_sums
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
) -> jax.Array:
    """fold_fragment_scores past a row's start for an item of unit_end's
    tile, with unit_end's checkpoint row sums and largest, loading the row
    sums before it sums the lanes."""
    probs = anchored_probs(scores, row_max_ref[rows, :])
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
    is_row_start: bool,
    checkpoint_o: jax.Array | None = None,
) -> None:
    """Add one fragment's f32 probs @ V to its rows of o_ref, a pipeline
    head's (group rows, head_dim) accumulator; is_row_start as in
    fold_fragment_scores. checkpoint_o, a unit's checkpoint
    (UnitCheckpoint), takes the rows written."""
    values = weighted_values
    if not is_row_start:
        values = o_ref[rows, :] + values
    o_ref[rows, :] = values
    if checkpoint_o is not None:
        checkpoint_o[rows, :] = values


@dataclasses.dataclass(frozen=True, eq=False)
class FlaggedTailOps:
    """The ops with which the body after a checked first body finishes it:
    once is_flagged replayed that body's tiles, its pending softmaxes add
    nothing to the row sums and give zero probabilities, so their
    accumulates add zero, and the other pending accumulates add nothing (a
    select, not a product: the flagged values may be inf). zeroed holds the
    (fragment, tile) keys of the softmaxes run.

    ops are the loop's FragmentOps, which also name an item's rows (rows),
    mask its scores (masked_scores) and hold the row's anchor and lane
    partial sums (row_max, row_sum); every pending softmax is of an item of
    mode plain_mode.
    """

    ops: FragmentOps
    is_flagged: jax.Array
    plain_mode: Hashable
    zeroed: set = dataclasses.field(default_factory=set)

    def k_tile(self, tile: Hashable) -> jax.Array:
        return self.ops.k_tile(tile)

    def v_tile(self, tile: Hashable) -> jax.Array:
        return self.ops.v_tile(tile)

    def q_fragment(self, item: WorkItem) -> jax.Array:
        return self.ops.q_fragment(item)

    def softmax(self, item: WorkItem, scores: jax.Array) -> jax.Array:
        assert item.mode == self.plain_mode, item
        self.zeroed.add((item.fragment, item.tile))
        ops, rows = self.ops, self.ops.rows(item)
        probs = anchored_probs(
            ops.masked_scores(item, scores), ops.row_max[rows, :]
        )
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
    """One fragment's lhs @ rhs^T (transpose) or lhs @ rhs on QK_MXU between
    two segments: rhs goes through REPLAY_STAGING_REGISTER and the f32 result
    through slot, a QK result slot that holds nothing there
    (MxuPipeline.free_slots). PV_MXU keeps its tile, its staged tile and the
    results in flight."""
    pltpu.matmul_push_rhs(
        rhs,
        staging_register=REPLAY_STAGING_REGISTER,
        mxu_index=QK_MXU,
        transpose=transpose,
    )
    pltpu.matmul_acc_lhs(slot, lhs, REPLAY_STAGING_REGISTER)
    return pltpu.matmul_pop(slot)


def rebase_fragment(
    row_max_ref: jax.Array,
    row_sum_ref: jax.Array,
    scores: jax.Array,
    is_row_start: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """The exact rebasing softmax of one fragment's masked scores, for a
    replay: raise the rows' anchor in row_max_ref to cover them (or start it
    at a row start), rescale row_sum_ref's lane partial sums to it and add
    the fragment's. The refs hold the fragment's rows. Returns the f32
    probabilities and the accumulator's rescale."""
    tile_max = across_lanes(scores.max(axis=-1))
    prev_max = row_max_ref[...]
    new_max = jnp.where(is_row_start, tile_max, jnp.maximum(prev_max, tile_max))
    probs = anchored_probs(scores, new_max)
    rescale = jnp.exp2(prev_max - new_max)
    lane_sums = lane_partial_sums(probs)
    row_sum_ref[...] = jnp.where(
        is_row_start, lane_sums, rescale * row_sum_ref[...] + lane_sums
    )
    row_max_ref[...] = new_max
    return probs, rescale


def accumulate_rebased(
    o_ref: jax.Array,
    weighted_values: jax.Array,
    rescale: jax.Array,
    is_row_start: jax.Array,
) -> None:
    """Add a replayed fragment's f32 probs @ V to its rows of the
    accumulator, o_ref, under rebase_fragment's rescale; a row start takes
    the fragment's alone."""
    lane_tiles = weighted_values.shape[-1] // NUM_LANES
    o_ref[...] = jnp.where(
        is_row_start,
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
) -> None:
    """Fold a replayed fragment's masked log2-unit scores into its rows of
    the row state, the refs, with the exact rebasing softmax
    (rebase_fragment), then add its probs @ V (accumulate_rebased): V the
    tile v_tile loads, the matmul on QK_MXU through value_slot
    (replay_matmul)."""
    probs, rescale = rebase_fragment(
        row_max_ref, row_sum_ref, scores, is_row_start
    )
    weighted_values = replay_matmul(
        value_slot, probs.astype(jnp.bfloat16), v_tile()
    )
    accumulate_rebased(o_ref, weighted_values, rescale, is_row_start)


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
            rescale = jnp.exp2(
                self.checkpoint_sum[anchor_slot, rows, :]
                - self.row_anchor(rows)
            )
            self.checkpoint_sum[slot, rows, :] *= rescale
            self.checkpoint_o[slot, rows, :] *= jnp.tile(
                rescale, (1, lane_tiles)
            )

        self.for_each_fragment(_rebase)


def finish_row_sums(
    traits: FwdTraits, state: RowStateRefs, parity: jax.Array | int
) -> None:
    """Reduce the lane partial sums of the row at parity to row sums, every
    lane a row's value, and set each pipeline head's guard flag: whether one
    of its rows could hold a score past the overflow guard.

    Runs after the row's last softmax and before its finish.
    """
    group_bq = traits.group_bq
    for g in range(traits.head_fold):
        row_sum_ref = state.row_sum.at[parity, pl.ds(g * group_bq, group_bq)]

        def _reduce_chunk(chunk, max_sum, row_sum_ref=row_sum_ref):
            rows = pl.ds(chunk * ROW_SUM_CHUNK_ROWS, ROW_SUM_CHUNK_ROWS)
            row_sum = across_lanes(row_sum_ref[rows, :].sum(axis=-1))
            row_sum_ref[rows, :] = row_sum
            return jnp.maximum(
                max_sum,
                row_sum.reshape(-1, NUM_SUBLANES, NUM_LANES).max(axis=0),
            )

        # Note (david): unrolled with one vreg of running maxima, the chunks'
        # lane sums overlap on the XLUs; a rolled loop reducing each chunk's
        # flag to a scalar ran them one after another, and per-lane flags
        # held a chunk of mask values live across the loop.
        max_sum = lax.fori_loop(
            0,
            group_bq // ROW_SUM_CHUNK_ROWS,
            _reduce_chunk,
            jnp.zeros((NUM_SUBLANES, NUM_LANES), jnp.float32),
            unroll=True,
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
    """Stage the lse and the normalized output of finish's row from state
    (row_finish_state), once its last softmax has run and its output stage
    slot is free (wait_row_out_slot). reduces_sums first reduces the row's
    lane partial sums and sets its guard flag (finish_row_sums); a replay
    leaves whole row sums."""
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
