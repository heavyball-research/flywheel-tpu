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

On a chip with MXU accumulators a head_dim 256 dense build
(dense_uses_mxu_pipeline) runs its rows through DenseMxuLoop in place of the
dot_general fragment pipeline of kernels/fwd_dense.py's dense_fwd_body. A row
is one pipeline, from its first tile to its drain, so nothing ping-pongs
between its kv blocks, and a causal row visits its diagonal block first: its
diagonal items start the fragments' rows, and every later tile is one
unmasked loop body, the first row's diagonal-only row included.

The overflow guard checks a row's kv blocks by check units of one block each
(DenseMxuLoop.checked_body). A unit's last tile also writes the row state it
leaves into a checkpoint, the next unit's entry state. The next block's first
body finishes that tile in its first qk_pop ticks; the row sums its own first
tile writes bound the unit's, and at its end they decide whether the unit
replays: from its checkpoint, through the first body's tiles. The next body
adds nothing for the items the replay redid. A row's final unit, its last
FINAL_UNIT_SPAN blocks, replays after the drain when the row's own sums
flag it (DenseMxuLoop.finish_row). The row state stays in place at
MXU_ROW_PARITY.
"""

import dataclasses
import functools
from collections.abc import Callable
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
    ScoreMask,
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
    FlaggedTailOps,
    LargestSums,
    RowCheckpoints,
    UnitCheckpoint,
    accumulate_fragment,
    across_lanes,
    finish_row_sums,
    first_final_unit,
    fold_fragment_scores,
    fold_replayed_fragment,
    fragment_rows,
    guard_flag,
    mxu_pipeline_applies,
    replay_matmul,
    scaled_logits,
)
from flywheel_tpu.softmax_attention.mxu_pipeline import (
    FRAGMENT_ROWS,
    MxuPipeline,
    PipelineState,
    Tail,
    WorkItem,
)
from flywheel_tpu.softmax_attention.schedule import FwdSchedule
from flywheel_tpu.softmax_attention.traits import FwdTraits
from flywheel_tpu.softmax_attention.variant import Variant, VariantOperands

# The smallest q and kv block the explicit loop takes: a shorter row pays its
# row start, drain and finish on fewer work items than the loop gains.
MIN_MXU_BLOCK = 1024
# The row-state parity an explicit row holds from its first tile to its finish.
MXU_ROW_PARITY = 0
# The fragments that meet one kv tile of the diagonal on it.
FRAGMENTS_PER_TILE = MXU_COLUMNS // FRAGMENT_ROWS
# A checked block's first body checks the unit before, and its body
# SECOND_BODY finishes the first one's items (DenseMxuLoop.run_second_body).
SECOND_BODY = 1


def dense_uses_mxu_pipeline(
    *,
    variant: Variant,
    block_sizes: BlockSizes,
    head_dim: int,
    head_dim_v: int,
    head_fold: int,
    transposed_pv: bool,
) -> bool:
    """Whether a dense build runs its fragments on the explicit MXU pipeline
    (fwd_mxu.mxu_pipeline_applies, plus what this loop supports).

    The loop runs one q head a pipeline head (head_fold 1), whole loop bodies
    of LOOP_BODY_TILES tiles per kv block, and blocks of at least MIN_MXU_BLOCK
    rows. A causal mask needs a static offset that is a multiple of block_kv
    and square blocks, so every row's last kv block holds the same diagonal
    triangle; the runtime-offset kernel keeps the dot_general loop. The loop
    scales its scores by the static Q scale, so it takes no kv descales.
    """
    block_q, block_kv = block_sizes.block_q, block_sizes.block_kv
    is_causal = variant.window[1] == 0
    has_static_diagonal = not variant.runtime_causal_offset and (
        not is_causal
        or (block_q == block_kv and variant.causal_offset % block_kv == 0)
    )
    return (
        head_fold == 1
        and has_static_diagonal
        and not variant.has_kv_scales
        and min(block_q, block_kv) >= MIN_MXU_BLOCK
        and block_kv % (LOOP_BODY_TILES * MXU_COLUMNS) == 0
        and mxu_pipeline_applies(
            variant=variant,
            block_sizes=block_sizes,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            transposed_pv=transposed_pv,
            group_rows=block_q,
        )
    )


def dense_has_checked_units(num_kv_blocks: int) -> bool:
    """Whether a dense build whose rows visit at most num_kv_blocks kv blocks
    has rows with a check unit before their final unit."""
    return num_kv_blocks > FINAL_UNIT_SPAN


class FragmentMode(NamedTuple):
    """The static flags of an explicit work item: is_row_start, the item is
    its fragment's first of the row; is_diagonal, it sits on the causal
    diagonal and masks; is_row_end, it is its fragment's last, so it also
    finishes the fragment's rows."""

    is_row_start: bool = False
    is_diagonal: bool = False
    is_row_end: bool = False


PLAIN = FragmentMode()
ROW_START = FragmentMode(is_row_start=True)
# A causal row visits its diagonal block first, so a diagonal item is also its
# fragment's first.
DIAGONAL = FragmentMode(is_row_start=True, is_diagonal=True)
ROW_END = FragmentMode(is_row_end=True)


def tile_items(
    fragments: range, tile: int, mode: FragmentMode = PLAIN
) -> tuple[WorkItem, ...]:
    """fragments against kv tile tile, in order."""
    return tuple(
        WorkItem(fragment=fragment, tile=tile, mode=mode)
        for fragment in fragments
    )


def body_items(
    traits: FwdTraits, last_tile_mode: FragmentMode = PLAIN
) -> tuple[WorkItem, ...]:
    """A loop body's items: every fragment against LOOP_BODY_TILES tiles,
    counted from the body's first; a row's last body ends its fragments on
    its last tile (last_tile_mode ROW_END)."""
    fragments = range(traits.bq // FRAGMENT_ROWS)
    return sum(
        (
            tile_items(
                fragments,
                tile,
                last_tile_mode if tile == LOOP_BODY_TILES - 1 else PLAIN,
            )
            for tile in range(LOOP_BODY_TILES)
        ),
        (),
    )


def row_start_items(traits: FwdTraits) -> tuple[WorkItem, ...]:
    """A full row's first segment: its first tile starts every fragment's
    row, and a plain tile follows, as a loop body needs before it."""
    fragments = range(traits.bq // FRAGMENT_ROWS)
    return tile_items(fragments, 0, ROW_START) + tile_items(fragments, 1)


def diagonal_items(
    traits: FwdTraits, *, ends_row: bool
) -> tuple[tuple[WorkItem, ...], tuple[WorkItem, ...]]:
    """A causal row's diagonal block, last tile first: tile t meets the
    fragments from FRAGMENTS_PER_TILE * t on, the first FRAGMENTS_PER_TILE of
    them on the diagonal. Returns (tiles from the last down to 1, tile 0);
    tile 0 meets every fragment, so the block ends as a loop body does, and
    it ends the row's fragments when ends_row (a diagonal-only row).

    A tile's diagonal items alternate with its first plain ones: a diagonal
    item holds its scores through the anchor's cross-lane max, which a run of
    them cannot hide."""
    num_fragments = traits.bq // FRAGMENT_ROWS

    def _items(tile, ends_row=False):
        first = tile * FRAGMENTS_PER_TILE
        diagonal = tile_items(
            range(first, first + FRAGMENTS_PER_TILE),
            tile,
            DIAGONAL._replace(is_row_end=ends_row),
        )
        plain = tile_items(
            range(first + FRAGMENTS_PER_TILE, num_fragments),
            tile,
            PLAIN._replace(is_row_end=ends_row),
        )
        paired = sum(zip(diagonal, plain), ())
        return paired + diagonal[len(plain) :] + plain[len(diagonal) :]

    head = sum(
        (
            _items(tile)
            for tile in reversed(range(1, traits.num_k_compute_tiles))
        ),
        (),
    )
    return head, _items(0, ends_row)


class CheckRefs(NamedTuple):
    """The scratch of a build's check units: the entry row sums and
    accumulators of NUM_CHECKPOINTS units, (NUM_CHECKPOINTS, bq, ...), and
    the SMEM (1,) flag of the unit a first body checks."""

    checkpoint_sum: jax.Array
    checkpoint_o: jax.Array
    unit_guard: jax.Array


@dataclasses.dataclass(eq=False)
class RowEnd:
    """What a row's last items finish into: the output staging slot and the
    pipeline head's sink logit (None without sinks). largest_sum collects the
    largest row sum of the fragments finished so far while the row's last
    segments trace; the guard checks it once."""

    out_slot: jax.Array
    sink: jax.Array | None
    largest_sum: jax.Array | None = None


@dataclasses.dataclass(frozen=True, eq=False)
class DenseFragmentOps:
    """The explicit pipeline's FragmentOps of a dense row: q fragments from
    the row's Q staging slot q_slot, K and V tiles of the kv block at kv_slot
    counted from first_tile, and the row state at MXU_ROW_PARITY. A row-end
    item reduces its fragment's row sums and writes its normalized output
    rows into o_stage at row_end's slot. unit_end names a check unit's last
    tile (UnitCheckpoint), and largest collects the row sums the items of
    tile key 0 write, when given.

    A fold only reads the row state, so ops without a block (kv_slot None)
    finish a tail: the items a segment leaves in flight push no tile.
    """

    traits: FwdTraits
    variant: Variant
    operands: VariantOperands
    q_refs: QRefs
    kv_refs: KVRefs
    state: RowStateRefs
    o_stage: jax.Array
    q_slot: jax.Array
    kv_slot: jax.Array | None = None
    first_tile: jax.Array | int = 0
    row_end: RowEnd | None = None
    unit_end: UnitCheckpoint | None = None
    largest: LargestSums | None = None

    def with_tiles(
        self,
        kv_slot: jax.Array,
        first_tile: jax.Array | int,
        row_end: RowEnd | None = None,
    ) -> "DenseFragmentOps":
        return dataclasses.replace(
            self, kv_slot=kv_slot, first_tile=first_tile, row_end=row_end
        )

    def _tile(self, buf: jax.Array, tile: int) -> jax.Array:
        assert self.kv_slot is not None, "a tail's ops push no kv tile."
        tile_start = (self.first_tile + tile) * MXU_COLUMNS
        if isinstance(tile_start, int):
            keys = pl.ds(tile_start, MXU_COLUMNS)
        else:
            keys = pl.ds(pl.multiple_of(tile_start, MXU_COLUMNS), MXU_COLUMNS)
        return buf[self.kv_slot, keys, :]

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

    def masked_scores(self, item: WorkItem, scores: jax.Array) -> jax.Array:
        """item's raw scores as log2-unit logits, a diagonal item's masked."""
        logits = scaled_logits(self.variant, scores)
        if item.mode.is_diagonal:
            # A diagonal item's tile is static and counted from the block's
            # first key, which sits on its q block's first row.
            logits = self.variant.mask_logits(
                logits,
                item.fragment * FRAGMENT_ROWS,
                item.tile * MXU_COLUMNS,
                logits.shape,
            )
        return logits

    def is_unit_end(self, item: WorkItem) -> bool:
        return self.unit_end is not None and item.tile == self.unit_end.tile

    def softmax(self, item: WorkItem, scores: jax.Array) -> jax.Array:
        logits = self.masked_scores(item, scores)
        if self.is_unit_end(item):
            return fold_fragment_scores(
                self.row_max,
                self.row_sum,
                self.rows(item),
                logits,
                is_row_start=item.mode.is_row_start,
                checkpoint_sum=self.unit_end.row_sum,
            )
        row_sum_ref = self.row_sum
        probs = fold_fragment_scores(
            self.row_max,
            row_sum_ref,
            fragment_rows(item.fragment),
            logits,
            is_row_start=item.mode.is_row_start,
            largest=self.largest if item.tile == 0 else None,
        )
        if item.mode.is_row_end:
            # The fragment's last lane partial sums reduce to its row sums,
            # every lane a row's value, as the row finish reads them.
            rows = fragment_rows(item.fragment)
            row_sum = across_lanes(row_sum_ref[rows, :].sum(axis=-1))
            row_sum_ref[rows, :] = row_sum
            end = self.row_end
            end.largest_sum = (
                row_sum
                if end.largest_sum is None
                else jnp.maximum(end.largest_sum, row_sum)
            )
        return probs

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
        # normalized (sink folded) instead of back to the accumulator, as
        # fold_out_sinks and finish_row_out would leave them.
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
    """An explicit row while it runs: how many kv blocks it visits, its head
    group's first head, its ops, the kv ring's carry (dense_fwd_body's) and
    the maker of its RowEnd, one per traced copy of the row's last segment."""

    num_visits: jax.Array
    base_head: jax.Array
    ops: DenseFragmentOps
    kv_carry: tuple
    row_end: Callable[[], RowEnd]


class LoopEntry(NamedTuple):
    """Where a row's rolled loop takes over from its first segments: the
    pipeline state they leave, the first visit the loop runs and how many
    bodies of that visit they ran."""

    state: PipelineState
    first_visit: int
    first_body: int


@dataclasses.dataclass(frozen=True, eq=False)
class DenseMxuLoop:
    """The rows of a dense build on the explicit MXU pipeline, over the
    refs of dense_fwd_body. A row keeps its row state at MXU_ROW_PARITY;
    under blocks.is_diagonal_first (a causal build) it visits its last kv
    block, the one that holds the diagonal, first. check_refs holds the
    scratch of the check units of a build with checked units
    (dense_has_checked_units), else None."""

    traits: FwdTraits
    variant: Variant
    operands: VariantOperands
    schedule: FwdSchedule
    pipeline: MxuPipeline
    blocks: DenseBlockStream
    q_refs: QRefs
    kv_refs: KVRefs
    out_refs: OutRefs
    state: RowStateRefs
    mask: ScoreMask
    check_refs: CheckRefs | None

    def __post_init__(self) -> None:
        traits, schedule = self.traits, self.schedule
        assert traits.head_fold == 1, traits.head_fold
        assert (self.check_refs is not None) == self.has_checked_units
        # A first body's checkpointed items, its first tile's, never sit in
        # the tail the next body finishes.
        assert self.num_fragments >= self.pipeline.delays.pv_pop
        if not self.blocks.is_diagonal_first:
            return
        bq, bkv = traits.bq, traits.bkv
        causal_offset = self.mask.causal_offset
        assert bq == bkv and causal_offset % bkv == 0, causal_offset
        last_row = schedule.first_row + schedule.num_rows - 1
        if last_row * bq + causal_offset >= schedule.num_kv_blocks * bkv:
            raise ValueError(
                "an explicit causal build needs the diagonal in every"
                f" row's last kv block; row {last_row} has none at"
                f" causal_offset {causal_offset}."
            )

    @property
    def bodies_per_block(self) -> int:
        return self.traits.num_k_compute_tiles // LOOP_BODY_TILES

    @property
    def num_fragments(self) -> int:
        return self.traits.bq // FRAGMENT_ROWS

    @property
    def has_checked_units(self) -> bool:
        return dense_has_checked_units(self.schedule.num_kv_blocks)

    def final_unit(self, num_visits: jax.Array) -> jax.Array | int:
        """The first visit of a row's final unit: the check units before it
        are checked (checked_body), and the row's own sums check it after the
        drain (replay_row_end); 0 in a build without checked units."""
        if not self.has_checked_units:
            return 0
        return first_final_unit(num_visits)

    @functools.cached_property
    def loop_items(self) -> tuple[WorkItem, ...]:
        return body_items(self.traits)

    def visit_block(self, si: jax.Array, visit: jax.Array | int) -> jax.Array:
        """The kv block of row si's visit visit."""
        lo, hi, _ = self.schedule.row_interval(si)
        if self.blocks.is_diagonal_first:
            return jnp.where(visit == 0, hi, lo + visit - 1)
        else:
            return lo + visit

    def step_block(
        self,
        si: jax.Array,
        visit: jax.Array | int,
        num_visits: jax.Array,
        ahead: tuple[BlockDesc, BlockDesc],
        base_head: jax.Array,
        kv_carry: tuple,
    ) -> tuple[tuple, jax.Array, jax.Array]:
        """Step the kv ring at the top of row si's visit visit: wait for the
        next visit's block, whose first K tile this visit stages, and start
        the block two visits ahead; ahead holds the descriptors of the next
        row's first two visits. Returns the shifted carry and the slots of
        this visit's block and the next one's."""
        kv_slot, next_slot = kv_carry[:2]
        ahead_block = self.ahead_block(
            si, visit, num_visits, ahead, base_head, kv_carry
        )
        self.wait_next_block(kv_carry)
        return self.start_ahead_block(kv_carry, ahead_block), kv_slot, next_slot

    def ahead_block(
        self,
        si: jax.Array,
        visit: jax.Array | int,
        num_visits: jax.Array,
        ahead: tuple[BlockDesc, BlockDesc],
        base_head: jax.Array,
        kv_carry: tuple,
    ) -> tuple[BlockDesc, jax.Array, jax.Array]:
        """The descriptor, ring slot and freshness of the block two visits
        after row si's visit visit (step_block)."""
        blocks = self.blocks
        _, next_slot, _, _, _, next_desc = kv_carry
        first_next_row_desc, second_next_row_desc = ahead
        ahead_desc = select_desc(
            visit + 2 < num_visits,
            blocks.desc(base_head, self.visit_block(si, visit + 2)),
            select_desc(
                visit + 2 == num_visits,
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
        """Start ahead_block (ahead_block's value) when it is fresh; returns
        the kv carry one visit on."""
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
        slot: jax.Array,
        next_slot: jax.Array,
        body: jax.Array,
        probs: tuple[jax.Array, ...],
    ) -> tuple[jax.Array, ...]:
        """One loop body: tiles body * LOOP_BODY_TILES on of the block at slot;
        it stages the next body's first K tile, the next block's past the
        block's last body."""
        pipeline = self.pipeline
        first_tile = body * LOOP_BODY_TILES
        is_block_end = body == self.bodies_per_block - 1
        next_ops = ops.with_tiles(
            jnp.where(is_block_end, next_slot, slot),
            jnp.where(is_block_end, 0, first_tile + LOOP_BODY_TILES),
        )
        body_state = pipeline.advance(
            PipelineState(probs=probs, tail=assumed),
            self.loop_items,
            ops.with_tiles(slot, first_tile),
            next_k_tile=lambda: next_ops.k_tile(0),
        )
        pipeline.check_loop_tail(body_state.tail, assumed)
        return body_state.probs

    def loop_tail(
        self, ops: DenseFragmentOps, state_before: PipelineState
    ) -> Tail:
        """The tail a loop body assumes after the segment that left
        state_before."""
        pipeline = self.pipeline
        return pipeline.tail_of(
            self.loop_items,
            ops,
            slot_base=state_before.tail.end_slot(pipeline.num_slots),
            has_staged_next_k=True,
        )

    def is_loop_tail(
        self, state_before: PipelineState, ops: DenseFragmentOps
    ) -> bool:
        """Whether a loop body may follow the segment that left
        state_before."""
        pipeline = self.pipeline
        signatures = [
            tail.signature(pipeline.delays, pipeline.num_slots)
            for tail in (state_before.tail, self.loop_tail(ops, state_before))
        ]
        return signatures[0] == signatures[1]

    def run_rest_of_row(
        self,
        si: jax.Array,
        row: DenseRow,
        ahead: tuple[BlockDesc, BlockDesc],
        entry: LoopEntry,
        slot: jax.Array,
    ) -> tuple[tuple, jax.Array, jax.Array]:
        """The rolled visits of row si after its first segments (entry), then
        its last body, which ends the row's fragments, and the drain; slot is
        the slot of the block the first segments ran. Returns the kv carry,
        the last visit's slot and the row's guard flag."""
        traits, pipeline = self.traits, self.pipeline
        num_visits, _, ops, kv_carry, row_end = row
        state_before, first_visit, _ = entry
        assumed = self.loop_tail(ops, state_before)
        pipeline.check_loop_tail(state_before.tail, assumed)
        if self.has_checked_units:
            run_visit = self.run_checked_visit
        else:
            run_visit = self.run_visit
        kv_carry, probs, last_slot = lax.fori_loop(
            first_visit,
            num_visits,
            partial(run_visit, si, row, ahead, entry, assumed),
            (kv_carry, state_before.probs, slot),
        )
        end_ops = ops.with_tiles(
            last_slot, traits.num_k_compute_tiles - LOOP_BODY_TILES, row_end()
        )
        last_state = pipeline.advance(
            PipelineState(probs=probs, tail=assumed),
            body_items(traits, ROW_END),
            end_ops,
        )
        pipeline.drain(last_state)
        return (
            kv_carry,
            last_slot,
            guard_flag(traits, end_ops.row_end.largest_sum),
        )

    def run_visit(
        self,
        si: jax.Array,
        row: DenseRow,
        ahead: tuple[BlockDesc, BlockDesc],
        entry: LoopEntry,
        assumed: Tail,
        visit: jax.Array,
        carry: tuple,
    ) -> tuple:
        """Row si's visit visit in a build without checked units: the block
        step, then its bodies after those the row's first segments ran; the
        last visit leaves its last body to the row's end. carry is the visit
        loop's (kv carry, probs, slot of the visit before)."""
        num_visits, base_head, ops, _, _ = row
        _, first_visit, first_visit_body = entry
        kv_carry, probs, _ = carry
        kv_carry, slot, next_slot = self.step_block(
            si, visit, num_visits, ahead, base_head, kv_carry
        )
        probs = lax.fori_loop(
            jnp.where(visit == first_visit, first_visit_body, 0),
            jnp.where(
                visit == num_visits - 1,
                self.bodies_per_block - 1,
                self.bodies_per_block,
            ),
            partial(self.run_body, ops, assumed, slot, next_slot),
            probs,
        )
        return kv_carry, probs, slot

    def run_checked_visit(
        self,
        si: jax.Array,
        row: DenseRow,
        ahead: tuple[BlockDesc, BlockDesc],
        entry: LoopEntry,
        assumed: Tail,
        visit: jax.Array,
        carry: tuple,
    ) -> tuple:
        """Row si's visit visit in a build with checked units, as run_visit:
        its first body checks the unit before (checked_body) unless the row's
        first segments ran it, finishing that unit's last tile, which writes
        the visit's checkpoint (unit_end_ops); the block two visits ahead
        starts once that unit's block can no longer replay from its slot;
        the second body finishes the first one's items (run_second_body);
        the block's last body writes the next unit's checkpoint on its last
        tile, and so do the bodies between, the last visit's included."""
        num_visits, base_head, ops, _, _ = row
        _, first_visit, first_visit_body = entry
        kv_carry, probs, prev_slot = carry
        slot, next_slot = kv_carry[:2]
        self.wait_next_block(kv_carry)
        unit_end_tail = dataclasses.replace(
            assumed,
            ops=self.unit_end_ops(ops, visit, LOOP_BODY_TILES - 1),
        )

        def _checked(probs):
            state = self.checked_body(
                row,
                visit,
                prev_slot,
                slot,
                PipelineState(probs=probs, tail=unit_end_tail),
            )
            self.pipeline.check_loop_tail(state.tail, assumed)
            return state.probs

        if first_visit_body == 0:
            probs = _checked(probs)
        else:
            probs = lax.cond(
                visit == first_visit, lambda probs: probs, _checked, probs
            )
        kv_carry = self.start_ahead_block(
            kv_carry,
            self.ahead_block(si, visit, num_visits, ahead, base_head, kv_carry),
        )
        second_body = partial(
            self.run_second_body, ops, assumed, (slot, next_slot)
        )
        is_last_visit = visit == num_visits - 1
        # The last visit writes over the checkpoint of its own unit, which the
        # row's final unit does not read.
        writing_ops = self.unit_end_ops(
            ops,
            visit + jnp.logical_not(is_last_visit).astype(jnp.int32),
            LOOP_BODY_TILES - 1,
        )
        if self.bodies_per_block == SECOND_BODY + 1:
            # The second body is the block's last; the last visit's ends the
            # row's fragments.
            return (
                kv_carry,
                lax.cond(
                    is_last_visit,
                    lambda probs: probs,
                    partial(second_body, writing_ops),
                    probs,
                ),
                slot,
            )
        # Note (david): the second body is its own segment, not a loop body,
        # so the ahead block's DMA start joins its basic block.
        probs = lax.fori_loop(
            SECOND_BODY + 1,
            jnp.where(
                is_last_visit,
                self.bodies_per_block - 1,
                self.bodies_per_block,
            ),
            partial(self.run_body, writing_ops, assumed, slot, next_slot),
            second_body(ops, probs),
        )
        return kv_carry, probs, slot

    def run_second_body(
        self,
        ops: DenseFragmentOps,
        assumed: Tail,
        slots: tuple[jax.Array, jax.Array],
        body_ops: DenseFragmentOps,
        probs: tuple[jax.Array, ...],
    ) -> tuple[jax.Array, ...]:
        """Body SECOND_BODY of a checked visit whose block and the next one
        sit at slots, its items with body_ops; it finishes the first body's
        items with ops (FlaggedTailOps): once a replay redid them, they add
        nothing."""
        slot, next_slot = slots
        tail = dataclasses.replace(
            assumed,
            ops=FlaggedTailOps(ops, self.check_refs.unit_guard[0] > 0, PLAIN),
        )
        return self.run_body(
            body_ops, tail, slot, next_slot, SECOND_BODY, probs
        )

    def unit_end_ops(
        self, ops: DenseFragmentOps, unit: jax.Array | int, tile: int
    ) -> DenseFragmentOps:
        """ops whose items of tile key tile, the last tile of check unit
        unit - 1, also write the row state they leave into unit's checkpoint
        (UnitCheckpoint)."""
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
        prev_slot: jax.Array,
        slot: jax.Array,
        state_before: PipelineState,
    ) -> PipelineState:
        """Run visit visit's first body, of the block at slot, after
        state_before. The row sums its first tile's items write bound the
        final sums of check unit visit - 1, the block at prev_slot, which
        replays through the body's tiles at its end (replay_unit) when they
        flag it, unless it is in the row's final unit. Returns the body's
        pipeline state."""
        pipeline, check_refs = self.pipeline, self.check_refs
        num_visits, _, ops, _, _ = row
        largest = LargestSums()
        next_ops = ops.with_tiles(slot, LOOP_BODY_TILES)
        state = pipeline.advance(
            state_before,
            self.loop_items,
            dataclasses.replace(ops.with_tiles(slot, 0), largest=largest),
            next_k_tile=lambda: next_ops.k_tile(0),
        )
        unit = visit - 1
        check_refs.unit_guard[0] = jnp.where(
            unit < self.final_unit(num_visits),
            guard_flag(self.traits, largest.value),
            0,
        )

        @pl.when(check_refs.unit_guard[0] > 0)
        def _replay():
            self.replay_unit(
                ops.q_slot,
                visit,
                prev_slot,
                slot,
                pipeline.free_slots(state.tail, REPLAY_RESULT_SLOTS),
            )

        return state

    def run_causal_row(
        self, si: jax.Array, row: DenseRow, ahead: tuple[BlockDesc, BlockDesc]
    ) -> tuple[tuple, jax.Array, jax.Array]:
        """A causal row: its diagonal block, then the interior blocks.
        Returns run_rest_of_row's values."""
        traits, pipeline = self.traits, self.pipeline
        num_visits, base_head, ops, kv_carry, row_end = row
        kv_carry, slot, next_slot = self.step_block(
            si, 0, num_visits, ahead, base_head, kv_carry
        )
        diagonal_ops = ops.with_tiles(slot, 0)
        head, last_tile = diagonal_items(traits, ends_row=False)
        head_state = pipeline.advance(
            pipeline.idle_state(),
            head,
            diagonal_ops,
            next_k_tile=lambda: diagonal_ops.k_tile(0),
        )

        def _with_interior(kv_carry):
            row_with_carry = row._replace(kv_carry=kv_carry)
            next_ops = ops.with_tiles(next_slot, 0)
            diagonal_state = pipeline.advance(
                head_state,
                last_tile,
                self.last_diagonal_tile_ops(diagonal_ops),
                next_k_tile=lambda: next_ops.k_tile(0),
            )
            if self.is_loop_tail(diagonal_state, ops):
                return self.run_rest_of_row(
                    si,
                    row_with_carry,
                    ahead,
                    LoopEntry(diagonal_state, 1, 0),
                    slot,
                )
            # A diagonal tile shorter than a body's tail holds row-start items
            # in its tail, so the first interior body runs as its own segment.
            bridge_state = self.run_bridge(row, slot, next_slot, diagonal_state)
            return self.run_rest_of_row(
                si, row_with_carry, ahead, LoopEntry(bridge_state, 1, 1), slot
            )

        def _diagonal_only(kv_carry):
            end_ops = ops.with_tiles(slot, 0, row_end())
            _, end_tile = diagonal_items(traits, ends_row=True)
            pipeline.drain(pipeline.advance(head_state, end_tile, end_ops))
            return (
                kv_carry,
                slot,
                guard_flag(traits, end_ops.row_end.largest_sum),
            )

        # A build of one kv block has diagonal-only rows.
        if self.schedule.num_kv_blocks == 1:
            return _diagonal_only(kv_carry)
        return lax.cond(
            num_visits > 1, _with_interior, _diagonal_only, kv_carry
        )

    def last_diagonal_tile_ops(
        self, diagonal_ops: DenseFragmentOps
    ) -> DenseFragmentOps:
        """The ops of a causal row's diagonal tile 0, the last tile of its
        diagonal block, check unit 0, which writes unit 1's checkpoint in a
        build with checked units."""
        if not self.has_checked_units:
            return diagonal_ops
        return self.unit_end_ops(diagonal_ops, unit=1, tile=0)

    def run_bridge(
        self,
        row: DenseRow,
        slot: jax.Array,
        next_slot: jax.Array,
        diagonal_state: PipelineState,
    ) -> PipelineState:
        """A causal row's first interior body, of the block at next_slot, as
        its own segment after its diagonal block (at slot), which left
        diagonal_state; it checks unit 0 in a build with checked units."""
        if self.has_checked_units:
            return self.checked_body(row, 1, slot, next_slot, diagonal_state)
        next_ops = row.ops.with_tiles(next_slot, 0)
        return self.pipeline.advance(
            diagonal_state,
            self.loop_items,
            next_ops,
            next_k_tile=lambda: next_ops.k_tile(LOOP_BODY_TILES),
        )

    def run_full_row(
        self, si: jax.Array, row: DenseRow, ahead: tuple[BlockDesc, BlockDesc]
    ) -> tuple[tuple, jax.Array, jax.Array]:
        """A row without a mask: its row-start segment, then the loop.
        Returns run_rest_of_row's values."""
        pipeline = self.pipeline
        if self.has_checked_units:
            # The first visit's first body is the row-start segment, which
            # checks nothing.
            self.check_refs.unit_guard[0] = jnp.int32(0)
        slot = row.kv_carry[0]
        start_ops = row.ops.with_tiles(slot, 0)
        start_state = pipeline.advance(
            pipeline.idle_state(),
            row_start_items(self.traits),
            start_ops,
            next_k_tile=lambda: start_ops.k_tile(LOOP_BODY_TILES),
        )
        return self.run_rest_of_row(
            si, row, ahead, LoopEntry(start_state, 0, 1), slot
        )

    def replay_unit(
        self,
        q_slot: jax.Array,
        visit: jax.Array | int,
        prev_slot: jax.Array,
        slot: jax.Array,
        result_slots: tuple[jax.Array, ...],
    ) -> None:
        """Recompute check unit visit - 1, the block at prev_slot, and the
        first body of visit visit, at slot, into the row state with the exact
        rebasing softmax, between that body and the next: from the unit's
        checkpoint, or from the row start at unit 0. Visit visit's checkpoint
        is written again as its first tile starts and rebased at the end to
        the anchor its tiles raised (rebase_checkpoint), so a checkpoint
        always holds the row state on the row's current anchor."""
        unit = visit - 1
        checkpoints = self.row_checkpoints

        @pl.when(unit > 0)
        def _restore():
            checkpoints.copy_checkpoint(unit, is_restore=True)

        self.replay_block(
            q_slot,
            prev_slot,
            self.traits.num_k_compute_tiles,
            unit == 0,
            result_slots,
        )
        checkpoints.copy_checkpoint(visit, is_restore=False)
        checkpoints.save_anchor(unit)
        self.replay_block(q_slot, slot, LOOP_BODY_TILES, False, result_slots)
        checkpoints.rebase_checkpoint(visit, unit)

    def replay_block(
        self,
        q_slot: jax.Array,
        kv_slot: jax.Array,
        num_tiles: int,
        starts_row: jax.Array | bool,
        result_slots: tuple[jax.Array, ...],
    ) -> None:
        """Recompute the first num_tiles tiles of the kv block at kv_slot into
        the row state, every fragment against each tile on QK_MXU through
        result_slots (replay_fragment), leaving lane partial sums. starts_row
        marks the row's first block: its tile 0 starts every fragment's row,
        and in a causal build it is the diagonal block, whose tile t meets
        the fragments from FRAGMENTS_PER_TILE * t on and masks them."""
        is_diagonal = jnp.logical_and(self.blocks.is_diagonal_first, starts_row)
        ops = self.row_ops(q_slot)

        def _replay_tile(tile, carry):
            tile_ops = ops.with_tiles(kv_slot, tile)
            is_row_start = jnp.logical_and(starts_row, tile == 0)

            def _replay_fragment(fragment, fragment_carry):
                self.replay_fragment(
                    tile_ops, fragment, is_row_start, is_diagonal, result_slots
                )
                return fragment_carry

            return lax.fori_loop(
                jnp.where(is_diagonal, tile * FRAGMENTS_PER_TILE, 0),
                self.num_fragments,
                _replay_fragment,
                carry,
            )

        lax.fori_loop(0, num_tiles, _replay_tile, None)

    def replay_fragment(
        self,
        ops: DenseFragmentOps,
        fragment: jax.Array,
        is_row_start: jax.Array,
        is_diagonal: jax.Array,
        result_slots: tuple[jax.Array, ...],
    ) -> None:
        """Fold fragment fragment against tile key 0 of ops into its rows of
        the row state with the exact rebasing softmax
        (fwd_mxu.fold_replayed_fragment), both matmuls on QK_MXU through
        result_slots, the (scores, probs @ V) slots; is_diagonal masks the
        scores as a diagonal block's."""
        variant, state = self.variant, self.state
        score_slot, value_slot = result_slots
        item = WorkItem(fragment=fragment, tile=0)
        logits = scaled_logits(
            variant,
            replay_matmul(
                score_slot, ops.q_fragment(item), ops.k_tile(0), transpose=True
            ),
        )
        if self.blocks.is_diagonal_first:
            masked = variant.mask_logits(
                logits,
                fragment * FRAGMENT_ROWS,
                ops.first_tile * MXU_COLUMNS,
                logits.shape,
            )
            logits = jnp.where(is_diagonal, masked, logits)
        rows = fragment_rows(fragment)
        fold_replayed_fragment(
            state.row_max.at[rows],
            state.row_sum.at[MXU_ROW_PARITY, rows],
            state.o.at[MXU_ROW_PARITY, rows],
            logits,
            is_row_start,
            value_slot,
            partial(ops.v_tile, 0),
        )

    @property
    def row_checkpoints(self) -> RowCheckpoints:
        """The checkpoints of a row's check units (check_refs)."""
        return RowCheckpoints(
            checkpoint_sum=self.check_refs.checkpoint_sum,
            checkpoint_o=self.check_refs.checkpoint_o,
            row_sum=self.state.row_sum,
            o=self.state.o,
            parity=MXU_ROW_PARITY,
            row_anchor=lambda rows: self.state.row_max[rows, :],
        )

    def replay_row_end(
        self,
        finish: RowFinish,
        num_visits: jax.Array,
        q_slot: jax.Array,
        slot: jax.Array,
    ) -> None:
        """Recompute the final unit (final_unit) of the row of finish after
        its drain, from the unit's checkpoint or the row start, its blocks in
        visit order, each staged again into slot, the row's last block's
        slot, which the last visit leaves as it found it."""
        final_visit = self.final_unit(num_visits)
        # A drained pipeline holds no result.
        result_slots = self.pipeline.qk_slots[:REPLAY_RESULT_SLOTS]
        if self.has_checked_units:

            @pl.when(final_visit > 0)
            def _restore():
                self.row_checkpoints.copy_checkpoint(
                    final_visit, is_restore=True
                )

        def _replay_visit(visit, carry):
            desc = self.blocks.desc(
                finish.base_head, self.visit_block(finish.si, visit)
            )
            self.blocks.start(desc, slot)
            wait_rotated_kv_block(self.traits, self.kv_refs, slot)
            self.replay_block(
                q_slot,
                slot,
                self.traits.num_k_compute_tiles,
                visit == 0,
                result_slots,
            )
            return carry

        lax.fori_loop(final_visit, num_visits, _replay_visit, None)

    def normalize_row_out(self, out_slot: jax.Array) -> None:
        """Normalize a replayed row's output into its stage slot, as
        finish_row_out does, whose slot wait the row already ran."""
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
        unscaled: the scores take the Q scale (scaled_logits)."""
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
        """An explicit row: one pipeline from its first tile to its drain,
        whose last items finish the row's sums and output rows, then the
        guard replay, the lse and the output copy. carry is dense_fwd_body's
        row carry."""
        traits, blocks = self.traits, self.blocks
        q_loaded, q_inflight, q_slot, kv_carry, num_out_copies = carry
        lo, hi, _ = self.schedule.row_interval(si)
        num_visits = hi - lo + 1
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
        if blocks.is_diagonal_first:
            run = self.run_causal_row
        else:
            run = self.run_full_row
        kv_carry, last_slot, row_guard = run(
            si,
            DenseRow(
                num_visits,
                base_head,
                self.row_ops(q_slot),
                kv_carry,
                lambda: RowEnd(out_slot, sink),
            ),
            (first_next_desc, second_next_desc),
        )
        self.finish_row(
            g, finish, sink, row_guard, num_visits, q_slot, last_slot
        )
        return q_loaded, q_inflight, q_slot, kv_carry, num_out_copies + 1

    def row_sink(self, base_head: jax.Array) -> jax.Array | None:
        """The sink logit of the pipeline head at base_head, None without
        sinks."""
        if not self.variant.has_sinks:
            return None
        return head_sinks(
            self.traits,
            self.operands.sinks_ref,
            self.schedule.heads_outer_batch,
            base_head,
        )[0]

    def row_ops(self, q_slot: jax.Array) -> DenseFragmentOps:
        """The ops of the row whose Q sits in staging slot q_slot, without a
        kv block yet (DenseFragmentOps.with_tiles)."""
        return DenseFragmentOps(
            self.traits,
            self.variant,
            self.operands,
            self.q_refs,
            self.kv_refs,
            self.state,
            self.out_refs.o_stage,
            q_slot,
        )

    def finish_row(
        self,
        g: jax.Array,
        finish: RowFinish,
        sink: jax.Array | None,
        row_guard: jax.Array,
        num_visits: jax.Array,
        q_slot: jax.Array,
        last_slot: jax.Array,
    ) -> None:
        """What a row's last items leave to do: its lse, the replay of the
        final unit of a row whose guard flag row_guard is set (replay_row_end),
        whose lane partial sums reduce before its own lse, sinks and output,
        and the output copy. q_slot and last_slot hold the row's Q and its
        last block."""
        traits, variant = self.traits, self.variant
        out_slot = row_out_slot(traits, finish)
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
            self.replay_row_end(finish, num_visits, q_slot, last_slot)
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
