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

"""Union attention: the tokens of each union attend the union's gathered
rows, a dense MQA over the union's positions with each token masked to the
positions it selects; the TensorCore half of the union route.

A call runs its unions as one stream of periods of PERIOD_KEYS keys, two
MXU key tiles each, in a software pipeline written with the explicit MXU
operations. Period k opens with the P @ V of period k - 2, value half m on
MXU m, into accumulators that stay in the MRB until the union ends; then
MXU m runs the QK of tile m of period k over all of the union's score
rows, while the scores of period k - 1 pop into the softmax that stages
their fp8 probabilities and the keys of period k + 1 stage from the
gathered rows. The pipeline runs across unions without draining: a union's
second period finishes the union before.

The softmax takes a fixed anchor per row: the row's maximum over the
union's first period. A later score more than log2(FP8_MAX) above it would
overflow the fp8 probabilities, which a per-lane sum over the guard limit
flags; a flagged union is replayed with the online max and rescale.

Score rows are a union's tokens times its heads, token-major (row t * H + h);
scores are in log2 units.
"""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from flywheel_tpu.core.tpu import NUM_LANES
from flywheel_tpu.softmax_attention.config import LOG2E, vmem_limit_bytes
from flywheel_tpu.softmax_attention.kernels.decode import lane_broadcast
from flywheel_tpu.softmax_attention.kernels.mla_decode import (
    LATENT_DIM,
    ROPE_DIM,
)
from flywheel_tpu.sparse_attention.dsa.kernels.gather import (
    NOPE_WORDS,
    QUAD_TOKENS,
    ROPE_TOKENS_PER_ROW,
)
from flywheel_tpu.sparse_attention.dsa.kernels.masked_dense import (
    FLAG_QUERY,
    FP8_DTYPE,
    KEY_ROW_DIM,
    MASKED_KEY,
    MAX_BLOCK_Q,
    NO_KEY_MAX,
    ROPE_LANES,
    bitmap_rows,
    selected_keys,
    stage_keys,
)

# Note (david): the gathered rows of a union start at a multiple of
# UNION_BLOCK_KV and run through whole blocks, which the routing prices.
UNION_BLOCK_KV = 1024
MXU_TILE = 256
PERIOD_TILES = 2
PERIOD_KEYS = PERIOD_TILES * MXU_TILE
# Note (david): a union's second period finishes the union before it, so
# every union takes at least two.
MIN_PERIODS = 2
NUM_MXUS = 2
FRAGMENT_ROWS = 32
QK_PASSES = 3
KEY_LANES = QK_PASSES * MXU_TILE
# Note (david): the rope pass latches its keys twice, so a period latches an
# even number of RHS tiles per MXU and every period uses the staging
# registers in the same order.
ROPE_PASS_SPLIT = 2
PV_LATCHES = PERIOD_TILES
# Note (david): a period's P @ V takes the tiles of two periods back, whose
# probabilities the period before staged, ahead of its QK.
PV_LAG = 2
LATCHES_PER_PERIOD = QK_PASSES - 1 + ROPE_PASS_SPLIT + PV_LATCHES
NUM_STAGING_REGISTERS = 2
# Note (david): period k stages period k + 1's keys while its P @ V reads
# period k - 2's. Slot counts are powers of two, which keeps the slot
# arithmetic of a period's scalar prologue short.
NUM_KEY_SLOTS = 4
NUM_ROW_SLOTS = 8
# Note (david): a period's row copy starts ROW_LEAD periods ahead of it and
# is waited one period before it: the copies queue behind each other and
# the union copies, and a shorter lead left the waits stalling periods.
ROW_LEAD = 7
NUM_PROBABILITY_SLOTS = 2
# Note (david): a union's queries and bitmap land while the union two ahead
# of it still runs; row state and output live a union past their last use.
NUM_UNION_SLOTS = 4
NUM_STATE_SLOTS = 2
NUM_OUTPUT_SLOTS = 2
FP8_MAX = 448.0
HALF_NO_KEY = 0.5 * NO_KEY_MAX
# The schedule's rows: the header, then a row per union. Lanes of a union's
# row: the call's row of its first token, its tokens, its size, its first
# gathered row, its periods (union_periods) and the next union with tokens.
ROW_FIRST_TOKEN = 0
ROW_NUM_TOKENS = 1
ROW_SIZE = 2
ROW_FIRST_ROW = 3
ROW_NUM_PERIODS = 4
ROW_NEXT_UNION = 5
ROW_WORDS = 16
# Lanes of the schedule's header: the call's first and last unions with
# tokens and its real tokens.
HEADER_FIRST_UNION = 0
HEADER_LAST_UNION = 1
HEADER_NUM_REAL = 2
HEADER_WORDS = 16
HEADER_ROW = 0
SEM_NOPE = 0
SEM_ROPE = 1
SEM_QUERY = 2
SEM_OUTPUT = 3
NUM_SEMS = 4
NUM_SEM_SLOTS = max(NUM_ROW_SLOTS, NUM_UNION_SLOTS, NUM_OUTPUT_SLOTS)
# Lanes of an output slot's pending copy in SMEM: its first token and its
# tokens, 0 when none is in flight.
PENDING_FIRST_TOKEN = 0
PENDING_NUM_TOKENS = 1
PENDING_WORDS = 2


class UnionAttentionParams(NamedTuple):
    """Static parameters of one union attention call. has_previous_out: out
    aliases an earlier call's out; else the call zeros the rows past its real
    tokens. force_replay: every union replays, which tests the replay."""

    union_tokens: int
    num_heads: int
    num_unions: int
    num_slabs: int
    bitmap_rows: int
    sm_scale_log2: float
    has_previous_out: bool
    force_replay: bool


class UnionAttentionRefs(NamedTuple):
    """The union attention kernel's operands, outputs and scratch."""

    schedule: jax.Array
    kv_scale: jax.Array
    q_nope: jax.Array
    q_pe_t: jax.Array
    nope_rows: jax.Array
    rope_rows: jax.Array
    bitmap: jax.Array
    previous_out: jax.Array | None
    out: jax.Array
    nope_ring: jax.Array
    rope_ring: jax.Array
    keys: jax.Array
    rope_words: jax.Array
    probabilities: jax.Array
    anchor: jax.Array
    row_sum: jax.Array
    query_buffer: jax.Array
    q_pe_stage: jax.Array
    bitmap_stage: jax.Array
    out_stage: jax.Array
    replay_out: jax.Array
    pending_out: jax.Array
    replay: jax.Array
    sems: jax.Array


class UnionRow(NamedTuple):
    """A union's row of the schedule."""

    first_token: jax.Array
    num_tokens: jax.Array
    size: jax.Array
    first_row: jax.Array
    num_periods: jax.Array
    next_union: jax.Array


class Period(NamedTuple):
    """A period of the stream: its union, its place in the union, and the
    union's rank among the call's unions with periods, which picks the
    union's slots."""

    union: jax.Array
    position: jax.Array
    ordinal: jax.Array


class UnionAttentionKernel:
    """One union attention call over the schedule's periods."""

    def __init__(
        self,
        params: UnionAttentionParams,
        refs: UnionAttentionRefs,
        accumulators: tuple,
    ):
        self.params = params
        self.refs = refs
        num_fragments = params.union_tokens * params.num_heads // FRAGMENT_ROWS
        self.num_fragments = num_fragments
        # Note (david): run_scoped hands the accumulators QK first, then
        # P @ V, each MXU's fragments together.
        per_mxu = [
            accumulators[i * num_fragments : (i + 1) * num_fragments]
            for i in range(2 * NUM_MXUS)
        ]
        self.qk_accumulators = per_mxu[:NUM_MXUS]
        self.pv_accumulators = per_mxu[NUM_MXUS:]
        self.first_union = refs.schedule[HEADER_ROW, HEADER_FIRST_UNION]
        self.last_union = refs.schedule[HEADER_ROW, HEADER_LAST_UNION]
        self.num_periods = lax.fori_loop(
            self.first_union,
            self.last_union + 1,
            lambda union, total: (
                total + refs.schedule[HEADER_ROW + 1 + union, ROW_NUM_PERIODS]
            ),
            jnp.int32(0),
        )
        self.kv_scale = refs.kv_scale[0]
        # Note (david): K and V share the cache's kv_scale, so it folds into
        # the score scale and the output normalization.
        self.score_scale = self.kv_scale * params.sm_scale_log2

    def union_row(self, union: jax.Array) -> UnionRow:
        # Note (david): a cursor before the first union or past the last
        # (the stream's ends) reads the last union's row, whose copies are
        # real; only stale work then uses them.
        union = jnp.where(
            (union < 0) | (union > self.last_union), self.last_union, union
        )
        row = HEADER_ROW + 1 + union
        return UnionRow(*(
            self.refs.schedule[row, lane] for lane in range(ROW_NEXT_UNION + 1)
        ))

    def next_period(self, period: Period) -> Period:
        row = self.union_row(period.union)
        is_union_end = period.position + 1 >= row.num_periods
        return Period(
            jnp.where(is_union_end, row.next_union, period.union),
            jnp.where(is_union_end, 0, period.position + 1),
            period.ordinal + is_union_end.astype(jnp.int32),
        )

    def fragment_rows(self, fragment: int) -> tuple:
        """Fragment's (token, head window) in the union's token-major rows."""
        per_token = self.params.num_heads // FRAGMENT_ROWS
        return fragment // per_token, pl.ds(
            (fragment % per_token) * FRAGMENT_ROWS, FRAGMENT_ROWS
        )

    def row_copies(self, period: Period, slot: jax.Array) -> tuple:
        """The period's nope and rope copies: one contiguous range of its
        union's gathered rows each."""
        first_row = pl.multiple_of(
            self.union_row(period.union).first_row
            + period.position * PERIOD_KEYS,
            PERIOD_KEYS,
        )
        rope_rows = PERIOD_KEYS // ROPE_TOKENS_PER_ROW
        first_rope_row = pl.multiple_of(
            first_row // ROPE_TOKENS_PER_ROW, rope_rows
        )
        refs = self.refs
        return (
            pltpu.make_async_copy(
                refs.nope_rows.at[pl.ds(first_row, PERIOD_KEYS)],
                refs.nope_ring.at[slot],
                refs.sems.at[SEM_NOPE, slot],
            ),
            pltpu.make_async_copy(
                refs.rope_rows.at[pl.ds(first_rope_row, rope_rows)],
                refs.rope_ring.at[slot],
                refs.sems.at[SEM_ROPE, slot],
            ),
        )

    def union_copies(self, union: jax.Array, slot: jax.Array) -> tuple:
        """The union's query and bitmap copies into its union slot."""
        refs = self.refs
        row = self.union_row(union)
        sem = refs.sems.at[SEM_QUERY, slot]
        tokens = pl.ds(row.first_token, row.num_tokens)
        staged = pl.ds(0, row.num_tokens)
        rows = self.params.bitmap_rows
        return (
            pltpu.make_async_copy(
                refs.q_nope.at[tokens],
                refs.query_buffer.at[slot, staged, :, pl.ds(0, LATENT_DIM)],
                sem,
            ),
            pltpu.make_async_copy(
                refs.q_pe_t.at[tokens], refs.q_pe_stage.at[slot, staged], sem
            ),
            pltpu.make_async_copy(
                refs.bitmap.at[
                    pl.ds(row.first_token * rows, row.num_tokens * rows)
                ],
                refs.bitmap_stage.at[slot, pl.ds(0, row.num_tokens * rows)],
                sem,
            ),
        )

    def output_copy(
        self, first_token: jax.Array, num_tokens: jax.Array, slot: jax.Array
    ):
        return pltpu.make_async_copy(
            self.refs.out_stage.at[slot, pl.ds(0, num_tokens)],
            self.refs.out.at[pl.ds(first_token, num_tokens)],
            self.refs.sems.at[SEM_OUTPUT, slot],
        )

    def wait_output(self, slot: int | jax.Array) -> None:
        """Waits the output copy in flight from slot, if any."""
        pending = self.refs.pending_out
        num_tokens = pending[slot, PENDING_NUM_TOKENS]

        @pl.when(num_tokens > 0)
        def _wait():
            self.output_copy(
                pending[slot, PENDING_FIRST_TOKEN], num_tokens, slot
            ).wait()

    def start_rows(self, period: Period, index: jax.Array) -> None:
        """Starts period index's row copies, if it is one of the stream's."""

        @pl.when(index < self.num_periods)
        def _start():
            for copy in self.row_copies(period, lax.rem(index, NUM_ROW_SLOTS)):
                copy.start()

    def wait_rows(self, period: Period, index: jax.Array) -> None:
        @pl.when(index < self.num_periods)
        def _wait():
            for copy in self.row_copies(period, lax.rem(index, NUM_ROW_SLOTS)):
                copy.wait()

    def place_queries(self, union: jax.Array, ordinal: jax.Array) -> None:
        """Waits for the union's copies and puts its rope queries in their
        lanes."""
        refs = self.refs
        slot = lax.rem(ordinal, NUM_UNION_SLOTS)
        for copy in self.union_copies(union, slot):
            copy.wait()
        # Note (david): the rope queries arrive heads minor and narrower than a
        # lane tile, so no copy window can address their lanes; a transpose and
        # vector stores place them.
        q_pe = jnp.swapaxes(refs.q_pe_stage[slot].astype(jnp.float32), 1, 2)
        refs.query_buffer[slot, :, :, pl.ds(LATENT_DIM, ROPE_DIM)] = (
            q_pe.astype(refs.query_buffer.dtype)
        )

    def stage(self, period: Period, index: jax.Array) -> None:
        """Period index's keys into its key slot, each token masked to its
        bitmap; tokens past the union's size are zero."""
        first_key = period.position * PERIOD_KEYS
        union_slot = lax.rem(period.ordinal, NUM_UNION_SLOTS)
        row_slot = lax.rem(index, NUM_ROW_SLOTS)

        def _mask_lanes():
            is_selected = selected_keys(
                self.refs.bitmap_stage.at[union_slot],
                first_key,
                block_q=self.params.union_tokens,
                block_kv=PERIOD_KEYS,
                num_slabs=self.params.num_slabs,
                rows_per_token=self.params.bitmap_rows,
            )
            lane = lax.broadcasted_iota(jnp.int32, is_selected.shape, 1)
            return jnp.where(
                lane >= ROPE_LANES, jnp.where(is_selected, 0.0, MASKED_KEY), 0.0
            )

        stage_keys(
            self.refs.nope_ring.at[row_slot],
            self.refs.rope_ring.at[row_slot],
            self.refs.keys.at[lax.rem(index, NUM_KEY_SLOTS)],
            self.refs.rope_words,
            self.union_row(period.union).size - first_key,
            _mask_lanes,
        )

    def key_tile(
        self, index: jax.Array, tile: int, lane_tile: int, is_current: bool
    ) -> jax.Array:
        """Lane tile lane_tile of key tile `tile` of period index. A period's
        reads of its own and the period before's slots carry no memory
        dependency on the staging of the next period's slot."""
        keys = self.refs.keys
        if is_current:
            keys = pltpu.annotate(keys, no_hazard_no_deps=True)
        return keys[
            lax.rem(index + NUM_KEY_SLOTS, NUM_KEY_SLOTS),
            pl.ds(tile * MXU_TILE, MXU_TILE),
            pl.ds(lane_tile * MXU_TILE, MXU_TILE),
        ]

    def latch_operands(self, index: jax.Array, mxu: int) -> list:
        """Period index's RHS tiles on mxu in latch order, (tile, transpose):
        the value halves of period index - 2, then its tile's QK passes."""
        pv = [
            functools.partial(self.key_tile, index - PV_LAG, tile, mxu, True)
            for tile in range(PV_LATCHES)
        ]
        qk = [
            functools.partial(self.key_tile, index, mxu, lane_tile, True)
            for lane_tile in range(QK_PASSES - 1)
        ] + [
            functools.partial(self.key_tile, index, mxu, QK_PASSES - 1, True)
        ] * ROPE_PASS_SPLIT
        return [(operand, False) for operand in pv] + [
            (operand, True) for operand in qk
        ]

    def push(self, operand, transpose: bool, latch: int, mxu: int) -> None:
        pltpu.matmul_push_rhs(
            operand(), latch % NUM_STAGING_REGISTERS, mxu, transpose=transpose
        )

    def push_first(self, index: jax.Array, mxu: int) -> None:
        """Pushes period index's first RHS tile on mxu: value half mxu of the
        first tile of period index - PV_LAG."""
        self.push(
            functools.partial(self.key_tile, index - PV_LAG, 0, mxu, True),
            False,
            0,
            mxu,
        )

    def matmul(
        self, accumulator, lhs, mxu: int, latch: int | None, operands, index
    ) -> None:
        """One accumulate; at a latch, the next RHS tile pushes."""
        staged = None if latch is None else latch % NUM_STAGING_REGISTERS
        pltpu.matmul_acc_lhs(accumulator, lhs, load_staged_rhs=staged)
        if latch is None:
            return
        if latch + 1 < LATCHES_PER_PERIOD:
            self.push(*operands[latch + 1], latch + 1, mxu)
        else:
            self.push_first(index + 1, mxu)

    def consume(
        self, prev: Period, index: jax.Array, fragment: int, is_first: bool
    ) -> None:
        """Pops QK fragment `fragment` of both tiles of period index - 1 and
        stages its probabilities; the first period of a union sets each
        row's anchor to its maximum over the period."""
        refs = self.refs
        rows = pl.ds(fragment * FRAGMENT_ROWS, FRAGMENT_ROWS)
        state = lax.rem(prev.ordinal + NUM_STATE_SLOTS, NUM_STATE_SLOTS)
        anchors, row_sums = self.row_state()
        scores = [
            pltpu.matmul_pop(self.qk_accumulators[m][fragment])
            * self.score_scale
            for m in range(NUM_MXUS)
        ]
        if is_first:
            halves = [
                jnp.maximum(s[:, :NUM_LANES], s[:, NUM_LANES:]) for s in scores
            ]
            tile_max = lane_broadcast(
                jnp.maximum(*halves).max(axis=1, keepdims=True), NUM_LANES
            )
            # Note (david): a row with no key in the period keeps NO_KEY_MAX,
            # so its masked scores give 0; a later key then overflows the
            # guard.
            anchor = jnp.where(tile_max > HALF_NO_KEY, tile_max, NO_KEY_MAX)
            anchors[state, rows] = anchor
        else:
            anchor = anchors[state, rows]
        anchor = lane_broadcast(anchor, MXU_TILE)
        slot = lax.rem(index - 1 + NUM_PROBABILITY_SLOTS, NUM_PROBABILITY_SLOTS)
        partial = None if is_first else row_sums[state, rows]
        for m, s in enumerate(scores):
            probabilities = jnp.exp2(s - anchor)
            # Note (david): probabilities round to fp8 for P @ V, as the sparse
            # MLA's do; past FP8_MAX the guard replays the union.
            refs.probabilities[slot, m, rows] = probabilities.astype(FP8_DTYPE)
            halves = probabilities[:, :NUM_LANES] + probabilities[:, NUM_LANES:]
            partial = halves if partial is None else partial + halves
        row_sums[state, rows] = partial

    def row_state(self) -> tuple:
        """The anchors and partial sums as a period sees them: each access
        touches its own fragment's rows of one union's slot, so none depends
        on another through memory."""
        return (
            pltpu.annotate(self.refs.anchor, no_hazard_no_deps=True),
            pltpu.annotate(self.refs.row_sum, no_hazard_no_deps=True),
        )

    def finish_fragment(
        self, done: jax.Array, done_ordinal: jax.Array, fragment: int, flags
    ):
        """Normalizes fragment `fragment` of the finished union's rows out of
        the P @ V accumulators into its output slot; returns the guard flags
        with the fragment's lanes added: a valid row whose partial sum went
        past FP8_MAX or out of range."""
        refs = self.refs
        rows = pl.ds(fragment * FRAGMENT_ROWS, FRAGMENT_ROWS)
        state = lax.rem(done_ordinal + NUM_STATE_SLOTS, NUM_STATE_SLOTS)
        partial = self.row_state()[1][state, rows]
        row_sum = partial.sum(axis=1, keepdims=True)
        inverse = lane_broadcast(
            jnp.where(row_sum > 0.0, self.kv_scale / row_sum, 0.0), MXU_TILE
        )
        token, heads = self.fragment_rows(fragment)
        slot = lax.rem(done_ordinal + NUM_OUTPUT_SLOTS, NUM_OUTPUT_SLOTS)
        for m in range(NUM_MXUS):
            out = pltpu.matmul_pop(self.pv_accumulators[m][fragment]) * inverse
            refs.out_stage[
                slot, token, heads, pl.ds(m * MXU_TILE, MXU_TILE)
            ] = out.astype(refs.out_stage.dtype)
        is_valid = token < self.union_row(done).num_tokens
        is_unsafe = jnp.where((partial <= FP8_MAX) | ~is_valid, 0.0, 1.0)
        return is_unsafe if flags is None else jnp.maximum(flags, is_unsafe)

    def finish_union(
        self, done: jax.Array, done_ordinal: jax.Array, flags
    ) -> None:
        """Starts the finished union's output copy and records it for replay
        if its guard flags are set; the union before the first has none."""
        refs = self.refs
        row = self.union_row(done)
        slot = lax.rem(done_ordinal + NUM_OUTPUT_SLOTS, NUM_OUTPUT_SLOTS)
        is_real = done >= 0
        is_forced = self.params.force_replay
        is_flagged = is_real & ((jnp.max(flags) > 0.0) | is_forced)
        count = refs.replay[0]
        refs.replay[1 + count] = done
        refs.replay[0] = count + is_flagged.astype(jnp.int32)

        @pl.when(is_real)
        def _start():
            self.output_copy(row.first_token, row.num_tokens, slot).start()
            refs.pending_out[slot, PENDING_FIRST_TOKEN] = row.first_token
            refs.pending_out[slot, PENDING_NUM_TOKENS] = row.num_tokens

    def start_union(self, cur: Period) -> None:
        """In a union's second period: the next union's queries land and
        place, and the copies of the union after it start into the slot
        the union before the current one held."""
        next_union = self.union_row(cur.union).next_union
        self.place_queries(next_union, cur.ordinal + 1)
        after = self.union_row(next_union).next_union
        for copy in self.union_copies(
            after, lax.rem(cur.ordinal + 2, NUM_UNION_SLOTS)
        ):
            copy.start()

    def qk_latch(self, qk_pass: int, fragment: int) -> int | None:
        rope_fragments = self.num_fragments // ROPE_PASS_SPLIT
        if qk_pass < QK_PASSES - 1:
            return PV_LATCHES + qk_pass if fragment == 0 else None
        if fragment % rope_fragments:
            return None
        return PV_LATCHES + qk_pass + fragment // rope_fragments

    def run_period(
        self,
        carry: tuple,
        rows: tuple,
        index: jax.Array,
        is_second: bool,
    ) -> None:
        """Period index: P @ V of period index - PV_LAG, QK of its tiles with
        the softmax of the period before's and the staging of the next
        period's keys; a union's second period also finishes the union
        before. rows: the periods index + 2 and index + ROW_LEAD, whose row
        copies the period waits at its end and starts."""
        _, cur, _, before = carry
        after, newest = rows
        self.start_rows(newest, index + ROW_LEAD)
        if is_second:
            self.start_union(cur)
            self.wait_output(
                lax.rem(cur.ordinal - 1 + NUM_OUTPUT_SLOTS, NUM_OUTPUT_SLOTS)
            )
        operands = [self.latch_operands(index, m) for m in range(NUM_MXUS)]
        self.issue_pv(index, operands)
        flags = self.issue_qk(carry, index, operands, is_second)
        if is_second:
            self.finish_union(before, cur.ordinal - 1, flags)
        self.wait_rows(after, index + 2)

    def issue_qk(
        self, carry: tuple, index: jax.Array, operands: list, is_second: bool
    ):
        """The period's QK in program order, with the softmax of the period
        before's fragments ahead of the first pass that reuses their
        accumulators, the next period's staging and, in a second period,
        the finished union's output; returns its guard flags."""
        prev, cur, upcoming, before = carry
        query_slot = lax.rem(cur.ordinal, NUM_UNION_SLOTS)
        # Note (david): the union's queries were placed a union earlier; a
        # second period places the next union's into another slot.
        queries = pltpu.annotate(self.refs.query_buffer, no_hazard_no_deps=True)
        flags = None
        for qk_pass in range(QK_PASSES):
            if qk_pass == 1:
                self.stage(upcoming, index + 1)
            for f in range(self.num_fragments):
                if qk_pass == 0:
                    self.consume(prev, index, f, is_second)
                if qk_pass == 1 and is_second:
                    flags = self.finish_fragment(
                        before, cur.ordinal - 1, f, flags
                    )
                token, heads = self.fragment_rows(f)
                lhs = queries[
                    query_slot,
                    token,
                    heads,
                    pl.ds(qk_pass * MXU_TILE, MXU_TILE),
                ]
                latch = self.qk_latch(qk_pass, f)
                for m in range(NUM_MXUS):
                    self.matmul(
                        self.qk_accumulators[m][f],
                        lhs,
                        m,
                        latch,
                        operands[m],
                        index,
                    )
        return flags

    def issue_pv(self, index: jax.Array, operands: list) -> None:
        """P @ V of period index - PV_LAG's two tiles, value half m on MXU m;
        their probabilities were staged in the period before, so the period
        opens on MXU work that waits for none of its own."""
        slot = lax.rem(
            index - PV_LAG + NUM_PROBABILITY_SLOTS, NUM_PROBABILITY_SLOTS
        )
        for tile in range(PV_LATCHES):
            for g in range(self.num_fragments):
                lhs = self.refs.probabilities[
                    slot, tile, pl.ds(g * FRAGMENT_ROWS, FRAGMENT_ROWS)
                ]
                for m in range(NUM_MXUS):
                    self.matmul(
                        self.pv_accumulators[m][g],
                        lhs,
                        m,
                        tile if g == 0 else None,
                        operands[m],
                        index,
                    )

    def step(self, index: jax.Array, carry: tuple) -> tuple:
        """Runs period index; the carry holds the periods index - 1 to
        index + 1 and index + ROW_LEAD - 1, and the union before index's."""
        prev, cur, upcoming, ahead, before = carry
        # Note (david): the next period stages period index + 2's rows, waited
        # at this period's end, off the next one's start.
        rows = (self.next_period(upcoming), self.next_period(ahead))
        period = (prev, cur, upcoming, before)
        lax.cond(
            cur.position == 1,
            functools.partial(self.run_period, period, rows, index, True),
            functools.partial(self.run_period, period, rows, index, False),
        )
        before = jnp.where(upcoming.position == 0, cur.union, before)
        return cur, upcoming, rows[0], rows[1], before

    def start_stream(self) -> tuple:
        """The first union's queries and the first periods' rows, the first
        period's keys staged and its first RHS tiles pushed; returns the
        first step's carry."""
        first = Period(self.first_union, jnp.int32(0), jnp.int32(0))
        second_union = self.union_row(first.union).next_union
        for ordinal, union in enumerate((first.union, second_union)):
            for copy in self.union_copies(union, jnp.int32(ordinal)):
                copy.start()
        self.place_queries(first.union, jnp.int32(0))
        upcoming = self.next_period(first)
        periods = [first, upcoming]
        for _ in range(ROW_LEAD - 2):
            periods.append(self.next_period(periods[-1]))
        for index, period in enumerate(periods):
            self.start_rows(period, jnp.int32(index))
        for index, period in enumerate(periods[:2]):
            self.wait_rows(period, jnp.int32(index))
        self.stage(first, jnp.int32(0))
        for m in range(NUM_MXUS):
            self.push_first(jnp.int32(0), m)
        before_first = Period(jnp.int32(-1), jnp.int32(1), jnp.int32(-1))
        return before_first, first, upcoming, periods[-1], jnp.int32(-1)

    def drain(self, carry: tuple) -> None:
        """Empties the MXUs (the last push is latched by one stale matmul)
        and waits the copies still in flight."""
        last = carry[0]
        refs = self.refs
        lhs = refs.query_buffer[
            0, 0, pl.ds(0, FRAGMENT_ROWS), pl.ds(0, MXU_TILE)
        ]
        for m in range(NUM_MXUS):
            for f in range(self.num_fragments):
                pltpu.matmul_pop(self.qk_accumulators[m][f])
            pltpu.matmul_acc_lhs(
                self.qk_accumulators[m][0], lhs, load_staged_rhs=0
            )
            pltpu.matmul_pop(self.qk_accumulators[m][0])
        for copy in self.union_copies(
            last.union, lax.rem(last.ordinal + 2, NUM_UNION_SLOTS)
        ):
            copy.wait()
        for slot in range(NUM_OUTPUT_SLOTS):
            self.wait_output(slot)

    def zero_padding(self) -> None:
        """Zero the output rows past the call's real tokens, which no union
        holds."""
        refs = self.refs
        num_tokens = refs.out.shape[0]
        union_tokens = self.params.union_tokens
        first_padding = refs.schedule[HEADER_ROW, HEADER_NUM_REAL]
        num_chunks = (
            num_tokens - first_padding + union_tokens - 1
        ) // union_tokens

        @pl.when(num_chunks > 0)
        def _zero():
            refs.out_stage[0] = jnp.zeros(
                refs.out_stage.shape[1:], refs.out_stage.dtype
            )

            def _chunk(chunk, carry):
                start = first_padding + chunk * union_tokens
                copy = self.output_copy(
                    start, jnp.minimum(union_tokens, num_tokens - start), 0
                )
                copy.start()
                copy.wait()
                return carry

            lax.fori_loop(0, num_chunks, _chunk, 0)

    def initialize(self) -> None:
        """Static lanes of the buffers: token t's query rows hold FLAG_QUERY
        in mask lane t, zeros in the other mask lanes and past them; the
        keys hold zeros past KEY_ROW_DIM. Stale scratch could hold NaN, which
        a zero lane turns into NaN."""
        refs = self.refs
        refs.query_buffer[...] = jnp.zeros(
            refs.query_buffer.shape, refs.query_buffer.dtype
        )
        shape = refs.query_buffer.shape[1:-1] + (MAX_BLOCK_Q,)
        flags = jnp.where(
            lax.broadcasted_iota(jnp.int32, shape, 0)
            == lax.broadcasted_iota(jnp.int32, shape, 2),
            FLAG_QUERY,
            0.0,
        ).astype(refs.query_buffer.dtype)
        for slot in range(NUM_UNION_SLOTS):
            refs.query_buffer[
                slot, :, :, pl.ds(LATENT_DIM + ROPE_LANES, MAX_BLOCK_Q)
            ] = flags
        tail = KEY_LANES - KEY_ROW_DIM
        refs.keys[:, :, pl.ds(KEY_ROW_DIM, tail)] = jnp.zeros(
            refs.keys.shape[:2] + (tail,), refs.keys.dtype
        )
        for slot in range(NUM_OUTPUT_SLOTS):
            refs.pending_out[slot, PENDING_NUM_TOKENS] = 0
        refs.replay[0] = 0

    def replay_scores(self, fragment: jax.Array) -> list:
        """The scaled scores of one fragment of a replayed period against its
        two key tiles, a tile on each MXU, one RHS at a time."""
        per_token = self.params.num_heads // FRAGMENT_ROWS
        heads = pl.ds(
            pl.multiple_of(
                lax.rem(fragment, per_token) * FRAGMENT_ROWS, FRAGMENT_ROWS
            ),
            FRAGMENT_ROWS,
        )
        scores = []
        for m in range(NUM_MXUS):
            for qk_pass in range(QK_PASSES):
                pltpu.matmul_push_rhs(
                    self.key_tile(0, m, qk_pass, False), 0, m, transpose=True
                )
                lhs = self.refs.query_buffer[
                    0,
                    fragment // per_token,
                    heads,
                    pl.ds(qk_pass * MXU_TILE, MXU_TILE),
                ]
                pltpu.matmul_acc_lhs(
                    self.qk_accumulators[m][0], lhs, load_staged_rhs=0
                )
            scores.append(
                pltpu.matmul_pop(self.qk_accumulators[m][0]) * self.score_scale
            )
        return scores

    def replay_fragment(self, fragment: jax.Array) -> None:
        """One fragment of a replayed period: the online max and rescale of
        its rows, then its P @ V into the replay's output rows."""
        refs = self.refs
        rows = pl.ds(
            pl.multiple_of(fragment * FRAGMENT_ROWS, FRAGMENT_ROWS),
            FRAGMENT_ROWS,
        )
        scores = self.replay_scores(fragment)
        halves = [
            jnp.maximum(s[:, :NUM_LANES], s[:, NUM_LANES:]) for s in scores
        ]
        tile_max = lane_broadcast(
            jnp.maximum(*halves).max(axis=1, keepdims=True), NUM_LANES
        )
        old_max = refs.anchor[0, rows]
        new_max = jnp.maximum(old_max, tile_max)
        rescale = jnp.exp2(old_max - new_max)
        refs.anchor[0, rows] = new_max
        partial = rescale * refs.row_sum[0, rows]
        for m, s in enumerate(scores):
            probabilities = jnp.exp2(s - lane_broadcast(new_max, MXU_TILE))
            partial += (
                probabilities[:, :NUM_LANES] + probabilities[:, NUM_LANES:]
            )
            for value_half in range(NUM_MXUS):
                pltpu.matmul_push_rhs(
                    self.key_tile(0, m, value_half, False), 0, value_half
                )
                pltpu.matmul_acc_lhs(
                    self.pv_accumulators[value_half][0],
                    probabilities.astype(FP8_DTYPE),
                    load_staged_rhs=0,
                )
        refs.row_sum[0, rows] = partial
        rescale = lane_broadcast(rescale, MXU_TILE)
        for m in range(NUM_MXUS):
            values = pl.ds(m * MXU_TILE, MXU_TILE)
            refs.replay_out[rows, values] = rescale * refs.replay_out[
                rows, values
            ] + pltpu.matmul_pop(self.pv_accumulators[m][0])

    def replay_union(self, union: jax.Array) -> None:
        """Recomputes a flagged union with the online max and rescale, a
        period and a fragment at a time, and rewrites its output rows."""
        refs = self.refs
        row = self.union_row(union)
        for copy in self.union_copies(union, jnp.int32(0)):
            copy.start()
        self.place_queries(union, jnp.int32(0))
        refs.anchor[0] = jnp.full(
            refs.anchor.shape[1:], NO_KEY_MAX, jnp.float32
        )
        refs.row_sum[0] = jnp.zeros(refs.row_sum.shape[1:], jnp.float32)
        refs.replay_out[...] = jnp.zeros(refs.replay_out.shape, jnp.float32)

        @pl.loop(0, row.num_periods)
        def _period(position):
            period = Period(union, position, jnp.int32(0))
            for copy in self.row_copies(period, jnp.int32(0)):
                copy.start()
                copy.wait()
            self.stage(period, jnp.int32(0))
            pl.loop(0, self.num_fragments)(self.replay_fragment)

        for f in range(self.num_fragments):
            rows = pl.ds(f * FRAGMENT_ROWS, FRAGMENT_ROWS)
            row_sum = refs.row_sum[0, rows].sum(axis=1, keepdims=True)
            inverse = jnp.where(row_sum > 0.0, self.kv_scale / row_sum, 0.0)
            token, heads = self.fragment_rows(f)
            refs.out_stage[0, token, heads] = (
                refs.replay_out[rows] * lane_broadcast(inverse, LATENT_DIM)
            ).astype(refs.out_stage.dtype)
        copy = self.output_copy(row.first_token, row.num_tokens, 0)
        copy.start()
        copy.wait()

    def run(self) -> None:
        self.initialize()

        @pl.when(self.num_periods > 0)
        def _stream():
            num_steps = self.num_periods + MIN_PERIODS
            self.drain(
                lax.fori_loop(0, num_steps, self.step, self.start_stream())
            )

            @pl.loop(0, self.refs.replay[0])
            def _replay(index):
                self.replay_union(self.refs.replay[1 + index])

        if not self.params.has_previous_out:
            self.zero_padding()


def split_refs(refs: tuple, params: UnionAttentionParams) -> UnionAttentionRefs:
    """The pallas_call's refs, in dsa_union_attention_pallas's order, as
    UnionAttentionRefs; previous_out None when absent."""
    refs = list(refs)
    if not params.has_previous_out:
        index = UnionAttentionRefs._fields.index("previous_out")
        refs.insert(index, None)
    return UnionAttentionRefs(*refs)


def union_attention_kernel(
    *refs: jax.Array, params: UnionAttentionParams
) -> None:
    refs = split_refs(refs, params)
    num_fragments = params.union_tokens * params.num_heads // FRAGMENT_ROWS
    accumulators = [
        pltpu.ACC(m)((FRAGMENT_ROWS, MXU_TILE), jnp.float32)
        for _ in range(2)
        for m in range(NUM_MXUS)
        for _ in range(num_fragments)
    ]

    def _body(*accumulator_refs):
        UnionAttentionKernel(params, refs, accumulator_refs).run()

    pl.run_scoped(_body, *accumulators)


def union_periods(num_tokens: jax.Array, size: jax.Array) -> jax.Array:
    """A union's periods: its size in periods of PERIOD_KEYS keys, at least
    MIN_PERIODS, with tokens; none without."""
    return jnp.where(
        num_tokens > 0, jnp.maximum(pl.cdiv(size, PERIOD_KEYS), MIN_PERIODS), 0
    )


def scratch_shapes(params: UnionAttentionParams, q_dtype: jnp.dtype) -> list:
    """The kernel's scratch, in UnionAttentionRefs order."""
    union_tokens, num_heads = params.union_tokens, params.num_heads
    num_rows = union_tokens * num_heads
    row_state = pltpu.VMEM((NUM_STATE_SLOTS, num_rows, NUM_LANES), jnp.float32)
    return [
        pltpu.VMEM((NUM_ROW_SLOTS, PERIOD_KEYS, NOPE_WORDS), jnp.int32),
        pltpu.VMEM(
            (NUM_ROW_SLOTS, PERIOD_KEYS // ROPE_TOKENS_PER_ROW, NOPE_WORDS),
            jnp.int32,
        ),
        pltpu.VMEM((NUM_KEY_SLOTS, PERIOD_KEYS, KEY_LANES), FP8_DTYPE),
        pltpu.VMEM((PERIOD_KEYS // QUAD_TOKENS, NUM_LANES), jnp.int32),
        pltpu.VMEM(
            (NUM_PROBABILITY_SLOTS, PERIOD_TILES, num_rows, MXU_TILE), FP8_DTYPE
        ),
        row_state,
        row_state,
        pltpu.VMEM(
            (NUM_UNION_SLOTS, union_tokens, num_heads, KEY_LANES), q_dtype
        ),
        pltpu.VMEM(
            (NUM_UNION_SLOTS, union_tokens, ROPE_DIM, num_heads), q_dtype
        ),
        pltpu.VMEM(
            (NUM_UNION_SLOTS, union_tokens * params.bitmap_rows, NUM_LANES),
            jnp.int32,
        ),
        pltpu.VMEM(
            (NUM_OUTPUT_SLOTS, union_tokens, num_heads, LATENT_DIM), q_dtype
        ),
        pltpu.VMEM((num_rows, LATENT_DIM), jnp.float32),
        pltpu.SMEM((NUM_OUTPUT_SLOTS, PENDING_WORDS), jnp.int32),
        pltpu.SMEM((params.num_unions + 1,), jnp.int32),
        pltpu.SemaphoreType.DMA((NUM_SEMS, NUM_SEM_SLOTS)),
    ]


# Note (david): the route calls this once a chunk; a jit traces and lowers
# the kernel once for all of them.
@functools.partial(
    jax.jit,
    static_argnames=(
        "union_tokens",
        "num_unions",
        "sm_scale",
        "num_slabs",
        "force_replay",
    ),
)
def dsa_union_attention_pallas(
    q_nope: jax.Array,
    q_pe_t: jax.Array,
    nope_rows: jax.Array,
    rope_rows: jax.Array,
    bitmap: jax.Array,
    schedule: jax.Array,
    kv_scale: jax.Array,
    *,
    union_tokens: int,
    num_unions: int,
    sm_scale: float,
    num_slabs: int,
    previous_out: jax.Array | None,
    force_replay: bool = False,
) -> jax.Array:
    """The union attention pallas_call on checked operands.

    Args:
      q_nope: (T, H, LATENT_DIM) bf16.
      q_pe_t: (T, ROPE_DIM, H) bf16, the rope queries heads minor.
      nope_rows: (R, NOPE_WORDS) int32 gathered nope rows, as the cache holds
        them; a union's rows contiguous from its first gathered row, which is
        a multiple of UNION_BLOCK_KV, with its periods' rows within R.
      rope_rows: (R / ROPE_TOKENS_PER_ROW, NOPE_WORDS) int32 gathered rope
        rows, as the cache holds them.
      bitmap: (T * bitmap_rows(num_slabs), NUM_LANES) int32 membership
        bitmaps of each token over its union's positions.
      schedule: (1 + U, NUM_LANES) int32 rows the union kernel writes: the
        header, then a row for each of the num_unions unions read.
      kv_scale: (1,) f32.
      union_tokens: the most tokens a union holds, at most MAX_BLOCK_Q.
      sm_scale: positive scale of q_nope . latent + q_pe . rope.
      num_slabs: the bitmaps' slabs a token.
      previous_out: an earlier call's out, written into in place, or None.
      force_replay: replay every union, as an overflowing one is.

    Returns:
      out (T, H, LATENT_DIM) bf16: the unions' rows, and without
      previous_out zeros past the call's real tokens; other rows are
      previous_out's, or unspecified without it.
    """
    num_heads = q_nope.shape[1]
    params = UnionAttentionParams(
        union_tokens=union_tokens,
        num_heads=num_heads,
        num_unions=num_unions,
        num_slabs=num_slabs,
        bitmap_rows=bitmap_rows(num_slabs),
        sm_scale_log2=sm_scale * LOG2E,
        has_previous_out=previous_out is not None,
        force_replay=force_replay,
    )
    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    inputs = [kv_scale, q_nope, q_pe_t, nope_rows, rope_rows, bitmap]
    in_specs = [pl.BlockSpec(memory_space=pltpu.SMEM)] + [hbm_spec] * 5
    aliases = {}
    if previous_out is not None:
        aliases = {1 + len(inputs): 0}
        inputs.append(previous_out)
        in_specs.append(hbm_spec)
    return pl.pallas_call(
        functools.partial(union_attention_kernel, params=params),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(1,),
            in_specs=in_specs,
            out_specs=hbm_spec,
            scratch_shapes=scratch_shapes(params, q_nope.dtype),
        ),
        out_shape=jax.ShapeDtypeStruct(q_nope.shape, q_nope.dtype),
        input_output_aliases=aliases,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=vmem_limit_bytes(),
            disable_bounds_checks=True,
        ),
        name="dsa_union_attention",
    )(schedule, *inputs)
