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

"""The explicit MXU pipeline's tick schedule against a model of the MXUs.

The model keeps each MXU's staging register, loaded tile and result slots as
numpy values and rejects what the hardware forbids: a push into a full
register, a load from an empty one, an issue into a slot whose result is
still due and a pop of a slot nothing was issued into. So a schedule that
passes here computes softmax attention and breaks no MXU rule, without a TPU.

A dense static row plan (fwd_dense_mxu.static_row_plan) runs item by item
against the masked softmax, with the explicit loop's masks and deferred
anchors.
"""

import dataclasses

import numpy as np
import pytest

from flywheel_tpu.softmax_attention import fwd_mxu, mxu_pipeline
from flywheel_tpu.softmax_attention.config import (
    DEFAULT_MASK_VALUE,
    BlockSizes,
)
from flywheel_tpu.softmax_attention.kernels import fwd_dense_mxu
from flywheel_tpu.softmax_attention.mxu_pipeline import (
    FRAGMENT_ROWS,
    NUM_SLOTS,
    MxuPipeline,
    PipelineDelays,
    PipelineShape,
    PipelineState,
    WorkItem,
)
from flywheel_tpu.softmax_attention.schedule import (
    MxuRowTiles,
    mxu_row_groups,
)
from flywheel_tpu.softmax_attention.softmax import UNANCHORED_ROW_MAX
from flywheel_tpu.softmax_attention.traits import FwdTraits
from flywheel_tpu.softmax_attention.variant import Variant

KEYS = 256
HEAD_DIM = 256
# A static plan's q block: its fragments fill a segment.
BLOCK_Q = 1024
MASK_VALUE = DEFAULT_MASK_VALUE


@dataclasses.dataclass(frozen=True)
class Slot:
    """A result slot of the model, shaped as an accumulator ref is."""

    mxu: int
    index: int

    @property
    def shape(self):
        return (FRAGMENT_ROWS, KEYS)


class MxuModel:
    """Two MXUs: per MXU a staging register, a loaded tile and result slots."""

    def __init__(self):
        self.staged = [None, None]
        self.loaded = [None, None]
        self.results = {}
        self.num_pushes = [0, 0]

    def push_rhs(self, rhs, *, staging_register, mxu_index, transpose=False):
        assert staging_register == mxu_pipeline.STAGING_REGISTER
        if self.staged[mxu_index] is not None:
            raise AssertionError(f"MXU {mxu_index}: push into a full register")
        rhs = np.asarray(rhs, np.float64)
        self.staged[mxu_index] = rhs.T if transpose else rhs
        self.num_pushes[mxu_index] += 1

    def acc_lhs(self, slot, lhs, load_staged_rhs=None):
        mxu = slot.mxu
        if load_staged_rhs is not None:
            if self.staged[mxu] is None:
                raise AssertionError(f"MXU {mxu}: load from an empty register")
            self.loaded[mxu], self.staged[mxu] = self.staged[mxu], None
        if slot in self.results:
            raise AssertionError(f"{slot}: issued before its pop")
        self.results[slot] = np.asarray(lhs, np.float64) @ self.loaded[mxu]

    def pop(self, slot):
        if slot not in self.results:
            raise AssertionError(f"{slot}: popped with nothing issued")
        return self.results.pop(slot)

    def check_idle(self):
        assert not self.results, sorted(self.results, key=repr)
        assert self.staged == [None, None]


@pytest.fixture(name="mxus")
def mxus_fixture(monkeypatch):
    model = MxuModel()
    monkeypatch.setattr(
        mxu_pipeline.pltpu, "matmul_push_rhs", model.push_rhs, raising=False
    )
    monkeypatch.setattr(
        mxu_pipeline.pltpu, "matmul_acc_lhs", model.acc_lhs, raising=False
    )
    monkeypatch.setattr(
        mxu_pipeline.pltpu, "matmul_pop", model.pop, raising=False
    )
    return model


def make_pipeline(shape=PipelineShape()):
    return MxuPipeline(
        delays=shape.delays,
        qk_slots=tuple(Slot(0, index) for index in range(shape.num_slots)),
        pv_slots=tuple(Slot(1, index) for index in range(shape.num_slots)),
    )


class Attention:
    """Static-anchor softmax attention of one q block over kv tiles; a
    fragment's row state starts at the first tile it meets."""

    def __init__(self, num_fragments, num_tiles, seed=0):
        generator = np.random.default_rng(seed)
        self.q = generator.standard_normal(
            (num_fragments * FRAGMENT_ROWS, HEAD_DIM)
        ) / np.sqrt(HEAD_DIM)
        self.k = generator.standard_normal((num_tiles * KEYS, HEAD_DIM))
        self.v = generator.standard_normal((num_tiles * KEYS, HEAD_DIM))
        self.anchor = {}
        self.row_sum = np.zeros((self.q.shape[0], 1))
        self.out = np.zeros((self.q.shape[0], HEAD_DIM))

    def ops(self, tile_offset=0):
        return AttentionOps(self, tile_offset)

    def expected(self, tiles_of_fragment):
        """(row_sum, out) when fragment f meets tiles_of_fragment(f), in
        order."""
        row_sum = np.zeros_like(self.row_sum)
        out = np.zeros_like(self.out)
        for fragment in range(self.q.shape[0] // FRAGMENT_ROWS):
            rows = slice(
                fragment * FRAGMENT_ROWS, (fragment + 1) * FRAGMENT_ROWS
            )
            anchor = None
            for tile in tiles_of_fragment(fragment):
                keys = slice(tile * KEYS, (tile + 1) * KEYS)
                scores = self.q[rows] @ self.k[keys].T
                if anchor is None:
                    anchor = scores.max(axis=1, keepdims=True)
                probs = np.exp2(scores - anchor)
                row_sum[rows] += probs.sum(axis=1, keepdims=True)
                out[rows] += probs @ self.v[keys]
        return row_sum, out


class AttentionOps:
    """An Attention's FragmentOps, its tiles counted from tile_offset."""

    v_handle_words = 1

    def __init__(self, attention, tile_offset):
        self.attention = attention
        self.tile_offset = tile_offset

    def v_handle(self, tile):
        return (tile + self.tile_offset,)

    def v_tile_at(self, handle):
        (tile,) = handle
        return self.attention.v[tile * KEYS : (tile + 1) * KEYS]

    def _keys(self, tile):
        tile = tile + self.tile_offset
        return slice(tile * KEYS, (tile + 1) * KEYS)

    def _rows(self, item):
        return slice(
            item.fragment * FRAGMENT_ROWS, (item.fragment + 1) * FRAGMENT_ROWS
        )

    def k_tile(self, tile):
        return self.attention.k[self._keys(tile)]

    def v_tile(self, tile):
        return self.attention.v[self._keys(tile)]

    def q_fragment(self, item):
        return self.attention.q[self._rows(item)]

    def softmax(self, item, scores):
        attention = self.attention
        if item.fragment not in attention.anchor:
            attention.anchor[item.fragment] = scores.max(axis=1, keepdims=True)
        probs = np.exp2(scores - attention.anchor[item.fragment])
        attention.row_sum[self._rows(item)] += probs.sum(axis=1, keepdims=True)
        return probs

    def accumulate(self, item, weighted_values):
        self.attention.out[self._rows(item)] += weighted_values


def tile_items(num_fragments, tile=0, first_fragment=0):
    return tuple(
        WorkItem(fragment=fragment, tile=tile)
        for fragment in range(first_fragment, num_fragments)
    )


def check(attention, mxus, tiles_of_fragment):
    mxus.check_idle()
    row_sum, out = attention.expected(tiles_of_fragment)
    np.testing.assert_allclose(attention.row_sum, row_sum, rtol=1e-9)
    np.testing.assert_allclose(attention.out, out, rtol=1e-9, atol=1e-9)


class MaskedRow:
    """A dense row of one q block against its visits, run item by item with
    the explicit loop's softmax: an item's scores take the mask of its tile,
    a row start writes the row state, and under a left bound a masked item
    sets the anchor of the rows whose anchor tile it is
    (fwd_mxu.fragment_anchor). out() is the normalized output."""

    def __init__(self, geometry, q_first, num_keys, seed=0):
        generator = np.random.default_rng(seed)
        self.geometry, self.q_first = geometry, q_first
        self.q = generator.standard_normal((BLOCK_Q, HEAD_DIM)) / 16
        self.k = generator.standard_normal((num_keys, HEAD_DIM))
        self.v = generator.standard_normal((num_keys, HEAD_DIM))
        self.anchor = np.full((BLOCK_Q, 1), np.nan)
        self.row_sum = np.full((BLOCK_Q, 1), np.nan)
        self.acc = np.full((BLOCK_Q, HEAD_DIM), np.nan)

    def run(self, fragment, first_key, *, is_row_start, is_masked):
        rows = slice(fragment * FRAGMENT_ROWS, (fragment + 1) * FRAGMENT_ROWS)
        keys = np.arange(first_key, first_key + KEYS)
        in_axis = (keys >= 0) & (keys < self.k.shape[0])
        clipped = np.clip(keys, 0, self.k.shape[0] - 1)
        scores = self.q[rows] @ self.k[clipped].T
        bounds = fwd_mxu.tile_bounds(self.geometry, first_key).from_row(
            self.q_first
        )
        first_row = fragment * FRAGMENT_ROWS
        if is_masked:
            visible = np.asarray(bounds.visible_keys(first_row))
            scores = np.where(visible, scores, MASK_VALUE)
        else:
            assert np.asarray(bounds.visible_keys(first_row)).all()
            assert in_axis.all()
        maximum = scores.max(axis=1, keepdims=True)
        held = UNANCHORED_ROW_MAX if is_row_start else self.anchor[rows]
        if self.geometry.has_deferred_anchors and is_masked:
            takes = np.asarray(bounds.takes_anchor(first_row))[:, :1]
            anchor = np.where(takes, maximum, held)
        else:
            anchor = maximum if is_row_start else held
        self.anchor[rows] = anchor
        probs = np.exp2(scores - anchor)
        values = probs @ self.v[clipped]
        if is_row_start:
            self.row_sum[rows] = probs.sum(axis=1, keepdims=True)
            self.acc[rows] = values
        else:
            self.row_sum[rows] += probs.sum(axis=1, keepdims=True)
            self.acc[rows] += values

    def out(self):
        return self.acc / self.row_sum

    def expected(self):
        rows = self.q_first + np.arange(BLOCK_Q)[:, None]
        keys = np.arange(self.k.shape[0])[None, :]
        geometry = self.geometry
        visible = np.ones((BLOCK_Q, self.k.shape[0]), bool)
        if geometry.left is not None:
            visible &= keys >= rows + geometry.shift - geometry.left
        if geometry.right is not None:
            visible &= keys <= rows + geometry.shift + geometry.right
        scores = np.where(visible, self.q @ self.k.T, -np.inf)
        probs = np.exp2(scores - np.max(scores, axis=1, keepdims=True))
        return (probs @ self.v) / probs.sum(axis=1, keepdims=True)


def plan_traits(block_kv):
    return FwdTraits(
        block_sizes=BlockSizes(
            block_q=BLOCK_Q,
            block_kv=block_kv,
            block_kv_compute=KEYS,
            block_q_compute=KEYS,
        ),
        num_q_heads=1,
        num_kv_heads=1,
        head_dim_qk=HEAD_DIM,
        head_dim_v=HEAD_DIM,
        head_fold=1,
        token_major=None,
        transposed_pv=False,
        is_per_seq=False,
        guard_threshold=60.0,
        uses_mxu_pipeline=True,
    )


@pytest.mark.parametrize(
    "block_kv,window,seqlen_q,seqlen_k,row",
    [
        pytest.param(1024, (-1, 0), 4096, 4096, 2, id="causal"),
        pytest.param(1024, (-1, 0), 4096, 4096, 0, id="causal-first-row"),
        pytest.param(512, (-1, 0), 2048, 2560, 1, id="causal-two-blocks"),
        pytest.param(1024, (-1, 128), 4096, 4096, 3, id="right-bound"),
        pytest.param(1024, (127, 0), 4096, 4096, 2, id="left-127"),
        pytest.param(1024, (127, 0), 4096, 4096, 0, id="left-127-first-row"),
        pytest.param(1024, (1500, 0), 4096, 4096, 1, id="left-clipped"),
        pytest.param(1024, (1500, 0), 4096, 4096, 3, id="left-1500"),
        pytest.param(1024, (300, 200), 4096, 4096, 3, id="both-bounds"),
        pytest.param(1024, (2047, 0), 4096, 4096, 3, id="anchor-on-interior"),
        pytest.param(1024, (2047, 0), 4096, 4096, 1, id="anchor-clipped"),
    ],
)
def test_static_row_plan_runs_the_masked_softmax(
    block_kv, window, seqlen_q, seqlen_k, row
):
    """A static plan's items, the interior visits' between its edges, give
    the masked softmax of a row: every visible key is met once, every row
    takes its anchor where it first sees a key."""
    traits = plan_traits(block_kv)
    offset = seqlen_k - seqlen_q
    variant = Variant(
        softmax_scale=1.0, window_size=window, causal_offset=offset
    )
    num_kv_blocks = seqlen_k // block_kv
    blocks_per_row = BLOCK_Q // block_kv
    plan = fwd_dense_mxu.static_row_plan(
        traits,
        variant,
        offset,
        fwd_dense_mxu.PlanRows(
            0, seqlen_q // BLOCK_Q - 1, blocks_per_row, num_kv_blocks
        ),
        PipelineDelays(),
    )
    assert plan is not None
    left, right = variant.window
    geometry = fwd_mxu.MaskGeometry(
        shift=offset, left=left, right=right, key_end=seqlen_k
    )
    masked_row = MaskedRow(geometry, row * BLOCK_Q, seqlen_k)
    frame = row * blocks_per_row
    tiles_a_block = block_kv // KEYS
    fragments = range(BLOCK_Q // FRAGMENT_ROWS)

    def run_segments(segments, *, starts):
        for segment in segments:
            for item in segment.items:
                first_key = ((frame + segment.block) * tiles_a_block) * KEYS
                masked_row.run(
                    item.fragment,
                    first_key + item.tile * KEYS,
                    is_row_start=starts and item.mode.is_row_start,
                    is_masked=item.mode.is_masked,
                )

    run_segments(plan.left, starts=True)
    if plan.right:
        interior_end = min(frame + plan.right[0].block, num_kv_blocks)
        interior_first = 0
        if plan.interior_first is not None:
            interior_first = max(frame + plan.interior_first, 0)
        interior = range(interior_first, interior_end)
        for block_index, block in enumerate(interior):
            for tile in range(tiles_a_block):
                for fragment in fragments:
                    is_first = not plan.left and block_index == 0 and tile == 0
                    masked_row.run(
                        fragment,
                        (block * tiles_a_block + tile) * KEYS,
                        is_row_start=is_first,
                        # The first interior body after left segments is
                        # masked (DenseMxuLoop.run_row_start).
                        is_masked=bool(plan.left)
                        and block_index == 0
                        and tile < fwd_mxu.LOOP_BODY_TILES,
                    )
        if interior:
            run_segments(plan.right, starts=False)
        else:
            run_segments(
                (plan.right_row_start or plan.right[0], *plan.right[1:]),
                starts=True,
            )
    np.testing.assert_allclose(
        masked_row.out(), masked_row.expected(), rtol=1e-9, atol=1e-9
    )


@pytest.mark.parametrize("num_fragments", [16, 32, 64, 128])
def test_tile_segments_then_drain(mxus, num_fragments):
    num_tiles = 5
    attention = Attention(num_fragments, num_tiles)
    pipeline = make_pipeline()
    ops = attention.ops()
    state = pipeline.idle_state()
    for tile in range(num_tiles):
        is_last = tile == num_tiles - 1
        state = pipeline.advance(
            state,
            tile_items(num_fragments, tile),
            ops,
            next_k_tile=None if is_last else (lambda t=tile + 1: ops.k_tile(t)),
        )
    pipeline.drain(state)
    check(attention, mxus, lambda fragment: range(num_tiles))
    # One push of every tile into each MXU.
    assert mxus.num_pushes == [num_tiles, num_tiles]


@pytest.mark.parametrize("num_fragments", [32, 64])
def test_loop_body_assumes_its_own_tail(mxus, num_fragments):
    """A rolled tile loop traces its body once: the body finishes the
    tiles before it, as far back as pv_pop items reach, through ops bound to
    those tiles, and leaves the same tail. A lap-long body leaves items of
    two bodies in flight."""
    num_tiles = 7
    attention = Attention(num_fragments, num_tiles)
    pipeline = make_pipeline()
    items = tile_items(num_fragments)
    num_back = -(-pipeline.delays.pv_pop // num_fragments)
    state = pipeline.idle_state()
    for tile in range(num_back):
        state = pipeline.advance(
            state,
            items,
            attention.ops(tile),
            next_k_tile=lambda t=tile + 1: attention.ops(t).k_tile(0),
        )
    for tile in range(num_back, num_tiles):
        is_last = tile == num_tiles - 1
        assumed = pipeline.tail_of(
            items,
            attention.ops(tile - 1),
            slot_base=(tile - 1) * num_fragments % NUM_SLOTS,
            has_staged_next_k=True,
        )
        pipeline.check_loop_tail(state.tail, assumed)
        state = pipeline.advance(
            PipelineState(probs=state.probs, tail=assumed),
            items,
            attention.ops(tile),
            next_k_tile=None
            if is_last
            else (lambda t=tile + 1: attention.ops(t).k_tile(0)),
        )
    pipeline.drain(state)
    check(attention, mxus, lambda fragment: range(num_tiles))


@pytest.mark.parametrize("num_fragments", [1, 2, 8])
def test_lap_of_short_tiles(mxus, num_fragments):
    """Tiles of fewer items than a K push lasts, a lap of them per segment:
    the pushes wait on the loads before them, and every pop still meets its
    issue."""
    tiles_per_lap = NUM_SLOTS // num_fragments
    num_laps = 4
    num_tiles = tiles_per_lap * num_laps
    attention = Attention(num_fragments, num_tiles)
    pipeline = make_pipeline()
    ops = attention.ops()
    state = pipeline.idle_state()
    for lap in range(num_laps):
        items = sum(
            (
                tile_items(num_fragments, lap * tiles_per_lap + tile)
                for tile in range(tiles_per_lap)
            ),
            (),
        )
        is_last = lap == num_laps - 1
        state = pipeline.advance(
            state,
            items,
            ops,
            next_k_tile=None
            if is_last
            else (lambda t=(lap + 1) * tiles_per_lap: ops.k_tile(t)),
        )
    pipeline.drain(state)
    check(attention, mxus, lambda fragment: range(num_tiles))
    assert mxus.num_pushes == [num_tiles, num_tiles]


def tiles_items(num_fragments, first_tile, num_tiles):
    return sum(
        (
            tile_items(num_fragments, tile)
            for tile in range(first_tile, first_tile + num_tiles)
        ),
        (),
    )


@pytest.mark.parametrize(
    "num_fragments,block_tiles",
    [(4, 2), (4, 4), (4, 6), (8, 2), (16, 2), (24, 2)],
)
def test_block_pipeline_runs_rows_inside_one_block(
    mxus, num_fragments, block_tiles
):
    """A kv block too short for the default depth takes a shallower
    pipeline on a smaller ring (fwd_mxu.pipeline_shape): its start segment
    fits in the block, its loop bodies tile the block and follow their own
    tail, a replay finds its slots free, and every pop meets its issue."""
    shape = fwd_mxu.pipeline_shape(num_fragments, block_tiles)
    assert shape != PipelineShape()
    tiles = fwd_mxu.row_tiles(num_fragments, shape)
    assert tiles.start_tiles <= block_tiles
    assert block_tiles % tiles.body_tiles == 0
    num_bodies = 3
    num_tiles = tiles.start_tiles + num_bodies * tiles.body_tiles
    attention = Attention(num_fragments, num_tiles)
    pipeline = make_pipeline(shape)
    body_items = tiles_items(num_fragments, 0, tiles.body_tiles)
    state = pipeline.advance(
        pipeline.idle_state(),
        tiles_items(num_fragments, 0, tiles.start_tiles),
        attention.ops(),
        next_k_tile=lambda: attention.ops().k_tile(tiles.start_tiles),
    )
    for first in range(tiles.start_tiles, num_tiles, tiles.body_tiles):
        in_flight = set(mxus.results)
        for slot in pipeline.free_slots(
            state.tail, fwd_mxu.REPLAY_RESULT_SLOTS
        ):
            assert slot not in in_flight
        assumed = pipeline.tail_of(
            body_items,
            attention.ops(first - tiles.body_tiles),
            has_staged_next_k=True,
        )
        pipeline.check_loop_tail(state.tail, assumed)
        next_first = first + tiles.body_tiles
        state = pipeline.advance(
            PipelineState(probs=state.probs, tail=assumed),
            body_items,
            attention.ops(first),
            next_k_tile=None
            if next_first == num_tiles
            else (lambda t=next_first: attention.ops(t).k_tile(0)),
        )
    pipeline.drain(state)
    check(attention, mxus, lambda fragment: range(num_tiles))


@pytest.mark.parametrize(
    "num_fragments,block_tiles", [(4, 16), (8, 8), (16, 4), (32, 2), (64, 2)]
)
def test_block_of_default_depth_keeps_default_pipeline(
    num_fragments, block_tiles
):
    assert fwd_mxu.pipeline_shape(num_fragments, block_tiles) == (
        PipelineShape()
    )


def test_fewer_items_than_a_lap(mxus):
    """A whole problem of two items: one segment and the drain."""
    attention = Attention(2, 1)
    pipeline = make_pipeline()
    state = pipeline.advance(
        pipeline.idle_state(), tile_items(2), attention.ops()
    )
    pipeline.drain(state)
    check(attention, mxus, lambda fragment: range(1))


def test_causal_suffix_segment(mxus):
    """A diagonal block's tiles, last tile first: tile t meets the fragments
    from 16 t on, so the segment ends with a full tile, as every other one
    does."""
    num_fragments, num_tiles = 128, 8
    fragments_per_tile = KEYS // FRAGMENT_ROWS
    attention = Attention(num_fragments, num_tiles)
    pipeline = make_pipeline()
    ops = attention.ops()
    items = sum(
        (
            tile_items(num_fragments, tile, tile * fragments_per_tile)
            for tile in reversed(range(num_tiles))
        ),
        (),
    )
    state = pipeline.advance(pipeline.idle_state(), items, ops)
    full_tile = pipeline.tail_of(tile_items(num_fragments), ops)
    pipeline.check_loop_tail(state.tail, full_tile)
    pipeline.drain(state)
    check(
        attention,
        mxus,
        lambda fragment: reversed(range(fragment // fragments_per_tile + 1)),
    )


def test_row_start_tile_cannot_precede_a_loop_body(mxus):
    """A loop body finishes the items in flight as plain ones, so the
    segment before it must end with a plain tile: a row's start segment is
    its first tile and the one after."""
    num_fragments, num_tiles = 64, 4
    attention = Attention(num_fragments, num_tiles)
    pipeline = make_pipeline()
    ops = attention.ops()
    plain_items = tile_items(num_fragments)
    start_items = tuple(item._replace(mode="row_start") for item in plain_items)
    assumed = pipeline.tail_of(plain_items, ops)
    state = pipeline.advance(pipeline.idle_state(), start_items, ops)
    with pytest.raises(AssertionError, match="tail the body assumes"):
        pipeline.check_loop_tail(state.tail, assumed)
    pipeline.drain(state)

    attention = Attention(num_fragments, num_tiles)
    ops = attention.ops()
    state = pipeline.advance(
        pipeline.idle_state(), start_items + tile_items(num_fragments, 1), ops
    )
    pipeline.check_loop_tail(
        state.tail, pipeline.tail_of(start_items + plain_items, ops)
    )
    for tile in range(2, num_tiles):
        state = pipeline.advance(state, tile_items(num_fragments, tile), ops)
    pipeline.drain(state)
    check(attention, mxus, lambda fragment: range(num_tiles))


def test_free_slots_hold_no_result_between_segments(mxus):
    """The slots a replay between two segments may use: a matmul through
    them leaves every result in flight where it was."""
    num_fragments = 32
    attention = Attention(num_fragments, 3)
    pipeline = make_pipeline()
    ops = attention.ops()
    state = pipeline.idle_state()
    for tile in range(2):
        state = pipeline.advance(
            state,
            tile_items(num_fragments, tile),
            ops,
            next_k_tile=lambda t=tile + 1: ops.k_tile(t),
        )
    in_flight = set(mxus.results)
    for slot in pipeline.free_slots(state.tail, 2):
        assert slot not in in_flight
        mxus.results[slot] = None
        mxus.pop(slot)
    state = pipeline.advance(state, tile_items(num_fragments, 2), ops)
    pipeline.drain(state)
    check(attention, mxus, lambda fragment: range(3))


def test_dropped_pop_is_caught(mxus, monkeypatch):
    """A schedule that skips one PV pop leaves a result in its slot: the
    slot's next issue, a lap later, fails."""
    num_fragments = 32
    attention = Attention(num_fragments, 4)
    pipeline = make_pipeline()
    ops = attention.ops()
    real_pop = mxus.pop
    popped = []

    def dropping_pop(slot):
        if slot.mxu == 1:
            popped.append(slot)
        if len(popped) == 1 and slot.mxu == 1:
            return np.zeros(slot.shape)
        return real_pop(slot)

    monkeypatch.setattr(mxu_pipeline.pltpu, "matmul_pop", dropping_pop)
    state = pipeline.idle_state()
    with pytest.raises(AssertionError, match="issued before its pop"):
        for tile in range(4):
            state = pipeline.advance(
                state, tile_items(num_fragments, tile), ops
            )
        pipeline.drain(state)


def test_loop_body_must_keep_the_slot_ring_phase(mxus):
    """A body whose item count is no multiple of the slot ring ends on
    another slot than it started on, so it cannot follow itself."""
    del mxus
    pipeline = make_pipeline()
    attention = Attention(80, 3)
    ops = attention.ops()
    items = tile_items(80)
    state = pipeline.advance(pipeline.idle_state(), items, ops)
    assumed = pipeline.tail_of(items, ops)
    pipeline.check_loop_tail(state.tail, assumed)
    state = pipeline.advance(
        PipelineState(probs=state.probs, tail=assumed), items, ops
    )
    with pytest.raises(AssertionError, match="tail the body assumes"):
        pipeline.check_loop_tail(state.tail, assumed)


def test_drain_refuses_a_staged_k(mxus):
    del mxus
    pipeline = make_pipeline()
    attention = Attention(64, 2)
    ops = attention.ops()
    state = pipeline.advance(
        pipeline.idle_state(),
        tile_items(64),
        ops,
        next_k_tile=lambda: ops.k_tile(1),
    )
    with pytest.raises(AssertionError, match="stage"):
        pipeline.drain(state)


@pytest.mark.parametrize(
    "row_tiles,num_rows,group_rows,groups,stream_tiles",
    [
        # Runs of one-tile rows fill start segments of four tiles, up to the
        # group capacity; the last row of a group streams the tiles that
        # complete the segment.
        (
            [1, 1, 1, 1, 1, 1, 9, 1, 2, 1],
            10,
            4,
            [(0, 4), (4, 2), (6, 1), (7, 1), (8, 1), (9, 1)],
            [1, 1, 1, 1, 1, 3, 10, 4, 4, 4],
        ),
        (
            [1, 1, 1, 1, 1, 7, 7],
            5,
            2,
            [(0, 2), (2, 2), (4, 1)],
            [1, 3, 1, 3, 4],
        ),
    ],
)
def test_row_groups(row_tiles, num_rows, group_rows, groups, stream_tiles):
    tiles = MxuRowTiles(unit_tiles=2, start_tiles=4, body_tiles=2)
    fields = mxu_row_groups(
        np.asarray(row_tiles, np.int32), num_rows, tiles, group_rows
    )
    num_groups = int(fields["num_groups"])
    assert num_groups == len(groups)
    assert [
        (int(first), int(size))
        for first, size in zip(
            fields["group_first"][:num_groups],
            fields["group_size"][:num_groups],
        )
    ] == groups
    assert list(np.asarray(fields["stream_tiles"])[:num_rows]) == stream_tiles
