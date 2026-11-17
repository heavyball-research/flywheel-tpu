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
"""

import dataclasses

import numpy as np
import pytest

from flywheel_tpu.softmax_attention import mxu_pipeline
from flywheel_tpu.softmax_attention.mxu_pipeline import (
    FRAGMENT_ROWS,
    MxuPipeline,
    PipelineDelays,
    WorkItem,
)

KEYS = 256
HEAD_DIM = 256
NUM_SLOTS = 64


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


def make_pipeline(delays=PipelineDelays()):
    return MxuPipeline(
        delays=delays,
        qk_slots=tuple(Slot(0, index) for index in range(NUM_SLOTS)),
        pv_slots=tuple(Slot(1, index) for index in range(NUM_SLOTS)),
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

    def __init__(self, attention, tile_offset):
        self.attention = attention
        self.tile_offset = tile_offset

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


@pytest.mark.parametrize("num_fragments", [64, 128])
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


def test_loop_body_assumes_its_own_tail(mxus):
    """A rolled tile loop traces its body once: the body finishes the tile
    before it through ops bound to that tile, and leaves the same tail."""
    num_fragments, num_tiles = 64, 6
    attention = Attention(num_fragments, num_tiles)
    pipeline = make_pipeline()
    items = tile_items(num_fragments)
    state = pipeline.advance(
        pipeline.idle_state(),
        items,
        attention.ops(0),
        next_k_tile=lambda: attention.ops(1).k_tile(0),
    )
    for tile in range(1, num_tiles):
        is_last = tile == num_tiles - 1
        assumed = pipeline.tail_of(
            items, attention.ops(tile - 1), has_staged_next_k=True
        )
        pipeline.check_loop_tail(state.tail, assumed)
        state = pipeline.advance(
            mxu_pipeline.PipelineState(probs=state.probs, tail=assumed),
            items,
            attention.ops(tile),
            next_k_tile=None
            if is_last
            else (lambda t=tile + 1: attention.ops(t).k_tile(0)),
        )
    pipeline.drain(state)
    check(attention, mxus, lambda fragment: range(num_tiles))


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
    pipeline.check_loop_tail(state.tail, pipeline.tail_of(plain_items, ops))
    for tile in range(2, num_tiles):
        state = pipeline.advance(state, tile_items(num_fragments, tile), ops)
    pipeline.drain(state)
    check(attention, mxus, lambda fragment: range(num_tiles))


def test_short_segment_is_rejected(mxus):
    del mxus
    pipeline = make_pipeline()
    attention = Attention(32, 1)
    with pytest.raises(AssertionError, match="at least"):
        pipeline.advance(pipeline.idle_state(), tile_items(32), attention.ops())


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
        mxu_pipeline.PipelineState(probs=state.probs, tail=assumed), items, ops
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
