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

"""Tick schedule of the forward fragment pipeline on explicit MXU ops.

On a chip whose MXUs have result accumulators (TPU v7x) a forward main loop
issues its matmuls itself: it pushes a K or V tile into an MXU's staging
register, streams q fragments against the loaded tile into result slots, and
pops each result some ticks later. MXU 0 runs QK and MXU 1 runs PV.

A work item is one q fragment (FRAGMENT_ROWS rows of one pipeline head)
against one kv tile (as many keys as the MXU has columns). Tick t issues the
QK of item t and runs one later stage of three older items:

    QK issue of item t
    PV issue of item t - pv_issue
    PV pop and accumulate of item t - pv_pop
    QK pop and softmax of item t - qk_pop

so no stage waits on a result its own tick issued, and both MXUs stream
without a gap. A loop emits its items in segments, one straight-line run of
ticks each; the items a segment leaves in flight are its tail, which the next
segment (or a drain) finishes. Nothing here knows attention: a FragmentOps
object reads the operands and folds the results.
"""

import dataclasses
from collections.abc import Callable, Hashable, Sequence
from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from flywheel_tpu.core.tpu import BF16_TILE_ROWS, NUM_SUBLANES

# The q rows of one work item: one bf16 tile, the least an MXU op streams.
FRAGMENT_ROWS = BF16_TILE_ROWS
QK_MXU = 0
PV_MXU = 1
# A K or V tile is loaded from the register the tick before last staged it in;
# each MXU's pushes and loads alternate, so one register serves a stream.
STAGING_REGISTER = 0


def slot_ring_rows() -> int:
    """The q rows that fill one ring of an MXU's result slots on this chip."""
    return pltpu.get_tpu_info().num_accumulators * NUM_SUBLANES


@dataclasses.dataclass(frozen=True)
class PipelineDelays:
    """Tick distances of one work item's stages from its QK issue, and how
    many ticks before its first use a K or V tile is pushed.

    qk_pop and pv_pop - pv_issue cover an MXU result's latency; pv_issue -
    qk_pop is how long a fragment's probabilities stay in registers. A
    transposed K push runs as long as k_push_lead ticks and a V push as long
    as v_push_lead.
    """

    qk_pop: int = 28
    pv_issue: int = 32
    pv_pop: int = 60
    k_push_lead: int = 16
    v_push_lead: int = 8

    def __post_init__(self) -> None:
        assert 0 < self.qk_pop < self.pv_issue < self.pv_pop, self
        assert self.k_push_lead >= 1, self
        assert 1 <= self.v_push_lead <= self.pv_issue, self

    @property
    def num_carried_probs(self) -> int:
        """The fragments whose probabilities a segment hands to the next."""
        return self.pv_issue - self.qk_pop


class WorkItem(NamedTuple):
    """One q fragment against one kv tile.

    fragment and tile are the caller's static keys of the q fragment and the
    kv tile; consecutive items of a segment with equal tile share one K and
    one V push. mode is whatever static flag the ops read (mask, row start).
    """

    fragment: Hashable
    tile: Hashable
    mode: Hashable = None


class FragmentOps(Protocol):
    """What a segment's items read and fold, bound to their traced context."""

    def k_tile(self, tile: Hashable) -> jax.Array:
        """The (keys, head_dim) K rows of tile; pushed transposed."""

    def v_tile(self, tile: Hashable) -> jax.Array:
        """The (keys, head_dim) V rows of tile."""

    def q_fragment(self, item: WorkItem) -> jax.Array:
        """The (FRAGMENT_ROWS, head_dim) q rows of item."""

    def softmax(self, item: WorkItem, scores: jax.Array) -> jax.Array:
        """Fold item's f32 (FRAGMENT_ROWS, keys) scores into the softmax row
        state; returns its (FRAGMENT_ROWS, keys) bf16 probabilities."""

    def accumulate(self, item: WorkItem, weighted_values: jax.Array) -> None:
        """Add item's f32 (FRAGMENT_ROWS, head_dim) probs @ V to its rows."""


@dataclasses.dataclass(frozen=True, eq=False)
class Tail:
    """The static side of what a segment leaves in flight: its items, the ops
    that finish them, its first item's result slot, and whether it pushed the
    K tile of the next segment's first item."""

    items: tuple[WorkItem, ...]
    ops: FragmentOps
    slot_base: int
    has_staged_next_k: bool

    def end_slot(self, num_slots: int) -> int:
        """The result slot of the item that follows the segment."""
        return (self.slot_base + len(self.items)) % num_slots

    def signature(self, delays: PipelineDelays, num_slots: int) -> tuple:
        """What the next segment's schedule reads of this tail. Two tails
        with equal signatures are interchangeable to it, which is the
        invariant of a loop body that assumes its own tail: the last pv_pop
        items (fragment, mode, whether each starts a tile), the slot ring's
        phase, and how far back the last tile starts, as far as a push can
        see."""
        starts_tile = [
            position == 0 or item.tile != self.items[position - 1].tile
            for position, item in enumerate(self.items)
        ]
        last_start = max(
            position for position, starts in enumerate(starts_tile) if starts
        )
        in_flight = slice(len(self.items) - delays.pv_pop, None)
        return (
            tuple(
                (item.fragment, item.mode, starts)
                for item, starts in zip(
                    self.items[in_flight], starts_tile[in_flight]
                )
            ),
            self.end_slot(num_slots),
            min(len(self.items) - last_start, delays.k_push_lead + 1),
            self.has_staged_next_k,
        )


class PipelineState(NamedTuple):
    """What a segment hands to the next: probs, the bf16 probabilities of the
    fragments between their softmax and their PV issue, oldest first (a loop
    carries them), and the static tail (None when nothing is in flight)."""

    probs: tuple[jax.Array, ...]
    tail: Tail | None


class _TimedItem(NamedTuple):
    """An item on a segment's timeline: its QK issue tick (negative in the
    tail), its result slot, whether it is its tile's first item, the tick of
    its tile's first item, and the tick of the first item of the tile before
    its own (None when unknown)."""

    tick: int
    item: WorkItem
    ops: FragmentOps
    slot: int
    starts_tile: bool
    tile_tick: int
    prev_tile_tick: int | None


def _timed_items(
    items: tuple[WorkItem, ...],
    ops: FragmentOps,
    first_tick: int,
    slot_base: int,
    num_slots: int,
    prev_tile_tick: int | None,
) -> list[_TimedItem]:
    """items on a timeline from first_tick on, the first in result slot
    slot_base."""
    timeline = []
    tile_tick = first_tick
    for position, item in enumerate(items):
        tick = first_tick + position
        starts_tile = position == 0 or item.tile != items[position - 1].tile
        if starts_tile and position:
            prev_tile_tick, tile_tick = tile_tick, tick
        timeline.append(
            _TimedItem(
                tick=tick,
                item=item,
                ops=ops,
                slot=(slot_base + position) % num_slots,
                starts_tile=starts_tile,
                tile_tick=tile_tick,
                prev_tile_tick=prev_tile_tick,
            )
        )
    return timeline


def _load_register(timed: _TimedItem) -> int | None:
    """The staging register timed's issue loads its tile from; None keeps
    the loaded tile."""
    if timed.starts_tile:
        return STAGING_REGISTER
    else:
        return None


@dataclasses.dataclass(frozen=True, eq=False)
class MxuPipeline:
    """The result slots of both MXUs and the tick schedule over them.

    Build one per kernel with allocate(), inside the kernel body: the slots
    are the kernel's whole accumulator space.
    """

    delays: PipelineDelays
    qk_slots: tuple[jax.Array, ...]
    pv_slots: tuple[jax.Array, ...]

    @classmethod
    def allocate(
        cls, delays: PipelineDelays = PipelineDelays()
    ) -> "MxuPipeline":
        """A pipeline over every accumulator of the chip's first two MXUs."""
        info = pltpu.get_tpu_info()
        assert info.num_accumulators, info.chip_version
        num_slots = slot_ring_rows() // FRAGMENT_ROWS
        slot_type = jax.ShapeDtypeStruct(
            (FRAGMENT_ROWS, info.mxu_column_size), jnp.float32
        )
        # A slot must hold a result from its issue to its pop.
        assert max(delays.qk_pop, delays.pv_pop - delays.pv_issue) < num_slots
        return cls(
            delays=delays,
            qk_slots=tuple(
                jax.empty_ref(slot_type, memory_space=pltpu.ACC(QK_MXU))
                for _ in range(num_slots)
            ),
            pv_slots=tuple(
                jax.empty_ref(slot_type, memory_space=pltpu.ACC(PV_MXU))
                for _ in range(num_slots)
            ),
        )

    @property
    def num_slots(self) -> int:
        return len(self.qk_slots)

    def idle_state(self) -> PipelineState:
        """The state with nothing in flight, before a row's first segment
        and after a drain."""
        empty_probs = jnp.zeros(self.qk_slots[0].shape, jnp.bfloat16)
        return PipelineState(
            probs=(empty_probs,) * self.delays.num_carried_probs, tail=None
        )

    def check_loop_tail(self, actual: Tail, assumed: Tail) -> None:
        """Assert that a segment built to follow assumed may follow actual:
        a loop body traces once, against the tail it assumes."""
        actual_signature = actual.signature(self.delays, self.num_slots)
        assumed_signature = assumed.signature(self.delays, self.num_slots)
        assert actual_signature == assumed_signature, (
            "the segment before a loop body, and the body itself, must leave"
            f" the tail the body assumes; got {actual_signature} against"
            f" {assumed_signature}."
        )

    def tail_of(
        self,
        items: tuple[WorkItem, ...],
        ops: FragmentOps,
        *,
        slot_base: int = 0,
        has_staged_next_k: bool = False,
    ) -> Tail:
        """The tail a segment of items leaves, for a loop body to assume as
        its previous iteration's; ops finishes those items."""
        return Tail(
            items=items,
            ops=ops,
            slot_base=slot_base,
            has_staged_next_k=has_staged_next_k,
        )

    def advance(
        self,
        state: PipelineState,
        items: tuple[WorkItem, ...],
        ops: FragmentOps,
        *,
        next_k_tile: Callable[[], jax.Array] | None = None,
    ) -> PipelineState:
        """Emit one tick per item: issue its QK and finish what the ticks
        before left due, the tail's first.

        next_k_tile returns the K tile of the next segment's first item; it
        is pushed k_push_lead ticks before this segment ends, so that item's
        QK loads it without waiting. A segment holds at least pv_pop items,
        so only one segment's items are ever in flight at a boundary.
        """
        assert len(items) >= self.delays.pv_pop, (
            f"a segment holds at least {self.delays.pv_pop} items; got"
            f" {len(items)}."
        )
        tail = state.tail
        slot_base = 0 if tail is None else tail.end_slot(self.num_slots)
        new_tail = Tail(
            items=items,
            ops=ops,
            slot_base=slot_base,
            has_staged_next_k=next_k_tile is not None,
        )
        probs = self._emit(state, new_tail, len(items), next_k_tile)
        return PipelineState(probs=probs, tail=new_tail)

    def free_slots(self, tail: Tail, count: int) -> tuple[jax.Array, ...]:
        """count QK result slots that hold no result between the segment
        that left tail and the next one: the next items' own."""
        first = tail.end_slot(self.num_slots)
        return tuple(
            self.qk_slots[(first + offset) % self.num_slots]
            for offset in range(count)
        )

    def drain(self, state: PipelineState) -> None:
        """Finish everything in flight; emits pv_pop ticks."""
        assert (
            not state.tail.has_staged_next_k
        ), "a drained segment must not stage a next K tile."
        self._emit(state, None, self.delays.pv_pop, None)

    def _emit(
        self,
        state: PipelineState,
        segment: Tail | None,
        num_ticks: int,
        next_k_tile: Callable[[], jax.Array] | None,
    ) -> tuple[jax.Array, ...]:
        """Emit ticks [0, num_ticks): segment's items (None: a drain) after
        the tail of state. Returns the probabilities the ticks leave between
        softmax and PV issue."""
        delays = self.delays
        timeline = self._timeline(state.tail, segment)
        by_tick = {timed.tick: timed for timed in timeline}
        self._check_slots(timeline)

        # The probabilities of the tail's fragments whose PV is still due.
        probs = {}
        carried_ticks = range(-delays.pv_issue, -delays.qk_pop)
        for tick, value in zip(carried_ticks, state.probs, strict=True):
            if tick in by_tick:
                probs[tick] = value

        k_pushes, v_pushes = self._push_ticks(
            timeline, state.tail, num_ticks, next_k_tile
        )
        for tick in range(num_ticks):
            self._emit_pushes(k_pushes.get(tick, ()), v_pushes.get(tick, ()))
            self._emit_tick(tick, by_tick, probs)
        if segment is None:
            return state.probs
        end = len(segment.items)
        return tuple(
            probs[tick]
            for tick in range(end - delays.pv_issue, end - delays.qk_pop)
        )

    def _timeline(
        self, tail: Tail | None, segment: Tail | None
    ) -> list[_TimedItem]:
        """The items of tail (ticks below 0), then those of segment (ticks
        from 0 on)."""
        if tail is None:
            timeline = []
        else:
            timeline = _timed_items(
                tail.items,
                tail.ops,
                -len(tail.items),
                tail.slot_base,
                self.num_slots,
                None,
            )
        if segment is None:
            return timeline
        return timeline + _timed_items(
            segment.items,
            segment.ops,
            0,
            segment.slot_base,
            self.num_slots,
            timeline[-1].tile_tick if timeline else None,
        )

    @staticmethod
    def _emit_pushes(
        k_pushes: Sequence[Callable[[], jax.Array]],
        v_pushes: Sequence[Callable[[], jax.Array]],
    ) -> None:
        """Push one tick's K tiles, then its V tiles."""
        for push in k_pushes:
            pltpu.matmul_push_rhs(
                push(),
                staging_register=STAGING_REGISTER,
                mxu_index=QK_MXU,
                transpose=True,
            )
        for push in v_pushes:
            pltpu.matmul_push_rhs(
                push(),
                staging_register=STAGING_REGISTER,
                mxu_index=PV_MXU,
            )

    def _emit_tick(
        self,
        tick: int,
        by_tick: dict[int, _TimedItem],
        probs: dict[int, jax.Array],
    ) -> None:
        """Emit tick's QK issue, then the PV issue, the PV pop and the QK
        pop of the older items due at it."""
        delays = self.delays
        issued = by_tick.get(tick)
        if issued is not None:
            pltpu.matmul_acc_lhs(
                self.qk_slots[issued.slot],
                issued.ops.q_fragment(issued.item),
                _load_register(issued),
            )
        weighted = by_tick.get(tick - delays.pv_issue)
        if weighted is not None:
            pltpu.matmul_acc_lhs(
                self.pv_slots[weighted.slot],
                probs.pop(weighted.tick),
                _load_register(weighted),
            )
        finished = by_tick.get(tick - delays.pv_pop)
        if finished is not None:
            finished.ops.accumulate(
                finished.item, pltpu.matmul_pop(self.pv_slots[finished.slot])
            )
        scored = by_tick.get(tick - delays.qk_pop)
        if scored is not None:
            probs[scored.tick] = scored.ops.softmax(
                scored.item, pltpu.matmul_pop(self.qk_slots[scored.slot])
            )

    def _push_ticks(
        self,
        timeline: list[_TimedItem],
        tail: Tail | None,
        num_ticks: int,
        next_k_tile: Callable[[], jax.Array] | None,
    ) -> tuple[dict[int, list], dict[int, list]]:
        """The K and V pushes of ticks [0, num_ticks), keyed by tick. A push
        follows the load of the tile before it, which frees the register."""
        delays = self.delays
        k_pushes: dict[int, list] = {}
        v_pushes: dict[int, list] = {}
        last_tile_tick = None
        for timed in timeline:
            if not timed.starts_tile:
                continue
            last_tile_tick = timed.tick
            ops, tile = timed.ops, timed.item.tile
            lead_k_tick = timed.tick - delays.k_push_lead
            lead_v_tick = timed.tick + delays.pv_issue - delays.v_push_lead
            if timed.prev_tile_tick is None:
                k_tick, v_tick = lead_k_tick, lead_v_tick
            else:
                after_load = timed.prev_tile_tick + 1
                k_tick = max(lead_k_tick, after_load)
                v_tick = max(lead_v_tick, after_load + delays.pv_issue)
            is_staged = (
                timed.tick == 0 and tail is not None and tail.has_staged_next_k
            )
            if timed.tick == 0 and not is_staged:
                k_tick = max(k_tick, 0)
            if 0 <= k_tick < num_ticks and timed.tick >= 0 and not is_staged:
                k_pushes.setdefault(k_tick, []).append(
                    lambda ops=ops, tile=tile: ops.k_tile(tile)
                )
            if 0 <= v_tick < num_ticks:
                v_pushes.setdefault(v_tick, []).append(
                    lambda ops=ops, tile=tile: ops.v_tile(tile)
                )
        if next_k_tile is not None:
            k_tick = max(num_ticks - delays.k_push_lead, last_tile_tick + 1)
            assert k_tick < num_ticks, (
                "the segment's last tile leaves no tick to stage the next K"
                " tile."
            )
            k_pushes.setdefault(k_tick, []).append(next_k_tile)
        return k_pushes, v_pushes

    def _check_slots(self, timeline: list[_TimedItem]) -> None:
        """Assert that no result slot is reissued before its result is
        popped."""
        delays = self.delays
        for issue_delay, pop_delay, name in (
            (0, delays.qk_pop, "QK"),
            (delays.pv_issue, delays.pv_pop, "PV"),
        ):
            pop_tick_of_slot: dict[int, int] = {}
            for timed in timeline:
                issue_tick = timed.tick + issue_delay
                held_until = pop_tick_of_slot.get(timed.slot)
                # A tick issues before it pops, so an equal tick collides.
                assert held_until is None or held_until < issue_tick, (
                    f"{name} result slot {timed.slot} is reissued at tick"
                    f" {issue_tick}, before its pop at tick {held_until}; size"
                    " the segments so the slot ring lines up."
                )
                pop_tick_of_slot[timed.slot] = timed.tick + pop_delay
