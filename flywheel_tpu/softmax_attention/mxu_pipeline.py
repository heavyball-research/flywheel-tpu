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
issues its matmuls itself: MXU 0 runs QK and MXU 1 runs PV. A work item is one
q fragment (FRAGMENT_ROWS rows of one pipeline head) against one kv tile.
Tick t issues the QK of item t and runs one later stage of three older items:

    PV issue of item t - pv_issue
    PV pop and accumulate of item t - pv_pop
    QK pop and softmax of item t - qk_pop

Item t takes result slot t mod NUM_SLOTS of each MXU. A loop emits its items
in segments, one straight-line run of ticks each; the items still in flight
at a segment's end are its tail. Nothing here knows attention: a FragmentOps
object reads the operands and folds the results.
"""

import dataclasses
from collections.abc import Callable, Hashable, Sequence
from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from flywheel_tpu.core.tpu import BF16_TILE_ROWS

# The q rows of one work item: one bf16 tile, the least an MXU op streams.
FRAGMENT_ROWS = BF16_TILE_ROWS
# The result slots of each MXU's ring. A slot number is static, so a rolled
# loop body holds whole laps of NUM_SLOTS items.
NUM_SLOTS = 32
QK_MXU = 0
PV_MXU = 1
# Each MXU's pushes and loads alternate, so one staging register serves it.
STAGING_REGISTER = 0


@dataclasses.dataclass(frozen=True)
class PipelineDelays:
    """Tick distances of a work item's stages from its QK issue, and how many
    ticks before its first use a K or V tile is pushed."""

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

    @property
    def push_horizon(self) -> int:
        """The ticks back within which an earlier tile delays a push: its
        first item's load frees the register."""
        return max(self.k_push_lead, self.pv_issue) + 1


class WorkItem(NamedTuple):
    """One q fragment against one kv tile, by the caller's static keys.
    Consecutive items of a segment with equal tile share one K and one V
    push; mode is whatever static flag the ops read."""

    fragment: Hashable
    tile: Hashable
    mode: Hashable = None


class FragmentOps(Protocol):
    """What a segment's items read and fold, bound to their traced context.

    v_handle gives the v_handle_words int32 scalars that locate a tile's V
    rows and v_tile_at reads them back, for a V push due after its segment.
    """

    v_handle_words: int

    def v_handle(self, tile: Hashable) -> tuple[jax.Array, ...]:
        """The scalars that locate the V rows of tile."""

    def v_tile_at(self, handle: tuple[jax.Array, ...]) -> jax.Array:
        """The (keys, head_dim) V rows handle locates."""

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


def tile_starts(items: Sequence[WorkItem]) -> tuple[bool, ...]:
    """Whether each item of a segment starts a tile."""
    return tuple(
        position == 0 or item.tile != items[position - 1].tile
        for position, item in enumerate(items)
    )


class PendingPush(NamedTuple):
    """A V push: its tick (counted from the segment's end once due after
    it), the ops and key of its tile, and its place among the handles the
    segment before carried, if that segment left it due."""

    tick: int
    ops: FragmentOps
    tile: Hashable
    carried: int | None = None


@dataclasses.dataclass(frozen=True, eq=False)
class Tail:
    """The static side of what the segments so far leave in flight: their
    last items (at most pv_pop, oldest first) with the ops that finish each,
    the first one's result slot, the ticks since the last tile started, the
    V pushes still due, and whether the next segment's first K is staged."""

    items: tuple[WorkItem, ...]
    item_ops: tuple[FragmentOps, ...]
    starts_tile: tuple[bool, ...]
    slot_base: int
    ticks_since_tile: int
    pending_v_pushes: tuple[PendingPush, ...]
    has_staged_next_k: bool

    def __post_init__(self) -> None:
        assert (
            len(self.item_ops) == len(self.items) == len(self.starts_tile)
        ), self

    @property
    def ops(self) -> FragmentOps:
        """The ops that finish the tail's newest item."""
        return self.item_ops[-1]

    def with_ops(self, ops: FragmentOps) -> "Tail":
        """The tail with every item and V push in flight finished by ops."""
        return dataclasses.replace(
            self,
            item_ops=(ops,) * len(self.items),
            pending_v_pushes=tuple(
                push._replace(ops=ops) for push in self.pending_v_pushes
            ),
        )

    def end_slot(self, num_slots: int) -> int:
        """The result slot of the item that follows the tail."""
        return (self.slot_base + len(self.items)) % num_slots

    def signature(
        self,
        delays: PipelineDelays,
        num_slots: int,
        *,
        has_mode_free_accumulate: bool = False,
    ) -> tuple:
        """What the next segment's schedule reads of this tail: a loop body
        may follow any tail whose signature equals the one it assumes. Which
        ops finish an item is the caller's to keep equivalent.
        has_mode_free_accumulate drops the mode of the items whose softmax
        has run, for ops whose accumulate reads no mode."""
        assert len(self.items) <= delays.pv_pop, self
        first_pending = len(self.items) - delays.qk_pop
        return (
            tuple(
                (
                    item.fragment,
                    item.mode
                    if position >= first_pending or not has_mode_free_accumulate
                    else None,
                    starts,
                )
                for position, (item, starts) in enumerate(
                    zip(self.items, self.starts_tile)
                )
            ),
            self.end_slot(num_slots),
            self.ticks_since_tile,
            tuple(
                (push.tick, push.ops.v_handle_words)
                for push in self.pending_v_pushes
            ),
            self.has_staged_next_k,
        )


class PipelineState(NamedTuple):
    """What a segment hands to the next. probs is what a loop carries: the
    bf16 probabilities of the fragments between softmax and PV issue (oldest
    first), then the handles of the V pushes still due. tail is None when
    nothing is in flight."""

    probs: tuple[jax.Array, ...]
    tail: Tail | None


class TimedItem(NamedTuple):
    """An item on a segment's timeline: its QK issue tick (negative in the
    tail), its ops, its result slot and whether it starts a tile."""

    tick: int
    item: WorkItem
    ops: FragmentOps
    slot: int
    starts_tile: bool


def _split_handles(
    words: tuple[jax.Array, ...], pushes: tuple[PendingPush, ...]
) -> list[tuple[jax.Array, ...]]:
    """words, the handles of pushes laid end to end, one per push."""
    handles = []
    for push in pushes:
        num_words = push.ops.v_handle_words
        handles.append(tuple(words[:num_words]))
        words = words[num_words:]
    assert not words, words
    return handles


def _load_register(timed: TimedItem) -> int | None:
    """The staging register timed's issue loads; None keeps the loaded
    tile."""
    return STAGING_REGISTER if timed.starts_tile else None


def _v_push_tick(
    delays: PipelineDelays, tile_tick: int, prev_tile_tick: int
) -> int:
    """The tick of the V push of the tile that starts at tile_tick: after the
    load of the tile before it, which started at prev_tile_tick."""
    lead_v_tick = tile_tick + delays.pv_issue - delays.v_push_lead
    return max(lead_v_tick, prev_tile_tick + 1 + delays.pv_issue)


def num_carried_v_pushes(delays: PipelineDelays, tile_items: int) -> int:
    """How many V pushes a segment of tiles of tile_items items each leaves
    due after it: its last tiles, whose V the next segment still reads."""
    num_carried = 0
    while (
        _v_push_tick(
            delays,
            -(num_carried + 1) * tile_items,
            -(num_carried + 2) * tile_items,
        )
        >= 0
    ):
        num_carried += 1
    return num_carried


def _segment_pushes(
    delays: PipelineDelays,
    tail: "Tail",
    segment: tuple[TimedItem, ...],
    num_ticks: int,
    has_next_k: bool,
) -> tuple[dict[int, list[TimedItem]], dict[int, list], tuple, int | None]:
    """The static push plan of ticks [0, num_ticks) after tail: the K
    pushes and the V pushes by tick, the V pushes due later, and the tick of
    the next segment's K push (has_next_k). A push follows the load of the
    tile before it, which frees the register."""
    k_plan: dict[int, list[TimedItem]] = {}
    v_plan: dict[int, list] = {}
    pending = []
    for index, push in enumerate(tail.pending_v_pushes):
        push = push._replace(carried=index)
        if push.tick < num_ticks:
            v_plan.setdefault(push.tick, []).append(push)
        else:
            pending.append(push._replace(tick=push.tick - num_ticks))
    prev_tile_tick = -tail.ticks_since_tile
    for timed in segment:
        if not timed.starts_tile:
            continue
        lead_k_tick = timed.tick - delays.k_push_lead
        k_tick = max(lead_k_tick, prev_tile_tick + 1)
        v_tick = _v_push_tick(delays, timed.tick, prev_tile_tick)
        if not (timed.tick == 0 and tail.has_staged_next_k):
            k_plan.setdefault(max(k_tick, 0), []).append(timed)
        push = PendingPush(v_tick, timed.ops, timed.item.tile)
        if v_tick < num_ticks:
            v_plan.setdefault(v_tick, []).append(push)
        else:
            pending.append(push._replace(tick=v_tick - num_ticks))
        prev_tile_tick = timed.tick
    next_k_tick = None
    # A last tile of one item frees the register only at the segment's last
    # tick, so the next segment pushes its first K tile itself.
    if has_next_k and prev_tile_tick + 1 < num_ticks:
        next_k_tick = max(num_ticks - delays.k_push_lead, prev_tile_tick + 1)
    return k_plan, v_plan, tuple(pending), next_k_tick


def _empty_tail(delays: PipelineDelays, slot_base: int = 0) -> Tail:
    """The tail of an idle pipeline whose next item takes slot slot_base."""
    return Tail(
        items=(),
        item_ops=(),
        starts_tile=(),
        slot_base=slot_base,
        ticks_since_tile=delays.push_horizon,
        pending_v_pushes=(),
        has_staged_next_k=False,
    )


def _advanced_tail(
    delays: PipelineDelays,
    tail: Tail,
    items: tuple[WorkItem, ...],
    ops: FragmentOps,
    *,
    has_next_k: bool,
) -> Tail:
    """The tail a segment of items, finished by ops, leaves after tail."""
    segment = _segment_items(tail, items, ops)
    _, _, pending, next_k_tick = _segment_pushes(
        delays, tail, segment, len(items), has_next_k
    )
    last_tile_tick = max(timed.tick for timed in segment if timed.starts_tile)
    all_items = tail.items + items
    all_ops = tail.item_ops + (ops,) * len(items)
    all_starts = tail.starts_tile + tile_starts(items)
    first_kept = max(len(all_items) - delays.pv_pop, 0)
    return Tail(
        items=all_items[first_kept:],
        item_ops=all_ops[first_kept:],
        starts_tile=all_starts[first_kept:],
        slot_base=(tail.slot_base + first_kept) % NUM_SLOTS,
        ticks_since_tile=min(len(items) - last_tile_tick, delays.push_horizon),
        pending_v_pushes=pending,
        has_staged_next_k=next_k_tick is not None,
    )


def _segment_items(
    tail: Tail, items: tuple[WorkItem, ...], ops: FragmentOps
) -> tuple[TimedItem, ...]:
    """items on the timeline after tail, from tick 0 on."""
    first_slot = tail.end_slot(NUM_SLOTS)
    return tuple(
        TimedItem(
            tick=position,
            item=item,
            ops=ops,
            slot=(first_slot + position) % NUM_SLOTS,
            starts_tile=starts,
        )
        for position, (item, starts) in enumerate(
            zip(items, tile_starts(items))
        )
    )


def _tail_items(tail: Tail) -> tuple[TimedItem, ...]:
    """tail's items on the timeline, at the ticks before 0."""
    first_tick = -len(tail.items)
    return tuple(
        TimedItem(
            tick=first_tick + position,
            item=item,
            ops=ops,
            slot=(tail.slot_base + position) % NUM_SLOTS,
            starts_tile=starts,
        )
        for position, (item, ops, starts) in enumerate(
            zip(tail.items, tail.item_ops, tail.starts_tile)
        )
    )


@dataclasses.dataclass(frozen=True, eq=False)
class MxuPipeline:
    """A ring of NUM_SLOTS result slots of each of the chip's first two MXUs
    and the tick schedule over them. Build one per kernel with allocate(),
    inside the kernel body."""

    delays: PipelineDelays
    qk_slots: tuple[jax.Array, ...]
    pv_slots: tuple[jax.Array, ...]
    has_mode_free_accumulate: bool = False

    def __post_init__(self) -> None:
        assert len(self.qk_slots) == len(self.pv_slots) == NUM_SLOTS
        delays = self.delays
        # A slot must hold a result from its issue to its pop.
        assert max(delays.qk_pop, delays.pv_pop - delays.pv_issue) < NUM_SLOTS

    @property
    def num_slots(self) -> int:
        return NUM_SLOTS

    @classmethod
    def allocate(
        cls,
        delays: PipelineDelays = PipelineDelays(),
        *,
        has_mode_free_accumulate: bool = False,
    ) -> "MxuPipeline":
        """A pipeline over fresh result slots. has_mode_free_accumulate
        (Tail.signature) is for loops whose rows start their accumulators at
        zero."""
        tpu_info = pltpu.get_tpu_info()
        assert tpu_info.num_accumulators, tpu_info.chip_version
        slot_type = jax.ShapeDtypeStruct(
            (FRAGMENT_ROWS, tpu_info.mxu_column_size), jnp.float32
        )
        return cls(
            delays=delays,
            has_mode_free_accumulate=has_mode_free_accumulate,
            qk_slots=tuple(
                jax.empty_ref(slot_type, memory_space=pltpu.ACC(QK_MXU))
                for _ in range(NUM_SLOTS)
            ),
            pv_slots=tuple(
                jax.empty_ref(slot_type, memory_space=pltpu.ACC(PV_MXU))
                for _ in range(NUM_SLOTS)
            ),
        )

    def idle_state(self) -> PipelineState:
        """The state with nothing in flight."""
        empty_probs = jnp.zeros(self.qk_slots[0].shape, jnp.bfloat16)
        return PipelineState(
            probs=(empty_probs,) * self.delays.num_carried_probs, tail=None
        )

    def check_loop_tail(self, actual: Tail, assumed: Tail) -> None:
        """Assert that a segment built to follow assumed may follow actual:
        a loop body traces once, against the tail it assumes."""
        actual_signature, assumed_signature = (
            tail.signature(
                self.delays,
                NUM_SLOTS,
                has_mode_free_accumulate=self.has_mode_free_accumulate,
            )
            for tail in (actual, assumed)
        )
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
        """The tail a loop body of items assumes from its previous
        iterations: back-to-back runs of items after an idle pipeline, as
        many as reach pv_pop items back, the last run's first item in slot
        slot_base. Each run stages the next one's first K tile, the last
        where has_staged_next_k."""
        num_runs = -(-self.delays.pv_pop // len(items))
        tail = _empty_tail(
            self.delays, (slot_base - (num_runs - 1) * len(items)) % NUM_SLOTS
        )
        for run in range(num_runs):
            is_last = run == num_runs - 1
            tail = _advanced_tail(
                self.delays,
                tail,
                items,
                ops,
                has_next_k=has_staged_next_k if is_last else True,
            )
        return tail

    def advance(
        self,
        state: PipelineState,
        items: tuple[WorkItem, ...],
        ops: FragmentOps,
        *,
        next_k_tile: Callable[[], jax.Array] | None = None,
    ) -> PipelineState:
        """Emit one tick per item: issue its QK and finish what the ticks
        before left due. next_k_tile returns the K tile of the next segment's
        first item, pushed k_push_lead ticks before this segment ends."""
        assert items, "a segment holds at least one item."
        has_next_k = next_k_tile is not None
        tail = state.tail
        if tail is None:
            tail = _empty_tail(self.delays)
        new_tail = _advanced_tail(
            self.delays, tail, items, ops, has_next_k=has_next_k
        )
        probs = self._emit(state, items, ops, next_k_tile)
        return PipelineState(probs=probs, tail=new_tail)

    def free_slots(self, tail: Tail, count: int) -> tuple[jax.Array, ...]:
        """count QK result slots that hold no result between the segment
        that left tail and the next one."""
        assert count <= NUM_SLOTS - self.delays.qk_pop, count
        first = tail.end_slot(NUM_SLOTS)
        return tuple(
            self.qk_slots[(first + offset) % NUM_SLOTS]
            for offset in range(count)
        )

    def drain(self, state: PipelineState) -> None:
        """Finish everything in flight; emits pv_pop ticks."""
        assert (
            not state.tail.has_staged_next_k
        ), "a drained segment must not stage a next K tile."
        self._emit(state, (), None, None)

    def _emit(
        self,
        state: PipelineState,
        items: tuple[WorkItem, ...],
        ops: FragmentOps | None,
        next_k_tile: Callable[[], jax.Array] | None,
    ) -> tuple[jax.Array, ...]:
        """Emit the ticks of a segment of items (none: a drain of pv_pop
        ticks) after state's tail; returns the next state's probs."""
        delays = self.delays
        tail = state.tail
        if tail is None:
            tail = _empty_tail(delays)
        segment = _segment_items(tail, items, ops)
        timeline = _tail_items(tail) + segment
        by_tick = {timed.tick: timed for timed in timeline}
        self._check_slots(timeline)

        num_ticks = len(items) if items else delays.pv_pop
        k_plan, v_plan, pending, next_k_tick = _segment_pushes(
            delays, tail, segment, num_ticks, next_k_tile is not None
        )
        assert items or not pending, "a drain leaves no V push due."

        # The probabilities of the tail's fragments whose PV is still due,
        # then the handles of its V pushes due.
        num_probs = delays.num_carried_probs
        probs = {}
        carried_ticks = range(-delays.pv_issue, -delays.qk_pop)
        for tick, value in zip(carried_ticks, state.probs[:num_probs]):
            if tick in by_tick:
                probs[tick] = value
        handles = _split_handles(state.probs[num_probs:], tail.pending_v_pushes)

        def _v_tile(push):
            if push.carried is None:
                return push.ops.v_tile(push.tile)
            return push.ops.v_tile_at(handles[push.carried])

        for tick in range(num_ticks):
            for timed in k_plan.get(tick, ()):
                self._push(QK_MXU, timed.ops.k_tile(timed.item.tile))
            if tick == next_k_tick:
                self._push(QK_MXU, next_k_tile())
            for push in v_plan.get(tick, ()):
                self._push(PV_MXU, _v_tile(push))
            self._emit_tick(tick, by_tick, probs)
        if not items:
            return state.probs[:num_probs]
        new_probs = self._carried_probs(probs, by_tick, len(items))
        new_handles = tuple(
            word
            for push in pending
            for word in (
                push.ops.v_handle(push.tile)
                if push.carried is None
                else handles[push.carried]
            )
        )
        return new_probs + new_handles

    def _carried_probs(
        self,
        probs: dict[int, jax.Array],
        by_tick: dict[int, TimedItem],
        num_ticks: int,
    ) -> tuple[jax.Array, ...]:
        """The probabilities a segment of num_ticks ticks leaves between
        softmax and PV issue; a tick that holds no item carries zeros."""
        delays = self.delays
        empty_probs = jnp.zeros(self.qk_slots[0].shape, jnp.bfloat16)
        return tuple(
            probs[tick] if tick in by_tick else empty_probs
            for tick in range(
                num_ticks - delays.pv_issue, num_ticks - delays.qk_pop
            )
        )

    @staticmethod
    def _push(mxu: int, tile: jax.Array) -> None:
        """Push a K tile (transposed, QK_MXU) or a V tile (PV_MXU)."""
        pltpu.matmul_push_rhs(
            tile,
            staging_register=STAGING_REGISTER,
            mxu_index=mxu,
            transpose=mxu == QK_MXU,
        )

    def _emit_tick(
        self,
        tick: int,
        by_tick: dict[int, TimedItem],
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

    def _check_slots(self, timeline: Sequence[TimedItem]) -> None:
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
