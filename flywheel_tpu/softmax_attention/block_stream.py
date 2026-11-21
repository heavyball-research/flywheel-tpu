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

"""The kv block stream of a dense or packed varlen forward main loop.

A loop stages its kv blocks two ahead of the one it runs, in the flattened
order of its head groups' rows. A block's descriptor is the inputs of its
staging DMAs, (kv head, first kv token), which DenseBlockStream and
VarlenBlockStream read from their schedules.
"""

import dataclasses
from functools import reduce

import jax
import jax.numpy as jnp

from flywheel_tpu.softmax_attention.fwd_blocks import (
    KVRefs,
    start_kv_block,
    wait_kv_block,
)
from flywheel_tpu.softmax_attention.schedule import FwdSchedule, ScheduleRow
from flywheel_tpu.softmax_attention.traits import FwdTraits

BlockDesc = tuple[jax.Array, ...]


def select_desc(
    predicate: jax.Array, desc_true: BlockDesc, desc_false: BlockDesc
) -> BlockDesc:
    return tuple(
        jnp.where(predicate, field_true, field_false)
        for field_true, field_false in zip(desc_true, desc_false)
    )


@dataclasses.dataclass(frozen=True, eq=False)
class BlockStream:
    """What the dense and the varlen block streams share: whether a block
    repeats the one before it, a block's staging DMAs, and the row order."""

    traits: FwdTraits
    schedule: FwdSchedule
    kv_refs: KVRefs

    @property
    def reuses_blocks(self) -> bool:
        """Whether a block whose DMA inputs equal the previous block's runs
        from the previous block's slot instead of being staged again."""
        # Note (david): a block whose DMA inputs equal the previous block's
        # holds the same staged data, so it runs from the previous block's
        # slot; dense GQA at one kv block per row otherwise re-staged each
        # block once per q head. Paged blocks start their DMAs inside the
        # copy and a rotary K is rotated in place, so both load every block.
        return not (self.traits.rotary_dim and self.traits.rotary_k)

    def is_fresh(self, desc: BlockDesc, prev_desc: BlockDesc) -> jax.Array:
        if self.reuses_blocks:
            differs = [
                field != prev_field
                for field, prev_field in zip(desc, prev_desc)
            ]
            return reduce(jnp.logical_or, differs)
        else:
            return jnp.bool_(True)

    def start(self, desc: BlockDesc, slot: jax.Array | int) -> None:
        kv_head, kv_tok = desc
        start_kv_block(self.traits, self.kv_refs, kv_head, kv_tok, slot)

    def wait(self, slot: jax.Array | int) -> None:
        wait_kv_block(self.traits, self.kv_refs, slot)

    def row_after(
        self, g: jax.Array, si: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """The (group, row) after row si of group g; the grid's last row is its
        own successor."""
        num_rows = self.schedule.num_rows
        num_head_groups = self.schedule.num_head_groups
        is_row_end = si == num_rows - 1
        is_grid_end = jnp.logical_and(g == num_head_groups - 1, is_row_end)
        next_g = jnp.where(
            is_row_end, jnp.minimum(g + 1, num_head_groups - 1), g
        )
        next_si = jnp.where(is_grid_end, si, jnp.where(is_row_end, 0, si + 1))
        return next_g, next_si


@dataclasses.dataclass(frozen=True, eq=False)
class DenseBlockStream(BlockStream):
    """The kv blocks of a dense loop's flattened block order. Under
    is_diagonal_first a row runs its last block (the one that holds the
    causal diagonal) first, then its other blocks in order."""

    is_diagonal_first: bool

    def desc(self, head: jax.Array, kv: jax.Array) -> BlockDesc:
        """The DMA inputs of kv block kv of head's group."""
        heads_outer_batch = self.schedule.heads_outer_batch
        q_heads_per_kv_head = self.schedule.q_heads_per_kv_head
        # Note (david): the heads-outer fold flattens (heads, batch) with
        # B = heads_outer_batch, so q index n * B + b reads kv index
        # (n // q_heads_per_kv_head) * B + b; the batch-outer fold flattens
        # (batch, heads), so consecutive q indices share one kv index.
        if heads_outer_batch is None:
            kv_head = head // q_heads_per_kv_head
        else:
            kv_head = (
                head // heads_outer_batch // q_heads_per_kv_head
            ) * heads_outer_batch + head % heads_outer_batch
        desc = (kv_head, kv * self.traits.bkv)
        # Note (david): the descriptors ride the loop carry, which needs strong
        # int32 scalars.
        return tuple(jnp.asarray(field, jnp.int32) for field in desc)

    def first_two_blocks(
        self, g: jax.Array, si: jax.Array
    ) -> tuple[BlockDesc, BlockDesc, tuple]:
        """The descriptors of the first two blocks of the flattened order from
        row si of group g on, and that row's (lo, hi, ctx)."""
        group_heads = self.traits.group_heads
        lo, hi, ctx = self.schedule.row_interval(si)
        after_g, after_si = self.row_after(g, si)
        after_lo, after_hi, _ = self.schedule.row_interval(after_si)
        if self.is_diagonal_first:
            first_kv, after_kv = hi, after_hi
        else:
            first_kv, after_kv = lo, after_lo
        first_desc = self.desc(g * group_heads, first_kv)
        second_desc = select_desc(
            lo < hi,
            self.desc(
                g * group_heads, lo if self.is_diagonal_first else lo + 1
            ),
            self.desc(after_g * group_heads, after_kv),
        )
        return first_desc, second_desc, (lo, hi, ctx)


@dataclasses.dataclass(frozen=True, eq=False)
class VarlenBlockStream(BlockStream):
    """The kv blocks of a varlen loop's flattened block order: the DMA inputs
    of each block of a head group's row and the descriptors that lead into
    the next row."""

    def desc(
        self, head: jax.Array, kv: jax.Array, ctx: ScheduleRow
    ) -> BlockDesc:
        """The DMA inputs of kv block kv of head's group in the row of ctx."""
        # A packed build has no heads-outer fold: consecutive q heads share
        # one kv head.
        kv_head = head // self.schedule.q_heads_per_kv_head
        desc = (kv_head, self.schedule.kv_token(kv, ctx))
        # Note (david): the descriptors ride the loop carry, which needs strong
        # int32 scalars.
        return tuple(jnp.asarray(field, jnp.int32) for field in desc)

    def first_two_blocks(
        self, g: jax.Array, si: jax.Array
    ) -> tuple[BlockDesc, BlockDesc, tuple]:
        """The descriptors of the first two blocks of the flattened order from
        row si of group g on, and that row's (lo, hi, ctx)."""
        group_heads = self.traits.group_heads
        lo, hi, ctx = self.schedule.row_interval(si)
        after_g, after_si = self.row_after(g, si)
        after_lo, _, after_ctx = self.schedule.row_interval(after_si)
        first_desc = self.desc(g * group_heads, lo, ctx)
        second_desc = select_desc(
            lo < hi,
            self.desc(g * group_heads, lo + 1, ctx),
            self.desc(after_g * group_heads, after_lo, after_ctx),
        )
        return first_desc, second_desc, (lo, hi, ctx)

    def ahead_desc(
        self,
        base_head: jax.Array,
        kv: jax.Array,
        hi: jax.Array,
        ctx: ScheduleRow,
        ahead: tuple[BlockDesc, BlockDesc],
    ) -> BlockDesc:
        """The descriptor of the block two after kv block kv of the row, where
        ahead holds the next row's first two in the flattened order."""
        first_next_row_desc, second_ahead_desc = ahead
        return select_desc(
            kv + 2 <= hi,
            self.desc(base_head, kv + 2, ctx),
            select_desc(kv + 1 == hi, first_next_row_desc, second_ahead_desc),
        )
