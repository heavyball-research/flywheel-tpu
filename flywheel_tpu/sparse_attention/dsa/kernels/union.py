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

"""SparseCore union of the top-k selections of each run of tokens: the
positions any of its tokens selects, once each, each token's membership
bitmap over the union, the bits of the union indices of the positions it
selects, and the union's metadata row for the gather of its positions."""

import functools
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu, tpu_sc as plsc

from flywheel_tpu.core.tpu import NUM_LANES
from flywheel_tpu.sparse_attention.dsa.kernels.gather import (
    CHUNK_EMPTY_TOKENS,
    CHUNK_SLOTS,
    CHUNK_WORDS,
    META_POSITION,
    META_SLOTS,
    META_TABLE_OFFSET,
    META_WORDS,
    UNUSED_INDEX,
    gather_compiler_params,
    sparse_core_mesh,
)
from flywheel_tpu.sparse_attention.dsa.kernels.masked_dense import (
    SLAB_POSITIONS,
    bitmap_rows,
    clear_bitmap,
    set_bitmap_bits,
    wait_last_copies,
)
from flywheel_tpu.sparse_attention.dsa.kernels.union_attention import (
    HEADER_FIRST_UNION,
    HEADER_LAST_UNION,
    HEADER_NUM_REAL,
    HEADER_ROW,
    HEADER_WORDS,
    ROW_FIRST_ROW,
    ROW_FIRST_TOKEN,
    ROW_NEXT_UNION,
    ROW_NUM_PERIODS,
    ROW_NUM_TOKENS,
    ROW_SIZE,
    ROW_WORDS,
    union_periods,
)
from flywheel_tpu.topk.kernels.sparse_core_topk import sparse_core_vmem_words

# Lanes of a union's row: its first token's call row, its tokens (consecutive,
# in one request), that token's position, its index row (negative: in the
# previous indices), its previous-indices row and its block-table offset.
UNION_WORDS = 16
UNION_FIRST_TOKEN = 0
UNION_NUM_TOKENS = 1
UNION_FIRST_POSITION = 2
UNION_INDEX_ROW = 3
UNION_PREVIOUS_ROW = 4
UNION_TABLE_OFFSET = 5
# Lane of a chunk's count row: the unions it unites, its first ones. The
# union table holds a block per chunk: its count row, then its union rows.
COUNT_UNIONS = 0
# Lane of the call's chunk row: the first word of its chunk's block.
CHUNK_BLOCK_WORD = 0
# Rows of the call rows' scratch: the schedule header and the gather's chunk
# row.
CALL_SCHEDULE_HEADER = 0
CALL_GATHER_ROW = 1
NUM_CALL_ROWS = 2
# Note (david): a union's positions leave in copies of UNION_PAGE slots, so
# its row holds whole pages.
UNION_PAGE = 256
# Note (david): the gather reads a union's positions in whole chunks, so the
# slots through the end of its last chunk hold UNUSED_INDEX.
UNION_FILL_SLOTS = CHUNK_SLOTS
NUM_INDEX_BUFFERS = 2
NUM_TOKEN_BITMAPS = 2
SEM_INDEX = 0
SEM_BITMAP = 1
SEM_UNION = 2
NUM_UNION_SEMS = 3
UNION_UNROLL = 4


def num_union_workers() -> int:
    """The union kernel's workers: every subcore of every SparseCore."""
    sparse_core = pltpu.get_tpu_info().sparse_core
    return sparse_core.num_cores * sparse_core.num_subcores


def padded_union_slots(num_slots: int) -> int:
    """A chunk's union rows in the table: its slots padded to a multiple of
    the workers, as a worker loads a whole run of rows."""
    num_workers = num_union_workers()
    return pl.cdiv(num_slots, num_workers) * num_workers


def schedule_rows(num_unions: int) -> int:
    """Rows of a schedule of num_unions unions: the header and a row per
    union."""
    return HEADER_ROW + 1 + num_unions


def union_scratch_shapes(
    num_unions: int,
    union_tokens: int,
    top_k: int,
    union_slots: int,
    num_slabs: int,
    max_seqlen_k: int,
) -> tuple[tuple[int, ...], ...]:
    """The shapes of the union kernel's int32 VMEM scratch, in UnionRefs
    order."""
    lanes = pltpu.get_tpu_info().sparse_core.num_lanes
    unions_per_worker = pl.cdiv(num_unions, num_union_workers())
    return (
        (unions_per_worker * UNION_WORDS,),
        (NUM_INDEX_BUFFERS, union_tokens, top_k),
        (NUM_TOKEN_BITMAPS, bitmap_rows(num_slabs), NUM_LANES),
        (pl.cdiv(max_seqlen_k, lanes) * lanes,),
        (union_slots,),
        (unions_per_worker, META_WORDS),
        (unions_per_worker, ROW_WORDS),
        (NUM_CALL_ROWS, UNION_WORDS),
    )


def union_vmem_words(
    num_unions: int,
    union_tokens: int,
    top_k: int,
    union_slots: int,
    num_slabs: int,
    max_seqlen_k: int,
) -> int:
    """Subcore VMEM words of the union kernel's scratch."""
    shapes = union_scratch_shapes(
        num_unions, union_tokens, top_k, union_slots, num_slabs, max_seqlen_k
    )
    return sum(math.prod(shape) for shape in shapes)


class UnionRefs(NamedTuple):
    """The union kernel's operands, outputs and scratch."""

    indices_hbm: jax.Array
    previous_indices_hbm: jax.Array
    unions_hbm: jax.Array
    chunk_hbm: jax.Array
    union_hbm: jax.Array
    meta_hbm: jax.Array
    bitmap_hbm: jax.Array
    schedule_hbm: jax.Array
    gather_row_hbm: jax.Array
    union_row_vmem: jax.Array
    index_vmem: jax.Array
    bitmap_vmem: jax.Array
    owner_vmem: jax.Array
    union_vmem: jax.Array
    meta_vmem: jax.Array
    schedule_vmem: jax.Array
    call_vmem: jax.Array
    sems: jax.Array


class UnionWorker:
    """A worker's run of unions. A union's tokens mark each position they
    select with the first of them that selects it, -(token + 1) in
    owner_vmem; then token by token, a position's first selection gives it
    the next union index, which replaces its mark. So a union lists its
    positions in the order of their first selection, and no owner word is
    read before its union writes it.

    A token's slot is used as the gather reads it: its index in [0, the
    token's position] and the slot not past that position. Its bitmap holds
    the bit of each union index it selects (set_bitmap_bits).
    """

    def __init__(
        self,
        refs: UnionRefs,
        *,
        first_word: jax.Array,
        first_union: jax.Array,
        num_unions: jax.Array,
        num_slots: int,
        top_k: int,
        num_slabs: int,
        num_lanes: int,
    ):
        self.refs = refs
        self.first_word = first_word
        self.first_union = first_union
        self.num_unions = num_unions
        self.num_slots = num_slots
        self.unions_per_worker = refs.union_row_vmem.shape[0] // UNION_WORDS
        self.top_k = top_k
        self.union_slots = refs.union_vmem.shape[0]
        self.num_slabs = num_slabs
        self.rows_per_token = refs.bitmap_vmem.shape[1]
        self.lanes = num_lanes
        self.num_vectors = top_k // num_lanes
        self.lane = lax.broadcasted_iota(jnp.int32, (num_lanes,), 0)

    def row_vector(self, local_union) -> jax.Array:
        return self.refs.union_row_vmem[
            pl.ds(local_union * UNION_WORDS, self.lanes)
        ]

    def union_row(self, local_union):
        """(first token, tokens, first position) of the worker's union."""
        row = self.row_vector(local_union)
        return (
            row[UNION_FIRST_TOKEN],
            row[UNION_NUM_TOKENS],
            row[UNION_FIRST_POSITION],
        )

    def index_copy(self, indices_hbm, row, buffer, token):
        return pltpu.make_async_copy(
            indices_hbm.at[row],
            self.refs.index_vmem.at[buffer, token],
            self.refs.sems.at[SEM_INDEX, buffer],
        )

    def index_copies(self, local_union, *, is_start: bool) -> None:
        """Starts, or waits, the copies of the union's index rows: a token's
        row from the indices, or from the previous indices before them."""
        refs = self.refs
        row = self.row_vector(local_union)
        num_tokens = row[UNION_NUM_TOKENS]
        buffer = lax.rem(local_union, NUM_INDEX_BUFFERS)

        @pl.loop(0, num_tokens)
        def _token(token):
            if not is_start:
                self.index_copy(refs.indices_hbm, 0, buffer, token).wait()
                return
            index_row = row[UNION_INDEX_ROW] + token
            pl.when(index_row >= 0)(
                self.index_copy(
                    refs.indices_hbm, index_row, buffer, token
                ).start
            )
            pl.when(index_row < 0)(
                self.index_copy(
                    refs.previous_indices_hbm,
                    row[UNION_PREVIOUS_ROW] + token,
                    buffer,
                    token,
                ).start
            )

    def bitmap_copy(self, token_row, slot):
        rows = self.rows_per_token
        return pltpu.make_async_copy(
            self.refs.bitmap_vmem.at[slot],
            self.refs.bitmap_hbm.at[pl.ds(token_row * rows, rows)],
            self.refs.sems.at[SEM_BITMAP, slot],
        )

    def union_copy(self, local_union, page):
        first_slot = (self.first_union + local_union) * self.union_slots
        return pltpu.make_async_copy(
            self.refs.union_vmem.at[pl.ds(page * UNION_PAGE, UNION_PAGE)],
            self.refs.union_hbm.at[
                pl.ds(first_slot + page * UNION_PAGE, UNION_PAGE)
            ],
            self.refs.sems.at[SEM_UNION, 0],
        )

    def selected(self, buffer, token, vector, position):
        """(index, is_used) of a vector of the token's slots; an unused
        slot's index reads as 0."""
        index = self.refs.index_vmem[
            buffer, token, pl.ds(vector * self.lanes, self.lanes)
        ]
        slot = vector * self.lanes + self.lane
        is_used = (index >= 0) & (index <= position) & (slot <= position)
        return jnp.where(is_used, index, 0), is_used

    def mark_owners(self, buffer, num_tokens, first_position):
        """Each selected position's mark: the union's last token marks
        first, so the first token that selects it marks it last."""

        @pl.loop(0, num_tokens)
        def _token(step):
            token = num_tokens - 1 - step
            mark = jnp.full((self.lanes,), -1 - token, jnp.int32)

            @plsc.parallel_loop(0, self.num_vectors, unroll=UNION_UNROLL)
            def _vector(vector):
                index, is_used = self.selected(
                    buffer, token, vector, first_position + token
                )
                plsc.store_scatter(
                    self.refs.owner_vmem, [index], mark, mask=is_used
                )

    def assign_union_indices(self, buffer, token, position, size, slot):
        """The token's first selections take union indices from size on, and
        its bitmap goes to bitmap_vmem's slot; returns the union's new size.
        A token's used indices are distinct, so its vectors touch distinct
        owner words."""
        refs = self.refs
        clear_bitmap(refs.bitmap_vmem, slot, self.num_slabs, self.lanes)

        def _vector(vector, size):
            index, is_used = self.selected(buffer, token, vector, position)
            owner = plsc.load_gather(refs.owner_vmem, [index])
            is_first = is_used & (owner == -1 - token)
            union_index = size + plsc.cumsum(is_first.astype(jnp.int32)) - 1
            plsc.store_scatter(
                refs.owner_vmem, [index], union_index, mask=is_first
            )
            plsc.store_compressed(
                refs.union_vmem.at[pl.ds(size, self.lanes)],
                index,
                mask=is_first,
            )
            member = jnp.where(is_first, union_index, owner)
            set_bitmap_bits(refs.bitmap_vmem.at[slot], member, is_used)
            return size + plsc.all_reduce_population_count(is_first)[0]

        return plsc.parallel_loop(
            0, self.num_vectors, unroll=UNION_UNROLL, carry=size
        )(_vector)

    def unite(self, local_union, buffer, num_issued):
        """The union's positions and its tokens' bitmaps, each token's
        leaving as it is done; returns (union size, bitmaps issued)."""
        first_token, num_tokens, first_position = self.union_row(local_union)

        def _token(token, carry):
            size, num_issued = carry
            slot = lax.rem(num_issued, NUM_TOKEN_BITMAPS)
            pl.when(num_issued >= NUM_TOKEN_BITMAPS)(
                self.bitmap_copy(first_token, slot).wait
            )
            size = self.assign_union_indices(
                buffer, token, first_position + token, size, slot
            )
            self.bitmap_copy(first_token + token, slot).start()
            return size, num_issued + 1

        return lax.fori_loop(0, num_tokens, _token, (jnp.int32(0), num_issued))

    def finish_union(self, local_union, size):
        """UNUSED_INDEX after the union's positions, their copy out and the
        union's metadata row: the gather fills its size's slots, each one's
        position at most its last token's; returns the pages in flight."""
        refs = self.refs
        unused = jnp.full((self.lanes,), UNUSED_INDEX, jnp.int32)
        for slot in range(0, UNION_FILL_SLOTS, self.lanes):
            refs.union_vmem[pl.ds(size + slot, self.lanes)] = unused
        num_pages = (size + UNION_FILL_SLOTS + UNION_PAGE - 1) // UNION_PAGE

        @pl.loop(0, num_pages)
        def _page(page):
            self.union_copy(local_union, page).start()

        row = self.row_vector(local_union)
        num_tokens = row[UNION_NUM_TOKENS]
        last_position = jnp.where(
            num_tokens > 0, row[UNION_FIRST_POSITION] + num_tokens - 1, -1
        )
        meta = meta_vector(
            self.lane, row[UNION_TABLE_OFFSET], last_position, size
        )
        union = self.first_union + local_union
        schedule = lane_vector(
            self.lane,
            {
                ROW_FIRST_TOKEN: row[UNION_FIRST_TOKEN],
                ROW_NUM_TOKENS: num_tokens,
                ROW_SIZE: size,
                ROW_FIRST_ROW: union * self.union_slots,
                ROW_NUM_PERIODS: union_periods(num_tokens, size),
                ROW_NEXT_UNION: jnp.where(
                    union + 1 < self.num_unions, union + 1, self.num_slots
                ),
            },
        )
        for word in range(0, META_WORDS, self.lanes):
            refs.meta_vmem[local_union, pl.ds(word, self.lanes)] = (
                meta if word == 0 else jnp.zeros_like(meta)
            )
        for word in range(0, ROW_WORDS, self.lanes):
            refs.schedule_vmem[local_union, pl.ds(word, self.lanes)] = (
                schedule if word == 0 else jnp.zeros_like(schedule)
            )
        return num_pages

    def wait_union(self, num_pages):
        @pl.loop(0, num_pages)
        def _page(page):
            self.union_copy(0, page).wait()

    def load_rows(self) -> None:
        """Loads the union rows from the worker's first on."""
        refs = self.refs
        pltpu.sync_copy(
            refs.unions_hbm.at[
                pl.ds(
                    self.first_word + self.first_union * UNION_WORDS,
                    refs.union_row_vmem.shape[0],
                )
            ],
            refs.union_row_vmem,
        )

    def run_union(self, num_local, local_union, carry):
        num_issued, num_pages = carry
        _, num_tokens, first_position = self.union_row(local_union)
        buffer = lax.rem(local_union, NUM_INDEX_BUFFERS)
        pl.when(local_union + 1 < num_local)(
            functools.partial(self.index_copies, local_union + 1, is_start=True)
        )
        self.index_copies(local_union, is_start=False)
        self.mark_owners(buffer, num_tokens, first_position)
        # Note (david): the union stage is free once the union before's
        # pages have left; they leave while this union marks.
        self.wait_union(num_pages)
        size, num_issued = self.unite(local_union, buffer, num_issued)
        return num_issued, self.finish_union(local_union, size)

    def run(self, num_local) -> None:
        """Unites the worker's first num_local unions and stores their
        metadata and schedule rows."""
        self.index_copies(0, is_start=True)
        num_issued, num_pages = lax.fori_loop(
            0,
            num_local,
            functools.partial(self.run_union, num_local),
            (jnp.int32(0), jnp.int32(0)),
        )
        self.wait_union(num_pages)
        wait_last_copies(
            num_issued,
            NUM_TOKEN_BITMAPS,
            lambda issued: self.bitmap_copy(
                0, lax.rem(issued, NUM_TOKEN_BITMAPS)
            ).wait(),
        )

        @pl.loop(0, num_local)
        def _store(local_union):
            union = self.first_union + local_union
            pltpu.sync_copy(
                self.refs.meta_vmem.at[local_union],
                self.refs.meta_hbm.at[union],
            )
            pltpu.sync_copy(
                self.refs.schedule_vmem.at[local_union],
                self.refs.schedule_hbm.at[
                    HEADER_ROW + 1 + union, pl.ds(0, ROW_WORDS)
                ],
            )


def lane_vector(lane: jax.Array, words: dict[int, jax.Array]) -> jax.Array:
    """A vector of the given words at their lanes, zeros elsewhere."""
    vector = jnp.zeros_like(lane)
    for index, word in words.items():
        vector = jnp.where(lane == index, word, vector)
    return vector


def meta_vector(
    lane: jax.Array,
    table_offset: jax.Array,
    last_position: jax.Array,
    num_slots: jax.Array,
) -> jax.Array:
    """The first lanes of a gather metadata row."""
    return jnp.where(
        lane == META_TABLE_OFFSET,
        table_offset,
        jnp.where(
            lane == META_POSITION,
            last_position,
            jnp.where(lane == META_SLOTS, num_slots, 0),
        ),
    )


def read_row_word(
    refs: UnionRefs, row_hbm: jax.Array, lane: int, num_lanes: int
) -> jax.Array:
    """The lane of a row of UNION_WORDS words in HBM."""
    pltpu.sync_copy(row_hbm, refs.union_row_vmem.at[pl.ds(0, UNION_WORDS)])
    return refs.union_row_vmem[pl.ds(0, num_lanes)][lane]


def store_call_rows(
    refs: UnionRefs,
    num_unions: jax.Array,
    *,
    num_slots: int,
    num_real: int,
    lane: jax.Array,
) -> None:
    """The schedule's header and the gather's chunk row of the call's
    num_unions unions out of num_slots: the gather skips the slots past
    them."""
    rows = {
        CALL_SCHEDULE_HEADER: {
            HEADER_FIRST_UNION: jnp.where(num_unions > 0, 0, num_slots),
            HEADER_LAST_UNION: num_unions - 1,
            HEADER_NUM_REAL: jnp.int32(num_real),
        },
        CALL_GATHER_ROW: {CHUNK_EMPTY_TOKENS: num_slots - num_unions},
    }
    num_lanes = lane.shape[0]
    for row, words in rows.items():
        vector = lane_vector(lane, words)
        for word in range(0, UNION_WORDS, num_lanes):
            refs.call_vmem[row, pl.ds(word, num_lanes)] = (
                vector if word == 0 else jnp.zeros_like(vector)
            )
    pltpu.sync_copy(
        refs.call_vmem.at[CALL_SCHEDULE_HEADER],
        refs.schedule_hbm.at[HEADER_ROW, pl.ds(0, HEADER_WORDS)],
    )
    pltpu.sync_copy(
        refs.call_vmem.at[CALL_GATHER_ROW, pl.ds(0, CHUNK_WORDS)],
        refs.gather_row_hbm,
    )


def union_kernel(
    *refs: jax.Array,
    top_k: int,
    num_slabs: int,
    num_slots: int,
    num_real: int,
    num_subcores: int,
    num_lanes: int,
) -> None:
    """The workers split the chunk's first unions, as many as its count row
    names, into even runs; each unites its run and stores the run's
    metadata and schedule rows, and the first worker the call rows."""
    refs = UnionRefs(*refs)
    num_workers = num_union_workers()
    worker = lax.axis_index("core") * num_subcores + lax.axis_index("subcore")
    block_word = read_row_word(
        refs, refs.chunk_hbm, CHUNK_BLOCK_WORD, num_lanes
    )
    num_unions = read_row_word(
        refs,
        refs.unions_hbm.at[pl.ds(block_word, UNION_WORDS)],
        COUNT_UNIONS,
        num_lanes,
    )
    lane = lax.broadcasted_iota(jnp.int32, (num_lanes,), 0)
    pl.when(worker == 0)(
        functools.partial(
            store_call_rows,
            refs,
            num_unions,
            num_slots=num_slots,
            num_real=num_real,
            lane=lane,
        )
    )
    per_worker = (num_unions + num_workers - 1) // num_workers
    union = UnionWorker(
        refs,
        first_word=block_word + UNION_WORDS,
        first_union=worker * per_worker,
        num_unions=num_unions,
        num_slots=num_slots,
        top_k=top_k,
        num_slabs=num_slabs,
        num_lanes=num_lanes,
    )
    union.load_rows()
    num_local = jnp.clip(num_unions - worker * per_worker, 0, per_worker)
    pl.when(num_local > 0)(functools.partial(union.run, num_local))


def dsa_union_pallas(
    indices: jax.Array,
    previous_indices: jax.Array,
    unions: jax.Array,
    chunk_row: jax.Array,
    *,
    num_tokens: int,
    union_tokens: int,
    union_slots: int,
    num_slabs: int,
    max_seqlen_k: int,
    num_slots: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """One chunk's union on the SparseCores.

    Args:
      indices: (R, top_k) int32 positions within each token's request, of
        the rows UNION_INDEX_ROW names; a token's used indices distinct and
        below max_seqlen_k, top_k a multiple of the SparseCore's lanes.
      previous_indices: (R', top_k) int32 rows UNION_PREVIOUS_ROW names, for
        a token whose UNION_INDEX_ROW is negative.
      unions: the union table: per chunk a count row (lane COUNT_UNIONS) and
        padded_union_slots(num_slots) union rows, the ones with tokens first.
      chunk_row: (UNION_WORDS,) int32, lane CHUNK_BLOCK_WORD the chunk's
        first word in the table.
      num_tokens, union_tokens, num_slabs, max_seqlen_k: the call's tokens,
        a union's most tokens, a bitmap's slabs, the positions' bound.
      union_slots: slots of a union's row, whole UNION_PAGE, at least
        union_tokens * top_k + UNION_FILL_SLOTS.
      num_slots: the chunk's slots, the unions of its schedule.

    Returns:
      (union, meta, bitmap, schedule, gather_row): each union's positions in
      its row, UNUSED_INDEX through UNION_FILL_SLOTS after them; the gather's
      metadata rows; the tokens' bitmaps over their union; the attention's
      schedule rows; the gather's chunk row, past the unions' slots empty.
    """
    top_k = indices.shape[1]
    slots = padded_union_slots(num_slots)
    check_union_operands(
        unions.shape[0] % ((1 + slots) * UNION_WORDS),
        slots,
        top_k,
        union_tokens,
        union_slots,
        num_slabs,
        max_seqlen_k,
    )
    scratch_shapes = union_scratch_shapes(
        slots, union_tokens, top_k, union_slots, num_slabs, max_seqlen_k
    )
    sparse_core = pltpu.get_tpu_info().sparse_core
    body = functools.partial(
        union_kernel,
        top_k=top_k,
        num_slabs=num_slabs,
        num_slots=num_slots,
        num_real=num_tokens,
        num_subcores=sparse_core.num_subcores,
        num_lanes=sparse_core.num_lanes,
    )
    kernel = pl.kernel(
        body,
        out_type=union_out_types(slots, union_slots, num_tokens, num_slabs),
        mesh=sparse_core_mesh(),
        scratch_types=(
            *(pltpu.VMEM(shape, jnp.int32) for shape in scratch_shapes),
            pltpu.SemaphoreType.DMA(
                (NUM_UNION_SEMS, max(NUM_INDEX_BUFFERS, NUM_TOKEN_BITMAPS))
            ),
        ),
        compiler_params=gather_compiler_params(),
        name="dsa_union",
    )
    return kernel(indices, previous_indices, unions, chunk_row)


def union_out_types(
    chunk_slots: int, union_slots: int, num_tokens: int, num_slabs: int
) -> tuple[jax.ShapeDtypeStruct, ...]:
    """The union kernel's int32 outputs, in dsa_union_pallas's order."""
    shapes = (
        (chunk_slots * union_slots,),
        (chunk_slots, META_WORDS),
        (num_tokens * bitmap_rows(num_slabs), NUM_LANES),
        (schedule_rows(chunk_slots), NUM_LANES),
        (CHUNK_WORDS,),
    )
    return tuple(jax.ShapeDtypeStruct(shape, jnp.int32) for shape in shapes)


def check_union_operands(
    table_remainder: int,
    num_unions: int,
    top_k: int,
    union_tokens: int,
    union_slots: int,
    num_slabs: int,
    max_seqlen_k: int,
) -> None:
    """Fails unless the union table holds whole chunk blocks of num_unions
    rows and a union, its bitmaps and the scratch fit."""
    lanes = pltpu.get_tpu_info().sparse_core.num_lanes
    max_size = union_tokens * top_k
    needed_slots = max_size + UNION_FILL_SLOTS
    vmem_words = union_vmem_words(
        num_unions, union_tokens, top_k, union_slots, num_slabs, max_seqlen_k
    )
    if table_remainder:
        raise ValueError(
            "The union table must hold whole blocks of a count row and"
            f" {num_unions} union rows; {table_remainder} words are left."
        )
    elif top_k % lanes:
        raise ValueError(f"top_k must be a multiple of {lanes}; got {top_k}.")
    elif union_slots < needed_slots or union_slots % UNION_PAGE:
        raise ValueError(
            f"union_slots must be at least {needed_slots} in whole"
            f" {UNION_PAGE}; got {union_slots}."
        )
    elif max_size > num_slabs * SLAB_POSITIONS:
        raise ValueError(
            f"{num_slabs} bitmap slabs cover {num_slabs * SLAB_POSITIONS}"
            f" union indices; a union holds up to {max_size}."
        )
    elif vmem_words > sparse_core_vmem_words():
        raise ValueError(
            f"The union kernel needs {vmem_words} words of SparseCore VMEM at"
            f" max_seqlen_k {max_seqlen_k}; a subcore holds"
            f" {sparse_core_vmem_words()}."
        )
