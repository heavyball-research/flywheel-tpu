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
positions any of its tokens selects, once each, and each token's membership
bitmap over the union, the bits of the union indices of the positions it
selects."""

import functools
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu, tpu_sc as plsc

from flywheel_tpu.core.tpu import NUM_LANES, NUM_SUBLANES
from flywheel_tpu.sparse_attention.dsa.kernels.gather import (
    CHUNK_SLOTS,
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
from flywheel_tpu.topk.kernels.sparse_core_topk import sparse_core_vmem_words

# Lanes of a union's row: the call's row of its first token, its tokens, and
# its first token's position; its tokens are consecutive in one request.
UNION_WORDS = 16
UNION_FIRST_TOKEN = 0
UNION_NUM_TOKENS = 1
UNION_FIRST_POSITION = 2
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


def worker_unions(num_unions: int) -> tuple[int, int]:
    """(unions a worker takes, words of its sizes row in whole DMA
    granules)."""
    sparse_core = pltpu.get_tpu_info().sparse_core
    unions_per_worker = pl.cdiv(
        num_unions, sparse_core.num_cores * sparse_core.num_subcores
    )
    return (
        unions_per_worker,
        pl.cdiv(unions_per_worker, NUM_SUBLANES) * NUM_SUBLANES,
    )


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
    unions_per_worker, sizes_words = worker_unions(num_unions)
    return (
        (unions_per_worker * UNION_WORDS,),
        (NUM_INDEX_BUFFERS, union_tokens, top_k),
        (NUM_TOKEN_BITMAPS, bitmap_rows(num_slabs), NUM_LANES),
        (pl.cdiv(max_seqlen_k, lanes) * lanes,),
        (union_slots,),
        # A lane vector past the last size, for its read-modify-write.
        (sizes_words + lanes,),
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
    unions_hbm: jax.Array
    union_hbm: jax.Array
    sizes_hbm: jax.Array
    bitmap_hbm: jax.Array
    union_row_vmem: jax.Array
    index_vmem: jax.Array
    bitmap_vmem: jax.Array
    owner_vmem: jax.Array
    union_vmem: jax.Array
    size_vmem: jax.Array
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
        first_union: jax.Array,
        top_k: int,
        num_slabs: int,
        num_lanes: int,
    ):
        self.refs = refs
        self.first_union = first_union
        self.unions_per_worker = refs.union_row_vmem.shape[0] // UNION_WORDS
        self.top_k = top_k
        self.union_slots = refs.union_vmem.shape[0]
        self.num_slabs = num_slabs
        self.rows_per_token = refs.bitmap_vmem.shape[1]
        self.lanes = num_lanes
        self.num_vectors = top_k // num_lanes
        self.lane = lax.broadcasted_iota(jnp.int32, (num_lanes,), 0)

    def union_row(self, local_union):
        """(first token, tokens, first position) of the worker's union."""
        row = self.refs.union_row_vmem[
            pl.ds(local_union * UNION_WORDS, self.lanes)
        ]
        return (
            row[UNION_FIRST_TOKEN],
            row[UNION_NUM_TOKENS],
            row[UNION_FIRST_POSITION],
        )

    def index_copies(self, local_union, *, is_start: bool) -> None:
        """Starts, or waits, the copies of the union's index rows."""
        first_token, num_tokens, _ = self.union_row(local_union)
        buffer = lax.rem(local_union, NUM_INDEX_BUFFERS)

        @pl.loop(0, num_tokens)
        def _token(token):
            copy = pltpu.make_async_copy(
                self.refs.indices_hbm.at[
                    pl.ds((first_token + token) * self.top_k, self.top_k)
                ],
                self.refs.index_vmem.at[buffer, token],
                self.refs.sems.at[SEM_INDEX, buffer],
            )
            if is_start:
                copy.start()
            else:
                copy.wait()

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
        union's size; returns the pages in flight."""
        refs = self.refs
        unused = jnp.full((self.lanes,), UNUSED_INDEX, jnp.int32)
        for slot in range(0, UNION_FILL_SLOTS, self.lanes):
            refs.union_vmem[pl.ds(size + slot, self.lanes)] = unused
        num_pages = (size + UNION_FILL_SLOTS + UNION_PAGE - 1) // UNION_PAGE

        @pl.loop(0, num_pages)
        def _page(page):
            self.union_copy(local_union, page).start()

        sizes = refs.size_vmem[pl.ds(local_union, self.lanes)]
        refs.size_vmem[pl.ds(local_union, self.lanes)] = jnp.where(
            self.lane == 0, size, sizes
        )
        return num_pages

    def wait_union(self, num_pages):
        @pl.loop(0, num_pages)
        def _page(page):
            self.union_copy(0, page).wait()

    def load_rows(self) -> jax.Array:
        """Loads the worker's union rows; returns how many of them it runs:
        through the last that holds tokens."""
        refs = self.refs
        pltpu.sync_copy(
            refs.unions_hbm.at[
                pl.ds(
                    self.first_union * UNION_WORDS,
                    refs.union_row_vmem.shape[0],
                )
            ],
            refs.union_row_vmem,
        )

        def _count(local_union, count):
            _, num_tokens, _ = self.union_row(local_union)
            return jnp.where(num_tokens > 0, local_union + 1, count)

        return lax.fori_loop(0, self.unions_per_worker, _count, jnp.int32(0))

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

    def run(self, num_local, sizes_row) -> None:
        """Unites the worker's first num_local unions."""
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
        pltpu.sync_copy(
            self.refs.size_vmem.at[pl.ds(0, sizes_row.shape[0])], sizes_row
        )


def union_kernel(
    *refs: jax.Array,
    unions_per_worker: int,
    top_k: int,
    num_slabs: int,
    num_subcores: int,
    num_lanes: int,
) -> None:
    """Each worker unites its run of unions_per_worker unions, through the
    last that holds tokens; one without tokens before it unites nothing."""
    refs = UnionRefs(*refs)
    worker = lax.axis_index("core") * num_subcores + lax.axis_index("subcore")
    union = UnionWorker(
        refs,
        first_union=worker * unions_per_worker,
        top_k=top_k,
        num_slabs=num_slabs,
        num_lanes=num_lanes,
    )
    num_local = union.load_rows()
    pl.when(num_local > 0)(
        functools.partial(union.run, num_local, refs.sizes_hbm.at[worker])
    )


def dsa_union_pallas(
    topk_indices: jax.Array,
    unions: jax.Array,
    *,
    union_tokens: int,
    union_slots: int,
    num_slabs: int,
    max_seqlen_k: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Each union's positions and its tokens' bitmaps on the SparseCores.

    Args:
      topk_indices: (T, top_k) int32 positions within each token's request,
        as dsa_sparse_mla takes them; top_k a multiple of the SparseCore's
        lanes. A token's used indices are distinct and below max_seqlen_k.
      unions: (U, UNION_WORDS) int32 union rows of at most union_tokens
        tokens each, in any order; every token in at most one union.
      union_tokens: the most tokens a union holds.
      union_slots: slots of a union's row in whole UNION_PAGE, at least
        union_tokens * top_k + UNION_FILL_SLOTS.
      num_slabs: slabs of SLAB_POSITIONS union indices a bitmap covers, at
        least union_tokens * top_k.
      max_seqlen_k: a static bound on the positions.

    Returns:
      (union, sizes, bitmap): (U * union_slots,) int32, union u's positions
      in slots [u * union_slots, + sizes[u]), then UNUSED_INDEX through
      UNION_FILL_SLOTS slots more, the slots after them unspecified; (U,)
      int32 union sizes; and (T * bitmap_rows(num_slabs), NUM_LANES) int32
      bitmaps laid out as dsa_selection_bitmap_pallas's, over the token's
      union. The outputs of a union without tokens, and a token outside the
      unions' rows, are unspecified.
    """
    num_tokens, top_k = topk_indices.shape
    num_unions = unions.shape[0]
    check_union_operands(
        num_unions, top_k, union_tokens, union_slots, num_slabs, max_seqlen_k
    )
    scratch_shapes = union_scratch_shapes(
        num_unions, union_tokens, top_k, union_slots, num_slabs, max_seqlen_k
    )
    sparse_core = pltpu.get_tpu_info().sparse_core
    num_workers = sparse_core.num_cores * sparse_core.num_subcores
    unions_per_worker, sizes_words = worker_unions(num_unions)
    unions = jnp.pad(
        unions, ((0, num_workers * unions_per_worker - num_unions), (0, 0))
    )
    body = functools.partial(
        union_kernel,
        unions_per_worker=unions_per_worker,
        top_k=top_k,
        num_slabs=num_slabs,
        num_subcores=sparse_core.num_subcores,
        num_lanes=sparse_core.num_lanes,
    )
    union, sizes, bitmap = pl.kernel(
        body,
        out_type=(
            jax.ShapeDtypeStruct((num_unions * union_slots,), jnp.int32),
            jax.ShapeDtypeStruct((num_workers, sizes_words), jnp.int32),
            jax.ShapeDtypeStruct(
                (num_tokens * bitmap_rows(num_slabs), NUM_LANES), jnp.int32
            ),
        ),
        mesh=sparse_core_mesh(),
        scratch_types=(
            *(pltpu.VMEM(shape, jnp.int32) for shape in scratch_shapes),
            pltpu.SemaphoreType.DMA(
                (NUM_UNION_SEMS, max(NUM_INDEX_BUFFERS, NUM_TOKEN_BITMAPS))
            ),
        ),
        compiler_params=gather_compiler_params(),
        name="dsa_union",
    )(topk_indices.reshape(-1), unions.reshape(-1))
    sizes = sizes[:, :unions_per_worker].reshape(-1)[:num_unions]
    return union, sizes, bitmap


def check_union_operands(
    num_unions: int,
    top_k: int,
    union_tokens: int,
    union_slots: int,
    num_slabs: int,
    max_seqlen_k: int,
) -> None:
    """Fails unless a union, its bitmaps and the scratch fit."""
    lanes = pltpu.get_tpu_info().sparse_core.num_lanes
    max_size = union_tokens * top_k
    needed_slots = max_size + UNION_FILL_SLOTS
    vmem_words = union_vmem_words(
        num_unions, union_tokens, top_k, union_slots, num_slabs, max_seqlen_k
    )
    if top_k % lanes:
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
