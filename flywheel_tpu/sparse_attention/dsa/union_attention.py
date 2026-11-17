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

"""DSA attention of given indices with one gather per union: the
UNION_TOKENS consecutive tokens of a request that form a union share one
SparseCore gather of the positions any of them selects, whose rows the
union attention attends, each token masked to the positions it selects.

The unions run in chunks that follow the call's token chunks: chunk c takes
the unions whose last token lies in its tokens. Each chunk is a union, a
gather of its positions and a union attention over the gathered rows; from
the indexer's scores, the chunk's top-k, union and gather run in one
SparseCore call.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl

from flywheel_tpu.sparse_attention.dsa.kernels.gather import NOPE_WORDS
from flywheel_tpu.sparse_attention.dsa.kernels.union import (
    CHUNK_BLOCK_WORD,
    COUNT_UNIONS,
    UNION_FILL_SLOTS,
    UNION_FIRST_POSITION,
    UNION_FIRST_TOKEN,
    UNION_INDEX_ROW,
    UNION_NUM_TOKENS,
    UNION_PREVIOUS_ROW,
    UNION_TABLE_OFFSET,
    UNION_WORDS,
    UnionShape,
    dsa_topk_union_gather_pallas,
    dsa_union_pallas,
    num_union_workers,
    padded_union_slots,
    union_vmem_words,
)
from flywheel_tpu.sparse_attention.dsa.kernels.union_attention import (
    UNION_BLOCK_KV,
    dsa_union_attention_pallas,
)
from flywheel_tpu.sparse_attention.dsa.masked_dense import bitmap_slabs
from flywheel_tpu.sparse_attention.dsa.routing import (
    cumulative_lens,
    request_q_lens,
)
from flywheel_tpu.sparse_attention.dsa.sparse_mla import (
    SplitCache,
    gather_chunk_rows,
    lane_rows,
    split_cache,
)
from flywheel_tpu.topk import topk
from flywheel_tpu.topk.kernels.sparse_core_topk import sparse_core_vmem_words

UNION_TOKENS = 8
# Note (david): a window of tokens cuts at most two requests, one at each
# end; the one its start cuts adds at most one union.
EDGE_REQUESTS = 2
EDGE_UNIONS = 1
# Note (david): a chunk's kernels lower once per shape, however many chunks
# of a call run them.
unite_chunk = jax.jit(dsa_union_pallas, static_argnames=("shape",))
topk_union_gather_chunk = jax.jit(
    dsa_topk_union_gather_pallas,
    static_argnames=("geometry", "shape", "page_size", "pages_per_seq"),
)


def after[Value](value: Value, anchor: object) -> Value:
    """value, which XLA may use only once anchor exists. A SparseCore call
    with a pinned operand launches after anchor; a SparseCore call's output
    in value or anchor waits for the call there."""
    return lax.optimization_barrier((value, anchor))[0]


class TokenUnions(NamedTuple):
    """(U,) int32 union columns: first token, tokens, first token's
    position and request. Unions past the call's hold no tokens and start
    at cu_q_lens[B]."""

    first_token: jax.Array
    num_tokens: jax.Array
    first_position: jax.Array
    request: jax.Array


def max_unions(num_tokens: int, batch: int, min_q_len: int) -> int:
    """The most unions num_tokens tokens make in requests of at least
    min_q_len tokens out of batch: each request's last union may be
    partial."""
    num_partial = min(batch, num_tokens // min_q_len) * (UNION_TOKENS - 1)
    return max(1, (num_tokens + num_partial) // UNION_TOKENS)


def token_unions(
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    is_union: jax.Array,
    num_unions: int,
) -> TokenUnions:
    """The tokens of the requests is_union marks in unions of UNION_TOKENS
    from the request's first, the last maybe partial; num_unions unions,
    the ones that hold tokens first, in token order."""
    batch = kv_lens.shape[0]
    q_lens = request_q_lens(cu_q_lens)
    union_starts = cumulative_lens(
        jnp.where(is_union, pl.cdiv(q_lens, UNION_TOKENS), 0)
    )
    union = jnp.arange(num_unions, dtype=jnp.int32)
    request = jnp.minimum(
        jnp.searchsorted(
            union_starts[1:], union, side="right", method="compare_all"
        ),
        batch - 1,
    ).astype(jnp.int32)
    # Note (david): one gather of the requests' columns costs what a gather of
    # one column does.
    q_start, q_end, first_position, union_start = jnp.stack(
        [cu_q_lens[:-1], cu_q_lens[1:], kv_lens - q_lens, union_starts[:-1]],
        axis=1,
    )[request].T
    token_in_request = (union - union_start) * UNION_TOKENS
    first_token = q_start + token_in_request
    is_real = union < union_starts[batch]
    return TokenUnions(
        first_token=jnp.where(is_real, first_token, cu_q_lens[batch]),
        num_tokens=jnp.where(
            is_real, jnp.minimum(UNION_TOKENS, q_end - first_token), 0
        ),
        first_position=first_position + token_in_request,
        request=request,
    )


def union_slots(top_k: int) -> int:
    """Slots of a union's row: its largest size and the fill after it, in
    whole kv blocks of the union attention."""
    return pl.cdiv(UNION_TOKENS * top_k + UNION_FILL_SLOTS, UNION_BLOCK_KV) * (
        UNION_BLOCK_KV
    )


def union_fits(num_unions: int, top_k: int, max_seqlen_k: int) -> bool:
    """Whether the union kernel's scratch for num_unions unions fits a
    SparseCore subcore's VMEM at max_seqlen_k."""
    vmem_words = union_vmem_words(
        num_unions,
        UNION_TOKENS,
        top_k,
        union_slots(top_k),
        bitmap_slabs(UNION_TOKENS * top_k),
        max_seqlen_k,
    )
    return vmem_words <= sparse_core_vmem_words()


def chunk_union_slots(
    window_tokens: int, batch: int, min_q_len: int, num_unions: int
) -> int:
    """Slots of a union chunk: the most unions whose last token lies in a
    window of window_tokens tokens of requests of at least min_q_len tokens
    out of batch, at most the call's num_unions."""
    num_requests = min(batch, window_tokens // min_q_len + EDGE_REQUESTS)
    return min(
        num_unions,
        (window_tokens + (UNION_TOKENS - 1) * num_requests) // UNION_TOKENS
        + EDGE_UNIONS,
    )


def window_unions(
    unions: TokenUnions, windows: list[tuple[int, int]], num_slots: int
) -> tuple[TokenUnions, jax.Array]:
    """(C, num_slots) columns of each window's unions, the ones whose last
    token lies in its [first, end) tokens, in its first slots, the other
    slots without tokens; and each window's (C,) count of them."""
    first, end = (
        jnp.asarray(bound, jnp.int32)[:, None] for bound in zip(*windows)
    )
    last_token = unions.first_token + unions.num_tokens - 1
    is_window = (
        (unions.num_tokens > 0) & (last_token >= first) & (last_token < end)
    )
    # Note (david): a union's rank among its window's unions counts the ones
    # up to it, so the window's union u is the first whose rank passes u.
    rank = jnp.cumsum(is_window.astype(jnp.int32), axis=1)
    slot = jnp.arange(num_slots, dtype=jnp.int32)
    union = jnp.minimum(
        jnp.sum(rank[:, None, :] <= slot[None, :, None], axis=-1),
        rank.shape[1] - 1,
    )
    num_unions = rank[:, -1]
    first_token, num_tokens, first_position, request = jnp.moveaxis(
        jnp.stack(unions, axis=-1)[union], -1, 0
    )
    window = TokenUnions(
        first_token=first_token,
        num_tokens=jnp.where(slot < num_unions[:, None], num_tokens, 0),
        first_position=first_position,
        request=request,
    )
    return window, num_unions


def union_table(
    windows: TokenUnions,
    num_unions: jax.Array,
    index_starts: list[int],
    pages_per_seq: int,
) -> jax.Array:
    """(C * (1 + P) * UNION_WORDS,) int32 union table of the windows'
    unions, P padded_union_slots of their slots: chunk c's block holds its
    count row, then its union rows, whose index rows count from the token
    index_starts[c]."""
    num_chunks, num_slots = windows.first_token.shape
    index_first = jnp.asarray(index_starts, jnp.int32)[:, None]
    previous_first = jnp.asarray(
        index_starts[:1] + index_starts[:-1], jnp.int32
    )[:, None]
    columns = {
        UNION_FIRST_TOKEN: windows.first_token,
        UNION_NUM_TOKENS: windows.num_tokens,
        UNION_FIRST_POSITION: windows.first_position,
        UNION_INDEX_ROW: windows.first_token - index_first,
        UNION_PREVIOUS_ROW: windows.first_token - previous_first,
        UNION_TABLE_OFFSET: windows.request * pages_per_seq,
    }
    rows = lane_rows(
        {lane: column.reshape(-1) for lane, column in columns.items()},
        UNION_WORDS,
    ).reshape(num_chunks, num_slots, UNION_WORDS)
    padding = padded_union_slots(num_slots) - num_slots
    count_rows = lane_rows({COUNT_UNIONS: num_unions}, UNION_WORDS)
    return jnp.concatenate(
        [count_rows[:, None], jnp.pad(rows, ((0, 0), (0, padding), (0, 0)))],
        axis=1,
    ).reshape(-1)


class UnionSelection(NamedTuple):
    """A union chunk's union kernel call: the union table it was pinned
    with, its positions, the gather's metadata rows and chunk row, the
    tokens' bitmaps and the union attention's schedule."""

    table: jax.Array
    positions: jax.Array
    meta: jax.Array
    bitmap: jax.Array
    schedule: jax.Array
    gather_row: jax.Array


class GatheredRows(NamedTuple):
    """A union chunk's gather: the pass's flat block table it was pinned
    with, and the (nope, rope) rows of its positions, slot s's from row
    s * union_slots, as the cache holds them."""

    block_table: jax.Array
    nope_rows: jax.Array
    rope_rows: jax.Array


class SelectedChunk(NamedTuple):
    """A union chunk's top-k, union and gather call: the (T_c, top_k) top-k
    of its tokens, its union and its gathered rows."""

    indices: jax.Array
    selection: UnionSelection
    gathered: GatheredRows


class UnionPass(NamedTuple):
    """The operands every union chunk of a call shares: the queries, the
    gather's view of the cache and the union table of the num_chunks chunks
    of num_slots slots. Each kernel reads its chunk's part itself."""

    q_nope: jax.Array
    q_pe: jax.Array
    cache: SplitCache
    table: jax.Array
    num_chunks: int
    num_slots: int
    top_k: int
    sm_scale: float
    kv_scale: jax.Array
    max_seqlen_k: int
    force_replay: bool

    @property
    def num_slabs(self) -> int:
        return bitmap_slabs(UNION_TOKENS * self.top_k)

    @property
    def union_shape(self) -> UnionShape:
        return UnionShape(
            num_tokens=self.q_nope.shape[0],
            union_tokens=UNION_TOKENS,
            union_slots=self.cache.top_k,
            num_slabs=self.num_slabs,
            max_seqlen_k=self.max_seqlen_k,
            num_slots=self.num_slots,
        )

    def queries(self, anchor: object) -> tuple[jax.Array, jax.Array]:
        """(q_nope, q_pe heads minor) as the union attention reads them,
        computed after anchor if one is given."""
        q_nope, q_pe = self.q_nope, self.q_pe
        if anchor is not None:
            q_nope, q_pe = after((q_nope, q_pe), anchor)
        return q_nope, jnp.swapaxes(q_pe, 1, 2)

    def chunk_row(self, chunk: int) -> jax.Array:
        """The union kernel's (UNION_WORDS,) row naming chunk's block of the
        table, a constant."""
        row = np.zeros(UNION_WORDS, np.int32)
        block_words = (1 + padded_union_slots(self.num_slots)) * UNION_WORDS
        row[CHUNK_BLOCK_WORD] = chunk * block_words
        return jnp.asarray(row)

    def select(
        self,
        chunk: int,
        indices: jax.Array,
        previous_indices: jax.Array,
        anchor: object,
    ) -> UnionSelection:
        """The union of chunk's unions on the SparseCores, launched after
        anchor if one is given; indices and previous_indices hold the rows
        the chunk's union rows name."""
        table = self.table
        if anchor is not None:
            table = after(table, anchor)
        outputs = unite_chunk(
            indices,
            previous_indices,
            table,
            self.chunk_row(chunk),
            shape=self.union_shape,
        )
        return UnionSelection(table, *outputs)

    def select_and_gather(
        self,
        chunk: int,
        scores: jax.Array,
        row_lengths: jax.Array,
        previous_indices: jax.Array | None,
        anchor: object,
    ) -> SelectedChunk:
        """The top-k of chunk's tokens from their flat scores and
        row_lengths, the union of chunk's unions from it and the gather of
        their positions in one SparseCore call, launched after anchor if one
        is given. previous_indices: the chunk before's top-k, which its
        union rows name before the chunk's first token; None for the first
        chunk, whose unions start in its tokens."""
        table = self.table
        if anchor is not None:
            table = after(table, anchor)
        num_tokens = scores.shape[0]
        if previous_indices is None:
            previous_indices = lax.empty((num_tokens, self.top_k), jnp.int32)
        # Note (david): a top-k whose subcores share a row, of fewer rows than
        # workers, hangs the gather that reuses its VMEM, so the call's top-k
        # selects a row or more a worker: empty rows after the chunk's.
        num_padding = max(num_union_workers() - num_tokens, 0)
        scores = jnp.pad(scores, ((0, num_padding), (0, 0), (0, 0)))
        row_lengths = jnp.pad(row_lengths, (0, num_padding))
        geometry = topk.flat_topk_geometry(
            scores, row_lengths, self.top_k, sparse_core_vmem_words()
        )
        cache = self.cache
        indices, *union_outputs, nope_rows, rope_rows, _ = (
            topk_union_gather_chunk(
                row_lengths,
                scores,
                previous_indices,
                table,
                self.chunk_row(chunk),
                cache.block_table,
                cache.nope_cache,
                cache.rope_cache,
                geometry=geometry,
                shape=self.union_shape,
                page_size=cache.page_size,
                pages_per_seq=cache.pages_per_seq,
            )
        )
        return SelectedChunk(
            indices[:num_tokens],
            UnionSelection(table, *union_outputs),
            GatheredRows(cache.block_table, nope_rows, rope_rows),
        )

    def gather(self, selection: UnionSelection, anchor: object) -> GatheredRows:
        """The gather of the selection's positions on the SparseCores, the
        slots past its unions skipped, launched after anchor if one is
        given."""
        # Note (david): the pass's flat block table is its own, so its pin
        # leaves the call's block table to the other routes.
        cache = self.cache
        if anchor is not None:
            cache = cache._replace(block_table=after(cache.block_table, anchor))
        nope_rows, rope_rows, _ = gather_chunk_rows(
            selection.gather_row,
            selection.positions,
            selection.meta,
            num_tokens=self.num_slots,
            writes_cache_rows=True,
            **cache._asdict(),
        )
        return GatheredRows(cache.block_table, nope_rows, rope_rows)

    def attend(
        self,
        selection: UnionSelection,
        gathered: GatheredRows,
        queries: tuple[jax.Array, jax.Array],
        out: jax.Array,
        anchor: object,
    ) -> jax.Array:
        """out with the rows of the selection's tokens written in place: the
        union attention over the gathered rows, after anchor if one is
        given."""
        if anchor is not None:
            out = after(out, anchor)
        return dsa_union_attention_pallas(
            *queries,
            gathered.nope_rows,
            gathered.rope_rows.reshape(-1, NOPE_WORDS),
            selection.bitmap,
            selection.schedule,
            self.kv_scale,
            union_tokens=UNION_TOKENS,
            num_unions=self.num_slots,
            sm_scale=self.sm_scale,
            num_slabs=self.num_slabs,
            previous_out=out,
            force_replay=self.force_replay,
        )


def union_pass(
    q_nope: jax.Array,
    q_pe: jax.Array,
    nope_cache: jax.Array,
    rope_cache: jax.Array,
    block_table: jax.Array,
    unions: TokenUnions,
    starts: list[int],
    chunk_tokens: int,
    *,
    index_starts: list[int],
    num_slots: int,
    top_k: int,
    sm_scale: float,
    kv_scale: jax.Array,
    max_seqlen_k: int,
    force_replay: bool = False,
) -> UnionPass:
    """The union chunks of the token chunks of chunk_tokens from starts:
    chunk c's window runs from chunk c - 1's end to its own, and its index
    rows count from the token index_starts[c]. force_replay: the union
    attention replays every union."""
    ends = [first + chunk_tokens for first in starts]
    windows, num_unions = window_unions(
        unions, list(zip([0] + ends[:-1], ends)), num_slots
    )
    cache = split_cache(nope_cache, rope_cache, block_table, union_slots(top_k))
    return UnionPass(
        q_nope=q_nope,
        q_pe=q_pe,
        cache=cache,
        table=union_table(
            windows, num_unions, index_starts, cache.pages_per_seq
        ),
        num_chunks=len(starts),
        num_slots=num_slots,
        top_k=top_k,
        sm_scale=sm_scale,
        kv_scale=jnp.asarray(kv_scale, jnp.float32).reshape(1),
        max_seqlen_k=max_seqlen_k,
        force_replay=force_replay,
    )


def attend_given(
    unions: UnionPass, indices: jax.Array, out: jax.Array
) -> jax.Array:
    """out with the unions' rows written in place, from the (T, top_k)
    indices of every token. The queries' layout copies run under gather 0;
    gather c + 1 and union c + 2 launch before attention c and run under
    it."""
    num_chunks = unions.num_chunks
    selections, gathered = [None] * num_chunks, [None] * num_chunks

    def unite(chunk, anchor):
        selections[chunk] = unions.select(chunk, indices, indices, anchor)

    def gather(chunk, anchor):
        gathered[chunk] = unions.gather(selections[chunk], anchor)

    unite(0, None)
    if num_chunks > 1:
        unite(1, None)
    gather(0, selections[0].meta)
    step = queries = unions.queries(gathered[0].block_table)
    for chunk in range(num_chunks):
        launched = []
        if chunk + 1 < num_chunks:
            gather(chunk + 1, step)
            launched += [
                selections[chunk + 1].meta,
                gathered[chunk + 1].block_table,
            ]
        if chunk + 2 < num_chunks:
            unite(chunk + 2, step)
            launched.append(selections[chunk + 2].table)
        out = unions.attend(
            selections[chunk],
            gathered[chunk],
            queries,
            out,
            (step, launched),
        )
        step = out
    return out
