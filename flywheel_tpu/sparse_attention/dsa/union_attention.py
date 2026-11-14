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
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from flywheel_tpu.sparse_attention.dsa.kernels.gather import (
    META_POSITION,
    META_SLOTS,
    META_TABLE_OFFSET,
    META_WORDS,
    NOPE_WORDS,
)
from flywheel_tpu.sparse_attention.dsa.kernels.union import (
    UNION_FILL_SLOTS,
    UNION_FIRST_POSITION,
    UNION_FIRST_TOKEN,
    UNION_NUM_TOKENS,
    UNION_WORDS,
    dsa_union_pallas,
    union_vmem_words,
)
from flywheel_tpu.sparse_attention.dsa.kernels.union_attention import (
    PERIOD_KEYS,
    UNION_BLOCK_KV,
    dsa_union_attention_pallas,
    union_schedule,
)
from flywheel_tpu.sparse_attention.dsa.masked_dense import bitmap_slabs
from flywheel_tpu.sparse_attention.dsa.routing import (
    cumulative_lens,
    request_q_lens,
)
from flywheel_tpu.sparse_attention.dsa.sparse_mla import (
    TARGET_CHUNKS,
    chunk_row,
    chunk_starts,
    gather_chunk_rows,
    lane_rows,
    split_cache,
)
from flywheel_tpu.topk.kernels.sparse_core_topk import sparse_core_vmem_words

UNION_TOKENS = 8
# Note (david): a chunk's gathered rows take its unions' largest size of HBM,
# which MAX_CHUNK_UNIONS bounds.
MAX_CHUNK_UNIONS = 64


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
    the ones that hold tokens first."""
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


class UnionSelection(NamedTuple):
    """A call's unions, their positions (each union's row of union_slots),
    sizes, and each token's membership bitmap over its union's positions; a
    pure function of the indices, cu_q_lens and kv_lens."""

    unions: TokenUnions
    positions: jax.Array
    sizes: jax.Array
    bitmap: jax.Array


def union_selection(
    topk_indices: jax.Array, unions: TokenUnions, *, max_seqlen_k: int
) -> UnionSelection:
    """The UnionSelection of checked operands, on the SparseCores."""
    top_k = topk_indices.shape[1]
    union_rows = lane_rows(
        {
            UNION_FIRST_TOKEN: unions.first_token,
            UNION_NUM_TOKENS: unions.num_tokens,
            UNION_FIRST_POSITION: unions.first_position,
        },
        UNION_WORDS,
    )
    positions, sizes, bitmap = dsa_union_pallas(
        topk_indices,
        union_rows,
        union_tokens=UNION_TOKENS,
        union_slots=union_slots(top_k),
        num_slabs=bitmap_slabs(UNION_TOKENS * top_k),
        max_seqlen_k=max_seqlen_k,
    )
    return UnionSelection(unions, positions, sizes, bitmap)


def chunk_schedules(
    selection: UnionSelection,
    starts: list[int],
    num_unions: int,
    union_rows: int,
    num_real: jax.Array,
) -> jax.Array:
    """The union attention's schedule of each chunk of unions [first,
    first + num_unions), first in starts, a row a chunk: a chunk's union j
    reads the gathered rows from j * union_rows. num_real is the token past
    which a call without previous_out zeros the rows."""
    union = jnp.asarray(starts, jnp.int32)[:, None] + jnp.arange(num_unions)
    unions = selection.unions
    columns = jnp.stack(
        [unions.first_token, unions.num_tokens, selection.sizes], axis=1
    )[union]
    first_rows = jnp.broadcast_to(
        jnp.arange(num_unions, dtype=jnp.int32)[:, None] * union_rows,
        (len(starts), num_unions, 1),
    )
    return union_schedule(
        jnp.concatenate([columns, first_rows], axis=-1),
        num_real,
        PERIOD_KEYS,
    )


def union_token_meta(
    selection: UnionSelection, pages_per_seq: int
) -> jax.Array:
    """The gather's (U, META_WORDS) rows of the unions: a union gathers its
    positions as a token at its last token's position."""
    unions = selection.unions
    has_tokens = unions.num_tokens > 0
    return lane_rows(
        {
            META_TABLE_OFFSET: unions.request * pages_per_seq,
            META_POSITION: jnp.where(
                has_tokens, unions.first_position + unions.num_tokens - 1, -1
            ),
            META_SLOTS: jnp.where(has_tokens, selection.sizes, 0),
        },
        META_WORDS,
    )


def union_attention(
    q_nope: jax.Array,
    q_pe: jax.Array,
    nope_cache: jax.Array,
    rope_cache: jax.Array,
    selection: UnionSelection,
    block_table: jax.Array,
    previous_out: jax.Array,
    *,
    top_k: int,
    sm_scale: float,
    kv_scale: jax.Array,
    force_replay: bool = False,
) -> jax.Array:
    """dsa_sparse_mla's out rows of the unions' tokens from their union
    selection of top_k indices, written into previous_out in place: chunk
    after chunk of unions, the gather of each union's positions, then the
    union attention of the chunk's unions over their gathered rows. A
    chunk's gather depends on nothing of the chunk before's attention, so
    the SparseCore runs it under that attention. force_replay: the union
    attention replays every union, as it does one whose fixed-anchor softmax
    overflows."""
    num_unions = selection.sizes.shape[0]
    unions_per_chunk = min(MAX_CHUNK_UNIONS, pl.cdiv(num_unions, TARGET_CHUNKS))
    starts = chunk_starts(num_unions, unions_per_chunk)
    cache = split_cache(nope_cache, rope_cache, block_table, union_slots(top_k))
    token_meta = union_token_meta(selection, cache.pages_per_seq)
    schedules = chunk_schedules(
        selection, starts, unions_per_chunk, cache.top_k, q_nope.shape[0]
    )
    # Note (david): the kernel reads q_pe with the heads minor, as XLA lays
    # (T, 128, ROPE_DIM) out, so the swap is a bitcast.
    q_pe_t = jnp.swapaxes(q_pe, 1, 2)
    kv_scale = jnp.asarray(kv_scale, jnp.float32).reshape(1)
    out = previous_out
    for chunk, first in enumerate(starts):
        nope_rows, rope_rows, _ = gather_chunk_rows(
            chunk_row(first, first),
            selection.positions,
            token_meta,
            num_tokens=unions_per_chunk,
            writes_cache_rows=True,
            **cache._asdict(),
        )
        out = dsa_union_attention_pallas(
            q_nope,
            q_pe_t,
            nope_rows,
            rope_rows.reshape(-1, NOPE_WORDS),
            selection.bitmap,
            schedules[chunk],
            kv_scale,
            union_tokens=UNION_TOKENS,
            sm_scale=sm_scale,
            num_slabs=bitmap_slabs(UNION_TOKENS * top_k),
            previous_out=out,
            force_replay=force_replay,
        )
    return out
