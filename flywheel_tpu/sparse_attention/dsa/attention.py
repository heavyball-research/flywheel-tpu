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

"""DeepSeek sparse attention (DSA) of one layer: the lightning indexer's
scores, their top-k, and the attention over the selected latents, each
request in the mode routing.py gives it.

Tokens run in chunks: the SparseCore selects and gathers one chunk while the
TensorCore scores the next, and in a step with union requests it selects,
unites and gathers one chunk while the TensorCore scores or attends another.
dsa_indices and dsa_attention_from_indices split a layer into selection and
attention for callers that run them apart.
"""

import collections
import functools
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from flywheel_tpu.core.tpu import NUM_LANES
from flywheel_tpu.sparse_attention.dsa.cache_write import (
    check_requests,
    token_positions,
)
from flywheel_tpu.sparse_attention.dsa.indexer import (
    dsa_indexer_scores,
    score_columns,
)
from flywheel_tpu.sparse_attention.dsa.kernels.gather import (
    CHUNK_FIRST_TOKEN,
    CHUNK_SLOTS,
    CHUNK_WORDS,
    GATHERED_WORDS_PER_SLOT,
    NOPE_WORDS,
    UNUSED_INDEX,
    fused_topk_vmem_words,
    gather_slots,
)
from flywheel_tpu.sparse_attention.dsa.kernels.indexer import BLOCK_TOKENS
from flywheel_tpu.sparse_attention.dsa.kernels.masked_dense import (
    masked_dense_plan,
)
from flywheel_tpu.sparse_attention.dsa.kernels.sparse_mla import (
    tokens_per_step,
)
from flywheel_tpu.sparse_attention.dsa.masked_dense import (
    SUPPORTED_HEADS,
    masked_dense_attention,
    masked_dense_config_of,
    selection_bitmap,
)
from flywheel_tpu.sparse_attention.dsa.routing import (
    MAX_MASKED_DENSE_KV_LEN,
    MIN_MASKED_DENSE_Q_LEN,
    MIN_UNION_Q_LEN,
    MODE_MASKED_DENSE,
    MODE_SPARSE,
    MODE_UNION,
    causal_indices,
    cumulative_lens,
    request_modes,
    request_q_lens,
)
from flywheel_tpu.sparse_attention.dsa.sparse_mla import (
    ChunkAttention,
    attend_in_chunks,
    check_chunk_tokens,
    check_indices,
    check_operands,
    chunk_attention,
    chunk_outputs,
    chunk_row,
    chunk_starts,
    default_chunk_tokens as sparse_mla_chunk_tokens,
    dsa_sparse_mla,
    slot_indices,
)
from flywheel_tpu.sparse_attention.dsa.union_attention import (
    UnionPass,
    after,
    attend_given,
    chunk_union_slots,
    max_unions,
    token_unions,
    union_fits,
    union_pass,
)
from flywheel_tpu.topk import topk
from flywheel_tpu.topk.kernels.sparse_core_topk import WORD_BYTES

# Positions a token attends.
DSA_TOPK = 2048

# Note (david): chunks of about PIPELINE_CHUNK_TOKENS keep the pipeline's fill
# and drain short; each chunk adds a call of each kernel.
PIPELINE_CHUNK_TOKENS = 256
MIN_PIPELINE_CHUNKS = 4
MAX_PIPELINE_CHUNKS = 8
MIN_CHUNK_TOKENS = 128
# Note (david): chunk c attends after chunk c + ATTEND_LAG's scores, so the
# SparseCore runs its queued calls back to back; scores issued after an
# attention would hold the next call back until that attention ends.
ATTEND_LAG = 3
# Note (david): ATTEND_LAG + 1 chunks of rows and scores are live at once;
# together they fit HBM >> PIPELINE_HBM_SHIFT, each under a SparseCore limit.
PIPELINE_HBM_SHIFT = 1
# Note (david): in a step with union requests, chunk c attends after chunk
# c + UNION_ATTEND_LAG's scores, so its union and gather each run under a
# TensorCore step.
UNION_ATTEND_LAG = 2
MAX_SPARSE_CORE_TENSOR_BYTES = 16 << 30
ROUTE_SPARSE = "sparse"
ROUTE_MASKED_DENSE = "masked_dense"
ROUTE_UNION = "union"
ROUTES = (ROUTE_SPARSE, ROUTE_MASKED_DENSE, ROUTE_UNION)
# Note (david): the heads the union mode was measured at; its masked-dense
# items grow with the heads.
UNION_HEADS = (64,)
# The fewest tokens of a request on the pinned union route.
PINNED_UNION_Q_LEN = 1
# Note (david): a call under MIN_ROUTED_TOKENS tokens (decode) compiles the
# sparse MLA alone: routing on device costs it more than the dense modes save.
MIN_ROUTED_TOKENS = 512
# Note (david): the routed sparse pass runs the chunks of its first
# SPARSE_HEAD_TOKENS tokens in the call's scope, each gather under the dense
# passes or the chunk before's attention; a chunk its extent does not reach
# still costs a call of each kernel, so the chunks after them run in a loop,
# one at a time.
SPARSE_HEAD_TOKENS = 256


def max_chunk_tokens(top_k: int, max_seqlen_k: int) -> int:
    """Tokens of the largest chunk, in whole q blocks of the indexer, whose
    live gathered rows and scores fit the pipeline's share of the HBM, and
    whose gathered nope rows and scores each fit a SparseCore tensor."""
    num_slots = gather_slots(top_k)
    nope_bytes = WORD_BYTES * num_slots * NOPE_WORDS
    score_bytes = WORD_BYTES * score_columns(max_seqlen_k)
    token_bytes = WORD_BYTES * num_slots * GATHERED_WORDS_PER_SLOT + score_bytes
    live_bytes = (ATTEND_LAG + 1) * token_bytes
    hbm_bytes = pltpu.get_tpu_info().hbm_capacity_bytes >> PIPELINE_HBM_SHIFT
    max_tokens = min(
        hbm_bytes // live_bytes,
        MAX_SPARSE_CORE_TENSOR_BYTES // max(nope_bytes, score_bytes),
    )
    return max(BLOCK_TOKENS, max_tokens // BLOCK_TOKENS * BLOCK_TOKENS)


def default_chunk_tokens(num_tokens: int, max_tokens: int) -> int:
    """Tokens per chunk of a call of num_tokens: whole q blocks that split the
    tokens evenly near PIPELINE_CHUNK_TOKENS, within max_tokens."""
    num_chunks = min(
        MAX_PIPELINE_CHUNKS,
        max(MIN_PIPELINE_CHUNKS, pl.cdiv(num_tokens, PIPELINE_CHUNK_TOKENS)),
    )
    num_chunks = max(num_chunks, pl.cdiv(num_tokens, max_tokens))
    target = max(MIN_CHUNK_TOKENS, pl.cdiv(num_tokens, num_chunks))
    rounded = pl.cdiv(target, BLOCK_TOKENS) * BLOCK_TOKENS
    return min(num_tokens, max_tokens, rounded)


def chunk_lengths(
    cu_q_lens: jax.Array, kv_lens: jax.Array, first: int, num_tokens: int
) -> tuple[jax.Array, jax.Array]:
    """cu_q_lens and kv_lens of the call's tokens [first, first + num_tokens)
    as a call of their own: a request keeps its tokens inside them, and its
    kv_len drops by its tokens past them, so that each token sees the same
    positions."""
    q_ends = cu_q_lens[1:]
    past = jnp.clip(q_ends - (first + num_tokens), 0, q_ends - cu_q_lens[:-1])
    return jnp.clip(cu_q_lens - first, 0, num_tokens), kv_lens - past


class IndexerChunks(NamedTuple):
    """The indexer's and the top-k's operands of a call's chunks."""

    queries: jax.Array
    weights: jax.Array
    index_cache: jax.Array
    cu_q_lens: jax.Array
    kv_lens: jax.Array
    block_table: jax.Array
    row_lengths: jax.Array
    max_seqlen_k: int
    top_k: int

    def scores(
        self, first: int, num_tokens: int, anchor: jax.Array | None
    ) -> jax.Array:
        """Index scores of the call's tokens [first, first + num_tokens),
        computed after anchor if one is given."""
        tokens = slice(first, first + num_tokens)
        queries = self.queries[tokens]
        return dsa_indexer_scores(
            queries if anchor is None else after(queries, anchor),
            self.weights[tokens],
            self.index_cache,
            *chunk_lengths(self.cu_q_lens, self.kv_lens, first, num_tokens),
            self.block_table,
            max_seqlen_k=self.max_seqlen_k,
        )

    def select(
        self, first: int, num_tokens: int, scores: jax.Array
    ) -> jax.Array:
        """The (num_tokens, top_k) top-k of the call's tokens [first, first +
        num_tokens) from their scores."""
        return topk.top_k(
            scores,
            self.top_k,
            row_lengths=self.row_lengths[first : first + num_tokens],
        )


class Gathered(NamedTuple):
    """A chunk's first token, token count and gathered rows."""

    first: int
    num_tokens: int
    rows: tuple[jax.Array, jax.Array, jax.Array]


def assemble_indices(
    chunk_indices: list[tuple[int, jax.Array]], num_tokens: int
) -> jax.Array:
    """The call's first num_tokens (T, top_k) indices from each chunk's
    (first token, indices): the last chunk ends at the call's end, and its
    tokens that the chunk before it holds keep that chunk's rows."""
    pieces = []
    covered = 0
    for first, indices in chunk_indices:
        pieces.append(indices[covered - first :])
        covered = first + indices.shape[0]
    return jnp.concatenate(pieces)[:num_tokens]


class SplitChunk(NamedTuple):
    """A ChunkAttention as its array leaves and the hashable rest, so that a
    jit of the chunk's kernels traces and lowers once per chunk size."""

    arrays: list[jax.Array]
    layout: tuple

    @classmethod
    def of(cls, chunk: ChunkAttention) -> "SplitChunk":
        leaves, treedef = jax.tree.flatten(chunk)
        arrays = [leaf for leaf in leaves if isinstance(leaf, jax.Array)]
        statics = tuple(
            None if isinstance(leaf, jax.Array) else leaf for leaf in leaves
        )
        return cls(arrays, (treedef, statics))


def joined(arrays: list[jax.Array], layout: tuple) -> ChunkAttention:
    """The ChunkAttention that SplitChunk.of split into arrays and layout."""
    treedef, statics = layout
    remaining = iter(arrays)
    leaves = [next(remaining) if leaf is None else leaf for leaf in statics]
    return jax.tree.unflatten(treedef, leaves)


@functools.partial(jax.jit, static_argnames=("layout", "num_tokens", "top_k"))
def select_and_gather(
    arrays: list[jax.Array],
    row: jax.Array,
    row_lengths: jax.Array,
    scores: jax.Array,
    *,
    layout: tuple,
    num_tokens: int,
    top_k: int,
) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
    """select_and_gather_rows of a split chunk's num_tokens tokens, top_k
    of each."""
    chunk = joined(arrays, layout)
    geometry = topk.flat_topk_geometry(
        scores, row_lengths, top_k, fused_topk_vmem_words()
    )
    return chunk.select_and_gather_rows(
        row, num_tokens, row_lengths, scores, geometry
    )


@functools.partial(jax.jit, static_argnames=("layout", "num_tokens"))
def attend_gathered(
    arrays: list[jax.Array],
    rows: tuple[jax.Array, jax.Array, jax.Array],
    previous: tuple[jax.Array, ...] | None,
    first_token: jax.Array,
    scalars: jax.Array,
    *,
    layout: tuple,
    num_tokens: int,
) -> tuple[jax.Array, ...]:
    """attend_rows of a split chunk's num_tokens tokens from first_token,
    traced, so that chunks of one size share one compile."""
    return joined(arrays, layout).attend_rows(
        rows,
        previous,
        first_token=first_token,
        num_tokens=num_tokens,
        scalars=scalars,
    )


def attend_chunk(
    chunk: ChunkAttention,
    gathered: Gathered,
    previous: tuple[jax.Array, ...] | None,
    anchor: jax.Array | None,
) -> tuple[jax.Array, ...]:
    """The attention of a gathered chunk, run after anchor if one is
    given."""
    split = SplitChunk.of(chunk)
    scalars = chunk.scalars
    return attend_gathered(
        split.arrays,
        gathered.rows,
        previous,
        jnp.int32(gathered.first),
        scalars if anchor is None else after(scalars, anchor),
        layout=split.layout,
        num_tokens=gathered.num_tokens,
    )


def run_chunks(
    indexer: IndexerChunks, chunk: ChunkAttention, chunk_tokens: int
) -> tuple[tuple[jax.Array, ...], jax.Array]:
    """Score, select, gather and attend every chunk in the pinned order.
    Returns the attention's (out, [lse]) and the (T_pad, top_k) indices.
    The TensorCore scores chunk c + ATTEND_LAG before it attends chunk c."""
    gathered = collections.deque()
    outputs = None
    anchor = None
    selections = []
    padded_tokens = chunk.q_nope.shape[0]
    starts = chunk_starts(padded_tokens, chunk_tokens)
    lag = min(ATTEND_LAG, len(starts) - 1)
    split = SplitChunk.of(chunk)
    for first in starts:
        scores = indexer.scores(first, chunk_tokens, anchor)
        anchor = scores
        rows, indices = select_and_gather(
            split.arrays,
            chunk_row(0, first),
            indexer.row_lengths[first : first + chunk_tokens],
            scores,
            layout=split.layout,
            num_tokens=chunk_tokens,
            top_k=indexer.top_k,
        )
        gathered.append(Gathered(first, chunk_tokens, rows))
        selections.append((first, indices))
        if len(gathered) > lag:
            outputs = attend_chunk(chunk, gathered.popleft(), outputs, scores)
            anchor = outputs[0]
    while gathered:
        outputs = attend_chunk(chunk, gathered.popleft(), outputs, None)
    return outputs, assemble_indices(selections, padded_tokens)


def attend_selected(
    indexer: IndexerChunks,
    unions: UnionPass,
    starts: list[int],
    chunk_tokens: int,
    out: jax.Array,
) -> tuple[jax.Array, list[tuple[int, jax.Array]]]:
    """out with the unions' rows written in place, and each chunk's (first
    token, top-k); scores c + UNION_ATTEND_LAG runs before attention c."""
    num_chunks = len(starts)
    selected, selections, gathered = ([None] * num_chunks for _ in range(3))

    def score(chunk, anchor):
        scores = indexer.scores(starts[chunk], chunk_tokens, anchor)
        selected[chunk] = indexer.select(starts[chunk], chunk_tokens, scores)
        return scores

    def unite(chunk, anchor):
        selections[chunk] = unions.select(
            chunk, selected[chunk], selected[max(chunk - 1, 0)], anchor
        )
        return [selected[chunk], selections[chunk].table]

    def gather(chunk, anchor):
        gathered[chunk] = unions.gather(selections[chunk], anchor)
        return [selections[chunk].meta, gathered[chunk].block_table]

    step = score(0, None)
    if num_chunks > 1:
        step = score(1, step)
    launched = unite(0, step)
    step = queries = unions.queries((step, launched))
    launched = gather(0, step)
    if num_chunks > 1:
        launched += unite(1, step)
    for chunk in range(num_chunks):
        later = chunk + UNION_ATTEND_LAG
        if later < num_chunks:
            step = score(later, (step, launched))
            launched = []
        if chunk + 1 < num_chunks:
            launched += gather(chunk + 1, step)
        out = unions.attend(
            selections[chunk], gathered[chunk], queries, out, (step, launched)
        )
        step, launched = out, []
        if later < num_chunks:
            launched = unite(later, step)
    return out, list(zip(starts, selected))


class LayerOperands(NamedTuple):
    """One DSA layer call's checked operands and static parameters."""

    index_queries: jax.Array
    index_weights: jax.Array
    index_cache: jax.Array
    q_nope: jax.Array
    q_pe: jax.Array
    nope_cache: jax.Array
    rope_cache: jax.Array
    cu_q_lens: jax.Array
    kv_lens: jax.Array
    block_table: jax.Array
    kv_scale: jax.Array
    max_seqlen_k: int
    sm_scale: float
    top_k: int
    return_lse: bool
    chunk_tokens: int | None


# A route's attention outputs, (out, [lse]) as dsa_sparse_mla returns them,
# and its (T, top_k) indices.
RouteOutputs = tuple[jax.Array | tuple[jax.Array, jax.Array], jax.Array]


def indexer_chunks(
    index_queries: jax.Array,
    index_weights: jax.Array,
    index_cache: jax.Array,
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    block_table: jax.Array,
    *,
    max_seqlen_k: int,
    top_k: int,
    padded_tokens: int,
) -> IndexerChunks:
    """The indexer's and the top-k's operands of a call's chunks, its tokens
    padded to padded_tokens."""
    num_tokens = index_queries.shape[0]
    token_pad = ((0, padded_tokens - num_tokens), (0, 0))
    _, position = token_positions(cu_q_lens, kv_lens, padded_tokens)
    return IndexerChunks(
        queries=jnp.pad(index_queries, (*token_pad, (0, 0))),
        weights=jnp.pad(index_weights, token_pad),
        index_cache=index_cache,
        cu_q_lens=cu_q_lens,
        kv_lens=kv_lens,
        block_table=block_table,
        row_lengths=position + 1,
        max_seqlen_k=max_seqlen_k,
        top_k=top_k,
    )


def layer_indexer(layer: LayerOperands, padded_tokens: int) -> IndexerChunks:
    """indexer_chunks of the layer."""
    return indexer_chunks(
        layer.index_queries,
        layer.index_weights,
        layer.index_cache,
        layer.cu_q_lens,
        layer.kv_lens,
        layer.block_table,
        max_seqlen_k=layer.max_seqlen_k,
        top_k=layer.top_k,
        padded_tokens=padded_tokens,
    )


def whole_steps(num_tokens: int) -> int:
    """num_tokens up to whole steps of the sparse MLA."""
    step_tokens = tokens_per_step(num_tokens)
    return pl.cdiv(num_tokens, step_tokens) * step_tokens


def sparse_route(layer: LayerOperands) -> RouteOutputs:
    """The layer through the indexer, its top-k, the gather and the sparse
    MLA, pipelined over chunks of tokens."""
    num_tokens, num_heads, _ = layer.q_nope.shape
    chunk = chunk_attention(
        layer.q_nope,
        layer.q_pe,
        layer.nope_cache,
        layer.rope_cache,
        layer.cu_q_lens,
        layer.kv_lens,
        layer.block_table,
        top_k=layer.top_k,
        sm_scale=layer.sm_scale,
        kv_scale=layer.kv_scale,
        return_lse=layer.return_lse,
    )
    padded_tokens = chunk.q_nope.shape[0]
    indexer = layer_indexer(layer, padded_tokens)
    chunk_tokens = call_chunk_tokens(
        layer.chunk_tokens, padded_tokens, layer.top_k, layer.max_seqlen_k
    )
    if chunk_tokens < padded_tokens:
        outputs, indices = run_chunks(indexer, chunk, chunk_tokens)
    else:
        scores = indexer.scores(0, padded_tokens, None)
        indices = topk.top_k(
            scores, layer.top_k, row_lengths=indexer.row_lengths
        )
        outputs = attend_in_chunks(
            chunk, indices, sparse_mla_chunk_tokens(padded_tokens)
        )
    return chunk_outputs(outputs, num_tokens, num_heads), indices[:num_tokens]


def selected_indices(
    indexer: IndexerChunks, num_tokens: int, chunk_tokens: int | None
) -> jax.Array:
    """The first num_tokens tokens' (T, top_k) indices: the indexer's scores
    and their top-k chunk by chunk, the SparseCore selecting a chunk while
    the TensorCore scores the next."""
    padded_tokens = indexer.queries.shape[0]
    chunk_tokens = call_chunk_tokens(
        chunk_tokens, padded_tokens, indexer.top_k, indexer.max_seqlen_k
    )
    selections = []
    for first in chunk_starts(padded_tokens, chunk_tokens):
        scores = indexer.scores(first, chunk_tokens, None)
        selections.append((first, indexer.select(first, chunk_tokens, scores)))
    return assemble_indices(selections, num_tokens)


def by_selection(
    indexer: IndexerChunks,
    num_tokens: int,
    causal: Callable[[jax.Array], object],
    selected: Callable[[], object],
) -> object:
    """selected(), or with no indexer causal(each position a token sees, in
    order, as (T, top_k) indices) when every request holds at most top_k
    positions, which each token's exact top-k selects whole. The step's
    lengths pick on device unless max_seqlen_k rules the indexer out."""

    def _causal():
        return causal(
            causal_indices(
                indexer.cu_q_lens, indexer.kv_lens, num_tokens, indexer.top_k
            )
        )

    if indexer.max_seqlen_k <= indexer.top_k:
        return _causal()
    return lax.cond(
        jnp.max(indexer.kv_lens) <= indexer.top_k, _causal, selected
    )


def routed_indices(
    indexer: IndexerChunks, num_tokens: int, chunk_tokens: int | None
) -> jax.Array:
    """Every token's (T, top_k) indices for the routed attention:
    selected_indices, or by_selection's causal indices."""
    return by_selection(
        indexer,
        num_tokens,
        lambda indices: indices,
        functools.partial(selected_indices, indexer, num_tokens, chunk_tokens),
    )


class RoutePlan(NamedTuple):
    """A call's static route: the modes its requests may take, the mode
    every request takes when the route is pinned, else None, the most
    unions the union mode builds, and the fewest tokens of a union
    request."""

    modes: tuple[int, ...]
    pinned_mode: int | None
    num_unions: int
    min_union_q_len: int = MIN_UNION_Q_LEN

    @property
    def is_sparse_only(self) -> bool:
        return self.modes == (MODE_SPARSE,)

    def union_slots(self, window_tokens: int, batch: int) -> int:
        """Slots of a union chunk whose window holds window_tokens tokens of
        batch requests."""
        return chunk_union_slots(
            window_tokens, batch, self.min_union_q_len, self.num_unions
        )

    def request_modes(
        self, cu_q_lens: jax.Array, kv_lens: jax.Array
    ) -> jax.Array:
        """(B,) int32 mode of each request: the pinned one, or the rule's."""
        if self.pinned_mode is None:
            return request_modes(
                cu_q_lens, kv_lens, has_union=MODE_UNION in self.modes
            )
        else:
            return jnp.full(kv_lens.shape, self.pinned_mode, jnp.int32)


def route_plan(
    route: str | None,
    *,
    num_tokens: int,
    num_heads: int,
    batch: int,
    top_k: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    return_lse: bool,
) -> RoutePlan:
    """The plan of route, or of the rule if None; fails unless the pinned
    mode serves the call. The rule compiles the sparse MLA alone under
    MIN_ROUTED_TOKENS tokens or when max_seqlen_q leaves no other mode; it
    adds masked-dense from MIN_MASKED_DENSE_Q_LEN at the heads it takes, and
    the union mode at UNION_HEADS heads without lse within the union
    kernel's VMEM."""
    if route == ROUTE_SPARSE:
        return RoutePlan((MODE_SPARSE,), MODE_SPARSE, 0)
    elif route == ROUTE_MASKED_DENSE:
        check_masked_dense_heads(num_heads)
        return RoutePlan((MODE_MASKED_DENSE,), MODE_MASKED_DENSE, 0)
    elif route == ROUTE_UNION:
        num_unions = max_unions(num_tokens, batch, PINNED_UNION_Q_LEN)
        check_union(num_heads, return_lse, num_unions, top_k, max_seqlen_k)
        return RoutePlan(
            (MODE_UNION,), MODE_UNION, num_unions, PINNED_UNION_Q_LEN
        )
    elif route is not None:
        raise ValueError(f"route must be None or one of {ROUTES}; got {route}.")
    num_unions = max_unions(num_tokens, batch, MIN_UNION_Q_LEN)
    has_union = (
        max_seqlen_q >= MIN_UNION_Q_LEN
        and num_heads in UNION_HEADS
        and not return_lse
        and union_fits(num_unions, top_k, max_seqlen_k)
    )
    has_masked_dense = (
        max_seqlen_q >= MIN_MASKED_DENSE_Q_LEN and num_heads in SUPPORTED_HEADS
    )
    modes = (MODE_SPARSE,)
    if has_masked_dense:
        modes += (MODE_MASKED_DENSE,)
    if has_union:
        modes += (MODE_UNION,)
    if num_tokens < MIN_ROUTED_TOKENS or modes == (MODE_SPARSE,):
        return RoutePlan((MODE_SPARSE,), None, 0)
    else:
        return RoutePlan(modes, None, num_unions if has_union else 0)


class RoutedOperands(NamedTuple):
    """The operands every pass of a routed call shares, the queries padded
    to whole steps of the sparse MLA."""

    q_nope: jax.Array
    q_pe: jax.Array
    nope_cache: jax.Array
    rope_cache: jax.Array
    topk_indices: jax.Array
    cu_q_lens: jax.Array
    kv_lens: jax.Array
    block_table: jax.Array
    kv_scale: jax.Array
    sm_scale: float
    return_lse: bool


def gather_row(first_token: jax.Array) -> jax.Array:
    """The gather's (CHUNK_WORDS,) int32 chunk row of the tokens from the
    traced first_token, their index rows from the first on."""
    lane = jnp.arange(CHUNK_WORDS, dtype=jnp.int32)
    return jnp.where(lane == CHUNK_FIRST_TOKEN, first_token, 0)


def sparse_extent(
    cu_q_lens: jax.Array, is_sparse: jax.Array, num_tokens: int
) -> tuple[jax.Array, jax.Array]:
    """The first and end token of the requests is_sparse marks, in whole
    steps of the sparse MLA and within the call's requests: (num_tokens, 0)
    when it marks none."""
    step_tokens = tokens_per_step(num_tokens)
    first = jnp.min(jnp.where(is_sparse, cu_q_lens[:-1], num_tokens))
    last = jnp.max(jnp.where(is_sparse, cu_q_lens[1:], 0))
    end = jnp.minimum(pl.cdiv(last, step_tokens) * step_tokens, cu_q_lens[-1])
    return first // step_tokens * step_tokens, end


class SparsePass(NamedTuple):
    """The sparse MLA of sparse_extent's tokens: the call cut at the
    extent's end, in chunks of chunk_tokens from its first token; a chunk's
    steps past the end keep the outputs' rows. A token of another mode
    inside the extent gets the sparse MLA's row, its same attention."""

    operands: RoutedOperands
    first: jax.Array
    end: jax.Array
    cu_q_lens: jax.Array
    kv_lens: jax.Array
    indices: jax.Array
    chunk_tokens: int

    @classmethod
    def of(
        cls, operands: RoutedOperands, is_sparse: jax.Array, chunk_tokens: int
    ) -> "SparsePass":
        """The pass of the requests is_sparse marks."""
        num_tokens = operands.q_nope.shape[0]
        first, end = sparse_extent(operands.cu_q_lens, is_sparse, num_tokens)
        topk_indices = operands.topk_indices
        return cls(
            operands,
            first,
            end,
            *chunk_lengths(operands.cu_q_lens, operands.kv_lens, 0, end),
            jnp.pad(
                topk_indices,
                ((0, num_tokens - topk_indices.shape[0]), (0, 0)),
                constant_values=UNUSED_INDEX,
            ),
            chunk_tokens,
        )

    @property
    def num_head_chunks(self) -> int:
        return min(
            pl.cdiv(self.operands.q_nope.shape[0], self.chunk_tokens),
            pl.cdiv(SPARSE_HEAD_TOKENS, self.chunk_tokens),
        )

    def chunk(self) -> ChunkAttention:
        operands = self.operands
        return chunk_attention(
            operands.q_nope,
            operands.q_pe,
            operands.nope_cache,
            operands.rope_cache,
            self.cu_q_lens,
            self.kv_lens,
            operands.block_table,
            top_k=operands.topk_indices.shape[1],
            sm_scale=operands.sm_scale,
            kv_scale=operands.kv_scale,
            return_lse=operands.return_lse,
        )

    def first_token(self, index: int | jax.Array) -> jax.Array:
        num_tokens = self.operands.q_nope.shape[0]
        return jnp.minimum(
            self.first + index * self.chunk_tokens,
            num_tokens - self.chunk_tokens,
        )

    def gather(
        self, chunk: ChunkAttention, index: int | jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        first_token = self.first_token(index)
        chunk_indices = slot_indices(
            lax.dynamic_slice_in_dim(
                self.indices, first_token, self.chunk_tokens
            ),
            chunk.cache.top_k,
        )
        return chunk.gather_rows(
            gather_row(first_token),
            chunk_indices.reshape(-1),
            self.chunk_tokens,
        )

    def attend(
        self,
        chunk: ChunkAttention,
        index: int | jax.Array,
        rows: tuple[jax.Array, jax.Array, jax.Array],
        outputs: tuple[jax.Array, ...],
    ) -> tuple[jax.Array, ...]:
        return tuple(
            chunk.attend_rows(
                rows,
                outputs,
                first_token=self.first_token(index),
                num_tokens=self.chunk_tokens,
                scalars=chunk.scalars,
                writes_padding=False,
            )
        )

    def head(self, outputs: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
        """The chunks of the first SPARSE_HEAD_TOKENS tokens, each gather
        under the chunk before's attention."""
        chunk = self.chunk()
        rows = self.gather(chunk, 0)
        for index in range(self.num_head_chunks):
            if index + 1 < self.num_head_chunks:
                next_rows = self.gather(chunk, index + 1)
            else:
                next_rows = None
            outputs = self.attend(chunk, index, rows, outputs)
            rows = next_rows
        return outputs

    def tail(self, outputs: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
        """The chunks after the head's, one at a time in a loop."""

        def _attend(index, outputs):
            # Note (david): a value the SparseCore lays out is relaid out
            # where it enters a loop, so a loop chunk builds its own
            # operands.
            chunk = self.chunk()
            return self.attend(chunk, index, self.gather(chunk, index), outputs)

        num_chunks = pl.cdiv(self.end - self.first, self.chunk_tokens)
        # Note (david): a loop that runs no trip still costs its entry, so the
        # loop runs only when the extent reaches past the head.
        return lax.cond(
            num_chunks > self.num_head_chunks,
            lambda outputs: lax.fori_loop(
                self.num_head_chunks, num_chunks, _attend, outputs
            ),
            lambda outputs: outputs,
            outputs,
        )


def lse_rows(lse: jax.Array, step_tokens: int) -> jax.Array:
    """(H, T) lse as the sparse MLA's (T / G, G, NUM_LANES) rows."""
    num_heads = lse.shape[0]
    rows = jnp.pad(lse.T, ((0, 0), (0, NUM_LANES - num_heads)))
    return rows.reshape(-1, step_tokens, NUM_LANES)


def masked_dense_pass(
    operands: RoutedOperands,
    is_masked: jax.Array,
    outputs: tuple[jax.Array, ...] | None,
    *,
    max_seqlen_q: int,
    max_seqlen_k: int,
) -> tuple[jax.Array, ...]:
    """The causal masked-dense MLA of the requests is_masked marks, written
    into outputs (initial_outputs', which come without lse), or into new
    (out, [lse rows]) if None, whose other rows are unspecified but the
    padding's, zeros and +inf; their selection bitmaps cover max_seqlen_k
    positions."""
    num_tokens, num_heads, _ = operands.q_nope.shape
    top_k = operands.topk_indices.shape[1]
    cu_q_lens, kv_lens = operands.cu_q_lens, operands.kv_lens
    has_bitmap = max_seqlen_k > top_k
    if has_bitmap:
        # Note (david): a request outside the pass holds no positions, so its
        # tokens take no bitmap.
        bitmap = selection_bitmap(
            operands.topk_indices,
            cu_q_lens,
            jnp.where(is_masked, kv_lens, 0),
            max_seqlen_k=max_seqlen_k,
        )
    else:
        bitmap = None
    if outputs is None:
        call_cu_q_lens, call_kv_lens, previous_out = cu_q_lens, kv_lens, None
    else:
        # Note (david): the padding rows hold their zeros already, so a last
        # request without attended tokens covers them and the kernel writes
        # none.
        call_cu_q_lens = jnp.append(cu_q_lens, num_tokens)
        call_kv_lens = jnp.append(kv_lens, 0)
        previous_out = outputs[0]
    attended_q_lens = jnp.where(is_masked, request_q_lens(cu_q_lens), 0)
    num_extra = call_kv_lens.shape[0] - kv_lens.shape[0]
    plan = masked_dense_plan(
        cumulative_lens(jnp.pad(attended_q_lens, (0, num_extra))),
        call_kv_lens,
        num_tokens=num_tokens,
        max_seqlen_q=max_seqlen_q,
        config=masked_dense_config_of(
            num_heads=num_heads,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            has_bitmap=has_bitmap,
        ),
    )
    result = masked_dense_attention(
        operands.q_nope,
        operands.q_pe,
        operands.nope_cache,
        operands.rope_cache,
        bitmap,
        call_cu_q_lens,
        call_kv_lens,
        operands.block_table,
        plan=plan,
        sm_scale=operands.sm_scale,
        kv_scale=operands.kv_scale,
        top_k=top_k,
        max_seqlen_k=max_seqlen_k,
        return_lse=operands.return_lse,
        previous_out=previous_out,
    )
    if not operands.return_lse:
        return (result,)
    else:
        out, lse = result
        return out, lse_rows(lse, tokens_per_step(num_tokens))


def sparse_chunk_tokens(chunk_tokens: int | None, num_tokens: int) -> int:
    """The sparse MLA's chunk of a call of num_tokens tokens: chunk_tokens,
    a multiple of the attention step's tokens, or its default if None; at
    most the call's tokens."""
    if chunk_tokens is None:
        chunk_tokens = sparse_mla_chunk_tokens(num_tokens)
    else:
        check_chunk_tokens(chunk_tokens, num_tokens)
    return min(chunk_tokens, num_tokens)


def attend_routed(
    q_nope: jax.Array,
    q_pe: jax.Array,
    nope_cache: jax.Array,
    rope_cache: jax.Array,
    topk_indices: jax.Array | None,
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    block_table: jax.Array,
    plan: RoutePlan,
    *,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float,
    kv_scale: jax.Array,
    return_lse: bool,
    chunk_tokens: int | None,
    with_union: bool,
    union_chunk_tokens: int | None = None,
    indexer: IndexerChunks | None = None,
) -> RouteOutputs:
    """The attention with each request on its plan's mode, each pass writing
    only its requests' rows, of topk_indices, or of the indexer's top-k when
    indexer is given; returns the outputs and the (T, top_k) indices.
    Without union requests: the masked-dense pass, then the sparse pass.
    With them (with_union): the union chunks, the sparse head, the
    masked-dense pass when a request takes it, then the sparse tail."""
    num_tokens, num_heads, _ = q_nope.shape
    padded_tokens = whole_steps(num_tokens)
    token_pad = ((0, padded_tokens - num_tokens), (0, 0), (0, 0))
    q_nope, q_pe = jnp.pad(q_nope, token_pad), jnp.pad(q_pe, token_pad)
    modes = plan.request_modes(cu_q_lens, kv_lens)
    q_lens = request_q_lens(cu_q_lens)
    is_masked, is_union, is_sparse = (
        (modes == mode) & (q_lens > 0)
        for mode in (MODE_MASKED_DENSE, MODE_UNION, MODE_SPARSE)
    )
    if plan.pinned_mode is None:
        masked_seqlen_k = min(max_seqlen_k, MAX_MASKED_DENSE_KV_LEN)
    else:
        masked_seqlen_k = max_seqlen_k
    route = RouteContext(
        q_nope=q_nope,
        q_pe=q_pe,
        nope_cache=nope_cache,
        rope_cache=rope_cache,
        cu_q_lens=cu_q_lens,
        kv_lens=kv_lens,
        block_table=block_table,
        plan=plan,
        is_masked=is_masked,
        is_union=is_union,
        is_sparse=is_sparse,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        masked_seqlen_k=masked_seqlen_k,
        sm_scale=sm_scale,
        kv_scale=kv_scale,
        return_lse=return_lse,
        chunk_tokens=chunk_tokens,
        union_chunk_tokens=union_chunk_tokens,
        num_tokens=num_tokens,
    )
    if with_union:
        outputs, indices = attend_with_union(route, topk_indices, indexer)
    else:
        outputs, indices = attend_without_union(route, topk_indices)
    return chunk_outputs(outputs, num_tokens, num_heads), indices


class RouteContext(NamedTuple):
    """One routed call's padded queries, operands, request masks and static
    lengths."""

    q_nope: jax.Array
    q_pe: jax.Array
    nope_cache: jax.Array
    rope_cache: jax.Array
    cu_q_lens: jax.Array
    kv_lens: jax.Array
    block_table: jax.Array
    plan: RoutePlan
    is_masked: jax.Array
    is_union: jax.Array
    is_sparse: jax.Array
    max_seqlen_q: int
    max_seqlen_k: int
    masked_seqlen_k: int
    sm_scale: float
    kv_scale: jax.Array
    return_lse: bool
    chunk_tokens: int | None
    union_chunk_tokens: int | None
    num_tokens: int

    @property
    def padded_tokens(self) -> int:
        return self.q_nope.shape[0]


def route_passes(
    route: RouteContext, topk_indices: jax.Array
) -> tuple[Callable, "SparsePass"]:
    """The route's masked-dense pass and sparse pass over topk_indices."""
    operands = RoutedOperands(
        route.q_nope,
        route.q_pe,
        route.nope_cache,
        route.rope_cache,
        topk_indices,
        route.cu_q_lens,
        route.kv_lens,
        route.block_table,
        route.kv_scale,
        route.sm_scale,
        route.return_lse,
    )
    masked_dense = functools.partial(
        masked_dense_pass,
        operands,
        route.is_masked,
        max_seqlen_q=route.max_seqlen_q,
        max_seqlen_k=route.masked_seqlen_k,
    )
    sparse = SparsePass.of(
        operands,
        route.is_sparse,
        sparse_chunk_tokens(route.chunk_tokens, route.padded_tokens),
    )
    return masked_dense, sparse


def attend_without_union(
    route: RouteContext, topk_indices: jax.Array
) -> tuple[tuple, jax.Array]:
    """The masked-dense pass, then the sparse pass, of topk_indices."""
    masked_dense, sparse = route_passes(route, topk_indices)
    if MODE_MASKED_DENSE in route.plan.modes:
        outputs = masked_dense(None)
    else:
        outputs = initial_outputs(
            route.cu_q_lens, route.q_nope.shape, route.q_nope.dtype
        )
    if MODE_SPARSE in route.plan.modes:
        outputs = sparse.tail(sparse.head(outputs))
    return outputs, topk_indices


def attend_unions(
    route: RouteContext,
    starts: list[int],
    union_chunk: int,
    top_k: int,
    given_indices: jax.Array | None,
    indexer: IndexerChunks | None,
) -> tuple[jax.Array, jax.Array]:
    """The union chunks' out and the (T, top_k) indices: given_indices, or
    the indexer's top-k when given_indices is None."""
    unions = union_pass(
        route.q_nope,
        route.q_pe,
        route.nope_cache,
        route.rope_cache,
        route.block_table,
        token_unions(
            route.cu_q_lens,
            route.kv_lens,
            route.is_union,
            route.plan.num_unions,
        ),
        starts,
        union_chunk,
        # Note (david): the indexer's chunks select their own tokens; given
        # indices hold every token's row.
        index_starts=starts if given_indices is None else [0] * len(starts),
        num_slots=route.plan.union_slots(union_chunk, route.kv_lens.shape[0]),
        top_k=top_k,
        sm_scale=route.sm_scale,
        kv_scale=route.kv_scale,
        max_seqlen_k=route.max_seqlen_k,
    )
    out = initial_outputs(
        route.cu_q_lens, route.q_nope.shape, route.q_nope.dtype
    )[0]
    if given_indices is not None:
        return attend_given(unions, given_indices, out), given_indices
    out, selections = attend_selected(indexer, unions, starts, union_chunk, out)
    return out, assemble_indices(selections, route.num_tokens)


def attend_with_union(
    route: RouteContext,
    topk_indices: jax.Array | None,
    indexer: IndexerChunks | None,
) -> tuple[tuple, jax.Array]:
    """The union chunks, the sparse head, the masked-dense pass when a
    request takes it, then the sparse tail."""
    top_k = indexer.top_k if indexer is not None else topk_indices.shape[1]
    union_chunk = call_chunk_tokens(
        route.union_chunk_tokens, route.padded_tokens, top_k, route.max_seqlen_k
    )
    starts = chunk_starts(route.padded_tokens, union_chunk)
    unions = functools.partial(
        attend_unions, route, starts, union_chunk, top_k, indexer=indexer
    )
    # Note (david): only the union chunks sit in the selection's cond, so the
    # other passes see their operands as without the union mode.
    if indexer is None:
        out, indices = unions(topk_indices)
    else:
        out, indices = by_selection(
            indexer, route.num_tokens, unions, lambda: unions(None)
        )
    masked_dense, sparse = route_passes(route, indices)
    outputs = (out,)
    if MODE_SPARSE in route.plan.modes:
        outputs = sparse.head(outputs)
    if MODE_MASKED_DENSE in route.plan.modes:
        outputs = lax.cond(
            jnp.any(route.is_masked),
            masked_dense,
            lambda outputs: outputs,
            outputs,
        )
    if MODE_SPARSE in route.plan.modes:
        outputs = sparse.tail(outputs)
    return outputs, indices


def initial_outputs(
    cu_q_lens: jax.Array, out_shape: tuple[int, ...], out_dtype: jnp.dtype
) -> tuple[jax.Array]:
    """The routed passes' (out,) before any pass writes it: the padding rows
    past cu_q_lens[B] zeros if there are any, the other rows unspecified, as
    each pass writes its own requests' rows."""
    # Note (david): both branches make their buffer, so neither passes one
    # through, which a lax.cond copies.
    return lax.cond(
        cu_q_lens[-1] < out_shape[0],
        lambda: (jnp.zeros(out_shape, out_dtype),),
        lambda: (lax.empty(out_shape, out_dtype),),
    )


def call_route(
    plan: RoutePlan,
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    num_tokens: int,
    sparse: Callable[[], object],
    routed: Callable[[bool], object],
) -> object:
    """sparse() for a plan of the sparse MLA alone, routed(with_union) for
    a pinned dense mode; under the rule, one picked on device: sparse()
    where every request attends sparse and the padding rows fill no chunk
    of the sparse MLA, which the routed sparse pass skips, else
    routed(with_union) by whether a request takes the union mode."""
    has_union = MODE_UNION in plan.modes
    if plan.is_sparse_only:
        return sparse()
    elif plan.pinned_mode is not None:
        return routed(has_union)
    modes = plan.request_modes(cu_q_lens, kv_lens)
    q_lens = request_q_lens(cu_q_lens)
    padded_tokens = whole_steps(num_tokens)
    num_padding = padded_tokens - cu_q_lens[-1]
    takes_sparse = jnp.all((modes == MODE_SPARSE) | (q_lens == 0)) & (
        num_padding < sparse_mla_chunk_tokens(padded_tokens)
    )
    branches = [sparse, functools.partial(routed, False)]
    if has_union:
        branches.append(functools.partial(routed, True))
        is_union = (modes == MODE_UNION) & (q_lens > 0)
        index = jnp.where(takes_sparse, 0, jnp.where(jnp.any(is_union), 2, 1))
    else:
        index = jnp.where(takes_sparse, 0, 1)
    return lax.switch(index, branches)


def routed(
    layer: LayerOperands,
    max_seqlen_q: int,
    route: str | None,
    return_indices: bool,
) -> jax.Array | tuple[jax.Array, ...]:
    """The layer through route, or by the rule if None, as call_route picks."""
    num_tokens, num_heads, _ = layer.q_nope.shape
    plan = route_plan(
        route,
        num_tokens=num_tokens,
        num_heads=num_heads,
        batch=layer.kv_lens.shape[0],
        top_k=layer.top_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=layer.max_seqlen_k,
        return_lse=layer.return_lse,
    )

    def _sparse():
        return with_indices(sparse_route(layer), return_indices)

    def _routed(with_union):
        indexer = layer_indexer(layer, whole_steps(num_tokens))
        indices = None
        if not with_union:
            indices = routed_indices(indexer, num_tokens, layer.chunk_tokens)
        route_outputs = attend_routed(
            layer.q_nope,
            layer.q_pe,
            layer.nope_cache,
            layer.rope_cache,
            indices,
            layer.cu_q_lens,
            layer.kv_lens,
            layer.block_table,
            plan,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=layer.max_seqlen_k,
            sm_scale=layer.sm_scale,
            kv_scale=layer.kv_scale,
            return_lse=layer.return_lse,
            chunk_tokens=None,
            union_chunk_tokens=layer.chunk_tokens,
            with_union=with_union,
            indexer=indexer if with_union else None,
        )
        return with_indices(route_outputs, return_indices)

    return call_route(
        plan, layer.cu_q_lens, layer.kv_lens, num_tokens, _sparse, _routed
    )


def with_indices(
    route_outputs: RouteOutputs, return_indices: bool
) -> jax.Array | tuple[jax.Array, ...]:
    """A route's out, (out, lse), (out, indices) or (out, lse, indices)."""
    outputs, indices = route_outputs
    if not return_indices:
        return outputs
    elif isinstance(outputs, tuple):
        return (*outputs, indices)
    else:
        return outputs, indices


@functools.partial(
    jax.jit,
    static_argnames=(
        "max_seqlen_k",
        "sm_scale",
        "top_k",
        "return_lse",
        "return_indices",
        "chunk_tokens",
        "max_seqlen_q",
        "route",
    ),
)
def dsa_attention(
    index_queries: jax.Array,
    index_weights: jax.Array,
    index_cache: jax.Array,
    q_nope: jax.Array,
    q_pe: jax.Array,
    nope_cache: jax.Array,
    rope_cache: jax.Array,
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    block_table: jax.Array,
    *,
    max_seqlen_k: int,
    sm_scale: float,
    kv_scale: jax.Array,
    top_k: int = DSA_TOPK,
    return_lse: bool = False,
    return_indices: bool = False,
    chunk_tokens: int | None = None,
    max_seqlen_q: int | None = None,
    route: str | None = None,
) -> jax.Array | tuple[jax.Array, ...]:
    """One DSA layer on a ragged batch: each token attends the top_k cached
    positions its request's indexer scores highest among those it sees.

    Each request attends in the mode dsa_attention_from_indices gives it. A
    step whose requests all attend sparse takes the sparse pipeline, which
    selects and attends chunk by chunk; any other step selects every token
    first, with no indexer when every request holds at most top_k
    positions, then attends.

    Args:
      index_queries: (T, index_heads, INDEX_HEAD_DIM) float8_e4m3fn, as
        dsa_indexer_scores takes them.
      index_weights: (T, index_heads) f32, as dsa_indexer_scores takes
        them.
      index_cache: index_cache_shape(num_pages, page_size) uint8 index-K
        cache, the batch's keys already written.
      q_nope: (T, H, LATENT_DIM) bf16, as dsa_sparse_mla takes them.
      q_pe: (T, H, ROPE_DIM) bf16.
      nope_cache, rope_cache: dsa_cache_shapes(num_pages, page_size) int32,
        the batch's latents already written.
      cu_q_lens: (B + 1,) int32 from 0, nondecreasing; rows past cu_q_lens[B]
        are padding, with a zero output.
      kv_lens: (B,) int32 tokens of each request in the caches, its tokens
        the last q_len of them; at most max_seqlen_k.
      block_table: (B, pages_per_seq) int32 page of each request page in
        both caches, which share one page size.
      max_seqlen_k: a static bound on kv_lens.
      sm_scale: scale of q_nope . latent + q_pe . rope.
      kv_scale: f32 scalar scale of the fp8 latent cache; traced.
      top_k: positions a token attends, a multiple of 128.
      return_lse: also return the natural-log lse.
      return_indices: also return each token's (T, top_k) int32 selection,
        for dsa_attention_from_indices.
      chunk_tokens: tokens per sparse pipeline chunk, a multiple of the
        attention step's tokens; None picks it from T and max_chunk_tokens.
      max_seqlen_q: a static bound on the q_lens, as
        dsa_attention_from_indices takes it. A decode step passes 1.
      route: as dsa_attention_from_indices takes it.

    Returns:
      out (T, H, LATENT_DIM) bf16, then lse (H, T) f32 with return_lse, as
      dsa_sparse_mla returns them, then the indices with return_indices.
    """
    num_tokens = q_nope.shape[0]
    check_operands(
        q_nope, q_pe, nope_cache, rope_cache, cu_q_lens, kv_lens, block_table
    )
    check_layer_operands(index_queries, index_cache, q_nope, nope_cache, top_k)
    if max_seqlen_q is None:
        max_seqlen_q = num_tokens
    layer = LayerOperands(
        index_queries,
        index_weights,
        index_cache,
        q_nope,
        q_pe,
        nope_cache,
        rope_cache,
        cu_q_lens,
        kv_lens,
        block_table,
        kv_scale,
        max_seqlen_k,
        sm_scale,
        top_k,
        return_lse,
        chunk_tokens,
    )
    return routed(layer, max_seqlen_q, route, return_indices)


@functools.partial(
    jax.jit,
    static_argnames=(
        "max_seqlen_k",
        "sm_scale",
        "return_lse",
        "chunk_tokens",
        "max_seqlen_q",
        "route",
    ),
)
def dsa_attention_from_indices(
    q_nope: jax.Array,
    q_pe: jax.Array,
    nope_cache: jax.Array,
    rope_cache: jax.Array,
    topk_indices: jax.Array,
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    block_table: jax.Array,
    *,
    max_seqlen_k: int,
    sm_scale: float,
    kv_scale: jax.Array,
    return_lse: bool = False,
    chunk_tokens: int | None = None,
    max_seqlen_q: int | None = None,
    route: str | None = None,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """dsa_sparse_mla's attention of given indices, for the layers that reuse
    another layer's selection.

    Each request attends in the mode the rule (routing.py) gives it from
    its q_len and kv_len: sparse, masked-dense over the paged cache, or in
    unions (union_attention.py). Where the union mode does not serve the
    call (other than UNION_HEADS heads, return_lse, or a union kernel past
    the SparseCore's VMEM), sparse takes its place. One call serves
    requests of every mode in any order, a mode's pass skipped when no
    request takes it; a step whose requests all attend sparse takes the
    sparse MLA alone. Calls under MIN_ROUTED_TOKENS tokens compile the
    sparse MLA alone.

    Args:
      q_nope, q_pe, nope_cache, rope_cache, topk_indices, cu_q_lens,
        kv_lens, block_table, sm_scale, kv_scale, return_lse: as
        dsa_sparse_mla takes them; a token reads only its first
        min(top_k, position + 1) slots.
      max_seqlen_k: a static bound on kv_lens.
      chunk_tokens: the sparse MLA's chunk_tokens.
      max_seqlen_q: a static bound on the q_lens, which sizes the
        masked-dense q blocks; T if None. Under MIN_UNION_Q_LEN every
        request attends sparse.
      route: ROUTE_SPARSE, ROUTE_MASKED_DENSE or ROUTE_UNION puts every
        request on that mode, and fails where the mode does not serve the
        call; None routes each request by the rule.

    Returns:
      out (T, H, LATENT_DIM) bf16, or (out, lse) with return_lse, as
      dsa_sparse_mla returns them.
    """
    num_tokens, num_heads, _ = q_nope.shape
    check_operands(
        q_nope, q_pe, nope_cache, rope_cache, cu_q_lens, kv_lens, block_table
    )
    check_indices(topk_indices, num_tokens)
    if max_seqlen_q is None:
        max_seqlen_q = num_tokens
    plan = route_plan(
        route,
        num_tokens=num_tokens,
        num_heads=num_heads,
        batch=kv_lens.shape[0],
        top_k=topk_indices.shape[1],
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        return_lse=return_lse,
    )
    operands = (q_nope, q_pe, nope_cache, rope_cache, topk_indices)
    operands += (cu_q_lens, kv_lens, block_table)
    sparse = functools.partial(
        dsa_sparse_mla,
        *operands,
        sm_scale=sm_scale,
        kv_scale=kv_scale,
        return_lse=return_lse,
        chunk_tokens=chunk_tokens,
    )

    def _routed(with_union):
        return attend_routed(
            *operands,
            plan,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            sm_scale=sm_scale,
            kv_scale=kv_scale,
            return_lse=return_lse,
            chunk_tokens=chunk_tokens,
            with_union=with_union,
        )[0]

    return call_route(plan, cu_q_lens, kv_lens, num_tokens, sparse, _routed)


@functools.partial(
    jax.jit, static_argnames=("max_seqlen_k", "top_k", "chunk_tokens")
)
def dsa_indices(
    index_queries: jax.Array,
    index_weights: jax.Array,
    index_cache: jax.Array,
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    block_table: jax.Array,
    *,
    max_seqlen_k: int,
    top_k: int = DSA_TOPK,
    chunk_tokens: int | None = None,
) -> jax.Array:
    """Each token's (T, top_k) selection for dsa_attention_from_indices, as
    dsa_attention returns it with return_indices.

    From MIN_ROUTED_TOKENS tokens, a step whose requests all hold at most
    top_k positions takes each position a token sees, with no indexer.

    Args:
      index_queries, index_weights, index_cache, cu_q_lens, kv_lens,
        block_table, max_seqlen_k, top_k: as dsa_attention takes them.
      chunk_tokens: tokens per chunk of scores, a multiple of the attention
        step's tokens; None picks it from T and max_chunk_tokens.

    Returns:
      (T, top_k) int32 positions within each token's request, -1 for an
      unused slot, as top_k returns them.
    """
    num_tokens = index_queries.shape[0]
    check_requests(cu_q_lens, kv_lens, block_table)
    check_top_k(top_k)
    indexer = indexer_chunks(
        index_queries,
        index_weights,
        index_cache,
        cu_q_lens,
        kv_lens,
        block_table,
        max_seqlen_k=max_seqlen_k,
        top_k=top_k,
        padded_tokens=whole_steps(num_tokens),
    )
    if num_tokens < MIN_ROUTED_TOKENS:
        return selected_indices(indexer, num_tokens, chunk_tokens)
    else:
        return routed_indices(indexer, num_tokens, chunk_tokens)


def check_masked_dense_heads(num_heads: int) -> None:
    """Fails unless the masked-dense mode takes num_heads heads."""
    if num_heads not in SUPPORTED_HEADS:
        raise NotImplementedError(
            f"The masked-dense mode takes {SUPPORTED_HEADS} heads; got"
            f" {num_heads}."
        )


def check_union(
    num_heads: int,
    return_lse: bool,
    num_unions: int,
    top_k: int,
    max_seqlen_k: int,
) -> None:
    """Fails unless the union mode serves the call: UNION_HEADS heads, no
    lse, and num_unions unions within the union kernel's VMEM."""
    if num_heads not in UNION_HEADS or return_lse:
        raise NotImplementedError(
            f"The union mode takes {UNION_HEADS} heads and returns no lse;"
            f" got {num_heads} heads and return_lse {return_lse}."
        )
    elif not union_fits(num_unions, top_k, max_seqlen_k):
        raise NotImplementedError(
            f"The union kernel of {num_unions} unions at max_seqlen_k"
            f" {max_seqlen_k} overflows a SparseCore subcore's VMEM."
        )


def call_chunk_tokens(
    chunk_tokens: int | None, padded_tokens: int, top_k: int, max_seqlen_k: int
) -> int:
    """The chunk size of a call of padded_tokens tokens: chunk_tokens, a
    multiple of the attention step's tokens, or default_chunk_tokens within
    max_chunk_tokens if None; at most the call's tokens."""
    if chunk_tokens is None:
        return default_chunk_tokens(
            padded_tokens, max_chunk_tokens(top_k, max_seqlen_k)
        )
    check_chunk_tokens(chunk_tokens, padded_tokens)
    return min(chunk_tokens, padded_tokens)


def check_top_k(top_k: int) -> None:
    """Fails unless the gather takes top_k."""
    if top_k % CHUNK_SLOTS:
        raise ValueError(
            f"top_k must be a multiple of {CHUNK_SLOTS}; got {top_k}."
        )


def check_layer_operands(
    index_queries: jax.Array,
    index_cache: jax.Array,
    q_nope: jax.Array,
    nope_cache: jax.Array,
    top_k: int,
) -> None:
    """Fails unless the indexer and the sparse MLA share one token count and
    one page size, and the gather takes top_k."""
    if index_queries.shape[0] != q_nope.shape[0]:
        raise ValueError(
            "index_queries and q_nope must share their token count; got"
            f" {index_queries.shape} and {q_nope.shape}."
        )
    elif index_cache.shape[-1] != nope_cache.shape[1]:
        raise ValueError(
            "The index cache and the latent cache share block_table, so their"
            f" page sizes must match; got {index_cache.shape[-1]} and"
            f" {nope_cache.shape[1]}."
        )
    check_top_k(top_k)
