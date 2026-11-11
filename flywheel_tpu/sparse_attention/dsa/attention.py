"""DeepSeek sparse attention (DSA) of one layer: the lightning indexer's
scores, their top-k, and the sparse MLA over the selected latents.

Tokens run in chunks: the SparseCore selects and gathers one chunk while the
TensorCore scores the next. dsa_indices and dsa_attention_from_indices split
a layer into selection and attention for callers that run them apart.
"""

import collections
import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from flywheel_tpu.sparse_attention.dsa.cache_write import (
    check_requests,
    token_positions,
)
from flywheel_tpu.sparse_attention.dsa.indexer import (
    dsa_indexer_scores,
    score_columns,
)
from flywheel_tpu.sparse_attention.dsa.kernels.gather import (
    CHUNK_SLOTS,
    GATHERED_WORDS_PER_SLOT,
    NOPE_WORDS,
    fused_topk_vmem_words,
    gather_slots,
)
from flywheel_tpu.sparse_attention.dsa.kernels.indexer import BLOCK_TOKENS
from flywheel_tpu.sparse_attention.dsa.kernels.masked_dense import (
    MaskedDenseConfig,
)
from flywheel_tpu.sparse_attention.dsa.kernels.sparse_mla import (
    tokens_per_step,
)
from flywheel_tpu.sparse_attention.dsa.masked_dense import (
    SUPPORTED_HEADS,
    masked_dense_config_of,
    selection_bitmap,
)
from flywheel_tpu.sparse_attention.dsa.routing import (
    RequestRoutes,
    bitmap_us,
    causal_indices,
    indexer_us,
    mixed_attention,
    mixed_us,
    route_requests,
    sparse_pipeline_us,
    sparse_us,
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
MAX_SPARSE_CORE_TENSOR_BYTES = 16 << 30
ROUTE_MIXED = "mixed"
ROUTE_SPARSE = "sparse"
ROUTES = (ROUTE_MIXED, ROUTE_SPARSE)
# Note (david): a call under MIN_ROUTED_TOKENS tokens (decode) takes the sparse
# MLA: choosing on device costs it more than masked-dense saves. Past
# MIXED_MAX_SEQLEN_K keys no bitmaps are built, so the mixed route serves only
# steps whose requests all hold at most top_k positions.
MIN_ROUTED_TOKENS = 512
MIXED_MAX_SEQLEN_K = 32768


def after(value: jax.Array, anchor: jax.Array) -> jax.Array:
    """value, which XLA may use only once anchor exists. Both must be
    TensorCore values, as XLA moves a SparseCore call's done past barriers."""
    return lax.optimization_barrier((value, anchor))[0]


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
        row_lengths = indexer.row_lengths[first : first + chunk_tokens]
        indices = topk.top_k(scores, indexer.top_k, row_lengths=row_lengths)
        selections.append((first, indices))
    return assemble_indices(selections, num_tokens)


def routed_indices(
    indexer: IndexerChunks, num_tokens: int, chunk_tokens: int | None
) -> jax.Array:
    """Every token's (T, top_k) indices for the mixed attention:
    selected_indices, or with no indexer each position a token sees, in
    order, when every request holds at most top_k positions, which each
    token's exact top-k selects whole. The step's lengths pick on device
    unless max_seqlen_k rules the indexer out."""
    causal = functools.partial(
        causal_indices,
        indexer.cu_q_lens,
        indexer.kv_lens,
        num_tokens,
        indexer.top_k,
    )
    if indexer.max_seqlen_k <= indexer.top_k:
        return causal()
    return lax.cond(
        jnp.max(indexer.kv_lens) <= indexer.top_k,
        causal,
        functools.partial(selected_indices, indexer, num_tokens, chunk_tokens),
    )


class MixedRoute(NamedTuple):
    """The mixed route of a call: whether its masked-dense MLA takes
    selection bitmaps, its blocks, and each request's path."""

    has_bitmap: bool
    config: MaskedDenseConfig
    routes: RequestRoutes
    max_seqlen_q: int
    max_seqlen_k: int


def mixed_route_of(
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    *,
    num_tokens: int,
    num_heads: int,
    top_k: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
) -> MixedRoute:
    """The mixed route of a call of num_tokens tokens. Without bitmaps, at
    max_seqlen_k within top_k or past MIXED_MAX_SEQLEN_K, it serves only a
    step whose requests all hold at most top_k positions."""
    has_bitmap = top_k < max_seqlen_k <= MIXED_MAX_SEQLEN_K
    if not has_bitmap:
        max_seqlen_k = min(max_seqlen_k, top_k)
    config = masked_dense_config_of(
        num_heads=num_heads,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        has_bitmap=has_bitmap,
    )
    routes = route_requests(
        cu_q_lens, kv_lens, config, num_heads, whole_steps(num_tokens)
    )
    return MixedRoute(has_bitmap, config, routes, max_seqlen_q, max_seqlen_k)


def attend_mixed(
    q_nope: jax.Array,
    q_pe: jax.Array,
    nope_cache: jax.Array,
    rope_cache: jax.Array,
    topk_indices: jax.Array,
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    block_table: jax.Array,
    route: MixedRoute,
    *,
    sm_scale: float,
    kv_scale: jax.Array,
    return_lse: bool,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """The mixed attention of given indices on route, its selection bitmaps
    built from them."""
    if route.has_bitmap:
        bitmap = selection_bitmap(
            topk_indices, cu_q_lens, kv_lens, max_seqlen_k=route.max_seqlen_k
        )
    else:
        bitmap = None
    return mixed_attention(
        q_nope,
        q_pe,
        nope_cache,
        rope_cache,
        topk_indices,
        bitmap,
        cu_q_lens,
        kv_lens,
        block_table,
        route.routes,
        config=route.config,
        max_seqlen_q=route.max_seqlen_q,
        sm_scale=sm_scale,
        kv_scale=kv_scale,
        max_seqlen_k=route.max_seqlen_k,
        return_lse=return_lse,
    )


def mixed_route(layer: LayerOperands, route: MixedRoute) -> RouteOutputs:
    """The layer with each request on its own path: routed_indices, or
    with no bitmaps each position a token sees, then the mixed attention."""
    num_tokens = layer.q_nope.shape[0]
    if route.has_bitmap:
        indices = routed_indices(
            layer_indexer(layer, whole_steps(num_tokens)),
            num_tokens,
            layer.chunk_tokens,
        )
    else:
        indices = causal_indices(
            layer.cu_q_lens, layer.kv_lens, num_tokens, layer.top_k
        )
    outputs = attend_mixed(
        layer.q_nope,
        layer.q_pe,
        layer.nope_cache,
        layer.rope_cache,
        indices,
        layer.cu_q_lens,
        layer.kv_lens,
        layer.block_table,
        route,
        sm_scale=layer.sm_scale,
        kv_scale=layer.kv_scale,
        return_lse=layer.return_lse,
    )
    return outputs, indices


def takes_sparse(route: str | None, num_tokens: int, num_heads: int) -> bool:
    """Whether a call takes the sparse MLA with no choice on device."""
    if route is None:
        return (
            num_tokens < MIN_ROUTED_TOKENS or num_heads not in SUPPORTED_HEADS
        )
    else:
        return route == ROUTE_SPARSE


def routed(
    layer: LayerOperands,
    max_seqlen_q: int,
    route: str | None,
    return_indices: bool,
) -> jax.Array | tuple[jax.Array, ...]:
    """The layer through route, or if None, through the route of the lower
    estimate that the step's lengths pick on device. Each route builds only
    the outputs that with_indices keeps."""
    num_tokens, num_heads, _ = layer.q_nope.shape

    def sparse():
        return with_indices(sparse_route(layer), return_indices)

    if takes_sparse(route, num_tokens, num_heads):
        return sparse()
    mixed_plan = mixed_route_of(
        layer.cu_q_lens,
        layer.kv_lens,
        num_tokens=num_tokens,
        num_heads=num_heads,
        top_k=layer.top_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=layer.max_seqlen_k,
    )

    def mixed():
        return with_indices(mixed_route(layer, mixed_plan), return_indices)

    if route == ROUTE_MIXED:
        return mixed()
    is_within_top_k = jnp.max(layer.kv_lens) <= layer.top_k
    mixed_estimate = mixed_us(whole_steps(num_tokens), mixed_plan.routes)
    if mixed_plan.has_bitmap:
        mixed_estimate += bitmap_us(num_tokens) + jnp.where(
            is_within_top_k, 0.0, indexer_us(layer.cu_q_lens, layer.kv_lens)
        )
    else:
        mixed_estimate = jnp.where(is_within_top_k, mixed_estimate, jnp.inf)
    return lax.cond(
        mixed_estimate < sparse_pipeline_us(num_tokens), mixed, sparse
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

    The mixed route runs each request on the cheaper of masked-dense and
    sparse MLA, and skips the indexer when every request holds at most top_k
    positions. With route None the choice is made on device, and both routes
    are compiled. Calls under MIN_ROUTED_TOKENS tokens take the sparse
    pipeline.

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
      max_seqlen_q: a static bound on the q_lens, which sizes the
        masked-dense q blocks; T if None. A decode step passes 1.
      route: ROUTE_MIXED or ROUTE_SPARSE pins the route, ROUTE_MIXED only
        at max_seqlen_k within top_k or MIXED_MAX_SEQLEN_K; None picks it
        on device per step.

    Returns:
      out (T, H, LATENT_DIM) bf16, then lse (H, T) f32 with return_lse, as
      dsa_sparse_mla returns them, then the indices with return_indices.
    """
    num_tokens = q_nope.shape[0]
    check_operands(
        q_nope, q_pe, nope_cache, rope_cache, cu_q_lens, kv_lens, block_table
    )
    check_layer_operands(index_queries, index_cache, q_nope, nope_cache, top_k)
    check_route(route, max_seqlen_k, top_k)
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
    another layer's selection. Each request runs on the cheaper of the sparse
    MLA and masked-dense attention over the selection bitmaps built from its
    indices; calls under MIN_ROUTED_TOKENS tokens or past MIXED_MAX_SEQLEN_K
    keys take the sparse MLA.

    Args:
      q_nope, q_pe, nope_cache, rope_cache, topk_indices, cu_q_lens,
        kv_lens, block_table, sm_scale, kv_scale, return_lse: as
        dsa_sparse_mla takes them; a token reads only its first
        min(top_k, position + 1) slots.
      max_seqlen_k: a static bound on kv_lens.
      chunk_tokens: the sparse MLA's chunk_tokens.
      max_seqlen_q: a static bound on the q_lens, which sizes the
        masked-dense q blocks; T if None.
      route: ROUTE_MIXED or ROUTE_SPARSE pins the route; None picks it on
        device per step.

    Returns:
      out (T, H, LATENT_DIM) bf16, or (out, lse) with return_lse, as
      dsa_sparse_mla returns them.
    """
    num_tokens, num_heads, _ = q_nope.shape
    check_operands(
        q_nope, q_pe, nope_cache, rope_cache, cu_q_lens, kv_lens, block_table
    )
    check_indices(topk_indices, num_tokens)
    top_k = topk_indices.shape[1]
    check_route(route, max_seqlen_k, top_k)
    if max_seqlen_q is None:
        max_seqlen_q = num_tokens
    sparse = functools.partial(
        dsa_sparse_mla,
        q_nope,
        q_pe,
        nope_cache,
        rope_cache,
        topk_indices,
        cu_q_lens,
        kv_lens,
        block_table,
        sm_scale=sm_scale,
        kv_scale=kv_scale,
        return_lse=return_lse,
        chunk_tokens=chunk_tokens,
    )
    if takes_sparse(route, num_tokens, num_heads) or (
        route is None and max_seqlen_k > MIXED_MAX_SEQLEN_K
    ):
        return sparse()
    mixed_plan = mixed_route_of(
        cu_q_lens,
        kv_lens,
        num_tokens=num_tokens,
        num_heads=num_heads,
        top_k=top_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    )
    mixed = functools.partial(
        attend_mixed,
        q_nope,
        q_pe,
        nope_cache,
        rope_cache,
        topk_indices,
        cu_q_lens,
        kv_lens,
        block_table,
        mixed_plan,
        sm_scale=sm_scale,
        kv_scale=kv_scale,
        return_lse=return_lse,
    )
    if route == ROUTE_MIXED:
        return mixed()
    mixed_estimate = mixed_us(whole_steps(num_tokens), mixed_plan.routes)
    if mixed_plan.has_bitmap:
        mixed_estimate += bitmap_us(num_tokens)
    return lax.cond(mixed_estimate < sparse_us(num_tokens), mixed, sparse)


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


def check_route(route: str | None, max_seqlen_k: int, top_k: int) -> None:
    """Fails unless route is None or a route the call can take."""
    if route is not None and route not in ROUTES:
        raise ValueError(f"route must be None or one of {ROUTES}; got {route}.")
    elif route == ROUTE_MIXED and max_seqlen_k > max(top_k, MIXED_MAX_SEQLEN_K):
        raise ValueError(
            "The mixed route needs max_seqlen_k within top_k or"
            f" {MIXED_MAX_SEQLEN_K}; got {max_seqlen_k}."
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
