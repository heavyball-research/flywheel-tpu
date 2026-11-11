"""Per-request routing of a DSA layer between the masked-dense MLA and the
sparse MLA, and the mixed attention that runs both in one call.

A request takes the path its estimated time favors: the masked-dense MLA's
time grows with q_len times kv_len, the sparse MLA's with q_len alone. The
mixed attention runs the masked-dense MLA over its requests while the
SparseCore gathers one window of tokens, then attends that window sparse,
writing it into the masked-dense output.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from flywheel_tpu.core.tpu import NUM_LANES
from flywheel_tpu.sparse_attention.dsa.cache_write import token_positions
from flywheel_tpu.sparse_attention.dsa.kernels.gather import (
    CHUNK_FIRST_INDEX_ROW,
    CHUNK_FIRST_TOKEN,
    CHUNK_WORDS,
)
from flywheel_tpu.sparse_attention.dsa.kernels.masked_dense import (
    MaskedDenseConfig,
)
from flywheel_tpu.sparse_attention.dsa.kernels.sparse_mla import (
    tokens_per_step,
)
from flywheel_tpu.sparse_attention.dsa.masked_dense import (
    masked_dense_attention,
)
from flywheel_tpu.sparse_attention.dsa.sparse_mla import (
    chunk_attention,
    chunk_outputs,
    slot_indices,
)

# Note (david): estimated device times in microseconds. A masked-dense item (a
# q block by a kv block) costs MASKED_SCORE_US per score, MASKED_KEY_US per
# staged key and MASKED_ITEM_US more. A sparse token costs SPARSE_TOKEN_US, or
# SPARSE_PIPELINE_TOKEN_US with its indexer and top-k; a row of the mixed
# attention's window costs WINDOW_TOKEN_US.
MASKED_SCORE_US = 2.2e-6
MASKED_KEY_US = 2.7e-4
MASKED_ITEM_US = 0.86
MASKED_CALL_US = 60.0
SPARSE_TOKEN_US = 1.52
SPARSE_CALL_US = 65.0
SPARSE_PIPELINE_TOKEN_US = 1.75
WINDOW_TOKEN_US = 0.76
# Note (david): the indexer and top-k cost per token and per position a
# token scores, and the selection bitmaps per token.
INDEXER_CALL_US = 40.0
INDEXER_TOKEN_US = 0.09
INDEXER_PAIR_US = 2.3e-5
BITMAP_TOKEN_US = 0.035
# Note (david): the mixed attention attends a window of T // MIXED_WINDOW_SHARE
# rows sparse: wider serves more decode requests but adds rows to every step.
MIXED_WINDOW_SHARE = 16


class RequestRoutes(NamedTuple):
    """Each request's path in the mixed attention, and its estimates."""

    is_masked: jax.Array  # (B,) bool
    window_start: jax.Array  # () int32, first token row attended sparse
    masked_us: jax.Array  # () f32, the masked-dense requests' time


def request_q_lens(cu_q_lens: jax.Array) -> jax.Array:
    return cu_q_lens[1:] - cu_q_lens[:-1]


def cumulative_lens(lens: jax.Array) -> jax.Array:
    """(B + 1,) int32 offsets of consecutive runs of lens tokens."""
    return jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.cumsum(lens)])


def window_tokens(num_tokens: int) -> int:
    """Token rows the mixed attention of num_tokens rows attends sparse:
    whole steps of the sparse MLA."""
    step_tokens = tokens_per_step(num_tokens)
    num_steps = num_tokens // MIXED_WINDOW_SHARE // step_tokens
    return max(1, num_steps) * step_tokens


def masked_request_us(
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    config: MaskedDenseConfig,
    num_heads: int,
) -> jax.Array:
    """(B,) f32 masked-dense time of each request at config: its items, each
    q block with its kv blocks up to the block's mean last position."""
    block_q, block_kv, _ = config
    q_lens = request_q_lens(cu_q_lens)
    num_q_blocks = (q_lens + block_q - 1) // block_q
    mean_frontier = kv_lens - q_lens + (q_lens + block_q) // 2
    num_kv_blocks = jnp.maximum((mean_frontier + block_kv - 1) // block_kv, 1)
    item_us = (
        MASKED_SCORE_US * block_q * num_heads * block_kv
        + MASKED_KEY_US * block_kv
        + MASKED_ITEM_US
    )
    return item_us * (num_q_blocks * num_kv_blocks).astype(jnp.float32)


def route_requests(
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    config: MaskedDenseConfig,
    num_heads: int,
    num_tokens: int,
) -> RequestRoutes:
    """Each request's path in a call of num_tokens rows: the window starts at
    the first request that favors the sparse MLA, or ends at the last row if
    none does; requests inside it take the sparse MLA, the rest masked-dense."""
    step_tokens = tokens_per_step(num_tokens)
    window = window_tokens(num_tokens)
    q_lens = request_q_lens(cu_q_lens)
    masked_us = masked_request_us(cu_q_lens, kv_lens, config, num_heads)
    prefers_sparse = (q_lens > 0) & (
        SPARSE_TOKEN_US * q_lens.astype(jnp.float32) <= masked_us
    )
    q_starts, q_ends = cu_q_lens[:-1], cu_q_lens[1:]
    first_sparse = jnp.min(jnp.where(prefers_sparse, q_starts, num_tokens))
    window_start = jnp.minimum(
        first_sparse // step_tokens * step_tokens, num_tokens - window
    )
    is_in_window = (q_starts >= window_start) & (
        q_ends <= window_start + window
    )
    is_masked = (q_lens > 0) & jnp.logical_not(is_in_window)
    return RequestRoutes(
        is_masked=is_masked,
        window_start=window_start,
        masked_us=jnp.sum(jnp.where(is_masked, masked_us, 0.0)),
    )


def mixed_us(num_tokens: int, routes: RequestRoutes) -> jax.Array:
    """The mixed attention's time in a call of num_tokens rows, its gather
    under the masked-dense MLA, without the indices' and bitmaps' work."""
    return (
        MASKED_CALL_US
        + routes.masked_us
        + WINDOW_TOKEN_US * window_tokens(num_tokens)
    )


def sparse_us(num_tokens: int) -> float:
    """The sparse MLA's time of num_tokens tokens."""
    return SPARSE_CALL_US + SPARSE_TOKEN_US * num_tokens


def sparse_pipeline_us(num_tokens: int) -> float:
    """The sparse pipeline's time of num_tokens tokens: its gather,
    attention, indexer and top-k."""
    return SPARSE_CALL_US + SPARSE_PIPELINE_TOKEN_US * num_tokens


def bitmap_us(num_tokens: int) -> float:
    return BITMAP_TOKEN_US * num_tokens


def indexer_us(cu_q_lens: jax.Array, kv_lens: jax.Array) -> jax.Array:
    """The indexer's and top-k's time: per token and per position a token
    scores."""
    q_lens = request_q_lens(cu_q_lens).astype(jnp.float32)
    num_tokens = jnp.sum(q_lens)
    num_pairs = jnp.sum(
        q_lens * (kv_lens.astype(jnp.float32) - 0.5 * (q_lens - 1.0))
    )
    return (
        INDEXER_CALL_US
        + INDEXER_TOKEN_US * num_tokens
        + INDEXER_PAIR_US * num_pairs
    )


def lse_rows(lse: jax.Array, num_tokens: int, step_tokens: int) -> jax.Array:
    """(H, T) lse as the sparse MLA's (num_tokens / G, G, NUM_LANES) rows,
    +inf for the token rows past T."""
    num_heads, num_real = lse.shape
    rows = jnp.pad(lse.T, ((0, 0), (0, NUM_LANES - num_heads)))
    rows = jnp.pad(
        rows, ((0, num_tokens - num_real), (0, 0)), constant_values=jnp.inf
    )
    return rows.reshape(-1, step_tokens, NUM_LANES)


def window_row(window_start: jax.Array) -> jax.Array:
    """The gather's (CHUNK_WORDS,) int32 chunk row of the window that starts
    at token row window_start, its index rows the call's."""
    lane = jnp.arange(CHUNK_WORDS, dtype=jnp.int32)
    is_start = (lane == CHUNK_FIRST_INDEX_ROW) | (lane == CHUNK_FIRST_TOKEN)
    return jnp.where(is_start, window_start, 0)


def mixed_attention(
    q_nope: jax.Array,
    q_pe: jax.Array,
    nope_cache: jax.Array,
    rope_cache: jax.Array,
    topk_indices: jax.Array,
    bitmap: jax.Array | None,
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    block_table: jax.Array,
    routes: RequestRoutes,
    *,
    config: MaskedDenseConfig,
    max_seqlen_q: int,
    sm_scale: float,
    kv_scale: jax.Array,
    max_seqlen_k: int,
    return_lse: bool,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """The masked-dense MLA of routes' masked requests and the sparse MLA of
    routes' window of token rows in one output.

    The SparseCore gathers the window's rows while the TensorCore runs the
    masked-dense MLA; the sparse MLA then writes the window's rows into that
    output. A masked request's tokens inside the window are attended twice,
    and the sparse MLA's rows stay.
    """
    num_tokens, num_heads, _ = q_nope.shape
    top_k = topk_indices.shape[1]
    chunk = chunk_attention(
        q_nope,
        q_pe,
        nope_cache,
        rope_cache,
        cu_q_lens,
        kv_lens,
        block_table,
        top_k=top_k,
        sm_scale=sm_scale,
        kv_scale=kv_scale,
        return_lse=return_lse,
    )
    padded_tokens = chunk.q_nope.shape[0]
    window = window_tokens(padded_tokens)
    rows = chunk.gather_rows(
        window_row(routes.window_start),
        slot_indices(topk_indices, chunk.cache.top_k).reshape(-1),
        window,
    )
    masked = masked_dense_attention(
        q_nope,
        q_pe,
        nope_cache,
        rope_cache,
        bitmap,
        cu_q_lens,
        kv_lens,
        block_table,
        attended_cu_q_lens=cumulative_lens(
            jnp.where(routes.is_masked, request_q_lens(cu_q_lens), 0)
        ),
        config=config,
        max_seqlen_q=max_seqlen_q,
        sm_scale=sm_scale,
        kv_scale=kv_scale,
        top_k=top_k,
        max_seqlen_k=max_seqlen_k,
        return_lse=return_lse,
    )
    token_pad = ((0, padded_tokens - num_tokens), (0, 0), (0, 0))
    if return_lse:
        previous = (
            jnp.pad(masked[0], token_pad),
            lse_rows(masked[1], padded_tokens, tokens_per_step(padded_tokens)),
        )
    else:
        previous = (jnp.pad(masked, token_pad),)
    outputs = chunk.attend_rows(
        rows,
        previous,
        first_token=routes.window_start,
        num_tokens=window,
        scalars=chunk.scalars,
    )
    return chunk_outputs(outputs, num_tokens, num_heads)


def causal_indices(
    cu_q_lens: jax.Array, kv_lens: jax.Array, num_tokens: int, top_k: int
) -> jax.Array:
    """(T, top_k) int32 top-k of tokens that see at most top_k positions:
    each position the token sees, then -1."""
    _, position = token_positions(cu_q_lens, kv_lens, num_tokens)
    slot = jnp.arange(top_k, dtype=jnp.int32)
    return jnp.where(slot[None, :] <= position[:, None], slot[None, :], -1)
