"""The DSA lightning indexer's scores on a paged index-K cache."""

import functools

import jax
import jax.numpy as jnp

from flywheel_tpu.softmax_attention.config import PAGE_SIZE_MULTIPLE
from flywheel_tpu.sparse_attention.dsa.cache_write import check_requests
from flywheel_tpu.sparse_attention.dsa.kernels.indexer import (
    CACHE_GROUPS,
    DECODE_BLOCK_KV,
    FP8_DTYPE,
    INDEX_HEAD_DIM,
    INDEX_HEADS_MULTIPLE,
    KEY_GROUPS,
    MAX_PREFILL_BLOCK_KV,
    SCALE_GROUP,
    WORD_BYTES,
    indexer_scores_pallas,
    prefill_block_kv,
)

__all__ = [
    "CACHE_GROUPS",
    "FP8_DTYPE",
    "INDEX_HEAD_DIM",
    "INDEX_HEADS_MULTIPLE",
    "KEY_GROUPS",
    "SCALE_GROUP",
    "WORD_BYTES",
    "check_index_cache",
    "dsa_indexer_scores",
    "index_cache_shape",
    "score_columns",
]
CACHE_DTYPE = jnp.dtype(jnp.uint8)
WEIGHT_DTYPE = jnp.dtype(jnp.float32)
# The score rows' width is a whole number of both phases' kv blocks.
SCORE_COLUMN_ALIGN = max(DECODE_BLOCK_KV, MAX_PREFILL_BLOCK_KV)


def index_cache_shape(
    num_pages: int, page_size: int
) -> tuple[int, int, int, int]:
    """The index-K cache's (num_pages, CACHE_GROUPS, WORD_BYTES, page_size)
    uint8 shape: tokens along lanes, byte i of group g < KEY_GROUPS a token's
    fp8 key dim 4g + i, group SCALE_GROUP its UE8M0 scale exponent in byte 0
    and zeros."""
    if page_size % PAGE_SIZE_MULTIPLE:
        raise ValueError(
            f"page_size={page_size} must be a multiple of {PAGE_SIZE_MULTIPLE}."
        )
    return (num_pages, CACHE_GROUPS, WORD_BYTES, page_size)


def check_index_cache(index_cache: jax.Array) -> None:
    """Fails unless index_cache has index_cache_shape and uint8."""
    if (
        index_cache.ndim != 4
        or index_cache.shape[1:3] != (CACHE_GROUPS, WORD_BYTES)
        or index_cache.shape[3] % PAGE_SIZE_MULTIPLE
    ):
        raise ValueError(
            "index_cache must be (num_pages, CACHE_GROUPS, WORD_BYTES,"
            f" page_size) with page_size a multiple of {PAGE_SIZE_MULTIPLE};"
            f" got {index_cache.shape}."
        )
    if index_cache.dtype != CACHE_DTYPE:
        raise ValueError(f"index_cache must be uint8; got {index_cache.dtype}.")


def score_columns(max_seqlen_k: int) -> int:
    """Columns of dsa_indexer_scores's rows for kv lengths up to
    max_seqlen_k."""
    return -(-max_seqlen_k // SCORE_COLUMN_ALIGN) * SCORE_COLUMN_ALIGN


def check_queries(queries: jax.Array, weights: jax.Array) -> None:
    """Fails unless queries and weights are (T, H, INDEX_HEAD_DIM)
    float8_e4m3fn and (T, H) f32 with H a multiple of INDEX_HEADS_MULTIPLE."""
    num_heads = weights.shape[-1]
    if (
        queries.shape[1:] != (num_heads, INDEX_HEAD_DIM)
        or weights.shape != (queries.shape[0], num_heads)
        or num_heads % INDEX_HEADS_MULTIPLE
    ):
        raise ValueError(
            "queries and weights must be (T, H, INDEX_HEAD_DIM) and (T, H)"
            f" with H a multiple of {INDEX_HEADS_MULTIPLE}; got"
            f" {queries.shape} and {weights.shape}."
        )
    if queries.dtype != FP8_DTYPE or weights.dtype != WEIGHT_DTYPE:
        raise ValueError(
            "queries and weights must be float8_e4m3fn and float32; got"
            f" {queries.dtype} and {weights.dtype}."
        )


def pages_fit_blocks(page_size: int, num_heads: int) -> bool:
    """Whether page_size divides or is a multiple of both phases' kv
    blocks."""
    return all(
        block_kv % page_size == 0 or page_size % block_kv == 0
        for block_kv in (DECODE_BLOCK_KV, prefill_block_kv(num_heads))
    )


@functools.partial(jax.jit, static_argnames=("max_seqlen_k",))
def dsa_indexer_scores(
    queries: jax.Array,
    weights: jax.Array,
    index_cache: jax.Array,
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    block_table: jax.Array,
    *,
    max_seqlen_k: int,
) -> jax.Array:
    """Index scores of a ragged chunk of tokens against their requests'
    cached keys:

        scores[t, s] = k_scale[s] * sum_h weights[t, h] * relu(q[t, h] . k[s])

    relu(q . k) and the weights are rounded to bf16 before the f32 head sum.
    Token i of a request with q_len tokens sees positions [0, kv_len - q_len
    + i + 1); only that prefix of its row is written. One call serves
    decode, chunked prefill and mixed batches.

    Args:
      queries: (T, H, INDEX_HEAD_DIM) float8_e4m3fn, H index heads a
        multiple of INDEX_HEADS_MULTIPLE, packed token-major: request r's at
        rows [cu_q_lens[r], cu_q_lens[r + 1]), the rows past cu_q_lens[B]
        padding.
      weights: (T, H) f32, weights_proj * H^-0.5 * q_scale *
        softmax_scale.
      index_cache: index_cache_shape(num_pages, page_size) uint8, its chunk
        tokens already written; page_size divides or is a multiple of
        DECODE_BLOCK_KV and prefill_block_kv(H).
      cu_q_lens: (B + 1,) int32 from 0, nondecreasing; a request of no tokens
        is inactive.
      kv_lens: (B,) int32 tokens of each request in the cache, its chunk the
        last q_len of them; at most max_seqlen_k.
      block_table: (B, pages_per_seq) int32 cache page of each request page;
        entries past cdiv(kv_lens, page_size) may hold anything.
      max_seqlen_k: a static bound on kv_lens, which sizes the rows.

    Returns:
      scores (T, score_columns(max_seqlen_k) / NUM_LANES, NUM_LANES) f32:
      token t's column s at [t, s // NUM_LANES, s % NUM_LANES]. Entries past
      a row's causal prefix, and padding rows, are unspecified. Each row is
      contiguous in HBM for top_k; a reshape to (T, columns) costs a
      relayout copy.
    """
    check_queries(queries, weights)
    check_index_cache(index_cache)
    check_requests(cu_q_lens, kv_lens, block_table)
    num_heads = weights.shape[-1]
    page_size = index_cache.shape[-1]
    capacity = block_table.shape[-1] * page_size
    if not pages_fit_blocks(page_size, num_heads):
        raise NotImplementedError(
            f"page_size={page_size} must divide or be a multiple of"
            f" {DECODE_BLOCK_KV} and {prefill_block_kv(num_heads)}."
        )
    if not 1 <= max_seqlen_k <= capacity:
        raise ValueError(
            f"max_seqlen_k must be in [1, the {capacity} tokens block_table"
            f" holds]; got {max_seqlen_k}."
        )
    return indexer_scores_pallas(
        queries,
        weights,
        index_cache,
        cu_q_lens,
        kv_lens,
        block_table,
        num_columns=score_columns(max_seqlen_k),
    )
