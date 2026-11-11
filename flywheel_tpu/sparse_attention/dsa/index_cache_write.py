"""Ragged write of new keys into the paged DSA index-K cache."""

import functools

import jax
import jax.numpy as jnp

from flywheel_tpu.sparse_attention.dsa.cache_write import check_requests
from flywheel_tpu.sparse_attention.dsa.indexer import (
    INDEX_HEAD_DIM,
    check_index_cache,
)
from flywheel_tpu.sparse_attention.dsa.kernels.index_cache_write import (
    index_cache_write_pallas,
)

KEY_DTYPE = jnp.dtype(jnp.bfloat16)


# Note (david): the cache is donated so that the write aliases the caller's
# buffer instead of a copy of the whole pool; under an outer jax.jit, donate
# it on that jit instead.
@functools.partial(jax.jit, donate_argnames=("index_cache",))
def dsa_index_cache_write(
    keys: jax.Array,
    index_cache: jax.Array,
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    block_table: jax.Array,
) -> jax.Array:
    """Quantizes each request's chunk of new index keys and writes them into
    its pages: a token's key becomes float8_e4m3fn under the UE8M0 scale of
    its max abs, at its cache lane.

    Every other byte of the cache keeps its value. HBM traffic follows the
    chunks: only the 128-lane tiles a chunk starts or ends inside are read.

    Args:
      keys: (T, INDEX_HEAD_DIM) bf16 Hadamard-rotated keys, token-major
        packed: request r's at rows [cu_q_lens[r], cu_q_lens[r + 1]), the rows
        past cu_q_lens[B] padding.
      index_cache: index_cache_shape(num_pages, page_size) uint8; updated in
        place.
      cu_q_lens: (B + 1,) int32 cumulative chunk lengths from 0; a request of
        none is inactive and writes nothing.
      kv_lens: (B,) int32 tokens of each request after the write, its chunk at
        [kv_lens - q_len, kv_lens).
      block_table: (B, pages_per_seq) int32 cache page of each request page;
        entries past cdiv(kv_lens, page_size) may hold anything.

    Returns:
      index_cache with the chunks written.
    """
    check_index_cache(index_cache)
    check_requests(cu_q_lens, kv_lens, block_table)
    if keys.ndim != 2 or keys.shape[1] != INDEX_HEAD_DIM:
        raise ValueError(
            f"keys must be (T, {INDEX_HEAD_DIM}); got {keys.shape}."
        )
    if keys.dtype != KEY_DTYPE:
        raise ValueError(f"keys must be bfloat16; got {keys.dtype}.")
    return index_cache_write_pallas(
        keys, index_cache, cu_q_lens, kv_lens, block_table
    )
