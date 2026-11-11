"""The split fp8 latent cache of DSA sparse MLA and its ragged write.

A token's 512 fp8 nope dims are one 128-word int32 row, byte b of word w
holding dim 128 b + w; its 64 fp8 rope dims are 16 words banded the same way
(byte b of word w is dim 16 b + w), 8 tokens a 128-word row.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax

from flywheel_tpu.softmax_attention.kernels.mla_decode import (
    LATENT_DIM,
    ROPE_DIM,
)
from flywheel_tpu.sparse_attention.dsa.kernels.cache_write import (
    dsa_cache_write_pallas,
)
from flywheel_tpu.sparse_attention.dsa.kernels.gather import (
    IGNORED_ROW,
    NOPE_WORDS,
    ROPE_TOKENS_PER_ROW,
    ROPE_WORDS,
)

FP8_CACHE_DTYPE = jnp.dtype(jnp.float8_e4m3fn)
CACHE_WORD_DTYPE = jnp.dtype(jnp.int32)
BYTES_PER_WORD = 4


def dsa_cache_shapes(
    num_pages: int, page_size: int
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """(nope, rope) int32 cache shapes: (num_pages, page_size, 128) and
    (num_pages, page_size / 8, 128)."""
    check_page_size(page_size)
    return (
        (num_pages, page_size, NOPE_WORDS),
        (num_pages, page_size // ROPE_TOKENS_PER_ROW, NOPE_WORDS),
    )


def check_page_size(page_size: int) -> None:
    """Fails unless page_size is a power of two that holds whole rope rows."""
    if page_size < ROPE_TOKENS_PER_ROW or page_size & (page_size - 1):
        raise ValueError(
            "page_size must be a power of two of at least"
            f" {ROPE_TOKENS_PER_ROW} tokens; got {page_size}."
        )


def check_caches(nope_cache: jax.Array, rope_cache: jax.Array) -> int:
    """Fails unless the caches are dsa_cache_shapes int32 arrays; returns
    their page size."""
    num_pages, page_size = nope_cache.shape[:2]
    if {nope_cache.dtype, rope_cache.dtype} != {CACHE_WORD_DTYPE} or (
        nope_cache.shape,
        rope_cache.shape,
    ) != dsa_cache_shapes(num_pages, page_size):
        raise ValueError(
            "The caches must be int32 dsa_cache_shapes(num_pages, page_size);"
            f" got {nope_cache.dtype} {nope_cache.shape} and"
            f" {rope_cache.dtype} {rope_cache.shape}."
        )
    return page_size


def check_requests(
    cu_q_lens: jax.Array, kv_lens: jax.Array, block_table: jax.Array
) -> None:
    """Fails unless cu_q_lens, kv_lens and block_table are int32 (B + 1,),
    (B,) and (B, pages_per_seq) with B >= 1."""
    batch = block_table.shape[0]
    if (
        block_table.ndim != 2
        or batch == 0
        or cu_q_lens.shape != (batch + 1,)
        or kv_lens.shape != (batch,)
    ):
        raise ValueError(
            "block_table, cu_q_lens and kv_lens must be (B, pages_per_seq),"
            f" (B + 1,) and (B,) with B >= 1; got {block_table.shape},"
            f" {cu_q_lens.shape} and {kv_lens.shape}."
        )
    elif {cu_q_lens.dtype, kv_lens.dtype, block_table.dtype} != {
        jnp.dtype(jnp.int32)
    }:
        raise ValueError(
            "cu_q_lens, kv_lens and block_table must be int32; got"
            f" {cu_q_lens.dtype}, {kv_lens.dtype} and {block_table.dtype}."
        )


def token_positions(
    cu_q_lens: jax.Array, kv_lens: jax.Array, num_tokens: int
) -> tuple[jax.Array, jax.Array]:
    """(request, position) of each of num_tokens packed tokens, (T,) int32.
    Padding rows are request 0 at position -1."""
    batch = kv_lens.shape[0]
    token = jnp.arange(num_tokens, dtype=jnp.int32)
    # Note (david): counting the requests that end by a token avoids a gather.
    request = jnp.sum(
        cu_q_lens[None, 1:] <= token[:, None], axis=1, dtype=jnp.int32
    )
    is_padding = request >= batch
    request = jnp.where(is_padding, 0, request)
    position = (
        token_values(cu_q_lens, kv_lens - cu_q_lens[1:], num_tokens) + token
    )
    return request, jnp.where(is_padding, -1, position)


def token_values(
    cu_q_lens: jax.Array, values: jax.Array, num_tokens: int
) -> jax.Array:
    """(T,) int32: each token's entry of its request in the (B,) int32
    values, the last request's for padding rows."""
    # Note (david): a gather from a short table lowers to one select pass per
    # entry, so the value is the first entry plus the steps before each token.
    token = jnp.arange(num_tokens, dtype=jnp.int32)
    is_past = cu_q_lens[None, 1:-1] <= token[:, None]
    steps = values[1:] - values[:-1]
    return values[0] + jnp.sum(
        jnp.where(is_past, steps[None, :], 0), axis=1, dtype=jnp.int32
    )


def band_words(values: jax.Array, num_words: int) -> jax.Array:
    """fp8 (T, 4 num_words) token rows as (T, num_words) int32 banded words:
    byte b of word w is dim num_words b + w."""
    num_tokens = values.shape[0]
    bands = values.reshape(num_tokens, BYTES_PER_WORD, num_words)
    return lax.bitcast_convert_type(bands.transpose(0, 2, 1), CACHE_WORD_DTYPE)


# Note (david): the caches are donated so that the write aliases the caller's
# buffers instead of copies of the whole pool; under an outer jax.jit, donate
# them on that jit instead.
@functools.partial(jax.jit, donate_argnames=("nope_cache", "rope_cache"))
def dsa_cache_write(
    new_nope: jax.Array,
    new_rope: jax.Array,
    nope_cache: jax.Array,
    rope_cache: jax.Array,
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    block_table: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Writes each request's chunk of new latent tokens into its pages of the
    split cache; every other cache word keeps its value.

    Args:
      new_nope: (T, LATENT_DIM) float8_e4m3fn latents, quantized by the
        caller with the cache's kv_scale, packed token-major: request r's at
        rows [cu_q_lens[r], cu_q_lens[r + 1]), the rows past cu_q_lens[B]
        padding.
      new_rope: (T, ROPE_DIM) float8_e4m3fn ropes, likewise.
      nope_cache, rope_cache: dsa_cache_shapes(num_pages, page_size) int32;
        updated in place.
      cu_q_lens: (B + 1,) int32 cumulative chunk lengths from 0.
      kv_lens: (B,) int32 tokens of each request after the write, its chunk at
        [kv_lens - q_len, kv_lens).
      block_table: (B, pages_per_seq) int32 cache page of each request page;
        entries past cdiv(kv_lens, page_size) may hold anything.

    Returns:
      (nope_cache, rope_cache) with the chunks written.
    """
    num_tokens = new_nope.shape[0]
    page_size = check_caches(nope_cache, rope_cache)
    check_requests(cu_q_lens, kv_lens, block_table)
    if new_nope.shape != (num_tokens, LATENT_DIM) or new_rope.shape != (
        num_tokens,
        ROPE_DIM,
    ):
        raise ValueError(
            f"new_nope and new_rope must be (T, {LATENT_DIM}) and (T,"
            f" {ROPE_DIM}); got {new_nope.shape} and {new_rope.shape}."
        )
    elif {new_nope.dtype, new_rope.dtype} != {FP8_CACHE_DTYPE}:
        raise NotImplementedError(
            "dsa_cache_write takes float8_e4m3fn latents only; got"
            f" {new_nope.dtype} and {new_rope.dtype}."
        )
    request, position = token_positions(cu_q_lens, kv_lens, num_tokens)
    page = block_table[request, jnp.maximum(position, 0) // page_size]
    rows = jnp.where(
        position >= 0, page * page_size + position % page_size, IGNORED_ROW
    )
    nope_rows, rope_rows = dsa_cache_write_pallas(
        rows,
        band_words(new_nope, NOPE_WORDS),
        band_words(new_rope, ROPE_WORDS),
        nope_cache.reshape(-1, NOPE_WORDS),
        rope_cache.reshape(-1, NOPE_WORDS),
    )
    return (
        nope_rows.reshape(nope_cache.shape),
        rope_rows.reshape(rope_cache.shape),
    )
