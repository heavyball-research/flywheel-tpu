"""Ragged write of prefill chunks into the paged MLA latent cache."""

import functools

import jax
import jax.numpy as jnp

from .kernels.mla_cache_write import mla_cache_write_pallas
from .kernels.mla_decode import (
    CACHE_ROW_DIM,
    LATENT_DIM,
    ROPE_DIM,
    cache_packing,
    cache_tile_rows,
)
from .mla_decode import CACHE_DTYPES

# Note (david): a group's copies overlap the group before; a short group hides
# little and a long one lengthens the unhidden first reads and last writes.
GROUP_TOKENS = 1024


# Note (david): the cache is donated so that the write aliases the caller's
# buffer instead of a copy of the whole pool; under an outer jax.jit, donate
# it on that jit instead.
@functools.partial(
    jax.jit, static_argnames=("interpret",), donate_argnames=("kv_cache",)
)
def mla_cache_write(
    new_kv_c: jax.Array,
    new_k_pe: jax.Array,
    kv_cache: jax.Array,
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    block_table: jax.Array,
    *,
    interpret: bool = False,
) -> jax.Array:
    """Writes each request's chunk of new latent tokens into its pages, each
    token's cache row its latent, its rope and zeros.

    Every other bit of the cache keeps its value. HBM traffic follows the
    chunks' tokens: only the cache tiles a chunk starts or ends inside are
    read, to keep their other tokens.

    Args:
      new_kv_c: (T, LATENT_DIM) new latents in the cache's dtype, token-major
        packed: request r's at rows [cu_q_lens[r], cu_q_lens[r + 1]), the
        rows past cu_q_lens[B] padding. A T of whole cache tiles is read in
        place; any other T is padded to them first, a copy of the new tokens.
      new_k_pe: (T, ROPE_DIM) new ropes, likewise.
      kv_cache: mla_cache_shape(num_pages, page_size, dtype), bf16 or
        float8_e4m3fn, with page_size whole cache tiles; updated in place.
      cu_q_lens: (B + 1,) int32 cumulative chunk lengths from 0; a request of
        none is inactive and writes nothing.
      kv_lens: (B,) int32 tokens of each request after the write, its chunk at
        [kv_lens - q_len, kv_lens).
      block_table: (B, pages_per_seq) int32 cache page of each request page;
        entries past cdiv(kv_lens, page_size) may hold anything.
      interpret: run the Pallas interpreter.

    Returns:
      kv_cache with the chunks written.
    """
    cache_dtype = jnp.dtype(kv_cache.dtype)
    packing = cache_packing(cache_dtype)
    tile_rows = cache_tile_rows(cache_dtype)
    num_tokens = new_kv_c.shape[0]
    batch = block_table.shape[0]
    if cache_dtype not in CACHE_DTYPES:
        raise NotImplementedError(
            "mla_cache_write supports bfloat16 and float8_e4m3fn caches only;"
            f" got {cache_dtype}."
        )
    elif (
        kv_cache.ndim != 4
        or kv_cache.shape[2:] != (packing, CACHE_ROW_DIM)
        or (kv_cache.shape[1] * packing) % tile_rows
    ):
        raise ValueError(
            f"A {cache_dtype} kv_cache must be (num_pages, page_size /"
            f" {packing}, {packing}, {CACHE_ROW_DIM}) with page_size a multiple"
            f" of {tile_rows}; got {kv_cache.shape}."
        )
    elif new_kv_c.shape != (num_tokens, LATENT_DIM) or new_k_pe.shape != (
        num_tokens,
        ROPE_DIM,
    ):
        raise ValueError(
            f"new_kv_c and new_k_pe must be (T, {LATENT_DIM}) and (T,"
            f" {ROPE_DIM}); got {new_kv_c.shape} and {new_k_pe.shape}."
        )
    elif {new_kv_c.dtype, new_k_pe.dtype} != {cache_dtype}:
        raise ValueError(
            "new_kv_c and new_k_pe must be in the cache's dtype"
            f" {cache_dtype}; got {new_kv_c.dtype} and {new_k_pe.dtype}."
        )
    elif (
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
    else:
        return mla_cache_write_pallas(
            new_kv_c,
            new_k_pe,
            kv_cache,
            cu_q_lens,
            kv_lens,
            block_table,
            group_tokens=GROUP_TOKENS,
            interpret=interpret,
        )
