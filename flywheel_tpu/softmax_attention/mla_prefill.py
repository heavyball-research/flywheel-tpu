"""Absorbed MLA chunked prefill on a paged latent cache."""

import functools

import jax
import jax.numpy as jnp

from .kernels.mla_decode import (
    CACHE_ROW_DIM,
    LATENT_DIM,
    ROPE_DIM,
    cache_packing,
    cache_tile_rows,
)
from .kernels.mla_prefill import default_mla_prefill_config, mla_prefill_pallas
from .mla_decode import CACHE_DTYPES, FP8_CACHE_DTYPE, QUERY_DTYPE


@functools.partial(
    jax.jit,
    static_argnames=(
        "sm_scale",
        "causal",
        "return_lse",
        "max_seqlen_q",
        "max_seqlen_k",
        "interpret",
    ),
)
def flash_mla_prefill(
    q_nope: jax.Array,
    q_pe: jax.Array,
    kv_cache: jax.Array,
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    block_table: jax.Array,
    *,
    sm_scale: float,
    kv_scale: jax.Array | None = None,
    causal: bool = True,
    return_lse: bool = False,
    max_seqlen_q: int,
    max_seqlen_k: int,
    interpret: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Attends a ragged chunk of queries to their requests' cache as MQA with
    one key head of the latent and the rope, and the latent as the value. The
    chunk's own latents must already be in the cache.

    Args:
      q_nope: (T, H, LATENT_DIM) bf16 queries with W_UK absorbed, packed
        token-major; H is 8, 16, 32, 64 or 128.
      q_pe: (T, H, ROPE_DIM) bf16 rope queries.
      kv_cache: mla_cache_shape(num_pages, page_size, dtype), bf16 or
        float8_e4m3fn, with page_size whole cache tiles of the dtype; read
        only.
      cu_q_lens: (B + 1,) int32 from 0, nondecreasing: request r's queries
        are rows [cu_q_lens[r], cu_q_lens[r + 1]); the rows past cu_q_lens[B]
        are padding, whose outputs are unspecified.
      kv_lens: (B,) int32 tokens of each request in the cache, its chunk the
        last q_len of them.
      block_table: (B, pages_per_seq) int32 cache page of each request page;
        entries past cdiv(kv_lens, page_size) may hold anything.
      sm_scale: scale of q_nope . latent + q_pe . rope.
      kv_scale: f32 scalar of an fp8 cache, whose real values are fp8 *
        kv_scale, or None for a bf16 cache. It is traced, so a new scale
        compiles no new kernel.
      causal: chunk token i sees the positions up to kv_lens - q_len + i;
        without it every token sees all kv_lens positions.
      return_lse: also return the natural-log lse.
      max_seqlen_q: a static bound on the q_lens, which sizes the q blocks and
        the schedule.
      max_seqlen_k: a static bound on kv_lens, which sizes the kv blocks.
      interpret: run the Pallas interpreter.

    Returns:
      out (T, H, LATENT_DIM) bf16, or (out, lse) with return_lse, lse (H, T)
      f32, +inf for a row that sees no key.
    """
    num_tokens, num_heads, _ = q_nope.shape
    batch = kv_lens.shape[0]
    cache_dtype = jnp.dtype(kv_cache.dtype)
    shapes_match = (
        q_pe.shape == (num_tokens, num_heads, ROPE_DIM)
        and q_nope.shape[-1] == LATENT_DIM
        and cu_q_lens.shape == (batch + 1,)
        and block_table.ndim == 2
        and block_table.shape[0] == batch
    )
    tile_rows = cache_tile_rows(cache_dtype)
    is_cache_shape = kv_cache.ndim == 4 and kv_cache.shape[2:] == (
        cache_packing(cache_dtype),
        CACHE_ROW_DIM,
    )
    page_size = kv_cache.shape[1] * kv_cache.shape[2] if is_cache_shape else 0
    if not shapes_match:
        raise ValueError(
            "flash_mla_prefill takes q_nope (T, H, LATENT_DIM), q_pe (T, H,"
            " ROPE_DIM), cu_q_lens (B + 1,), kv_lens (B,) and block_table (B,"
            f" pages_per_seq); got {q_nope.shape}, {q_pe.shape},"
            f" {cu_q_lens.shape}, {kv_lens.shape} and {block_table.shape}."
        )
    elif num_heads not in (8, 16, 32, 64, 128):
        raise NotImplementedError(
            "flash_mla_prefill supports 8, 16, 32, 64 or 128 query heads; got"
            f" {num_heads}."
        )
    elif {q_nope.dtype, q_pe.dtype} != {QUERY_DTYPE}:
        raise NotImplementedError(
            "flash_mla_prefill supports bfloat16 queries only; got"
            f" {q_nope.dtype} and {q_pe.dtype}."
        )
    elif {cu_q_lens.dtype, kv_lens.dtype, block_table.dtype} != {
        jnp.dtype(jnp.int32)
    }:
        raise ValueError(
            "cu_q_lens, kv_lens and block_table must be int32; got"
            f" {cu_q_lens.dtype}, {kv_lens.dtype} and {block_table.dtype}."
        )
    elif cache_dtype not in CACHE_DTYPES:
        raise NotImplementedError(
            "flash_mla_prefill supports bfloat16 and float8_e4m3fn caches"
            f" only; got {cache_dtype}."
        )
    elif not is_cache_shape or page_size % tile_rows:
        raise ValueError(
            f"A {cache_dtype} kv_cache must be mla_cache_shape(num_pages,"
            f" page_size, dtype) with page_size a multiple of {tile_rows};"
            f" got {kv_cache.shape}."
        )
    elif (cache_dtype == FP8_CACHE_DTYPE) != (kv_scale is not None):
        raise ValueError(
            "kv_scale must be given for an fp8 cache and only for it; the"
            f" cache is {cache_dtype}."
        )
    elif not 1 <= max_seqlen_q <= num_tokens:
        raise ValueError(
            f"max_seqlen_q must be in [1, T={num_tokens}]; got {max_seqlen_q}."
        )
    elif not 1 <= max_seqlen_k <= block_table.shape[1] * page_size:
        raise ValueError(
            "max_seqlen_k must be in [1, the"
            f" {block_table.shape[1] * page_size} tokens block_table holds];"
            f" got {max_seqlen_k}."
        )
    else:
        config = default_mla_prefill_config(
            num_heads=num_heads,
            page_size=page_size,
            pages_per_seq=block_table.shape[1],
            cache_dtype=cache_dtype,
            return_lse=return_lse,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
        )
        return mla_prefill_pallas(
            q_nope,
            q_pe,
            kv_cache,
            cu_q_lens,
            kv_lens,
            block_table,
            kv_scale=kv_scale,
            sm_scale=sm_scale,
            causal=causal,
            return_lse=return_lse,
            max_seqlen_q=max_seqlen_q,
            interpret=interpret,
            config=config,
        )
