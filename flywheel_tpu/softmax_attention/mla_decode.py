"""Absorbed MLA decode on a paged latent cache."""

import functools

import jax
import jax.numpy as jnp

from .kernels.mla_decode import (
    CACHE_ROW_DIM,
    LATENT_DIM,
    ROPE_DIM,
    cache_packing,
    cache_tile_rows,
    default_mla_config,
    mla_decode_pallas,
)

SCORE_DTYPES = (jnp.dtype(jnp.float32), jnp.dtype(jnp.bfloat16))
QUERY_DTYPE = jnp.dtype(jnp.bfloat16)
# Note (david): float8_e4m3fn is MLA v2's fp8 KV dtype, and its extra mantissa
# bit suits the latent better than e5m2.
FP8_CACHE_DTYPE = jnp.dtype(jnp.float8_e4m3fn)
CACHE_DTYPES = (QUERY_DTYPE, FP8_CACHE_DTYPE)
# Note (david): the append's code grows with s_q, and the s_q * H query rows
# grow every item's score temporaries, so the kernel is checked up to
# MAX_QUERY_TOKENS only. It also keeps a request's new tokens within one
# cache tile's length, so they span at most two pages.
MAX_QUERY_TOKENS = 16


def mla_cache_shape(
    num_pages: int, page_size: int, dtype: jnp.dtype
) -> tuple[int, int, int, int]:
    """flash_mla_decode's cache shape in MLA v2's word layout, (num_pages,
    page_size / packing, packing, CACHE_ROW_DIM): a row-major reshape of
    (num_pages, page_size, CACHE_ROW_DIM) with packing tokens a 32-bit
    word."""
    packing = cache_packing(dtype)
    if page_size % packing:
        raise ValueError(
            f"page_size={page_size} must hold whole {packing}-token words."
        )
    return (num_pages, page_size // packing, packing, CACHE_ROW_DIM)


# Note (david): the cache is donated so that the in-place append aliases the
# caller's buffer instead of a copy of the whole pool; under an outer jax.jit,
# donate it on that jit instead.
@functools.partial(
    jax.jit,
    static_argnames=(
        "sm_scale",
        "causal",
        "s_dtype",
        "return_lse",
        "interpret",
        "max_seqlen_k",
    ),
    donate_argnames=("kv_cache",),
)
def flash_mla_decode(
    q_nope: jax.Array,
    q_pe: jax.Array,
    new_kv_c: jax.Array,
    new_k_pe: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    block_table: jax.Array,
    *,
    sm_scale: float,
    kv_scale: jax.Array | None = None,
    causal: bool = True,
    s_dtype: jnp.dtype = jnp.float32,
    return_lse: bool = False,
    interpret: bool = False,
    max_seqlen_k: int | None = None,
) -> tuple[jax.Array, ...]:
    """Appends s_q new latent tokens per request (1 <= s_q <=
    MAX_QUERY_TOKENS), then attends each request's cache as MQA with one key
    head of the latent and the rope, and the latent as the value.

    Each request anchors its softmax on the max score of its first lane tile
    of keys; a score far past its anchor makes the call replay every request
    with the online softmax. The probabilities round to the cache's dtype for
    P @ V, as MLA v2's do.

    Args:
      q_nope: (B, s_q, H, LATENT_DIM) bf16 queries with W_UK absorbed.
      q_pe: (B, s_q, H, ROPE_DIM) bf16 rope queries.
      new_kv_c: (B, s_q, LATENT_DIM) new latents in the cache's dtype,
        quantized by the caller with kv_scale for an fp8 cache.
      new_k_pe: (B, s_q, ROPE_DIM) new ropes, likewise.
      kv_cache: mla_cache_shape(num_pages, page_size, dtype), bf16 or
        float8_e4m3fn, with page_size whole VMEM tiles of the dtype; updated
        in place.
      kv_lens: (B,) int32 tokens of each request after the append, the new
        ones at [kv_lens - s_q, kv_lens); 0 marks an inactive request, which
        appends nothing and gets a zero output and an lse of +inf.
      block_table: (B, pages_per_seq) int32 cache page of each request page;
        entries past cdiv(kv_lens, page_size) may hold anything.
      sm_scale: scale of q_nope . latent + q_pe . rope.
      kv_scale: f32 scalar of an fp8 cache, whose real values are fp8 *
        kv_scale, or None for a bf16 cache. It is traced, so a new scale
        compiles no new kernel.
      causal: query token i sees the positions before kv_lens - s_q + i + 1;
        without it every query token sees all kv_lens positions.
      s_dtype: f32 or bf16, the score dtype of the online-softmax replay.
      return_lse: also return the natural-log lse.
      interpret: run the Pallas interpreter.
      max_seqlen_k: a static bound on kv_lens, as serving knows it, which the
        config sizes its KV blocks by; None bounds them by block_table's
        pages. Only the config depends on it: a kv_lens past it still attends
        every token.

    Returns:
      (out, kv_cache), or (out, kv_cache, lse) with return_lse: out
      (B, s_q, H, LATENT_DIM) bf16 and lse (B, H, s_q) f32.
    """
    batch, query_tokens, num_heads, _ = q_nope.shape
    cache_dtype = jnp.dtype(kv_cache.dtype)
    packing = cache_packing(cache_dtype)
    for name, shape, expected in (
        ("q_nope", q_nope.shape, (batch, query_tokens, num_heads, LATENT_DIM)),
        ("q_pe", q_pe.shape, (batch, query_tokens, num_heads, ROPE_DIM)),
        ("new_kv_c", new_kv_c.shape, (batch, query_tokens, LATENT_DIM)),
        ("new_k_pe", new_k_pe.shape, (batch, query_tokens, ROPE_DIM)),
        ("kv_lens", kv_lens.shape, (batch,)),
    ):
        if shape != expected:
            raise ValueError(f"{name} must be {expected}; got {shape}.")
    if jnp.dtype(s_dtype) not in SCORE_DTYPES:
        raise ValueError(f"s_dtype must be f32 or bf16; got {s_dtype!r}.")
    if cache_dtype not in CACHE_DTYPES:
        raise NotImplementedError(
            "flash_mla_decode supports bfloat16 and float8_e4m3fn caches only;"
            f" got {cache_dtype}."
        )
    if {q_nope.dtype, q_pe.dtype} != {QUERY_DTYPE}:
        raise NotImplementedError(
            "flash_mla_decode supports bfloat16 queries only; got"
            f" {q_nope.dtype} and {q_pe.dtype}."
        )
    if {new_kv_c.dtype, new_k_pe.dtype} != {cache_dtype}:
        raise ValueError(
            "new_kv_c and new_k_pe must be in the cache's dtype"
            f" {cache_dtype}; got {new_kv_c.dtype} and {new_k_pe.dtype}."
        )
    if (cache_dtype == FP8_CACHE_DTYPE) != (kv_scale is not None):
        raise ValueError(
            "kv_scale must be given for an fp8 cache and only for it; the"
            f" cache is {cache_dtype}."
        )
    tile_rows = cache_tile_rows(cache_dtype)
    if (
        kv_cache.ndim != 4
        or kv_cache.shape[2:] != (packing, CACHE_ROW_DIM)
        or (kv_cache.shape[1] * packing) % tile_rows
    ):
        raise ValueError(
            f"A {cache_dtype} kv_cache must be (num_pages, page_size /"
            f" {packing}, {packing}, {CACHE_ROW_DIM}) with page_size a multiple"
            f" of {tile_rows}; got {kv_cache.shape}."
        )
    if block_table.ndim != 2 or block_table.shape[0] != batch:
        raise ValueError(
            f"block_table must be (B, pages_per_seq) with B={batch}; got"
            f" {block_table.shape}."
        )
    if not 1 <= query_tokens <= MAX_QUERY_TOKENS:
        raise NotImplementedError(
            f"flash_mla_decode supports 1 to {MAX_QUERY_TOKENS} query tokens"
            f" per request; got s_q={query_tokens}."
        )
    page_size = kv_cache.shape[1] * packing
    capacity = block_table.shape[1] * page_size
    if max_seqlen_k is None:
        max_seqlen_k = capacity
    elif not query_tokens <= max_seqlen_k <= capacity:
        raise ValueError(
            f"max_seqlen_k must be in [s_q, the {capacity} tokens block_table"
            f" holds]; got {max_seqlen_k} at s_q={query_tokens}."
        )
    config = default_mla_config(
        batch=batch,
        page_size=page_size,
        pages_per_seq=block_table.shape[1],
        num_query_rows=query_tokens * num_heads,
        query_tokens=query_tokens,
        cache_dtype=cache_dtype,
        return_lse=return_lse,
        max_seqlen_k=max_seqlen_k,
    )
    return mla_decode_pallas(
        q_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        kv_cache,
        kv_lens,
        block_table,
        kv_scale=kv_scale,
        sm_scale=sm_scale,
        s_dtype=jnp.dtype(s_dtype),
        # Note (david): a single query token sits at the end of its cache, so
        # causal masks nothing there and both settings run the same kernel.
        causal=causal and query_tokens > 1,
        return_lse=return_lse,
        interpret=interpret,
        config=config,
    )
