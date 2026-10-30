"""write_kv_cache_pages: new K/V rows into the merged interleaved paged KV
cache, for callers that keep the cache themselves (tpu-inference's
prefill)."""

import functools

import jax

from .kernels.kv_cache_write import kv_cache_write_pallas
from .kv_layout import check_merged_kv_dtype, merged_cache_heads


@functools.partial(
    jax.jit, static_argnames=("interpret",), donate_argnames=("kv_cache",)
)
def write_kv_cache_pages(
    k: jax.Array,
    v: jax.Array,
    kv_cache: jax.Array,
    slices: jax.Array,
    num_slices: jax.Array,
    *,
    interpret: bool = False,
) -> jax.Array:
    """Copy new K/V rows into the merged interleaved cache; returns the cache,
    updated in place (it is donated).

    k and v are (T, H, D) and kv_cache is (P, S, merged_cache_heads(H), D), all
    of one dtype: bf16, or fp8 already quantized by the caller. slices is
    i32[3, n] and num_slices i32[1]: column j < num_slices[0] of slices is
    (cache_row_start, new_row_start, length) in rows of the flat (P * S) cache
    view and of k / v, each piece inside one page; later columns are ignored.
    """
    _, num_kv_heads, head_dim = k.shape
    _, _, cache_heads, cache_head_dim = kv_cache.shape
    check_merged_kv_dtype(kv_cache.dtype)
    if k.dtype != kv_cache.dtype or v.dtype != kv_cache.dtype:
        raise NotImplementedError(
            f"k, v and the cache must share one dtype; got {k.dtype},"
            f" {v.dtype}, {kv_cache.dtype}."
        )
    if v.shape != k.shape or (cache_heads, cache_head_dim) != (
        merged_cache_heads(num_kv_heads, kv_cache.dtype),
        head_dim,
    ):
        raise ValueError(
            "k and v must be (T, H, D) of a (P, S, merged_cache_heads(H), D)"
            f" cache; got {k.shape=}, {v.shape=}, {kv_cache.shape=}."
        )
    if head_dim % 128:
        raise ValueError(f"head_dim must be a multiple of 128; got {head_dim}.")
    return kv_cache_write_pallas(
        k, v, kv_cache, slices, num_slices, interpret=interpret
    )
