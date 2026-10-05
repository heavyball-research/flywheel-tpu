"""Packed multi-token KV-cache attention: a ragged append, then the kv-outer
extend kernel over the updated pages."""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .block_sizes import vmem_limit_bytes
from .flash_fwd_kvcache import SUPPORTED_HEAD_DIMS
from .flash_fwd_kvcache_extend import flash_attn_kvcache_extend_pallas

# Note (david): the extend kernel reads whole pages of a multiple of 128
# tokens, so a contiguous cache row is viewed as a run of 128-token virtual
# pages (same bytes).
VIRTUAL_PAGE_SIZE = 128


def append_kernel(
    cu_seqlens_ref: jax.Array,
    cache_seqlens_ref: jax.Array,
    block_table_ref: jax.Array,
    num_active_ref: jax.Array,
    *refs: jax.Array,
    page_size: int,
    pages_per_seq: int,
    is_merged: bool,
) -> None:
  # Note (david): each part is (new-token rows, VMEM bounce buffer, cache
  # rows); a merged cache has one part whose rows already hold a token's K
  # heads then its V heads.
  if is_merged:
    kv_ref, _, kv_out_ref, kv_vmem, dma_sems = refs
    parts = ((kv_ref, kv_vmem, kv_out_ref),)
  else:
    (k_ref, v_ref, _, _, key_out_ref, value_out_ref, key_vmem, value_vmem,
     dma_sems) = refs
    parts = ((k_ref, key_vmem, key_out_ref), (v_ref, value_vmem, value_out_ref))

  def _append_request(request, _):
    query_start = cu_seqlens_ref[request]
    query_end = cu_seqlens_ref[request + 1]

    @pl.when(query_end > query_start)
    def _append_tokens():
      prefix_len = cache_seqlens_ref[request]
      first_page = prefix_len // page_size
      end_page = pl.cdiv(prefix_len + query_end - query_start, page_size)

      def _copy_page(logical_page, _):
        first_token = jnp.maximum(prefix_len, logical_page * page_size)
        end_token = jnp.minimum(
            prefix_len + query_end - query_start,
            (logical_page + 1) * page_size,
        )
        num_tokens = end_token - first_token
        source_start = query_start + first_token - prefix_len
        physical_page = block_table_ref[request * pages_per_seq + logical_page]
        destination_start = physical_page * page_size + first_token % page_size
        incoming = [
            pltpu.make_async_copy(
                source_ref.at[pl.ds(source_start, num_tokens)],
                vmem.at[pl.ds(0, num_tokens)],
                dma_sems.at[index],
            )
            for index, (source_ref, vmem, _) in enumerate(parts)
        ]
        for copy in incoming:
          copy.start()
        for copy in incoming:
          copy.wait()
        outgoing = [
            pltpu.make_async_copy(
                vmem.at[pl.ds(0, num_tokens)],
                destination_ref.at[pl.ds(destination_start, num_tokens)],
                dma_sems.at[index],
            )
            for index, (_, vmem, destination_ref) in enumerate(parts)
        ]
        for copy in outgoing:
          copy.start()
        for copy in outgoing:
          copy.wait()

      lax.fori_loop(first_page, end_page, _copy_page, None)

  lax.fori_loop(0, num_active_ref[0], _append_request, None)


def append_ragged(
    k_cache: jax.Array,
    v_cache: jax.Array | None,
    k: jax.Array,
    v: jax.Array,
    cu_seqlens_q: jax.Array,
    cache_seqlens: jax.Array,
    block_table: jax.Array,
    num_active: jax.Array,
    *,
    page_size: int,
    pages_per_seq: int,
    interpret: bool,
    merged_cache: bool = False,
) -> tuple[jax.Array, ...]:
  caches = (k_cache,) if merged_cache else (k_cache, v_cache)
  # Note (david): a token-row view that leaves the (heads, head_dim) tiling
  # untouched, so XLA emits no copy.
  flat_caches = tuple(cache.reshape(-1, *cache.shape[-2:]) for cache in caches)
  if merged_cache:
    # Note (david): the new tokens become whole (K heads, V heads) rows and
    # are copied like the other layouts' rows. Copying each part into its
    # head block of the page instead would slice the head axis in HBM, the
    # access that gave wrong attention in the serving step at TP 2/4 when the
    # extend kernel read that way; with one KV head that slice is not even
    # tile-aligned.
    new_tokens = (jnp.concatenate([k, v], axis=1),)
  else:
    new_tokens = (k, v)
  hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
  scalar_prefetches = (cu_seqlens_q, cache_seqlens, block_table, num_active)
  call = pl.pallas_call(
      functools.partial(append_kernel, page_size=page_size,
                        pages_per_seq=pages_per_seq, is_merged=merged_cache),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=len(scalar_prefetches), grid=(1,),
          in_specs=(hbm_spec,) * (len(new_tokens) + len(flat_caches)),
          out_specs=(hbm_spec,) * len(flat_caches),
          scratch_shapes=(
              *(pltpu.VMEM((page_size, *rows.shape[1:]), rows.dtype)
                for rows in new_tokens),
              pltpu.SemaphoreType.DMA((len(new_tokens),)),
          ),
      ),
      out_shape=tuple(
          jax.ShapeDtypeStruct(cache.shape, cache.dtype)
          for cache in flat_caches
      ),
      input_output_aliases={
          len(scalar_prefetches) + len(new_tokens) + index: index
          for index in range(len(flat_caches))
      },
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=vmem_limit_bytes()),
      interpret=pltpu.InterpretParams() if interpret else False,
      name="flash_attn_ragged_cache_append",
  )
  updated_flat_caches = call(*scalar_prefetches, *new_tokens, *flat_caches)
  return tuple(
      updated_cache.reshape(cache.shape)
      for updated_cache, cache in zip(updated_flat_caches, caches)
  )


def flash_attn_kvcache_varlen(
    q: jax.Array,
    k_cache: jax.Array,
    v_cache: jax.Array | None,
    k: jax.Array | None,
    v: jax.Array | None,
    cu_seqlens_q: jax.Array,
    cache_seqlens: jax.Array,
    cache_batch_idx: jax.Array,
    block_table: jax.Array | None,
    num_active: int | jax.Array | None,
    *,
    q_scale: float,
    causal: bool,
    return_lse: bool,
    interpret: bool,
    merged_cache: bool = False,
) -> tuple[jax.Array, ...]:
  """Append packed new tokens once, then attend with the kv-outer extend
  kernel, which reads the updated pages only.

  k_cache and v_cache are the K/V pair, or with merged_cache (v_cache=None)
  k_cache is one (..., 2 * num_kv_heads, head_dim) cache holding a token's K
  heads then its V heads. Returns (out, *updated caches) plus lse
  (num_query_heads, total_q) float32 when return_lse.
  """
  if merged_cache:
    if v_cache is not None or k_cache.ndim != 4 or k_cache.shape[2] % 2:
      raise ValueError(
          "merged_cache takes one (rows, row_tokens, 2 * kv_heads, head_dim)"
          f" cache and v_cache=None; got {k_cache.shape=}."
      )
    cache_shape = (*k_cache.shape[:2], k_cache.shape[2] // 2, k_cache.shape[3])
  elif v_cache is None:
    raise ValueError(
        "A K/V pair cache needs v_cache; one merged cache needs merged_cache."
    )
  else:
    cache_shape = k_cache.shape
  batch = cu_seqlens_q.shape[0] - 1
  num_query_heads, head_dim = q.shape[-2:]
  row_tokens, num_kv_heads = cache_shape[1], cache_shape[2]
  if head_dim not in SUPPORTED_HEAD_DIMS:
    raise ValueError(
        f"head_dim must be one of {SUPPORTED_HEAD_DIMS}; got {head_dim}."
    )
  if num_kv_heads > 1 and num_kv_heads % 2:
    raise ValueError("token-major cache needs an even head axis or MQA.")
  if row_tokens <= 0 or row_tokens % VIRTUAL_PAGE_SIZE:
    raise ValueError(
        "Cache capacity/page_size must be a positive multiple of"
        f" {VIRTUAL_PAGE_SIZE}."
    )
  if block_table is None:
    page_size = VIRTUAL_PAGE_SIZE
    pages_per_seq = row_tokens // VIRTUAL_PAGE_SIZE
    page_table = (
        cache_batch_idx[:, None] * pages_per_seq
        + jnp.arange(pages_per_seq, dtype=jnp.int32)[None]
    ).reshape(-1)
  else:
    page_size = row_tokens
    pages_per_seq = block_table.size // batch
    if pages_per_seq == 0:
      raise ValueError(
          "block_table must reserve at least one page per sequence."
      )
    page_table = block_table
  num_active_array = jnp.asarray(
      batch if num_active is None else num_active, jnp.int32
  ).reshape(1)
  caches = (k_cache,) if merged_cache else (k_cache, v_cache)
  if q.shape[0] == 0 and return_lse:
    return (q, *caches, jnp.empty((num_query_heads, 0), jnp.float32))
  elif q.shape[0] == 0:
    return (q, *caches)
  else:
    has_new = k is not None
    if has_new:
      updated_caches = append_ragged(
          k_cache, v_cache, k, v, cu_seqlens_q, cache_seqlens, page_table,
          num_active_array, page_size=page_size, pages_per_seq=pages_per_seq,
          interpret=interpret, merged_cache=merged_cache,
      )
    else:
      updated_caches = caches
    total_lengths = cache_seqlens + (jnp.diff(cu_seqlens_q) if has_new else 0)
    page_caches = tuple(
        cache.reshape(-1, page_size, *cache.shape[-2:])
        if block_table is None
        else cache
        for cache in updated_caches
    )
    extend_outputs = flash_attn_kvcache_extend_pallas(
        q, page_caches[0], None if merged_cache else page_caches[1],
        cu_seqlens_q, total_lengths, page_table, num_active_array,
        q_scale=q_scale, causal=causal, return_lse=return_lse,
        interpret=interpret, merged_cache=merged_cache,
    )
    if return_lse:
      out, lse = extend_outputs
      return (out, *updated_caches, lse)
    else:
      return (extend_outputs, *updated_caches)
