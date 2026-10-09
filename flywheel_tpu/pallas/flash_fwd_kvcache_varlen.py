"""Packed multi-token KV-cache attention: a ragged append into the paged
merged cache, then the paged varlen kernel over the updated pages."""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .block_sizes import vmem_limit_bytes
from .flash_fwd_varlen_paged import flash_attn_varlen_paged
from .kv_cache_write import INTERLEAVE_STEP_ELEMENTS


def append_kernel(
    cu_seqlens_ref: jax.Array,
    cache_seqlens_ref: jax.Array,
    block_table_ref: jax.Array,
    num_active_ref: jax.Array,
    k_ref: jax.Array,
    v_ref: jax.Array,
    _: jax.Array,
    kv_out_ref: jax.Array,
    k_vmem: jax.Array,
    v_vmem: jax.Array,
    kv_vmem: jax.Array,
    dma_sems: jax.Array,
    *,
    page_size: int,
    pages_per_seq: int,
    step_rows: int,
) -> None:
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
                staging_vmem.at[pl.ds(0, num_tokens)],
                dma_sems.at[sem_index],
            )
            for sem_index, (source_ref, staging_vmem) in enumerate(
                ((k_ref, k_vmem), (v_ref, v_vmem)))
        ]
        for copy in incoming:
          copy.start()
        for copy in incoming:
          copy.wait()

        # Note (david): K and V are interleaved in VMEM so that each page's new
        # rows go out as whole cache rows in one DMA.
        def _interleave_step(step, carry):
          rows = pl.ds(pl.multiple_of(step * step_rows, step_rows), step_rows)
          # Note (david): K head h lands in the low and V head h in the high
          # 16 bits of its row pair's u32 word; the halves move as integers,
          # so every bit pattern (NaN payloads, subnormals) survives.
          k_bits = lax.bitcast_convert_type(k_vmem[rows],
                                            jnp.uint16).astype(jnp.uint32)
          v_bits = lax.bitcast_convert_type(v_vmem[rows],
                                            jnp.uint16).astype(jnp.uint32)
          kv_vmem[rows] = pltpu.bitcast(k_bits | (v_bits << 16), jnp.bfloat16)
          return carry

        lax.fori_loop(0, pl.cdiv(num_tokens, step_rows), _interleave_step,
                      None)
        outgoing = pltpu.make_async_copy(
            kv_vmem.at[pl.ds(0, num_tokens)],
            kv_out_ref.at[pl.ds(destination_start, num_tokens)],
            dma_sems.at[0],
        )
        outgoing.start()
        outgoing.wait()

      lax.fori_loop(first_page, end_page, _copy_page, None)

  lax.fori_loop(0, num_active_ref[0], _append_request, None)


def append_ragged(
    kv_cache: jax.Array,
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
) -> jax.Array:
  """Write packed new tokens k, v (total, nheads_k, head_dim) after each
  active request's cache_seqlens into its pages of the merged interleaved
  (num_pages, page_size, 2 * nheads_k, head_dim) cache; returns the cache."""
  # Note (david): a token-row view that leaves the (heads, head_dim) tiling
  # untouched, so XLA emits no copy.
  flat_cache = kv_cache.reshape(-1, *kv_cache.shape[-2:])
  _, num_kv_heads, head_dim = k.shape
  hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
  scalar_prefetches = (cu_seqlens_q, cache_seqlens, block_table, num_active)
  max_step_rows = max(1, INTERLEAVE_STEP_ELEMENTS // (num_kv_heads * head_dim))
  step_rows = 1 << (max_step_rows.bit_length() - 1)
  if page_size % step_rows:
    raise ValueError(
        f"an interleave step of {step_rows} rows does not divide"
        f" {page_size=}.")
  call = pl.pallas_call(
      functools.partial(
          append_kernel, page_size=page_size, pages_per_seq=pages_per_seq,
          step_rows=step_rows),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=len(scalar_prefetches), grid=(1,),
          in_specs=(hbm_spec, hbm_spec, hbm_spec),
          out_specs=hbm_spec,
          scratch_shapes=(
              pltpu.VMEM((page_size, num_kv_heads, head_dim), k.dtype),
              pltpu.VMEM((page_size, num_kv_heads, head_dim), v.dtype),
              pltpu.VMEM((page_size, 2 * num_kv_heads, head_dim),
                         kv_cache.dtype),
              pltpu.SemaphoreType.DMA((2,)),
          ),
      ),
      out_shape=jax.ShapeDtypeStruct(flat_cache.shape, flat_cache.dtype),
      input_output_aliases={len(scalar_prefetches) + 2: 0},
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=vmem_limit_bytes()),
      interpret=pltpu.InterpretParams() if interpret else False,
      name="flash_attn_ragged_cache_append",
  )
  return call(*scalar_prefetches, k, v, flat_cache).reshape(kv_cache.shape)


def flash_attn_kvcache_varlen(
    q: jax.Array,
    kv_cache: jax.Array,
    k: jax.Array | None,
    v: jax.Array | None,
    cu_seqlens_q: jax.Array,
    cache_seqlens: jax.Array,
    block_table: jax.Array,
    num_active: int | jax.Array | None,
    *,
    q_scale: float,
    causal: bool,
    return_lse: bool,
    interpret: bool,
) -> tuple[jax.Array, ...]:
  """Append packed new tokens once, then attend with the paged varlen kernel,
  which reads the updated pages only.

  q is (total_q, nheads, head_dim) packed by cu_seqlens_q; kv_cache is the
  merged interleaved (num_pages, page_size, 2 * nheads_k, head_dim) cache and
  block_table its flat (batch * pages_per_seq,) page table. Returns (out,
  kv_cache) plus lse (nheads, total_q) float32 when return_lse.
  """
  batch = cu_seqlens_q.shape[0] - 1
  num_query_heads = q.shape[1]
  page_size = kv_cache.shape[1]
  pages_per_seq = block_table.size // batch
  if pages_per_seq == 0:
    raise ValueError("block_table must reserve at least one page per sequence.")
  if q.shape[0] == 0 and return_lse:
    return (q, kv_cache, jnp.empty((num_query_heads, 0), jnp.float32))
  elif q.shape[0] == 0:
    return (q, kv_cache)
  else:
    num_active_array = jnp.asarray(
        batch if num_active is None else num_active, jnp.int32).reshape(1)
    has_new = k is not None
    if has_new:
      updated_kv_cache = append_ragged(
          kv_cache, k, v, cu_seqlens_q, cache_seqlens, block_table,
          num_active_array, page_size=page_size, pages_per_seq=pages_per_seq,
          interpret=interpret,
      )
    else:
      updated_kv_cache = kv_cache
    # Note (david): pinning every boundary past num_active to
    # cu_seqlens_q[num_active] empties the inactive requests, so they own no q
    # block and the kernel never reads their lengths or table rows; their rows
    # become packed padding (out = 0, lse = -inf).
    num_active_scalar = num_active_array[0]
    kept_cu_seqlens_q = jnp.where(
        jnp.arange(batch + 1, dtype=jnp.int32) <= num_active_scalar,
        cu_seqlens_q, cu_seqlens_q[num_active_scalar])
    total_lengths = cache_seqlens + (
        jnp.diff(cu_seqlens_q) if has_new else 0)
    outputs = flash_attn_varlen_paged(
        q.transpose(1, 0, 2), updated_kv_cache, kept_cu_seqlens_q,
        total_lengths, block_table.reshape(batch, pages_per_seq),
        max_seqlen_q=q.shape[0], max_seqlen_k=pages_per_seq * page_size,
        causal=causal, q_scale=q_scale, return_lse=return_lse,
        interpret=interpret,
    )
    if return_lse:
      out, lse = outputs
      return (out.transpose(1, 0, 2), updated_kv_cache, lse)
    else:
      return (outputs.transpose(1, 0, 2), updated_kv_cache)
