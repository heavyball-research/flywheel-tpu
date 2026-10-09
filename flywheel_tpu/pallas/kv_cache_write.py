"""Write new K/V rows into the merged interleaved paged KV cache.

The cache is (num_pages, page_size, 2 * nheads_k, head_dim) bf16 whose token
rows interleave each kv head's K and V rows, [k0, v0, k1, v1, ...]. A bf16
row pair (2h, 2h + 1) shares one u32 word per lane, so K and V cannot be
written into their rows separately (a size-1 window of the tiled head axis
does not compile, and a strided one is not expressible); the writer stages
both in VMEM, interleaves them there and writes whole rows.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Note (david): the scoped VMEM the staging is sized against; the compiler
# gets KV_WRITE_VMEM_HEADROOM_BYTES more for the interleave temporaries.
KV_WRITE_VMEM_LIMIT_BYTES = 40 * 1024 * 1024
KV_WRITE_VMEM_HEADROOM_BYTES = 16 * 1024 * 1024
# Note (david): more in-flight page DMAs raise HBM utilization, but 128 per
# program regressed on v6e (scalar registers run out), so 64 is the cap.
MAX_PIECES_PER_BLOCK = 64
# Note (david): one interleave step packs about 8 u32 vregs (32 KiB) per
# operand.
INTERLEAVE_STEP_ELEMENTS = 8192


def kv_write_pieces_per_block(page_size: int, num_kv_heads: int,
                              head_dim: int) -> int:
  """Page pieces one writer program stages: each holds a page of K, of V and
  of interleaved rows, within KV_WRITE_VMEM_LIMIT_BYTES (1 MiB reserved)."""
  piece_bytes = 2 * page_size * 2 * num_kv_heads * head_dim * 2
  num_fitting_pieces = (KV_WRITE_VMEM_LIMIT_BYTES - 1024 * 1024) // piece_bytes
  if num_fitting_pieces < 1:
    raise ValueError(
        f"one page piece needs {piece_bytes} bytes of VMEM, past the"
        f" {KV_WRITE_VMEM_LIMIT_BYTES}-byte writer budget.")
  return min(1 << (num_fitting_pieces.bit_length() - 1), MAX_PIECES_PER_BLOCK)


def kv_cache_write_kernel(
    slices_ref: jax.Array,
    num_slices_ref: jax.Array,
    k_hbm: jax.Array,
    v_hbm: jax.Array,
    cache_hbm: jax.Array,
    out_hbm: jax.Array,
    k_vmem: jax.Array,
    v_vmem: jax.Array,
    kv_vmem: jax.Array,
    dma_sem: jax.Array,
    *,
    step_rows: int,
) -> None:
  """Write one block of page pieces into the flat cache.

  slices_ref is SMEM i32[3, padded] of (cache_row, new_row, length) columns
  and num_slices_ref is SMEM i32[1]. k_hbm and v_hbm are (T, H, D); out_hbm
  is the flat (P * S, 2H, D) cache, aliased to cache_hbm. k_vmem and v_vmem
  are (pieces, S, H, D) and kv_vmem is (pieces, S, 2H, D).
  """
  del cache_hbm
  pieces_per_block = k_vmem.shape[0]
  first_piece = pl.program_id(0) * pieces_per_block
  num_slices = num_slices_ref[0]

  def _piece_slice(i):
    slice_index = first_piece + i
    is_live = slice_index < num_slices
    return tuple(lax.select(is_live, slices_ref[field, slice_index], 0)
                 for field in range(3))

  copies = []
  for i in range(pieces_per_block):
    _, new_row_start, length = _piece_slice(i)
    for src_hbm, dst_vmem in ((k_hbm, k_vmem), (v_hbm, v_vmem)):
      copy = pltpu.make_async_copy(src_hbm.at[pl.ds(new_row_start, length)],
                                   dst_vmem.at[i, pl.ds(0, length)], dma_sem)
      copy.start()
      copies.append(copy)
  for copy in copies:
    copy.wait()

  # Note (david): interleaving only the live pieces and their live rows keeps
  # a 2-piece step in a 64-piece block at 2 pages of packing, not 64.
  def _interleave_piece(i, carry):
    length = slices_ref[2, first_piece + i]

    def _interleave_step(step, carry):
      row_window = pl.ds(pl.multiple_of(step * step_rows, step_rows),
                         step_rows)
      # Note (david): K head h lands in the low and V head h in the high 16
      # bits of its row pair's u32 word; the halves move as integers, so every
      # bit pattern (NaN payloads, subnormals) survives.
      k_bits = lax.bitcast_convert_type(k_vmem[i, row_window],
                                        jnp.uint16).astype(jnp.uint32)
      v_bits = lax.bitcast_convert_type(v_vmem[i, row_window],
                                        jnp.uint16).astype(jnp.uint32)
      kv_vmem[i, row_window] = pltpu.bitcast(k_bits | (v_bits << 16),
                                             jnp.bfloat16)
      return carry

    return lax.fori_loop(0, pl.cdiv(length, step_rows), _interleave_step,
                         carry)

  lax.fori_loop(0, jnp.minimum(num_slices - first_piece, pieces_per_block),
                _interleave_piece, 0)

  copies.clear()
  for i in range(pieces_per_block):
    cache_row_start, _, length = _piece_slice(i)
    copy = pltpu.make_async_copy(kv_vmem.at[i, pl.ds(0, length)],
                                 out_hbm.at[pl.ds(cache_row_start, length)],
                                 dma_sem)
    copy.start()
    copies.append(copy)
  for copy in copies:
    copy.wait()


@functools.partial(jax.jit, static_argnames=("interpret",),
                   donate_argnames=("kv_cache",))
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

  k and v are (T, H, D) and kv_cache is (P, S, 2H, D), all bf16. slices is
  i32[3, n] and num_slices i32[1]: column j < num_slices[0] of slices is
  (cache_row_start, new_row_start, length) in rows of the flat (P * S) cache
  view and of k / v, each piece inside one page; later columns are ignored.
  """
  _, num_kv_heads, head_dim = k.shape
  num_pages, page_size, cache_heads, cache_head_dim = kv_cache.shape
  if v.shape != k.shape or (cache_heads, cache_head_dim) != (
      2 * num_kv_heads, head_dim):
    raise ValueError(
        f"k and v must be (T, H, D) of a (P, S, 2H, D) cache; got {k.shape=},"
        f" {v.shape=}, {kv_cache.shape=}.")
  if head_dim % 128:
    raise ValueError(f"head_dim must be a multiple of 128; got {head_dim}.")
  if any(operand.dtype != jnp.bfloat16 for operand in (k, v, kv_cache)):
    raise NotImplementedError("the KV writer takes bfloat16 only.")
  pieces_per_block = kv_write_pieces_per_block(
      page_size, num_kv_heads, head_dim)
  # Note (david): the kernel indexes every piece of its last block, so the
  # table is padded with empty pieces to a whole number of blocks.
  padded_num_slices = (
      -(-slices.shape[1] // pieces_per_block) * pieces_per_block)
  slices = jnp.pad(slices.astype(jnp.int32),
                   ((0, 0), (0, padded_num_slices - slices.shape[1])))
  flat_cache = kv_cache.reshape(num_pages * page_size, cache_heads, head_dim)
  kv_stage_shape = (pieces_per_block, page_size, num_kv_heads, head_dim)
  any_spec = pl.BlockSpec(memory_space=pl.ANY)
  max_step_rows = max(1, INTERLEAVE_STEP_ELEMENTS // (num_kv_heads * head_dim))
  step_rows = 1 << (max_step_rows.bit_length() - 1)
  if page_size % step_rows:
    raise ValueError(
        f"an interleave step of {step_rows} rows does not divide"
        f" {page_size=}.")
  kv_write_call = pl.pallas_call(
      functools.partial(kv_cache_write_kernel, step_rows=step_rows),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=2,
          in_specs=[any_spec, any_spec, any_spec],
          out_specs=any_spec,
          grid=(pl.cdiv(num_slices[0], pieces_per_block),),
          scratch_shapes=[
              pltpu.VMEM(kv_stage_shape, k.dtype),
              pltpu.VMEM(kv_stage_shape, v.dtype),
              pltpu.VMEM((pieces_per_block, page_size, cache_heads, head_dim),
                         kv_cache.dtype),
              pltpu.SemaphoreType.DMA,
          ]),
      out_shape=jax.ShapeDtypeStruct(flat_cache.shape, flat_cache.dtype),
      # Note (david): the kernel writes only the new rows, so the output must
      # alias the cache (operand 4, after slices, num_slices, k and v) to keep
      # every other row.
      input_output_aliases={4: 0},
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=(KV_WRITE_VMEM_LIMIT_BYTES
                            + KV_WRITE_VMEM_HEADROOM_BYTES)),
      interpret=pltpu.InterpretParams() if interpret else False,
      name="flash_attn_kv_cache_write",
  )
  return kv_write_call(slices, num_slices.astype(jnp.int32), k, v,
                       flat_cache).reshape(kv_cache.shape)
