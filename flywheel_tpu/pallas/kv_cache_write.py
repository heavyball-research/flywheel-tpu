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


def interleave_kv_rows(k: jax.Array, v: jax.Array) -> jax.Array:
  """(rows, 2H, D) bf16 [k0, v0, k1, v1, ...] from (rows, H, D) bf16 k, v.

  K head h lands in the low and V head h in the high 16 bits of its row
  pair's u32 word. The halves move as integers, so every bit pattern (NaN
  payloads, subnormals) survives.
  """
  low = lax.bitcast_convert_type(k, jnp.uint16).astype(jnp.uint32)
  high = lax.bitcast_convert_type(v, jnp.uint16).astype(jnp.uint32)
  return pltpu.bitcast(low | (high << 16), jnp.bfloat16)


def interleave_step_rows(num_kv_heads: int, head_dim: int,
                         page_size: int) -> int:
  """Token rows one interleave step packs: a power of two dividing the page,
  about INTERLEAVE_STEP_ELEMENTS u32 words per operand."""
  rows = max(1, INTERLEAVE_STEP_ELEMENTS // (num_kv_heads * head_dim))
  rows = 1 << (rows.bit_length() - 1)
  if page_size % rows:
    raise ValueError(
        f"an interleave step of {rows} rows does not divide {page_size=}.")
  return rows


def kv_write_pieces_per_block(page_size: int, num_kv_heads: int,
                              head_dim: int) -> int:
  """Page pieces one writer program stages: each holds a page of K, of V and
  of interleaved rows, within KV_WRITE_VMEM_LIMIT_BYTES (1 MiB reserved)."""
  piece_bytes = 2 * page_size * 2 * num_kv_heads * head_dim * 2
  pieces = (KV_WRITE_VMEM_LIMIT_BYTES - 1024 * 1024) // piece_bytes
  if pieces < 1:
    raise ValueError(
        f"one page piece needs {piece_bytes} bytes of VMEM, past the"
        f" {KV_WRITE_VMEM_LIMIT_BYTES}-byte writer budget.")
  return min(1 << (pieces.bit_length() - 1), MAX_PIECES_PER_BLOCK)


def _write_kernel(
    slices_ref,  # SMEM i32[3, padded]: (cache_row, new_row, length)
    num_slices_ref,  # SMEM i32[1]
    k_hbm,  # [T, H, D]
    v_hbm,  # [T, H, D]
    cache_hbm,  # [P * S, 2H, D], aliased to out_hbm
    out_hbm,  # [P * S, 2H, D]
    k_buf,  # VMEM [pieces, S, H, D]
    v_buf,  # VMEM [pieces, S, H, D]
    kv_buf,  # VMEM [pieces, S, 2H, D]
    sem,
    *,
    step_rows: int,
):
  del cache_hbm
  pieces = k_buf.shape[0]
  first = pl.program_id(0) * pieces
  count = num_slices_ref[0]

  def _piece(i):
    index = first + i
    live = index < count
    return tuple(lax.select(live, slices_ref[field, index], 0)
                 for field in range(3))

  copies = []
  for i in range(pieces):
    _, new_start, length = _piece(i)
    for source, buffer in ((k_hbm, k_buf), (v_hbm, v_buf)):
      copy = pltpu.make_async_copy(source.at[pl.ds(new_start, length)],
                                   buffer.at[i, pl.ds(0, length)], sem)
      copy.start()
      copies.append(copy)
  for copy in copies:
    copy.wait()

  # Note (david): only live pieces and their live rows are interleaved; a
  # 2-piece step in a 64-piece block packs 2 pages, not 64.
  def _interleave_piece(i, carry):
    length = slices_ref[2, first + i]

    def _interleave_step(step, carry):
      rows = pl.ds(pl.multiple_of(step * step_rows, step_rows), step_rows)
      kv_buf[i, rows] = interleave_kv_rows(k_buf[i, rows], v_buf[i, rows])
      return carry

    return lax.fori_loop(0, pl.cdiv(length, step_rows), _interleave_step,
                         carry)

  lax.fori_loop(0, jnp.minimum(count - first, pieces), _interleave_piece, 0)

  copies.clear()
  for i in range(pieces):
    cache_start, _, length = _piece(i)
    copy = pltpu.make_async_copy(kv_buf.at[i, pl.ds(0, length)],
                                 out_hbm.at[pl.ds(cache_start, length)], sem)
    copy.start()
    copies.append(copy)
  for copy in copies:
    copy.wait()


@functools.partial(jax.jit, static_argnames=("interpret",),
                   donate_argnames=("kv_cache",))
def write_kv_cache_pages(
    k: jax.Array,  # [T, H, D]
    v: jax.Array,  # [T, H, D]
    kv_cache: jax.Array,  # [P, S, 2H, D]
    slices: jax.Array,  # i32[3, n]
    num_slices: jax.Array,  # i32[1]
    *,
    interpret: bool = False,
) -> jax.Array:
  """Copy new K/V rows into the merged interleaved cache; returns the cache,
  updated in place (it is donated).

  Column j < num_slices[0] of slices is (cache_row_start, new_row_start,
  length) in rows of the flat (P * S) cache view and of k / v, each piece
  inside one page; later columns are ignored.
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
  if any(x.dtype != jnp.bfloat16 for x in (k, v, kv_cache)):
    raise NotImplementedError("the KV writer takes bfloat16 only.")
  pieces = kv_write_pieces_per_block(page_size, num_kv_heads, head_dim)
  # Note (david): the kernel indexes every piece of its last block, so the
  # table is padded with empty pieces to a whole number of blocks.
  padded = -(-slices.shape[1] // pieces) * pieces
  slices = jnp.pad(slices.astype(jnp.int32),
                   ((0, 0), (0, padded - slices.shape[1])))
  flat_cache = kv_cache.reshape(num_pages * page_size, cache_heads, head_dim)
  stage = (pieces, page_size, num_kv_heads, head_dim)
  any_space = pl.BlockSpec(memory_space=pl.ANY)
  call = pl.pallas_call(
      functools.partial(
          _write_kernel,
          step_rows=interleave_step_rows(num_kv_heads, head_dim, page_size)),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=2,
          in_specs=[any_space, any_space, any_space],
          out_specs=any_space,
          grid=(pl.cdiv(num_slices[0], pieces),),
          scratch_shapes=[
              pltpu.VMEM(stage, k.dtype),
              pltpu.VMEM(stage, v.dtype),
              pltpu.VMEM((pieces, page_size, cache_heads, head_dim),
                         kv_cache.dtype),
              pltpu.SemaphoreType.DMA,
          ]),
      out_shape=jax.ShapeDtypeStruct(flat_cache.shape, flat_cache.dtype),
      # Note (david): operands are slices, num_slices, k, v, cache; the cache
      # (operand 4) is the output.
      input_output_aliases={4: 0},
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=(KV_WRITE_VMEM_LIMIT_BYTES
                            + KV_WRITE_VMEM_HEADROOM_BYTES)),
      interpret=pltpu.InterpretParams() if interpret else False,
      name="flash_attn_kv_cache_write",
  )
  return call(slices, num_slices.astype(jnp.int32), k, v,
              flat_cache).reshape(kv_cache.shape)
