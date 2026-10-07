"""Read-only varlen forward kernel over a paged KV cache.

Each sequence attends its own cache prefix [0, seqused_k[r]), staged page by
page through its block table row. The cache is one merged (num_pages,
page_size, 2 * nheads_k, head_dim) pool whose token rows interleave each kv
head's K and V rows, [k0, v0, k1, v1, ...]. Every head group is one q head,
which stages only its kv head's (K, V) row pair of each page.
"""

import math
from functools import cache, partial

import jax
import jax.numpy as jnp

from .block_sizes import (
  F32_BYTES,
  FWD_BLOCKS,
  FWD_KV_COMPUTE_BLOCKS,
  FWD_Q_COMPUTE_BLOCKS,
  NUM_LANES,
  NUM_SUBLANES,
  BlockSizes,
  PagedKVInfo,
  next_pow2,
  pick_tile,
  round_up,
  vmem_limit_bytes,
)
from .flash_fwd import MIN_Q_TILES_PER_BLOCK
from .fwd_pipeline import forward_common, fwd_body, overflow_guard_threshold
from .loop_schedule import make_paged_fwd_schedule, per_seq_qblk_prefix

# Note (david): each page is its own DMA, so smaller pages would leave the
# kv stream descriptor-bound.
PAGE_SIZE_MULTIPLE = 128
# Note (david): the distance-1 fragment pipeline keeps five score-shaped f32
# temporaries live, plus a one-byte bounds mask per score.
SCORE_TEMPORARY_BYTES = 5 * F32_BYTES + 1
# Note (david): one q head per group. The fragment pipeline is fully unrolled
# over head_fold * compute tiles, and folding a kv head's q heads together
# measured 2.1x slower on v6e (h32 k4 d256, 1024-token chunks over 16K: 22.6
# against 10.6 ms) with a 15x longer compile, while re-staging the kv head per
# q head hides under the compute.
PAGED_HEAD_FOLD = 1
# Note (david): a 2048 x 2048 q x kv block ran 3.4x slower than 2048 x 1024
# on v6e (Qwen3-4B's 32:8 heads of 128, a 2040-token chunk over a 16K prefix:
# 7.72 against 2.19 ms) though it fits VMEM, and 1024 x 2048 ran at par, so
# the block area is capped at 2048 x 1024.
MAX_PAGED_BLOCK_AREA = 2048 * 1024


def flash_fwd_varlen_paged_kernel(
    cu_q_ref: jax.Array,
    seqused_k_ref: jax.Array,
    cu_qblk_ref: jax.Array,
    block_table_ref: jax.Array,
    q_hbm: jax.Array,
    kv_hbm: jax.Array,
    *refs: jax.Array,
    paged: PagedKVInfo,
    causal: bool,
    num_head_groups: int,
    q_heads_per_kv_head: int,
    bq: int,
    bkv: int,
    **body,
) -> None:
  schedule = make_paged_fwd_schedule(
      cu_q_ref, seqused_k_ref, cu_qblk_ref, block_table_ref,
      num_head_groups=num_head_groups,
      q_heads_per_kv_head=q_heads_per_kv_head,
      padded_total_q=q_hbm.shape[1],
      bq=bq, bkv=bkv,
      page_size=paged.page_size, pages_per_seq=paged.pages_per_seq,
      causal=causal, num_rows=cu_qblk_ref[cu_qblk_ref.shape[0] - 1],
  )
  # Note (david): K and V share the one merged cache operand.
  fwd_body(schedule, q_hbm, kv_hbm, kv_hbm, refs, bq=bq, bkv=bkv, paged=paged,
           **body)


def paged_kv_info(
    *,
    num_kv_heads: int,
    page_size: int,
    pages_per_seq: int,
    interpret: bool,
) -> PagedKVInfo:
  """Staging layout of one paged build; interpret=False gives the TPU one."""
  return PagedKVInfo(
      num_kv_heads=num_kv_heads,
      page_size=page_size,
      pages_per_seq=pages_per_seq,
      is_bitcast_load=not interpret,
  )


def vmem_buffer_bytes(shape: tuple[int, ...], dtype: jnp.dtype) -> int:
  """Bytes of one VMEM scratch buffer under Mosaic's v6 memref tiling.

  A second-minor axis shorter than one large tile pads only to a power of two,
  so a (2, head_dim) bf16 head pair costs no pad.
  """
  itemsize = jnp.dtype(dtype).itemsize
  *leading, rows, lanes = shape
  packing = F32_BYTES // itemsize
  large_tile_rows = NUM_SUBLANES * packing
  if rows >= large_tile_rows:
    tile_rows = large_tile_rows
  else:
    tile_rows = max(packing, next_pow2(min(rows, NUM_SUBLANES)))
  return (math.prod(leading) * round_up(rows, tile_rows)
          * round_up(lanes, NUM_LANES) * itemsize)


def paged_scratch_shapes(
    block_sizes: BlockSizes,
    *,
    head_dim: int,
    return_lse: bool,
    rotary_dtype: jnp.dtype | None,
) -> list[tuple[tuple[int, ...], jnp.dtype]]:
  """(shape, dtype) of each VMEM scratch buffer forward_common allocates for
  one paged build, in its order: Q stage, (K, V) pair staging, K and V planes,
  bounds rows, Q rotary coefficients, out stage, row max, row sum, f32
  accumulator, rescale, lse."""
  bq, bkv = block_sizes.block_q, block_sizes.block_kv
  num_stages = block_sizes.num_stages
  bf16, f32 = jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float32)
  shapes = [
      ((num_stages, bq, head_dim), bf16),
      ((num_stages, bkv, 1, 2, head_dim), bf16),
      ((2, bkv, head_dim), bf16),
      ((1, 2, bkv), jnp.dtype(jnp.int32)),
  ]
  if rotary_dtype is not None:
    # Note (david): prepare_rotary's one batch plane is shared by every head,
    # so one Q coefficient group is staged.
    shapes.append((
        (num_stages, 1, 2, bq, head_dim // 2), jnp.dtype(rotary_dtype)))
  shapes += [
      ((num_stages, bq, head_dim), bf16),
      ((bq, NUM_LANES), f32),
      ((2, bq, NUM_LANES), f32),
      ((2, bq, head_dim), f32),
      ((bq, NUM_LANES), f32),
  ]
  if return_lse:
    shapes.append(((num_stages, bq, NUM_LANES), f32))
  return shapes


def estimate_vmem_bytes(
    block_sizes: BlockSizes,
    *,
    head_dim: int,
    return_lse: bool,
    rotary_dtype: jnp.dtype | None,
) -> int:
  """Scoped VMEM bytes of one paged build: its scratch plus the score
  temporaries of one compute fragment."""
  scratch_bytes = sum(
      vmem_buffer_bytes(shape, dtype)
      for shape, dtype in paged_scratch_shapes(
          block_sizes, head_dim=head_dim, return_lse=return_lse,
          rotary_dtype=rotary_dtype))
  temporary_bytes = (
      block_sizes.block_q_compute * block_sizes.block_kv_compute
      * SCORE_TEMPORARY_BYTES)
  return scratch_bytes + temporary_bytes


@cache
def resolve_paged_tiles(
    *,
    max_seqlen_q_bucket: int,
    max_seqlen_k_bucket: int,
    page_size: int,
    head_dim: int,
    return_lse: bool,
    rotary_dtype: jnp.dtype | None,
    block_sizes: BlockSizes | None = None,
) -> BlockSizes:
  """block_sizes of one paged build.

  Among the builds whose TPU staging fits vmem_limit_bytes() and whose q x kv
  block area is at most MAX_PAGED_BLOCK_AREA, picks the one that streams the
  longest sequence's kv prefix the fewest times, then the smaller q block,
  then the larger kv block. A pinned block_sizes is only checked against the
  budget.
  """
  if block_sizes is not None:
    candidate_blocks = [block_sizes]
  else:
    # Note (david): block_kv is whole pages, so a 384- or 640-token page takes
    # five or three pages (1920 tokens), where no power-of-two block ends on a
    # page edge; smaller kv blocks halve the page count.
    kv_block_cap = max(min(max_seqlen_k_bucket, FWD_BLOCKS[0]), page_size)
    max_kv_pages = kv_block_cap // page_size
    kv_blocks = [page_size * (max_kv_pages >> shift)
                 for shift in range(max_kv_pages.bit_length())]
    q_block_cap = max(max_seqlen_q_bucket, FWD_BLOCKS[-1])
    candidate_blocks = [
        BlockSizes(
            block_q=block_q,
            block_kv=block_kv,
            block_kv_compute=pick_tile(FWD_KV_COMPUTE_BLOCKS, block_kv),
            block_q_compute=pick_tile(
                FWD_Q_COMPUTE_BLOCKS, block_q // MIN_Q_TILES_PER_BLOCK),
        )
        for block_q in FWD_BLOCKS if block_q <= q_block_cap
        for block_kv in kv_blocks
        if block_q * block_kv <= MAX_PAGED_BLOCK_AREA
    ]
  estimates = {
      blocks: estimate_vmem_bytes(
          blocks, head_dim=head_dim, return_lse=return_lse,
          rotary_dtype=rotary_dtype)
      for blocks in candidate_blocks
  }
  fitting = [
      blocks for blocks, estimate in estimates.items()
      if estimate <= vmem_limit_bytes()
  ]
  if not fitting:
    blocks, estimate = min(estimates.items(), key=lambda item: item[1])
    raise ValueError(
        "paged attention does not fit the scoped VMEM budget of"
        f" {vmem_limit_bytes()} bytes: its smallest build (block_q="
        f"{blocks.block_q}, block_kv={blocks.block_kv}) needs ~{estimate}"
        f" bytes at {head_dim=}."
    )

  def _prefix_streams(blocks):
    return (-(-max_seqlen_q_bucket // blocks.block_q), blocks.block_q,
            -blocks.block_kv)

  return min(fitting, key=_prefix_streams)


def flash_attn_varlen_paged(
    q: jax.Array,
    kv_cache: jax.Array,
    cu_seqlens_q: jax.Array,
    seqused_k: jax.Array,
    block_table: jax.Array,
    *,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool,
    q_scale: float,
    return_lse: bool,
    interpret: bool,
    block_sizes: BlockSizes | None = None,
    rotary: tuple[jax.Array, None] | None = None,
    rotary_interleaved: bool = True,
) -> jax.Array | tuple[jax.Array, jax.Array]:
  """Packed head-major attention over a paged KV cache, read-only.

  q is (nheads, total_q, head_dim) bf16 packed by cu_seqlens_q, whose first
  entry is 0. kv_cache is the bf16 merged (num_pages, page_size, 2 * nheads_k,
  head_dim) pool whose token rows interleave each kv head's K and V rows,
  [k0, v0, k1, v1, ...]. seqused_k is each sequence's kv length with its new
  tokens, which the cache must already hold; block_table entries past those
  tokens' pages are never dereferenced. Causal rows align to the end of their
  sequence's keys, and q_scale is softmax_scale * log2(e). Rows past
  cu_seqlens_q[-1] are out = 0, lse = -inf. max_seqlen_q and max_seqlen_k
  only pick the tiles, so an underestimate is slow, not wrong.
  rotary is prepare_rotary's (q coefficients, None) pair.
  """
  num_q_heads, total_q, head_dim = q.shape
  if kv_cache.ndim != 4 or kv_cache.shape[2] % 2:
    raise ValueError(
        "kv_cache must be one merged (num_pages, page_size, 2 * nheads_k,"
        f" head_dim) cache; got {kv_cache.shape}."
    )
  _, page_size, cache_heads, cache_head_dim = kv_cache.shape
  num_kv_heads = cache_heads // 2
  if q.dtype != jnp.bfloat16 or kv_cache.dtype != jnp.bfloat16:
    raise NotImplementedError(
        "paged attention supports bfloat16 q and cache only; got"
        f" {q.dtype}, {kv_cache.dtype}."
    )
  if cache_head_dim != head_dim or head_dim % NUM_LANES:
    raise ValueError(
        f"head_dim must be a multiple of {NUM_LANES} shared by q and the"
        f" cache; got q {head_dim}, cache {cache_head_dim}."
    )
  if page_size % PAGE_SIZE_MULTIPLE:
    raise ValueError(
        f"page_size must be a multiple of {PAGE_SIZE_MULTIPLE}; got"
        f" {page_size}."
    )
  if num_q_heads % num_kv_heads:
    raise ValueError(
        f"nheads={num_q_heads} must be a multiple of nheads_k={num_kv_heads}."
    )
  if total_q == 0:
    raise ValueError("q needs at least one token row.")

  cu_seqlens_q = jnp.asarray(cu_seqlens_q)
  seqused_k = jnp.asarray(seqused_k)
  block_table = jnp.asarray(block_table)
  batch = cu_seqlens_q.shape[0] - 1
  if (seqused_k.shape != (batch,)
      or not jnp.issubdtype(seqused_k.dtype, jnp.integer)):
    raise ValueError(
        f"seqused_k must be a ({batch},) integer array; got"
        f" {seqused_k.shape} {seqused_k.dtype}."
    )
  if (block_table.ndim != 2 or block_table.shape[0] != batch
      or block_table.shape[1] == 0
      or not jnp.issubdtype(block_table.dtype, jnp.integer)):
    raise ValueError(
        f"block_table must be a (batch={batch}, max_pages_per_seq >= 1)"
        f" integer array; got {block_table.shape} {block_table.dtype}."
    )
  pages_per_seq = block_table.shape[1]
  capacity = pages_per_seq * page_size

  if block_sizes is not None:
    if block_sizes.block_kv % page_size:
      raise ValueError(
          f"block_kv={block_sizes.block_kv} must be whole pages of"
          f" page_size={page_size}.")
    if (block_sizes.block_q % block_sizes.block_q_compute
        or block_sizes.block_q // block_sizes.block_q_compute
        < MIN_Q_TILES_PER_BLOCK):
      raise ValueError(
          "the static-anchor softmax needs block_q to hold >= 2 whole q"
          f" compute tiles; got block_q={block_sizes.block_q},"
          f" block_q_compute={block_sizes.block_q_compute}."
      )
  if rotary is None:
    rotary_dtype = None
  else:
    rotary_dtype = jnp.dtype(rotary[0].dtype)
  # Note (david): the bounds key the tile cache, so power-of-two buckets keep
  # a wobbling longest sequence on one build. The caps hold because no q
  # block passes the 128-padded buffer and no sequence outgrows its table row.
  block_sizes = resolve_paged_tiles(
      max_seqlen_q_bucket=min(
          next_pow2(max_seqlen_q), round_up(total_q, NUM_LANES)),
      max_seqlen_k_bucket=min(next_pow2(max_seqlen_k), capacity),
      page_size=page_size,
      head_dim=head_dim,
      return_lse=return_lse,
      rotary_dtype=rotary_dtype,
      block_sizes=block_sizes,
  )
  bq = block_sizes.block_q
  bkv = block_sizes.block_kv
  # Note (david): fwd_body keys a staged block by (sequence, kv head, kv
  # block, loaded 8-row tiles) in one int32.
  num_kv_keys = (batch * num_kv_heads * -(-capacity // bkv)
                 * (bkv // NUM_SUBLANES + 1))
  if num_kv_keys > 2**31 - 1:
    raise ValueError(
        f"{batch=} sequences of {num_kv_heads} kv heads and capacity"
        f" {capacity} tokens in {bkv}-token blocks need {num_kv_keys} staging"
        " keys, past int32."
    )
  paged = paged_kv_info(
      num_kv_heads=num_kv_heads, page_size=page_size,
      pages_per_seq=pages_per_seq, interpret=interpret)

  padded_total_q = round_up(total_q, bq)
  num_pad_q = padded_total_q - total_q
  q_in = jnp.pad(q, ((0, 0), (0, num_pad_q), (0, 0)))
  if rotary is None:
    rotary_in = None
  else:
    rotary_in = (
        jnp.pad(rotary[0], ((0, 0), (0, 0), (0, num_pad_q), (0, 0))), None)

  cu_q = cu_seqlens_q.astype(jnp.int32)
  smem_operands = [
      cu_q,
      seqused_k.astype(jnp.int32),
      per_seq_qblk_prefix(cu_q, bq),
      block_table.astype(jnp.int32).reshape(-1),
  ]
  kernel = partial(
      flash_fwd_varlen_paged_kernel,
      causal=causal,
      num_head_groups=num_q_heads // PAGED_HEAD_FOLD,
      q_heads_per_kv_head=num_q_heads // num_kv_heads,
  )
  outputs = forward_common(
      kernel, smem_operands, q_in, kv_cache, None,
      block_sizes=block_sizes,
      num_kv_heads=num_kv_heads,
      head_fold=PAGED_HEAD_FOLD,
      transposed_pv=False,
      return_lse=return_lse,
      kernel_name="flash_attn_varlen_paged_fwd",
      interpret=interpret,
      token_major=None,
      is_per_seq=True,
      rotary=rotary_in,
      rotary_interleaved=rotary_interleaved,
      paged=paged,
      guard_threshold=overflow_guard_threshold(capacity),
      window=(None, None),
      causal_offset=0,
      softcap=0.0,
      q_scale=q_scale,
  )
  if return_lse:
    out, lse = outputs
    return out[:, :total_q], lse[:, :total_q]
  else:
    return outputs[:, :total_q]
