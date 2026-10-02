"""Varlen forward kernel reading K/V from a paged KV cache.

The q side is the per-seq varlen schedule. Each sequence's kv axis is its own
cache prefix [0, seqused_k[r]), staged page by page through its block table
row; the cache already holds every attended token, so the kernel only reads
it.
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
  VMEM_LIMIT_BYTES,
  BlockSizes,
  PagedKVInfo,
  next_pow2,
  pick_tile,
  round_up,
)
from .flash_fwd import MIN_Q_TILES_PER_BLOCK
from .fwd_pipeline import forward_common, fwd_body, overflow_guard_threshold
from .loop_schedule import make_paged_fwd_schedule, per_seq_qblk_prefix

# Note (david): each page is its own DMA, so smaller pages would leave the
# kv stream descriptor-bound.
PAGE_SIZE_MULTIPLE = 128
# Note (david): five live score-shaped f32 temporaries of the distance-1
# fragment pipeline plus a one-byte bounds mask per score, the term the extend
# kernel's estimator tracks against Mosaic. No Mosaic report of a paged build
# has checked it yet.
SCORE_TEMPORARY_BYTES = 5 * F32_BYTES + 1


def flash_fwd_varlen_paged_kernel(
    cu_q_ref: jax.Array,
    seqused_k_ref: jax.Array,
    cu_qblk_ref: jax.Array,
    block_table_ref: jax.Array,
    q_hbm: jax.Array,
    k_hbm: jax.Array,
    *refs: jax.Array,
    paged: PagedKVInfo,
    causal: bool,
    num_head_groups: int,
    q_heads_per_kv_head: int,
    bq: int,
    bkv: int,
    **body,
) -> None:
  # Note (david): a merged cache is one operand whose V heads follow its K
  # heads, so V reads the K ref.
  if paged.is_merged:
    v_hbm = k_hbm
  else:
    v_hbm, *refs = refs
  schedule = make_paged_fwd_schedule(
      cu_q_ref, seqused_k_ref, cu_qblk_ref, block_table_ref,
      num_head_groups=num_head_groups,
      q_heads_per_kv_head=q_heads_per_kv_head,
      padded_total_q=q_hbm.shape[1],
      bq=bq, bkv=bkv,
      page_size=paged.page_size, pages_per_seq=paged.pages_per_seq,
      left=None, right=0 if causal else None,
      num_rows=cu_qblk_ref[cu_qblk_ref.shape[0] - 1],
  )
  fwd_body(schedule, q_hbm, k_hbm, v_hbm, refs, bq=bq, bkv=bkv, paged=paged,
           **body)


def paged_kv_info(
    *,
    num_kv_heads: int,
    kv_heads_per_group: int,
    page_size: int,
    pages_per_seq: int,
    is_merged: bool,
    interpret: bool,
) -> PagedKVInfo:
  """Staging layout of one paged build; interpret=False gives the TPU one."""
  # Note (david): a one-head K/V pair is staged head-major, (1, tokens,
  # head_dim), since (page_size, 1, head_dim) holds the same bytes as
  # (1, page_size, head_dim) with no head pair to pack and no 8-sublane pad.
  # The pair-packed load bitcasts refs, which only the TPU build supports,
  # and needs an even staged head axis.
  is_cache_head_major = not is_merged and num_kv_heads == 1
  staged_kv_heads = 2 * num_kv_heads if is_merged else num_kv_heads
  is_bitcast_load = (
      not interpret and not is_cache_head_major and staged_kv_heads % 2 == 0)
  return PagedKVInfo(
      num_kv_heads=num_kv_heads,
      kv_heads_per_group=kv_heads_per_group,
      page_size=page_size,
      pages_per_seq=pages_per_seq,
      is_merged=is_merged,
      is_cache_head_major=is_cache_head_major,
      is_bitcast_load=is_bitcast_load,
  )


def vmem_buffer_bytes(shape: tuple[int, ...], dtype: jnp.dtype) -> int:
  """Bytes of one VMEM scratch buffer under Mosaic's v6 memref tiling.

  The minor axis pads to whole 128-lane tiles. A second-minor axis at least
  one large tile tall (16 bf16 or 8 32-bit rows) pads to that tile; a shorter
  one to the smallest power of two covering it, no less than one packed row
  group, so a (2, head_dim) bf16 head pair costs no pad.
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
    paged: PagedKVInfo,
    *,
    q_heads_per_kv_head: int,
    head_dim: int,
    return_lse: bool,
    rotary_dtype: jnp.dtype | None,
) -> list[tuple[tuple[int, ...], jnp.dtype]]:
  """(shape, dtype) of every VMEM scratch buffer forward_common allocates for
  one paged build, in its allocation order: Q stage, KV staging (one merged
  buffer or a K/V pair), bounds rows, Q rotary coefficients, output stage,
  row max, row sum, f32 accumulator, rescale, then the lse stage."""
  bq, bkv = block_sizes.block_q, block_sizes.block_kv
  num_stages = block_sizes.num_stages
  head_fold = paged.kv_heads_per_group * q_heads_per_kv_head
  folded_bq = head_fold * bq
  bf16, f32 = jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float32)
  if head_fold == 1:
    q_stage_shape = (num_stages, bq, head_dim)
  else:
    q_stage_shape = (num_stages, head_fold, bq, head_dim)
  if paged.is_cache_head_major:
    kv_staging_shape = (num_stages, 1, bkv, head_dim)
  elif paged.is_bitcast_load:
    kv_staging_shape = (
        num_stages, bkv, paged.staged_kv_heads // 2, 2, head_dim)
  else:
    kv_staging_shape = (num_stages, bkv, paged.staged_kv_heads, head_dim)
  num_kv_staging = 1 if paged.is_merged else 2
  shapes = [(q_stage_shape, bf16)]
  shapes += [(kv_staging_shape, bf16)] * num_kv_staging
  shapes.append(((1, 2, bkv), jnp.dtype(jnp.int32)))
  if rotary_dtype is not None:
    # Note (david): prepare_rotary's coefficients have one batch plane shared
    # by every head, and nheads is a multiple of any paged fold, so the
    # pipeline stages a single coefficient group, for Q only.
    shapes.append((
        (num_stages, 1, 2, bq, head_dim // 2), jnp.dtype(rotary_dtype)))
  shapes += [
      (q_stage_shape, bf16),
      ((folded_bq, NUM_LANES), f32),
      ((2, folded_bq, NUM_LANES), f32),
      ((2, folded_bq, head_dim), f32),
      ((folded_bq, NUM_LANES), f32),
  ]
  if return_lse:
    shapes.append(((num_stages, bq, round_up(head_fold, NUM_LANES)), f32))
  return shapes


def estimate_vmem_bytes(
    block_sizes: BlockSizes,
    paged: PagedKVInfo,
    *,
    q_heads_per_kv_head: int,
    head_dim: int,
    return_lse: bool,
    rotary_dtype: jnp.dtype | None,
) -> int:
  """Scoped VMEM of one paged build, in bytes: paged_scratch_shapes under
  vmem_buffer_bytes, plus the score temporaries of one compute fragment."""
  scratch_bytes = sum(
      vmem_buffer_bytes(shape, dtype)
      for shape, dtype in paged_scratch_shapes(
          block_sizes, paged, q_heads_per_kv_head=q_heads_per_kv_head,
          head_dim=head_dim, return_lse=return_lse,
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
    pages_per_seq: int,
    num_kv_heads: int,
    q_heads_per_kv_head: int,
    head_dim: int,
    is_merged: bool,
    return_lse: bool,
    rotary_dtype: jnp.dtype | None,
    block_sizes: BlockSizes | None = None,
    kv_heads_per_group: int | None = None,
) -> tuple[BlockSizes, int]:
  """(block_sizes, kv_heads_per_group) of one paged build.

  block_q ranges over the FWD_BLOCKS entries up to the max_seqlen_q bucket
  (one 128-row block at least). block_kv is the most whole pages that fit
  both the max_seqlen_k bucket and the largest FWD_BLOCKS entry, one page at
  least, and compute tiles come from FWD_Q_COMPUTE_BLOCKS (two per q block)
  and FWD_KV_COMPUTE_BLOCKS. kv_heads_per_group ranges over the divisors of
  num_kv_heads. Among the builds whose estimate_vmem_bytes, for the TPU
  staging layout, fits VMEM_LIMIT_BYTES, the pair minimizes how often the
  longest sequence's prefix is streamed, ceil(bucket / block_q) *
  (num_kv_heads / kv_heads_per_group); a tie takes the larger group, whose
  smaller q block pads short sequences less. A pinned block_sizes or
  kv_heads_per_group fixes its axis of the search. Raises when no build
  fits; tiles never shrink past these candidates.
  """
  if block_sizes is not None:
    candidate_blocks = [block_sizes]
  else:
    # Note (david): any multiple-of-128 page size gets whole-page blocks, so
    # a 384- or 640-token page takes five or three pages (1920 tokens) where
    # no power-of-two block ends on a page edge.
    kv_block_cap = max(min(max_seqlen_k_bucket, FWD_BLOCKS[0]), page_size)
    block_kv = kv_block_cap // page_size * page_size
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
    ]
  if kv_heads_per_group is not None:
    candidate_groups = [kv_heads_per_group]
  else:
    candidate_groups = [
        group for group in range(1, num_kv_heads + 1)
        if num_kv_heads % group == 0
    ]
  estimates = {
      (blocks, group): estimate_vmem_bytes(
          blocks,
          paged_kv_info(
              num_kv_heads=num_kv_heads, kv_heads_per_group=group,
              page_size=page_size, pages_per_seq=pages_per_seq,
              is_merged=is_merged, interpret=False),
          q_heads_per_kv_head=q_heads_per_kv_head, head_dim=head_dim,
          return_lse=return_lse, rotary_dtype=rotary_dtype)
      for blocks in candidate_blocks for group in candidate_groups
  }
  fitting = [
      candidate for candidate, estimate in estimates.items()
      if estimate <= VMEM_LIMIT_BYTES
  ]
  if not fitting:
    (blocks, group), estimate = min(
        estimates.items(), key=lambda item: item[1])
    raise ValueError(
        "paged attention does not fit the scoped VMEM budget of"
        f" {VMEM_LIMIT_BYTES} bytes: its smallest build (block_q="
        f"{blocks.block_q}, block_kv={blocks.block_kv},"
        f" kv_heads_per_group={group}, head_fold"
        f" {group * q_heads_per_kv_head}) needs ~{estimate} bytes. A head"
        f" group holds at least {q_heads_per_kv_head=} q heads of"
        f" {head_dim=}."
    )

  def _prefix_streams(candidate):
    blocks, group = candidate
    num_q_blocks = -(-max_seqlen_q_bucket // blocks.block_q)
    return (num_q_blocks * (num_kv_heads // group), -group, blocks.block_q)

  return min(fitting, key=_prefix_streams)


def flash_attn_varlen_paged(
    q: jax.Array,
    k_cache: jax.Array,
    v_cache: jax.Array | None,
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
    kv_heads_per_group: int | None = None,
    rotary: tuple[jax.Array, None] | None = None,
    rotary_interleaved: bool = True,
) -> jax.Array | tuple[jax.Array, jax.Array]:
  """Packed head-major attention over a paged KV cache, read-only.

  q: (nheads, total_q, head_dim) bf16, packed by cu_seqlens_q, (batch + 1,)
  nondecreasing int boundaries; cu_seqlens_q[0] may exceed 0. The cache is
  bf16, either one merged (num_pages, page_size, 2 * nheads_k, head_dim) pool
  whose token rows hold the K heads, then the V heads (v_cache None), or a
  (num_pages, page_size, nheads_k, head_dim) K/V pair. seqused_k: (batch,)
  kv length per sequence, its new tokens included; the cache must already
  hold them. block_table: (batch, pages_per_seq) int; entries past a
  sequence's first ceil(seqused_k / page_size) pages are never dereferenced.

  Query row t of sequence r attends keys [0, seqused_k[r] - q_len[r] + t]
  when causal, [0, seqused_k[r]) otherwise; a row with no visible key is
  unspecified. q_scale multiplies each staged Q tile (softmax_scale *
  log2(e)). Rows [cu_seqlens_q[-1], total_q) are out = 0 and lse = -inf.
  Rows below cu_seqlens_q[0] are unspecified, for the caller to restore: the
  first block of each head group starts on the 8-row grid, or lower when its
  window is pulled back to fit the bq-padded q buffer.

  max_seqlen_q / max_seqlen_k bound the longest sequence's q rows and kv
  length (new tokens included); they pick the tiles only, so an
  underestimate is slow, not wrong. Their power-of-two buckets, capped at the
  128-padded total_q and at the table row's capacity, key
  resolve_paged_tiles. block_sizes (block_kv whole pages) and
  kv_heads_per_group, how many KV heads one head group folds with all their
  q heads, pin the tiles that resolve_paged_tiles otherwise picks; a build
  over the scoped VMEM budget raises. rotary is the (q coefficients, None)
  pair of prepare_rotary; the cache holds K already rotated. Returns out
  (nheads, total_q, head_dim), plus lse (nheads, total_q) float32 when
  return_lse.
  """
  is_merged = v_cache is None
  if q.ndim != 3:
    raise ValueError(f"q must be (nheads, total_q, head_dim); got {q.shape}.")
  num_q_heads, total_q, head_dim = q.shape
  if k_cache.ndim != 4:
    raise ValueError(
        "k_cache must be (num_pages, page_size, heads, head_dim); got"
        f" {k_cache.shape}."
    )
  num_pages, page_size, cache_heads, cache_head_dim = k_cache.shape
  if is_merged:
    if cache_heads % 2:
      raise ValueError(
          "a merged cache holds 2 * nheads_k heads per token row; got"
          f" {cache_heads}."
      )
    num_kv_heads = cache_heads // 2
    cache_operands = (k_cache,)
  else:
    if v_cache.shape != k_cache.shape:
      raise ValueError(
          "a K/V cache pair must share one shape; got"
          f" {k_cache.shape=}, {v_cache.shape=}."
      )
    num_kv_heads = cache_heads
    cache_operands = (k_cache, v_cache)
  if any(operand.dtype != jnp.bfloat16 for operand in (q, *cache_operands)):
    raise NotImplementedError(
        "paged attention supports bfloat16 q and cache only; got"
        f" {tuple(str(x.dtype) for x in (q, *cache_operands))}."
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
  if num_kv_heads <= 0 or num_q_heads % num_kv_heads:
    raise ValueError(
        f"nheads={num_q_heads} must be a multiple of nheads_k={num_kv_heads}."
    )
  if total_q == 0:
    raise ValueError("q needs at least one token row.")

  cu_seqlens_q = jnp.asarray(cu_seqlens_q)
  seqused_k = jnp.asarray(seqused_k)
  block_table = jnp.asarray(block_table)
  if (cu_seqlens_q.ndim != 1 or cu_seqlens_q.shape[0] < 2
      or not jnp.issubdtype(cu_seqlens_q.dtype, jnp.integer)):
    raise ValueError(
        "cu_seqlens_q must be a 1-D integer array of batch + 1 boundaries;"
        f" got {cu_seqlens_q.shape} {cu_seqlens_q.dtype}."
    )
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
  for arg_name, seqlen_bound in (("max_seqlen_q", max_seqlen_q),
                                 ("max_seqlen_k", max_seqlen_k)):
    if type(seqlen_bound) is not int or seqlen_bound <= 0:
      raise ValueError(
          f"{arg_name} must be a positive static int; got {seqlen_bound!r}.")

  if kv_heads_per_group is not None and (
      type(kv_heads_per_group) is not int or kv_heads_per_group <= 0
      or num_kv_heads % kv_heads_per_group):
    raise ValueError(
        f"kv_heads_per_group must be a positive divisor of {num_kv_heads=};"
        f" got {kv_heads_per_group!r}."
    )
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
  q_heads_per_kv_head = num_q_heads // num_kv_heads
  if rotary is None:
    rotary_dtype = None
  else:
    rotary_dtype = jnp.dtype(rotary[0].dtype)
  # Note (david): the bounds ride in the tile cache key, so power-of-two
  # buckets keep a wobbling longest sequence on one build; no q block need
  # pass the 128-padded buffer and no sequence outgrows its table row.
  block_sizes, kv_heads_per_group = resolve_paged_tiles(
      max_seqlen_q_bucket=min(
          next_pow2(max_seqlen_q), round_up(total_q, NUM_LANES)),
      max_seqlen_k_bucket=min(next_pow2(max_seqlen_k), capacity),
      page_size=page_size,
      pages_per_seq=pages_per_seq,
      num_kv_heads=num_kv_heads,
      q_heads_per_kv_head=q_heads_per_kv_head,
      head_dim=head_dim,
      is_merged=is_merged,
      return_lse=bool(return_lse),
      rotary_dtype=rotary_dtype,
      block_sizes=block_sizes,
      kv_heads_per_group=kv_heads_per_group,
  )
  bq = block_sizes.block_q
  head_fold = kv_heads_per_group * q_heads_per_kv_head
  paged = paged_kv_info(
      num_kv_heads=num_kv_heads, kv_heads_per_group=kv_heads_per_group,
      page_size=page_size, pages_per_seq=pages_per_seq, is_merged=is_merged,
      interpret=bool(interpret))
  if is_merged:
    k_in, v_in = k_cache, None
  elif paged.is_cache_head_major:
    k_in, v_in = (
        cache.reshape(num_pages, 1, page_size, head_dim)
        for cache in cache_operands)
  else:
    k_in, v_in = k_cache, v_cache

  padded_total_q = round_up(total_q, bq)
  num_pad_q = padded_total_q - total_q
  q_in = jnp.pad(q, ((0, 0), (0, num_pad_q), (0, 0)))
  if rotary is None:
    rotary_in = None
  else:
    q_coeff, k_coeff = rotary
    if k_coeff is not None:
      raise ValueError(
          "a paged cache holds K already rotated; pass no K coefficients.")
    rotary_in = (
        jnp.pad(q_coeff, ((0, 0), (0, 0), (0, num_pad_q), (0, 0))), None)

  cu_q = cu_seqlens_q.astype(jnp.int32)
  smem_operands = [
      cu_q,
      seqused_k.astype(jnp.int32),
      per_seq_qblk_prefix(cu_q, bq),
      block_table.astype(jnp.int32).reshape(-1),
  ]
  kernel = partial(
      flash_fwd_varlen_paged_kernel,
      causal=bool(causal),
      num_head_groups=num_q_heads // head_fold,
      q_heads_per_kv_head=q_heads_per_kv_head,
  )
  outputs = forward_common(
      kernel, smem_operands, q_in, k_in, v_in,
      block_sizes=block_sizes,
      num_kv_heads=num_kv_heads,
      head_fold=head_fold,
      transposed_pv=False,
      return_lse=bool(return_lse),
      kernel_name="flash_attn_varlen_paged_fwd",
      interpret=bool(interpret),
      token_major=None,
      is_per_seq=True,
      rotary=rotary_in,
      rotary_interleaved=rotary_interleaved,
      paged=paged,
      guard_threshold=overflow_guard_threshold(capacity),
      window=(None, None),
      causal_offset=0,
      softcap=0.0,
      q_scale=float(q_scale),
  )
  if return_lse:
    out, lse = outputs
    return out[:, :total_q], lse[:, :total_q]
  else:
    return outputs[:, :total_q]
