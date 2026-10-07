"""Arithmetic block schedules driving the kernels' nested loops.

Block coordinates, bounds and flags are derived from loop indices by scalar
arithmetic, so the kernels iterate exactly the surviving blocks with no
schedule operand. In-kernel integer division uses jnp's floor semantics.
"""

import dataclasses
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from .block_sizes import NUM_SUBLANES


class PerSeqIntervals(NamedTuple):
  """row_interval context of the per-seq schedule (all traced scalars).

  bottom_right_offset is kv_end - q_end, the sequence's causal offset;
  q_window_start is the block's DMA-clamped first q token; kv_window_base is
  the aligned base of the sequence's kv windows. seq_idx and kv_used_end are
  set only by the paged schedule: the block's sequence, whose block table row
  its kv DMAs read, and that sequence's kv length. There kv_end is the q
  block's kv frontier (one past the last key any of its rows attends), which
  is what the block loads and owns; bottom_right_offset still uses kv_used_end.
  """
  q_start: jax.Array
  q_end: jax.Array
  kv_start: jax.Array
  kv_end: jax.Array
  bottom_right_offset: jax.Array
  q_window_start: jax.Array
  kv_window_base: jax.Array
  seq_idx: jax.Array | None = None
  kv_used_end: jax.Array | None = None


BlockIndex = jax.Array | int
ScheduleContext = PerSeqIntervals | None


@dataclasses.dataclass(frozen=True)
class FwdSchedule:
  """Nested-loop bounds and per-block arithmetic for fwd_body.

  Loop order: head group g (static bound), q block row si (static bound,
  qi = first_row + si), kv block in [lo, hi] (runtime bounds).

  Attributes:
    row_interval: si -> (lo, hi, ctx); hi inclusive.
    block_flags: (si, kv, ctx) -> (is_partial, needs_bounds).
    q_token, kv_token, q_bounds, blend_start: set only by the per-seq
      schedule, which addresses HBM by token offset and synthesizes its bounds
      in-register. q_token(ctx) and kv_token(kv, ctx) are the clamped packed
      window starts; q_bounds(ctx, kv, k_abs) is (lower, span), the absolute q
      lower-bound row plus, when has_q_span, the row of span widths: a q row
      attends a column iff lower <= q, and q - lower < span under has_q_span.
      blend_start(ctx) is the first row the block owns, and fwd_body blends
      rows [q_token, blend_start) of the output stage from the previous block
      before the packed write.
    has_q_span: per-seq only; q_bounds returns a span row (a left window bounds
      q from above), otherwise its span is None.
    heads_outer_batch: dense head-major only. None is the batch-outer fold
      (q fold index h reads kv fold index h // q_heads_per_kv_head); an int B
      is a heads-outer fold of B sequences (q fold index n * B + b reads kv
      fold index (n // q_heads_per_kv_head) * B + b).
    packed_q_end: per-seq only; cu_seqlens_q[-1], where the packed padding
      rows fwd_body zero-fills begin.
    kv_page, kv_frontier: set only by the paged schedule. kv_page(seq_idx,
      kv, page) is the physical page holding the page-th page of kv block kv
      of sequence seq_idx; kv_frontier(ctx, q_last) is one past the last key
      that rows up to packed row q_last attend, so fwd_body skips compute
      tiles at or past it.
  """
  num_head_groups: int
  num_rows: int | jax.Array
  first_row: int
  num_kv_blocks: int
  q_heads_per_kv_head: int
  row_interval: Callable[
      [BlockIndex], tuple[BlockIndex, BlockIndex, ScheduleContext]]
  block_flags: Callable[
      [BlockIndex, BlockIndex, ScheduleContext], tuple[jax.Array, jax.Array]]
  q_token: Callable[[PerSeqIntervals], jax.Array] | None = None
  kv_token: Callable[[BlockIndex, PerSeqIntervals], jax.Array] | None = None
  q_bounds: Callable[
      [PerSeqIntervals, BlockIndex, jax.Array],
      tuple[jax.Array, jax.Array | None]] | None = None
  has_q_span: bool = False
  blend_start: Callable[[PerSeqIntervals], jax.Array] | None = None
  heads_outer_batch: int | None = None
  kv_page: Callable[[jax.Array, BlockIndex, int], jax.Array] | None = None
  packed_q_end: jax.Array | None = None
  kv_frontier: Callable[[PerSeqIntervals, jax.Array], jax.Array] | None = None


class DenseFwdParams(NamedTuple):
  """Static (hashable) schedule inputs, resolved at build time."""
  num_head_groups: int
  q_heads_per_kv_head: int
  num_kv_blocks: int
  bq: int
  bkv: int
  left: int | None
  right: int | None
  offset: int
  first_row: int
  last_row: int
  heads_outer_batch: int | None = None


def dense_row_bounds(
    qi: BlockIndex, params: DenseFwdParams
) -> tuple[BlockIndex, BlockIndex]:
  """(lo, hi) kv-block interval of q block row qi; hi inclusive.

  A kv block survives iff floor((q_first + offset - left) / bkv) <= kv <=
  floor((q_last + offset + right) / bkv).
  """
  # Note (david): host ints stay pure Python. Host builders run inside the
  # caller's jit trace, where a jnp op would stage a tracer the builder cannot
  # branch on.
  is_traced = isinstance(qi, jax.Array)
  maximum = jnp.maximum if is_traced else max
  minimum = jnp.minimum if is_traced else min
  q_first = qi * params.bq
  q_last = q_first + params.bq - 1
  if params.left is None:
    lo = jnp.int32(0) if is_traced else 0
  else:
    lo = maximum(0, (q_first + params.offset - params.left) // params.bkv)
  if params.right is None:
    hi = params.num_kv_blocks - 1
  else:
    hi = minimum(
        params.num_kv_blocks - 1,
        (q_last + params.offset + params.right) // params.bkv,
    )
  return lo, hi


def dense_fwd_params(
    *,
    num_q_heads: int,
    head_fold: int,
    q_heads_per_kv_head: int,
    seqlen_q: int,
    seqlen_kv: int,
    bq: int,
    bkv: int,
    left: int | None,
    right: int | None,
    offset: int,
    heads_outer_batch: int | None = None,
) -> DenseFwdParams:
  """Host-side schedule resolution, including the nonempty q block row range.

  lo and hi are monotone in qi, so rows with no surviving block form a prefix
  and/or suffix and [first_row, last_row] is contiguous. Raises when every row
  is empty.
  """
  if seqlen_q % bq or seqlen_kv % bkv:
    raise ValueError(f"{bq=}/{bkv=} must divide {seqlen_q=}/{seqlen_kv=}.")
  if num_q_heads % head_fold:
    raise ValueError(f"{head_fold=} must divide {num_q_heads=}.")
  num_q_blocks = seqlen_q // bq
  num_kv_blocks = seqlen_kv // bkv
  rowless_params = DenseFwdParams(
      num_q_heads // head_fold, q_heads_per_kv_head,
      num_kv_blocks, bq, bkv, left, right, offset, 0, 0)
  nonempty_rows = []
  for qi in range(num_q_blocks):
    kv_lo, kv_hi = dense_row_bounds(qi, rowless_params)
    if kv_lo <= kv_hi:
      nonempty_rows.append(qi)
  if not nonempty_rows:
    raise ValueError(
        f"every q block row is fully masked ({left=}, {right=}, {offset=}).")
  assert nonempty_rows == list(range(nonempty_rows[0], nonempty_rows[-1] + 1))
  return rowless_params._replace(
      first_row=nonempty_rows[0], last_row=nonempty_rows[-1],
      heads_outer_batch=heads_outer_batch)


def make_dense_fwd_schedule(params: DenseFwdParams) -> FwdSchedule:
  """Dense schedule; params.offset may be a runtime scalar."""
  def _row_interval(si):
    lo, hi = dense_row_bounds(params.first_row + si, params)
    return lo, hi, None

  def _block_flags(si, kv, _):
    q_first = (params.first_row + si) * params.bq
    q_last = q_first + params.bq - 1
    k_first = kv * params.bkv
    k_last = k_first + params.bkv - 1
    if params.right is None:
      is_inside_right_edge = jnp.bool_(True)
    else:
      is_inside_right_edge = q_first + params.offset + params.right >= k_last
    if params.left is None:
      is_inside_left_edge = jnp.bool_(True)
    else:
      is_inside_left_edge = q_last + params.offset - params.left <= k_first
    is_full = jnp.logical_and(is_inside_right_edge, is_inside_left_edge)
    return jnp.logical_not(is_full), jnp.bool_(False)

  return FwdSchedule(
      num_head_groups=params.num_head_groups,
      num_rows=params.last_row - params.first_row + 1,
      first_row=params.first_row,
      num_kv_blocks=params.num_kv_blocks,
      q_heads_per_kv_head=params.q_heads_per_kv_head,
      row_interval=_row_interval,
      block_flags=_block_flags,
      heads_outer_batch=params.heads_outer_batch,
  )


def make_runtime_offset_dense_fwd_schedule(
    params: DenseFwdParams, offset: jax.Array) -> FwdSchedule:
  """Dense causal schedule whose absolute Q offset is a runtime scalar.

  The nonempty row range was resolved on host from the static offset the
  caller validated; only the in-kernel bounds and flags read the runtime one.
  """
  return make_dense_fwd_schedule(params._replace(offset=offset))


class VarlenSchedule(NamedTuple):
  """Build-time shape of the runtime (cu_seqlens-driven) per-seq schedule.

  Attributes:
    num_q_blocks: q blocks of the padded packed q axis.
    num_kv_blocks: kv blocks of the padded packed kv axis.
    q_heads_per_kv_head: maps a q head to its kv head.
    left/right: the window around each sequence's own bottom-right diagonal;
      None is unbounded, and causal is (None, 0).
  """

  num_q_blocks: int
  num_kv_blocks: int
  q_heads_per_kv_head: int
  left: int | None
  right: int | None


def seq_index_at(pos: jax.Array, cu_src_ref: jax.Array) -> jax.Array:
  """The sequence pos belongs to: the last s with cu_src[s] <= pos.

  Branchless binary lifting, unrolled at trace time over the static boundary
  count; cu_src[0] == 0 <= pos, so s = 0 starts valid. Works on SMEM refs
  in-kernel and on jnp arrays on host.
  """
  last_seq = cu_src_ref.shape[0] - 2
  seq_idx = jnp.int32(0)
  step = 1 << ((last_seq + 1).bit_length() - 1)
  while step:
    candidate = seq_idx + step
    clamped_candidate = jnp.minimum(candidate, last_seq)
    is_reached = jnp.logical_and(
        candidate <= last_seq, cu_src_ref[clamped_candidate] <= pos)
    seq_idx = jnp.where(is_reached, candidate, seq_idx)
    step //= 2
  return seq_idx


def align_to_sublane_tile(token: jax.Array) -> jax.Array:
  """Snap a token offset down to the sublane tile the token axis is tiled in.

  Mosaic rejects a dynamic second-minor offset it cannot prove tile-aligned,
  and it can prove nothing about a cu_seqlens scalar read from SMEM, so the
  per-seq schedule puts every window base on this grid and asserts it with
  pl.multiple_of.
  """
  return (token // NUM_SUBLANES) * NUM_SUBLANES


def per_seq_qblk_prefix(cu_seqlens_q: jax.Array, bq: int) -> jax.Array:
  """Host-side (jit-trace) prefix sum of per-sequence q-block counts.

  Entry r is the flat index of sequence r's first q block; the last entry is
  the runtime total the kernel's middle loop runs to. A sequence covers
  ceil((cu_q[r + 1] - align(cu_q[r])) / bq) blocks on the aligned grid, and an
  empty sequence covers none. Rows past cu_q[-1] get no block.
  """
  cu_q = jnp.asarray(cu_seqlens_q, jnp.int32)
  num_blocks_per_seq = jnp.where(
      jnp.diff(cu_q) > 0,
      -(-(cu_q[1:] - align_to_sublane_tile(cu_q[:-1])) // bq), 0)
  return jnp.concatenate(
      [jnp.zeros((1,), jnp.int32),
       jnp.cumsum(num_blocks_per_seq).astype(jnp.int32)])


def make_per_seq_fwd_schedule(
    cu_q_ref: jax.Array,
    cu_k_ref: jax.Array | None,
    cu_qblk_ref: jax.Array,
    *,
    num_head_groups: int,
    q_heads_per_kv_head: int,
    padded_total_q: int,
    padded_total_k: int,
    bq: int,
    bkv: int,
    left: int | None,
    right: int | None,
    num_rows: jax.Array,
) -> FwdSchedule:
  """Per-sequence q blocks on the aligned token grid; no block crosses a
  sequence boundary.

  si is a flat index over every sequence's q blocks (cu_qblk_ref is
  per_seq_qblk_prefix's output in SMEM; num_rows its last entry). Block i of
  sequence r starts at align(cu_q[r]) + i * bq, and its kv interval covers only
  sequence r's kv blocks on the grid from align(cu_k[r]), cut to the window.
  left / right bound the window around the sequence's own bottom-right
  diagonal: q attends k iff q + offset - left <= k <= q + offset + right, with
  offset = cu_k[r + 1] - cu_q[r + 1]. None is unbounded, and causal is
  (None, 0). Empty sequences own no si.

  Aligning a base down drags up to 7 rows of the previous sequence into the
  first block, and the last block runs past q_end. Those q rows are garbage:
  fwd_body blends rows [q_token, blend_start) from the previous block's stage
  before the packed write, and tail rows are overwritten by the next block or,
  past cu_q[-1] (packed_q_end), written as out = 0, lse = -inf. Stray kv
  columns and the window edges are masked through q_bounds, so this schedule
  needs no bounds operand.
  """
  # Note (david): pl.multiple_of is an unchecked hint, and a lying hint
  # corrupts data silently instead of failing the verifier. A base of
  # align(cu) + i * bq is tile-aligned only while bq and bkv are, and the
  # clamped bases (padded_total - block) need the padded totals aligned too.
  if (bq % NUM_SUBLANES or bkv % NUM_SUBLANES
      or padded_total_q % NUM_SUBLANES or padded_total_k % NUM_SUBLANES):
    raise ValueError(
        f"{bq=}, {bkv=}, {padded_total_q=} and {padded_total_k=} must be"
        f" multiples of {NUM_SUBLANES}."
    )
  num_kv_blocks = padded_total_k // bkv

  def _row_interval(si):
    seq_idx = seq_index_at(si, cu_qblk_ref)
    block_in_seq = si - cu_qblk_ref[seq_idx]
    q_start, q_end = cu_q_ref[seq_idx], cu_q_ref[seq_idx + 1]
    kv_start, kv_end = cu_k_ref[seq_idx], cu_k_ref[seq_idx + 1]
    kv_window_base = align_to_sublane_tile(kv_start)
    q_block_start = align_to_sublane_tile(q_start) + block_in_seq * bq
    q_window_start = jnp.minimum(q_block_start, padded_total_q - bq)
    bottom_right_offset = kv_end - q_end
    if right is None:
      kv_hi_token = kv_end - 1
    else:
      last_valid_q = jnp.minimum(q_block_start + bq, q_end) - 1
      kv_hi_token = jnp.minimum(
          kv_end - 1, last_valid_q + bottom_right_offset + right)
    # Note (david): a degenerate cu (kv_len == 0) still gets one kv block,
    # because every row must write its output.
    hi = jnp.clip((kv_hi_token - kv_window_base) // bkv, 0, num_kv_blocks - 1)
    if left is None:
      lo = jnp.int32(0)
    else:
      # Note (david): a window clamped at the padded end recomputes rows of its
      # own sequence below q_block_start, which the packed write keeps, so the
      # left edge starts from the clamped window's first own row.
      first_kept_q = jnp.maximum(q_start, q_window_start)
      kv_lo_token = jnp.maximum(
          kv_start, first_kept_q + bottom_right_offset - left)
      lo = jnp.clip((kv_lo_token - kv_window_base) // bkv, 0, hi)
    ctx = PerSeqIntervals(
        q_start, q_end, kv_start, kv_end, bottom_right_offset,
        q_window_start, kv_window_base)
    return lo, hi, ctx

  def _q_token(ctx):
    return pl.multiple_of(ctx.q_window_start, NUM_SUBLANES)

  def _kv_token(kv, ctx):
    return pl.multiple_of(
        jnp.minimum(ctx.kv_window_base + kv * bkv, padded_total_k - bkv),
        NUM_SUBLANES)

  def _block_flags(_, kv, ctx):
    k_first = ctx.kv_window_base + kv * bkv
    k_last = k_first + bkv - 1
    # Note (david): a block needs the mask iff it crosses the sequence's kv end
    # (which covers every window clamped at the padded axis end), starts before
    # the sequence (the aligned base reaches into the previous one), or a
    # window edge crosses it. The right edge is checked from the clamped
    # q_window_start, since a window clamped at the padded end recomputes the
    # same sequence's lower rows, which the packed write keeps; the left edge
    # from the last row the write keeps. Rows past q_end are overwritten or
    # discarded, so they never force the mask.
    needs_mask = jnp.logical_or(k_last >= ctx.kv_end, k_first < ctx.kv_start)
    if right is not None:
      needs_mask = jnp.logical_or(
          needs_mask,
          k_last > ctx.q_window_start + ctx.bottom_right_offset + right)
    if left is not None:
      last_kept_q = jnp.minimum(ctx.q_window_start + bq, ctx.q_end) - 1
      needs_mask = jnp.logical_or(
          needs_mask, k_first < last_kept_q + ctx.bottom_right_offset - left)
    return jnp.bool_(False), needs_mask

  def _q_bounds(ctx, kv, k_abs):
    # Note (david): kept q rows are always below q_end (rows at or above it are
    # overwritten or discarded with the pad), so only a left window needs an
    # upper bound. Invalid kv columns get the first q id past the physical
    # block, which fails the lower-bound check for every row and wraps every
    # row past the unsigned span check.
    if right is None:
      lower = jnp.broadcast_to(ctx.q_start, k_abs.shape)
    else:
      lower = jnp.maximum(
          ctx.q_start, k_abs - ctx.bottom_right_offset - right)
    owned_lo = jnp.maximum(ctx.kv_start, ctx.kv_window_base + kv * bkv)
    is_owned = jnp.logical_and(k_abs >= owned_lo, k_abs < ctx.kv_end)
    q_lower = jnp.where(is_owned, lower, ctx.q_window_start + bq)
    if left is None:
      return q_lower, None
    else:
      q_upper = k_abs - ctx.bottom_right_offset + left
      return q_lower, jnp.maximum(q_upper + 1 - q_lower, 0)

  return FwdSchedule(
      num_head_groups=num_head_groups,
      num_rows=num_rows,
      first_row=0,
      num_kv_blocks=num_kv_blocks,
      q_heads_per_kv_head=q_heads_per_kv_head,
      row_interval=_row_interval,
      block_flags=_block_flags,
      q_token=_q_token,
      kv_token=_kv_token,
      q_bounds=_q_bounds,
      has_q_span=left is not None,
      blend_start=lambda ctx: ctx.q_start,
      packed_q_end=cu_q_ref[cu_q_ref.shape[0] - 1],
  )


def make_paged_fwd_schedule(
    cu_q_ref: jax.Array,
    seqused_k_ref: jax.Array,
    cu_qblk_ref: jax.Array,
    block_table_ref: jax.Array,
    *,
    num_head_groups: int,
    q_heads_per_kv_head: int,
    padded_total_q: int,
    bq: int,
    bkv: int,
    page_size: int,
    pages_per_seq: int,
    causal: bool,
    num_rows: jax.Array,
) -> FwdSchedule:
  """Per-sequence q blocks attending a paged KV cache.

  The q side is make_per_seq_fwd_schedule's. Sequence r's kv axis is its own
  [0, seqused_k[r]), read through its row of the flat (batch * pages_per_seq,)
  block_table_ref. Rows past cu_q[-1] get no q block; fwd_body zero-fills them
  from packed_q_end.
  """
  num_kv_blocks = -(-(pages_per_seq * page_size) // bkv)
  pages_per_block = bkv // page_size
  # Note (david): only the per-seq row_interval reads cu_k_ref, and it is
  # replaced below, so the base schedule is built without one.
  per_seq_schedule = make_per_seq_fwd_schedule(
      cu_q_ref, None, cu_qblk_ref,
      num_head_groups=num_head_groups,
      q_heads_per_kv_head=q_heads_per_kv_head,
      padded_total_q=padded_total_q,
      padded_total_k=num_kv_blocks * bkv,
      bq=bq, bkv=bkv, left=None, right=0 if causal else None,
      num_rows=num_rows,
  )

  def _kv_frontier(kv_end, bottom_right_offset, q_last):
    if causal:
      return jnp.minimum(kv_end, q_last + bottom_right_offset + 1)
    else:
      return kv_end

  def _row_interval(si):
    seq_idx = seq_index_at(si, cu_qblk_ref)
    block_in_seq = si - cu_qblk_ref[seq_idx]
    q_start, q_end = cu_q_ref[seq_idx], cu_q_ref[seq_idx + 1]
    kv_start, kv_used_end = jnp.int32(0), seqused_k_ref[seq_idx]
    q_block_start = align_to_sublane_tile(q_start) + block_in_seq * bq
    q_window_start = jnp.minimum(q_block_start, padded_total_q - bq)
    bottom_right_offset = kv_used_end - q_end
    # Note (david): the block loads and owns keys only up to its own causal
    # frontier, so a q block early in a chunk neither fetches nor scores the
    # keys only later rows see. Rows of the window outside the sequence are
    # overwritten or discarded, so the frontier follows the last owned row.
    last_valid_q = jnp.minimum(q_block_start + bq, q_end) - 1
    kv_end = _kv_frontier(kv_used_end, bottom_right_offset, last_valid_q)
    # Note (david): kv_len == 0 still gets one kv block, as on the per-seq
    # schedule; its page DMAs clip to zero tokens.
    hi = jnp.clip((kv_end - 1) // bkv, 0, num_kv_blocks - 1)
    ctx = PerSeqIntervals(
        q_start, q_end, kv_start, kv_end, bottom_right_offset,
        q_window_start, kv_start, seq_idx, kv_used_end)
    return jnp.int32(0), hi, ctx

  def _kv_page(seq_idx, kv, page):
    # Note (david): a lookup past the row's last page clamps into the row; its
    # DMA clips to zero tokens, so the entry is never dereferenced.
    table_index = jnp.minimum(kv * pages_per_block + page, pages_per_seq - 1)
    return block_table_ref[seq_idx * pages_per_seq + table_index]

  return dataclasses.replace(
      per_seq_schedule,
      row_interval=_row_interval,
      kv_page=_kv_page,
      kv_frontier=lambda ctx, q_last: _kv_frontier(
          ctx.kv_end, ctx.bottom_right_offset, q_last),
  )


@dataclasses.dataclass(frozen=True)
class BwdSchedule:
  """Kv-outer nested-loop schedule for bwd_body: group -> kv -> qi.

  Attributes:
    row_interval: kv -> (qlo, qhi, ctx); qhi inclusive.
    block_flags: (kv, qi, ctx) -> (is_dq_first, is_dq_last, needs_mask).
  """
  num_head_groups: int
  num_kv_blocks: int
  num_q_blocks: int
  row_interval: Callable[
      [BlockIndex], tuple[BlockIndex, BlockIndex, ScheduleContext]]
  block_flags: Callable[
      [BlockIndex, BlockIndex, ScheduleContext],
      tuple[jax.Array, jax.Array, jax.Array | bool]]


class DenseBwdParams(NamedTuple):
  num_heads: int
  num_q_blocks: int
  num_kv_blocks: int
  bq: int
  bkv: int
  causal: bool
  offset: int


def first_visible_q_block(
    kv: BlockIndex, params: DenseBwdParams
) -> BlockIndex:
  """Smallest qi whose block's last row reaches kv block kv."""
  # Note (david): host ints stay pure Python. The host builder runs inside the
  # caller's jit trace, where a jnp op would stage a tracer that int() rejects.
  is_traced = isinstance(kv, jax.Array)
  if params.causal:
    first_row_numerator = kv * params.bkv - params.offset - params.bq + 1
    maximum = jnp.maximum if is_traced else max
    return maximum(0, -((-first_row_numerator) // params.bq))
  elif is_traced:
    return jnp.int32(0)
  else:
    return 0


def dense_bwd_params(
    *,
    num_heads: int,
    num_q_blocks: int,
    num_kv_blocks: int,
    bq: int,
    bkv: int,
    causal: bool,
    offset: int,
) -> DenseBwdParams:
  params = DenseBwdParams(
      num_heads, num_q_blocks, num_kv_blocks, bq, bkv, causal, offset)
  # Note (david): the first visible q block is monotone in kv, so the last kv
  # block is the worst case.
  if causal and int(
      first_visible_q_block(num_kv_blocks - 1, params)) >= num_q_blocks:
    raise ValueError(
        f"kv block {num_kv_blocks - 1} (of {num_kv_blocks}) is unreachable"
        f" by every q block under causal_offset={offset}, bq={bq},"
        f" bkv={bkv}; its dk/dv would never be written."
    )
  return params


def make_dense_bwd_schedule(params: DenseBwdParams) -> BwdSchedule:
  num_q_blocks, num_kv_blocks = params.num_q_blocks, params.num_kv_blocks
  bq, bkv, offset = params.bq, params.bkv, params.offset

  def _row_interval(kv):
    return (first_visible_q_block(kv, params), jnp.int32(num_q_blocks - 1),
            None)

  def _block_flags(kv, qi, _):
    # Note (david): the Python-bool needs_mask cases keep bwd_body's static
    # single-copy dispatch.
    if params.causal:
      last_kv_block = jnp.minimum(
          num_kv_blocks - 1, (qi * bq + bq - 1 + offset) // bkv)
    else:
      last_kv_block = num_kv_blocks - 1
    if not params.causal:
      needs_mask = False
    elif (num_q_blocks - 1) * bq + offset < bkv - 1:
      needs_mask = True
    else:
      needs_mask = qi * bq + offset < kv * bkv + bkv - 1
    return kv == 0, kv == last_kv_block, needs_mask

  return BwdSchedule(
      num_head_groups=params.num_heads,
      num_kv_blocks=num_kv_blocks,
      num_q_blocks=num_q_blocks,
      row_interval=_row_interval,
      block_flags=_block_flags,
  )
