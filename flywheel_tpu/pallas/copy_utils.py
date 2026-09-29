"""DMA stream helpers shared by the forward and backward kernels."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl

from .block_sizes import NUM_LANES, QKVLayout, TokenMajorInfo


def advance(
    key: jax.Array,
    key_next: jax.Array,
    start_cur: Callable[[jax.Array], None],
    start_next: Callable[[jax.Array], None],
    wait: Callable[[jax.Array], None],
    loaded: jax.Array,
    inflight: jax.Array,
    slot: jax.Array,
    num_stages: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """One ring-buffer step of a DMA stream; returns (loaded, inflight, slot).

  Invariant at the top of a step: slot holds (or is receiving) the block for
  key loaded unless loaded == -1, and at most this step's own block is in
  flight, prefetched by the previous step as key inflight. The step waits for
  key's block, starting it first on a prefetch miss, then prefetches key_next
  into the following slot when it differs from key.
  """
  is_new = key != loaded
  is_cold = jnp.logical_and(is_new, key != inflight)
  next_slot = lax.rem(slot + 1, num_stages)

  @pl.when(is_cold)
  def _():
    start_cur(next_slot)

  @pl.when(is_new)
  def _():
    wait(next_slot)

  new_slot = jnp.where(is_new, next_slot, slot)
  should_prefetch = key_next != key

  @pl.when(should_prefetch)
  def _():
    start_next(lax.rem(new_slot + 1, num_stages))

  new_inflight = jnp.where(should_prefetch, key_next, -1)
  return key, new_inflight, new_slot


def drain(
    num_copies: jax.Array,
    num_stages: int,
    wait: Callable[[int | jax.Array], None],
) -> None:
  """Wait out the last num_stages of num_copies started copies, oldest first.

  The copies still in flight occupy slots (num_copies - num_stages) ..
  (num_copies - 1); each wait is guarded by its copy having started at all.
  """
  for pending in range(num_stages, 0, -1):
    @pl.when(num_copies >= pending)
    def _wait_pending(pending=pending):
      wait((num_copies - pending) % num_stages)


def hbm_head_slice(
    ref: jax.Array,
    head: jax.Array | int,
    token_start: jax.Array | int,
    num_rows: int,
    layout: QKVLayout,
    fold: int,
) -> jax.Array:
  """Window of fold consecutive heads of a head-major ref as one HBM region.

  ref is (num_heads, seq, head_dim), or (num_heads, head_dim, seq) for
  SEQ_MINOR; head is the fold group's base head and the window spans num_rows
  tokens from token_start, so it needs no block alignment.
  """
  # Note (david): slicing the 3-D ref directly keeps the same HBM region; a
  # jnp.reshape outside the kernel made XLA materialize a copy whenever
  # head_dim needed lane padding.
  row_window = pl.ds(token_start, num_rows)
  head_window = head if fold == 1 else pl.ds(head, fold)
  if layout == QKVLayout.HEAD_DIM_MINOR:
    return ref.at[head_window, row_window, :]
  else:
    return ref.at[head_window, :, row_window]


def hbm_tm_slice(
    ref: jax.Array,
    head: jax.Array | int,
    token_start: jax.Array | int,
    num_rows: int,
    heads_per_row: int,
    width: int,
    is_batched: bool,
    fold: int,
) -> jax.Array:
  """Window of fold consecutive heads of a token-major ref, as lane columns.

  head is flat (batch * heads_per_row + local head) and, at fold > 1, the
  group's base head; heads_per_row and width are this stream's per-row head
  count and head_dim. is_batched addresses 3-D (batch, seq, heads * width)
  refs, otherwise 2-D (total, heads * width) refs. Consecutive heads occupy
  consecutive lanes, so a fold group is one contiguous column window; builders
  keep fold dividing heads_per_row, so a group never crosses a batch row.
  """
  batch_idx = head // heads_per_row
  local_head = head % heads_per_row
  row_window = pl.ds(token_start, num_rows)
  # Note (david): Mosaic needs a 128-aligned lane offset for an HBM DMA window.
  # A sub-tile width (d64) relies on builders enforcing
  # fold * width % 128 == 0; local_head is a multiple of fold, so the offset is
  # aligned, but the compiler cannot derive that from a traced head.
  if width % NUM_LANES:
    assert (fold * width) % NUM_LANES == 0, (fold, width)
    lane_start = pl.multiple_of(local_head * width, NUM_LANES)
  else:
    lane_start = local_head * width
  lane_window = pl.ds(lane_start, fold * width)
  if is_batched:
    return ref.at[batch_idx, row_window, lane_window]
  else:
    return ref.at[row_window, lane_window]


def hbm_window(
    ref: jax.Array,
    head: jax.Array | int,
    token_start: jax.Array | int,
    num_rows: int,
    *,
    width: int,
    heads_per_row: int,
    layout: QKVLayout,
    token_major: TokenMajorInfo | None,
    fold: int,
) -> jax.Array:
  """One stream's HBM window for whichever layout the build runs.

  Head-major (token_major is None) takes a leading-axis head window and layout
  picks the token axis; token-major takes a lane-axis column window of the
  stream's per-head width and ignores layout.
  """
  if token_major is None:
    return hbm_head_slice(ref, head, token_start, num_rows, layout, fold)
  else:
    return hbm_tm_slice(
        ref, head, token_start, num_rows, heads_per_row, width,
        token_major.batch is not None, fold)


def fold_row(
    ref: jax.Array,
    fold_idx: jax.Array | int,
    block: int | None,
    head_fold: int,
    layout: QKVLayout = QKVLayout.HEAD_DIM_MINOR,
    lane_width: int | None = None,
) -> jax.Array:
  """Select fold group fold_idx from a per-slot VMEM scratch ref.

  head_fold == 1 slices the plain (block, head_dim) ref, or (head_dim, block)
  for SEQ_MINOR. head_fold > 1 indexes the leading fold axis, or, when
  lane_width is set (token-major), the lane column window at
  fold_idx * lane_width.
  """
  # Note (david): fold_idx may be traced. Leading-axis indexing and
  # 128-aligned lane offsets are valid dynamically, but a sub-tile lane_width
  # (d64) gives only 64-aligned offsets, which Mosaic accepts for a static
  # fold_idx only, so those builds pass a Python int.
  if head_fold == 1 and layout == QKVLayout.HEAD_DIM_MINOR:
    return ref.at[pl.ds(fold_idx * block, block), :]
  elif head_fold == 1:
    return ref.at[:, pl.ds(fold_idx * block, block)]
  elif lane_width is not None:
    return ref.at[:, pl.ds(fold_idx * lane_width, lane_width)]
  else:
    return ref.at[fold_idx]
