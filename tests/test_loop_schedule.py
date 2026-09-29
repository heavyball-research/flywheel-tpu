"""Host replays of the block schedules against the element-level mask.

Replaying a schedule's nested loops must visit exactly the blocks the mask
makes non-empty and flag exactly the ones it does not fully cover. The oracle
is the mask built straight from its definition, never a second copy of the
schedule.
"""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

from flywheel_tpu.pallas.loop_schedule import (
    PerSeqIntervals,
    dense_bwd_params,
    dense_fwd_params,
    make_dense_bwd_schedule,
    make_dense_fwd_schedule,
    make_per_seq_fwd_schedule,
    per_seq_qblk_prefix,
)
from flywheel_tpu.pallas.seqlen_info import extend_cu_seqlens


def static_mask(q_len, k_len, left, right, offset):
  """q + offset - left <= k <= q + offset + right, None meaning unbounded."""
  q = np.arange(q_len)[:, None]
  k = np.arange(k_len)[None, :]
  left_visible = True if left is None else q + offset - left <= k
  right_visible = True if right is None else k <= q + offset + right
  return np.broadcast_to(left_visible & right_visible, (q_len, k_len))


def packed_mask(cu_q, cu_k, q_len, k_len, left, right):
  """Same sequence and flash_attn's bottom-right window on that sequence's own
  lengths (None unbounded, causal is (None, 0)); the pad tail is one more
  sequence."""
  cu_q = [*cu_q, q_len]
  cu_k = [*cu_k, k_len]
  mask = np.zeros((q_len, k_len), bool)
  for seq in range(len(cu_q) - 1):
    offset = cu_k[seq + 1] - cu_q[seq + 1]
    for row in range(cu_q[seq], cu_q[seq + 1]):
      for col in range(cu_k[seq], cu_k[seq + 1]):
        mask[row, col] = (
            (left is None or row + offset - left <= col)
            and (right is None or col <= row + offset + right))
  return mask


def block_truth(mask, bq, bkv):
  """Per (q block, kv block): any real work, and fully unmasked."""
  blocks = mask.reshape(mask.shape[0] // bq, bq, mask.shape[1] // bkv, bkv)
  return blocks.any(axis=(1, 3)), blocks.all(axis=(1, 3))


def replay_dense_fwd(params, head_fold):
  """Walks the schedule the way fwd_body's nested loops will."""
  sched = make_dense_fwd_schedule(params)
  entries = []
  for group in range(sched.num_head_groups):
    head = group * head_fold
    kv_head = head // sched.q_heads_per_kv_head
    for si in range(sched.num_rows):
      lo, hi, ctx = sched.row_interval(np.int32(si))
      lo, hi = int(lo), int(hi)
      # Note (david): rows between first_row and last_row are never empty.
      assert lo <= hi
      for kv in range(lo, hi + 1):
        partial, _ = sched.block_flags(np.int32(si), np.int32(kv), ctx)
        entries.append((head, sched.first_row + si, kv_head, kv, kv == lo,
                        kv == hi, bool(partial)))
  return entries


DENSE_CASES = [
    # (num_q_heads, qhpkh, seq_q, seq_kv, bq, bkv, causal, offset, window, fold)
    (4, 1, 1024, 1024, 256, 256, True, 0, None, 1),
    (4, 1, 1024, 1024, 256, 256, False, 0, None, 1),
    (4, 2, 1024, 2048, 256, 512, True, 1024, None, 1),
    (4, 1, 1024, 1024, 256, 128, False, 0, (256, 128), 1),
    (4, 1, 1024, 1024, 512, 256, True, -512, None, 1),
    (8, 1, 1024, 1024, 1024, 1024, True, 0, None, 4),
    (4, 1, 2048, 1024, 512, 256, False, 0, (None, 0), 1),
]


@pytest.mark.parametrize("case", DENSE_CASES)
def test_dense_fwd_schedule_matches_the_mask(case):
  # Note (david): the dense mask is static, so the schedule is held to it
  # exactly: every non-empty block in head/q-major, kv-minor order, flagged
  # partial on exactly the blocks the mask cuts.
  (num_q_heads, qhpkh, seq_q, seq_kv, bq, bkv, causal, offset, window,
   fold) = case
  if window is not None:
    left, right = window
  elif causal:
    left, right = None, 0
  else:
    left, right = None, None
  nonempty, unmasked = block_truth(
      static_mask(seq_q, seq_kv, left, right, offset), bq, bkv)
  expected = [
      (head, qi, head // qhpkh, int(kv), pos == 0, pos == kvs.size - 1,
       not unmasked[qi, kv])
      for head in range(0, num_q_heads, fold)
      for qi in range(seq_q // bq)
      for kvs in [np.nonzero(nonempty[qi])[0]]
      for pos, kv in enumerate(kvs)
  ]
  params = dense_fwd_params(
      num_q_heads=num_q_heads, head_fold=fold, q_heads_per_kv_head=qhpkh,
      seqlen_q=seq_q, seqlen_kv=seq_kv, bq=bq, bkv=bkv,
      left=left, right=right, offset=offset)
  assert replay_dense_fwd(params, fold) == expected


class PerSeqRow(NamedTuple):
  si: int
  staged_q_start: int
  aligned_q_start: int
  owned_rows: tuple[int, int]
  ctx: PerSeqIntervals
  blocks: list[tuple[int, int, bool]]


def replay_per_seq_fwd(cu_q, cu_k, q_pad, kv_pad, bq, bkv, window):
  """Walks one head group of the per-seq schedule; each visited block is
  (kv, first kv token, needs_bounds)."""
  cu_q_ext = extend_cu_seqlens(jnp.asarray(cu_q, jnp.int32), q_pad)
  cu_k_ext = extend_cu_seqlens(jnp.asarray(cu_k, jnp.int32), kv_pad)
  cu_qblk = per_seq_qblk_prefix(cu_q_ext, bq)
  num_rows = int(cu_qblk[-1])
  left, right = window
  sched = make_per_seq_fwd_schedule(
      cu_q_ext, cu_k_ext, cu_qblk, num_head_groups=1, q_heads_per_kv_head=1,
      padded_total_q=q_pad, padded_total_k=kv_pad, bq=bq, bkv=bkv,
      left=left, right=right, num_rows=num_rows)
  rows = []
  for si in range(num_rows):
    lo, hi, ctx = sched.row_interval(jnp.int32(si))
    # Note (david): ctx only carries the start clamped to the padded end, so
    # the unclamped one is rebuilt from the layout: sequence r's i-th block
    # starts at align8(cu_q[r]) + i * bq.
    seq = int(np.searchsorted(np.asarray(cu_qblk), si, side="right")) - 1
    aligned_q_start = (int(cu_q_ext[seq]) // 8 * 8
                       + (si - int(cu_qblk[seq])) * bq)
    blocks = []
    for kv in range(int(lo), int(hi) + 1):
      _, needs_bounds = sched.block_flags(jnp.int32(si), jnp.int32(kv), ctx)
      blocks.append(
          (kv, int(sched.kv_token(jnp.int32(kv), ctx)), bool(needs_bounds)))
    rows.append(PerSeqRow(
        si=si,
        staged_q_start=int(sched.q_token(ctx)),
        aligned_q_start=aligned_q_start,
        owned_rows=(max(int(ctx.q_start), aligned_q_start),
                    min(int(ctx.q_end), aligned_q_start + bq)),
        ctx=ctx,
        blocks=blocks,
    ))
  return sched, rows


PER_SEQ_CASES = [
    # (cu_q, cu_k, q_pad, kv_pad, bq, bkv)
    ([0, 512], [0, 512], 512, 512, 128, 128),
    ([0, 37, 90, 112], [0, 37, 90, 112], 128, 128, 128, 128),
    ([0, 256, 512], [0, 512, 1024], 512, 1024, 128, 128),
    ([0, 40, 100], [0, 200, 500], 128, 512, 128, 128),
    ([0, 0, 90], [0, 128, 400], 128, 512, 128, 128),
    ([0, 300, 364, 1024], [0, 590, 823, 1900], 1024, 2048, 256, 256),
    # Note (david): a misaligned start spills this sequence into an extra
    # aligned block (126 tokens, 2 blocks), with bq != bkv.
    ([0, 100, 226], [0, 150, 400], 256, 512, 128, 256),
    # Note (david): the last block clamps from 416 to 384 at the padded end, so
    # its recomputed rows sit below the intended diagonal and a kv block
    # between the two diagonals must not take the mask-free path.
    ([0, 128, 128, 165, 421], [0, 128, 128, 165, 421], 512, 512, 128, 128),
]

# Note (david): (left, right) around each sequence's bottom-right diagonal:
# full, causal, causal sliding, left-only, both edges off the diagonal, and the
# diagonal alone; the left edge reaches back past several kv blocks.
PER_SEQ_WINDOWS = [
    (None, None), (None, 0), (40, 0), (300, None), (37, 20), (0, 0)]


def kept_rows(row, bq):
  # Note (david): the packed-direct write publishes every own-sequence row of
  # the clamped window, clamp overlaps included, so all of them must be exact.
  return np.arange(max(int(row.ctx.q_start), row.staged_q_start),
                   min(int(row.ctx.q_end), row.staged_q_start + bq))


@pytest.mark.parametrize("window", PER_SEQ_WINDOWS)
@pytest.mark.parametrize("case", PER_SEQ_CASES)
def test_per_seq_fwd_schedule_windows_and_masks(case, window):
  cu_q, cu_k, q_pad, kv_pad, bq, bkv = case
  _, rows = replay_per_seq_fwd(cu_q, cu_k, q_pad, kv_pad, bq, bkv, window)
  mask = packed_mask(cu_q, cu_k, q_pad, kv_pad, *window)

  # Note (david): Mosaic DMAs need every token offset on the 8-row tile,
  # aligned cu_seqlens or not.
  for row in rows:
    assert row.aligned_q_start % 8 == 0 and row.staged_q_start % 8 == 0, row.si
    for kv, k_tok, _ in row.blocks:
      assert (int(row.ctx.kv_window_base) + kv * bkv) % 8 == 0, k_tok
      assert k_tok % 8 == 0, k_tok

  # Note (david): the owned ranges must partition [0, q_pad) in si order, so
  # every token is computed exactly once however the staged windows overlap.
  cursor = 0
  for row in rows:
    start, end = row.owned_rows
    assert start == cursor and start < end, (row.si, row.owned_rows)
    assert row.staged_q_start <= start and end <= row.staged_q_start + bq, (
        row.si, row.owned_rows)
    cursor = end
  assert cursor == q_pad

  for row in rows:
    valid_rows = kept_rows(row, bq)
    needed = set(np.nonzero(mask[valid_rows].any(axis=0))[0])
    covered = set()
    for _, k_tok, _ in row.blocks:
      covered |= set(range(k_tok, min(k_tok + bkv, kv_pad)))
    assert needed <= covered, (row.si, sorted(needed - covered)[:4])
    # Note (david): the interval runs from the block owning the first needed
    # column to the block owning the last one on the aligned kv grid; a row
    # with nothing to attend still keeps one block. The first block is exact
    # only when every kept row sees a key (len_k >= len_q under a right
    # bound): a row with none cannot pull the left edge in.
    visited = [kv for kv, _, _ in row.blocks]
    kv_window_base = int(row.ctx.kv_window_base)
    if needed:
      assert visited[-1] == (max(needed) - kv_window_base) // bkv, (
          row.si, visited)
      first_needed_block = (min(needed) - kv_window_base) // bkv
      if mask[valid_rows].any(axis=1).all():
        assert visited[0] == first_needed_block, (row.si, visited)
      else:
        assert visited[0] <= first_needed_block, (row.si, visited)
    else:
      assert len(visited) == 1, row.si
    # Note (david): unflagged blocks run the mask-free pipeline copy, so every
    # kept (row, column) pair they load must already be valid.
    for kv, k_tok, needs_bounds in row.blocks:
      assert needs_bounds or mask[np.ix_(valid_rows,
                                         np.arange(k_tok, k_tok + bkv))].all(), (
          row.si, kv)


@pytest.mark.parametrize("window", PER_SEQ_WINDOWS)
@pytest.mark.parametrize("case", PER_SEQ_CASES)
def test_per_seq_q_bounds_match_kept_rows(case, window):
  cu_q, cu_k, q_pad, kv_pad, bq, bkv = case
  sched, rows = replay_per_seq_fwd(cu_q, cu_k, q_pad, kv_pad, bq, bkv, window)
  mask = packed_mask(cu_q, cu_k, q_pad, kv_pad, *window)
  for row in rows:
    q_ids = kept_rows(row, bq)
    flagged_blocks = [block for block in row.blocks if block[2]]
    for kv, k_tok, _ in flagged_blocks:
      k_abs = jnp.arange(k_tok, k_tok + bkv, dtype=jnp.int32)[None, :]
      lower, span = sched.q_bounds(row.ctx, jnp.int32(kv), k_abs)
      assert (span is not None) == sched.has_q_span
      lower = np.asarray(lower)[0]
      actual = q_ids[:, None] >= lower[None, :]
      if span is not None:
        # Note (david): the kernel's one unsigned comparison, where a row below
        # lower wraps past any span.
        relative_ids = (q_ids[:, None] - lower[None, :]).astype(np.uint32)
        actual &= relative_ids < np.asarray(span)[0].astype(np.uint32)
      cols = np.arange(k_tok, k_tok + bkv)
      # Note (david): columns below the sequence start or below this block's
      # own window start belong to another block and must be fully masked;
      # the reference only speaks for the columns this block owns.
      owned = cols >= max(int(row.ctx.kv_start),
                          int(row.ctx.kv_window_base) + kv * bkv)
      np.testing.assert_array_equal(
          actual[:, owned], mask[np.ix_(q_ids, cols)][:, owned],
          err_msg=str((row.si, kv)))
      assert not actual[:, ~owned].any(), (row.si, kv)


@pytest.mark.parametrize("case", PER_SEQ_CASES)
def test_per_seq_packed_write_blend_covers_every_row(case):
  # Note (david): host replay of fwd_body's packed-direct write: blocks write
  # their whole staged window in si order after blending rows [staged start,
  # q_start) from the previous block's stage, and every packed row must end up
  # with its owning block's value. The blend fixes the write, not the mask, so
  # it does not depend on the window.
  cu_q, cu_k, q_pad, kv_pad, bq, bkv = case
  _, rows = replay_per_seq_fwd(cu_q, cu_k, q_pad, kv_pad, bq, bkv, (None, 0))
  owner = np.full(q_pad, -1)
  for row in rows:
    owner[slice(*row.owned_rows)] = row.si
  assert (owner >= 0).all()

  out = np.full(q_pad, -1)
  prev_stage, prev_start = np.full(bq, -1), 0
  for row in rows:
    q_start, q_end = int(row.ctx.q_start), int(row.ctx.q_end)
    packed = row.staged_q_start + np.arange(bq)
    # Note (david): a block computes every own-sequence row of its window;
    # rows of other sequences are garbage until blended.
    stage = np.where((packed >= q_start) & (packed < q_end),
                     owner[packed % q_pad], -1)
    blend_width = max(0, q_start - row.staged_q_start)
    src_offset = row.staged_q_start - prev_start
    # Note (david): the in-kernel copy moves whole 8-row tiles and must stay
    # inside the previous stage.
    assert blend_width == 0 or (
        src_offset >= 0 and src_offset % 8 == 0
        and src_offset + -(-blend_width // 8) * 8 <= bq), (
            row.si, src_offset, blend_width)
    stage[:blend_width] = prev_stage[src_offset:src_offset + blend_width]
    out[row.staged_q_start:row.staged_q_start + bq] = stage
    prev_stage, prev_start = stage, row.staged_q_start
  np.testing.assert_array_equal(out, owner)


BWD_CASES = [
    # (num_heads, num_q_blocks, num_kv_blocks, bq, bkv, causal, offset)
    (2, 8, 8, 128, 128, True, 0),
    (2, 8, 8, 128, 128, False, 0),
    (2, 8, 4, 128, 256, True, 512),
    (2, 4, 8, 256, 128, True, 0),
]


@pytest.mark.parametrize("case", BWD_CASES)
def test_dense_bwd_schedule_matches_the_mask(case):
  # Note (david): kv-outer swaps the roles: each kv block keeps the q blocks
  # that reach it, and its first/last flags gate the dk/dv init and flush.
  num_heads, nq, nkv, bq, bkv, causal, offset = case
  left, right = (None, 0) if causal else (None, None)
  nonempty, _ = block_truth(
      static_mask(nq * bq, nkv * bkv, left, right, offset), bq, bkv)
  expected = [
      (head, int(qi), kv, pos == 0, pos == qis.size - 1)
      for head in range(num_heads)
      for kv in range(nkv)
      for qis in [np.nonzero(nonempty[:, kv])[0]]
      for pos, qi in enumerate(qis)
  ]
  sched = make_dense_bwd_schedule(dense_bwd_params(
      num_heads=num_heads, num_q_blocks=nq, num_kv_blocks=nkv, bq=bq, bkv=bkv,
      causal=causal, offset=offset))
  visits = []
  for head in range(sched.num_head_groups):
    for kv in range(sched.num_kv_blocks):
      qlo, qhi, _ = sched.row_interval(np.int32(kv))
      qlo, qhi = int(qlo), int(qhi)
      visits.extend((head, qi, kv, qi == qlo, qi == qhi)
                    for qi in range(qlo, qhi + 1))
  assert visits == expected
