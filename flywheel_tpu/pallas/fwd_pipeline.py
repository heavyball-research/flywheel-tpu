"""Forward fragment pipeline shared by the batch and varlen flash kernels."""

import math
from collections.abc import Callable, Sequence
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .block_sizes import (
  BF16_TILE_ROWS,
  DEFAULT_MASK_VALUE,
  NN_DIM_NUMBERS,
  NT_DIM_NUMBERS,
  NUM_LANES,
  NUM_SUBLANES,
  BlockSizes,
  PagedKVInfo,
  QKVLayout,
  TokenMajorInfo,
  from_head_minor,
  round_up,
  vmem_limit_bytes,
)
from .copy_utils import (
  advance,
  bf16_half,
  drain,
  fold_row,
  hbm_window,
  load_staged_words,
  staged_words,
)
from .loop_schedule import FwdSchedule
from .mask import apply_packed_bounds_mask
from .rotary import ROTARY_DIM_MULTIPLE, rotate_tile

SEM_Q = 0
SEM_K = 1
SEM_V = 2
SEM_OUT = 3
SEM_LSE = 4
NUM_SEMS = 5
SEM_ROTARY_Q = NUM_SEMS
SEM_ROTARY_K = NUM_SEMS + 1

# Note (david): the guard keeps score <= e^T, row_sum <= kv_seq_len * e^T and
# |o| <= row_sum * max|v| under the f32 max (e^88.7), assuming |v| <= 2^12 and
# keeping a 2-nat margin. Below a 20-nat threshold the static anchor cannot be
# guarded safely.
F32_MAX_LOG = 88.7
V_ABS_MAX_LOG2 = 12
GUARD_MARGIN_NATS = 2.0
MIN_GUARD_THRESHOLD_NATS = 20.0


def overflow_guard_threshold(kv_seq_len: int) -> float:
  """Static-anchor overflow-guard threshold in log2 units."""
  threshold = (
      F32_MAX_LOG - math.log(kv_seq_len) - V_ABS_MAX_LOG2 * math.log(2.0)
      - GUARD_MARGIN_NATS
  )
  if threshold < MIN_GUARD_THRESHOLD_NATS:
    raise ValueError(
        f"overflow-guard threshold {threshold:.1f} nats is below the 20-nat"
        f" floor for kv_seq_len={kv_seq_len}; the static anchor cannot be"
        " guarded safely at this sequence length."
    )
  # Note (david): the kernels compare anchors in log2 units, while the floor
  # above is checked in nats.
  return threshold / math.log(2.0)


def fwd_body(
    schedule: FwdSchedule,
    q_hbm: jax.Array,
    k_hbm: jax.Array,
    v_hbm: jax.Array,
    refs: Sequence[jax.Array],
    *,
    bq: int,
    bkv: int,
    bkv_compute: int,
    bq_compute: int,
    num_stages: int,
    head_dim_v: int,
    qkv_layout: QKVLayout,
    guard_threshold: float,
    head_fold: int,
    transposed_pv: bool,
    window: tuple[int | None, int | None],
    causal_offset: int,
    softcap: float,
    return_lse: bool,
    token_major: TokenMajorInfo | None,
    q_scale: float,
    rotary_dim: int = 0,
    rotary_interleaved: bool = True,
    rotary_batch: int = 1,
    rotary_q_heads: int = 1,
    rotary_k_heads: int = 1,
    rotary_groups: int = 1,
    rotary_k: bool = True,
    paged: PagedKVInfo | None = None,
) -> None:
  """Run the fragment pipeline over every block the schedule yields.

  refs are forward_common's rotary operands, outputs and scratch, in its
  allocation order. The logits run in log2 units: Q reaches the QK matmul
  pre-multiplied by log2(e). paged (per-seq only) reads K and V from the one
  merged interleaved cache operand (k_hbm is v_hbm). A per-seq schedule writes
  the packed padding rows past schedule.packed_q_end as out = 0, lse = -inf.
  """
  is_per_seq = schedule.q_token is not None
  is_paged = paged is not None

  remaining_refs = list(refs)
  if rotary_dim:
    rotary_q_hbm = remaining_refs.pop(0)
    rotary_k_hbm = remaining_refs.pop(0) if rotary_k else None
  else:
    rotary_q_hbm = rotary_k_hbm = None
  o_hbm = remaining_refs.pop(0)
  lse_hbm = remaining_refs.pop(0) if return_lse else None
  q_buf = remaining_refs.pop(0)
  if is_paged:
    # Note (david): a paged block stages its kv head's (K, V) row pair as a
    # (bkv, 1, 2, head_dim) block and unpacks it once into dense K and V
    # planes, so every q tile of the block reads plain tiles.
    kv_pair_buf, kv_planes = remaining_refs.pop(0), remaining_refs.pop(0)
    k_buf = v_buf = None
  else:
    k_buf, v_buf = remaining_refs.pop(0), remaining_refs.pop(0)
    kv_pair_buf = kv_planes = None
  bounds_buf = remaining_refs.pop(0) if is_per_seq else None
  if rotary_dim:
    rotary_q_buf = remaining_refs.pop(0)
    rotary_k_buf = remaining_refs.pop(0) if rotary_k else None
  else:
    rotary_q_buf = rotary_k_buf = None
  if return_lse:
    *pipeline_refs, lse_stage = remaining_refs
  else:
    pipeline_refs, lse_stage = remaining_refs, None
  # Note (david): the exact unpack pins this variable-arity tail against
  # forward_common's scratch list; a silent mismatch would shift every ref.
  (o_stage, row_max_scratch, row_sum_scratch, o_scratch, rescale_scratch,
   guard_smem, parity_smem, sems) = pipeline_refs
  if return_lse and is_per_seq:
    lse_stage[...] = jnp.zeros(lse_stage.shape, jnp.float32)

  is_head_dim_minor = qkv_layout == QKVLayout.HEAD_DIM_MINOR
  head_dim_v_repeats = pl.cdiv(head_dim_v, NUM_LANES)
  guard_detect_bound = math.exp2(guard_threshold)
  has_static_mask = window != (None, None)
  o_layout = QKVLayout.SEQ_MINOR if transposed_pv else QKVLayout.HEAD_DIM_MINOR
  num_head_groups = schedule.num_head_groups
  num_rows = schedule.num_rows
  q_heads_per_kv_head = schedule.q_heads_per_kv_head
  heads_outer_batch = schedule.heads_outer_batch
  bkv_repeats = bkv_compute // NUM_LANES
  num_k_compute_tiles = bkv // bkv_compute
  num_q_compute_tiles = bq // bq_compute
  num_compute_tiles = num_k_compute_tiles * num_q_compute_tiles
  # Note (david): fragment skipping assumes the diagonal sits at qi == kv with
  # bq == bkv (zero-offset square causal, or no mask). Any other window masks
  # every fragment of a partial block, which costs compute, never correctness.
  # A runtime causal_offset is a traced scalar, so the isinstance keeps it off.
  can_skip_fragments = (
      window[0] is None and window[1] in (0, None)
      and isinstance(causal_offset, int) and causal_offset == 0 and bq == bkv
  )
  # Note (david): a paged q block's frontier is a runtime scalar, so its masked
  # blocks skip each kv compute tile past the frontier of the q tile's last row.
  # Unmasked blocks lie below every row's frontier and skip nothing.
  skips_past_frontier = schedule.kv_frontier is not None

  if token_major is None:
    width_qk = width_v = q_heads_per_row = kv_heads_per_row = None
  else:
    width_qk, width_v = token_major.head_dim_qk, token_major.head_dim_v
    q_heads_per_row = token_major.num_q_heads
    kv_heads_per_row = token_major.num_kv_heads
  is_lane_fold = token_major is not None and head_fold > 1
  lane_width_qk = width_qk if is_lane_fold else None
  lane_width_v = width_v if is_lane_fold else None
  # Note (david): a sub-tile lane width (d64) breaks two Mosaic rules: a
  # slot-sliced scratch ref only takes lane windows at whole-tile offset and
  # size, and the head's half of the loaded tile must then be a static value
  # slice. So scratch is sliced per 128-lane pair and g must be a Python int.
  is_lane_paired = is_lane_fold and lane_width_qk % NUM_LANES != 0
  heads_per_lane_tile = NUM_LANES // lane_width_qk if is_lane_paired else 1

  def _fold_lane(ref, g, block, width, layout=qkv_layout):
    if is_lane_paired:
      tile_start = (g // heads_per_lane_tile) * NUM_LANES
      return ref.at[:, pl.ds(tile_start, NUM_LANES)], g % heads_per_lane_tile
    else:
      return fold_row(ref, g, block, head_fold, layout, width), None

  def _take_half(value, half, width):
    if half is None:
      return value
    else:
      return value[..., half * width:(half + 1) * width]

  def _set_half(value, half, width, new_half):
    # Note (david): jnp .at[].set lowers to scatter, which Pallas TPU lacks, so
    # replace the half with a static split + concatenate.
    halves = [value[..., i * width:(i + 1) * width]
              for i in range(value.shape[-1] // width)]
    halves[half] = new_half
    return jnp.concatenate(halves, axis=-1)

  def _q_copy(head, tok, slot):
    return pltpu.make_async_copy(
        hbm_window(q_hbm, head, tok, bq, width=width_qk,
                   heads_per_row=q_heads_per_row, layout=qkv_layout,
                   token_major=token_major, fold=head_fold),
        q_buf.at[slot], sems.at[SEM_Q, slot])

  def _rotary_copies(coeff_hbm, coeff_buf, head, tok, block, slot,
                     heads_per_batch, sem_plane):
    for g in range(rotary_groups):
      batch_idx = 0 if rotary_batch == 1 else (head + g) // heads_per_batch
      rows = pl.ds(tok, block)
      coeff_window = (coeff_hbm.at[:, batch_idx, rows, :] if is_head_dim_minor
                      else coeff_hbm.at[:, batch_idx, :, rows])
      yield pltpu.make_async_copy(
          coeff_window, coeff_buf.at[slot, g], sems.at[sem_plane, slot])

  def _rotate_staged(buf, coeff_buf, slot, block):
    for g in range(head_fold):
      head_ref, half = _fold_lane(buf.at[slot], g, block, lane_width_qk)
      coeff_group = 0 if rotary_groups == 1 else g
      cos = coeff_buf[slot, coeff_group, 0]
      sin = coeff_buf[slot, coeff_group, 1]
      staged = head_ref[...]
      head_value = _take_half(staged, half, lane_width_qk)
      rotated = rotate_tile(head_value, cos, sin,
                            interleaved=rotary_interleaved,
                            head_dim_minor=is_head_dim_minor)
      head_ref[...] = (rotated if half is None
                       else _set_half(staged, half, lane_width_qk, rotated))

  def _start_q_row(head, tok, slot):
    _q_copy(head, tok, slot).start()
    if rotary_dim:
      for dma in _rotary_copies(rotary_q_hbm, rotary_q_buf, head, tok, bq,
                                slot, rotary_q_heads, SEM_ROTARY_Q):
        dma.start()

  def _wait_q_row(slot):
    _q_copy(0, 0, slot).wait()
    if rotary_dim:
      for dma in _rotary_copies(rotary_q_hbm, rotary_q_buf, 0, 0, bq, slot,
                                rotary_q_heads, SEM_ROTARY_Q):
        dma.wait()
      _rotate_staged(q_buf, rotary_q_buf, slot, bq)
    if q_scale != 1.0:
      # Note (david): each staged Q tile is waited on exactly once per row, so
      # scaling it here replaces an HBM pre-scaled Q copy with the same f32
      # multiply and cast, leaving the logits unchanged.
      q_ref = q_buf.at[slot]
      q_ref[...] = (
          q_ref[...].astype(jnp.float32) * q_scale).astype(q_buf.dtype)

  def _kv_copies(kv_head, tok, slot):
    return [
        pltpu.make_async_copy(
            hbm_window(k_hbm, kv_head, tok, bkv, width=width_qk,
                       heads_per_row=kv_heads_per_row, layout=qkv_layout,
                       token_major=token_major, fold=head_fold),
            k_buf.at[slot], sems.at[SEM_K, slot]),
        pltpu.make_async_copy(
            hbm_window(v_hbm, kv_head, tok, bkv, width=width_v,
                       heads_per_row=kv_heads_per_row, layout=qkv_layout,
                       token_major=token_major, fold=head_fold),
            v_buf.at[slot], sems.at[SEM_V, slot]),
    ]

  def _start_kv(kv_head, tok, slot):
    for dma in _kv_copies(kv_head, tok, slot):
      dma.start()
    if rotary_dim and rotary_k:
      for dma in _rotary_copies(rotary_k_hbm, rotary_k_buf, kv_head, tok, bkv,
                                slot, rotary_k_heads, SEM_ROTARY_K):
        dma.start()

  def _wait_kv(slot):
    for dma in _kv_copies(0, 0, slot):
      dma.wait()
    if rotary_dim and rotary_k:
      for dma in _rotary_copies(rotary_k_hbm, rotary_k_buf, 0, 0, bkv, slot,
                                rotary_k_heads, SEM_ROTARY_K):
        dma.wait()
      _rotate_staged(k_buf, rotary_k_buf, slot, bkv)

  def _o_copy(head, tok, slot):
    return pltpu.make_async_copy(
        o_stage.at[slot],
        hbm_window(o_hbm, head, tok, bq, width=width_v,
                   heads_per_row=q_heads_per_row, layout=o_layout,
                   token_major=token_major, fold=head_fold),
        sems.at[SEM_OUT, slot])

  # Note (david): a paged group is one q head, which reads its kv head's
  # (K, V) row pair; the pair of one page is the 2-row window [2 * kv_head,
  # 2 * kv_head + 2) of the interleaved cache's head axis, one DMA.
  assert not is_paged or head_fold == 1, head_fold

  def _paged_buffer_tile(slot, token_start, num_tokens):
    return kv_pair_buf.at[slot, pl.ds(token_start, num_tokens), 0, :, :]

  def _paged_kv_source(physical_page, num_tokens, kv_head):
    head_row = pl.multiple_of(2 * kv_head, 2)
    return k_hbm.at[physical_page, pl.ds(0, num_tokens), pl.ds(head_row, 2), :]

  def _paged_load_rows(kv_end, kv):
    # Note (david): each page DMA is clipped to the 8-aligned tokens the
    # sequence owns in that page, so a block table entry past the sequence's
    # pages is never dereferenced, and kv_end <= pages_per_seq * page_size
    # keeps every dereferenced entry inside the sequence's own table row.
    return jnp.clip(
        (kv_end + NUM_SUBLANES - 1) // NUM_SUBLANES * NUM_SUBLANES - kv * bkv,
        0, bkv)

  def _paged_kv_key(seq_idx, kv_head, kv_end, kv):
    # Note (david): a block's staged content is one kv head of its sequence's
    # kv block, cut at the q block's frontier kv_end, so the key carries the
    # head and the loaded extent; consecutive q blocks of one sequence reuse a
    # staged block only when they load the same rows of the same head.
    # flash_attn_varlen_paged bounds the key below 2**31.
    return (((seq_idx * paged.num_kv_heads + kv_head) * schedule.num_kv_blocks
             + kv) * (bkv // NUM_SUBLANES + 1)
            + _paged_load_rows(kv_end, kv) // NUM_SUBLANES)

  def _paged_page_tokens(kv_end, kv):
    load_size = _paged_load_rows(kv_end, kv)
    return [
        pl.multiple_of(
            jnp.clip(load_size - page * paged.page_size, 0, paged.page_size),
            NUM_SUBLANES)
        for page in range(bkv // paged.page_size)
    ]

  def _start_paged_kv(seq_idx, kv_head, kv_end, kv, slot):
    for page, num_tokens in enumerate(_paged_page_tokens(kv_end, kv)):
      @pl.when(num_tokens > 0)
      def _start_page(page=page, num_tokens=num_tokens):
        physical_page = schedule.kv_page(seq_idx, kv, page)
        pltpu.make_async_copy(
            _paged_kv_source(physical_page, num_tokens, kv_head),
            _paged_buffer_tile(slot, page * paged.page_size, num_tokens),
            sems.at[SEM_K, slot]).start()

  def _wait_paged_kv(ctx, kv, slot):
    # Note (david): every page of a block signals the one semaphore, and a
    # DMA wait only needs the destination size, so each page is waited with
    # its own staged window as both ends.
    for page, num_tokens in enumerate(_paged_page_tokens(ctx.kv_end, kv)):
      @pl.when(num_tokens > 0)
      def _wait_page(page=page, num_tokens=num_tokens):
        staged = _paged_buffer_tile(slot, page * paged.page_size, num_tokens)
        pltpu.make_async_copy(staged, staged, sems.at[SEM_K, slot]).wait()

    # Note (david): staged rows from valid_rows on hold cache slots past
    # seqused_k (uninitialized, possibly NaN) or stale VMEM past the load.
    # Their keys are masked, but a zero probability times a NaN value is still
    # NaN in the pv matmul, so a partial block zeroes those (K, V) rows once,
    # right after its load lands. Only the rest of the last compute tile below
    # the frontier (tile 0 at least) is ever read, since fwd_body skips the
    # tiles past it.
    valid_rows = jnp.clip(
        jnp.minimum(ctx.kv_used_end - kv * bkv,
                    _paged_load_rows(ctx.kv_end, kv)),
        0, bkv)
    read_rows = jnp.minimum(
        bkv,
        jnp.maximum(
            bkv_compute,
            (valid_rows + bkv_compute - 1) // bkv_compute * bkv_compute))

    @pl.when(valid_rows < read_rows)
    def _zero_value_tail():
      token_offsets = jnp.arange(NUM_SUBLANES, dtype=jnp.int32)[
          :, None, None, None]

      def _zero_tile(tile, carry):
        tile_start = pl.multiple_of(tile * NUM_SUBLANES, NUM_SUBLANES)
        tile_index = (slot, pl.ds(tile_start, NUM_SUBLANES))
        staged_tile = kv_pair_buf[tile_index]
        kv_pair_buf[tile_index] = jnp.where(
            tile_start + token_offsets < valid_rows, staged_tile,
            jnp.zeros_like(staged_tile))
        return carry

      lax.fori_loop(
          valid_rows // NUM_SUBLANES, read_rows // NUM_SUBLANES, _zero_tile,
          None)

  def _unpack_pair_planes(slot, num_live_tiles):
    # Note (david): a staged (K, V) row pair is one u32 word per lane, K in
    # its low half and V in its high half; each live compute tile is split
    # into the K and V planes once, not once per q tile.
    for tile in range(num_k_compute_tiles):
      @pl.when(tile < num_live_tiles)
      def _unpack_tile(tile=tile):
        tile_rows = pl.ds(tile * bkv_compute, bkv_compute)
        if paged.is_bitcast_load:
          words = load_staged_words(
              staged_words(kv_pair_buf, slot), tile * bkv_compute,
              bkv_compute, 1)
          for part in range(2):
            kv_planes[part, tile_rows, :] = bf16_half(words, part)
        else:
          for part in range(2):
            kv_planes[part, tile_rows, :] = kv_pair_buf[
                slot, tile_rows, 0, part, :]

  def _block_step(group, si, kv, lo, hi, ctx, next_row_entry, next_row_paged,
                  carry):
    (q_loaded, q_inflight, q_slot, kv_loaded, kv_inflight, kv_slot,
     prev_out_tok, num_out_copies) = carry

    if is_per_seq:
      q_tok = schedule.q_token(ctx)
      kv_tok = schedule.kv_token(kv, ctx)
      # Note (david): kv_token clamps kv + 1 past the row end, and the is_last
      # select below discards that value.
      next_block_kv_tok = schedule.kv_token(kv + 1, ctx)
    else:
      q_tok = (schedule.first_row + si) * bq
      kv_tok = kv * bkv
      next_block_kv_tok = (kv + 1) * bkv
    base_head = group * head_fold
    # Note (david): the heads-outer fold flattens (heads, batch) with
    # B = heads_outer_batch, so q index n * B + b reads kv index
    # (n // q_heads_per_kv_head) * B + b; the batch-outer fold flattens
    # (batch, heads), so consecutive q indices share one kv index.
    if heads_outer_batch is None:
      kv_head = base_head // q_heads_per_kv_head
    else:
      kv_head = (
          (base_head // heads_outer_batch // q_heads_per_kv_head)
          * heads_outer_batch + base_head % heads_outer_batch
      )
    row_key = group * num_rows + si
    if is_paged:
      kv_key = _paged_kv_key(ctx.seq_idx, kv_head, ctx.kv_end, kv)
    else:
      # Note (david): stream keys must be unique per DMA content, which on the
      # per-seq schedule only the token offset is, not the kv block index.
      kv_axis_len = schedule.num_kv_blocks * bkv
      kv_key = kv_head * kv_axis_len + kv_tok
    is_first = kv == lo
    is_last = kv == hi
    is_partial, needs_bounds = schedule.block_flags(si, kv, ctx)
    out_slot = num_out_copies % num_stages
    # Note (david): at the row end the lookahead is the next row's first block;
    # at the very last block that entry equals the current one, so no key
    # changes and no prefetch fires.
    (next_row_head, next_row_kv_head, next_row_q_tok, next_row_kv_tok,
     next_row_key) = next_row_entry
    lookahead_head = jnp.where(is_last, next_row_head, base_head)
    lookahead_q_tok = jnp.where(is_last, next_row_q_tok, q_tok)
    lookahead_kv_head = jnp.where(is_last, next_row_kv_head, kv_head)
    lookahead_kv_tok = jnp.where(is_last, next_row_kv_tok, next_block_kv_tok)
    lookahead_row_key = jnp.where(is_last, next_row_key, row_key)
    if is_paged:
      next_row_kv, next_row_seq, next_row_kv_end = next_row_paged
      lookahead_kv = jnp.where(is_last, next_row_kv, kv + 1)
      lookahead_seq = jnp.where(is_last, next_row_seq, ctx.seq_idx)
      lookahead_kv_end = jnp.where(is_last, next_row_kv_end, ctx.kv_end)
      lookahead_kv_key = _paged_kv_key(
          lookahead_seq, lookahead_kv_head, lookahead_kv_end, lookahead_kv)
    else:
      lookahead_kv_key = lookahead_kv_head * kv_axis_len + lookahead_kv_tok

    q_loaded, q_inflight, q_slot = advance(
        row_key,
        lookahead_row_key,
        lambda slot: _start_q_row(base_head, q_tok, slot),
        lambda slot: _start_q_row(lookahead_head, lookahead_q_tok, slot),
        _wait_q_row,
        q_loaded, q_inflight, q_slot, num_stages,
    )
    if is_paged:
      kv_loaded, kv_inflight, kv_slot = advance(
          kv_key,
          lookahead_kv_key,
          lambda slot: _start_paged_kv(
              ctx.seq_idx, kv_head, ctx.kv_end, kv, slot),
          lambda slot: _start_paged_kv(
              lookahead_seq, lookahead_kv_head, lookahead_kv_end,
              lookahead_kv, slot),
          lambda slot: _wait_paged_kv(ctx, kv, slot),
          kv_loaded, kv_inflight, kv_slot, num_stages,
      )
      # Note (david): the block's compute tiles below the frontier, and tile 0
      # always, since every q tile's first fragment runs.
      num_live_tiles = jnp.maximum(
          1, pl.cdiv(jnp.clip(ctx.kv_end - kv_tok, 0, bkv), bkv_compute))
      _unpack_pair_planes(kv_slot, num_live_tiles)
    else:
      kv_loaded, kv_inflight, kv_slot = advance(
          kv_key,
          lookahead_kv_key,
          lambda slot: _start_kv(kv_head, kv_tok, slot),
          lambda slot: _start_kv(lookahead_kv_head, lookahead_kv_tok, slot),
          _wait_kv,
          kv_loaded, kv_inflight, kv_slot, num_stages,
      )

    # Note (david): o_stage is one output ring shared by the whole fold group,
    # so wait out the slot's previous DMA before any head writes into it.
    @pl.when(is_last)
    def _wait_prev_out():
      @pl.when(num_out_copies >= num_stages)
      def _():
        _o_copy(0, 0, out_slot).wait()

    # Note (david): row_sum and o scratch ping-pong, so a block's first kv tile
    # reads the previous block's exit state while later tiles update in place.
    # Each head keeps its own parity and guard flag, so one head's overflow
    # never replays another.
    for g in range(head_fold):
      parity_smem[g] = jnp.where(is_first, 0, 1 - parity_smem[g])
      guard_smem[g] = 0

    def _head_parities(g):
      new_parity = parity_smem[g]
      return new_parity, 1 - new_parity

    def _is_kv_tile_live(kv_compute_index, q_compute_index):
      # Note (david): a kv tile at or past the frontier of a q tile's last row
      # is masked for every row of it, so it is skipped. Tile 0 always runs: it
      # carries a head's row state from the old parity to the new one, which
      # the later tiles of the block then read.
      q_last = q_tok + (q_compute_index + 1) * bq_compute - 1
      return jnp.logical_or(
          kv_compute_index == 0,
          kv_tok + kv_compute_index * bkv_compute
          < schedule.kv_frontier(ctx, q_last))

    def _paged_fragment(kv_compute_index, kv_part):
      # Note (david): the block's K (kv_part 0) or V (kv_part 1) plane tile.
      tile_start = pl.multiple_of(kv_compute_index * bkv_compute, NUM_LANES)
      return kv_planes[kv_part, pl.ds(tile_start, bkv_compute), :]

    def _qk(g, kv_compute_index, q_compute_index):
      q_ref, q_half = _fold_lane(q_buf.at[q_slot], g, bq, lane_width_qk)
      slice_q = pl.ds(q_compute_index * bq_compute, bq_compute)
      if is_paged:
        q = q_ref[slice_q, :]
        k = _paged_fragment(kv_compute_index, 0)
        dim_numbers = NT_DIM_NUMBERS
      else:
        k_ref, k_half = _fold_lane(k_buf.at[kv_slot], g, bkv, lane_width_qk)
        slice_k = pl.ds(kv_compute_index * bkv_compute, bkv_compute)
        if is_head_dim_minor:
          q = _take_half(q_ref[slice_q, :], q_half, lane_width_qk)
          k = _take_half(k_ref[slice_k, :], k_half, lane_width_qk)
          dim_numbers = NT_DIM_NUMBERS
        else:
          q = q_ref[:, slice_q].T
          k = k_ref[:, slice_k]
          dim_numbers = NN_DIM_NUMBERS
      raw_logits = lax.dot_general(
          q, k, dim_numbers, preferred_element_type=jnp.float32)
      assert raw_logits.shape == (bq_compute, bkv_compute)
      if softcap:
        # Note (david): the softcap is applied before masking.
        return jnp.tanh(raw_logits) * softcap
      else:
        return raw_logits

    def _online_softmax(
        g, kv_compute_index, q_compute_index, logits, needs_mask,
        guard_recovery, apply_bounds,
    ):
      head_rows = pl.ds(g * bq, bq)
      new_parity, old_parity = _head_parities(g)
      row_sum_new_ref = row_sum_scratch.at[new_parity, head_rows]
      row_max_scratch_g = row_max_scratch.at[head_rows]
      rescale_scratch_g = rescale_scratch.at[head_rows]

      slice_q = pl.ds(q_compute_index * bq_compute, bq_compute)
      # Note (david): per-seq bounds live in unshifted q coordinates, while the
      # static window lives in coordinates shifted by causal_offset.
      q_unshifted = q_tok + q_compute_index * bq_compute
      q_abs = q_unshifted + causal_offset
      k_abs = kv_tok + kv_compute_index * bkv_compute

      def _apply_window_mask(scores):
        left, right = window
        q_ids = q_abs + lax.broadcasted_iota(jnp.int32, scores.shape, 0)
        k_ids = k_abs + lax.broadcasted_iota(jnp.int32, scores.shape, 1)
        if left is None:
          left_masked = scores
        else:
          left_masked = jnp.where(
              q_ids - left <= k_ids, scores, DEFAULT_MASK_VALUE)
        if right is None:
          return left_masked
        else:
          return jnp.where(
              k_ids <= q_ids + right, left_masked, DEFAULT_MASK_VALUE)

      window_masked = lax.cond(
          needs_mask, _apply_window_mask, lambda scores: scores, logits)

      if apply_bounds:
        # Note (david): apply_bounds is static so interior blocks never trace
        # these VPU ops per score tile. The per-seq bounds rows cover the whole
        # block, so they sit in slot 0; the span row exists only under a left
        # window, so every other build keeps a single comparison.
        q_ids = q_unshifted + lax.broadcasted_iota(
            jnp.int32, window_masked.shape, 0)
        slice_k = pl.ds(kv_compute_index * bkv_compute, bkv_compute)
        staged_bounds = bounds_buf.at[0]
        if schedule.has_q_span:
          scores = apply_packed_bounds_mask(
              window_masked,
              DEFAULT_MASK_VALUE,
              q_ids,
              staged_bounds[0:1, slice_k],
              staged_bounds[1:2, slice_k],
          )
        else:
          scores = jnp.where(
              q_ids >= staged_bounds[0:1, slice_k], window_masked,
              DEFAULT_MASK_VALUE,
          )
      else:
        scores = window_masked

      read_parity = jnp.where(kv_compute_index == 0, old_parity, new_parity)
      row_sum_read_ref = row_sum_scratch.at[read_parity, head_rows]

      state_shape = row_max_scratch_g[slice_q, :].shape
      frag_row_max = lax.broadcast_in_dim(
          scores.max(axis=-1), state_shape, (0,))
      is_row_start = jnp.logical_and(is_first, kv_compute_index == 0)
      row_max_prev = row_max_scratch_g[slice_q, :]
      if guard_recovery:
        # Note (david): the recovery replay rebases the anchor onto the
        # overflowing fragment; the ordinary pass keeps the anchor it first
        # picked.
        should_rebase = frag_row_max - row_max_prev > guard_threshold
        row_max = jnp.where(
            is_row_start,
            frag_row_max,
            jnp.where(
                should_rebase,
                jnp.maximum(row_max_prev, frag_row_max),
                row_max_prev,
            ),
        )
      else:
        row_max = jnp.where(is_row_start, frag_row_max, row_max_prev)
      # Note (david): logits are in log2 units, so exp2 drops the per-element
      # multiply jnp.exp lowers to on TPU.
      probs = jnp.exp2(scores - jnp.tile(row_max, (1, bkv_repeats)))
      frag_row_sum = lax.broadcast_in_dim(
          probs.sum(axis=-1), state_shape, (0,))
      if guard_recovery:
        rescale = jnp.exp2(row_max_prev - row_max)
        row_sum_prev = rescale * row_sum_read_ref[slice_q, :]
      else:
        row_sum_prev = row_sum_read_ref[slice_q, :]
      row_sum_new = jnp.where(
          is_row_start, frag_row_sum, row_sum_prev + frag_row_sum)
      row_sum_new_ref[slice_q, :] = row_sum_new
      row_max_scratch_g[slice_q, :] = row_max
      if guard_recovery:
        rescale_scratch_g[slice_q, :] = jnp.where(is_row_start, 1.0, rescale)
      else:
        # Note (david): row_sum upper-bounds every single score of the row, so
        # this detector may request a needless replay but never misses a score
        # above the guard bound. Check once the row has consumed the block, at
        # its last live tile when tiles past the frontier are skipped.
        is_last_kv_tile = kv_compute_index == num_k_compute_tiles - 1
        if skips_past_frontier and apply_bounds:
          is_last_kv_tile = jnp.logical_or(
              is_last_kv_tile,
              jnp.logical_not(
                  _is_kv_tile_live(kv_compute_index + 1, q_compute_index)))
        has_overflow_risk = jnp.any(
            jnp.logical_not(row_sum_new <= guard_detect_bound))
        guard_smem[g] = jnp.maximum(
            guard_smem[g],
            jnp.where(is_last_kv_tile, has_overflow_risk.astype(jnp.int32), 0),
        )
      return probs, is_row_start

    def _pv(
        g, kv_compute_index, q_compute_index, probs, is_row_start,
        guard_recovery,
    ):
      head_rows = pl.ds(g * bq, bq)
      new_parity, old_parity = _head_parities(g)
      rescale_scratch_g = rescale_scratch.at[head_rows]

      if is_paged:
        o_curr = lax.dot_general(
            probs, _paged_fragment(kv_compute_index, 1), NN_DIM_NUMBERS,
            preferred_element_type=jnp.float32)
      else:
        v_ref, v_half = _fold_lane(v_buf.at[kv_slot], g, bkv, lane_width_v)
        slice_k = pl.ds(kv_compute_index * bkv_compute, bkv_compute)
        if is_head_dim_minor:
          o_curr = lax.dot_general(
              probs, _take_half(v_ref[slice_k, :], v_half, lane_width_v),
              NN_DIM_NUMBERS, preferred_element_type=jnp.float32)
        elif transposed_pv:
          # Note (david): contracting the SEQ_MINOR (head_dim, bkv_compute) v
          # slab against probs yields o^T and fills the MXU's K and N dims with
          # bkv_compute and bq_compute instead of capping N at head_dim.
          o_curr = lax.dot_general(
              v_ref[:, slice_k], probs, NT_DIM_NUMBERS,
              preferred_element_type=jnp.float32)
        else:
          o_curr = lax.dot_general(
              probs, v_ref[:, slice_k], NT_DIM_NUMBERS,
              preferred_element_type=jnp.float32)

      slice_q = pl.ds(q_compute_index * bq_compute, bq_compute)
      read_parity = jnp.where(kv_compute_index == 0, old_parity, new_parity)
      if transposed_pv:
        # Note (david): q is the lane axis of the transposed o scratch, and
        # Mosaic only slices lanes dynamically at an offset that is static per
        # unrolled copy, as q_compute_index is.
        o_new_ref = o_scratch.at[new_parity, :, head_rows]
        o_read_ref = o_scratch.at[read_parity, :, head_rows]
        tile_index = (slice(None), slice_q)
      else:
        o_new_ref = o_scratch.at[new_parity, head_rows]
        o_read_ref = o_scratch.at[read_parity, head_rows]
        tile_index = (slice_q, slice(None))
      o_read = o_read_ref[tile_index]
      if not guard_recovery:
        o_prev = o_read
      elif transposed_pv:
        # Note (david): every lane of the rescale scratch holds the row value,
        # so lane 0 transposed to (1, bq_compute) broadcasts over the head_dim_v
        # sublanes without any lane tiling.
        o_prev = rescale_scratch_g[slice_q, 0:1].T * o_read
      else:
        o_prev = jnp.tile(
            rescale_scratch_g[slice_q, :], (1, head_dim_v_repeats)
        )[..., :o_scratch.shape[-1]] * o_read
      o_new_ref[tile_index] = jnp.where(is_row_start, o_curr, o_prev + o_curr)

    def _run_schedule(is_causal, guard_recovery, static_g, apply_bounds):
      # Note (david): static_g None is the fast path, one distance-1 pipeline
      # across every folded head so head g + 1's first QK overlaps head g's
      # last PV. An int static_g is the rare per-head overflow replay, kept
      # simple on purpose.
      if static_g is None:
        num_fragments = head_fold * num_compute_tiles

        def _fragment_coords(frag_idx):
          g = frag_idx // num_compute_tiles
          tile_idx = frag_idx % num_compute_tiles
          return g, tile_idx // num_q_compute_tiles, tile_idx % num_q_compute_tiles
      else:
        num_fragments = num_compute_tiles

        def _fragment_coords(frag_idx):
          return (static_g, frag_idx // num_q_compute_tiles,
                  frag_idx % num_q_compute_tiles)

      # Note (david): a pinned g replaces the loop-carried copy as a Python int,
      # because sub-tile (d64) lane windows need static offsets; a static g
      # costs nothing on the other builds.
      def _resolve_g(g):
        return static_g if static_g is not None else g

      first_g, first_k, first_q = (
          jnp.int32(coord) for coord in _fragment_coords(0))
      logits = _qk(_resolve_g(first_g), first_k, first_q)
      probs_first, is_row_start_first = _online_softmax(
          _resolve_g(first_g), first_k, first_q, logits, needs_mask=is_causal,
          guard_recovery=guard_recovery, apply_bounds=apply_bounds,
      )

      def _pipeline_step(frag_idx, carry):
        probs_prev, is_row_start_prev, prev_g, prev_k, prev_q = carry
        current_g, current_k, current_q = _fragment_coords(frag_idx)
        if skips_past_frontier and apply_bounds:
          is_active = _is_kv_tile_live(current_k, current_q)
        elif not can_skip_fragments:
          is_active = True
        else:
          is_below_diagonal = (
              (current_q + 1) * bq_compute > current_k * bkv_compute)
          is_active = jnp.logical_or(not is_causal, is_below_diagonal)

        def _process_active(_):
          if can_skip_fragments:
            needs_mask = jnp.logical_and(
                is_causal,
                current_q * bq_compute < (current_k + 1) * bkv_compute,
            )
          else:
            needs_mask = is_causal
          logits = _qk(_resolve_g(current_g), current_k, current_q)
          probs_curr, is_row_start_curr = _online_softmax(
              _resolve_g(current_g), current_k, current_q, logits, needs_mask,
              guard_recovery=guard_recovery, apply_bounds=apply_bounds,
          )
          _pv(
              _resolve_g(prev_g), prev_k, prev_q, probs_prev,
              is_row_start_prev, guard_recovery=guard_recovery,
          )
          return (probs_curr, is_row_start_curr, current_g, current_k,
                  current_q)

        return lax.cond(is_active, _process_active, lambda _: carry, operand=None)

      probs_last, is_row_start_last, last_g, last_k, last_q = lax.fori_loop(
          1,
          num_fragments,
          _pipeline_step,
          (probs_first, is_row_start_first, first_g, first_k, first_q),
          unroll=True,
      )
      _pv(
          _resolve_g(last_g), last_k, last_q, probs_last, is_row_start_last,
          guard_recovery=guard_recovery,
      )

    def _run_group(is_causal, apply_bounds):
      if is_lane_paired:
        # Note (david): one static-g pipeline per head, since the fused
        # pipeline's loop-carried g would make the 64-lane window offsets
        # dynamic; this gives up the overlap across head boundaries.
        for g in range(head_fold):
          _run_schedule(
              is_causal, guard_recovery=False, static_g=g,
              apply_bounds=apply_bounds,
          )
      else:
        _run_schedule(
            is_causal, guard_recovery=False, static_g=None,
            apply_bounds=apply_bounds,
        )

      for g in range(head_fold):
        @pl.when(guard_smem[g] > 0)
        def _recover(g=g):
          _run_schedule(
              is_causal, guard_recovery=True, static_g=g,
              apply_bounds=apply_bounds,
          )

    # Note (david): two static copies of the fragment pipeline, one mask-free
    # and one applying every mask the build has, because the masks cost about
    # as many VPU ops as the softmax itself. One copy per (causal, bounds) pair
    # measured 2.6x slower at block 2048. A build carries either the static
    # window (dense) or the per-seq bounds (varlen), never both, and
    # has_static_mask still gates the former so a full-attention build never
    # loses its upper triangle.
    if is_per_seq:
      needs_any_mask = jnp.logical_or(is_partial, needs_bounds)
    else:
      needs_any_mask = is_partial
    for is_masked in (False, True):
      is_selected = (
          needs_any_mask if is_masked else jnp.logical_not(needs_any_mask))

      @pl.when(is_selected)
      def _run_block(is_masked=is_masked):
        if is_masked and is_per_seq:
          # Note (david): synthesize the bounds rows once for the whole (1, bkv)
          # kv window; evaluated per fragment they spilled vector registers the
          # MXU then waited on (+13.4% on a 2048/256 prefill).
          staged_bounds = bounds_buf.at[0]
          k_ids = kv_tok + lax.broadcasted_iota(jnp.int32, (1, bkv), 1)
          q_lower, q_span = schedule.q_bounds(ctx, kv, k_ids)
          staged_bounds[0:1, :] = q_lower
          if schedule.has_q_span:
            staged_bounds[1:2, :] = q_span
        _run_group(
            is_causal=is_masked and has_static_mask,
            apply_bounds=is_masked and is_per_seq,
        )

    @pl.when(is_last)
    def _finalize():
      def _finalize_head(g):
        head_rows = pl.ds(g * bq, bq)
        new_parity, _ = _head_parities(g)
        row_sum_new_ref = row_sum_scratch.at[new_parity, head_rows]
        # Note (david): o_stage keeps an explicit fold axis (leading on
        # head-major, lanes on token-major) while o_scratch stays flat-row
        # addressed, so only o_stage goes through _fold_lane.
        o_new_ref = (
            o_scratch.at[new_parity, :, head_rows] if transposed_pv
            else o_scratch.at[new_parity, head_rows]
        )
        o_stage_g, o_half = _fold_lane(
            o_stage.at[out_slot], g, bq, lane_width_v, layout=o_layout
        )

        def _finalize_fragment(q_compute_index, carry):
          slice_q = pl.ds(q_compute_index * bq_compute, bq_compute)
          row_sum = row_sum_new_ref[slice_q, :]
          if transposed_pv:
            tile_index = (slice(None), slice_q)
            inverse_row_sum = (1.0 / row_sum[:, 0:1]).T
          else:
            tile_index = (slice_q, slice(None))
            inverse_row_sum = jnp.tile(
                1.0 / row_sum, (1, head_dim_v_repeats)
            )[..., :o_scratch.shape[-1]]
          o_normalized = (
              o_new_ref[tile_index] * inverse_row_sum).astype(o_stage.dtype)
          if o_half is None:
            o_stage_g[tile_index] = o_normalized
          else:
            # Note (david): the stage tile holds both heads of the pair, and the
            # head loop is sequential, so the other half is either untouched or
            # already this step's final value.
            o_stage_g[tile_index] = _set_half(
                o_stage_g[tile_index], o_half, lane_width_v, o_normalized)
          return carry

        lax.fori_loop(
            0,
            num_q_compute_tiles,
            _finalize_fragment,
            None,
            unroll=True,
        )

        if return_lse:
          # Note (david): the static anchor keeps sum_j exp2(logit_j) ==
          # exp2(row_max) * row_sum, so lse is row_max + log2(row_sum) in log2
          # units; scale by ln(2) for the natural-log lse that is returned.
          row_max = row_max_scratch.at[head_rows][:, 0:1]
          lse_row = (
              (row_max + jnp.log2(row_sum_new_ref[:, 0:1])) * math.log(2.0))
          if is_per_seq:
            lse_stage[out_slot, :, g] = lse_row[:, 0]
          else:
            lse_stage[g, :] = lse_row[:, 0]

      for g in range(head_fold):
        _finalize_head(g)

    if is_per_seq:
      # Note (david): a per-seq window sits on the 8-row grid and may reach
      # below blend_start into rows the previous output block computed
      # correctly. That stage is still resident (num_stages >= 2 and only its
      # outbound DMA reads it), so the rows are copied over from it. The bf16
      # stage is accessed in whole 16-row tiles, but the two windows are only
      # 8-aligned apart, so each own tile reads the two source tiles it
      # straddles and picks their 8-row halves in f32, where 8 rows are one
      # whole (8, 128) tile.
      num_blend_rows = schedule.blend_start(ctx) - q_tok

      @pl.when(jnp.logical_and(is_last, num_blend_rows > 0))
      def _blend_head_rows():
        prev_slot = (num_out_copies - 1) % num_stages
        src_row_offset = q_tok - prev_out_tok
        src_half = src_row_offset & (BF16_TILE_ROWS - 1)
        half_rows = BF16_TILE_ROWS // 2

        def _tile_index(stage_ref, slot, row_start):
          return (slot,) + (slice(None),) * (len(stage_ref.shape) - 3) + (
              pl.ds(pl.multiple_of(row_start, BF16_TILE_ROWS), BF16_TILE_ROWS),
              slice(None))

        def _blend_stage(stage_ref, tile):
          own_index = _tile_index(stage_ref, out_slot, tile * BF16_TILE_ROWS)
          own_rows = stage_ref[own_index].astype(jnp.float32)
          lo_start = src_row_offset - src_half + tile * BF16_TILE_ROWS
          # Note (david): the upper source tile is used only when src_half is 8
          # and this tile's upper half blends, and then it lies inside the
          # source stage; the clamp keeps the unused read in bounds otherwise.
          hi_start = jnp.minimum(lo_start + BF16_TILE_ROWS,
                                 stage_ref.shape[-2] - BF16_TILE_ROWS)
          lo = stage_ref[_tile_index(stage_ref, prev_slot, lo_start)].astype(
              jnp.float32)
          hi = stage_ref[_tile_index(stage_ref, prev_slot, hi_start)].astype(
              jnp.float32)
          half_ids = lax.broadcasted_iota(
              jnp.int32, lo[..., :half_rows, :].shape, lo.ndim - 2)
          is_src_on_tile = half_ids + src_half < half_rows
          prev_rows = jnp.concatenate([
              jnp.where(is_src_on_tile, lo[..., :half_rows, :],
                        lo[..., half_rows:, :]),
              jnp.where(is_src_on_tile, lo[..., half_rows:, :],
                        hi[..., :half_rows, :]),
          ], axis=-2)
          row_ids = tile * BF16_TILE_ROWS + lax.broadcasted_iota(
              jnp.int32, own_rows.shape, own_rows.ndim - 2)
          stage_ref[own_index] = jnp.where(
              row_ids < num_blend_rows, prev_rows, own_rows).astype(
                  stage_ref.dtype)

        def _blend_tile(tile, carry):
          _blend_stage(o_stage, tile)
          if return_lse:
            _blend_stage(lse_stage, tile)
          return carry

        lax.fori_loop(
            0, (num_blend_rows + BF16_TILE_ROWS - 1) // BF16_TILE_ROWS,
            _blend_tile, None)

    if is_per_seq:
      packed_q_end = schedule.packed_q_end

      # Note (david): rows at or past cu_seqlens_q[-1] belong to no sequence
      # and read out = 0 (lse = -inf). A window reaching them zeroes them in
      # its stage, which covers the partial tile at the packed end; the fill
      # after the loop covers the whole tiles past it.
      @pl.when(jnp.logical_and(is_last, q_tok + bq > packed_q_end))
      def _zero_packed_padding():
        o_slot = o_stage.at[out_slot]
        staged_out = o_slot[...]
        out_rows = q_tok + lax.broadcasted_iota(
            jnp.int32, staged_out.shape, staged_out.ndim - 2)
        o_slot[...] = jnp.where(
            out_rows < packed_q_end, staged_out, jnp.zeros_like(staged_out))
        if return_lse:
          lse_slot = lse_stage.at[out_slot]
          staged_lse = lse_slot[...]
          lse_rows = q_tok + lax.broadcasted_iota(
              jnp.int32, staged_lse.shape, 0)
          lse_slot[...] = jnp.where(
              lse_rows < packed_q_end, staged_lse, -jnp.inf)

    @pl.when(is_last)
    def _start_out():
      # Note (david): the output and LSE copies start only after the boundary
      # blend has repaired the rows shared with the previous block.
      _o_copy(base_head, q_tok, out_slot).start()
      if return_lse:
        if is_per_seq:
          # Note (david): the head group is a leading untiled axis and its
          # 128-lane minor axis is copied whole, avoiding the rejected dynamic
          # minor-axis slice at the head offset.
          lse_copies = [(lse_stage.at[out_slot],
                         lse_hbm.at[base_head // head_fold, pl.ds(q_tok, bq), :])]
        elif head_fold % NUM_SUBLANES == 0:
          # Note (david): HBM's leading f32 tile is 8 rows, so an 8-aligned fold
          # is one group-wide copy; a smaller fold copies row by row, since
          # later groups start at h = 2 or 4 and would break tile alignment.
          lse_copies = [(lse_stage,
                         lse_hbm.at[pl.ds(base_head, head_fold),
                                    pl.ds(q_tok, bq)])]
        else:
          lse_copies = [
              (lse_stage.at[pl.ds(local_head, 1), :],
               lse_hbm.at[pl.ds(base_head + local_head, 1), pl.ds(q_tok, bq)])
              for local_head in range(head_fold)
          ]
        for lse_src, lse_dst in lse_copies:
          lse_copy = pltpu.make_async_copy(
              lse_src, lse_dst, sems.at[SEM_LSE, 0])
          lse_copy.start()
          lse_copy.wait()

    if is_per_seq:
      next_prev_out_tok = jnp.where(is_last, q_tok, prev_out_tok)
    else:
      next_prev_out_tok = prev_out_tok

    return (q_loaded, q_inflight, q_slot, kv_loaded, kv_inflight, kv_slot,
            next_prev_out_tok, num_out_copies + is_last.astype(jnp.int32))

  def _row_body(g, si, carry):
    lo, hi, ctx = schedule.row_interval(si)
    # Note (david): past the very last row of the last group the lookahead is
    # the current row itself, so the keys stay equal and nothing is
    # prefetched. There next_si == si, so next_ctx == ctx and hi stays valid.
    is_row_end = si == num_rows - 1
    is_grid_end = jnp.logical_and(g == num_head_groups - 1, is_row_end)
    next_g = jnp.where(is_row_end, jnp.minimum(g + 1, num_head_groups - 1), g)
    next_si = jnp.where(is_grid_end, si, jnp.where(is_row_end, 0, si + 1))
    next_lo, _, next_ctx = schedule.row_interval(next_si)
    next_head = next_g * head_fold
    next_kv = jnp.where(is_grid_end, hi, next_lo)
    if is_per_seq:
      next_q_tok = schedule.q_token(next_ctx)
      next_kv_tok = schedule.kv_token(next_kv, next_ctx)
    else:
      next_q_tok = (schedule.first_row + next_si) * bq
      next_kv_tok = next_kv * bkv
    if heads_outer_batch is None:
      next_kv_head = next_head // q_heads_per_kv_head
    else:
      next_kv_head = (
          (next_head // heads_outer_batch // q_heads_per_kv_head)
          * heads_outer_batch + next_head % heads_outer_batch
      )
    next_row_entry = (
        next_head,
        next_kv_head,
        next_q_tok,
        next_kv_tok,
        next_g * num_rows + next_si,
    )
    if is_paged:
      next_row_paged = (next_kv, next_ctx.seq_idx, next_ctx.kv_end)
    else:
      next_row_paged = None
    return lax.fori_loop(
        lo, hi + 1,
        lambda kv, block_carry: _block_step(
            g, si, kv, lo, hi, ctx, next_row_entry, next_row_paged,
            block_carry),
        carry,
    )

  minus_one = jnp.int32(-1)
  zero = jnp.int32(0)
  init_carry = (minus_one, minus_one, zero, minus_one, minus_one, zero, zero,
                zero)
  *_, num_out_copies = lax.fori_loop(
      0, num_head_groups,
      lambda g, group_carry: lax.fori_loop(
          0, num_rows, lambda si, row_carry: _row_body(g, si, row_carry),
          group_carry),
      init_carry,
  )

  drain(num_out_copies, num_stages, lambda slot: _o_copy(0, 0, slot).wait())

  if is_per_seq:
    # Note (david): packed padding rows own no q block, so once every output
    # DMA has landed they are written from a cleared stage. The last block
    # already zeroed its window's padding rows, so the fill starts at the next
    # 8-row tile; with no block at all, cu_seqlens_q is all 0 and so is that
    # tile.
    packed_q_end = schedule.packed_q_end
    padded_total_q = o_hbm.shape[-2]
    fill_start = (
        (packed_q_end + NUM_SUBLANES - 1) // NUM_SUBLANES * NUM_SUBLANES)
    num_fills = pl.cdiv(jnp.maximum(padded_total_q - fill_start, 0), bq)

    @pl.when(num_fills > 0)
    def _clear_fill_stage():
      o_stage[0] = jnp.zeros(o_stage.shape[1:], o_stage.dtype)
      if return_lse:
        lse_stage[0] = jnp.full(lse_stage.shape[1:], -jnp.inf, jnp.float32)

    def _fill_rows(fill_index, carry):
      row_start = pl.multiple_of(fill_start + fill_index * bq, NUM_SUBLANES)
      num_fill_rows = pl.multiple_of(
          jnp.minimum(padded_total_q - row_start, bq), NUM_SUBLANES)
      fill_rows = pl.ds(0, num_fill_rows)
      # Note (david): only a head-major fold stages a leading fold axis; a
      # token-major fold rides the lanes.
      if token_major is None and head_fold > 1:
        o_fill = o_stage.at[0, :, fill_rows, :]
      else:
        o_fill = o_stage.at[0, fill_rows, :]
      for group in range(num_head_groups):
        o_fill_copy = (
            o_fill,
            hbm_window(o_hbm, group * head_fold, row_start, num_fill_rows,
                       width=width_v, heads_per_row=q_heads_per_row,
                       layout=o_layout, token_major=token_major,
                       fold=head_fold),
            SEM_OUT,
        )
        if return_lse:
          fill_copies = [o_fill_copy, (
              lse_stage.at[0, fill_rows, :],
              lse_hbm.at[group, pl.ds(row_start, num_fill_rows), :],
              SEM_LSE,
          )]
        else:
          fill_copies = [o_fill_copy]
        for fill_src, fill_dst, sem_index in fill_copies:
          fill_copy = pltpu.make_async_copy(
              fill_src, fill_dst, sems.at[sem_index, 0])
          fill_copy.start()
          fill_copy.wait()
      return carry

    lax.fori_loop(0, num_fills, _fill_rows, None)


def forward_common(
    kernel: Callable[..., None],
    smem_operands: list[jax.Array],
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    block_sizes: BlockSizes,
    num_kv_heads: int,
    head_fold: int,
    transposed_pv: bool,
    return_lse: bool,
    kernel_name: str,
    interpret: bool,
    token_major: TokenMajorInfo | None,
    is_per_seq: bool = False,
    rotary: tuple[jax.Array, jax.Array | None] | None = None,
    rotary_interleaved: bool = True,
    paged: PagedKVInfo | None = None,
    **body,
) -> jax.Array | tuple[jax.Array, jax.Array]:
  """Validate q/k/v, allocate scratch and run a fwd_body kernel.

  kernel is the mode kernel with its schedule already bound; every fwd_body
  static that follows from block_sizes or the q/k/v shapes is bound here.
  rotary is the (2, batch, tokens, head_dim / 2) Q and K coefficient pair
  (K None when K arrives rotated). Returns o shaped like q (head_dim_v wide),
  or (o, lse) with a (num_q_heads, q_seq_len) float32 natural-log lse.

  paged (per-seq, head-major q only) makes k / v the paged cache operands:
  k the merged cache and v None, or the K and V caches of a pair, each laid
  out as paged describes. The kernel then takes (q, k[, v], *rotary).
  """
  bq, bkv = block_sizes.block_q, block_sizes.block_kv
  bkv_compute = block_sizes.block_kv_compute
  bq_compute = block_sizes.block_q_compute
  num_stages = block_sizes.num_stages
  qkv_layout = block_sizes.qkv_layout

  if bq % bq_compute:
    raise ValueError(f"{bq=} must be a multiple of {bq_compute=}.")
  if bq_compute % NUM_SUBLANES:
    raise ValueError(
        f"{bq_compute=} must be a multiple of {NUM_SUBLANES}."
    )
  if is_per_seq and bq % BF16_TILE_ROWS:
    # Note (david): the per-seq head blend steps the output stage in whole
    # 16-row tiles.
    raise ValueError(
        f"per-seq {bq=} must be a multiple of {BF16_TILE_ROWS}.")

  if paged is None and k.shape[:-1] != v.shape[:-1]:
    raise ValueError(
        f"'key' {k.shape} and 'value' {v.shape} must have the same leading"
        " dimensions."
    )

  if paged is not None:
    if qkv_layout != QKVLayout.HEAD_DIM_MINOR:
      raise ValueError(
          f"a paged KV cache needs a HEAD_DIM_MINOR layout; got {qkv_layout=}."
      )
    num_q_heads, q_seq_len, head_dim_qk = q.shape
    head_dim_v = head_dim_qk
    # Note (david): seqused_k bounds every sequence's kv blocks, so the kv
    # axis only needs its block count; the padded capacity stands in for its
    # length.
    kv_seq_len = round_up(paged.pages_per_seq * paged.page_size, bkv)
  elif token_major is not None:
    batch = token_major.batch
    q_heads_per_row = token_major.num_q_heads
    kv_heads_per_row = token_major.num_kv_heads
    head_dim_qk, head_dim_v = token_major.head_dim_qk, token_major.head_dim_v
    expected_rank = 2 if batch is None else 3
    if q.ndim != expected_rank or k.ndim != expected_rank or v.ndim != expected_rank:
      raise ValueError(
          f"token-major with batch={batch} expects rank-{expected_rank}"
          f" q/k/v; got {q.shape=}, {k.shape=}, {v.shape=}."
      )
    if batch is not None and (q.shape[0] != batch or k.shape[0] != batch):
      raise ValueError(
          f"q/k lead with batch={batch}; got {q.shape=}, {k.shape=}."
      )
    if q.shape[-1] != q_heads_per_row * head_dim_qk:
      raise ValueError(
          f"q's fused width {q.shape[-1]} != num_heads {q_heads_per_row} *"
          f" head_dim {head_dim_qk}."
      )
    if k.shape[-1] != kv_heads_per_row * head_dim_qk:
      raise ValueError(
          f"k's fused width {k.shape[-1]} != num_kv_heads {kv_heads_per_row} *"
          f" head_dim {head_dim_qk}."
      )
    if v.shape[-1] != kv_heads_per_row * head_dim_v:
      raise ValueError(
          f"v's fused width {v.shape[-1]} != num_kv_heads {kv_heads_per_row} *"
          f" head_dim_v {head_dim_v}."
      )
    if head_dim_v % NUM_LANES and (head_fold * head_dim_v) % NUM_LANES:
      # Note (david): a sub-tile head_dim (d64) relies on an even head_fold so
      # every DMA group spans whole 128-lane tiles.
      raise ValueError(
          f"token-major requires head_dim_v % 128 == 0, or a head_fold that"
          f" pads the DMA group to a 128-lane multiple; got {head_dim_v=},"
          f" {head_fold=}."
      )
    q_seq_len = q.shape[-2]
    kv_seq_len = k.shape[-2]
    num_q_heads = (batch or 1) * q_heads_per_row
  else:
    num_q_heads, q_seq_len, head_dim_qk = q.shape
    head_dim_v = v.shape[-1]

    if k.ndim != 3:
      raise ValueError(
          f"Expected 3-dim 'key' tensor. Instead got a {k.ndim}-dim one."
      )
    if k.shape[2] != head_dim_qk:
      raise ValueError(
          f"Expected 'key' head dimension to be: {head_dim_qk}. Instead got:"
          f" {k.shape[2]}."
      )
    if k.shape[0] != num_kv_heads:
      raise ValueError(
          f"'key' has {k.shape[0]} heads but the schedule was built for"
          f" {num_kv_heads} kv heads."
      )

    kv_seq_len = k.shape[1]

  if head_fold > 1 and head_dim_v != head_dim_qk:
    # Note (david): v_buf, o_stage and the f32 o_scratch all scale with
    # head_dim_v, which no tuned fold was measured against; a fold sized for
    # head_dim_qk can miss the scoped VMEM budget by tens of MB.
    raise ValueError(
        f"head_fold > 1 requires head_dim_v == head_dim_qk; got {head_dim_v}"
        f" vs {head_dim_qk}."
    )
  if bkv % bkv_compute:
    raise ValueError(f"{bkv=} must be a multiple of {bkv_compute=}.")
  if bkv_compute % NUM_LANES:
    raise ValueError(f"{bkv_compute=} must be a multiple of {NUM_LANES}.")

  if q_seq_len % bq or kv_seq_len % bkv:
    raise ValueError(f"{bq=} and {bkv=} must divide the sequence lengths.")
  if rotary is None:
    rotary_dim, rotary_batch, rotary_groups = 0, 1, 1
    rotary_q_heads, rotary_k_heads = num_q_heads, num_kv_heads
    rotary_k, rotary_dtype = True, q.dtype
    rotary_operands = []
  else:
    if type(rotary_interleaved) is not bool:
      raise ValueError("rotary_interleaved must be a static bool.")
    if len(rotary) != 2:
      raise ValueError("rotary must contain Q and K coefficient arrays.")
    rotary_q_coeff, rotary_k_coeff = rotary
    rotary_k = rotary_k_coeff is not None
    if (rotary_q_coeff.ndim != 4 or rotary_q_coeff.shape[0] != 2
        or rotary_q_coeff.shape[1] == 0
        or rotary_q_coeff.shape[2] != q_seq_len
        or (rotary_k and (rotary_k_coeff.ndim != 4
                          or rotary_q_coeff.shape[:2] != rotary_k_coeff.shape[:2]
                          or rotary_k_coeff.shape[2] != kv_seq_len
                          or rotary_q_coeff.shape[3] != rotary_k_coeff.shape[3]))):
      raise ValueError("rotary coefficients must be (2, batch, tokens, head_dim/2).")
    rotary_dim = 2 * rotary_q_coeff.shape[3]
    rotary_batch = rotary_q_coeff.shape[1]
    # Note (david): float32 coefficients keep the in-kernel f32 rotation
    # identical to an f32 RoPE computed outside the kernel.
    rotary_dtype = rotary_q_coeff.dtype
    # Note (david): rotate_tile rotates every dim of the head, so only full
    # rotary is supported.
    if (rotary_dim <= 0 or rotary_dim % ROTARY_DIM_MULTIPLE
        or rotary_dim != head_dim_qk
        or rotary_dtype not in (q.dtype, jnp.dtype(jnp.float32))
        or (rotary_k and rotary_k_coeff.dtype != rotary_dtype)):
      raise ValueError("Invalid rotary dimension or coefficient dtype.")
    if num_q_heads % rotary_batch or num_kv_heads % rotary_batch:
      raise ValueError("rotary batch must divide the flattened head counts.")
    if token_major is not None and rotary_batch not in (1, token_major.batch):
      raise ValueError("rotary batch must match token-major batch or be shared.")
    rotary_q_heads = num_q_heads // rotary_batch
    rotary_k_heads = num_kv_heads // rotary_batch
    # Note (david): a dense head-major fold may span batch rows, and only then
    # does each folded head need its own coefficient plane.
    if rotary_q_heads % head_fold or rotary_k_heads % head_fold:
      rotary_groups = head_fold
    else:
      rotary_groups = 1
    rotary_operands = [coeff for coeff in rotary if coeff is not None]
  num_rotary_streams = (1 + int(rotary_k)) if rotary_dim else 0

  fwd_kernel = partial(
      kernel,
      bq=bq, bkv=bkv, bkv_compute=bkv_compute, bq_compute=bq_compute,
      num_stages=num_stages, head_dim_v=head_dim_v, qkv_layout=qkv_layout,
      head_fold=head_fold, transposed_pv=transposed_pv, return_lse=return_lse,
      token_major=token_major, rotary_dim=rotary_dim,
      rotary_interleaved=rotary_interleaved, rotary_batch=rotary_batch,
      rotary_q_heads=rotary_q_heads, rotary_k_heads=rotary_k_heads,
      rotary_groups=rotary_groups, rotary_k=rotary_k, paged=paged, **body,
  )

  folded_bq = head_fold * bq
  o_layout = QKVLayout.SEQ_MINOR if transposed_pv else QKVLayout.HEAD_DIM_MINOR
  vmem = pltpu.VMEM

  def _staged_shape(rows, head_dim, layout):
    if head_fold == 1:
      return (num_stages, *from_head_minor((rows, head_dim), layout))
    elif token_major is not None:
      # Note (david): a token-major group is head_fold * head_dim consecutive
      # lanes of one row, so the fold rides the lane axis and one DMA per
      # stream lands the whole group.
      return (num_stages, rows, head_fold * head_dim)
    else:
      # Note (david): an explicit fold axis rank-matches the 3-D HBM head
      # slices, so XLA never materializes a reshaped q/k/v copy, which
      # dominated wall time at short sequences when head_dim < 128 needed lane
      # padding.
      return (num_stages, head_fold, *from_head_minor((rows, head_dim), layout))

  q_buf = vmem(_staged_shape(bq, head_dim_qk, qkv_layout), q.dtype)
  if paged is None:
    k_buf = vmem(_staged_shape(bkv, head_dim_qk, qkv_layout), k.dtype)
    v_buf = vmem(_staged_shape(bkv, head_dim_v, qkv_layout), v.dtype)
    scratch_shapes = [q_buf, k_buf, v_buf]
  else:
    # Note (david): a paged block stages one kv head's (K, V) row pair, then
    # unpacks it into one K and one V plane; (2, head_dim) bf16 rows pad to
    # nothing, so the pair staging costs what two dense planes would.
    scratch_shapes = [
        q_buf, vmem((num_stages, bkv, 1, 2, head_dim_qk), k.dtype),
        vmem((2, bkv, head_dim_qk), k.dtype)]
  if is_per_seq:
    # Note (david): the per-seq kernel synthesizes its bounds rows (the q lower
    # bound, plus the span under a left window) once per block in-kernel, so
    # it needs one slot, not a kv-stream ring. Both rows share one (8, 128)
    # int32 tile, so the span row costs no extra VMEM.
    scratch_shapes.append(vmem((1, 2, bkv), jnp.int32))
  if rotary_dim:
    scratch_shapes += [
        vmem((num_stages, rotary_groups, 2,
              *from_head_minor((block, rotary_dim // 2), qkv_layout)),
             rotary_dtype)
        for block in ((bq, bkv) if rotary_k else (bq,))
    ]
  o_stage = vmem(_staged_shape(bq, head_dim_v, o_layout), q.dtype)
  row_max_scratch = vmem((folded_bq, NUM_LANES), jnp.float32)
  row_sum_scratch = vmem((2, folded_bq, NUM_LANES), jnp.float32)
  o_scratch = vmem(
      (2, *from_head_minor((folded_bq, head_dim_v), o_layout)), jnp.float32)
  rescale_scratch = vmem((folded_bq, NUM_LANES), jnp.float32)
  guard_smem = pltpu.SMEM((head_fold,), jnp.int32)
  parity_smem = pltpu.SMEM((head_fold,), jnp.int32)
  sems = pltpu.SemaphoreType.DMA((NUM_SEMS + num_rotary_streams, num_stages))
  scratch_shapes += [o_stage, row_max_scratch, row_sum_scratch, o_scratch,
                     rescale_scratch, guard_smem, parity_smem, sems]
  if return_lse and is_per_seq:
    scratch_shapes.append(vmem((num_stages, bq, NUM_LANES), jnp.float32))
  elif return_lse:
    scratch_shapes.append(vmem((head_fold, bq), jnp.float32))

  if qkv_layout == QKVLayout.SEQ_MINOR:
    q_in, k_in, v_in, *rotary_in = (
        operand.swapaxes(-1, -2) for operand in (q, k, v, *rotary_operands))
  else:
    q_in, k_in, v_in, *rotary_in = (q, k, v, *rotary_operands)
  if paged is not None:
    kernel_inputs = [q_in, k_in, *rotary_in]
  else:
    kernel_inputs = [q_in, k_in, v_in, *rotary_in]
  smem_spec = pl.BlockSpec(memory_space=pltpu.SMEM)
  any_spec = pl.BlockSpec(memory_space=pl.ANY)
  in_specs = (
      [smem_spec] * len(smem_operands) + [any_spec] * len(kernel_inputs))

  if token_major is None:
    o_shape = (
        num_q_heads, *from_head_minor((q_seq_len, head_dim_v), o_layout))
  else:
    o_shape = (
        *q.shape[:-2], q_seq_len, token_major.num_q_heads * head_dim_v)
  # Note (david): per-seq LSE gives every head group an explicit minor axis of
  # whole 128-lane tiles, so fwd_body DMAs complete tiles at offset zero and
  # groups never alias in the physical layout; only the first head_fold lanes
  # are returned.
  lse_shape = (
      (num_q_heads // head_fold, q_seq_len, NUM_LANES)
      if is_per_seq else (num_q_heads, q_seq_len)
  )
  if return_lse:
    out_specs = [any_spec, any_spec]
    out_shape = [
        jax.ShapeDtypeStruct(o_shape, q.dtype),
        jax.ShapeDtypeStruct(lse_shape, jnp.float32),
    ]
  else:
    out_specs = any_spec
    out_shape = jax.ShapeDtypeStruct(o_shape, q.dtype)

  with jax.named_scope(kernel_name):
    outputs = pl.pallas_call(
        fwd_kernel,
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shape,
        scratch_shapes=scratch_shapes,
        name=kernel_name,
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_limit_bytes()),
        interpret=pltpu.InterpretParams() if interpret else False,
    )(*smem_operands, *kernel_inputs)
  if return_lse:
    o_kernel, lse_kernel = outputs
  else:
    o_kernel = outputs
  # Note (david): transposed_pv emits (heads, head_dim_v, seq); for d64 the swap
  # back is the {1,2,0} layout XLA already prefers, so it folds into a bitcast.
  o = o_kernel.swapaxes(-1, -2) if transposed_pv else o_kernel
  if not return_lse:
    return o
  elif is_per_seq:
    return o, lse_kernel[:, :, :head_fold].swapaxes(1, 2).reshape(
        num_q_heads, q_seq_len)
  else:
    return o, lse_kernel
