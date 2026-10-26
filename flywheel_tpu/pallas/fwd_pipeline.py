"""Forward fragment pipeline shared by the batch and varlen flash kernels."""

import math
from collections.abc import Callable, Sequence
from functools import partial, reduce
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .block_sizes import (
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
  hbm_bf16_tile_rows,
  round_up,
  vmem_limit_bytes,
)
from .copy_utils import (
  advance,
  drain,
  fold_row,
  hbm_window,
  kv_heads_per_word,
  kv_word_halves,
  kv_word_rows,
  load_staged_words,
  staged_words,
)
from .loop_schedule import FwdSchedule, window_fragment_plans
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
PENDING_WRITE_PARITY = 0
PENDING_READ_PARITY = 1
PENDING_ROW_START = 2
NUM_PENDING_FIELDS = 3
# Note (david): an empty pending PV reads and writes this trash o slot, so the
# next block issues the pending PV without a branch.
TRASH_PARITY = 2
NUM_O_SLOTS = 3
# Note (david): the kv stream prefetches two blocks ahead, so its ring holds
# the running block, the next one and the one in flight.
NUM_KV_STAGES = 3
# Note (david): Mosaic tiles a bf16 VMEM scratch whose row count is a multiple
# of 16 as (16, 128) (v7x does; v6e may keep (8, 128)), and a dynamic vector
# load or store must start on that tile. Dynamic row accesses into bf16 stages
# therefore step in whole 16-row tiles, which is legal under either tiling;
# HBM DMAs stay on the hbm_bf16_tile_rows() grid.
BF16_TILE_ROWS = 2 * NUM_SUBLANES

# Note (david): the guard keeps score <= e^T, row_sum <= kv_seq_len * e^T and
# |o| <= row_sum * max|v| under the f32 max (e^88.7), assuming |v| <= 2^12 and
# keeping a 2-nat margin. Below a 20-nat threshold the static anchor cannot be
# guarded safely.
F32_MAX_LOG = 88.7
V_ABS_MAX_LOG2 = 12
GUARD_MARGIN_NATS = 2.0
MIN_GUARD_THRESHOLD_NATS = 20.0

# Note (david): a masked score is exactly DEFAULT_MASK_VALUE and real logits
# sit far inside half of it, so these bounds never mistake one for the other;
# exp2(score - UNANCHORED_ROW_MAX) is exactly 0 for every score.
MASKED_ROW_BOUND = 0.5 * DEFAULT_MASK_VALUE
UNANCHORED_ROW_MAX = -DEFAULT_MASK_VALUE
UNANCHORED_ROW_BOUND = -MASKED_ROW_BOUND


class RowFinish(NamedTuple):
  """A q block row's output finish, deferred until its last PV has run.

  prev_out_tok is the previous output block's q_tok, out_count the row's
  output copy index and parity that of its final row state. is_live 0 marks
  no row.
  """
  is_live: jax.Array
  si: jax.Array
  q_tok: jax.Array
  base_head: jax.Array
  num_blend_rows: jax.Array
  prev_out_tok: jax.Array
  out_count: jax.Array
  parity: jax.Array


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
    q_scale: float | jax.Array,
    rotary_dim: int = 0,
    rotary_interleaved: bool = False,
    rotary_batch: int = 1,
    rotary_q_heads: int = 1,
    rotary_k_heads: int = 1,
    rotary_groups: int = 1,
    rotary_k: bool = True,
    paged: PagedKVInfo | None = None,
    gqa_fold: int = 1,
    v_scale: float | jax.Array = 1.0,
) -> None:
  """Run the fragment pipeline over every block the schedule yields.

  refs are forward_common's rotary operands, outputs and scratch, in its
  allocation order. The logits run in log2 units: Q reaches the QK matmul
  pre-multiplied by log2(e). paged (per-seq only) reads K and V from the one
  merged interleaved cache operand (k_hbm is v_hbm). A per-seq schedule writes
  the packed padding rows past schedule.packed_q_end as out = 0, lse = -inf.

  gqa_fold > 1 (paged only) stages gqa_fold consecutive q heads of one kv head
  per group and stacks them over the rows of every score fragment: fragment
  row r holds head r // bq_compute at token r % bq_compute of the q tile, so
  bq and bq_compute count tokens per head. A paged fp8 cache is widened to
  bf16 K and V planes, and v_scale multiplies the output. q_scale and v_scale
  are static floats, or (paged) f32 scalars the kernel read from SMEM.
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
    # Note (david): a paged block stages its kv head's K and V rows as a
    # (bkv, 1, kv_word_rows, head_dim) block and unpacks it once into dense K
    # and V planes, so every q tile of the block reads plain tiles.
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
  (o_stage, row_max_scratch, row_sum_scratch, o_scratch, pending_probs,
   pending_v, guard_smem, parity_smem, pending_smem, sems) = pipeline_refs
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
  # Note (david): a group stages group_heads q heads. head_fold runs one
  # fragment pipeline per head, while gqa_fold stacks its heads over the rows
  # of one fragment, so each K and V tile the MXU holds serves frag_rows rows.
  # The row state of pipeline head g is its group_bq rows, ordered (q tile,
  # folded head, token).
  assert gqa_fold == 1 or (is_paged and head_fold == 1), (gqa_fold, head_fold)
  group_heads = head_fold * gqa_fold
  group_bq = gqa_fold * bq
  frag_rows = gqa_fold * bq_compute
  token_tile_shape = (bq_compute, bkv_compute)
  # Note (david): a partial block of a dense window (causal included) masks
  # every fragment of the generic copy, about as many VPU ops as the softmax.
  # Under bq % bkv == 0 and a static offset its fragment flags are static per
  # block offset, so each offset runs its own branch-free copy over the band
  # alone; runtime fragment flags measured 2x slower than masking everything,
  # since a branch per fragment breaks the QK/PV overlap. Any other build
  # masks every fragment of a partial block, which costs compute, never
  # correctness. A runtime causal_offset is a traced scalar, so the isinstance
  # keeps it off.
  if (has_static_mask and not is_per_seq
      and isinstance(causal_offset, int) and bq % bkv == 0):
    window_plans = window_fragment_plans(
        bq=bq, bkv=bkv, bq_compute=bq_compute, bkv_compute=bkv_compute,
        left=window[0], right=window[1], causal_offset=causal_offset)
  else:
    window_plans = None
  # Note (david): a per-seq q block's kv frontier is a runtime scalar, so a
  # masked block runs only its kv compute tiles below it, in a tile loop that
  # indexes the tile at runtime; only a sublane token axis allows that.
  skips_past_frontier = schedule.num_live_tiles is not None
  assert skips_past_frontier == is_per_seq, (skips_past_frontier, is_per_seq)
  assert not skips_past_frontier or (
      is_head_dim_minor and not transposed_pv), (qkv_layout, transposed_pv)
  # Note (david): only a left window can mask a row's whole first fragment
  # while later keys stay visible; anchoring on that masked score would push
  # the row's first visible key past the guard into a replay.
  defers_row_start = window[0] is not None or schedule.has_q_span
  # Note (david): a per-seq row is often one or two blocks, so its row-end
  # work is paid every block or two: on v6e (32:8 d128, 512 x 512, one-block
  # rows) each row cost 0.97 us on top of its 1.89 us block, 0.50 of it the
  # unoverlapped last PV. Dense rows are long and keep the row-end drain.
  defers_row_finish = is_per_seq

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
                   token_major=token_major, fold=group_heads),
        q_buf.at[slot], sems.at[SEM_Q, slot])

  def _rotary_copies(coeff_hbm, coeff_buf, head, tok, block, slot,
                     heads_per_batch, sem_plane):
    for g in range(rotary_groups):
      batch_idx = 0 if rotary_batch == 1 else (head + g) // heads_per_batch
      rows = pl.ds(tok, block)
      coeff_window = (coeff_hbm.at[batch_idx, rows, :] if is_head_dim_minor
                      else coeff_hbm.at[batch_idx, :, rows])
      yield pltpu.make_async_copy(
          coeff_window, coeff_buf.at[slot, g], sems.at[sem_plane, slot])

  def _rotate_staged(buf, coeff_buf, slot, block):
    for g in range(head_fold):
      head_ref, half = _fold_lane(buf.at[slot], g, block, lane_width_qk)
      coeff_group = 0 if rotary_groups == 1 else g
      staged = head_ref[...]
      head_value = _take_half(staged, half, lane_width_qk)
      rotated = rotate_tile(head_value, coeff_buf[slot, coeff_group],
                            interleaved=rotary_interleaved,
                            head_dim_minor=is_head_dim_minor)
      head_ref[...] = (rotated if half is None
                       else _set_half(staged, half, lane_width_qk, rotated))

  def _rotate_gqa_folded_q(slot):
    # Note (david): the gqa-folded heads share the paged build's one Q
    # coefficient plane, and a rolled loop keeps one copy of the rotation.
    def _rotate_head(h, carry):
      head_ref = q_buf.at[slot, h]
      head_ref[...] = rotate_tile(
          head_ref[...], rotary_q_buf[slot, 0],
          interleaved=rotary_interleaved, head_dim_minor=is_head_dim_minor)
      return carry

    lax.fori_loop(0, gqa_fold, _rotate_head, None)

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
      if gqa_fold > 1:
        _rotate_gqa_folded_q(slot)
      else:
        _rotate_staged(q_buf, rotary_q_buf, slot, bq)
    if isinstance(q_scale, jax.Array) or q_scale != 1.0:
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

  def _o_copy(head, tok, slot):
    return pltpu.make_async_copy(
        o_stage.at[slot],
        hbm_window(o_hbm, head, tok, bq, width=width_v,
                   heads_per_row=q_heads_per_row, layout=o_layout,
                   token_major=token_major, fold=group_heads),
        sems.at[SEM_OUT, slot])

  # Note (david): a paged group is the gqa_fold q heads of one kv head, which
  # read its (K, V) rows; those of one page are the u32 word window of the
  # interleaved cache's head axis that holds them, one DMA: a bf16 head's
  # [2 * kv_head, 2 * kv_head + 2), an fp8 head pair's 4 rows.
  assert not is_paged or head_fold == 1, head_fold
  if is_paged:
    word_rows = kv_word_rows(kv_pair_buf.dtype)
    heads_per_word = kv_heads_per_word(kv_pair_buf.dtype)
    num_cache_pages = k_hbm.shape[0]
  else:
    word_rows = heads_per_word = num_cache_pages = None

  def _paged_buffer_tile(slot, token_start, num_tokens):
    return kv_pair_buf.at[slot, pl.ds(token_start, num_tokens), 0, :, :]

  def _paged_kv_source(physical_page, num_tokens, kv_head):
    head_row = pl.multiple_of(kv_head // heads_per_word * word_rows, word_rows)
    return k_hbm.at[
        physical_page, pl.ds(0, num_tokens), pl.ds(head_row, word_rows), :]

  def _paged_page_rows(kv_load_end, kv):
    # Note (david): every page DMA of a block is issued, so its wait needs no
    # branch. A page past the load extent moves one sublane tile of a clamped
    # page, whose rows the unpack zeroes.
    num_load_rows = jnp.clip(kv_load_end - kv * bkv, 0, bkv)
    return [
        pl.multiple_of(
            jnp.clip(num_load_rows - page * paged.page_size, NUM_SUBLANES,
                     paged.page_size),
            NUM_SUBLANES)
        for page in range(bkv // paged.page_size)
    ]

  # Note (david): the kv stream prefetches two blocks ahead in a
  # NUM_KV_STAGES ring. Mosaic's scheduler sinks a DMA start towards its
  # wait, so a start and wait in one basic block leave the DMA no time to
  # land, and a wait inside a copy splits the copy into two basic blocks with
  # their own pipeline fill and drain (v6e, 512 x 512: 1574 + 1553 cycles
  # against 2330 for the whole copy). Blocks j and j + 1 hold at most two
  # slots, so the slot block j + 2 takes is free for its start anywhere in
  # block j's step.
  def _block_desc(head, kv, ctx):
    """The DMA inputs of kv block kv of head's group in the row of ctx."""
    # Note (david): the heads-outer fold flattens (heads, batch) with
    # B = heads_outer_batch, so q index n * B + b reads kv index
    # (n // q_heads_per_kv_head) * B + b; the batch-outer fold flattens
    # (batch, heads), so consecutive q indices share one kv index.
    if heads_outer_batch is None:
      kv_head = head // q_heads_per_kv_head
    else:
      kv_head = ((head // heads_outer_batch // q_heads_per_kv_head)
                 * heads_outer_batch + head % heads_outer_batch)
    if is_paged:
      desc = (ctx.page_base, kv_head, ctx.kv_load_end, ctx.kv_used_end, kv)
    elif is_per_seq:
      desc = (kv_head, schedule.kv_token(kv, ctx))
    else:
      desc = (kv_head, kv * bkv)
    # Note (david): the descriptors ride the loop carry, which needs strong
    # int32 scalars.
    return tuple(jnp.asarray(field, jnp.int32) for field in desc)

  def _select_desc(predicate, desc_true, desc_false):
    return tuple(jnp.where(predicate, field_true, field_false)
                 for field_true, field_false in zip(desc_true, desc_false))

  # Note (david): a block whose DMA inputs equal the previous block's holds
  # the same staged data, so it runs from the previous block's slot; dense GQA
  # at one kv block per row otherwise re-staged each block once per q head
  # (v6e, 32:4 d256, 1024 tokens: 63.4 against 60.6 us). Paged blocks start
  # their DMAs inside the copy and a rotary K is rotated in place, so both
  # load every block.
  reuses_blocks = not is_paged and not (rotary_dim and rotary_k)

  def _is_fresh(desc, prev_desc):
    if reuses_blocks:
      differs = [field != prev_field
                 for field, prev_field in zip(desc, prev_desc)]
      return reduce(jnp.logical_or, differs)
    else:
      return jnp.bool_(True)

  def _start_block_data(desc, slot):
    if is_paged:
      page_base, kv_head, kv_load_end, _, kv = desc
      for page, num_rows in enumerate(_paged_page_rows(kv_load_end, kv)):
        physical_page = jnp.clip(
            schedule.kv_page(page_base, kv, page), 0, num_cache_pages - 1)
        pltpu.make_async_copy(
            _paged_kv_source(physical_page, num_rows, kv_head),
            _paged_buffer_tile(slot, page * paged.page_size, num_rows),
            sems.at[SEM_K, slot]).start()
    else:
      kv_head, kv_tok = desc
      for dma in _kv_copies(kv_head, kv_tok, slot):
        dma.start()
      if rotary_dim and rotary_k:
        for dma in _rotary_copies(rotary_k_hbm, rotary_k_buf, kv_head, kv_tok,
                                  bkv, slot, rotary_k_heads, SEM_ROTARY_K):
          dma.start()

  def _wait_block_data(desc, slot):
    if is_paged:
      _, _, kv_load_end, _, kv = desc
      # Note (david): every page signals the one semaphore by its size, and a
      # DMA wait only needs a destination of the awaited size, so one wait on
      # a window as long as all the pages together covers the block.
      num_rows = pl.multiple_of(
          sum(_paged_page_rows(kv_load_end, kv)), NUM_SUBLANES)
      staged_window = _paged_buffer_tile(slot, 0, num_rows)
      pltpu.make_async_copy(
          staged_window, staged_window, sems.at[SEM_K, slot]).wait()
    else:
      for dma in _kv_copies(0, 0, slot):
        dma.wait()
      if rotary_dim and rotary_k:
        for dma in _rotary_copies(rotary_k_hbm, rotary_k_buf, 0, 0, bkv, slot,
                                  rotary_k_heads, SEM_ROTARY_K):
          dma.wait()

  def _paged_fragment(kv_compute_index, kv_part):
    """The (bkv_compute, head_dim) K (kv_part 0) or V (kv_part 1) plane tile
    of the unpacked paged block."""
    tile_start = pl.multiple_of(kv_compute_index * bkv_compute, NUM_LANES)
    return kv_planes[kv_part, pl.ds(tile_start, bkv_compute), :]

  def _v_tile(g, kv_compute_index, kv_slot):
    """A fragment's V operand as loaded, before a lane pair's half is
    picked."""
    if is_paged:
      return _paged_fragment(kv_compute_index, 1)
    else:
      v_ref, _ = _fold_lane(v_buf.at[kv_slot], g, bkv, lane_width_v)
      slice_k = pl.ds(kv_compute_index * bkv_compute, bkv_compute)
      return v_ref[slice_k, :] if is_head_dim_minor else v_ref[:, slice_k]

  def _pv(g, q_compute_index, probs, v_tile, is_row_start, rescale,
          write_parity, read_parity):
    """v_tile is _v_tile's operand, and rescale the fragment's recovery
    rescale (None on the ordinary pass). The PV of a fragment reads o from
    read_parity and writes it to write_parity."""
    head_rows = pl.ds(g * group_bq, group_bq)
    v_half = g % heads_per_lane_tile if is_lane_paired else None

    if is_paged:
      o_curr = lax.dot_general(
          probs, v_tile, NN_DIM_NUMBERS, preferred_element_type=jnp.float32)
    elif is_head_dim_minor:
      o_curr = lax.dot_general(
          probs, _take_half(v_tile, v_half, lane_width_v),
          NN_DIM_NUMBERS, preferred_element_type=jnp.float32)
    elif transposed_pv:
      # Note (david): contracting the SEQ_MINOR (head_dim, bkv_compute) v
      # slab against probs yields o^T and fills the MXU's K and N dims with
      # bkv_compute and bq_compute instead of capping N at head_dim.
      o_curr = lax.dot_general(
          v_tile, probs, NT_DIM_NUMBERS, preferred_element_type=jnp.float32)
    else:
      o_curr = lax.dot_general(
          probs, v_tile, NT_DIM_NUMBERS, preferred_element_type=jnp.float32)

    slice_q = pl.ds(q_compute_index * frag_rows, frag_rows)
    if transposed_pv:
      # Note (david): q is the lane axis of the transposed o scratch, and
      # Mosaic only slices lanes dynamically at an offset that is static per
      # unrolled copy, as q_compute_index is.
      o_new_ref = o_scratch.at[write_parity, :, head_rows]
      o_read_ref = o_scratch.at[read_parity, :, head_rows]
      tile_index = (slice(None), slice_q)
    else:
      o_new_ref = o_scratch.at[write_parity, head_rows]
      o_read_ref = o_scratch.at[read_parity, head_rows]
      tile_index = (slice_q, slice(None))
    o_read = o_read_ref[tile_index]
    if rescale is None:
      o_prev = o_read
    elif transposed_pv:
      # Note (david): every lane of the rescale holds the row value, so lane
      # 0 transposed to (1, bq_compute) broadcasts over the head_dim_v
      # sublanes without any lane tiling.
      o_prev = rescale[:, 0:1].T * o_read
    else:
      o_prev = jnp.tile(
          rescale, (1, head_dim_v_repeats)
      )[..., :o_scratch.shape[-1]] * o_read
    o_new_ref[tile_index] = jnp.where(is_row_start, o_curr, o_prev + o_curr)

  # Note (david): the mask-free copy leaves its last fragment's PV pending in
  # scratch (a pl.when copy returns no values), and the next block's copy
  # issues it after its own first QK and softmax, so the MXU never drains
  # between blocks. Two row-state parities still suffice: block k's pending PV
  # writes block k's new parity before block k + 1 reads it, and block k's
  # replay runs inside block k's copy; a replay of the last head redoes the
  # pending fragment, so it cancels the pending PV. Under defers_row_finish
  # every copy carries its last PV out and a row's output finish runs after
  # the next row's first block, so rows alternate parity to keep the finished
  # row's state intact until then.
  pending_g, pending_k, pending_q = (
      head_fold - 1, num_k_compute_tiles - 1, num_q_compute_tiles - 1)

  def _issue_pending_pv():
    _pv(pending_g, pending_q, pending_probs[...], pending_v[...],
        pending_smem[PENDING_ROW_START] > 0, None,
        write_parity=pending_smem[PENDING_WRITE_PARITY],
        read_parity=pending_smem[PENDING_READ_PARITY])

  def _cancel_pending_pv():
    pending_smem[PENDING_WRITE_PARITY] = TRASH_PARITY
    pending_smem[PENDING_READ_PARITY] = TRASH_PARITY

  def _block_step(group, si, kv, lo, hi, ctx, row, ahead, carry):
    """One kv block of row si. ahead holds the descriptors of the next row's
    first two blocks in the flattened block order, and carry this block's and
    the next block's ring slot, freshness and descriptor."""
    q_tok, base_head, q_slot, row_parity, prev_row_finish = row
    (kv_slot, next_slot, is_fresh, is_next_fresh, block_desc,
     next_block_desc) = carry
    if is_per_seq:
      kv_tok = schedule.kv_token(kv, ctx)
    else:
      kv_tok = kv * bkv
    is_first = kv == lo
    is_partial, needs_bounds = schedule.block_flags(si, kv, ctx)
    if skips_past_frontier:
      num_live_tiles = schedule.num_live_tiles(kv, ctx)
    else:
      num_live_tiles = None
    # Note (david): row_sum and o scratch ping-pong, so a block's first kv tile
    # reads the previous block's exit state while later tiles update in place;
    # each head keeps its own guard flag, so one head's overflow never replays
    # another. The parity is re-read from SMEM at every use: held as a loop
    # value it lengthened every copy's schedule by about 30% (v6e, 512 x 512
    # paged: 3537 against 2872 bundles).
    parity_smem[0] = lax.rem(row_parity + kv - lo, 2)

    def _block_parities():
      new_parity = parity_smem[0]
      return new_parity, 1 - new_parity

    first_next_row_desc, second_ahead_desc = ahead
    ahead_desc = _select_desc(
        kv + 2 <= hi, _block_desc(base_head, kv + 2, ctx),
        _select_desc(kv + 1 == hi, first_next_row_desc, second_ahead_desc))
    is_ahead_fresh = _is_fresh(ahead_desc, next_block_desc).astype(jnp.int32)
    ahead_slot = jnp.where(
        is_ahead_fresh > 0, lax.rem(next_slot + 1, NUM_KV_STAGES), next_slot)
    for g in range(head_fold):
      guard_smem[g] = 0

    @pl.when(is_fresh > 0)
    def _wait_fresh_block():
      _wait_block_data(block_desc, kv_slot)

    if reuses_blocks:
      @pl.when(is_ahead_fresh > 0)
      def _start_fresh_ahead():
        _start_block_data(ahead_desc, ahead_slot)

    def _qk(g, kv_compute_index, q_compute_index):
      slice_q = pl.ds(q_compute_index * bq_compute, bq_compute)
      if is_paged:
        if gqa_fold > 1:
          # Note (david): the q tile's tokens of every folded head, stacked
          # head-major over the fragment rows; bq_compute is whole bf16 tiles,
          # so the reshape moves no data.
          q = q_buf[q_slot, :, slice_q, :].reshape(
              frag_rows, q_buf.shape[-1])
        else:
          q_ref, _ = _fold_lane(q_buf.at[q_slot], g, bq, lane_width_qk)
          q = q_ref[slice_q, :]
        k = _paged_fragment(kv_compute_index, 0)
        dim_numbers = NT_DIM_NUMBERS
      else:
        q_ref, q_half = _fold_lane(q_buf.at[q_slot], g, bq, lane_width_qk)
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
      assert raw_logits.shape == (frag_rows, bkv_compute)
      if softcap:
        # Note (david): the softcap is applied before masking.
        return jnp.tanh(raw_logits) * softcap
      else:
        return raw_logits

    def _masked_scores(kv_compute_index, q_compute_index, logits, needs_mask,
                       apply_bounds):
      # Note (david): per-seq bounds live in unshifted q coordinates, while the
      # static window lives in coordinates shifted by causal_offset.
      q_unshifted = q_tok + q_compute_index * bq_compute
      q_abs = q_unshifted + causal_offset
      k_abs = kv_tok + kv_compute_index * bkv_compute

      def _apply_window_mask(scores):
        left, right = window
        q_ids = q_abs + lax.broadcasted_iota(jnp.int32, token_tile_shape, 0)
        k_ids = k_abs + lax.broadcasted_iota(jnp.int32, token_tile_shape, 1)
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

      # Note (david): a mask depends on the row's token alone, so a gqa fold
      # masks the (gqa_fold, bq_compute, bkv_compute) view of the fragment,
      # and one token-plane comparison broadcasts over its heads.
      if gqa_fold > 1:
        token_logits = logits.reshape(gqa_fold, *token_tile_shape)
      else:
        token_logits = logits
      window_masked = lax.cond(
          needs_mask, _apply_window_mask, lambda scores: scores, token_logits)

      if apply_bounds:
        # Note (david): apply_bounds is static so interior blocks never trace
        # these VPU ops per score tile. The bounds rows sit in one slot per kv
        # compute tile, since the tile loop indexes the tile at runtime and
        # Mosaic slices lanes only at static offsets; the span row exists only
        # under a left window, so every other build keeps a single comparison.
        q_ids = q_unshifted + lax.broadcasted_iota(
            jnp.int32, token_tile_shape, 0)
        staged_bounds = bounds_buf.at[kv_compute_index]
        if schedule.has_q_span:
          scores = apply_packed_bounds_mask(
              window_masked,
              DEFAULT_MASK_VALUE,
              q_ids,
              staged_bounds[0:1, :],
              staged_bounds[1:2, :],
          )
        else:
          scores = jnp.where(
              q_ids >= staged_bounds[0:1, :], window_masked,
              DEFAULT_MASK_VALUE,
          )
      else:
        scores = window_masked
      return scores.reshape(logits.shape)

    def _tile_parity(kv_compute_index, is_tile_zero, new_parity, old_parity):
      """The parity a kv tile reads the row state from, the old one at tile 0;
      is_tile_zero is a static bool where the caller knows, else None."""
      if is_tile_zero is None:
        return jnp.where(kv_compute_index == 0, old_parity, new_parity)
      else:
        return old_parity if is_tile_zero else new_parity

    def _flag_overflow_risk(g, row_sum, is_checked):
      # Note (david): row_sum upper-bounds every single score of the row, so
      # this detector may request a needless replay but never misses a score
      # above the guard bound. Checking at the block's last live tile is
      # enough, since a tile past a q tile's own frontier adds exactly 0 to
      # every anchored row's sum.
      has_overflow_risk = jnp.any(
          jnp.logical_not(row_sum <= guard_detect_bound))
      guard_smem[g] = jnp.maximum(
          guard_smem[g],
          jnp.where(is_checked, has_overflow_risk.astype(jnp.int32), 0),
      )

    def _online_softmax(
        g, kv_compute_index, q_compute_index, logits, needs_mask,
        guard_recovery, apply_bounds, window_fragment=None,
        is_tile_zero=None, last_kv_tile=num_k_compute_tiles - 1,
    ):
      """window_fragment is the fragment's static plan entry on a window
      block copy, None on the generic copies. is_tile_zero is a static bool
      when the caller knows whether kv_compute_index is 0, else None.
      last_kv_tile is the block's last kv compute tile that runs, where the
      ordinary pass checks the guard; None leaves the check to the caller."""
      head_rows = pl.ds(g * group_bq, group_bq)
      new_parity, old_parity = _block_parities()
      row_sum_new_ref = row_sum_scratch.at[new_parity, head_rows]
      row_max_scratch_g = row_max_scratch.at[head_rows]

      slice_q = pl.ds(q_compute_index * frag_rows, frag_rows)
      scores = _masked_scores(kv_compute_index, q_compute_index, logits,
                              needs_mask, apply_bounds)

      read_parity = _tile_parity(
          kv_compute_index, is_tile_zero, new_parity, old_parity)
      row_sum_read_ref = row_sum_scratch.at[read_parity, head_rows]

      state_shape = row_max_scratch_g[slice_q, :].shape
      frag_row_max = lax.broadcast_in_dim(
          scores.max(axis=-1), state_shape, (0,))
      if window_fragment is None:
        is_row_start = jnp.logical_and(
            is_first,
            kv_compute_index == 0 if is_tile_zero is None else is_tile_zero)
      else:
        # Note (david): a band clipped at kv 0 starts at the row's first block.
        is_row_start = jnp.logical_or(
            window_fragment.starts_band,
            jnp.logical_and(is_first, kv_compute_index == 0))
      row_max_prev = row_max_scratch_g[slice_q, :]
      if defers_row_start:
        # Note (david): a row whose fragments so far all lay before its window
        # holds UNANCHORED_ROW_MAX, which zeroes every probability it scores,
        # so its row_sum and o stay exactly 0 until its first visible key
        # anchors it, and no rescale is due then.
        was_unanchored = row_max_prev >= UNANCHORED_ROW_BOUND
        takes_frag_anchor = jnp.logical_or(is_row_start, was_unanchored)
        frag_anchor = jnp.where(
            frag_row_max > MASKED_ROW_BOUND, frag_row_max, UNANCHORED_ROW_MAX)
      else:
        takes_frag_anchor = is_row_start
        frag_anchor = frag_row_max
      if guard_recovery:
        # Note (david): the recovery replay rebases the anchor onto the
        # overflowing fragment; the ordinary pass keeps the anchor it first
        # picked.
        should_rebase = frag_row_max - row_max_prev > guard_threshold
        row_max = jnp.where(
            takes_frag_anchor,
            frag_anchor,
            jnp.where(
                should_rebase,
                jnp.maximum(row_max_prev, frag_row_max),
                row_max_prev,
            ),
        )
      else:
        row_max = jnp.where(takes_frag_anchor, frag_anchor, row_max_prev)
      # Note (david): logits are in log2 units, so exp2 drops the per-element
      # multiply jnp.exp lowers to on TPU.
      probs = jnp.exp2(scores - jnp.tile(row_max, (1, bkv_repeats)))
      frag_row_sum = lax.broadcast_in_dim(
          probs.sum(axis=-1), state_shape, (0,))
      if guard_recovery and defers_row_start:
        # Note (david): an unanchored row's state is exactly 0, but
        # exp2(UNANCHORED_ROW_MAX - anchor) is inf, and inf * 0 is NaN.
        rescale = jnp.where(
            was_unanchored, 0.0, jnp.exp2(row_max_prev - row_max))
        row_sum_prev = rescale * row_sum_read_ref[slice_q, :]
      elif guard_recovery:
        rescale = jnp.exp2(row_max_prev - row_max)
        row_sum_prev = rescale * row_sum_read_ref[slice_q, :]
      else:
        rescale = None
        row_sum_prev = row_sum_read_ref[slice_q, :]
      row_sum_new = jnp.where(
          is_row_start, frag_row_sum, row_sum_prev + frag_row_sum)
      row_sum_new_ref[slice_q, :] = row_sum_new
      row_max_scratch_g[slice_q, :] = row_max
      if guard_recovery:
        # Note (david): the rescale rides the pipeline carry to this fragment's
        # PV. A per-q-tile scratch would race: once skipped fragments make two
        # fragments of one q tile adjacent, the later softmax overwrites it
        # before the earlier fragment's PV reads it.
        rescale = jnp.where(is_row_start, 1.0, rescale)
      elif window_fragment is not None:
        _flag_overflow_risk(g, row_sum_new, window_fragment.checks_guard)
      elif last_kv_tile is not None:
        _flag_overflow_risk(g, row_sum_new, kv_compute_index == last_kv_tile)
      return probs, is_row_start, rescale

    def _block_pv(
        g, kv_compute_index, q_compute_index, probs, is_row_start, rescale,
        is_tile_zero=None,
    ):
      """is_tile_zero as in _online_softmax."""
      new_parity, old_parity = _block_parities()
      _pv(g, q_compute_index, probs, _v_tile(g, kv_compute_index, kv_slot),
          is_row_start, rescale, write_parity=new_parity,
          read_parity=_tile_parity(
              kv_compute_index, is_tile_zero, new_parity, old_parity))

    def _carry_out_pv(probs, is_row_start, kv_compute_index=pending_k,
                      is_tile_zero=num_k_compute_tiles == 1):
      """Leave the PV of the block's last fragment (head pending_g, q tile
      pending_q, kv tile kv_compute_index) to the next block; is_tile_zero as
      in _online_softmax."""
      new_parity, old_parity = _block_parities()
      pending_probs[...] = probs
      pending_v[...] = _v_tile(pending_g, kv_compute_index, kv_slot)
      pending_smem[PENDING_WRITE_PARITY] = new_parity
      pending_smem[PENDING_READ_PARITY] = _tile_parity(
          kv_compute_index, is_tile_zero, new_parity, old_parity)
      pending_smem[PENDING_ROW_START] = jnp.asarray(is_row_start, jnp.int32)

    def _run_schedule(is_causal, guard_recovery, static_g, apply_bounds,
                      window_fragments=None, *, skips_dead_tiles=False,
                      takes_pending=False, carries_out=False):
      """takes_pending issues the previous block's pending PV after the first
      softmax, and carries_out leaves the last fragment's PV pending; a
      replay passes neither. skips_dead_tiles runs a bounds-masked block's
      live tiles alone."""
      def _after_first_softmax():
        if takes_pending:
          _issue_pending_pv()
          if not carries_out:
            _cancel_pending_pv()

      # Note (david): static_g None is the fast path, one distance-1 pipeline
      # across every folded head so head g + 1's first QK overlaps head g's
      # last PV. An int static_g is the rare per-head overflow replay, kept
      # simple on purpose.
      heads = range(head_fold) if static_g is None else (static_g,)
      if skips_dead_tiles:
        # Note (david): the live tiles run in a loop whose body is the q tiles
        # of one kv tile, straight-line, with the pending PV in the carry; a
        # cond around every fragment instead ran 17.4 against 3.78 ms (v6e,
        # 32:8 d128, paged (512, 2048, 128, 512)). A live tile past a q tile's
        # own frontier scores fully masked fragments, which leave every
        # anchored row's state as it was. Tile 0 is peeled so the carry always
        # holds a pending PV, and with two tiles per block only tile 0 is live.
        last_q_tile = num_q_compute_tiles - 1
        has_tile_loop = num_k_compute_tiles > 2
        last_live_tile = num_live_tiles - 1 if has_tile_loop else 0

        def _tile_fragments(g, kv_compute_index, is_tile_zero, pending_pv):
          for q_compute_index in range(num_q_compute_tiles):
            logits = _qk(g, kv_compute_index, q_compute_index)
            probs, is_row_start, rescale = _online_softmax(
                g, kv_compute_index, q_compute_index, logits, is_causal,
                guard_recovery=guard_recovery, apply_bounds=True,
                is_tile_zero=is_tile_zero,
                last_kv_tile=None if has_tile_loop else 0,
            )
            if pending_pv is None:
              _after_first_softmax()
            else:
              _block_pv(*pending_pv)
            pending_pv = (g, kv_compute_index, q_compute_index, probs,
                          is_row_start, rescale, is_tile_zero)
          return pending_pv

        pending_pv = None
        for g in heads:
          pending_pv = _tile_fragments(g, 0, True, pending_pv)
          if has_tile_loop:
            def _tile_step(kv_compute_index, carry, g=g):
              # Note (david): the carried PV is tile 0's on the first
              # iteration, so its parity stays runtime.
              pending = (g, kv_compute_index - 1, last_q_tile, *carry, None)
              *_, probs, is_row_start, rescale, _ = _tile_fragments(
                  g, kv_compute_index, False, pending)
              return probs, is_row_start, rescale

            carry = lax.fori_loop(
                1, num_live_tiles, _tile_step, tuple(pending_pv[3:6]))
            pending_pv = (g, last_live_tile, last_q_tile, *carry, None)
        if carries_out:
          assert not guard_recovery
          _, last_k, _, probs, is_row_start, _, is_tile_zero = pending_pv
          _carry_out_pv(probs, is_row_start, last_k, is_tile_zero)
        else:
          _block_pv(*pending_pv)
        if has_tile_loop and not guard_recovery:
          # Note (david): once the loop ends each head's new-parity row sums
          # are its last live tile's, so one check replaces a runtime tile
          # compare in every fragment.
          for g in heads:
            new_parity, _ = _block_parities()
            _flag_overflow_risk(
                g,
                row_sum_scratch[new_parity, pl.ds(g * group_bq, group_bq), :],
                True)
      elif window_fragments is not None:
        assert not carries_out
        pending_pv = None
        for g in heads:
          for fragment in window_fragments:
            logits = _qk(g, fragment.kv_tile, fragment.q_tile)
            probs, is_row_start, rescale = _online_softmax(
                g, fragment.kv_tile, fragment.q_tile, logits,
                fragment.needs_mask, guard_recovery=guard_recovery,
                apply_bounds=False, window_fragment=fragment,
            )
            if pending_pv is None:
              _after_first_softmax()
            else:
              _block_pv(*pending_pv)
            pending_pv = (g, fragment.kv_tile, fragment.q_tile, probs,
                          is_row_start, rescale)
        _block_pv(*pending_pv)
      else:
        if static_g is None:
          num_fragments = head_fold * num_compute_tiles

          def _fragment_coords(frag_idx):
            g = frag_idx // num_compute_tiles
            tile_idx = frag_idx % num_compute_tiles
            return (g, tile_idx // num_q_compute_tiles,
                    tile_idx % num_q_compute_tiles)
        else:
          num_fragments = num_compute_tiles

          def _fragment_coords(frag_idx):
            return (static_g, frag_idx // num_q_compute_tiles,
                    frag_idx % num_q_compute_tiles)

        # Note (david): a pinned g replaces the loop-carried copy as a Python
        # int, because sub-tile (d64) lane windows need static offsets; a
        # static g costs nothing on the other builds.
        def _resolve_g(g):
          return static_g if static_g is not None else g

        first_g, first_k, first_q = (
            jnp.int32(coord) for coord in _fragment_coords(0))
        logits = _qk(_resolve_g(first_g), first_k, first_q)
        probs_first, is_row_start_first, rescale_first = _online_softmax(
            _resolve_g(first_g), first_k, first_q, logits,
            needs_mask=is_causal, guard_recovery=guard_recovery,
            apply_bounds=apply_bounds,
        )
        _after_first_softmax()

        def _pipeline_step(frag_idx, carry):
          (probs_prev, is_row_start_prev, rescale_prev, prev_g, prev_k,
           prev_q) = carry
          current_g, current_k, current_q = _fragment_coords(frag_idx)
          logits = _qk(_resolve_g(current_g), current_k, current_q)
          probs_curr, is_row_start_curr, rescale_curr = _online_softmax(
              _resolve_g(current_g), current_k, current_q, logits, is_causal,
              guard_recovery=guard_recovery, apply_bounds=apply_bounds,
          )
          _block_pv(
              _resolve_g(prev_g), prev_k, prev_q, probs_prev,
              is_row_start_prev, rescale_prev,
          )
          return (probs_curr, is_row_start_curr, rescale_curr, current_g,
                  current_k, current_q)

        (probs_last, is_row_start_last, rescale_last, last_g, last_k,
         last_q) = lax.fori_loop(
            1,
            num_fragments,
            _pipeline_step,
            (probs_first, is_row_start_first, rescale_first, first_g, first_k,
             first_q),
            unroll=True,
        )
        if carries_out:
          # Note (david): this copy runs every fragment, so its last fragment
          # is the static pending one.
          assert not guard_recovery
          _carry_out_pv(probs_last, is_row_start_last)
        else:
          _block_pv(
              _resolve_g(last_g), last_k, last_q, probs_last,
              is_row_start_last, rescale_last,
          )

    def _run_group(is_causal, apply_bounds, window_fragments=None,
                   skips_dead_tiles=False, carries_out=False):
      # Note (david): the unpack (or K rotation) leads the copy in program
      # order, where the scheduler overlaps it with the pending PV, which
      # reads only scratch.
      if is_paged:
        _, staged_kv_head, kv_load_end, kv_used_end, staged_kv = block_desc
        valid_rows = jnp.clip(
            jnp.minimum(kv_used_end - staged_kv * bkv,
                        jnp.clip(kv_load_end - staged_kv * bkv, 0, bkv)),
            0, bkv)
        # Note (david): a staged word holds the kv head's K and V rows, split
        # into planes once per compute tile, not once per q tile. Staged rows
        # from valid_rows on may hold NaN (cache slots past seqused_k or stale
        # VMEM), and a zero probability times a NaN value is still NaN, so
        # they unpack as zero values; the keys are masked and skip the
        # select, which keeps it off the path to the first QK.
        for tile in range(num_k_compute_tiles):
          tile_rows = pl.ds(tile * bkv_compute, bkv_compute)

          def _valid_only(rows, tile=tile):
            row_ids = tile * bkv_compute + lax.broadcasted_iota(
                jnp.int32, rows.shape, 0)
            return jnp.where(row_ids < valid_rows, rows, jnp.zeros_like(rows))

          if paged.is_bitcast_load:
            words = load_staged_words(
                staged_words(kv_pair_buf, kv_slot), tile * bkv_compute,
                bkv_compute, 1)
            k_rows, _ = kv_word_halves(
                words, staged_kv_head, kv_pair_buf.dtype)
            _, v_rows = kv_word_halves(
                _valid_only(words), staged_kv_head, kv_pair_buf.dtype)
          else:
            k_rows, v_rows = (
                kv_pair_buf[kv_slot, tile_rows, 0,
                            kv_part * heads_per_word
                            + staged_kv_head % heads_per_word, :]
                for kv_part in range(2))
            v_rows = _valid_only(v_rows)
          # Note (david): an fp8 block is widened to bf16 once per staged
          # block rather than fed to the MXU as fp8, which makes Mosaic
          # rescale every f32 score and output tile for e4m3fn's bias (v6e:
          # 0.356 against 0.349 ms a 1024-token chunk).
          for kv_part, plane_rows in enumerate((k_rows, v_rows)):
            kv_planes[kv_part, tile_rows, :] = plane_rows.astype(
                kv_planes.dtype)
      elif rotary_dim and rotary_k:
        _rotate_staged(k_buf, rotary_k_buf, kv_slot, bkv)
      # Note (david): a path that loads every block starts the block two ahead
      # inside the copy, where the page DMAs' scalar work rides the copy's
      # idle scalar slots.
      if not reuses_blocks:
        _start_block_data(ahead_desc, ahead_slot)
      if is_lane_paired:
        # Note (david): one static-g pipeline per head, since the fused
        # pipeline's loop-carried g would make the 64-lane window offsets
        # dynamic; this gives up the overlap across head boundaries, while the
        # first head still takes the pending PV in and the last carries out.
        for g in range(head_fold):
          _run_schedule(
              is_causal, guard_recovery=False, static_g=g,
              apply_bounds=apply_bounds, window_fragments=window_fragments,
              skips_dead_tiles=skips_dead_tiles,
              takes_pending=g == 0,
              carries_out=carries_out and g == head_fold - 1,
          )
      else:
        _run_schedule(
            is_causal, guard_recovery=False, static_g=None,
            apply_bounds=apply_bounds, window_fragments=window_fragments,
            skips_dead_tiles=skips_dead_tiles,
            takes_pending=True, carries_out=carries_out,
        )

      for g in range(head_fold):
        @pl.when(guard_smem[g] > 0)
        def _recover(g=g):
          _run_schedule(
              is_causal, guard_recovery=True, static_g=g,
              apply_bounds=apply_bounds, window_fragments=window_fragments,
              skips_dead_tiles=skips_dead_tiles,
          )
          if g == pending_g:
            _cancel_pending_pv()

    def _stage_bounds():
      # Note (david): synthesize the bounds rows once for the whole (1, bkv)
      # kv window; evaluated per fragment they spilled vector registers the
      # MXU then waited on (+13.4% on a 2048/256 prefill).
      k_ids = kv_tok + lax.broadcasted_iota(jnp.int32, (1, bkv), 1)
      q_lower, q_span = schedule.q_bounds(ctx, kv, k_ids)
      for tile in range(num_k_compute_tiles):
        tile_lanes = slice(tile * bkv_compute, (tile + 1) * bkv_compute)
        bounds_buf[tile, 0:1, :] = q_lower[:, tile_lanes]
        if schedule.has_q_span:
          bounds_buf[tile, 1:2, :] = q_span[:, tile_lanes]

    # Note (david): a mask-free copy of the fragment pipeline and masked ones,
    # because the masks cost about as many VPU ops as the softmax itself: one
    # per partial-block offset under window_plans, else one applying every
    # mask the build has. One copy per (causal, bounds) pair measured 2.6x
    # slower at block 2048. A build carries either the static window (dense)
    # or the per-seq bounds (varlen), never both, and has_static_mask still
    # gates the former so a full-attention build never loses its upper
    # triangle.
    if is_per_seq:
      needs_any_mask = jnp.logical_or(is_partial, needs_bounds)
    else:
      needs_any_mask = is_partial

    @pl.when(jnp.logical_not(needs_any_mask))
    def _run_unmasked_block():
      _run_group(is_causal=False, apply_bounds=False, carries_out=True)

    # Note (david): each tile loop step ends a basic block the fragment
    # pipeline cannot overlap across, so only a block with dead tiles takes
    # the loop: running every masked block through it slowed the packed
    # varlen kernel by 7% (v6e, 32:8 d128: 7.35 against 6.85 ms).
    if skips_past_frontier and num_k_compute_tiles > 1:
      skips_dead_tiles = num_live_tiles < num_k_compute_tiles
      runs_every_tile = jnp.logical_and(
          needs_any_mask, jnp.logical_not(skips_dead_tiles))

      @pl.when(jnp.logical_and(needs_any_mask, skips_dead_tiles))
      def _run_live_tile_block():
        _stage_bounds()
        _run_group(is_causal=has_static_mask, apply_bounds=True,
                   skips_dead_tiles=True, carries_out=defers_row_finish)
    else:
      runs_every_tile = needs_any_mask

    if window_plans is not None:
      for block_offset, fragments in window_plans.items():
        @pl.when(jnp.logical_and(runs_every_tile,
                                 kv_tok - q_tok == block_offset))
        def _run_window_block(fragments=fragments):
          _run_group(
              is_causal=True, apply_bounds=False, window_fragments=fragments)
    else:
      @pl.when(runs_every_tile)
      def _run_masked_block():
        if is_per_seq:
          _stage_bounds()
        _run_group(is_causal=has_static_mask, apply_bounds=is_per_seq,
                   carries_out=defers_row_finish)

    if defers_row_finish:
      # Note (david): the previous row's last PV ran inside this block's copy,
      # and its final parity stays untouched until this row's second block.
      @pl.when(jnp.logical_and(is_first, prev_row_finish.is_live > 0))
      def _finish_previous_row():
        _finish_row_out(prev_row_finish)

    return (next_slot, ahead_slot, is_next_fresh, is_ahead_fresh,
            next_block_desc, ahead_desc)

  def _blend_stage_rows(stage_ref, finish):
    """Per-seq: copy the rows of the row's stage slot that the previous
    output block owns from that block's slot."""
    # Note (david): a per-seq window sits on the hbm_bf16_tile_rows() grid
    # and may reach below blend_start into rows the previous output block
    # computed correctly. That stage is still resident (num_stages >= 2 and
    # only its outbound DMA reads it), so the rows are copied over from it.
    # The bf16 stage is accessed in whole 16-row tiles, but the two windows
    # can be only 8-aligned apart, so each own tile reads the two source
    # tiles it straddles and picks their 8-row halves in f32, where 8 rows
    # are one whole (8, 128) tile.
    out_slot = lax.rem(finish.out_count, num_stages)
    num_blend_rows = finish.num_blend_rows

    # Note (david): rows below a head group's first block belong to no
    # sequence, and the previous output block is the last group's, so the
    # first block never blends.
    @pl.when(jnp.logical_and(num_blend_rows > 0, finish.si > 0))
    def _blend_head_rows():
      prev_slot = lax.rem(finish.out_count + num_stages - 1, num_stages)
      src_row_offset = finish.q_tok - finish.prev_out_tok
      src_offset_in_tile = src_row_offset & (BF16_TILE_ROWS - 1)
      half_rows = BF16_TILE_ROWS // 2

      def _tile_index(slot, row_start):
        return (slot,) + (slice(None),) * (len(stage_ref.shape) - 3) + (
            pl.ds(pl.multiple_of(row_start, BF16_TILE_ROWS), BF16_TILE_ROWS),
            slice(None))

      def _blend_tile(tile, carry):
        own_index = _tile_index(out_slot, tile * BF16_TILE_ROWS)
        own_rows = stage_ref[own_index].astype(jnp.float32)
        src_lower_start = (
            src_row_offset - src_offset_in_tile + tile * BF16_TILE_ROWS)
        # Note (david): the upper source tile is used only when
        # src_offset_in_tile is 8 and this tile's upper half blends, and then
        # it lies inside the source stage; the clamp keeps the unused read in
        # bounds otherwise.
        src_upper_start = jnp.minimum(src_lower_start + BF16_TILE_ROWS,
                                      stage_ref.shape[-2] - BF16_TILE_ROWS)
        src_lower = stage_ref[
            _tile_index(prev_slot, src_lower_start)].astype(jnp.float32)
        src_upper = stage_ref[
            _tile_index(prev_slot, src_upper_start)].astype(jnp.float32)
        half_ids = lax.broadcasted_iota(
            jnp.int32, src_lower[..., :half_rows, :].shape,
            src_lower.ndim - 2)
        is_src_on_tile = half_ids + src_offset_in_tile < half_rows
        prev_rows = jnp.concatenate([
            jnp.where(is_src_on_tile, src_lower[..., :half_rows, :],
                      src_lower[..., half_rows:, :]),
            jnp.where(is_src_on_tile, src_lower[..., half_rows:, :],
                      src_upper[..., :half_rows, :]),
        ], axis=-2)
        row_ids = tile * BF16_TILE_ROWS + lax.broadcasted_iota(
            jnp.int32, own_rows.shape, own_rows.ndim - 2)
        stage_ref[own_index] = jnp.where(
            row_ids < num_blend_rows, prev_rows, own_rows).astype(
                stage_ref.dtype)
        return carry

      lax.fori_loop(
          0, (num_blend_rows + BF16_TILE_ROWS - 1) // BF16_TILE_ROWS,
          _blend_tile, None)

  def _fill_packed_padding(stage_ref, finish, fill_value):
    """Per-seq: write fill_value into the rows of the row's stage slot at or
    past cu_seqlens_q[-1]."""
    # Note (david): rows at or past cu_seqlens_q[-1] belong to no sequence
    # and read out = 0 (lse = -inf). A window reaching them fills them in its
    # stage, which covers the partial tile at the packed end; the fill after
    # the loop covers the whole tiles past it.
    packed_q_end = schedule.packed_q_end

    @pl.when(finish.q_tok + bq > packed_q_end)
    def _fill_rows():
      stage_slot = stage_ref.at[lax.rem(finish.out_count, num_stages)]
      staged = stage_slot[...]
      rows = finish.q_tok + lax.broadcasted_iota(
          jnp.int32, staged.shape, staged.ndim - 2)
      stage_slot[...] = jnp.where(
          rows < packed_q_end, staged,
          jnp.asarray(fill_value, staged.dtype))

  def _finish_row_out(finish):
    """Normalize the row's output into its output stage slot and start its
    write; its last PV has run, and its final o and row_sum sit at
    finish.parity."""
    out_slot = lax.rem(finish.out_count, num_stages)

    # Note (david): o_stage is one output ring shared by the whole fold group,
    # so wait out the slot's previous DMA before any head writes into it.
    @pl.when(finish.out_count >= num_stages)
    def _wait_prev_out():
      _o_copy(0, 0, out_slot).wait()

    def _finalize_head(g):
      head_rows = pl.ds(g * group_bq, group_bq)
      row_sum_ref = row_sum_scratch.at[finish.parity, head_rows]
      # Note (david): o_stage keeps an explicit fold axis (leading on
      # head-major, lanes on token-major) while o_scratch stays flat-row
      # addressed, so only o_stage goes through _fold_lane. A gqa fold
      # writes all of its heads at once.
      o_ref = (
          o_scratch.at[finish.parity, :, head_rows] if transposed_pv
          else o_scratch.at[finish.parity, head_rows]
      )
      if gqa_fold > 1:
        o_stage_g, o_half = o_stage.at[out_slot], None
      else:
        o_stage_g, o_half = _fold_lane(
            o_stage.at[out_slot], g, bq, lane_width_v, layout=o_layout
        )

      def _finalize_fragment(q_compute_index, carry):
        slice_q = pl.ds(q_compute_index * frag_rows, frag_rows)
        row_sum = row_sum_ref[slice_q, :]
        # Note (david): a quantized cache's V descale is one per-tensor
        # factor, folded into the softmax normalization.
        if transposed_pv:
          tile_index = (slice(None), slice_q)
          inverse_row_sum = (v_scale / row_sum[:, 0:1]).T
        else:
          tile_index = (slice_q, slice(None))
          inverse_row_sum = jnp.tile(
              v_scale / row_sum, (1, head_dim_v_repeats)
          )[..., :o_scratch.shape[-1]]
        o_normalized = (
            o_ref[tile_index] * inverse_row_sum).astype(o_stage.dtype)
        if gqa_fold > 1:
          o_stage_g[:, pl.ds(q_compute_index * bq_compute, bq_compute),
                    :] = o_normalized.reshape(
                        gqa_fold, bq_compute, o_normalized.shape[-1])
        elif o_half is None:
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

    for g in range(head_fold):
      _finalize_head(g)

    if is_per_seq:
      _blend_stage_rows(o_stage, finish)
      _fill_packed_padding(o_stage, finish, 0.0)
    # Note (david): the output copy starts only after the boundary blend has
    # repaired the rows shared with the previous block.
    _o_copy(finish.base_head, finish.q_tok, out_slot).start()

  def _finish_row_lse(group, finish):
    """Stage and write the row's lse; its last softmax has run, and its final
    row_sum sits at finish.parity."""
    out_slot = lax.rem(finish.out_count, num_stages)
    for g in range(head_fold):
      head_rows = pl.ds(g * group_bq, group_bq)
      row_sum_ref = row_sum_scratch.at[finish.parity, head_rows]
      # Note (david): the static anchor keeps sum_j exp2(logit_j) ==
      # exp2(row_max) * row_sum, so lse is row_max + log2(row_sum) in log2
      # units; scale by ln(2) for the natural-log lse that is returned.
      if gqa_fold > 1:
        # Note (david): every lane of a row state holds the row's value, and
        # the rows of a q tile run (folded head, token), so each head's token
        # rows reach its own lse lane through one lane select per head.
        lane_ids = lax.broadcasted_iota(
            jnp.int32, (bq_compute, NUM_LANES), 1)
        for q_compute_index in range(num_q_compute_tiles):
          tile_rows = pl.ds(q_compute_index * frag_rows, frag_rows)
          tile_lse = (
              (row_max_scratch[tile_rows, :]
               + jnp.log2(row_sum_ref[tile_rows, :])) * math.log(2.0)
          ).reshape(gqa_fold, bq_compute, NUM_LANES)
          head_lanes = tile_lse[0]
          for h in range(1, gqa_fold):
            head_lanes = jnp.where(lane_ids == h, tile_lse[h], head_lanes)
          lse_stage[out_slot,
                    pl.ds(q_compute_index * bq_compute, bq_compute), :] = (
                        head_lanes)
      else:
        row_max = row_max_scratch.at[head_rows][:, 0:1]
        lse_row = (
            (row_max + jnp.log2(row_sum_ref[:, 0:1])) * math.log(2.0))
        if is_per_seq:
          lse_stage[out_slot, :, g] = lse_row[:, 0]
        else:
          lse_stage[g, :] = lse_row[:, 0]

    if is_per_seq:
      _blend_stage_rows(lse_stage, finish)
      _fill_packed_padding(lse_stage, finish, -jnp.inf)
      # Note (david): the head group is a leading untiled axis and its
      # 128-lane minor axis is copied whole, avoiding the rejected dynamic
      # minor-axis slice at the head offset.
      lse_copies = [(lse_stage.at[out_slot],
                     lse_hbm.at[group, pl.ds(finish.q_tok, bq), :])]
    elif head_fold % NUM_SUBLANES == 0:
      # Note (david): HBM's leading f32 tile is 8 rows, so an 8-aligned fold
      # is one group-wide copy; a smaller fold copies row by row, since later
      # groups start at h = 2 or 4 and would break tile alignment.
      lse_copies = [(lse_stage,
                     lse_hbm.at[pl.ds(finish.base_head, head_fold),
                                pl.ds(finish.q_tok, bq)])]
    else:
      lse_copies = [
          (lse_stage.at[pl.ds(local_head, 1), :],
           lse_hbm.at[pl.ds(finish.base_head + local_head, 1),
                      pl.ds(finish.q_tok, bq)])
          for local_head in range(head_fold)
      ]
    for lse_src, lse_dst in lse_copies:
      lse_copy = pltpu.make_async_copy(lse_src, lse_dst, sems.at[SEM_LSE, 0])
      lse_copy.start()
      lse_copy.wait()

  def _row_after(g, si):
    """The (group, row) after row si of group g; the grid's last row is its
    own successor."""
    is_row_end = si == num_rows - 1
    is_grid_end = jnp.logical_and(g == num_head_groups - 1, is_row_end)
    next_g = jnp.where(is_row_end, jnp.minimum(g + 1, num_head_groups - 1), g)
    next_si = jnp.where(is_grid_end, si, jnp.where(is_row_end, 0, si + 1))
    return next_g, next_si

  def _first_two_blocks(g, si):
    """The descriptors of the first two blocks of the flattened order from row
    si of group g on, and that row's (lo, hi, ctx)."""
    lo, hi, ctx = schedule.row_interval(si)
    after_g, after_si = _row_after(g, si)
    after_lo, _, after_ctx = schedule.row_interval(after_si)
    first_desc = _block_desc(g * group_heads, lo, ctx)
    second_desc = _select_desc(
        lo < hi, _block_desc(g * group_heads, lo + 1, ctx),
        _block_desc(after_g * group_heads, after_lo, after_ctx))
    return first_desc, second_desc, (lo, hi, ctx)

  def _carried_finish(row_state):
    # Note (david): a loop carry drops q_token's pl.multiple_of, and the
    # output DMA window must start on the hbm_bf16_tile_rows() grid, where
    # q_token put it.
    row_parity, finish = row_state
    return row_parity, finish._replace(
        q_tok=pl.multiple_of(finish.q_tok, hbm_bf16_tile_rows()))

  def _row_body(g, si, carry):
    """carry ends with (the row's first block's new parity, the previous
    row's RowFinish) under defers_row_finish, else with ()."""
    (q_loaded, q_inflight, q_slot, kv_carry, prev_out_tok, num_out_copies,
     row_state) = carry
    if defers_row_finish:
      row_parity, prev_row_finish = _carried_finish(row_state)
    else:
      row_parity, prev_row_finish = 0, None
    lo, hi, ctx = schedule.row_interval(si)
    # Note (david): past the grid's last row the lookahead is that row itself,
    # so the q stream prefetches nothing, and the kv stream reloads blocks the
    # kernel waits out after its loop.
    next_g, next_si = _row_after(g, si)
    first_next_desc, second_next_desc, (_, _, next_ctx) = _first_two_blocks(
        next_g, next_si)
    base_head = g * group_heads
    next_head = next_g * group_heads
    if is_per_seq:
      q_tok = schedule.q_token(ctx)
      next_q_tok = schedule.q_token(next_ctx)
    else:
      q_tok = (schedule.first_row + si) * bq
      next_q_tok = (schedule.first_row + next_si) * bq

    q_loaded, q_inflight, q_slot = advance(
        g * num_rows + si,
        next_g * num_rows + next_si,
        lambda slot: _start_q_row(base_head, q_tok, slot),
        lambda slot: _start_q_row(next_head, next_q_tok, slot),
        _wait_q_row,
        q_loaded, q_inflight, q_slot, num_stages,
    )
    kv_carry = lax.fori_loop(
        lo, hi + 1,
        lambda kv, block_carry: _block_step(
            g, si, kv, lo, hi, ctx,
            (q_tok, base_head, q_slot, row_parity, prev_row_finish),
            (first_next_desc, second_next_desc), block_carry),
        kv_carry,
    )

    finish = RowFinish(
        is_live=jnp.int32(1), si=si, q_tok=q_tok, base_head=base_head,
        num_blend_rows=(schedule.blend_start(ctx) - q_tok if is_per_seq
                        else jnp.int32(0)),
        prev_out_tok=prev_out_tok, out_count=num_out_copies,
        parity=lax.rem(row_parity + hi - lo, 2))
    if defers_row_finish:
      if return_lse:
        _finish_row_lse(g, finish)
      next_row_state = (1 - finish.parity, finish)
    else:
      @pl.when(pending_smem[PENDING_WRITE_PARITY] != TRASH_PARITY)
      def _drain_pending_pv():
        _issue_pending_pv()

      _cancel_pending_pv()
      _finish_row_out(finish)
      if return_lse:
        _finish_row_lse(g, finish)
      next_row_state = ()
    next_prev_out_tok = q_tok if is_per_seq else prev_out_tok
    return (q_loaded, q_inflight, q_slot, kv_carry, next_prev_out_tok,
            num_out_copies + 1, next_row_state)

  _cancel_pending_pv()
  first_desc, second_desc, _ = _first_two_blocks(jnp.int32(0), jnp.int32(0))
  _start_block_data(first_desc, 0)
  is_second_fresh = _is_fresh(second_desc, first_desc).astype(jnp.int32)
  second_slot = is_second_fresh

  @pl.when(is_second_fresh > 0)
  def _start_second():
    _start_block_data(second_desc, second_slot)

  minus_one = jnp.int32(-1)
  zero = jnp.int32(0)
  if defers_row_finish:
    init_row_state = (zero, RowFinish(*([zero] * len(RowFinish._fields))))
  else:
    init_row_state = ()
  init_carry = (minus_one, minus_one, zero,
                (zero, second_slot, jnp.int32(1), is_second_fresh,
                 first_desc, second_desc),
                zero, zero, init_row_state)
  *_, kv_carry, _, num_out_copies, row_state = lax.fori_loop(
      0, num_head_groups,
      lambda g, group_carry: lax.fori_loop(
          0, num_rows, lambda si, row_carry: _row_body(g, si, row_carry),
          group_carry),
      init_carry,
  )
  (last_slot, after_last_slot, is_last_fresh, is_after_last_fresh, last_desc,
   after_last_desc) = kv_carry

  if defers_row_finish:
    # Note (david): the last row has no next row to run its pending PV and
    # finish its output.
    _, last_row_finish = _carried_finish(row_state)

    @pl.when(pending_smem[PENDING_WRITE_PARITY] != TRASH_PARITY)
    def _drain_last_pending_pv():
      _issue_pending_pv()

    _cancel_pending_pv()

    @pl.when(last_row_finish.is_live > 0)
    def _finish_last_row():
      _finish_row_out(last_row_finish)

  @pl.when(is_last_fresh > 0)
  def _wait_last():
    _wait_block_data(last_desc, last_slot)

  @pl.when(is_after_last_fresh > 0)
  def _wait_after_last():
    _wait_block_data(after_last_desc, after_last_slot)

  drain(num_out_copies, num_stages, lambda slot: _o_copy(0, 0, slot).wait())

  if is_per_seq:
    # Note (david): packed padding rows own no q block, so once every output
    # DMA has landed they are written from a cleared stage. The last block
    # already zeroed its window's padding rows, so the fill starts at the next
    # HBM tile; with no block at all, cu_seqlens_q is all 0 and so is that
    # tile.
    packed_q_end = schedule.packed_q_end
    padded_total_q = o_hbm.shape[-2]
    tile_rows = hbm_bf16_tile_rows()
    fill_start = (packed_q_end + tile_rows - 1) // tile_rows * tile_rows
    num_fills = pl.cdiv(jnp.maximum(padded_total_q - fill_start, 0), bq)

    @pl.when(num_fills > 0)
    def _clear_fill_stage():
      o_stage[0] = jnp.zeros(o_stage.shape[1:], o_stage.dtype)
      if return_lse:
        lse_stage[0] = jnp.full(lse_stage.shape[1:], -jnp.inf, jnp.float32)

    def _fill_rows(fill_index, carry):
      row_start = pl.multiple_of(fill_start + fill_index * bq, tile_rows)
      num_fill_rows = pl.multiple_of(
          jnp.minimum(padded_total_q - row_start, bq), tile_rows)
      fill_rows = pl.ds(0, num_fill_rows)
      # Note (david): only a head-major fold stages a leading fold axis; a
      # token-major fold rides the lanes.
      if token_major is None and group_heads > 1:
        o_fill = o_stage.at[0, :, fill_rows, :]
      else:
        o_fill = o_stage.at[0, fill_rows, :]
      for group in range(num_head_groups):
        o_fill_copy = (
            o_fill,
            hbm_window(o_hbm, group * group_heads, row_start, num_fill_rows,
                       width=width_v, heads_per_row=q_heads_per_row,
                       layout=o_layout, token_major=token_major,
                       fold=group_heads),
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
    v: jax.Array | None,
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
    rotary_interleaved: bool = False,
    paged: PagedKVInfo | None = None,
    gqa_fold: int = 1,
    **body,
) -> jax.Array | tuple[jax.Array, jax.Array]:
  """Validate q/k/v, allocate scratch and run a fwd_body kernel.

  kernel is the mode kernel with its schedule already bound; every fwd_body
  static that follows from block_sizes or the q/k/v shapes is bound here.
  rotary is prepare_rotary's packed (batch, tokens, head_dim) Q and K pair
  (K None when K arrives rotated). Returns o shaped like q (head_dim_v wide),
  or (o, lse) with a (num_q_heads, q_seq_len) float32 natural-log lse.

  paged (per-seq, head-major q only) makes k the merged interleaved cache
  paged describes and v None; the kernel then takes (q, k, *rotary). Only a
  paged build takes gqa_fold > 1, the q heads per kv head that fwd_body
  stacks over the rows of one score fragment.
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
  if gqa_fold > 1 and (paged is None or head_fold != 1 or transposed_pv
                       or gqa_fold > NUM_LANES):
    # Note (david): a folded group's lse rides one 128-lane row per token.
    raise ValueError(
        f"{gqa_fold=} needs a paged build with head_fold == 1, no transposed"
        f" pv and at most {NUM_LANES} folded heads; got {head_fold=},"
        f" {transposed_pv=}.")
  if gqa_fold > 1 and bq_compute % BF16_TILE_ROWS:
    # Note (david): a folded q tile collapses (gqa_fold, bq_compute) into
    # fragment rows, which moves no data only on whole bf16 tiles.
    raise ValueError(
        f"a gqa fold needs {bq_compute=} to be a multiple of"
        f" {BF16_TILE_ROWS}.")

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
    if gqa_fold > 1 and num_q_heads != num_kv_heads * gqa_fold:
      raise ValueError(
          "a gqa fold stacks all of a kv head's q heads; got"
          f" {num_q_heads=}, {num_kv_heads=}, {gqa_fold=}.")
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

  # Note (david): a per-seq schedule clamps its windows to the axis end and
  # checks that each block fits, so only the dense grid needs the divisibility.
  if not is_per_seq and (q_seq_len % bq or kv_seq_len % bkv):
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
    if (rotary_q_coeff.ndim != 3
        or rotary_q_coeff.shape[0] == 0
        or rotary_q_coeff.shape[1] != q_seq_len
        or (rotary_k and (rotary_k_coeff.ndim != 3
                          or rotary_q_coeff.shape[0] != rotary_k_coeff.shape[0]
                          or rotary_k_coeff.shape[1] != kv_seq_len
                          or rotary_q_coeff.shape[2] != rotary_k_coeff.shape[2]))):
      raise ValueError("rotary coefficients must be (batch, tokens, head_dim).")
    rotary_dim = rotary_q_coeff.shape[2]
    rotary_batch = rotary_q_coeff.shape[0]
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
      rotary_groups=rotary_groups, rotary_k=rotary_k, paged=paged,
      gqa_fold=gqa_fold, **body,
  )

  group_heads = head_fold * gqa_fold
  folded_bq = group_heads * bq
  o_layout = QKVLayout.SEQ_MINOR if transposed_pv else QKVLayout.HEAD_DIM_MINOR
  vmem = pltpu.VMEM

  def _staged_shape(rows, head_dim, layout, stages=num_stages):
    if group_heads == 1:
      return (stages, *from_head_minor((rows, head_dim), layout))
    elif token_major is not None:
      # Note (david): a token-major group is head_fold * head_dim consecutive
      # lanes of one row, so the fold rides the lane axis and one DMA per
      # stream lands the whole group.
      return (stages, rows, head_fold * head_dim)
    else:
      # Note (david): an explicit fold axis rank-matches the 3-D HBM head
      # slices, so XLA never materializes a reshaped q/k/v copy, which
      # dominated wall time at short sequences when head_dim < 128 needed lane
      # padding.
      return (stages, group_heads,
              *from_head_minor((rows, head_dim), layout))

  q_buf = vmem(_staged_shape(bq, head_dim_qk, qkv_layout), q.dtype)
  if paged is None:
    k_buf = vmem(
        _staged_shape(bkv, head_dim_qk, qkv_layout, NUM_KV_STAGES), k.dtype)
    v_buf = vmem(
        _staged_shape(bkv, head_dim_v, qkv_layout, NUM_KV_STAGES), v.dtype)
    scratch_shapes = [q_buf, k_buf, v_buf]
  else:
    # Note (david): a paged block stages the u32 word rows holding one kv
    # head's K and V, then unpacks them into one bf16 K and one bf16 V plane;
    # a word's (kv_word_rows, head_dim) rows pad to nothing, so the staging
    # costs what two dense planes of the cache dtype would.
    scratch_shapes = [
        q_buf,
        vmem((NUM_KV_STAGES, bkv, 1, kv_word_rows(k.dtype), head_dim_qk),
             k.dtype),
        vmem((2, bkv, head_dim_qk), q.dtype)]
  if is_per_seq:
    # Note (david): the per-seq kernel synthesizes its bounds rows (the q lower
    # bound, plus the span under a left window) once per block in-kernel, so
    # it needs one block's rows, not a kv-stream ring. Both rows share one
    # (8, 128) int32 tile, so the span row costs no extra VMEM.
    scratch_shapes.append(
        vmem((bkv // bkv_compute, 2, bkv_compute), jnp.int32))
  if rotary_dim:
    scratch_shapes += [
        vmem((stages, rotary_groups,
              *from_head_minor((block, rotary_dim), qkv_layout)),
             rotary_dtype)
        for block, stages in (((bq, num_stages), (bkv, NUM_KV_STAGES))
                              if rotary_k else ((bq, num_stages),))
    ]
  o_stage = vmem(_staged_shape(bq, head_dim_v, o_layout), q.dtype)
  row_max_scratch = vmem((folded_bq, NUM_LANES), jnp.float32)
  row_sum_scratch = vmem((2, folded_bq, NUM_LANES), jnp.float32)
  o_scratch = vmem(
      (NUM_O_SLOTS, *from_head_minor((folded_bq, head_dim_v), o_layout)),
      jnp.float32)
  # Note (david): a lane-paired fold's pending V tile is its pair's 128 lanes,
  # as fwd_body loads it.
  pending_probs = vmem((gqa_fold * bq_compute, bkv_compute), jnp.float32)
  is_lane_paired = (token_major is not None and head_fold > 1
                    and head_dim_qk % NUM_LANES != 0)
  v_tile_width = NUM_LANES if is_lane_paired else head_dim_v
  pending_v = vmem(
      from_head_minor((bkv_compute, v_tile_width), qkv_layout),
      q.dtype if paged is not None else v.dtype)
  guard_smem = pltpu.SMEM((head_fold,), jnp.int32)
  parity_smem = pltpu.SMEM((1,), jnp.int32)
  pending_smem = pltpu.SMEM((NUM_PENDING_FIELDS,), jnp.int32)
  sems = pltpu.SemaphoreType.DMA(
      (NUM_SEMS + num_rotary_streams, max(num_stages, NUM_KV_STAGES)))
  scratch_shapes += [o_stage, row_max_scratch, row_sum_scratch, o_scratch,
                     pending_probs, pending_v, guard_smem, parity_smem,
                     pending_smem, sems]
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
    # Note (david): the pool is pinned to HBM because memory space assignment
    # otherwise moves a pool that fits its VMEM budget there and back around
    # every call, the whole pool however few pages the call reads (v6e, 16 MB
    # e5m2 pool: 0.369 against 0.339 ms a chunk). The spec stays pl.ANY: a
    # layout copy XLA inserts before the call escapes the constraint and can
    # land in VMEM, where DMAs compiled for an HBM operand halt the chip.
    kernel_inputs = [
        q_in, pltpu.with_memory_space_constraint(k_in, pltpu.HBM),
        *rotary_in]
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
  # groups never alias in the physical layout; only the first group_heads lanes
  # are returned.
  lse_shape = (
      (num_q_heads // group_heads, q_seq_len, NUM_LANES)
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
    return o, lse_kernel[:, :, :group_heads].swapaxes(1, 2).reshape(
        num_q_heads, q_seq_len)
  else:
    return o, lse_kernel
