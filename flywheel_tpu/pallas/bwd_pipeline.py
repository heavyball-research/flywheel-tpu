"""Kv-outer backward kernel body and its launcher."""

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .block_sizes import (
  DEFAULT_MASK_VALUE,
  LOG2E,
  NN_DIM_NUMBERS,
  NT_DIM_NUMBERS,
  NUM_LANES,
  VMEM_LIMIT_BYTES,
  QKVLayout,
  TokenMajorInfo,
  default_block,
)
from .copy_utils import advance, drain, fold_row, hbm_head_slice, hbm_window
from .loop_schedule import BwdSchedule
from .mask import apply_causal_kv_major

# Note (david): contracting axis 0 of both ds and k gives dq = ds^T @ k without
# materializing the transpose.
DQ_DIM_NUMBERS = (((0,), (0,)), ((), ()))

SEM_Q = 0
SEM_DO = 1
SEM_LSE = 2
SEM_DI = 3
SEM_K = 4
SEM_V = 5
SEM_ODQ = 6
SEM_ODK = 7
SEM_ODV = 8
NUM_SEMS_BWD = 9

# Note (david): the dq accumulator and stage buffers cap the useful backward
# block at 1024, one rung below the forward's ladder.
BWD_BLOCKS = (1024, 512, 256, 128)

default_bwd_block = functools.partial(default_block, candidates=BWD_BLOCKS)


def bwd_body(
    schedule: BwdSchedule,
    refs: tuple[jax.Array, ...],
    *,
    bq: int,
    bkv: int,
    bq_compute: int,
    bkv_compute: int,
    num_stages: int,
    scale: float,
    causal: bool,
    causal_offset: int,
    out_dtype: jnp.dtype,
    head_fold: int,
    kv_fold: int,
    token_major: TokenMajorInfo | None,
) -> None:
  """Kv-outer backward over schedule.

  refs are q, k, v, do, lse, di, dq, dk, dv, then call_bwd's scratch in order.
  dk/dv accumulate per kv block and dq over the whole group; each streams out
  as soon as it is complete.
  """
  # Note (david): the exact unpack pins this ref list against call_bwd's
  # operand and scratch order; a mismatch fails here instead of shifting every
  # later ref.
  (q_hbm, k_hbm, v_hbm, do_hbm, lse_hbm, di_hbm, dq_hbm, dk_hbm, dv_hbm,
   q_buf, do_buf, lse_buf, di_buf, k_buf, v_buf,
   dq_stage, dk_stage, dv_stage, dq_acc, dk_acc, dv_acc, sems) = refs

  f32 = jnp.float32
  num_q_tiles = bq // bq_compute
  num_kv_tiles = bkv // bkv_compute
  # Note (david): scores stay in log2 units so the softmax recompute can use
  # exp2; lse arrives pre-scaled by LOG2E to match.
  score_scale = scale * LOG2E
  # Note (david): fully masked q rows (causal_offset < 0) carry lse == -inf, so
  # exp2(qk - lse) overflows to inf and 0 * inf = NaN would reach dk/dv.
  should_guard_lse = causal and causal_offset < 0

  if token_major is None:
    lane_width_qk = None
    lane_width_v = None
    q_heads_per_row = None
    kv_heads_per_row = None
  else:
    lane_width_qk = token_major.head_dim_qk
    lane_width_v = token_major.head_dim_v
    q_heads_per_row = token_major.num_q_heads
    kv_heads_per_row = token_major.num_kv_heads

  # Note (david): Mosaic takes a lane window of a scratch ref only at whole-tile
  # offset and size once another axis is sliced too, so 64-wide heads are
  # addressed a lane tile (a head pair) at a time: loads take the head's half
  # out of the value and stores go through a tile-wide read-modify-write. The
  # head loop is sequential Python, so the partner half is never stale.
  is_lane_paired = (
      token_major is not None and token_major.head_dim_qk % NUM_LANES != 0)
  if is_lane_paired:
    heads_per_lane_tile = NUM_LANES // token_major.head_dim_qk
  else:
    heads_per_lane_tile = 1

  def _fold_head(ref, head_idx, lane_width=None, fold=head_fold):
    # Note (david): fold == 1 passes the ref through, which under GQA makes
    # every head of the group share the kv side's single buffer.
    if fold == 1:
      return ref
    else:
      return fold_row(ref, head_idx, None, fold, lane_width=lane_width)

  def _fold_lane(ref, head_idx, lane_width, fold=head_fold):
    if is_lane_paired and fold > 1:
      tile_start = (head_idx // heads_per_lane_tile) * NUM_LANES
      return (ref.at[:, pl.ds(tile_start, NUM_LANES)],
              head_idx % heads_per_lane_tile)
    else:
      return _fold_head(ref, head_idx, lane_width, fold=fold), None

  def _take_half(value, half, lane_width):
    if half is None:
      return value
    else:
      return value[..., half * lane_width:(half + 1) * lane_width]

  def _read_modify_write(ref, half, lane_width, rows, update):
    current = ref[rows, :]
    if half is None:
      updated = update(current)
      ref[rows, :] = updated
    else:
      # Note (david): jnp .at[].set lowers to scatter, which Pallas TPU lacks,
      # so replace the half with a static split + concatenate.
      updated = update(current[:, half * lane_width:(half + 1) * lane_width])
      lane_slices = [
          current[:, idx * lane_width:(idx + 1) * lane_width]
          for idx in range(current.shape[-1] // lane_width)]
      lane_slices[half] = updated
      ref[rows, :] = jnp.concatenate(lane_slices, axis=-1)
    return updated

  def _stream_copy(hbm, buf, sem, lane_width, head, block_idx, block_size,
                   slot, fold, heads_per_row, is_output=False):
    window = hbm_window(
        hbm, head, block_idx * block_size, block_size, width=lane_width,
        heads_per_row=heads_per_row, layout=QKVLayout.HEAD_DIM_MINOR,
        token_major=token_major, fold=fold)
    if is_output:
      return pltpu.make_async_copy(buf.at[slot], window, sems.at[sem, slot])
    else:
      return pltpu.make_async_copy(window, buf.at[slot], sems.at[sem, slot])

  def _start_all(copies):
    for copy in copies:
      copy.start()

  def _wait_all(copies):
    for copy in copies:
      copy.wait()

  def _qrow_copies(head, q_block, slot):
    copies = [
        _stream_copy(q_hbm, q_buf, SEM_Q, lane_width_qk, head, q_block, bq,
                     slot, head_fold, q_heads_per_row),
        _stream_copy(do_hbm, do_buf, SEM_DO, lane_width_v, head, q_block, bq,
                     slot, head_fold, q_heads_per_row),
    ]
    # Note (david): lse/di are (heads, 1, seq) refs with the q block on the
    # minor axis, which is exactly the SEQ_MINOR head slice.
    for hbm, buf, sem in (
        (lse_hbm, lse_buf, SEM_LSE), (di_hbm, di_buf, SEM_DI)):
      window = hbm_head_slice(
          hbm, head, q_block * bq, bq, QKVLayout.SEQ_MINOR, head_fold)
      copies.append(
          pltpu.make_async_copy(window, buf.at[slot], sems.at[sem, slot]))
    return copies

  def _kv_copies(kv_head, kv_block, slot):
    return (
        _stream_copy(k_hbm, k_buf, SEM_K, lane_width_qk, kv_head, kv_block,
                     bkv, slot, kv_fold, kv_heads_per_row),
        _stream_copy(v_hbm, v_buf, SEM_V, lane_width_v, kv_head, kv_block,
                     bkv, slot, kv_fold, kv_heads_per_row),
    )

  def _dq_out_copy(head, q_block, slot):
    return _stream_copy(
        dq_hbm, dq_stage, SEM_ODQ, lane_width_qk, head, q_block, bq, slot,
        head_fold, q_heads_per_row, is_output=True)

  def _dkv_out_copies(kv_head, kv_block, slot):
    return (
        _stream_copy(dk_hbm, dk_stage, SEM_ODK, lane_width_qk, kv_head,
                     kv_block, bkv, slot, kv_fold, kv_heads_per_row,
                     is_output=True),
        _stream_copy(dv_hbm, dv_stage, SEM_ODV, lane_width_v, kv_head,
                     kv_block, bkv, slot, kv_fold, kv_heads_per_row,
                     is_output=True),
    )

  def _block_step(group, kv, qi, qlo, qhi, ctx, next_row_start, carry):
    (q_loaded, q_inflight, q_slot, kv_loaded, kv_inflight, kv_slot,
     num_dkv_out, num_dq_out) = carry

    q_head = group * head_fold
    kv_head = group * kv_fold
    q_row_key = group * schedule.num_q_blocks + qi
    kv_key = group * schedule.num_kv_blocks + kv
    is_first_q_block = qi == qlo
    is_last_q_block = qi == qhi
    is_dq_first, is_dq_last, should_mask = schedule.block_flags(kv, qi, ctx)
    # Note (david): past the last q block of a kv row the lookahead targets the
    # next row's first entry; on the final entry that is the current one, and
    # equal keys issue no prefetch.
    (next_row_group, next_row_kv, next_row_qi, next_row_q_key,
     next_row_kv_key) = next_row_start
    next_group = jnp.where(is_last_q_block, next_row_group, group)
    next_q_head = next_group * head_fold
    next_kv_head = next_group * kv_fold
    next_kv = jnp.where(is_last_q_block, next_row_kv, kv)
    next_qi = jnp.where(is_last_q_block, next_row_qi, qi + 1)
    next_q_row_key = jnp.where(is_last_q_block, next_row_q_key, q_row_key + 1)
    next_kv_key = jnp.where(is_last_q_block, next_row_kv_key, kv_key)

    q_loaded, q_inflight, q_slot = advance(
        q_row_key, next_q_row_key,
        lambda slot: _start_all(_qrow_copies(q_head, qi, slot)),
        lambda slot: _start_all(_qrow_copies(next_q_head, next_qi, slot)),
        lambda slot: _wait_all(_qrow_copies(0, 0, slot)),
        q_loaded, q_inflight, q_slot, num_stages)
    kv_loaded, kv_inflight, kv_slot = advance(
        kv_key, next_kv_key,
        lambda slot: _start_all(_kv_copies(kv_head, kv, slot)),
        lambda slot: _start_all(_kv_copies(next_kv_head, next_kv, slot)),
        lambda slot: _wait_all(_kv_copies(0, 0, slot)),
        kv_loaded, kv_inflight, kv_slot, num_stages)

    # Note (david): an output slot is reused only when its row flushes, so
    # only then must the slot's previous copy be waited out.
    @pl.when(is_last_q_block)
    def _wait_prev_dkv_out():
      @pl.when(num_dkv_out >= num_stages)
      def _():
        _wait_all(_dkv_out_copies(0, 0, num_dkv_out % num_stages))

    dq_slot = num_dq_out % num_stages

    @pl.when(is_dq_last)
    def _wait_prev_dq_out():
      @pl.when(num_dq_out >= num_stages)
      def _():
        _dq_out_copy(0, 0, dq_slot).wait()

    def _compute_q_tile(q_tile, is_masked):
      q_start = qi * bq + q_tile * bq_compute
      q_tile_rows = pl.ds(q_tile * bq_compute, bq_compute)
      dq_acc_rows = pl.ds(q_start, bq_compute)

      # Note (david): one head of the fold per iteration keeps the live score
      # tile a single head's, so only the staged buffers and accumulators scale
      # with the fold. The schedule is head-independent, so every head shares
      # the same first/last flags. Explicit cross-head pipelining measured
      # identical.
      def _compute_head(head_idx):
        q_ref, q_half = _fold_lane(q_buf.at[q_slot], head_idx, lane_width_qk)
        q_head_tile = _take_half(q_ref[q_tile_rows, :], q_half, lane_width_qk)
        do_ref, do_half = _fold_lane(do_buf.at[q_slot], head_idx, lane_width_v)
        do_head_tile = _take_half(
            do_ref[q_tile_rows, :], do_half, lane_width_v)
        lse_row = _fold_head(lse_buf.at[q_slot], head_idx)[:, q_tile_rows]
        di_row = _fold_head(di_buf.at[q_slot], head_idx)[:, q_tile_rows]
        k_ref, k_half = _fold_lane(
            k_buf.at[kv_slot], head_idx, lane_width_qk, fold=kv_fold)
        v_ref, v_half = _fold_lane(
            v_buf.at[kv_slot], head_idx, lane_width_v, fold=kv_fold)
        dq_acc_ref, dq_acc_half = _fold_lane(dq_acc, head_idx, lane_width_qk)
        dk_acc_ref, dk_acc_half = _fold_lane(
            dk_acc, head_idx, lane_width_qk, fold=kv_fold)
        dv_acc_ref, dv_acc_half = _fold_lane(
            dv_acc, head_idx, lane_width_v, fold=kv_fold)

        # Note (david): kv fragments run to completion one at a time, not
        # distance-1 pipelined like the forward: dp does not depend on p, so
        # intra-fragment ILP already hides the exp/ds work, and the d256
        # ablation measured the pipeline as pure register pressure (+23% vector
        # spill traffic, -2..4% throughput at every seq_len).
        def _kv_tile_step(kv_tile, dq_partial):
          kv_tile_rows = pl.ds(kv_tile * bkv_compute, bkv_compute)
          k_tile = _take_half(k_ref[kv_tile_rows, :], k_half, lane_width_qk)
          v_tile = _take_half(v_ref[kv_tile_rows, :], v_half, lane_width_v)
          qk = lax.dot_general(
              k_tile, q_head_tile, NT_DIM_NUMBERS,
              preferred_element_type=f32) * score_scale
          k_start = kv * bkv + kv_tile * bkv_compute
          # Note (david): is_masked selects one of two static copies of this
          # pipeline, so it is a Python bool, never a tracer.
          if is_masked and causal:
            masked_qk = apply_causal_kv_major(
                qk, DEFAULT_MASK_VALUE, q_start + causal_offset, k_start)
          else:
            masked_qk = qk
          unguarded_p = jnp.exp2(masked_qk - lse_row)
          if should_guard_lse:
            p = jnp.where(jnp.isfinite(lse_row), unguarded_p, 0.0)
          else:
            p = unguarded_p
          dp = lax.dot_general(
              v_tile, do_head_tile, NT_DIM_NUMBERS, preferred_element_type=f32)
          ds = (dp - di_row) * p
          dk_tile = scale * lax.dot_general(
              ds, q_head_tile, NN_DIM_NUMBERS, preferred_element_type=f32)
          dv_tile = lax.dot_general(
              p, do_head_tile, NN_DIM_NUMBERS, preferred_element_type=f32)
          if head_idx > 0 and kv_fold == 1:
            # Note (david): under GQA every head after the first accumulates
            # into the group's one shared dk/dv.
            should_init_dkv = jnp.bool_(False)
          else:
            should_init_dkv = jnp.logical_and(is_first_q_block, q_tile == 0)
          _read_modify_write(
              dk_acc_ref, dk_acc_half, lane_width_qk, kv_tile_rows,
              lambda prev: jnp.where(
                  should_init_dkv, jnp.zeros_like(dk_tile), prev) + dk_tile)
          _read_modify_write(
              dv_acc_ref, dv_acc_half, lane_width_v, kv_tile_rows,
              lambda prev: jnp.where(
                  should_init_dkv, jnp.zeros_like(dv_tile), prev) + dv_tile)
          return dq_partial + scale * lax.dot_general(
              ds, k_tile, DQ_DIM_NUMBERS, preferred_element_type=f32)

        dq_init = jnp.zeros((bq_compute, q_head_tile.shape[-1]), f32)
        dq_tile = lax.fori_loop(
            0, num_kv_tiles, _kv_tile_step, dq_init, unroll=True)
        dq_acc_tile = _read_modify_write(
            dq_acc_ref, dq_acc_half, lane_width_qk, dq_acc_rows,
            lambda prev: jnp.where(
                is_dq_first, jnp.zeros_like(dq_tile), prev) + dq_tile)
        dq_stage_ref, dq_stage_half = _fold_lane(
            dq_stage.at[dq_slot], head_idx, lane_width_qk)

        @pl.when(is_dq_last)
        def _stage_dq():
          _read_modify_write(
              dq_stage_ref, dq_stage_half, lane_width_qk, q_tile_rows,
              lambda _: dq_acc_tile.astype(out_dtype))

      for head_idx in range(head_fold):
        _compute_head(head_idx)

    def _run_q_tiles(is_masked):
      lax.fori_loop(
          0, num_q_tiles,
          lambda q_tile, _: _compute_q_tile(q_tile, is_masked), None,
          unroll=True,
      )

    # Note (david): two static copies of the q-tile pipeline, one with no mask
    # and one with the causal mask, picked per schedule step: the mask costs
    # VPU ops per score element that a block below the diagonal should not
    # pay. The schedule hands back a Python bool when the choice is static.
    if isinstance(should_mask, bool):
      _run_q_tiles(should_mask)
    else:
      for is_masked in (False, True):
        @pl.when(should_mask if is_masked else jnp.logical_not(should_mask))
        def _run_q_tiles_for_mask(is_masked=is_masked):
          _run_q_tiles(is_masked)

    @pl.when(is_dq_last)
    def _flush_dq():
      _dq_out_copy(q_head, qi, dq_slot).start()

    @pl.when(is_last_q_block)
    def _flush_dkv():
      slot = num_dkv_out % num_stages
      dk_stage[slot] = dk_acc[...].astype(out_dtype)
      dv_stage[slot] = dv_acc[...].astype(out_dtype)
      _start_all(_dkv_out_copies(kv_head, kv, slot))

    return (q_loaded, q_inflight, q_slot, kv_loaded, kv_inflight, kv_slot,
            jnp.where(is_last_q_block, num_dkv_out + 1, num_dkv_out),
            jnp.where(is_dq_last, num_dq_out + 1, num_dq_out))

  num_head_groups = schedule.num_head_groups
  num_kv_blocks = schedule.num_kv_blocks

  def _kv_row(group, kv, carry):
    qlo, qhi, ctx = schedule.row_interval(kv)
    # Note (david): the last row of the last group looks ahead to its own
    # entry, and equal keys issue no prefetch.
    is_last_kv = kv == num_kv_blocks - 1
    is_last_row = jnp.logical_and(group == num_head_groups - 1, is_last_kv)
    next_group = jnp.where(
        is_last_kv, jnp.minimum(group + 1, num_head_groups - 1), group)
    next_kv = jnp.where(is_last_row, kv, jnp.where(is_last_kv, 0, kv + 1))
    next_qlo, _, _ = schedule.row_interval(next_kv)
    next_qi = jnp.where(is_last_row, qhi, next_qlo)
    next_row_start = (
        next_group,
        next_kv,
        next_qi,
        next_group * schedule.num_q_blocks + next_qi,
        next_group * num_kv_blocks + next_kv,
    )
    return lax.fori_loop(
        qlo, qhi + 1,
        lambda qi, step_carry: _block_step(
            group, kv, qi, qlo, qhi, ctx, next_row_start, step_carry),
        carry,
    )

  initial_carry = (-1, -1, 0, -1, -1, 0, 0, 0)
  *_, num_dkv_out, num_dq_out = lax.fori_loop(
      0, num_head_groups,
      lambda group, group_carry: lax.fori_loop(
          0, num_kv_blocks,
          lambda kv, row_carry: _kv_row(group, kv, row_carry), group_carry),
      initial_carry,
  )

  drain(num_dq_out, num_stages, lambda slot: _dq_out_copy(0, 0, slot).wait())
  drain(num_dkv_out, num_stages,
        lambda slot: _wait_all(_dkv_out_copies(0, 0, slot)))


def bwd_out_shape(
    token_major: TokenMajorInfo | None,
    dtype: jnp.dtype,
    num_heads: int,
    num_kv_heads: int,
    q_seq_len: int,
    kv_seq_len: int,
    head_dim_qk: int,
    head_dim_v: int,
) -> list[jax.ShapeDtypeStruct]:
  """(dq, dk, dv) output structs, laid out like q, k and v."""
  if token_major is None:
    return [
        jax.ShapeDtypeStruct((num_heads, q_seq_len, head_dim_qk), dtype),
        jax.ShapeDtypeStruct((num_kv_heads, kv_seq_len, head_dim_qk), dtype),
        jax.ShapeDtypeStruct((num_kv_heads, kv_seq_len, head_dim_v), dtype),
    ]
  else:
    batch_axis = () if token_major.batch is None else (token_major.batch,)
    return [
        jax.ShapeDtypeStruct(
            (*batch_axis, q_seq_len, token_major.num_q_heads * head_dim_qk),
            dtype),
        jax.ShapeDtypeStruct(
            (*batch_axis, kv_seq_len, token_major.num_kv_heads * head_dim_qk),
            dtype),
        jax.ShapeDtypeStruct(
            (*batch_axis, kv_seq_len,
             token_major.num_kv_heads * token_major.head_dim_v),
            dtype),
    ]


def call_bwd(
    kernel: Callable[..., None],
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    do: jax.Array,
    lse_log2: jax.Array,
    delta: jax.Array,
    *,
    bq: int,
    bkv: int,
    head_fold: int,
    kv_fold: int,
    num_stages: int,
    num_heads: int,
    num_kv_heads: int,
    q_seq_len: int,
    kv_seq_len: int,
    head_dim_qk: int,
    head_dim_v: int,
    interpret: bool,
    kernel_name: str,
    token_major: TokenMajorInfo | None,
) -> list[jax.Array]:
  """Allocate a bwd_body kernel's stage buffers and accumulators and run it."""
  vmem = pltpu.VMEM
  any_spec = pl.BlockSpec(memory_space=pl.ANY)

  # Note (david): each staged buffer and accumulator carries its family's fold
  # on the axis its HBM window folds, so one DMA per stream moves a whole
  # group: a leading axis on head-major, a fold-times wider lane axis on
  # token-major. lse/di stay head-major under both layouts, so their fold
  # always leads.
  is_head_major = token_major is None
  q_fold_axis = (head_fold,) if is_head_major and head_fold > 1 else ()
  kv_fold_axis = (kv_fold,) if is_head_major and kv_fold > 1 else ()
  lse_fold_axis = (head_fold,) if head_fold > 1 else ()
  q_lane_fold = 1 if is_head_major else head_fold
  kv_lane_fold = 1 if is_head_major else kv_fold
  q_width = q_lane_fold * head_dim_qk
  do_width = q_lane_fold * head_dim_v
  k_width = kv_lane_fold * head_dim_qk
  v_width = kv_lane_fold * head_dim_v
  # Note (david): the order must match bwd_body's ref unpacking.
  scratch_shapes = [
      vmem((num_stages, *q_fold_axis, bq, q_width), q.dtype),
      vmem((num_stages, *q_fold_axis, bq, do_width), q.dtype),
      vmem((num_stages, *lse_fold_axis, 1, bq), jnp.float32),
      vmem((num_stages, *lse_fold_axis, 1, bq), jnp.float32),
      vmem((num_stages, *kv_fold_axis, bkv, k_width), q.dtype),
      vmem((num_stages, *kv_fold_axis, bkv, v_width), q.dtype),
      vmem((num_stages, *q_fold_axis, bq, q_width), q.dtype),
      vmem((num_stages, *kv_fold_axis, bkv, k_width), q.dtype),
      vmem((num_stages, *kv_fold_axis, bkv, v_width), q.dtype),
      vmem((*q_fold_axis, q_seq_len, q_width), jnp.float32),
      vmem((*kv_fold_axis, bkv, k_width), jnp.float32),
      vmem((*kv_fold_axis, bkv, v_width), jnp.float32),
      pltpu.SemaphoreType.DMA((NUM_SEMS_BWD, num_stages)),
  ]
  hbm_operands = [q, k, v, do, lse_log2, delta]

  with jax.named_scope(kernel_name):
    return pl.pallas_call(
        kernel,
        in_specs=[any_spec] * len(hbm_operands),
        out_specs=[any_spec, any_spec, any_spec],
        out_shape=bwd_out_shape(
            token_major, q.dtype, num_heads, num_kv_heads, q_seq_len,
            kv_seq_len, head_dim_qk, head_dim_v),
        scratch_shapes=scratch_shapes,
        name=kernel_name,
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=VMEM_LIMIT_BYTES),
        interpret=pltpu.InterpretParams() if interpret else False,
    )(*hbm_operands)
