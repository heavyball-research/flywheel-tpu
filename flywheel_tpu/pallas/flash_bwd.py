"""Backward kernels and the flash_attn_bwd entry point."""

import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ..tuned_block_sizes import get_tuned_config
from .block_sizes import (
  DEFAULT_MASK_VALUE,
  LOG2E,
  MIN_NUM_STAGES,
  NN_DIM_NUMBERS,
  NT_DIM_NUMBERS,
  NUM_LANES,
  NUM_SUBLANES,
  PAIRED_HEAD_DIM,
  TokenMajorInfo,
  pick_tile,
  round_up,
  vmem_limit_bytes,
)
from .bwd_pipeline import (
  DQ_DIM_NUMBERS,
  NUM_SEMS_BWD,
  SEM_DI,
  SEM_DO,
  SEM_K,
  SEM_LSE,
  SEM_ODK,
  SEM_ODQ,
  SEM_ODV,
  SEM_Q,
  SEM_V,
  bwd_body,
  bwd_out_shape,
  call_bwd,
  default_bwd_block,
)
from .copy_utils import advance, drain, hbm_tm_slice
from .loop_schedule import (
  DenseBwdParams,
  dense_bwd_params,
  make_dense_bwd_schedule,
)
from .mask import apply_causal_kv_major

# Note (david): dq_acc is the only scratch that grows with q_seq_len; capping it
# at half the VMEM budget leaves the other half to the block-sized stage
# buffers and the live score tiles. None is half of vmem_limit_bytes(); tests
# set a byte count to force q-row chunking.
DQ_ACC_VMEM_LIMIT_BYTES: int | None = None

# Note (david): every general-kernel scratch scales with head_fold; this
# whole-scratch cap stays below the scoped limit to leave room for the
# pipeline's live temporaries: 80 MiB of v6e's 100 MiB.
FOLD_VMEM_LIMIT_FRACTION = 0.8

# Note (david): the head-fold gain saturates by about 8 heads. The single-block
# kernel halves its target past FULL_FOLD_MAX_BLOCK so the fold-scaled buffers
# still fit VMEM.
MAX_SINGLE_BLOCK_HEAD_FOLD = 8
FULL_FOLD_MAX_BLOCK = 512

# Note (david): a flat min(bkv, 256) breaks a bkv like 384, so the kv compute
# tile is the largest of these dividing bkv; bkv % NUM_SUBLANES == 0 makes the
# search always hit.
BKV_COMPUTE_CANDIDATES = (256, 128, 64, 32, 16, 8)


def fold_vmem_bytes(
    head_fold: int,
    kv_fold: int,
    q_seq_len: int,
    bq: int,
    bkv: int,
    head_dim_qk: int,
    head_dim_v: int,
    num_stages: int,
    itemsize: int,
) -> tuple[int, int]:
  """(dq_acc bytes, total scratch bytes) of a general-kernel build at a fold.

  The staged terms over-count on purpose: this is a fail-fast gate, not a
  memory model. dq_acc comes back separately because it is the one scratch that
  grows with q_seq_len, so it carries its own tighter limit.
  """
  # Note (david): VMEM pads every minor head_dim axis to a whole lane tile, so a
  # sub-tile head_dim costs its padded width, which the slack does not cover.
  padded_head_dim_qk = round_up(head_dim_qk, NUM_LANES)
  padded_head_dim_v = round_up(head_dim_v, NUM_LANES)
  f32_bytes = jnp.dtype(jnp.float32).itemsize
  staged_q_bytes = num_stages * head_fold * itemsize * (
      2 * bq * (padded_head_dim_qk + padded_head_dim_v)
      + bq * padded_head_dim_qk)
  staged_kv_bytes = num_stages * kv_fold * itemsize * (
      3 * bkv * (padded_head_dim_qk + padded_head_dim_v))
  dq_acc_bytes = q_seq_len * head_fold * padded_head_dim_qk * f32_bytes
  dkv_acc_bytes = (
      kv_fold * bkv * (padded_head_dim_qk + padded_head_dim_v) * f32_bytes)
  return dq_acc_bytes, (
      staged_q_bytes + staged_kv_bytes + dq_acc_bytes + dkv_acc_bytes)


def flash_bwd_kernel(
    *refs: jax.Array, dense_params: DenseBwdParams, **body) -> None:
  bwd_body(make_dense_bwd_schedule(dense_params), refs, **body)


def flash_bwd_folded_kernel(
    q_hbm: jax.Array,
    k_hbm: jax.Array,
    v_hbm: jax.Array,
    do_hbm: jax.Array,
    lse_hbm: jax.Array,
    di_hbm: jax.Array,
    dq_hbm: jax.Array,
    dk_hbm: jax.Array,
    dv_hbm: jax.Array,
    q_buf: jax.Array,
    do_buf: jax.Array,
    lse_buf: jax.Array,
    di_buf: jax.Array,
    k_buf: jax.Array,
    v_buf: jax.Array,
    dq_stage: jax.Array,
    dk_stage: jax.Array,
    dv_stage: jax.Array,
    sems: jax.Array,
    *,
    num_head_groups: int,
    head_fold: int,
    seq_len: int,
    num_stages: int,
    scale: float,
    causal: bool,
    out_dtype: jnp.dtype,
    token_major: TokenMajorInfo | None,
) -> None:
  """Single-block (block == seq_len) backward over head_fold heads per step.

  Batching heads amortizes the per-step DMA and loop overhead that dominates
  many-head short-sequence builds. With one kv block dq needs no reduction and
  is written straight out.
  """
  f32 = jnp.float32
  # Note (david): scores stay in log2 units so the softmax recompute can use
  # exp2; lse arrives pre-scaled by LOG2E to match.
  score_scale = scale * LOG2E
  if token_major is None:
    lane_width_qk = None
    lane_width_v = None
  else:
    lane_width_qk = token_major.head_dim_qk
    lane_width_v = token_major.head_dim_v

  def _group_window(hbm, lane_width, group):
    if token_major is None:
      return hbm.at[pl.ds(group * head_fold, head_fold)]
    else:
      return hbm_tm_slice(
          hbm, group * head_fold, 0, seq_len, token_major.num_q_heads,
          lane_width, token_major.batch is not None, head_fold)

  input_streams = (
      (q_hbm, q_buf, SEM_Q, lane_width_qk),
      (do_hbm, do_buf, SEM_DO, lane_width_v),
      (k_hbm, k_buf, SEM_K, lane_width_qk),
      (v_hbm, v_buf, SEM_V, lane_width_v),
  )
  output_streams = (
      (dq_hbm, dq_stage, SEM_ODQ, lane_width_qk),
      (dk_hbm, dk_stage, SEM_ODK, lane_width_qk),
      (dv_hbm, dv_stage, SEM_ODV, lane_width_v),
  )

  def _input_copies(group, slot):
    copies = [
        pltpu.make_async_copy(
            _group_window(hbm, lane_width, group), buf.at[slot],
            sems.at[sem, slot])
        for hbm, buf, sem, lane_width in input_streams]
    # Note (david): lse/di stay head-major (heads, 1, seq) under both layouts,
    # so their fold always rides the leading axis.
    for hbm, buf, sem in (
        (lse_hbm, lse_buf, SEM_LSE), (di_hbm, di_buf, SEM_DI)):
      copies.append(pltpu.make_async_copy(
          hbm.at[pl.ds(group * head_fold, head_fold)], buf.at[slot],
          sems.at[sem, slot]))
    return copies

  def _output_copies(group, slot):
    return [
        pltpu.make_async_copy(
            stage.at[slot], _group_window(hbm, lane_width, group),
            sems.at[sem, slot])
        for hbm, stage, sem, lane_width in output_streams]

  def _start_all(copies):
    for copy in copies:
      copy.start()

  def _wait_all(copies):
    for copy in copies:
      copy.wait()

  def _head_index(slot, head_idx, lane_width):
    if token_major is None:
      return (slot, head_idx)
    else:
      return (slot, slice(None), pl.ds(head_idx * lane_width, lane_width))

  def _group_step(group, carry):
    loaded_key, inflight_key, in_slot, num_out = carry
    next_group = jnp.minimum(group + 1, num_head_groups - 1)
    loaded_key, inflight_key, in_slot = advance(
        group, next_group,
        lambda slot: _start_all(_input_copies(group, slot)),
        lambda slot: _start_all(_input_copies(next_group, slot)),
        lambda slot: _wait_all(_input_copies(0, slot)),
        loaded_key, inflight_key, in_slot, num_stages)

    out_slot = num_out % num_stages

    @pl.when(num_out >= num_stages)
    def _():
      _wait_all(_output_copies(0, out_slot))

    # Note (david): one head per iteration keeps the live score tile a single
    # head's (seq_len, seq_len), so VMEM scales with head_fold only through the
    # staged buffers. Mosaic already overlaps the independent heads' matmuls;
    # explicit cross-head pipelining measured identical.
    for head_idx in range(head_fold):
      q_head = q_buf[_head_index(in_slot, head_idx, lane_width_qk)]
      do_head = do_buf[_head_index(in_slot, head_idx, lane_width_v)]
      k_head = k_buf[_head_index(in_slot, head_idx, lane_width_qk)]
      v_head = v_buf[_head_index(in_slot, head_idx, lane_width_v)]
      lse_row = lse_buf[in_slot, head_idx]
      di_row = di_buf[in_slot, head_idx]
      qk = score_scale * lax.dot_general(
          k_head, q_head, NT_DIM_NUMBERS, preferred_element_type=f32)
      if causal:
        masked_qk = apply_causal_kv_major(qk, DEFAULT_MASK_VALUE, 0, 0)
      else:
        masked_qk = qk
      p = jnp.exp2(masked_qk - lse_row)
      dp = lax.dot_general(
          v_head, do_head, NT_DIM_NUMBERS, preferred_element_type=f32)
      ds = (dp - di_row) * p
      dk_head = scale * lax.dot_general(
          ds, q_head, NN_DIM_NUMBERS, preferred_element_type=f32)
      dv_head = lax.dot_general(
          p, do_head, NN_DIM_NUMBERS, preferred_element_type=f32)
      dq_head = scale * lax.dot_general(
          ds, k_head, DQ_DIM_NUMBERS, preferred_element_type=f32)
      dq_stage[_head_index(out_slot, head_idx, lane_width_qk)] = (
          dq_head.astype(out_dtype))
      dk_stage[_head_index(out_slot, head_idx, lane_width_qk)] = (
          dk_head.astype(out_dtype))
      dv_stage[_head_index(out_slot, head_idx, lane_width_v)] = (
          dv_head.astype(out_dtype))

    _start_all(_output_copies(group, out_slot))
    return loaded_key, inflight_key, in_slot, num_out + 1

  *_, num_out = lax.fori_loop(0, num_head_groups, _group_step, (-1, -1, 0, 0))

  drain(num_out, num_stages, lambda slot: _wait_all(_output_copies(0, slot)))


def flash_attn_bwd_folded(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    do: jax.Array,
    lse_log2: jax.Array,
    delta: jax.Array,
    *,
    num_heads: int,
    q_seq_len: int,
    head_dim_qk: int,
    head_dim_v: int,
    head_fold: int,
    scale: float,
    causal: bool,
    interpret: bool,
    num_stages: int,
    token_major: TokenMajorInfo | None,
) -> list[jax.Array]:
  """Launch flash_bwd_folded_kernel with one head group of scratch per stage."""
  vmem = pltpu.VMEM
  any_spec = pl.BlockSpec(memory_space=pl.ANY)

  # Note (david): head-major folds on a leading axis and token-major on the
  # lanes, matching the single group-wide DMA window each layout issues per
  # stream.
  if token_major is None:
    q_group_shape = (num_stages, head_fold, q_seq_len, head_dim_qk)
    v_group_shape = (num_stages, head_fold, q_seq_len, head_dim_v)
  else:
    q_group_shape = (num_stages, q_seq_len, head_fold * head_dim_qk)
    v_group_shape = (num_stages, q_seq_len, head_fold * head_dim_v)
  lse_group_shape = (num_stages, head_fold, 1, q_seq_len)
  # Note (david): the order must match flash_bwd_folded_kernel's scratch
  # parameters.
  scratch_shapes = [
      vmem(q_group_shape, q.dtype),
      vmem(v_group_shape, q.dtype),
      vmem(lse_group_shape, jnp.float32),
      vmem(lse_group_shape, jnp.float32),
      vmem(q_group_shape, q.dtype),
      vmem(v_group_shape, q.dtype),
      vmem(q_group_shape, q.dtype),
      vmem(q_group_shape, q.dtype),
      vmem(v_group_shape, q.dtype),
      pltpu.SemaphoreType.DMA((NUM_SEMS_BWD, num_stages)),
  ]
  kernel = functools.partial(
      flash_bwd_folded_kernel, num_head_groups=num_heads // head_fold,
      head_fold=head_fold, seq_len=q_seq_len, scale=scale, causal=causal,
      out_dtype=q.dtype, num_stages=num_stages, token_major=token_major)
  operands = (q, k, v, do, lse_log2, delta)
  with jax.named_scope("flash_attn_bwd_folded"):
    return pl.pallas_call(
        kernel,
        in_specs=[any_spec] * len(operands),
        out_specs=[any_spec, any_spec, any_spec],
        out_shape=bwd_out_shape(
            token_major, q.dtype, num_heads, num_heads, q_seq_len, q_seq_len,
            head_dim_qk, head_dim_v),
        scratch_shapes=scratch_shapes,
        name="flash_attn_bwd_folded",
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_limit_bytes()),
        interpret=pltpu.InterpretParams() if interpret else False,
    )(*operands)


def flash_attn_bwd_q_chunked(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    o: jax.Array,
    lse: jax.Array,
    do: jax.Array,
    *,
    chunk_rows: int,
    block_kv: int,
    causal: bool,
    causal_offset: int,
    **knobs: int | float | bool | None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Dense backward as one rectangular backward per q-row chunk -> (dq, dk, dv).

  dq rows are disjoint, so the chunks concatenate; dk/dv sum in f32, so each
  chunk's own out_dtype rounding is the only one added. A causal chunk drops
  the kv blocks past its last row, exactly the ones the unchunked schedule
  skips for those rows. A full-mask chunk keeps causal_offset, so equal-sized
  chunks share one compiled kernel.
  """
  q_seq_len, kv_seq_len = q.shape[-2], k.shape[-2]
  f32 = jnp.float32
  dq_chunks = []
  dk = jnp.zeros(k.shape, f32)
  dv = jnp.zeros(v.shape, f32)
  for start in range(0, q_seq_len, chunk_rows):
    stop = min(start + chunk_rows, q_seq_len)
    if causal and stop + causal_offset <= 0:
      # Note (david): every row sits below the diagonal, so dq is exactly 0 and
      # dk/dv get nothing; the dense schedule would reject the empty chunk.
      dq_chunks.append(jnp.zeros_like(q[..., start:stop, :]))
      continue
    if causal:
      kv_stop = min(kv_seq_len, round_up(stop + causal_offset, block_kv))
      chunk_causal_offset = causal_offset + start
    else:
      kv_stop = kv_seq_len
      chunk_causal_offset = causal_offset
    chunk_dq, chunk_dk, chunk_dv = flash_attn_bwd(
        q[..., start:stop, :], k[..., :kv_stop, :], v[..., :kv_stop, :],
        o[..., start:stop, :], lse[:, start:stop], do[..., start:stop, :],
        causal=causal, causal_offset=chunk_causal_offset, block_kv=block_kv,
        **knobs)
    dq_chunks.append(chunk_dq)
    dk = dk.at[..., :kv_stop, :].add(chunk_dk.astype(f32))
    dv = dv.at[..., :kv_stop, :].add(chunk_dv.astype(f32))
  return (jnp.concatenate(dq_chunks, axis=-2), dk.astype(q.dtype),
          dv.astype(q.dtype))


def flash_attn_bwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    o: jax.Array,
    lse: jax.Array,
    do: jax.Array,
    *,
    causal: bool = False,
    causal_offset: int = 0,
    softmax_scale: float | None = None,
    block: int | None = None,
    block_kv: int | None = None,
    block_q_compute: int | None = None,
    block_kv_compute: int | None = None,
    head_fold: int | None = None,
    num_stages: int | None = None,
    interpret: bool = False,
    token_major: bool = False,
    head_dim: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Splash-attention backward -> (dq, dk, dv), laid out like q, k and v.

  Head-major q/o/do are (num_heads, q_seq_len, head_dim) and k/v are
  (num_kv_heads, kv_seq_len, head_dim), with num_kv_heads dividing num_heads
  (MHA or GQA/MQA). token_major=True takes (batch, seq, heads * head_dim) refs
  with a static head_dim that is a multiple of 128, or 64 (paired, MHA-only).
  lse is (num_heads, q_seq_len) under both layouts: the natural-log logsumexp
  of the scaled logits that the forward saved.

  The causal mask keeps q_idx + causal_offset >= kv_idx.

  head_fold heads share one schedule step: an explicit MHA fold runs the
  single-block kernel (square, zero offset, block == seq_len). GQA/MQA always
  folds each kv head's q heads and rejects an explicit head_fold. Leaving
  block, block_kv, block_q_compute, block_kv_compute, head_fold and num_stages
  all None lets the tuned table set them together. A build whose f32 dq
  accumulator overflows VMEM runs as q-row chunks.
  """
  if num_stages is not None and (
      type(num_stages) is not int or num_stages < MIN_NUM_STAGES):
    raise ValueError(
        f"num_stages must be at least {MIN_NUM_STAGES}; got {num_stages!r}.")
  if token_major:
    if (head_dim is None or type(head_dim) is not int
        or (head_dim % NUM_LANES and head_dim != PAIRED_HEAD_DIM)):
      raise ValueError(
          f"token-major flash_attn_bwd requires a static head_dim with"
          f" head_dim % 128 == 0, or head_dim == 64 (paired); got"
          f" {head_dim!r}."
      )
    # Note (david): k/v carry the kv sequence and o/do the q one, so across all
    # five only the rank is shared; a rectangular backward differs in length.
    if any(operand.ndim != 3 for operand in (q, k, v, o, do)):
      raise ValueError(
          f"token-major q/k/v/o/do must all be (batch, seq, heads * head_dim);"
          f" got {q.shape=}, {k.shape=}, {v.shape=}, {o.shape=}, {do.shape=}."
      )
    batch = q.shape[0]
    if k.shape[0] != batch:
      raise ValueError(
          f"token-major k must lead with q's batch {batch}; got"
          f" {k.shape=}."
      )
    if v.shape[:-1] != k.shape[:-1]:
      raise ValueError(
          f"token-major v must share k's leading dims {k.shape[:-1]} (both"
          f" carry the KV sequence); got {v.shape=}."
      )
    if q.shape[-1] % head_dim or k.shape[-1] % head_dim:
      raise ValueError(
          f"head_dim {head_dim} must divide q and k fused widths; got"
          f" {q.shape=}, {k.shape=}.")
    heads_per_row = q.shape[-1] // head_dim
    kv_heads_per_row = k.shape[-1] // head_dim
    if heads_per_row % kv_heads_per_row:
      raise ValueError(
          f"k's fused width implies {kv_heads_per_row} kv heads, which must"
          f" divide q's {heads_per_row} (MHA or GQA/MQA); got {q.shape=},"
          f" {k.shape=}.")
    head_dim_qk = head_dim
    head_dim_v = v.shape[-1] // kv_heads_per_row
    if (v.shape[-1] % kv_heads_per_row
        or (head_dim_v % NUM_LANES and head_dim_v != PAIRED_HEAD_DIM)):
      raise ValueError(
          f"v fused width {v.shape[-1]} must split into {kv_heads_per_row}"
          f" heads of a 128-multiple (or 64, paired) head_dim.")
    if head_dim_qk % NUM_LANES or head_dim_v % NUM_LANES:
      # Note (david): a d64 fold must span whole lane tiles on both the q and kv
      # streams, which the unfolded kv side of GQA/MQA cannot; MLA cannot fold
      # at all, and an odd per-row head count leaves no pair to fold.
      if head_dim_v != head_dim_qk:
        raise NotImplementedError(
            f"token-major d64 requires head_dim_v == head_dim (no MLA); got"
            f" {head_dim_qk} vs {head_dim_v}.")
      if kv_heads_per_row != heads_per_row:
        raise NotImplementedError(
            "token-major d64 is MHA-only (GQA/MQA's kv stream cannot be"
            " paired); pass head-major inputs instead.")
      if heads_per_row % 2:
        raise ValueError(
            f"token-major d64 requires an even per-row head count; got"
            f" {heads_per_row}.")
    # Note (david): o holds one vector per q head whatever the kv head count,
    # so its fused width is heads_per_row * head_dim_v, which matches v's only
    # under MHA.
    expected_o_shape = (*q.shape[:-1], heads_per_row * head_dim_v)
    if o.shape != expected_o_shape or do.shape != o.shape:
      raise ValueError(
          f"token-major o must be {expected_o_shape} (q's leading dims -- it"
          f" carries the Q sequence -- and q's per-row head count times"
          f" head_dim_v) and do must match o; got {o.shape=}, {do.shape=}."
      )
    q_seq_len = q.shape[-2]
    kv_seq_len = k.shape[-2]
    num_heads = batch * heads_per_row
    num_kv_heads = batch * kv_heads_per_row
    token_major_info = TokenMajorInfo(
        batch=batch, num_q_heads=heads_per_row,
        num_kv_heads=kv_heads_per_row, head_dim_qk=head_dim_qk,
        head_dim_v=head_dim_v)
  else:
    token_major_info = None
    if head_dim is not None:
      raise ValueError(
          "head_dim is a token-major-only parameter for flash_attn_bwd."
      )
    if any(operand.ndim != 3 for operand in (q, k, v, o, do)):
      raise ValueError(
          "q/k/v/o/do must all be head-major (num_heads, seq_len, head_dim);"
          f" got q {q.shape}, k {k.shape}, v {v.shape}, o {o.shape},"
          f" do {do.shape}.")
    num_heads, q_seq_len, head_dim_qk = q.shape
    num_kv_heads, kv_seq_len = k.shape[0], k.shape[1]
    head_dim_v = v.shape[-1]
    if k.shape[-1] != head_dim_qk or num_heads % num_kv_heads:
      raise ValueError(
          f"k {k.shape} must carry q's head_dim {head_dim_qk} and a head"
          f" count dividing q's {num_heads} (MHA or GQA/MQA); got q"
          f" {q.shape}.")
    expected_v_shape = (num_kv_heads, kv_seq_len, head_dim_v)
    expected_o_shape = (num_heads, q_seq_len, head_dim_v)
    if v.shape != expected_v_shape:
      raise ValueError(f"v {v.shape} must be {expected_v_shape}.")
    if o.shape != expected_o_shape:
      raise ValueError(f"o {o.shape} must be {expected_o_shape}.")
    if do.shape != o.shape:
      raise ValueError(f"do {do.shape} must match o {o.shape}.")
  if lse.shape != (num_heads, q_seq_len):
    raise ValueError(f"lse {lse.shape} must be (num_heads, q_seq_len).")

  is_gqa = num_kv_heads != num_heads
  if is_gqa and head_fold is not None:
    raise ValueError(
        f"head_fold is MHA-only: a GQA/MQA backward already folds each kv"
        f" head's {num_heads // num_kv_heads} q heads into one group; got"
        f" {head_fold=}.")

  if softmax_scale is None:
    scale = 1.0 / math.sqrt(head_dim_qk)
  else:
    scale = softmax_scale

  is_rectangular = q_seq_len != kv_seq_len
  # Note (david): only a caller-chosen fold raises when illegal; the tuned
  # table's and this function's own picks get clamped instead.
  is_explicit_head_fold = head_fold is not None
  # Note (david): the block knobs, head_fold and num_stages are co-tuned
  # (head_fold > 1 needs block == seq_len, block_kv_compute must divide block),
  # so the table applies only when all of them are unset. It is keyed on
  # square, zero-offset MHA shapes.
  is_tunable = (
      block is None and block_kv is None and block_q_compute is None
      and block_kv_compute is None
      and head_fold is None and num_stages is None
      and not is_rectangular and causal_offset == 0 and not is_gqa
  )
  if is_tunable:
    tuned_config = get_tuned_config(
        "bwd",
        token_major=token_major,
        num_heads=token_major_info.num_q_heads if token_major else num_heads,
        batch=token_major_info.batch if token_major else None,
        seq_len=q_seq_len,
        head_dim=head_dim_qk,
        head_dim_v=head_dim_v,
        causal=causal,
        max_seqlen=None,
        return_lse=False,
    )
  else:
    tuned_config = None
  if tuned_config is None:
    num_stages = MIN_NUM_STAGES if num_stages is None else num_stages
  else:
    (
        block,
        block_kv,
        block_q_compute,
        block_kv_compute,
        num_stages,
        head_fold,
        qkv_layout_name,
        is_transposed_pv,
    ) = tuned_config
    assert qkv_layout_name == "head_dim_minor" and not is_transposed_pv, (
        tuned_config)

  bq = default_bwd_block(q_seq_len) if block is None else block
  # Note (david): lse/di put seq on the lane axis, so a per-block slice at
  # qi * bq is only tile-aligned when bq is a multiple of NUM_LANES.
  if bq % NUM_LANES:
    raise ValueError(f"block {bq} must be a multiple of {NUM_LANES}.")
  if q_seq_len % bq:
    raise ValueError(f"block {bq} must divide q_seq_len {q_seq_len}.")
  num_q_blocks = q_seq_len // bq

  bq_compute = bq if block_q_compute is None else block_q_compute
  if bq_compute % NUM_LANES or bq % bq_compute:
    raise ValueError(
        f"block_q_compute {bq_compute} must be a multiple of {NUM_LANES}"
        f" and divide block {bq}.")

  bkv = default_bwd_block(kv_seq_len) if block_kv is None else block_kv
  if bkv % NUM_SUBLANES:
    raise ValueError(f"block_kv {bkv} must be a multiple of {NUM_SUBLANES}.")
  if kv_seq_len % bkv:
    raise ValueError(f"block_kv {bkv} must divide kv_seq_len {kv_seq_len}.")
  num_kv_blocks = kv_seq_len // bkv

  # Note (david): the general kernel folds only a GQA/MQA group or a d64 lane
  # pair; any other fold goes to the single-block kernel.
  if is_gqa:
    # Note (david): a GQA/MQA schedule group is one kv head plus its q heads,
    # so the q-side fold is the group itself.
    resolved_head_fold = num_heads // num_kv_heads
  elif head_fold is not None:
    resolved_head_fold = head_fold
  elif (is_rectangular or causal_offset != 0 or num_q_blocks != 1
        or head_dim_v != head_dim_qk):
    # Note (david): the single-block kernel is square, zero-offset and
    # one-block only, and its fold was never modelled for MLA's wider do/v/dv.
    resolved_head_fold = 1
  else:
    # Note (david): the fold-scaled buffers grow with the block and head_dim,
    # so the target halves for large blocks and shrinks per extra lane tile.
    if bq <= FULL_FOLD_MAX_BLOCK:
      block_fold_target = MAX_SINGLE_BLOCK_HEAD_FOLD
    else:
      block_fold_target = MAX_SINGLE_BLOCK_HEAD_FOLD // 2
    fold_target = max(
        1, block_fold_target // max(1, head_dim_qk // NUM_LANES))
    resolved_head_fold = 1
    while (resolved_head_fold * 2 <= min(fold_target, num_heads)
           and num_heads % (resolved_head_fold * 2) == 0):
      resolved_head_fold *= 2

  # Note (david): a fold group is head_fold consecutive flat heads, which stays
  # inside one token-major batch row only if head_fold also divides the per-row
  # head count. Automatic picks are keyed on the flat count, so they halve to
  # the largest legal group instead of raising.
  if (is_explicit_head_fold and token_major and resolved_head_fold > 1
      and token_major_info.num_q_heads % resolved_head_fold):
    raise ValueError(
        f"token-major head_fold {resolved_head_fold} must also divide the"
        f" per-row head count {token_major_info.num_q_heads} (a fold group"
        " cannot cross a batch row boundary)."
    )
  while (token_major and resolved_head_fold > 1
         and token_major_info.num_q_heads % resolved_head_fold):
    resolved_head_fold //= 2

  # Note (david): on d64 the fold is the DMA-alignment mechanism (a group must
  # span whole lane tiles), so it must be even, even where folding does not
  # pay; 2 always divides the per-row head count, which was checked to be even.
  is_lane_paired = token_major and head_dim_qk % NUM_LANES != 0
  if is_lane_paired and resolved_head_fold % 2:
    if is_explicit_head_fold:
      raise ValueError(
          f"token-major d64 requires an even head_fold (a DMA group must"
          f" span whole 128-lane tiles); got {resolved_head_fold}.")
    head_fold = 2
  else:
    head_fold = resolved_head_fold

  # Note (david): under GQA/MQA the group's kv side is one physical head, so
  # only the q side folds.
  kv_fold = 1 if is_gqa else head_fold

  bkv_compute = (pick_tile(BKV_COMPUTE_CANDIDATES, bkv)
                 if block_kv_compute is None else block_kv_compute)
  if bkv_compute % NUM_SUBLANES or bkv % bkv_compute:
    raise ValueError(
        f"block_kv_compute {bkv_compute} must be a multiple of {NUM_SUBLANES}"
        f" and divide block_kv {bkv}.")

  dense_params = dense_bwd_params(
      num_heads=num_heads // head_fold, num_q_blocks=num_q_blocks,
      num_kv_blocks=num_kv_blocks, bq=bq, bkv=bkv, causal=causal,
      offset=causal_offset)

  dq_acc_bytes, scratch_bytes = fold_vmem_bytes(
      head_fold, kv_fold, q_seq_len=q_seq_len, bq=bq, bkv=bkv,
      head_dim_qk=head_dim_qk, head_dim_v=head_dim_v, num_stages=num_stages,
      itemsize=q.dtype.itemsize)
  dq_acc_limit = DQ_ACC_VMEM_LIMIT_BYTES or vmem_limit_bytes() // 2
  fold_limit = int(vmem_limit_bytes() * FOLD_VMEM_LIMIT_FRACTION)
  # Note (david): the general kernel keeps the whole group's dq in VMEM, so a
  # build past that gate runs as q-row chunks that each fit. Chunks are
  # equalized rather than filled to the gate, so a full-mask build compiles one
  # kernel, not two.
  chunk_blocks = dq_acc_limit // (dq_acc_bytes // num_q_blocks)
  should_chunk = (
      (head_fold == 1 or is_gqa or is_lane_paired)
      and dq_acc_bytes > dq_acc_limit and chunk_blocks > 0)
  is_single_block_fold = head_fold > 1 and not (is_gqa or is_lane_paired)

  def _lse_log2_and_delta():
    if token_major:
      # Note (david): one lane window per head, never a reshape of the fused
      # axis: splitting heads * head_dim moves the (8, 128)-tiled sublane axis
      # from seq to heads, so XLA re-tiles and materializes o * do in f32 (three
      # HBM passes, about 370us per backward at 16k tokens on v6e). Per-head
      # accumulation can differ from head-major's by 1 f32 ULP.
      per_head_delta = []
      for head_idx in range(token_major_info.num_q_heads):
        lane_start = head_idx * head_dim_v
        lane_stop = lane_start + head_dim_v
        o_head = lax.slice_in_dim(o, lane_start, lane_stop, axis=-1)
        do_head = lax.slice_in_dim(do, lane_start, lane_stop, axis=-1)
        per_head_delta.append(jnp.einsum(
            "bsd,bsd->bs", o_head.astype(jnp.float32),
            do_head.astype(jnp.float32)))
      # Note (david): stacking at -2 puts heads next to seq in the kernel's
      # flat batch-major, head-minor order, so no transpose follows.
      delta = jnp.stack(per_head_delta, axis=-2).reshape(num_heads, q_seq_len)
    else:
      delta = jnp.einsum("hsd,hsd->hs", o.astype(jnp.float32),
                         do.astype(jnp.float32))
    # Note (david): lse is pre-scaled into log2 units once so both kernels
    # recompute p with exp2 and no per-element multiply; delta stays in nats
    # since it never enters the exp.
    lse_log2 = (lse.astype(jnp.float32) * LOG2E)[:, None, :]
    return lse_log2, delta[:, None, :]

  if should_chunk:
    num_chunks = -(-num_q_blocks // chunk_blocks)
    return flash_attn_bwd_q_chunked(
        q, k, v, o, lse, do,
        chunk_rows=-(-num_q_blocks // num_chunks) * bq,
        causal=causal, causal_offset=causal_offset,
        softmax_scale=scale, block=bq, block_kv=bkv,
        block_q_compute=bq_compute, block_kv_compute=bkv_compute,
        head_fold=None if is_gqa else head_fold, num_stages=num_stages,
        interpret=interpret, token_major=token_major, head_dim=head_dim)
  elif is_single_block_fold:
    if bq_compute != bq:
      raise ValueError(
          "head_fold > 1 does not support split Q compute tiles;"
          f" block_q_compute must equal block ({bq}), got {bq_compute}.")
    # Note (david): automatic picks never fold a rectangular, offset or MLA
    # build, so those raises only fire for an explicit head_fold.
    if is_rectangular:
      raise ValueError(
          f"head_fold > 1 requires q_seq_len == kv_seq_len (square); got"
          f" {q_seq_len} vs {kv_seq_len}.")
    if causal_offset != 0:
      raise ValueError(
          f"head_fold > 1 requires causal_offset == 0 (square, zero-offset"
          f" diagonal); got {causal_offset}.")
    if num_q_blocks != 1:
      raise ValueError(
          f"head_fold > 1 requires a single block (block == seq_len); got"
          f" block {bq}, seq_len {q_seq_len}.")
    if num_heads % head_fold:
      raise ValueError(
          f"head_fold {head_fold} must divide num_heads {num_heads}.")
    # Note (david): the folded do/v/dv buffers scale with head_dim_v, so a
    # fold sized for head_dim_qk overshoots VMEM (16 heads at seq 512 with
    # head_dim_qk 256 and head_dim_v 512 want 80MB of the 64MB scoped budget).
    if head_dim_v != head_dim_qk:
      raise ValueError(
          f"head_fold > 1 requires head_dim_v == head_dim_qk; got"
          f" {head_dim_v} vs {head_dim_qk}.")
    lse_log2, delta = _lse_log2_and_delta()
    return flash_attn_bwd_folded(
        q, k, v, do, lse_log2, delta, num_heads=num_heads,
        q_seq_len=q_seq_len, head_dim_qk=head_dim_qk, head_dim_v=head_dim_v,
        head_fold=head_fold, scale=float(scale),
        causal=causal, interpret=interpret,
        num_stages=num_stages, token_major=token_major_info)
  else:
    if num_heads % head_fold:
      raise ValueError(
          f"head_fold {head_fold} must divide num_heads {num_heads}.")
    if head_fold > 1 and scratch_bytes > fold_limit:
      raise ValueError(
          f"head_fold {head_fold} needs ~{scratch_bytes} bytes of scratch VMEM"
          f" at block {bq}, block_kv {bkv}, num_stages {num_stages}, above the"
          f" {fold_limit} byte limit; use a smaller head_fold or"
          " smaller blocks.")
    if dq_acc_bytes > dq_acc_limit:
      raise ValueError(
          f"dq accumulator needs seq_len {q_seq_len} * head_fold {head_fold} *"
          f" head_dim {head_dim_qk} * 4 = {dq_acc_bytes} bytes of VMEM, above"
          f" the {dq_acc_limit} byte limit and not one q block's dq"
          f" fits; lower block {bq}, or shard the sequence across chips.")
    lse_log2, delta = _lse_log2_and_delta()
    # Note (david): the general kernel runs unfolded for MHA, one kv head's q
    # group for GQA/MQA, or an even DMA-alignment fold for d64 token-major, so
    # dense_params' group count is num_heads // head_fold.
    assert head_fold == 1 or is_gqa or is_lane_paired, (
        head_fold, num_kv_heads)
    kernel = functools.partial(
        flash_bwd_kernel, dense_params=dense_params,
        bq=bq, bkv=bkv, bq_compute=bq_compute, bkv_compute=bkv_compute,
        num_stages=num_stages, scale=float(scale),
        causal=causal, causal_offset=causal_offset, out_dtype=q.dtype,
        head_fold=head_fold, kv_fold=kv_fold, token_major=token_major_info,
    )
    dq, dk, dv = call_bwd(
        kernel, q, k, v, do, lse_log2, delta,
        bq=bq, bkv=bkv, head_fold=head_fold, kv_fold=kv_fold,
        num_stages=num_stages,
        num_heads=num_heads, num_kv_heads=num_kv_heads, q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        head_dim_qk=head_dim_qk, head_dim_v=head_dim_v, interpret=interpret,
        kernel_name="flash_attn_bwd", token_major=token_major_info,
    )
    if causal and causal_offset < 0:
      # Note (david): q blocks wholly below the diagonal never enter the
      # schedule, so the kernel never writes their dq; zero it rather than
      # trust the output buffer.
      is_valid_row = jnp.arange(q_seq_len) >= -causal_offset
      return jnp.where(is_valid_row[None, :, None], dq, 0), dk, dv
    else:
      return dq, dk, dv
