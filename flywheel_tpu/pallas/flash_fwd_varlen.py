"""Varlen forward kernel driven by a schedule table built from the cu_seqlens.

The per-seq schedule cuts each sequence into its own q blocks, one SMEM table
row each, and synthesizes its masks in-kernel: the sequence bounds plus the
causal or window edges around each sequence's own bottom-right diagonal.
"""

from functools import partial

import jax

from .block_sizes import BlockSizes, QKVLayout, TokenMajorInfo
from .fwd_pipeline import forward_common, fwd_body
from .loop_schedule import (
  ScheduleTableLayout,
  VarlenSchedule,
  make_per_seq_fwd_schedule,
  varlen_schedule_table,
)
from .seqlen_info import check_cu_seqlens_pair


def flash_fwd_varlen_kernel(
    table_ref: jax.Array,
    q_hbm: jax.Array,
    k_hbm: jax.Array,
    v_hbm: jax.Array,
    *refs: jax.Array,
    varlen_sched: VarlenSchedule,
    layout: ScheduleTableLayout,
    num_head_groups: int,
    bq: int,
    bkv: int,
    bkv_compute: int,
    **body,
) -> None:
  schedule = make_per_seq_fwd_schedule(
      table_ref,
      layout=layout,
      num_head_groups=num_head_groups,
      q_heads_per_kv_head=varlen_sched.q_heads_per_kv_head,
      padded_total_k=varlen_sched.padded_total_k,
      bq=bq, bkv=bkv, bkv_compute=bkv_compute,
      left=varlen_sched.left, right=varlen_sched.right,
  )
  fwd_body(schedule, q_hbm, k_hbm, v_hbm, refs, bq=bq, bkv=bkv,
           bkv_compute=bkv_compute, **body)


def flash_attn_forward_varlen(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    cu_seqlens_q: jax.Array,
    cu_seqlens_k: jax.Array,
    *,
    varlen_sched: VarlenSchedule,
    block_sizes: BlockSizes,
    head_fold: int,
    transposed_pv: bool,
    token_major: TokenMajorInfo | None,
    **body,
) -> jax.Array | tuple[jax.Array, jax.Array]:
  if token_major is None:
    num_q_heads = q.shape[0]
  else:
    num_q_heads = (token_major.batch or 1) * token_major.num_q_heads
  check_cu_seqlens_pair(cu_seqlens_k, cu_seqlens_q)
  # Note (david): per-seq blocks address HBM at arbitrary token offsets, which
  # only a second-minor (sublane) token axis supports.
  if transposed_pv or block_sizes.qkv_layout != QKVLayout.HEAD_DIM_MINOR:
    raise ValueError(
        "the per-seq varlen schedule requires a HEAD_DIM_MINOR qkv_layout"
        " and transposed_pv=False (token axis must stay second-minor)."
    )
  # Note (david): padding rows past cu_seqlens_q[-1] own no q block and come
  # back out = 0, lse = -inf, so they cost no attention; kv tokens past
  # cu_seqlens_k[-1] are masked wherever a block loads them. Resolving the
  # schedule once per call spares every q block a search of the cu_seqlens.
  table, layout = varlen_schedule_table(
      cu_seqlens_q, cu_seqlens_k, varlen_sched=varlen_sched,
      bq=block_sizes.block_q, bkv=block_sizes.block_kv,
      bkv_compute=block_sizes.block_kv_compute)

  kernel = partial(
      flash_fwd_varlen_kernel,
      varlen_sched=varlen_sched,
      layout=layout,
      num_head_groups=num_q_heads // head_fold,
  )
  return forward_common(
      kernel, [table], q, k, v,
      block_sizes=block_sizes,
      head_fold=head_fold, transposed_pv=transposed_pv,
      kernel_name="flash_attn_mha_static_t1_guarded_fwd_varlen_per_seq",
      token_major=token_major,
      is_per_seq=True,
      **body,
  )
