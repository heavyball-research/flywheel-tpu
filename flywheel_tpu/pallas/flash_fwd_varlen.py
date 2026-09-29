"""Varlen forward kernel driven by the extended cu_seqlens pair in SMEM.

The per-seq schedule cuts each sequence into its own q blocks and synthesizes
its masks in-kernel: the sequence bounds plus the causal or window edges
around each sequence's own bottom-right diagonal.
"""

from functools import partial

import jax

from .block_sizes import BlockSizes, QKVLayout, TokenMajorInfo
from .fwd_pipeline import forward_common, fwd_body
from .loop_schedule import (
  VarlenSchedule,
  make_per_seq_fwd_schedule,
  per_seq_qblk_prefix,
)
from .seqlen_info import (
  check_cu_seqlens_pair,
  extend_cu_seqlens,
  split_pad_tail,
)


def flash_fwd_varlen_kernel(
    cu_q_ref: jax.Array,
    cu_k_ref: jax.Array,
    cu_qblk_ref: jax.Array,
    q_hbm: jax.Array,
    k_hbm: jax.Array,
    v_hbm: jax.Array,
    *refs: jax.Array,
    varlen_sched: VarlenSchedule,
    num_head_groups: int,
    bq: int,
    bkv: int,
    **body,
) -> None:
  schedule = make_per_seq_fwd_schedule(
      cu_q_ref, cu_k_ref, cu_qblk_ref,
      num_head_groups=num_head_groups,
      q_heads_per_kv_head=varlen_sched.q_heads_per_kv_head,
      padded_total_q=varlen_sched.num_q_blocks * bq,
      padded_total_k=varlen_sched.num_kv_blocks * bkv,
      bq=bq, bkv=bkv, left=varlen_sched.left, right=varlen_sched.right,
      num_rows=cu_qblk_ref[cu_qblk_ref.shape[0] - 1],
  )
  fwd_body(schedule, q_hbm, k_hbm, v_hbm, refs, bq=bq, bkv=bkv, **body)


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
  # Note (david): validate before extending, which would otherwise turn a
  # malformed pair into a raw concatenate error or into one fabricated
  # sequence.
  check_cu_seqlens_pair(cu_seqlens_k, cu_seqlens_q)
  # Note (david): per-seq blocks address HBM at arbitrary token offsets, which
  # only a second-minor (sublane) token axis supports.
  if transposed_pv or block_sizes.qkv_layout != QKVLayout.HEAD_DIM_MINOR:
    raise ValueError(
        "the per-seq varlen schedule requires a HEAD_DIM_MINOR qkv_layout"
        " and transposed_pv=False (token axis must stay second-minor)."
    )
  # Note (david): giving the pad tail a square sequence of its own keeps pad
  # rows off every unclaimed kv token.
  cu_q_split, cu_k_split = split_pad_tail(
      extend_cu_seqlens(cu_seqlens_q, q.shape[-2]),
      extend_cu_seqlens(cu_seqlens_k, k.shape[-2]))
  smem_operands = [cu_q_split, cu_k_split,
                   per_seq_qblk_prefix(cu_q_split, block_sizes.block_q)]

  kernel = partial(
      flash_fwd_varlen_kernel,
      varlen_sched=varlen_sched,
      num_head_groups=num_q_heads // head_fold,
  )
  return forward_common(
      kernel, smem_operands, q, k, v,
      block_sizes=block_sizes,
      head_fold=head_fold, transposed_pv=transposed_pv,
      kernel_name="flash_attn_mha_static_t1_guarded_fwd_varlen_per_seq",
      token_major=token_major,
      is_per_seq=True,
      **body,
  )
