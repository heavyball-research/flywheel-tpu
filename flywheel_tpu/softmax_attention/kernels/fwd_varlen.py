"""Varlen forward kernel driven by a schedule table built from the cu_seqlens.

The per-seq schedule cuts each sequence into its own q blocks, one SMEM table
row each, and synthesizes its masks in-kernel: the sequence bounds plus the
causal or window edges around each sequence's own bottom-right diagonal.
"""

from functools import partial

import jax

from ..config import BlockSizes, QKVLayout, TokenMajorInfo
from ..fwd_pipeline import forward_common, fwd_body, qkv_fwd_traits
from ..schedule import (
    ScheduleTableLayout,
    VarlenSchedule,
    make_per_seq_fwd_schedule,
    varlen_schedule_table,
)
from ..seqlen import check_cu_seqlens_pair
from ..traits import FwdTraits
from ..variant import Variant, VariantOperands


def flash_fwd_varlen_kernel(
    table_ref: jax.Array,
    *refs: jax.Array,
    traits: FwdTraits,
    variant: Variant,
    varlen_sched: VarlenSchedule,
    layout: ScheduleTableLayout,
    num_head_groups: int,
) -> None:
    if variant.has_sinks:
        sinks_ref, q_hbm, k_hbm, v_hbm, *refs = refs
    else:
        sinks_ref = None
        q_hbm, k_hbm, v_hbm, *refs = refs
    schedule = make_per_seq_fwd_schedule(
        table_ref,
        layout=layout,
        num_head_groups=num_head_groups,
        q_heads_per_kv_head=varlen_sched.q_heads_per_kv_head,
        padded_total_k=varlen_sched.padded_total_k,
        bq=traits.bq,
        bkv=traits.bkv,
        bkv_compute=traits.bkv_compute,
        left=varlen_sched.left,
        right=varlen_sched.right,
    )
    fwd_body(
        traits,
        variant,
        schedule,
        q_hbm,
        k_hbm,
        v_hbm,
        refs,
        operands=VariantOperands(sinks_ref=sinks_ref),
    )


def flash_attn_forward_varlen(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    cu_seqlens_q: jax.Array,
    cu_seqlens_k: jax.Array,
    sinks: jax.Array | None = None,
    *,
    varlen_sched: VarlenSchedule,
    block_sizes: BlockSizes,
    num_kv_heads: int,
    head_fold: int,
    transposed_pv: bool,
    token_major: TokenMajorInfo | None,
    guard_threshold: float,
    variant: Variant,
    interpret: bool,
    rotary: tuple[jax.Array, jax.Array | None] | None = None,
    rotary_interleaved: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    check_cu_seqlens_pair(cu_seqlens_k, cu_seqlens_q)
    # Note (david): per-seq blocks address HBM at arbitrary token offsets, which
    # only a second-minor (sublane) token axis supports.
    if transposed_pv or block_sizes.qkv_layout != QKVLayout.HEAD_DIM_MINOR:
        raise ValueError(
            "the per-seq varlen schedule requires a HEAD_DIM_MINOR qkv_layout"
            " and transposed_pv=False (token axis must stay second-minor)."
        )
    traits = qkv_fwd_traits(
        q,
        v,
        rotary,
        rotary_interleaved,
        block_sizes=block_sizes,
        num_kv_heads=num_kv_heads,
        head_fold=head_fold,
        transposed_pv=transposed_pv,
        token_major=token_major,
        is_per_seq=True,
        guard_threshold=guard_threshold,
    )
    # Note (david): padding rows past cu_seqlens_q[-1] own no q block and come
    # back out = 0, lse = -inf, so they cost no attention; kv tokens past
    # cu_seqlens_k[-1] are masked wherever a block loads them. Resolving the
    # schedule once per call spares every q block a search of the cu_seqlens.
    table, layout = varlen_schedule_table(
        cu_seqlens_q,
        cu_seqlens_k,
        varlen_sched=varlen_sched,
        bq=block_sizes.block_q,
        bkv=block_sizes.block_kv,
        bkv_compute=block_sizes.block_kv_compute,
    )
    smem_operands = [table]
    variant.check_sinks_operand(sinks)
    if sinks is not None:
        # Note (deepolar): packed rows carry no batch, so the flat q head is the
        # sequence head itself.
        num_seq_heads = (
            q.shape[0] if token_major is None else token_major.num_q_heads
        )
        if sinks.shape != (num_seq_heads,):
            raise ValueError(
                f"sinks must be ({num_seq_heads},), one per q head; got"
                f" {sinks.shape}."
            )
        smem_operands.append(variant.sink_logits(sinks))

    kernel = partial(
        flash_fwd_varlen_kernel,
        varlen_sched=varlen_sched,
        layout=layout,
        num_head_groups=traits.num_q_heads // head_fold,
    )
    return forward_common(
        kernel,
        smem_operands,
        q,
        k,
        v,
        traits,
        variant,
        kernel_name="flash_attn_mha_static_t1_guarded_fwd_varlen_per_seq",
        interpret=interpret,
        rotary=rotary,
    )
