"""Dense (batch) forward kernel and the forward kernel builder."""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp

from .block_sizes import (
  LOG2E,
  NUM_LANES,
  PAIRED_HEAD_DIM,
  BlockSizes,
  QKVLayout,
  TokenMajorInfo,
)
from .flash_fwd_varlen import flash_attn_forward_varlen
from .fwd_pipeline import forward_common, fwd_body, overflow_guard_threshold
from .loop_schedule import (
  DenseFwdParams,
  VarlenSchedule,
  dense_fwd_params,
  make_dense_fwd_schedule,
  make_runtime_offset_dense_fwd_schedule,
)

# Note (david): the static-anchor softmax runs a distance-1 fragment pipeline,
# which must not revisit a q slice while its rescale is still pending.
MIN_Q_TILES_PER_BLOCK = 2


def flash_fwd_kernel(
    q_hbm: jax.Array,
    k_hbm: jax.Array,
    v_hbm: jax.Array,
    *refs: jax.Array,
    dense_params: DenseFwdParams,
    **body,
) -> None:
  fwd_body(
      make_dense_fwd_schedule(dense_params), q_hbm, k_hbm, v_hbm, refs, **body)


def flash_attn_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    dense_params: DenseFwdParams,
    **body,
) -> jax.Array | tuple[jax.Array, jax.Array]:
  return forward_common(
      partial(flash_fwd_kernel, dense_params=dense_params),
      [], q, k, v,
      kernel_name="flash_attn_mha_static_t1_guarded_fwd",
      **body,
  )


def flash_fwd_runtime_offset_kernel(
    offset_ref: jax.Array,
    q_hbm: jax.Array,
    k_hbm: jax.Array,
    v_hbm: jax.Array,
    *refs: jax.Array,
    dense_params: DenseFwdParams,
    **body,
) -> None:
  runtime_offset = offset_ref[0]
  body = dict(body)
  body["causal_offset"] = runtime_offset
  fwd_body(
      make_runtime_offset_dense_fwd_schedule(dense_params, runtime_offset),
      q_hbm, k_hbm, v_hbm, refs, **body,
  )


def flash_attn_forward_runtime_offset(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    runtime_offset: jax.Array,
    *,
    dense_params: DenseFwdParams,
    **body,
) -> jax.Array | tuple[jax.Array, jax.Array]:
  offset = jnp.asarray(runtime_offset, jnp.int32).reshape(1)
  return forward_common(
      partial(flash_fwd_runtime_offset_kernel, dense_params=dense_params),
      [offset], q, k, v,
      kernel_name="flash_attn_mha_runtime_offset_fwd",
      **body,
  )


def make_flash_attn_mha(
    num_heads: int,
    seqlen_q: int,
    seqlen_kv: int,
    *,
    causal: bool,
    causal_offset: int = 0,
    window: tuple[int | None, int | None] | None = None,
    block_sizes: BlockSizes,
    num_kv_heads: int | None = None,
    interpret: bool = False,
    head_fold: int = 1,
    transposed_pv: bool = False,
    softcap: float = 0.0,
    return_lse: bool = False,
    varlen_max_seqlen_kv: int | None = None,
    token_major: TokenMajorInfo | None = None,
    q_scale: float = 1.0,
    heads_outer_batch: int | None = None,
    runtime_causal_offset: bool = False,
) -> Callable[..., jax.Array | tuple[jax.Array, jax.Array]]:
  """Build the MHA forward kernel for one static attention configuration.

  The returned callable takes (q, k, v), or (q, k, v, cu_seqlens_q,
  cu_seqlens_k) when varlen_max_seqlen_kv is set, or (q, k, v, causal_offset)
  under runtime_causal_offset, plus optional rotary and rotary_interleaved
  kwargs, and returns o, or (o, lse) under return_lse.

  num_heads / num_kv_heads: per-row counts under token_major, flat otherwise;
    num_kv_heads=None is MHA and MQA is num_kv_heads=1.
  causal / causal_offset / window: causal alone is window=(None, 0) shifted by
    causal_offset; an explicit window fully determines (left, right). Under
    varlen both are bottom-right per sequence and causal_offset must be 0.
  block_sizes: block_q must hold at least two block_q_compute tiles.
  head_fold: consecutive heads folded into one physical block; 1 disables it.
  transposed_pv: compute PV as o^T [head_dim, q_block], which requires
    block_sizes.qkv_layout == SEQ_MINOR; the output is swapped back.
  softcap: Gemma-2 tanh softcap; Q must then carry softmax_scale / softcap
    instead of softmax_scale * log2(e).
  q_scale: scale applied in-kernel to each staged Q tile. The softmax runs in
    exp2, so Q must reach the QK matmul multiplied by log2(e).
  return_lse: also return the natural-log logsumexp, (num_q_heads, seqlen_q)
    float32.
  varlen_max_seqlen_kv: switches to the cu_seqlens-driven block schedule;
    only its presence is read.
  token_major: (batch, seq, heads * head_dim) or (total, heads * head_dim)
    q/k/v addressing; None is head-major (heads, seq, head_dim).
  heads_outer_batch: B for a heads-outer dense fold of B sequences, where q
    fold index n * B + b reads kv fold index (n // q_heads_per_kv_head) * B + b;
    None keeps the batch-outer fold.
  runtime_causal_offset: dense causal only; the causal offset becomes a
    device-side int32 scalar argument of the returned callable, and the static
    causal_offset must stay 0.
  """
  if runtime_causal_offset and (not causal or window is not None
                                or varlen_max_seqlen_kv is not None):
    raise ValueError(
        "runtime_causal_offset requires dense causal attention without an"
        " explicit window.")
  if type(transposed_pv) is not bool:
    raise TypeError("transposed_pv must be a bool.")
  if type(return_lse) is not bool:
    raise TypeError("return_lse must be a bool.")
  if return_lse and transposed_pv:
    raise NotImplementedError(
        "return_lse is only wired for transposed_pv=False; got"
        f" {transposed_pv=}."
    )
  if softcap < 0.0:
    raise ValueError(f"softcap must be non-negative; got {softcap}.")
  if token_major is not None and block_sizes.qkv_layout != QKVLayout.HEAD_DIM_MINOR:
    raise ValueError(
        "token-major requires qkv_layout == HEAD_DIM_MINOR: a SEQ_MINOR"
        " slab cannot be sliced out of a (.., seq, heads*head_dim) ref."
    )
  # Note (david): the transposed PV matmul contracts v's block_kv axis in
  # place, which only a SEQ_MINOR v has. With the token-major check above this
  # also rejects token-major x transposed_pv, since they demand opposite
  # layouts.
  if transposed_pv and block_sizes.qkv_layout != QKVLayout.SEQ_MINOR:
    raise ValueError(
        "transposed_pv=True requires block_sizes.qkv_layout =="
        " QKVLayout.SEQ_MINOR (v must already be the [head_dim, block_kv]"
        " slab the transposed PV matmul contracts directly); got"
        f" {block_sizes.qkv_layout!r}."
    )
  if block_sizes.block_q // block_sizes.block_q_compute < MIN_Q_TILES_PER_BLOCK:
    raise ValueError(
        "the static-anchor softmax requires >= 2 q tiles per block"
        " (block_q_compute < block_q): the distance-1 pipeline must not"
        " revisit a q slice while its rescale alpha is still pending."
    )
  is_varlen = varlen_max_seqlen_kv is not None
  if token_major is None:
    is_lane_paired = False
  else:
    is_lane_paired = (token_major.head_dim_qk % NUM_LANES != 0
                    or token_major.head_dim_v % NUM_LANES != 0)
    if is_lane_paired and (token_major.head_dim_qk, token_major.head_dim_v) != (
        PAIRED_HEAD_DIM, PAIRED_HEAD_DIM):
      raise ValueError(
          f"token-major requires head_dim % 128 == 0, or head_dim =="
          f" head_dim_v == 64 (paired); got {token_major!r}."
      )
    # Note (david): the schedule reads num_heads / num_kv_heads while the DMA
    # windows read token_major, so a disagreement would silently address the
    # wrong lanes.
    expected_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
    if (token_major.num_q_heads, token_major.num_kv_heads) != (
        num_heads, expected_kv_heads):
      raise ValueError(
          f"token_major carries per-row heads"
          f" ({token_major.num_q_heads}, {token_major.num_kv_heads}) but the"
          f" builder was given ({num_heads}, {expected_kv_heads})."
      )
    if token_major.batch is None and not is_varlen:
      raise ValueError(
          "token-major with batch=None is 2-D packed storage, which only"
          " the runtime (varlen) schedule consumes; dense x 2-D has no"
          " caller. Pass batch for dense, or varlen_max_seqlen_kv."
      )
    if head_fold > 1 and num_heads % head_fold:
      raise ValueError(
          f"token-major head_fold must divide the PER-ROW num_heads (a fold"
          f" group cannot straddle a batch boundary); got {head_fold=},"
          f" per-row {num_heads=}."
      )
  # Note (david): token-major head counts are per fused row while the schedule
  # runs on flat heads, so widen them by the batch rows (2-D storage is one
  # row of heads).
  num_batch_rows = 1 if token_major is None else (token_major.batch or 1)
  num_q_heads = num_heads * num_batch_rows
  guard_threshold = overflow_guard_threshold(seqlen_kv)

  num_kv_heads = (
      num_q_heads if num_kv_heads is None else num_kv_heads * num_batch_rows)
  if num_q_heads % num_kv_heads:
    raise ValueError(f"{num_kv_heads=} must divide {num_q_heads=}.")
  q_heads_per_kv_head = num_q_heads // num_kv_heads

  if type(head_fold) is not int:
    raise TypeError(f"head_fold must be an int, got {type(head_fold)}.")
  if head_fold <= 0:
    raise ValueError(f"head_fold must be positive, got {head_fold}.")
  if is_lane_paired and head_fold % 2:
    raise ValueError(
        f"token-major d64 requires an even head_fold (a DMA group must span"
        f" whole 128-lane tiles); got {head_fold=}."
    )
  if head_fold > 1:
    if q_heads_per_kv_head != 1:
      raise ValueError(
          "head_fold > 1 requires one query head per kv head (no GQA); got"
          f" {q_heads_per_kv_head=}."
      )
    if num_q_heads % head_fold:
      raise ValueError(f"{head_fold=} must divide {num_q_heads=}.")
    # Note (david): a d64 lane-pair fold is the DMA-alignment mechanism, not an
    # optimization, so it skips the single-block restriction that is only a
    # performance heuristic for other dense folds.
    if is_varlen:
      has_foldable_blocks = block_sizes.block_q == block_sizes.block_kv
    else:
      has_foldable_blocks = is_lane_paired or (
          block_sizes.block_q == seqlen_q and block_sizes.block_kv == seqlen_q)
    if not has_foldable_blocks:
      raise ValueError(
          "head_fold > 1 requires the whole sequence to be one physical block"
          " for dense attention or square physical blocks for varlen attention:"
          f" block_q={block_sizes.block_q}, block_kv={block_sizes.block_kv},"
          f" seq_len={seqlen_q}, varlen={is_varlen}."
      )
  if heads_outer_batch is not None:
    if type(heads_outer_batch) is not int or heads_outer_batch <= 0:
      raise ValueError(
          "heads_outer_batch must be a positive int; got"
          f" {heads_outer_batch!r}."
      )
    if token_major is not None or is_varlen:
      raise ValueError(
          "heads_outer_batch is a head-major dense fold; got"
          f" token_major={token_major!r},"
          f" varlen_max_seqlen_kv={varlen_max_seqlen_kv!r}."
      )
    if head_fold != 1:
      raise ValueError(
          "heads_outer_batch requires head_fold == 1 (a group of consecutive"
          f" fold indices would span sequences); got {head_fold=}."
      )
    if num_q_heads % heads_outer_batch or num_kv_heads % heads_outer_batch:
      raise ValueError(
          f"heads_outer_batch={heads_outer_batch} must divide the folded"
          f" {num_q_heads=} and {num_kv_heads=}."
      )
  # Note (david): packed causal and window edges sit on each sequence's own
  # bottom-right diagonal, which no single static offset expresses once q and
  # kv pack independently, so the per-seq schedule carries them and the kernel
  # is built without a static mask.
  if is_varlen and causal_offset != 0:
    raise ValueError(
        "a packed forward derives its bottom-right offset per sequence from"
        f" cu_seqlens; a kernel-wide causal_offset={causal_offset} would be"
        " applied on top of it. Pass causal_offset=0."
    )
  if window is not None:
    left, right = window
  elif causal:
    left, right = None, 0
  else:
    left, right = None, None

  if is_varlen:
    forward = flash_attn_forward_varlen
    schedule_kwargs = {"varlen_sched": VarlenSchedule(
        num_q_blocks=seqlen_q // block_sizes.block_q,
        num_kv_blocks=seqlen_kv // block_sizes.block_kv,
        q_heads_per_kv_head=q_heads_per_kv_head,
        left=left,
        right=right,
    )}
    static_window = (None, None)
  else:
    forward = (flash_attn_forward_runtime_offset
               if runtime_causal_offset else flash_attn_forward)
    schedule_kwargs = {"dense_params": dense_fwd_params(
        num_q_heads=num_q_heads,
        head_fold=head_fold,
        q_heads_per_kv_head=q_heads_per_kv_head,
        seqlen_q=seqlen_q,
        seqlen_kv=seqlen_kv,
        bq=block_sizes.block_q,
        bkv=block_sizes.block_kv,
        left=left,
        right=right,
        offset=causal_offset,
        heads_outer_batch=heads_outer_batch,
    )}
    static_window = (left, right)
  # Note (david): Q carries softmax_scale / softcap rather than log2(e), so the
  # log2(e) the exp2 softmax needs is folded into the tanh output scale; 0.0
  # stays 0.0, so the kernel never traces the tanh.
  log2_softcap = softcap * LOG2E
  return partial(
      forward,
      **schedule_kwargs,
      block_sizes=block_sizes,
      num_kv_heads=num_kv_heads,
      interpret=interpret,
      guard_threshold=guard_threshold,
      head_fold=head_fold,
      transposed_pv=transposed_pv,
      window=static_window,
      causal_offset=causal_offset,
      softcap=log2_softcap,
      return_lse=return_lse,
      token_major=token_major,
      q_scale=float(q_scale),
  )
