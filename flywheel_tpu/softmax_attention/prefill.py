"""flash_attn_func and flash_attn_varlen_func served by the TPU Pallas
kernels.

Arguments the kernels do not implement are accepted but raise
NotImplementedError on any non-default value.
"""

import functools
import math

import jax
import jax.numpy as jnp

from ..core.tiling import round_up
from ..core.tpu import NUM_LANES
from .autograd import attn_with_vjp, forward_only_attn
from .config import BlockSizes, next_pow2
from .dispatch import (
    default_block_sizes,
    get_kernel,
    resolve_paged_block_sizes,
)
from .inputs import (
    HEAD_MAJOR_PACKED_AXIS,
    HEAD_MAJOR_SEQ_AXIS,
    ROTARY_TOKEN_AXIS,
    TOKEN_MAJOR_PACKED_AXIS,
    TOKEN_MAJOR_SEQ_AXIS,
    check_common_kwargs,
    check_sinks,
    descale_scalar,
    fused_q_scale,
    pad_axis,
    pad_seqlen,
    paged_window,
    parse_qkv,
)
from .kernels.fwd_dense import make_flash_attn_mha
from .kernels.fwd_paged import check_paged_inputs, flash_attn_varlen_paged
from .rotary import prepare_rotary


def flash_attn_func(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    attention_chunk: int | None = None,
    softcap: float = 0.0,
    alibi_slopes: jax.Array | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    return_softmax_lse: bool = False,
    interpret: bool = False,
    head_dim: int | None = None,
    token_major: bool = False,
    *,
    rotary_cos: jax.Array | None = None,
    rotary_sin: jax.Array | None = None,
    rotary_interleaved: bool = False,
    rotary_k: bool = True,
    heads_outer_fold: bool = False,
    learnable_sink: jax.Array | None = None,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """flash_attn.flash_attn_func on TPU, JAX arrays in and out.

    Head-major (default): q (batch, nheads, seqlen_q, headdim_qk), k (batch,
    nheads_k, seqlen_k, headdim_qk), v (batch, nheads_k, seqlen_k, headdim_v),
    out (batch, nheads, seqlen_q, headdim_v) -- the kernel's own head-major
    layout with the batch split off, so no operand is relaid out. Token-major
    (token_major=True) fuses the heads into the last axis, (batch, seqlen,
    nheads * head_dim), and head_dim gives the split: a multiple of 128, or 64
    for MHA with an even head count. nheads % nheads_k == 0, and all inputs are
    bfloat16.

    causal is bottom-right aligned on the original lengths: query i attends key
    j iff j <= i + seqlen_k - seqlen_q. seqlen_q may be any length; seqlen_k
    must be a multiple of 128 unless the mask's right bound is 0. Rows with no
    reachable key return out = 0 and lse = -inf. return_softmax_lse adds lse
    (batch, nheads, seqlen_q) float32, the natural-log logsumexp of the scaled,
    masked scores. softcap > 0 applies
    tanh(softmax_scale * q @ k^T / softcap) * softcap.

    rotary_cos / rotary_sin: fused full-head RoPE tables (seqlen_ro,
    headdim_qk / 2) in q's dtype or float32, headdim_qk divisible by 16. K is
    rotated at positions 0..seqlen_k - 1 and Q from seqlen_k - seqlen_q on, so
    seqlen_q <= seqlen_k. rotary_interleaved pairs adjacent dimensions instead
    of the two halves; rotary_k=False rotates Q only.

    heads_outer_fold (head-major only): q/k/v and out are heads-outer, (nheads,
    batch, seqlen, headdim), again with no relayout; supports Q-only RoPE and no
    lse. attention_chunk is not implemented. interpret runs the Pallas TPU
    interpreter.

    learnable_sink: None or (nheads,) float sink logits, on the scale of the
    softcapped scores (gpt-oss attention sinks, FlashAttention's
    learnable_sink): exp(learnable_sink[h]) joins head h's softmax denominator
    with no value row, and lse includes it. A row with no reachable key then
    returns out = 0 and lse = learnable_sink[h]. Each logit must be finite or
    -inf (no sink); +inf and NaN are not checked and give undefined results.

    Gradients cover MHA and GQA/MQA with causal or full masks; window_size,
    softcap, RoPE, heads_outer_fold and learnable_sink are forward-only.
    """
    window_size = check_common_kwargs(
        window_size,
        dropout_p=dropout_p,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
    )
    if attention_chunk is not None:
        raise NotImplementedError("attention_chunk is not implemented on TPU.")
    if type(heads_outer_fold) is not bool:
        raise TypeError("heads_outer_fold must be a bool.")
    has_rotary_tables = rotary_cos is not None or rotary_sin is not None
    if heads_outer_fold and (
        token_major or return_softmax_lse or (has_rotary_tables and rotary_k)
    ):
        raise NotImplementedError(
            "heads_outer_fold is a forward-only head-major layout; it supports"
            " fused RoPE only as Q-only rotation (rotary_k=False) and no"
            " return_softmax_lse."
        )
    expected_rank = 3 if token_major else 4
    nheads, nheads_k, head_dim_qk, head_dim_v = parse_qkv(
        q,
        k,
        v,
        head_dim,
        token_major,
        expected_rank,
        "hbsd" if heads_outer_fold else "bhsd",
    )
    if token_major:
        seq_axis = TOKEN_MAJOR_SEQ_AXIS
        batch = q.shape[0]
    else:
        seq_axis = HEAD_MAJOR_SEQ_AXIS
        batch = q.shape[1] if heads_outer_fold else q.shape[0]
    seqlen_q, seqlen_k = q.shape[seq_axis], k.shape[seq_axis]
    rotary = prepare_rotary(
        rotary_cos,
        rotary_sin,
        head_dim=head_dim_qk,
        dtype=q.dtype,
        rotate_k=rotary_k,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        interleaved=rotary_interleaved,
    )

    padded_seqlen_q = pad_seqlen(seqlen_q)
    padded_seqlen_k = pad_seqlen(seqlen_k)
    num_pad_q = padded_seqlen_q - seqlen_q
    num_pad_kv = padded_seqlen_k - seqlen_k
    # Note (david): a right bound of 0 keeps every real query off the padded kv
    # tail, so seqlen_k pads for free; without it the dense kernel cannot hide
    # that tail. Padded q rows need no mask: they are sliced off, and the zero
    # cotangent of that slice cancels their k/v gradient contributions.
    is_right_bounded = causal or window_size[1] == 0
    if num_pad_kv and not is_right_bounded:
        raise ValueError(
            f"flash_attn_func is dense-only: seqlen_k={seqlen_k} is not a"
            " multiple of 128 and the mask has no right bound at 0, so the"
            " padded kv tail would be attended. Align seqlen_k to a multiple of"
            " 128, or use flash_attn_varlen_func."
        )

    q = pad_axis(q, seq_axis, num_pad_q)
    k, v = (pad_axis(operand, seq_axis, num_pad_kv) for operand in (k, v))
    if rotary is None:
        rotary_coeffs = None
    else:
        q_coeff, k_coeff = rotary
        rotary_coeffs = (
            pad_axis(q_coeff, ROTARY_TOKEN_AXIS, num_pad_q),
            None
            if k_coeff is None
            else pad_axis(k_coeff, ROTARY_TOKEN_AXIS, num_pad_kv),
        )

    softmax_scale = (
        1.0 / math.sqrt(head_dim_qk) if softmax_scale is None else softmax_scale
    )
    # Note (david): bottom-right alignment anchors the causal diagonal on the
    # original lengths, not the padded ones.
    causal_offset = seqlen_k - seqlen_q
    # Note (david): token-major kernels are built from per-row heads plus the
    # batch, while head-major folds the batch into the head axis.
    if token_major:
        kernel_num_heads, kernel_num_kv_heads = nheads, nheads_k
        token_major_batch = batch
    else:
        kernel_num_heads = batch * nheads
        kernel_num_kv_heads = batch * nheads_k
        token_major_batch = None
    build_kernel = functools.partial(
        get_kernel,
        num_heads=kernel_num_heads,
        num_kv_heads=kernel_num_kv_heads,
        seqlen_q=padded_seqlen_q,
        seqlen_kv=padded_seqlen_k,
        causal=bool(causal),
        window_size=window_size,
        interpret=bool(interpret),
        softcap=float(softcap),
        head_dim=head_dim_qk,
        head_dim_v=head_dim_v,
        causal_offset=causal_offset,
        token_major=bool(token_major),
        kernel_batch=token_major_batch,
        q_scale=float(fused_q_scale(softmax_scale, softcap)),
    )
    # Note (david): token-major kernels address heads as lane-axis column
    # windows of the fused refs, and either head-major layout is the kernel's
    # leading axis split in two, so no operand is relaid out.
    if token_major:
        kernel_q, kernel_k, kernel_v = q, k, v
    else:
        kernel_q, kernel_k, kernel_v = (
            operand.reshape(-1, *operand.shape[2:]) for operand in (q, k, v)
        )
    sinks = check_sinks(learnable_sink, nheads)
    sink_args = () if sinks is None else (sinks,)
    if heads_outer_fold:
        # Note (david): flash_attn_bwd has no heads-outer kv map, so this layout
        # is forward-only. Q-only RoPE stays valid: the dense coefficients carry
        # a singleton batch shared by every kernel row, and each heads-outer row
        # is one sequence at positions 0..seqlen-1.
        reason = "heads_outer_fold" if rotary_coeffs is None else "Fused RoPE"
        kernel_output = forward_only_attn(
            build_kernel(return_lse=False, heads_outer_batch=batch),
            reason,
            kernel_q,
            kernel_k,
            kernel_v,
            sink_args,
            rotary_coeffs,
            rotary_interleaved,
        )
    elif rotary_coeffs is not None or sinks is not None:
        reason = "learnable_sink" if rotary_coeffs is None else "Fused RoPE"
        kernel_output = forward_only_attn(
            build_kernel(return_lse=bool(return_softmax_lse)),
            reason,
            kernel_q,
            kernel_k,
            kernel_v,
            sink_args,
            rotary_coeffs,
            rotary_interleaved,
        )
    elif return_softmax_lse:
        # Note (david): lse has no gradient of its own, so this path stays the
        # bare kernel call.
        kernel_output = build_kernel(return_lse=True)(
            kernel_q, kernel_k, kernel_v
        )
    else:
        if window_size != (-1, -1):
            bwd_unsupported = "window_size"
        elif softcap != 0.0:
            bwd_unsupported = "softcap"
        else:
            bwd_unsupported = ""
        kernel_output = attn_with_vjp(
            kernel_q,
            kernel_k,
            kernel_v,
            build_kernel=build_kernel,
            softmax_scale=softmax_scale,
            causal=bool(causal),
            causal_offset=causal_offset,
            interpret=bool(interpret),
            bwd_unsupported=bwd_unsupported,
            token_major=token_major,
            head_dim=head_dim_qk if token_major else None,
        )
    if return_softmax_lse:
        kernel_out, kernel_lse = kernel_output
    else:
        kernel_out = kernel_output

    if token_major:
        out_padded = kernel_out
    else:
        out_padded = kernel_out.reshape(
            *q.shape[:2], padded_seqlen_q, head_dim_v
        )
    out_sliced = jax.lax.slice_in_dim(out_padded, 0, seqlen_q, axis=seq_axis)
    # Note (david): a right bound leaves the first -causal_offset - right rows
    # with no reachable key. The kernel skips their blocks, so its output there
    # is unspecified; match flash_attn with out = 0 and lse = -inf.
    if causal:
        num_masked_rows = -causal_offset
    elif window_size[1] != -1:
        num_masked_rows = -causal_offset - window_size[1]
    else:
        num_masked_rows = 0
    if num_masked_rows > 0:
        is_valid_row = jnp.arange(seqlen_q) >= num_masked_rows
        row_broadcast_index = tuple(
            slice(None) if axis == seq_axis else None
            for axis in range(out_sliced.ndim)
        )
        out = jnp.where(is_valid_row[row_broadcast_index], out_sliced, 0)
    else:
        out = out_sliced
    if return_softmax_lse:
        # Note (david): the kernel lse is (batch * nheads, padded_seqlen_q)
        # batch-major in both layouts.
        lse = kernel_lse.reshape(batch, nheads, padded_seqlen_q)[
            :, :, :seqlen_q
        ]
        if num_masked_rows > 0:
            # Note (deepolar): a masked row's only softmax term is its sink.
            empty_lse = -jnp.inf if sinks is None else sinks[None, :, None]
            return out, jnp.where(is_valid_row[None, None, :], lse, empty_lse)
        else:
            return out, lse
    else:
        return out


def _flash_attn_func_pcp_causal_offset(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    causal_offset: jax.Array | int,
    *,
    softmax_scale: float | None = None,
    return_softmax_lse: bool = False,
    heads_outer_fold: bool = False,
    interpret: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Internal dense PCP forward with a device-side Q-position offset.

    Q/K/V are flash_attn_func's head-major (1, nheads, seqlen, headdim), or
    (nheads, 1, seqlen, headdim) with heads_outer_fold. The PCP caller
    guarantees 0 <= offset and offset + q_len <= kv_len before padding. This is
    not a general arbitrary-offset attention API.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(
            "runtime-offset attention requires 4-D head-major Q/K/V."
        )
    batch_axis, head_axis = (1, 0) if heads_outer_fold else (0, 1)
    batch, num_heads = q.shape[batch_axis], q.shape[head_axis]
    q_len, head_dim = q.shape[HEAD_MAJOR_SEQ_AXIS], q.shape[3]
    if batch != 1 or k.shape[batch_axis] != 1 or v.shape[batch_axis] != 1:
        raise ValueError("runtime-offset attention requires batch_size == 1.")
    if k.shape != v.shape or k.shape[3] != head_dim:
        raise ValueError(
            "K/V must share shape and Q/K head dimensions must match."
        )
    num_kv_heads = k.shape[head_axis]
    kv_len = k.shape[HEAD_MAJOR_SEQ_AXIS]
    if num_heads % num_kv_heads:
        raise ValueError("Q heads must be divisible by KV heads.")
    if q_len <= 0 or kv_len < q_len:
        raise ValueError(
            "runtime-offset attention requires a positive Q length, a"
            " positive KV length, and kv_len >= q_len."
        )
    if any(operand.dtype != jnp.bfloat16 for operand in (q, k, v)):
        raise NotImplementedError(
            "runtime-offset attention supports bf16 only."
        )
    offset_array = jnp.asarray(causal_offset)
    if offset_array.size != 1 or not jnp.issubdtype(
        offset_array.dtype, jnp.integer
    ):
        raise ValueError("causal_offset must be one integer scalar.")
    if softmax_scale is None:
        softmax_scale = head_dim**-0.5
    padded_q_len = pad_seqlen(q_len)
    padded_kv_len = pad_seqlen(kv_len)
    q = pad_axis(q, HEAD_MAJOR_SEQ_AXIS, padded_q_len - q_len)
    k = pad_axis(k, HEAD_MAJOR_SEQ_AXIS, padded_kv_len - kv_len)
    v = pad_axis(v, HEAD_MAJOR_SEQ_AXIS, padded_kv_len - kv_len)
    kernel = make_flash_attn_mha(
        num_heads,
        padded_q_len,
        padded_kv_len,
        causal=True,
        causal_offset=0,
        block_sizes=default_block_sizes(padded_q_len, padded_kv_len),
        num_kv_heads=num_kv_heads,
        interpret=interpret,
        return_lse=return_softmax_lse,
        q_scale=float(fused_q_scale(softmax_scale, 0.0)),
        heads_outer_batch=1 if heads_outer_fold else None,
        runtime_causal_offset=True,
    )
    kernel_output = kernel(
        *(operand.reshape(-1, *operand.shape[2:]) for operand in (q, k, v)),
        offset_array.reshape(()),
    )
    if return_softmax_lse:
        kernel_out, kernel_lse = kernel_output
    else:
        kernel_out = kernel_output
    out = kernel_out.reshape(*q.shape[:2], padded_q_len, head_dim)[:, :, :q_len]
    if return_softmax_lse:
        return (
            out,
            kernel_lse.reshape(batch, num_heads, padded_q_len)[:, :, :q_len],
        )
    else:
        return out


def paged_varlen_attn(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array | None,
    cu_seqlens_q: jax.Array,
    cu_seqlens_k: jax.Array | None,
    max_seqlen_q: int,
    max_seqlen_k: int,
    *,
    softmax_scale: float | None,
    causal: bool,
    window_size: tuple[int, int],
    softcap: float,
    return_softmax_lse: bool,
    interpret: bool,
    head_dim: int | None,
    token_major: bool,
    blocks_override: tuple[int, int, int, int] | None,
    rotary_cos: jax.Array | None,
    rotary_sin: jax.Array | None,
    rotary_interleaved: bool,
    rotary_k: bool,
    block_table: jax.Array,
    seqused_k: jax.Array | None,
    num_active: jax.Array | int | None,
    k_descale: float | jax.Array | None,
    v_descale: float | jax.Array | None,
    nheads_k: int | None,
    sinks: jax.Array | None,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """flash_attn_varlen_func's paged KV cache branch (block_table given)."""
    if token_major:
        raise NotImplementedError(
            "a paged KV cache takes head-major q (nheads, total_q, headdim);"
            " token_major=True is not supported."
        )
    if head_dim is not None:
        raise ValueError(
            "head_dim is only meaningful for token-major inputs; a paged call"
            " takes head-major q."
        )
    is_right_bounded, window_left = paged_window(causal, window_size)
    if cu_seqlens_k is not None:
        raise ValueError(
            "with block_table, pass cu_seqlens_k=None: seqused_k gives each"
            " sequence's kv length."
        )
    if v is not None:
        raise ValueError(
            "with block_table, pass one merged (num_pages, page_size,"
            " cache_heads, headdim) cache as k and v=None; separate paged K and"
            " V caches are not supported."
        )
    if seqused_k is None:
        raise ValueError(
            "block_table needs seqused_k, the (batch,) kv length of each"
            " sequence (its new tokens included)."
        )
    if rotary_cos is not None and rotary_k:
        raise NotImplementedError(
            "a paged KV cache holds K already rotated; pass rotary_k=False to"
            " rotate Q only."
        )
    for arg_name, seqlen_bound in (
        ("max_seqlen_q", max_seqlen_q),
        ("max_seqlen_k", max_seqlen_k),
    ):
        if isinstance(seqlen_bound, jax.core.Tracer):
            raise ValueError(
                f"{arg_name} must be a concrete int (it picks the kernel's"
                " block sizes); pass a static bucket bound, not a traced"
                " value."
            )
        if int(seqlen_bound) <= 0:
            raise ValueError(
                f"{arg_name} must be positive; got {seqlen_bound}."
            )
    if q.ndim != 3:
        raise ValueError(
            "with block_table, q must be (nheads, total_q, headdim); got"
            f" {q.shape=}."
        )

    cu_seqlens_q = jnp.asarray(cu_seqlens_q)
    seqused_k = jnp.asarray(seqused_k)
    if (
        cu_seqlens_q.ndim != 1
        or cu_seqlens_q.shape[0] < 2
        or not jnp.issubdtype(cu_seqlens_q.dtype, jnp.integer)
    ):
        raise ValueError(
            "cu_seqlens_q must be a 1-D integer array of batch + 1 boundaries;"
            f" got {cu_seqlens_q.shape} {cu_seqlens_q.dtype}."
        )
    batch = cu_seqlens_q.shape[0] - 1
    cu_seqlens_q = cu_seqlens_q.astype(jnp.int32)
    # Note (david): under a trace even a closed-over concrete array indexes to
    # a tracer, so the check runs on the indexed entry.
    first_boundary = cu_seqlens_q[0]
    if (
        not isinstance(first_boundary, jax.core.Tracer)
        and int(first_boundary) != 0
    ):
        raise ValueError(
            "with block_table, cu_seqlens_q must start at 0; got"
            f" cu_seqlens_q[0]={int(first_boundary)}."
        )
    if num_active is None:
        rotary_seqused_k = seqused_k
    else:
        num_active = jnp.asarray(num_active)
        if num_active.size != 1 or not jnp.issubdtype(
            num_active.dtype, jnp.integer
        ):
            raise ValueError(
                "num_active must be one integer; got"
                f" {num_active.shape} {num_active.dtype}."
            )
        num_active = num_active.reshape(())
        # Note (david): a traced num_active is the caller's contract, like the
        # other traced metadata; out of range, the cu_seqlens_q lookup below
        # would clamp or wrap it instead of raising.
        if (
            not isinstance(num_active, jax.core.Tracer)
            and not 0 <= int(num_active) <= batch
        ):
            raise ValueError(
                f"num_active must be in [0, {batch}]; got {int(num_active)}."
            )
        num_active = num_active.astype(jnp.int32)
        # Note (david): pinning every boundary past num_active to
        # cu_seqlens_q[num_active] empties the inactive sequences, so they own
        # no q block, the kernel never reads their seqused_k or block_table
        # entries, and their rows become packed padding.
        cu_seqlens_q = jnp.where(
            jnp.arange(batch + 1, dtype=jnp.int32) <= num_active,
            cu_seqlens_q,
            cu_seqlens_q[num_active],
        )
        rotary_seqused_k = jnp.where(
            jnp.arange(batch, dtype=jnp.int32) < num_active, seqused_k, 0
        )

    total_q, head_dim_qk = q.shape[1], q.shape[2]
    rotary = prepare_rotary(
        rotary_cos,
        rotary_sin,
        head_dim=head_dim_qk,
        dtype=q.dtype,
        rotate_k=rotary_k,
        seqlen_q=total_q,
        seqlen_k=int(max_seqlen_k),
        interleaved=rotary_interleaved,
        cu_q=cu_seqlens_q,
        seqused_k=None if rotary_cos is None else rotary_seqused_k,
        max_seqlen_k=max_seqlen_k,
    )
    if blocks_override is None:
        paged_block_sizes = None
    else:
        block_q, block_kv, block_q_compute, block_kv_compute = blocks_override
        paged_block_sizes = BlockSizes(
            block_q=block_q,
            block_kv=block_kv,
            block_kv_compute=block_kv_compute,
            block_q_compute=block_q_compute,
        )
    softmax_scale = (
        1.0 / math.sqrt(head_dim_qk) if softmax_scale is None else softmax_scale
    )
    k_scale = descale_scalar(k_descale, "k_descale")
    v_scale = descale_scalar(v_descale, "v_descale")
    kernel_sinks = check_sinks(sinks, q.shape[0])
    num_kv_heads = check_paged_inputs(
        q,
        k,
        cu_seqlens_q,
        seqused_k,
        block_table,
        kernel_sinks,
        nheads_k=nheads_k,
        block_sizes=paged_block_sizes,
    )
    block_sizes = resolve_paged_block_sizes(
        q,
        k,
        block_table,
        num_kv_heads=num_kv_heads,
        max_seqlen_q=int(max_seqlen_q),
        max_seqlen_k=int(max_seqlen_k),
        return_lse=bool(return_softmax_lse),
        rotary=rotary,
        has_window=window_left is not None,
        block_sizes=paged_block_sizes,
    )
    kernel = functools.partial(
        flash_attn_varlen_paged,
        block_sizes=block_sizes,
        causal=is_right_bounded,
        window_left=window_left,
        softcap=float(softcap),
        q_scale=float(fused_q_scale(softmax_scale, softcap)),
        return_lse=bool(return_softmax_lse),
        interpret=bool(interpret),
        nheads_k=nheads_k,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    # Note (david): K and V share the one merged cache operand k.
    return forward_only_attn(
        lambda q, kv_cache, unused_v, *kernel_args, **rotary_kwargs: kernel(
            q, kv_cache, *kernel_args, **rotary_kwargs
        ),
        "flash_attn_varlen_func",
        q,
        k,
        None,
        (cu_seqlens_q, seqused_k, block_table, kernel_sinks),
        rotary,
        rotary_interleaved,
    )


def flash_attn_varlen_func(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array | None,
    cu_seqlens_q: jax.Array,
    cu_seqlens_k: jax.Array | None,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: jax.Array | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    return_softmax_lse: bool = False,
    interpret: bool = False,
    head_dim: int | None = None,
    token_major: bool = False,
    block_sizes: tuple[int, int, int, int] | None = None,
    *,
    rotary_cos: jax.Array | None = None,
    rotary_sin: jax.Array | None = None,
    rotary_interleaved: bool = False,
    rotary_k: bool = True,
    block_table: jax.Array | None = None,
    seqused_k: jax.Array | None = None,
    num_active: jax.Array | int | None = None,
    k_descale: float | jax.Array | None = None,
    v_descale: float | jax.Array | None = None,
    nheads_k: int | None = None,
    learnable_sink: jax.Array | None = None,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """flash_attn.flash_attn_varlen_func on TPU for packed sequences.

    Head-major (default): q (nheads, total_q, headdim_qk), k (nheads_k,
    total_k, headdim_qk), v (nheads_k, total_k, headdim_v), out (nheads,
    total_q, headdim_v) -- the kernel's own head-major layout, so no operand is
    relaid out. Token-major (token_major=True) fuses the heads into the last
    axis, (total, nheads * headdim), and takes head_dim as flash_attn_func
    does. cu_seqlens_q / cu_seqlens_k are
    (num_seqs + 1,) int cumulative offsets and may be tracers; buffer rows past
    cu_seqlens[-1] are padding that never attends or is attended. Padding q
    rows return out = 0 and lse = -inf, as with a paged KV cache.

    causal and window_size are bottom-right aligned per sequence, as
    flash_attn_func aligns them on one sequence's own lengths; a row with no
    visible key is unspecified, so causal needs len_k(s) >= len_q(s).
    max_seqlen_q / max_seqlen_k must be static ints: they only cap the block
    sizes of the block-skipping schedule and are bucketed to powers of two, so
    an underestimate is slow, not wrong.

    Forward-only: jax.grad / jax.vjp raise NotImplementedError.
    return_softmax_lse adds lse (nheads, total_q) float32 in the varlen
    layout. RoPE follows flash_attn_func with per-sequence positions; the table
    must cover max_seqlen_k.

    block_sizes (TPU extension): (block_q, block_kv, block_q_compute,
    block_kv_compute) pins the forward tiles instead of the tuned or analytic
    ones. block_q and block_kv must be multiples of 128, and each packed total
    then pads to a multiple of its own block.

    Paged KV cache (block_table given; read-only, as in FlashAttention): k is
    one merged (num_pages, page_size, cache_heads, headdim) pool laid out as
    flash_attn_with_kvcache's (bf16 [k0, v0, k1, v1, ...], or fp8 with
    k_descale / v_descale and an explicit nheads_k), and v is None; headdim and
    page_size are multiples of 128, any nheads_k. block_table: (batch,
    max_pages_per_seq) int, never read past a sequence's own pages.
    cu_seqlens_k must be None and seqused_k ((batch,) int) is each sequence's
    kv length, its new tokens included, which the cache must already hold;
    max_seqlen_k bounds it. q / out stay head-major; token_major, rotary_k=True
    and a window right bound other than -1 or 0 raise, and RoPE rotates Q
    only, at positions seqused_k - len_q + t. window_size and softcap act as on
    packed K/V. cu_seqlens_q[0] must be 0; a concrete one is checked, a traced
    one is the caller's contract. A one-token sequence is an ordinary sequence.
    num_active (int or int32 scalar, possibly traced) keeps only the first
    num_active sequences; rows past the last kept one return out = 0 and lse =
    -inf. A concrete num_active outside [0, batch] raises; a traced one must
    lie in that range, which is not checked on device.
    learnable_sink: None or (nheads,) float sink logits, on the scale of the
    softcapped scores; exp(learnable_sink[h]) joins head h's softmax
    denominator with no value row (gpt-oss attention sinks), and lse includes
    it. Each logit must be finite or -inf (no sink); +inf and NaN are not
    checked and give undefined results. Padding rows and rows past num_active
    still return out = 0 and lse = -inf.
    """
    window_size = check_common_kwargs(
        window_size,
        dropout_p=dropout_p,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
    )
    if block_sizes is None:
        blocks_override = None
    else:
        blocks_override = tuple(block_sizes)
        if len(blocks_override) != 4 or not all(
            type(block) is int and block > 0 for block in blocks_override
        ):
            raise ValueError(
                "block_sizes must be 4 positive ints (block_q, block_kv,"
                f" block_q_compute, block_kv_compute); got {blocks_override!r}."
            )
        if blocks_override[0] % NUM_LANES or blocks_override[1] % NUM_LANES:
            raise ValueError(
                "block_q and block_kv must be multiples of 128 (the kernel's"
                f" native tile); got {blocks_override!r}."
            )
    if block_table is not None:
        return paged_varlen_attn(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            return_softmax_lse=return_softmax_lse,
            interpret=interpret,
            head_dim=head_dim,
            token_major=token_major,
            blocks_override=blocks_override,
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            rotary_interleaved=rotary_interleaved,
            rotary_k=rotary_k,
            block_table=block_table,
            seqused_k=seqused_k,
            num_active=num_active,
            k_descale=k_descale,
            v_descale=v_descale,
            nheads_k=nheads_k,
            sinks=learnable_sink,
        )
    elif seqused_k is not None or num_active is not None:
        raise ValueError(
            "seqused_k and num_active describe a paged KV cache and need"
            " block_table; packed K/V takes cu_seqlens_k."
        )
    elif any(arg is not None for arg in (k_descale, v_descale, nheads_k)):
        raise ValueError(
            "k_descale, v_descale and nheads_k describe an fp8 paged pool and"
            " need block_table; packed K/V is bf16."
        )
    else:
        expected_rank = 2 if token_major else 3
        nheads, nheads_k, head_dim_qk, head_dim_v = parse_qkv(
            q, k, v, head_dim, token_major, expected_rank, "htd"
        )
        if token_major:
            token_axis = TOKEN_MAJOR_PACKED_AXIS
        else:
            token_axis = HEAD_MAJOR_PACKED_AXIS
        total_q, total_k = q.shape[token_axis], k.shape[token_axis]

        cu_seqlens_q = jnp.asarray(cu_seqlens_q)
        cu_seqlens_k = jnp.asarray(cu_seqlens_k)
        if cu_seqlens_q.ndim != 1 or cu_seqlens_q.shape != cu_seqlens_k.shape:
            raise ValueError(
                "cu_seqlens_q and cu_seqlens_k must be equal-shaped 1-D"
                f" arrays; got {cu_seqlens_q.shape=}, {cu_seqlens_k.shape=}."
            )
        for arg_name, seqlen_bound in (
            ("max_seqlen_q", max_seqlen_q),
            ("max_seqlen_k", max_seqlen_k),
        ):
            if isinstance(seqlen_bound, jax.core.Tracer):
                # Note (david): a non-concrete bound is a ValueError in the
                # public contract, not a TypeError.
                raise ValueError(
                    f"{arg_name} must be a concrete int (it picks the kernel's"
                    " block sizes); pass a static bucket bound, not a traced"
                    " value."
                )
            if int(seqlen_bound) <= 0:
                raise ValueError(
                    f"{arg_name} must be positive; got {seqlen_bound}."
                )

        rotary = prepare_rotary(
            rotary_cos,
            rotary_sin,
            head_dim=head_dim_qk,
            dtype=q.dtype,
            rotate_k=rotary_k,
            seqlen_q=total_q,
            seqlen_k=total_k,
            interleaved=rotary_interleaved,
            cu_q=cu_seqlens_q,
            cu_k=cu_seqlens_k,
            max_seqlen_k=max_seqlen_k,
        )

        # Note (david): cu_seqlens needs no adjustment for the padding, because
        # rows past cu_seqlens_q[-1] own no q block and the kernel writes them
        # as zeros.
        if blocks_override is None:
            padded_total_q = pad_seqlen(total_q)
            padded_total_k = pad_seqlen(total_k)
        else:
            padded_total_q = round_up(total_q, blocks_override[0])
            padded_total_k = round_up(total_k, blocks_override[1])
        num_pad_q = padded_total_q - total_q
        num_pad_kv = padded_total_k - total_k
        q = pad_axis(q, token_axis, num_pad_q)
        k, v = (pad_axis(operand, token_axis, num_pad_kv) for operand in (k, v))
        if rotary is None:
            rotary_coeffs = None
        else:
            q_coeff, k_coeff = rotary
            rotary_coeffs = (
                pad_axis(q_coeff, ROTARY_TOKEN_AXIS, num_pad_q),
                None
                if k_coeff is None
                else pad_axis(k_coeff, ROTARY_TOKEN_AXIS, num_pad_kv),
            )

        softmax_scale = (
            1.0 / math.sqrt(head_dim_qk)
            if softmax_scale is None
            else softmax_scale
        )
        # Note (david): the max_seqlen bounds only cap the block sizes but ride
        # in the kernel cache key, so power-of-two buckets keep a wobbling
        # longest sequence on one compiled kernel.
        max_seqlen_k_bucket = min(padded_total_k, next_pow2(max_seqlen_k))
        max_seqlen_q_bucket = min(padded_total_q, next_pow2(max_seqlen_q))
        # Note (david): causal_offset stays 0 because packed causal and window
        # masking is per-sequence bottom-right, derived from the cu_seqlens pair
        # in the kernel.
        build_kernel = functools.partial(
            get_kernel,
            num_heads=nheads,
            num_kv_heads=nheads_k,
            seqlen_q=padded_total_q,
            seqlen_kv=padded_total_k,
            causal=bool(causal),
            window_size=window_size,
            interpret=bool(interpret),
            softcap=float(softcap),
            head_dim=head_dim_qk,
            head_dim_v=head_dim_v,
            causal_offset=0,
            varlen_max_seqlen_kv=max_seqlen_k_bucket,
            varlen_max_seqlen_q=max_seqlen_q_bucket,
            token_major=bool(token_major),
            q_scale=float(fused_q_scale(softmax_scale, softcap)),
            blocks_override=blocks_override,
        )
        sinks = check_sinks(learnable_sink, nheads)
        sink_args = () if sinks is None else (sinks,)
        # Note (david): varlen has no backward kernel, so every path, lse
        # included, is forward-only and jax.grad names it instead of failing in
        # the tracer.
        kernel_output = forward_only_attn(
            build_kernel(return_lse=bool(return_softmax_lse)),
            "flash_attn_varlen_func",
            q,
            k,
            v,
            (cu_seqlens_q, cu_seqlens_k, *sink_args),
            rotary_coeffs,
            rotary_interleaved,
        )
        if return_softmax_lse:
            out, kernel_lse = kernel_output
        else:
            out = kernel_output
        out = jax.lax.slice_in_dim(out, 0, total_q, axis=token_axis)
        if return_softmax_lse:
            # Note (david): the kernel emits lse as (nheads, padded_total_q) in
            # both layouts.
            return out, kernel_lse[:, :total_q]
        else:
            return out
