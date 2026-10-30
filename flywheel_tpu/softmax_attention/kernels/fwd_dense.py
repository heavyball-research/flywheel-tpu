"""Dense (batch) forward kernels: the static schedule and the runtime causal
offset one."""

import dataclasses
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ...core.tpu import NUM_LANES
from ..config import (
    QKVLayout,
    TokenMajorInfo,
    from_head_minor,
    vmem_limit_bytes,
)
from ..fwd_blocks import (
    NUM_KV_STAGES,
    NUM_O_SLOTS,
    NUM_PENDING_FIELDS,
    NUM_SEMS,
)
from ..fwd_call import (
    check_qkv_operands,
    check_rotary_operands,
    rotary_fields,
    staged_shape,
)
from ..fwd_pipeline import fwd_body
from ..schedule import (
    DenseFwdParams,
    make_dense_fwd_schedule,
    make_runtime_offset_dense_fwd_schedule,
)
from ..traits import FwdTraits
from ..variant import Variant, VariantOperands


def flash_fwd_kernel(
    *refs: jax.Array,
    traits: FwdTraits,
    variant: Variant,
    dense_params: DenseFwdParams,
) -> None:
    if variant.has_sinks:
        sinks_ref, q_hbm, k_hbm, v_hbm, *refs = refs
    else:
        sinks_ref = None
        q_hbm, k_hbm, v_hbm, *refs = refs
    fwd_body(
        traits,
        variant,
        make_dense_fwd_schedule(dense_params),
        q_hbm,
        k_hbm,
        v_hbm,
        refs,
        operands=VariantOperands(sinks_ref=sinks_ref),
    )


def dense_scratch_shapes(
    traits: FwdTraits,
    *,
    q_dtype: jnp.dtype,
    k_dtype: jnp.dtype,
    v_dtype: jnp.dtype,
    rotary_dtype: jnp.dtype | None,
    return_lse: bool,
) -> list[pl.MemoryRef]:
    """Every scratch buffer of one dense build, in the order fwd_body
    unpacks them: Q, K and V stages, Q and (under rotary_k) K rotary
    coefficients, out stage, row max, row sum, f32 accumulator, pending
    probs, pending V tile, the SMEM guard flags, block parity and pending-PV
    fields, the DMA semaphores and the lse rows of each folded head.

    Q, K, V, their rotary coefficients and the pending V tile are staged in
    traits.qkv_layout, the out stage and the accumulator in traits.o_layout.
    rotary_dtype is None without rotary."""
    bq, bkv = traits.bq, traits.bkv
    bq_compute, bkv_compute = traits.bq_compute, traits.bkv_compute
    num_stages = traits.num_stages
    qkv_layout, o_layout = traits.qkv_layout, traits.o_layout
    head_dim_qk, head_dim_v = traits.head_dim_qk, traits.head_dim_v
    head_fold = traits.head_fold
    rotary_dim, rotary_k = traits.rotary_dim, traits.rotary_k
    num_rotary_streams = (1 + int(rotary_k)) if rotary_dim else 0
    folded_bq = traits.group_heads * bq
    vmem = pltpu.VMEM

    scratch_shapes = [
        vmem(
            staged_shape(traits, bq, head_dim_qk, qkv_layout, num_stages),
            q_dtype,
        ),
        vmem(
            staged_shape(traits, bkv, head_dim_qk, qkv_layout, NUM_KV_STAGES),
            k_dtype,
        ),
        vmem(
            staged_shape(traits, bkv, head_dim_v, qkv_layout, NUM_KV_STAGES),
            v_dtype,
        ),
    ]
    if rotary_dim:
        scratch_shapes += [
            vmem(
                (
                    stages,
                    traits.rotary_groups,
                    *from_head_minor((block, rotary_dim), qkv_layout),
                ),
                rotary_dtype,
            )
            for block, stages in (
                ((bq, num_stages), (bkv, NUM_KV_STAGES))
                if rotary_k
                else ((bq, num_stages),)
            )
        ]
    # Note (david): a lane-paired fold's pending V tile is its pair's 128 lanes,
    # as the forward loop loads it.
    v_tile_width = NUM_LANES if traits.is_lane_paired else head_dim_v
    scratch_shapes += [
        vmem(
            staged_shape(traits, bq, head_dim_v, o_layout, num_stages), q_dtype
        ),
        vmem((folded_bq, NUM_LANES), jnp.float32),
        vmem((2, folded_bq, NUM_LANES), jnp.float32),
        vmem(
            (NUM_O_SLOTS, *from_head_minor((folded_bq, head_dim_v), o_layout)),
            jnp.float32,
        ),
        vmem((bq_compute, bkv_compute), jnp.float32),
        vmem(from_head_minor((bkv_compute, v_tile_width), qkv_layout), v_dtype),
        pltpu.SMEM((head_fold,), jnp.int32),
        pltpu.SMEM((1,), jnp.int32),
        pltpu.SMEM((NUM_PENDING_FIELDS,), jnp.int32),
        pltpu.SemaphoreType.DMA(
            (NUM_SEMS + num_rotary_streams, max(num_stages, NUM_KV_STAGES))
        ),
    ]
    if return_lse:
        scratch_shapes.append(vmem((head_fold, bq), jnp.float32))
    return scratch_shapes


def dense_pallas_call(
    kernel: Callable[..., None],
    smem_operands: list[jax.Array],
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    traits: FwdTraits,
    variant: Variant,
    *,
    kernel_name: str,
    interpret: bool,
    rotary: tuple[jax.Array, jax.Array | None] | None = None,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Validate q/k/v against traits, allocate the dense scratch and run a
    dense kernel.

    kernel is the mode kernel with its schedule already bound; traits and
    variant are bound here. smem_operands lead the kernel's inputs, among them
    the variant's traced values, which the kernel reads into VariantOperands.
    rotary is prepare_rotary's packed (batch, tokens, head_dim) Q and K pair
    (K None when K arrives rotated) that traits' rotary fields describe.
    Returns o shaped like q (head_dim_v wide), or (o, lse) with a (num_q_heads,
    q_seq_len) float32 natural-log lse.
    """
    assert not (traits.is_paged or traits.is_per_seq), traits
    bq, bkv = traits.bq, traits.bkv
    qkv_layout, o_layout = traits.qkv_layout, traits.o_layout
    num_q_heads, head_dim_v = traits.num_q_heads, traits.head_dim_v
    transposed_pv = traits.transposed_pv
    token_major = traits.token_major
    return_lse = variant.return_lse
    assert (rotary is None) == (traits.rotary_dim == 0), traits.rotary_dim

    q_seq_len, kv_seq_len = check_qkv_operands(q, k, v, traits)

    # Note (david): a per-seq schedule clamps its windows to the axis end and
    # checks that each block fits, so only the dense grid needs the
    # divisibility.
    if q_seq_len % bq or kv_seq_len % bkv:
        raise ValueError(f"{bq=} and {bkv=} must divide the sequence lengths.")
    rotary_dtype, rotary_operands = check_rotary_operands(
        rotary, traits, q.dtype, q_seq_len, kv_seq_len
    )
    scratch_shapes = dense_scratch_shapes(
        traits,
        q_dtype=q.dtype,
        k_dtype=k.dtype,
        v_dtype=v.dtype,
        rotary_dtype=rotary_dtype,
        return_lse=return_lse,
    )

    if qkv_layout == QKVLayout.SEQ_MINOR:
        q_in, k_in, v_in, *rotary_in = (
            operand.swapaxes(-1, -2) for operand in (q, k, v, *rotary_operands)
        )
    else:
        q_in, k_in, v_in, *rotary_in = (q, k, v, *rotary_operands)
    kernel_inputs = [q_in, k_in, v_in, *rotary_in]
    smem_spec = pl.BlockSpec(memory_space=pltpu.SMEM)
    any_spec = pl.BlockSpec(memory_space=pl.ANY)
    in_specs = [smem_spec] * len(smem_operands) + [any_spec] * len(
        kernel_inputs
    )

    if token_major is None:
        o_shape = (
            num_q_heads,
            *from_head_minor((q_seq_len, head_dim_v), o_layout),
        )
    else:
        o_shape = (
            *q.shape[:-2],
            q_seq_len,
            token_major.num_q_heads * head_dim_v,
        )
    lse_shape = (num_q_heads, q_seq_len)
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
            partial(kernel, traits=traits, variant=variant),
            in_specs=in_specs,
            out_specs=out_specs,
            out_shape=out_shape,
            scratch_shapes=scratch_shapes,
            name=kernel_name,
            compiler_params=pltpu.CompilerParams(
                vmem_limit_bytes=vmem_limit_bytes()
            ),
            interpret=pltpu.InterpretParams() if interpret else False,
        )(*smem_operands, *kernel_inputs)
    o_kernel, lse_kernel = outputs if return_lse else (outputs, None)
    # Note (david): transposed_pv emits (heads, head_dim_v, seq); for d64 the
    # swap back is the {1,2,0} layout XLA already prefers, so it folds into a
    # bitcast.
    o = o_kernel.swapaxes(-1, -2) if transposed_pv else o_kernel
    if not return_lse:
        return o
    else:
        return o, lse_kernel


def check_dense_sinks(
    sinks: jax.Array,
    num_flat_heads: int,
    heads_outer_batch: int | None,
    token_major: TokenMajorInfo | None,
) -> None:
    """Raise unless sinks is (nheads,), one logit per sequence head."""
    # Note (deepolar): a head-major kernel sees batch * nheads flat heads, so
    # only heads-outer and token-major pin nheads. Batch-outer cannot tell
    # nheads from batch, so it only checks that the length divides the flat
    # heads: every flat head then reads inside the sinks, but a wrong length
    # that divides them passes.
    if token_major is not None:
        is_valid = sinks.shape == (token_major.num_q_heads,)
    elif heads_outer_batch is not None:
        is_valid = sinks.shape == (num_flat_heads // heads_outer_batch,)
    else:
        is_valid = (
            sinks.ndim == 1
            and sinks.shape[0] > 0
            and num_flat_heads % sinks.shape[0] == 0
        )
    if not is_valid:
        raise ValueError(
            "sinks must be (nheads,), one logit per sequence head, for"
            f" {num_flat_heads} flat q heads; got {sinks.shape}."
        )


def flash_attn_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    sinks: jax.Array | None = None,
    *,
    dense_params: DenseFwdParams,
    traits: FwdTraits,
    variant: Variant,
    interpret: bool,
    rotary: tuple[jax.Array, jax.Array | None] | None = None,
    rotary_interleaved: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    variant.check_sinks_operand(sinks)
    if sinks is not None:
        check_dense_sinks(
            sinks,
            q.shape[0],
            dense_params.heads_outer_batch,
            traits.token_major,
        )
    smem_operands = [] if sinks is None else [variant.sink_logits(sinks)]
    return dense_pallas_call(
        partial(flash_fwd_kernel, dense_params=dense_params),
        smem_operands,
        q,
        k,
        v,
        dataclasses.replace(
            traits, **rotary_fields(rotary, rotary_interleaved)
        ),
        variant,
        kernel_name="flash_attn_mha_static_t1_guarded_fwd",
        interpret=interpret,
        rotary=rotary,
    )


def flash_fwd_runtime_offset_kernel(
    offset_ref: jax.Array,
    q_hbm: jax.Array,
    k_hbm: jax.Array,
    v_hbm: jax.Array,
    *refs: jax.Array,
    traits: FwdTraits,
    variant: Variant,
    dense_params: DenseFwdParams,
) -> None:
    runtime_offset = offset_ref[0]
    fwd_body(
        traits,
        variant,
        make_runtime_offset_dense_fwd_schedule(dense_params, runtime_offset),
        q_hbm,
        k_hbm,
        v_hbm,
        refs,
        operands=VariantOperands(causal_offset=runtime_offset),
    )


def flash_attn_forward_runtime_offset(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    runtime_offset: jax.Array,
    *,
    dense_params: DenseFwdParams,
    traits: FwdTraits,
    variant: Variant,
    interpret: bool,
    rotary: tuple[jax.Array, jax.Array | None] | None = None,
    rotary_interleaved: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    variant.check_sinks_operand(None)
    offset = jnp.asarray(runtime_offset, jnp.int32).reshape(1)
    return dense_pallas_call(
        partial(flash_fwd_runtime_offset_kernel, dense_params=dense_params),
        [offset],
        q,
        k,
        v,
        dataclasses.replace(
            traits, **rotary_fields(rotary, rotary_interleaved)
        ),
        variant,
        kernel_name="flash_attn_mha_runtime_offset_fwd",
        interpret=interpret,
        rotary=rotary,
    )
