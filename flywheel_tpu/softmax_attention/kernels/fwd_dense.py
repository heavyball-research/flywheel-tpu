"""Dense (batch) forward kernels: the static schedule and the runtime causal
offset one."""

import dataclasses
from functools import partial

import jax
import jax.numpy as jnp

from ..config import TokenMajorInfo
from ..fwd_pipeline import forward_common, fwd_body, rotary_fields
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
    return forward_common(
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
    return forward_common(
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
