"""Differentiation rules around the forward kernels."""

import functools
from collections.abc import Callable

import jax

from .dispatch import resolve_bwd_config
from .kernels.bwd import flash_attn_bwd


def attn_with_vjp(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    build_kernel: Callable[..., functools.partial],
    softmax_scale: float,
    causal: bool,
    causal_offset: int,
    interpret: bool,
    bwd_unsupported: str,
    token_major: bool,
    head_dim: int | None,
) -> jax.Array:
    # Note (david): the forward kernel fuses q_scale into Q, so the backward
    # differentiates the raw q/k/v against the true softmax_scale. Only the
    # custom_vjp forward asks the kernel for lse, so the primal skips its
    # compute and DMA.
    @jax.custom_vjp
    def _attn(q, k, v):
        return build_kernel(return_lse=False)(q, k, v)

    def _attn_fwd(q, k, v):
        out, lse = build_kernel(return_lse=True)(q, k, v)
        return out, (q, k, v, out, lse)

    def _attn_bwd(residuals, dout):
        # Note (david): raise only when a gradient is requested, so forward-only
        # configurations still run their forward.
        if bwd_unsupported:
            raise NotImplementedError(
                f"The TPU backward kernel does not support {bwd_unsupported};"
                " that configuration is forward-only."
            )
        block_sizes, head_fold = resolve_bwd_config(
            *residuals[:3],
            causal=causal,
            causal_offset=causal_offset,
            token_major=token_major,
            head_dim=head_dim,
        )
        return flash_attn_bwd(
            *residuals,
            dout,
            block_sizes=block_sizes,
            head_fold=head_fold,
            causal=causal,
            causal_offset=causal_offset,
            softmax_scale=softmax_scale,
            interpret=interpret,
            token_major=token_major,
            head_dim=head_dim,
        )

    _attn.defvjp(_attn_fwd, _attn_bwd)
    return _attn(q, k, v)


def forward_only_attn(
    kernel: Callable[..., jax.Array | tuple[jax.Array, jax.Array]],
    reason: str,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array | None,
    kernel_args: tuple[jax.Array, ...],
    rotary: tuple[jax.Array, jax.Array | None] | None,
    rotary_interleaved: bool,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    # Note (david): a bare pallas_call has no differentiation rule, so jax.grad
    # would fail deep inside the tracer instead of naming the unsupported
    # feature. The rotary coefficients are primal inputs so that table gradients
    # raise as well.
    @jax.custom_vjp
    def _forward(q, k, v, rotary):
        if rotary is None:
            return kernel(q, k, v, *kernel_args)
        else:
            return kernel(
                q,
                k,
                v,
                *kernel_args,
                rotary=rotary,
                rotary_interleaved=rotary_interleaved,
            )

    def _raise_unsupported(*unused_args):
        raise NotImplementedError(
            f"{reason} is forward-only; backward is unsupported."
        )

    _forward.defvjp(_raise_unsupported, _raise_unsupported)
    return _forward(q, k, v, rotary)
