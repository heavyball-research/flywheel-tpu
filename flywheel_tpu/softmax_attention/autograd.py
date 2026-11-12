# Copyright 2026 heavyball-research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Differentiation rules around the forward kernels."""

import functools
from collections.abc import Callable

import jax

from flywheel_tpu.softmax_attention.dispatch import resolve_bwd_config
from flywheel_tpu.softmax_attention.kernels.bwd import flash_attn_bwd
from flywheel_tpu.softmax_attention.variant import Variant


def attn_with_vjp(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    build_kernel: Callable[..., functools.partial],
    variant: Variant,
    interpret: bool,
    token_major: bool,
    head_dim: int | None,
) -> jax.Array:
    # Note (david): the forward kernel fuses q_scale into Q, so the backward
    # differentiates the raw q/k/v against the true softmax_scale. Only the
    # custom_vjp forward asks the kernel for lse, so the primal skips its
    # compute and DMA.
    @jax.custom_vjp
    def _attn(q, k, v):
        return build_kernel(variant=variant.with_lse(False))(q, k, v)

    def _attn_fwd(q, k, v):
        out, lse = build_kernel(variant=variant.with_lse(True))(q, k, v)
        return out, (q, k, v, out, lse)

    def _attn_bwd(residuals, dout):
        # Note (david): raise only when a gradient is requested, so forward-only
        # configurations still run their forward.
        variant.check_backward()
        block_sizes, head_fold = resolve_bwd_config(
            *residuals[:3],
            variant=variant,
            token_major=token_major,
            head_dim=head_dim,
        )
        return flash_attn_bwd(
            *residuals,
            dout,
            block_sizes=block_sizes,
            head_fold=head_fold,
            variant=variant,
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
