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

"""Wrapper-side pieces the forward kernels share around their pallas_call.

rotary_fields turns prepare_rotary's packed coefficient pair into the
FwdTraits rotary fields of a dense, varlen or paged build. The dense and
varlen kernels, which read q, k and v as their own contiguous arrays, also
share the operand checks against their traits and the shape of a staged q,
k, v or out scratch buffer.
"""

import jax
import jax.numpy as jnp

from flywheel_tpu.softmax_attention.config import QKVLayout, from_head_minor
from flywheel_tpu.softmax_attention.traits import FwdTraits


def rotary_fields(
    rotary: tuple[jax.Array, jax.Array | None] | None,
    rotary_interleaved: bool,
) -> dict[str, int | bool]:
    """The FwdTraits rotary fields of prepare_rotary's packed (batch, tokens,
    head_dim) Q and K coefficient pair (K None when K arrives rotated); none
    without rotary."""
    if rotary is None:
        return {}
    if len(rotary) != 2:
        raise ValueError("rotary must contain Q and K coefficient arrays.")
    rotary_q_coeff, rotary_k_coeff = rotary
    rotary_k = rotary_k_coeff is not None
    if (
        rotary_q_coeff.ndim != 3
        or rotary_q_coeff.shape[0] == 0
        or (
            rotary_k
            and (
                rotary_k_coeff.ndim != 3
                or rotary_q_coeff.shape[0] != rotary_k_coeff.shape[0]
                or rotary_q_coeff.shape[2] != rotary_k_coeff.shape[2]
            )
        )
    ):
        raise ValueError(
            "rotary coefficients must be (batch, tokens, head_dim)."
        )
    if rotary_q_coeff.shape[2] <= 0 or (
        rotary_k and rotary_k_coeff.dtype != rotary_q_coeff.dtype
    ):
        raise ValueError("Invalid rotary dimension or coefficient dtype.")
    return {
        "rotary_dim": rotary_q_coeff.shape[2],
        "rotary_batch": rotary_q_coeff.shape[0],
        "rotary_interleaved": rotary_interleaved,
        "rotary_k": rotary_k,
    }


def check_qkv_operands(
    q: jax.Array, k: jax.Array, v: jax.Array, traits: FwdTraits
) -> tuple[int, int]:
    """Raise unless q, k and v match traits' heads, head widths and layout
    (token-major or head-major); returns the q and kv sequence lengths."""
    num_q_heads, num_kv_heads = traits.num_q_heads, traits.num_kv_heads
    head_dim_qk, head_dim_v = traits.head_dim_qk, traits.head_dim_v
    token_major = traits.token_major

    if k.shape[:-1] != v.shape[:-1]:
        raise ValueError(
            f"'key' {k.shape} and 'value' {v.shape} must have the same leading"
            " dimensions."
        )

    if token_major is not None:
        batch = token_major.batch
        q_heads_per_row = token_major.num_q_heads
        kv_heads_per_row = token_major.num_kv_heads
        expected_rank = 2 if batch is None else 3
        if (
            q.ndim != expected_rank
            or k.ndim != expected_rank
            or v.ndim != expected_rank
        ):
            raise ValueError(
                f"token-major with batch={batch} expects rank-{expected_rank}"
                f" q/k/v; got {q.shape=}, {k.shape=}, {v.shape=}."
            )
        if batch is not None and (q.shape[0] != batch or k.shape[0] != batch):
            raise ValueError(
                f"q/k lead with batch={batch}; got {q.shape=}, {k.shape=}."
            )
        if q.shape[-1] != q_heads_per_row * head_dim_qk:
            raise ValueError(
                f"q's fused width {q.shape[-1]} != num_heads {q_heads_per_row}"
                f" * head_dim {head_dim_qk}."
            )
        if k.shape[-1] != kv_heads_per_row * head_dim_qk:
            raise ValueError(
                f"k's fused width {k.shape[-1]} != num_kv_heads"
                f" {kv_heads_per_row} * head_dim {head_dim_qk}."
            )
        if v.shape[-1] != kv_heads_per_row * head_dim_v:
            raise ValueError(
                f"v's fused width {v.shape[-1]} != num_kv_heads"
                f" {kv_heads_per_row} * head_dim_v {head_dim_v}."
            )
        q_seq_len = q.shape[-2]
        kv_seq_len = k.shape[-2]
    else:
        q_seq_len = q.shape[1]

        if (
            q.ndim != 3
            or q.shape[0] != num_q_heads
            or q.shape[2] != head_dim_qk
        ):
            raise ValueError(
                f"'query' {q.shape} must be (num_q_heads={num_q_heads}, seq,"
                f" head_dim={head_dim_qk})."
            )
        if v.ndim != 3 or v.shape[2] != head_dim_v:
            raise ValueError(
                f"'value' {v.shape} must be (num_kv_heads, seq,"
                f" head_dim_v={head_dim_v})."
            )
        if k.ndim != 3:
            raise ValueError(
                f"Expected 3-dim 'key' tensor. Instead got a {k.ndim}-dim one."
            )
        if k.shape[2] != head_dim_qk:
            raise ValueError(
                f"Expected 'key' head dimension to be: {head_dim_qk}. Instead"
                f" got: {k.shape[2]}."
            )
        if k.shape[0] != num_kv_heads:
            raise ValueError(
                f"'key' has {k.shape[0]} heads but the schedule was built for"
                f" {num_kv_heads} kv heads."
            )

        kv_seq_len = k.shape[1]
    return q_seq_len, kv_seq_len


def check_rotary_operands(
    rotary: tuple[jax.Array, jax.Array | None] | None,
    traits: FwdTraits,
    q_dtype: jnp.dtype,
    q_seq_len: int,
    kv_seq_len: int,
) -> tuple[jnp.dtype | None, list[jax.Array]]:
    """Raise unless rotary's coefficients cover q's and k's tokens in q's
    dtype or float32; returns their dtype (None without rotary) and the
    coefficient operands the kernel reads, Q's then K's under traits.rotary_k.
    """
    if rotary is None:
        return None, []
    rotary_q_coeff, rotary_k_coeff = rotary
    if rotary_q_coeff.shape[1] != q_seq_len or (
        traits.rotary_k and rotary_k_coeff.shape[1] != kv_seq_len
    ):
        raise ValueError(
            "rotary coefficients must be (batch, tokens, head_dim)."
        )
    # Note (david): float32 coefficients keep the in-kernel f32 rotation
    # identical to an f32 RoPE computed outside the kernel.
    rotary_dtype = rotary_q_coeff.dtype
    if rotary_dtype not in (q_dtype, jnp.dtype(jnp.float32)):
        raise ValueError("Invalid rotary dimension or coefficient dtype.")
    return rotary_dtype, [coeff for coeff in rotary if coeff is not None]


def staged_shape(
    traits: FwdTraits,
    rows: int,
    head_dim: int,
    layout: QKVLayout,
    stages: int,
) -> tuple[int, ...]:
    """The shape of a dense or varlen build's staged q, k, v or out buffer:
    stages slots, each holding one head group's rows x head_dim tiles in
    layout."""
    group_heads = traits.group_heads
    if group_heads == 1:
        return (stages, *from_head_minor((rows, head_dim), layout))
    elif traits.token_major is not None:
        # Note (david): a token-major group is head_fold * head_dim
        # consecutive lanes of one row, so the fold rides the lane axis and
        # one DMA per stream lands the whole group.
        return (stages, rows, traits.head_fold * head_dim)
    else:
        # Note (david): an explicit fold axis rank-matches the 3-D HBM head
        # slices, so XLA never materializes a reshaped q/k/v copy, which
        # dominated wall time at short sequences when head_dim < 128 needed
        # lane padding.
        return (
            stages,
            group_heads,
            *from_head_minor((rows, head_dim), layout),
        )
