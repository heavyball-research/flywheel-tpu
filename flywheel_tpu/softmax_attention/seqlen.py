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

"""Packed-sequence (cu_seqlens) preparation for the varlen forward kernel."""

import jax
import jax.numpy as jnp


def check_cu_seqlens_pair(
    cu_seqlens_k: jax.Array, cu_seqlens_q: jax.Array
) -> None:
    """Shape rules every packed-sequence build shares; values may be tracers."""
    cu_seqlens_k = jnp.asarray(cu_seqlens_k)
    cu_seqlens_q = jnp.asarray(cu_seqlens_q)
    if cu_seqlens_k.ndim != 1 or cu_seqlens_k.shape != cu_seqlens_q.shape:
        raise ValueError(
            "cu_seqlens_q and cu_seqlens_k must be equal-shaped 1-D arrays (one"
            f" entry per sequence boundary); got {cu_seqlens_q.shape},"
            f" {cu_seqlens_k.shape}."
        )
    if cu_seqlens_k.shape[0] < 2:
        raise ValueError(
            "cu_seqlens needs at least 2 entries (0 and the total); got"
            f" {cu_seqlens_k.shape[0]}."
        )
