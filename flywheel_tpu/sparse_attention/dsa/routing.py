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

"""Per-request routing of a DSA layer among its three attention modes.

A request takes a mode by a static rule on its q_len and kv_len alone: few
tokens attend sparse (each token's own gathered rows); more attend in unions
(union_attention.py); many, up to masked_dense_kv_len_bound positions, attend
masked-dense over the paged cache. The rule is a pure function of the step's
lengths, so the layers of a step share it.
"""

import jax
import jax.numpy as jnp

from flywheel_tpu.sparse_attention.dsa.cache_write import token_positions

MODE_SPARSE = 0
MODE_MASKED_DENSE = 1
MODE_UNION = 2
# Note (david): a union costs about one kernel item a request and a sparse
# token one gathered selection, so requests under MIN_UNION_Q_LEN tokens attend
# sparse. Masked-dense pays a whole q block a request and scores every position
# up to the token at every head, so it takes requests of at least
# MIN_MASKED_DENSE_Q_LEN tokens within a bound of positions: next to the union
# mode, MAX_MASKED_DENSE_KV_LEN_WITH_UNION; next to sparse alone, more, by the
# heads.
MIN_UNION_Q_LEN = 5
MIN_MASKED_DENSE_Q_LEN = 64
MAX_MASKED_DENSE_KV_LEN_WITH_UNION = 3072
MAX_MASKED_DENSE_KV_LEN_BY_HEADS = {16: 24576, 32: 15360, 64: 8192, 128: 4096}


def masked_dense_kv_len_bound(num_heads: int, has_union: bool) -> int:
    """The most positions of a request that attends masked-dense."""
    if has_union:
        return MAX_MASKED_DENSE_KV_LEN_WITH_UNION
    else:
        return MAX_MASKED_DENSE_KV_LEN_BY_HEADS[num_heads]


def request_q_lens(cu_q_lens: jax.Array) -> jax.Array:
    return cu_q_lens[1:] - cu_q_lens[:-1]


def cumulative_lens(lens: jax.Array) -> jax.Array:
    """(B + 1,) int32 offsets of consecutive runs of lens tokens."""
    return jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.cumsum(lens)])


def request_modes(
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    *,
    max_masked_dense_kv_len: int,
    has_union: bool,
) -> jax.Array:
    """(B,) int32 mode of each request by the rule, masked-dense within
    max_masked_dense_kv_len positions; without has_union, the requests the
    rule gives the union mode attend sparse."""
    q_lens = request_q_lens(cu_q_lens)
    if has_union:
        union_mode = MODE_UNION
    else:
        union_mode = MODE_SPARSE
    is_masked = (q_lens >= MIN_MASKED_DENSE_Q_LEN) & (
        kv_lens <= max_masked_dense_kv_len
    )
    return jnp.where(
        is_masked,
        MODE_MASKED_DENSE,
        jnp.where(q_lens >= MIN_UNION_Q_LEN, union_mode, MODE_SPARSE),
    ).astype(jnp.int32)


def causal_indices(
    cu_q_lens: jax.Array, kv_lens: jax.Array, num_tokens: int, top_k: int
) -> jax.Array:
    """(T, top_k) int32 top-k of tokens that see at most top_k positions:
    each position the token sees, then -1."""
    _, position = token_positions(cu_q_lens, kv_lens, num_tokens)
    slot = jnp.arange(top_k, dtype=jnp.int32)
    return jnp.where(slot[None, :] <= position[:, None], slot[None, :], -1)
