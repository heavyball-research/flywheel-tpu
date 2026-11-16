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

"""Random MoE problems, the f32 reference the MoE tests compare against, and
the check they share."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

INTERPRET = jax.default_backend() != "tpu"
TPU_ONLY = pytest.mark.skipif(
    INTERPRET, reason="too large for the Pallas interpreter"
)
# Note (david): the outputs are O(1), and the kernels round each scaled row
# to bf16 before the f32 combine and the sum to bf16 after it, a few bf16
# ulps in all.
RTOL = ATOL = 2e-2
ACTIVATIONS = {"silu": jax.nn.silu, "gelu": jax.nn.gelu}


def randn(
    rng: np.random.Generator, shape: tuple[int, ...], scale: float
) -> jax.Array:
    """Normal samples drawn in f32, scaled, cast to bf16."""
    return jnp.asarray(
        rng.standard_normal(shape, dtype=np.float32) * scale
    ).astype(jnp.bfloat16)


def make_moe_problem(
    num_tokens: int,
    hidden: int,
    intermediate: int,
    num_experts: int,
    seed: int = 0,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """bf16 hidden_states [T, hidden], router_logits [T, E], w_gate and w_up
    [E, hidden, intermediate] and w2 [E, intermediate, hidden]."""
    rng = np.random.default_rng(seed)
    # Note (david): weights scaled by 1 / sqrt(fan_in), as models initialize
    # them, keep every activation O(1) at any hidden size, so the sum of a
    # token's top_k rows does not cancel far below the rounding error of its
    # terms.
    hidden_states = randn(rng, (num_tokens, hidden), 1.0)
    router_logits = randn(rng, (num_tokens, num_experts), 1.0)
    w_gate = randn(rng, (num_experts, hidden, intermediate), hidden**-0.5)
    w_up = randn(rng, (num_experts, hidden, intermediate), hidden**-0.5)
    w2 = randn(rng, (num_experts, intermediate, hidden), intermediate**-0.5)
    return hidden_states, router_logits, w_gate, w_up, w2


def softmax_topk_routing(
    router_logits: jax.Array, top_k: int
) -> tuple[jax.Array, jax.Array]:
    """(topk_weights, topk_ids): lax.top_k's experts and the f32 softmax over
    their logits."""
    top_logits, topk_ids = jax.lax.top_k(router_logits, top_k)
    return jax.nn.softmax(top_logits.astype(jnp.float32), axis=-1), topk_ids


def moe_reference(
    hidden_states: jax.Array,
    topk_weights: jax.Array,
    topk_ids: jax.Array,
    w_gate: jax.Array,
    w_up: jax.Array,
    w2: jax.Array,
    activation: str,
) -> jax.Array:
    """The f32 MoE, out[t] = sum_k w[t, k] * mlp_{e[t, k]}(h[t]), by grouped
    matmuls over the (token, k) pairs sorted by expert, with the intermediate
    rounded to bf16 where the kernel rounds it.

    bf16 x bf16 products are exact in f32, so the f32 accumulation leaves
    only the summation order to differ from the kernel.
    """
    num_tokens, top_k = topk_ids.shape
    flat_ids = topk_ids.reshape(-1)
    order = jnp.argsort(flat_ids, stable=True)
    group_sizes = jnp.bincount(flat_ids, length=w2.shape[0]).astype(jnp.int32)
    rows = hidden_states[order // top_k]
    gate = jax.lax.ragged_dot(
        rows, w_gate, group_sizes, preferred_element_type=jnp.float32
    )
    up = jax.lax.ragged_dot(
        rows, w_up, group_sizes, preferred_element_type=jnp.float32
    )
    activated = (ACTIVATIONS[activation](gate) * up).astype(jnp.bfloat16)
    down = jax.lax.ragged_dot(
        activated, w2, group_sizes, preferred_element_type=jnp.float32
    )
    sorted_weights = topk_weights.reshape(-1)[order].astype(jnp.float32)
    scaled = down * sorted_weights[:, None]
    unsorted = jnp.zeros_like(scaled).at[order].set(scaled)
    return unsorted.reshape(num_tokens, top_k, -1).sum(axis=1)


def assert_close(out: jax.Array, ref: jax.Array) -> None:
    out_f32, ref_f32 = out.astype(jnp.float32), ref.astype(jnp.float32)
    max_abs_err = float(jnp.max(jnp.abs(out_f32 - ref_f32)))
    assert jnp.allclose(
        out_f32, ref_f32, rtol=RTOL, atol=ATOL
    ), f"max abs err {max_abs_err}"
