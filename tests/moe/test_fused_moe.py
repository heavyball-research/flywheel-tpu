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

"""fused_moe and fused_moe_softmax_topk against an f32 reference.

The grid is FlashInfer's bf16 fused-MoE grid: hidden 2048, 128 experts,
top-8, 3 and 777 tokens, intermediate 768 and its TP4 and TP8 slices 192 and
96, which prepare_expert_weights pads to 256 and 128. The experts of the
benchmarked models and the options that run other paths of the fused MLP are
added to it. Off TPU the kernels run in the Pallas TPU interpreter, on the
smallest cases.
"""

import pytest

from flywheel_tpu.moe import (
    fused_moe,
    fused_moe_softmax_topk,
    prepare_expert_weights,
)
from tests.moe.reference import (
    INTERPRET,
    TPU_ONLY,
    assert_close,
    make_moe_problem,
    moe_reference,
    softmax_topk_routing,
)

# (tokens, hidden, intermediate, experts, top_k)
SHAPES = [
    pytest.param(3, 2048, 768, 128, 8, id="i768-t3"),
    pytest.param(777, 2048, 768, 128, 8, id="i768-t777", marks=TPU_ONLY),
    pytest.param(3, 2048, 192, 128, 8, id="i192-t3"),
    pytest.param(777, 2048, 192, 128, 8, id="i192-t777", marks=TPU_ONLY),
    pytest.param(3, 2048, 96, 128, 8, id="i96-t3"),
    pytest.param(777, 2048, 96, 128, 8, id="i96-t777", marks=TPU_ONLY),
    pytest.param(
        256, 2048, 768, 128, 8, id="qwen3-30b-a3b-t256", marks=TPU_ONLY
    ),
    pytest.param(
        4096, 2048, 768, 128, 8, id="qwen3-30b-a3b-t4096", marks=TPU_ONLY
    ),
    pytest.param(
        256, 2048, 512, 512, 10, id="qwen3-next-80b-a3b-t256", marks=TPU_ONLY
    ),
    pytest.param(
        4096, 2048, 512, 512, 10, id="qwen3-next-80b-a3b-t4096", marks=TPU_ONLY
    ),
    pytest.param(
        256, 2048, 1408, 256, 8, id="h2048-i1408-e256-k8-t256", marks=TPU_ONLY
    ),
    pytest.param(
        4096,
        2048,
        1408,
        256,
        8,
        id="h2048-i1408-e256-k8-t4096",
        marks=TPU_ONLY,
    ),
]

# (tokens, hidden, intermediate, experts, top_k, activation, options)
OPTIONS = [
    pytest.param(
        777,
        2048,
        768,
        128,
        8,
        "silu",
        dict(weight_block_columns=256),
        id="weight-blocks",
    ),
    pytest.param(
        777,
        2048,
        192,
        128,
        8,
        "silu",
        dict(weight_block_columns=128),
        id="weight-blocks-i192",
    ),
    pytest.param(
        777, 2048, 768, 128, 8, "silu", dict(tail_bucket=False), id="no-tail"
    ),
    pytest.param(777, 2048, 768, 128, 8, "gelu", {}, id="gelu"),
    pytest.param(777, 2048, 96, 128, 8, "gelu", {}, id="gelu-i96"),
]


@pytest.mark.parametrize("tokens, hidden, intermediate, experts, top_k", SHAPES)
def test_fused_moe(tokens, hidden, intermediate, experts, top_k):
    hidden_states, router_logits, w_gate, w_up, w2 = make_moe_problem(
        tokens, hidden, intermediate, experts
    )
    topk_weights, topk_ids = softmax_topk_routing(router_logits, top_k)
    out = fused_moe(
        hidden_states,
        topk_weights,
        topk_ids,
        *prepare_expert_weights(w_gate, w_up, w2),
        interpret=INTERPRET,
    )
    ref = moe_reference(
        hidden_states, topk_weights, topk_ids, w_gate, w_up, w2, "silu"
    )
    assert_close(out, ref)


@pytest.mark.parametrize("tokens, hidden, intermediate, experts, top_k", SHAPES)
def test_fused_moe_softmax_topk(tokens, hidden, intermediate, experts, top_k):
    hidden_states, router_logits, w_gate, w_up, w2 = make_moe_problem(
        tokens, hidden, intermediate, experts
    )
    out = fused_moe_softmax_topk(
        hidden_states,
        router_logits,
        *prepare_expert_weights(w_gate, w_up, w2),
        top_k=top_k,
        interpret=INTERPRET,
    )
    topk_weights, topk_ids = softmax_topk_routing(router_logits, top_k)
    ref = moe_reference(
        hidden_states, topk_weights, topk_ids, w_gate, w_up, w2, "silu"
    )
    assert_close(out, ref)


@TPU_ONLY
@pytest.mark.parametrize(
    "tokens, hidden, intermediate, experts, top_k, activation, options", OPTIONS
)
def test_fused_moe_options(
    tokens, hidden, intermediate, experts, top_k, activation, options
):
    hidden_states, router_logits, w_gate, w_up, w2 = make_moe_problem(
        tokens, hidden, intermediate, experts
    )
    topk_weights, topk_ids = softmax_topk_routing(router_logits, top_k)
    out = fused_moe(
        hidden_states,
        topk_weights,
        topk_ids,
        *prepare_expert_weights(w_gate, w_up, w2),
        activation=activation,
        **options,
    )
    ref = moe_reference(
        hidden_states, topk_weights, topk_ids, w_gate, w_up, w2, activation
    )
    assert_close(out, ref)
