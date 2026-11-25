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

"""dsa_attention_from_indices, each request in the mode the rule gives it,
against the reference sparse MLA with each token's selection as the mask."""

import jax
import jax.numpy as jnp
import pytest

from flywheel_tpu.sparse_attention.dsa.attention import (
    dsa_attention_from_indices,
)
from flywheel_tpu.sparse_attention.dsa.routing import (
    MAX_MASKED_DENSE_KV_LEN_WITH_UNION,
    MIN_MASKED_DENSE_Q_LEN,
    MIN_UNION_Q_LEN,
)
from tests.softmax_attention.mla.reference import SM_SCALE, check_is_allclose
from tests.sparse_attention.dsa_reference import (
    LENGTH_CELLS,
    PADDED_TOKENS,
    TOP_KS,
    DsaParam,
    dsa_split_cache,
    grid_params,
    make_dsa_case,
    param_id,
    prefix_lens,
    sparse_mla_reference,
)

if jax.default_backend() != "tpu":
    pytest.skip("the DSA kernels need a SparseCore", allow_module_level=True)

# Note (david): test_dsa_sparse_mla's tolerances.
OUT_TOL = dict(abs_tol=1.8e-2, rel_tol=2.01 / 128, cos_diff_tol=5.5e-4)
# Note (david): the batch of several chunks holds a sparse request next to
# masked-dense requests at 16 and 32 heads and next to union requests at 64;
# at 128 heads its requests all attend sparse. The smaller batches take the
# sparse MLA alone.
HEADS = (16, 32, 64, 128)
# Note (david): a batch on the rule's thresholds: a request under
# MIN_UNION_Q_LEN tokens, one of MIN_MASKED_DENSE_Q_LEN tokens and one of a
# token fewer, both at the masked-dense bound next to the union mode, and one
# of many tokens past that bound. The three modes meet at 64 heads; sparse
# and masked-dense meet at the other head counts, which have no union mode.
MIXED_Q_LENS = (
    MIN_UNION_Q_LEN - 1,
    MIN_MASKED_DENSE_Q_LEN,
    MIN_MASKED_DENSE_Q_LEN - 1,
    641,
)
MIXED_KV_LENS = (
    16384,
    MAX_MASKED_DENSE_KV_LEN_WITH_UNION,
    MAX_MASKED_DENSE_KV_LEN_WITH_UNION,
    12289,
)


def mixed_params(heads: int) -> list[DsaParam]:
    """The batch of MIXED_Q_LENS at every top_k of TOP_KS."""
    return [
        DsaParam(
            q_lens=MIXED_Q_LENS,
            prefix_lens=prefix_lens(MIXED_Q_LENS, MIXED_KV_LENS),
            h_q=heads,
            top_k=top_k,
            padded_tokens=PADDED_TOKENS,
            seed=len(LENGTH_CELLS),
            has_stale_slots=True,
        )
        for top_k in TOP_KS
    ]


# Note (david): the slots past each token's positions name positions it sees.
PARAMS = [
    param
    for heads in HEADS
    for param in (
        *grid_params(h_q=heads, has_stale_slots=True),
        *mixed_params(heads),
    )
]


@pytest.mark.parametrize("param", PARAMS, ids=param_id)
def test_dsa_attention_from_indices(param):
    """The output of every request, each in the rule's mode, matches the
    reference with each token's selection as mask."""
    case = make_dsa_case(param)
    out_ref, _ = sparse_mla_reference(case)
    nope_cache, rope_cache = dsa_split_cache(case.kv_cache)
    out = dsa_attention_from_indices(
        jnp.asarray(case.q_nope),
        jnp.asarray(case.q_pe),
        jnp.asarray(nope_cache),
        jnp.asarray(rope_cache),
        jnp.asarray(case.topk_indices),
        jnp.asarray(case.cu_q_lens),
        jnp.asarray(case.kv_lens),
        jnp.asarray(case.block_table),
        max_seqlen_k=int(case.kv_lens.max()),
        sm_scale=SM_SCALE,
        kv_scale=jnp.float32(case.kv_scale),
    )
    assert check_is_allclose(
        "out", out, out_ref.astype(jnp.bfloat16), **OUT_TOL
    )
