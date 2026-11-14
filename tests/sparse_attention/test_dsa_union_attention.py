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

"""The union attention of some of a call's unions, against the reference
sparse MLA on their tokens' rows."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flywheel_tpu.sparse_attention.dsa.union_attention import (
    max_unions,
    token_unions,
    union_attention,
    union_selection,
)
from tests.softmax_attention.mla_reference import SM_SCALE, check_is_allclose
from tests.sparse_attention.dsa_reference import (
    DsaParam,
    dsa_split_cache,
    make_dsa_case,
    sparse_mla_reference,
)

if jax.default_backend() != "tpu":
    pytest.skip("the DSA kernels need a SparseCore", allow_module_level=True)

# Note (david): test_dsa_sparse_mla's tolerances.
OUT_TOL = dict(abs_tol=1.8e-2, rel_tol=2.01 / 128, cos_diff_tol=5.5e-4)
# Partial last unions; the dropped requests leave unions without tokens
# between and after the kept ones.
SUBSET_PARAM = DsaParam(
    q_lens=(30, 17, 45, 9),
    prefix_lens=(2500, 4000, 3100, 6000),
    h_q=64,
    page_size=128,
    seed=19,
)
KEPT_REQUESTS = (0, 2)


@pytest.mark.parametrize("force_replay", [False, True])
def test_union_subset(force_replay):
    """Only the kept requests' unions are attended, their rows right, also
    through the replay an overflowing union takes, and every other row keeps
    previous_out's."""
    case = make_dsa_case(SUBSET_PARAM)
    out_ref, _ = sparse_mla_reference(case)
    nope_cache, rope_cache = dsa_split_cache(case.kv_cache)
    num_tokens, batch = case.q_nope.shape[0], case.kv_lens.shape[0]
    unions = token_unions(
        jnp.asarray(case.cu_q_lens),
        jnp.asarray(case.kv_lens),
        jnp.ones((batch,), bool),
        max_unions(num_tokens, batch, 1),
    )
    is_kept = jnp.isin(unions.request, jnp.asarray(KEPT_REQUESTS))
    unions = unions._replace(
        num_tokens=jnp.where(is_kept, unions.num_tokens, 0)
    )
    top_k = case.topk_indices.shape[1]
    selection = union_selection(
        jnp.asarray(case.topk_indices),
        unions,
        max_seqlen_k=int(case.kv_lens.max()),
    )
    attend = jax.jit(
        functools.partial(
            union_attention,
            top_k=top_k,
            sm_scale=SM_SCALE,
            force_replay=force_replay,
        )
    )
    previous_out = jnp.ones(case.q_nope.shape, jnp.bfloat16)
    out = attend(
        jnp.asarray(case.q_nope),
        jnp.asarray(case.q_pe),
        jnp.asarray(nope_cache),
        jnp.asarray(rope_cache),
        selection,
        jnp.asarray(case.block_table),
        previous_out,
        kv_scale=jnp.float32(case.kv_scale),
    )
    rows = np.concatenate([
        np.arange(case.cu_q_lens[request], case.cu_q_lens[request + 1])
        for request in KEPT_REQUESTS
    ])
    assert check_is_allclose(
        "out", out[rows], out_ref[rows].astype(jnp.bfloat16), **OUT_TOL
    )
    other_rows = np.setdiff1d(np.arange(num_tokens), rows)
    np.testing.assert_array_equal(np.asarray(out[other_rows], np.float32), 1.0)
