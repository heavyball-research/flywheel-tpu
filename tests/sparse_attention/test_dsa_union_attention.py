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

from flywheel_tpu.sparse_attention.dsa.sparse_mla import chunk_starts
from flywheel_tpu.sparse_attention.dsa.union_attention import (
    attend_given,
    chunk_union_slots,
    max_unions,
    token_unions,
    union_pass,
)
from tests.softmax_attention.mla_reference import SM_SCALE, check_is_allclose
from tests.sparse_attention.dsa_reference import (
    dsa_split_cache,
    grid_params,
    make_dsa_case,
    param_id,
    sparse_mla_reference,
)

if jax.default_backend() != "tpu":
    pytest.skip("the DSA kernels need a SparseCore", allow_module_level=True)

# Note (david): test_dsa_sparse_mla's tolerances.
OUT_TOL = dict(abs_tol=1.8e-2, rel_tol=2.01 / 128, cos_diff_tol=5.5e-4)
# Note (david): a few selected slots hold -1 or positions past their token,
# and the slots past each token's positions name positions it sees.
PARAMS = grid_params(h_q=64, has_dropped_slots=True, has_stale_slots=True)
# The dropped request leaves unions without tokens between the kept ones.
KEPT_REQUESTS = (0, 2)
# Note (david): chunks whose window ends cut a kept request's union; the last
# chunk overlaps the one before it.
MAX_CHUNK_TOKENS = 64


@pytest.mark.parametrize("force_replay", [False, True])
@pytest.mark.parametrize("param", PARAMS, ids=param_id)
def test_union_subset(param, force_replay):
    """Only the kept requests' unions are attended, their rows right, also
    through the replay an overflowing union takes, and every other row keeps
    previous_out's."""
    case = make_dsa_case(param)
    out_ref, _ = sparse_mla_reference(case)
    nope_cache, rope_cache = dsa_split_cache(case.kv_cache)
    num_tokens, batch = case.q_nope.shape[0], case.kv_lens.shape[0]
    num_unions = max_unions(num_tokens, batch, 1)
    unions = token_unions(
        jnp.asarray(case.cu_q_lens),
        jnp.asarray(case.kv_lens),
        jnp.ones((batch,), bool),
        num_unions,
    )
    is_kept = jnp.isin(unions.request, jnp.asarray(KEPT_REQUESTS))
    unions = unions._replace(
        num_tokens=jnp.where(is_kept, unions.num_tokens, 0)
    )
    chunk_tokens = min(MAX_CHUNK_TOKENS, num_tokens)
    starts = chunk_starts(num_tokens, chunk_tokens)
    passes = functools.partial(
        union_pass,
        unions=unions,
        starts=starts,
        chunk_tokens=chunk_tokens,
        index_starts=[0] * len(starts),
        num_slots=chunk_union_slots(chunk_tokens, batch, 1, num_unions),
        top_k=case.topk_indices.shape[1],
        sm_scale=SM_SCALE,
        max_seqlen_k=int(case.kv_lens.max()),
        force_replay=force_replay,
    )

    @jax.jit
    def attend(
        q_nope, q_pe, nope_cache, rope_cache, block_table, indices, out, scale
    ):
        return attend_given(
            passes(
                q_nope,
                q_pe,
                nope_cache,
                rope_cache,
                block_table,
                kv_scale=scale,
            ),
            indices,
            out,
        )

    previous_out = jnp.ones(case.q_nope.shape, jnp.bfloat16)
    out = attend(
        jnp.asarray(case.q_nope),
        jnp.asarray(case.q_pe),
        jnp.asarray(nope_cache),
        jnp.asarray(rope_cache),
        jnp.asarray(case.block_table),
        jnp.asarray(case.topk_indices),
        previous_out,
        jnp.float32(case.kv_scale),
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
