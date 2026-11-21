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

"""DSA masked-dense MLA against the dense MLA reference with each token's
top-k selection as the mask."""

import jax
import jax.numpy as jnp
import pytest

from flywheel_tpu.sparse_attention.dsa.masked_dense import dsa_masked_dense
from tests.softmax_attention.mla_reference import SM_SCALE, check_is_allclose
from tests.sparse_attention.dsa_reference import (
    dsa_split_cache,
    grid_params,
    make_dsa_case,
    param_id,
    sparse_mla_reference,
)

if jax.default_backend() != "tpu":
    pytest.skip(
        "the selection bitmap needs a SparseCore", allow_module_level=True
    )

# Note (david): the dense fp8 MLA tolerances: the probabilities round to fp8
# for P @ V, and the row sums add the unrounded ones.
OUT_TOL = dict(abs_tol=1.8e-2, rel_tol=2.01 / 128, cos_diff_tol=5.5e-4)
LSE_TOL = dict(abs_tol=1e-6, rel_tol=8.01 / 65536)

# Note (david): the slots past each token's positions name positions it sees.
PARAMS = grid_params(has_stale_slots=True)


@pytest.mark.parametrize("param", PARAMS, ids=param_id)
def test_dsa_masked_dense(param):
    """The masked-dense output and lse match the reference with each token's
    selection as mask."""
    case = make_dsa_case(param)
    out_ref, lse_ref = sparse_mla_reference(case)
    nope_cache, rope_cache = dsa_split_cache(case.kv_cache)
    out, lse = dsa_masked_dense(
        jnp.asarray(case.q_nope),
        jnp.asarray(case.q_pe),
        jnp.asarray(nope_cache),
        jnp.asarray(rope_cache),
        jnp.asarray(case.topk_indices),
        jnp.asarray(case.cu_q_lens),
        jnp.asarray(case.kv_lens),
        jnp.asarray(case.block_table),
        sm_scale=SM_SCALE,
        kv_scale=jnp.float32(case.kv_scale),
        max_seqlen_k=int(case.kv_lens.max()),
        max_seqlen_q=max(param.q_lens),
        return_lse=True,
    )
    assert check_is_allclose(
        "out", out, out_ref.astype(jnp.bfloat16), **OUT_TOL
    )
    assert check_is_allclose("lse", lse, lse_ref, **LSE_TOL)
