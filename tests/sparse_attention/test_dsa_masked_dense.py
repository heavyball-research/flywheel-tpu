"""DSA masked-dense MLA against the dense MLA reference with each token's
top-k selection as the mask."""

import jax
import jax.numpy as jnp
import pytest

from flywheel_tpu.sparse_attention.dsa.masked_dense import dsa_masked_dense
from tests.softmax_attention.mla_reference import SM_SCALE, check_is_allclose
from tests.sparse_attention.dsa_reference import (
    DsaParam,
    dsa_split_cache,
    make_dsa_case,
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

PARAMS = [
    # Decode: one token a request, one of them within top_k positions.
    DsaParam(q_lens=(1, 1, 1, 1), prefix_lens=(3000, 9000, 70, 20000)),
    # Ragged prefill: one request within top_k from an empty cache, two past
    # it after long prefixes.
    DsaParam(q_lens=(300, 1, 77), prefix_lens=(0, 5000, 2100), seed=1),
    # Slots past their token, and padding rows after the batch.
    DsaParam(
        q_lens=(40, 3),
        prefix_lens=(2030, 100),
        has_stray_indices=True,
        padded_tokens=5,
        seed=2,
    ),
    # Every request within top_k: no bitmap, slots past the token ignored.
    DsaParam(
        q_lens=(64, 200),
        prefix_lens=(1900, 0),
        has_stray_indices=True,
        seed=3,
    ),
    # A chunk that crosses top_k positions: its early tokens' slots past
    # their first position + 1 stale, which the bitmap must ignore.
    DsaParam(
        q_lens=(600, 1),
        prefix_lens=(1700, 5000),
        h_q=64,
        has_stale_slots=True,
        seed=8,
    ),
    # Fewer heads and a small page; a shorter top-k.
    DsaParam(
        q_lens=(9, 2), prefix_lens=(4000, 300), h_q=16, page_size=16, seed=4
    ),
    DsaParam(
        q_lens=(5, 30),
        prefix_lens=(700, 6000),
        page_size=256,
        top_k=256,
        seed=5,
    ),
]


def param_id(param: DsaParam) -> str:
    return "-".join([
        "q" + "_".join(map(str, param.q_lens)),
        "p" + "_".join(map(str, param.prefix_lens)),
        f"h{param.h_q}",
        f"ps{param.page_size}",
        f"k{param.top_k}",
        f"pad{param.padded_tokens}",
        "stray" if param.has_stray_indices else "clean",
        "stale" if param.has_stale_slots else "fresh",
    ])


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
