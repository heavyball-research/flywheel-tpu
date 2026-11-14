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

"""dsa_attention end to end, against the reference top-k of the indexer
kernel's scores and the reference sparse MLA over the selected latents.

test_dsa_indexer checks those scores.
"""

import dataclasses
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flywheel_tpu.sparse_attention.dsa.attention import (
    ROUTE_MASKED_DENSE,
    ROUTE_SPARSE,
    ROUTE_UNION,
    dsa_attention,
    dsa_attention_from_indices,
)
from flywheel_tpu.sparse_attention.dsa.indexer import (
    INDEX_HEAD_DIM,
    dsa_indexer_scores,
    index_cache_shape,
)
from tests.softmax_attention.mla_reference import SM_SCALE, check_is_allclose
from tests.sparse_attention import dsa_reference as ref
from tests.sparse_attention.dsa_reference import (
    DsaParam,
    dsa_split_cache,
    make_dsa_case,
    sparse_mla_reference,
)

NUM_INDEX_HEADS = ref.DEEPSEEK_V32_INDEX_HEADS

if jax.default_backend() != "tpu":
    pytest.skip("the DSA kernels need a SparseCore", allow_module_level=True)

# Note (david): test_dsa_sparse_mla's tolerances.
OUT_TOL = dict(abs_tol=1.8e-2, rel_tol=2.01 / 128, cos_diff_tol=5.5e-4)
LSE_TOL = dict(abs_tol=1e-6, rel_tol=8.01 / 65536)
# Note (david): keys of varied magnitude exercise the UE8M0 scales.
KEY_SCALE_RANGE = (0.01, 10.0)

# (param, chunk_tokens) of calls under MIN_ROUTED_TOKENS tokens, which take
# the sparse pipeline.
PIPELINE_CASES = [
    # One decode token: a single chunk; the gather halves its slot chunks.
    (DsaParam(q_lens=(1,), prefix_lens=(9000,), page_size=128), None),
    # Decode: rows short of, at and far past top_k; the last chunk overlaps
    # the one before it.
    (
        DsaParam(
            q_lens=(1,) * 18,
            prefix_lens=(3000, 9000, 70, 20000, 2047, 2048) * 3,
            page_size=128,
            seed=1,
        ),
        16,
    ),
    # Ragged prefill from an empty cache and after long prefixes in the
    # default chunks; T is not a whole number of attention steps.
    (
        DsaParam(
            q_lens=(300, 1, 77),
            prefix_lens=(0, 5000, 2100),
            page_size=128,
            seed=2,
        ),
        None,
    ),
    # Decode and prefill requests, fewer heads, a shorter top-k, padding.
    (
        DsaParam(
            q_lens=(40, 1, 3, 1),
            prefix_lens=(2030, 3000, 100, 9000),
            h_q=16,
            page_size=256,
            top_k=256,
            padded_tokens=5,
            seed=3,
        ),
        16,
    ),
    # More chunks than ATTEND_LAG + 1: attentions run between the later
    # chunks' scores; the last chunk overlaps the one before it.
    (
        DsaParam(
            q_lens=(40, 1, 3, 1),
            prefix_lens=(2030, 3000, 100, 9000),
            h_q=16,
            page_size=256,
            top_k=256,
            padded_tokens=5,
            seed=11,
        ),
        8,
    ),
    # top_k a multiple of 128 but not of ROPE_BLOCK_SLOTS: each call selects
    # top_k and gathers whole rope blocks, the slots past top_k unused.
    (
        DsaParam(
            q_lens=(40, 1, 3, 1),
            prefix_lens=(2030, 3000, 100, 9000),
            h_q=16,
            page_size=256,
            top_k=384,
            padded_tokens=5,
            seed=12,
        ),
        16,
    ),
    # Decode at top_k 128 in larger chunks: rows short of, at and far past
    # top_k.
    (
        DsaParam(
            q_lens=(1,) * 72,
            prefix_lens=(50, 127, 128, 3000, 9000, 20000) * 12,
            page_size=128,
            top_k=128,
            seed=13,
        ),
        64,
    ),
]
# Ragged prefill from an empty cache and after prefixes, every request
# within top_k positions, where the routed path runs no indexer.
WITHIN_TOP_K_PARAM = DsaParam(
    q_lens=(300, 77, 140),
    prefix_lens=(0, 1500, 1900),
    page_size=128,
    padded_tokens=3,
    seed=5,
)
# The idle-rank decode batch: every token at position 0, padding rows.
IDLE_DECODE_PARAM = DsaParam(
    q_lens=(1,) * 13,
    prefix_lens=(0,) * 13,
    h_q=64,
    page_size=128,
    padded_tokens=3,
    seed=8,
)
# A static bound past top_k for a batch within it: the step's lengths rule
# the indexer out on device, and the masked-dense MLA takes bitmaps.
BOUND_PAST_TOP_K = 4096


class RouteCase(NamedTuple):
    param: DsaParam
    route: str | None
    max_seqlen_q: int | None = None
    max_seqlen_k: int | None = None
    return_lse: bool = True


ROUTE_CASES = [
    # Within top_k: the rule, and masked-dense under a static bound past
    # top_k.
    RouteCase(WITHIN_TOP_K_PARAM, None),
    RouteCase(
        WITHIN_TOP_K_PARAM, ROUTE_MASKED_DENSE, max_seqlen_k=BOUND_PAST_TOP_K
    ),
    # Decode at short contexts on masked-dense, its q blocks one token.
    RouteCase(
        DsaParam(
            q_lens=(1,) * 16,
            prefix_lens=tuple(range(60, 1000, 60)),
            h_q=64,
            page_size=128,
            seed=6,
        ),
        ROUTE_MASKED_DENSE,
        max_seqlen_q=1,
    ),
    RouteCase(IDLE_DECODE_PARAM, None, max_seqlen_q=1),
    RouteCase(IDLE_DECODE_PARAM, ROUTE_MASKED_DENSE, max_seqlen_q=1),
    # Prefill requests past top_k alone: masked-dense within
    # MAX_MASKED_DENSE_KV_LEN, the others sparse, as lse rules the union
    # mode out.
    RouteCase(
        DsaParam(
            q_lens=(400, 77, 40),
            prefix_lens=(2500, 3000, 6000),
            page_size=128,
            padded_tokens=2,
            seed=7,
        ),
        None,
    ),
    # Decode requests around a prefill chunk that crosses top_k: the
    # decodes sparse on both sides of the masked-dense chunk, the one past
    # MAX_MASKED_DENSE_KV_LEN too.
    RouteCase(
        DsaParam(
            q_lens=(1,) * 9 + (560,) + (1,) * 7,
            prefix_lens=(3000, 9000, 70, 20000, 2047, 2048, 5000, 0, 8191)
            + (1700,)
            + (4000, 6000, 1, 300, 9100, 2500, 7000),
            h_q=64,
            page_size=128,
            padded_tokens=3,
            seed=9,
        ),
        None,
    ),
    # top_k a multiple of 128 but not of ROPE_BLOCK_SLOTS: the sparse
    # pass's gather takes whole rope blocks; the bitmaps and the
    # masked-dense MLA take top_k as it is.
    RouteCase(
        DsaParam(
            q_lens=(1,) * 9 + (300,) + (1,) * 7,
            prefix_lens=(3000, 9000, 70, 20000, 383, 384, 5000, 0, 8191)
            + (200,)
            + (4000, 6000, 1, 300, 9100, 2500, 7000),
            h_q=64,
            page_size=128,
            top_k=384,
            padded_tokens=3,
            seed=14,
        ),
        None,
    ),
    # Decode requests first, then a chunk: the sparse pass ends at the last
    # decode.
    RouteCase(
        DsaParam(
            q_lens=(1,) * 80 + (480,),
            prefix_lens=tuple(range(2100, 2100 + 80 * 50, 50)) + (2400,),
            h_q=64,
            page_size=128,
            seed=10,
        ),
        None,
    ),
    # A 2048-token chunk past top_k: its top-k and its selection bitmaps
    # compile in one jit.
    RouteCase(
        DsaParam(
            q_lens=(2048,),
            prefix_lens=(1000,),
            h_q=16,
            page_size=128,
            top_k=512,
            seed=11,
        ),
        None,
    ),
    # A DP-padded step: decodes, then more padding rows than a sparse
    # chunk, which the routed path leaves zero.
    RouteCase(
        DsaParam(
            q_lens=(1,) * 64,
            prefix_lens=tuple(range(100, 100 + 64 * 60, 60)),
            h_q=64,
            page_size=128,
            padded_tokens=512,
            seed=23,
        ),
        None,
    ),
    # Every mode in one step, without lse: union requests past
    # MAX_MASKED_DENSE_KV_LEN and under MIN_MASKED_DENSE_Q_LEN around a
    # masked-dense chunk and decodes.
    RouteCase(
        DsaParam(
            q_lens=(1, 3, 300, 80, 1, 205, 16, 1),
            prefix_lens=(20000, 9000, 9000, 2000, 70, 12000, 1500, 5000),
            h_q=64,
            page_size=128,
            padded_tokens=2,
            seed=19,
        ),
        None,
        return_lse=False,
    ),
]
# Batches of every mode in vLLM's order (decodes, then prefill chunks) and
# in others, for the routed attention of given indices: union requests
# whose last union is partial next to other modes' tokens, sparse requests
# on both sides of the dense ones, and padding rows.
MIXED_ORDER_PARAMS = [
    DsaParam(
        q_lens=(1, 1, 3, 1, 300, 80, 16, 205),
        prefix_lens=(20000, 70, 9000, 5000, 9000, 2000, 1500, 12000),
        h_q=64,
        page_size=128,
        padded_tokens=3,
        seed=20,
    ),
    DsaParam(
        q_lens=(300, 1, 80, 3, 16, 205, 1, 1),
        prefix_lens=(9000, 20000, 2000, 9000, 1500, 12000, 70, 5000),
        h_q=64,
        page_size=128,
        padded_tokens=3,
        seed=21,
    ),
    DsaParam(
        q_lens=(205, 80, 300, 1, 3, 16, 1, 1),
        prefix_lens=(12000, 2000, 9000, 20000, 9000, 1500, 70, 5000),
        h_q=64,
        page_size=128,
        seed=22,
    ),
]
# Given indices on the union route.
UNION_PARAMS = [
    # Each request's last union partial, slots past their token or the
    # cache, padding rows.
    DsaParam(
        q_lens=(300, 77, 5),
        prefix_lens=(2500, 3000, 6000),
        h_q=64,
        page_size=128,
        padded_tokens=3,
        has_stray_indices=True,
        seed=15,
    ),
    # A chunk that crosses top_k: its early tokens' stale slots ignored.
    DsaParam(
        q_lens=(600, 1),
        prefix_lens=(1700, 5000),
        h_q=64,
        has_stale_slots=True,
        seed=16,
    ),
    # A shorter top-k, dropped slots.
    DsaParam(
        q_lens=(40, 3, 1, 90),
        prefix_lens=(2030, 100, 9000, 700),
        h_q=64,
        page_size=256,
        top_k=256,
        has_dropped_slots=True,
        seed=17,
    ),
    # More unions than a chunk holds: the last chunk overlaps the one before
    # it.
    DsaParam(
        q_lens=(1100,),
        prefix_lens=(3000,),
        h_q=64,
        page_size=128,
        top_k=512,
        seed=18,
    ),
]
# Note (david): more rows than a SparseCore subcore holds lengths for at
# once, in requests of LARGE_CHUNK_Q_LEN tokens from an empty cache.
LARGE_CHUNK_TOKENS = 32768
LARGE_CHUNK_Q_LEN = 256


def param_id(param: DsaParam) -> str:
    return "-".join([
        "q" + "_".join(map(str, param.q_lens[:4])),
        "p" + "_".join(map(str, param.prefix_lens[:4])),
        f"b{len(param.q_lens)}",
        f"h{param.h_q}",
        f"ps{param.page_size}",
        f"k{param.top_k}",
        f"pad{param.padded_tokens}",
    ])


@dataclasses.dataclass(frozen=True)
class IndexOperands:
    """The indexer's side of a DSA batch, on the latent cache's pages."""

    cache: np.ndarray  # index_cache_shape uint8, every request's keys written
    q_fp8: np.ndarray  # (T, NUM_INDEX_HEADS, INDEX_HEAD_DIM) float8_e4m3fn
    weights: np.ndarray  # (T, NUM_INDEX_HEADS) f32


def make_index_operands(
    rng: np.random.Generator,
    num_tokens: int,
    num_pages: int,
    kv_lens: np.ndarray,
    block_table: np.ndarray,
    page_size: int,
) -> IndexOperands:
    """An index-K cache of random bytes with every cached position of each
    request written as a UE8M0-quantized key, and fp8 queries with their
    weights."""
    cache = rng.integers(
        0, 256, index_cache_shape(num_pages, page_size), dtype=np.uint8
    )
    cu_kv_lens = np.concatenate([[0], np.cumsum(kv_lens)]).astype(np.int32)
    keys = rng.standard_normal((int(cu_kv_lens[-1]), INDEX_HEAD_DIM))
    keys *= rng.uniform(*KEY_SCALE_RANGE, (len(keys), 1))
    k_fp8, k_scale = ref.ue8m0_quantize(keys.astype(np.float32))
    cache = ref.write_index_chunks(
        cache, k_fp8, k_scale, cu_kv_lens, kv_lens, block_table
    )
    queries = rng.standard_normal(
        (num_tokens, NUM_INDEX_HEADS, INDEX_HEAD_DIM)
    ).astype(np.float32)
    q_fp8, q_scale = ref.ue8m0_quantize(queries)
    weights_proj = rng.standard_normal((num_tokens, NUM_INDEX_HEADS))
    return IndexOperands(
        cache=cache,
        q_fp8=q_fp8,
        weights=ref.index_weights(weights_proj.astype(np.float32), q_scale),
    )


def reference_indices(case, index, request_operands, top_k) -> np.ndarray:
    """The reference top-k of the indexer kernel's scores over each row's
    causal prefix."""
    scores = dsa_indexer_scores(
        jnp.asarray(index.q_fp8),
        jnp.asarray(index.weights),
        jnp.asarray(index.cache),
        *request_operands,
        max_seqlen_k=int(case.kv_lens.max()),
    )
    scores = np.asarray(scores).reshape(scores.shape[0], -1)
    lengths = ref.indexer_row_lengths(
        case.cu_q_lens, case.kv_lens, scores.shape[0]
    )
    return ref.indexer_topk(scores, lengths, top_k)


def pipeline_case_id(case) -> str:
    param, chunk_tokens = case
    return f"{param_id(param)}-c{chunk_tokens}"


def route_case_id(case: RouteCase) -> str:
    return "-".join([
        param_id(case.param),
        f"{case.route}",
        f"msq{case.max_seqlen_q}",
        f"msk{case.max_seqlen_k}",
        f"lse{int(case.return_lse)}",
    ])


@pytest.mark.parametrize("case", PIPELINE_CASES, ids=pipeline_case_id)
def test_sparse_pipeline(case):
    """The sparse pipeline's output and selection match the reference."""
    param, chunk_tokens = case
    check_attention(param, chunk_tokens)


@pytest.mark.parametrize("case", ROUTE_CASES, ids=route_case_id)
def test_routes(case):
    """Each route's output and selection match the reference."""
    check_attention(
        case.param,
        None,
        max_seqlen_q=case.max_seqlen_q,
        max_seqlen_k=case.max_seqlen_k,
        route=case.route,
        return_lse=case.return_lse,
    )


def test_large_chunks():
    """A LARGE_CHUNK_TOKENS chunk on the sparse route matches the reference."""
    num_requests = LARGE_CHUNK_TOKENS // LARGE_CHUNK_Q_LEN + 1
    param = DsaParam(
        q_lens=(LARGE_CHUNK_Q_LEN,) * num_requests,
        prefix_lens=(0,) * num_requests,
        h_q=16,
        page_size=256,
        top_k=LARGE_CHUNK_Q_LEN,
        seed=4,
    )
    check_attention(param, LARGE_CHUNK_TOKENS, route=ROUTE_SPARSE)


@pytest.mark.parametrize("param", UNION_PARAMS, ids=param_id)
def test_union_route(param):
    """The union route's output of given indices matches the reference."""
    check_given_indices(param, ROUTE_UNION)


@pytest.mark.parametrize("param", MIXED_ORDER_PARAMS, ids=param_id)
def test_mixed_orders(param):
    """The rule's output of given indices matches the reference whatever the
    order of the requests' modes."""
    check_given_indices(param, None)


def check_given_indices(param: DsaParam, route: str | None) -> None:
    """dsa_attention_from_indices of param's batch through route against
    the reference sparse MLA."""
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
        route=route,
    )
    assert check_is_allclose(
        "out", out, out_ref.astype(jnp.bfloat16), **OUT_TOL
    )


def make_selection(param: DsaParam, max_seqlen_k: int | None):
    """(case, index operands, request operands, reference indices) of
    param's batch, its block table as wide as max_seqlen_k if given."""
    case = make_dsa_case(param)
    if max_seqlen_k is not None:
        width = -(-max_seqlen_k // param.page_size)
        table_pad = ((0, 0), (0, width - case.block_table.shape[1]))
        case = dataclasses.replace(
            case, block_table=np.pad(case.block_table, table_pad)
        )
    rng = np.random.default_rng(param.seed)
    index = make_index_operands(
        rng,
        case.q_nope.shape[0],
        case.kv_cache.shape[0],
        case.kv_lens,
        case.block_table,
        param.page_size,
    )
    request_operands = tuple(
        jnp.asarray(value)
        for value in (case.cu_q_lens, case.kv_lens, case.block_table)
    )
    indices = reference_indices(case, index, request_operands, param.top_k)
    return case, index, request_operands, indices


def check_attention(
    param: DsaParam,
    chunk_tokens: int | None,
    *,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    route: str | None = None,
    return_lse: bool = True,
) -> None:
    """dsa_attention of param's batch against the reference sparse MLA;
    max_seqlen_k defaults to the batch's longest request."""
    case, index, request_operands, indices = make_selection(param, max_seqlen_k)
    out_ref, lse_ref = sparse_mla_reference(
        dataclasses.replace(case, topk_indices=indices)
    )
    nope_cache, rope_cache = dsa_split_cache(case.kv_cache)
    *outputs, selection = dsa_attention(
        jnp.asarray(index.q_fp8),
        jnp.asarray(index.weights),
        jnp.asarray(index.cache),
        jnp.asarray(case.q_nope),
        jnp.asarray(case.q_pe),
        jnp.asarray(nope_cache),
        jnp.asarray(rope_cache),
        *request_operands,
        max_seqlen_k=max_seqlen_k or int(case.kv_lens.max()),
        sm_scale=SM_SCALE,
        kv_scale=jnp.float32(case.kv_scale),
        top_k=param.top_k,
        return_lse=return_lse,
        return_indices=True,
        chunk_tokens=chunk_tokens,
        max_seqlen_q=max_seqlen_q,
        route=route,
    )
    assert check_is_allclose(
        "out", outputs[0], out_ref.astype(jnp.bfloat16), **OUT_TOL
    )
    if return_lse:
        assert check_is_allclose("lse", outputs[1], lse_ref, **LSE_TOL)
    check_selection(case, np.asarray(selection), indices, param.top_k)


def check_selection(case, selection, indices, top_k) -> None:
    """Each real token's returned selection holds the reference's positions
    in its first min(top_k, position + 1) slots, in any order."""
    lengths = ref.indexer_row_lengths(
        case.cu_q_lens, case.kv_lens, selection.shape[0]
    )
    for token, length in enumerate(lengths):
        used = min(top_k, int(length))
        np.testing.assert_array_equal(
            np.sort(selection[token, :used]),
            np.sort(indices[token, :used]),
            err_msg=f"token {token}",
        )
