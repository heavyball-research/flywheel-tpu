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
    ROUTE_MIXED,
    ROUTE_SPARSE,
    dsa_attention,
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
# within top_k positions, where the mixed route runs no indexer.
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


ROUTE_CASES = [
    # Within top_k: the route picked on device, and the mixed route under a
    # static bound past top_k.
    RouteCase(WITHIN_TOP_K_PARAM, None),
    RouteCase(WITHIN_TOP_K_PARAM, ROUTE_MIXED, max_seqlen_k=BOUND_PAST_TOP_K),
    # Decode at short contexts, its q blocks one token.
    RouteCase(
        DsaParam(
            q_lens=(1,) * 16,
            prefix_lens=tuple(range(60, 1000, 60)),
            h_q=64,
            page_size=128,
            seed=6,
        ),
        ROUTE_MIXED,
        max_seqlen_q=1,
    ),
    RouteCase(IDLE_DECODE_PARAM, None, max_seqlen_q=1),
    RouteCase(IDLE_DECODE_PARAM, ROUTE_MIXED, max_seqlen_q=1),
    # Prefill requests past top_k alone: masked-dense, but the tokens in
    # the window.
    RouteCase(
        DsaParam(
            q_lens=(400, 77, 40),
            prefix_lens=(2500, 3000, 6000),
            page_size=128,
            padded_tokens=2,
            seed=7,
        ),
        ROUTE_MIXED,
    ),
    # Decode requests around a prefill chunk that crosses top_k: the decodes
    # before it sparse, the chunk and the decodes after it masked-dense.
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
        ROUTE_MIXED,
    ),
    # top_k a multiple of 128 but not of ROPE_BLOCK_SLOTS: the window's
    # gather takes whole rope blocks; the bitmaps and the masked-dense MLA
    # take top_k as it is.
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
        ROUTE_MIXED,
    ),
    # More decode requests than the mixed route's window holds: the requests
    # past it take masked-dense.
    RouteCase(
        DsaParam(
            q_lens=(1,) * 80 + (480,),
            prefix_lens=tuple(range(2100, 2100 + 80 * 50, 50)) + (2400,),
            h_q=64,
            page_size=128,
            seed=10,
        ),
        ROUTE_MIXED,
    ),
    # A 2048-token chunk past top_k: its top-k and its selection bitmaps
    # compile in one jit.
    RouteCase(
        DsaParam(
            q_lens=(2048,),
            prefix_lens=(1500,),
            h_q=16,
            page_size=128,
            top_k=512,
            seed=11,
        ),
        ROUTE_MIXED,
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
) -> None:
    """dsa_attention of param's batch against the reference sparse MLA;
    max_seqlen_k defaults to the batch's longest request."""
    case, index, request_operands, indices = make_selection(param, max_seqlen_k)
    out_ref, lse_ref = sparse_mla_reference(
        dataclasses.replace(case, topk_indices=indices)
    )
    nope_cache, rope_cache = dsa_split_cache(case.kv_cache)
    out, lse, selection = dsa_attention(
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
        return_lse=True,
        return_indices=True,
        chunk_tokens=chunk_tokens,
        max_seqlen_q=max_seqlen_q,
        route=route,
    )
    assert check_is_allclose(
        "out", out, out_ref.astype(jnp.bfloat16), **OUT_TOL
    )
    assert check_is_allclose("lse", lse, lse_ref, **LSE_TOL)
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
