"""flash_mla_decode against FlashMLA's dense decode test.

mla_reference ports FlashMLA's cases, reference and tolerances. The grid is
FlashMLA's dense correctness grid, run with a bf16 and an fp8 cache, and its
dense corner cases, plus the grid's causal bf16 cells again at few query
heads, cells at batch and page sizes FlashMLA does not reach, and edge cells
that pin kv_lens so that the new tokens straddle a cache word, a cache tile, a
page or a KV block, or make up a request's whole cache. Every cell runs at
both s_dtypes and checks out, lse and the fused append: the new tokens' rows
hold (new_kv_c, new_k_pe, 0) and every other cache row keeps its bits. The
guard cells put one key far past its request's anchor.
"""

import dataclasses

import jax
import jax.numpy as jnp
import pytest

from flywheel_tpu.softmax_attention.kernels.mla_decode import (
    MlaDecodeConfig,
    default_mla_config,
    mla_decode_pallas,
)
from flywheel_tpu.softmax_attention.mla_decode import (
    FP8_CACHE_DTYPE,
    LATENT_DIM,
    MAX_QUERY_TOKENS,
    flash_mla_decode,
)
from tests.softmax_attention.mla_reference import (
    SM_SCALE,
    MlaParam,
    check_is_allclose,
    check_kv_cache,
    logical_cache,
    make_case,
    pages_per_seq,
    quantize_case,
    reference,
)

INTERPRET = jax.default_backend() != "tpu"
PAGE_SIZES = (64, 512)
S_DTYPES = {"f32": jnp.float32, "bf16": jnp.bfloat16}
# FlashMLA's tolerances for out (both sides rounded to bf16) and for lse.
OUT_TOL = dict(abs_tol=8e-4, rel_tol=2.01 / 128, cos_diff_tol=5e-6)
LSE_TOL = dict(abs_tol=1e-6, rel_tol=8.01 / 65536)
# Note (david): FlashMLA's softmax is f32 only, so bf16 scores get their own
# lse tolerance, measured on the TPU, as the Pallas interpreter does not
# reproduce their rounding. Each probability rounds to bf16 before the row
# sum, which only long rows average out, and a row that sees only a few keys
# (a request that holds only its new tokens, causal) has an lse of a few
# rounded scores. out keeps OUT_TOL, as both score dtypes round the
# probabilities to bf16 for P @ V.
BF16_SCORES_LSE_TOL = dict(
    abs_tol=4.3e-3, rel_tol=LSE_TOL["rel_tol"], cos_diff_tol=8.6e-7
)
# Note (david): an fp8 cache's probabilities round to fp8 for P @ V, as MLA
# v2's p_same_dtype_as_v rounds them, which requests of a few keys do not
# average out, so its out takes a wider tolerance. Its lse keeps the bf16
# cache's, as the reference attends the dequantized cache and the row sums
# add the unrounded probabilities.
FP8_OUT_TOL = dict(abs_tol=1.8e-2, rel_tol=2.01 / 128, cos_diff_tol=5.5e-4)
LSE_TOLS = {"f32": LSE_TOL, "bf16": BF16_SCORES_LSE_TOL}
# Note (david): the quick checkpoint (pytest --subset) keeps every small
# f32-score cell and samples the rest: the big cells, the drawn-length cells
# at s_q > 1 or with an fp8 cache, which multiply the grid and its interpret
# time, and the bf16-score cells, which change only the softmax's dtype. The
# cells that pin kv_lens, where the tile, page and block boundaries are, keep
# every f32 cell whatever their size.
BIG_CELL_TOKENS = 8192
SUBSET_STRIDE = 4
BF16 = jnp.dtype(jnp.bfloat16)


def param_id(param: MlaParam) -> str:
    if param.kv_lens is None:
        lengths = [f"sk{param.s_k}", "varlen" if param.is_varlen else "fixed"]
    else:
        lengths = ["kv" + "_".join(str(kv_len) for kv_len in param.kv_lens)]
    return "-".join([
        f"b{param.b}",
        f"sq{param.s_q}",
        *lengths,
        "causal" if param.is_causal else "full",
        "fp8" if param.is_fp8 else "bf16",
        f"ps{param.page_size}",
        f"h{param.h_q}",
        *(["zero"] if param.have_zero_seqlen_k else []),
    ])


def grid(params):
    """params at every s_dtype as (param, s_dtype) pytest params, the big,
    the drawn-length s_q > 1 or fp8, and the bf16-score cells marked for the
    --subset sample."""
    cells = []
    for param in params:
        for s_dtype in S_DTYPES:
            stride = 1
            if param.kv_lens is None:
                if param.b * param.s_k > BIG_CELL_TOKENS:
                    stride *= SUBSET_STRIDE
                if param.s_q > 1:
                    stride *= SUBSET_STRIDE
                if param.is_fp8:
                    stride *= SUBSET_STRIDE
            if s_dtype != "f32":
                stride *= SUBSET_STRIDE
            if stride > 1:
                marks = [pytest.mark.subset_sample(stride)]
            else:
                marks = []
            cells.append(
                pytest.param(
                    param,
                    s_dtype,
                    id=f"{param_id(param)}-s{s_dtype}",
                    marks=marks,
                )
            )
    return cells


# FlashMLA's correctness grid, dense cases only. FlashMLA runs fp8 caches in
# its sparse (top-k) cells only, so the dense fp8 cells are this grid's own.
CORRECTNESS_PARAMS = [
    MlaParam(b, s_q, s_k, is_varlen, is_causal, is_fp8, page_size=page_size)
    for page_size in PAGE_SIZES
    for b in (1, 2, 6, 64)
    for s_q in (1, 2, 4)
    for s_k in (20, 140, 4096)
    for is_varlen in (False, True)
    for is_causal in (False, True)
    for is_fp8 in (False, True)
]
# Note (david): FlashMLA runs at DeepSeek's full head count only, while a
# TP-sharded DeepSeek sees a fraction of the heads per chip, so the grid
# repeats there, causal and bf16.
FEW_HEADS_PARAMS = [
    MlaParam(b, s_q, s_k, is_varlen, True, page_size=page_size, h_q=16)
    for page_size in PAGE_SIZES
    for b in (1, 2, 6, 64)
    for s_q in (1, 2, 4)
    for s_k in (20, 140, 4096)
    for is_varlen in (False, True)
]
# Note (david): FlashMLA's dense corner cases, where about half the requests
# are inactive (kv_lens = 0), at one more s_q. Seed 0 makes the single-request
# batch all inactive. The fp8 cache runs the small batches only: inactive
# requests never reach the fp8 path, and the big batch costs much host work a
# cell.
CORNER_PARAMS = [
    MlaParam(
        b,
        s_q,
        s_k,
        is_varlen=True,
        is_causal=is_causal,
        is_fp8=is_fp8,
        have_zero_seqlen_k=True,
        page_size=page_size,
    )
    for page_size in PAGE_SIZES
    for is_fp8 in (False, True)
    for b, s_k in ((128, 4096), (6, 140), (1, 20))
    for s_q in (1, 2)
    for is_causal in (False, True)
    if not (is_fp8 and b * s_k > BIG_CELL_TOKENS)
]
# Note (david): past a lane tile of requests the lse spans a second lane
# group; a page of one cache tile puts the most pages in a KV block, and a
# page of three tiles is no power of two.
GEOMETRY_PARAMS = [
    param
    for s_q in (1, 4)
    for is_fp8, one_tile, three_tiles in ((False, 16, 48), (True, 32, 96))
    for param in (
        MlaParam(130, s_q, 20, is_varlen=True, is_causal=True, is_fp8=is_fp8),
        *(
            MlaParam(
                6,
                s_q,
                1100,
                is_varlen=True,
                is_causal=True,
                is_fp8=is_fp8,
                page_size=page_size,
                h_q=16,
            )
            for page_size in (one_tile, three_tiles)
        ),
    )
]

# Note (david): a straddle at KV_BLOCK sits on a KV block boundary of the
# kernel for its page size, whatever block it picks: at the block of a request
# far longer than one block, which straddle_param checks is a whole number of
# the blocks the cell's own pages give.
KV_BLOCK = "kv_block"
LONG_REQUEST_TOKENS = 1 << 20


def kv_block_tokens(
    page_size: int,
    is_fp8: bool,
    s_q: int,
    h_q: int,
    max_kv_len: int,
    batch: int,
) -> int:
    """Tokens of the kernel's KV block when make_case's longest request holds
    max_kv_len tokens and the call passes no max_seqlen_k."""
    num_pages = pages_per_seq(max_kv_len, page_size)
    return default_mla_config(
        batch=batch,
        page_size=page_size,
        pages_per_seq=num_pages,
        num_query_rows=s_q * h_q,
        query_tokens=s_q,
        cache_dtype=FP8_CACHE_DTYPE if is_fp8 else BF16,
        return_lse=True,
        max_seqlen_k=num_pages * page_size,
    ).block_kv


def straddle_param(
    boundary: int | str,
    s_q: int,
    page_size: int,
    h_q: int,
    is_causal: bool,
    is_fp8: bool,
) -> MlaParam:
    """A cell whose requests put their s_q new tokens around position
    boundary, a KV block boundary for KV_BLOCK: kv_lens boundary + d for d in
    0, 1, s_q // 2, s_q - 1 and s_q. d = 0 ends the new tokens at boundary,
    d = s_q starts them there, and the rest straddle it."""
    splits = sorted({0, 1, s_q // 2, s_q - 1, s_q})
    if boundary == KV_BLOCK:
        boundary = kv_block_tokens(
            page_size, is_fp8, s_q, h_q, LONG_REQUEST_TOKENS, len(splits)
        )
        kv_lens = tuple(boundary + split for split in splits)
        cell_block = kv_block_tokens(
            page_size, is_fp8, s_q, h_q, kv_lens[-1], len(kv_lens)
        )
        assert boundary % cell_block == 0, (boundary, cell_block)
    else:
        kv_lens = tuple(boundary + split for split in splits)
    return MlaParam(
        len(kv_lens),
        s_q,
        kv_lens[-1],
        is_varlen=False,
        is_causal=is_causal,
        is_fp8=is_fp8,
        page_size=page_size,
        h_q=h_q,
        kv_lens=kv_lens,
    )


# Note (david): the new tokens of a request can straddle a cache tile, a page
# or a KV block, which makes the append patch two tiles of two pages and stage
# them into two items. FlashMLA's random lengths straddle a tile now and then,
# a page rarely and a block almost never, so each (s_q, page_size, boundary)
# pins kv_lens around one: a tile inside a page, a page (on a page of one tile
# every tile boundary is one) and a KV block, at FlashMLA's s_q, an odd s_q
# and MAX_QUERY_TOKENS. Past a block boundary by fewer than s_q - 1 tokens,
# the causal mask of the first query rows reaches into the block before. Every
# boundary runs causal; the KV blocks, where the append and the mask both
# cross staged blocks, run full as well. An fp8 cell's page holds whole fp8
# tiles, and its boundaries put the new tokens at every offset in a word.
BF16_STRADDLES = (
    (4, 64, 48),
    (4, 16, 16),
    (4, 64, 64),
    (4, 512, 512),
    (4, 16, KV_BLOCK),
    (4, 64, KV_BLOCK),
    (4, 512, KV_BLOCK),
    (2, 64, 48),
    (2, 64, 64),
    (2, 64, KV_BLOCK),
    (3, 64, KV_BLOCK),
    (MAX_QUERY_TOKENS, 16, 16),
    (MAX_QUERY_TOKENS, 64, KV_BLOCK),
)
FP8_STRADDLES = (
    (4, 64, 32),
    (4, 64, 48),
    (4, 32, 32),
    (4, 64, 64),
    (4, 512, 512),
    (4, 32, KV_BLOCK),
    (4, 64, KV_BLOCK),
    (4, 512, KV_BLOCK),
    (2, 64, 32),
    (2, 64, 64),
    (2, 64, KV_BLOCK),
    (3, 64, KV_BLOCK),
    (MAX_QUERY_TOKENS, 32, 32),
    (MAX_QUERY_TOKENS, 64, KV_BLOCK),
)
STRADDLE_PARAMS = [
    straddle_param(boundary, s_q, page_size, h_q, is_causal, is_fp8)
    for is_fp8, straddles in ((False, BF16_STRADDLES), (True, FP8_STRADDLES))
    for h_q in (16, 128)
    for s_q, page_size, boundary in straddles
    for is_causal in (True, False)
    if is_causal or boundary == KV_BLOCK
] + [
    # Note (david): a page of three tiles aligns its KV blocks to whole pages
    # and whole lane tiles, so they are no power of two long.
    straddle_param(KV_BLOCK, 4, page_size, 16, True, is_fp8)
    for is_fp8, page_size in ((False, 48), (True, 96))
]
# Note (david): a request whose kv_lens is s_q has no token before its new
# ones, so its whole cache is its appended tile and its first query row sees
# one key under causal. The inactive request between two of them is an item
# that only finishes its output.
FRESH_PARAMS = [
    MlaParam(
        3,
        s_q,
        s_q,
        is_varlen=False,
        is_causal=True,
        is_fp8=is_fp8,
        page_size=page_size,
        h_q=h_q,
        kv_lens=(s_q, 0, s_q),
    )
    for is_fp8, one_tile in ((False, 16), (True, 32))
    for h_q in (16, 128)
    for s_q, page_size in (
        (2, one_tile),
        (2, 64),
        (4, one_tile),
        (4, 64),
        (MAX_QUERY_TOKENS, one_tile),
    )
]


def check_decode(case, param, lse_tol, out, lse, kv_cache):
    """Assert out, lse and the appended cache match the reference."""
    out_ref, lse_ref, kv_cache_ref = reference(case, causal=param.is_causal)
    if param.is_fp8:
        out_tol = FP8_OUT_TOL
    else:
        out_tol = OUT_TOL
    passed = {
        "out": check_is_allclose("out", out, out_ref, **out_tol),
        "lse": check_is_allclose("lse", lse, lse_ref, **lse_tol),
        "kv_cache": check_kv_cache(
            "kv_cache", logical_cache(kv_cache), kv_cache_ref
        ),
    }
    assert all(passed.values()), passed


def run_decode(case, param, s_dtype):
    out, kv_cache, lse = flash_mla_decode(
        *case.operands(),
        kv_scale=case.kv_scale,
        sm_scale=SM_SCALE,
        causal=param.is_causal,
        s_dtype=S_DTYPES[s_dtype],
        return_lse=True,
        interpret=INTERPRET,
    )
    return out, lse, kv_cache


@pytest.mark.parametrize(
    "param, s_dtype",
    grid(
        CORRECTNESS_PARAMS
        + FEW_HEADS_PARAMS
        + CORNER_PARAMS
        + GEOMETRY_PARAMS
        + STRADDLE_PARAMS
        + FRESH_PARAMS
    ),
)
def test_flash_mla_decode(param, s_dtype):
    case = make_case(param)
    if param.have_zero_seqlen_k:
        assert (case.kv_lens == 0).any(), "the case drew no inactive request"
    check_decode(
        case, param, LSE_TOLS[s_dtype], *run_decode(case, param, s_dtype)
    )


def spiked_case(param, position, value):
    """make_case(param) with request 0's query rows all ones and its cache
    row at position holding value in every latent lane, so that key scores in
    proportion to value where the others score about 0. An fp8 param
    quantizes after the spike, so its kv_scale maps value to the largest fp8
    value."""
    case = make_case(dataclasses.replace(param, is_fp8=False))
    q_nope = case.q_nope.copy()
    q_nope[0] = 1
    kv_cache = case.kv_cache.copy()
    page = case.block_table[0, position // param.page_size]
    kv_cache[page, position % param.page_size, :LATENT_DIM] = value
    case = dataclasses.replace(case, q_nope=q_nope, kv_cache=kv_cache)
    if param.is_fp8:
        return quantize_case(case)
    else:
        return case


# Note (david): long requests at a small and a large query-row count, so the
# kernel splits them into several items and a spike can sit in a later item
# than the anchor's.
GUARD_PARAMS = [
    MlaParam(2, s_q, 4096, False, True, is_fp8, page_size=64, h_q=h_q)
    for is_fp8 in (False, True)
    for s_q, h_q in ((1, 16), (4, 128))
]


@pytest.mark.parametrize("param, s_dtype", grid(GUARD_PARAMS))
@pytest.mark.parametrize(
    "position, value, replays",
    [
        (3000, 0.45, False),
        (500, 2.0, False),
        (3000, 2.0, False),
        (3000, 5.0, True),
    ],
    ids=[
        "later-item-past-fp8-max",
        "first-item-inside-guard",
        "later-item-inside-guard",
        "later-item-past-guard",
    ],
)
def test_flash_mla_decode_anchor_guard(
    param, s_dtype, position, value, replays
):
    """A key far past its request's anchor. Inside the guard the anchored
    pass keeps its large probabilities; past it the anchored scores overflow
    f32, and only the exact replay can return finite, correct rows. Past
    fp8's largest value an fp8 cache's block scales its probabilities under
    it, and the other keys must keep their share of the row's mass."""
    case = spiked_case(param, position, value)
    if replays and s_dtype == "bf16":
        # Note (david): the replay's bf16 scores round the spike's score, and
        # with it its row's lse, which the spike all but makes up, to bf16's
        # relative precision, far past an abs_tol sized for scores near 0.
        lse_tol = dict(BF16_SCORES_LSE_TOL, rel_tol=2.0**-8)
    else:
        lse_tol = LSE_TOLS[s_dtype]
    check_decode(case, param, lse_tol, *run_decode(case, param, s_dtype))


@pytest.mark.parametrize("is_fp8", [False, True], ids=["bf16", "fp8"])
@pytest.mark.parametrize("block_kv", [384, 640, 2048])
def test_flash_mla_decode_replay_chunks(is_fp8, block_kv):
    """The replay walks an item a chunk at a time, the largest whole fraction
    of block_kv up to REPLAY_CHUNK_TILES lane tiles. These blocks take a
    chunk of a tile count that is no power of two, of a single tile and of
    REPLAY_CHUNK_TILES tiles, and a key past the guard makes each replay."""
    param = MlaParam(2, 4, 4096, False, True, is_fp8, page_size=64, h_q=16)
    case = spiked_case(param, 3000, 5.0)
    out, kv_cache, lse = mla_decode_pallas(
        *case.operands(),
        kv_scale=case.kv_scale,
        sm_scale=SM_SCALE,
        s_dtype=jnp.dtype(jnp.float32),
        causal=True,
        return_lse=True,
        interpret=INTERPRET,
        config=MlaDecodeConfig(block_kv, 2),
    )
    check_decode(case, param, LSE_TOL, out, lse, kv_cache)
