"""flash_mla_prefill against prefill_reference with FlashMLA's tolerances, on
ragged chunks and on FlashMLA's prefill-relevant test cases."""

import dataclasses
import random

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flywheel_tpu.softmax_attention.kernels.mla_prefill import (
    MlaPrefillConfig,
    mla_prefill_pallas,
    mla_prefill_schedule,
)
from flywheel_tpu.softmax_attention.mla_decode import (
    LATENT_DIM,
    mla_cache_shape,
)
from flywheel_tpu.softmax_attention.mla_prefill import flash_mla_prefill
from tests.softmax_attention.mla_reference import (
    QK_DIM,
    SM_SCALE,
    MlaParam,
    MlaPrefillParam,
    check_is_allclose,
    draw_kv_lens,
    make_prefill_case,
    prefill_reference,
    quantize_case,
    write_chunks,
)

INTERPRET = jax.default_backend() != "tpu"
# Note (david): FlashMLA's tolerances, as the decode test takes them; fp8
# keeps OUT_TOL, since this kernel rounds its probabilities to bf16, not fp8.
OUT_TOL = dict(abs_tol=8e-4, rel_tol=2.01 / 128, cos_diff_tol=5e-6)
LSE_TOL = dict(abs_tol=1e-6, rel_tol=8.01 / 65536)
# Note (david): FlashMLA's looser tolerances for its amplified keys.
AMPLIFIED_OUT_TOL = dict(abs_tol=1.0, rel_tol=1.0, cos_diff_tol=1e-5)
AMPLIFIED_LSE_TOL = dict(abs_tol=1e-5, rel_tol=4.01 / 65536)
SUBSET_STRIDE = 4


def param_id(param: MlaPrefillParam) -> str:
    return "-".join([
        "q" + "_".join(map(str, param.q_lens)),
        "p" + "_".join(map(str, param.prefix_lens)),
        f"h{param.h_q}",
        f"ps{param.page_size}",
        "fp8" if param.is_fp8 else "bf16",
        f"pad{param.padded_tokens}",
    ])


def check_cell(
    param,
    causal,
    return_lse=True,
    config=None,
    case=None,
    *,
    is_amplified=False,
    check_correctness=True,
    reruns=0,
):
    """The kernel's out, and lse with return_lse, on the chunk's rows against
    prefill_reference; config None runs flash_mla_prefill's own config.
    Without check_correctness only the out's finiteness is checked; reruns
    more calls must match the first bit for bit."""
    case = make_prefill_case(param) if case is None else case
    chunk = slice(0, case.cu_q_lens[-1])
    if check_correctness:
        out_ref, lse_ref, kv_cache = prefill_reference(case, causal=causal)
        references = {"out": out_ref[chunk], "lse": lse_ref[:, chunk]}
    else:
        kv_cache = write_chunks(case)
        references = None
    num_pages, page_size, _ = kv_cache.shape
    operands = (
        jnp.asarray(case.q_nope),
        jnp.asarray(case.q_pe),
        jnp.asarray(
            kv_cache.reshape(
                mla_cache_shape(num_pages, page_size, kv_cache.dtype)
            )
        ),
        jnp.asarray(case.cu_q_lens),
        jnp.asarray(case.kv_lens),
        jnp.asarray(case.block_table),
    )
    max_seqlen_q = max(1, *param.q_lens)

    def _call():
        if config is None:
            result = flash_mla_prefill(
                *operands,
                sm_scale=SM_SCALE,
                kv_scale=case.kv_scale,
                causal=causal,
                return_lse=return_lse,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=int(case.kv_lens.max()),
                interpret=INTERPRET,
            )
        else:
            result = mla_prefill_pallas(
                *operands,
                kv_scale=case.kv_scale,
                sm_scale=SM_SCALE,
                causal=causal,
                return_lse=return_lse,
                max_seqlen_q=max_seqlen_q,
                interpret=INTERPRET,
                config=config,
            )
        if return_lse:
            out, lse = result
            return {
                "out": np.asarray(out[chunk]),
                "lse": np.asarray(lse[:, chunk]),
            }
        else:
            return {"out": np.asarray(result[chunk])}

    results = _call()
    for _ in range(reruns):
        for name, again in _call().items():
            assert (
                results[name].tobytes() == again.tobytes()
            ), f"{name} differs on a rerun"
    if is_amplified:
        tolerances = {"out": AMPLIFIED_OUT_TOL, "lse": AMPLIFIED_LSE_TOL}
    else:
        tolerances = {"out": OUT_TOL, "lse": LSE_TOL}
    if check_correctness:
        passed = {
            name: check_is_allclose(
                name, value, references[name], **tolerances[name]
            )
            for name, value in results.items()
        }
        assert all(passed.values()), passed
    else:
        assert np.isfinite(results["out"]).all()


def grid(params, *, sampled):
    """params under both masks as pytest params; under --subset the sampled
    params and the full-mask cells keep one in SUBSET_STRIDE."""
    cells = []
    for param in params:
        for causal in (True, False):
            stride = SUBSET_STRIDE ** (int(sampled(param)) + int(not causal))
            if stride > 1:
                marks = [pytest.mark.subset_sample(stride)]
            else:
                marks = []
            cells.append(
                pytest.param(
                    param,
                    causal,
                    id=f"{param_id(param)}-{'causal' if causal else 'full'}",
                    marks=marks,
                )
            )
    return cells


# Note (david): ragged batches of mid, inactive, one-token and long chunks,
# then first chunks, long prefixes, decode-like chunks and extreme pages.
DEFAULT_PARAMS = [
    MlaPrefillParam(
        (37, 0, 1, 130),
        (300, 50, 0, 1000),
        h_q=h_q,
        page_size=page_size,
        is_fp8=is_fp8,
        padded_tokens=5,
    )
    for h_q in (16, 128)
    for page_size in (64, 512)
    for is_fp8 in (False, True)
] + [
    MlaPrefillParam((200,), (0,), h_q=16),
    MlaPrefillParam((200,), (0,), h_q=128, is_fp8=True, padded_tokens=3),
    MlaPrefillParam((64, 129), (3000, 5000), h_q=16),
    MlaPrefillParam((64, 129), (3000, 5000), h_q=16, is_fp8=True),
    MlaPrefillParam((1, 1, 1), (0, 63, 500), h_q=16, padded_tokens=2),
    # Note (david): a token's heads fill only part of a bf16 tile.
    MlaPrefillParam((37, 0, 1, 130), (300, 50, 0, 1000), h_q=8),
    MlaPrefillParam((37, 0, 1, 130), (300, 50, 0, 1000), h_q=8, is_fp8=True),
    # Note (david): a one-tile page puts the most pages in a kv block, and MLA
    # v2 serves fp8 on its largest page.
    MlaPrefillParam((37, 0, 130), (300, 50, 1000), h_q=16, page_size=16),
    MlaPrefillParam(
        (37, 0, 130), (300, 50, 1000), h_q=16, page_size=1024, is_fp8=True
    ),
]


@pytest.mark.parametrize(
    "param, causal",
    grid(DEFAULT_PARAMS, sampled=lambda param: param.h_q > 16 or param.is_fp8),
)
def test_flash_mla_prefill(param, causal):
    check_cell(param, causal)


def test_flash_mla_prefill_without_lse():
    check_cell(
        MlaPrefillParam((37, 0, 130), (300, 50, 70), padded_tokens=5),
        causal=True,
        return_lse=False,
    )


# Note (david): tiny q blocks make the chunks straddle cache words, tiles,
# pages, q blocks and kv blocks, and the kv blocks span several pages or part
# of one.
STRADDLE_Q_LENS = (3, 5, 2, 66, 1, 4, 9, 7, 0)
STRADDLE_PREFIX_LENS = (1, 15, 31, 63, 127, 128, 255, 0, 40)
STRADDLE_CELLS = [
    (
        MlaPrefillParam(
            STRADDLE_Q_LENS,
            STRADDLE_PREFIX_LENS,
            h_q=h_q,
            page_size=page_size,
            is_fp8=is_fp8,
            padded_tokens=3,
        ),
        MlaPrefillConfig(4, block_kv, 1),
    )
    for h_q in (16, 128)
    for is_fp8 in (False, True)
    for page_size, block_kv in ((64, 128), (512, 256), (96, 384))
] + [
    # Note (david): one q block split into few row groups and into many.
    (
        MlaPrefillParam(
            STRADDLE_Q_LENS, STRADDLE_PREFIX_LENS, h_q=16, is_fp8=is_fp8
        ),
        MlaPrefillConfig(32, 256, row_groups),
    )
    for is_fp8 in (False, True)
    for row_groups in (2, 16)
]


@pytest.mark.parametrize(
    "param, config, causal",
    [
        pytest.param(
            param,
            config,
            causal,
            id=(
                f"{param_id(param)}-bq{config.block_q}-bkv{config.block_kv}"
                f"-g{config.row_groups}-{'causal' if causal else 'full'}"
            ),
            marks=[pytest.mark.subset_sample(SUBSET_STRIDE)]
            if param.h_q > 16 or not causal
            else [],
        )
        for param, config in STRADDLE_CELLS
        for causal in (True, False)
    ],
)
def test_flash_mla_prefill_straddles(param, config, causal):
    check_cell(param, causal, config=config)


def spiked_prefill_case(param, position, value):
    """make_prefill_case(param) with request 0's chunk queries all ones and
    its prefix row at position holding value in every latent lane, so that key
    scores in proportion to value where the others score about 0. An fp8
    param quantizes after the spike, so its kv_scale maps value to the largest
    fp8 value."""
    assert position < param.prefix_lens[0], (position, param.prefix_lens)
    case = make_prefill_case(dataclasses.replace(param, is_fp8=False))
    q_nope = case.q_nope.copy()
    q_nope[case.cu_q_lens[0] : case.cu_q_lens[1]] = 1
    kv_cache = case.kv_cache.copy()
    page_size = kv_cache.shape[1]
    page = case.block_table[0, position // page_size]
    kv_cache[page, position % page_size, :LATENT_DIM] = value
    case = dataclasses.replace(case, q_nope=q_nope, kv_cache=kv_cache)
    if param.is_fp8:
        return quantize_case(case)
    else:
        return case


# Note (david): a long prefix puts the spike in a later item than its q
# block's anchor; a small spike stays inside the guard, and a large one
# overflows f32 and makes the call replay.
GUARD_PARAMS = [
    MlaPrefillParam((64, 37), (4000, 1000), h_q=h_q, is_fp8=is_fp8)
    for h_q in (16, 128)
    for is_fp8 in (False, True)
]


@pytest.mark.parametrize(
    "param",
    [
        pytest.param(
            param,
            id=param_id(param),
            marks=[pytest.mark.subset_sample(SUBSET_STRIDE)]
            if param.h_q > 16 or param.is_fp8
            else [],
        )
        for param in GUARD_PARAMS
    ],
)
@pytest.mark.parametrize(
    "position, value",
    [(500, 2.0), (3000, 2.0), (3000, 5.0)],
    ids=[
        "first-item-inside-guard",
        "later-item-inside-guard",
        "later-item-past-guard",
    ],
)
def test_flash_mla_prefill_anchor_guard(param, position, value):
    """Past the guard only the online-softmax replay returns finite, correct
    rows, for the spiked request and the other alike."""
    check_cell(
        param, causal=True, case=spiked_prefill_case(param, position, value)
    )


# Note (david): the replay walks one kv block and one row group at a time, so
# these configs vary both.
REPLAY_CELLS = [
    (
        MlaPrefillParam((37, 5), (1500, 300), h_q=16, page_size=page_size),
        config,
    )
    for page_size, config in (
        (64, MlaPrefillConfig(4, 128, 1)),
        (96, MlaPrefillConfig(4, 384, 1)),
        (512, MlaPrefillConfig(32, 256, 16)),
    )
]


@pytest.mark.parametrize(
    "param, config, causal",
    [
        pytest.param(
            param,
            config,
            causal,
            id=(
                f"{param_id(param)}-bq{config.block_q}-bkv{config.block_kv}"
                f"-g{config.row_groups}-{'causal' if causal else 'full'}"
            ),
        )
        for param, config in REPLAY_CELLS
        for causal in (True, False)
    ],
)
def test_flash_mla_prefill_replay(param, config, causal):
    """A key past the guard makes each of these configs replay."""
    check_cell(
        param,
        causal,
        config=config,
        case=spiked_prefill_case(param, 1000, 5.0),
    )


@pytest.mark.parametrize(
    "causal, num_q_blocks, kv_block_counts",
    [
        (True, 9, [1, 1, 2, 2, 3, 3, 4, 4, 8]),
        (False, 9, [4, 4, 4, 4, 4, 4, 4, 4, 8]),
    ],
    ids=["causal", "full"],
)
def test_mla_prefill_schedule(causal, num_q_blocks, kv_block_counts):
    """A first chunk of 512 tokens in 64-token q blocks takes, under causal,
    only the 128-token kv blocks up to each q block's last token; the
    inactive request takes none, and a 3-token chunk on a 997-token prefix
    takes all of its 1000 tokens'."""
    max_q_blocks = 12
    table, request_first_q_block = mla_prefill_schedule(
        jnp.array([0, 512, 512, 515], jnp.int32),
        jnp.array([512, 7, 1000], jnp.int32),
        block_q=64,
        block_kv=128,
        max_q_blocks=max_q_blocks,
        causal=causal,
    )
    table = np.asarray(table)
    counts = table[max_q_blocks : 2 * max_q_blocks]
    np.testing.assert_array_equal(counts[:num_q_blocks], kv_block_counts)
    np.testing.assert_array_equal(counts[num_q_blocks:], 0)
    np.testing.assert_array_equal(
        table[-2:], [num_q_blocks, sum(kv_block_counts)]
    )
    np.testing.assert_array_equal(request_first_q_block, [0, 8, 8])


# Note (david): FlashMLA's dense decode grid at the head counts this kernel
# takes, each request a chunk of s_q tokens on its drawn prefix; a request
# drawn empty is inactive, as the chunk is part of its cache.
def dense_decode_param(b, s_q, s_k, is_varlen, have_zero_seqlen_k, h_q, is_fp8):
    kv_lens = draw_kv_lens(
        MlaParam(
            b,
            s_q,
            s_k,
            is_varlen,
            is_causal=True,
            have_zero_seqlen_k=have_zero_seqlen_k,
            h_q=h_q,
        ),
        np.random.default_rng(0),
    ).tolist()
    return MlaPrefillParam(
        tuple(s_q if kv_len else 0 for kv_len in kv_lens),
        tuple(max(kv_len - s_q, 0) for kv_len in kv_lens),
        h_q=h_q,
        is_fp8=is_fp8,
    )


DENSE_DECODE_CELLS = [
    pytest.param(
        dense_decode_param(b, s_q, s_k, is_varlen, is_corner, h_q, is_fp8),
        causal,
        id=(
            f"b{b}-sq{s_q}-sk{s_k}{'-varlen' if is_varlen else ''}"
            f"{'-zero' if is_corner else ''}-h{h_q}"
            f"-{'fp8' if is_fp8 else 'bf16'}-{'causal' if causal else 'full'}"
        ),
        marks=[pytest.mark.subset_sample(SUBSET_STRIDE)],
    )
    for b, s_q, s_k, is_varlen, is_corner in [
        (b, s_q, s_k, is_varlen, False)
        for b in (1, 2, 6, 64)
        for s_q in (1, 2, 4)
        for s_k in (20, 140, 4096)
        for is_varlen in (False, True)
    ] + [(128, 2, 4096, True, True)]
    for h_q in (64, 128)
    for is_fp8 in (False, True)
    for causal in (False, True)
]


@pytest.mark.parametrize("param, causal", DENSE_DECODE_CELLS)
def test_flash_mla_prefill_dense_decode_grid(param, causal):
    check_cell(param, causal)


# Note (david): FlashMLA's dense MHA prefill grid in absorbed form, its
# lengths drawn as it draws them; it skips the reference at its largest cell
# and checks that reruns match bit for bit.
def fmha_param(mean_sq, mean_sk, is_varlen, h_q):
    if is_varlen:
        lengths = random.Random(0)
        q_lens = [
            int(max(lengths.normalvariate(mean_sq, mean_sq / 2), 1))
            for _ in range(2)
        ]
        kv_lens = [
            int(max(lengths.normalvariate(mean_sk, mean_sk / 2), q_len))
            for q_len in q_lens
        ]
    else:
        q_lens, kv_lens = [mean_sq] * 2, [mean_sk] * 2
    return MlaPrefillParam(
        tuple(q_lens),
        tuple(kv_len - q_len for kv_len, q_len in zip(kv_lens, q_lens)),
        h_q=h_q,
    )


@pytest.mark.skipif(INTERPRET, reason="the interpreter is too slow for them")
@pytest.mark.parametrize(
    "param, causal, check_correctness",
    [
        pytest.param(
            fmha_param(mean_len, mean_len, is_varlen, h_q),
            causal,
            not (mean_len == 8192 and h_q == 128),
            id=(
                f"s{mean_len}{'-varlen' if is_varlen else ''}-h{h_q}"
                f"-{'causal' if causal else 'full'}"
            ),
        )
        for mean_len in (4096, 8192)
        for is_varlen in (False, True)
        for h_q in (128, 32)
        for causal in (False, True)
    ],
)
def test_flash_mla_prefill_fmha_grid(param, causal, check_correctness):
    check_cell(param, causal, check_correctness=check_correctness, reruns=5)


# Note (david): FlashMLA's sparse prefill shapes as dense chunks, each s_kv
# a prefix; its top-k indices, attention sinks and topk lengths have no dense
# counterpart.
SPARSE_SHAPE_PARAMS = [
    MlaPrefillParam((s_q,), (s_kv,), h_q=h_q)
    for h_q in (128, 64)
    for s_kv in (128, 256, 512, 592, 1840, 1592, 1521, 95, 153, 114)
    for s_q in (1, 62, 213)
] + [
    MlaPrefillParam((s_q,), (s_kv,), h_q=h_q)
    for h_q in (128, 64)
    for s_kv in (32, 64)
    for s_q in (1, 1024)
]


@pytest.mark.parametrize(
    "param, causal", grid(SPARSE_SHAPE_PARAMS, sampled=lambda param: True)
)
def test_flash_mla_prefill_sparse_shapes(param, causal):
    check_cell(param, causal)


@pytest.mark.skipif(INTERPRET, reason="the interpreter is too slow for them")
@pytest.mark.parametrize(
    "param, check_correctness",
    [
        pytest.param(
            MlaPrefillParam((s_q,), (s_kv,), h_q=h_q),
            s_q < 70000,
            id=f"q{s_q}-p{s_kv}-h{h_q}",
        )
        for h_q in (128, 64)
        for s_q, s_kv in ((1234, 4321), (4096, 2048), (70000, 256))
    ],
)
def test_flash_mla_prefill_sparse_long_shapes(param, check_correctness):
    """FlashMLA runs its longest chunk without a reference."""
    check_cell(param, causal=True, check_correctness=check_correctness)


# Note (david): FlashMLA's k amplifier scales a few keys far up, which takes
# the anchored softmax far from its anchor.
K_AMPLIFIER_PORTION = 0.02
K_AMPLIFIER_RATIO = 2.0**8
AMPLIFIED_PARAMS = [
    MlaPrefillParam((s_q,), (s_kv,), h_q=h_q)
    for h_q in (128, 64)
    for s_kv in (592, 1840, 1592, 1521, 95, 153, 114)
    for s_q in (62, 213)
]


def amplified_prefill_case(param):
    """make_prefill_case(param) with FlashMLA's k amplifier on request 0's
    prefix: K_AMPLIFIER_PORTION of its rows, drawn with replacement, each
    scaled by a uniform factor in [1, K_AMPLIFIER_RATIO)."""
    case = make_prefill_case(param)
    rng = np.random.default_rng(param.seed)
    prefix = param.prefix_lens[0]
    positions = rng.integers(0, prefix, int(prefix * K_AMPLIFIER_PORTION))
    factors = rng.random(len(positions)) * (K_AMPLIFIER_RATIO - 1) + 1
    kv_cache = case.kv_cache.copy()
    page_size = kv_cache.shape[1]
    for position, factor in zip(positions.tolist(), factors.tolist()):
        page = case.block_table[0, position // page_size]
        row = kv_cache[page, position % page_size]
        row[:QK_DIM] = (row[:QK_DIM].astype(np.float32) * factor).astype(
            row.dtype
        )
    return dataclasses.replace(case, kv_cache=kv_cache)


@pytest.mark.parametrize(
    "param, causal", grid(AMPLIFIED_PARAMS, sampled=lambda param: True)
)
def test_flash_mla_prefill_amplified_keys(param, causal):
    check_cell(
        param,
        causal,
        case=amplified_prefill_case(param),
        is_amplified=True,
    )
