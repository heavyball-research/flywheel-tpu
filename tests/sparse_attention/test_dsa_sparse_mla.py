"""DSA sparse MLA and its split-cache write against the dense MLA reference
with each token's top-k selection as the mask."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from flywheel_tpu.softmax_attention.mla_decode import LATENT_DIM
from flywheel_tpu.sparse_attention.dsa.cache_write import dsa_cache_write
from flywheel_tpu.sparse_attention.dsa.kernels.gather import (
    CHUNK_WORDS,
    FLAG_ROW_SLOTS,
    META_SLOTS,
    NOPE_WORDS,
    ROPE_ROW_SLOTS,
    ROPE_WORDS,
    gather_compiler_params,
    sparse_core_mesh,
)
from flywheel_tpu.sparse_attention.dsa.sparse_mla import (
    chunk_attention,
    chunk_outputs,
    chunk_row,
    chunk_starts,
    default_chunk_tokens,
    dsa_sparse_mla,
)
from flywheel_tpu.topk.kernels.sparse_core_topk import sparse_core_vmem_words
from tests.softmax_attention.mla_reference import SM_SCALE, check_is_allclose
from tests.sparse_attention.dsa_reference import (
    DsaParam,
    dsa_split_cache,
    make_dsa_case,
    sparse_mla_reference,
)

if jax.default_backend() != "tpu":
    pytest.skip("the DSA kernels need a SparseCore", allow_module_level=True)

# Note (david): the dense fp8 MLA tolerances: the probabilities round to fp8
# for P @ V, and the row sums add the unrounded ones.
OUT_TOL = dict(abs_tol=1.8e-2, rel_tol=2.01 / 128, cos_diff_tol=5.5e-4)
LSE_TOL = dict(abs_tol=1e-6, rel_tol=8.01 / 65536)

# Each case runs with its chunk_tokens, None for the default chunks.
CASES = [
    # One token: the gather cuts its slots into half chunks.
    pytest.param(
        DsaParam(q_lens=(1,), prefix_lens=(5000,)), None, id="one_token"
    ),
    # Ragged prefill from an empty cache and after long prefixes, in several
    # chunks.
    pytest.param(
        DsaParam(q_lens=(300, 1, 77), prefix_lens=(0, 5000, 2100), seed=1),
        None,
        id="ragged_prefill",
    ),
    # Slots past their token, and padding rows after the batch.
    pytest.param(
        DsaParam(
            q_lens=(40, 3),
            prefix_lens=(2030, 100),
            has_stray_indices=True,
            padded_tokens=5,
            seed=2,
        ),
        None,
        id="stray_indices",
    ),
    # Fewer heads and small pages.
    pytest.param(
        DsaParam(
            q_lens=(9, 2), prefix_lens=(4000, 300), h_q=16, page_size=16, seed=4
        ),
        None,
        id="few_heads",
    ),
    # Large pages and a shorter top-k.
    pytest.param(
        DsaParam(
            q_lens=(5, 30),
            prefix_lens=(700, 6000),
            page_size=256,
            top_k=256,
            seed=5,
        ),
        None,
        id="short_top_k",
    ),
    # Attention steps of one block of slots, of several, of all top_k and of
    # padding alone, in chunks of one step; slots past each token's positions
    # name positions it sees.
    pytest.param(
        DsaParam(
            q_lens=(1,) * 10,
            prefix_lens=(0, 130, 200, 255, 500, 700, 300, 64, 1000, 2047),
            has_dropped_slots=True,
            has_stale_slots=True,
            padded_tokens=6,
            seed=7,
        ),
        4,
        id="step_kinds",
    ),
    # top_k a multiple of 128 but not of ROPE_BLOCK_SLOTS: tokens short of
    # one block of slots, between it and top_k, and past top_k; slots past
    # each token's positions name positions it sees.
    pytest.param(
        DsaParam(
            q_lens=(5, 30, 1),
            prefix_lens=(100, 6000, 300),
            page_size=256,
            top_k=384,
            has_stale_slots=True,
            seed=8,
        ),
        None,
        id="top_k_384",
    ),
    # top_k of 128, fewer heads, in chunks of one step.
    pytest.param(
        DsaParam(
            q_lens=(9, 2, 1),
            prefix_lens=(4000, 60, 127),
            h_q=16,
            top_k=128,
            has_stale_slots=True,
            padded_tokens=3,
            seed=10,
        ),
        4,
        id="top_k_128",
    ),
]
WRITE_CASES = [pytest.param(case.values[0], id=case.id) for case in CASES]
# Note (david): every byte NaN in float8_e4m3fn (S.1111.111), of either sign.
NAN_WORDS = (0x7F7F7F7F, -1)
# Note (david): words of a subcore's VMEM the poison leaves to the compiler.
VMEM_SLACK_WORDS = 1024
POISON_PARAM = DsaParam(
    q_lens=(300, 1, 1, 1),
    prefix_lens=(0, 0, 100, 3000),
    has_dropped_slots=True,
    has_stale_slots=True,
    padded_tokens=3,
    seed=9,
)


def request_operands(case):
    return tuple(
        jnp.asarray(value)
        for value in (case.cu_q_lens, case.kv_lens, case.block_table)
    )


@pytest.mark.parametrize("param, chunk_tokens", CASES)
def test_dsa_sparse_mla(param, chunk_tokens):
    """The sparse MLA output and lse match the reference with each token's
    selection as mask."""
    case = make_dsa_case(param)
    out_ref, lse_ref = sparse_mla_reference(case)
    nope_cache, rope_cache = dsa_split_cache(case.kv_cache)
    out, lse = dsa_sparse_mla(
        jnp.asarray(case.q_nope),
        jnp.asarray(case.q_pe),
        jnp.asarray(nope_cache),
        jnp.asarray(rope_cache),
        jnp.asarray(case.topk_indices),
        *request_operands(case),
        sm_scale=SM_SCALE,
        kv_scale=jnp.float32(case.kv_scale),
        return_lse=True,
        chunk_tokens=chunk_tokens,
    )
    assert check_is_allclose(
        "out", out, out_ref.astype(jnp.bfloat16), **OUT_TOL
    )
    assert check_is_allclose("lse", lse, lse_ref, **LSE_TOL)


@pytest.mark.parametrize("param", WRITE_CASES)
def test_dsa_cache_write(param):
    """The cache write replaces the batch's rows and keeps every other word."""
    case = make_dsa_case(param)
    nope_after, rope_after = dsa_split_cache(case.kv_cache)
    # Note (david): the batch's rows start as random words; all others must
    # keep their bits.
    rng = np.random.default_rng(param.seed)
    nope_before = nope_after.reshape(-1, NOPE_WORDS).copy()
    rope_before = rope_after.reshape(-1, ROPE_WORDS).copy()
    page_size = case.kv_cache.shape[1]
    for r, (start, end) in enumerate(
        zip(case.cu_q_lens[:-1], case.cu_q_lens[1:])
    ):
        positions = np.arange(case.kv_lens[r] - (end - start), case.kv_lens[r])
        rows = case.block_table[r, positions // page_size] * page_size + (
            positions % page_size
        )
        for before, words in (
            (nope_before, NOPE_WORDS),
            (rope_before, ROPE_WORDS),
        ):
            before[rows] = rng.integers(
                -(2**31), 2**31, (len(rows), words), dtype=np.int64
            )
    nope, rope = dsa_cache_write(
        jnp.asarray(case.new_kv[:, :LATENT_DIM]),
        jnp.asarray(case.new_kv[:, LATENT_DIM:]),
        jnp.asarray(nope_before.reshape(nope_after.shape)),
        jnp.asarray(rope_before.reshape(rope_after.shape)),
        *request_operands(case),
    )
    np.testing.assert_array_equal(np.asarray(nope), nope_after)
    np.testing.assert_array_equal(np.asarray(rope), rope_after)


def test_poisoned_staging():
    """The output matches the reference with fp8 NaN in every word the
    attention must not read."""
    case = make_dsa_case(POISON_PARAM)
    out_ref, lse_ref = sparse_mla_reference(case)
    nope_cache, rope_cache = poisoned_split_cache(case)
    out, lse = poisoned_sparse_mla(
        jnp.asarray(case.q_nope),
        jnp.asarray(case.q_pe),
        jnp.asarray(nope_cache),
        jnp.asarray(rope_cache),
        jnp.asarray(case.topk_indices),
        *request_operands(case),
        kv_scale=jnp.float32(case.kv_scale),
    )
    assert check_is_allclose(
        "out", out, out_ref.astype(jnp.bfloat16), **OUT_TOL
    )
    assert check_is_allclose("lse", lse, lse_ref, **LSE_TOL)


def poisoned_split_cache(case) -> tuple[np.ndarray, np.ndarray]:
    """The split caches with fp8 NaN words, both signs in turn, in every
    token row no request holds."""
    nope, rope = dsa_split_cache(case.kv_cache)
    page_size = case.kv_cache.shape[1]
    is_cached = np.zeros(nope.shape[0] * page_size, bool)
    for r, kv_len in enumerate(case.kv_lens.tolist()):
        positions = np.arange(kv_len)
        is_cached[
            case.block_table[r, positions // page_size] * page_size
            + positions % page_size
        ] = True
    poison = np.asarray(NAN_WORDS, np.int32)[
        np.arange(is_cached.size) % len(NAN_WORDS)
    ]
    rows = []
    for cache, words in ((nope, NOPE_WORDS), (rope, ROPE_WORDS)):
        flat = cache.reshape(-1, words).copy()
        flat[~is_cached] = poison[~is_cached, None]
        rows.append(flat.reshape(cache.shape))
    return tuple(rows)


def poison_sparse_core_vmem() -> jax.Array:
    """Fills nearly all of each subcore's VMEM with fp8 NaN words; the zeros
    it returns order the later calls that take them after it."""
    num_lanes = pltpu.get_tpu_info().sparse_core.num_lanes
    num_rows = (sparse_core_vmem_words() - VMEM_SLACK_WORDS) // num_lanes

    def body(out_hbm, zeros_vmem, poison_vmem):
        @pl.loop(0, num_rows)
        def _row(row):
            poison_vmem[row] = jnp.full((num_lanes,), NAN_WORDS[-1], jnp.int32)

        for lane in range(0, CHUNK_WORDS, num_lanes):
            zeros_vmem[pl.ds(lane, num_lanes)] = jnp.zeros(
                (num_lanes,), jnp.int32
            )

        is_first_worker = (lax.axis_index("core") == 0) & (
            lax.axis_index("subcore") == 0
        )

        @pl.when(is_first_worker)
        def _write_zeros():
            pltpu.sync_copy(zeros_vmem, out_hbm)

    return pl.kernel(
        body,
        out_type=jax.ShapeDtypeStruct((CHUNK_WORDS,), jnp.int32),
        mesh=sparse_core_mesh(),
        scratch_types=(
            pltpu.VMEM((CHUNK_WORDS,), jnp.int32),
            pltpu.VMEM((num_rows, num_lanes), jnp.int32),
        ),
        compiler_params=gather_compiler_params(),
        name="poison_vmem",
    )()


def poison_unwritten(
    rows: jax.Array, slots: jax.Array, slots_per_row: int
) -> jax.Array:
    """Gathered rows with fp8 NaN words in each token's rows past its slots,
    slots_per_row slots a row."""
    num_tokens = slots.shape[0]
    by_token = rows.reshape(num_tokens, -1, rows.shape[1])
    first_slot = jnp.arange(by_token.shape[1]) * slots_per_row
    is_unwritten = first_slot[None, :] >= slots[:, None]
    return jnp.where(is_unwritten[..., None], NAN_WORDS[-1], by_token).reshape(
        rows.shape
    )


@jax.jit
def poisoned_sparse_mla(
    q_nope, q_pe, nope_cache, rope_cache, topk_indices, *requests, kv_scale
):
    """dsa_sparse_mla's chunks with return_lse, each gather after a VMEM
    poison and each attention over rows poisoned past its steps' slots."""
    num_tokens, num_heads, _ = q_nope.shape
    top_k = topk_indices.shape[1]
    chunk = chunk_attention(
        q_nope,
        q_pe,
        nope_cache,
        rope_cache,
        *requests,
        top_k=top_k,
        sm_scale=SM_SCALE,
        kv_scale=kv_scale,
        return_lse=True,
    )
    padded_tokens = chunk.q_nope.shape[0]
    chunk_tokens = min(default_chunk_tokens(padded_tokens), padded_tokens)
    outputs = None
    for first in chunk_starts(padded_tokens, chunk_tokens):
        row = jnp.maximum(chunk_row(first, first), poison_sparse_core_vmem())
        nope_rows, rope_rows, flag_rows = chunk.gather_rows(
            row, topk_indices.reshape(-1), chunk_tokens
        )
        slots = chunk.token_meta[first : first + chunk_tokens, META_SLOTS]
        rows = (
            poison_unwritten(nope_rows, slots, 1),
            poison_unwritten(rope_rows, slots, ROPE_ROW_SLOTS),
            poison_unwritten(flag_rows, slots, FLAG_ROW_SLOTS),
        )
        outputs = chunk.attend_rows(
            rows,
            outputs,
            first_token=first,
            num_tokens=chunk_tokens,
            scalars=chunk.scalars,
        )
    return chunk_outputs(outputs, num_tokens, num_heads)
