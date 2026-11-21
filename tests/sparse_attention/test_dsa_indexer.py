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

"""dsa_index_cache_write against the host reference bit for bit, and
dsa_indexer_scores and their top-k against the f32 host reference on every
row's causal prefix."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flywheel_tpu.sparse_attention.dsa.index_cache_write import (
    dsa_index_cache_write,
)
from flywheel_tpu.sparse_attention.dsa.indexer import (
    INDEX_HEAD_DIM,
    dsa_indexer_scores,
    index_cache_shape,
)
from tests.sparse_attention import dsa_reference as ref

# Note (david): the Pallas interpreter cannot read through the indexer's
# bitcast key ref, and the repeated cache write is far too slow there.
if jax.default_backend() != "tpu":
    pytest.skip(
        "the Pallas interpreter cannot run these kernels",
        allow_module_level=True,
    )

# block_table's entry for the pages past a request's last one.
INVALID_PAGE = 2147480000
# Note (david): the kernel rounds relu(q . k) and the weights to bf16, so the
# bound is on the error over the row's largest sum of |w| * relu(q . k).
SCORE_TOLERANCE = 2e-2
# Of the f32 reference's top-k, the share a row's top-k must hold.
MIN_TOPK_RECALL = 0.99
SEED = 0
# The score cells: each request's kv_len, and the rows of a request.
SCORE_KV_LENS = (
    (1, 63, 127, 64, 0, 4096),
    (70000, 3, 128 * 257 + 5),
)
SCORE_ROWS = (1, 3, 8)
INDEX_HEADS = (ref.DEEPSEEK_V32_INDEX_HEADS, ref.GLM52_INDEX_HEADS)


@dataclasses.dataclass(frozen=True)
class IndexerParam:
    """Request r appends q_lens[r] tokens after prefix_lens[r] cached ones,
    then padded_tokens rows."""

    q_lens: tuple[int, ...]
    prefix_lens: tuple[int, ...]
    page_size: int = 256
    padded_tokens: int = 0
    num_heads: int = ref.DEEPSEEK_V32_INDEX_HEADS


@dataclasses.dataclass(frozen=True)
class IndexerCase:
    keys: np.ndarray  # (T, INDEX_HEAD_DIM) bf16, the chunk's new keys
    cache: np.ndarray  # the byte cache with every request's prefix written
    q_fp8: np.ndarray  # (T, num_heads, INDEX_HEAD_DIM) float8_e4m3fn
    weights: np.ndarray  # (T, num_heads) f32
    cu_q_lens: np.ndarray
    kv_lens: np.ndarray
    block_table: np.ndarray


def paged_layout(rng, kv_lens, page_size):
    """A cache of random bytes and a block table owning cdiv(kv_len,
    page_size) shuffled pages a request, INVALID_PAGE past them."""
    pages_per_seq = max(1, -(-int(kv_lens.max()) // page_size))
    num_requests = len(kv_lens)
    num_pages = num_requests * pages_per_seq + 1
    order = rng.permutation(num_pages)[: num_requests * pages_per_seq]
    block_table = order.reshape(num_requests, pages_per_seq).astype(np.int32)
    for r, kv_len in enumerate(kv_lens.tolist()):
        block_table[r, -(-kv_len // page_size) :] = INVALID_PAGE
    cache = rng.integers(
        0, 256, index_cache_shape(num_pages, page_size), dtype=np.uint8
    )
    return cache, block_table


def make_case(param: IndexerParam) -> IndexerCase:
    rng = np.random.default_rng(SEED)
    q_lens = np.array(param.q_lens, np.int32)
    prefix_lens = np.array(param.prefix_lens, np.int32)
    kv_lens = prefix_lens + q_lens
    cu_q_lens = np.zeros(len(q_lens) + 1, np.int32)
    cu_q_lens[1:] = np.cumsum(q_lens)
    num_tokens = int(cu_q_lens[-1]) + param.padded_tokens
    cache, block_table = paged_layout(rng, kv_lens, param.page_size)
    cu_prefix = np.zeros(len(q_lens) + 1, np.int32)
    cu_prefix[1:] = np.cumsum(prefix_lens)
    # Note (david): keys of varied magnitude exercise the UE8M0 scales.
    prefix_keys = rng.standard_normal((int(cu_prefix[-1]), INDEX_HEAD_DIM))
    prefix_keys *= rng.uniform(0.01, 10.0, (len(prefix_keys), 1))
    k_fp8, k_scale = ref.ue8m0_quantize(prefix_keys.astype(np.float32))
    cache = ref.write_index_chunks(
        cache, k_fp8, k_scale, cu_prefix, prefix_lens, block_table
    )
    keys = rng.standard_normal((num_tokens, INDEX_HEAD_DIM))
    keys *= rng.uniform(0.01, 10.0, (num_tokens, 1))
    queries = rng.standard_normal(
        (num_tokens, param.num_heads, INDEX_HEAD_DIM)
    ).astype(np.float32)
    q_fp8, q_scale = ref.ue8m0_quantize(queries)
    weights_proj = rng.standard_normal((num_tokens, param.num_heads))
    return IndexerCase(
        keys=keys.astype(jnp.bfloat16),
        cache=cache,
        q_fp8=q_fp8,
        weights=ref.index_weights(weights_proj.astype(np.float32), q_scale),
        cu_q_lens=cu_q_lens,
        kv_lens=kv_lens,
        block_table=block_table,
    )


def written_cache(case: IndexerCase) -> np.ndarray:
    """The reference cache after the chunk's keys are written."""
    num_chunk = int(case.cu_q_lens[-1])
    k_fp8, k_scale = ref.ue8m0_quantize(
        case.keys[:num_chunk].astype(np.float32)
    )
    return ref.write_index_chunks(
        case.cache,
        k_fp8,
        k_scale,
        case.cu_q_lens,
        case.kv_lens,
        case.block_table,
    )


def param_id(param: IndexerParam) -> str:
    return "-".join([
        "q" + "_".join(map(str, param.q_lens[:6])),
        "p" + "_".join(map(str, param.prefix_lens[:6])),
        f"b{len(param.q_lens)}",
        f"ps{param.page_size}",
        f"pad{param.padded_tokens}",
        f"h{param.num_heads}",
    ])


def grid_param(
    q_lens: tuple[int, ...],
    kv_lens: tuple[int, ...],
    num_heads: int = ref.DEEPSEEK_V32_INDEX_HEADS,
) -> IndexerParam:
    return IndexerParam(
        q_lens,
        ref.prefix_lens(q_lens, kv_lens),
        padded_tokens=ref.PADDED_TOKENS,
        num_heads=num_heads,
    )


# A request of kv_len keys scores min(rows, kv_len) rows; one of none is
# inactive.
ROW_CELLS = [
    (tuple(min(rows, kv_len) for kv_len in kv_lens), kv_lens)
    for kv_lens in SCORE_KV_LENS
    for rows in SCORE_ROWS
]
WRITE_PARAMS = [grid_param(*cell) for cell in ref.LENGTH_CELLS]
SCORE_PARAMS = [
    grid_param(*cell, num_heads=num_heads)
    for cell in (*ref.LENGTH_CELLS, *ROW_CELLS)
    for num_heads in INDEX_HEADS
]


@pytest.mark.parametrize("param", WRITE_PARAMS, ids=param_id)
def test_dsa_index_cache_write(param):
    """The byte cache after the chunk write matches the host reference."""
    case = make_case(param)
    keys, cache, cu_q_lens, kv_lens, block_table = (
        jnp.asarray(values)
        for values in (
            case.keys,
            case.cache,
            case.cu_q_lens,
            case.kv_lens,
            case.block_table,
        )
    )
    cache = dsa_index_cache_write(keys, cache, cu_q_lens, kv_lens, block_table)
    expected = written_cache(case)
    mismatches = np.argwhere(np.asarray(cache) != expected)
    assert len(mismatches) == 0, (
        f"{len(mismatches)} bytes differ, first (page, group, byte, slot):"
        f" {mismatches[:4].tolist()}"
    )


def topk_recall(scores, expected, lengths):
    """The smallest share of a live row's expected top-k in its top-k."""
    got = ref.indexer_topk(scores, lengths)
    want = ref.indexer_topk(expected, lengths)
    recalls = [
        len(set(got[row]) & set(want[row][want[row] >= 0]))
        / np.count_nonzero(want[row] >= 0)
        for row in np.flatnonzero(lengths)
    ]
    return min(recalls)


@pytest.mark.parametrize("param", SCORE_PARAMS, ids=param_id)
def test_dsa_indexer_scores(param):
    """Index scores and their top-k match the f32 reference on each prefix."""
    case = make_case(param)
    cache = dsa_index_cache_write(
        jnp.asarray(case.keys),
        jnp.asarray(case.cache),
        jnp.asarray(case.cu_q_lens),
        jnp.asarray(case.kv_lens),
        jnp.asarray(case.block_table),
    )
    scores = np.asarray(
        dsa_indexer_scores(
            jnp.asarray(case.q_fp8),
            jnp.asarray(case.weights),
            cache,
            jnp.asarray(case.cu_q_lens),
            jnp.asarray(case.kv_lens),
            jnp.asarray(case.block_table),
            max_seqlen_k=int(case.kv_lens.max()),
        )
    )
    scores = scores.reshape(scores.shape[0], -1)
    cache = written_cache(case)
    reference_args = (
        case.q_fp8,
        case.weights,
        cache,
        case.cu_q_lens,
        case.kv_lens,
        case.block_table,
        scores.shape[1],
    )
    expected = ref.indexer_scores(*reference_args)
    term_sums = ref.indexer_scores(
        reference_args[0], np.abs(reference_args[1]), *reference_args[2:]
    )
    lengths = ref.indexer_row_lengths(
        case.cu_q_lens, case.kv_lens, scores.shape[0]
    )
    visible = np.arange(scores.shape[1])[None, :] < lengths[:, None]
    prefix = np.where(visible, scores, 0.0)
    assert np.isfinite(prefix).all(), "a visible score is not finite"
    row_scale = np.max(np.where(visible, term_sums, 0.0), axis=1, keepdims=True)
    error = np.abs(prefix - np.where(visible, expected, 0.0)) / np.maximum(
        row_scale, np.finfo(np.float32).tiny
    )
    assert error.max() <= SCORE_TOLERANCE, (
        f"max error {error.max():.3e} at (row, column)"
        f" {np.unravel_index(error.argmax(), error.shape)}"
    )
    masked = np.where(visible, scores, -np.inf)
    assert topk_recall(masked, expected, lengths) >= MIN_TOPK_RECALL
