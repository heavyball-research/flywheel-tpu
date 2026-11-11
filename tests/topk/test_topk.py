"""flywheel_tpu.topk against jax.lax.top_k on the masked rows, as sets, ties
either way. The tail past each row's length holds +large values or NaN,
which must never be selected; the transforms are checked by mapping their
output back to positions.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flywheel_tpu.topk import (
    top_k,
    top_k_page_table_transform,
    top_k_ragged_transform,
)
from flywheel_tpu.topk.kernels.sparse_core_topk import TILE_COLUMNS

TAIL_VALUE = 1e30
# A k that is a multiple of every chip's SparseCore lane count, as flat
# scores need.
FLAT_K = 2048
# Lengths at the path boundaries: empty, below and at k, just above k, mid
# direct, sampled, full row.
FLAT_BOUNDARY_LENGTHS = (
    0,
    1,
    FLAT_K - 1,
    FLAT_K,
    FLAT_K + 1,
    3 * FLAT_K,
    4 * FLAT_K + 7,
    9 * FLAT_K + 128,
)
pytestmark = pytest.mark.skipif(
    jax.default_backend() != "tpu", reason="top_k runs on SparseCore"
)


def row_scores(rng, num_rows, num_columns, kind):
    if kind == "normal":
        return rng.standard_normal((num_rows, num_columns), dtype=np.float32)
    if kind == "ties":
        # Non-negative, rounded to 1/8: long runs of equal scores and zeros.
        x = np.maximum(rng.standard_normal((num_rows, num_columns)), 0.0)
        return (np.round(x * 8) / 8).astype(np.float32)
    if kind == "equal":
        return np.full((num_rows, num_columns), -0.5, np.float32)
    if kind == "signed_zeros":
        return rng.choice(
            np.array([-0.0, 0.0, -1.0, 1.0], np.float32),
            (num_rows, num_columns),
        )
    if kind == "mostly_neg_inf":
        x = rng.standard_normal((num_rows, num_columns), dtype=np.float32)
        x[rng.random((num_rows, num_columns)) < 0.97] = -np.inf
        return x
    if kind == "trend":
        trend = np.arange(num_columns, dtype=np.float32) / num_columns * 4
        return trend + rng.standard_normal(
            (num_rows, num_columns), dtype=np.float32
        )
    raise ValueError(kind)


def check_rows(positions, scores, row_lengths, k):
    """Each row holds distinct positions of its k largest valid scores, then
    -1; ties at the k-th largest may go either way."""
    num_columns = scores.shape[1]
    masked = jnp.where(
        jnp.arange(num_columns)[None, :] < row_lengths[:, None],
        scores,
        -jnp.inf,
    )
    kth_index = min(k, num_columns) - 1
    kth = np.asarray(jax.lax.top_k(masked, kth_index + 1)[0][:, kth_index])
    scores = np.asarray(scores)
    assert positions.shape == (scores.shape[0], k)
    for row, length in enumerate(np.asarray(row_lengths)):
        got = positions[row]
        chosen = got[got >= 0]
        expected = min(k, length)
        assert len(chosen) == expected, (row, length, len(chosen))
        assert np.all(got[expected:] == -1), row
        assert len(np.unique(chosen)) == expected, row
        assert np.all(chosen < length), row
        if length > k:
            values = scores[row, :length]
            picked = values[chosen]
            assert np.all(picked >= kth[row]), row
            assert np.sum(picked > kth[row]) == np.sum(values > kth[row]), row


def make_case(seed, num_rows, num_columns, k, kind, dtype, tail, lengths):
    """Stored scores in dtype, their float32 values, and row lengths."""
    rng = np.random.default_rng(seed)
    values = row_scores(rng, num_rows, num_columns, kind)
    values = np.asarray(jnp.asarray(values).astype(dtype).astype(jnp.float32))
    if lengths == "full":
        row_lengths = np.full(num_rows, num_columns, np.int32)
    else:
        row_lengths = rng.integers(0, num_columns + 1, num_rows).astype(
            np.int32
        )
        boundaries = np.clip([0, 1, k - 1, k, k + 1], 0, num_columns)
        num_set = min(num_rows - 1, len(boundaries))
        row_lengths[:num_set] = boundaries[:num_set]
        row_lengths[-1] = num_columns
    columns = np.arange(num_columns)[None, :]
    stored = np.where(columns < row_lengths[:, None], values, tail)
    return jnp.asarray(stored).astype(dtype), jnp.asarray(values), row_lengths


def run_case(
    num_rows, num_columns, k, dtype, lengths, kind="normal", tail=None
):
    tail = TAIL_VALUE if tail is None else tail
    stored, values, row_lengths = make_case(
        num_rows * 31 + num_columns,
        num_rows,
        num_columns,
        k,
        kind,
        dtype,
        tail,
        lengths,
    )
    lengths_arg = None if lengths == "full" else jnp.asarray(row_lengths)
    got_values, positions = top_k(
        stored, k, row_lengths=lengths_arg, return_values=True
    )
    positions = np.asarray(positions)
    check_rows(positions, values, jnp.asarray(row_lengths), k)
    want_values = np.where(
        positions >= 0,
        np.take_along_axis(np.asarray(values), np.maximum(positions, 0), 1),
        -np.inf,
    )
    assert got_values.dtype == stored.dtype
    np.testing.assert_array_equal(
        np.asarray(got_values.astype(jnp.float32)), want_values
    )


@pytest.mark.parametrize(
    "num_rows,num_columns,k",
    [(1, 151936, 1), (3, 1000, 7), (40, 32768, 2048)],
)
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_full_rows(num_rows, num_columns, k, dtype):
    run_case(num_rows, num_columns, k, dtype, "full")


@pytest.mark.parametrize(
    "num_rows,num_columns,k",
    [(5, 20480, 64), (33, 9000, 300)],
)
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_row_lengths(num_rows, num_columns, k, dtype):
    """Random lengths, NaN past them."""
    run_case(num_rows, num_columns, k, dtype, "random", tail=np.nan)


@pytest.mark.parametrize("num_rows", [2, 40])
@pytest.mark.parametrize("kind", ["ties", "equal", "mostly_neg_inf"])
def test_hard_scores(num_rows, kind):
    run_case(num_rows, 65536, 2048, jnp.float32, "random", kind=kind)


def test_k_past_columns():
    """k larger than N: every row's positions, then -1."""
    run_case(4, 300, 512, jnp.float32, "random")


def transform_case(num_rows, num_columns, seed):
    rng = np.random.default_rng(seed)
    scores = rng.standard_normal((num_rows, num_columns), dtype=np.float32)
    row_lengths = rng.integers(0, num_columns + 1, num_rows).astype(np.int32)
    return rng, scores, row_lengths


def test_ragged_transform():
    num_rows, num_columns, k = 3, 5000, 50
    rng, scores, row_lengths = transform_case(num_rows, num_columns, k)
    offsets = rng.integers(0, 1 << 24, num_rows).astype(np.int32)
    out = np.asarray(
        top_k_ragged_transform(
            jnp.asarray(scores),
            jnp.asarray(offsets),
            jnp.asarray(row_lengths),
            k,
        )
    )
    positions = np.where(out >= 0, out - offsets[:, None], -1)
    check_rows(positions, jnp.asarray(scores), jnp.asarray(row_lengths), k)


@pytest.mark.parametrize(
    "num_rows,num_columns,k,page_size,has_row_to_batch",
    [
        (8, 20000, 64, 1, True),
        (3, 5000, 50, 16, False),
    ],
)
def test_page_table_transform(
    num_rows, num_columns, k, page_size, has_row_to_batch
):
    """Random page tables, so each physical row maps back to one position, on
    both lookup paths."""
    rng, scores, row_lengths = transform_case(num_rows, num_columns, k)
    pages = -(-num_columns // page_size)
    num_batches = max(1, num_rows // 2) if has_row_to_batch else num_rows
    table = np.stack([
        rng.permutation(pages * num_batches)[:pages] for _ in range(num_batches)
    ]).astype(np.int32)
    row_to_batch = rng.integers(0, num_batches, num_rows).astype(np.int32)
    if not has_row_to_batch:
        row_to_batch = np.arange(num_rows, dtype=np.int32)
    out = np.asarray(
        top_k_page_table_transform(
            jnp.asarray(scores),
            jnp.asarray(table),
            jnp.asarray(row_lengths),
            k,
            row_to_batch=jnp.asarray(row_to_batch)
            if has_row_to_batch
            else None,
            page_size=page_size,
        )
    )
    positions = np.full_like(out, -1)
    for row in range(num_rows):
        logical = {page: i for i, page in enumerate(table[row_to_batch[row]])}
        used = out[row] >= 0
        physical = out[row][used]
        positions[row][used] = [
            logical[p // page_size] * page_size + p % page_size
            for p in physical
        ]
    check_rows(positions, jnp.asarray(scores), jnp.asarray(row_lengths), k)


def run_flat_case(row_lengths, row_stride, kind, tail, seed):
    """top_k on flat scores of row_stride columns, tail past each row's
    length."""
    rng = np.random.default_rng(seed)
    row_lengths = np.asarray(row_lengths, np.int32)
    values = row_scores(rng, len(row_lengths), row_stride, kind)
    columns = np.arange(row_stride)[None, :]
    stored = np.where(columns < row_lengths[:, None], values, tail)
    flat = stored.astype(np.float32).reshape(len(row_lengths), -1, TILE_COLUMNS)
    positions = top_k(
        jnp.asarray(flat), FLAT_K, row_lengths=jnp.asarray(row_lengths)
    )
    check_rows(
        np.asarray(positions),
        jnp.asarray(values),
        jnp.asarray(row_lengths),
        FLAT_K,
    )


@pytest.mark.parametrize("num_rows", [17, 70])
@pytest.mark.parametrize("kind", ["normal", "ties", "signed_zeros"])
def test_flat_mixed_lengths(num_rows, kind):
    row_stride = 10 * FLAT_K + 1024
    rng = np.random.default_rng(num_rows)
    lengths = rng.integers(0, row_stride + 1, num_rows)
    lengths[: len(FLAT_BOUNDARY_LENGTHS)] = FLAT_BOUNDARY_LENGTHS
    lengths[-1] = row_stride
    run_flat_case(lengths, row_stride, kind, TAIL_VALUE, seed=num_rows)


@pytest.mark.parametrize("num_rows", [2, 40])
@pytest.mark.parametrize("kind", ["equal", "mostly_neg_inf", "trend"])
def test_flat_hard_scores(num_rows, kind):
    row_stride = 64 * FLAT_K
    lengths = np.linspace(row_stride // 3, row_stride, num_rows).astype(
        np.int32
    )
    lengths -= lengths % 7
    run_flat_case(lengths, row_stride, kind, np.nan, seed=7)


def test_flat_many_rows():
    """More rows than a subcore holds lengths for at once, so lengths load in
    blocks."""
    num_rows, row_stride = 32768, 3072
    rng = np.random.default_rng(num_rows)
    lengths = rng.integers(0, row_stride + 1, num_rows)
    run_flat_case(lengths, row_stride, "normal", TAIL_VALUE, seed=num_rows)


def test_flat_single_row():
    """One row cut across every subcore of a SparseCore."""
    row_stride = 64 * FLAT_K
    run_flat_case([row_stride], row_stride, "normal", TAIL_VALUE, seed=0)
