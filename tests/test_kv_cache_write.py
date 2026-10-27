"""The host-sliced page writer (tpu-inference's prefill cache fill) against a NumPy-built
merged cache, bit for bit. flash_attn_with_kvcache never calls it, so the kvcache tests
leave it uncovered."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flywheel_tpu.pallas.copy_utils import merged_cache_heads
from flywheel_tpu.pallas.kv_cache_write import write_kv_cache_pages
from tests.test_util import merge_kv_heads, randn

INTERPRET = jax.default_backend() != "tpu"
KV_DTYPES = [jnp.bfloat16, jnp.float8_e4m3fn, jnp.float8_e5m2]


@pytest.mark.parametrize("dtype", KV_DTYPES)
@pytest.mark.parametrize("nheads_k", [1, 2, 3, 8])
def test_write_kv_cache_pages(dtype, nheads_k):
    rng = np.random.default_rng(0)
    num_pages, page_size, d = 6, 128, 128
    cache_shape = (num_pages, page_size, merged_cache_heads(nheads_k, dtype), d)
    cache = randn(rng, cache_shape, jnp.float32).astype(dtype)
    num_tokens = 300
    k = randn(rng, (num_tokens, nheads_k, d), jnp.float32).astype(dtype)
    v = randn(rng, (num_tokens, nheads_k, d), jnp.float32).astype(dtype)
    # Note (david): (cache_row_start, new_row_start, length) pieces, each inside one page:
    # a page tail, a whole page and a page head, out of token order.
    pieces = [
        (4 * page_size + 100, 0, 28),
        (1 * page_size, 28, 128),
        (3 * page_size, 156, 100),
    ]
    slices = np.zeros((3, 8), np.int32)
    for column, piece in enumerate(pieces):
        slices[:, column] = piece
    expected = np.array(cache.reshape(num_pages * page_size, *cache.shape[2:]))
    merged = np.asarray(merge_kv_heads(k, v))
    for cache_row, new_row, length in pieces:
        expected[cache_row : cache_row + length] = merged[
            new_row : new_row + length
        ]
    out = write_kv_cache_pages(
        k,
        v,
        cache,
        jnp.asarray(slices),
        jnp.asarray([len(pieces)], jnp.int32),
        interpret=INTERPRET,
    )
    np.testing.assert_array_equal(
        np.asarray(out).reshape(expected.shape).view(np.uint8),
        expected.view(np.uint8),
    )
