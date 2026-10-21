"""The merged-cache KV writers (the host-sliced page writer and the ragged append of
flash_attn_with_kvcache) against a NumPy-built merged cache, bit for bit."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flywheel_tpu.pallas.copy_utils import merged_cache_heads
from flywheel_tpu.pallas.flash_fwd_kvcache_varlen import append_ragged
from flywheel_tpu.pallas.kv_cache_write import write_kv_cache_pages
from tests.test_util import merge_kv_heads, randn

INTERPRET = jax.default_backend() != "tpu"
KV_DTYPES = [jnp.bfloat16, jnp.float8_e4m3fn, jnp.float8_e5m2]


def random_cache(rng, num_pages, page_size, nheads_k, d, dtype):
    cache_heads = merged_cache_heads(nheads_k, dtype)
    return randn(rng, (num_pages, page_size, cache_heads, d), jnp.float32).astype(dtype)


def bits(array):
    return np.asarray(jax.lax.bitcast_convert_type(array, jnp.uint8 if array.dtype.itemsize == 1 else jnp.uint16))


@pytest.mark.parametrize("dtype", KV_DTYPES)
@pytest.mark.parametrize("nheads_k", [1, 2, 3, 8])
def test_write_kv_cache_pages(dtype, nheads_k):
    rng = np.random.default_rng(0)
    num_pages, page_size, d = 6, 128, 128
    cache = random_cache(rng, num_pages, page_size, nheads_k, d, dtype)
    num_tokens = 300
    k = randn(rng, (num_tokens, nheads_k, d), jnp.float32).astype(dtype)
    v = randn(rng, (num_tokens, nheads_k, d), jnp.float32).astype(dtype)
    # Note (david): (cache_row_start, new_row_start, length) pieces, each inside one page:
    # a page tail, a whole page and a page head, out of token order.
    pieces = [(4 * page_size + 100, 0, 28), (1 * page_size, 28, 128), (3 * page_size, 156, 144 - 16)]
    slices = np.zeros((3, 8), np.int32)
    for column, piece in enumerate(pieces):
        slices[:, column] = piece
    expected = np.array(cache.reshape(num_pages * page_size, *cache.shape[2:]))
    merged = np.asarray(merge_kv_heads(k, v))
    for cache_row, new_row, length in pieces:
        expected[cache_row : cache_row + length] = merged[new_row : new_row + length]
    out = write_kv_cache_pages(
        k, v, cache, jnp.asarray(slices), jnp.asarray([len(pieces)], jnp.int32), interpret=INTERPRET
    )
    np.testing.assert_array_equal(bits(out.reshape(expected.shape)), bits(jnp.asarray(expected)))


@pytest.mark.parametrize("dtype", KV_DTYPES)
@pytest.mark.parametrize("nheads_k", [1, 3, 8])
def test_append_ragged(dtype, nheads_k):
    rng = np.random.default_rng(1)
    page_size, d, pages_per_seq = 128, 128, 3
    batch = 3
    num_pages = batch * pages_per_seq
    cache = random_cache(rng, num_pages, page_size, nheads_k, d, dtype)
    block_table = rng.permutation(num_pages).astype(np.int32).reshape(batch, pages_per_seq)
    cache_seqlens = np.array([5, 0, 200], np.int32)
    q_lens = np.array([130, 7, 100], np.int32)
    cu_seqlens = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    k = randn(rng, (int(cu_seqlens[-1]), nheads_k, d), jnp.float32).astype(dtype)
    v = randn(rng, (int(cu_seqlens[-1]), nheads_k, d), jnp.float32).astype(dtype)
    expected = np.array(cache.reshape(num_pages * page_size, *cache.shape[2:]))
    merged = np.asarray(merge_kv_heads(k, v))
    for request in range(batch):
        for token in range(q_lens[request]):
            position = cache_seqlens[request] + token
            page = block_table[request, position // page_size]
            expected[page * page_size + position % page_size] = merged[cu_seqlens[request] + token]
    out = append_ragged(
        cache, k, v, jnp.asarray(cu_seqlens), jnp.asarray(cache_seqlens),
        jnp.asarray(block_table.reshape(-1)), jnp.asarray([batch], jnp.int32),
        page_size=page_size, pages_per_seq=pages_per_seq, interpret=INTERPRET,
    )
    np.testing.assert_array_equal(bits(out.reshape(expected.shape)), bits(jnp.asarray(expected)))
