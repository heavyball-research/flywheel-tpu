"""The merged interleaved KV cache writer vs a NumPy scatter."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flywheel_tpu.pallas.kv_cache_write import (
    kv_write_pieces_per_block,
    write_kv_cache_pages,
)

INTERPRET = jax.default_backend() != "tpu"


def random_bits(rng, shape):
  """(device bf16 array, its uint16 bits on the host): every sign, exponent
  and mantissa pattern except NaNs and subnormals, which become +-inf and
  +-0.

  Note (david): on TPU any bf16 XLA op may canonicalize NaN payloads and
  flush subnormals, including the relayout copy XLA inserts for a (T, 1, D)
  operand at one kv head, so those patterns would test XLA, not the writer.
  """
  bits = rng.integers(0, 2**16, shape, dtype=np.uint16)
  exponent = (bits >> 7) & 0xFF
  special = ((exponent == 0) | (exponent == 0xFF)) & ((bits & 0x7F) != 0)
  bits[special] &= np.uint16(0xFF80)
  return (jax.lax.bitcast_convert_type(jnp.asarray(bits), jnp.bfloat16),
          bits)


def write_slices(lengths, prefixes, table, page_size):
  """The writer's (cache_row, new_row, length) pieces for requests that
  append lengths[r] tokens after prefixes[r], packed back to back."""
  pieces, new_row = [], 0
  for request, (length, prefix) in enumerate(zip(lengths, prefixes)):
    position = prefix
    while position < prefix + length:
      page = position // page_size
      end = min(prefix + length, (page + 1) * page_size)
      pieces.append((table[request, page] * page_size + position % page_size,
                     new_row, end - position))
      new_row += end - position
      position = end
  return np.asarray(pieces, np.int32).T.reshape(3, -1)


@pytest.mark.parametrize("num_kv_heads,head_dim,page_size", [
    (1, 128, 128), (2, 128, 128), (8, 128, 128), (4, 256, 128),
    (3, 256, 256), (32, 256, 128)])
def test_write_kv_cache_pages_matches_scatter(num_kv_heads, head_dim,
                                              page_size):
  # Note (david): ragged appends cross page edges, start mid-page and fill a
  # page exactly; the table has spare columns past num_slices.
  rng = np.random.default_rng(num_kv_heads * 7 + head_dim)
  lengths, prefixes = (5, page_size + 9, 0, 2 * page_size), (3, 0, 7, 0)
  pages_per_seq = 4
  table = rng.permutation(len(lengths) * pages_per_seq).reshape(
      len(lengths), pages_per_seq)
  num_pages = table.size
  slices = write_slices(lengths, prefixes, table, page_size)
  total = sum(lengths)
  k, k_bits = random_bits(rng, (total + 3, num_kv_heads, head_dim))
  v, v_bits = random_bits(rng, k.shape)
  cache, expected = random_bits(
      rng, (num_pages, page_size, 2 * num_kv_heads, head_dim))
  expected = expected.copy()
  flat = expected.reshape(num_pages * page_size, num_kv_heads, 2, head_dim)
  for cache_row, new_row, length in slices.T:
    rows = slice(new_row, new_row + length)
    flat[cache_row:cache_row + length, :, 0] = k_bits[rows]
    flat[cache_row:cache_row + length, :, 1] = v_bits[rows]
  padded = np.pad(slices, ((0, 0), (0, 5)))
  updated = write_kv_cache_pages(
      k, v, cache, jnp.asarray(padded), jnp.asarray([slices.shape[1]],
                                                    jnp.int32),
      interpret=INTERPRET)
  np.testing.assert_array_equal(np.asarray(updated).view(np.uint16), expected)


@pytest.mark.parametrize("num_kv_heads,head_dim,page_size,expected", [
    (8, 128, 128, 32), (4, 256, 128, 32), (32, 256, 128, 4),
    (64, 256, 128, 2), (1, 128, 64, 64)])
def test_kv_write_pieces_per_block(num_kv_heads, head_dim, page_size,
                                   expected):
  # Note (david): a piece stages two pages of 2H heads (1 MiB at Qwen3-4B's
  # 8 x 128); the count is the largest power of two within the 39 MiB left
  # of the budget, at most 64.
  assert kv_write_pieces_per_block(page_size, num_kv_heads,
                                   head_dim) == expected
