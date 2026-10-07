"""Paged single-token decode over the merged cache, whose token rows hold each
KV head's K then its V: [k0, v0, k1, v1, ...]."""

import jax.numpy as jnp
import numpy as np
import pytest

from flywheel_tpu import flash_attn_with_kvcache
from tests.test_flash_attn_kvcache import (
    INTERPRET,
    decode_reference,
    decode_with_cache_copies,
    gather_pages,
    interleave_kv,
    random_paged_decode_inputs,
)

BATCH = 6
PAGE_SIZE = 128


def ragged_lengths(capacity):
  # Note (david): a mid-page row, a page boundary, an empty row, the last
  # slot, a block-crossing row and a long row.
  return jnp.array(
      [37, PAGE_SIZE, 0, capacity - 1, capacity // 2 + 3, capacity * 3 // 4],
      jnp.int32)


def assert_matches_reference(out, lse, q, k_cache, v_cache, total_seqlens,
                             rows, window_size=(-1, -1)):
  # Note (david): the wrapper rounds the scaled q to bf16, so a one-key row's
  # lse, a raw score, can be off by a few 1e-3 in absolute terms.
  out_ref, lse_ref = decode_reference(
      q, k_cache, v_cache, total_seqlens, rows, window_size)
  np.testing.assert_allclose(out.astype(jnp.float32),
                             out_ref.astype(jnp.float32), rtol=2e-2, atol=3e-2)
  if lse is not None:
    finite = jnp.isfinite(lse_ref)
    np.testing.assert_allclose(lse[finite], lse_ref[finite], rtol=2e-3,
                               atol=1e-2)
    np.testing.assert_array_equal(jnp.isneginf(lse), jnp.isneginf(lse_ref))


def run_interleaved_decode(num_query_heads, num_kv_heads, head_dim, capacity,
                           append, return_lse, window_size=(-1, -1),
                           num_active=None, seed=0):
  """Decode once on the interleaved pool; returns (out, lse, updated pool,
  expected pool, q, gathered expected K, gathered expected V, total lengths,
  active rows)."""
  q, k_pages, v_pages, block_table, k, v = random_paged_decode_inputs(
      seed, BATCH, capacity, PAGE_SIZE, num_query_heads, num_kv_heads,
      head_dim)
  cache_seqlens = ragged_lengths(capacity)
  active = BATCH if num_active is None else num_active
  rows = jnp.arange(active, dtype=jnp.int32)
  if append:
    append_slots = (block_table[rows, cache_seqlens[:active] // PAGE_SIZE],
                    cache_seqlens[:active] % PAGE_SIZE)
    expected_k = k_pages.at[append_slots].set(k[:active, 0])
    expected_v = v_pages.at[append_slots].set(v[:active, 0])
    total_seqlens = cache_seqlens + 1
  else:
    k = v = None
    expected_k, expected_v = k_pages, v_pages
    total_seqlens = cache_seqlens
  pool = interleave_kv(k_pages, v_pages)
  pointer = pool.unsafe_buffer_pointer()
  outputs = flash_attn_with_kvcache(
      q, pool, None, k, v, cache_seqlens=cache_seqlens,
      block_table=block_table, window_size=window_size,
      num_active=None if num_active is None else jnp.int32(num_active),
      return_softmax_lse=return_lse, interpret=INTERPRET)
  if return_lse:
    out, lse, updated = outputs
  else:
    (out, updated), lse = outputs, None
  # Note (david): the donated 4-D pool must be updated in place, never copied
  # or relaid out; interpret mode does not honor aliasing.
  assert INTERPRET or updated.unsafe_buffer_pointer() == pointer
  table = block_table[:active]
  return (out[:active], None if lse is None else lse[:active], updated,
          interleave_kv(expected_k, expected_v), q[:active],
          gather_pages(expected_k, table), gather_pages(expected_v, table),
          total_seqlens[:active], rows)


@pytest.mark.parametrize(
    ("num_query_heads", "num_kv_heads", "head_dim", "capacity", "append",
     "return_lse"),
    [
        (32, 8, 128, 512, True, True),
        # Note (david): head_dim 256 with 8 KV heads caps block_kv at 1024, so
        # a 2048-token row spans two blocks of whole pages.
        (32, 8, 256, 2048, True, False),
        # Note (david): one 2048-token block of two 1024-token compute
        # fragments.
        (16, 4, 128, 2048, True, True),
        (16, 4, 256, 512, False, True),
        (32, 4, 256, 512, True, True),
        (32, 32, 128, 512, True, True),
        (32, 32, 256, 512, True, False),
        (8, 1, 128, 512, True, True),
        (8, 1, 256, 512, False, False),
        # Note (david): KV head counts the blocked [K heads, V heads] row could
        # not DMA as tile-aligned head blocks.
        (12, 3, 128, 512, True, True),
        (24, 6, 256, 512, True, True),
        (24, 12, 128, 512, True, False),
        (8, 2, 64, 512, True, True),
        (16, 4, 64, 512, True, False),
        (12, 3, 64, 512, True, True),
        (8, 8, 64, 512, False, True),
        (4, 1, 64, 512, True, True),
    ],
)
def test_interleaved_decode_matches_reference(
    num_query_heads, num_kv_heads, head_dim, capacity, append, return_lse):
  (out, lse, updated, expected_pool, q, expected_k, expected_v, total_seqlens,
   rows) = run_interleaved_decode(
       num_query_heads, num_kv_heads, head_dim, capacity, append, return_lse,
       seed=num_query_heads + num_kv_heads + head_dim)
  # Note (david): the append is a byte copy into the interleaved row, so the
  # whole pool is checked exactly.
  np.testing.assert_array_equal(updated, expected_pool)
  assert_matches_reference(out, lse, q, expected_k, expected_v, total_seqlens,
                           rows)


@pytest.mark.parametrize(
    ("num_query_heads", "num_kv_heads", "head_dim"),
    [(32, 8, 128), (16, 4, 256), (32, 4, 256), (32, 32, 128), (8, 1, 256)],
)
def test_interleaved_decode_matches_contiguous_pair(
    num_query_heads, num_kv_heads, head_dim):
  # Note (david): the contiguous K/V pair decodes the same keys and values
  # through its own staging, which may score two heads per MXU pass where the
  # interleaved build scores one, so out and lse match to a bf16 ulp.
  capacity = 512
  (out, lse, _, _, q, expected_k, expected_v, _, _) = run_interleaved_decode(
      num_query_heads, num_kv_heads, head_dim, capacity, True, True, seed=5)
  q_pair, k_pages, v_pages, block_table, k, v = random_paged_decode_inputs(
      5, BATCH, capacity, PAGE_SIZE, num_query_heads, num_kv_heads, head_dim)
  out_pair, lse_pair, updated_k, updated_v = decode_with_cache_copies(
      q_pair, gather_pages(k_pages, block_table),
      gather_pages(v_pages, block_table), k, v,
      cache_seqlens=ragged_lengths(capacity), return_softmax_lse=True,
      interpret=INTERPRET)
  np.testing.assert_array_equal(updated_k, expected_k)
  np.testing.assert_array_equal(updated_v, expected_v)
  np.testing.assert_allclose(out.astype(jnp.float32),
                             out_pair.astype(jnp.float32), rtol=2e-2,
                             atol=2e-3)
  np.testing.assert_allclose(lse, lse_pair, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize(
    ("num_query_heads", "num_kv_heads", "head_dim", "window_size",
     "num_active"),
    [(16, 4, 128, (5, 0), 4), (12, 3, 256, (200, 0), None),
     (8, 1, 128, (-1, -1), 2)],
)
def test_interleaved_decode_window_and_padding_rows(
    num_query_heads, num_kv_heads, head_dim, window_size, num_active):
  # Note (david): padding rows past num_active must append nothing, so the
  # exact pool check covers them too.
  (out, lse, updated, expected_pool, q, expected_k, expected_v, total_seqlens,
   rows) = run_interleaved_decode(
       num_query_heads, num_kv_heads, head_dim, 512, True, True,
       window_size=window_size, num_active=num_active, seed=17)
  np.testing.assert_array_equal(updated, expected_pool)
  assert_matches_reference(out, lse, q, expected_k, expected_v, total_seqlens,
                           rows, window_size)
