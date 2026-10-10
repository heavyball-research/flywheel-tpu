"""Single-token decode through flash_attn_with_kvcache and its kernel."""

import functools
import math
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flywheel_tpu import flash_attn_with_kvcache
from flywheel_tpu.flash_attn_interface import fused_q_scale
from flywheel_tpu.pallas.flash_fwd_kvcache import flash_attn_kvcache_pallas

INTERPRET = jax.default_backend() != "tpu"


def decode_reference(q, k_cache, v_cache, cache_seqlens, cache_batch_idx,
                     window_size=(-1, -1)):
  """(out, lse) of one decode step over the mapped contiguous cache rows."""
  scale = 1.0 / math.sqrt(q.shape[-1])
  keys = k_cache[cache_batch_idx].astype(jnp.float32)
  values = v_cache[cache_batch_idx].astype(jnp.float32)
  group = q.shape[2] // keys.shape[2]
  keys, values = (jnp.repeat(cache, group, axis=2) for cache in (keys, values))
  # The decode wrapper folds softmax_scale * log2(e) into q and rounds it to
  # bf16 before the kernel, so the reference scores start from that same
  # rounded q. Scaling the unrounded q instead is off by the rounding: with
  # one visible key the lse is a single dot product, the error has a std of
  # 1.6e-3 at head_dim 128 or 256, and it passes the lse tolerance below for
  # 1-2% of the elements, whatever the kernel does.
  q_scaled = (
      q.astype(jnp.float32) * fused_q_scale(scale, 0.0)).astype(q.dtype)
  scores = math.log(2.0) * jnp.einsum(
      "bqhd,bkhd->bhqk", q_scaled.astype(jnp.float32), keys)
  key_positions = jnp.arange(k_cache.shape[1])[None, :]
  query_positions = (cache_seqlens - 1)[:, None]
  left, right = window_size
  left_visible = True if left == -1 else key_positions >= query_positions - left
  right_visible = (
      True if right == -1 else key_positions <= query_positions + right)
  visible = (
      (key_positions < cache_seqlens[:, None]) & left_visible & right_visible)
  scores = jnp.where(visible[:, None, None, :], scores, -jnp.inf)
  probabilities = jnp.where(
      (cache_seqlens > 0)[:, None, None, None],
      jax.nn.softmax(scores, axis=-1), 0.0)
  out = jnp.einsum("bhqk,bkhd->bqhd", probabilities, values)
  lse = jax.scipy.special.logsumexp(scores, axis=-1)
  return out.astype(q.dtype), lse


def assert_decode_matches(out, lse, q, k_cache, v_cache, total_seqlens,
                          cache_batch_idx, window_size=(-1, -1),
                          out_tol=(2e-2, 3e-2)):
  out_ref, lse_ref = decode_reference(
      q, k_cache, v_cache, total_seqlens, cache_batch_idx, window_size)
  np.testing.assert_allclose(
      out.astype(jnp.float32), out_ref.astype(jnp.float32),
      rtol=out_tol[0], atol=out_tol[1])
  finite = jnp.isfinite(lse_ref)
  np.testing.assert_allclose(lse[finite], lse_ref[finite], rtol=2e-3,
                             atol=3e-3)
  np.testing.assert_array_equal(jnp.isneginf(lse), jnp.isneginf(lse_ref))


def random_decode_inputs(seed, batch, cache_batch, capacity, num_query_heads,
                         num_kv_heads, head_dim):
  """(q, k_cache, v_cache, k, v) for one contiguous-cache decode step."""
  keys = jax.random.split(jax.random.PRNGKey(seed), 5)
  cache_shape = (cache_batch, capacity, num_kv_heads, head_dim)
  new_shape = (batch, 1, num_kv_heads, head_dim)
  return (
      jax.random.normal(keys[0], (batch, 1, num_query_heads, head_dim),
                        jnp.bfloat16),
      jax.random.normal(keys[1], cache_shape, jnp.bfloat16),
      jax.random.normal(keys[2], cache_shape, jnp.bfloat16),
      jax.random.normal(keys[3], new_shape, jnp.bfloat16),
      jax.random.normal(keys[4], new_shape, jnp.bfloat16),
  )


def decode_with_cache_copies(q, k_cache, v_cache, *args, **kwargs):
  # Note (david): the API donates the caches, which deletes the caller's
  # arrays, and the reference checks still need them.
  return flash_attn_with_kvcache(
      q, jnp.copy(k_cache), jnp.copy(v_cache), *args, **kwargs)


def interleave_kv(k_pages, v_pages):
  """The merged (..., 2 * heads, head_dim) pool, [k0, v0, k1, v1, ...]."""
  *leading, num_kv_heads, head_dim = k_pages.shape
  return jnp.stack([k_pages, v_pages], axis=-2).reshape(
      *leading, 2 * num_kv_heads, head_dim)


def paged_decode_with_cache_copy(q, k_pages, v_pages, *args, **kwargs):
  """flash_attn_with_kvcache on the merged pool of k_pages and v_pages.

  Returns (out, lse, updated_k, updated_v), the updated pool split back into
  its K and V heads so exact checks against both page pools cover every byte.
  """
  out, lse, updated = flash_attn_with_kvcache(
      q, interleave_kv(k_pages, v_pages), None, *args, **kwargs)
  return (out, lse, updated[..., 0::2, :], updated[..., 1::2, :])


def random_paged_cache(key, batch, pages_per_seq, page_size, num_kv_heads,
                       head_dim):
  # Note (david): a random page permutation keeps a paged row from being
  # secretly contiguous in HBM.
  k_key, v_key, perm_key = jax.random.split(key, 3)
  num_pages = batch * pages_per_seq
  shape = (num_pages, page_size, num_kv_heads, head_dim)
  block_table = jax.random.permutation(perm_key, num_pages).reshape(
      batch, pages_per_seq).astype(jnp.int32)
  return (jax.random.normal(k_key, shape, jnp.bfloat16),
          jax.random.normal(v_key, shape, jnp.bfloat16), block_table)


def random_paged_decode_inputs(seed, batch, capacity, page_size,
                               num_query_heads, num_kv_heads, head_dim):
  """(q, k_pages, v_pages, block_table, k, v) for one paged decode step."""
  keys = jax.random.split(jax.random.PRNGKey(seed), 4)
  new_shape = (batch, 1, num_kv_heads, head_dim)
  k_pages, v_pages, block_table = random_paged_cache(
      keys[1], batch, capacity // page_size, page_size, num_kv_heads, head_dim)
  return (
      jax.random.normal(keys[0], (batch, 1, num_query_heads, head_dim),
                        jnp.bfloat16),
      k_pages, v_pages, block_table,
      jax.random.normal(keys[2], new_shape, jnp.bfloat16),
      jax.random.normal(keys[3], new_shape, jnp.bfloat16),
  )


def gather_pages(pages, block_table):
  batch, pages_per_seq = block_table.shape
  return pages[block_table].reshape(
      batch, pages_per_seq * pages.shape[1], *pages.shape[2:])


def decode_step_ms(step, k_cache, v_cache, warmup=3, iters=10, repeats=5):
  """Median wall-clock ms of a donating step(k_cache, v_cache) decode."""
  for _ in range(warmup):
    _, k_cache, v_cache = step(k_cache, v_cache)
  jax.block_until_ready((k_cache, v_cache))
  samples = []
  for _ in range(repeats):
    start = time.perf_counter()
    for _ in range(iters):
      out, k_cache, v_cache = step(k_cache, v_cache)
    jax.block_until_ready(out)
    samples.append((time.perf_counter() - start) / iters * 1e3)
  return sorted(samples)[len(samples) // 2]


def test_flash_attn_with_kvcache_appends_mapped_rows():
  q, k_cache, v_cache, k, v = random_decode_inputs(0, 2, 3, 256, 4, 4, 128)
  cache_seqlens = jnp.array([0, 129], jnp.int32)
  cache_batch_idx = jnp.array([2, 0], jnp.int32)
  out, lse, updated_k, updated_v = decode_with_cache_copies(
      q, k_cache, v_cache, k, v, cache_seqlens=cache_seqlens,
      cache_batch_idx=cache_batch_idx, causal=True, return_softmax_lse=True,
      interpret=INTERPRET)
  expected_k = k_cache.at[cache_batch_idx, cache_seqlens].set(k[:, 0])
  expected_v = v_cache.at[cache_batch_idx, cache_seqlens].set(v[:, 0])
  np.testing.assert_array_equal(updated_k, expected_k)
  np.testing.assert_array_equal(updated_v, expected_v)
  assert_decode_matches(out, lse, q, expected_k, expected_v, cache_seqlens + 1,
                        cache_batch_idx)


@pytest.mark.parametrize(
    ("head_dim", "num_query_heads", "num_kv_heads", "append", "window_size"),
    [
        (64, 2, 2, False, (-1, -1)),
        (128, 2, 2, True, (-1, -1)),
        (256, 2, 2, True, (-1, -1)),
        (128, 4, 2, True, (3, 0)),
        (256, 8, 2, True, (-1, -1)),
        (128, 8, 8, True, (-1, -1)),
        (128, 8, 1, True, (-1, -1)),
        (256, 4, 1, True, (3, 0)),
        (64, 1, 1, False, (-1, -1)),
    ],
)
def test_flash_attn_with_kvcache_matches_reference(
    head_dim, num_query_heads, num_kv_heads, append, window_size):
  capacity = 256
  q, k_cache, v_cache, k, v = random_decode_inputs(
      head_dim + num_kv_heads, 2, 2, capacity, num_query_heads, num_kv_heads,
      head_dim)
  cache_batch_idx = jnp.arange(2, dtype=jnp.int32)
  kwargs = dict(causal=False, window_size=window_size, return_softmax_lse=True,
                interpret=INTERPRET)
  if append:
    cache_seqlens = jnp.array([5, capacity - 1], jnp.int32)
    out, lse, updated_k, updated_v = decode_with_cache_copies(
        q, k_cache, v_cache, k, v, cache_seqlens=cache_seqlens, **kwargs)
    expected_k = k_cache.at[cache_batch_idx, cache_seqlens].set(k[:, 0])
    expected_v = v_cache.at[cache_batch_idx, cache_seqlens].set(v[:, 0])
    total_seqlens = cache_seqlens + 1
  else:
    cache_seqlens = jnp.array([0, capacity], jnp.int32)
    out, lse, updated_k, updated_v = decode_with_cache_copies(
        q, k_cache, v_cache, cache_seqlens=cache_seqlens, **kwargs)
    expected_k, expected_v = k_cache, v_cache
    total_seqlens = cache_seqlens
  np.testing.assert_array_equal(updated_k, expected_k)
  np.testing.assert_array_equal(updated_v, expected_v)
  assert_decode_matches(out, lse, q, expected_k, expected_v, total_seqlens,
                        cache_batch_idx, window_size)


def test_flash_attn_with_kvcache_recovers_when_later_block_changes_anchor():
  # Note (david): the second half's keys lift the row max from 0 to 256 after
  # the first block has frozen the softmax anchor.
  capacity, heads, head_dim = 4096, 2, 64
  q = jnp.ones((1, 1, heads, head_dim), jnp.bfloat16)
  half_shape = (1, capacity // 2, heads, head_dim)
  k_cache = jnp.concatenate(
      (jnp.zeros(half_shape, jnp.bfloat16),
       jnp.full(half_shape, 32.0, jnp.bfloat16)), axis=1)
  v_cache = jnp.concatenate(
      (jnp.ones(half_shape, jnp.bfloat16),
       jnp.full(half_shape, 3.0, jnp.bfloat16)), axis=1)
  out, lse, _, _ = decode_with_cache_copies(
      q, k_cache, v_cache, cache_seqlens=capacity, return_softmax_lse=True,
      interpret=INTERPRET)
  assert_decode_matches(out, lse, q, k_cache, v_cache,
                        jnp.array([capacity], jnp.int32),
                        jnp.array([0], jnp.int32))


def test_kvcache_compute_fragments_handle_masked_prefix_and_boundary_append():
  # Note (david): a 255-token window leaves whole compute fragments masked in
  # front of the live range, and 1799 + 1 = 1800 fills the last row of an
  # 8-sublane cache tile, the granule the cache copy rounds up to.
  q, k_cache, v_cache, k, v = random_decode_inputs(91, 2, 3, 2048, 4, 2, 64)
  cache_seqlens = jnp.array([512, 1799], jnp.int32)
  cache_batch_idx = jnp.array([2, 0], jnp.int32)
  out, lse, updated_k, updated_v = decode_with_cache_copies(
      q, k_cache, v_cache, k, v, cache_seqlens=cache_seqlens,
      cache_batch_idx=cache_batch_idx, window_size=(255, 0),
      return_softmax_lse=True, interpret=INTERPRET)
  expected_k = k_cache.at[cache_batch_idx, cache_seqlens].set(k[:, 0])
  expected_v = v_cache.at[cache_batch_idx, cache_seqlens].set(v[:, 0])
  np.testing.assert_array_equal(updated_k, expected_k)
  np.testing.assert_array_equal(updated_v, expected_v)
  assert_decode_matches(out, lse, q, expected_k, expected_v, cache_seqlens + 1,
                        cache_batch_idx, (255, 0), out_tol=(3e-2, 4e-2))


@pytest.mark.parametrize(
    ("head_dim", "capacity", "cache_seqlen"),
    [(128, 2048, 1537), (256, 1024, 900)],
)
def test_mha_token_major_cache_across_compute_fragments(
    head_dim, capacity, cache_seqlen):
  q, k_cache, v_cache, _, _ = random_decode_inputs(
      head_dim, 1, 1, capacity, 2, 2, head_dim)
  cache_seqlens = jnp.array([cache_seqlen], jnp.int32)
  out, lse, updated_k, updated_v = decode_with_cache_copies(
      q, k_cache, v_cache, cache_seqlens=cache_seqlens,
      return_softmax_lse=True, interpret=INTERPRET)
  np.testing.assert_array_equal(updated_k, k_cache)
  np.testing.assert_array_equal(updated_v, v_cache)
  assert_decode_matches(out, lse, q, k_cache, v_cache, cache_seqlens,
                        jnp.array([0], jnp.int32), out_tol=(3e-2, 4e-2))


@pytest.mark.parametrize(
    ("head_dim", "num_query_heads", "num_kv_heads", "append", "window_size",
     "page_size", "capacity"),
    [
        (128, 2, 2, True, (-1, -1), 128, 512),
        (128, 4, 2, True, (3, 0), 256, 512),
        (128, 8, 1, True, (-1, -1), 128, 512),
        # Note (david): head_dim 256 with 8 KV heads caps block_kv at 1024, so
        # a 2048-token row spans two blocks of whole pages.
        (256, 8, 8, True, (-1, -1), 128, 2048),
        (256, 8, 8, False, (-1, -1), 256, 2048),
        (64, 8, 8, False, (-1, -1), 128, 256),
        (64, 8, 2, False, (-1, -1), 128, 256),
    ],
)
def test_flash_attn_with_kvcache_block_table_matches_reference(
    head_dim, num_query_heads, num_kv_heads, append, window_size, page_size,
    capacity):
  batch = 4
  q, k_pages, v_pages, block_table, k, v = random_paged_decode_inputs(
      head_dim + page_size, batch, capacity, page_size, num_query_heads,
      num_kv_heads, head_dim)
  # Note (david): a page boundary, a block boundary, an empty row and the
  # last slot.
  cache_seqlens = jnp.array(
      [page_size, capacity // 2, 0, capacity - 1], jnp.int32)
  rows = jnp.arange(batch, dtype=jnp.int32)
  k_cache = gather_pages(k_pages, block_table)
  v_cache = gather_pages(v_pages, block_table)
  kwargs = dict(cache_seqlens=cache_seqlens, block_table=block_table,
                window_size=window_size, return_softmax_lse=True,
                interpret=INTERPRET)
  if append:
    out, lse, updated_k, updated_v = paged_decode_with_cache_copy(
        q, k_pages, v_pages, k, v, **kwargs)
    expected_k = k_cache.at[rows, cache_seqlens].set(k[:, 0])
    expected_v = v_cache.at[rows, cache_seqlens].set(v[:, 0])
    total_seqlens = cache_seqlens + 1
  else:
    out, lse, updated_k, updated_v = paged_decode_with_cache_copy(
        q, k_pages, v_pages, **kwargs)
    expected_k, expected_v = k_cache, v_cache
    total_seqlens = cache_seqlens
  # Note (david): the table covers every page, so the gathered pool checks both
  # the appended slots and that no other page byte moved, in the K and the V
  # heads of the merged pool alike.
  np.testing.assert_array_equal(gather_pages(updated_k, block_table),
                                expected_k)
  np.testing.assert_array_equal(gather_pages(updated_v, block_table),
                                expected_v)
  assert_decode_matches(out, lse, q, expected_k, expected_v, total_seqlens,
                        rows, window_size)


def test_flash_attn_with_kvcache_block_table_matches_contiguous_kernel():
  # Note (david): capacity 1024 resolves to one 1024-token block either way,
  # and 8 query heads per KV head make the pair build score one head per
  # fragment like the merged build, so paging and the interleaved row only
  # change addressing and the two must agree bit for bit.
  batch, page_size, capacity, head_dim = 3, 128, 1024, 128
  num_query_heads, num_kv_heads = 16, 2
  q, k_pages, v_pages, block_table, k, v = random_paged_decode_inputs(
      7, batch, capacity, page_size, num_query_heads, num_kv_heads, head_dim)
  cache_seqlens = jnp.array([1, 700, capacity - 1], jnp.int32)
  out_paged, lse_paged, _, _ = paged_decode_with_cache_copy(
      q, k_pages, v_pages, k, v, cache_seqlens=cache_seqlens,
      block_table=block_table, return_softmax_lse=True, interpret=INTERPRET)
  out_dense, lse_dense, _, _ = decode_with_cache_copies(
      q, gather_pages(k_pages, block_table),
      gather_pages(v_pages, block_table), k, v, cache_seqlens=cache_seqlens,
      return_softmax_lse=True, interpret=INTERPRET)
  np.testing.assert_array_equal(out_paged, out_dense)
  np.testing.assert_array_equal(lse_paged, lse_dense)


@pytest.mark.parametrize(("num_query_heads", "num_kv_heads", "head_dim"),
                         [(8, 1, 128), (6, 1, 256), (8, 2, 128)])
def test_flash_attn_with_kvcache_appends_step_after_step(
    num_query_heads, num_kv_heads, head_dim):
  # Note (david): one executable serves every position, and each step's
  # append must land on the cache the previous step returned: the positions
  # cross the 8-row and 16-row cache tiles (one KV head stages head-major,
  # whose token axis v7x tiles 16 deep) and end at the last slot.
  batch, capacity = 2, 256
  positions = [*range(0, 40), *range(capacity - 20, capacity)]
  q, k_cache, v_cache, _, _ = random_decode_inputs(
      17, batch, batch, capacity, num_query_heads, num_kv_heads, head_dim)
  k_cache = k_cache.at[:, positions].set(0.0)
  v_cache = v_cache.at[:, positions].set(0.0)
  expected_k, expected_v = k_cache, v_cache
  updated_k, updated_v = jnp.copy(k_cache), jnp.copy(v_cache)
  rows = jnp.arange(batch, dtype=jnp.int32)
  new_shape = (batch, 1, num_kv_heads, head_dim)
  for step, position in enumerate(positions):
    keys = jax.random.split(jax.random.PRNGKey(100 + step), 2)
    k = jax.random.normal(keys[0], new_shape, jnp.bfloat16)
    v = jax.random.normal(keys[1], new_shape, jnp.bfloat16)
    cache_seqlens = jnp.full((batch,), position, jnp.int32)
    out, lse, updated_k, updated_v = flash_attn_with_kvcache(
        q, updated_k, updated_v, k, v, cache_seqlens=cache_seqlens,
        return_softmax_lse=True, interpret=INTERPRET)
    expected_k = expected_k.at[rows, position].set(k[:, 0])
    expected_v = expected_v.at[rows, position].set(v[:, 0])
    np.testing.assert_array_equal(updated_k, expected_k)
    np.testing.assert_array_equal(updated_v, expected_v)
    assert_decode_matches(out, lse, q, expected_k, expected_v,
                          cache_seqlens + 1, rows)


def test_flash_attn_with_kvcache_block_table_changes_under_one_executable():
  # Note (david): the table and the lengths are runtime operands, so one
  # executable must serve both page layouts, each append landing where its
  # table says.
  batch, page_size, pages_per_seq, heads, head_dim = 2, 128, 2, 2, 128
  q = jnp.ones((batch, 1, heads, head_dim), jnp.bfloat16)
  kv_pages = jnp.zeros(
      (batch * pages_per_seq, page_size, 2 * heads, head_dim), jnp.bfloat16)
  k = jnp.ones((batch, 1, heads, head_dim), jnp.bfloat16)
  v = jnp.full_like(k, 2.0)
  table_a = jnp.array([[0, 1], [2, 3]], jnp.int32)
  table_b = jnp.array([[3, 2], [1, 0]], jnp.int32)
  lens_a = jnp.array([0, 129], jnp.int32)
  lens_b = jnp.array([130, 5], jnp.int32)
  compiled = flash_attn_with_kvcache.lower(
      q, kv_pages, None, k, v, cache_seqlens=lens_a, block_table=table_a,
      interpret=INTERPRET).compile()
  rows = jnp.arange(batch)
  for table, lens in ((table_a, lens_a), (table_b, lens_b)):
    _, kv_pages = compiled(
        q, kv_pages, None, k, v, cache_seqlens=lens, block_table=table)
    slots = (table[rows, lens // page_size], lens % page_size)
    slot_kv = kv_pages[slots]
    np.testing.assert_array_equal(slot_kv[..., 0::2, :], k[:, 0])
    np.testing.assert_array_equal(slot_kv[..., 1::2, :], v[:, 0])
  assert float(jnp.abs(kv_pages).sum()) == 4 * float(
      jnp.abs(k[0, 0]).sum() + jnp.abs(v[0, 0]).sum())


@pytest.mark.skipif(INTERPRET, reason="kernel timing needs a TPU")
def test_gqa_decode_costs_no_more_than_mha_on_the_same_cache():
  # Note (david): query heads sharing a KV head must ride one K/V read and one
  # MXU pass, so 16 query heads over a 2-head cache may cost at most 1.5x the
  # 2-head MHA step that moves the same bytes.
  batch, capacity, num_kv_heads, head_dim = 32, 2048, 2, 128
  lengths = jnp.asarray(np.random.default_rng(0).integers(
      1024, 2048, size=batch, dtype=np.int32))

  def step_ms(num_query_heads):
    q, k_cache, v_cache, k, v = random_decode_inputs(
        num_query_heads, batch, batch, capacity, num_query_heads,
        num_kv_heads, head_dim)

    # Note (david): donating on the outer jit keeps every step from copying
    # the caches first.
    @functools.partial(jax.jit, donate_argnums=(0, 1))
    def step(k_cache_arg, v_cache_arg):
      return flash_attn_with_kvcache(
          q, k_cache_arg, v_cache_arg, k, v, cache_seqlens=lengths)

    return decode_step_ms(step, k_cache, v_cache)

  mha_ms = step_ms(num_kv_heads)
  gqa_ms = step_ms(8 * num_kv_heads)
  assert gqa_ms <= 1.5 * mha_ms, f"{gqa_ms=:.3f} {mha_ms=:.3f}"


@pytest.mark.skipif(INTERPRET, reason="kernel timing needs a TPU")
def test_token_major_decode_costs_no_more_than_head_major():
  # Note (david): the token-major cache packs two heads per u32 lane, and
  # reading it may cost at most 1.1x the head-major layout that needs no
  # unpacking.
  batch, capacity, heads, head_dim = 32, 2048, 16, 128
  lengths = jnp.asarray(np.random.default_rng(0).integers(
      1024, 2048, size=batch, dtype=np.int32))
  batch_idx = jnp.arange(batch, dtype=jnp.int32)
  keys = jax.random.split(jax.random.PRNGKey(0), 5)
  q, k, v = (jax.random.normal(key, (batch, heads, head_dim), jnp.bfloat16)
             for key in keys[:3])

  def step_ms(head_major):
    cache_shape = ((batch, heads, capacity, head_dim) if head_major
                   else (batch, capacity, heads, head_dim))
    k_cache = jax.random.normal(keys[3], cache_shape, jnp.bfloat16)
    v_cache = jax.random.normal(keys[4], cache_shape, jnp.bfloat16)

    @functools.partial(jax.jit, donate_argnums=(0, 1))
    def step(k_cache_arg, v_cache_arg):
      return flash_attn_kvcache_pallas(
          q, k_cache_arg, v_cache_arg, k, v, lengths, batch_idx,
          cache_head_major=head_major, has_new=True, left=None,
          return_lse=False, interpret=False)

    return decode_step_ms(step, k_cache, v_cache)

  head_major_ms = step_ms(True)
  token_major_ms = step_ms(False)
  assert token_major_ms <= 1.1 * head_major_ms, (
      f"{token_major_ms=:.3f} {head_major_ms=:.3f}")


@pytest.mark.skipif(INTERPRET, reason="kernel timing needs a TPU")
def test_padding_rows_past_num_active_cost_nothing():
  # Note (david): one active row in a 64-row bucket may cost at most 1.3x that
  # row alone; a padding row that runs costs a block iteration plus a cache
  # tile write, ~0.14 ms on v6e.
  bucket, capacity, num_query_heads, num_kv_heads, head_dim = (
      64, 2048, 32, 8, 128)

  def step_ms(batch, num_active):
    q, k_cache, v_cache, k, v = random_decode_inputs(
        3, batch, batch, capacity, num_query_heads, num_kv_heads, head_dim)
    lengths = jnp.full((batch,), 1500, jnp.int32)

    @functools.partial(jax.jit, donate_argnums=(0, 1))
    def step(k_cache_arg, v_cache_arg):
      return flash_attn_with_kvcache(
          q, k_cache_arg, v_cache_arg, k, v, cache_seqlens=lengths,
          num_active=num_active)

    return decode_step_ms(step, k_cache, v_cache)

  single_ms = step_ms(1, None)
  bucket_ms = step_ms(bucket, 1)
  assert bucket_ms <= 1.3 * single_ms, f"{bucket_ms=:.3f} {single_ms=:.3f}"


@pytest.mark.parametrize(
    ("num_query_heads", "num_kv_heads", "head_dim", "append", "num_active"),
    [
        # Note (david): the Qwen3.8-27B TP=8 and TP=4 shards.
        (3, 1, 256, True, 4),
        (6, 1, 256, True, None),
        (4, 1, 128, False, None),
        (8, 2, 128, True, 4),
        (8, 2, 256, True, None),
        (8, 2, 256, False, None),
        # Note (david): Qwen3.5-4B, Qwen3.5-27B and Qwen3-4B, in that order.
        (16, 4, 256, True, None),
        (24, 4, 256, False, None),
        (32, 8, 128, True, 4),
        (8, 8, 128, False, None),
    ],
)
def test_flash_attn_with_kvcache_paged_merged_matches_contiguous_pair(
    num_query_heads, num_kv_heads, head_dim, append, num_active):
  # Note (david): the merged build reads the halves of staged (K, V) u32 words
  # one head per fragment, where the pair build may score two heads per pass
  # or read head-major rows, so a fragment may round differently: out and lse
  # match to a bf16 ulp. The append is a byte copy, so the whole pool is
  # checked exactly.
  batch, page_size, capacity = 6, 128, 512
  q, k_pages, v_pages, block_table, k, v = random_paged_decode_inputs(
      41, batch, capacity, page_size, num_query_heads, num_kv_heads, head_dim)
  cache_seqlens = jnp.array(
      [37, page_size, 0, capacity - 1, 200, 300], jnp.int32)
  active = batch if num_active is None else num_active
  if append:
    append_slots = (
        block_table[jnp.arange(active), cache_seqlens[:active] // page_size],
        cache_seqlens[:active] % page_size)
    expected_k = k_pages.at[append_slots].set(k[:active, 0])
    expected_v = v_pages.at[append_slots].set(v[:active, 0])
  else:
    k = v = None
    expected_k, expected_v = k_pages, v_pages
  kwargs = dict(
      cache_seqlens=cache_seqlens,
      num_active=None if num_active is None else jnp.int32(num_active),
      return_softmax_lse=True, interpret=INTERPRET)
  out_pair, lse_pair, _, _ = decode_with_cache_copies(
      q, gather_pages(k_pages, block_table),
      gather_pages(v_pages, block_table), k, v, **kwargs)
  merged = interleave_kv(k_pages, v_pages)
  pointer = merged.unsafe_buffer_pointer()
  out, lse, updated = flash_attn_with_kvcache(
      q, merged, None, k, v, block_table=block_table, **kwargs)
  # Note (david): donation must alias the updated pool onto the caller's
  # buffer; interpret mode does not honor aliasing.
  assert INTERPRET or updated.unsafe_buffer_pointer() == pointer
  np.testing.assert_allclose(out[:active].astype(jnp.float32),
                             out_pair[:active].astype(jnp.float32),
                             rtol=2e-2, atol=2e-3)
  np.testing.assert_allclose(lse[:active], lse_pair[:active], rtol=2e-3,
                             atol=2e-3)
  np.testing.assert_array_equal(updated, interleave_kv(expected_k, expected_v))


@pytest.mark.parametrize(("paged", "num_active"),
                         [(False, 3), (True, 3), (True, 0)])
def test_flash_attn_with_kvcache_num_active_skips_padding_rows(
    paged, num_active):
  # Note (david): padding rows carry what a bucketed decode hands over, length
  # 0 and row 0's storage, so a padding row that did run would append onto
  # row 0's slot 0 and fail the whole-store comparison.
  batch, num_query_heads, num_kv_heads, head_dim = 6, 8, 2, 128
  page_size, capacity = 128, 512
  keys = jax.random.split(jax.random.PRNGKey(29), 5)
  q = jax.random.normal(
      keys[0], (batch, 1, num_query_heads, head_dim), jnp.bfloat16)
  k = jax.random.normal(keys[1], (batch, 1, num_kv_heads, head_dim),
                        jnp.bfloat16)
  v = jax.random.normal(keys[2], (batch, 1, num_kv_heads, head_dim),
                        jnp.bfloat16)
  rows = jnp.arange(batch, dtype=jnp.int32)
  active = rows < num_active
  cache_seqlens = jnp.where(
      active, jnp.array([37, page_size, capacity - 1, 200, 5, 300], jnp.int32),
      0)
  active_rows, active_lens = rows[:num_active], cache_seqlens[:num_active]
  kwargs = dict(cache_seqlens=cache_seqlens, num_active=jnp.int32(num_active),
                return_softmax_lse=True, interpret=INTERPRET)
  if paged:
    k_store, v_store, block_table = random_paged_cache(
        keys[3], batch, capacity // page_size, page_size, num_kv_heads,
        head_dim)
    block_table = jnp.where(active[:, None], block_table, block_table[:1])
    kwargs["block_table"] = block_table
    append_slots = (block_table[active_rows, active_lens // page_size],
                    active_lens % page_size)
    active_view = functools.partial(
        gather_pages, block_table=block_table[:num_active])
    decode = paged_decode_with_cache_copy
  else:
    shape = (batch, capacity, num_kv_heads, head_dim)
    k_store = jax.random.normal(keys[3], shape, jnp.bfloat16)
    v_store = jax.random.normal(keys[4], shape, jnp.bfloat16)
    kwargs["cache_batch_idx"] = jnp.where(active, rows, 0)
    append_slots = (active_rows, active_lens)
    active_view = lambda store: store[:num_active]
    decode = decode_with_cache_copies
  expected_k = k_store.at[append_slots].set(k[:num_active, 0])
  expected_v = v_store.at[append_slots].set(v[:num_active, 0])
  out, lse, updated_k, updated_v = decode(q, k_store, v_store, k, v, **kwargs)
  np.testing.assert_array_equal(updated_k, expected_k)
  np.testing.assert_array_equal(updated_v, expected_v)
  assert_decode_matches(
      out[:num_active], lse[:num_active], q[:num_active],
      active_view(expected_k), active_view(expected_v), active_lens + 1,
      active_rows)
