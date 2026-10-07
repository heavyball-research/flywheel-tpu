"""Packed multi-token attention over a paged KV cache vs a NumPy reference.

flash_attn_varlen_func(block_table=...) reads the merged interleaved cache
append_ragged filled; flash_attn_with_kvcache's multi-token and packed calls
run that same append and kernel.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from flywheel_tpu import flash_attn_varlen_func, flash_attn_with_kvcache
from flywheel_tpu.pallas import fwd_pipeline
from flywheel_tpu.pallas.block_sizes import BlockSizes, vmem_limit_bytes
from flywheel_tpu.pallas.flash_fwd_kvcache_varlen import append_ragged
from flywheel_tpu.pallas.flash_fwd_varlen_paged import (
    estimate_vmem_bytes,
    flash_attn_varlen_paged,
    paged_scratch_shapes,
    resolve_paged_tiles,
    vmem_buffer_bytes,
)

INTERPRET = jax.default_backend() != "tpu"
OUT_TOL = dict(rtol=2e-2, atol=3e-2)
LSE_TOL = dict(rtol=2e-3, atol=4e-3)


def random_ragged_case(lengths=(3, 0, 5), prefixes=(127, 0, 251), heads=4,
                       kv_heads=2, head_dim=128, page_size=128, capacity=512,
                       padding=7):
  """(q, k_pages, v_pages, k, v, cu_seqlens_q, cache_seqlens, block_table)."""
  rng = np.random.default_rng(19)
  batch = len(lengths)
  cu = np.concatenate(([0], np.cumsum(lengths))).astype(np.int32)
  total = int(cu[-1]) + padding

  def normal(shape):
    return jnp.asarray(rng.normal(size=shape), jnp.bfloat16)

  q = normal((total, heads, head_dim))
  k = normal((total, kv_heads, head_dim))
  v = normal(k.shape)
  pages_per_seq = capacity // page_size
  table = rng.permutation(batch * pages_per_seq).reshape(batch, pages_per_seq)
  shape = (batch * pages_per_seq, page_size, kv_heads, head_dim)
  return (q, normal(shape), normal(shape), k, v,
          jnp.asarray(cu), jnp.asarray(prefixes, jnp.int32),
          jnp.asarray(table, jnp.int32))


def ragged_reference(q, kc, vc, k, v, cu, lengths, table, *, causal,
                     num_active=None, softmax_scale=None):
  """Appends k/v to the paged caches, then attends each request's rows.

  Returns (out, lse, k_pages, v_pages, top2), where top2 is the attention mass
  on the two heaviest keys of every attending (row, head).
  """
  q, kc, vc = (np.asarray(operand, dtype=np.float32).copy()
               for operand in (q, kc, vc))
  cu, lengths, table = map(np.asarray, (cu, lengths, table))
  active = len(lengths) if num_active is None else num_active
  heads, head_dim = q.shape[1:]
  page_size, kv_heads = kc.shape[1:3]
  repeats = heads // kv_heads
  scale = 1 / math.sqrt(head_dim) if softmax_scale is None else softmax_scale
  out = np.zeros_like(q)
  lse = np.full((heads, q.shape[0]), -np.inf, np.float32)
  top2 = [np.empty(0)]
  for seq in range(active):
    start, end = cu[seq:seq + 2]
    query_length = end - start
    prefix = lengths[seq]
    if k is None:
      total = prefix
    else:
      positions = prefix + np.arange(query_length)
      pages = table[seq, positions // page_size]
      kc[pages, positions % page_size] = np.asarray(k[start:end], np.float32)
      vc[pages, positions % page_size] = np.asarray(v[start:end], np.float32)
      total = prefix + query_length
    positions = np.arange(total)
    pages = table[seq, positions // page_size]
    # Note (david): (1, kv_heads, 1, head_dim, total) keys and (1, kv_heads, 1,
    # total, head_dim) values broadcast against (rows, kv_heads, repeats, 1,
    # head_dim).
    keys = kc[pages, positions % page_size].transpose(1, 2, 0)[None, :, None]
    values = vc[pages, positions % page_size].transpose(1, 0, 2)[None, :, None]
    # Note (david): 256-row chunks bound the (rows, heads, total) f32 score
    # temporaries of the long-prefix cases.
    for chunk_start in range(0, query_length, 256):
      rows = np.arange(chunk_start, min(chunk_start + 256, query_length))
      queries = q[start + rows].reshape(
          len(rows), kv_heads, repeats, 1, head_dim)
      scores = (queries @ keys)[..., 0, :] * scale
      if causal:
        visible = positions[None] <= (total - query_length + rows)[:, None]
      else:
        visible = np.ones((len(rows), total), bool)
      seen = visible.any(axis=-1)
      scores = np.where(visible[:, None, None], scores, -np.inf)
      anchor = np.where(seen[:, None, None, None],
                        scores.max(axis=-1, keepdims=True, initial=-np.inf), 0)
      weights = np.exp(scores - anchor)
      denom = weights.sum(axis=-1, keepdims=True)
      probs = weights / np.where(denom > 0, denom, 1)
      result = (probs[..., None, :] @ values)[..., 0, :].reshape(
          len(rows), heads, head_dim)
      out[start + rows] = np.where(seen[:, None, None], result, 0)
      with np.errstate(divide="ignore"):
        row_lse = (anchor + np.log(denom)).reshape(len(rows), heads)
      lse[:, start + rows] = np.where(seen[:, None], row_lse, -np.inf).T
      heavy = (np.partition(probs, total - 2, axis=-1)[..., -2:].sum(axis=-1)
               if total > 1 else probs.sum(axis=-1))
      top2.append(heavy.reshape(len(rows), heads)[seen].ravel())
  return out, lse, kc, vc, np.concatenate(top2)


def interleave(kc, vc):
  """The merged cache of K and V pages (pages, page_size, kv_heads, dim): each
  token row holds each head's K row then its V row, [k0, v0, k1, v1, ...]."""
  num_pages, page_size, kv_heads, head_dim = kc.shape
  return jnp.stack((kc, vc), axis=3).reshape(
      num_pages, page_size, 2 * kv_heads, head_dim)


def deinterleave(kv_cache):
  """interleave's inverse: the (K pages, V pages) of a merged cache."""
  num_pages, page_size, cache_heads, head_dim = kv_cache.shape
  pairs = kv_cache.reshape(num_pages, page_size, cache_heads // 2, 2, head_dim)
  return pairs[..., 0, :], pairs[..., 1, :]


def run_extend(q, kc, vc, k, v, cu, lengths, table, *, return_lse=True,
               num_active=None, **kwargs):
  """flash_attn_with_kvcache's packed call on the merged pool; returns (out,
  lse or None, updated_k, updated_v), the updated pool split back."""
  *result, cache = flash_attn_with_kvcache(
      q, interleave(kc, vc), None, k, v, block_table=table,
      cache_seqlens=lengths, cu_seqlens_q=cu,
      num_active=None if num_active is None else jnp.int32(num_active),
      return_softmax_lse=return_lse, interpret=INTERPRET, **kwargs)
  return (result[0], result[1] if return_lse else None, *deinterleave(cache))


def run_paged(q, kc, vc, k, v, cu, lengths, table, *, return_lse=True,
              num_active=None, max_seqlen_q=None, max_seqlen_k=None,
              **kwargs):
  """Appends k/v (when given) with append_ragged, then reads the merged pool
  through flash_attn_varlen_func; q/out stay token-major here. Returns (out,
  lse or None, k_pages, v_pages)."""
  active = table.shape[0] if num_active is None else num_active
  kv_cache = interleave(kc, vc)
  if k is None:
    seqused = lengths
  else:
    kv_cache = append_ragged(
        kv_cache, k, v, cu, lengths, table.reshape(-1),
        jnp.full((1,), active, jnp.int32), page_size=kc.shape[1],
        pages_per_seq=table.shape[1], interpret=INTERPRET)
    seqused = lengths + jnp.diff(cu)
  if max_seqlen_q is None:
    max_seqlen_q = max(
        1, int(np.diff(np.asarray(cu))[:active].max(initial=0)))
  if max_seqlen_k is None:
    max_seqlen_k = max(1, int(np.asarray(seqused)[:active].max(initial=0)))
  result = flash_attn_varlen_func(
      q.transpose(1, 0, 2), kv_cache, None, cu, None, max_seqlen_q,
      max_seqlen_k, return_softmax_lse=return_lse, interpret=INTERPRET,
      block_table=table, seqused_k=seqused,
      num_active=None if num_active is None else jnp.int32(num_active),
      **kwargs)
  out, lse = result if return_lse else (result, None)
  return out.transpose(1, 0, 2), lse, *deinterleave(kv_cache)


def assert_extend_matches(actual, expected, cu, lengths, num_active, *,
                          skip_unseen_rows=False):
  """Rows [cu[0], cu[num_active]) match the reference (skip_unseen_rows drops
  the rows that see no key), packed padding is out = 0 and lse = -inf, and
  the caches match exactly; lse is None when the call did not return it."""
  out, lse, updated_k, updated_v = actual
  cu, lengths = np.asarray(cu), np.asarray(lengths)
  real = int(cu[num_active])
  checked = np.arange(int(cu[0]), real)
  if skip_unseen_rows:
    checked = checked[np.isfinite(expected[1][0, checked])]
  out = np.asarray(out, np.float32)
  mismatched = ~np.isclose(
      out[checked], expected[0][checked], **OUT_TOL).all(axis=(1, 2))
  if lse is None:
    padding_lse = np.full(0, -np.inf)
  else:
    lse = np.asarray(lse)
    mismatched |= ~np.isclose(
        lse[:, checked], expected[1][:, checked], **LSE_TOL).all(axis=0)
    padding_lse = lse[:, real:]
  rows = checked[mismatched]
  seqs = np.searchsorted(cu, rows, side="right") - 1
  first_mismatches = [(int(row), int(seq), int(row - cu[seq]),
                       int(lengths[seq] + row - cu[seq]))
                      for row, seq in zip(rows[:8], seqs[:8])]
  assert not rows.size, (
      f"{rows.size}/{checked.size} checked rows differ; first (row, request,"
      f" offset, position): {first_mismatches}")
  np.testing.assert_array_equal(out[real:], 0, err_msg="packed padding out")
  np.testing.assert_array_equal(padding_lse, -np.inf,
                                err_msg="packed padding lse")
  np.testing.assert_array_equal(np.asarray(updated_k, np.float32), expected[2])
  np.testing.assert_array_equal(np.asarray(updated_v, np.float32), expected[3])


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("append", [False, True])
def test_ragged_cache_matches_reference(causal, append):
  q, kc, vc, k, v, cu, lengths, table = random_ragged_case()
  k, v = (k, v) if append else (None, None)
  expected = ragged_reference(q, kc, vc, k, v, cu, lengths, table,
                              causal=causal)
  actual = run_paged(q, kc, vc, k, v, cu, lengths, table, causal=causal)
  assert_extend_matches(actual, expected, cu, lengths, len(lengths))


@pytest.mark.parametrize("interleaved,causal,append", [
    (True, True, True), (False, False, False)])
def test_ragged_cache_rotary_q_matches_reference(interleaved, causal, append):
  # Note (david): the paged cache holds K already rotated, so only Q turns, at
  # its bottom-right position seqused_k - q_len + t.
  q, kc, vc, k, v, cu, lengths, table = random_ragged_case()
  k, v = (k, v) if append else (None, None)
  head_dim = q.shape[-1]
  frequency = 10000.0 ** (-np.arange(0, head_dim, 2) / head_dim)
  angle = np.arange(table.shape[1] * kc.shape[1])[:, None] * frequency
  cos, sin = np.cos(angle).astype(np.float32), np.sin(angle).astype(np.float32)
  if interleaved:
    first_lanes = np.arange(0, head_dim, 2)
    second_lanes = np.arange(1, head_dim, 2)
  else:
    first_lanes = np.arange(head_dim // 2)
    second_lanes = np.arange(head_dim // 2, head_dim)
  boundaries = np.asarray(cu)
  seqused = np.asarray(lengths) + (np.diff(boundaries) if append else 0)
  reference_q = np.array(q, np.float32)
  for seq in range(len(lengths)):
    start, end = boundaries[seq:seq + 2]
    positions = seqused[seq] - (end - start) + np.arange(end - start)
    cosine, sine = cos[positions][:, None], sin[positions][:, None]
    first = reference_q[start:end, :, first_lanes]
    second = reference_q[start:end, :, second_lanes]
    reference_q[start:end, :, first_lanes] = first * cosine - second * sine
    reference_q[start:end, :, second_lanes] = first * sine + second * cosine
  expected = ragged_reference(reference_q, kc, vc, k, v, cu, lengths, table,
                              causal=causal)
  actual = run_paged(q, kc, vc, k, v, cu, lengths, table,
                     causal=causal, rotary_cos=jnp.asarray(cos),
                     rotary_sin=jnp.asarray(sin),
                     rotary_interleaved=interleaved, rotary_k=False)
  assert_extend_matches(actual, expected, cu, lengths, len(lengths))


@pytest.mark.parametrize("head_dim,heads,kv_heads",
                         [(128, 4, 2), (128, 4, 1), (256, 4, 4)])
def test_ragged_cache_multiple_query_blocks_and_pages(head_dim, heads,
                                                      kv_heads):
  # Note (david): 128-token bounds (an underestimate is slow, not wrong) cap
  # both block sizes at one page, so the 131-row request spans two q blocks
  # and every cache several kv blocks.
  case = random_ragged_case(lengths=(131, 17), prefixes=(123, 255),
                            heads=heads, kv_heads=kv_heads, head_dim=head_dim,
                            padding=3)
  expected = ragged_reference(*case, causal=True)
  actual = run_paged(*case, causal=True, max_seqlen_q=128, max_seqlen_k=128)
  assert_extend_matches(actual, expected, case[5], case[6], 2)


@pytest.mark.parametrize("heads,kv_heads,dim", [
    (16, 16, 128), (24, 4, 256), (32, 16, 128), (6, 3, 128), (8, 1, 128),
    (8, 2, 256)])
def test_ragged_cache_head_tiles(heads, kv_heads, dim):
  # Note (david): every group stages its kv head's (K, V) row pair, a 2-row
  # window of the interleaved head axis at any kv head, so any KV head count
  # works, odd ones and one included.
  case = random_ragged_case(lengths=(3, 5), prefixes=(37, 128), heads=heads,
                            kv_heads=kv_heads, head_dim=dim, capacity=256,
                            padding=0)
  expected = ragged_reference(*case, causal=True)
  actual = run_paged(*case, causal=True)
  assert_extend_matches(actual, expected, case[5], case[6], 2)


@pytest.mark.parametrize("num_active", [0, 1, 3])
def test_ragged_cache_empty_prefix_and_inactive_metadata(num_active):
  # Note (david): empty sequences and inactive rows carry no valid page-table
  # entries or lengths, and must never read them.
  q, kc, vc, k, v, cu, lengths, table = random_ragged_case(
      lengths=(5, 0, 3), prefixes=(0, 0, 0))
  table = table.at[num_active:].set(-1).at[1].set(-1)
  lengths = lengths.at[num_active:].set(-100)
  expected = ragged_reference(q, kc, vc, k, v, cu, lengths, table,
                              causal=True, num_active=num_active)
  actual = run_paged(q, kc, vc, k, v, cu, lengths, table,
                     num_active=num_active, causal=True)
  assert_extend_matches(actual, expected, cu, lengths, num_active)


@pytest.mark.parametrize("causal", [False, True])
def test_ragged_read_only_short_cache_and_nan_padding(causal):
  # Note (david): NaN in every slot past the two live tokens catches any read
  # outside the valid range; the empty cache has no valid page at all. Rows
  # that see no key (the empty cache's, and causal rows above the short
  # cache's diagonal) are unspecified on the paged path and go unchecked.
  q, kc, vc, _, _, cu, lengths, table = random_ragged_case(
      lengths=(5, 3), prefixes=(2, 0), padding=0)
  kc = jnp.full_like(kc, jnp.nan).at[table[0, 0], :2].set(1)
  vc = jnp.full_like(vc, jnp.nan).at[table[0, 0], :2].set(2)
  table = table.at[1].set(-1)
  expected = ragged_reference(q, kc, vc, None, None, cu, lengths, table,
                              causal=causal)
  actual = run_paged(q, kc, vc, None, None, cu, lengths, table,
                     causal=causal)
  assert_extend_matches(actual, expected, cu, lengths, 2,
                        skip_unseen_rows=True)


def test_ragged_read_only_empty_cache_between_cached_requests():
  # Note (david): rows over an empty cache run no KV block, so they issue no
  # prefetch for the next request; a kernel that counts that prefetch as
  # issued waits on it forever. Those rows see no key, so they are
  # unspecified and go unchecked.
  q, kc, vc, _, _, cu, lengths, table = random_ragged_case(
      lengths=(4, 3, 9, 2), prefixes=(0, 130, 0, 5), padding=5)
  expected = ragged_reference(q, kc, vc, None, None, cu, lengths, table,
                              causal=False)
  actual = run_extend(q, kc, vc, None, None, cu, lengths, table,
                      causal=False)
  assert_extend_matches(actual, expected, cu, lengths, 4,
                        skip_unseen_rows=True)


def test_ragged_cache_shared_read_only_prefix_and_private_append_pages():
  q, kc, vc, k, v, cu, lengths, table = random_ragged_case(
      lengths=(2, 7), prefixes=(128, 128), padding=0)
  table = table.at[1, 0].set(table[0, 0])
  expected = ragged_reference(q, kc, vc, k, v, cu, lengths, table,
                              causal=True)
  actual = run_paged(q, kc, vc, k, v, cu, lengths, table,
                     return_lse=False, causal=True)
  assert_extend_matches(actual, expected, cu, lengths, 2)


def test_ragged_metadata_reuses_one_executable():
  q, kc, vc, k, v, cu, lengths, table = random_ragged_case(padding=0)
  kv_pages = interleave(kc, vc)

  def paged_attn(q, kv_pages, cu, seqused, table, num_active):
    # Note (david): the static bounds cover every metadata set below: the
    # whole packed buffer and a request's full table row.
    return flash_attn_varlen_func(
        q, kv_pages, None, cu, None, q.shape[1],
        table.shape[1] * kv_pages.shape[1], causal=True, interpret=INTERPRET,
        block_table=table, seqused_k=seqused, num_active=num_active)

  compiled = jax.jit(paged_attn).lower(
      q.transpose(1, 0, 2), kv_pages, cu, lengths, table,
      jnp.int32(3)).compile()
  for boundaries, active, lens, pages in (
      (cu, 3, lengths, table),
      (jnp.array([0, 1, 6, 8], jnp.int32), 2,
       jnp.array([0, 127, -100], jnp.int32), table[::-1]),
  ):
    kc, vc = deinterleave(kv_pages)
    expected = ragged_reference(q, kc, vc, k, v, boundaries, lens, pages,
                                causal=True, num_active=active)
    kv_pages = append_ragged(
        kv_pages, k, v, boundaries, lens, pages.reshape(-1),
        jnp.full((1,), active, jnp.int32), page_size=kv_pages.shape[1],
        pages_per_seq=pages.shape[1], interpret=INTERPRET)
    out = compiled(q.transpose(1, 0, 2), kv_pages, boundaries,
                   lens + jnp.diff(boundaries), pages, jnp.int32(active))
    assert_extend_matches(
        (out.transpose(1, 0, 2), None, *deinterleave(kv_pages)),
        expected, boundaries, lens, active)


def test_multitoken_contiguous_cache_and_mapping():
  q, kc, vc, k, v, cu, lengths, table = random_ragged_case(
      lengths=(3, 3), prefixes=(127, 509), padding=0)
  kc = kc[table].reshape(2, 512, 2, 128)
  vc = vc[table].reshape(2, 512, 2, 128)
  # Note (david): a contiguous cache is one 512-token page per row behind a
  # one-column block table (cache_batch_idx's role); the second request
  # fills its row exactly.
  mapping = jnp.array([[1], [0]], jnp.int32)
  expected = ragged_reference(q, kc, vc, k, v, cu, lengths, mapping,
                              causal=True)
  actual = run_paged(q, kc, vc, k, v, cu, lengths, mapping, causal=True)
  assert_extend_matches(actual, expected, cu, lengths, 2)


@pytest.mark.parametrize(
    "uniform,heads,kv_heads,causal,append,return_lse,num_active", [
        (False, 8, 1, True, True, True, None),
        (False, 8, 2, False, True, True, 2),
        (False, 16, 4, True, True, False, 2),
        (False, 32, 8, True, False, True, None),
        (True, 8, 1, False, True, True, None),
        (True, 16, 2, True, True, True, 2),
        (True, 32, 8, True, True, False, None),
        (True, 16, 4, False, False, True, None),
    ])
def test_multitoken_matches_reference(
    uniform, heads, kv_heads, causal, append, return_lse, num_active):
  # Note (david): the inactive request carries garbage lengths and table rows
  # that must never be read. No row sees a single key, whose lse carries Q's
  # full bf16 rounding (~LSE_TOL). uniform requests (no packed padding) stand
  # for the dense (batch, seqlen_q) calls.
  q, kc, vc, k, v, cu, lengths, table = random_ragged_case(
      lengths=(6, 6, 6) if uniform else (13, 1, 21), prefixes=(127, 3, 251),
      heads=heads, kv_heads=kv_heads, padding=0 if uniform else 7)
  active = len(lengths) if num_active is None else num_active
  lengths = lengths.at[active:].set(-100)
  table = table.at[active:].set(-1)
  k, v = (k, v) if append else (None, None)
  expected = ragged_reference(q, kc, vc, k, v, cu, lengths, table,
                              causal=causal, num_active=active)
  actual = run_extend(q, kc, vc, k, v, cu, lengths, table,
                      num_active=num_active, return_lse=return_lse,
                      causal=causal)
  assert_extend_matches(actual, expected, cu, lengths, active)


@pytest.mark.parametrize("page_size", [256, 512, 1024])
def test_ragged_cache_page_sizes(page_size):
  case = random_ragged_case(lengths=(5, 11), prefixes=(page_size - 2, 3),
                            page_size=page_size, capacity=2 * page_size,
                            heads=2, kv_heads=1, head_dim=128)
  expected = ragged_reference(*case, causal=True)
  actual = run_paged(*case, causal=True)
  assert_extend_matches(actual, expected, case[5], case[6], 2)


@pytest.mark.parametrize("append", [False, True])
def test_empty_packed_allocation(append):
  # Note (david): a zero-row buffer is outside the varlen contract (the
  # packed path rejects it too), so the allocation holds packed padding only.
  q, kc, vc, k, v, cu, lengths, table = random_ragged_case(
      lengths=(0, 0), prefixes=(0, 0), padding=8)
  k, v = (k, v) if append else (None, None)
  expected = ragged_reference(q, kc, vc, k, v, cu, lengths, table,
                              causal=True)
  actual = run_paged(q, kc, vc, k, v, cu, lengths, table, causal=True)
  assert_extend_matches(actual, expected, cu, lengths, 2)


@pytest.mark.parametrize("heads,kv_heads", [(132, 6), (128, 8)])
def test_ragged_cache_many_heads(heads, kv_heads):
  # Note (david): one q head per group, so 128+ q heads at d256 are as many
  # groups, each re-staging its kv head.
  case = random_ragged_case(lengths=(67, 11), prefixes=(3, 128), heads=heads,
                            kv_heads=kv_heads, head_dim=256, capacity=256,
                            padding=0)
  expected = ragged_reference(*case, causal=True)
  actual = run_paged(*case, causal=True)
  assert_extend_matches(actual, expected, case[5], case[6], 2)


def test_ragged_cache_prefill_gqa_regression():
  case = random_ragged_case(lengths=(237,), prefixes=(2048,), heads=32,
                            kv_heads=8, capacity=3072, padding=19)
  expected = ragged_reference(*case, causal=True)
  actual = run_paged(*case, causal=True)
  assert_extend_matches(actual, expected, case[5], case[6], 1)


def test_multitoken_replays_when_later_block_raises_anchor():
  # The cached keys jump from 0 to 32 at position 1024, so the later blocks'
  # scores sit hundreds of log2 units above the anchor the first block set: the
  # fixed-anchor pass must notice and replay with the rescaling update instead
  # of overflowing. Only out is checked: Q is scaled in bf16, a relative error
  # that at an lse near 370 exceeds the lse tolerance.
  q, kc, vc, k, v, cu, lengths, table = random_ragged_case(
      lengths=(64,), prefixes=(2048,), heads=4, kv_heads=2, capacity=4096,
      padding=0)
  late_pages = table[0, 1024 // kc.shape[1]:]
  kc = jnp.zeros_like(kc).at[late_pages].set(32.0)
  vc = jnp.ones_like(vc).at[late_pages].set(3.0)
  q, k, v = jnp.ones_like(q), jnp.full_like(k, 32.0), jnp.full_like(v, 3.0)
  case = (q, kc, vc, k, v, cu, lengths, table)
  expected = ragged_reference(*case, causal=True)
  actual = run_extend(*case, return_lse=False, causal=True)
  assert_extend_matches(actual, expected, case[5], case[6], 1)


@pytest.mark.skipif(INTERPRET,
                    reason="requires TPU optimized buffer assignment")
def test_ragged_cache_has_no_pool_sized_temporary():
  q = jax.ShapeDtypeStruct((8, 16, 128), jnp.bfloat16)
  cache = jax.ShapeDtypeStruct((1024, 128, 8, 128), jnp.bfloat16)

  def paged_attn(q, kv_cache, cu, seqused, table):
    return flash_attn_varlen_func(
        q, kv_cache, None, cu, None, 16, 256 * 128, causal=True,
        block_table=table, seqused_k=seqused)

  compiled = jax.jit(paged_attn).lower(
      q, cache, jax.ShapeDtypeStruct((3,), jnp.int32),
      jax.ShapeDtypeStruct((2,), jnp.int32),
      jax.ShapeDtypeStruct((2, 256), jnp.int32)).compile()
  memory = compiled.memory_analysis()
  pool_bytes = math.prod(cache.shape) * jnp.dtype(cache.dtype).itemsize
  # Note (david): the read-only kernel aliases nothing. A cache layout
  # conversion or a dense gather of the cache would allocate at least half the
  # pool; query transforms and scratch are far smaller.
  assert memory.temp_size_in_bytes < pool_bytes // 4, memory


@pytest.mark.parametrize("case,error,message", [
    ("seqused_k_without_block_table", ValueError, "seqused_k"),
    ("block_table_without_seqused_k", ValueError, "seqused_k"),
    ("cu_seqlens_k_with_block_table", ValueError, "cu_seqlens_k"),
    ("token_major", NotImplementedError, "(?i)token.major"),
    ("rotary_k", NotImplementedError, "rotary_k"),
    ("window_size", NotImplementedError, "window"),
    ("softcap", NotImplementedError, "softcap"),
    ("cu_seqlens_q_start", ValueError, "cu_seqlens_q"),
    ("num_active_past_batch", ValueError, "num_active"),
    ("num_active_jax_negative", ValueError, "num_active"),
    ("num_active_float", ValueError, "num_active"),
    ("float32", NotImplementedError, "(?i)bf16|bfloat16"),
    ("head_dim_64", ValueError, "head_dim"),
    ("kv_pair", ValueError, "merged"),
    ("odd_cache_heads", ValueError, "merged"),
])
def test_paged_varlen_rejects_unsupported_calls(case, error, message):
  # Note (david): each case changes one knob of an otherwise valid paged call.
  heads, kv_heads, total, head_dim = 4, 2, 8, 128
  q = jnp.zeros((heads, total, head_dim), jnp.bfloat16)
  cu = jnp.array([0, 3, total], jnp.int32)
  operands = [q, jnp.zeros((2, 128, 2 * kv_heads, head_dim), jnp.bfloat16),
              None, cu, None]
  kwargs = dict(causal=True, interpret=INTERPRET,
                block_table=jnp.array([[0], [1]], jnp.int32),
                seqused_k=jnp.array([5, 9], jnp.int32))
  if case == "seqused_k_without_block_table":
    packed_kv = jnp.zeros((kv_heads, total, head_dim), jnp.bfloat16)
    operands[1:] = [packed_kv, packed_kv, cu, cu]
    kwargs["block_table"] = None
  elif case == "block_table_without_seqused_k":
    kwargs["seqused_k"] = None
  elif case == "cu_seqlens_k_with_block_table":
    operands[4] = cu
  elif case == "token_major":
    operands[0] = q.transpose(1, 0, 2).reshape(total, heads * head_dim)
    kwargs.update(token_major=True, head_dim=head_dim)
  elif case == "rotary_k":
    rope_table = jnp.ones((128, head_dim // 2), jnp.float32)
    kwargs.update(rotary_cos=rope_table, rotary_sin=rope_table, rotary_k=True)
  elif case == "window_size":
    kwargs["window_size"] = (16, 0)
  elif case == "softcap":
    kwargs["softcap"] = 5.0
  elif case == "cu_seqlens_q_start":
    operands[3] = jnp.array([2, 3, total], jnp.int32)
  elif case == "num_active_past_batch":
    kwargs["num_active"] = 3
  elif case == "num_active_jax_negative":
    kwargs["num_active"] = jnp.int32(-1)
  elif case == "num_active_float":
    kwargs["num_active"] = jnp.float32(1.0)
  elif case == "float32":
    operands[:2] = [operand.astype(jnp.float32) for operand in operands[:2]]
  elif case == "kv_pair":
    pages = jnp.zeros((2, 128, kv_heads, head_dim), jnp.bfloat16)
    operands[1:3] = [pages, pages]
  elif case == "odd_cache_heads":
    operands[1] = jnp.zeros((2, 128, 3, head_dim), jnp.bfloat16)
  else:
    operands[:2] = [operand[..., :64] for operand in operands[:2]]
  with pytest.raises(error, match=message):
    flash_attn_varlen_func(*operands, total, 128, **kwargs)


def paged_tiles(max_seqlen_q_bucket, max_seqlen_k_bucket, *, head_dim=128,
                page_size=128, block_sizes=None):
  """resolve_paged_tiles with no lse or rotary."""
  return resolve_paged_tiles(
      max_seqlen_q_bucket=max_seqlen_q_bucket,
      max_seqlen_k_bucket=max_seqlen_k_bucket, page_size=page_size,
      head_dim=head_dim, return_lse=False, rotary_dtype=None,
      block_sizes=block_sizes)


@pytest.mark.parametrize("shape,dtype,expected_bytes", [
    ((2, 512, 12, 128), jnp.bfloat16, 2 * 512 * 16 * 128 * 2),
    ((2, 512, 1, 2, 128), jnp.bfloat16, 2 * 512 * 2 * 128 * 2),
    ((2, 1, 2, 128, 64), jnp.float32, 2 * 2 * 128 * 128 * 4),
    ((1, 2, 512), jnp.int32, 2 * 512 * 4),
])
def test_vmem_buffer_bytes_pads_to_mosaic_tiles(shape, dtype, expected_bytes):
  # Note (david): an odd second-minor axis pads to a power of two (12 to 16),
  # a staged bf16 (K, V) row pair costs nothing, a half-lane rotary plane pads
  # to 128, and the two int32 bounds rows pad to 2, not 8.
  assert vmem_buffer_bytes(shape, jnp.dtype(dtype)) == expected_bytes


class CapturedPallasCall(Exception):
  """Stops forward_common at its intercepted pallas_call."""


@pytest.mark.parametrize("heads,kv_heads,head_dim,interpret,return_lse,rotary", [
    (32, 8, 128, False, False, False),
    (32, 8, 128, True, True, True),
    (6, 3, 128, False, True, False),
    (8, 1, 256, False, True, True),
])
def test_paged_vmem_estimate_tracks_forward_common_scratch(
    monkeypatch, heads, kv_heads, head_dim, interpret, return_lse, rotary):
  # Note (david): forward_common's pallas_call is intercepted before any
  # lowering, so the TPU build (interpret=False) is inspected on CPU as well.
  captured = []

  def _capture_pallas_call(*unused_args, scratch_shapes, **unused_kwargs):
    captured.extend(scratch_shapes)
    raise CapturedPallasCall

  monkeypatch.setattr(fwd_pipeline.pl, "pallas_call", _capture_pallas_call)
  page_size, pages_per_seq, total_q = 128, 4, 200
  cache = jnp.zeros((8, page_size, 2 * kv_heads, head_dim), jnp.bfloat16)
  blocks = BlockSizes(block_q=256, block_kv=256, block_kv_compute=128,
                      block_q_compute=128)
  if rotary:
    rotary_dtype = jnp.dtype(jnp.float32)
    rotary_pair = (
        jnp.zeros((2, 1, total_q, head_dim // 2), rotary_dtype), None)
  else:
    rotary_dtype, rotary_pair = None, None
  with pytest.raises(CapturedPallasCall):
    flash_attn_varlen_paged(
        jnp.zeros((heads, total_q, head_dim), jnp.bfloat16), cache,
        jnp.array([0, 90, 180], jnp.int32), jnp.array([300, 500], jnp.int32),
        jnp.zeros((2, pages_per_seq), jnp.int32), max_seqlen_q=90,
        max_seqlen_k=500, causal=True, q_scale=1.0, return_lse=return_lse,
        interpret=interpret, block_sizes=blocks, rotary=rotary_pair)
  allocated = [(tuple(ref.shape), jnp.dtype(ref.dtype)) for ref in captured
               if ref.memory_space == pltpu.VMEM]
  assert allocated == paged_scratch_shapes(
      blocks, head_dim=head_dim, return_lse=return_lse,
      rotary_dtype=rotary_dtype)


@pytest.mark.parametrize(
    "head_dim,max_seqlen_q_bucket,max_seqlen_k_bucket,expected", [
        (128, 128, 40960, (128, 2048)),
        (128, 8192, 40960, (2048, 1024)),
        (128, 2048, 32768, (2048, 1024)),
        (128, 1024, 32768, (1024, 2048)),
        (128, 64, 128, (128, 128)),
        (128, 384, 640, (256, 640)),
        (256, 1024, 16384, (1024, 2048)),
        (256, 2048, 256, (2048, 256)),
    ])
def test_paged_tiles_minimize_prefix_streams_within_vmem(
    head_dim, max_seqlen_q_bucket, max_seqlen_k_bucket, expected):
  # Note (david): one q head per group keeps every build far inside the VMEM
  # budget, so the q block grows to the bucket (at most 2048 rows) and the
  # prefix is streamed once per q block; the kv block takes the most pages
  # within the 2048 x 1024 block area, so a 2048-row q block streams 1024-token
  # kv blocks. A 384-row bucket takes one 256-row
  # block fewer than three 128-row ones; a 640-token bucket (a five-page
  # table row) is one five-page block, whose kv compute tile is 128, the
  # largest one dividing it.
  blocks = paged_tiles(max_seqlen_q_bucket, max_seqlen_k_bucket,
                       head_dim=head_dim)
  assert (blocks.block_q, blocks.block_kv) == expected
  assert blocks.block_q_compute == min(256, blocks.block_q // 2)
  if blocks.block_kv == 640:
    assert blocks.block_kv_compute == 128
  else:
    assert blocks.block_kv_compute == min(512, blocks.block_kv)
  assert estimate_vmem_bytes(
      blocks, head_dim=head_dim, return_lse=False,
      rotary_dtype=None) <= vmem_limit_bytes()


def test_paged_tiles_pins_and_page_multiples():
  # Note (david): a pinned build is only checked against the budget; block_kv
  # holds the most whole pages, at least one, within the kv bucket and 2048
  # tokens, for any multiple-of-128 page size.
  pinned = BlockSizes(block_q=1024, block_kv=512, block_kv_compute=256,
                      block_q_compute=256)
  assert paged_tiles(8192, 40960, block_sizes=pinned) == pinned
  for page_size, kv_bucket, block_kv, block_kv_compute in [
      (384, 128, 384, 384), (512, 1536, 1536, 512), (640, 4096, 1920, 384),
      (4096, 8192, 4096, 512)]:
    blocks = paged_tiles(128, kv_bucket, page_size=page_size)
    assert (blocks.block_kv, blocks.block_kv_compute) == (
        block_kv, block_kv_compute)


def test_paged_tiles_reject_pinned_vmem_overflow():
  # Note (david): a pinned build past the budget raises; nothing shrinks
  # behind the pin. A 65536-token kv block stages ~200 MiB at d256.
  with pytest.raises(ValueError, match="VMEM"):
    paged_tiles(8192, 65536, head_dim=256, block_sizes=BlockSizes(
        block_q=2048, block_kv=65536, block_kv_compute=512,
        block_q_compute=256))


# Note (david): with N(0, 1) inputs attention spreads over the whole visible
# range, so an off-by-one causal limit or one misplaced page moves out by ~1/N,
# far below the bf16 tolerances once a prefix holds thousands of tokens. Here
# every key is a unit code and each query head aims at one or two chosen key
# positions with a 12-16 nat logit, so a visibility, paging or row-placement
# error moves out/lse by O(1). Stale cache slots hold copies of live codes, so
# attending one is loud too.
def probe_case(lengths, prefixes, *, heads=4, kv_heads=2, head_dim=128,
               page_size=128, pages_per_seq=None, padding=0, batch=None,
               tail_owned=False, unused_pages=None, causal=True, logit=12.0,
               softmax_scale=None, seed=7):
  """Needle-probe inputs in random_ragged_case's tuple layout.

  lengths/prefixes describe the active requests (num_active = len(lengths)).
  Slots past them, up to batch, are inactive with garbage cache_seqlens and
  block-table rows; with tail_owned their cu entries keep growing into the
  packed pad tail. unused_pages=None fills an active row's table past its live
  pages with foreign valid page ids, otherwise with that value. Even heads aim
  at the row itself, the row before it or a recent row, plus the first hidden
  key (causal) or a random request row (not causal). Odd heads aim at a
  visible position, half the time one next to a page, 256/512/1024/2048-token
  or 128-row chunk edge.
  """
  rng = np.random.default_rng(seed)
  active = len(lengths)
  batch = active if batch is None else batch
  scale = 1 / math.sqrt(head_dim) if softmax_scale is None else softmax_scale
  totals = [prefix + length for prefix, length in zip(prefixes, lengths)]
  live_pages = [-(-total // page_size) for total in totals]
  pages_per_seq = max(live_pages) if pages_per_seq is None else pages_per_seq
  num_pages = sum(live_pages) + 3

  owners = rng.permutation(num_pages)
  if unused_pages is None:
    table = rng.integers(0, num_pages, (batch, pages_per_seq))
  else:
    table = np.full((batch, pages_per_seq), unused_pages)
  table[active:] = rng.integers(-2**20, 2**20, (batch - active, pages_per_seq))
  first_page = np.cumsum([0, *live_pages])
  for seq in range(active):
    table[seq, :live_pages[seq]] = owners[first_page[seq]:first_page[seq + 1]]
  cache_seqlens = rng.integers(-1000, 10**6, batch)
  cache_seqlens[:active] = prefixes
  cu = np.zeros(batch + 1, np.int64)
  cu[1:active + 1] = np.cumsum(lengths)
  cu[active + 1:] = cu[active]
  total_q = int(cu[active]) + padding
  if tail_owned:
    cu[active + 1:] += np.sort(rng.integers(0, padding + 1, batch - active))

  def unit(shape):
    directions = rng.normal(size=shape)
    return directions / np.linalg.norm(directions, axis=-1, keepdims=True)

  codes = [unit((total, kv_heads, head_dim)) for total in totals]
  values = [rng.normal(size=(total, kv_heads, head_dim)) for total in totals]
  all_codes = np.concatenate(codes)
  kc = all_codes[rng.integers(0, len(all_codes), (num_pages, page_size))]
  vc = rng.normal(size=kc.shape)
  q = rng.normal(size=(total_q, heads, head_dim))
  k = rng.normal(size=(total_q, kv_heads, head_dim))
  v = rng.normal(size=k.shape)
  group = np.arange(heads) // (heads // kv_heads)
  for seq, (prefix, length, total) in enumerate(zip(prefixes, lengths, totals)):
    own = table[seq, :live_pages[seq]]
    # Note (david): stale slots of the request's own pages, the append target
    # included, copy its own codes: the strongest distractors for its queries.
    kc[own] = codes[seq][rng.integers(0, total, (len(own), page_size))]
    positions = np.arange(prefix)
    slots = (table[seq, positions // page_size], positions % page_size)
    kc[slots] = codes[seq][:prefix]
    vc[slots] = values[seq][:prefix]
    start = cu[seq]
    k[start:start + length] = codes[seq][prefix:]
    v[start:start + length] = values[seq][prefix:]
    if length == 0:
      continue
    rows = prefix + np.arange(length)
    last = rows if causal else np.full(length, total - 1)
    marks = np.concatenate(
        [np.arange(0, total + 1, step)
         for step in {page_size, 256, 512, 1024, 2048}]
        + [prefix + np.arange(0, length + 1, 128)])
    edges = np.unique(np.concatenate([marks - 1, marks]))
    edges = edges[(edges >= 0) & (edges < total)]
    for head in range(heads):
      if head % 2 == 0 and causal:
        choice = rng.integers(0, 4, length)
        target = np.select(
            [choice == 0, choice == 1, choice == 2],
            [rows, np.maximum(rows - 1, 0), rng.integers(prefix, rows + 1)],
            rng.integers(np.maximum(rows - 700, 0), rows + 1))
        second = np.where(rows + 1 < total, rows + 1, -1)
      elif head % 2 == 0:
        target = rng.integers(0, total, length)
        second = rng.integers(prefix, total, length)
      else:
        count = np.searchsorted(edges, last, side="right")
        target = np.where(rng.random(length) < 0.5,
                          edges[rng.integers(0, count)],
                          rng.integers(0, last + 1))
        second = np.full(length, -1)
      direction = codes[seq][target, group[head]] + np.where(
          second[:, None] >= 0, codes[seq][second, group[head]], 0)
      q[start:start + length, head] = logit / scale * direction

  def bf16(array):
    return jnp.asarray(array, jnp.bfloat16)

  return (bf16(q), bf16(kc), bf16(vc), bf16(k), bf16(v),
          jnp.asarray(cu, jnp.int32), jnp.asarray(cache_seqlens, jnp.int32),
          jnp.asarray(table, jnp.int32))


def assert_probe_is_sharp(top2):
  # Note (david): without needle-dominated attention the probe tests lose
  # their detection power.
  assert np.median(top2) > 0.9 and np.quantile(top2, 0.02) > 0.5, (
      f"probe attention is diffuse: median top-2 mass {np.median(top2):.3f},"
      f" 2% quantile {np.quantile(top2, 0.02):.3f}")


@pytest.mark.parametrize("padding,causal", [
    (0, True), (5, True), (27, True), (27, False)])
def test_paged_unaligned_sequence_boundaries(padding, causal):
  # Note (david): cu = 0 1 15 15 36 43 46 47 58 61 puts boundaries at every
  # residue mod 8, with an empty request in the middle, one-row requests, and
  # the last request ending at the buffer end or inside the pad tail.
  lengths = (1, 14, 0, 21, 7, 3, 1, 11, 3)
  case = probe_case(lengths, (0, 5, 64, 130, 250, 381, 127, 128, 255),
                    pages_per_seq=4, padding=padding, causal=causal)
  expected = ragged_reference(*case, causal=causal)
  assert_probe_is_sharp(expected[4])
  actual = run_paged(*case, causal=causal)
  assert_extend_matches(actual, expected, case[5], case[6], len(lengths))


@pytest.mark.parametrize("lengths,prefixes,page_size,causal", [
    ((1300,), (4000,), 256, True),
    ((700, 300), (1500, 2900), 128, True),
    ((300, 700), (2900, 1500), 128, False),
])
def test_paged_chunked_prefill_long_prefix(lengths, prefixes, page_size,
                                           causal):
  # Note (david): max_seqlen_q = 256 underestimates the chunks, which only caps
  # the q block, so one request splits across several q blocks and its causal
  # limit must advance row by row across every block edge.
  case = probe_case(lengths, prefixes, head_dim=128, page_size=page_size,
                    padding=13, causal=causal, logit=16.0)
  expected = ragged_reference(*case, causal=causal)
  assert_probe_is_sharp(expected[4])
  actual = run_paged(*case, causal=causal, max_seqlen_q=256)
  assert_extend_matches(actual, expected, case[5], case[6], len(lengths))


@pytest.mark.parametrize("page_size", [128, 384, 640, 1024, 2048])
def test_paged_causal_limits_inside_kv_blocks(page_size):
  # Note (david): a paged KV block holds whole pages, from many 128-token
  # pages through five 384-token or three 640-token pages (1920 tokens, on no
  # power-of-two edge) down to one 2048-token page, and the causal limits
  # (2000 -> 2096, 1023 -> 1068, 500 -> 520) cross 512/1024/2048 mid-chunk,
  # so the limit falls inside a block.
  lengths = (96, 45, 20)
  case = probe_case(lengths, (2000, 1023, 500), page_size=page_size,
                    padding=7, logit=16.0)
  expected = ragged_reference(*case, causal=True)
  assert_probe_is_sharp(expected[4])
  actual = run_paged(*case, causal=True)
  assert_extend_matches(actual, expected, case[5], case[6], len(lengths))


@pytest.mark.parametrize("tail_owned", [False, True])
def test_paged_mixed_prefixes_and_inactive_garbage(tail_owned):
  # Note (david): prefixes 0, 1 and both sides of a page edge sit next to a
  # long one, in a 9-slot batch whose 3 inactive slots carry garbage lengths
  # and table rows; active table entries past the live pages are -1.
  lengths = (9, 1, 17, 8, 35, 61)
  case = probe_case(lengths, (0, 1, 127, 128, 129, 1000), batch=9,
                    pages_per_seq=12, padding=21, tail_owned=tail_owned,
                    unused_pages=-1, logit=14.0)
  expected = ragged_reference(*case, causal=True, num_active=len(lengths))
  assert_probe_is_sharp(expected[4])
  actual = run_paged(*case, causal=True, num_active=len(lengths))
  assert_extend_matches(actual, expected, case[5], case[6], len(lengths))


@pytest.mark.parametrize("heads,kv_heads,return_lse", [
    (32, 8, False), (32, 8, True), (16, 4, False), (8, 2, True)])
def test_paged_serving_gqa_shape(heads, kv_heads, return_lse):
  # Note (david): the tpu-inference FlyWheel backend's prefill step for
  # Qwen3-4B: 64 padded slots with 320-page table rows, 3 active requests in a
  # 512-token bucket, called without lse when serving. The 16/4 and 8/2 cases
  # are one TP 2 / TP 4 shard's heads, where few KV heads share one merged
  # row. The bounds are the backend's: the token bucket, and the table row's
  # capacity for max_model_len.
  lengths = (150, 131, 169)
  scale = 1 / math.sqrt(128)
  case = probe_case(lengths, (600, 641, 577), heads=heads, kv_heads=kv_heads,
                    head_dim=128, batch=64, pages_per_seq=320, padding=62,
                    logit=16.0, softmax_scale=scale)
  expected = ragged_reference(*case, causal=True, num_active=len(lengths),
                              softmax_scale=scale)
  assert_probe_is_sharp(expected[4])
  actual = run_paged(*case, causal=True, num_active=len(lengths),
                     return_lse=return_lse, softmax_scale=scale,
                     max_seqlen_q=case[0].shape[0], max_seqlen_k=320 * 128)
  assert_extend_matches(actual, expected, case[5], case[6], len(lengths))


def test_paged_fresh_chunk_q_blocks_share_one_kv_block():
  # Note (david): a 600-token fresh chunk in 128-row q blocks over one
  # 2048-token kv block: each q block loads the block only up to its own
  # frontier, so consecutive q blocks stage the same kv block with different
  # extents and must not reuse each other's staging, and most kv compute
  # tiles of every q tile lie past its frontier and are skipped.
  lengths = (600,)
  case = probe_case(lengths, (0,), pages_per_seq=16, padding=11, logit=16.0)
  expected = ragged_reference(*case, causal=True)
  assert_probe_is_sharp(expected[4])
  actual = run_paged(*case, causal=True, max_seqlen_q=128,
                     max_seqlen_k=16 * 128)
  assert_extend_matches(actual, expected, case[5], case[6], len(lengths))


def test_paged_overflow_guard_checks_last_live_kv_tile():
  # Note (david): keys [0, 512) score 0 and keys [512, 600) score ~200 nats,
  # so rows past 512 anchor on the first kv tile and overflow on the second.
  # Every q tile's frontier is below the block's last kv tile, which is
  # skipped, so the overflow check must run at the last live tile for the
  # replay to rescue those rows.
  length, kv_heads, heads, head_dim = 600, 2, 4, 128
  num_pages, page_size = 16, 128
  scale = 1 / math.sqrt(head_dim)
  direction = np.zeros(head_dim)
  direction[0] = 1.0
  magnitude = math.sqrt(200.0 / scale)
  q = np.broadcast_to(magnitude * direction, (length, heads, head_dim))
  k = np.zeros((length, kv_heads, head_dim))
  k[512:] = magnitude * direction
  v = np.where(np.arange(length)[:, None, None] < 512, 1.0, 3.0) * np.ones(
      (length, kv_heads, head_dim))
  pages = np.zeros((num_pages, page_size, kv_heads, head_dim))

  def bf16(array):
    return jnp.asarray(array, jnp.bfloat16)

  case = (bf16(q), bf16(pages), bf16(pages), bf16(k), bf16(v),
          jnp.array([0, length], jnp.int32), jnp.array([0], jnp.int32),
          jnp.arange(num_pages, dtype=jnp.int32)[None])
  expected = ragged_reference(*case, causal=True)
  actual = run_paged(*case, causal=True, max_seqlen_k=num_pages * page_size)
  assert np.all(np.isfinite(np.asarray(actual[0], np.float32)))
  assert_extend_matches(actual, expected, case[5], case[6], 1)
