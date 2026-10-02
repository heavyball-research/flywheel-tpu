"""Packed multi-token attention over a paged KV cache.

The cache cases append the new tokens with append_ragged, then read the whole
cache through flash_attn_varlen_func(block_table=...). The kv-outer extend
kernel, which flash_attn_with_kvcache still runs for multi-token calls, keeps
its direct kernel tests and the empty-cache cases as the A/B reference.
"""

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from flywheel_tpu import flash_attn_varlen_func, flash_attn_with_kvcache
from flywheel_tpu.pallas import fwd_pipeline
from flywheel_tpu.pallas.block_sizes import (
    FWD_BLOCKS,
    VMEM_LIMIT_BYTES,
    BlockSizes,
)
from flywheel_tpu.pallas.flash_fwd_kvcache_extend import (
    flash_attn_kvcache_extend_pallas,
)
from flywheel_tpu.pallas.flash_fwd_kvcache_varlen import append_ragged
from flywheel_tpu.pallas.flash_fwd_varlen_paged import (
    estimate_vmem_bytes,
    flash_attn_varlen_paged,
    paged_kv_info,
    paged_scratch_shapes,
    resolve_paged_tiles,
    vmem_buffer_bytes,
)

INTERPRET = jax.default_backend() != "tpu"
TPU_ONLY = pytest.mark.skipif(
    INTERPRET, reason="pair-packed loads bitcast on TPU only")
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


def rope_tables(length, head_dim, theta=10000.0):
  """float32 (length, head_dim / 2) cos and sin tables."""
  frequency = theta ** (-np.arange(0, head_dim, 2) / head_dim)
  angle = np.arange(length)[:, None] * frequency
  return (jnp.asarray(np.cos(angle), jnp.float32),
          jnp.asarray(np.sin(angle), jnp.float32))


def rotate_queries(q, cos, sin, cu, seqused_k, *, interleaved):
  """q in float32 with each request's rows rotated at their bottom-right
  positions seqused_k[r] - q_len[r] + t; other rows stay as they are."""
  q = np.asarray(q, np.float32)
  cos, sin, cu, seqused_k = map(np.asarray, (cos, sin, cu, seqused_k))
  rotated = q.copy()
  half = q.shape[-1] // 2
  for seq in range(len(seqused_k)):
    start, end = cu[seq:seq + 2]
    positions = seqused_k[seq] - (end - start) + np.arange(end - start)
    cosine, sine = cos[positions][:, None], sin[positions][:, None]
    if interleaved:
      first, second = q[start:end, :, 0::2], q[start:end, :, 1::2]
      rotated[start:end, :, 0::2] = first * cosine - second * sine
      rotated[start:end, :, 1::2] = first * sine + second * cosine
    else:
      first, second = q[start:end, :, :half], q[start:end, :, half:]
      rotated[start:end, :, :half] = first * cosine - second * sine
      rotated[start:end, :, half:] = first * sine + second * cosine
  return rotated


def contiguous_case(q, kc, vc, k, v, cu, lengths, table):
  """A random_ragged_case over the contiguous K/V pair: each request's pages
  gathered into one (capacity, kv_heads, head_dim) cache row, which the
  reference reads as that row's single page."""
  batch, pages_per_seq = table.shape
  kc, vc = (pages[table].reshape(
      batch, pages_per_seq * pages.shape[1], *pages.shape[2:])
            for pages in (kc, vc))
  return (q, kc, vc, k, v, cu, lengths,
          jnp.arange(batch, dtype=jnp.int32)[:, None])


def run_extend(q, kc, vc, k, v, cu, lengths, table, *, layout="merged",
               return_lse=True, num_active=None, **kwargs):
  """flash_attn_with_kvcache on the "merged" pool (with the block table, the
  updated pool split back into its K and V heads) or on a contiguous_case's
  "pair" rows; returns (out, lse or None, updated_k, updated_v)."""
  kwargs.update(
      cache_seqlens=lengths, cu_seqlens_q=cu,
      num_active=None if num_active is None else jnp.int32(num_active),
      return_softmax_lse=return_lse, interpret=INTERPRET)
  if layout == "merged":
    *result, cache = flash_attn_with_kvcache(
        q, jnp.concatenate((kc, vc), axis=2), None, k, v, block_table=table,
        **kwargs)
    updated = jnp.split(cache, 2, axis=2)
  else:
    *result, updated_k, updated_v = flash_attn_with_kvcache(
        q, kc, vc, k, v, **kwargs)
    updated = (updated_k, updated_v)
  return (result[0], result[1] if return_lse else None, *updated)


def append_tokens(caches, k, v, cu, lengths, table, num_active):
  """append_ragged of k/v after the first num_active requests'
  cache_seqlens, into the (merged pool,) or (k_pages, v_pages) caches."""
  is_merged = len(caches) == 1
  return append_ragged(
      caches[0], None if is_merged else caches[1], k, v, cu, lengths,
      table.reshape(-1), jnp.full((1,), num_active, jnp.int32),
      page_size=caches[0].shape[1], pages_per_seq=table.shape[1],
      interpret=INTERPRET, merged_cache=is_merged)


def run_paged(q, kc, vc, k, v, cu, lengths, table, *, layout="merged",
              return_lse=True, num_active=None, max_seqlen_q=None,
              max_seqlen_k=None, **kwargs):
  """Appends k/v (when given) with append_ragged, then reads the whole paged
  cache through flash_attn_varlen_func: one "merged" (pages, page_size,
  2 * kv_heads, dim) pool with v=None, or the "pair" of K and V page pools.

  seqused_k is cache_seqlens plus the appended tokens. q and out are
  head-major in the call and in the reference's token-major layout here. The
  max_seqlen bounds default to the active requests' exact maxima. Returns
  (out, lse or None, k_pages, v_pages).
  """
  active = table.shape[0] if num_active is None else num_active
  if layout == "merged":
    caches = (jnp.concatenate((kc, vc), axis=2),)
  else:
    caches = (kc, vc)
  if k is None:
    seqused = lengths
  else:
    caches = append_tokens(caches, k, v, cu, lengths, table, active)
    seqused = lengths + jnp.diff(cu)
  if max_seqlen_q is None:
    max_seqlen_q = max(
        1, int(np.diff(np.asarray(cu))[:active].max(initial=0)))
  if max_seqlen_k is None:
    max_seqlen_k = max(1, int(np.asarray(seqused)[:active].max(initial=0)))
  if layout == "merged":
    k_cache, v_cache = caches[0], None
  else:
    k_cache, v_cache = caches
  result = flash_attn_varlen_func(
      q.transpose(1, 0, 2), k_cache, v_cache, cu, None, max_seqlen_q,
      max_seqlen_k, return_softmax_lse=return_lse, interpret=INTERPRET,
      block_table=table, seqused_k=seqused,
      num_active=None if num_active is None else jnp.int32(num_active),
      **kwargs)
  out, lse = result if return_lse else (result, None)
  if layout == "merged":
    k_pages, v_pages = jnp.split(k_cache, 2, axis=2)
  else:
    k_pages, v_pages = k_cache, v_cache
  return out.transpose(1, 0, 2), lse, k_pages, v_pages


def assert_extend_matches(actual, expected, cu, lengths, num_active, *,
                          skip_unseen_rows=False):
  """Owned rows [cu[0], cu[num_active]) match the reference, packed padding
  rows past them are out = 0 and lse = -inf, and the appended caches match
  exactly. lse is None when the call did not return it. Rows before cu[0]
  are the caller's; skip_unseen_rows also leaves out the rows that see no
  key, which the paged path does not specify."""
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


@pytest.mark.parametrize("rotary", [None, "interleaved", "half"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("append", [False, True])
@pytest.mark.parametrize("layout", ["pair", "merged"])
def test_ragged_cache_matches_reference(causal, append, layout, rotary):
  q, kc, vc, k, v, cu, lengths, table = random_ragged_case()
  k, v = (k, v) if append else (None, None)
  kwargs = dict(causal=causal)
  reference_q = q
  if rotary is not None:
    # Note (david): the cached K is already rotated, so only Q turns, at its
    # bottom-right position seqused_k - q_len + t; the reference attends with
    # Q rotated up front in f32.
    interleaved = rotary == "interleaved"
    cos, sin = rope_tables(table.shape[1] * kc.shape[1], q.shape[-1])
    seqused = lengths + (jnp.diff(cu) if append else 0)
    reference_q = rotate_queries(q, cos, sin, cu, seqused,
                                 interleaved=interleaved)
    kwargs.update(rotary_cos=cos, rotary_sin=sin,
                  rotary_interleaved=interleaved, rotary_k=False)
  expected = ragged_reference(reference_q, kc, vc, k, v, cu, lengths, table,
                              causal=causal)
  actual = run_paged(q, kc, vc, k, v, cu, lengths, table, layout=layout,
                     **kwargs)
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


@pytest.mark.parametrize("heads,kv_heads,dim,layout", [
    (6, 6, 128, "merged"), (16, 8, 128, "merged"), (16, 16, 128, "merged"),
    (24, 4, 256, "merged"), (32, 16, 128, "merged"), (6, 3, 128, "merged"),
    (6, 3, 128, "pair")])
def test_ragged_cache_head_tiles(heads, kv_heads, dim, layout):
  # Note (david): the paged path stages whole token rows, so any KV head
  # count works, odd ones included.
  case = random_ragged_case(lengths=(3, 5), prefixes=(37, 128), heads=heads,
                            kv_heads=kv_heads, head_dim=dim, capacity=256,
                            padding=0)
  expected = ragged_reference(*case, causal=True)
  actual = run_paged(*case, layout=layout, causal=True)
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
@pytest.mark.parametrize("layout", ["pair", "merged"])
def test_ragged_read_only_short_cache_and_nan_padding(layout, causal):
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
                     layout=layout, causal=causal)
  assert_extend_matches(actual, expected, cu, lengths, 2,
                        skip_unseen_rows=True)


@pytest.mark.parametrize("layout", ["pair", "merged"])
def test_ragged_read_only_empty_cache_between_cached_requests(layout):
  # Note (david): rows over an empty cache run no KV block, so they issue no
  # prefetch for the next request; a kernel that counts that prefetch as
  # issued waits on it forever.
  case = random_ragged_case(lengths=(4, 3, 9, 2), prefixes=(0, 130, 0, 5),
                            padding=5)
  q, kc, vc, _, _, cu, lengths, table = (
      case if layout == "merged" else contiguous_case(*case))
  expected = ragged_reference(q, kc, vc, None, None, cu, lengths, table,
                              causal=False)
  actual = run_extend(q, kc, vc, None, None, cu, lengths, table,
                      layout=layout, causal=False)
  assert_extend_matches(actual, expected, cu, lengths, 4)


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
  kv_pages = jnp.concatenate((kc, vc), axis=2)

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
    kc, vc = jnp.split(kv_pages, 2, axis=2)
    expected = ragged_reference(q, kc, vc, k, v, boundaries, lens, pages,
                                causal=True, num_active=active)
    (kv_pages,) = append_tokens((kv_pages,), k, v, boundaries, lens, pages,
                                active)
    out = compiled(q.transpose(1, 0, 2), kv_pages, boundaries,
                   lens + jnp.diff(boundaries), pages, jnp.int32(active))
    assert_extend_matches(
        (out.transpose(1, 0, 2), None, *jnp.split(kv_pages, 2, axis=2)),
        expected, boundaries, lens, active)


@pytest.mark.parametrize("layout", ["pair", "merged"])
@pytest.mark.parametrize("head_dim", [128, 256])
def test_multitoken_contiguous_cache_and_mapping(layout, head_dim):
  q, kc, vc, k, v, cu, lengths, table = random_ragged_case(
      lengths=(3, 3), prefixes=(127, 509), head_dim=head_dim, padding=0)
  kc = kc[table].reshape(2, 512, 2, head_dim)
  vc = vc[table].reshape(2, 512, 2, head_dim)
  # Note (david): a contiguous cache is one 512-token page per row, mapped to
  # its request through a one-column block table (cache_batch_idx's role);
  # the second request fills its row exactly.
  mapping = jnp.array([[1], [0]], jnp.int32)
  expected = ragged_reference(q, kc, vc, k, v, cu, lengths, mapping,
                              causal=True)
  actual = run_paged(q, kc, vc, k, v, cu, lengths, mapping, layout=layout,
                     causal=True)
  assert_extend_matches(actual, expected, cu, lengths, 2)


@pytest.mark.parametrize(
    "uniform,layout,heads,kv_heads,causal,append,return_lse,num_active", [
        (False, "merged", 8, 1, True, True, True, None),
        (False, "merged", 8, 2, False, True, True, 2),
        (False, "pair", 16, 4, True, True, False, 2),
        (False, "merged", 32, 8, True, False, True, None),
        (True, "pair", 8, 1, False, True, True, None),
        (True, "merged", 16, 2, True, True, True, 2),
        (True, "pair", 32, 8, True, True, False, None),
        (True, "merged", 16, 4, False, False, True, None),
    ])
def test_multitoken_kept_layouts_match_reference(
    uniform, layout, heads, kv_heads, causal, append, return_lse,
    num_active):
  # Note (david): the inactive request carries garbage lengths and table rows
  # that must never be read. No row sees a single key, whose lse carries Q's
  # full bf16 rounding (~LSE_TOL). uniform requests (no packed padding) stand
  # for the dense (batch, seqlen_q) calls.
  q, kc, vc, k, v, cu, lengths, table = random_ragged_case(
      lengths=(6, 6, 6) if uniform else (13, 1, 21), prefixes=(127, 3, 251),
      heads=heads, kv_heads=kv_heads, padding=0 if uniform else 7)
  batch = len(lengths)
  active = batch if num_active is None else num_active
  lengths = lengths.at[active:].set(-100)
  if layout == "merged":
    table = table.at[active:].set(-1)
  else:
    # Note (david): the K/V pair rows are contiguous caches, one 512-token
    # page per row, mapped to their requests through a one-column table.
    kc, vc = (pages[table].reshape(batch, 512, kv_heads, 128)
              for pages in (kc, vc))
    table = jnp.array([[2], [0], [1]], jnp.int32).at[active:].set(-1)
  k, v = (k, v) if append else (None, None)
  expected = ragged_reference(q, kc, vc, k, v, cu, lengths, table,
                              causal=causal, num_active=active)
  actual = run_paged(q, kc, vc, k, v, cu, lengths, table, layout=layout,
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


@pytest.mark.parametrize("heads,kv_heads,head_dim",
                         [(132, 6, 256), (128, 8, 256), (192, 1, 128)])
def test_ragged_cache_large_head_fold(heads, kv_heads, head_dim):
  # Note (david): the VMEM budget folds 3 of 6 and 4 of 8 KV heads; one KV
  # head under 192 q heads folds past one 128-lane lse row. 256 q heads on
  # one d256 KV head fit no fold (test_paged_tiles_reject_vmem_overflow).
  case = random_ragged_case(lengths=(67, 11), prefixes=(3, 128), heads=heads,
                            kv_heads=kv_heads, head_dim=head_dim,
                            capacity=256, padding=0)
  expected = ragged_reference(*case, causal=True)
  actual = run_paged(*case, causal=True)
  assert_extend_matches(actual, expected, case[5], case[6], 2)


@pytest.mark.parametrize("layout", ["pair", "merged"])
def test_ragged_cache_prefill_gqa_regression(layout):
  case = random_ragged_case(lengths=(237,), prefixes=(2048,), heads=32,
                            kv_heads=8, capacity=3072, padding=19)
  expected = ragged_reference(*case, causal=True)
  actual = run_paged(*case, layout=layout, causal=True)
  assert_extend_matches(actual, expected, case[5], case[6], 1)


@pytest.mark.skipif(INTERPRET,
                    reason="requires TPU optimized buffer assignment")
@pytest.mark.parametrize("layout", ["pair", "merged"])
def test_ragged_cache_has_no_pool_sized_temporary(layout):
  q = jax.ShapeDtypeStruct((8, 16, 128), jnp.bfloat16)
  cache = jax.ShapeDtypeStruct((1024, 128, 4, 128), jnp.bfloat16)
  if layout == "merged":
    caches = (jax.ShapeDtypeStruct(
        (*cache.shape[:2], 2 * cache.shape[2], cache.shape[3]), cache.dtype),
              None)
  else:
    caches = (cache, cache)

  def paged_attn(q, k_cache, v_cache, cu, seqused, table):
    return flash_attn_varlen_func(
        q, k_cache, v_cache, cu, None, 16, 256 * 128, causal=True,
        block_table=table, seqused_k=seqused)

  compiled = jax.jit(paged_attn).lower(
      q, *caches, jax.ShapeDtypeStruct((3,), jnp.int32),
      jax.ShapeDtypeStruct((2,), jnp.int32),
      jax.ShapeDtypeStruct((2, 256), jnp.int32)).compile()
  memory = compiled.memory_analysis()
  pool_bytes = 2 * math.prod(cache.shape) * jnp.dtype(cache.dtype).itemsize
  # Note (david): the read-only kernel aliases nothing. A cache layout
  # conversion or a dense gather of the cache would allocate at least half the
  # pool; query transforms and scratch are far smaller.
  assert memory.temp_size_in_bytes < pool_bytes // 4, memory


REJECTED_PAGED_CALLS = {
    # Note (david): case -> (exception, message pattern).
    "seqused_k_without_block_table": (ValueError, "seqused_k"),
    "num_active_without_block_table": (ValueError, "num_active"),
    "block_table_without_seqused_k": (ValueError, "seqused_k"),
    "cu_seqlens_k_with_block_table": (ValueError, "cu_seqlens_k"),
    "token_major": ((ValueError, NotImplementedError), "(?i)token.major"),
    "rotary_k": ((ValueError, NotImplementedError), "rotary_k"),
    "window_size": ((ValueError, NotImplementedError), "window"),
    "softcap": ((ValueError, NotImplementedError), "softcap"),
    "float32": ((ValueError, NotImplementedError), "(?i)bf16|bfloat16"),
    "head_dim_64": ((ValueError, NotImplementedError), "head_dim"),
    "num_active_past_batch": (ValueError, "num_active"),
    "num_active_numpy_past_batch": (ValueError, "num_active"),
    "num_active_jax_past_batch": (ValueError, "num_active"),
    "num_active_jax_negative": (ValueError, "num_active"),
    "num_active_float": (ValueError, "num_active"),
}


@pytest.mark.parametrize("case", list(REJECTED_PAGED_CALLS))
def test_paged_varlen_rejects_unsupported_calls(case):
  # Note (david): each case changes one knob of an otherwise valid paged
  # call: 2 requests of 3 and 5 rows over 5 and 9 cached tokens.
  error, message = REJECTED_PAGED_CALLS[case]
  heads, kv_heads, total = 4, 2, 8
  head_dim = 64 if case == "head_dim_64" else 128
  dtype = jnp.float32 if case == "float32" else jnp.bfloat16
  q = jnp.zeros((heads, total, head_dim), dtype)
  cu = jnp.array([0, 3, total], jnp.int32)
  operands = [q, jnp.zeros((2, 128, 2 * kv_heads, head_dim), dtype), None,
              cu, None]
  kwargs = dict(causal=True, interpret=INTERPRET,
                block_table=jnp.array([[0], [1]], jnp.int32),
                seqused_k=jnp.array([5, 9], jnp.int32))
  if case.endswith("_without_block_table"):
    packed_kv = jnp.zeros((kv_heads, total, head_dim), dtype)
    operands[1:] = [packed_kv, packed_kv, cu, cu]
    kwargs["block_table"] = None
    if case.startswith("num_active"):
      kwargs.update(seqused_k=None, num_active=jnp.int32(1))
  elif case == "block_table_without_seqused_k":
    kwargs["seqused_k"] = None
  elif case == "cu_seqlens_k_with_block_table":
    operands[4] = cu
  elif case == "token_major":
    operands[0] = q.transpose(1, 0, 2).reshape(total, heads * head_dim)
    kwargs.update(token_major=True, head_dim=head_dim)
  elif case == "rotary_k":
    cos, sin = rope_tables(128, head_dim)
    kwargs.update(rotary_cos=cos, rotary_sin=sin, rotary_k=True)
  elif case == "window_size":
    kwargs["window_size"] = (16, 0)
  elif case == "softcap":
    kwargs["softcap"] = 5.0
  elif case.startswith("num_active_"):
    # Note (david): a concrete NumPy or jax scalar is range-checked like an
    # int; batch is 2.
    kwargs["num_active"] = {
        "num_active_past_batch": 3,
        "num_active_numpy_past_batch": np.int32(3),
        "num_active_jax_past_batch": jnp.int32(3),
        "num_active_jax_negative": jnp.int32(-1),
        "num_active_float": jnp.float32(1.0),
    }[case]
  with pytest.raises(error, match=message):
    flash_attn_varlen_func(*operands, total, 128, **kwargs)


def paged_tiles(max_seqlen_q_bucket, max_seqlen_k_bucket, *, heads=32,
                kv_heads=8, head_dim=128, page_size=128, pages_per_seq=320,
                is_merged=True, return_lse=False, **pins):
  """resolve_paged_tiles for nheads/nheads_k heads, no rotary."""
  return resolve_paged_tiles(
      max_seqlen_q_bucket=max_seqlen_q_bucket,
      max_seqlen_k_bucket=max_seqlen_k_bucket, page_size=page_size,
      pages_per_seq=pages_per_seq, num_kv_heads=kv_heads,
      q_heads_per_kv_head=heads // kv_heads, head_dim=head_dim,
      is_merged=is_merged, return_lse=return_lse, rotary_dtype=None, **pins)


def tpu_vmem_bytes(blocks, group, *, heads=32, kv_heads=8, head_dim=128,
                   page_size=128, is_merged=True, return_lse=False):
  """estimate_vmem_bytes of the TPU (pair-packed) staging, no rotary."""
  paged = paged_kv_info(
      num_kv_heads=kv_heads, kv_heads_per_group=group, page_size=page_size,
      pages_per_seq=320, is_merged=is_merged, interpret=False)
  return estimate_vmem_bytes(
      blocks, paged, q_heads_per_kv_head=heads // kv_heads, head_dim=head_dim,
      return_lse=return_lse, rotary_dtype=None)


@pytest.mark.parametrize("shape,dtype,expected_bytes", [
    ((2, 512, 3, 128), jnp.bfloat16, 2 * 512 * 4 * 128 * 2),
    ((2, 512, 12, 128), jnp.bfloat16, 2 * 512 * 16 * 128 * 2),
    ((2, 512, 8, 2, 128), jnp.bfloat16, 2 * 512 * 8 * 2 * 128 * 2),
    ((2, 1, 512, 256), jnp.bfloat16, 2 * 512 * 256 * 2),
    ((2, 1, 2, 128, 64), jnp.float32, 2 * 2 * 128 * 128 * 4),
    ((1, 2, 512), jnp.int32, 2 * 512 * 4),
    ((4096, 128), jnp.float32, 4096 * 128 * 4),
])
def test_vmem_buffer_bytes_pads_to_mosaic_tiles(shape, dtype, expected_bytes):
  # Note (david): an odd staged head axis pads to a power of two (12 to 16),
  # a bf16 head pair costs nothing, and a half-lane rotary plane pads to 128.
  assert vmem_buffer_bytes(shape, jnp.dtype(dtype)) == expected_bytes


class CapturedPallasCall(Exception):
  """Carries the scratch_shapes of an intercepted pallas_call."""


@pytest.mark.parametrize(
    "heads,kv_heads,head_dim,layout,group,interpret,return_lse,rotary", [
        (32, 8, 128, "merged", 4, False, False, False),
        (32, 8, 128, "merged", 8, True, True, True),
        (6, 3, 128, "pair", 3, False, True, False),
        (3, 3, 128, "pair", 1, False, False, True),
        (8, 1, 256, "pair", 1, False, True, True),
        (8, 2, 128, "merged", 2, False, False, False),
    ])
def test_paged_vmem_estimate_tracks_forward_common_scratch(
    monkeypatch, heads, kv_heads, head_dim, layout, group, interpret,
    return_lse, rotary):
  # Note (david): forward_common's pallas_call is intercepted before any
  # lowering, so the TPU build (interpret=False: pair-packed u32 staging for
  # an even staged head axis) is inspected on CPU as well.
  captured = []

  def _capture_pallas_call(*unused_args, scratch_shapes, **unused_kwargs):
    captured.extend(scratch_shapes)
    raise CapturedPallasCall

  monkeypatch.setattr(fwd_pipeline.pl, "pallas_call", _capture_pallas_call)
  page_size, pages_per_seq, total_q = 128, 4, 200
  cache_heads = 2 * kv_heads if layout == "merged" else kv_heads
  cache = jnp.zeros((8, page_size, cache_heads, head_dim), jnp.bfloat16)
  blocks = BlockSizes(block_q=256, block_kv=256, block_kv_compute=128,
                      block_q_compute=128)
  rotary_dtype = jnp.dtype(jnp.float32) if rotary else None
  if rotary:
    q_coeff = jnp.zeros((2, 1, total_q, head_dim // 2), rotary_dtype)
    rotary_pair = (q_coeff, None)
  else:
    rotary_pair = None
  with pytest.raises(CapturedPallasCall):
    flash_attn_varlen_paged(
        jnp.zeros((heads, total_q, head_dim), jnp.bfloat16), cache,
        None if layout == "merged" else cache,
        jnp.array([0, 90, 180], jnp.int32), jnp.array([300, 500], jnp.int32),
        jnp.zeros((2, pages_per_seq), jnp.int32), max_seqlen_q=90,
        max_seqlen_k=500, causal=True, q_scale=1.0, return_lse=return_lse,
        interpret=interpret, block_sizes=blocks, kv_heads_per_group=group,
        rotary=rotary_pair)
  allocated = [(tuple(ref.shape), jnp.dtype(ref.dtype)) for ref in captured
               if ref.memory_space == pltpu.VMEM]
  paged = paged_kv_info(
      num_kv_heads=kv_heads, kv_heads_per_group=group, page_size=page_size,
      pages_per_seq=pages_per_seq, is_merged=layout == "merged",
      interpret=interpret)
  assert allocated == paged_scratch_shapes(
      blocks, paged, q_heads_per_kv_head=heads // kv_heads,
      head_dim=head_dim, return_lse=return_lse, rotary_dtype=rotary_dtype)


def assert_fewest_prefix_streams(blocks, group, max_seqlen_q_bucket, *,
                                 heads=32, kv_heads=8, head_dim=128,
                                 is_merged=True):
  """The tiles fit the VMEM budget, and every q block and KV head group
  that streams the longest prefix fewer times does not."""
  layout = dict(heads=heads, kv_heads=kv_heads, head_dim=head_dim,
                is_merged=is_merged)
  assert tpu_vmem_bytes(blocks, group, **layout) <= VMEM_LIMIT_BYTES

  def _streams(block_q, kv_group):
    return -(-max_seqlen_q_bucket // block_q) * (kv_heads // kv_group)

  for block_q in FWD_BLOCKS:
    for kv_group in range(1, kv_heads + 1):
      if (kv_heads % kv_group
          or block_q > max(max_seqlen_q_bucket, FWD_BLOCKS[-1])
          or _streams(block_q, kv_group) >= _streams(blocks.block_q, group)):
        continue
      fewer_streams = dataclasses.replace(
          blocks, block_q=block_q, block_q_compute=min(256, block_q // 2))
      assert tpu_vmem_bytes(fewer_streams, kv_group, **layout) > (
          VMEM_LIMIT_BYTES), (block_q, kv_group)


@pytest.mark.parametrize(
    "max_seqlen_q_bucket,max_seqlen_k_bucket,expected", [
        (128, 4096, (128, 2048, 8)),
        (128, 40960, (128, 2048, 8)),
        (2048, 4096, (512, 2048, 8)),
        (2048, 40960, (512, 2048, 8)),
        (8192, 4096, (512, 2048, 8)),
        (8192, 40960, (512, 2048, 8)),
        (64, 128, (128, 128, 8)),
        (384, 640, (256, 640, 8)),
    ])
@pytest.mark.parametrize("is_merged", [True, False])
def test_paged_tiles_minimize_prefix_streams_within_vmem(
    max_seqlen_q_bucket, max_seqlen_k_bucket, expected, is_merged):
  # Note (david): Qwen3-4B's 32 q / 8 KV heads at d128 on 128-token pages.
  # Past a 512-row q block, (512, all 8 KV heads), (1024, 4) and (2048, 2)
  # stream the prefix equally often and the next halving does not fit, so
  # the tie takes the widest group, whose q block pads short sequences least.
  # A 640-token bucket (a five-page table row) is one five-page block, whose
  # kv compute tile is 128, the largest one dividing it.
  blocks, group = paged_tiles(max_seqlen_q_bucket, max_seqlen_k_bucket,
                              is_merged=is_merged)
  assert (blocks.block_q, blocks.block_kv, group) == expected
  assert blocks.block_q_compute == min(256, blocks.block_q // 2)
  if blocks.block_kv == 640:
    assert blocks.block_kv_compute == 128
  else:
    assert blocks.block_kv_compute == min(512, blocks.block_kv)
  assert_fewest_prefix_streams(blocks, group, max_seqlen_q_bucket,
                               is_merged=is_merged)


@pytest.mark.parametrize("heads,max_seqlen_k_bucket,expected", [
    (64, 40960, (128, 2048, 8)), (128, 256, (128, 256, 4))])
def test_paged_tiles_trade_the_q_block_then_the_fold_for_vmem(
    heads, max_seqlen_k_bucket, expected):
  # Note (david): at d256 on 8 KV heads, 64 q heads keep all 8 KV heads in
  # one group on a 128-row q block, the widest of the tied (128, 8),
  # (256, 4), (512, 2) and (1024, 1). 128 q heads overflow even then, so
  # their group halves.
  blocks, group = paged_tiles(2048, max_seqlen_k_bucket, heads=heads,
                              head_dim=256)
  assert (blocks.block_q, blocks.block_kv, group) == expected
  assert_fewest_prefix_streams(blocks, group, 2048, heads=heads,
                               head_dim=256)


def test_paged_tiles_pins_and_page_multiples():
  # Note (david): a pinned axis is searched no further; block_kv holds the
  # most whole pages, at least one, within the kv bucket and 2048 tokens, for
  # any multiple-of-128 page size.
  pinned = BlockSizes(block_q=1024, block_kv=512, block_kv_compute=256,
                      block_q_compute=256)
  assert paged_tiles(8192, 40960, block_sizes=pinned) == (pinned, 4)
  blocks, group = paged_tiles(8192, 40960, kv_heads_per_group=2)
  assert (blocks.block_q, group) == (2048, 2)
  for page_size, kv_bucket, block_kv, block_kv_compute in [
      (256, 128, 256, 256), (512, 1536, 1536, 512), (2048, 4096, 2048, 512),
      (384, 128, 384, 384), (384, 4096, 1920, 384), (640, 4096, 1920, 384),
      (4096, 8192, 4096, 512)]:
    blocks, _ = paged_tiles(128, kv_bucket, page_size=page_size)
    assert (blocks.block_kv, blocks.block_kv_compute) == (
        block_kv, block_kv_compute)


def test_paged_tiles_reject_vmem_overflow():
  # Note (david): a head group holds every q head of its KV heads, so 256 q
  # heads on one d256 KV head need ~193 MiB at the smallest tiles; a pinned
  # build past the budget raises too, and nothing shrinks behind the pin.
  with pytest.raises(ValueError, match="VMEM"):
    flash_attn_varlen_func(
        jnp.zeros((256, 8, 256), jnp.bfloat16),
        jnp.zeros((2, 128, 2, 256), jnp.bfloat16), None,
        jnp.array([0, 8], jnp.int32), None, 8, 128, causal=True,
        interpret=INTERPRET, block_table=jnp.array([[0]], jnp.int32),
        seqused_k=jnp.array([8], jnp.int32))
  with pytest.raises(ValueError, match="VMEM"):
    paged_tiles(8192, 40960, kv_heads_per_group=8, block_sizes=BlockSizes(
        block_q=2048, block_kv=2048, block_kv_compute=512,
        block_q_compute=256))
  with pytest.raises(ValueError, match="VMEM"):
    paged_tiles(2048, 40960, heads=64, head_dim=256, kv_heads_per_group=8,
                block_sizes=BlockSizes(block_q=512, block_kv=2048,
                                       block_kv_compute=512,
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
  """Needle-probe extend inputs in random_ragged_case's tuple layout.

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


@pytest.mark.parametrize("leading_rows,causal", [(13, True), (16, False)])
def test_paged_rows_before_cu_seqlens_q_start_are_not_owned(leading_rows,
                                                            causal):
  # Note (david): serving starts cu_seqlens_q past the decode rows. The kernel
  # owns no row below cu_seqlens_q[0]; its first block, aligned down to 8
  # rows, may overwrite [align8(start), start), which the caller restores.
  # NaN in the leading rows must not reach any owned row.
  lengths = (14, 0, 21, 7, 1)
  q, kc, vc, k, v, cu, cache_seqlens, table = probe_case(
      lengths, (5, 64, 130, 250, 127), padding=9, causal=causal)
  q, k, v = (
      jnp.concatenate((jnp.full((leading_rows, *tokens.shape[1:]), jnp.nan,
                                tokens.dtype), tokens))
      for tokens in (q, k, v))
  case = (q, kc, vc, k, v, cu + leading_rows, cache_seqlens, table)
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
  # Note (david): the tpu-inference FlyWheel backend's extend layout for Qwen3-4B:
  # 64 padded slots with 320-page table rows, 3 active requests in a 512-token
  # bucket, called without lse when serving. The 16/4 and 8/2 cases are one
  # TP 2 / TP 4 shard's heads, where few KV heads share one merged row. The
  # bounds are the backend's: the token bucket, and the table row's capacity
  # for max_model_len.
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


def check_extend_kernel(case, *, causal, tilings, layout="pair"):
  """The extend kernel on caches the reference already appended to.

  layout "pair" passes separate K/V pools and "merged" one
  (pages, page_size, 2 * kv_heads, dim) pool.
  """
  q, kc, vc, k, v, cu, lengths, table = case
  expected = ragged_reference(q, kc, vc, k, v, cu, lengths, table,
                              causal=causal)
  k_pages = jnp.asarray(expected[2], jnp.bfloat16)
  v_pages = jnp.asarray(expected[3], jnp.bfloat16)
  if layout == "pair":
    cache_operands = (k_pages, v_pages)
  else:
    cache_operands = (jnp.concatenate((k_pages, v_pages), axis=2), None)
  out, lse = flash_attn_kvcache_extend_pallas(
      q, *cache_operands, cu, (lengths + jnp.diff(cu)).astype(jnp.int32),
      table.reshape(-1), jnp.int32(len(lengths)),
      q_scale=float(np.log2(np.e)) / np.sqrt(q.shape[-1]), causal=causal,
      return_lse=True, interpret=INTERPRET, tilings=tilings,
      merged_cache=layout == "merged")
  np.testing.assert_allclose(np.asarray(out, np.float32), expected[0],
                             **OUT_TOL)
  np.testing.assert_allclose(lse, expected[1], **LSE_TOL)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("tilings", [
    ((8, 128, 128, 1),),
    ((16, 256, 128, 1),),
    ((24, 512, 256, 1),),
    ((32, 256, 256, 1), (16, 512, 256, 1), (8, 128, 128, 1)),
    pytest.param(((32, 256, 128, 1), (8, 512, 256, 2)), marks=TPU_ONLY),
])
def test_extend_kernel_small_chunks_and_boundary_blends(causal, tilings):
  # Note (david): chunks far smaller than the requests force several KV passes
  # per request, 1-row requests share one 8-row output tile three times in a
  # row, and mixed tilings prefetch across staging buffers of different sizes.
  case = random_ragged_case(lengths=(13, 1, 1, 1, 0, 29, 3),
                            prefixes=(200, 0, 7, 128, 5, 300, 1),
                            capacity=512, padding=11)
  check_extend_kernel(case, causal=causal, tilings=tilings)


@pytest.mark.parametrize("tilings", [
    ((16, 256, 128, 1),),
    ((32, 256, 256, 1), (8, 1024, 512, 1)),
    pytest.param(((16, 256, 128, 2),), marks=TPU_ONLY),
    pytest.param(((32, 256, 256, 1), (8, 1024, 512, 2)), marks=TPU_ONLY),
])
def test_extend_kernel_gqa_multi_block_prefix(tilings):
  case = random_ragged_case(lengths=(40, 21, 5), prefixes=(700, 1000, 998),
                            heads=16, kv_heads=8, capacity=1024, padding=5)
  check_extend_kernel(case, causal=True, tilings=tilings)


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_extend_kernel_merged_cache_head_counts(kv_heads):
  # Note (david): one KV head makes the merged row a single K/V head pair; two
  # and four heads are head blocks inside one (8, 128) HBM tile of the row.
  case = random_ragged_case(lengths=(40, 1, 21), prefixes=(300, 129, 0),
                            heads=8, kv_heads=kv_heads, capacity=512,
                            padding=3)
  check_extend_kernel(case, causal=True, tilings=None, layout="merged")
