"""Packed multi-token extend through flash_attn_with_kvcache and its kernel."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flywheel_tpu import flash_attn_with_kvcache
from flywheel_tpu.pallas.flash_fwd_kvcache_extend import (
    flash_attn_kvcache_extend_pallas,
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


def extend_in_place(q, kc, vc, k, v, *, merged, **kwargs):
  """flash_attn_with_kvcache donating kc and vc (as one merged pool when
  merged); returns (*result, updated_k, updated_v) after checking the append
  landed in the donated buffers."""
  if merged:
    cache = jnp.concatenate((kc, vc), axis=2)
    pointers = (cache.unsafe_buffer_pointer(),)
    *result, updated = flash_attn_with_kvcache(q, cache, None, k, v, **kwargs)
    updated_pointers = (updated.unsafe_buffer_pointer(),)
    updated_k, updated_v = jnp.split(updated, 2, axis=2)
  else:
    pointers = (kc.unsafe_buffer_pointer(), vc.unsafe_buffer_pointer())
    *result, updated_k, updated_v = flash_attn_with_kvcache(q, kc, vc, k, v,
                                                            **kwargs)
    updated_pointers = (updated_k.unsafe_buffer_pointer(),
                        updated_v.unsafe_buffer_pointer())
  # Note (david): donation must append in place instead of copying the pool;
  # interpret mode does not honor aliasing.
  assert INTERPRET or updated_pointers == pointers
  return (*result, updated_k, updated_v)


def assert_extend_matches(actual, expected, cu, lengths, num_active):
  """Real rows match the reference, packed padding rows are out = 0 and
  lse = -inf, and the appended caches match exactly. lse is None when the
  call did not return it."""
  out, lse, updated_k, updated_v = actual
  cu, lengths = np.asarray(cu), np.asarray(lengths)
  real = int(cu[num_active])
  out = np.asarray(out, np.float32)
  if lse is None:
    lse_mismatched = np.zeros(real, bool)
    padding_lse = np.full(0, -np.inf)
  else:
    lse = np.asarray(lse)
    lse_mismatched = ~np.isclose(
        lse[:, :real], expected[1][:, :real], **LSE_TOL).all(axis=0)
    padding_lse = lse[:, real:]
  mismatched = ~np.isclose(
      out[:real], expected[0][:real], **OUT_TOL).all(axis=(1, 2))
  rows = np.flatnonzero(mismatched | lse_mismatched)
  seqs = np.searchsorted(cu, rows, side="right") - 1
  first_mismatches = [(int(row), int(seq), int(row - cu[seq]),
                       int(lengths[seq] + row - cu[seq]))
                      for row, seq in zip(rows[:8], seqs[:8])]
  assert not rows.size, (
      f"{rows.size}/{real} real rows differ; first (row, request, offset,"
      f" position): {first_mismatches}")
  np.testing.assert_array_equal(out[real:], 0, err_msg="packed padding out")
  np.testing.assert_array_equal(padding_lse, -np.inf,
                                err_msg="packed padding lse")
  np.testing.assert_array_equal(np.asarray(updated_k, np.float32), expected[2])
  np.testing.assert_array_equal(np.asarray(updated_v, np.float32), expected[3])


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("append", [False, True])
@pytest.mark.parametrize("layout", ["pair", "merged"])
def test_ragged_cache_matches_reference(causal, append, layout):
  case = random_ragged_case()
  q, kc, vc, k, v, cu, lengths, table = (
      case if layout == "merged" else contiguous_case(*case))
  k, v = (k, v) if append else (None, None)
  expected = ragged_reference(q, kc, vc, k, v, cu, lengths, table,
                              causal=causal)
  kwargs = dict(cache_seqlens=lengths, cu_seqlens_q=cu, causal=causal,
                return_softmax_lse=True, interpret=INTERPRET)
  if layout == "merged":
    kwargs["block_table"] = table
  actual = extend_in_place(q, kc, vc, k, v, merged=layout == "merged",
                           **kwargs)
  assert_extend_matches(actual, expected, cu, lengths, len(lengths))


@pytest.mark.parametrize("head_dim,heads,kv_heads",
                         [(64, 4, 2), (128, 4, 1), (256, 4, 4)])
def test_ragged_cache_multiple_query_blocks_and_pages(head_dim, heads,
                                                      kv_heads):
  case = random_ragged_case(lengths=(131, 17), prefixes=(123, 255),
                            heads=heads, kv_heads=kv_heads, head_dim=head_dim,
                            padding=3)
  expected = ragged_reference(*case, causal=True)
  actual = run_extend(*case, causal=True)
  assert_extend_matches(actual, expected, case[5], case[6], 2)


@pytest.mark.parametrize("heads,kv_heads,dim", [(6, 6, 64), (16, 8, 128),
                                               (16, 16, 128), (24, 4, 256),
                                               (32, 16, 128)])
def test_ragged_cache_head_tiles(heads, kv_heads, dim):
  case = random_ragged_case(lengths=(3, 5), prefixes=(37, 128), heads=heads,
                            kv_heads=kv_heads, head_dim=dim, capacity=256,
                            padding=0)
  expected = ragged_reference(*case, causal=True)
  actual = run_extend(*case, causal=True)
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
  actual = run_extend(q, kc, vc, k, v, cu, lengths, table,
                      num_active=num_active, causal=True)
  assert_extend_matches(actual, expected, cu, lengths, num_active)


def test_ragged_read_only_short_cache_and_nan_padding():
  # Note (david): NaN in every slot past the two live tokens catches any read
  # outside the valid range; the empty cache has no valid page at all.
  q, kc, vc, _, _, cu, lengths, table = random_ragged_case(
      lengths=(5, 3), prefixes=(2, 0), padding=0)
  kc = jnp.full_like(kc, jnp.nan).at[table[0, 0], :2].set(1)
  vc = jnp.full_like(vc, jnp.nan).at[table[0, 0], :2].set(2)
  table = table.at[1].set(-1)
  expected = ragged_reference(q, kc, vc, None, None, cu, lengths, table,
                              causal=True)
  actual = run_extend(q, kc, vc, None, None, cu, lengths, table,
                      causal=True)
  assert_extend_matches(actual, expected, cu, lengths, 2)


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
  actual = run_extend(q, kc, vc, k, v, cu, lengths, table,
                      return_lse=False, causal=True)
  assert_extend_matches(actual, expected, cu, lengths, 2)


def test_ragged_metadata_reuses_one_executable():
  q, kc, vc, k, v, cu, lengths, table = random_ragged_case(padding=0)
  kv_pages = jnp.concatenate((kc, vc), axis=2)
  compiled = flash_attn_with_kvcache.lower(
      q, kv_pages, None, k, v, cache_seqlens=lengths, cu_seqlens_q=cu,
      block_table=table, num_active=jnp.int32(3), causal=True,
      interpret=INTERPRET).compile()
  for boundaries, active, lens, pages in (
      (cu, 3, lengths, table),
      (jnp.array([0, 1, 6, 8], jnp.int32), 2, jnp.array([0, 127, -100]),
       table[::-1]),
  ):
    kc, vc = jnp.split(kv_pages, 2, axis=2)
    expected = ragged_reference(q, kc, vc, k, v, boundaries, lens, pages,
                                causal=True, num_active=active)
    out, kv_pages = compiled(q, kv_pages, None, k, v, cache_seqlens=lens,
                             cu_seqlens_q=boundaries, block_table=pages,
                             num_active=jnp.int32(active))
    assert_extend_matches((out, None, *jnp.split(kv_pages, 2, axis=2)),
                          expected, boundaries, lens, active)


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_multitoken_contiguous_cache_and_mapping(dense, head_dim):
  q, kc, vc, k, v, cu, lengths, table = random_ragged_case(
      lengths=(3, 3), prefixes=(127, 509), head_dim=head_dim, padding=0)
  kc = kc[table].reshape(2, 512, 2, head_dim)
  vc = vc[table].reshape(2, 512, 2, head_dim)
  mapping = jnp.array([1, 0], jnp.int32)
  # Note (david): a contiguous cache is one 512-token page per row.
  expected = ragged_reference(q, kc, vc, k, v, cu, lengths, mapping[:, None],
                              causal=True)
  if dense:
    q, k, v = (tokens.reshape(2, 3, *tokens.shape[1:]) for tokens in (q, k, v))
    expected_lse = expected[1].reshape(4, 2, 3).transpose(1, 0, 2)
  else:
    expected_lse = expected[1]
  out, lse, updated_k, updated_v = flash_attn_with_kvcache(
      q, kc, vc, k, v, cache_seqlens=lengths, cache_batch_idx=mapping,
      cu_seqlens_q=None if dense else cu, causal=True,
      return_softmax_lse=True, interpret=INTERPRET)
  np.testing.assert_allclose(
      np.asarray(out, np.float32).reshape(-1, 4, head_dim), expected[0],
      **OUT_TOL)
  np.testing.assert_allclose(lse, expected_lse, **LSE_TOL)
  np.testing.assert_array_equal(np.asarray(updated_k, np.float32), expected[2])
  np.testing.assert_array_equal(np.asarray(updated_v, np.float32), expected[3])


@pytest.mark.parametrize(
    "dense,paged,heads,kv_heads,causal,append,return_lse,num_active", [
        (False, True, 8, 1, True, True, True, None),
        (False, True, 8, 2, False, True, True, 2),
        (False, False, 16, 4, True, True, False, 2),
        (False, True, 32, 8, True, False, True, None),
        (True, False, 8, 1, False, True, True, None),
        (True, True, 16, 2, True, True, True, 2),
        (True, False, 32, 8, True, True, False, None),
        (True, True, 16, 4, False, False, True, None),
    ])
def test_multitoken_kept_layouts_match_reference(
    dense, paged, heads, kv_heads, causal, append, return_lse, num_active):
  # Note (david): the inactive request carries garbage lengths and page rows
  # (or cache rows) that must never be read. No row sees a single key, whose
  # lse carries Q's full bf16 rounding (~LSE_TOL).
  q, kc, vc, k, v, cu, lengths, table = random_ragged_case(
      lengths=(6, 6, 6) if dense else (13, 1, 21), prefixes=(127, 3, 251),
      heads=heads, kv_heads=kv_heads, padding=0 if dense else 7)
  batch = len(lengths)
  active = batch if num_active is None else num_active
  lengths = lengths.at[active:].set(-100)
  if paged:
    table = table.at[active:].set(-1)
    reference_table = table
    metadata = dict(block_table=table)
  else:
    # Note (david): a contiguous cache is one 512-token page per row, mapped
    # to its query row through cache_batch_idx.
    kc, vc = (pages[table].reshape(batch, 512, kv_heads, 128)
              for pages in (kc, vc))
    mapping = jnp.array([2, 0, 1], jnp.int32).at[active:].set(-1)
    reference_table = mapping[:, None]
    metadata = dict(cache_batch_idx=mapping)
  k, v = (k, v) if append else (None, None)
  expected = ragged_reference(q, kc, vc, k, v, cu, lengths, reference_table,
                              causal=causal, num_active=active)
  kwargs = dict(cache_seqlens=lengths, causal=causal,
                num_active=None if num_active is None else jnp.int32(active),
                return_softmax_lse=return_lse, interpret=INTERPRET, **metadata)
  if dense:
    q, k, v = (None if tokens is None
               else tokens.reshape(batch, 6, *tokens.shape[1:])
               for tokens in (q, k, v))
  else:
    kwargs["cu_seqlens_q"] = cu

  *result, updated_k, updated_v = extend_in_place(q, kc, vc, k, v,
                                                  merged=paged, **kwargs)
  # Note (david): a dense call returns out as (batch, seqlen_q, heads,
  # head_dim) and lse as (batch, heads, seqlen_q); the reference is packed.
  out = np.asarray(result[0], np.float32).reshape(-1, heads, 128)
  if not return_lse:
    lse = None
  elif dense:
    lse = np.asarray(result[1]).transpose(1, 0, 2).reshape(heads, -1)
  else:
    lse = np.asarray(result[1])
  assert_extend_matches((out, lse, updated_k, updated_v), expected, cu,
                        lengths, active)


@pytest.mark.parametrize("page_size", [256, 512, 1024])
def test_ragged_cache_page_sizes(page_size):
  case = random_ragged_case(lengths=(5, 11), prefixes=(page_size - 2, 3),
                            page_size=page_size, capacity=2 * page_size,
                            heads=2, kv_heads=1, head_dim=64)
  expected = ragged_reference(*case, causal=True)
  actual = run_extend(*case, causal=True)
  assert_extend_matches(actual, expected, case[5], case[6], 2)


@pytest.mark.parametrize("append", [False, True])
def test_empty_packed_allocation(append):
  q, kc, vc, k, v, cu, lengths, table = random_ragged_case(
      lengths=(0, 0), prefixes=(0, 0), padding=0)
  k, v = (k, v) if append else (None, None)
  expected = ragged_reference(q, kc, vc, k, v, cu, lengths, table,
                              causal=True)
  actual = run_extend(q, kc, vc, k, v, cu, lengths, table, causal=True)
  assert_extend_matches(actual, expected, cu, lengths, 2)


@pytest.mark.parametrize("heads,kv_heads", [(132, 6), (256, 1), (128, 8)])
def test_ragged_cache_large_head_fold(heads, kv_heads):
  case = random_ragged_case(lengths=(67, 11), prefixes=(3, 128), heads=heads,
                            kv_heads=kv_heads, head_dim=256, capacity=256,
                            padding=0)
  expected = ragged_reference(*case, causal=True)
  actual = run_extend(*case, causal=True)
  assert_extend_matches(actual, expected, case[5], case[6], 2)


@pytest.mark.parametrize("layout", ["pair", "merged"])
def test_ragged_cache_prefill_gqa_regression(layout):
  case = random_ragged_case(lengths=(237,), prefixes=(2048,), heads=32,
                            kv_heads=8, capacity=3072, padding=19)
  if layout == "pair":
    case = contiguous_case(*case)
  expected = ragged_reference(*case, causal=True)
  actual = run_extend(*case, layout=layout, causal=True)
  assert_extend_matches(actual, expected, case[5], case[6], 1)


@pytest.mark.parametrize("layout", ["pair", "merged"])
def test_extend_replays_when_later_block_raises_anchor(layout):
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
  if layout == "pair":
    case = contiguous_case(*case)
  expected = ragged_reference(*case, causal=True)
  actual = run_extend(*case, layout=layout, return_lse=False, causal=True)
  assert_extend_matches(actual, expected, case[5], case[6], 1)


@pytest.mark.skipif(INTERPRET,
                    reason="requires TPU optimized buffer assignment")
@pytest.mark.parametrize("layout", ["pair", "merged"])
def test_ragged_cache_has_no_pool_sized_temporary(layout):
  q = jax.ShapeDtypeStruct((16, 8, 128), jnp.bfloat16)
  kv = jax.ShapeDtypeStruct((16, 4, 128), jnp.bfloat16)
  kwargs = dict(cache_seqlens=jax.ShapeDtypeStruct((2,), jnp.int32),
                cu_seqlens_q=jax.ShapeDtypeStruct((3,), jnp.int32),
                causal=True)
  if layout == "merged":
    cache = jax.ShapeDtypeStruct((1024, 128, 4, 128), jnp.bfloat16)
    pool = jax.ShapeDtypeStruct(
        (*cache.shape[:2], 2 * cache.shape[2], cache.shape[3]), cache.dtype)
    compiled = flash_attn_with_kvcache.lower(
        q, pool, None, kv, kv,
        block_table=jax.ShapeDtypeStruct((2, 256), jnp.int32),
        **kwargs).compile()
  else:
    # Note (david): the same bytes per cache as the paged pool, as 16 rows.
    cache = jax.ShapeDtypeStruct((16, 8192, 4, 128), jnp.bfloat16)
    compiled = flash_attn_with_kvcache.lower(
        q, cache, cache, kv, kv, **kwargs).compile()
  memory = compiled.memory_analysis()
  pool_bytes = 2 * math.prod(cache.shape) * jnp.dtype(cache.dtype).itemsize
  assert memory.alias_size_in_bytes >= pool_bytes
  # Note (david): a cache layout conversion or a dense gather of the cache
  # would allocate at least half the pool; query transforms and scratch are
  # far smaller.
  assert memory.temp_size_in_bytes < pool_bytes // 4, memory


# Note (david): with N(0, 1) inputs attention spreads over the whole visible
# range, so an off-by-one causal limit or one misplaced page moves out by ~1/N,
# far below the bf16 tolerances once a prefix holds thousands of tokens. Here
# every key is a unit code and each query head aims at one or two chosen key
# positions with a 12-16 nat logit, so a visibility, paging or row-placement
# error moves out/lse by O(1). Stale cache slots hold copies of live codes, so
# attending one is loud too.
def probe_case(lengths, prefixes, *, heads=4, kv_heads=2, head_dim=64,
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
def test_extend_unaligned_sequence_boundaries(padding, causal):
  # Note (david): cu = 0 1 15 15 36 43 46 47 58 61 puts boundaries at every
  # residue mod 8, with an empty request in the middle, one-row requests, and
  # the last request ending at the buffer end or inside the pad tail.
  lengths = (1, 14, 0, 21, 7, 3, 1, 11, 3)
  case = probe_case(lengths, (0, 5, 64, 130, 250, 381, 127, 128, 255),
                    pages_per_seq=4, padding=padding, causal=causal)
  expected = ragged_reference(*case, causal=causal)
  assert_probe_is_sharp(expected[4])
  actual = run_extend(*case, causal=causal)
  assert_extend_matches(actual, expected, case[5], case[6], len(lengths))


@pytest.mark.parametrize("lengths,prefixes,page_size,causal", [
    ((1300,), (4000,), 256, True),
    ((700, 300), (1500, 2900), 128, True),
    ((300, 700), (2900, 1500), 128, False),
])
def test_extend_chunked_prefill_long_prefix(lengths, prefixes, page_size,
                                            causal):
  # Note (david): a q chunk longer than any kernel q chunk splits one request
  # across several kernel chunks, and its causal limit must advance row by
  # row across every chunk edge.
  case = probe_case(lengths, prefixes, head_dim=128, page_size=page_size,
                    padding=13, causal=causal, logit=16.0)
  expected = ragged_reference(*case, causal=causal)
  assert_probe_is_sharp(expected[4])
  actual = run_extend(*case, causal=causal)
  assert_extend_matches(actual, expected, case[5], case[6], len(lengths))


@pytest.mark.parametrize("page_size", [384, 640, 1024, 2048])
def test_extend_kv_tiles_end_inside_pages(page_size):
  # Note (david): power-of-two KV tiles end strictly inside these pages, and
  # the causal limits (2000 -> 2096, 1023 -> 1068, 500 -> 520) cross
  # 512/1024/2048 mid-chunk, so the limit falls inside a tile.
  lengths = (96, 45, 20)
  case = probe_case(lengths, (2000, 1023, 500), page_size=page_size,
                    padding=7, logit=16.0)
  expected = ragged_reference(*case, causal=True)
  assert_probe_is_sharp(expected[4])
  actual = run_extend(*case, causal=True)
  assert_extend_matches(actual, expected, case[5], case[6], len(lengths))


@pytest.mark.parametrize("tail_owned", [False, True])
def test_extend_mixed_prefixes_and_inactive_garbage(tail_owned):
  # Note (david): prefixes 0, 1 and both sides of a page edge sit next to a
  # long one, in a 9-slot batch whose 3 inactive slots carry garbage lengths
  # and table rows; active table entries past the live pages are -1.
  lengths = (9, 1, 17, 8, 35, 61)
  case = probe_case(lengths, (0, 1, 127, 128, 129, 1000), batch=9,
                    pages_per_seq=12, padding=21, tail_owned=tail_owned,
                    unused_pages=-1, logit=14.0)
  expected = ragged_reference(*case, causal=True, num_active=len(lengths))
  assert_probe_is_sharp(expected[4])
  actual = run_extend(*case, causal=True, num_active=len(lengths))
  assert_extend_matches(actual, expected, case[5], case[6], len(lengths))


@pytest.mark.parametrize("heads,kv_heads,return_lse", [
    (32, 8, False), (32, 8, True), (16, 4, False), (8, 2, True)])
def test_extend_serving_gqa_shape(heads, kv_heads, return_lse):
  # Note (david): the tpu-inference FlyWheel backend's extend layout for Qwen3-4B:
  # 64 padded slots with 320-page table rows, 3 active requests in a 512-token
  # bucket, called without lse when serving. The 16/4 and 8/2 cases are one
  # TP 2 / TP 4 shard's heads, where few KV heads share one merged row.
  lengths = (150, 131, 169)
  scale = 1 / math.sqrt(128)
  case = probe_case(lengths, (600, 641, 577), heads=heads, kv_heads=kv_heads,
                    head_dim=128, batch=64, pages_per_seq=320, padding=62,
                    logit=16.0, softmax_scale=scale)
  expected = ragged_reference(*case, causal=True, num_active=len(lengths),
                              softmax_scale=scale)
  assert_probe_is_sharp(expected[4])
  actual = run_extend(*case, causal=True, num_active=len(lengths),
                      return_lse=return_lse, softmax_scale=scale)
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
