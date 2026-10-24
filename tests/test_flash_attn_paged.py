"""Regression tests for packed multi-token attention over a paged KV cache vs a
NumPy reference, on paths the random grid of test_flash_attn_kvcache does not
reach.

flash_attn_varlen_func(block_table=...) reads a merged interleaved cache the test
filled on the host; flash_attn_with_kvcache's multi-token and packed calls append
the new tokens, then run that same kernel.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flywheel_tpu import flash_attn_varlen_func, flash_attn_with_kvcache
from tests.test_util import apply_rotary_emb

INTERPRET = jax.default_backend() != "tpu"
OUT_TOL = dict(rtol=2e-2, atol=3e-2)
LSE_TOL = dict(rtol=2e-3, atol=4e-3)


def bf16(array):
  return jnp.asarray(array, jnp.bfloat16)


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


def append_to_pages(kc, vc, k, v, cu, lengths, table):
  """f32 host copies of the K and V pages with each request's k/v rows written
  after its cache_seqlens; the pages an append writes are private to the
  request, so the order of the requests does not matter."""
  kc, vc = (np.asarray(pages, dtype=np.float32).copy() for pages in (kc, vc))
  if k is None:
    return kc, vc
  cu, lengths, table = map(np.asarray, (cu, lengths, table))
  page_size = kc.shape[1]
  for seq in range(len(lengths)):
    start, end = cu[seq:seq + 2]
    positions = lengths[seq] + np.arange(end - start)
    pages = table[seq, positions // page_size]
    kc[pages, positions % page_size] = np.asarray(k[start:end], np.float32)
    vc[pages, positions % page_size] = np.asarray(v[start:end], np.float32)
  return kc, vc


def ragged_reference(q, kc, vc, k, v, cu, lengths, table, *, causal,
                     window_left=None, softcap=0.0, sinks=None):
  """Appends k/v to the paged caches, then attends each request's rows.

  window_left hides the keys more than window_left positions before a row's
  bottom-right diagonal; softcap > 0 maps each scaled score s to
  softcap * tanh(s / softcap) before masking. sinks (heads,) adds exp(sinks[h])
  to head h's softmax denominator; a row that sees no key then has lse
  sinks[h].

  Returns (out, lse, k_pages, v_pages, top2), where top2 is the attention mass
  on the two heaviest keys of every attending (row, head).
  """
  kc, vc = append_to_pages(kc, vc, k, v, cu, lengths, table)
  q = np.asarray(q, dtype=np.float32)
  cu, lengths, table = map(np.asarray, (cu, lengths, table))
  heads, head_dim = q.shape[1:]
  page_size, kv_heads = kc.shape[1:3]
  repeats = heads // kv_heads
  scale = 1 / math.sqrt(head_dim)
  if sinks is not None:
    head_sinks = np.asarray(sinks, np.float32).reshape(kv_heads, repeats, 1)
  out = np.zeros_like(q)
  lse = np.full((heads, q.shape[0]), -np.inf, np.float32)
  top2 = [np.empty(0)]
  for seq in range(len(lengths)):
    start, end = cu[seq:seq + 2]
    query_length = end - start
    total = lengths[seq] + (0 if k is None else query_length)
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
      raw_scores = (queries @ keys)[..., 0, :] * scale
      if softcap:
        scores = softcap * np.tanh(raw_scores / softcap)
      else:
        scores = raw_scores
      diagonal = (total - query_length + rows)[:, None]
      if causal:
        right_visible = positions[None] <= diagonal
      else:
        right_visible = np.ones((len(rows), total), bool)
      if window_left is None:
        visible = right_visible
      else:
        visible = right_visible & (positions[None] >= diagonal - window_left)
      seen = visible.any(axis=-1)
      scores = np.where(visible[:, None, None], scores, -np.inf)
      anchor = np.where(seen[:, None, None, None],
                        scores.max(axis=-1, keepdims=True, initial=-np.inf), 0)
      if sinks is not None:
        anchor = np.where(seen[:, None, None, None],
                          np.maximum(anchor, head_sinks), head_sinks)
      weights = np.exp(scores - anchor)
      denom = weights.sum(axis=-1, keepdims=True)
      if sinks is not None:
        denom = denom + np.exp(head_sinks - anchor)
      probs = weights / np.where(denom > 0, denom, 1)
      result = (probs[..., None, :] @ values)[..., 0, :].reshape(
          len(rows), heads, head_dim)
      out[start + rows] = np.where(seen[:, None, None], result, 0)
      with np.errstate(divide="ignore"):
        row_lse = (anchor + np.log(denom)).reshape(len(rows), heads)
      if sinks is None:
        row_lse = np.where(seen[:, None], row_lse, -np.inf)
      lse[:, start + rows] = row_lse.T
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
               **kwargs):
  """flash_attn_with_kvcache's packed call on the merged pool; returns (out,
  lse or None, updated_k, updated_v), the updated pool split back."""
  *attention_outputs, kv_cache = flash_attn_with_kvcache(
      q, interleave(kc, vc), None, k, v, block_table=table,
      cache_seqlens=lengths, cu_seqlens_q=cu,
      return_softmax_lse=return_lse, interpret=INTERPRET, **kwargs)
  return (attention_outputs[0],
          attention_outputs[1] if return_lse else None,
          *deinterleave(kv_cache))


def run_paged(q, kc, vc, k, v, cu, lengths, table, *, return_lse=True,
              max_seqlen_q=None, max_seqlen_k=None, **kwargs):
  """Appends k/v (when given) on the host, then reads the merged pool through
  flash_attn_varlen_func; q/out stay token-major here. Returns (out, lse or
  None, k_pages, v_pages)."""
  kc, vc = (bf16(pages) for pages in append_to_pages(
      kc, vc, k, v, cu, lengths, table))
  seqused = lengths if k is None else lengths + jnp.diff(cu)
  max_seqlen_q = (
      max(1, int(np.diff(np.asarray(cu)).max(initial=0)))
      if max_seqlen_q is None else max_seqlen_q)
  max_seqlen_k = (
      max(1, int(np.asarray(seqused).max(initial=0)))
      if max_seqlen_k is None else max_seqlen_k)
  attention_outputs = flash_attn_varlen_func(
      q.transpose(1, 0, 2), interleave(kc, vc), None, cu, None, max_seqlen_q,
      max_seqlen_k, return_softmax_lse=return_lse, interpret=INTERPRET,
      block_table=table, seqused_k=seqused, **kwargs)
  out, lse = (attention_outputs if return_lse
              else (attention_outputs, None))
  return out.transpose(1, 0, 2), lse, kc, vc


def assert_extend_matches(actual, expected, cu, lengths, *,
                          skip_unseen_rows=False):
  """Every request's rows match the reference (skip_unseen_rows drops the rows
  that see no key), packed padding is out = 0 and lse = -inf, and the caches
  match exactly; lse is None when the call did not return it."""
  out, lse, updated_k, updated_v = actual
  cu, lengths = np.asarray(cu), np.asarray(lengths)
  packed_q_end = int(cu[-1])
  owned_rows = np.arange(int(cu[0]), packed_q_end)
  if skip_unseen_rows:
    checked = owned_rows[np.isfinite(expected[1][0, owned_rows])]
  else:
    checked = owned_rows
  out = np.asarray(out, np.float32)
  mismatched = ~np.isclose(
      out[checked], expected[0][checked], **OUT_TOL).all(axis=(1, 2))
  if lse is None:
    padding_lse = np.full(0, -np.inf)
  else:
    lse = np.asarray(lse)
    mismatched |= ~np.isclose(
        lse[:, checked], expected[1][:, checked], **LSE_TOL).all(axis=0)
    padding_lse = lse[:, packed_q_end:]
  rows = checked[mismatched]
  seqs = np.searchsorted(cu, rows, side="right") - 1
  first_mismatches = [(int(row), int(seq), int(row - cu[seq]),
                       int(lengths[seq] + row - cu[seq]))
                      for row, seq in zip(rows[:8], seqs[:8])]
  assert not rows.size, (
      f"{rows.size}/{checked.size} checked rows differ; first (row, request,"
      f" offset, position): {first_mismatches}")
  np.testing.assert_array_equal(out[packed_q_end:], 0,
                                err_msg="packed padding out")
  np.testing.assert_array_equal(padding_lse, -np.inf,
                                err_msg="packed padding lse")
  np.testing.assert_array_equal(np.asarray(updated_k, np.float32), expected[2])
  np.testing.assert_array_equal(np.asarray(updated_v, np.float32), expected[3])


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
  assert_extend_matches(actual, expected, case[5], case[6])


@pytest.mark.parametrize("heads,kv_heads,head_dim", [
    (16, 16, 128), (24, 4, 256), (32, 16, 128), (6, 3, 128), (8, 1, 128),
    (8, 2, 256)])
def test_ragged_cache_head_tiles(heads, kv_heads, head_dim):
  # Note (david): every group stages its kv head's (K, V) row pair, a 2-row
  # window of the interleaved head axis at any kv head, so any KV head count
  # works, odd ones and one included.
  case = random_ragged_case(lengths=(3, 5), prefixes=(37, 128), heads=heads,
                            kv_heads=kv_heads, head_dim=head_dim,
                            capacity=256, padding=0)
  expected = ragged_reference(*case, causal=True)
  actual = run_paged(*case, causal=True)
  assert_extend_matches(actual, expected, case[5], case[6])


@pytest.mark.parametrize("rotary_interleaved", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("heads,kv_heads", [(4, 4), (4, 1), (6, 3)])
def test_paged_rotary_q_at_cache_positions(heads, kv_heads, causal,
                                           rotary_interleaved):
  # Note (david): the cache holds K already rotated, so the paged path rotates
  # Q only (rotary_k=False), token t of a request at cache_seqlens + t, as
  # FlashAttention's kvcache test rotates the appended q; under a non-causal
  # mask FlashAttention puts every query at cache_seqlens instead. The tables
  # are f32, as serving passes them.
  q, kc, vc, k, v, cu, lengths, table = random_ragged_case(
      lengths=(3, 1, 130), prefixes=(127, 0, 251), heads=heads,
      kv_heads=kv_heads)
  seqlen_ro, head_dim = table.shape[1] * kc.shape[1], q.shape[-1]
  angle = np.random.default_rng(23).random(
      (seqlen_ro, head_dim // 2)) * 2 * math.pi
  cos = jnp.asarray(np.cos(angle), jnp.float32)
  sin = jnp.asarray(np.sin(angle), jnp.float32)
  q_ro = apply_rotary_emb(q, cos, sin, interleaved=rotary_interleaved,
                          seqlen_offsets=lengths, cu_seqlens=cu)
  expected = ragged_reference(q_ro, kc, vc, k, v, cu, lengths, table,
                              causal=causal)
  actual = run_paged(q, kc, vc, k, v, cu, lengths, table, causal=causal,
                     rotary_cos=cos, rotary_sin=sin,
                     rotary_interleaved=rotary_interleaved, rotary_k=False)
  assert_extend_matches(actual, expected, cu, lengths)


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
  assert_extend_matches(actual, expected, cu, lengths, skip_unseen_rows=True)


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
  assert_extend_matches(actual, expected, cu, lengths, skip_unseen_rows=True)


def test_ragged_cache_shared_read_only_prefix_and_private_append_pages():
  q, kc, vc, k, v, cu, lengths, table = random_ragged_case(
      lengths=(2, 7), prefixes=(128, 128), padding=0)
  table = table.at[1, 0].set(table[0, 0])
  expected = ragged_reference(q, kc, vc, k, v, cu, lengths, table,
                              causal=True)
  actual = run_extend(q, kc, vc, k, v, cu, lengths, table, return_lse=False,
                      causal=True)
  assert_extend_matches(actual, expected, cu, lengths)


@pytest.mark.parametrize("append", [False, True])
def test_empty_packed_allocation(append):
  # Note (david): a zero-row buffer is outside the varlen contract (the
  # packed path rejects it too), so the allocation holds packed padding only.
  q, kc, vc, k, v, cu, lengths, table = random_ragged_case(
      lengths=(0, 0), prefixes=(0, 0), padding=8)
  k, v = (k, v) if append else (None, None)
  expected = ragged_reference(q, kc, vc, k, v, cu, lengths, table,
                              causal=True)
  actual = run_extend(q, kc, vc, k, v, cu, lengths, table, causal=True)
  assert_extend_matches(actual, expected, cu, lengths)


@pytest.mark.parametrize("heads,kv_heads", [(132, 6), (128, 8)])
def test_ragged_cache_many_heads(heads, kv_heads):
  # Note (david): a group is one kv head whose 22 or 16 q heads fold into the
  # score fragment rows, so the q compute tile shrinks to 32 or 64 tokens.
  case = random_ragged_case(lengths=(67, 11), prefixes=(3, 128), heads=heads,
                            kv_heads=kv_heads, head_dim=256, capacity=256,
                            padding=0)
  expected = ragged_reference(*case, causal=True)
  actual = run_paged(*case, causal=True)
  assert_extend_matches(actual, expected, case[5], case[6])


def test_ragged_cache_prefill_gqa_regression():
  case = random_ragged_case(lengths=(237,), prefixes=(2048,), heads=32,
                            kv_heads=8, capacity=3072, padding=19)
  expected = ragged_reference(*case, causal=True)
  actual = run_paged(*case, causal=True)
  assert_extend_matches(actual, expected, case[5], case[6])


def test_multitoken_replays_when_later_block_raises_anchor():
  # Note (david): the cached keys jump from 0 to 32 at position 1024, so the
  # later blocks' scores sit hundreds of log2 units above the anchor the first
  # block set: the fixed-anchor pass must notice and replay with the rescaling
  # update instead of overflowing. Only out is checked: Q is scaled in bf16, a
  # relative error that at an lse near 370 exceeds the lse tolerance.
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
  assert_extend_matches(actual, expected, case[5], case[6])


# Note (david): with N(0, 1) inputs attention spreads over the whole visible
# range, so an off-by-one causal limit or one misplaced page moves out by ~1/N,
# far below the bf16 tolerances once a prefix holds thousands of tokens. Here
# every key is a unit code and each query head aims at one or two chosen key
# positions with a 12-16 nat logit, so a visibility, paging or row-placement
# error moves out/lse by O(1). Stale cache slots hold copies of live codes, so
# attending one is loud too.
def probe_case(lengths, prefixes, *, heads=4, kv_heads=2, head_dim=128,
               page_size=128, pages_per_seq=None, padding=0, causal=True,
               logit=12.0, window_left=None, seed=7):
  """Needle-probe inputs in random_ragged_case's tuple layout.

  An active row's table past its live pages holds foreign valid page ids.
  Even heads aim at the row itself, the row before it or a recent row, plus
  the first hidden key (causal) or a random request row (not causal). Odd
  heads aim at a visible position, half the time one next to a page,
  256/512/1024/2048-token or 128-row chunk edge. window_left keeps every
  target inside the row's window, and half the rows of each head aim at its
  first visible key plus the last hidden one instead.
  """
  rng = np.random.default_rng(seed)
  batch = len(lengths)
  scale = 1 / math.sqrt(head_dim)
  totals = [prefix + length for prefix, length in zip(prefixes, lengths)]
  live_pages = [-(-total // page_size) for total in totals]
  pages_per_seq = max(live_pages) if pages_per_seq is None else pages_per_seq
  num_pages = sum(live_pages) + 3

  owners = rng.permutation(num_pages)
  table = rng.integers(0, num_pages, (batch, pages_per_seq))
  first_page = np.cumsum([0, *live_pages])
  for seq in range(batch):
    table[seq, :live_pages[seq]] = owners[first_page[seq]:first_page[seq + 1]]
  # Note (david): drawn, then overwritten, to keep the random stream the probe
  # logits were tuned on.
  cache_seqlens = rng.integers(-1000, 10**6, batch)
  cache_seqlens[:] = prefixes
  cu = np.concatenate(([0], np.cumsum(lengths)))
  total_q = int(cu[-1]) + padding

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
      if window_left is None:
        aimed_target, aimed_second = target, second
      else:
        # Note (david): a missed left edge splits the attention of an edge row
        # onto the hidden key it also aims at.
        first_visible = np.maximum(rows - window_left, 0)
        at_edge = rng.random(length) < 0.5
        aimed_target = np.where(at_edge, first_visible,
                                np.maximum(target, first_visible))
        aimed_second = np.where(at_edge & (first_visible > 0),
                                first_visible - 1, second)
      direction = codes[seq][aimed_target, group[head]] + np.where(
          aimed_second[:, None] >= 0, codes[seq][aimed_second, group[head]], 0)
      q[start:start + length, head] = logit / scale * direction

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
  assert_extend_matches(actual, expected, case[5], case[6])


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
  assert_extend_matches(actual, expected, case[5], case[6])


@pytest.mark.parametrize("entry", ["varlen", "kvcache"])
@pytest.mark.parametrize(
    "lengths,prefixes,heads,kv_heads,causal,window_left,softcap", [
        ((700, 300), (1500, 2900), 8, 2, True, 255, 0.0),
        ((700, 300), (1500, 2900), 8, 2, True, 1000, 30.0),
        ((300, 700), (2900, 1500), 4, 2, False, 511, 0.0),
        ((96, 45, 20), (2000, 1023, 500), 32, 8, True, 127, 50.0),
    ])
def test_paged_window_edges_long_prefix(entry, lengths, prefixes, heads,
                                        kv_heads, causal, window_left,
                                        softcap):
  # Note (david): the windows are shorter than the chunks, so each q block
  # starts its kv interval past the prefix's first blocks, and the left edge
  # crosses page and kv block edges row by row. On the varlen entry
  # max_seqlen_q = 256 splits every chunk over several q blocks. A softcap of
  # 30 or 50 bends a 16-nat needle by over a nat, which the lse check sees.
  case = probe_case(lengths, prefixes, heads=heads, kv_heads=kv_heads,
                    padding=13, causal=causal, window_left=window_left,
                    logit=16.0)
  expected = ragged_reference(*case, causal=causal, window_left=window_left,
                              softcap=softcap)
  assert_probe_is_sharp(expected[4])
  kwargs = dict(causal=causal, window_size=(window_left, 0 if causal else -1),
                softcap=softcap)
  if entry == "varlen":
    actual = run_paged(*case, max_seqlen_q=256, **kwargs)
  else:
    actual = run_extend(*case, **kwargs)
  assert_extend_matches(actual, expected, case[5], case[6])


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
  assert_extend_matches(actual, expected, case[5], case[6])


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
  assert_extend_matches(actual, expected, case[5], case[6])


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

  case = (bf16(q), bf16(pages), bf16(pages), bf16(k), bf16(v),
          jnp.array([0, length], jnp.int32), jnp.array([0], jnp.int32),
          jnp.arange(num_pages, dtype=jnp.int32)[None])
  expected = ragged_reference(*case, causal=True)
  actual = run_paged(*case, causal=True, max_seqlen_k=num_pages * page_size)
  assert np.all(np.isfinite(np.asarray(actual[0], np.float32)))
  assert_extend_matches(actual, expected, case[5], case[6])


def run_decode(q, kc, vc, k, v, cu, lengths, table, **kwargs):
  """flash_attn_with_kvcache's single-token decode on the merged pool, one q, k
  and v row per request; returns run_extend's (out, lse, updated_k,
  updated_v), out and lse in packed row order."""
  del cu
  out, lse, kv_cache = flash_attn_with_kvcache(
      q[:, None], interleave(kc, vc), None, k[:, None], v[:, None],
      block_table=table, cache_seqlens=lengths, return_softmax_lse=True,
      interpret=INTERPRET, **kwargs)
  return (out[:, 0], lse[:, :, 0].T, *deinterleave(kv_cache))


@pytest.mark.parametrize("entry", ["varlen", "kvcache", "decode"])
@pytest.mark.parametrize("heads,kv_heads,window_left,softcap", [
    (8, 2, None, 0.0),
    (8, 2, 127, 0.0),
    (4, 4, None, 30.0),
    (64, 8, 127, 0.0),
])
def test_paged_attention_sinks(entry, heads, kv_heads, window_left, softcap):
  # Note (deepolar): sinks near the rows' lse move a large share of each
  # head's mass off the keys, so a sink dropped from either the output or the
  # lse fails the check. 64:8 heads with a 128-token window are gpt-oss's
  # sliding layers. A request with no new tokens owns no rows, so it checks
  # that the sinks stay with their heads across requests.
  if entry == "decode":
    lengths, prefixes, padding = (1, 1, 1), (0, 300, 900), 0
  else:
    lengths, prefixes, padding = (131, 17, 0), (123, 900, 64), 5
  case = random_ragged_case(lengths=lengths, prefixes=prefixes, heads=heads,
                            kv_heads=kv_heads, capacity=1024, padding=padding)
  sinks = jnp.asarray(
      np.random.default_rng(3).uniform(-2.0, 8.0, heads), jnp.float32)
  expected = ragged_reference(*case, causal=True, window_left=window_left,
                              softcap=softcap, sinks=sinks)
  kwargs = dict(causal=True,
                window_size=(-1 if window_left is None else window_left, 0),
                softcap=softcap, learnable_sink=sinks)
  if entry == "varlen":
    actual = run_paged(*case, **kwargs)
  elif entry == "kvcache":
    actual = run_extend(*case, **kwargs)
  else:
    actual = run_decode(*case, **kwargs)
  assert_extend_matches(actual, expected, case[5], case[6])


@pytest.mark.parametrize("window_left", [127, None])
def test_paged_decode_attention_sinks_gpt_oss(window_left):
  # Note (deepolar): gpt-oss decodes 64 q heads on 8 kv heads at head_dim 64;
  # the paged extend kernel still needs head_dim 128, so decode alone runs it.
  case = random_ragged_case(lengths=(1, 1, 1), prefixes=(0, 300, 900),
                            heads=64, kv_heads=8, head_dim=64, capacity=1024,
                            padding=0)
  sinks = jnp.asarray(
      np.random.default_rng(3).uniform(-2.0, 8.0, 64), jnp.float32)
  expected = ragged_reference(*case, causal=True, window_left=window_left,
                              sinks=sinks)
  actual = run_decode(
      *case, causal=True, learnable_sink=sinks,
      window_size=(-1 if window_left is None else window_left, 0))
  assert_extend_matches(actual, expected, case[5], case[6])


@pytest.mark.parametrize("dtype", [jnp.float8_e4m3fn, jnp.float8_e5m2])
@pytest.mark.parametrize("entry", ["kvcache", "decode"])
@pytest.mark.parametrize("heads,kv_heads,window_left", [
    (8, 2, None), (64, 8, 127), (4, 4, None)])
def test_paged_fp8_attention_sinks(dtype, entry, heads, kv_heads, window_left):
  # Note (deepolar): a sink joins after the descales, so it must not take the
  # K descale and the V descale must still scale the output. A sink call then
  # equals the plain call exactly: out_s = out * sigmoid(lse - sink) and
  # lse_s = logaddexp(lse, sink).
  if entry == "decode":
    lengths, prefixes, padding = (1, 1, 1), (0, 300, 900), 0
  else:
    lengths, prefixes, padding = (131, 17, 0), (123, 900, 64), 5
  q, kc, vc, k, v, cu, lens, table = random_ragged_case(
      lengths=lengths, prefixes=prefixes, heads=heads, kv_heads=kv_heads,
      capacity=1024, padding=padding)
  kc, vc, k, v = (x.astype(dtype) for x in (kc, vc, k, v))
  case = (q, kc, vc, k, v, cu, lens, table)
  sinks = jnp.asarray(
      np.random.default_rng(3).uniform(-2.0, 8.0, heads), jnp.float32)
  kwargs = dict(causal=True, k_descale=0.5, v_descale=2.0,
                window_size=(-1 if window_left is None else window_left, 0))
  run = run_decode if entry == "decode" else run_extend
  out, lse, *_ = run(*case, **kwargs)
  sink_out, sink_lse, *_ = run(*case, learnable_sink=sinks, **kwargs)
  rows = np.arange(int(np.asarray(cu)[0]), int(np.asarray(cu)[-1]))
  out, sink_out = (np.asarray(x, np.float32)[rows] for x in (out, sink_out))
  lse, sink_lse = (np.asarray(x, np.float32)[:, rows] for x in (lse, sink_lse))
  head_sinks = np.asarray(sinks)[:, None]
  keep = 1 / (1 + np.exp(head_sinks - lse))
  assert np.abs(sink_lse - np.logaddexp(lse, head_sinks)).max() < 1e-2
  assert np.abs(sink_out - out * keep.T[:, :, None]).max() < 2e-2
