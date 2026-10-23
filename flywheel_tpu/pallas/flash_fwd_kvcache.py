"""Single-token KV-cache decode: one persistent Pallas call appends every row's
new token and attends its whole cache."""

import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ..tuned_block_sizes import (
  KVCacheConfig,
  get_tuned_kvcache_config,
  kvcache_config_key,
  warn_untuned,
)
from .block_sizes import (
  BF16_BYTES,
  F32_BYTES,
  LOG2E,
  NN_DIM_NUMBERS,
  NT_DIM_NUMBERS,
  NUM_LANES,
  NUM_SUBLANES,
  STAGES,
  next_pow2,
  round_up,
  vmem_limit_bytes,
)
from .copy_utils import (
  MIN_BLOCK_KV,
  bank_safe_word_stride,
  kv_heads_per_word,
  kv_word_halves,
  kv_word_rows,
  load_kv_fragment,
  load_staged_words,
  merged_cache_heads,
  resolve_merged_kv_heads,
  staged_words,
)
from .fwd_pipeline import BF16_TILE_ROWS, overflow_guard_threshold

SUPPORTED_HEAD_DIMS = (64, 128, 256)
DEFAULT_NUM_KV_STAGES = 2
NUM_IO_STAGES = 2
SEM_K = 0
SEM_V = 1
SEM_K_WRITE = 2
SEM_V_WRITE = 3
SEM_Q = 4
SEM_OUT = 5
NUM_SEMS = 6
BLOCK_KV_COMPUTE_CANDIDATES = (1024, 512, 384, 256, 128)
STAGED_KV_ELEMENT_BUDGET = 8 * 1024 * 1024
# Note (david): a row runs only the kv compute tiles that hold its keys, but a
# bigger block still issues a copy for every page in it. At Qwen3-4B's 32:8
# heads of 128 with a 32K-token table row on v6e, 256 rows of 64-1024 tokens
# took 1.15 ms with 4096-token blocks against 1.08 ms with 1024-token ones
# (1.82 ms when every tile of the block ran), while a 16K context took 15.53
# against 15.33 ms.
MAX_BLOCK_KV = 1024


def lane_broadcast(value: jax.Array, width: int) -> jax.Array:
  """(rows, width) from (rows, w) with w == 1 or w dividing width.

  A (rows, 128) value whose lanes all hold the row's number repeats along
  the lane axis as whole vregs, which costs no data movement, where a
  (rows, 1) slice loaded from VMEM needs a cross-lane broadcast per use.
  """
  rows, current = value.shape
  if current == width:
    return value
  elif current == 1:
    return jnp.broadcast_to(value, (rows, width))
  elif width < current:
    return value[:, :width]
  else:
    assert width % current == 0, (current, width)
    return jnp.concatenate([value] * (width // current), axis=1)


def static_anchor_update(
    scores: jax.Array,
    row_max_prev: jax.Array,
    row_sum_prev: jax.Array,
    *,
    is_first: bool | jax.Array,
    guard_threshold: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Online-softmax step in log2 units over (rows, w) running state, w == 1
  or NUM_LANES with every lane holding its row's value.

  Returns (probabilities, row_max, row_sum, rescale), the state at the
  width it came in.
  """
  # Note (david): row_max moves only once a fragment's max exceeds it by more
  # than guard_threshold, which keeps every exp2 finite without tracking the
  # exact running max.
  fragment_max = scores.max(axis=-1, keepdims=True)
  should_raise_max = jnp.logical_and(
      jnp.logical_not(is_first),
      fragment_max - row_max_prev > guard_threshold,
  )
  row_max = jnp.where(
      is_first,
      fragment_max,
      jnp.where(
          should_raise_max,
          jnp.maximum(row_max_prev, fragment_max),
          row_max_prev,
      ),
  )
  probabilities = jnp.exp2(scores - lane_broadcast(row_max, scores.shape[-1]))
  fragment_sum = probabilities.sum(axis=-1, keepdims=True)
  rescale = jnp.where(is_first, 1.0, jnp.exp2(row_max_prev - row_max))
  row_sum = jnp.where(
      is_first,
      fragment_sum,
      rescale * row_sum_prev + fragment_sum,
  )
  return probabilities, row_max, row_sum, rescale


def load_kv_pair_fragment(
    kv_stage: jax.Array,
    slot: jax.Array,
    kv_head: int,
    kv_compute_index: int,
    *,
    block_kv_compute: int,
    is_bitcast_load: bool,
    kv_heads_per_fragment: int = 1,
) -> tuple[jax.Array, jax.Array]:
  """(keys, values) of kv_head, each (block_kv_compute, head_dim) in the
  cache dtype, read from a merged-cache staging slot.

  With kv_heads_per_fragment 2 (an fp8 cache), kv_head starts a u32 word's
  head pair and each of keys and values is (2 * block_kv_compute, head_dim):
  row 2 * t holds token t of kv_head, row 2 * t + 1 token t of kv_head + 1.

  A staged token row holds every KV head's K and V rows: (stages, block_kv,
  word_stride, kv_word_rows, head_dim) with is_bitcast_load, its first
  cache_heads / kv_word_rows word rows live, else (stages, block_kv,
  cache_heads, head_dim).
  """
  head_dim = kv_stage.shape[-1]
  dtype = kv_stage.dtype
  heads_per_word = kv_heads_per_word(dtype)
  compute_start = pl.multiple_of(
      kv_compute_index * block_kv_compute, MIN_BLOCK_KV
  )
  if kv_heads_per_fragment == 2:
    if not is_bitcast_load:
      pair_row = kv_head // heads_per_word * kv_word_rows(dtype)
      return tuple(
          kv_stage[
              slot,
              pl.ds(compute_start, block_kv_compute),
              pl.ds(pair_row + kv_part * heads_per_word, heads_per_word),
              :,
          ].reshape(2 * block_kv_compute, head_dim)
          for kv_part in range(2)
      )
    # Note: a word's low 16 bits are the pair's two K bytes and its high 16
    # bits their two V bytes, so one u32 -> u16 pack per half yields a
    # head-interleaved fp8 tile, as the bf16 head-pair bitcast does; no byte
    # unpacking per head.
    word_stride = kv_stage.shape[2]
    words = load_staged_words(
        staged_words(kv_stage, slot),
        compute_start * word_stride + kv_head // heads_per_word,
        block_kv_compute,
        word_stride,
    )
    return (
        pltpu.bitcast(words.astype(jnp.uint16), dtype),
        pltpu.bitcast((words >> 16).astype(jnp.uint16), dtype),
    )
  if not is_bitcast_load:
    key_row = (kv_head // heads_per_word * kv_word_rows(dtype)
               + kv_head % heads_per_word)
    keys, values = (
        kv_stage[
            slot,
            pl.ds(compute_start, block_kv_compute),
            pl.ds(key_row + kv_part * heads_per_word, 1),
            :,
        ].reshape(block_kv_compute, head_dim)
        for kv_part in range(2)
    )
    return keys, values
  else:
    # Note (david): the K and V of KV head h share u32 word row token *
    # word_stride + h // heads_per_word, so one strided word load yields
    # both. At one KV head this costs 1.4-1.5x the head-major decode of
    # an unmerged K/V cache (6 query heads, head_dim 256, v6e), yet splitting
    # the pair into dense K/V buffers per block (1.45-2.25x), folding the heads
    # into the row width (6.5-12x) and scoring the pair interleaved
    # (1.46-2.02x) all measured worse. Scoring it interleaved at one query
    # head per KV head (32 KV heads of 256) doubles the MXU pushes and lost
    # too: 9750 against 7149 v7x bundles per 512-token block, 9.38 against
    # 6.19 ms a compute-only v6e step at batch 32.
    word_stride = kv_stage.shape[2]
    words = load_staged_words(
        staged_words(kv_stage, slot),
        compute_start * word_stride + kv_head // heads_per_word,
        block_kv_compute,
        word_stride,
    )
    return kv_word_halves(words, kv_head, dtype)


def kvcache_kernel(
    cache_seqlens_ref: jax.Array,
    cache_batch_idx_ref: jax.Array,
    num_active_ref: jax.Array,
    v_scale_ref: jax.Array,
    *refs: jax.Array,
    batch: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_kv: int,
    block_kv_compute: int,
    num_kv_stages: int,
    is_cache_head_major: bool,
    is_bitcast_load: bool,
    return_lse: bool,
    has_new: bool,
    left: int | None,
    query_tokens: int,
    causal: bool,
    log2_softcap: float,
    guard_threshold: float,
    page_size: int | None,
    pages_per_seq: int | None,
    is_merged_cache: bool,
) -> None:
  remaining_refs = list(refs)
  block_table_ref = remaining_refs.pop(0) if page_size is not None else None
  num_active = num_active_ref[0]
  if is_merged_cache:
    # Note (david): a merged build's new token arrives as one
    # (staged_kv_heads, head_dim) row already in the cache's head order, so
    # the append writes it as a ready token row.
    (
        q_ref,
        new_kv_ref,
        k_cache_ref,
        out_ref,
        k_out_ref,
        *remaining_refs,
    ) = remaining_refs
    k_ref = v_ref = v_cache_ref = v_out_ref = None
  else:
    (
        q_ref,
        k_ref,
        v_ref,
        k_cache_ref,
        v_cache_ref,
        out_ref,
        k_out_ref,
        v_out_ref,
        *remaining_refs,
    ) = remaining_refs
    new_kv_ref = None
  lse_ref = remaining_refs.pop(0) if return_lse else None
  (
      k_buffers,
      v_buffers,
      k_write_buffers,
      v_write_buffers,
      q_buffers,
      out_buffers,
      row_max_ref,
      row_sum_ref,
      accumulator_ref,
      sems,
  ) = remaining_refs

  query_heads_per_kv_head = num_query_heads // num_kv_heads
  cache_dtype = k_cache_ref.dtype
  # Note (david): a merged build stages whole token rows, every KV head's
  # (K, V) pair (and an fp8 cache's zero pad head).
  if is_merged_cache:
    staged_kv_heads = merged_cache_heads(num_kv_heads, cache_dtype)
  else:
    staged_kv_heads = num_kv_heads
  word_rows = kv_word_rows(cache_dtype)
  num_kv_compute_fragments = block_kv // block_kv_compute
  # Note (david): a bitcast head-pair tile interleaves its two heads per token,
  # so both heads' query groups ride one MXU pass with the off-parity columns
  # masked. That is free while the pair's query rows fit one 8-sublane f32
  # tile; past it the masked half doubles the softmax work, so larger groups
  # unpack one head per fragment. A merged row's u32 word pairs a head's K
  # with its V instead, so a merged build always unpacks one head per
  # fragment.
  # Note: an fp8 merged word holds a head pair's K bytes in its low half and
  # V bytes in its high half, so the pair rides one fragment like the bf16
  # head-pair bitcast.
  is_pair_word_fragment = (
      is_merged_cache
      and kv_heads_per_word(cache_dtype) == 2
      and NUM_SUBLANES // 2 < 2 * query_heads_per_kv_head <= NUM_SUBLANES
      and num_kv_heads % 2 == 0
  )
  if (
      is_bitcast_load
      and 2 * query_heads_per_kv_head <= NUM_SUBLANES
      and not is_merged_cache
  ) or is_pair_word_fragment:
    kv_heads_per_fragment = 2
  else:
    kv_heads_per_fragment = 1
  # Note (david): a merged build whose KV head serves fewer than 8 query
  # heads scores several KV heads through one 8-row query tile: each head's
  # keys are pushed once, as before, but the LHS carries the whole group's
  # query rows, and the head keeps its own rows of the result. The group then
  # runs one softmax over a full f32 tile instead of one per head over a
  # mostly empty one, and its running max, sum and accumulator are whole
  # tiles. At 32 KV heads of 256 (MHA) the v7x 512-token block loop went
  # from 7149 to 5223 bundles (RPA v3: 7165), losing the per-head cross-lane
  # broadcasts, sublane rotates and single-row state stores.
  if (
      is_merged_cache
      and not is_pair_word_fragment
      and NUM_SUBLANES % query_heads_per_kv_head == 0
      and num_kv_heads % (NUM_SUBLANES // query_heads_per_kv_head) == 0
  ):
    kv_heads_per_group = NUM_SUBLANES // query_heads_per_kv_head
  else:
    kv_heads_per_group = 1
  num_head_groups = num_kv_heads // kv_heads_per_fragment // kv_heads_per_group
  rows_per_fragment = (
      query_heads_per_kv_head * kv_heads_per_fragment * kv_heads_per_group
  )
  fragment_keys = kv_heads_per_fragment * block_kv_compute
  # Note (david): row r of a group's tile belongs to the group's KV head
  # r // query_heads_per_kv_head.
  if kv_heads_per_group == 1:
    score_row_head = pv_row_head = None
  else:
    score_row_head = lax.broadcasted_iota(
        jnp.int32, (rows_per_fragment, fragment_keys), 0
    ) // query_heads_per_kv_head
    pv_row_head = lax.broadcasted_iota(
        jnp.int32, (rows_per_fragment, head_dim), 0
    ) // query_heads_per_kv_head
  # Note (david): a folded window lays a KV head's rows out token by token, so
  # row r is window token t = r // group, which causal stops
  # query_tokens - 1 - t keys short of total_len.
  if causal and query_tokens > 1:
    row_key_offset = (
        lax.broadcasted_iota(jnp.int32, (rows_per_fragment, fragment_keys), 0)
        % query_heads_per_kv_head
        // (query_heads_per_kv_head // query_tokens)
        - (query_tokens - 1)
    )
  else:
    row_key_offset = None
  if kv_heads_per_fragment == 1:
    parity_mask = None
  else:
    column_index = jnp.arange(fragment_keys, dtype=jnp.int32)
    row_parity = (
        jnp.arange(rows_per_fragment, dtype=jnp.int32)
        // query_heads_per_kv_head
    )
    parity_mask = (
        column_index[None, :] % kv_heads_per_fragment == row_parity[:, None]
    )
  if is_merged_cache:
    kv_pair_fragment = functools.partial(
        load_kv_pair_fragment,
        block_kv_compute=block_kv_compute,
        is_bitcast_load=is_bitcast_load,
        kv_heads_per_fragment=kv_heads_per_fragment,
    )
  else:
    kv_fragment = functools.partial(
        load_kv_fragment,
        block_kv_compute=block_kv_compute,
        kv_heads_per_fragment=kv_heads_per_fragment,
        is_cache_head_major=is_cache_head_major,
        is_bitcast_load=is_bitcast_load,
    )

  # Note (david): every cache DMA on the token axis moves whole cache_tile_rows
  # tiles. A head-major stage puts tokens on its bf16 sublane axis, which v7x
  # tiles 16 rows deep (BF16_TILE_ROWS; v6e 8), and Mosaic sizes a DMA and its
  # wait between the 8-row HBM tiles and that stage in whole 16-row tiles, so
  # an 8-row remainder was dropped from the copy and the wait returned before
  # the DMA landed (v7x read stale tokens; the leftover semaphore count then
  # released later waits early). A token-major stage puts tokens on a leading
  # axis, where each token is whole tiles and the 8-row granule is exact.
  cache_tile_rows = BF16_TILE_ROWS if is_cache_head_major else NUM_SUBLANES

  def _cache_copy_size(cache_len, block_index):
    copy_limit = (
        (cache_len + int(has_new) + (cache_tile_rows - 1))
        // cache_tile_rows * cache_tile_rows
    )
    copy_size = jnp.clip(
        copy_limit - block_index * block_kv, 0, block_kv
    )
    return pl.multiple_of(copy_size, cache_tile_rows)

  def _cache_tile(ref, hbm_row, token_start, num_tokens):
    if is_cache_head_major:
      return ref.at[
          hbm_row, slice(None), pl.ds(token_start, num_tokens), slice(None)
      ]
    else:
      return ref.at[
          hbm_row, pl.ds(token_start, num_tokens), slice(None), slice(None)
      ]

  def _buffer_tile(buffers, slot, token_start, num_tokens):
    if is_cache_head_major:
      return buffers.at[slot, :, pl.ds(token_start, num_tokens), :]
    elif is_bitcast_load:
      # Note (david): the staged rows past staged_kv_heads are the bank pad,
      # which no DMA fills.
      word_stride = buffers.shape[2]
      return buffers.at[slot].reshape(
          block_kv, word_rows * word_stride, head_dim
      ).at[pl.ds(token_start, num_tokens), pl.ds(0, staged_kv_heads), :]
    else:
      return buffers.at[slot, pl.ds(token_start, num_tokens), :, :]

  # Note (david): a merged row carries each head's V next to its K, so a
  # merged build stages, appends and writes back whole token rows through the
  # K buffers only, one contiguous DMA per page or appended tile.
  k_load = (k_cache_ref, k_buffers, SEM_K)
  k_write = (k_write_buffers, k_out_ref, SEM_K_WRITE)
  if is_merged_cache:
    load_parts = (k_load,)
    write_parts = (k_write,)
    value_buffers = k_buffers
  else:
    load_parts = (k_load, (v_cache_ref, v_buffers, SEM_V))
    write_parts = (k_write, (v_write_buffers, v_out_ref, SEM_V_WRITE))
    value_buffers = v_buffers

  def _load_segments(cache_row, block_index, load_size):
    # Note (david): a paged block is one piece per page, each a full page
    # except the clipped last one (RPA sizing; zero-length tails are issued
    # as-is). block_index * pages_per_block + page never leaves the row's table
    # entries because the table is padded to whole blocks and the caller keeps
    # cache_len + has_new <= capacity.
    if page_size is None:
      return [(cache_row, block_index * block_kv, load_size, 0)]
    else:
      pages_per_block = block_kv // page_size
      table_base = cache_row * pages_per_seq + block_index * pages_per_block
      return [
          (
              block_table_ref[table_base + page],
              0,
              pl.multiple_of(
                  jnp.clip(load_size - page * page_size, 0, page_size),
                  cache_tile_rows,
              ),
              page * page_size,
          )
          for page in range(pages_per_block)
      ]

  def _start_kv(cache_row, cache_len, block_index, slot):
    load_size = _cache_copy_size(cache_len, block_index)

    @pl.when(load_size > 0)
    def _start():
      for hbm_row, hbm_start, num_tokens, buffer_start in _load_segments(
          cache_row, block_index, load_size
      ):
        for cache_ref, buffers, sem_index in load_parts:
          pltpu.make_async_copy(
              _cache_tile(cache_ref, hbm_row, hbm_start, num_tokens),
              _buffer_tile(buffers, slot, buffer_start, num_tokens),
              sems.at[sem_index, slot, 0],
          ).start()

  def _wait_kv(cache_len, block_index, slot):
    load_size = _cache_copy_size(cache_len, block_index)

    # Note (david): every segment of a block signals the same K (or V)
    # semaphore and a DMA semaphore counts bytes, so one wait sized to the
    # whole staged window covers them all (the RPA pattern); src == dst
    # because only the size matters.
    @pl.when(load_size > 0)
    def _wait():
      for _, buffers, sem_index in load_parts:
        tile = _buffer_tile(buffers, slot, 0, load_size)
        pltpu.make_async_copy(tile, tile, sems.at[sem_index, slot, 0]).wait()

  def _cache_write_copies(cache_row, cache_len, slot):
    if page_size is None:
      hbm_row = cache_row
      tile_start = pl.multiple_of(
          cache_len // cache_tile_rows * cache_tile_rows, cache_tile_rows
      )
    else:
      # Note (david): page_size is a multiple of 128, so a write tile at a
      # tile-aligned offset never straddles a page and the paged append stays
      # a single write.
      hbm_row = block_table_ref[
          cache_row * pages_per_seq + cache_len // page_size
      ]
      tile_start = pl.multiple_of(
          cache_len % page_size // cache_tile_rows * cache_tile_rows,
          cache_tile_rows,
      )
    if is_cache_head_major:
      # Note (david): one copy per head, since the head axis of a cache row is
      # not contiguous.
      return [
          pltpu.make_async_copy(
              buffers.at[slot, pl.ds(head_offset, 1), :, :],
              out_ref_hbm.at[
                  hbm_row,
                  pl.ds(head_offset, 1),
                  pl.ds(tile_start, cache_tile_rows),
                  :,
              ],
              sems.at[sem_index, slot, head_offset],
          )
          for head_offset in range(num_kv_heads)
          for buffers, out_ref_hbm, sem_index in write_parts
      ]
    else:
      return [
          pltpu.make_async_copy(
              buffers.at[slot].reshape(
                  cache_tile_rows, staged_kv_heads, head_dim
              )
              if is_bitcast_load
              else buffers.at[slot],
              out_ref_hbm.at[hbm_row, pl.ds(tile_start, cache_tile_rows), :, :],
              sems.at[sem_index, slot, 0],
          )
          for buffers, out_ref_hbm, sem_index in write_parts
      ]

  def _q_copy(batch_index, slot):
    return pltpu.make_async_copy(
        q_ref.at[batch_index], q_buffers.at[slot], sems.at[SEM_Q, slot, 0]
    )

  def _output_copy(batch_index, slot):
    return pltpu.make_async_copy(
        out_buffers.at[slot], out_ref.at[batch_index],
        sems.at[SEM_OUT, slot, 0],
    )

  # Note (david): staged V rows a block never loads keep whatever VMEM held;
  # start them finite so a zero probability times a stale row stays zero.
  if is_merged_cache and is_bitcast_load:
    # Note: through its (tokens * word_stride, head_dim) word view a slot
    # zeroes in whole 8-row stores; the 5-D view's one-word token rows took
    # one store per row. v7x, 32:8 heads, two 2048-token stages: 18450 ->
    # 2322 bundles a call.
    for zero_slot in range(num_kv_stages):
      slot_words = staged_words(value_buffers, zero_slot)
      slot_words[...] = jnp.zeros(slot_words.shape, jnp.uint32)
  else:
    value_words = (
        value_buffers.bitcast(jnp.uint32) if is_bitcast_load
        else value_buffers
    )
    value_words[...] = jnp.zeros(value_words.shape, value_words.dtype)

  def _block_range(cache_len):
    total_len = cache_len + int(has_new)
    if left is None:
      first_key = jnp.int32(0)
    else:
      first_key = jnp.maximum(0, total_len - 1 - left)
    return first_key, first_key // block_kv, pl.cdiv(total_len, block_kv)

  def _process_sequence(local_batch_index, sequence_carry):
    initial_slot, is_first_prefetched = sequence_carry
    cache_row = cache_batch_idx_ref[local_batch_index]
    cache_len = cache_seqlens_ref[local_batch_index]
    total_len = cache_len + int(has_new)
    first_key, first_block, end_block = _block_range(cache_len)

    row_max_ref[...] = jnp.full(
        row_max_ref.shape, -jnp.inf, jnp.float32
    )
    row_sum_ref[...] = jnp.zeros(row_sum_ref.shape, jnp.float32)
    accumulator_ref[...] = jnp.zeros(
        accumulator_ref.shape, q_ref.dtype
    )

    @pl.when(
        jnp.logical_and(
            first_block < end_block, jnp.logical_not(is_first_prefetched)
        )
    )
    def _prefetch_first_block():
      _start_kv(cache_row, cache_len, first_block, initial_slot)

    next_local_batch_index = jnp.minimum(
        local_batch_index + 1, num_active - 1
    )
    has_next_sequence = local_batch_index + 1 < num_active
    next_cache_row = cache_batch_idx_ref[next_local_batch_index]
    next_cache_len = cache_seqlens_ref[next_local_batch_index]
    _, next_first_block, next_end_block = _block_range(next_cache_len)
    next_has_blocks = jnp.logical_and(
        has_next_sequence, next_first_block < next_end_block
    )
    q_slot = lax.rem(local_batch_index, NUM_IO_STAGES)
    _q_copy(local_batch_index, q_slot).wait()
    queries = q_buffers[q_slot]
    next_q_slot = lax.rem(q_slot + 1, NUM_IO_STAGES)

    @pl.when(has_next_sequence)
    def _prefetch_next_q():
      _q_copy(next_local_batch_index, next_q_slot).start()

    def _score_and_softmax(
        slot,
        head_group,
        kv_compute_index,
        is_valid,
        fragment_has_valid_key,
        is_first,
        row_max_prev,
        row_sum_prev,
    ):
      # Note (david): the query heads of one head group are consecutive rows of
      # q, so one (rows_per_fragment, head_dim) tile rides a single K load and
      # MXU pass.
      group_start = head_group * rows_per_fragment
      query = queries[group_start : group_start + rows_per_fragment, :]
      if is_merged_cache:
        # Note (david): a head's K and V share their u32 words, so the score's
        # word load also yields the values, held until this fragment's pv.
        # Loading the words again for pv cost 1.30x the blocked layout's MHA
        # decode (32 KV heads of 256, batch 32, v6e), whose 32-row word stride
        # takes the costly strided load; one load is at parity.
        scores = None
        values = []
        for group_member in range(kv_heads_per_group):
          key, value = kv_pair_fragment(
              k_buffers,
              slot,
              (head_group * kv_heads_per_group + group_member)
              * kv_heads_per_fragment,
              kv_compute_index,
          )
          values.append(value)
          # Note (david): K and V stay in the cache dtype; the MXU widens an
          # fp8 latched operand to bf16 itself.
          member_scores = lax.dot_general(
              query, key, NT_DIM_NUMBERS, preferred_element_type=jnp.float32
          )
          if scores is None:
            scores = member_scores
          else:
            scores = jnp.where(
                score_row_head == group_member, member_scores, scores
            )
      else:
        key = kv_fragment(k_buffers, slot, head_group, kv_compute_index)
        values = None
        scores = jnp.dot(query, key.T, preferred_element_type=jnp.float32)
      # Note (david): Q carries softmax_scale / softcap, so log2(e) rides in the
      # tanh output scale. The cap runs before the mask, since tanh would lift
      # a masked -inf to -softcap.
      if log2_softcap:
        capped_scores = jnp.tanh(scores) * log2_softcap
      else:
        capped_scores = scores
      scores = jnp.where(is_valid, capped_scores, -jnp.inf)

      if left is None:
        probabilities, row_max, row_sum, rescale = static_anchor_update(
            scores,
            row_max_prev,
            row_sum_prev,
            is_first=is_first,
            guard_threshold=guard_threshold,
        )
      else:
        # Note (david): a sliding window can leave a fragment with no visible
        # key, whose all -inf scores would turn the update into NaN; score it
        # as zeros and keep the old state. The state stays empty until the
        # first visible key, hence is_first = row_sum_prev == 0.
        safe_scores = jnp.where(
            fragment_has_valid_key, scores, jnp.zeros_like(scores)
        )
        (
            candidate_probabilities,
            candidate_row_max,
            candidate_row_sum,
            candidate_rescale,
        ) = static_anchor_update(
            safe_scores,
            row_max_prev,
            row_sum_prev,
            is_first=row_sum_prev == 0,
            guard_threshold=guard_threshold,
        )
        probabilities = jnp.where(
            fragment_has_valid_key,
            candidate_probabilities,
            jnp.zeros_like(candidate_probabilities),
        )
        row_max = jnp.where(
            fragment_has_valid_key, candidate_row_max, row_max_prev
        )
        row_sum = jnp.where(
            fragment_has_valid_key, candidate_row_sum, row_sum_prev
        )
        rescale = jnp.where(
            fragment_has_valid_key,
            candidate_rescale,
            jnp.ones_like(candidate_rescale),
        )
      return probabilities, rescale, values, row_max, row_sum

    def _pv(
        slot,
        head_group,
        kv_compute_index,
        probabilities,
        rescale,
        merged_values,
    ):
      group_slice = pl.ds(head_group * rows_per_fragment, rows_per_fragment)
      probabilities = probabilities.astype(jnp.bfloat16)
      if is_merged_cache:
        weighted_values = None
        for group_member, value in enumerate(merged_values):
          member_values = lax.dot_general(
              probabilities, value, NN_DIM_NUMBERS,
              preferred_element_type=jnp.float32,
          )
          if weighted_values is None:
            weighted_values = member_values
          else:
            weighted_values = jnp.where(
                pv_row_head == group_member, member_values, weighted_values
            )
      else:
        value = kv_fragment(value_buffers, slot, head_group, kv_compute_index)
        weighted_values = jnp.dot(
            probabilities, value, preferred_element_type=jnp.float32
        )
      # Note (david): the accumulator is stored in q's dtype while rescale and
      # weighted_values are f32; the explicit upcast keeps the kernel tracing
      # under jax_numpy_dtype_promotion=strict.
      accumulator_ref[group_slice, :] = (
          accumulator_ref[group_slice, :].astype(jnp.float32)
          * lane_broadcast(rescale, head_dim)
          + weighted_values
      ).astype(q_ref.dtype)

    def _attend_block(block_index, has_prefetched_next_sequence):
      slot = lax.rem(
          initial_slot + block_index - first_block, num_kv_stages
      )
      next_block = block_index + 1
      next_slot = lax.rem(slot + 1, num_kv_stages)

      @pl.when(next_block < end_block)
      def _prefetch_next_block():
        _start_kv(cache_row, cache_len, next_block, next_slot)

      should_prefetch_next_sequence = jnp.logical_and(
          next_block == end_block, next_has_blocks
      )

      @pl.when(should_prefetch_next_sequence)
      def _prefetch_next_sequence():
        _start_kv(
            next_cache_row, next_cache_len, next_first_block, next_slot
        )

      _wait_kv(cache_len, block_index, slot)

      if has_new:
        append_block = cache_len // block_kv

        @pl.when(block_index == append_block)
        def _append():
          # Note (david): the append reads, updates and writes back the whole
          # cache tile holding the new token, cache_tile_rows tokens at a
          # tile-aligned offset, which the block load has staged in full.
          local_index = cache_len - block_index * block_kv
          local_tile_start = pl.multiple_of(
              local_index // cache_tile_rows * cache_tile_rows,
              cache_tile_rows,
          )
          new_row = local_index - local_tile_start
          token_row = jnp.arange(cache_tile_rows, dtype=jnp.int32)
          tile_tokens = pl.ds(local_tile_start, cache_tile_rows)
          if is_cache_head_major:
            tile_index = (slot, slice(None), tile_tokens, slice(None))
            token_rows = token_row[None, :, None]
            expand = lambda token: token[:, None, :]
          elif is_bitcast_load:
            tile_index = (slot, tile_tokens,
                          pl.ds(0, staged_kv_heads // word_rows),
                          slice(None), slice(None))
            token_rows = token_row[:, None, None, None]
            expand = lambda token: token.reshape(
                staged_kv_heads // word_rows, word_rows, head_dim
            )[None]
          else:
            tile_index = (slot, tile_tokens, slice(None), slice(None))
            token_rows = token_row[:, None, None]
            expand = lambda token: token[None]
          is_new_row = token_rows == new_row
          if is_merged_cache:
            new_tokens = (new_kv_ref[local_batch_index, :, :],)
          else:
            new_tokens = (
                k_ref[local_batch_index, :, :], v_ref[local_batch_index, :, :]
            )
          updated_tiles = []
          for (_, buffers, _), new_token in zip(load_parts, new_tokens):
            staged_tile = buffers[tile_index]
            updated_tile = jnp.where(is_new_row, expand(new_token), staged_tile)
            buffers[tile_index] = updated_tile
            updated_tiles.append(updated_tile)

          write_slot = lax.rem(local_batch_index, num_kv_stages)

          @pl.when(local_batch_index >= num_kv_stages)
          def _wait_previous_cache_write():
            previous_batch_index = local_batch_index - num_kv_stages
            previous_cache_row = cache_batch_idx_ref[previous_batch_index]
            previous_cache_len = cache_seqlens_ref[previous_batch_index]
            for copy in _cache_write_copies(
                previous_cache_row, previous_cache_len, write_slot
            ):
              copy.wait()

          for (write_buffers, _, _), updated_tile in zip(
              write_parts, updated_tiles
          ):
            write_buffers[write_slot] = updated_tile
          for copy in _cache_write_copies(cache_row, cache_len, write_slot):
            copy.start()

      block_start = block_index * block_kv
      # Note (david): the running state is read as whole (rows, 128) tiles
      # with every lane holding its row's value, so every broadcast against
      # the scores or the accumulator is a vreg repeat, not a cross-lane op.
      group_slices = [
          pl.ds(head_group * rows_per_fragment, rows_per_fragment)
          for head_group in range(num_head_groups)
      ]

      def _fragment_masks(kv_compute_index):
        key_start = block_start + kv_compute_index * block_kv_compute
        # Note (david): build the mask at full fragment shape; a (1, N) mask
        # needs a sublane-broadcast relayout inside every fragment's select,
        # which measured ~40% of MQA decode time.
        key_position = key_start + lax.broadcasted_iota(
            jnp.int32, (rows_per_fragment, fragment_keys), 1
        ) // kv_heads_per_fragment
        if row_key_offset is None:
          key_end = total_len
        else:
          key_end = total_len + row_key_offset
        if left is None:
          is_in_window = key_position < key_end
        else:
          is_in_window = (
              (key_position < key_end) & (key_position >= first_key)
          )
        if parity_mask is None:
          is_valid = is_in_window
        else:
          is_valid = is_in_window & parity_mask
        fragment_has_valid_key = jnp.logical_and(
            key_start < total_len,
            key_start + block_kv_compute > first_key,
        )
        return is_valid, fragment_has_valid_key

      def _attend_every_tile():
        fragment_masks = [
            _fragment_masks(kv_compute_index)
            for kv_compute_index in range(num_kv_compute_fragments)
        ]
        # Note (david): software pipeline; each fragment's scores are issued
        # one step ahead of its own pv, so the MXU has the next matmul queued
        # while pv runs.
        pending_fragment = None
        for fragment_index in range(
            num_head_groups * num_kv_compute_fragments
        ):
          head_group, kv_compute_index = divmod(
              fragment_index, num_kv_compute_fragments
          )
          group_slice = group_slices[head_group]
          if kv_compute_index == 0:
            row_max = row_max_ref[group_slice, :]
            row_sum = row_sum_ref[group_slice, :]
          probabilities, rescale, values, row_max, row_sum = (
              _score_and_softmax(
                  slot,
                  head_group,
                  kv_compute_index,
                  *fragment_masks[kv_compute_index],
                  block_index == first_block if kv_compute_index == 0
                  else False,
                  row_max,
                  row_sum,
              )
          )
          if kv_compute_index == num_kv_compute_fragments - 1:
            row_max_ref[group_slice, :] = row_max
            row_sum_ref[group_slice, :] = row_sum
          if pending_fragment is not None:
            _pv(slot, *pending_fragment)
          pending_fragment = (
              head_group, kv_compute_index, probabilities, rescale, values
          )
        _pv(slot, *pending_fragment)

      def _attend_live_tiles():
        def _attend_tile(kv_compute_index, row_states):
          is_valid, fragment_has_valid_key = _fragment_masks(kv_compute_index)
          # Note (david): without a window the row's first tile is tile 0 of
          # its first block; a window's state starts at its first visible key
          # instead (row_sum_prev == 0 in _score_and_softmax).
          is_first = jnp.logical_and(
              block_index == first_block, kv_compute_index == 0
          )
          # Note (david): each head group's scores are issued one step ahead
          # of the previous group's pv; the pipeline drains at every tile.
          next_row_states = []
          pending_fragment = None
          for head_group, (row_max_prev, row_sum_prev) in enumerate(
              row_states
          ):
            probabilities, rescale, values, row_max, row_sum = (
                _score_and_softmax(
                    slot,
                    head_group,
                    kv_compute_index,
                    is_valid,
                    fragment_has_valid_key,
                    is_first,
                    row_max_prev,
                    row_sum_prev,
                )
            )
            if pending_fragment is not None:
              _pv(slot, *pending_fragment)
            pending_fragment = (
                head_group, kv_compute_index, probabilities, rescale, values
            )
            next_row_states.append((row_max, row_sum))
          _pv(slot, *pending_fragment)
          return tuple(next_row_states)

        first_fragment = (
            jnp.maximum(first_key - block_start, 0) // block_kv_compute
        )
        end_fragment = jnp.minimum(
            pl.cdiv(total_len - block_start, block_kv_compute),
            num_kv_compute_fragments,
        )
        row_states = lax.fori_loop(
            first_fragment,
            end_fragment,
            _attend_tile,
            tuple(
                (row_max_ref[group_slice, :], row_sum_ref[group_slice, :])
                for group_slice in group_slices
            ),
        )
        for group_slice, (row_max, row_sum) in zip(group_slices, row_states):
          row_max_ref[group_slice, :] = row_max
          row_sum_ref[group_slice, :] = row_sum

      if num_kv_compute_fragments == 1:
        _attend_every_tile()
      else:
        # Note (david): a block whose every compute tile holds keys runs one
        # pipeline over all its fragments, as before; only a partial block
        # (a row's last, or a window's first) loops over the tiles holding
        # its keys, as RPA v3 runs cdiv(kv_len, bkv_csz) tiles. Running every
        # tile of a partial block made a row cost the whole block: at
        # Qwen3-4B's 16:4 heads, batch 128 and kv_len 256 on v6e, the
        # 8192-token blocks of 2048-token tiles of #14's v7x entries took
        # 4.33 us a row against 1.67 us looping over the first tile alone
        # (RPA v3: 3.32 us). The loop drains the pipeline at every tile, so
        # full blocks keep the single pipeline: looping over them took 1.21x
        # as long at 2048-token blocks of 256-token tiles and kv_len 4608.
        is_block_full = jnp.logical_and(
            block_start + block_kv <= total_len, first_key <= block_start
        )
        pl.when(is_block_full)(_attend_every_tile)
        pl.when(jnp.logical_not(is_block_full))(_attend_live_tiles)
      return jnp.logical_or(
          has_prefetched_next_sequence, should_prefetch_next_sequence
      )

    has_prefetched_next_sequence = lax.fori_loop(
        first_block, end_block, _attend_block, False
    )
    current_has_blocks = first_block < end_block
    should_prefetch_after_empty = jnp.logical_and(
        jnp.logical_not(current_has_blocks), next_has_blocks
    )

    @pl.when(should_prefetch_after_empty)
    def _prefetch_after_empty_sequence():
      _start_kv(
          next_cache_row, next_cache_len, next_first_block, initial_slot
      )

    output_slot = lax.rem(local_batch_index, NUM_IO_STAGES)

    @pl.when(local_batch_index >= NUM_IO_STAGES)
    def _wait_previous_output():
      _output_copy(local_batch_index - NUM_IO_STAGES, output_slot).wait()

    row_sum = row_sum_ref[...]
    # Note (david): a quantized cache's V descale is one per-tensor factor,
    # so it rides in the softmax normalization. It arrives in SMEM, so a
    # new descale value compiles no new kernel.
    inverse_row_sum = jnp.where(
        row_sum > 0, pl.reciprocal(row_sum, approx=True) * v_scale_ref[0],
        0.0
    )
    out_buffers[output_slot] = (
        accumulator_ref[...].astype(jnp.float32)
        * lane_broadcast(inverse_row_sum, head_dim)
    ).astype(q_ref.dtype)
    if return_lse:
      lse_log2 = row_max_ref[...] + jnp.log2(row_sum)
      lse = jnp.where(
          row_sum > 0, lse_log2 * math.log(2.0), -jnp.inf
      ).astype(jnp.float32)
      lse_ref[pl.ds(local_batch_index, 1), :, :] = lse[None, :, :]
    _output_copy(local_batch_index, output_slot).start()
    num_blocks = end_block - first_block
    next_initial_slot = lax.rem(
        initial_slot + jnp.maximum(num_blocks, 0), num_kv_stages
    )
    return (
        next_initial_slot,
        jnp.logical_or(
            has_prefetched_next_sequence, should_prefetch_after_empty
        ),
    )

  @pl.when(num_active > 0)
  def _prefetch_first_q():
    _q_copy(0, 0).start()

  lax.fori_loop(0, num_active, _process_sequence, (0, False))

  # Note (david): the kernel must not exit with DMAs in flight, and the loop
  # leaves the last NUM_IO_STAGES output copies and the last num_kv_stages
  # cache writes running. The candidate rows are the static tail of a full
  # batch, each guarded against running past the active prefix.
  def _drain(tail_length, wait_row):
    for tail_offset in range(min(batch, tail_length)):
      batch_index = num_active - 1 - tail_offset

      @pl.when(batch_index >= 0)
      def _wait_tail_row():
        wait_row(batch_index)

  def _wait_output(batch_index):
    _output_copy(batch_index, lax.rem(batch_index, NUM_IO_STAGES)).wait()

  _drain(NUM_IO_STAGES, _wait_output)

  if has_new:

    def _wait_cache_write(batch_index):
      cache_row = cache_batch_idx_ref[batch_index]
      cache_len = cache_seqlens_ref[batch_index]
      write_slot = lax.rem(batch_index, num_kv_stages)
      for copy in _cache_write_copies(cache_row, cache_len, write_slot):
        copy.wait()

    _drain(num_kv_stages, _wait_cache_write)


def resolve_block_kv_compute(block_kv: int) -> int:
  return next(
      candidate for candidate in BLOCK_KV_COMPUTE_CANDIDATES
      if block_kv % candidate == 0
  )


def validate_kvcache_config(
    config: KVCacheConfig,
    *,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    capacity: int,
    kv_itemsize: int,
) -> KVCacheConfig:
  """Validate a (block_kv, block_kv_compute, num_kv_stages) config."""
  block_kv, block_kv_compute, num_kv_stages = config
  if block_kv % MIN_BLOCK_KV or capacity % block_kv:
    raise ValueError(
        f"block_kv={block_kv} must be a {MIN_BLOCK_KV}-aligned divisor"
        f" of capacity={capacity}."
    )
  if block_kv_compute % MIN_BLOCK_KV or block_kv % block_kv_compute:
    raise ValueError(
        f"block_kv_compute={block_kv_compute} must be a"
        f" {MIN_BLOCK_KV}-aligned divisor of block_kv={block_kv}."
    )
  if num_kv_stages not in STAGES:
    raise ValueError(
        f"num_kv_stages must be one of {STAGES}; got {num_kv_stages}."
    )
  # Note (david): this must mirror the scratch shapes below; the extra
  # BF16_TILE_ROWS rows per stage bound the cache write tile, and the two
  # extra heads per token bound a staged block's bank pad word row.
  kv_staging_bytes = (
      2 * num_kv_stages * (block_kv + BF16_TILE_ROWS) * (num_kv_heads + 2)
      * head_dim * kv_itemsize)
  q_o_buffer_bytes = 2 * 2 * num_query_heads * head_dim * BF16_BYTES
  row_state_bytes = 2 * num_query_heads * NUM_LANES * F32_BYTES
  accumulator_bytes = num_query_heads * head_dim * BF16_BYTES
  required_vmem = (
      kv_staging_bytes + q_o_buffer_bytes + row_state_bytes + accumulator_bytes)
  if required_vmem > vmem_limit_bytes():
    raise ValueError(
        f"KV-cache config requires {required_vmem} VMEM bytes, exceeding"
        f" limit {vmem_limit_bytes()}."
    )
  return config


def flash_attn_kvcache_pallas(
    q: jax.Array,
    k_cache: jax.Array,
    v_cache: jax.Array | None,
    k: jax.Array,
    v: jax.Array,
    cache_seqlens: jax.Array,
    cache_batch_idx: jax.Array,
    *,
    cache_head_major: bool,
    has_new: bool,
    left: int | None,
    return_lse: bool,
    interpret: bool,
    block_table: jax.Array | None = None,
    num_active: int | jax.Array | None = None,
    merged_cache: bool = False,
    softcap: float = 0.0,
    causal: bool = False,
    nheads_k: int | None = None,
    v_scale: float | jax.Array = 1.0,
) -> tuple[jax.Array, ...]:
  """One-token cache append and decode attention for every row in one call.

  q: (batch, num_query_heads, head_dim), already scaled into log2 units, or
  by softmax_scale / softcap when softcap > 0, which applies Gemma-2's
  tanh(s / softcap) * softcap to the scaled scores. A (batch, window,
  num_query_heads, head_dim) q is a window already in the cache, folded into
  the query heads so each row's cache is read once; out and lse keep q's
  layout, and causal bounds window token t at cache_seqlens - window + 1 + t.
  k_cache/v_cache: (rows, capacity, num_kv_heads, head_dim), or head-major
  (rows, num_kv_heads, capacity, head_dim) with cache_head_major. With
  block_table, the flat (batch * pages_per_seq,) int32 page table, the caches
  are (num_pages, page_size, ...) pools and the per-row capacity is
  pages_per_seq * page_size.

  merged_cache (v_cache=None, token-major) makes k_cache one
  (rows, capacity, merged_cache_heads(nheads_k), head_dim) array whose token
  rows hold the KV heads' K and V rows in copy_utils' merged order (bf16
  [k0, v0, k1, v1, ...], RPA v3's; fp8 [k0, k1, v0, v1, ...]), and returns
  (out, kv_cache[, lse]) instead of (out, k_cache, v_cache[, lse]). nheads_k
  defaults to half the cache's head axis; an fp8 cache, which pads an odd
  head count, needs it. An fp8 merged cache feeds K and V to the MXU as
  stored, k and v must be in its dtype, and v_scale, a scalar that may be
  traced, multiplies the output (q carries the K scale). lse is (batch,
  num_query_heads, 128) float32, (batch, window, num_query_heads, 128) for a
  window, with each value broadcast over lanes.

  num_active: rows at or past it are padding (RPA's distribution[-1]
  contract); the kernel never reads their cache_seqlens, cache_batch_idx or
  block_table entries, appends nothing for them and leaves their out and lse
  rows unspecified. None means every row is active.
  """
  assert (v_cache is None) == merged_cache, (
      "merged_cache takes one cache operand and v_cache=None, a pair both."
  )
  assert not merged_cache or (
      not cache_head_major and k_cache.shape[2] % 2 == 0
  ), "merged_cache takes one token-major cache with an even head axis."
  cache_operands = (k_cache,) if merged_cache else (k_cache, v_cache)
  cache_shape = k_cache.shape
  is_window = q.ndim == 4
  if is_window:
    batch, query_tokens, window_heads, head_dim = q.shape
  else:
    batch, num_query_heads, head_dim = q.shape
    query_tokens = 1
  if merged_cache:
    # Note (david): a merged build stages, appends and writes back whole token
    # rows, so no DMA cuts the head axis, and a staged u32 word holds whole
    # heads' K and V: the head count needs no tile alignment or parity.
    row_tokens = cache_shape[1]
    num_kv_heads = resolve_merged_kv_heads(
        cache_shape[2], k_cache.dtype, nheads_k)
  elif k_cache.dtype != jnp.bfloat16:
    raise NotImplementedError(
        f"the K/V cache pair is bfloat16 only; got {k_cache.dtype}."
    )
  elif cache_head_major:
    num_kv_heads, row_tokens = cache_shape[1], cache_shape[2]
  else:
    row_tokens, num_kv_heads = cache_shape[1], cache_shape[2]
    if num_kv_heads != 1 and num_kv_heads % 2:
      # Note (david): the packed load bitcasts two bf16 heads into one u32
      # lane. The plain strided read an odd axis would need leaves a singleton
      # minor-but-one buffer dim that Mosaic pads to a full 8-sublane tile,
      # measured 5-10x slower at batch >= 32 and behind the RPA v3 baseline.
      raise ValueError(
          "token-major cache needs an even head axis (the packed load carries"
          f" two bf16 heads per u32 lane); got num_kv_heads={num_kv_heads}."
      )
  # Note (david): with a singleton head axis a token-major
  # (rows, capacity, 1, head_dim) cache is the same bytes as head-major
  # (rows, 1, capacity, head_dim), whose tile is a dense (block_kv, head_dim)
  # window with no head pair to pack and no 8-sublane pad.
  is_token_major_mqa = (
      not cache_head_major and not merged_cache and num_kv_heads == 1
  )
  is_cache_head_major = cache_head_major or is_token_major_mqa
  if is_token_major_mqa:
    kernel_caches = tuple(
        cache.reshape(cache.shape[0], 1, row_tokens, head_dim)
        for cache in cache_operands
    )
  else:
    kernel_caches = cache_operands
  if block_table is None:
    page_size = None
    pages_per_seq = None
    capacity = row_tokens
  else:
    page_size = row_tokens
    pages_per_seq = block_table.shape[0] // batch
    capacity = pages_per_seq * page_size
  if head_dim not in SUPPORTED_HEAD_DIMS:
    raise ValueError(
        f"head_dim must be one of {SUPPORTED_HEAD_DIMS}; got {head_dim}."
    )
  if is_window:
    assert window_heads % num_kv_heads == 0
    assert not has_new and left is None, "a window is already in the cache"
    # Note (david): putting window token t of query head h * group + j at
    # folded head (h * query_tokens + t) * group + j lets KV head h's rows
    # share one read of its keys.
    group = window_heads // num_kv_heads
    num_query_heads = num_kv_heads * query_tokens * group
    q = (
        q.reshape(batch, query_tokens, num_kv_heads, group, head_dim)
        .transpose(0, 2, 1, 3, 4)
        .reshape(batch, num_query_heads, head_dim)
    )
  if capacity % MIN_BLOCK_KV:
    raise ValueError(
        f"capacity must be a multiple of {MIN_BLOCK_KV}; got {capacity}."
    )
  # Note (david): softcap stays out of the tuned key because decode is bound by
  # the cache read; the tanh did not move the v6e step time. q is always
  # bf16, so the key's dtype is the cache's.
  tuned_key_fields = dict(
      dtype=k_cache.dtype,
      num_query_heads=num_query_heads,
      num_kv_heads=num_kv_heads,
      head_dim=head_dim,
      capacity=capacity,
      batch=batch,
      append=has_new,
      sliding_window=left,
      return_lse=return_lse,
  )
  tuned_config = get_tuned_kvcache_config(**tuned_key_fields)
  # Note (david): a paged block is whole pages, so a tuned block_kv that does
  # not tile the page cannot be used as-is.
  if (
      tuned_config is not None
      and page_size is not None
      and tuned_config[0] % page_size
  ):
    warn_untuned(
        kvcache_config_key(**tuned_key_fields),
        f"the entry's block_kv {tuned_config[0]} is not a multiple of the"
        f" page size {page_size}, and a paged block must be whole pages",
    )
    tuned_config = None
  if tuned_config is not None:
    block_kv, block_kv_compute, num_kv_stages = validate_kvcache_config(
        tuned_config,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        capacity=capacity,
        kv_itemsize=k_cache.dtype.itemsize,
    )
  else:
    granule = MIN_BLOCK_KV if page_size is None else page_size
    target_block_kv = min(
        capacity,
        STAGED_KV_ELEMENT_BUDGET // head_dim // next_pow2(2 * num_kv_heads),
        MAX_BLOCK_KV,
    )
    aligned_target = max(target_block_kv // granule * granule, granule)
    if page_size is None:
      # Note (david): capacity is a multiple of granule, so the divisor search
      # always ends.
      block_kv = next(
          block for block in range(aligned_target, 0, -granule)
          if capacity % block == 0
      )
    else:
      # Note (david): the table is padded to whole blocks below, so a page
      # count with few factors (37 pages of 256 tokens) keeps the target block
      # instead of one page per pipeline step.
      block_kv = aligned_target
    block_kv_compute = resolve_block_kv_compute(block_kv)
    num_kv_stages = DEFAULT_NUM_KV_STAGES
  # Note (david): a row's keys stay within the caller's capacity, which the
  # padded table does not raise, so the guard is sized on it.
  guard_threshold = overflow_guard_threshold(capacity)
  if page_size is not None:
    # Note (david): the padded entries lie past every row's length and each
    # page's load is clipped to it, so page 0 stands in and is never read.
    padded_pages_per_seq = round_up(pages_per_seq, block_kv // page_size)
    block_table = jnp.pad(
        block_table.reshape(batch, pages_per_seq),
        ((0, 0), (0, padded_pages_per_seq - pages_per_seq)),
    ).reshape(-1)
    pages_per_seq = padded_pages_per_seq
  kv_spec = pl.BlockSpec(
      (batch, num_kv_heads, head_dim), lambda *_: (0, 0, 0)
  )
  hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
  aliased_out_shapes = (
      jax.ShapeDtypeStruct(q.shape, q.dtype),
      *(jax.ShapeDtypeStruct(cache.shape, cache.dtype)
        for cache in kernel_caches),
  )
  if return_lse:
    lse_shape = (batch, num_query_heads, NUM_LANES)
    out_shapes = (
        *aliased_out_shapes, jax.ShapeDtypeStruct(lse_shape, jnp.float32)
    )
    out_specs = (hbm_spec,) * len(aliased_out_shapes) + (
        pl.BlockSpec(lse_shape, lambda *_: (0, 0, 0)),
    )
  else:
    out_shapes = aliased_out_shapes
    out_specs = (hbm_spec,) * len(aliased_out_shapes)
  # Note (david): the pair-packed load bitcasts refs, which only the TPU build
  # supports; interpret mode reads the staged heads as bf16.
  is_bitcast_load = not interpret and not is_cache_head_major
  staged_kv_heads = cache_shape[2] if merged_cache else num_kv_heads
  word_rows = kv_word_rows(k_cache.dtype)

  def _staging_shape(num_tokens, word_stride):
    if is_cache_head_major:
      return (num_kv_stages, num_kv_heads, num_tokens, head_dim)
    elif is_bitcast_load:
      return (num_kv_stages, num_tokens, word_stride, word_rows, head_dim)
    else:
      return (num_kv_stages, num_tokens, staged_kv_heads, head_dim)

  # Note (david): only the staged blocks are read with strided word loads,
  # so only they take the bank pad; the write tiles stay dense for their
  # one-DMA write-back. A write tile is one cache tile of tokens: 16 on the
  # head-major token axis (the bf16 sublane tile v7x uses), 8 token-major
  # (see cache_tile_rows in the kernel).
  token_words = staged_kv_heads // word_rows
  cache_buffer_shape = _staging_shape(
      block_kv, bank_safe_word_stride(token_words))
  write_tile_rows = BF16_TILE_ROWS if is_cache_head_major else NUM_SUBLANES
  write_buffer_shape = _staging_shape(write_tile_rows, token_words)
  # Note (david): a merged build never touches the V staging, so it shrinks to
  # the write-tile size instead of a whole KV block.
  value_buffer_shape = (
      write_buffer_shape if merged_cache else cache_buffer_shape
  )
  io_buffer_shape = (NUM_IO_STAGES, num_query_heads, head_dim)
  state_shape = (num_query_heads, NUM_LANES)
  scratch_shapes = (
      pltpu.VMEM(cache_buffer_shape, k_cache.dtype),
      pltpu.VMEM(value_buffer_shape, k_cache.dtype),
      pltpu.VMEM(write_buffer_shape, k_cache.dtype),
      pltpu.VMEM(write_buffer_shape, k_cache.dtype),
      pltpu.VMEM(io_buffer_shape, q.dtype),
      pltpu.VMEM(io_buffer_shape, q.dtype),
      pltpu.VMEM(state_shape, jnp.float32),
      pltpu.VMEM(state_shape, jnp.float32),
      pltpu.VMEM((num_query_heads, head_dim), q.dtype),
      pltpu.SemaphoreType.DMA((NUM_SEMS, num_kv_stages, num_kv_heads)),
  )
  if num_active is None:
    num_active_array = jnp.full((1,), batch, jnp.int32)
  else:
    num_active_array = jnp.asarray(num_active, jnp.int32).reshape(1)
  v_scale_array = jnp.asarray(v_scale, jnp.float32).reshape(1)
  if page_size is None:
    scalar_prefetches = (
        cache_seqlens, cache_batch_idx, num_active_array, v_scale_array
    )
  else:
    scalar_prefetches = (
        cache_seqlens, cache_batch_idx, num_active_array, v_scale_array,
        block_table,
    )
  num_prefetch = len(scalar_prefetches)
  if merged_cache:
    # Note (david): the new K and V are interleaved into the cache's head order
    # here, a (batch, staged_kv_heads, head_dim) op far below one cache read,
    # so the kernel appends a ready token row and writes each appended tile
    # back with one contiguous DMA.
    new_kv_shape = (batch, staged_kv_heads, head_dim)
    head_pad = ((0, 0), (0, staged_kv_heads // 2 - num_kv_heads), (0, 0))
    word_heads_shape = (batch, -1, kv_heads_per_word(k_cache.dtype), head_dim)
    new_kv = jnp.stack(
        [jnp.pad(new_rows, head_pad).reshape(word_heads_shape)
         for new_rows in (k, v)],
        axis=2,
    ).reshape(new_kv_shape)
    dense_inputs = (q, new_kv)
    new_token_specs = (pl.BlockSpec(new_kv_shape, lambda *_: (0, 0, 0)),)
  else:
    dense_inputs = (q, k, v)
    new_token_specs = (kv_spec, kv_spec)
  # Note (david): q aliases out (same shape and dtype) and each cache aliases
  # its own output, so the append lands in place.
  input_output_aliases = {num_prefetch: 0} | {
      num_prefetch + len(dense_inputs) + index: 1 + index
      for index in range(len(kernel_caches))
  }
  kernel = functools.partial(
      kvcache_kernel,
      batch=batch,
      num_query_heads=num_query_heads,
      num_kv_heads=num_kv_heads,
      head_dim=head_dim,
      block_kv=block_kv,
      block_kv_compute=block_kv_compute,
      num_kv_stages=num_kv_stages,
      is_cache_head_major=is_cache_head_major,
      is_bitcast_load=is_bitcast_load,
      return_lse=return_lse,
      has_new=has_new,
      left=left,
      query_tokens=query_tokens,
      causal=causal,
      log2_softcap=softcap * LOG2E,
      guard_threshold=guard_threshold,
      page_size=page_size,
      pages_per_seq=pages_per_seq,
      is_merged_cache=merged_cache,
  )
  call = pl.pallas_call(
      kernel,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=num_prefetch,
          grid=(1,),
          in_specs=(hbm_spec, *new_token_specs)
          + (hbm_spec,) * len(kernel_caches),
          out_specs=out_specs,
          scratch_shapes=scratch_shapes,
      ),
      out_shape=out_shapes,
      input_output_aliases=input_output_aliases,
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("arbitrary",),
          vmem_limit_bytes=vmem_limit_bytes(),
          disable_bounds_checks=True,
          disable_semaphore_checks=True,
      ),
      name="flash_attn_with_kvcache",
      interpret=pltpu.InterpretParams() if interpret else False,
  )
  kernel_outputs = call(*scalar_prefetches, *dense_inputs, *kernel_caches)
  num_caches = len(cache_operands)
  if is_token_major_mqa:
    caches_out = tuple(
        out_cache.reshape(operand.shape)
        for out_cache, operand in zip(
            kernel_outputs[1:1 + num_caches], cache_operands
        )
    )
  else:
    caches_out = tuple(kernel_outputs[1:1 + num_caches])
  if is_window:
    out = (
        kernel_outputs[0]
        .reshape(batch, num_kv_heads, query_tokens, group, head_dim)
        .transpose(0, 2, 1, 3, 4)
        .reshape(batch, query_tokens, window_heads, head_dim)
    )
    lse_out = tuple(
        lse.reshape(batch, num_kv_heads, query_tokens, group, NUM_LANES)
        .transpose(0, 2, 1, 3, 4)
        .reshape(batch, query_tokens, window_heads, NUM_LANES)
        for lse in kernel_outputs[1 + num_caches:]
    )
  else:
    out = kernel_outputs[0]
    lse_out = tuple(kernel_outputs[1 + num_caches:])
  return (out, *caches_out, *lse_out)
