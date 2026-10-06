"""Single-token KV-cache decode: one persistent Pallas call appends every row's
new token and attends its whole cache."""

import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ..tuned_block_sizes import KVCacheConfig, get_tuned_kvcache_config
from .block_sizes import (
  BF16_BYTES,
  F32_BYTES,
  NUM_LANES,
  NUM_SUBLANES,
  STAGES,
  next_pow2,
  vmem_limit_bytes,
)
from .copy_utils import MIN_BLOCK_KV, load_kv_fragment
from .fwd_pipeline import overflow_guard_threshold

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
WIDE_HEAD_DIM = 256
WIDE_HEAD_MIN_KV_HEADS = 8
WIDE_HEAD_MAX_BLOCK_KV = 1024


def static_anchor_update(
    scores: jax.Array,
    row_max_prev: jax.Array,
    row_sum_prev: jax.Array,
    *,
    is_first: bool | jax.Array,
    guard_threshold: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Online-softmax step in log2 units over (rows, 1) running state.

  Returns (probabilities, row_max, row_sum, rescale).
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
  probabilities = jnp.exp2(scores - row_max)
  fragment_sum = probabilities.sum(axis=-1, keepdims=True)
  rescale = jnp.where(is_first, 1.0, jnp.exp2(row_max_prev - row_max))
  row_sum = jnp.where(
      is_first,
      fragment_sum,
      rescale * row_sum_prev + fragment_sum,
  )
  return probabilities, row_max, row_sum, rescale


def kvcache_kernel(
    cache_seqlens_ref: jax.Array,
    cache_batch_idx_ref: jax.Array,
    num_active_ref: jax.Array,
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
    guard_threshold: float,
    page_size: int | None,
    pages_per_seq: int | None,
    is_merged_cache: bool,
    is_kv_pair_tile: bool,
) -> None:
  remaining_refs = list(refs)
  block_table_ref = remaining_refs.pop(0) if page_size is not None else None
  num_active = num_active_ref[0]
  if is_merged_cache:
    (
        q_ref,
        k_ref,
        v_ref,
        kv_cache_ref,
        out_ref,
        kv_out_ref,
        *remaining_refs,
    ) = remaining_refs
    k_cache_ref = v_cache_ref = kv_cache_ref
    k_out_ref = v_out_ref = kv_out_ref
    # Note (david): a K/V pair-tile build copies whole (..., 2, head_dim) pair
    # tiles, so it addresses the operand without picking a K or V part.
    k_plane, v_plane = (None, None) if is_kv_pair_tile else (0, 1)
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
    k_plane = v_plane = None
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

  def _cache_view(ref, plane, index):
    # Note (david): a merged operand holds the K heads then the V heads on its
    # head axis, so a part is a head-block view of the row.
    if plane is None:
      return ref.at[index]
    else:
      row = ref.at[index[0], :, pl.ds(plane * num_kv_heads, num_kv_heads), :]
      return row.at[index[1:]]

  query_heads_per_kv_head = num_query_heads // num_kv_heads
  staged_kv_heads = 2 if is_kv_pair_tile else num_kv_heads
  num_kv_compute_fragments = block_kv // block_kv_compute
  # Note (david): a bitcast head-pair tile interleaves its two heads per token,
  # so both heads' query groups ride one MXU pass with the off-parity columns
  # masked. That is free while the pair's query rows fit one 8-sublane f32
  # tile; past it the masked half doubles the softmax work, so larger groups
  # unpack one head per fragment.
  if (
      is_bitcast_load
      and 2 * query_heads_per_kv_head <= NUM_SUBLANES
      and not is_kv_pair_tile
  ):
    kv_heads_per_fragment = 2
  else:
    kv_heads_per_fragment = 1
  num_head_groups = num_kv_heads // kv_heads_per_fragment
  rows_per_fragment = query_heads_per_kv_head * kv_heads_per_fragment
  fragment_keys = kv_heads_per_fragment * block_kv_compute
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
  kv_fragment = functools.partial(
      load_kv_fragment,
      block_kv_compute=block_kv_compute,
      kv_heads_per_fragment=kv_heads_per_fragment,
      is_kv_pair_tile=is_kv_pair_tile,
      is_cache_head_major=is_cache_head_major,
      is_bitcast_load=is_bitcast_load,
  )

  def _cache_copy_size(cache_len, block_index):
    copy_limit = (
        (cache_len + int(has_new) + (NUM_SUBLANES - 1))
        // NUM_SUBLANES * NUM_SUBLANES
    )
    copy_size = jnp.clip(
        copy_limit - block_index * block_kv, 0, block_kv
    )
    return pl.multiple_of(copy_size, NUM_SUBLANES)

  def _cache_tile(ref, plane, hbm_row, token_start, num_tokens):
    if is_cache_head_major:
      index = (
          hbm_row, slice(None), pl.ds(token_start, num_tokens), slice(None)
      )
    else:
      index = (
          hbm_row, pl.ds(token_start, num_tokens), slice(None), slice(None)
      )
    return _cache_view(ref, plane, index)

  def _buffer_tile(buffers, slot, token_start, num_tokens):
    if is_cache_head_major:
      return buffers.at[slot, :, pl.ds(token_start, num_tokens), :]
    elif is_bitcast_load:
      return buffers.at[slot].reshape(
          block_kv, staged_kv_heads, head_dim
      ).at[pl.ds(token_start, num_tokens), :, :]
    else:
      return buffers.at[slot, pl.ds(token_start, num_tokens), :, :]

  # Note (david): a K/V pair tile carries V next to K, so a pair-tile build
  # stages, appends and writes back through the K buffers only.
  k_load = (k_cache_ref, k_plane, k_buffers, SEM_K)
  k_write = (k_write_buffers, k_out_ref, k_plane, SEM_K_WRITE)
  if is_kv_pair_tile:
    load_parts = (k_load,)
    write_parts = (k_write,)
    value_buffers = k_buffers
  else:
    load_parts = (k_load, (v_cache_ref, v_plane, v_buffers, SEM_V))
    write_parts = (
        k_write, (v_write_buffers, v_out_ref, v_plane, SEM_V_WRITE)
    )
    value_buffers = v_buffers

  def _load_segments(cache_row, block_index, load_size):
    # Note (david): a paged block is one piece per page, each a full page
    # except the clipped last one (RPA sizing; zero-length tails are issued
    # as-is). block_index * pages_per_block + page never leaves the row's table
    # entries because capacity % block_kv == 0 and the caller keeps
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
                  NUM_SUBLANES,
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
        for cache_ref, plane, buffers, sem_index in load_parts:
          pltpu.make_async_copy(
              _cache_tile(cache_ref, plane, hbm_row, hbm_start, num_tokens),
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
      for _, _, buffers, sem_index in load_parts:
        tile = _buffer_tile(buffers, slot, 0, load_size)
        pltpu.make_async_copy(tile, tile, sems.at[sem_index, slot, 0]).wait()

  def _cache_write_copies(cache_row, cache_len, slot):
    if page_size is None:
      hbm_row = cache_row
      tile_start = pl.multiple_of(
          cache_len // NUM_SUBLANES * NUM_SUBLANES, NUM_SUBLANES
      )
    else:
      # Note (david): page_size is a multiple of 128, so an 8-token tile at an
      # 8-aligned offset never straddles a page and the paged append stays a
      # single write.
      hbm_row = block_table_ref[
          cache_row * pages_per_seq + cache_len // page_size
      ]
      tile_start = pl.multiple_of(
          cache_len % page_size // NUM_SUBLANES * NUM_SUBLANES, NUM_SUBLANES
      )
    if is_cache_head_major:
      # Note (david): one copy per head, since the head axis of a cache row is
      # not contiguous.
      return [
          pltpu.make_async_copy(
              buffers.at[slot, pl.ds(head_offset, 1), :, :],
              _cache_view(
                  out_ref_hbm,
                  plane,
                  (
                      hbm_row,
                      pl.ds(head_offset, 1),
                      pl.ds(tile_start, NUM_SUBLANES),
                      slice(None),
                  ),
              ),
              sems.at[sem_index, slot, head_offset],
          )
          for head_offset in range(num_kv_heads)
          for buffers, out_ref_hbm, plane, sem_index in write_parts
      ]
    else:
      return [
          pltpu.make_async_copy(
              buffers.at[slot].reshape(NUM_SUBLANES, staged_kv_heads, head_dim)
              if is_bitcast_load
              else buffers.at[slot],
              _cache_view(
                  out_ref_hbm,
                  plane,
                  (
                      hbm_row,
                      pl.ds(tile_start, NUM_SUBLANES),
                      slice(None),
                      slice(None),
                  ),
              ),
              sems.at[sem_index, slot, 0],
          )
          for buffers, out_ref_hbm, plane, sem_index in write_parts
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
  value_words = (
      value_buffers.bitcast(jnp.uint32) if is_bitcast_load else value_buffers
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
        block_index,
        slot,
        head_group,
        kv_compute_index,
        is_valid,
        fragment_has_valid_key,
        row_max_carry,
        row_sum_carry,
    ):
      # Note (david): the query heads of one head group are consecutive rows of
      # q, so one (rows_per_fragment, head_dim) tile rides a single K load and
      # MXU pass.
      group_start = head_group * rows_per_fragment
      group_slice = pl.ds(group_start, rows_per_fragment)
      query = queries[group_start : group_start + rows_per_fragment, :]
      key = kv_fragment(k_buffers, slot, head_group, kv_compute_index, 0)
      scores = jnp.where(
          is_valid,
          jnp.dot(query, key.T, preferred_element_type=jnp.float32),
          -jnp.inf,
      )

      if kv_compute_index == 0:
        row_max_prev = row_max_ref[group_slice, 0:1]
        row_sum_prev = row_sum_ref[group_slice, 0:1]
      else:
        row_max_prev = row_max_carry
        row_sum_prev = row_sum_carry
      if left is None:
        probabilities, row_max, row_sum, rescale = static_anchor_update(
            scores,
            row_max_prev,
            row_sum_prev,
            is_first=(
                block_index == first_block if kv_compute_index == 0 else False
            ),
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
      if kv_compute_index == num_kv_compute_fragments - 1:
        row_max_ref[group_slice, :] = jnp.broadcast_to(
            row_max, (rows_per_fragment, NUM_LANES)
        )
        row_sum_ref[group_slice, :] = jnp.broadcast_to(
            row_sum, (rows_per_fragment, NUM_LANES)
        )
      return probabilities, rescale, row_max, row_sum

    def _pv(slot, head_group, kv_compute_index, probabilities, rescale):
      group_slice = pl.ds(head_group * rows_per_fragment, rows_per_fragment)
      value = kv_fragment(value_buffers, slot, head_group, kv_compute_index, 1)
      weighted_values = jnp.dot(
          probabilities.astype(jnp.bfloat16),
          value,
          preferred_element_type=jnp.float32,
      )
      # Note (david): the accumulator is stored in q's dtype while rescale and
      # weighted_values are f32; the explicit upcast keeps the kernel tracing
      # under jax_numpy_dtype_promotion=strict.
      accumulator_ref[group_slice, :] = (
          accumulator_ref[group_slice, :].astype(jnp.float32) * rescale
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
          local_index = cache_len - block_index * block_kv
          local_tile_start = pl.multiple_of(
              local_index // NUM_SUBLANES * NUM_SUBLANES, NUM_SUBLANES
          )
          new_key = k_ref[local_batch_index, :, :]
          new_value = v_ref[local_batch_index, :, :]
          token_row = jnp.arange(NUM_SUBLANES, dtype=jnp.int32)
          tile_tokens = pl.ds(local_tile_start, NUM_SUBLANES)
          if is_cache_head_major:
            tile_index = (slot, slice(None), tile_tokens, slice(None))
            token_rows = token_row[None, :, None]
            expand = lambda token: token[:, None, :]
          elif is_bitcast_load:
            tile_index = (slot, tile_tokens, slice(None), slice(None),
                          slice(None))
            token_rows = token_row[:, None, None, None]
            expand = lambda token: token.reshape(
                staged_kv_heads // 2, 2, head_dim
            )[None]
          else:
            tile_index = (slot, tile_tokens, slice(None), slice(None))
            token_rows = token_row[:, None, None]
            expand = lambda token: token[None]
          is_new_row = token_rows == local_index - local_tile_start
          if is_kv_pair_tile:
            new_tokens = (jnp.concatenate([new_key, new_value], axis=0),)
          else:
            new_tokens = (new_key, new_value)
          updated_tiles = []
          for (_, _, buffers, _), new_token in zip(load_parts, new_tokens):
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

          for (write_buffers, _, _, _), updated_tile in zip(
              write_parts, updated_tiles
          ):
            write_buffers[write_slot] = updated_tile
          for copy in _cache_write_copies(cache_row, cache_len, write_slot):
            copy.start()

      valid_masks = []
      fragment_has_valid_keys = []
      for kv_compute_index in range(num_kv_compute_fragments):
        key_start = (
            block_index * block_kv
            + kv_compute_index * block_kv_compute
        )
        # Note (david): build the mask at full fragment shape; a (1, N) mask
        # needs a sublane-broadcast relayout inside every fragment's select,
        # which measured ~40% of MQA decode time.
        key_position = key_start + lax.broadcasted_iota(
            jnp.int32, (rows_per_fragment, fragment_keys), 1
        ) // kv_heads_per_fragment
        if left is None:
          is_in_window = key_position < total_len
        else:
          is_in_window = (
              (key_position < total_len) & (key_position >= first_key)
          )
        if parity_mask is None:
          is_valid = is_in_window
        else:
          is_valid = is_in_window & parity_mask
        valid_masks.append(is_valid)
        fragment_has_valid_keys.append(
            jnp.logical_and(
                key_start < total_len,
                key_start + block_kv_compute > first_key,
            )
        )

      # Note (david): software pipeline; each fragment's scores are issued one
      # step ahead of its own pv, so the MXU has the next matmul queued while
      # pv runs. pending_fragment is (head_group, kv_compute_index,
      # probabilities, rescale, row_max, row_sum).
      pending_fragment = None
      for fragment_index in range(num_head_groups * num_kv_compute_fragments):
        head_group, kv_compute_index = divmod(
            fragment_index, num_kv_compute_fragments
        )
        current_fragment = _score_and_softmax(
            block_index,
            slot,
            head_group,
            kv_compute_index,
            valid_masks[kv_compute_index],
            fragment_has_valid_keys[kv_compute_index],
            *(
                pending_fragment[4:]
                if pending_fragment is not None
                else (None, None)
            ),
        )
        if pending_fragment is not None:
          _pv(slot, *pending_fragment[:4])
        pending_fragment = (head_group, kv_compute_index, *current_fragment)
      _pv(slot, *pending_fragment[:4])
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

    row_sum = row_sum_ref[:, 0:1]
    inverse_row_sum = jnp.where(
        row_sum > 0, pl.reciprocal(row_sum, approx=True), 0.0
    )
    out_buffers[output_slot] = (
        accumulator_ref[...].astype(jnp.float32) * inverse_row_sum
    ).astype(q_ref.dtype)
    if return_lse:
      lse_log2 = row_max_ref[:, 0:1] + jnp.log2(row_sum)
      lse = jnp.where(
          row_sum > 0, lse_log2 * math.log(2.0), -jnp.inf
      ).astype(jnp.float32)
      lse_ref[pl.ds(local_batch_index, 1), :, :] = jnp.broadcast_to(
          lse[None, :, :], (1, num_query_heads, NUM_LANES)
      )
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


def resolve_block_kv(
    capacity: int, num_kv_heads: int, head_dim: int, granule: int = MIN_BLOCK_KV
) -> int:
  """Largest VMEM-sized multiple of granule that divides capacity.

  granule is the page size for paged caches (a block is whole pages) and
  MIN_BLOCK_KV otherwise; capacity is a multiple of it either way.
  """
  packed_kv_heads = next_pow2(2 * num_kv_heads)
  budget_block_kv = min(
      capacity,
      STAGED_KV_ELEMENT_BUDGET // head_dim // packed_kv_heads,
  )
  if head_dim == WIDE_HEAD_DIM and num_kv_heads >= WIDE_HEAD_MIN_KV_HEADS:
    target_block_kv = min(budget_block_kv, WIDE_HEAD_MAX_BLOCK_KV)
  else:
    target_block_kv = budget_block_kv
  aligned_target = max(target_block_kv // granule * granule, granule)
  return next(
      block for block in range(aligned_target, 0, -granule)
      if capacity % block == 0
  )


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
  # NUM_SUBLANES rows per stage are the cache write tile.
  kv_staging_bytes = (
      2 * num_kv_stages * (block_kv + NUM_SUBLANES) * num_kv_heads * head_dim
      * BF16_BYTES)
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
) -> tuple[jax.Array, ...]:
  """One-token cache append and decode attention for every row in one call.

  q: (batch, num_query_heads, head_dim), already scaled into log2 units.
  k_cache/v_cache: (rows, capacity, num_kv_heads, head_dim), or head-major
  (rows, num_kv_heads, capacity, head_dim) with cache_head_major. With
  block_table, the flat (batch * pages_per_seq,) int32 page table, the caches
  are (num_pages, page_size, ...) pools and the per-row capacity is
  pages_per_seq * page_size.

  merged_cache (v_cache=None, token-major) makes k_cache one
  (rows, capacity, 2 * num_kv_heads, head_dim) array holding a token's K heads
  followed by its V heads, and returns (out, kv_cache[, lse]) instead of
  (out, k_cache, v_cache[, lse]); lse is (batch, num_query_heads, 128) float32
  with each value broadcast over lanes.

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
  batch, num_query_heads, head_dim = q.shape
  if merged_cache:
    row_tokens, num_kv_heads = cache_shape[1], cache_shape[2] // 2
    if num_kv_heads != 1 and num_kv_heads % 2:
      raise ValueError(
          "a merged cache needs one KV head per shard or an even number (the"
          " packed load carries two bf16 heads per u32 lane); got"
          f" num_kv_heads={num_kv_heads}."
      )
    # Note (david): past one KV head, K and V are head-block DMAs out of the
    # merged row, which need a tile-aligned head offset and count. Measured on
    # v6e (jax 0.11.0): 6 or 12 KV heads fail to compile, and 2 or 4 at
    # head_dim 64 read and append the wrong bytes without an error.
    is_head_block_aligned = num_kv_heads % NUM_SUBLANES == 0 or (
        head_dim >= NUM_LANES and NUM_SUBLANES % num_kv_heads == 0
    )
    if num_kv_heads > 1 and not is_head_block_aligned:
      raise ValueError(
          "a merged cache needs 1, 2, 4 or a multiple of 8 KV heads per shard"
          " (1 or a multiple of 8 at head_dim 64), since its K and V head"
          " blocks are DMA'd at tile-aligned offsets; got"
          f" num_kv_heads={num_kv_heads}, head_dim={head_dim}."
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
  # Note (david): one merged KV head leaves a (2, head_dim) head axis whose
  # one-head slice is not tile-aligned, so the build stages the whole K/V pair
  # tile and reads K from lane 0 and V from lane 1 of its u32 words. That costs
  # 1.4-1.5x the head-major decode of an unmerged K/V cache (6 query heads,
  # head_dim 256, v6e), yet splitting the pair into dense K/V buffers per
  # block (1.45-2.25x), folding the heads into the row width (6.5-12x) and
  # scoring the pair interleaved (1.46-2.02x) all measured worse.
  is_kv_pair_tile = merged_cache and num_kv_heads == 1
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
  if capacity % MIN_BLOCK_KV:
    raise ValueError(
        f"capacity must be a multiple of {MIN_BLOCK_KV}; got {capacity}."
    )
  tuned_config = get_tuned_kvcache_config(
      dtype=q.dtype,
      num_query_heads=num_query_heads,
      num_kv_heads=num_kv_heads,
      head_dim=head_dim,
      capacity=capacity,
      batch=batch,
      append=has_new,
      sliding_window=left,
      return_lse=return_lse,
  )
  # Note (david): a paged block is whole pages, so a tuned block_kv that does
  # not tile the page cannot be used as-is.
  if tuned_config is not None and (
      page_size is None or tuned_config[0] % page_size == 0
  ):
    block_kv, block_kv_compute, num_kv_stages = validate_kvcache_config(
        tuned_config,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        capacity=capacity,
    )
  else:
    block_kv = resolve_block_kv(
        capacity, num_kv_heads, head_dim,
        granule=MIN_BLOCK_KV if page_size is None else page_size,
    )
    block_kv_compute = resolve_block_kv_compute(block_kv)
    num_kv_stages = DEFAULT_NUM_KV_STAGES
  guard_threshold = overflow_guard_threshold(capacity)
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
  staged_kv_heads = 2 if is_kv_pair_tile else num_kv_heads

  def _staging_shape(num_tokens):
    if is_cache_head_major:
      return (num_kv_stages, num_kv_heads, num_tokens, head_dim)
    elif is_bitcast_load:
      return (num_kv_stages, num_tokens, staged_kv_heads // 2, 2, head_dim)
    else:
      return (num_kv_stages, num_tokens, staged_kv_heads, head_dim)

  cache_buffer_shape = _staging_shape(block_kv)
  write_buffer_shape = _staging_shape(NUM_SUBLANES)
  # Note (david): a pair-tile build never touches the V staging, so it shrinks
  # to the write-tile size instead of a whole KV block.
  value_buffer_shape = (
      write_buffer_shape if is_kv_pair_tile else cache_buffer_shape
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
  if page_size is None:
    scalar_prefetches = (cache_seqlens, cache_batch_idx, num_active_array)
  else:
    scalar_prefetches = (
        cache_seqlens, cache_batch_idx, num_active_array, block_table
    )
  num_prefetch = len(scalar_prefetches)
  dense_inputs = (q, k, v)
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
      guard_threshold=guard_threshold,
      page_size=page_size,
      pages_per_seq=pages_per_seq,
      is_merged_cache=merged_cache,
      is_kv_pair_tile=is_kv_pair_tile,
  )
  call = pl.pallas_call(
      kernel,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=num_prefetch,
          grid=(1,),
          in_specs=(hbm_spec, kv_spec, kv_spec)
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
  if is_token_major_mqa:
    num_caches = len(cache_operands)
    restored_caches = tuple(
        out_cache.reshape(operand.shape)
        for out_cache, operand in zip(
            kernel_outputs[1:1 + num_caches], cache_operands
        )
    )
    return (
        kernel_outputs[0], *restored_caches, *kernel_outputs[1 + num_caches:]
    )
  else:
    return kernel_outputs
