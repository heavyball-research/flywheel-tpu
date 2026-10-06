"""Cached-prefix extend attention over a paged KV cache, kv-outer.

The q-outer varlen pipeline re-streams the visible KV prefix once per q block;
this kernel loads a request's new-token queries once per static q chunk and
walks its KV prefix once per chunk, like the decode kernel.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .block_sizes import (
    BF16_BYTES,
    F32_BYTES,
    MAX_VMEM_LIMIT_BYTES,
    NUM_LANES,
    NUM_SUBLANES,
    round_up,
    vmem_limit_bytes,
)
from .flash_fwd_kvcache import (
    MIN_BLOCK_KV,
    SUPPORTED_HEAD_DIMS,
    load_kv_fragment,
    static_anchor_update,
)
from .fwd_pipeline import overflow_guard_threshold

TilingSpec = tuple[int, int, int, int]

SEM_K = 0
SEM_V = 1
SEM_Q = 2
SEM_OUT = 3
SEM_LSE = 4
NUM_SEMS = 5
NUM_KV_STAGES = 2
# Note (david): the wrapper allocates and the kernel unpacks these per tiling
# in one order: K and V staging, token-major q load and output stage,
# head-major q, accumulator and row state, then the LSE stage when returned.
# A merged cache has one staging buffer, whole K/V rows, in place of the two.
NUM_TILING_SCRATCH = 7
BLOCK_KV_COMPUTE_CANDIDATES = (1024, 512, 256, 128)
# Note (david): each request runs one monomorphic chunk loop per tiling whose
# trip count selects it, so no tiling sits behind a runtime branch. Every
# tiling owns all of its staging: a short pass working in a window of a long
# tiling's stages measured 3-4x slower (a branch inside a shared block loop,
# 2.2x). Long chunks favor many rows against small per-head KV fragments,
# short suffixes few rows against large pair-packed ones. Entries are
# (rows, block_kv, block_kv_compute, kv_heads_per_fragment), rows descending.
def default_tilings(query_heads_per_kv_head: int) -> tuple[TilingSpec, ...]:
  """The tilings resolve_tilings starts from, for the core's VMEM."""
  if vmem_limit_bytes() >= MAX_VMEM_LIMIT_BYTES:
    return (
        (384, 512, 512, 1),
        (128, 512, 512, 1),
        (32, 1024, 1024, 2),
        (8, 1024, 1024, 2),
    )
  # On a core with less VMEM than v6e's, every tiling's K/V staging is a block
  # of every head whatever its rows, so the four above squeeze each other down
  # to one-page blocks or out of VMEM. One tiling per build instead; a
  # request's last chunk runs it padded. A score fragment holds rows times the
  # KV head's query heads, so groups of 4 or more query heads take fewer rows
  # and longer key blocks. 1K chunk at 8K context, 32 query heads of 256, a
  # merged cache, on v7x, ms: 4 KV heads 0.619 (128, 1024, 512); 8 KV heads
  # 0.888; 32 KV heads 1.978 (256, 256, 256); 16 KV heads 1.357, where
  # (256, 512, 256) took 1.193 but sits near the VMEM limit, which the estimate
  # undercounts by about 12 MiB at 256 rows.
  # TODO: tune across head dims, chunk sizes and short requests, and close the
  # VMEM estimate gap.
  if query_heads_per_kv_head >= 4:
    return ((128, 1024, 512, 1),)
  return ((256, 256, 256, 1),)


def extend_kernel(
    cu_seqlens_ref: jax.Array,
    total_lengths_ref: jax.Array,
    block_table_ref: jax.Array,
    num_active_ref: jax.Array,
    q_ref: jax.Array,
    *refs: jax.Array,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    tilings: tuple[TilingSpec, ...],
    num_kv_stages: int,
    is_cache_head_major: bool,
    is_bitcast_load: bool,
    return_lse: bool,
    causal: bool,
    should_zero_padding: bool,
    q_scale: float,
    guard_threshold: float,
    page_size: int,
    pages_per_seq: int,
    is_merged_cache: bool,
    q_rows: int,
    lse_width: int,
) -> None:
  remaining_refs = list(refs)
  if is_merged_cache:
    # Note (david): merged caches stage whole K/V head rows to preserve the
    # serving-step cache layout, then select each part's heads in VMEM.
    k_cache_ref = v_cache_ref = remaining_refs.pop(0)
    # Each row is staged once: K and V loads read the same buffer.
    staging_parts = ((k_cache_ref, 0, SEM_K),)
  else:
    k_cache_ref, v_cache_ref = remaining_refs.pop(0), remaining_refs.pop(0)
    staging_parts = ((k_cache_ref, 0, SEM_K), (v_cache_ref, 1, SEM_V))
  out_ref = remaining_refs.pop(0)
  lse_ref = remaining_refs.pop(0) if return_lse else None
  boundary_out_ref = remaining_refs.pop(0)
  boundary_lse_ref = remaining_refs.pop(0) if return_lse else None
  num_tilings = len(tilings)
  num_scratch_per_tiling = (
      NUM_TILING_SCRATCH + int(return_lse) - int(is_merged_cache)
  )
  tiling_scratch = [
      # The merged cache's one staging buffer serves as both K and V staging.
      (remaining_refs[first_ref],) * int(is_merged_cache)
      + tuple(remaining_refs[first_ref:first_ref + num_scratch_per_tiling])
      + (() if return_lse else (None,))
      for first_ref in range(
          0, num_scratch_per_tiling * num_tilings, num_scratch_per_tiling
      )
  ]
  guard_smem, sems = remaining_refs[num_scratch_per_tiling * num_tilings:]
  guard_detect_bound = math.exp2(guard_threshold)

  num_active = num_active_ref[0]
  max_chunk = tilings[0][0]
  query_heads_per_kv_head = num_query_heads // num_kv_heads
  staged_heads = 2 * num_kv_heads if is_merged_cache else num_kv_heads

  def _cache_tile(ref, page, num_tokens):
    if is_cache_head_major:
      return ref.at[page, :, pl.ds(0, num_tokens), :]
    else:
      return ref.at[page, pl.ds(0, num_tokens), :, :]

  def _buffer_tile(tiling, buffers, slot, token_start, num_tokens):
    block_kv = tilings[tiling][1]
    if is_cache_head_major:
      return buffers.at[slot, :, pl.ds(token_start, num_tokens), :]
    elif is_bitcast_load:
      return buffers.at[slot].reshape(
          block_kv, staged_heads, head_dim
      ).at[pl.ds(token_start, num_tokens), :, :]
    else:
      return buffers.at[slot, pl.ds(token_start, num_tokens), :, :]

  def _cache_copy_size(tiling, total_len, block_index):
    block_kv = tilings[tiling][1]
    copy_limit = (
        (total_len + (NUM_SUBLANES - 1)) // NUM_SUBLANES * NUM_SUBLANES
    )
    return pl.multiple_of(
        jnp.clip(copy_limit - block_index * block_kv, 0, block_kv),
        NUM_SUBLANES,
    )

  def _start_kv(tiling, sequence, total_len, block_index, slot):
    block_kv = tilings[tiling][1]
    k_buffers, v_buffers = tiling_scratch[tiling][:2]
    load_size = _cache_copy_size(tiling, total_len, block_index)
    pages_per_block = block_kv // page_size
    table_base = sequence * pages_per_seq + block_index * pages_per_block

    @pl.when(load_size > 0)
    def _start():
      for page in range(pages_per_block):
        num_tokens = pl.multiple_of(
            jnp.clip(load_size - page * page_size, 0, page_size), NUM_SUBLANES
        )
        physical_page = block_table_ref[table_base + page]
        for cache_ref, part, sem_index in staging_parts:
          pltpu.make_async_copy(
              _cache_tile(cache_ref, physical_page, num_tokens),
              _buffer_tile(
                  tiling, (k_buffers, v_buffers)[part], slot,
                  page * page_size, num_tokens,
              ),
              sems.at[sem_index, slot, tiling],
          ).start()

  def _wait_kv(tiling, total_len, block_index, slot):
    k_buffers, v_buffers = tiling_scratch[tiling][:2]
    load_size = _cache_copy_size(tiling, total_len, block_index)

    # Note (david): a DMA semaphore counts bytes, so one wait sized to the
    # staged window covers every page copy of the block; src == dst because
    # only the size matters.
    @pl.when(load_size > 0)
    def _wait():
      for _, part, sem_index in staging_parts:
        tile = _buffer_tile(
            tiling, (k_buffers, v_buffers)[part], slot, 0, load_size
        )
        pltpu.make_async_copy(
            tile, tile, sems.at[sem_index, slot, tiling]
        ).wait()

  def _zero_value_tail(tiling, total_len, block_index, slot):
    # Note (david): cache slots past a request's length are uninitialized and
    # possibly NaN. Their keys are masked to -inf, but a zero probability times
    # a NaN value is still NaN in the pv matmul, so the staged V rows
    # [total_len, align8(total_len)) of the block's last tile are zeroed.
    block_kv = tilings[tiling][1]
    v_buffers = tiling_scratch[tiling][1]
    tail_start = total_len // NUM_SUBLANES * NUM_SUBLANES

    @pl.when(
        jnp.logical_and(
            total_len % NUM_SUBLANES != 0,
            block_index == total_len // block_kv,
        )
    )
    def _zero():
      local_start = pl.multiple_of(
          tail_start - block_index * block_kv, NUM_SUBLANES
      )
      token_row = jnp.arange(NUM_SUBLANES, dtype=jnp.int32)
      tile_tokens = pl.ds(local_start, NUM_SUBLANES)
      if is_cache_head_major:
        tile_index = (slot, slice(None), tile_tokens, slice(None))
        token_rows = token_row[None, :, None]
      elif is_bitcast_load:
        tile_index = (slot, tile_tokens, slice(None), slice(None),
                      slice(None))
        token_rows = token_row[:, None, None, None]
      else:
        tile_index = (slot, tile_tokens, slice(None), slice(None))
        token_rows = token_row[:, None, None]
      staged_tile = v_buffers[tile_index]
      v_buffers[tile_index] = jnp.where(
          token_rows < total_len - tail_start,
          staged_tile,
          jnp.zeros_like(staged_tile),
      )

  # Note (david): staged V rows a block never loads keep whatever VMEM held;
  # start them finite so a zero probability times a stale row stays zero.
  for tiling in range(num_tilings):
    v_buffers = tiling_scratch[tiling][1]
    value_words = (
        v_buffers.bitcast(jnp.uint32) if is_bitcast_load else v_buffers
    )
    value_words[...] = jnp.zeros(value_words.shape, value_words.dtype)

  def _out_wait(tiling, num_rows):
    _, _, _, out_stage, _, _, _, lse_stage = tiling_scratch[tiling]
    # Note (david): a row count carried through a loop loses its 8-alignment;
    # restate it, or a 256-lane LSE stage (more than 128 query heads) fails
    # tile verification.
    aligned_rows = pl.multiple_of(num_rows, NUM_SUBLANES)

    @pl.when(aligned_rows > 0)
    def _wait():
      tile = out_stage.at[pl.ds(0, aligned_rows), :, :]
      pltpu.make_async_copy(tile, tile, sems.at[SEM_OUT, 0, tiling]).wait()
      if return_lse:
        lse_tile = lse_stage.at[pl.ds(0, aligned_rows), :]
        pltpu.make_async_copy(
            lse_tile, lse_tile, sems.at[SEM_LSE, 0, tiling]
        ).wait()

  def _pending_wait(pending_tiling, pending_rows):
    for tiling in range(num_tilings):

      @pl.when(pending_tiling == tiling)
      def _wait(tiling=tiling):
        _out_wait(tiling, pending_rows)

  def _out_start(tiling, buffer_start, row_start, num_rows):
    _, _, _, out_stage, _, _, _, lse_stage = tiling_scratch[tiling]
    pltpu.make_async_copy(
        out_stage.at[pl.ds(buffer_start, num_rows), :, :],
        out_ref.at[pl.ds(row_start, num_rows), :, :],
        sems.at[SEM_OUT, 0, tiling],
    ).start()
    if return_lse:
      pltpu.make_async_copy(
          lse_stage.at[pl.ds(buffer_start, num_rows), :],
          lse_ref.at[pl.ds(row_start, num_rows), :],
          sems.at[SEM_LSE, 0, tiling],
      ).start()

  def _tiling_of(rows_left):
    # Note (david): tiling rows are strictly descending, so counting the
    # smaller tilings that still cover rows_left indexes the smallest covering
    # one; a chunk wider than max_chunk is a full chunk and runs tiling 0.
    return jnp.where(
        rows_left > max_chunk,
        0,
        sum(
            (rows_left <= tiling_rows).astype(jnp.int32)
            for tiling_rows, _, _, _ in tilings[1:]
        ),
    )

  def _process_sequence(sequence, sequence_carry):
    query_start = cu_seqlens_ref[sequence]
    query_end = cu_seqlens_ref[sequence + 1]
    query_len = query_end - query_start
    total_len = total_lengths_ref[sequence]
    has_rows = query_len > 0
    has_blocks = jnp.logical_and(has_rows, total_len > 0)
    aligned_start = query_start // NUM_SUBLANES * NUM_SUBLANES
    aligned_end = (
        (query_end + (NUM_SUBLANES - 1)) // NUM_SUBLANES * NUM_SUBLANES
    )
    num_chunks = jnp.where(
        has_rows, pl.cdiv(query_end - aligned_start, max_chunk), 0
    )
    next_sequence = jnp.minimum(sequence + 1, num_active - 1)
    next_start = cu_seqlens_ref[next_sequence]
    next_end = cu_seqlens_ref[next_sequence + 1]
    next_total_len = total_lengths_ref[next_sequence]
    next_has_blocks = jnp.logical_and(
        sequence + 1 < num_active,
        jnp.logical_and(next_end > next_start, next_total_len > 0),
    )
    next_request_tiling = _tiling_of(
        (next_end + (NUM_SUBLANES - 1)) // NUM_SUBLANES * NUM_SUBLANES
        - next_start // NUM_SUBLANES * NUM_SUBLANES
    )
    visible_offset = total_len - query_len if causal else 0
    last_chunk_tiling = _tiling_of(
        aligned_end - (aligned_start + (num_chunks - 1) * max_chunk)
    )

    def _process_chunk(tiling, chunk, chunk_carry):
      rows, block_kv, block_kv_compute, kv_heads_per_fragment = (
          tilings[tiling]
      )
      (
          k_buffers,
          v_buffers,
          q_load,
          out_stage,
          q_major,
          accumulator_ref,
          row_state_ref,
          lse_stage,
      ) = tiling_scratch[tiling]
      (
          is_first_prefetched,
          previous_tile,
          pending_rows,
          pending_tiling,
          *slots,
      ) = chunk_carry
      kv_slot = slots[tiling]
      pass_blocks = jnp.where(has_blocks, pl.cdiv(total_len, block_kv), 0)
      window_start = pl.multiple_of(
          aligned_start + chunk * max_chunk, NUM_SUBLANES
      )
      is_last_chunk = chunk == num_chunks - 1
      rows_left = aligned_end - window_start
      # Note (david): the q operand is only 8-aligned, so a window that would
      # run past it loads from an earlier base instead; its leading rows repeat
      # the previous chunk (or belong to earlier requests) and are never
      # written.
      load_start = pl.multiple_of(
          jnp.minimum(window_start, q_rows - rows), NUM_SUBLANES
      )
      write_offset = pl.multiple_of(window_start - load_start, NUM_SUBLANES)
      row_shift = load_start - query_start
      write_rows = pl.multiple_of(
          jnp.clip(rows_left, 0, max_chunk), NUM_SUBLANES
      )
      blend_rows = query_start - window_start
      should_keep_tile = jnp.logical_and(
          is_last_chunk, query_end % NUM_SUBLANES != 0
      )
      # Note (david): the next pass (this request's next chunk or the next
      # request's first) gets its first block prefetched into its own tiling's
      # staging, at the slot that tiling's next pass starts from.
      next_tiling = jnp.where(
          is_last_chunk, next_request_tiling, _tiling_of(rows_left - max_chunk)
      )
      next_pass_has_blocks = jnp.where(
          is_last_chunk, next_has_blocks, has_blocks
      )
      next_pass_sequence = jnp.where(is_last_chunk, next_sequence, sequence)
      next_pass_total_len = jnp.where(
          is_last_chunk, next_total_len, total_len
      )
      next_slots = list(slots)
      next_slots[tiling] = lax.rem(kv_slot + pass_blocks, num_kv_stages)

      num_kv_compute_fragments = block_kv // block_kv_compute
      num_head_groups = num_kv_heads // kv_heads_per_fragment
      group_heads = query_heads_per_kv_head * kv_heads_per_fragment
      fragment_keys = kv_heads_per_fragment * block_kv_compute
      mask_shape = (group_heads, rows, fragment_keys)
      group_rows = group_heads * rows

      def _kv_fragment(buffers, slot, head_group, kv_compute_index, kv_part):
        staged_head = (
            head_group + kv_part * num_kv_heads
            if is_merged_cache else head_group
        )
        return load_kv_fragment(
            buffers, slot, staged_head, kv_compute_index, kv_part,
            block_kv_compute=block_kv_compute,
            kv_heads_per_fragment=kv_heads_per_fragment,
            is_kv_pair_tile=False,
            is_cache_head_major=is_cache_head_major,
            is_bitcast_load=is_bitcast_load,
        )

      q_copy = pltpu.make_async_copy(
          q_ref.at[pl.ds(load_start, rows), :, :],
          q_load,
          sems.at[SEM_Q, 0, tiling],
      )
      q_copy.start()

      @pl.when(
          jnp.logical_and(
              pass_blocks > 0, jnp.logical_not(is_first_prefetched)
          )
      )
      def _prefetch_first_block():
        _start_kv(tiling, sequence, total_len, 0, kv_slot)

      q_copy.wait()
      # Note (david): fold the softmax scale (log2 units) into the staged rows
      # and lay them out head-major, so a head group's queries are one
      # contiguous window.
      q_major[...] = (
          q_load[...].astype(jnp.float32) * q_scale
      ).astype(q_ref.dtype).transpose(1, 0, 2)

      def _attend_block(block_index, _):
        slot = lax.rem(kv_slot + block_index, num_kv_stages)
        is_last_block = block_index + 1 == pass_blocks

        @pl.when(block_index + 1 < pass_blocks)
        def _prefetch_next_block():
          _start_kv(
              tiling, sequence, total_len, block_index + 1,
              lax.rem(slot + 1, num_kv_stages),
          )

        for target_tiling in range(num_tilings):

          @pl.when(
              jnp.logical_and(
                  jnp.logical_and(is_last_block, next_pass_has_blocks),
                  next_tiling == target_tiling,
              )
          )
          def _prefetch_next_pass(target_tiling=target_tiling):
            _start_kv(
                target_tiling, next_pass_sequence, next_pass_total_len, 0,
                next_slots[target_tiling],
            )

        _wait_kv(tiling, total_len, block_index, slot)
        _zero_value_tail(tiling, total_len, block_index, slot)
        _block_pipeline(block_index, slot, fixed_anchor=True)

      def _block_pipeline(block_index, slot, fixed_anchor):
        # One block's masks, scores, softmax and PV, shared by its two
        # callers: _attend_block with fixed_anchor=True, and the replay after
        # the pass with False. fixed_anchor is the dense kernel's softmax: each
        # row keeps the max of its first fragment as its anchor, so no fragment
        # rescales the row sum or the accumulator; a pass whose row sums end
        # past the overflow bound replays with the rescaling update.
        def _parity_mask():
          # Note (david): a pair-packed tile interleaves its two heads per
          # token, so each query head masks the off-parity columns. Full-shape
          # iota compares are born in the layout the select needs.
          head_index = lax.broadcasted_iota(jnp.int32, mask_shape, 0)
          column_index = lax.broadcasted_iota(jnp.int32, mask_shape, 2)
          return head_index // query_heads_per_kv_head == column_index % 2

        def _build_masks():
          parity = _parity_mask() if kv_heads_per_fragment == 2 else None
          valid_masks = []
          for kv_compute_index in range(num_kv_compute_fragments):
            key_start = (
                block_index * block_kv + kv_compute_index * block_kv_compute
            )
            key_position = key_start + lax.broadcasted_iota(
                jnp.int32, (rows, fragment_keys), 1
            ) // kv_heads_per_fragment
            if causal:
              # Note (david): clamp to [0, total_len) so every row, including
              # rows outside the request or rows that see nothing, keeps key 0
              # and no row's softmax goes NaN; those rows are zeroed at the
              # output.
              row_limit = jnp.clip(
                  visible_offset + row_shift + lax.broadcasted_iota(
                      jnp.int32, (rows, fragment_keys), 0
                  ),
                  0,
                  total_len - 1,
              )
              is_visible = key_position <= row_limit
            else:
              is_visible = key_position < total_len
            if parity is None:
              valid_masks.append(is_visible[None])
            else:
              valid_masks.append(jnp.logical_and(is_visible[None], parity))
          return valid_masks

        def _score_and_softmax(valid_masks, head_group, kv_compute_index,
                              row_max_carry, row_sum_carry):
          head_slice = slice(
              head_group * group_heads, (head_group + 1) * group_heads
          )
          query = q_major[head_slice, :, :].reshape(group_rows, head_dim)
          key = _kv_fragment(k_buffers, slot, head_group, kv_compute_index, 0)
          raw_scores = jnp.dot(
              query, key.T, preferred_element_type=jnp.float32
          ).reshape(mask_shape)
          if valid_masks is None:
            scores = raw_scores
          else:
            scores = jnp.where(
                valid_masks[kv_compute_index], raw_scores, -jnp.inf
            )
          if kv_compute_index == 0:
            row_max_prev = row_state_ref[head_slice, :, 0:1]
            row_sum_prev = row_state_ref[head_slice, :, 1:2]
            is_first = block_index == 0
          else:
            row_max_prev, row_sum_prev = row_max_carry, row_sum_carry
            is_first = False
          if fixed_anchor:
            row_max = jnp.where(
                is_first, scores.max(axis=-1, keepdims=True), row_max_prev
            )
            probabilities = jnp.exp2(scores - row_max)
            fragment_sum = probabilities.sum(axis=-1, keepdims=True)
            row_sum = jnp.where(
                is_first, fragment_sum, row_sum_prev + fragment_sum
            )
            rescale = None
          else:
            probabilities, row_max, row_sum, rescale = static_anchor_update(
                scores,
                row_max_prev,
                row_sum_prev,
                is_first=is_first,
                guard_threshold=guard_threshold,
            )
          if kv_compute_index == num_kv_compute_fragments - 1:
            # Note (david): one 128-lane state tile per row; lane 0 holds
            # row_max and the other lanes row_sum (read back from lane 1).
            row_state_ref[head_slice, :, :] = jnp.concatenate(
                [
                    row_max,
                    jnp.broadcast_to(
                        row_sum, (group_heads, rows, NUM_LANES - 1)
                    ),
                ],
                axis=-1,
            )
          return probabilities, rescale, is_first, row_max, row_sum

        def _pv(head_group, kv_compute_index, probabilities, rescale, is_first):
          head_slice = slice(
              head_group * group_heads, (head_group + 1) * group_heads
          )
          value = _kv_fragment(v_buffers, slot, head_group, kv_compute_index, 1)
          weighted_values = jnp.dot(
              probabilities.reshape(group_rows, fragment_keys).astype(
                  jnp.bfloat16
              ),
              value,
              preferred_element_type=jnp.float32,
          ).reshape(group_heads, rows, head_dim)
          accumulated_prev = accumulator_ref[head_slice, :, :].astype(
              jnp.float32
          )
          if rescale is None:
            rescaled = accumulated_prev + weighted_values
          else:
            rescaled = accumulated_prev * rescale + weighted_values
          if kv_compute_index == 0:
            # Note (david): the accumulator is not cleared between passes, so
            # a pass's first fragment overwrites it instead of rescaling stale
            # values.
            accumulated = jnp.where(is_first, weighted_values, rescaled)
          else:
            accumulated = rescaled
          accumulator_ref[head_slice, :, :] = accumulated

        def _run_pipeline(valid_masks):
          # Note (david): software pipeline; each fragment's scores are issued
          # one step ahead of its own pv, so the MXU has the next matmul queued
          # while pv runs. pending_fragment is (head_group, kv_compute_index,
          # probabilities, rescale, is_first, row_max, row_sum).
          pending_fragment = None
          for fragment_index in range(
              num_head_groups * num_kv_compute_fragments
          ):
            head_group, kv_compute_index = divmod(
                fragment_index, num_kv_compute_fragments
            )
            current_fragment = _score_and_softmax(
                valid_masks, head_group, kv_compute_index,
                *(
                    pending_fragment[5:]
                    if pending_fragment is not None
                    else (None, None)
                ),
            )
            if pending_fragment is not None:
              _pv(*pending_fragment[:5])
            pending_fragment = (head_group, kv_compute_index, *current_fragment)
          _pv(*pending_fragment[:5])

        # Note (david): a block every row of the window sees in full needs no
        # causal mask: its keys are initialized cache entries (so finite), and
        # rows outside the request or before their first visible key are
        # zeroed at the output anyway. Prefix blocks are the common case, so
        # they get a static copy of the fragment pipeline instead of a runtime
        # select.
        block_end = (block_index + 1) * block_kv
        is_block_cached = block_end <= total_len
        if causal:
          is_fully_visible = jnp.logical_and(
              is_block_cached,
              block_end <= visible_offset + jnp.maximum(row_shift, 0) + 1,
          )
        else:
          is_fully_visible = is_block_cached

        @pl.when(is_fully_visible)
        def _unmasked_block():
          if kv_heads_per_fragment == 2:
            _run_pipeline([_parity_mask()] * num_kv_compute_fragments)
          else:
            _run_pipeline(None)

        @pl.when(jnp.logical_not(is_fully_visible))
        def _masked_block():
          _run_pipeline(_build_masks())

      lax.fori_loop(0, pass_blocks, _attend_block, None)

      # A row sum bounds each of its probabilities, and once a score
      # overflows the sum stays inf or NaN, so one check after the pass finds
      # any row whose scores rose too far above its anchor. The replay stages
      # each block synchronously in the slot the next pass's prefetch does not
      # hold (the one this pass's last block used).
      row_sums = row_state_ref[:, :, 1:2]
      guard_smem[0] = jnp.any(
          jnp.logical_not(row_sums <= guard_detect_bound)
      ).astype(jnp.int32)

      @pl.when(jnp.logical_and(pass_blocks > 0, guard_smem[0] > 0))
      def _replay_with_rescaling():
        replay_slot = lax.rem(kv_slot + pass_blocks + 1, num_kv_stages)

        def _replay_block(block_index, _):
          _start_kv(tiling, sequence, total_len, block_index, replay_slot)
          _wait_kv(tiling, total_len, block_index, replay_slot)
          _zero_value_tail(tiling, total_len, block_index, replay_slot)
          _block_pipeline(block_index, replay_slot, fixed_anchor=False)

        lax.fori_loop(0, pass_blocks, _replay_block, None)

      # Note (david): the previous pass's output DMA (possibly from another
      # tiling's stage) must land before this pass's write starts; with HBM
      # writes serialized, a later window that repeats a shared boundary tile
      # always lands last.
      _pending_wait(pending_tiling, pending_rows)

      row_offset = row_shift + lax.broadcasted_iota(jnp.int32, (1, rows, 1), 1)
      is_in_request = jnp.logical_and(row_offset >= 0, row_offset < query_len)
      if causal:
        is_row_visible = jnp.logical_and(
            is_in_request, visible_offset + row_offset >= 0
        )
      else:
        is_row_visible = is_in_request

      @pl.when(pass_blocks > 0)
      def _normalize():
        row_sum = row_state_ref[:, :, 1:2]
        should_keep = jnp.logical_and(is_row_visible, row_sum > 0)
        inverse_row_sum = jnp.where(
            should_keep, pl.reciprocal(row_sum, approx=True), 0.0
        )
        out_stage[...] = (
            accumulator_ref[...].astype(jnp.float32) * inverse_row_sum
        ).astype(q_ref.dtype).transpose(1, 0, 2)
        if return_lse:
          lse_log2 = row_state_ref[:, :, 0:1] + jnp.log2(row_sum)
          head_major_lse = jnp.where(
              should_keep, lse_log2 * math.log(2.0), -jnp.inf
          )
          token_major_lse = head_major_lse.reshape(num_query_heads, rows).T
          if lse_width > num_query_heads:
            padded_lse = jnp.concatenate(
                [token_major_lse,
                 jnp.full((rows, lse_width - num_query_heads),
                          -jnp.inf, jnp.float32)],
                axis=1,
            )
          else:
            padded_lse = token_major_lse
          lse_stage[...] = padded_lse.astype(jnp.float32)

      @pl.when(pass_blocks == 0)
      def _nothing_visible():
        out_stage[...] = jnp.zeros(out_stage.shape, out_stage.dtype)
        if return_lse:
          lse_stage[...] = jnp.full(lse_stage.shape, -jnp.inf, jnp.float32)

      # Note (david): this request's first 8-aligned tile may hold rows of the
      # previous request, whose last tile the boundary refs kept.
      @pl.when(window_start == previous_tile)
      def _blend_boundary_tile():
        tile_index = (
            pl.ds(write_offset, NUM_SUBLANES), slice(None), slice(None)
        )
        current_tile = out_stage[tile_index]
        tile_rows = lax.broadcasted_iota(jnp.int32, current_tile.shape, 0)
        out_stage[tile_index] = jnp.where(
            tile_rows < blend_rows, boundary_out_ref[...], current_tile
        )
        if return_lse:
          lse_index = (pl.ds(write_offset, NUM_SUBLANES), slice(None))
          current_lse = lse_stage[lse_index]
          lse_tile_rows = lax.broadcasted_iota(jnp.int32, current_lse.shape, 0)
          lse_stage[lse_index] = jnp.where(
              lse_tile_rows < blend_rows, boundary_lse_ref[...], current_lse
          )

      @pl.when(should_keep_tile)
      def _keep_last_tile():
        last_tile = pl.multiple_of(
            aligned_end - NUM_SUBLANES - load_start, NUM_SUBLANES
        )
        boundary_out_ref[...] = out_stage[
            pl.ds(last_tile, NUM_SUBLANES), :, :
        ]
        if return_lse:
          boundary_lse_ref[...] = lse_stage[pl.ds(last_tile, NUM_SUBLANES), :]

      _out_start(tiling, write_offset, window_start, write_rows)

      next_previous_tile = jnp.where(
          is_last_chunk,
          jnp.where(should_keep_tile, aligned_end - NUM_SUBLANES, -1),
          previous_tile,
      )
      # Note (david): the next pass's first block is prefetched from this
      # pass's last block, so a pass with no block (rows over an empty cache
      # without an append) prefetched nothing, and the next pass must start
      # its own load instead of waiting on one that was never issued.
      return (
          jnp.logical_and(next_pass_has_blocks, pass_blocks > 0),
          next_previous_tile,
          write_rows,
          jnp.int32(tiling),
          *next_slots,
      )

    # Note (david): one monomorphic chunk loop per tiling, picked by its trip
    # count instead of a runtime branch; full chunks always run tiling 0 and a
    # request's last chunk runs exactly one tiling.
    carry = sequence_carry
    for tiling in range(num_tilings):
      if tiling == 0:
        num_trips = jnp.where(
            last_chunk_tiling == 0, num_chunks, jnp.maximum(num_chunks - 1, 0)
        )
        first_chunk = 0
      else:
        num_trips = jnp.where(
            jnp.logical_and(has_rows, last_chunk_tiling == tiling), 1, 0
        )
        first_chunk = num_chunks - 1
      carry = lax.fori_loop(
          0,
          num_trips,
          lambda trip, chunk_carry, tiling=tiling, first_chunk=first_chunk: (
              _process_chunk(tiling, first_chunk + trip, chunk_carry)
          ),
          carry,
      )
    return carry

  initial_carry = (
      (False, jnp.int32(-1), jnp.int32(0), jnp.int32(0))
      + (jnp.int32(0),) * num_tilings
  )
  _, _, pending_rows, pending_tiling, *_ = lax.fori_loop(
      0, num_active, _process_sequence, initial_carry
  )
  _pending_wait(pending_tiling, pending_rows)

  if should_zero_padding:
    # Note (david): the last request's own tile already zeroed its tail rows,
    # so the fill of the remaining padding rows starts on the 8-aligned grid,
    # from the first tiling's (largest) output stage.
    _, _, _, fill_stage, _, _, _, fill_lse_stage = tiling_scratch[0]
    fill_start = (
        (cu_seqlens_ref[num_active] + (NUM_SUBLANES - 1))
        // NUM_SUBLANES * NUM_SUBLANES
    )
    num_fills = pl.cdiv(jnp.maximum(q_rows - fill_start, 0), max_chunk)

    @pl.when(num_fills > 0)
    def _clear_stage():
      fill_stage[...] = jnp.zeros(fill_stage.shape, fill_stage.dtype)
      if return_lse:
        fill_lse_stage[...] = jnp.full(
            fill_lse_stage.shape, -jnp.inf, jnp.float32
        )

    def _fill_chunk(fill_index, _):
      row_start = pl.multiple_of(
          fill_start + fill_index * max_chunk, NUM_SUBLANES
      )
      num_rows = pl.multiple_of(
          jnp.clip(q_rows - row_start, 0, max_chunk), NUM_SUBLANES
      )
      _out_start(0, 0, row_start, num_rows)
      _out_wait(0, num_rows)

    lax.fori_loop(0, num_fills, _fill_chunk, None)


def estimate_vmem_bytes(
    tilings: list[TilingSpec],
    *,
    num_query_heads: int,
    num_kv_heads: int,
    staged_heads: int,
    num_staging_buffers: int,
    head_dim: int,
    return_lse: bool,
    lse_width: int,
) -> int:
  """Scoped VMEM of one extend build, in bytes.

  Tracks what Mosaic reported for rejected v6e builds to within ~1%.
  """
  query_heads_per_kv_head = num_query_heads // num_kv_heads
  # Per query head and row: three bf16 head_dim buffers (token-major q load and
  # output stage, head-major q), the f32 accumulator and a 128-lane f32 row
  # state.
  row_buffer_bytes = num_query_heads * (
      3 * head_dim * BF16_BYTES + head_dim * F32_BYTES + NUM_LANES * F32_BYTES
  )
  if return_lse:
    bytes_per_row = row_buffer_bytes + lse_width * F32_BYTES
  else:
    bytes_per_row = row_buffer_bytes
  total_bytes = num_query_heads * NUM_SUBLANES * head_dim * BF16_BYTES
  temporary_bytes = 0
  for rows, block_kv, block_kv_compute, kv_heads_per_fragment in tilings:
    total_bytes += (
        num_staging_buffers * NUM_KV_STAGES * block_kv * staged_heads
        * head_dim * BF16_BYTES
    )
    total_bytes += rows * bytes_per_row
    # Note (david): five live score-shaped f32 temporaries of the two
    # pipelined fragments, plus the one-byte boolean masks of one block.
    group_heads = query_heads_per_kv_head * kv_heads_per_fragment
    fragment_keys = kv_heads_per_fragment * block_kv_compute
    temporary_bytes = max(
        temporary_bytes,
        rows * group_heads * fragment_keys
        * (5 * F32_BYTES + block_kv // block_kv_compute + 1),
    )
  return total_bytes + temporary_bytes


def resolve_tilings(
    total_rows: int,
    *,
    num_query_heads: int,
    num_kv_heads: int,
    staged_heads: int,
    num_staging_buffers: int,
    head_dim: int,
    capacity: int,
    page_size: int,
    is_pair_packed: bool,
    return_lse: bool,
    lse_width: int,
) -> tuple[TilingSpec, ...]:
  """Default static tilings for one build.

  Each of default_tilings() is fit to the cache (whole-page blocks dividing
  the capacity) and capped at the 8-aligned token count; a tiling whose rows
  no longer fall below the previous one's is dropped. Blocks then shrink, from
  the last tiling back, and finally the first tiling's rows, until the build
  fits the scoped VMEM budget.
  """
  q_rows = round_up(total_rows, NUM_SUBLANES)

  def _fit_block(target_block_kv):
    block_kv = max(
        min(target_block_kv, capacity) // page_size * page_size, page_size
    )
    return next(
        block for block in range(block_kv, 0, -page_size)
        if capacity % block == 0
    )

  tilings = []
  for rows, block_kv, block_kv_compute, kv_heads_per_fragment in (
      default_tilings(num_query_heads // num_kv_heads)
  ):
    fitted_block_kv = _fit_block(block_kv)
    fitted_block_kv_compute = next(
        candidate
        for candidate in (block_kv_compute, *BLOCK_KV_COMPUTE_CANDIDATES)
        if candidate <= fitted_block_kv and fitted_block_kv % candidate == 0
    )
    fitted_rows = min(rows, q_rows)
    if not tilings or fitted_rows < tilings[-1][0]:
      tilings.append((
          fitted_rows,
          fitted_block_kv,
          fitted_block_kv_compute,
          kv_heads_per_fragment if is_pair_packed else 1,
      ))

  while estimate_vmem_bytes(
      tilings, num_query_heads=num_query_heads, num_kv_heads=num_kv_heads,
      staged_heads=staged_heads, num_staging_buffers=num_staging_buffers,
      head_dim=head_dim, return_lse=return_lse, lse_width=lse_width,
  ) > vmem_limit_bytes():
    shrinkable = [
        index for index, (_, block_kv, _, _) in enumerate(tilings)
        if block_kv > page_size
    ]
    if shrinkable:
      last_shrinkable = shrinkable[-1]
      rows, block_kv, block_kv_compute, kv_heads_per_fragment = (
          tilings[last_shrinkable]
      )
      smaller_block_kv = _fit_block(block_kv // 2)
      tilings[last_shrinkable] = (
          rows,
          smaller_block_kv,
          min(block_kv_compute, smaller_block_kv),
          kv_heads_per_fragment,
      )
    elif tilings[0][0] > NUM_SUBLANES:
      tilings[0] = (tilings[0][0] - NUM_SUBLANES, *tilings[0][1:])
      tilings = [tilings[0]] + [
          spec for spec in tilings[1:] if spec[0] < tilings[0][0]
      ]
    else:
      raise ValueError(
          "Extend attention does not fit the scoped VMEM budget:"
          f" {num_query_heads=}, {num_kv_heads=}, {head_dim=}."
      )
  return tuple(tilings)


def flash_attn_kvcache_extend_pallas(
    q: jax.Array,
    k_cache: jax.Array,
    v_cache: jax.Array | None,
    cu_seqlens_q: jax.Array,
    total_lengths: jax.Array,
    block_table: jax.Array,
    num_active: int | jax.Array,
    *,
    q_scale: float,
    causal: bool,
    return_lse: bool,
    interpret: bool = False,
    should_zero_padding: bool = True,
    tilings: tuple[TilingSpec, ...] | None = None,
    merged_cache: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
  """Packed extend attention against a paged cache that already holds the
  new tokens.

  q: (total_q, heads, head_dim) bf16, packed by cu_seqlens_q. k_cache/v_cache:
  (num_pages, page_size, kv_heads, head_dim). With merged_cache=True
  (v_cache=None), k_cache is (num_pages, page_size, 2 * kv_heads, head_dim):
  every K head of a token, then every V head. total_lengths: (batch,) visible
  cache length per request including its new tokens. block_table: flat
  (batch * pages_per_seq,) int32. num_active: int scalar or (1,) int32;
  requests at or past it are never read.

  Query row t of request r (0-based within the request) attends keys
  [0, total - query_len + t] when causal, [0, total) otherwise. Rows of a
  request with no visible key return out = 0 and lse = -inf.
  should_zero_padding extends that to packed padding rows and rows of inactive
  requests; without it those rows are unspecified (ragged_paged_attention's
  contract), which saves one output DMA per chunk of padding.

  tilings: optional explicit tuple of (rows, block_kv, block_kv_compute,
  kv_heads_per_fragment), rows strictly descending; full chunks run the
  first, a request's last chunk the smallest one whose rows cover it. None
  resolves the defaults within the VMEM budget. kv_heads_per_fragment 2
  scores a pair-packed tile against both heads' queries with an off-parity
  column mask (token-major caches with an even head axis, TPU builds only).

  Returns out (total_q, heads, head_dim), plus lse (heads, total_q) float32
  when return_lse.
  """
  num_active_array = jnp.asarray(num_active, jnp.int32).reshape(1)
  if merged_cache:
    if (v_cache is not None or k_cache.ndim != 4 or k_cache.shape[2] < 2
        or k_cache.shape[2] % 2):
      raise ValueError(
          "merged_cache takes one (pages, page_size, 2 * kv_heads, dim) pool"
          f" and v_cache=None; got {k_cache.shape=}."
      )
    cache_shape = (*k_cache.shape[:2], k_cache.shape[2] // 2, k_cache.shape[3])
  elif v_cache is None:
    raise ValueError(
        "A K/V pair pool needs v_cache; one merged pool needs merged_cache."
    )
  else:
    cache_shape = k_cache.shape
  total_rows, num_query_heads, head_dim = q.shape
  _, page_size, num_kv_heads = cache_shape[:3]
  batch = cu_seqlens_q.shape[0] - 1
  pages_per_seq = block_table.shape[0] // batch
  capacity = pages_per_seq * page_size
  if head_dim not in SUPPORTED_HEAD_DIMS:
    raise ValueError(
        f"head_dim must be one of {SUPPORTED_HEAD_DIMS}; got {head_dim}."
    )
  if page_size % MIN_BLOCK_KV:
    raise ValueError(
        f"page_size must be a multiple of {MIN_BLOCK_KV}; got {page_size}."
    )
  if num_kv_heads > 1 and num_kv_heads % 2:
    raise ValueError("token-major cache needs an even head axis or MQA.")
  if num_query_heads % num_kv_heads:
    raise ValueError(
        f"{num_query_heads=} must be a multiple of {num_kv_heads=}."
    )
  is_cache_head_major = num_kv_heads == 1 and not merged_cache
  # Note (david): the pair-packed load bitcasts refs, which only the TPU build
  # supports; interpret mode reads the staged heads as bf16.
  is_bitcast_load = not interpret and not is_cache_head_major
  lse_width = round_up(num_query_heads, NUM_LANES)
  q_rows = round_up(total_rows, NUM_SUBLANES)
  # A merged cache stages whole token rows, K heads then V heads, once, in a
  # single buffer; a K/V pair stages each part in its own.
  staged_heads = 2 * num_kv_heads if merged_cache else round_up(num_kv_heads, 2)
  num_staging_buffers = 1 if merged_cache else 2
  if tilings is None:
    tiling_specs = resolve_tilings(
        total_rows, num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads, staged_heads=staged_heads,
        num_staging_buffers=num_staging_buffers,
        head_dim=head_dim, capacity=capacity,
        page_size=page_size,
        is_pair_packed=is_bitcast_load and not merged_cache,
        return_lse=return_lse, lse_width=lse_width,
    )
  else:
    tiling_specs = tuple(tuple(spec) for spec in tilings)
  row_counts = [spec[0] for spec in tiling_specs]
  if (not tiling_specs or row_counts != sorted(set(row_counts), reverse=True)
      or any(rows <= 0 or rows % NUM_SUBLANES or rows > q_rows
             for rows in row_counts)):
    raise ValueError(
        "tiling rows must be distinct, descending, positive multiples of 8 no"
        f" larger than the 8-aligned token count {q_rows}; got {tiling_specs}."
    )
  for _, block_kv, block_kv_compute, kv_heads_per_fragment in tiling_specs:
    if block_kv % page_size or capacity % block_kv:
      raise ValueError(
          f"block_kv={block_kv} must be whole pages of {page_size} dividing"
          f" capacity={capacity}."
      )
    if block_kv_compute % MIN_BLOCK_KV or block_kv % block_kv_compute:
      raise ValueError(
          f"block_kv_compute={block_kv_compute} must be a"
          f" {MIN_BLOCK_KV}-aligned divisor of block_kv={block_kv}."
      )
    if kv_heads_per_fragment not in (1, 2):
      raise ValueError(
          f"kv_heads_per_fragment must be 1 or 2; got {kv_heads_per_fragment}."
      )
    if kv_heads_per_fragment == 2 and (not is_bitcast_load or merged_cache):
      raise ValueError(
          "kv_heads_per_fragment=2 needs pair-packed bitcast loads (a"
          " token-major cache with an even head axis, compiled for TPU)."
      )

  cache_operands = (k_cache,) if merged_cache else (k_cache, v_cache)
  if is_cache_head_major:
    # Note (david): with one KV head, (..., page_size, 1, head_dim) holds the
    # same bytes as (..., 1, page_size, head_dim), whose page tile is a dense
    # (tokens, head_dim) window with no head pair to pack.
    kernel_caches = tuple(
        cache.reshape(cache.shape[0], 1, page_size, head_dim)
        for cache in cache_operands
    )
  else:
    kernel_caches = cache_operands

  # Note (david): the kernel stages 8-aligned token windows of q as-is
  # (token-major, no host-side scale or transpose), so only an unaligned packed
  # length needs padding.
  if q_rows == total_rows:
    padded_q = q
  else:
    padded_q = jnp.pad(q, ((0, q_rows - total_rows), (0, 0), (0, 0)))

  def _staging_shape(block_kv):
    if is_cache_head_major:
      return (NUM_KV_STAGES, num_kv_heads, block_kv, head_dim)
    elif is_bitcast_load:
      return (NUM_KV_STAGES, block_kv, staged_heads // 2, 2, head_dim)
    else:
      return (NUM_KV_STAGES, block_kv, staged_heads, head_dim)

  scratch_shapes = [
      pltpu.VMEM((NUM_SUBLANES, num_query_heads, head_dim), q.dtype)
  ]
  if return_lse:
    scratch_shapes.append(
        pltpu.VMEM((NUM_SUBLANES, lse_width), jnp.float32)
    )
  for rows, block_kv, _, _ in tiling_specs:
    scratch_shapes += [
        pltpu.VMEM(_staging_shape(block_kv), k_cache.dtype)
        for _ in range(num_staging_buffers)
    ]
    scratch_shapes += [
        pltpu.VMEM((rows, num_query_heads, head_dim), q.dtype),
        pltpu.VMEM((rows, num_query_heads, head_dim), q.dtype),
        pltpu.VMEM((num_query_heads, rows, head_dim), q.dtype),
        pltpu.VMEM((num_query_heads, rows, head_dim), jnp.float32),
        pltpu.VMEM((num_query_heads, rows, NUM_LANES), jnp.float32),
    ]
    if return_lse:
      scratch_shapes.append(pltpu.VMEM((rows, lse_width), jnp.float32))
  scratch_shapes.append(pltpu.SMEM((1,), jnp.int32))
  scratch_shapes.append(
      pltpu.SemaphoreType.DMA((NUM_SEMS, NUM_KV_STAGES, len(tiling_specs)))
  )
  if return_lse:
    out_shapes = (
        jax.ShapeDtypeStruct(padded_q.shape, q.dtype),
        jax.ShapeDtypeStruct((q_rows, lse_width), jnp.float32),
    )
  else:
    out_shapes = (jax.ShapeDtypeStruct(padded_q.shape, q.dtype),)
  hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
  kernel = functools.partial(
      extend_kernel,
      num_query_heads=num_query_heads,
      num_kv_heads=num_kv_heads,
      head_dim=head_dim,
      tilings=tiling_specs,
      num_kv_stages=NUM_KV_STAGES,
      is_cache_head_major=is_cache_head_major,
      is_bitcast_load=is_bitcast_load,
      return_lse=return_lse,
      causal=causal,
      should_zero_padding=should_zero_padding,
      q_scale=q_scale,
      guard_threshold=overflow_guard_threshold(capacity),
      page_size=page_size,
      pages_per_seq=pages_per_seq,
      is_merged_cache=merged_cache,
      q_rows=q_rows,
      lse_width=lse_width,
  )
  scalar_prefetches = (
      cu_seqlens_q, total_lengths, block_table, num_active_array
  )
  kernel_inputs = (padded_q, *kernel_caches)
  call = pl.pallas_call(
      kernel,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=len(scalar_prefetches),
          grid=(1,),
          in_specs=(hbm_spec,) * len(kernel_inputs),
          out_specs=(hbm_spec,) * len(out_shapes),
          scratch_shapes=tuple(scratch_shapes),
      ),
      out_shape=out_shapes,
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("arbitrary",),
          vmem_limit_bytes=vmem_limit_bytes(),
          disable_bounds_checks=True,
          disable_semaphore_checks=True,
      ),
      name="flash_attn_varlen_paged_kvcache",
      interpret=pltpu.InterpretParams() if interpret else False,
  )
  kernel_outputs = call(*scalar_prefetches, *kernel_inputs)
  if q_rows == total_rows:
    out = kernel_outputs[0]
  else:
    out = kernel_outputs[0][:total_rows]
  if return_lse:
    return out, kernel_outputs[1][:total_rows, :num_query_heads].T
  else:
    return out
