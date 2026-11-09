"""Absorbed MLA chunked prefill: a ragged chunk of queries attends its
requests' paged latent cache as MQA, every head of a q block folded into the
score rows, with the static-anchor softmax and an online-softmax replay when
its overflow guard trips."""

import functools
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ...core.tpu import (
    BF16_BYTES,
    BF16_TILE_ROWS,
    F32_BYTES,
    NUM_LANES,
    NUM_SUBLANES,
)
from ..config import LOG2E, NT_DIM_NUMBERS, next_pow2, vmem_limit_bytes
from ..softmax import overflow_guard_threshold
from .decode import lane_broadcast
from .mla_decode import CACHE_ROW_DIM, LATENT_DIM, cache_tile_rows, mla_block_kv

# Note (david): the copies of the next two items run while one computes, as an
# item computes for about as long as its block's copy takes at few query rows.
NUM_KV_STAGES = 3
# Note (david): a masked item skips its kv block's dead tiles, as fwd_varlen
# does; a tile is one MXU pass wide.
TILE_COLUMNS = 2 * NUM_LANES
# Note (david): a q block's copies land while the block before it computes,
# and its output leaves while the block after it does.
NUM_QUERY_SLOTS = 2
# Note (david): finite, not -inf, so a row that sees no key of a block (or none
# at all) keeps every running value finite.
MASKED_SCORE = -1e30
REQUEST_SHIFT = 16
Q_BLOCK_MASK = (1 << REQUEST_SHIFT) - 1
SEM_KV = 0
SEM_QUERY = 1
SEM_OUTPUT = 2


class MlaPrefillConfig(NamedTuple):
    """Query tokens of a q block, cache tokens of a kv block, and the row
    groups an item splits the q block's rows into."""

    block_q: int
    block_kv: int
    row_groups: int


class QueryBlock(NamedTuple):
    """Scalars of one q block of the schedule."""

    ordinal: jax.Array
    request: jax.Array
    q_start: jax.Array
    q_count: jax.Array
    kv_len: jax.Array
    first_position: jax.Array
    num_kv_blocks: jax.Array


def mla_prefill_vmem_bytes(
    config: MlaPrefillConfig,
    *,
    num_heads: int,
    cache_dtype: jnp.dtype,
    return_lse: bool,
) -> int:
    """VMEM bytes of mla_prefill_pallas's scratch at config, plus an item's
    score temporaries."""
    block_q, block_kv, row_groups = config
    itemsize = jnp.dtype(cache_dtype).itemsize
    num_rows = block_q * num_heads
    group_rows = num_rows // row_groups
    # Note (david): a bf16 (tokens, heads, lanes) buffer tiles its heads in
    # whole bf16 tiles.
    staged_rows = block_q * pl.cdiv(num_heads, BF16_TILE_ROWS) * BF16_TILE_ROWS
    kv_ring_bytes = NUM_KV_STAGES * block_kv * CACHE_ROW_DIM * itemsize
    if itemsize < BF16_BYTES:
        widened_bytes = block_kv * CACHE_ROW_DIM * BF16_BYTES
    else:
        widened_bytes = 0
    query_bytes = (
        NUM_QUERY_SLOTS
        * staged_rows
        * (CACHE_ROW_DIM + NUM_LANES + LATENT_DIM)
        * BF16_BYTES
    )
    state_bytes = (
        num_rows * (3 * NUM_LANES + LATENT_DIM) + NUM_SUBLANES * NUM_LANES
    ) * F32_BYTES
    if return_lse:
        lse_bytes = (
            NUM_QUERY_SLOTS
            * pl.cdiv(block_q, NUM_SUBLANES)
            * NUM_SUBLANES
            * NUM_LANES
            * F32_BYTES
        )
    else:
        lse_bytes = 0
    # Note (david): f32 scores, their f32 exponentials and the bf16
    # probabilities of two row groups are live at once.
    score_bytes = 2 * group_rows * block_kv * (2 * F32_BYTES + BF16_BYTES)
    return (
        kv_ring_bytes
        + widened_bytes
        + query_bytes
        + state_bytes
        + lse_bytes
        + score_bytes
    )


def default_mla_prefill_config(
    *,
    num_heads: int,
    page_size: int,
    pages_per_seq: int,
    cache_dtype: jnp.dtype,
    return_lse: bool,
    max_seqlen_q: int,
    max_seqlen_k: int,
) -> MlaPrefillConfig:
    """A q block of about target_rows score rows, no longer than the longest
    chunk, in row groups of group_rows rows, and a kv block of about
    target_block_kv tokens fitted to the pages and the longest request; past
    the VMEM budget the q block halves, then the kv block."""
    # Note (david): big q blocks amortize the fixed costs of an item and a q
    # block; bigger kv blocks or row groups spill the scores, and smaller row
    # groups starve the MXU.
    target_rows = 4096
    group_rows = 256
    target_block_kv = min(512, max(NUM_LANES, next_pow2(max_seqlen_k)))
    vmem_budget = 0.75 * vmem_limit_bytes()
    block_q = min(max(1, target_rows // num_heads), next_pow2(max_seqlen_q))
    bound_pages = pl.cdiv(
        min(max_seqlen_k, pages_per_seq * page_size), page_size
    )
    while True:
        block_kv = mla_block_kv(page_size, bound_pages, target_block_kv)
        config = MlaPrefillConfig(
            block_q, block_kv, max(1, block_q * num_heads // group_rows)
        )
        shrunk_block_kv = mla_block_kv(
            page_size, bound_pages, target_block_kv // 2
        )
        if (
            mla_prefill_vmem_bytes(
                config,
                num_heads=num_heads,
                cache_dtype=cache_dtype,
                return_lse=return_lse,
            )
            <= vmem_budget
        ):
            return config
        elif block_q > 1:
            block_q //= 2
        elif shrunk_block_kv < block_kv:
            target_block_kv //= 2
        else:
            raise ValueError(
                "flash_mla_prefill needs more than 0.75 of the"
                f" {vmem_limit_bytes()}-byte VMEM limit at a one-token q"
                f" block and a {block_kv}-token kv block."
            )


def mla_prefill_schedule(
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    *,
    block_q: int,
    block_kv: int,
    max_q_blocks: int,
    causal: bool,
) -> tuple[jax.Array, jax.Array]:
    """The call's q blocks as one flat int32 table, a pure function of
    cu_q_lens and kv_lens, so XLA builds it once for every layer of a step,
    and each request's first q block.

    Entry j < max_q_blocks is q block j's request << REQUEST_SHIFT | its index
    in the request; entry max_q_blocks + j is its kv block count, every kv
    block up to its last token's causal frontier; the last two entries are
    the q block count and the item count. An inactive request has no q block.
    """
    batch = kv_lens.shape[0]
    q_begin = cu_q_lens[:-1]
    q_end = cu_q_lens[1:]
    request_q_blocks = (
        jnp.maximum(q_end - q_begin, 0) + block_q - 1
    ) // block_q
    q_block_end = jnp.cumsum(request_q_blocks).astype(jnp.int32)
    request_first_q_block = q_block_end - request_q_blocks
    num_q_blocks = jnp.minimum(q_block_end[-1], max_q_blocks)
    ordinal = jnp.arange(max_q_blocks, dtype=jnp.int32)
    request = jnp.minimum(
        jnp.searchsorted(q_block_end, ordinal, side="right"), batch - 1
    ).astype(jnp.int32)
    q_block = ordinal - request_first_q_block[request]
    q_start = q_begin[request] + q_block * block_q
    q_count = jnp.minimum(block_q, q_end[request] - q_start)
    kv_len = kv_lens[request]
    if causal:
        frontier = kv_len - q_end[request] + q_start + q_count
    else:
        frontier = kv_len
    is_live = ordinal < num_q_blocks
    # Note (david): a q block whose rows see no key still takes an item, which
    # writes its output; only lengths that break the contract get there.
    num_kv_blocks = jnp.where(
        is_live, jnp.maximum((frontier + block_kv - 1) // block_kv, 1), 0
    )
    entries = jnp.where(is_live, request * (1 << REQUEST_SHIFT) + q_block, 0)
    table = jnp.concatenate([
        entries,
        num_kv_blocks,
        num_q_blocks[None],
        jnp.sum(num_kv_blocks)[None],
    ]).astype(jnp.int32)
    return table, request_first_q_block


def mla_prefill_kernel(
    schedule_ref: jax.Array,
    cu_q_lens_ref: jax.Array,
    kv_lens_ref: jax.Array,
    block_table_ref: jax.Array,
    q_nope_hbm: jax.Array,
    q_pe_hbm: jax.Array,
    cache_hbm: jax.Array,
    *refs: jax.Array,
    page_size: int,
    pages_per_seq: int,
    max_q_blocks: int,
    config: MlaPrefillConfig,
    num_heads: int,
    sm_scale_log2: float,
    is_causal: bool,
    return_lse: bool,
    is_quantized: bool,
    guard_threshold: float,
) -> None:
    """Run the schedule's items, (q block, kv block) pairs in q block order,
    with the static-anchor softmax, and replay every q block with the online
    softmax if the overflow guard tripped.

    Score rows are a q block's tokens times its heads, token-major (row
    t * H + h). With is_quantized, refs start with the SMEM (1,) f32 kv_scale
    and the scratch ends with the widened block; with return_lse, the outputs
    end with q block j's lse at [j, token, head] and the scratch with its
    stage. guard_threshold is the guard's bound in log2 units.
    """
    # Note (david): K and V share the fp8 cache's one kv_scale, so it folds
    # into the score scale and the output normalization instead of scaling
    # cache elements.
    if is_quantized:
        kv_scale_ref, out_hbm, *refs = refs
        kv_scale = kv_scale_ref[0]
        score_scale = kv_scale * sm_scale_log2
    else:
        out_hbm, *refs = refs
        kv_scale = 1.0
        score_scale = sm_scale_log2
    if return_lse:
        lse_hbm, *refs = refs
    else:
        lse_hbm = None
    (
        kv_ring,
        query_buffer,
        q_pe_stage,
        out_stage,
        row_max_ref,
        row_sum_ref,
        accumulator_ref,
        row_token_ref,
        max_row_sum_ref,
        should_replay_ref,
        dma_sems,
        *refs,
    ) = refs
    if is_quantized:
        widened_block, *refs = refs
    else:
        widened_block = None
    if return_lse:
        (lse_stage,) = refs
    else:
        lse_stage = None
    block_q, block_kv, num_groups = config
    num_rows = block_q * num_heads
    row_dim = query_buffer.shape[-1]
    rope_dim = q_pe_stage.shape[-1]
    latent_dim = out_stage.shape[-1]
    tile_rows = cache_tile_rows(kv_ring.dtype)
    group_tokens = block_q // num_groups
    group_rows = group_tokens * num_heads
    num_q_blocks = schedule_ref[2 * max_q_blocks]
    num_items = schedule_ref[2 * max_q_blocks + 1]
    # Note (david): the word layout is a row-major reshape of the token rows,
    # so every copy addresses token rows, as flash_mla_decode's do. The Pallas
    # interpreter takes no ref reshape and is handed the token rows.
    if len(cache_hbm.shape) == 3:
        cache_token_rows = cache_hbm
    else:
        cache_token_rows = cache_hbm.reshape(
            cache_hbm.shape[0], page_size, row_dim
        )

    def _query_block(ordinal):
        # Note (david): the prefetch cursor walks past the last q block on the
        # last items, and its reads must stay in the table.
        entry_index = jnp.minimum(ordinal, max_q_blocks - 1)
        entry = schedule_ref[entry_index]
        request = lax.shift_right_logical(entry, REQUEST_SHIFT)
        q_end = cu_q_lens_ref[request + 1]
        q_start = cu_q_lens_ref[request] + (entry & Q_BLOCK_MASK) * block_q
        kv_len = kv_lens_ref[request]
        return QueryBlock(
            ordinal,
            request,
            q_start,
            jnp.minimum(block_q, q_end - q_start),
            kv_len,
            kv_len - q_end + q_start,
            schedule_ref[max_q_blocks + entry_index],
        )

    def _next_item(block, kv_block):
        is_block_end = kv_block + 1 >= block.num_kv_blocks
        return (
            jnp.where(is_block_end, block.ordinal + 1, block.ordinal),
            jnp.where(is_block_end, 0, kv_block + 1),
        )

    def _staged_tokens(num_tokens):
        """num_tokens rounded up to whole cache tiles, the unit of a copy
        window."""
        return pl.multiple_of(
            (num_tokens + tile_rows - 1) // tile_rows * tile_rows, tile_rows
        )

    def _physical_page(request, page_index, num_tokens):
        # Note (david): a copy of no tokens still forms its address, so it
        # reads page 0, never a block_table entry past the request's pages.
        table_index = request * pages_per_seq + jnp.minimum(
            page_index, pages_per_seq - 1
        )
        return jnp.where(num_tokens > 0, block_table_ref[table_index], 0)

    def _start_kv_copies(block, kv_block, slot):
        """Copy the kv block's cache tokens below kv_len, their last tile
        whole, to the front of the ring slot."""
        block_start = kv_block * block_kv
        kv_sem = dma_sems.at[SEM_KV, slot]
        if block_kv >= page_size:
            pages_per_block = block_kv // page_size
            for page in range(pages_per_block):
                num_tokens = _staged_tokens(
                    jnp.clip(
                        block.kv_len - block_start - page * page_size,
                        0,
                        page_size,
                    )
                )
                page_id = _physical_page(
                    block.request, kv_block * pages_per_block + page, num_tokens
                )
                pltpu.make_async_copy(
                    cache_token_rows.at[page_id, pl.ds(0, num_tokens)],
                    kv_ring.at[slot, pl.ds(page * page_size, num_tokens)],
                    kv_sem,
                ).start()
        else:
            blocks_per_page = page_size // block_kv
            num_tokens = _staged_tokens(
                jnp.clip(block.kv_len - block_start, 0, block_kv)
            )
            page_id = _physical_page(
                block.request, kv_block // blocks_per_page, num_tokens
            )
            page_offset = pl.multiple_of(
                lax.rem(kv_block, blocks_per_page) * block_kv, tile_rows
            )
            pltpu.make_async_copy(
                cache_token_rows.at[page_id, pl.ds(page_offset, num_tokens)],
                kv_ring.at[slot, pl.ds(0, num_tokens)],
                kv_sem,
            ).start()

    def _wait_kv_copies(block, kv_block, slot):
        # Note (david): every copy of an item signals its slot's semaphore,
        # which counts bytes, so one wait sized to the item's tokens covers
        # them all.
        num_tokens = _staged_tokens(
            jnp.clip(block.kv_len - kv_block * block_kv, 0, block_kv)
        )

        @pl.when(num_tokens > 0)
        def _wait():
            staged = kv_ring.at[slot, pl.ds(0, num_tokens)]
            pltpu.make_async_copy(
                staged, staged, dma_sems.at[SEM_KV, slot]
            ).wait()

    def _query_copies(block, slot):
        query_sem = dma_sems.at[SEM_QUERY, slot]
        tokens = pl.ds(block.q_start, block.q_count)
        return (
            pltpu.make_async_copy(
                q_nope_hbm.at[tokens],
                query_buffer.at[
                    slot, pl.ds(0, block.q_count), :, pl.ds(0, latent_dim)
                ],
                query_sem,
            ),
            pltpu.make_async_copy(
                q_pe_hbm.at[tokens],
                q_pe_stage.at[slot, pl.ds(0, block.q_count)],
                query_sem,
            ),
        )

    def _output_copies(block, slot):
        output_sem = dma_sems.at[SEM_OUTPUT, slot]
        out_copy = pltpu.make_async_copy(
            out_stage.at[slot, pl.ds(0, block.q_count)],
            out_hbm.at[pl.ds(block.q_start, block.q_count)],
            output_sem,
        )
        if return_lse:
            return (
                out_copy,
                pltpu.make_async_copy(
                    lse_stage.at[slot], lse_hbm.at[block.ordinal], output_sem
                ),
            )
        else:
            return (out_copy,)

    def _group_rows(group):
        """Row group group's slice of the score rows; group is a static int,
        or traced in the replay's rolled loop."""
        if isinstance(group, int):
            first_row = group * group_rows
        else:
            first_row = pl.multiple_of(group * group_rows, NUM_SUBLANES)
        return pl.ds(first_row, group_rows)

    def _key_window(first_column, num_columns):
        """The kv block's tokens [first_column, first_column + num_columns);
        first_column is a static int, or traced in a live-tile loop."""
        if isinstance(first_column, int):
            return pl.ds(first_column, num_columns)
        else:
            return pl.ds(
                pl.multiple_of(first_column, TILE_COLUMNS), num_columns
            )

    def _group_scores(
        query_slot, group, kv_block_ref, first_column, num_columns
    ):
        """(rows, scores): row group group's slice of the score rows and its
        log2-unit scores against the kv block's tokens [first_column,
        first_column + num_columns)."""
        queries = query_buffer[
            query_slot, pl.ds(group * group_tokens, group_tokens)
        ].reshape(group_rows, row_dim)
        scores = (
            lax.dot_general(
                queries,
                kv_block_ref[_key_window(first_column, num_columns)],
                NT_DIM_NUMBERS,
                preferred_element_type=jnp.float32,
            )
            * score_scale
        )
        return _group_rows(group), scores

    def _visible_keys(block, kv_block, group, first_column, num_columns):
        """(group_rows, num_columns) bool: which of the kv block's tokens
        [first_column, first_column + num_columns) each row of row group
        group sees."""
        column = lax.broadcasted_iota(jnp.int32, (group_rows, num_columns), 1)
        first_key = kv_block * block_kv + first_column
        if is_causal:
            # Note (david): row r is query token r // H, read from a table
            # staged once: an iota // H here is hoisted out of every row group
            # and its spills stall the MXU.
            token = lane_broadcast(
                row_token_ref[_group_rows(group)], num_columns
            )
            return token >= column + (first_key - block.first_position)
        else:
            return column < block.kv_len - first_key

    def _lane_sums(values):
        """values (rows, n * NUM_LANES) summed pairwise over its lane tiles
        into (rows, NUM_LANES): each lane keeps its own partial sum."""
        lane_tiles = [
            values[:, i : i + NUM_LANES]
            for i in range(0, values.shape[1], NUM_LANES)
        ]
        while len(lane_tiles) > 1:
            pairs = [a + b for a, b in zip(lane_tiles[0::2], lane_tiles[1::2])]
            lane_tiles = pairs + lane_tiles[2 * len(pairs) :]
        return lane_tiles[0]

    def _attend(
        block,
        kv_block,
        kv_block_ref,
        query_slot,
        num_columns,
        is_masked,
        first_column=0,
        may_anchor=True,
    ):
        """One item's static-anchor update of the q block's rows by the kv
        block's tokens [first_column, first_column + num_columns), a row group
        at a time, each group's P @ V after the next group's softmax.
        may_anchor False (a later live tile of an item) never anchors."""
        is_first_block = kv_block == 0

        def _softmax(group):
            rows, scores = _group_scores(
                query_slot, group, kv_block_ref, first_column, num_columns
            )
            if is_masked:
                visible = _visible_keys(
                    block, kv_block, group, first_column, num_columns
                )
            else:
                visible = None
            if may_anchor:
                if visible is None:
                    first_tile = scores[:, :NUM_LANES]
                else:
                    first_tile = jnp.where(
                        visible[:, :NUM_LANES],
                        scores[:, :NUM_LANES],
                        MASKED_SCORE,
                    )
                # Note (david): the first lane tile's max lets the exp wait
                # for one tile, key 0 (seen by every row) makes it a real
                # score, and later items keep it, so nothing is rescaled.
                anchor = jnp.where(
                    is_first_block,
                    lane_broadcast(
                        first_tile.max(axis=1, keepdims=True), NUM_LANES
                    ),
                    row_max_ref[rows],
                )
                row_max_ref[rows] = anchor
                previous_sum = jnp.where(is_first_block, 0.0, row_sum_ref[rows])
            else:
                anchor = row_max_ref[rows]
                previous_sum = row_sum_ref[rows]
            if visible is None:
                shifted = scores - lane_broadcast(anchor, num_columns)
            else:
                # Note (david): masked after the anchor is subtracted, so a
                # masked key is exactly 0 even for a row anchored on the mask.
                shifted = jnp.where(
                    visible,
                    scores - lane_broadcast(anchor, num_columns),
                    MASKED_SCORE,
                )
            probabilities = jnp.exp2(shifted)
            # Note (david): row_sum keeps a partial sum per lane, which the q
            # block's finish adds up once, so no item reduces across lanes.
            row_sum_ref[rows] = previous_sum + _lane_sums(probabilities)
            return rows, probabilities.astype(jnp.bfloat16)

        def _accumulate(rows, probabilities):
            pv = jnp.dot(
                probabilities,
                kv_block_ref[
                    _key_window(first_column, num_columns), :latent_dim
                ],
                preferred_element_type=jnp.float32,
            )
            if may_anchor:
                previous = jnp.where(is_first_block, 0.0, accumulator_ref[rows])
            else:
                previous = accumulator_ref[rows]
            accumulator_ref[rows] = previous + pv

        pending = _softmax(0)
        for group in range(1, num_groups):
            current = _softmax(group)
            _accumulate(*pending)
            pending = current
        _accumulate(*pending)

    def _wait_output(ordinal):
        for copy in _output_copies(
            _query_block(ordinal), lax.rem(ordinal, NUM_QUERY_SLOTS)
        ):
            copy.wait()

    def _finish_query_block(block, slot):
        previous_ordinal = block.ordinal - NUM_QUERY_SLOTS
        pl.when(previous_ordinal >= 0)(
            functools.partial(_wait_output, previous_ordinal)
        )
        row_sum = lane_broadcast(
            row_sum_ref[...].sum(axis=1, keepdims=True), NUM_LANES
        )
        # Note (david): an anchored row sum bounds each of its terms, so its
        # max trips the guard when a score may have overflowed, NaN included;
        # rows past q_count hold no output and stay out.
        real_row_sum = jnp.where(
            row_token_ref[...] < block.q_count, row_sum, 0.0
        )
        max_row_sum_ref[...] = jnp.maximum(
            max_row_sum_ref[...],
            real_row_sum.reshape(
                num_rows // NUM_SUBLANES, NUM_SUBLANES, NUM_LANES
            ).max(axis=0),
        )
        _stage_outputs(slot, row_max_ref[...], row_sum)
        for copy in _output_copies(block, slot):
            copy.start()

    def _stage_outputs(slot, row_max, row_sum):
        """The q block's normalized rows into out_stage at slot, and with
        return_lse their lse into lse_stage; row_sum is each row's total."""
        has_keys = row_max > 0.5 * MASKED_SCORE
        inverse_row_sum = jnp.where(has_keys, kv_scale / row_sum, 0.0)
        out_rows = (
            (accumulator_ref[...] * lane_broadcast(inverse_row_sum, latent_dim))
            .reshape(block_q, num_heads, latent_dim)
            .astype(out_stage.dtype)
        )
        if return_lse:
            lse = jnp.where(
                has_keys, (row_max + jnp.log2(row_sum)) * math.log(2.0), jnp.inf
            )
            # Note (david): a token's lse leaves as one lane-dense row, lane h
            # its head h, for a select and a sum over its heads' rows.
            row = lax.broadcasted_iota(jnp.int32, (num_rows, NUM_LANES), 0)
            head = row - row // num_heads * num_heads
            is_head_lane = (
                lax.broadcasted_iota(jnp.int32, (num_rows, NUM_LANES), 1)
                == head
            )
            token_lse = (
                jnp.where(is_head_lane, lse, 0.0)
                .reshape(block_q, num_heads, NUM_LANES)
                .sum(axis=1)
            )
            stores = ((out_stage, out_rows), (lse_stage, token_lse))
        else:
            stores = ((out_stage, out_rows),)
        for stage, value in stores:
            stage[slot] = value

    def _prepare_kv_block(block, kv_block, kv_slot):
        """The staged kv block's ref for the matmuls: rows past kv_len (the
        staged tail tile's, or a slot's stale or never written rows) zeroed,
        and an fp8 block widened to bf16."""
        valid_tokens = block.kv_len - kv_block * block_kv
        is_partial = valid_tokens < block_kv
        # Note (david): those rows may hold NaN, which a zero probability
        # still turns into a NaN P @ V; only a request's last block has them.
        if is_quantized:

            @pl.when(is_partial)
            def _widen_partial_block():
                token_row = lax.broadcasted_iota(
                    jnp.int32, widened_block.shape, 0
                )
                widened_block[...] = jnp.where(
                    token_row < valid_tokens,
                    kv_ring[kv_slot].astype(widened_block.dtype),
                    jnp.zeros(widened_block.shape, widened_block.dtype),
                )

            @pl.when(jnp.logical_not(is_partial))
            def _widen_block():
                widened_block[...] = kv_ring[kv_slot].astype(
                    widened_block.dtype
                )

            return widened_block
        else:

            @pl.when(is_partial)
            def _zero_rows_past_kv_len():
                token_row = lax.broadcasted_iota(
                    jnp.int32, kv_ring.shape[1:], 0
                )
                kv_ring[kv_slot] = jnp.where(
                    token_row < valid_tokens,
                    kv_ring[kv_slot],
                    jnp.zeros(kv_ring.shape[1:], kv_ring.dtype),
                )

            return kv_ring.at[kv_slot]

    def _run_item(item, carry):
        block_ordinal, kv_block, prefetch_ordinal, prefetch_kv_block = carry
        prefetch_item = item + NUM_KV_STAGES - 1
        prefetch_block = _query_block(prefetch_ordinal)

        # Note (david): the copies go to the slot the previous item has just
        # read, so they start after its last load.
        @pl.when(prefetch_item < num_items)
        def _prefetch():
            _start_kv_copies(
                prefetch_block,
                prefetch_kv_block,
                lax.rem(prefetch_item, NUM_KV_STAGES),
            )

        prefetch_ordinal, prefetch_kv_block = _next_item(
            prefetch_block, prefetch_kv_block
        )
        block = _query_block(block_ordinal)
        query_slot = lax.rem(block_ordinal, NUM_QUERY_SLOTS)
        kv_slot = lax.rem(item, NUM_KV_STAGES)

        @pl.when(kv_block == 0)
        def _start_query_block():
            for copy in _query_copies(block, query_slot):
                copy.wait()
            # Note (david): q_pe is narrower than a lane tile, which no copy
            # window can address, so it reaches its lanes through vector
            # stores.
            query_buffer[query_slot, :, :, pl.ds(latent_dim, rope_dim)] = (
                q_pe_stage[query_slot]
            )
            next_ordinal = block_ordinal + 1

            @pl.when(next_ordinal < num_q_blocks)
            def _start_next_query_copies():
                for copy in _query_copies(
                    _query_block(next_ordinal),
                    lax.rem(next_ordinal, NUM_QUERY_SLOTS),
                ):
                    copy.start()

        _wait_kv_copies(block, kv_block, kv_slot)
        kv_block_ref = _prepare_kv_block(block, kv_block, kv_slot)
        if is_causal:
            masked_from = block.first_position + 1
            live_end = block.first_position + block.q_count
        else:
            masked_from = block.kv_len
            live_end = block.kv_len
        needs_mask = (kv_block + 1) * block_kv > masked_from
        attend = functools.partial(
            _attend, block, kv_block, kv_block_ref, query_slot
        )
        if block_kv % TILE_COLUMNS == 0 and block_kv > TILE_COLUMNS:
            num_tiles = block_kv // TILE_COLUMNS
            # Note (david): no row looks past live_end. A loop step ends a
            # basic block and drains the MXU, so only an item with dead tiles
            # loops.
            live_tiles = jnp.clip(
                (live_end - kv_block * block_kv + TILE_COLUMNS - 1)
                // TILE_COLUMNS,
                1,
                num_tiles,
            )
            has_dead_tiles = live_tiles < num_tiles

            @pl.when(jnp.logical_and(needs_mask, has_dead_tiles))
            def _attend_live_tiles():
                attend(TILE_COLUMNS, True)
                if num_tiles > 2:

                    def _attend_tile(tile, carry):
                        attend(
                            TILE_COLUMNS,
                            True,
                            first_column=tile * TILE_COLUMNS,
                            may_anchor=False,
                        )
                        return carry

                    lax.fori_loop(1, live_tiles, _attend_tile, 0)

            pl.when(
                jnp.logical_and(needs_mask, jnp.logical_not(has_dead_tiles))
            )(functools.partial(attend, block_kv, True))
        else:
            pl.when(needs_mask)(functools.partial(attend, block_kv, True))
        pl.when(jnp.logical_not(needs_mask))(
            functools.partial(attend, block_kv, False)
        )
        pl.when(kv_block == block.num_kv_blocks - 1)(
            functools.partial(_finish_query_block, block, query_slot)
        )
        return (
            *_next_item(block, kv_block),
            prefetch_ordinal,
            prefetch_kv_block,
        )

    def _replay_group(block, kv_block, kv_block_ref, group, carry):
        """Row group group's online-softmax update by the whole kv block."""
        rows, scores = _group_scores(0, group, kv_block_ref, 0, block_kv)
        visible = _visible_keys(block, kv_block, group, 0, block_kv)
        is_first_block = kv_block == 0
        previous_max = jnp.where(
            is_first_block, MASKED_SCORE, row_max_ref[rows]
        )
        row_max = jnp.maximum(
            previous_max,
            lane_broadcast(
                jnp.where(visible, scores, MASKED_SCORE).max(
                    axis=1, keepdims=True
                ),
                NUM_LANES,
            ),
        )
        rescale = jnp.exp2(previous_max - row_max)
        probabilities = jnp.exp2(
            jnp.where(
                visible,
                scores - lane_broadcast(row_max, block_kv),
                MASKED_SCORE,
            )
        )
        row_max_ref[rows] = row_max
        row_sum_ref[rows] = rescale * jnp.where(
            is_first_block, 0.0, row_sum_ref[rows]
        ) + _lane_sums(probabilities)
        pv = jnp.dot(
            probabilities.astype(jnp.bfloat16),
            kv_block_ref[:, :latent_dim],
            preferred_element_type=jnp.float32,
        )
        accumulator_ref[rows] = (
            jnp.where(is_first_block, 0.0, accumulator_ref[rows])
            * lane_broadcast(rescale, latent_dim)
            + pv
        )
        return carry

    def _replay_query_block(ordinal, carry):
        """q block ordinal again with the online softmax, a kv block at a
        time through ring slot 0 and a row group at a time, its outputs
        rewritten."""
        block = _query_block(ordinal)
        for copy in _query_copies(block, 0):
            copy.start()
        for copy in _query_copies(block, 0):
            copy.wait()
        query_buffer[0, :, :, pl.ds(latent_dim, rope_dim)] = q_pe_stage[0]

        def _replay_kv_block(kv_block, carry):
            _start_kv_copies(block, kv_block, 0)
            _wait_kv_copies(block, kv_block, 0)
            kv_block_ref = _prepare_kv_block(block, kv_block, 0)
            return lax.fori_loop(
                0,
                num_groups,
                functools.partial(_replay_group, block, kv_block, kv_block_ref),
                carry,
            )

        lax.fori_loop(0, block.num_kv_blocks, _replay_kv_block, 0)
        _stage_outputs(
            0,
            row_max_ref[...],
            lane_broadcast(
                row_sum_ref[...].sum(axis=1, keepdims=True), NUM_LANES
            ),
        )
        for copy in _output_copies(block, 0):
            copy.start()
        for copy in _output_copies(block, 0):
            copy.wait()
        return carry

    # Note (david): the cache's padding lanes are zero, but NaN * 0 is NaN, so
    # the queries' padding lanes must not hold the scratch's garbage.
    query_buffer[:, :, :, pl.ds(latent_dim, row_dim - latent_dim)] = jnp.zeros(
        (*query_buffer.shape[:-1], row_dim - latent_dim), query_buffer.dtype
    )
    max_row_sum_ref[...] = jnp.zeros(max_row_sum_ref.shape, jnp.float32)
    row_token_ref[...] = (
        lax.broadcasted_iota(jnp.int32, row_token_ref.shape, 0) // num_heads
    )
    first_block = _query_block(jnp.int32(0))

    @pl.when(num_q_blocks > 0)
    def _start_first_query_copies():
        for copy in _query_copies(first_block, 0):
            copy.start()

    prefetch_ordinal = jnp.int32(0)
    prefetch_kv_block = jnp.int32(0)
    for item in range(NUM_KV_STAGES - 1):
        prefetch_block = _query_block(prefetch_ordinal)
        pl.when(item < num_items)(
            functools.partial(
                _start_kv_copies, prefetch_block, prefetch_kv_block, item
            )
        )
        prefetch_ordinal, prefetch_kv_block = _next_item(
            prefetch_block, prefetch_kv_block
        )
    lax.fori_loop(
        0,
        num_items,
        _run_item,
        (jnp.int32(0), jnp.int32(0), prefetch_ordinal, prefetch_kv_block),
    )
    # Note (david): the kernel must not exit with copies in flight, and the
    # last q blocks' outputs still leave; each earlier one was waited for by
    # the q block that reused its slot.
    for distance in range(NUM_QUERY_SLOTS, 0, -1):
        ordinal = num_q_blocks - distance
        pl.when(ordinal >= 0)(functools.partial(_wait_output, ordinal))

    # Note (david): the ring is refilled before the guard is known, so the
    # replay reruns every q block; it runs only when the guard trips, so it is
    # one row group's code in rolled loops.
    should_replay_ref[0] = jnp.any(
        jnp.logical_not(max_row_sum_ref[...] <= 2.0**guard_threshold)
    ).astype(jnp.int32)

    @pl.when(should_replay_ref[0] > 0)
    def _replay():
        lax.fori_loop(0, num_q_blocks, _replay_query_block, 0)


def mla_prefill_pallas(
    q_nope: jax.Array,
    q_pe: jax.Array,
    kv_cache: jax.Array,
    cu_q_lens: jax.Array,
    kv_lens: jax.Array,
    block_table: jax.Array,
    *,
    kv_scale: jax.Array | None,
    sm_scale: float,
    causal: bool,
    return_lse: bool,
    max_seqlen_q: int,
    interpret: bool,
    config: MlaPrefillConfig,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """flash_mla_prefill's pallas_call on operands it has checked, at config.

    Returns out, or (out, lse) with return_lse, as flash_mla_prefill does.
    """
    num_tokens, num_heads, latent_dim = q_nope.shape
    rope_dim = q_pe.shape[-1]
    batch = kv_lens.shape[0]
    _, page_words, packing, row_dim = kv_cache.shape
    page_size = page_words * packing
    pages_per_seq = block_table.shape[1]
    cache_dtype = kv_cache.dtype
    is_quantized = kv_scale is not None
    block_q, block_kv, row_groups = config
    tile_rows = cache_tile_rows(cache_dtype)
    if (
        block_q < 1
        or block_q % row_groups
        or block_kv % NUM_LANES
        or block_kv % tile_rows
        or (block_kv % page_size and page_size % block_kv)
    ):
        raise ValueError(
            f"{config} needs a q block of at least one token in whole row"
            " groups and a kv block of whole lane tiles and"
            f" {tile_rows}-token cache tiles that holds whole"
            f" {page_size}-token pages or a whole fraction of one."
        )
    elif (
        batch >= 1 << (31 - REQUEST_SHIFT)
        or pl.cdiv(max_seqlen_q, block_q) > Q_BLOCK_MASK
    ):
        raise ValueError(
            f"batch {batch} or {pl.cdiv(max_seqlen_q, block_q)} q blocks a"
            " request overflow a schedule entry."
        )
    else:
        # Note (david): a request of n tokens takes at most n // block_q + 1 q
        # blocks, so the table holds what any cu_q_lens within the bounds needs.
        max_q_blocks = max(
            1,
            min(
                num_tokens // block_q + min(batch, num_tokens),
                batch * pl.cdiv(max_seqlen_q, block_q),
            ),
        )
        schedule, request_first_q_block = mla_prefill_schedule(
            cu_q_lens,
            kv_lens,
            block_q=block_q,
            block_kv=block_kv,
            max_q_blocks=max_q_blocks,
            causal=causal,
        )
        # Note (david): the Pallas interpreter takes no ref reshape, so an
        # interpreted call hands the kernel the token rows.
        if interpret:
            cache_operand = kv_cache.reshape(
                kv_cache.shape[0], page_size, row_dim
            )
        else:
            cache_operand = kv_cache
        hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
        out_shape = jax.ShapeDtypeStruct(q_nope.shape, q_nope.dtype)
        if return_lse:
            out_shapes = [
                out_shape,
                jax.ShapeDtypeStruct(
                    (max_q_blocks, block_q, NUM_LANES), jnp.float32
                ),
            ]
            lse_scratch = [
                pltpu.VMEM((NUM_QUERY_SLOTS, block_q, NUM_LANES), jnp.float32)
            ]
        else:
            out_shapes = [out_shape]
            lse_scratch = []
        if is_quantized:
            quantized_scratch = [pltpu.VMEM((block_kv, row_dim), jnp.bfloat16)]
            quantized_inputs = [kv_scale.reshape(1).astype(jnp.float32)]
        else:
            quantized_scratch = []
            quantized_inputs = []
        num_rows = block_q * num_heads
        scratch_shapes = [
            pltpu.VMEM((NUM_KV_STAGES, block_kv, row_dim), cache_dtype),
            pltpu.VMEM(
                (NUM_QUERY_SLOTS, block_q, num_heads, row_dim), q_nope.dtype
            ),
            pltpu.VMEM(
                (NUM_QUERY_SLOTS, block_q, num_heads, rope_dim), q_pe.dtype
            ),
            pltpu.VMEM(
                (NUM_QUERY_SLOTS, block_q, num_heads, latent_dim), q_nope.dtype
            ),
            pltpu.VMEM((num_rows, NUM_LANES), jnp.float32),
            pltpu.VMEM((num_rows, NUM_LANES), jnp.float32),
            pltpu.VMEM((num_rows, latent_dim), jnp.float32),
            pltpu.VMEM((num_rows, NUM_LANES), jnp.int32),
            pltpu.VMEM((NUM_SUBLANES, NUM_LANES), jnp.float32),
            pltpu.SMEM((1,), jnp.int32),
            pltpu.SemaphoreType.DMA((3, max(NUM_KV_STAGES, NUM_QUERY_SLOTS))),
            *quantized_scratch,
            *lse_scratch,
        ]
        kernel = functools.partial(
            mla_prefill_kernel,
            page_size=page_size,
            pages_per_seq=pages_per_seq,
            max_q_blocks=max_q_blocks,
            config=config,
            num_heads=num_heads,
            sm_scale_log2=sm_scale * LOG2E,
            is_causal=causal,
            return_lse=return_lse,
            is_quantized=is_quantized,
            # Note (david): sized on the block table's capacity, which bounds
            # every kv_lens.
            guard_threshold=overflow_guard_threshold(pages_per_seq * page_size),
        )
        scalar_prefetches = (
            schedule,
            cu_q_lens,
            kv_lens,
            block_table.reshape(-1),
        )
        inputs = [q_nope, q_pe, cache_operand, *quantized_inputs]
        in_specs = [hbm_spec] * 3 + [
            pl.BlockSpec(memory_space=pltpu.SMEM)
        ] * len(quantized_inputs)
        call = pl.pallas_call(
            kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=len(scalar_prefetches),
                grid=(1,),
                in_specs=in_specs,
                out_specs=[hbm_spec] * len(out_shapes),
                scratch_shapes=scratch_shapes,
            ),
            out_shape=out_shapes,
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("arbitrary",),
                vmem_limit_bytes=vmem_limit_bytes(),
            ),
            name="flash_mla_prefill",
            interpret=pltpu.InterpretParams() if interpret else False,
        )
        out, *lse_blocks = call(*scalar_prefetches, *inputs)
        if not return_lse:
            return out
        else:
            # Note (david): q block j keeps token i's lse at [j, i, head]; this
            # gather back to (H, T) costs far less than one cache read.
            token = jnp.arange(num_tokens, dtype=jnp.int32)
            request = jnp.minimum(
                jnp.searchsorted(cu_q_lens[1:], token, side="right"), batch - 1
            )
            token_in_request = token - cu_q_lens[request]
            ordinal = jnp.clip(
                request_first_q_block[request] + token_in_request // block_q,
                0,
                max_q_blocks - 1,
            )
            lse = lse_blocks[0][ordinal, token_in_request % block_q, :num_heads]
            return out, lse.T
