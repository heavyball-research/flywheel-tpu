"""Absorbed MLA decode: append new latent tokens, then attend as MQA."""

import enum
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
from ..dispatch import get_device_variant_name, get_tuned_mla_config
from ..softmax import overflow_guard_threshold
from .decode import lane_broadcast

LATENT_DIM = 512
ROPE_DIM = 64
# Note (david): MLA v2's cache row, so both kernels read one cache: the latent,
# the rope, then zeros that pad the row to whole lane tiles.
CACHE_ROW_DIM = 640
# Note (david): an item computes all its block_kv columns whatever its
# request's length and pays a fixed cost of its own, so a block longer than
# most requests wastes columns and a short one multiplies the items. A batch
# whose max_seqlen_k fits WHOLE_REQUEST_TOKENS takes one item a request. Past
# it the bound sits past most of the batch's requests, so the block takes a
# BOUND_DIVISOR-th of it. A batch of fewer than MIN_ITEMS items takes shorter
# blocks, so that its steps' copies and compute still overlap.
WHOLE_REQUEST_TOKENS = 2048
BOUND_DIVISOR = 4
MIN_BLOCK_KV = 512
MIN_ITEMS = 64
# Note (david): v7x keeps longer blocks than v6e, as shorter ones slow down the
# long requests that take most of a batch's time there.
MAX_BLOCK_KV = {"TPU v6e": 1024, "TPU v7": 2048}
# Note (david): the Pallas interpreter has no chip to ask, so it takes v6e's
# blocks, as vmem_limit_bytes takes v6e's limit.
INTERPRET_CHIP = "TPU v6e"
# Note (david): longer steps and more items a step stopped paying off.
TARGET_STEP_TOKENS = 16384
MAX_ITEMS_PER_STEP = 8
# Note (david): a third ring slot gained nothing, and a step of one group loses
# the overlap of one group's PV with the next group's softmax.
NUM_STAGES = 2
NUM_GROUPS = 2
# Note (david): the replay attends an item a chunk at a time, the largest whole
# fraction of block_kv up to REPLAY_CHUNK_TILES lane tiles, so its code is one
# chunk's.
REPLAY_CHUNK_TILES = 4
# Note (david): the append runs in chunks through NUM_APPEND_BUFFERS buffers,
# so its VMEM does not grow with the batch.
APPEND_CHUNK_REQUESTS = 64
NUM_APPEND_BUFFERS = 3
# Note (david): the rest of the VMEM limit is Mosaic's room for a step's score
# temporaries.
VMEM_BUDGET_FRACTION = 0.75
# Note (david): finite, not -inf, so an item that sees no key (an inactive
# request's or a padding item's) leaves every running value finite.
MASKED_SCORE = -1e30
FP8_MAX = float(jnp.finfo(jnp.float8_e4m3fn).max)
REQUEST_SHIFT = 16
KV_BLOCK_MASK = (1 << REQUEST_SHIFT) - 1
SEM_KV = 0
SEM_QUERY = 1
SEM_OUTPUT = 2
SEM_NEW_TOKENS = 3
SEM_APPEND_READ = 4
SEM_APPEND_WRITE = 5
NUM_SEM_ROWS = 6


class MlaDecodeConfig(NamedTuple):
    """Tokens an item and items a step."""

    block_kv: int
    items_per_step: int


class KvSource(enum.Enum):
    """Where a step's copies take an item's tokens from."""

    # Note (david): the tokens before the appended tiles, which do not wait
    # for the append.
    CACHE_BEFORE_APPEND = enum.auto()
    # Note (david): the appended tiles, from the append buffers.
    APPEND_BUFFER = enum.auto()
    # Note (david): every token, after the append's write-back.
    CACHE = enum.auto()


class ScheduleItem(NamedTuple):
    """Scalars of one item at a static position in its step."""

    step_position: int
    request: jax.Array
    kv_block: jax.Array
    kv_len: jax.Array
    is_first_block: jax.Array
    is_last_block: jax.Array
    query_slot: jax.Array


def cache_packing(dtype: jnp.dtype) -> int:
    """Tokens a 32-bit cache word holds."""
    return 4 // jnp.dtype(dtype).itemsize


def cache_tile_rows(dtype: jnp.dtype) -> int:
    """Rows of a (rows, NUM_LANES) VMEM tile in dtype."""
    return NUM_SUBLANES * cache_packing(dtype)


def mla_block_kv(
    page_size: int, pages_per_seq: int, target_block_kv: int
) -> int:
    """Tokens of one item: a whole fraction of a page when a page holds
    target_block_kv tokens whole, else whole pages and whole lane tiles about
    target_block_kv long, no more than a request's pages hold."""
    if page_size >= target_block_kv and page_size % target_block_kv == 0:
        return target_block_kv
    else:
        block_unit = math.lcm(page_size, NUM_LANES)
        max_units = pl.cdiv(pages_per_seq * page_size, block_unit)
        return block_unit * max(
            1, min(target_block_kv // block_unit, max_units)
        )


def mla_append_chunk(batch: int, items_per_step: int) -> int:
    """Requests in one chunk of the append, so that the buffers hold every
    request of the first NUM_STAGES steps, which read them."""
    return min(
        batch,
        max(
            APPEND_CHUNK_REQUESTS,
            pl.cdiv(NUM_STAGES * items_per_step, NUM_APPEND_BUFFERS),
        ),
    )


def max_append_tokens(query_tokens: int, cache_dtype: jnp.dtype) -> int:
    """Tokens of the cache tiles that hold a request's new tokens, at most."""
    tile_rows = cache_tile_rows(cache_dtype)
    # Note (david): the first new token may sit anywhere in its tile.
    return (pl.cdiv(query_tokens - 1, tile_rows) + 1) * tile_rows


def mla_vmem_bytes(
    config: MlaDecodeConfig,
    *,
    batch: int,
    num_query_rows: int,
    query_tokens: int,
    cache_dtype: jnp.dtype,
    return_lse: bool,
) -> int:
    """VMEM bytes of mla_decode_pallas's scratch at config, plus a step's
    score temporaries."""
    block_kv, items_per_step = config
    num_groups = min(NUM_GROUPS, items_per_step)
    itemsize = jnp.dtype(cache_dtype).itemsize
    tile_rows = cache_tile_rows(cache_dtype)
    padded_rows = pl.cdiv(num_query_rows, BF16_TILE_ROWS) * BF16_TILE_ROWS
    padded_tokens = pl.cdiv(query_tokens, tile_rows) * tile_rows
    append_rows = NUM_APPEND_BUFFERS * mla_append_chunk(batch, items_per_step)
    kv_ring_bytes = (
        NUM_STAGES * items_per_step * block_kv * CACHE_ROW_DIM * itemsize
    )
    query_ring_bytes = (
        NUM_STAGES
        * items_per_step
        * padded_rows
        * (CACHE_ROW_DIM + NUM_LANES)
        * BF16_BYTES
    )
    new_token_bytes = (
        (append_rows + 1) * padded_tokens * (LATENT_DIM + NUM_LANES) * itemsize
    )
    append_tile_bytes = (
        append_rows
        * max_append_tokens(query_tokens, cache_dtype)
        * CACHE_ROW_DIM
        * itemsize
    )
    output_ring_bytes = (
        2 * items_per_step * padded_rows * LATENT_DIM * BF16_BYTES
    )
    softmax_state_bytes = padded_rows * (3 * NUM_LANES + LATENT_DIM) * F32_BYTES
    if return_lse:
        lse_bytes = (
            2 * pl.cdiv(batch, NUM_LANES) * padded_rows * NUM_LANES * F32_BYTES
        )
    else:
        lse_bytes = 0
    # Note (david): a group's f32 scores and its probabilities are live at
    # once, and with more than one group the next group's too.
    live_items = pl.cdiv(items_per_step, num_groups) + int(num_groups > 1)
    score_bytes = live_items * padded_rows * block_kv * (F32_BYTES + itemsize)
    return (
        kv_ring_bytes
        + query_ring_bytes
        + new_token_bytes
        + append_tile_bytes
        + output_ring_bytes
        + softmax_state_bytes
        + lse_bytes
        + score_bytes
    )


def default_mla_config(
    *,
    batch: int,
    page_size: int,
    pages_per_seq: int,
    num_query_rows: int,
    query_tokens: int,
    cache_dtype: jnp.dtype,
    return_lse: bool,
    max_seqlen_k: int,
) -> MlaDecodeConfig:
    """The config of one build: the target block and items cap of its
    tuned_configs.json entry, else the chip's block for max_seqlen_k, the
    call's bound on kv_lens. The block is fitted to the pages, and a step
    holds TARGET_STEP_TOKENS tokens but no more items than the cap or than
    the batch can have; past the VMEM budget the step halves, then the block.
    """
    tuned = get_tuned_mla_config(
        cache_dtype=cache_dtype,
        num_query_rows=num_query_rows,
        batch=batch,
        max_seqlen_k=max_seqlen_k,
    )
    if tuned is not None:
        target_block_kv, max_items_per_step = tuned
    else:
        if jax.default_backend() == "tpu":
            chip = get_device_variant_name()
        else:
            chip = INTERPRET_CHIP
        if chip not in MAX_BLOCK_KV:
            raise NotImplementedError(
                f"flash_mla_decode has no default config for {chip}."
            )
        bound = max(MIN_BLOCK_KV, next_pow2(max_seqlen_k))
        if bound <= WHOLE_REQUEST_TOKENS:
            target_block_kv = bound
        else:
            target_block_kv = min(
                max(MIN_BLOCK_KV, bound // BOUND_DIVISOR), MAX_BLOCK_KV[chip]
            )
        while (
            target_block_kv > MIN_BLOCK_KV
            and batch * pl.cdiv(max_seqlen_k, target_block_kv) < MIN_ITEMS
        ):
            target_block_kv //= 2
        max_items_per_step = MAX_ITEMS_PER_STEP
    shape = {
        "batch": batch,
        "num_query_rows": num_query_rows,
        "query_tokens": query_tokens,
        "cache_dtype": cache_dtype,
        "return_lse": return_lse,
    }
    vmem_budget = VMEM_BUDGET_FRACTION * vmem_limit_bytes()
    bound_pages = pl.cdiv(
        min(max_seqlen_k, pages_per_seq * page_size), page_size
    )
    while True:
        block_kv = mla_block_kv(page_size, bound_pages, target_block_kv)
        # Note (david): a step computes every item it holds, the padding past
        # the schedule's last item too, so it holds no more items than the
        # batch can have.
        items_per_step = max(
            1,
            min(
                max_items_per_step,
                TARGET_STEP_TOKENS // block_kv,
                batch * pl.cdiv(max_seqlen_k, block_kv),
            ),
        )
        config = MlaDecodeConfig(block_kv, items_per_step)
        while (
            config.items_per_step > 1
            and mla_vmem_bytes(config, **shape) > vmem_budget
        ):
            config = MlaDecodeConfig(block_kv, config.items_per_step // 2)
        if mla_vmem_bytes(config, **shape) <= vmem_budget:
            return config
        # Note (david): a block is whole lane tiles.
        elif (
            target_block_kv // 2 < NUM_LANES
            or mla_block_kv(page_size, bound_pages, target_block_kv // 2)
            >= block_kv
        ):
            raise ValueError(
                f"flash_mla_decode needs more than {VMEM_BUDGET_FRACTION} of"
                f" the {vmem_limit_bytes()}-byte VMEM limit at one"
                f" {block_kv}-token item a step (batch {batch},"
                f" {num_query_rows} query rows)."
            )
        else:
            target_block_kv //= 2


def mla_decode_schedule(
    kv_lens: jax.Array, *, block_kv: int, max_items: int
) -> jax.Array:
    """The call's items as one flat int32 array, a pure function of kv_lens,
    so XLA builds it once for every layer that shares them.

    Entry i < max_items is item i's request << REQUEST_SHIFT | KV block; the
    last entry is the item count. A request takes cdiv(kv_lens, block_kv)
    items, an inactive one a single item that only writes its zero output.
    """
    batch = kv_lens.shape[0]
    request_items = jnp.maximum((kv_lens + block_kv - 1) // block_kv, 1)
    request_item_end = jnp.cumsum(request_items)
    request_item_start = request_item_end - request_items
    num_items = request_item_end[-1]
    item_index = jnp.arange(max_items, dtype=jnp.int32)
    # Note (david): every request has an item, so a running count of the
    # request starts gives each item its request.
    request_starts = (
        jnp.zeros((max_items,), jnp.int32)
        .at[request_item_start[1:]]
        .add(1, mode="drop")
    )
    request = jnp.cumsum(request_starts)
    kv_block = item_index - request_item_start[request]
    # Note (david): the entries past the item count pad the last step; they
    # continue the last request with a block past its keys, so they copy
    # nothing and leave its running state as it was.
    is_padding = item_index >= num_items
    request = jnp.where(is_padding, batch - 1, request)
    kv_block = jnp.where(is_padding, request_items[batch - 1], kv_block)
    return jnp.concatenate(
        [request * (1 << REQUEST_SHIFT) + kv_block, num_items[None]]
    ).astype(jnp.int32)


def mla_decode_kernel(
    schedule_ref: jax.Array,
    kv_lens_ref: jax.Array,
    block_table_ref: jax.Array,
    q_nope_hbm: jax.Array,
    q_pe_hbm: jax.Array,
    new_kv_c_hbm: jax.Array,
    new_k_pe_hbm: jax.Array,
    _: jax.Array,
    *refs: jax.Array,
    batch: int,
    page_size: int,
    pages_per_seq: int,
    config: MlaDecodeConfig,
    append_chunk: int,
    num_heads: int,
    sm_scale_log2: float,
    s_dtype: jnp.dtype,
    is_causal: bool,
    return_lse: bool,
    is_quantized: bool,
    guard_threshold: float,
) -> None:
    """Append every request's new tokens, run the schedule's steps with the
    static-anchor softmax, and replay them with the online softmax if the
    overflow guard tripped.

    Query, new-token and output rows are token-major (row t * H + h). With
    is_quantized, refs start with the SMEM (1,) f32 kv_scale; with
    return_lse, the lse block holds request b at [b // 128, :, b % 128].
    The append runs in chunks of append_chunk requests, and request b's new
    tokens and tiles sit at row b % (NUM_APPEND_BUFFERS * append_chunk) of
    the append buffers.
    """
    # Note (david): K and V share the fp8 cache's one kv_scale, so it folds
    # into the score scale and the output normalization instead of scaling
    # cache elements.
    if is_quantized:
        kv_scale_ref, output_hbm, cache_hbm, *refs = refs
        kv_scale = kv_scale_ref[0]
        score_scale = kv_scale * sm_scale_log2
    else:
        output_hbm, cache_hbm, *refs = refs
        kv_scale = 1.0
        score_scale = sm_scale_log2
    if return_lse:
        lse_ref, *refs = refs
    else:
        lse_ref = None
    (
        kv_ring,
        query_ring,
        q_pe_ring,
        new_latent,
        new_rope,
        new_token_rows,
        append_tiles,
        output_ring,
        row_max_ref,
        row_sum_ref,
        accumulator_ref,
        max_row_sum_ref,
        should_replay_ref,
        dma_sems,
    ) = refs
    block_kv, items_per_step = config
    num_groups = min(NUM_GROUPS, items_per_step)
    num_query_rows, row_dim = query_ring.shape[1:]
    latent_dim = new_latent.shape[-1]
    rope_dim = new_rope.shape[-1]
    query_tokens = new_latent.shape[1]
    cache_dtype = kv_ring.dtype
    tile_rows = cache_tile_rows(cache_dtype)
    num_query_slots = query_ring.shape[0]
    num_output_slots = output_ring.shape[0]
    max_items = schedule_ref.shape[0] - 1
    num_items = schedule_ref[max_items]
    num_steps = pl.cdiv(num_items, items_per_step)
    num_append_rows = append_tiles.shape[0]
    num_chunks = pl.cdiv(batch, append_chunk)
    # Note (david): the word layout is a row-major reshape of the token rows,
    # and a copy into the staging's own tiles converts the tiling at full
    # bandwidth, so every copy addresses token rows. The Pallas interpreter
    # takes no ref reshape and is handed the token rows.
    if len(cache_hbm.shape) == 3:
        cache_token_rows = cache_hbm
    else:
        cache_token_rows = cache_hbm.reshape(
            cache_hbm.shape[0], page_size, row_dim
        )

    def _tile_aligned(num_tokens):
        # Note (david): a copy window on the token axis must be whole staging
        # tiles.
        return pl.multiple_of(num_tokens, tile_rows)

    def _schedule_item(step, step_position):
        # Note (david): the copies of the steps past the last one still read
        # an entry.
        entry = schedule_ref[
            jnp.minimum(step * items_per_step + step_position, max_items - 1)
        ]
        request = lax.shift_right_logical(entry, REQUEST_SHIFT)
        kv_block = entry & KV_BLOCK_MASK
        kv_len = kv_lens_ref[request]
        last_kv_block = jnp.maximum(pl.cdiv(kv_len, block_kv), 1) - 1
        return ScheduleItem(
            step_position,
            request,
            kv_block,
            kv_len,
            kv_block == 0,
            kv_block == last_kv_block,
            lax.rem(request, num_query_slots),
        )

    def _append_tile_range(kv_len):
        """[start, end) of the cache tiles that hold the request's new
        tokens; empty for an inactive request."""
        start = jnp.maximum(kv_len - query_tokens, 0) // tile_rows * tile_rows
        end = (kv_len + tile_rows - 1) // tile_rows * tile_rows
        return start, end

    def _physical_page(request, page_index, num_tokens):
        # Note (david): a copy of no tokens still forms its address, so it
        # reads page 0, never a block_table entry past the request's pages.
        table_index = request * pages_per_seq + jnp.minimum(
            page_index, pages_per_seq - 1
        )
        return jnp.where(num_tokens > 0, block_table_ref[table_index], 0)

    def _append_row(request):
        """Row of the request's new tokens and tiles in the append buffers."""
        return lax.rem(request, num_append_rows)

    def _start_item_copies(
        item, slot, is_live_step, *, kv_source, copy_queries
    ):
        """Start the item's copies into its place in the ring slot: its
        tokens from kv_source, and its request's query rows when the item
        starts the request."""
        append_start, append_end = _append_tile_range(item.kv_len)
        item_start = item.kv_block * block_kv
        item_end = item_start + block_kv
        slot_offset = item.step_position * block_kv
        kv_sem = dma_sems.at[SEM_KV, slot]
        if kv_source in (KvSource.CACHE_BEFORE_APPEND, KvSource.CACHE):
            if kv_source == KvSource.CACHE_BEFORE_APPEND:
                cached_end = jnp.minimum(append_start, item_end)
            else:
                cached_end = jnp.minimum(append_end, item_end)
            if block_kv >= page_size:
                pages_per_item = block_kv // page_size
                for page in range(pages_per_item):
                    page_start = item_start + page * page_size
                    num_tokens = _tile_aligned(
                        jnp.where(
                            is_live_step,
                            jnp.clip(cached_end - page_start, 0, page_size),
                            0,
                        )
                    )
                    page_id = _physical_page(
                        item.request,
                        item.kv_block * pages_per_item + page,
                        num_tokens,
                    )
                    pltpu.make_async_copy(
                        cache_token_rows.at[page_id, pl.ds(0, num_tokens)],
                        kv_ring.at[
                            slot,
                            pl.ds(slot_offset + page * page_size, num_tokens),
                        ],
                        kv_sem,
                    ).start()
            else:
                items_per_page = page_size // block_kv
                num_tokens = _tile_aligned(
                    jnp.where(
                        is_live_step,
                        jnp.clip(cached_end - item_start, 0, block_kv),
                        0,
                    )
                )
                page_id = _physical_page(
                    item.request,
                    item.kv_block // items_per_page,
                    num_tokens,
                )
                page_offset = _tile_aligned(
                    lax.rem(item.kv_block, items_per_page) * block_kv
                )
                pltpu.make_async_copy(
                    cache_token_rows.at[
                        page_id, pl.ds(page_offset, num_tokens)
                    ],
                    kv_ring.at[slot, pl.ds(slot_offset, num_tokens)],
                    kv_sem,
                ).start()
        elif kv_source == KvSource.APPEND_BUFFER:
            copy_start = jnp.maximum(append_start, item_start)
            num_tokens = _tile_aligned(
                jnp.where(
                    is_live_step,
                    jnp.clip(
                        jnp.minimum(append_end, item_end) - copy_start,
                        0,
                        block_kv,
                    ),
                    0,
                )
            )

            # Note (david): only a request's last item or two hold its
            # appended tiles, and a copy costs its issue whatever it
            # moves, so the other items issue none.
            @pl.when(num_tokens > 0)
            def _start_append_tile_copy():
                pltpu.make_async_copy(
                    append_tiles.at[
                        _append_row(item.request),
                        pl.ds(
                            _tile_aligned(copy_start - append_start),
                            num_tokens,
                        ),
                    ],
                    kv_ring.at[
                        slot,
                        pl.ds(
                            slot_offset
                            + _tile_aligned(copy_start - item_start),
                            num_tokens,
                        ),
                    ],
                    kv_sem,
                ).start()

        if copy_queries:
            is_request_start = jnp.logical_and(
                jnp.logical_and(is_live_step, item.is_first_block),
                item.kv_len > 0,
            )

            @pl.when(is_request_start)
            def _start_query_copies():
                query_sem = dma_sems.at[SEM_QUERY, slot]
                pltpu.make_async_copy(
                    q_nope_hbm.at[item.request],
                    query_ring.at[item.query_slot, :, pl.ds(0, latent_dim)],
                    query_sem,
                ).start()
                pltpu.make_async_copy(
                    q_pe_hbm.at[item.request],
                    q_pe_ring.at[item.query_slot],
                    query_sem,
                ).start()

    def _start_step_copies(step, slot, *, kv_source, copy_queries):
        is_live_step = step < num_steps
        for step_position in range(items_per_step):
            _start_item_copies(
                _schedule_item(step, step_position),
                slot,
                is_live_step,
                kv_source=kv_source,
                copy_queries=copy_queries,
            )

    def _zero_rows_past_kv_len(slot, items):
        """Zero the staged rows past each request's kv_len in its last tile.

        These rows hold whatever the cache holds there, possibly NaN, and
        P @ V multiplies them by 0. The other staged rows hold keys or the
        zeros and keys of earlier steps, so they stay finite.
        """
        # Note (david): branch-free: an item with no row to zero rewrites its
        # first tile unchanged.
        for item in items:
            item_start = item.kv_block * block_kv
            needs_zeroing = jnp.logical_and(
                item.is_last_block, lax.rem(item.kv_len, tile_rows) != 0
            )
            last_tile_start = (item.kv_len - 1) // tile_rows * tile_rows
            tile_offset = jnp.where(
                needs_zeroing, last_tile_start - item_start, 0
            )
            kept_rows = jnp.where(
                needs_zeroing,
                item.kv_len - item_start - tile_offset,
                tile_rows,
            )
            rows = kv_ring.at[
                slot,
                pl.ds(
                    _tile_aligned(item.step_position * block_kv + tile_offset),
                    tile_rows,
                ),
            ]
            rows[...] = jnp.where(
                lax.broadcasted_iota(jnp.int32, rows.shape, 0) < kept_rows,
                rows[...],
                jnp.zeros(rows.shape, rows.dtype),
            )

    def _anchored_softmax(item, scores, visible, row_max, row_sum):
        """The static-anchor step of one item: (probabilities, row_max,
        row_sum, fp8_unscale), row_max the request's anchor; fp8_unscale is
        None for a bf16 cache."""
        # Note (david): a request anchors on the max of its first lane tile of
        # keys, so its exp waits for one tile's max, not the block's; key 0,
        # which every query row sees, makes the anchor a real score. Later
        # items keep the anchor, so no state is ever rescaled.
        first_tile_scores = jnp.where(
            visible[:, :NUM_LANES], scores[:, :NUM_LANES], MASKED_SCORE
        )
        row_max = jnp.where(
            item.is_first_block,
            lane_broadcast(
                first_tile_scores.max(axis=1, keepdims=True), NUM_LANES
            ),
            row_max,
        )
        # Note (david): f32 whatever s_dtype is, as a key's exp argument grows
        # with its distance past the anchor and bf16 would round it in
        # proportion.
        probabilities = jnp.exp2(
            jnp.where(
                visible,
                scores - lane_broadcast(row_max, block_kv),
                MASKED_SCORE,
            )
        )
        # Note (david): summed pairwise over the lane tiles, so each lane keeps
        # its own sum for the fp8 bound below.
        lane_tiles = [
            probabilities[:, i : i + NUM_LANES]
            for i in range(0, block_kv, NUM_LANES)
        ]
        while len(lane_tiles) > 1:
            pairs = [a + b for a, b in zip(lane_tiles[0::2], lane_tiles[1::2])]
            lane_tiles = pairs + lane_tiles[2 * len(pairs) :]
        lane_sums = lane_tiles[0]
        row_sum = jnp.where(item.is_first_block, 0.0, row_sum) + lane_broadcast(
            lane_sums.sum(axis=1, keepdims=True), NUM_LANES
        )
        if not is_quantized:
            return probabilities, row_max, row_sum, None
        # Note (david): an fp8 cache's P @ V takes fp8 probabilities, and an
        # anchored one passes 1 for a key past its anchor. Each is at most its
        # lane's sum, so a row whose lane sums pass FP8_MAX is scaled under it
        # and its P @ V scaled back. The row sum keeps the unscaled values, so
        # a NaN or inf bound still trips the guard.
        max_lane_sum = lane_broadcast(
            lane_sums.max(axis=1, keepdims=True), NUM_LANES
        )
        fp8_scale = jnp.minimum(1.0, FP8_MAX / max_lane_sum)
        scaled_probabilities = probabilities * lane_broadcast(
            fp8_scale, block_kv
        )
        return scaled_probabilities, row_max, row_sum, 1.0 / fp8_scale

    def _visible_keys(item, first_column, num_columns):
        """(rows, num_columns) bool: which of the item's columns
        [first_column, first_column + num_columns) each query row sees."""
        visible_tokens = item.kv_len - item.kv_block * block_kv - first_column
        if is_causal:
            # Note (david): query row r is query token r // H, which sees the
            # positions before kv_len - s_q + 1 + r // H.
            query_token = (
                lax.broadcasted_iota(jnp.int32, (num_query_rows, NUM_LANES), 0)
                // num_heads
            )
            visible_count = lane_broadcast(
                visible_tokens - (query_tokens - 1) + query_token, num_columns
            )
        else:
            visible_count = visible_tokens
        return (
            lax.broadcasted_iota(jnp.int32, (num_query_rows, num_columns), 1)
            < visible_count
        )

    def _attend_step(slot, items, state):
        """The step's items through num_groups groups, each group's PV after
        the next group's QK and softmax. Returns the state after the step,
        each item's (row_max, row_sum, accumulator) and the rows' largest row
        sum over the step's items."""
        group_size = pl.cdiv(len(items), num_groups)
        item_groups = [
            items[i : i + group_size] for i in range(0, len(items), group_size)
        ]
        item_states = {}

        def _score_and_softmax(group, row_max, row_sum, max_row_sum):
            group_scores = {}
            for item in group:
                # Note (david): an fp8 block reaches the MXU as fp8 in both
                # matmuls: Mosaic widens it against the bf16 queries, and
                # P @ V takes fp8 probabilities.
                group_scores[item.step_position] = (
                    lax.dot_general(
                        query_ring[item.query_slot],
                        kv_ring[
                            slot,
                            pl.ds(item.step_position * block_kv, block_kv),
                            :row_dim,
                        ],
                        NT_DIM_NUMBERS,
                        preferred_element_type=jnp.float32,
                    )
                    * score_scale
                )
            group_probabilities = {}
            for item in group:
                probabilities, row_max, row_sum, fp8_unscale = (
                    _anchored_softmax(
                        item,
                        group_scores[item.step_position],
                        _visible_keys(item, 0, block_kv),
                        row_max,
                        row_sum,
                    )
                )
                if max_row_sum is None:
                    max_row_sum = row_sum
                else:
                    max_row_sum = jnp.maximum(max_row_sum, row_sum)
                # Note (david): the probabilities round to the cache's dtype,
                # as MLA v2's p_same_dtype_as_v rounds them, so an fp8
                # cache's P @ V runs at the MXU's fp8 rate.
                group_probabilities[item.step_position] = (
                    probabilities.astype(cache_dtype),
                    fp8_unscale,
                )
                item_states[item.step_position] = (row_max, row_sum)
            return group_probabilities, row_max, row_sum, max_row_sum

        def _accumulate_pv(group, group_probabilities, accumulator):
            for item in group:
                probabilities, fp8_unscale = group_probabilities[
                    item.step_position
                ]
                raw_pv = jnp.dot(
                    probabilities,
                    kv_ring[
                        slot,
                        pl.ds(item.step_position * block_kv, block_kv),
                        :latent_dim,
                    ],
                    preferred_element_type=jnp.float32,
                )
                if fp8_unscale is None:
                    pv = raw_pv
                else:
                    pv = raw_pv * lane_broadcast(fp8_unscale, latent_dim)
                carried = jnp.where(item.is_first_block, 0.0, accumulator)
                accumulator = carried + pv
                item_states[item.step_position] = (
                    *item_states[item.step_position],
                    accumulator,
                )
            return accumulator

        row_max, row_sum, accumulator = state
        first_group, *later_groups = item_groups
        group_probabilities, row_max, row_sum, max_row_sum = _score_and_softmax(
            first_group, row_max, row_sum, None
        )
        previous_group = (first_group, group_probabilities)
        for group in later_groups:
            group_probabilities, row_max, row_sum, max_row_sum = (
                _score_and_softmax(group, row_max, row_sum, max_row_sum)
            )
            accumulator = _accumulate_pv(*previous_group, accumulator)
            previous_group = (group, group_probabilities)
        accumulator = _accumulate_pv(*previous_group, accumulator)
        return (row_max, row_sum, accumulator), item_states, max_row_sum

    def _output_copy(request, output_slot):
        return pltpu.make_async_copy(
            output_ring.at[output_slot],
            output_hbm.at[request],
            dma_sems.at[SEM_OUTPUT, output_slot],
        )

    def _finish_request(item, row_max, row_sum, accumulator):
        output_slot = lax.rem(item.request, num_output_slots)

        @pl.when(item.request >= num_output_slots)
        def _wait_previous_output():
            _output_copy(item.request - num_output_slots, output_slot).wait()

        # Note (david): an inactive request saw no key, so its output is 0
        # and its lse +inf whatever its masked state holds.
        has_keys = item.kv_len > 0
        inverse_row_sum = jnp.where(has_keys, kv_scale / row_sum, 0.0)
        output_ring[output_slot] = (
            accumulator * lane_broadcast(inverse_row_sum, latent_dim)
        ).astype(output_ring.dtype)
        _output_copy(item.request, output_slot).start()
        if return_lse:
            lse = jnp.where(
                has_keys,
                (row_max + jnp.log2(row_sum)) * math.log(2.0),
                jnp.inf,
            )
            lse_tile = item.request // NUM_LANES
            is_request_lane = lax.broadcasted_iota(
                jnp.int32, lse.shape, 1
            ) == lax.rem(item.request, NUM_LANES)
            lse_ref[lse_tile] = jnp.where(
                is_request_lane, lse, lse_ref[lse_tile]
            )

    def _wait_step_copies(slot, items):
        """Wait for the copies of the step's cached tokens, appended tiles and
        new requests' query rows into the ring slot."""
        # Note (david): every copy of a step signals its slot's semaphore,
        # which counts bytes, so one wait sized to the step's tokens covers
        # them all.
        step_tokens = 0
        num_new_requests = 0
        for item in items:
            _, append_end = _append_tile_range(item.kv_len)
            step_tokens = step_tokens + jnp.clip(
                append_end - item.kv_block * block_kv, 0, block_kv
            )
            num_new_requests = num_new_requests + jnp.where(
                jnp.logical_and(item.is_first_block, item.kv_len > 0), 1, 0
            )

        @pl.when(step_tokens > 0)
        def _wait_kv_copies():
            staged = kv_ring.at[slot, pl.ds(0, _tile_aligned(step_tokens))]
            pltpu.make_async_copy(
                staged, staged, dma_sems.at[SEM_KV, slot]
            ).wait()

        @pl.when(num_new_requests > 0)
        def _wait_query_copies():
            query_rows = query_ring.at[
                pl.ds(0, num_new_requests), :, pl.ds(0, latent_dim)
            ]
            pltpu.make_async_copy(
                query_rows, query_rows, dma_sems.at[SEM_QUERY, slot]
            ).wait()
            q_pe_rows = q_pe_ring.at[pl.ds(0, num_new_requests)]
            pltpu.make_async_copy(
                q_pe_rows, q_pe_rows, dma_sems.at[SEM_QUERY, slot]
            ).wait()

    def _run_step(step, carry):
        """One step of the anchored pass."""
        slot = lax.rem(step, NUM_STAGES)
        items = [
            _schedule_item(step, step_position)
            for step_position in range(items_per_step)
        ]
        _wait_step_copies(slot, items)
        _zero_rows_past_kv_len(slot, items)
        # Note (david): q_pe is narrower than a lane tile, which no DMA window
        # can address, so it reaches its lanes of the query rows through
        # vector stores.
        for item in items:
            query_ring[item.query_slot, :, pl.ds(latent_dim, rope_dim)] = (
                q_pe_ring[item.query_slot]
            )
        state, item_states, max_row_sum = _attend_step(
            slot,
            items,
            (row_max_ref[...], row_sum_ref[...], accumulator_ref[...]),
        )
        row_max_ref[...], row_sum_ref[...], accumulator_ref[...] = state
        # Note (david): an anchored row sum only grows within a request and
        # bounds each of its terms, so the largest any item left passes the
        # guard bound exactly when a score may have overflowed (or is NaN,
        # which maximum keeps).
        max_row_sum_ref[...] = jnp.maximum(max_row_sum_ref[...], max_row_sum)
        # Note (david): the copies overwrite the slot the items have just
        # read, so they start after its last load.
        _start_step_copies(
            step + NUM_STAGES, slot, kv_source=KvSource.CACHE, copy_queries=True
        )
        for item in items:
            pl.when(item.is_last_block)(
                functools.partial(
                    _finish_request, item, *item_states[item.step_position]
                )
            )
        return carry

    # Note (david): the replay runs only when the guard trips, so its code size
    # matters and its speed does not. Unrolled as the anchored step is, it
    # pushed the code every call runs past the instruction memory and slowed
    # down the anchored pass without ever replaying. It runs one item at a
    # time in a rolled loop instead, and an item replay_chunk columns at a
    # time, each chunk an online-softmax update of the state refs.
    replay_chunk = NUM_LANES * max(
        tiles
        for tiles in range(REPLAY_CHUNK_TILES, 0, -1)
        if block_kv % (tiles * NUM_LANES) == 0
    )

    def _replay_chunk(item, slot, chunk, carry):
        first_column = chunk * replay_chunk
        first_row = pl.multiple_of(
            item.step_position * block_kv + first_column, replay_chunk
        )
        scores = (
            lax.dot_general(
                query_ring[item.query_slot],
                kv_ring[slot, pl.ds(first_row, replay_chunk), :row_dim],
                NT_DIM_NUMBERS,
                preferred_element_type=jnp.float32,
            )
            * score_scale
        )
        chunk_scores = jnp.where(
            _visible_keys(item, first_column, replay_chunk),
            scores.astype(s_dtype),
            jnp.asarray(MASKED_SCORE, s_dtype),
        )
        is_first = jnp.logical_and(item.is_first_block, chunk == 0)
        previous_max = jnp.where(is_first, MASKED_SCORE, row_max_ref[...])
        previous_sum = jnp.where(is_first, 0.0, row_sum_ref[...])
        chunk_max = chunk_scores.max(axis=1, keepdims=True).astype(jnp.float32)
        row_max = jnp.maximum(
            previous_max, lane_broadcast(chunk_max, NUM_LANES)
        )
        rescale = jnp.exp2(previous_max - row_max)
        probabilities = jnp.exp2(
            chunk_scores - lane_broadcast(row_max.astype(s_dtype), replay_chunk)
        )
        row_max_ref[...] = row_max
        row_sum_ref[...] = rescale * previous_sum + lane_broadcast(
            probabilities.astype(jnp.float32).sum(axis=1, keepdims=True),
            NUM_LANES,
        )
        pv = jnp.dot(
            probabilities.astype(cache_dtype),
            kv_ring[slot, pl.ds(first_row, replay_chunk), :latent_dim],
            preferred_element_type=jnp.float32,
        )
        carried = jnp.where(is_first, 0.0, accumulator_ref[...])
        accumulator_ref[...] = (
            carried * lane_broadcast(rescale, latent_dim) + pv
        )
        return carry

    def _replay_item(step, slot, step_position, carry):
        item = _schedule_item(step, step_position)
        query_ring[item.query_slot, :, pl.ds(latent_dim, rope_dim)] = q_pe_ring[
            item.query_slot
        ]
        lax.fori_loop(
            0,
            block_kv // replay_chunk,
            functools.partial(_replay_chunk, item, slot),
            0,
        )

        @pl.when(item.is_last_block)
        def _finish():
            _finish_request(
                item, row_max_ref[...], row_sum_ref[...], accumulator_ref[...]
            )

        return carry

    def _run_replay_step(step, carry):
        """One step of the online-softmax replay."""
        slot = lax.rem(step, NUM_STAGES)
        items = [
            _schedule_item(step, step_position)
            for step_position in range(items_per_step)
        ]
        _wait_step_copies(slot, items)
        _zero_rows_past_kv_len(slot, items)
        lax.fori_loop(
            0, items_per_step, functools.partial(_replay_item, step, slot), 0
        )
        _start_step_copies(
            step + NUM_STAGES, slot, kv_source=KvSource.CACHE, copy_queries=True
        )
        return carry

    def _append_windows(request):
        """(num_tokens, windows) of the cache tiles that hold the request's
        new tokens, a window (page, page_offset, buffer_offset, num_tokens)
        for each of the two pages they can span."""
        # Note (david): a page holds whole tiles and a request's new tokens
        # fit in one tile's length, so they span at most two pages.
        append_start, append_end = _append_tile_range(kv_lens_ref[request])
        num_tokens = append_end - append_start
        page_index = append_start // page_size
        page_offset = append_start - page_index * page_size
        first_tokens = jnp.minimum(num_tokens, page_size - page_offset)
        second_tokens = num_tokens - first_tokens
        return num_tokens, (
            (
                _physical_page(request, page_index, first_tokens),
                _tile_aligned(page_offset),
                0,
                _tile_aligned(first_tokens),
            ),
            (
                _physical_page(request, page_index + 1, second_tokens),
                0,
                _tile_aligned(jnp.where(second_tokens > 0, first_tokens, 0)),
                _tile_aligned(second_tokens),
            ),
        )

    def _chunk_requests(chunk):
        """(first request, request count) of the append chunk."""
        first_request = chunk * append_chunk
        return first_request, jnp.minimum(append_chunk, batch - first_request)

    def _new_token_copies(chunk):
        first_request, num_requests = _chunk_requests(chunk)
        buffer = chunk % NUM_APPEND_BUFFERS
        return tuple(
            pltpu.make_async_copy(
                source.at[pl.ds(first_request, num_requests)],
                destination.at[pl.ds(buffer * append_chunk, num_requests)],
                dma_sems.at[SEM_NEW_TOKENS, buffer],
            )
            for source, destination in (
                (new_kv_c_hbm, new_latent),
                (new_k_pe_hbm, new_rope),
            )
        )

    def _wait_append_copies(sem_row, buffer, request, carry):
        num_tokens, _ = _append_windows(request)

        @pl.when(num_tokens > 0)
        def _wait_tiles():
            tiles = append_tiles.at[
                _append_row(request), pl.ds(0, _tile_aligned(num_tokens))
            ]
            pltpu.make_async_copy(
                tiles, tiles, dma_sems.at[sem_row, buffer]
            ).wait()

        return carry

    def _for_chunk_requests(chunk, body):
        """Run body(buffer, request, carry) over the chunk's requests."""
        first_request, num_requests = _chunk_requests(chunk)
        lax.fori_loop(
            first_request,
            first_request + num_requests,
            functools.partial(body, chunk % NUM_APPEND_BUFFERS),
            0,
        )

    def _start_append_reads(buffer, request, carry):
        _, windows = _append_windows(request)
        for page_id, page_offset, buffer_offset, num_tokens in windows:
            pltpu.make_async_copy(
                cache_token_rows.at[page_id, pl.ds(page_offset, num_tokens)],
                append_tiles.at[
                    _append_row(request), pl.ds(buffer_offset, num_tokens)
                ],
                dma_sems.at[SEM_APPEND_READ, buffer],
            ).start()
        return carry

    def _merge_new_tokens(buffer, request, carry):
        """Patch the request's new tokens into its tiles, keeping every other
        row's bits, and start their write-back to the cache."""
        _, windows = _append_windows(request)
        row = _append_row(request)
        kv_len = kv_lens_ref[request]
        append_start, _ = _append_tile_range(kv_len)
        new_token_start = kv_len - query_tokens
        new_token_rows[:, :latent_dim] = new_latent[row]
        new_token_rows[:, latent_dim : latent_dim + rope_dim] = new_rope[row]
        old_tiles = append_tiles[row]
        token_position = append_start + lax.broadcasted_iota(
            jnp.int32, old_tiles.shape, 0
        )
        merged_tiles = old_tiles
        for token in range(query_tokens):
            merged_tiles = jnp.where(
                token_position == new_token_start + token,
                jnp.broadcast_to(
                    new_token_rows[pl.ds(token, 1), :], old_tiles.shape
                ),
                merged_tiles,
            )
        append_tiles[row] = merged_tiles
        for page_id, page_offset, buffer_offset, num_tokens in windows:
            pltpu.make_async_copy(
                append_tiles.at[row, pl.ds(buffer_offset, num_tokens)],
                cache_token_rows.at[page_id, pl.ds(page_offset, num_tokens)],
                dma_sems.at[SEM_APPEND_WRITE, buffer],
            ).start()
        return carry

    def _start_chunk_reads(chunk):
        for copy in _new_token_copies(chunk):
            copy.start()
        _for_chunk_requests(chunk, _start_append_reads)

    def _wait_chunk_writes(chunk):
        _for_chunk_requests(
            chunk, functools.partial(_wait_append_copies, SEM_APPEND_WRITE)
        )

    def _append_chunk(index, carry):
        """Merge one chunk, last chunk first, and start the reads of the
        chunk before it. Each buffer signals its own semaphores."""
        chunk = num_chunks - 1 - index

        @pl.when(chunk > 0)
        def _start_previous_chunk_reads():
            # Note (david): the chunk before reuses the buffer of the chunk
            # NUM_APPEND_BUFFERS - 1 after it.
            reused_chunk = chunk - 1 + NUM_APPEND_BUFFERS

            @pl.when(reused_chunk < num_chunks)
            def _wait_reused_buffer():
                _wait_chunk_writes(reused_chunk)

            _start_chunk_reads(chunk - 1)

        for copy in _new_token_copies(chunk):
            copy.wait()
        _for_chunk_requests(
            chunk, functools.partial(_wait_append_copies, SEM_APPEND_READ)
        )
        _for_chunk_requests(chunk, _merge_new_tokens)
        return carry

    def _wait_last_append_writes():
        # Note (david): the chunks run last first, so the first ones may still
        # write back.
        for chunk in range(min(num_chunks, NUM_APPEND_BUFFERS)):
            _wait_chunk_writes(chunk)

    query_ring[:, :, pl.ds(latent_dim, row_dim - latent_dim)] = jnp.zeros(
        (num_query_slots, num_query_rows, row_dim - latent_dim),
        query_ring.dtype,
    )
    new_token_rows[...] = jnp.zeros(new_token_rows.shape, new_token_rows.dtype)
    row_max_ref[...] = jnp.full(row_max_ref.shape, MASKED_SCORE, jnp.float32)
    row_sum_ref[...] = jnp.zeros(row_sum_ref.shape, jnp.float32)
    accumulator_ref[...] = jnp.zeros(accumulator_ref.shape, jnp.float32)
    max_row_sum_ref[...] = jnp.zeros(max_row_sum_ref.shape, jnp.float32)
    _start_chunk_reads(num_chunks - 1)
    # Note (david): the first steps' cached tokens and queries do not depend
    # on the append, so they start before it and overlap it.
    for step in range(NUM_STAGES):
        kv_ring[step] = jnp.zeros(kv_ring.shape[1:], kv_ring.dtype)
        _start_step_copies(
            step,
            step,
            kv_source=KvSource.CACHE_BEFORE_APPEND,
            copy_queries=True,
        )
    lax.fori_loop(0, num_chunks, _append_chunk, 0)
    # Note (david): the first steps take their appended tiles from the
    # buffers, as the write-back may still run.
    for step in range(NUM_STAGES):
        _start_step_copies(
            step, step, kv_source=KvSource.APPEND_BUFFER, copy_queries=False
        )
    # Note (david): the later steps take their appended tiles from the cache.
    _wait_last_append_writes()
    lax.fori_loop(0, num_steps, _run_step, 0)
    # Note (david): only the last num_output_slots output copies still run;
    # each earlier one was waited for by the finish that reused its slot.
    for request in range(max(0, batch - num_output_slots), batch):
        _output_copy(request, request % num_output_slots).wait()

    # Note (david): the steps copy into the slot they have just read, so no
    # step can replay in place; the replay reruns the whole schedule once with
    # the online softmax, rewriting every output. Every write-back has landed,
    # so the replay stages every token from the cache.
    should_replay_ref[0] = jnp.any(
        jnp.logical_not(max_row_sum_ref[...] <= 2.0**guard_threshold)
    ).astype(jnp.int32)

    @pl.when(should_replay_ref[0] > 0)
    def _replay():
        row_max_ref[...] = jnp.full(
            row_max_ref.shape, MASKED_SCORE, jnp.float32
        )
        row_sum_ref[...] = jnp.zeros(row_sum_ref.shape, jnp.float32)
        accumulator_ref[...] = jnp.zeros(accumulator_ref.shape, jnp.float32)
        for step in range(NUM_STAGES):
            _start_step_copies(
                step, step, kv_source=KvSource.CACHE, copy_queries=True
            )
        lax.fori_loop(0, num_steps, _run_replay_step, 0)
        for request in range(max(0, batch - num_output_slots), batch):
            _output_copy(request, request % num_output_slots).wait()


def mla_decode_pallas(
    q_nope: jax.Array,
    q_pe: jax.Array,
    new_kv_c: jax.Array,
    new_k_pe: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    block_table: jax.Array,
    *,
    kv_scale: jax.Array | None,
    sm_scale: float,
    s_dtype: jnp.dtype,
    causal: bool,
    return_lse: bool,
    interpret: bool,
    config: MlaDecodeConfig,
) -> tuple[jax.Array, ...]:
    """flash_mla_decode's pallas_call on operands it has checked, at config.

    Returns (out, kv_cache), or (out, kv_cache, lse) with return_lse, as
    flash_mla_decode does.
    """
    batch, query_tokens, num_heads, latent_dim = q_nope.shape
    rope_dim = q_pe.shape[-1]
    _, page_words, packing, row_dim = kv_cache.shape
    page_size = page_words * packing
    pages_per_seq = block_table.shape[1]
    cache_dtype = kv_cache.dtype
    is_quantized = kv_scale is not None
    # Note (david): a request's query tokens fold into its query rows,
    # token-major, so one score matmul an item covers all s_q * H rows as the
    # MXU's M dimension.
    num_query_rows = query_tokens * num_heads
    block_kv, items_per_step = config
    # Note (david): an item is whole lane tiles, and whole pages or a whole
    # fraction of one.
    if (
        block_kv <= 0
        or block_kv % NUM_LANES
        or (block_kv % page_size and page_size % block_kv)
        or items_per_step < 1
    ):
        raise ValueError(
            f"{config} needs block_kv a multiple of {NUM_LANES} and either a"
            f" multiple or a divisor of page_size {page_size}, and"
            " items_per_step >= 1."
        )
    blocks_per_request = pl.cdiv(pages_per_seq * page_size, block_kv)
    if blocks_per_request >= KV_BLOCK_MASK or batch >= 1 << (
        31 - REQUEST_SHIFT
    ):
        raise ValueError(
            f"batch {batch} or {blocks_per_request} blocks a request overflow"
            " a schedule entry."
        )
    max_items = (
        pl.cdiv(batch * blocks_per_request, items_per_step) * items_per_step
    )
    schedule = mla_decode_schedule(
        kv_lens, block_kv=block_kv, max_items=max_items
    )
    q_nope_rows = q_nope.reshape(batch, num_query_rows, latent_dim)
    q_pe_rows = q_pe.reshape(batch, num_query_rows, rope_dim)
    # Note (david): the Pallas interpreter takes no ref reshape, so an
    # interpreted call hands the kernel the token rows.
    if interpret:
        cache_operand = kv_cache.reshape(kv_cache.shape[0], page_size, row_dim)
    else:
        cache_operand = kv_cache
    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    out_shapes = [
        jax.ShapeDtypeStruct(q_nope_rows.shape, q_nope.dtype),
        jax.ShapeDtypeStruct(cache_operand.shape, cache_dtype),
    ]
    out_specs = [hbm_spec, hbm_spec]
    if return_lse:
        lse_shape = (pl.cdiv(batch, NUM_LANES), num_query_rows, NUM_LANES)
        out_shapes.append(jax.ShapeDtypeStruct(lse_shape, jnp.float32))
        out_specs.append(pl.BlockSpec(lse_shape, lambda *_: (0, 0, 0)))
    num_query_slots = NUM_STAGES * items_per_step
    num_output_slots = 2 * items_per_step
    append_chunk = mla_append_chunk(batch, items_per_step)
    num_append_rows = NUM_APPEND_BUFFERS * append_chunk
    append_tokens = max_append_tokens(query_tokens, cache_dtype)
    scratch_shapes = (
        # kv_ring
        pltpu.VMEM(
            (NUM_STAGES, items_per_step * block_kv, row_dim), cache_dtype
        ),
        # query_ring
        pltpu.VMEM((num_query_slots, num_query_rows, row_dim), q_nope.dtype),
        # q_pe_ring
        pltpu.VMEM((num_query_slots, num_query_rows, rope_dim), q_pe.dtype),
        # new_latent
        pltpu.VMEM((num_append_rows, query_tokens, latent_dim), cache_dtype),
        # new_rope
        pltpu.VMEM((num_append_rows, query_tokens, rope_dim), cache_dtype),
        # new_token_rows
        pltpu.VMEM((query_tokens, row_dim), cache_dtype),
        # append_tiles
        pltpu.VMEM((num_append_rows, append_tokens, row_dim), cache_dtype),
        # output_ring
        pltpu.VMEM(
            (num_output_slots, num_query_rows, latent_dim), q_nope.dtype
        ),
        # row_max_ref
        pltpu.VMEM((num_query_rows, NUM_LANES), jnp.float32),
        # row_sum_ref
        pltpu.VMEM((num_query_rows, NUM_LANES), jnp.float32),
        # accumulator_ref
        pltpu.VMEM((num_query_rows, latent_dim), jnp.float32),
        # max_row_sum_ref
        pltpu.VMEM((num_query_rows, NUM_LANES), jnp.float32),
        # should_replay_ref
        pltpu.SMEM((1,), jnp.int32),
        # dma_sems
        pltpu.SemaphoreType.DMA((
            NUM_SEM_ROWS,
            max(NUM_STAGES, num_output_slots, NUM_APPEND_BUFFERS),
        )),
    )
    kernel = functools.partial(
        mla_decode_kernel,
        batch=batch,
        page_size=page_size,
        pages_per_seq=pages_per_seq,
        config=config,
        append_chunk=append_chunk,
        num_heads=num_heads,
        sm_scale_log2=sm_scale * LOG2E,
        s_dtype=s_dtype,
        is_causal=causal,
        return_lse=return_lse,
        is_quantized=is_quantized,
        # Note (david): sized on the block table's capacity, which bounds
        # every kv_lens.
        guard_threshold=overflow_guard_threshold(pages_per_seq * page_size),
    )
    scalar_prefetches = (schedule, kv_lens, block_table.reshape(-1))
    inputs = [q_nope_rows, q_pe_rows, new_kv_c, new_k_pe, cache_operand]
    in_specs = [hbm_spec] * len(inputs)
    if is_quantized:
        inputs.append(kv_scale.reshape(1))
        in_specs.append(pl.BlockSpec(memory_space=pltpu.SMEM))
    call = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalar_prefetches),
            grid=(1,),
            in_specs=in_specs,
            out_specs=out_specs,
            scratch_shapes=scratch_shapes,
        ),
        out_shape=out_shapes,
        # Note (david): the cache operand aliases output 1, so the append
        # lands in place.
        input_output_aliases={len(scalar_prefetches) + 4: 1},
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=vmem_limit_bytes(),
        ),
        name="flash_mla_decode",
        interpret=pltpu.InterpretParams() if interpret else False,
    )
    out, updated_cache, *lse_block = call(*scalar_prefetches, *inputs)
    out = out.reshape(q_nope.shape)
    updated_cache = updated_cache.reshape(kv_cache.shape)
    if not return_lse:
        return out, updated_cache
    # Note (david): the kernel keeps request b's lse in lane b % NUM_LANES of
    # tile b // NUM_LANES, its query rows token-major; this relayout costs far
    # less than one cache read.
    lse = lse_block[0].transpose(0, 2, 1).reshape(-1, query_tokens, num_heads)
    return out, updated_cache, lse[:batch].transpose(0, 2, 1)
